"""Fit and export the final merged pair ranker after OOF model selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import ROOT
from src.features import FEATURE_NAMES
from src.neural_pair import encode_pairs
from src.train_ranker import Context


MODEL_DIR = Path("D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B")


def prepare_groups(tokenizer, max_groups, recipe="raw", negative_mode="mixed"):
    ctx = Context(4, 1, seed=7, use_llm=True, allow_llm_generate=False)
    cat_rows = ctx.catalog.to_dict("records")
    items = [(row, cand, block, target) for i, (row, cand, block, target) in enumerate(
        zip(ctx.rows, ctx.cands, ctx.blocks, ctx.truth)) if ctx.hangul[i]]
    items += [(row, cand, block, ctx.truth[origin]) for row, cand, block, origin in zip(
        ctx.aug_rows, ctx.aug_cands, ctx.aug_blocks, ctx.aug_origin) if ctx.hangul[origin]]
    rng = np.random.default_rng(42)
    groups = []
    if recipe == "evidence":
        from src.pair_evidence import encode_pairs as evidence_encode
    mining_scores = None
    if negative_mode == "hard":
        import lightgbm as lgb
        from src.train_ranker import PARAMS
        from src.experiment import _stack, _predict_blocks
        blocks = [item[2] for item in items]
        labels = [(c == target).astype(np.float32) for _, c, _, target in items]
        xs, ys, group = _stack(blocks, labels)
        miner = lgb.train({**PARAMS, "num_threads": 6},
                          lgb.Dataset(xs, ys, group=group, feature_name=FEATURE_NAMES), num_boost_round=150)
        del xs, ys, group, labels
        gc.collect()
        mining_scores = _predict_blocks(lambda x: miner.predict(x, num_threads=6), blocks)
        del miner, blocks
        print("Full-training-only hard-negative miner fitted", flush=True)
    cols = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]
    for j in rng.permutation(len(items)):
        row, c, block, target = items[j]
        hit = np.flatnonzero(c == target)
        if not len(hit) or len(c) < 8:
            continue
        strength = np.maximum(block[:, cols[0]], block[:, cols[1]]) - .20 * block[:, cols[2]]
        negatives = [i for i in np.argsort(-strength, kind="stable") if i != hit[0]]
        if negative_mode == "hard":
            learned = [int(i) for i in np.argsort(-mining_scores[j], kind="stable") if i != hit[0]][:5]
            lexical = [i for i in negatives if i not in learned][:2]
            selected = [int(hit[0])] + learned + lexical
        else:
            hard = negatives[:3]
            other = [i for i in negatives if i not in hard]
            selected = [int(hit[0])] + hard + rng.choice(other, size=4, replace=False).tolist()
        choices = [cat_rows[i] for i in c[selected]]
        groups.append(evidence_encode(tokenizer, row, choices, ctx.beams([row["vessel_name"]])[0])
                      if recipe == "evidence" else encode_pairs(tokenizer, row, choices))
        if len(groups) >= max_groups:
            break
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-artifacts", type=Path, default=ROOT / "experiments" / "artifacts" / "ranker_v8")
    parser.add_argument("--max-groups", type=int, default=2400)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--recipe", choices=["raw", "evidence"], default="raw")
    parser.add_argument("--negative-mode", choices=["mixed", "hard"], default="mixed")
    parser.add_argument("--variant", choices=["evidence_only", "equal_neural_ensemble"], default="evidence_only")
    parser.add_argument("--raw-artifacts", type=Path, default=ROOT / "experiments/artifacts/ranker_v10_neural")
    args = parser.parse_args()
    if args.max_groups < 1:
        parser.error("--max-groups must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="left")
    fingerprint = hashlib.sha256(f"{args.max_groups}:{args.recipe}:{args.negative_mode}".encode())
    for path in [Path(__file__), ROOT / "train.csv", ROOT / "vessel_catalog.csv"] + [
        ROOT / "src" / name for name in ("neural_pair.py", "features.py", "retrieval.py", "augment.py", "hangul.py", "common.py")]:
        fingerprint.update(path.read_bytes())
    if args.recipe == "evidence":
        fingerprint.update((ROOT / "src/pair_evidence.py").read_bytes())
    digest = fingerprint.hexdigest()
    group_cache = ROOT / "experiments" / "cache" / f"final_pair_groups_{digest[:12]}.joblib"
    if group_cache.exists():
        cached = joblib.load(group_cache)
        assert cached["fingerprint"] == digest
        groups = cached["groups"]
    else:
        groups = prepare_groups(tokenizer, args.max_groups, args.recipe, args.negative_mode)
        joblib.dump({"fingerprint": digest, "groups": groups}, group_cache, compress=3)
    gc.collect()
    print(f"Final neural training groups: {len(groups)}", flush=True)
    if args.prepare_only:
        print(f"Prepared: {group_cache}", flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                            task_type="CAUSAL_LM", bias="none"))
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    yes, no = [tokenizer.convert_tokens_to_ids(w) for w in ("yes", "no")]
    start = time.perf_counter()
    order = np.random.default_rng(42).permutation(len(groups))
    loss_sum = 0.0
    for step, offset in enumerate(range(0, len(order), 2), 1):
        batch = order[offset:offset + 2]
        tokens = [seq for i in batch for seq in groups[i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        scores = (logits[:, yes] - logits[:, no]).reshape(len(batch), 8)
        loss = F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite final training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.item())
        if step % 100 == 0:
            print(f"final step {step}, loss {loss_sum/100:.4f}, elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            loss_sum = 0.0
            # Variable sequence lengths can otherwise retain nearly all 16GB
            # of a Windows GPU's allocator cache during long no-checkpoint runs.
            torch.cuda.empty_cache()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "training_adapter")
    model = model.merge_and_unload().eval()
    model_name = "reranker_evidence" if args.recipe == "evidence" else "reranker"
    model.save_pretrained(out / model_name)
    tokenizer.save_pretrained(out / model_name)
    license_path = MODEL_DIR / "LICENSE"
    if license_path.exists():
        shutil.copy(license_path, out / model_name / "LICENSE")
    if (MODEL_DIR / "README.md").exists():
        shutil.copy(MODEL_DIR / "README.md", out / model_name / "MODEL_CARD.md")
    revision = json.loads((MODEL_DIR / "codex_revision.json").read_text(encoding="utf-8"))
    (out / model_name / "NOTICE").write_text(
        f"{revision['repo']}\nOriginal model: https://huggingface.co/{revision['repo']}\n"
        f"Revision: {revision['revision']}\nThe original model card declares Apache-2.0.\n"
        "Modified: merged LoRA adapter trained on official DACON train/catalog pairs.\n",
        encoding="utf-8")
    base = json.loads((args.base_artifacts / "ensemble.json").read_text(encoding="utf-8"))
    for name in ["mappers.joblib", "ensemble.json"] + [m["file"] for m in base["members"]]:
        shutil.copy(args.base_artifacts / name, out / name)
    spec = {"feature_names": FEATURE_NAMES, "model_dir": model_name, "top_k": 20, "weight": .25,
            "recipe": args.recipe, "negative_mode": args.negative_mode,
            "lookup": False, "training_groups": len(groups), "seed": 42, "epochs": 1,
            "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4, "groups_per_batch": 2,
            "pruning": "prediction-preserving probability gap bound", "tonnage_encoding": "canonical float",
            "source_revision": revision,
            "source_hashes": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                              for name in ("neural_pair.py", "features.py", "retrieval.py", "common.py",
                                           "ensemble.py", "hangul.py", "llm_translit.py", "llm_cache.py")},
            "data_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ("train.csv", "vessel_catalog.csv")},
            "training_minutes": (time.perf_counter() - start) / 60}
    if args.recipe == "evidence":
        spec["views"] = [{"dir": model_name, "recipe": "evidence", "weight": 1.0}]
        if args.variant == "equal_neural_ensemble":
            shutil.copytree(args.raw_artifacts / "reranker", out / "reranker_raw", dirs_exist_ok=True)
            spec["views"].insert(0, {"dir": "reranker_raw", "recipe": "raw", "weight": 1.0})
        spec["variant"] = args.variant
        for name in ("pair_evidence.py", "pair_views.py"):
            spec["source_hashes"][name] = hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
        metadata = "pair_views.json"
    else:
        metadata = "neural.json"
    (out / metadata).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved final neural ensemble: {out}", flush=True)


if __name__ == "__main__":
    main()
