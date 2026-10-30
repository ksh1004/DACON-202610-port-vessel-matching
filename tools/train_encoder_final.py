"""Fit a full-data phonetic encoder and export merged BF16 model views."""
import argparse
import gc
import hashlib
import json
import random
import shutil
import time

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import ROOT
from src.encoder_experiment import MODEL_DIR
from src.encoder_phonetics import encode_pairs
from src.experiment import _stack, _predict_blocks
from src.features import FEATURE_NAMES
from src.train_ranker import Context, PARAMS


def groups(tokenizer, count):
    ctx = Context(4, 1, seed=7, use_llm=True, allow_llm_generate=False)
    cat = ctx.catalog.to_dict("records")
    items = [(r, c, b, t) for i, (r, c, b, t) in enumerate(zip(ctx.rows, ctx.cands, ctx.blocks, ctx.truth)) if ctx.hangul[i]]
    items += [(r, c, b, ctx.truth[o]) for r, c, b, o in zip(ctx.aug_rows, ctx.aug_cands, ctx.aug_blocks, ctx.aug_origin) if ctx.hangul[o]]
    blocks = [b for _, _, b, _ in items]
    labels = [(c == t).astype(np.float32) for _, c, _, t in items]
    xs, ys, group = _stack(blocks, labels)
    model = lgb.train({**PARAMS, "num_threads": 6}, lgb.Dataset(xs, ys, group=group, feature_name=FEATURE_NAMES), num_boost_round=150)
    del xs, ys, group, labels
    gc.collect()
    mining = _predict_blocks(lambda x: model.predict(x, num_threads=6), blocks)
    del model, blocks
    print("Full-training-only negative miner ready", flush=True)
    result = []
    cols = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]
    for i in np.random.default_rng(42).permutation(len(items)):
        row, c, block, truth = items[i]
        hit = np.flatnonzero(c == truth)
        if not len(hit) or len(c) < 8:
            continue
        strength = np.maximum(block[:, cols[0]], block[:, cols[1]])-.2*block[:, cols[2]]
        hard = [int(j) for j in np.argsort(-mining[i], kind="stable") if j != hit[0]][:5]
        lexical = [int(j) for j in np.argsort(-strength, kind="stable") if j != hit[0] and j not in hard][:2]
        selected = [int(hit[0])] + hard + lexical
        result.append(encode_pairs(tokenizer, row, [cat[j] for j in c[selected]], ctx.beams([row["vessel_name"]])[0]))
        if len(result) == count:
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/artifacts/ranker_v12_encoder")
    parser.add_argument("--max-groups", type=int, default=2400)
    parser.add_argument("--variant", choices=["encoder_only", "equal_ensemble"], default="equal_ensemble")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    digest = hashlib.sha256(str(args.max_groups).encode())
    files = [ROOT / "train.csv", ROOT / "vessel_catalog.csv", MODEL_DIR / "tokenizer.json", MODEL_DIR / "codex_revision.json",
             ROOT / "tools/train_encoder_final.py"] + [ROOT / "src" / name for name in
              ("encoder_pair.py", "encoder_phonetics.py", "neural_pair.py", "features.py", "retrieval.py", "augment.py", "train_ranker.py", "experiment.py", "common.py", "hangul.py")]
    for file in files:
        digest.update(file.read_bytes())
    cache = ROOT / "experiments/cache" / f"final_encoder_groups_{digest.hexdigest()[:12]}.joblib"
    if cache.exists():
        data = joblib.load(cache)
        assert data["fingerprint"] == digest.hexdigest()
        training = data["groups"]
    else:
        training = groups(tokenizer, args.max_groups)
        joblib.dump({"fingerprint": digest.hexdigest(), "groups": training}, cache, compress=3)
    gc.collect()
    print(f"Final encoder groups: {len(training)}; {cache}", flush=True)
    if args.prepare_only:
        return
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["query", "key", "value"],
                                           modules_to_save=["classifier"], task_type="SEQ_CLS", bias="none"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    ordering = np.random.default_rng(42).permutation(len(training))
    start = time.perf_counter()
    model.train(); losses = []
    for step, offset in enumerate(range(0, len(ordering), 2), 1):
        ids = ordering[offset:offset+2]
        tokens = [tokens for i in ids for tokens in training[i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(ids), 8).float()
            loss = F.cross_entropy(scores, torch.zeros(len(ids), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite encoder loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"final encoder step {step}, loss {np.mean(losses[-100:]):.4f}, elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            torch.cuda.empty_cache()
    out = (ROOT / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "training_adapter")
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(out / "encoder"); tokenizer.save_pretrained(out / "encoder")
    for name, destination in (("README.md", "MODEL_CARD.md"), ("LICENSE", "LICENSE")):
        shutil.copy(MODEL_DIR / name, out / "encoder" / destination)
    revision = json.loads((MODEL_DIR / "codex_revision.json").read_text())
    (out / "encoder/NOTICE").write_text(f"{revision['repo']}\nRevision: {revision['revision']}\nModified: merged LoRA and classifier trained on official DACON train/catalog pairs.\n", encoding="utf-8")
    base = ROOT / "experiments/artifacts/ranker_v8"
    ensemble = json.loads((base / "ensemble.json").read_text())
    for name in ["ensemble.json", "mappers.joblib"]+[m["file"] for m in ensemble["members"]]:
        shutil.copy(base / name, out / name)
    views = [{"dir": "encoder", "kind": "encoder", "recipe": "phonetic", "weight": 1.0}]
    if args.variant == "equal_ensemble":
        shutil.copytree(ROOT / "experiments/artifacts/ranker_v11_pairviews/reranker_evidence", out / "qwen", dirs_exist_ok=True)
        views.insert(0, {"dir": "qwen", "kind": "qwen", "recipe": "evidence", "weight": 1.0})
    modules = ("common.py", "features.py", "retrieval.py", "hangul.py", "ensemble.py", "llm_cache.py", "llm_translit.py",
               "neural_pair.py", "pair_evidence.py", "pair_views.py", "encoder_pair.py", "encoder_phonetics.py", "encoder_views.py")
    spec = {"feature_names": FEATURE_NAMES, "top_k": 20, "weight": .25, "views": views, "variant": args.variant,
            "lookup": False, "source_revision": revision, "training_groups": len(training), "epochs": 1, "seed": 42,
            "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4, "groups_per_batch": 2,
            "training_group_fingerprint": digest.hexdigest(), "training_minutes": (time.perf_counter()-start)/60,
            "data_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("train.csv", "vessel_catalog.csv")},
            "source_hashes": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() for name in modules}}
    (out / "encoder_views.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved final encoder views: {out}", flush=True)


if __name__ == "__main__":
    main()
