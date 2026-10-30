"""LoRA pair-ranker screen on strictly held-out vessels and candidate retrieval.

No call sign, vessel ID, record ID, or lookup table is exposed to the network.
Training negatives use training-only features/models. Validation top20 comes
from the strict LightGBM or archived v8 OOF. No validation truth adds candidates.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.augment import augment
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.ensemble import normalize
from src.experiment import TESTLIKE_MIX, build_testlike
from src.features import FEATURE_NAMES
from src.validate import load_folds


MODEL_DIR = "D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B"
PREFIX = ('<|im_start|>system\nDetermine whether the two provided records identify the same ship. '
          'Answer only yes or no.<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = ("Match a port record to a ship catalog entry. Names can have Korean transliteration, "
               "misspellings or omitted words. Tonnage can be missing or inaccurate. "
               "Compare name, numbers, tonnage, type and flag; missing fields do not indicate a mismatch.")


def record_text(row, catalog=False):
    keys = ["vessel_name", "gross_tonnage", "vessel_type", "flag_name"]
    if not catalog:
        keys += ["international_gross_tonnage"]
    values = {k: None if pd.isna(row.get(k)) else row.get(k) for k in keys}
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=2)
    parser.add_argument("--tag", default="neural_v1")
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--merge-eval", action="store_true")
    parser.add_argument("--eval-adapter", type=Path)
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--bound-prune", action="store_true", help="Skip neural inference when it cannot change the 25% blend winner")
    parser.add_argument("--v8-oof-eval", action="store_true", help="Compare against saved v8 OOF probabilities on identical candidates")
    parser.add_argument("--recipe", choices=["raw", "evidence"], default="raw")
    parser.add_argument("--negative-mode", choices=["mixed", "hard"], default="mixed")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    hangul = train.vessel_name.map(is_hangul).to_numpy()
    paths = sorted((ROOT / "experiments" / "cache").glob(f"strict_han_f{args.fold}_train_*.joblib"),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("Run strict_hangul_experiment first")
    key = paths[-1].stem.rsplit("_", 1)[-1]
    matrices = {s: joblib.load(ROOT / "experiments" / "cache" / f"strict_han_f{args.fold}_{s}_{key}.joblib")
                for s in ("train", "oof", "hard", "testlike")}
    aug = augment(train, 4, 1, seed=7)
    aug = aug.loc[hangul[aug.origin.to_numpy()] & (folds[aug.origin.to_numpy()] != args.fold)]
    original = train.loc[(folds != args.fold) & hangul]
    tr_df = pd.concat([original, aug], ignore_index=True)
    from types import SimpleNamespace
    tl = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=hangul))
    hard = augment(train, 2, 1, seed=2026)
    hard = hard.loc[hangul[hard.origin.to_numpy()] & (folds[hard.origin.to_numpy()] == args.fold)].copy()
    hard["sample_id"] = hard.sample_id + "_hard"
    dfs = {"train": tr_df, "oof": train.loc[(folds == args.fold) & hangul], "hard": hard,
           "testlike": tl.loc[folds[tl.origin.to_numpy()] == args.fold]}
    for s, df in dfs.items():
        assert matrices[s]["sample_ids"] == df.sample_id.tolist(), s
    train_vessels = set(tr_df.target_vessel_id)
    valid_vessels = set(dfs["oof"].target_vessel_id)
    assert not train_vessels & valid_vessels
    baseline = None if args.v8_oof_eval else lgb.Booster(
        model_file=str(ROOT / "experiments" / "artifacts" / "strict_han_v1" / f"base_fold{args.fold}.txt"))
    v8_scores, v8_identity, v8_positions = None, None, None
    if args.v8_oof_eval:
        from src.blend import blend
        from src.experiment import SCORES
        members = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9", "lgb_base_prior", "lgb_base_prior_a8"]
        v8_scores = blend([joblib.load(SCORES / f"{name}.joblib") for name in members], [3, 3, 3, 1, 1], "prob")
        v8_identity = joblib.load(SCORES / "hardcat_only_match_v1.identity.joblib")
        v8_positions = {s: {sample: i for i, sample in enumerate(v8_identity["sets"][s]["sample_ids"])}
                        for s in ("oof", "hard", "testlike")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="left")
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    token_yes, token_no = [tokenizer.convert_tokens_to_ids(w) for w in ("yes", "no")]
    cat_text = [record_text(row, catalog=True) for row in catalog.to_dict("records")]
    cat_rows = catalog.to_dict("records") if args.recipe == "evidence" else None
    if args.recipe == "evidence":
        from src.llm_cache import BeamCache
        from src.pair_evidence import encode_pairs as encode_evidence
        beam_cache = BeamCache(allow_generate=False)

    def encode(row, indices):
        if args.recipe == "evidence":
            return encode_evidence(tokenizer, row, [cat_rows[i] for i in indices],
                                   beam_cache([row["vessel_name"]])[0])
        query = record_text(row)
        texts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {cat_text[i]}" for i in indices]
        tokens = tokenizer(texts, add_special_tokens=False, truncation=True,
                           max_length=256 - len(prefix) - len(suffix)).input_ids
        return [prefix + ids + suffix for ids in tokens]

    rng = np.random.default_rng(42)
    groups = []
    col_name = FEATURE_NAMES.index("llm_ratio_max")
    col_key = FEATURE_NAMES.index("key_lev")
    col_gap = FEATURE_NAMES.index("best_gap")
    order = rng.permutation(len(tr_df))
    mining_scores = None
    if args.negative_mode == "hard" and not args.eval_adapter:
        from src.experiment import _predict_blocks
        miner = lgb.Booster(model_file=str(ROOT / "experiments" / "artifacts" / "strict_han_v1" / f"base_fold{args.fold}.txt"))
        mining_scores = _predict_blocks(lambda x: miner.predict(x, num_threads=4),
                                       [b[:, :len(FEATURE_NAMES)] for b in matrices["train"]["blocks"]])
        del miner
    for i in ([] if args.eval_adapter else order):
        c, b, target = matrices["train"]["candidates"][i], matrices["train"]["blocks"][i], matrices["train"]["truth"][i]
        hit = np.flatnonzero(c == target)
        if not len(hit) or len(c) < 8:
            continue
        # Fixed lexical/tonnage ordering is independent of every validation label.
        strength = np.maximum(b[:, col_name], b[:, col_key]) - .20 * b[:, col_gap]
        negatives = [j for j in np.argsort(-strength, kind="stable") if j != hit[0]]
        if args.negative_mode == "hard":
            learned = [int(j) for j in np.argsort(-mining_scores[i], kind="stable") if j != hit[0]][:5]
            lexical = [j for j in negatives if j not in learned][:2]
            pick = [int(hit[0])] + learned + lexical
        else:
            hard_pick = negatives[:3]
            remaining = [j for j in negatives if j not in hard_pick]
            random_pick = rng.choice(remaining, size=4, replace=False).tolist()
            pick = [int(hit[0])] + hard_pick + random_pick
        groups.append(encode(tr_df.iloc[i].to_dict(), c[pick]))
        if args.max_groups and len(groups) >= args.max_groups:
            break
    # Validation candidate lists are fixed before neural training.
    eval_data = {}
    for s in ("oof", "hard", "testlike"):
        m = matrices[s]
        eval_data[s] = []
        for row, c, block, target in zip(dfs[s].to_dict("records"), m["candidates"], m["blocks"], m["truth"]):
            if args.v8_oof_eval:
                j = v8_positions[s][row["sample_id"]]
                c = v8_identity["sets"][s]["candidates"][j]
                base = v8_scores[s][j].astype(float)
                base /= base.sum()
            else:
                base = baseline.predict(block[:, :baseline.num_feature()], num_threads=4)
            keep = np.argsort(-base, kind="stable")[:20]
            eval_data[s].append({"tokens": encode(row, c[keep]), "keep": keep,
                                 "base": base, "candidates": c, "truth": target})
    del matrices, baseline, v8_scores, v8_identity, v8_positions, mining_scores
    gc.collect()
    print(f"Neural training groups {len(groups)}, validation rows {sum(len(v) for v in eval_data.values())}; no vessel overlap", flush=True)
    protocol = {"fold": args.fold, "recipe": args.recipe, "negative_mode": args.negative_mode,
                "training_groups": len(groups), "max_tokens": 256, "seed": 42,
                "vessel_overlap": 0, "source": key,
                "training_at_token_cap": sum(len(tokens) == 256 for group in groups for tokens in group)}
    if args.prepare_only:
        print(json.dumps(protocol), flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False

    def batch_score(tokens):
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return logits[:, token_yes] - logits[:, token_no]

    def evaluate(label):
        model.eval()
        methods = ("baseline", "blend25") if args.bound_prune else ("baseline", "neural", "blend25")
        correct = {method: {} for method in methods}
        stored = {s: [] for s in eval_data}
        skipped = 0
        with torch.inference_mode():
            for s, rows in eval_data.items():
                for method in correct:
                    correct[method][s] = []
                for row_no, row in enumerate(rows):
                    keep, base = row["keep"], row["base"]
                    bp = base[keep] / base[keep].sum() if args.v8_oof_eval else normalize(base[keep], "prob")
                    # For every possible neural probability vector, a base
                    # probability gap > .25/.75 leaves the winner unchanged.
                    prune = args.bound_prune and (len(keep) == 1 or bp[0] - bp[1] > 1/3 + 1e-6)
                    if prune:
                        new = None
                        skipped += 1
                        blended_index = int(keep[0])
                    else:
                        new = batch_score(row["tokens"]).cpu().numpy()
                        blend = .75 * bp + .25 * normalize(new, "prob")
                        blended_index = int(keep[int(np.argmax(blend))])
                    pred_indices = {"baseline": int(np.argmax(base)), "blend25": blended_index}
                    if "neural" in correct:
                        pred_indices["neural"] = int(keep[int(np.argmax(new))])
                    for method, pred in pred_indices.items():
                        correct[method][s].append(bool(row["candidates"][pred] == row["truth"]))
                    stored[s].append({"keep": keep, "neural_scores": new, "candidates": row["candidates"],
                                      "base_scores": base, "truth": row["truth"], "bound_pruned": bool(prune)})
                    if (row_no + 1) % 200 == 0:
                        print(f"eval {label} {s}: {row_no + 1}/{len(rows)}", flush=True)
        metrics = {}
        for method, ok in correct.items():
            parts = {"clean": float(np.mean(ok["oof"]))}
            for scenario in ("gt_missing", "drop_word", "both"):
                parts[scenario] = float(np.mean(np.array(ok["testlike"])[dfs["testlike"].scenario.eq(scenario).to_numpy()]))
            metrics[method] = {"testlike_hangul": sum(TESTLIKE_MIX[s] * parts[s] for s in parts),
                               "scenarios": parts, "hard_hangul": float(np.mean(ok["hard"]))}
        print(json.dumps({"stage": label, "metrics": metrics}), flush=True)
        out = ROOT / "experiments" / "artifacts" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"stage": label, "sets": stored, "correct": correct, "protocol": protocol,
                     "source": key, "metrics": metrics, "bound_pruned": skipped,
                     "sample_ids": {s: dfs[s].sample_id.tolist() for s in eval_data}},
                    out / f"fold{args.fold}_{label}_scores.joblib", compress=3)
        return metrics

    result = {**protocol, "fold": args.fold, "source": key, "training_groups": len(groups),
              "max_tokens": 256, "negative_count": 7, "training_vessels_overlap": 0,
              "baseline_kind": "v8_saved_oof_probability" if args.v8_oof_eval else "strict_hangul_lgb"}
    if args.eval_adapter:
        model = PeftModel.from_pretrained(model, args.eval_adapter).merge_and_unload()
        result["merged"] = evaluate("merged")
        result["reused_adapter"] = str(args.eval_adapter)
        (ROOT / "experiments" / f"{args.tag}_fold{args.fold}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if not args.skip_pretrained:
        result["pretrained"] = evaluate("pretrained")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                            task_type="CAUSAL_LM", bias="none"))
    if not args.no_checkpoint:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    start = time.perf_counter()
    step = 0
    training_seconds = 0.0
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        ordering = np.random.default_rng(42 + epoch).permutation(len(groups))
        losses = []
        for offset in range(0, len(ordering), args.groups_per_batch):
            ids = ordering[offset:offset + args.groups_per_batch]
            tokens = [seq for i in ids for seq in groups[i]]
            scores = batch_score(tokens).reshape(len(ids), 8)
            loss = F.cross_entropy(scores, torch.zeros(len(ids), dtype=torch.long, device="cuda"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite neural loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step} loss {np.mean(losses[-50:]):.4f} elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            if args.no_checkpoint and step % 100 == 0:
                torch.cuda.empty_cache()
        training_seconds += time.perf_counter() - epoch_start
        out = ROOT / "experiments" / "artifacts" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out / f"fold{args.fold}_adapter")
        if args.merge_eval and epoch == args.epochs - 1:
            model = model.merge_and_unload()
        result[f"epoch{epoch + 1}"] = evaluate(f"epoch{epoch + 1}")
    out = ROOT / "experiments" / "artifacts" / args.tag
    if args.save_merged:
        model.save_pretrained(out / f"fold{args.fold}_merged")
        tokenizer.save_pretrained(out / f"fold{args.fold}_merged")
    result["training_minutes"] = training_seconds / 60
    result["fit_and_eval_minutes"] = (time.perf_counter() - start) / 60
    (ROOT / "experiments" / f"{args.tag}_fold{args.fold}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
