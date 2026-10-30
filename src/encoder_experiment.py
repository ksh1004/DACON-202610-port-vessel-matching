"""BGE LoRA ranking on train-fold-only negatives and archived v8 candidates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.augment import augment
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.encoder_pair import encode_pairs
from src.ensemble import normalize
from src.experiment import TESTLIKE_MIX, build_testlike, _predict_blocks
from src.features import FEATURE_NAMES
from src.llm_cache import BeamCache
from src.validate import load_folds


MODEL_DIR = Path("D:/KSH/hf_models/BAAI__bge-reranker-v2-m3")
REFERENCE = ROOT / "experiments/artifacts/neural_evidence_v1"


def prepare(fold, max_groups, tokenizer):
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    h = train.vessel_name.map(is_hangul).to_numpy()
    aug = augment(train, 4, 1, seed=7)
    trdf = pd.concat([train.loc[h & (folds != fold)],
                      aug.loc[h[aug.origin.to_numpy()] & (folds[aug.origin.to_numpy()] != fold)]], ignore_index=True)
    tl = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=h))
    hard = augment(train, 2, 1, seed=2026)
    hard = hard.loc[h[hard.origin.to_numpy()] & (folds[hard.origin.to_numpy()] == fold)].copy()
    hard["sample_id"] += "_hard"
    frames = {"oof": train.loc[h & (folds == fold)], "hard": hard,
              "testlike": tl.loc[folds[tl.origin.to_numpy()] == fold]}
    assert not set(trdf.target_vessel_id) & set(frames["oof"].target_vessel_id)
    source_path = max((ROOT / "experiments/cache").glob(f"strict_han_f{fold}_train_*.joblib"), key=lambda p: p.stat().st_mtime)
    reference_path = REFERENCE / f"fold{fold}_epoch1_scores.joblib"
    miner_path = ROOT / "experiments/artifacts/strict_han_v1" / f"base_fold{fold}.txt"
    digest = hashlib.sha256(f"{fold}:{max_groups}".encode())
    for path in [Path(__file__), ROOT / "src/encoder_pair.py", ROOT / "src/neural_pair.py",
                 ROOT / "src/augment.py", ROOT / "train.csv", ROOT / "vessel_catalog.csv",
                 MODEL_DIR / "tokenizer.json", MODEL_DIR / "codex_revision.json", reference_path, miner_path]:
        digest.update(path.read_bytes())
    digest.update(source_path.name.encode())
    cache = ROOT / "experiments/cache" / f"encoder_groups_f{fold}_{digest.hexdigest()[:12]}.joblib"
    if cache.exists():
        data = joblib.load(cache)
        assert data["fingerprint"] == digest.hexdigest()
        return data
    matrix = joblib.load(source_path)
    assert matrix["sample_ids"] == trdf.sample_id.tolist()
    miner = lgb.Booster(model_file=str(miner_path))
    mining = _predict_blocks(lambda x: miner.predict(x, num_threads=4),
                             [b[:, :len(FEATURE_NAMES)] for b in matrix["blocks"]])
    del miner
    beams = BeamCache(allow_generate=False)
    catrows = catalog.to_dict("records")
    columns = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]
    groups = []
    order = np.random.default_rng(42).permutation(len(trdf))
    for i in order:
        c, block, truth = matrix["candidates"][i], matrix["blocks"][i], matrix["truth"][i]
        positive = np.flatnonzero(c == truth)
        if not len(positive) or len(c) < 8:
            continue
        strength = np.maximum(block[:, columns[0]], block[:, columns[1]])-.2*block[:, columns[2]]
        learned = [int(j) for j in np.argsort(-mining[i], kind="stable") if j != positive[0]][:5]
        lexical = [int(j) for j in np.argsort(-strength, kind="stable") if j != positive[0] and j not in learned][:2]
        chosen = [int(positive[0])] + learned + lexical
        row = trdf.iloc[i].to_dict()
        groups.append(encode_pairs(tokenizer, row, [catrows[j] for j in c[chosen]], beams([row["vessel_name"]])[0]))
        if len(groups) == max_groups:
            break
    del matrix, mining
    reference = joblib.load(reference_path)
    evaluation = {}
    for s, frame in frames.items():
        assert frame.sample_id.tolist() == reference["sample_ids"][s]
        evaluation[s] = []
        for row, old in zip(frame.to_dict("records"), reference["sets"][s]):
            choices = [catrows[j] for j in old["candidates"][old["keep"]]]
            tokens = encode_pairs(tokenizer, row, choices, beams([row["vessel_name"]])[0])
            evaluation[s].append({"row": row, "reference": old, "tokens": tokens})
    protocol = {"fold": fold, "model": "BAAI/bge-reranker-v2-m3", "source_revision": json.loads((MODEL_DIR / "codex_revision.json").read_text()),
                "training_groups": len(groups), "max_tokens": 256, "seed": 42, "epochs": 1,
                "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4, "groups_per_batch": 2,
                "negative_mode": "five train-only LGB hard plus two lexical", "vessel_overlap": 0,
                "train_cache": source_path.name, "training_at_cap": sum(len(t) == 256 for g in groups for t in g),
                "candidate_caveat": "Historical v8 shared retrieval mapper; training retrieval/mapper held out strictly."}
    data = {"groups": groups, "evaluation": evaluation, "protocol": protocol, "fingerprint": digest.hexdigest()}
    joblib.dump(data, cache, compress=3)
    print(f"Prepared {cache}", flush=True)
    return data


def evaluate(scorer, data, folder, fold):
    methods = ("baseline", "reference_v11", "encoder25", "equal_ensemble")
    correct = {m: {} for m in methods}
    stored, sample_ids = {}, {}
    skipped = 0
    for s, rows in data["evaluation"].items():
        stored[s], sample_ids[s] = [], []
        for m in methods:
            correct[m][s] = []
        for number, item in enumerate(rows):
            row, reference = item["row"], item["reference"]
            c, base, keep = reference["candidates"], reference["base_scores"], reference["keep"]
            bp = base[keep]/base[keep].sum()
            prune = len(keep) < 2 or bp[0]-bp[1] > 1/3+1e-6
            winners = {m: int(np.argmax(base)) for m in methods}
            neural = None
            if is_hangul(row["vessel_name"]) and not prune:
                with torch.inference_mode():
                    enc = scorer.tokenizer.pad({"input_ids": item["tokens"]}, padding=True,
                                                pad_to_multiple_of=8, return_tensors="pt").to("cuda")
                    neural = scorer.model(**enc).logits.reshape(-1).float().cpu().numpy()
                old_prob, new_prob = normalize(reference["neural_scores"], "prob"), normalize(neural, "prob")
                winners["reference_v11"] = int(keep[int(np.argmax(.75*bp+.25*old_prob))])
                winners["encoder25"] = int(keep[int(np.argmax(.75*bp+.25*new_prob))])
                average = .75*bp+.125*old_prob.astype(float)+.125*new_prob.astype(float)
                winners["equal_ensemble"] = int(keep[int(np.argmax(average))])
            else:
                skipped += 1
            for m, j in winners.items():
                correct[m][s].append(bool(c[j] == reference["truth"]))
            stored[s].append({"candidates": c, "base_scores": base, "keep": keep,
                              "truth": reference["truth"], "neural_scores": neural,
                              "bound_pruned": prune, "routed_non_hangul": not is_hangul(row["vessel_name"])})
            sample_ids[s].append(row["sample_id"])
            if (number+1) % 200 == 0:
                print(f"encoder eval {s}: {number+1}/{len(rows)}", flush=True)
    metrics = {}
    scenarios = [r["row"].get("scenario") for r in data["evaluation"]["testlike"]]
    for method in methods:
        parts = {"clean": float(np.mean(correct[method]["oof"]))}
        for scenario in ("gt_missing", "drop_word", "both"):
            parts[scenario] = float(np.array(correct[method]["testlike"])[np.array(scenarios) == scenario].mean())
        metrics[method] = {"testlike_hangul": sum(TESTLIKE_MIX[s]*v for s, v in parts.items()),
                           "hard_hangul": float(np.mean(correct[method]["hard"])), "scenarios": parts}
    joblib.dump({"sets": stored, "sample_ids": sample_ids, "metrics": metrics,
                 "protocol": data["protocol"], "bound_pruned": skipped}, folder / f"fold{fold}_epoch1_scores.joblib", compress=3)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--tag", default="encoder_bge_v1")
    parser.add_argument("--max-groups", type=int, default=2400)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, args.max_groups, tokenizer)
    print(json.dumps(data["protocol"]), flush=True)
    if args.prepare_only:
        return
    from src.encoder_pair import EncoderPairScorer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=["query", "key", "value"], modules_to_save=["classifier"],
                                            task_type="SEQ_CLS", bias="none"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    order = np.random.default_rng(42).permutation(len(data["groups"]))
    start = time.perf_counter()
    losses = []
    model.train()
    for step, offset in enumerate(range(0, len(order), 2), 1):
        selected = order[offset:offset+2]
        tokens = [part for i in selected for part in data["groups"][i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(selected), 8).float()
            loss = F.cross_entropy(scores, torch.zeros(len(selected), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite encoder ranking loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"encoder step {step}, loss {np.mean(losses[-100:]):.4f}, elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            torch.cuda.empty_cache()
    folder = ROOT / "experiments/artifacts" / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder / f"fold{args.fold}_adapter")
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    merged = folder / f"fold{args.fold}_merged"
    model.save_pretrained(merged); tokenizer.save_pretrained(merged)
    del model, optimizer, enc, scores, loss
    gc.collect(); torch.cuda.empty_cache()
    scorer = EncoderPairScorer(merged)
    metrics = evaluate(scorer, data, folder, args.fold)
    result = {"protocol": data["protocol"], "metrics": metrics,
              "train_eval_minutes": (time.perf_counter()-start)/60,
              "deployment_precision": "merged, cast to BF16, saved and reloaded before evaluation"}
    (ROOT / "experiments" / f"{args.tag}_fold{args.fold}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
