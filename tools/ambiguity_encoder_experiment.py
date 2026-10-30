"""Fixed-budget loss ablation on the exact v12 BGE training examples."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np

from src.common import ROOT, compact_name, is_hangul
from tools.audit_observation_ambiguity import same_metadata, spans

TAG = "encoder_ambiguity_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG
PROTOCOL = ROOT / "experiments" / f"{TAG}_protocol.json"
ARMS = ("none", "targeted", "random")


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp, compress=3)
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint():
    return sha(PROTOCOL)


def check_lock():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["source_hashes"].items():
        assert sha(ROOT / path) == expected, path
    return protocol


def folder(arm):
    assert arm in ARMS
    return ROOT / "experiments/artifacts" / f"{TAG}_{arm}"


def masks(row, candidates, beams, fold, position):
    keys = {compact_name(b) for b in beams}
    keys.discard("")
    common = keys & spans(candidates[0]["vessel_name"])
    target = np.array([False] + [bool(common & spans(c["vessel_name"])) and same_metadata(row, candidates[0], c)
                                for c in candidates[1:]], dtype=bool)
    if not is_hangul(row["vessel_name"]):
        target[:] = False
    # A sham mask changes the same number of negatives in each identical query.
    # This RNG is used only for training sampling, never inference or features.
    rng = np.random.default_rng(20261029 + 10000 * fold + position)
    sham = np.zeros(len(candidates), dtype=bool)
    sham[rng.choice(np.arange(1, len(candidates)), int(target.sum()), replace=False)] = True
    return {"none": np.zeros_like(target), "targeted": target, "random": sham}


def prepare(fold):
    from transformers import AutoTokenizer
    from src.encoder_phonetic_experiment import MODEL_DIR
    from src.encoder_phonetics import encode_pairs
    from src.features import FEATURE_NAMES
    from src.focused_encoder_experiment import training_source
    from src.llm_cache import BeamCache

    check_lock()
    path = FOLDER / f"fold{fold}_data.joblib"
    if path.exists():
        result = joblib.load(path)
        assert result["fingerprint"] == fingerprint()
        return result
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    train, catalog, folds, eligible, frame, matrix, miner, baseline, source = training_source(fold, tokenizer)
    records = frame.to_dict("records")
    catrows = catalog.to_dict("records")
    beams = BeamCache(allow_generate=False)
    columns = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]
    groups, selected, weights = [], [], {a: [] for a in ARMS}
    for i in np.random.default_rng(42).permutation(len(frame)):
        c, block, truth = matrix["candidates"][i], matrix["blocks"][i][:, :len(FEATURE_NAMES)], matrix["truth"][i]
        positive = np.flatnonzero(c == truth)
        if not len(positive) or len(c) < 8:
            continue
        scores = miner.predict(block, num_threads=1)
        strength = np.maximum(block[:, columns[0]], block[:, columns[1]]) - .2 * block[:, columns[2]]
        hard = [int(j) for j in np.argsort(-scores, kind="stable") if j != positive[0]][:5]
        lexical = [int(j) for j in np.argsort(-strength, kind="stable") if j != positive[0] and j not in hard][:2]
        choice = c[[int(positive[0])] + hard + lexical]
        row = records[int(i)]
        names = beams([row["vessel_name"]])[0]
        choices = [catrows[int(j)] for j in choice]
        assert choices[0]["vessel_id"] == row["target_vessel_id"]
        tokens = encode_pairs(tokenizer, row, choices, names)
        position = len(groups)
        assert tokens == baseline["groups"][position], f"Legacy tokens differ: fold{fold} group{position}"
        mask = masks(row, choices, names, fold, position)
        for a in ARMS:
            weights[a].append(mask[a])
        groups.append(tokens)
        selected.append({"row": row, "candidates": choice.copy(), "beams": names})
        if len(groups) == 2400:
            break
    assert len(groups) == len(baseline["groups"]) == 2400
    assert not set(r["row"]["target_vessel_id"] for r in selected) & set(train.loc[folds == fold, "target_vessel_id"])
    weights = {a: np.stack(v) for a, v in weights.items()}
    np.testing.assert_array_equal(weights["targeted"].sum(axis=1), weights["random"].sum(axis=1))
    assert not any(v[:, 0].any() for v in weights.values())
    result = {"groups": groups, "evaluation": baseline["evaluation"], "selected": selected, "masks": weights,
              "fingerprint": fingerprint(), "protocol": {**baseline["protocol"], "experiment": TAG,
              "source": source, "training_token_groups_exactly_legacy": True, "vessel_overlap": 0},
              "token_sha256": hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()}
    dump(result, path)
    summary = {"fold": fold, "groups": len(groups), "affected_groups": int(weights["targeted"].any(axis=1).sum()),
               "relaxed_negatives": int(weights["targeted"].sum()), "random_overlap_with_target": int((weights["targeted"] & weights["random"]).sum()),
               "same_per_query_mask_counts": True, "all_19200_pairs_exactly_legacy": True, "vessel_overlap": 0,
               "fingerprint": result["fingerprint"], "token_sha256": result["token_sha256"]}
    (FOLDER / f"fold{fold}_preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    del matrix, miner, baseline
    gc.collect()
    return result


def seed():
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)


def train(fold, arm, data):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.ambiguity_loss import ranking_loss
    from src.encoder_phonetic_experiment import MODEL_DIR

    check_lock()
    out = folder(arm)
    out.mkdir(parents=True, exist_ok=True)
    merged = out / f"fold{fold}_merged"
    report = out / f"fold{fold}_training.json"
    if report.exists():
        assert json.loads(report.read_text())["fingerprint"] == fingerprint()
        return merged
    checks = json.loads((ROOT / "experiments" / f"{TAG}_setup_checks.json").read_text())
    assert checks["status"] == "PASS" and checks["fingerprint"] == fingerprint()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    seed()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    checkpoints = sorted(p for p in (out / f"fold{fold}_checkpoints").glob("step*/state.pt") if (p.parent / "ready.json").exists())
    completed = 0
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cuda", weights_only=False)
        assert state["fingerprint"] == fingerprint() and state["arm"] == arm and state["fold"] == fold
        model = PeftModel.from_pretrained(model, checkpoints[-1].parent, is_trainable=True)
        completed, initial_hash = state["completed"], state["initial_trainable_sha256"]
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                          target_modules=["query", "key", "value"], modules_to_save=["classifier"], task_type="SEQ_CLS", bias="none"))
        digest = hashlib.sha256()
        for n, p in model.named_parameters():
            if p.requires_grad:
                digest.update(n.encode()); digest.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
        initial_hash = digest.hexdigest()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if checkpoints:
        assert state["trainable_names"] == names
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        del state
    order = np.random.default_rng(42).permutation(len(data["groups"]))
    start = time.perf_counter(); losses = []
    model.train()
    for offset in range(completed * 2, len(order), 2):
        selected = order[offset:offset + 2]
        tokens = [t for i in selected for t in data["groups"][i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(selected), 8).float()
            loss = ranking_loss(scores, data["masks"][arm][selected], weight=.25)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ambiguity loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        step = offset // 2 + 1
        losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"ambiguity fold{fold} {arm} {step}/1200 loss{np.mean(losses[-100:]):.4f} seconds{time.perf_counter()-start:.1f}", flush=True)
            torch.cuda.empty_cache()
        if step % 200 == 0:
            cp = out / f"fold{fold}_checkpoints/step{step:06d}"
            model.save_pretrained(cp)
            state = {"fingerprint": fingerprint(), "arm": arm, "fold": fold, "completed": step,
                     "initial_trainable_sha256": initial_hash, "trainable_names": names,
                     "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            torch.save(state, cp / "state.pt")
            (cp / "ready.json").write_text(json.dumps({"step": step}), encoding="utf-8")
            del state
    model.save_pretrained(out / f"fold{fold}_adapter")
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged); tokenizer.save_pretrained(merged)
    result = {"fingerprint": fingerprint(), "fold": fold, "arm": arm, "groups": len(data["groups"]),
              "steps": len(order) // 2, "resumed_after_step": completed, "seconds_this_segment": time.perf_counter()-start,
              "initial_trainable_sha256": initial_hash, "token_sha256": data["token_sha256"], "finite_loss": True,
              "relaxed_negatives": int(data["masks"][arm].sum()), "mean_loss_this_segment": float(np.mean(losses)) if losses else None}
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, optimizer, enc, scores, loss
    gc.collect(); torch.cuda.empty_cache()
    return merged


def score(fold, arm, data):
    import torch
    from src.encoder_phonetic_experiment import evaluate
    from src.encoder_phonetics import EncoderPhoneticScorer

    check_lock()
    path = folder(arm) / f"fold{fold}_epoch1_scores.joblib"
    if path.exists():
        assert joblib.load(path)["protocol"]["ambiguity_fingerprint"] == fingerprint()
        return
    torch.set_num_threads(4)
    model = EncoderPhoneticScorer(folder(arm) / f"fold{fold}_merged")
    runtime = {**data, "protocol": {**data["protocol"], "ambiguity_arm": arm, "ambiguity_fingerprint": fingerprint()}}
    evaluate(model, runtime, folder(arm), fold)
    del model
    gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prepare", "train", "score", "run"), default="run")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 4])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = p.parse_args()
    for fold in args.folds:
        data = prepare(fold)
        for arm in args.arms:
            if args.stage in ("train", "run"):
                train(fold, arm, data)
            if args.stage in ("score", "run"):
                score(fold, arm, data)


if __name__ == "__main__":
    main()
