"""Train a phonetic character channel and score frozen v12 held-out candidates."""
import argparse
import hashlib
import json
import time

import joblib
import numpy as np

from src.common import ROOT, is_hangul, load_train, load_catalog
from src.learned_alignment import LearnedAlignment, observed
from src.encoder_phonetic_experiment import prepare, MODEL_DIR
from src.validate import load_folds
from transformers import AutoTokenizer


TAG = "learned_alignment_v1"


def train_fold(k):
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    h = train.vessel_name.map(is_hangul).to_numpy()
    left = train.loc[h & (folds != k)]
    right = catalog.set_index("vessel_id").vessel_name
    assert not set(left.target_vessel_id) & set(train.loc[folds == k].target_vessel_id)
    pairs = sorted(set((r.vessel_name, right[r.target_vessel_id]) for r in left.itertuples()))
    model = LearnedAlignment().fit(pairs, iterations=4)
    return model, {"fold": k, "unique_pairs": len(pairs), "training_entities": int(left.target_vessel_id.nunique()),
                   "validation_entity_overlap": 0, "synthetic_catalog_pairs": 0,
                   "pair_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "experiments/learned_alignment_protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    k, start = args.fold, time.perf_counter()
    folder = ROOT / "experiments/artifacts" / TAG
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"fold{k}_epoch1_scores.joblib"
    assert not out.exists()
    model, training = train_fold(k)
    np.save(folder / f"fold{k}_costs.npy", model.costs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(k, 2400, tokenizer)
    catalog = load_catalog()
    old = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{k}_epoch1_scores.joblib")
    calls = 0
    for s, items in data["evaluation"].items():
        for oldrow, item in zip(old["sets"][s], items):
            if oldrow["bound_pruned"] or not is_hangul(item["row"]["vessel_name"]):
                oldrow["neural_scores"] = None
                continue
            choices = oldrow["candidates"][oldrow["keep"]]
            oldrow["neural_scores"] = np.array([model.score(item["row"]["vessel_name"], catalog.vessel_name.iloc[j])
                                                for j in choices], dtype=float)
            assert np.isfinite(oldrow["neural_scores"]).all()
            calls += 1
        print(f"Alignment fold{k} {s} scored, memo {model.score.cache_info()}", flush=True)
    training.update(seconds=time.perf_counter()-start, active_rows=calls, registered=protocol)
    old["protocol"] = training
    old.pop("metrics", None)
    joblib.dump(old, out, compress=3)
    (ROOT / "experiments" / f"{TAG}_fold{k}.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    print(json.dumps(training, indent=2), flush=True)


if __name__ == "__main__":
    main()
