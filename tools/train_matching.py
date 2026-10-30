"""Fit final matching models after selecting a validated experiment configuration."""
import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRanker, Pool

from src.common import ROOT
from src.experiment import Lab, _predict_blocks, _stack
from src.feature_experiment import extend, fit_lgb
from src.features import FEATURE_NAMES
from src.match_features import EXTRA_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--hangul-weight", type=float, default=1.0)
    parser.add_argument("--mode", choices=["cat_only", "blend"], default="cat_only")
    parser.add_argument("--base-artifacts", type=Path)
    parser.add_argument("--blend-weight", type=float, default=2.0)
    args = parser.parse_args()
    lab = Lab()
    identity = extend(lab)
    ctx = lab.ctx
    X, y = ctx.blocks + ctx.aug_blocks, ctx.labels + ctx.aug_labels
    names = FEATURE_NAMES + EXTRA_FEATURE_NAMES
    primary = fit_lgb(X, y, names, hangul_weight=args.hangul_weight)
    raw = _predict_blocks(lambda b: primary.predict(b, num_threads=6), X)
    picks = [np.unique(np.r_[np.argsort(-sc, kind="stable")[:20], np.flatnonzero(t)]) for sc, t in zip(raw, y)]
    xs, ys, group = _stack([b[p] for b, p in zip(X, picks)], [t[p] for t, p in zip(y, picks)])
    weights = np.where(xs[:, FEATURE_NAMES.index("q_hangul")] > 0, args.hangul_weight, 1.0)
    second = CatBoostRanker(loss_function="YetiRank", iterations=400, learning_rate=.1, depth=6,
                           random_seed=42, thread_count=8, verbose=False, metric_period=1000)
    second.fit(Pool(xs, ys, group_id=np.repeat(np.arange(len(group)), group), group_weight=weights))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    primary.save_model(str(out / "matching_primary.txt"))
    second.save_model(str(out / "matching_second.cbm"))
    joblib.dump({"type": ctx.builder.type_mapper, "flag": ctx.builder.flag_mapper,
                 "feature_names": FEATURE_NAMES}, out / "mappers.joblib")
    spec = {"feature_names": names, "primary": "matching_primary.txt", "second": "matching_second.cbm",
            "top_k": 20, "mode": args.mode, "hangul_weight": args.hangul_weight,
            "training_fingerprint": identity["fingerprint"], "lookup": False}
    spec["source_hashes"] = {
        filename: hashlib.sha256((ROOT / "src" / filename).read_bytes()).hexdigest()
        for filename in ("match_features.py", "features.py", "retrieval.py")}
    if args.base_artifacts:
        base = json.loads((args.base_artifacts / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["ensemble.json"] + [m["file"] for m in base["members"]]:
            shutil.copy(args.base_artifacts / name, out / name)
        spec.update(base_ensemble=True, blend_weight=args.blend_weight)
    (out / "matching.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved matching ranker: {out}", flush=True)
    del lab, X, y, xs, ys
    gc.collect()


if __name__ == "__main__":
    main()
