"""Saved-tree and row/candidate ordering checks using frozen pair likelihoods."""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT
from tools import reverse_ranker_experiment as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    if args.partial:
        from tools import reverse_partial_experiment as module
        folder, feature_fn, suffix = module.FOLDER, module.features, "partial"
    else:
        folder, feature_fn, suffix = base.FOLDER, base.features, "reverse"
    cache = joblib.load(folder / "pair_scores.joblib")
    reports = []
    for fold in (0, 2):
        data = base.prepare(fold)
        saved = joblib.load(folder / f"fold{fold}_results.joblib")
        model = lgb.Booster(model_file=str(folder / f"fold{fold}_{suffix}.txt"))
        matrices, expected = [], []
        permutation_checks = 0
        for s, groups in data["evaluation"].items():
            for g, result in zip(groups, saved["sets"][s]):
                if not g["actual_hangul"]:
                    continue
                x = feature_fn(g, cache["scores"])
                matrices.append(x)
                expected.append(result["scores"])
                if permutation_checks < 32:
                    perm = np.random.default_rng(1000+permutation_checks).permutation(len(g["names"]))
                    changed = {**g, "names": [g["names"][i] for i in perm], "raw": g["raw"][perm]}
                    np.testing.assert_array_equal(feature_fn(changed, cache["scores"]), x[perm])
                    permutation_checks += 1
        x = np.vstack(matrices)
        expected = np.concatenate(expected)
        actual = model.predict(x, num_threads=4)
        np.testing.assert_array_equal(actual, expected)
        perm = np.random.default_rng(20261028).permutation(len(x))
        shuffled = model.predict(x[perm], num_threads=4)
        np.testing.assert_array_equal(shuffled, actual[perm])
        single = model.predict(x[:1], num_threads=1)
        np.testing.assert_array_equal(single, actual[:1])
        reports.append({"fold": fold, "candidate_rows": len(x), "exact_save_reload": True,
                        "exact_reordered_groups": True, "exact_single": True,
                        "candidate_permutation_groups": permutation_checks})
    report = {"status": "PASS", "partial": args.partial, "folds": reports,
              "scope": "Frozen likelihood cache plus final trees; not an end-to-end submission/package test"}
    (ROOT / "experiments" / f"{folder.name}_head_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
