"""Probe: train the ranker on test-like candidate groups (other train vessels removed).

At test time every train vessel is excluded, so the ranker only ever compares the truth with
non-train vessels. Here each training group keeps its own truth and drops candidates that are
other training-fold vessels. Eval groups use the same exclusion as tools/probe_exclude_train_vessels.

    python -m tools.probe_exclusion_training lgb_base catg_400
"""
import json
import sys

import joblib
import numpy as np

from src.experiment import LEARNERS, SCORES, Lab, _predict_blocks


def main() -> None:
    names = sys.argv[1:] or ["lgb_base"]
    lab = Lab()
    ctx = lab.ctx
    n_cat = len(ctx.retriever.ids)
    seen = {k: np.isin(np.arange(n_cat), ctx.truth[ctx.folds != k]) for k in sorted(set(ctx.folds))}

    def masked(scores):
        out = {}
        for s, rows in scores.items():
            origin = np.arange(len(ctx.train)) if s == "oof" else lab.sets[s].origin
            res = []
            for i, (cand, sc) in enumerate(zip(lab.cands(s), rows)):
                if sc is None or len(sc) == 0:
                    res.append(sc)
                    continue
                bad = seen[ctx.folds[origin[i]]][np.asarray(cand)]
                res.append(np.where(bad, -np.inf, sc) if bad.any() and not bad.all() else sc)
            out[s] = res
        return out

    results = {}
    for name in names:
        for mode in ("normal", "filtered"):
            scores = {s: [None] * n for s, n in lab.sizes().items()}
            for k, X, y, evals in lab.folds():
                if mode == "filtered":
                    tr_idx = np.flatnonzero(ctx.folds != k)
                    pick = np.flatnonzero(np.isin(ctx.aug_origin, tr_idx))
                    cands = [ctx.cands[i] for i in tr_idx] + [ctx.aug_cands[i] for i in pick]
                    assert len(cands) == len(X)
                    X2, y2 = [], []
                    for c, xb, yb in zip(cands, X, y):
                        keep = ~seen[k][np.asarray(c)] | (yb > 0)
                        X2.append(xb[keep])
                        y2.append(yb[keep])
                    X, y = X2, y2
                predict = LEARNERS[name]()(X, y)
                for s, (idx, blocks) in evals.items():
                    for i, sc in zip(idx, _predict_blocks(predict, blocks)):
                        scores[s][i] = sc
                print(f"{name} {mode} fold {k} done", flush=True)
            joblib.dump(scores, SCORES / f"excl_{name}_{mode}.joblib")
            results[f"{name}_{mode}"] = {"plain": lab.evaluate(scores), "excluded": lab.evaluate(masked(scores))}
            print(json.dumps({f"{name}_{mode}": results[f"{name}_{mode}"]["excluded"]}), flush=True)
    with open("experiments/exclusion_training_probe.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
