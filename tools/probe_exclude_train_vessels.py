"""Read-only probe: how much would excluding train vessels from candidates help?

Assumption tested: test vessels never appear in train (supported by v13 = v12 to 10 decimals).
In vessel-grouped CV this is simulated exactly: a validation row's candidates drop every
vessel that appears in the *training folds*. Uses saved v8-style OOF scores; no fitting.

    python -m tools.probe_exclude_train_vessels
"""
import json

import joblib
import numpy as np

from src.blend import blend
from src.experiment import SCORES, Lab

MEMBERS = {"cat_yeti400_prior": 3, "cat_yeti400_prior_a8": 3, "cat_yeti400_prior_a9": 3,
           "lgb_base_prior": 1, "lgb_base_prior_a8": 1}


def main() -> None:
    lab = Lab()
    ctx = lab.ctx
    scores = blend([joblib.load(SCORES / f"{m}.joblib") for m in MEMBERS], list(MEMBERS.values()), "prob")
    base = lab.evaluate(scores)

    fold_vessels = {k: set(ctx.truth[ctx.folds != k].tolist()) for k in sorted(set(ctx.folds))}
    masked = {}
    stats = {}
    for s, rows in scores.items():
        origin = np.arange(len(ctx.train)) if s == "oof" else lab.sets[s].origin
        out, changed = [], 0
        for i, (cand, sc) in enumerate(zip(lab.cands(s), rows)):
            if sc is None or len(sc) == 0:
                out.append(sc)
                continue
            seen = fold_vessels[ctx.folds[origin[i]]]
            bad = np.array([c in seen for c in cand], dtype=bool)
            sc = np.where(bad, -1e9, sc) if not bad.all() else sc
            changed += int(np.argmax(sc) != np.argmax(rows[i]))
            out.append(sc)
        masked[s] = out
        stats[s] = changed
    excl = lab.evaluate(masked)
    print(json.dumps({"base": base, "exclude_train_vessels": excl, "changed_top1": stats}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
