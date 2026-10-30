"""Read-only: error breakdown after train-vessel exclusion (v8 OOF scores).

    python -m tools.probe_exclusion_errors
"""
import collections
import json

import joblib
import numpy as np

from src.blend import blend
from src.experiment import SCORES, Lab
from tools.probe_exclude_train_vessels import MEMBERS


def main() -> None:
    lab = Lab()
    ctx = lab.ctx
    scores = blend([joblib.load(SCORES / f"{m}.joblib") for m in MEMBERS], list(MEMBERS.values()), "prob")
    seen = {k: np.isin(np.arange(len(ctx.retriever.ids)), ctx.truth[ctx.folds != k]) for k in sorted(set(ctx.folds))}
    tl = lab.sets["testlike"]
    report = {}
    for scen in ("gt_missing", "drop_word", "both"):
        idx = np.flatnonzero((tl.df["scenario"] == scen).to_numpy())
        c = collections.Counter()
        n_cand = []
        for i in idx:
            cand, sc = np.asarray(tl.cands[i]), np.asarray(scores["testlike"][i], dtype=float)
            k = ctx.folds[tl.origin[i]]
            keep = ~seen[k][cand]
            truth = tl.truth[i]
            c["rows"] += 1
            if truth not in cand:
                c["truth_not_in_candidates"] += 1
                continue
            cand, sc = cand[keep], sc[keep]
            n_cand.append(len(cand))
            rank = int((sc > sc[cand == truth][0]).sum()) + 1
            c["rank1" if rank == 1 else "rank2" if rank == 2 else "rank3_5" if rank <= 5
              else "rank6_20" if rank <= 20 else "rank_gt20"] += 1
        report[scen] = {**dict(c), "median_candidates_after_exclusion": float(np.median(n_cand))}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
