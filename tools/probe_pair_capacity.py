"""Optimistic oracle for a fixed probability blend; never a learned score."""
import json

import joblib
import numpy as np

from src.blend import blend
from src.common import ROOT, is_hangul
from src.experiment import SCORES
from tools.blend_matching import lightweight_lab


def main():
    identity = joblib.load(SCORES / "hardcat_only_match_v1.identity.joblib")
    lab = lightweight_lab(identity)
    names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9", "lgb_base_prior", "lgb_base_prior_a8"]
    scores = blend([joblib.load(SCORES / f"{n}.joblib") for n in names], [3, 3, 3, 1, 1], "prob")
    truth = {"oof": lab.ctx.truth, **{s: e.truth for s, e in lab.sets.items()}}
    report = {"caveat": "Optimistic truth-aware selector, not a model. Non-Hangul input predictions stay v8. Historical candidates shared.", "weights": {}}
    for weight in (.25, .5, 1.):
        perfect = {}
        for s, parts in scores.items():
            row_names = lab.ctx.train.vessel_name if s == "oof" else lab.sets[s].df.vessel_name
            perfect[s] = []
            for c, raw, target, name in zip(lab.cands(s), parts, truth[s], row_names):
                if not is_hangul(name):
                    perfect[s].append(raw)
                    continue
                p = raw.astype(float)
                keep = np.argsort(-p, kind="stable")[:20]
                bp = p[keep]/p[keep].sum()
                mixed = (1-weight)*bp
                hit = np.flatnonzero(c[keep] == target)
                mixed[int(hit[0]) if len(hit) else 0] += weight
                full = np.zeros(len(c))
                full[keep] = mixed
                perfect[s].append(full)
        report["weights"][str(weight)] = lab.evaluate(perfect)
    (ROOT / "experiments/pair_blend_oracle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
