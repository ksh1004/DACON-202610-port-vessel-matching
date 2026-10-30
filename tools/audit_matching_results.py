"""Paired vessel bootstrap and fold/few-shot sensitivity for new OOF blends."""
import argparse
import json

import joblib
import numpy as np

from src.blend import blend
from src.common import ROOT
from src.experiment import TESTLIKE_MIX, SCORES
from src.llm_translit import FEW_SHOT
from src.validate import load_folds
from tools.blend_matching import lightweight_lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new", nargs="+", default=["hardcat_only_match_v1"])
    args = parser.parse_args()
    identity = joblib.load(SCORES / f"{args.new[0]}.identity.joblib")
    lab = lightweight_lab(identity)
    ctx = lab.ctx
    names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9", "lgb_base_prior", "lgb_base_prior_a8"]
    raw = [joblib.load(SCORES / f"{n}.joblib") for n in names]
    v7 = blend([raw[i] for i in (0, 3, 4)], [3, 1, 1], "prob")
    v8 = blend(raw, [3, 3, 3, 1, 1], "prob")
    configs = {"v7": v7, "v8": v8}
    for name in args.new:
        scores = joblib.load(SCORES / f"{name}.joblib")
        for s in lab.sizes():
            assert len(scores[s]) == len(lab.cands(s))
            assert all(len(a) == len(b) for a, b in zip(scores[s], lab.cands(s)))
        configs[name + "+v8"] = blend(raw + [scores], [3, 3, 3, 1, 1, 2], "prob")
    contributions = {}
    metrics = {}
    for name, scores in configs.items():
        ok = {s: np.array([c[int(np.argmax(sc))] == target for c, sc, target in zip(
            lab.cands(s), scores[s], ctx.truth if s == "oof" else lab.sets[s].truth)]) for s in scores}
        values = ok["oof"].astype(float)
        values[ctx.hangul] *= TESTLIKE_MIX["clean"]
        tl = lab.sets["testlike"].df
        for scenario in ("gt_missing", "drop_word", "both"):
            mask = tl.scenario.eq(scenario).to_numpy()
            np.add.at(values, tl.origin.to_numpy()[mask], TESTLIKE_MIX[scenario] * ok["testlike"][mask])
        contributions[name] = values
        metrics[name] = lab.evaluate(scores)
    _, gi = np.unique(ctx.truth, return_inverse=True)
    count = int(gi.max()) + 1
    rng = np.random.default_rng(20261027)
    draws = rng.multinomial(count, np.full(count, 1/count), size=2000)
    hc = np.bincount(gi, weights=ctx.hangul, minlength=count)
    nc = np.bincount(gi, weights=~ctx.hangul, minlength=count)
    hd, nd = draws @ hc, draws @ nc
    folds = load_folds(ctx.train)
    shots = ctx.train.loc[ctx.train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"].unique()
    exclude = ~ctx.train.target_vessel_id.isin(shots).to_numpy()

    def weighted(v, mask):
        return float(.7*v[mask & ctx.hangul].mean()+.3*v[mask & ~ctx.hangul].mean())

    results = {}
    for name, values in contributions.items():
        item = {"metrics": metrics[name], "without_shot_vessels": weighted(values, exclude)}
        for reference in ("v7", "v8"):
            delta = values - contributions[reference]
            hs = np.bincount(gi, weights=delta*ctx.hangul, minlength=count)
            ns = np.bincount(gi, weights=delta*~ctx.hangul, minlength=count)
            bootstrap = .7*(draws@hs)/hd + .3*(draws@ns)/nd
            item["vs_" + reference] = {"delta": weighted(delta, np.ones(len(delta), dtype=bool)),
                                      "bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                      "delta_by_fold": {str(k): weighted(delta, folds == k) for k in range(5)},
                                      "delta_without_shots": weighted(delta, exclude)}
        results[name] = item
    output = {"bootstrap_unit": "target_vessel_id", "replicates": 2000,
              "caveat": "Exploratory intervals on reused validation; historical candidate mapper caveat remains.",
              "results": results}
    (ROOT / "experiments" / "matching_result_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
