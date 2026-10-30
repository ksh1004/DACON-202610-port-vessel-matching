"""Combine held-out neural predictions, preserving the v8 English branch."""
import json
from pathlib import Path

import joblib
import numpy as np

from src.blend import blend
from src.common import ROOT, weighted_accuracy, is_hangul
from src.ensemble import normalize
from src.experiment import TESTLIKE_MIX, SCORES
from src.llm_translit import FEW_SHOT
from src.validate import load_folds
from tools.blend_matching import lightweight_lab


def main():
    identity = joblib.load(SCORES / "hardcat_only_match_v1.identity.joblib")
    lab = lightweight_lab(identity)
    names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9", "lgb_base_prior", "lgb_base_prior_a8"]
    base_scores = blend([joblib.load(SCORES / f"{n}.joblib") for n in names], [3, 3, 3, 1, 1], "prob")
    truth = {"oof": lab.ctx.truth, **{s: es.truth for s, es in lab.sets.items()}}
    baseline = {s: np.array([c[int(np.argmax(sc))] for c, sc in zip(lab.cands(s), parts)]) for s, parts in base_scores.items()}
    predictions = {s: values.copy() for s, values in baseline.items()}
    covered = {s: np.zeros(len(values), dtype=bool) for s, values in baseline.items()}
    positions = {s: {sample: i for i, sample in enumerate(identity["sets"][s]["sample_ids"])} for s in baseline}
    folder = ROOT / "experiments" / "artifacts" / "neural_v8_cv"
    complete = []
    fold_metrics = {}
    skipped = 0
    non_hangul_variants = 0
    for k in range(5):
        p = folder / f"fold{k}_{'merged' if k == 4 else 'epoch1'}_scores.joblib"
        if not p.exists():
            continue
        data = joblib.load(p)
        skipped += data.get("bound_pruned", 0)
        complete.append(k)
        fold_metrics[str(k)] = data["metrics"]
        for s, rows in data["sets"].items():
            for sample, row in zip(data["sample_ids"][s], rows):
                i = positions[s][sample]
                assert not covered[s][i], (s, sample)
                np.testing.assert_array_equal(row["candidates"], lab.cands(s)[i])
                assert row["truth"] == truth[s][i]
                name = lab.ctx.train.vessel_name.iloc[i] if s == "oof" else lab.sets[s].df.vessel_name.iloc[i]
                if not is_hangul(name):
                    pred = int(np.argmax(row["base_scores"]))
                    non_hangul_variants += 1
                elif row["bound_pruned"]:
                    pred = int(np.argmax(row["base_scores"]))
                else:
                    keep = row["keep"]
                    bp = row["base_scores"][keep]
                    bp = bp / bp.sum()
                    mixed = .75 * bp + .25 * normalize(row["neural_scores"], "prob")
                    pred = int(keep[int(np.argmax(mixed))])
                predictions[s][i] = row["candidates"][pred]
                covered[s][i] = True
    report = {"complete_folds": complete, "complete": len(complete) == 5, "fold_metrics": fold_metrics,
              "covered_rows": {s: int(mask.sum()) for s, mask in covered.items()}, "bound_pruned": skipped,
              "non_hangul_variants_routed_to_baseline": non_hangul_variants,
              "candidate_caveat": "V8 historical shared candidate retrieval; neural training vessels are held out strictly."}
    if len(complete) < 5:
        (ROOT / "experiments" / "neural_cv_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return
    np.testing.assert_array_equal(covered["oof"], lab.ctx.hangul)
    assert covered["testlike"].all()
    # Evaluation includes every variant of a Hangul-origin record, including
    # variants whose remaining name no longer contains a Hangul syllable.
    np.testing.assert_array_equal(covered["hard"],
                                  lab.ctx.hangul[lab.sets["hard"].df.origin.to_numpy()])

    def metrics(pred):
        ok = {s: pred[s] == truth[s] for s in pred}
        orig = weighted_accuracy(lab.ctx.train.vessel_name, ok["oof"])
        hard = weighted_accuracy(lab.sets["hard"].names, ok["hard"])
        parts = {"clean": orig["hangul"]}
        for scenario in ("gt_missing", "drop_word", "both"):
            parts[scenario] = float(ok["testlike"][lab.sets["testlike"].df.scenario.eq(scenario).to_numpy()].mean())
        values = ok["oof"].astype(float)
        values[lab.ctx.hangul] *= TESTLIKE_MIX["clean"]
        for scenario in ("gt_missing", "drop_word", "both"):
            pick = lab.sets["testlike"].df.scenario.eq(scenario).to_numpy()
            np.add.at(values, lab.sets["testlike"].df.origin.to_numpy()[pick], TESTLIKE_MIX[scenario]*ok["testlike"][pick])
        tl_h = sum(TESTLIKE_MIX[s]*parts[s] for s in parts)
        return {"testlike": .7*tl_h+.3*orig["non_hangul"], "testlike_hangul": tl_h,
                "oof": orig["score"], "oof_hangul": orig["hangul"], "oof_non_hangul": orig["non_hangul"],
                "hard": hard["score"], "hard_hangul": hard["hangul"], "hard_non_hangul": hard["non_hangul"],
                "scenarios": parts}, values

    before, b = metrics(baseline)
    after, a = metrics(predictions)
    delta = a-b
    _, gi = np.unique(lab.ctx.truth, return_inverse=True)
    n = int(gi.max())+1
    draws = np.random.default_rng(20261027).multinomial(n, np.full(n,1/n), size=2000)
    h = lab.ctx.hangul
    hc = np.bincount(gi, weights=h, minlength=n)
    nc = np.bincount(gi, weights=~h, minlength=n)
    hs = np.bincount(gi, weights=delta*h, minlength=n)
    ns = np.bincount(gi, weights=delta*~h, minlength=n)
    bootstrap = .7*(draws@hs)/(draws@hc)+.3*(draws@ns)/(draws@nc)
    shot_ids = lab.ctx.train.loc[lab.ctx.train.vessel_name.isin({ko for ko,_ in FEW_SHOT}),"target_vessel_id"].unique()
    no_shots = ~lab.ctx.train.target_vessel_id.isin(shot_ids).to_numpy()
    folds = load_folds(lab.ctx.train)
    weighted = lambda v,m: float(.7*v[m&h].mean()+.3*v[m&~h].mean())
    report.update(baseline=before, neural=after, delta=after["testlike"]-before["testlike"],
                  bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist(),
                  without_shots={"baseline":weighted(b,no_shots),"neural":weighted(a,no_shots)},
                  delta_by_fold={str(k):weighted(delta,folds==k) for k in range(5)})
    (ROOT / "experiments" / "neural_cv_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    joblib.dump({"predictions":predictions,"baseline":baseline,"identity":identity},
                ROOT / "experiments" / "neural_cv_predictions.joblib",compress=3)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
