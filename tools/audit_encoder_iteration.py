"""Compare a new held-out pair scorer with archived v8 and v11 predictions."""
import argparse
import json

import joblib
import numpy as np

from src.blend import blend
from src.common import ROOT, is_hangul
from src.ensemble import normalize
from src.experiment import SCORES, TESTLIKE_MIX
from src.llm_translit import FEW_SHOT
from src.validate import load_folds
from src.pair_views import mix_views
from tools.blend_matching import lightweight_lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="encoder_bge_v1")
    args = parser.parse_args()
    identity = joblib.load(SCORES / "hardcat_only_match_v1.identity.joblib")
    lab = lightweight_lab(identity)
    members = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9", "lgb_base_prior", "lgb_base_prior_a8"]
    scores = blend([joblib.load(SCORES / f"{m}.joblib") for m in members], [3, 3, 3, 1, 1], "prob")
    truth = {"oof": lab.ctx.truth, **{s: e.truth for s, e in lab.sets.items()}}
    base = {s: np.array([c[int(np.argmax(p))] for c, p in zip(lab.cands(s), parts)]) for s, parts in scores.items()}
    ref = joblib.load(ROOT / "experiments/neural_evidence_v1_predictions.joblib")["predictions"]
    pred = {s: p.copy() for s, p in base.items()}
    ensemble = {s: p.copy() for s, p in base.items()}
    positions = {s: {sample: i for i, sample in enumerate(identity["sets"][s]["sample_ids"])} for s in base}
    covered = {s: np.zeros(len(p), dtype=bool) for s, p in base.items()}
    complete, protocols = [], {}
    for k in range(5):
        path = ROOT / "experiments/artifacts" / args.tag / f"fold{k}_epoch1_scores.joblib"
        if not path.exists():
            continue
        data = joblib.load(path)
        reference_path = ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{k}_epoch1_scores.joblib"
        reference_data = joblib.load(reference_path)
        complete.append(k)
        protocols[str(k)] = data.get("protocol", {})
        for s, rows in data["sets"].items():
            assert data["sample_ids"][s] == reference_data["sample_ids"][s]
            for sample, row, old_row in zip(data["sample_ids"][s], rows, reference_data["sets"][s]):
                i = positions[s][sample]
                assert not covered[s][i]
                np.testing.assert_array_equal(row["candidates"], lab.cands(s)[i])
                expected_base = scores[s][i].astype(float)
                expected_base /= expected_base.sum()
                np.testing.assert_allclose(row["base_scores"], expected_base, rtol=1e-12, atol=1e-12)
                assert row["truth"] == truth[s][i]
                name = lab.ctx.train.vessel_name.iloc[i] if s == "oof" else lab.sets[s].df.vessel_name.iloc[i]
                j = int(np.argmax(row["base_scores"]))
                ensemble_j = j
                if is_hangul(name) and not row["bound_pruned"]:
                    keep = row["keep"]
                    np.testing.assert_array_equal(keep, old_row["keep"])
                    assert not old_row["bound_pruned"]
                    bp = row["base_scores"][keep]
                    mixed = .75*bp/bp.sum() + .25*normalize(row["neural_scores"], "prob")
                    j = int(keep[int(np.argmax(mixed))])
                    average = mix_views(bp/bp.sum(), [old_row["neural_scores"], row["neural_scores"]], [1, 1])
                    ensemble_j = int(keep[int(np.argmax(average))])
                pred[s][i] = row["candidates"][j]
                ensemble[s][i] = row["candidates"][ensemble_j]
                covered[s][i] = True
    if not complete:
        raise RuntimeError("No complete fold scores")
    folds = load_folds(lab.ctx.train)
    h = lab.ctx.hangul
    scope = np.isin(folds, complete)
    np.testing.assert_array_equal(covered["oof"], scope & h)
    for s in ("hard", "testlike"):
        origins = lab.sets[s].df.origin.to_numpy()
        np.testing.assert_array_equal(covered[s], scope[origins] & h[origins])

    def metrics(prediction, selected=scope):
        ok = {s: prediction[s] == truth[s] for s in truth}
        values = ok["oof"].astype(float)
        values[h] *= TESTLIKE_MIX["clean"]
        parts = {"clean": float(ok["oof"][selected & h].mean())}
        tl = lab.sets["testlike"].df
        for scenario in ("gt_missing", "drop_word", "both"):
            pick = tl.scenario.eq(scenario).to_numpy()
            origins = tl.origin.to_numpy()[pick]
            parts[scenario] = float(ok["testlike"][pick][selected[origins]].mean())
            np.add.at(values, origins, TESTLIKE_MIX[scenario] * ok["testlike"][pick])
        hg = sum(TESTLIKE_MIX[s]*v for s, v in parts.items())
        ns = float(ok["oof"][selected & ~h].mean())
        hard_mask = selected[lab.sets["hard"].df.origin.to_numpy()]
        hard_h = lab.sets["hard"].names.map(is_hangul).to_numpy()
        hard = .7*ok["hard"][hard_mask & hard_h].mean()+.3*ok["hard"][hard_mask & ~hard_h].mean()
        return {"testlike": .7*hg+.3*ns, "testlike_hangul": hg,
                "hard": float(hard), "hard_hangul": float(ok["hard"][hard_mask & hard_h].mean()),
                "oof": float(.7*parts["clean"]+.3*ns), "scenarios": parts}, values

    bmet, b = metrics(base)
    rmet, r = metrics(ref)
    pmet, p = metrics(pred)
    emet, e = metrics(ensemble)
    delta = p-r
    _, group = np.unique(lab.ctx.truth[scope], return_inverse=True)
    n = int(group.max())+1
    draws = np.random.default_rng(20261028).multinomial(n, np.full(n, 1/n), size=2000)
    hc = np.bincount(group, weights=h[scope], minlength=n)
    nc = np.bincount(group, weights=~h[scope], minlength=n)
    dh = np.bincount(group, weights=delta[scope]*h[scope], minlength=n)
    dn = np.bincount(group, weights=delta[scope]*(~h[scope]), minlength=n)
    boot = .7*(draws@dh)/(draws@hc)+.3*(draws@dn)/(draws@nc)
    shot_ids = lab.ctx.train.loc[lab.ctx.train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"].unique()
    no_shots = scope & ~lab.ctx.train.target_vessel_id.isin(shot_ids).to_numpy()
    fold_delta = {str(k): metrics(pred, folds == k)[0]["testlike"]-metrics(ref, folds == k)[0]["testlike"] for k in complete}
    lost = int(np.sum(scope & (ref["oof"] == truth["oof"]) & (pred["oof"] != truth["oof"])))
    report = {"tag": args.tag, "complete_folds": complete, "complete": len(complete) == 5,
              "baseline_v8": bmet, "reference_v11": rmet, "candidate": pmet,
              "delta_vs_v11": pmet["testlike"]-rmet["testlike"],
              "bootstrap_95_vs_v11": np.quantile(boot, [.025, .975]).tolist(),
              "delta_by_fold_vs_v11": fold_delta, "clean_lost_rows_vs_v11": lost,
              "without_shots": {"reference_v11": metrics(ref, no_shots)[0], "candidate": metrics(pred, no_shots)[0]},
              "protocols": protocols,
              "candidate_caveat": "Exploratory reused validation; historical v8 retrieval mapper remains shared."}
    gates = [set(complete) == {1, 2}, report["delta_vs_v11"] >= .001,
             min(fold_delta.values()) >= -.002,
             pmet["hard_hangul"]-rmet["hard_hangul"] >= -.002, lost <= 1]
    report["screen_gate_pass"] = bool(all(gates))
    ensemble_delta = e-r
    eh = np.bincount(group, weights=ensemble_delta[scope]*h[scope], minlength=n)
    en = np.bincount(group, weights=ensemble_delta[scope]*(~h[scope]), minlength=n)
    eboot = .7*(draws@eh)/(draws@hc)+.3*(draws@en)/(draws@nc)
    efold = {str(k): metrics(ensemble, folds == k)[0]["testlike"]-metrics(ref, folds == k)[0]["testlike"] for k in complete}
    elost = int(np.sum(scope & (ref["oof"] == truth["oof"]) & (ensemble["oof"] != truth["oof"])))
    egates = [set(complete) == {1, 2}, emet["testlike"]-rmet["testlike"] >= .001,
              min(efold.values()) >= -.002, emet["hard_hangul"]-rmet["hard_hangul"] >= -.002, elost <= 1]
    report.update(equal_neural_ensemble=emet, ensemble_delta_vs_v11=emet["testlike"]-rmet["testlike"],
                  ensemble_bootstrap_95_vs_v11=np.quantile(eboot, [.025, .975]).tolist(),
                  ensemble_delta_by_fold_vs_v11=efold, ensemble_clean_lost_rows_vs_v11=elost,
                  ensemble_screen_gate_pass=bool(all(egates)))
    report["without_shots"]["ensemble"] = metrics(ensemble, no_shots)[0]
    out = ROOT / "experiments" / f"{args.tag}_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": pred, "ensemble": ensemble, "reference": ref, "baseline": base, "identity": identity, "covered": covered},
                ROOT / "experiments" / f"{args.tag}_predictions.joblib", compress=3)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
