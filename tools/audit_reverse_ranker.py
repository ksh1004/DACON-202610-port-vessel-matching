"""Paired v12 comparison for the preregistered direct reverse-name heads."""
import json

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.experiment import TESTLIKE_MIX
from src.llm_translit import FEW_SHOT
from src.validate import load_folds
from tools.blend_matching import lightweight_lab
from tools.reverse_ranker_experiment import FOLDER, PROTOCOL


def audit(artifact_folder=None, protocol_file=None):
    folder = FOLDER if artifact_folder is None else artifact_folder
    protocol = json.loads((PROTOCOL if protocol_file is None else protocol_file).read_text())
    archived = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archived["identity"])
    reference = archived["ensemble"]
    truth = {"oof": lab.ctx.truth, **{s: d.truth for s, d in lab.sets.items()}}
    positions = {s: {sid: i for i, sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in truth}
    predictions = {recipe: {m: {s: p.copy() for s, p in reference.items()}
                            for m in protocol["contenders"]} for recipe in ("sets", "controls")}
    seen = {s: np.zeros(len(v), dtype=bool) for s, v in reference.items()}
    complete, outside = [], {s: 0 for s in truth}
    for fold in range(5):
        path = folder / f"fold{fold}_results.joblib"
        if not path.exists():
            continue
        data = joblib.load(path)
        complete.append(fold)
        for recipe in predictions:
            for s, rows in data[recipe].items():
                for row in rows:
                    i = positions[s][row["sample_id"]]
                    assert row["truth"] == truth[s][i]
                    if recipe == "sets":
                        assert not seen[s][i]
                        seen[s][i] = True
                        outside[s] += int(not row["truth_in_top20"])
                    for m, pred in row["predictions"].items():
                        if pred is not None:
                            predictions[recipe][m][s][i] = pred
    if not complete:
        raise RuntimeError("No completed fold results")
    folds, h = load_folds(lab.ctx.train), lab.ctx.hangul
    scope = np.isin(folds, complete)
    np.testing.assert_array_equal(seen["oof"], scope & h)
    for s in ("hard", "testlike"):
        origins = lab.sets[s].df.origin.to_numpy()
        np.testing.assert_array_equal(seen[s], scope[origins] & h[origins])

    def metrics(p, mask=scope):
        ok = {s: p[s] == truth[s] for s in truth}
        parts = {"clean": float(ok["oof"][mask & h].mean())}
        values = ok["oof"].astype(float)
        values[h] *= TESTLIKE_MIX["clean"]
        tl = lab.sets["testlike"].df
        for scenario in ("gt_missing", "drop_word", "both"):
            pick = tl.scenario.eq(scenario).to_numpy()
            origins = tl.origin.to_numpy()[pick]
            parts[scenario] = float(ok["testlike"][pick][mask[origins]].mean())
            np.add.at(values, origins, TESTLIKE_MIX[scenario]*ok["testlike"][pick])
        han = sum(TESTLIKE_MIX[s]*v for s, v in parts.items())
        english = float(ok["oof"][mask & ~h].mean())
        hard = lab.sets["hard"]
        hm = mask[hard.df.origin.to_numpy()]
        hh = hard.names.map(is_hangul).to_numpy()
        hard_h = float(ok["hard"][hm & hh].mean())
        return {"testlike": .7*han+.3*english, "testlike_hangul": han,
                "hard": .7*hard_h+.3*float(ok["hard"][hm & ~hh].mean()),
                "hard_hangul": hard_h, "oof": .7*parts["clean"]+.3*english,
                "scenarios": parts}, values

    rm, rv = metrics(reference)
    shots = lab.ctx.train.loc[lab.ctx.train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"].unique()
    no_shots = scope & ~lab.ctx.train.target_vessel_id.isin(shots).to_numpy()
    _, group = np.unique(lab.ctx.truth[scope], return_inverse=True)
    count = int(group.max())+1
    draws = np.random.default_rng(20261028).multinomial(count, np.full(count, 1/count), size=2000)
    hc = np.bincount(group, weights=h[scope], minlength=count)
    nc = np.bincount(group, weights=~h[scope], minlength=count)
    reports = {}
    for method in protocol["contenders"]:
        pred, control = predictions["sets"][method], predictions["controls"][method]
        pm, pv = metrics(pred)
        cm, _ = metrics(control)
        difference = pv-rv
        dh = np.bincount(group, weights=difference[scope]*h[scope], minlength=count)
        dn = np.bincount(group, weights=difference[scope]*(~h[scope]), minlength=count)
        boot = .7*(draws@dh)/(draws@hc)+.3*(draws@dn)/(draws@nc)
        deltas = {str(k): metrics(pred, folds == k)[0]["testlike"]-metrics(reference, folds == k)[0]["testlike"] for k in complete}
        lost = int(np.sum(scope & (reference["oof"] == truth["oof"]) & (pred["oof"] != truth["oof"])))
        gate = protocol["screen_gate"]
        checks = {"folds": complete == protocol["screen_folds"],
                  "testlike": pm["testlike"]-rm["testlike"] >= gate["testlike_gain_min"],
                  "worst_fold": min(deltas.values()) >= gate["minimum_fold_delta"],
                  "hard_hangul": pm["hard_hangul"]-rm["hard_hangul"] >= gate["hard_hangul_delta_min"],
                  "drop_word": pm["scenarios"]["drop_word"]-rm["scenarios"]["drop_word"] >= gate["drop_word_delta_min"],
                  "clean_lost": lost <= gate["clean_lost_rows_max"],
                  "matched_control": pm["testlike"]-cm["testlike"] >= gate["gain_vs_matched_control_min"]}
        changes = {}
        for s in truth:
            mask = scope if s == "oof" else scope[lab.sets[s].df.origin.to_numpy()]
            old_ok, new_ok = reference[s] == truth[s], pred[s] == truth[s]
            changes[s] = {"gained": int(np.sum(mask & ~old_ok & new_ok)), "lost": int(np.sum(mask & old_ok & ~new_ok))}
        reports[method] = {"candidate": pm, "matched_control": cm,
                           "matched_control_recipe": protocol["matched_control"],
                           "delta_vs_v12": pm["testlike"]-rm["testlike"],
                           "delta_vs_matched_control": pm["testlike"]-cm["testlike"],
                           "bootstrap_95_vs_v12": np.quantile(boot, [.025, .975]).tolist(),
                           "delta_by_fold_vs_v12": deltas, "clean_lost_rows": lost,
                           "screen_checks": checks, "screen_gate_pass": bool(all(checks.values())),
                           "without_shots": {"reference_v12": metrics(reference, no_shots)[0], "candidate": metrics(pred, no_shots)[0]},
                           "changes": changes}
    selected = next((m for m in protocol["contenders"] if reports[m]["screen_gate_pass"]), None)
    report = {"tag": protocol["tag"], "complete_folds": complete, "reference_v12": rm,
              "contenders": reports, "selected": selected, "truth_outside_top20": outside,
              "actual_competition_submission": False, "caveat": protocol["caveat"]}
    out = ROOT / "experiments" / f"{protocol['tag']}_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": predictions, "reference": reference, "identity": archived["identity"], "covered": seen},
                ROOT / "experiments" / f"{protocol['tag']}_predictions.joblib", compress=3)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    audit()
