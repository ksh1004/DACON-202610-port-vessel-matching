"""Fixed Qwen/new-encoder mix evaluated against the frozen v12 OOF winner."""
import argparse
import json

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.experiment import TESTLIKE_MIX
from src.llm_translit import FEW_SHOT
from src.pair_views import mix_views
from src.validate import load_folds
from tools.blend_matching import lightweight_lab


def audit(tag, protocol_name="encoder_seed_probability_protocol.json"):
    archived = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archived["identity"])
    reference = archived["ensemble"]
    prediction = {s: values.copy() for s, values in reference.items()}
    truth = {"oof": lab.ctx.truth, **{s: values.truth for s, values in lab.sets.items()}}
    pos = {s: {key: i for i, key in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in reference}
    seen = {s: np.zeros(len(values), dtype=bool) for s, values in reference.items()}
    complete, protocols = [], {}
    for fold in range(5):
        path = ROOT / "experiments/artifacts" / tag / f"fold{fold}_epoch1_scores.joblib"
        if not path.exists():
            continue
        data = joblib.load(path)
        old = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_epoch1_scores.joblib")
        qwen = joblib.load(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_epoch1_scores.joblib")
        complete.append(fold)
        protocols[str(fold)] = data["protocol"]
        for s, rows in data["sets"].items():
            assert data["sample_ids"][s] == old["sample_ids"][s] == qwen["sample_ids"][s]
            for key, row, prior, qr in zip(data["sample_ids"][s], rows, old["sets"][s], qwen["sets"][s]):
                i = pos[s][key]
                assert not seen[s][i]
                for field in ("candidates", "keep", "base_scores"):
                    np.testing.assert_array_equal(row[field], prior[field])
                    np.testing.assert_array_equal(row[field], qr[field])
                assert row["truth"] == prior["truth"] == qr["truth"] == truth[s][i]
                assert row["bound_pruned"] == prior["bound_pruned"] == qr["bound_pruned"]
                name = lab.ctx.train.vessel_name.iloc[i] if s == "oof" else lab.sets[s].df.vessel_name.iloc[i]
                winner = int(np.argmax(row["base_scores"]))
                old_winner = winner
                if is_hangul(name) and not row["bound_pruned"]:
                    keep = row["keep"]
                    bp = row["base_scores"][keep]
                    bp = bp/bp.sum()
                    old_winner = int(keep[np.argmax(mix_views(bp, [qr["neural_scores"], prior["neural_scores"]], [1, 1]))])
                    winner = int(keep[np.argmax(mix_views(bp, [qr["neural_scores"], prior["neural_scores"], row["neural_scores"]], [2, 1, 1]))])
                assert row["candidates"][old_winner] == reference[s][i]
                prediction[s][i] = row["candidates"][winner]
                seen[s][i] = True
    if not complete:
        raise RuntimeError("No complete fold evaluation")
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
        for s in ("gt_missing", "drop_word", "both"):
            pick = tl.scenario.eq(s).to_numpy()
            origin = tl.origin.to_numpy()[pick]
            parts[s] = float(ok["testlike"][pick][mask[origin]].mean())
            np.add.at(values, origin, TESTLIKE_MIX[s]*ok["testlike"][pick])
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
    pm, pv = metrics(prediction)
    difference = pv-rv
    _, group = np.unique(lab.ctx.truth[scope], return_inverse=True)
    count = int(group.max())+1
    draws = np.random.default_rng(20261028).multinomial(count, np.full(count, 1/count), size=2000)
    hc = np.bincount(group, weights=h[scope], minlength=count)
    nc = np.bincount(group, weights=~h[scope], minlength=count)
    dh = np.bincount(group, weights=difference[scope]*h[scope], minlength=count)
    dn = np.bincount(group, weights=difference[scope]*(~h[scope]), minlength=count)
    boot = .7*(draws@dh)/(draws@hc)+.3*(draws@dn)/(draws@nc)
    fold_delta = {str(k): metrics(prediction, folds == k)[0]["testlike"]-metrics(reference, folds == k)[0]["testlike"] for k in complete}
    lost = int(np.sum(scope & (reference["oof"] == truth["oof"]) & (prediction["oof"] != truth["oof"])))
    shots = lab.ctx.train.loc[lab.ctx.train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"].unique()
    no_shots = scope & ~lab.ctx.train.target_vessel_id.isin(shots).to_numpy()
    protocol = json.loads((ROOT / "experiments" / protocol_name).read_text(encoding="utf-8"))
    gates = protocol["screen_gate"]
    checks = {"folds": complete == protocol["screen_folds"],
              "testlike": pm["testlike"]-rm["testlike"] >= gates["testlike_gain_min"],
              "worst_fold": min(fold_delta.values()) >= gates["minimum_fold_delta"],
              "hard_hangul": pm["hard_hangul"]-rm["hard_hangul"] >= gates["hard_hangul_delta_min"],
              "drop_word": pm["scenarios"]["drop_word"]-rm["scenarios"]["drop_word"] >= gates["drop_word_delta_min"],
              "clean_lost": lost <= gates["clean_lost_rows_max"]}
    changes = {}
    for s in truth:
        mask = scope if s == "oof" else scope[lab.sets[s].df.origin.to_numpy()]
        old_ok, new_ok = reference[s] == truth[s], prediction[s] == truth[s]
        changes[s] = {"gained": int(np.sum(mask & ~old_ok & new_ok)), "lost": int(np.sum(mask & old_ok & ~new_ok))}
    report = {"tag": tag, "reference": "v12", "complete_folds": complete, "complete": len(complete) == 5,
              "reference_v12": rm, "candidate": pm, "delta_vs_v12": pm["testlike"]-rm["testlike"],
              "bootstrap_95_vs_v12": np.quantile(boot, [.025, .975]).tolist(),
              "delta_by_fold_vs_v12": fold_delta, "clean_lost_rows_vs_v12": lost,
              "screen_checks": checks, "screen_gate_pass": bool(all(checks.values())),
              "without_shots": {"reference_v12": metrics(reference, no_shots)[0], "candidate": metrics(prediction, no_shots)[0]},
              "changes": changes, "protocols": protocols,
              "caveat": "Reused exploratory validation; historical v8 shared retrieval mapper exception remains."}
    output = ROOT / "experiments" / f"{tag}_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": prediction, "reference": reference,
                 "identity": archived["identity"], "covered": seen},
                ROOT / "experiments" / f"{tag}_predictions.joblib", compress=3)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="encoder_seed_probability_v1")
    parser.add_argument("--protocol", default="encoder_seed_probability_protocol.json")
    args = parser.parse_args()
    audit(args.tag, args.protocol)
