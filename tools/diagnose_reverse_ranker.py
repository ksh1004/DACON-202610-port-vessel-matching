"""Post-screen diagnostics only; no fitting, tuning, or contender selection."""
import json

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT, load_catalog
from src.experiment import TESTLIKE_MIX
from tools import reverse_ranker_experiment as base


def main():
    audit = json.loads((ROOT / "experiments/reverse_kanana_v1_audit.json").read_text())
    assert audit["selected"] is None
    archived = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    indices = {s: {sid: i for i, sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])}
               for s in archived["ensemble"]}
    reverse = joblib.load(base.FOLDER / "pair_scores.joblib")["scores"]
    catnames = load_catalog().vessel_name.to_numpy()
    correct = {m: {s: [] for s in ("oof", "hard", "testlike")}
               for m in ("strict_all_group_tree", "reverse_name_only", "new_head", "v12")}
    scenarios, importance, cases = [], {}, {"gains": [], "losses": []}
    for fold in (0, 2):
        data = base.prepare(fold)
        results = joblib.load(base.FOLDER / f"fold{fold}_results.joblib")
        full = lgb.Booster(model_file=str(ROOT / f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt"))
        head = lgb.Booster(model_file=str(base.FOLDER / f"fold{fold}_reverse.txt"))
        gain = head.feature_importance(importance_type="gain")
        importance[str(fold)] = {"reverse_total_gain_fraction": float(gain[78:].sum()/gain.sum()),
                                  "top_features": sorted(zip(head.feature_name(), gain.tolist()), key=lambda x: -x[1])[:15]}
        _, _, _, frames, _ = base.frames_and_training(fold)
        scenario_map = dict(zip(frames["testlike"].sample_id, frames["testlike"].scenario))
        for s, groups in data["evaluation"].items():
            for g, result in zip(groups, results["sets"][s]):
                i = indices[s][g["sample_id"]]
                prior = int(archived["ensemble"][s][i])
                preds = {"v12": prior, "new_head": prior,
                         "strict_all_group_tree": prior, "reverse_name_only": prior}
                if g["actual_hangul"]:
                    raw = full.predict(g["raw"], num_threads=4)
                    likelihood = np.array([reverse[(g["name"], n)][1] for n in g["names"]])
                    preds.update({"strict_all_group_tree": int(g["candidates"][np.argmax(raw)]),
                                  "reverse_name_only": int(g["candidates"][np.argmax(likelihood)]),
                                  "new_head": result["predictions"]["direct"]})
                    old_ok, new_ok = prior == g["truth"], preds["new_head"] == g["truth"]
                    typ = "gains" if new_ok and not old_ok else "losses" if old_ok and not new_ok else None
                    if typ and s == "testlike" and len(cases[typ]) < 12:
                        cases[typ].append({"observed": g["name"], "scenario": scenario_map[g["sample_id"]],
                                           "truth": str(catnames[g["truth"]]), "v12": str(catnames[prior]),
                                           "new_head": str(catnames[preds["new_head"]])})
                for m, pred in preds.items():
                    correct[m][s].append(pred == g["truth"])
                if s == "testlike":
                    scenarios.append(scenario_map[g["sample_id"]])
    summaries = {}
    english = (audit["reference_v12"]["oof"]-.7*audit["reference_v12"]["scenarios"]["clean"])/.3
    scenarios = np.array(scenarios)
    for m, sets in correct.items():
        parts = {"clean": float(np.mean(sets["oof"]))}
        for s in ("gt_missing", "drop_word", "both"):
            parts[s] = float(np.array(sets["testlike"])[scenarios == s].mean())
        summaries[m] = {"scenarios": parts,
                        "testlike": .7*sum(TESTLIKE_MIX[s]*v for s, v in parts.items())+.3*english,
                        "hard_all_hangul_origin_rows": float(np.mean(sets["hard"]))}
    assert abs(summaries["v12"]["testlike"]-audit["reference_v12"]["testlike"]) < 1e-12
    report = {"purpose": "Post-screen diagnosis, no new candidate selection or tuning",
              "complete_folds": [0, 2], "summaries": summaries, "feature_importance": importance,
              "cases": cases, "strict_all_group_tree_caveat": "Existing strict miner scored on historical shared-mapper top20 and full-pool raw78; no new training. Different training pool/sampler from2400 head. Diagnostic only.",
              "actual_competition_submission": False}
    (ROOT / "experiments/reverse_kanana_v1_diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "feature_importance": importance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
