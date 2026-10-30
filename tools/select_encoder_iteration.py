"""Select only variants that pass both preregistered screen and full gates."""
import hashlib
import json

from src.common import ROOT


def select():
    tag = "encoder_bge_phonetic"
    path = ROOT / "experiments" / f"{tag}_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    screen = json.loads((ROOT / "experiments" / f"{tag}_screen_snapshot.json").read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "experiments" / f"fresh_pairs_f1_{tag}.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "experiments/encoder_phonetic_protocol.json").read_text(encoding="utf-8"))
    if audit["complete_folds"] != list(range(5)) or not audit["complete"]:
        raise ValueError("Complete five-fold encoder evaluation required")
    ref, gate = audit["reference_v11"], protocol["full_gate"]
    decisions = {}
    for variant, metric, deltas, screen_ok, fresh_delta in (
        ("encoder_only", audit["candidate"], audit["delta_by_fold_vs_v11"], screen["screen_gate_pass"], fresh["candidate_delta_vs_v11"]),
        ("equal_ensemble", audit["equal_neural_ensemble"], audit["ensemble_delta_by_fold_vs_v11"], screen["ensemble_screen_gate_pass"], fresh["ensemble_delta_vs_v11"]),
    ):
        checks = {"screen": screen_ok, "fresh": fresh_delta >= protocol["fresh_guard"]["minimum_mean_accuracy_delta"],
                  "testlike": metric["testlike"]-ref["testlike"] >= gate["testlike_gain_min"],
                  "hard_hangul": metric["hard_hangul"]-ref["hard_hangul"] >= gate["hard_hangul_delta_min"],
                  "oof": metric["oof"]-ref["oof"] >= gate["oof_delta_min"],
                  "positive_folds": sum(value > 0 for value in deltas.values()) >= gate["positive_folds_min"]}
        decisions[variant] = {"passed": bool(all(checks.values())), "checks": checks, "metrics": metric}
    chosen = next((variant for variant in protocol["priority"] if decisions[variant]["passed"]), None)
    result = {"tag": tag, "selected_variant": chosen, "decisions": decisions,
              "cv_audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "reference_public_v11": .9039635945, "actual_competition_submission": False,
              "caveat": "Exploratory reused validation; operational criteria do not establish Public improvement."}
    (ROOT / "experiments/encoder_iteration_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(select(), indent=2), flush=True)
