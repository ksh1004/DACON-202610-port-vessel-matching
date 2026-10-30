"""Apply the predeclared complete-CV gates without changing their thresholds."""
import argparse
import hashlib
import json

from src.common import ROOT


def select(tag):
    path = ROOT / "experiments" / f"{tag}_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "experiments" / f"fresh_pairs_f3_{tag}.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "experiments/pair_iteration_protocol.json").read_text(encoding="utf-8"))
    if not audit["complete"] or audit["complete_folds"] != list(range(5)):
        raise ValueError("A complete five-fold evaluation is required for final model selection")
    gate = protocol["full_promotion_gate"]
    reference = audit["reference_v10"]
    decisions = {}
    for variant, metrics, deltas, fresh_delta in (
        ("evidence_only", audit["candidate"], audit["delta_by_fold_vs_v10"], fresh["candidate_delta_vs_v10"]),
        ("equal_neural_ensemble", audit["equal_neural_ensemble"], audit["ensemble_delta_by_fold_vs_v10"], fresh["ensemble_delta_vs_v10"]),
    ):
        checks = {"testlike": metrics["testlike"]-reference["testlike"] >= gate["testlike_gain_vs_v10_min"],
                  "hard_hangul": metrics["hard_hangul"]-reference["hard_hangul"] >= gate["hard_hangul_delta_min"],
                  "oof": metrics["oof"]-reference["oof"] >= gate["oof_delta_min"],
                  "positive_folds": sum(d > 0 for d in deltas.values()) >= gate["positive_testlike_folds_min"],
                  "fresh_stress": fresh_delta >= gate["fresh_stress_mean_delta_min"]}
        decisions[variant] = {"passed": bool(all(checks.values())), "checks": checks, "metrics": metrics}
    selected = next((v for v, result in decisions.items() if result["passed"]), None)
    report = {"tag": tag, "selected_variant": selected, "decisions": decisions,
              "cv_audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "actual_competition_submission": False,
              "caveat": "Operational gates on exploratory reused validation; not proof of Public improvement or 0.97."}
    (ROOT / "experiments/pair_iteration_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="neural_evidence_v1")
    print(json.dumps(select(parser.parse_args().tag), indent=2), flush=True)


if __name__ == "__main__":
    main()
