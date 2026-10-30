"""Evaluate the predeclared targeted loss against both paired controls."""
import argparse
import json

from tools.ambiguity_encoder_experiment import ARMS, PROTOCOL, TAG, check_lock, fingerprint, folder
from tools.audit_encoder_extension import audit
from src.common import ROOT


def run(stage="screen"):
    protocol = check_lock()
    reports = {a: audit(f"{TAG}_{a}", PROTOCOL.name) for a in ARMS}
    candidate, plain, sham = (reports[a] for a in ("targeted", "none", "random"))
    folds = candidate["complete_folds"]
    assert folds == plain["complete_folds"] == sham["complete_folds"]
    initial, token = {}, {}
    for a in ARMS:
        for fold in folds:
            report = json.loads((folder(a) / f"fold{fold}_training.json").read_text())
            assert report["fingerprint"] == fingerprint()
            assert report["finite_loss"] and report["steps"] == 1200
            initial.setdefault(fold, set()).add(report["initial_trainable_sha256"])
            token.setdefault(fold, set()).add(report["token_sha256"])
    assert all(len(v) == 1 for v in initial.values())
    assert all(len(v) == 1 for v in token.values())
    delta_plain = candidate["candidate"]["testlike"] - plain["candidate"]["testlike"]
    delta_sham = candidate["candidate"]["testlike"] - sham["candidate"]["testlike"]
    gate = protocol["screen_gate"] if stage == "screen" else protocol["full_gate"]
    if stage == "screen":
        checks = dict(candidate["screen_checks"])
    else:
        ref, new = candidate["reference_v12"], candidate["candidate"]
        checks = {"five_folds": folds == list(range(5)),
                  "testlike": new["testlike"] - ref["testlike"] >= gate["testlike_gain_min"],
                  "hard_hangul": new["hard_hangul"] - ref["hard_hangul"] >= gate["hard_hangul_delta_min"],
                  "drop_word": new["scenarios"]["drop_word"] - ref["scenarios"]["drop_word"] >= gate["drop_word_delta_min"],
                  "oof": new["oof"] - ref["oof"] >= gate["oof_delta_min"],
                  "positive_folds": sum(d > 0 for d in candidate["delta_by_fold_vs_v12"].values()) >= gate["positive_folds_min"]}
    checks.update(gain_vs_unmasked=delta_plain >= gate["gain_vs_unmasked_min"],
                  gain_vs_random=delta_sham >= gate["gain_vs_random_min"],
                  paired_initial_parameters=True, paired_training_tokens=True)
    report = {"tag": TAG, "stage": stage, "complete_folds": folds,
              "reference_v12": candidate["reference_v12"],
              "arms": {a: {"metrics": r["candidate"], "delta_vs_v12": r["delta_vs_v12"],
                             "changes": r["changes"], "bootstrap_95_vs_v12": r["bootstrap_95_vs_v12"]} for a, r in reports.items()},
              "delta_vs_unmasked": delta_plain, "delta_vs_random": delta_sham,
              "checks": checks, "passed": bool(all(checks.values())),
              "selected": f"{TAG}_targeted" if all(checks.values()) else None,
              "actual_competition_submission": False,
              "caveat": "Exploratory reused synthetic validation; historical v8 outer retrieval mapper exception remains. This tests a loss hypothesis, not Public0.97."}
    path = ROOT / "experiments" / f"{TAG}_{stage}.json"
    assert not path.exists(), "Preserve recorded stage decisions"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if stage == "screen":
        for arm, arm_report in reports.items():
            snapshot = ROOT / "experiments" / f"{TAG}_{arm}_screen_locked.json"
            assert not snapshot.exists()
            snapshot.write_text(json.dumps(arm_report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("screen", "full"), default="screen")
    run(p.parse_args().stage)
