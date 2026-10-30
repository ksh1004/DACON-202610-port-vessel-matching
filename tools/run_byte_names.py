"""Sequential fixed byte-name trial; expand only at registered gates."""
import json
import os
import subprocess
import sys

from src.common import ROOT
from tools.byte_names_experiment import check_lock
from tools.prepare_byte_names import FOLDER, PROTOCOL, TAG


def stage(name, module, *args):
    path = ROOT/"experiments"/f"{TAG}_{name}_run.log"
    print(f"Starting {name}: {path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=ROOT,
                       env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {name}", flush=True)


def audit():
    from tools.audit_reverse_ranker import audit as compare
    check_lock()
    return compare(FOLDER, PROTOCOL)


def main():
    protocol = check_lock(); selected = ROOT/"experiments"/f"{TAG}_selection.json"
    assert not selected.exists(), "Preserve completed selection"
    for fold in protocol["screen_folds"]: stage(f"fold{fold}", "tools.byte_names_experiment", "--folds", fold)
    report = audit()
    (ROOT/"experiments"/f"{TAG}_screen_locked.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    decision = {"tag": TAG, "selected": None, "stage": "screen", "screen": report, "actual_competition_submission": False}
    if report["selected"] is not None:
        fold = protocol["fresh_guard"]["fold"]
        stage("guard_training", "tools.byte_names_experiment", "--folds", fold, "--train-only")
        stage("fresh", "tools.fresh_byte_names")
        fresh = json.loads((ROOT/"experiments"/f"{TAG}_fresh.json").read_text())
        decision.update(stage="fresh", fresh=fresh)
        if fresh["delta_vs_v12"] >= protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
            for k in sorted(set(range(5))-set(protocol["screen_folds"])):
                stage(f"full_fold{k}", "tools.byte_names_experiment", "--folds", k)
            full = audit(); ref = full["reference_v12"]; r = full["contenders"]["residual30"]
            new, gate = r["candidate"], protocol["full_gate"]
            checks = {"five_folds": full["complete_folds"] == list(range(5)),
                      "testlike": new["testlike"]-ref["testlike"] >= gate["testlike_gain_min"],
                      "hard_hangul": new["hard_hangul"]-ref["hard_hangul"] >= gate["hard_hangul_delta_min"],
                      "drop_word": new["scenarios"]["drop_word"]-ref["scenarios"]["drop_word"] >= gate["drop_word_delta_min"],
                      "oof": new["oof"]-ref["oof"] >= gate["oof_delta_min"],
                      "positive_folds": sum(v>0 for v in r["delta_by_fold_vs_v12"].values()) >= gate["positive_folds_min"],
                      "matched_control": r["delta_vs_matched_control"] >= gate["gain_vs_matched_control_min"]}
            decision.update(stage="full", full=full, full_checks=checks)
            if all(checks.values()): decision["selected"] = "residual30"
    selected.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"selected": decision["selected"], "stage": decision["stage"]}), flush=True)
    if decision["selected"]: stage("release", "tools.release_byte_names")


if __name__ == "__main__": main()
