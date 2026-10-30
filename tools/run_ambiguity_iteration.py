"""Run the single registered loss candidate and its controls; never upload."""
import json
import os
import subprocess
import sys

from src.common import ROOT
from tools.ambiguity_encoder_experiment import TAG, check_lock


def stage(name, module, *args):
    path = ROOT / "experiments" / f"{TAG}_{name}_run.log"
    print(f"Starting {name}: {path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=ROOT,
                       env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {name}", flush=True)


def main():
    protocol = check_lock()
    path = ROOT / "experiments" / f"{TAG}_selection.json"
    assert not path.exists(), "Inspect the recorded selection instead of restarting"
    for fold in protocol["screen_folds"]:
        stage(f"screen_fold{fold}", "tools.ambiguity_encoder_experiment", "--folds", fold)
    stage("screen_audit", "tools.audit_ambiguity_iteration")
    screen = json.loads((ROOT / "experiments" / f"{TAG}_screen.json").read_text())
    decision = {"tag": TAG, "selected": None, "stage": "screen", "screen": screen,
                "actual_competition_submission": False}
    if screen["passed"]:
        fold = protocol["fresh_guard"]["fold"]
        stage(f"guard_fold{fold}", "tools.ambiguity_encoder_experiment", "--folds", fold, "--arms", "targeted")
        stage("fresh_guard", "tools.fresh_ambiguity_guard")
        fresh = json.loads((ROOT / "experiments" / f"{TAG}_fresh.json").read_text())
        decision.update(stage="fresh", fresh=fresh)
        if fresh["delta_vs_v12"] >= protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
            stage("guard_controls", "tools.ambiguity_encoder_experiment", "--folds", fold, "--arms", "none", "random")
            for remaining in sorted(set(range(5))-set(protocol["screen_folds"]+[fold])):
                stage(f"full_fold{remaining}", "tools.ambiguity_encoder_experiment", "--folds", remaining)
            stage("full_audit", "tools.audit_ambiguity_iteration", "--stage", "full")
            full = json.loads((ROOT / "experiments" / f"{TAG}_full.json").read_text())
            decision.update(stage="full", full=full)
            if full["passed"]: decision["selected"] = f"{TAG}_targeted"
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"selected": decision["selected"], "stage": decision["stage"]}), flush=True)
    if decision["selected"]:
        stage("release", "tools.release_ambiguity_encoder")


if __name__ == "__main__": main()
