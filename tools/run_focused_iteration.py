"""Continue A1/A2 only at their frozen gates; never upload a competition entry."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from src.common import ROOT
from tools.continue_encoder_followup import full_checks


def stage(label, module, *args):
    path = ROOT / "experiments" / f"run_data_{label}.log"
    print(f"Starting {label}: {path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=ROOT,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Passed execution {label}", flush=True)


def read(name):
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def attempt(mode, initial=False):
    tag = "encoder_focused20_v1" if mode == "focused" else "encoder_restore_v1"
    protocol_name = f"encoder_{mode}_protocol.json"
    protocol = read(protocol_name)
    if not initial:
        for k in protocol["screen_folds"]:
            stage(f"{mode}_f{k}", "src.focused_encoder_experiment", "--fold", k, "--mode", mode)
        stage(f"{mode}_screen", "tools.audit_encoder_extension", "--tag", tag, "--protocol", protocol_name)
    audit = read(f"{tag}_audit.json")
    assert audit["complete_folds"] == protocol["screen_folds"]
    shutil.copyfile(ROOT / "experiments" / f"{tag}_audit.json", ROOT / "experiments" / f"{tag}_screen_snapshot.json")
    if not audit["screen_gate_pass"]:
        return {"tag": tag, "mode": mode, "passed": False, "stage": "screen", "checks": audit["screen_checks"]}
    k = protocol["fresh_guard"]["fold"]
    stage(f"{mode}_f{k}", "src.focused_encoder_experiment", "--fold", k, "--mode", mode)
    stage(f"{mode}_fresh", "tools.fresh_encoder_extension", "--fold", k, "--tag", tag)
    fresh_file = f"fresh_extension_f{k}_{tag}.json"
    fresh = read(fresh_file)
    if fresh["delta_vs_v12"] < protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
        return {"tag": tag, "mode": mode, "passed": False, "stage": "fresh", "delta": fresh["delta_vs_v12"]}
    for k in sorted(set(range(5))-set(protocol["screen_folds"]+[k])):
        stage(f"{mode}_f{k}", "src.focused_encoder_experiment", "--fold", k, "--mode", mode)
    stage(f"{mode}_full_audit", "tools.audit_encoder_extension", "--tag", tag, "--protocol", protocol_name)
    audit = read(f"{tag}_audit.json")
    checks = full_checks(audit, protocol)
    return {"tag": tag, "mode": mode, "passed": bool(all(checks.values())), "stage": "full", "checks": checks,
            "cv_audit_sha256": hashlib.sha256((ROOT / "experiments" / f"{tag}_audit.json").read_bytes()).hexdigest(),
            "fresh_file": fresh_file, "fresh_delta": fresh["delta_vs_v12"]}


def main():
    first = ROOT / "experiments/encoder_focused20_v1_audit.json"
    deadline = time.monotonic()+3600
    print("Waiting for A1 screen", flush=True)
    while not first.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("A1 screen did not complete")
        time.sleep(10)
    trials = [attempt("focused", initial=True)]
    if not trials[-1]["passed"]:
        trials.append(attempt("restore"))
    selected = next((trial for trial in trials if trial["passed"]), None)
    result = {"reference": "v12", "reference_public": .9065951735, "selected": selected,
              "trials": trials, "actual_competition_submission": False}
    (ROOT / "experiments/focused_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if selected:
        stage("release", "tools.release_focused_encoder")


if __name__ == "__main__":
    main()
