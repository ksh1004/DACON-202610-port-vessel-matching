"""Evaluate the preregistered catalog warmup without competition uploads."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

from src.common import ROOT
from tools.continue_encoder_followup import full_checks


TAG = "encoder_catalog_warm_v1"
PROTOCOL = "catalog_warm_protocol.json"


def read(name):
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def stage(label, module, *args):
    path = ROOT / "experiments" / f"run_catalog_{label}.log"
    print(f"Starting {label}: {path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=ROOT,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {label}", flush=True)


def attempt():
    protocol = read(PROTOCOL)
    stage("warmup", "src.catalog_warm_experiment", "--stage", "warmup")
    for k in protocol["screen_folds"]:
        stage(f"f{k}", "src.catalog_warm_experiment", "--stage", "fold", "--fold", k)
    stage("screen", "tools.audit_encoder_extension", "--tag", TAG, "--protocol", PROTOCOL)
    audit = read(f"{TAG}_audit.json")
    assert audit["complete_folds"] == protocol["screen_folds"]
    shutil.copyfile(ROOT / "experiments" / f"{TAG}_audit.json", ROOT / "experiments" / f"{TAG}_screen_snapshot.json")
    if not audit["screen_gate_pass"]:
        return {"tag": TAG, "passed": False, "stage": "screen", "checks": audit["screen_checks"]}
    k = protocol["fresh_guard"]["fold"]
    stage(f"f{k}", "src.catalog_warm_experiment", "--stage", "fold", "--fold", k)
    stage("fresh", "tools.fresh_encoder_extension", "--fold", k, "--tag", TAG)
    fresh_file = f"fresh_extension_f{k}_{TAG}.json"
    fresh = read(fresh_file)
    if fresh["delta_vs_v12"] < protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
        return {"tag": TAG, "passed": False, "stage": "fresh", "delta": fresh["delta_vs_v12"]}
    for k in sorted(set(range(5))-set(protocol["screen_folds"]+[k])):
        stage(f"f{k}", "src.catalog_warm_experiment", "--stage", "fold", "--fold", k)
    stage("full_audit", "tools.audit_encoder_extension", "--tag", TAG, "--protocol", PROTOCOL)
    audit = read(f"{TAG}_audit.json")
    checks = full_checks(audit, protocol)
    return {"tag": TAG, "passed": bool(all(checks.values())), "stage": "full", "checks": checks,
            "cv_audit_sha256": hashlib.sha256((ROOT / "experiments" / f"{TAG}_audit.json").read_bytes()).hexdigest(),
            "fresh_file": fresh_file, "fresh_delta": fresh["delta_vs_v12"]}


def main():
    trial = attempt()
    selected = trial if trial["passed"] else None
    result = {"reference": "v12", "reference_public": .9065951735,
              "selected": selected, "trials": [trial], "actual_competition_submission": False}
    (ROOT / "experiments/catalog_warm_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if selected:
        stage("release", "tools.release_catalog_encoder")


if __name__ == "__main__":
    main()
