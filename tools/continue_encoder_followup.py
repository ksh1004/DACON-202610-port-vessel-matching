"""Continue preregistered trials sequentially after the running two-fold screen."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from src.common import ROOT


def stage(label, module, *args):
    logfile = ROOT / "experiments" / f"run_followup_{label}.log"
    print(f"Starting {label}: {logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=ROOT,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Passed execution {label}", flush=True)


def read(name):
    return json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))


def full_checks(audit, protocol):
    ref, new, gate = audit["reference_v12"], audit["candidate"], protocol["full_gate"]
    return {"five_folds": audit["complete_folds"] == list(range(5)),
            "testlike": new["testlike"]-ref["testlike"] >= gate["testlike_gain_min"],
            "hard_hangul": new["hard_hangul"]-ref["hard_hangul"] >= gate["hard_hangul_delta_min"],
            "oof": new["oof"]-ref["oof"] >= gate["oof_delta_min"],
            "drop_word": new["scenarios"]["drop_word"]-ref["scenarios"]["drop_word"] >= gate["drop_word_delta_min"],
            "positive_folds": sum(d > 0 for d in audit["delta_by_fold_vs_v12"].values()) >= gate["positive_folds_min"]}


def attempt(tag, kind):
    extended = kind == "expanded"
    protocol_name = "encoder_extension_protocol.json" if extended else "encoder_confidence_protocol.json"
    protocol = read(protocol_name)
    audit_module = "tools.audit_encoder_extension" if extended else "tools.audit_encoder_confidence"
    if not extended:
        for fold in protocol["screen_folds"]:
            stage(f"confidence_f{fold}", "tools.score_encoder_confidence", "--fold", fold)
        stage("confidence_screen", audit_module)
    audit = read(f"{tag}_audit.json")
    assert audit["complete_folds"] == protocol["screen_folds"]
    snapshot = ROOT / "experiments" / f"{tag}_screen_snapshot.json"
    shutil.copyfile(ROOT / "experiments" / f"{tag}_audit.json", snapshot)
    if not audit["screen_gate_pass"]:
        return {"tag": tag, "kind": kind, "passed": False, "stage": "screen", "checks": audit["screen_checks"]}
    fold = protocol["fresh_guard"]["fold"]
    if extended:
        stage("expanded_f0", "src.encoder_extended_experiment", "--fold", fold)
    stage(f"{kind}_fresh", "tools.fresh_encoder_extension" if extended else "tools.fresh_encoder_confidence",
          "--fold", fold, "--tag", tag)
    fresh = read(f"fresh_extension_f{fold}_{tag}.json")
    if fresh["delta_vs_v12"] < protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
        return {"tag": tag, "kind": kind, "passed": False, "stage": "fresh", "delta": fresh["delta_vs_v12"]}
    completed = set(protocol["screen_folds"] + ([fold] if extended else []))
    for k in sorted(set(range(5))-completed):
        stage(f"{kind}_f{k}", "src.encoder_extended_experiment" if extended else "tools.score_encoder_confidence", "--fold", k)
    stage(f"{kind}_full_audit", audit_module)
    audit = read(f"{tag}_audit.json")
    checks = full_checks(audit, protocol)
    return {"tag": tag, "kind": kind, "passed": bool(all(checks.values())), "stage": "full", "checks": checks,
            "cv_audit_sha256": hashlib.sha256((ROOT / "experiments" / f"{tag}_audit.json").read_bytes()).hexdigest(),
            "fresh_file": f"fresh_extension_f{fold}_{tag}.json", "fresh_delta": fresh["delta_vs_v12"]}


def main():
    screen = ROOT / "experiments/encoder_bge_full_v1_audit.json"
    print("Waiting for the running expanded-data two-fold screen", flush=True)
    deadline = time.monotonic()+3600
    while not screen.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Expanded-data screen did not complete")
        time.sleep(10)
    trials = [attempt("encoder_bge_full_v1", "expanded")]
    if not trials[-1]["passed"]:
        trials.append(attempt("encoder_confidence40_v1", "confidence"))
    chosen = next((t for t in trials if t["passed"]), None)
    result = {"reference": "v12", "reference_public": .9065951735,
              "selected": chosen, "trials": trials, "actual_competition_submission": False,
              "caveat": "Exploratory reused validation, sequential predeclared fallback; Public improvement unmeasured."}
    (ROOT / "experiments/encoder_followup_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if chosen:
        stage("release", "tools.release_encoder_followup")
    else:
        print("Neither trial passed; preserve v12 without a new submission ZIP.", flush=True)


if __name__ == "__main__":
    main()
