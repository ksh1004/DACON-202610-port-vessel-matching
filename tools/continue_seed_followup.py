"""Screen one extra seed's fixed parameter/probability averages in that order."""
import hashlib
import json
import shutil
import time

from src.common import ROOT
from tools.continue_encoder_followup import stage, read, full_checks


def trained(fold):
    return (ROOT / "experiments/artifacts/encoder_seed_average_v1" / f"fold{fold}_epoch1_scores.joblib").exists()


def fit(fold):
    if not trained(fold):
        stage(f"seed_average_f{fold}", "src.encoder_seed_average", "--fold", fold)


def attempt(probability=False):
    tag = "encoder_seed_probability_v1" if probability else "encoder_seed_average_v1"
    kind = "seed_probability" if probability else "seed_average"
    protocol_name = f"{kind}_protocol.json".replace("seed_", "encoder_seed_", 1)
    protocol = read(protocol_name)
    module = "tools.audit_seed_probability" if probability else "tools.audit_encoder_extension"
    if probability:
        for fold in protocol["screen_folds"]:
            stage(f"seed_probability_f{fold}", "tools.score_seed_probability", "--fold", fold)
        stage("seed_probability_screen", module, "--tag", tag, "--protocol", protocol_name)
    audit = read(f"{tag}_audit.json")
    assert audit["complete_folds"] == protocol["screen_folds"]
    shutil.copyfile(ROOT / "experiments" / f"{tag}_audit.json", ROOT / "experiments" / f"{tag}_screen_snapshot.json")
    if not audit["screen_gate_pass"]:
        return {"tag": tag, "kind": kind, "passed": False, "stage": "screen", "checks": audit["screen_checks"]}
    fold = protocol["fresh_guard"]["fold"]
    fit(fold)
    stage(f"{kind}_fresh", "tools.fresh_seed_probability" if probability else "tools.fresh_encoder_extension",
          "--fold", fold, "--tag", tag)
    fresh = read(f"fresh_extension_f{fold}_{tag}.json")
    if fresh["delta_vs_v12"] < protocol["fresh_guard"]["minimum_mean_accuracy_delta"]:
        return {"tag": tag, "kind": kind, "passed": False, "stage": "fresh", "delta": fresh["delta_vs_v12"]}
    for fold in range(5):
        fit(fold)
        if probability:
            output = ROOT / "experiments/artifacts" / tag / f"fold{fold}_epoch1_scores.joblib"
            if not output.exists():
                stage(f"seed_probability_f{fold}", "tools.score_seed_probability", "--fold", fold)
    stage(f"{kind}_full_audit", module, "--tag", tag, "--protocol", protocol_name)
    audit = read(f"{tag}_audit.json")
    checks = full_checks(audit, protocol)
    return {"tag": tag, "kind": kind, "passed": bool(all(checks.values())), "stage": "full", "checks": checks,
            "cv_audit_sha256": hashlib.sha256((ROOT / "experiments" / f"{tag}_audit.json").read_bytes()).hexdigest(),
            "fresh_file": f"fresh_extension_f{protocol['fresh_guard']['fold']}_{tag}.json", "fresh_delta": fresh["delta_vs_v12"]}


def main():
    path = ROOT / "experiments/encoder_seed_average_v1_audit.json"
    deadline = time.monotonic()+2400
    print("Waiting for seed-average screen", flush=True)
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Seed screen did not complete")
        time.sleep(10)
    trials = [attempt(False)]
    if not trials[-1]["passed"]:
        trials.append(attempt(True))
    chosen = next((trial for trial in trials if trial["passed"]), None)
    result = read("encoder_followup_selection.json")
    result["trials"].extend(trials)
    result["selected"] = chosen
    (ROOT / "experiments/encoder_followup_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if chosen:
        stage("seed_release", "tools.release_encoder_followup")
    else:
        print("All registered trials failed; preserve v12.", flush=True)


if __name__ == "__main__":
    main()
