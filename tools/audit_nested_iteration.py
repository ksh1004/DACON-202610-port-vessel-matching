"""Collect completed nested screens without changing any selection condition."""
import datetime
import hashlib
import json

from src.common import ROOT


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    state = read("experiments/nested_family_selection_20261029.json")
    family = read("experiments/nested_family_protocol_20261029.json")
    lock = read("experiments/nested_combiner_lock.json")
    checks = {}
    for source in (lock["source_and_input_hashes"], family["source_hashes"]):
        for path, expected in source.items():
            actual = sha(ROOT / path)
            assert actual == expected, path
            checks[path] = actual
    frozen = read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name, expected in frozen["source_hashes"].items():
        assert sha(ROOT / "src" / name) == expected, name
    release = read("experiments/release_v12_encoder.json")
    archive = ROOT / release["zip"]
    assert archive.stat().st_size == release["bytes"]
    assert sha(archive) == release["sha256"]
    reports = {}
    for tag in state["families"]:
        audit = read(f"experiments/{tag}_audit.json")
        assert audit == read(f"experiments/{tag}_screen_locked.json")
        assert audit["complete_folds"] == [0, 2]
        head_checks = read(f"experiments/{tag}_head_checks.json")
        assert head_checks["status"] == "PASS"
        reports[tag] = {
            "reference_testlike": audit["reference_v12"]["testlike"],
            "selected": audit["selected"],
            "contenders": {
                name: {key: value for key, value in result.items() if key in {
                    "candidate", "delta_vs_v12", "delta_vs_matched_control",
                    "bootstrap_95_vs_v12", "delta_by_fold_vs_v12",
                    "screen_checks", "screen_gate_pass", "clean_lost_rows", "changes"}}
                for name, result in audit["contenders"].items()
            },
            "truth_outside_top20": audit["truth_outside_top20"],
            "saved_head_checks": head_checks,
        }
    training = []
    for outer in (0, 2):
        for inner in (0, 1):
            folder = f"experiments/artifacts/nested_combiner_v1/outer{outer}_inner{inner}"
            prep = read(f"{folder}/preparation.json")
            fit, held, outside = (set(prep[k]) for k in ("fit_vessels", "meta_vessels", "outer_validation_vessels"))
            assert not fit & held and not outside & (fit | held)
            for kind in ("bge", "qwen"):
                record = read(f"{folder}/{kind}/training.json")
                assert (ROOT / folder / f"meta_{kind}_scores.joblib").exists()
                training.append({"outer": outer, "inner": inner, "kind": kind,
                                 "nn_groups": prep["nn_groups"], "meta_groups": prep["meta_groups"],
                                 "training_record": record})
    rejected = state["selected_family"] is None
    if rejected:
        assert list(state["families"]) == family["family_order"]
        assert all(r["selected"] is None for r in reports.values())
    result = {
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "client_date": "2026-10-29", "selected_family": state["selected_family"],
        "selected_contender": state["selected_contender"], "screens": reports,
        "new_inner_neural_models": training,
        "source_and_input_hashes_verified": checks,
        "frozen_v12_source_modules_verified": len(frozen["source_hashes"]),
        "v12_zip_sha256_verified": release["sha256"],
        "v12_zip_bytes_verified": archive.stat().st_size,
        "decision": "all four screens rejected; retain v12" if rejected else "locked screen qualifier requires fresh/full gates",
        "fresh_candidate_validation_performed": False,
        "full_five_fold_candidate_validation_performed": False,
        "new_final_model": False, "new_zip": False, "actual_competition_submission": False,
        "actual_best_public": 0.9065951735, "actual_server_seconds": 591,
        "actual_submission_count": 9, "target_public": 0.97,
        "lookup_used": False,
        "caveats": ["Repeated exploratory outer0/2 synthetic validation is not Public performance.",
                    "Historical outer v8 retrieval mapper exception remains; inner fits were strictly vessel-separated.",
                    "Saved-head checks use cached experts, not a new complete submission package.",
                    "Preparing private inner data for folds1/3/4 is not candidate validation on those folds."]}
    path = ROOT / "experiments/nested_iteration_20261029.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(path), "decision": result["decision"],
                      "source_and_input_files_verified": len(checks), "frozen_v12_modules": len(frozen["source_hashes"]),
                      "v12_zip_unchanged": True, "new_inner_neural_models": len(training)}, indent=2))


if __name__ == "__main__":
    main()
