"""Build and audit v13 only after every predeclared v12 comparison gate passes."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

from src.common import ROOT


def stage(label, args):
    path = ROOT / "experiments" / f"run_v13_{label}.log"
    start = time.perf_counter()
    print(f"Starting {label}: {path}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", *map(str, args)], cwd=ROOT,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
    print(f"Passed {label} ({time.perf_counter()-start:.1f}s)", flush=True)
    return path


def main():
    selection = json.loads((ROOT / "experiments/encoder_followup_selection.json").read_text(encoding="utf-8"))
    chosen = selection["selected"]
    assert chosen and chosen["passed"] and all(chosen["checks"].values())
    tag, kind = chosen["tag"], chosen["kind"]
    audit_path = ROOT / "experiments" / f"{tag}_audit.json"
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == chosen["cv_audit_sha256"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["complete_folds"] == list(range(5))
    snapshot = json.loads((ROOT / "experiments" / f"{tag}_screen_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["screen_gate_pass"]
    fresh = json.loads((ROOT / "experiments" / chosen["fresh_file"]).read_text(encoding="utf-8"))
    assert fresh["delta_vs_v12"] >= -.002
    artifact = ROOT / "experiments/artifacts/ranker_v13_followup"
    if kind == "expanded":
        stage("runtime", ["-m", "tools.check_encoder_runtime", "--tag", tag, "--fold", "3"])
        if not (artifact / "encoder_views.json").exists():
            stage("train", ["-m", "tools.train_encoder_final", "--out", artifact, "--max-groups", "100000"])
    elif kind in {"seed_average", "seed_probability"}:
        if kind == "seed_average":
            stage("runtime", ["-m", "tools.check_encoder_runtime", "--tag", tag, "--fold", "3"])
        else:
            stage("runtime", ["-m", "tools.check_seed_probability_runtime", "--fold", "3"])
        if not (artifact / "encoder_views.json").exists():
            stage("train", ["-m", "tools.train_seed_average_final", "--out", artifact, "--variant", kind])
    elif kind == "confidence":
        stage("runtime", ["-m", "tools.check_encoder_confidence_runtime", "--fold", "0"])
        source = ROOT / "experiments/artifacts/ranker_v12_encoder"
        artifact.mkdir(parents=True, exist_ok=True)
        spec = json.loads((source / "encoder_views.json").read_text(encoding="utf-8"))
        ensemble = json.loads((source / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["encoder_views.json", "ensemble.json", "mappers.joblib"]+[m["file"] for m in ensemble["members"]]:
            shutil.copyfile(source / name, artifact / name)
        for view in spec["views"]:
            shutil.copytree(source / view["dir"], artifact / view["dir"], dirs_exist_ok=True)
    else:
        raise ValueError(kind)
    spec_path = artifact / "encoder_views.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec.update(weight=.4 if kind == "confidence" else .25, variant=kind,
                validation_sha256=chosen["cv_audit_sha256"], reference="v12",
                modification={"expanded": "all eligible groups, one epoch", "confidence": "unchanged v12 weights; fixed 40% neural mixture",
                              "seed_average": "fixed 50/50 merged parameter mean, seeds42/2026", "seed_probability": "fixed Qwen/seed42/seed2026 probabilities 2/1/1"}[kind])
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == expected, name
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    stage("build", ["tools/build_package.py", "v13_followup", "--hangul-mode", "ranker", "--artifacts", artifact,
                    "--llm-dir", "D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505"])
    package = ROOT / "submit/v13_followup"
    smoke = stage("package", ["tools/make_submit.py", package, "--n", "3000", "--subset", "300", "--singles", "3"])
    target = stage("targeted", ["-m", "tools.check_package_neural", package])
    seconds = re.search(r"형식 확인, 실행 ([0-9.]+)초", smoke.read_text(encoding="utf-8"))
    assert seconds
    lines = [s for s in target.read_text(encoding="utf-8").splitlines() if s.startswith('{"status":')]
    branch = json.loads(lines[-1])
    assert branch["status"] == "PASS"
    signatures = [v["kind"]+":"+v["recipe"] for v in spec["views"]]
    expected_views = {v["kind"]+":"+v["recipe"]+(":"+v["dir"] if signatures.count(v["kind"]+":"+v["recipe"]) > 1 else "") for v in spec["views"]}
    assert set(branch["view_calls_full"]) == expected_views
    assert "callsign_map.json" not in (package / "script.py").read_text(encoding="utf-8")
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((package / "model/src" / name).read_bytes()).hexdigest() == expected, name
    for view in spec["views"]:
        assert (package / "model" / view["dir"] / "LICENSE").exists()
    archive = ROOT / "dist/v13_followup.zip"
    assert archive.stat().st_size < 10_000_000_000
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names))
        assert not any("training_adapter" in n or n.endswith("callsign_map.json") for n in names)
        assert zipped.testzip() is None
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(16*1024**2), b""):
            digest.update(block)
    result = {"zip": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size, "files": len(names),
              "sha256": digest.hexdigest(), "zip_crc": "PASS", "tag": tag, "variant": kind,
              "cv": audit["candidate"], "reference_v12": audit["reference_v12"],
              "delta_vs_v12": audit["delta_vs_v12"], "bootstrap_95_vs_v12": audit["bootstrap_95_vs_v12"],
              "delta_by_fold_vs_v12": audit["delta_by_fold_vs_v12"], "fresh": fresh,
              "package_execution": {"status": "PASS", "rows": 3000, "seconds": float(seconds.group(1)),
                                    "subset_rows": 300, "single_rows": 3, "sample_kind": "train-derived execution smoke test"},
              "neural_branch_execution": branch, "lookup_map": False, "training_adapter_packaged": False,
              "actual_competition_submission": False, "public_score": None,
              "reference_public_v12": .9065951735, "reference_server_seconds_v12": 591, "target_score": .97,
              "caveat": "Reused exploratory CV, historical v8 mapper exception; new Public and server time unmeasured."}
    (ROOT / "experiments/release_v13_followup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
