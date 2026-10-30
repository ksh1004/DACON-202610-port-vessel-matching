"""Build and verify a local candidate ZIP after complete CV gates pass.

This tool never uploads or submits anything to DACON.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

from src.common import ROOT
from tools.select_encoder_iteration import select


def stage(label, arguments, timeout=2400):
    logfile = ROOT / "experiments" / f"run_v12_{label}.log"
    start = time.perf_counter()
    print(f"Starting {label}: {logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if result.returncode:
        raise RuntimeError(f"{label} failed with code {result.returncode}: {logfile}")
    print(f"Passed {label} ({time.perf_counter()-start:.1f}s)", flush=True)
    return logfile


def main():
    selection = select()
    variant = selection["selected_variant"]
    if variant is None:
        print("No variant passed the predeclared full-CV gates; preserve v10.", flush=True)
        return
    stage("runtime", ["-m", "tools.check_encoder_runtime", "--fold", "3", "--variant", variant])
    artifact = ROOT / "experiments/artifacts/ranker_v12_encoder"
    spec_path = artifact / "encoder_views.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec["variant"] != variant:
            raise ValueError("Existing candidate uses a different variant")
        for name, value in spec["source_hashes"].items():
            assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == value, name
    else:
        stage("train", ["-m", "tools.train_encoder_final", "--out", str(artifact), "--variant", variant])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["validation_sha256"] = selection["cv_audit_sha256"]
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    stage("build", ["tools/build_package.py", "v12_encoder", "--hangul-mode", "ranker", "--artifacts", str(artifact),
                    "--llm-dir", "D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505"])
    package = ROOT / "submit/v12_encoder"
    smoke_log = stage("package", ["tools/make_submit.py", str(package), "--n", "3000", "--subset", "300", "--singles", "3"])
    target_log = stage("targeted", ["-m", "tools.check_package_neural", str(package)])
    seconds = re.search(r"형식 확인, 실행 ([0-9.]+)초", smoke_log.read_text(encoding="utf-8"))
    assert seconds, "Missing complete-package execution timing"
    target_lines = [line for line in target_log.read_text(encoding="utf-8").splitlines() if line.startswith('{"status":')]
    targeted = json.loads(target_lines[-1])
    assert targeted["status"] == "PASS"
    assert set(targeted["view_calls_full"]) == {v["kind"]+":"+v["recipe"] for v in spec["views"]}
    assert not list((package / "model").rglob("callsign_map.json"))
    assert "callsign_map.json" not in (package / "script.py").read_text(encoding="utf-8")
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((package / "model/src" / name).read_bytes()).hexdigest() == expected, name
    for view in spec["views"]:
        assert (package / "model" / view["dir"] / "LICENSE").is_file()
    archive = ROOT / "dist/v12_encoder.zip"
    assert archive.stat().st_size < 10*1024**3
    with zipfile.ZipFile(archive) as zipped:
        files = zipped.namelist()
        assert len(files) == len(set(files))
        assert not any("training_adapter" in name or name.endswith("callsign_map.json") for name in files)
        assert zipped.testzip() is None
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(16*1024**2), b""):
            digest.update(block)
    audit = json.loads((ROOT / "experiments/encoder_bge_phonetic_audit.json").read_text(encoding="utf-8"))
    metrics = audit["candidate"] if variant == "encoder_only" else audit["equal_neural_ensemble"]
    confidence = audit["bootstrap_95_vs_v11"] if variant == "encoder_only" else audit["ensemble_bootstrap_95_vs_v11"]
    report = {"zip": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size, "files": len(files),
              "sha256": digest.hexdigest(), "zip_crc": "PASS", "variant": variant,
              "cv": metrics, "reference_v11": audit["reference_v11"], "bootstrap_95_vs_v11": confidence,
              "package_execution": {"status": "PASS", "rows": 3000, "seconds": float(seconds.group(1)),
                                    "subset_rows": 300, "single_rows": 3, "sample_kind": "train-derived execution smoke test"},
              "neural_branch_execution": targeted, "lookup_map": False, "training_adapter_packaged": False,
              "actual_competition_submission": False, "public_score": None, "reference_public_v11": .9039635945, "target_score": .97,
              "caveat": "Exploratory reused CV, historical v8 retrieval mapper remains shared; server timing unmeasured."}
    (ROOT / "experiments/release_v12_encoder.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
