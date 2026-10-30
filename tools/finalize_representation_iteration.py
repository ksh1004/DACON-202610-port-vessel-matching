"""Collect the completed representation trial and the fixed dedicated4B probe."""
import datetime
import importlib.util
import json

import joblib
import numpy as np

from src.common import ROOT
from src.reranker4b_score import MODEL,REVISION
from tools.calibration_experiment import sha,write
from tools import explicit_input_experiment as explicit


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    ep=explicit.check();prior=read("experiments/explicit_input_iteration_20261030.json")
    assert prior["selected"] is None and prior["runtime_checks"]["status"]=="PASS"
    assert prior["protocol_fingerprint"]==explicit.fingerprint()
    tag="dedicated_reranker4b_v1"
    protocol=ROOT/"experiments"/f"{tag}_probe_protocol.json";p=json.loads(protocol.read_text())
    for f,h in p["source_hashes"].items():assert sha(ROOT/f)==h,f
    result=read(f"experiments/{tag}_probe.json");process=read(f"experiments/{tag}_process.json")
    assert process["returncode"]==0 and not result["admitted"] and result["exact_order_single_checks"]
    assert result["rows"]==result["distinct_vessels"]==48
    assert result["reference_correct"]==17 and result["candidate_correct"]==15
    assert result["net_gain"]==result["gained"]-result["lost"]==-2
    meta=json.loads((MODEL/"codex_revision.json").read_text())
    assert meta["revision"]==p["model"]["revision"]==REVISION
    assert meta["files_sha256"]==p["model"]["files_sha256"]
    assert sum(f.stat().st_size for f in MODEL.glob("model*.safetensors"))==meta["weight_bytes"]
    scored=joblib.load(ROOT/f"experiments/artifacts/{tag}/probe_scores.joblib")
    assert scored["fingerprint"]==sha(protocol) and len(scored["scores"])==48
    assert all(np.isfinite(v).all() and len(v)==20 for v in scored["scores"])
    assert importlib.util.find_spec("bitsandbytes") is None
    frozen=read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name,h in frozen["source_hashes"].items():assert sha(ROOT/"src"/name)==h,name
    release=read("experiments/release_v12_encoder.json");archive=ROOT/release["zip"]
    assert archive.stat().st_size==release["bytes"]
    assert prior["v12_zip_sha256_verified"]==release["sha256"]
    verification_time=datetime.datetime.fromisoformat(prior["completed_utc"]).timestamp()
    assert archive.stat().st_mtime<=verification_time
    report=dict(client_date="2026-10-30",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=None,
                explicit_representation_iteration=prior,dedicated4b_probe=result,dedicated4b_protocol=p,
                dedicated4b_process=process,dedicated4b_download=meta,
                verified_source_counts=dict(explicit=len(ep["source_hashes"]),dedicated4b=len(p["source_hashes"])),
                frozen_v12_source_modules_verified=len(frozen["source_hashes"]),
                v12_archive=dict(size=archive.stat().st_size,mtime_ns=archive.stat().st_mtime_ns,
                    full_sha256_certificate="experiments/explicit_input_iteration_20261030.json",
                    sha256=release["sha256"],note="Full archive hash verified earlier in this same iteration; reused after size/mtime and current13-source checks. No baseline files were changed by the new-model probe."),
                actual_best_public=.9065951735,actual_server_seconds=591,actual_submissions=9,target_public=.97,
                new_trainable_models=4,new_optimizer_updates=2400,additional_pretrained_checkpoint_loaded=True,
                additional4b_training=False,quantization_library_installed=False,quantization_tested=False,
                new_final_model=False,new_zip=False,actual_competition_submission=False,callsign_or_identity_lookup=False,
                fresh_candidate_validation=False,full_five_fold_candidate_validation=False,
                caveats=["Input inspection found no truncation in the inspected official pairs; representation changes did not improve v12.",
                         "Dedicated4B admission uses a reused hard48-vessel diagnostic, not competition accuracy or an independent Public estimate.",
                         "No4-bit model or new package was produced. BF16 packaging budget was a size feasibility estimate, not a measured new ZIP."])
    path=ROOT/"experiments/representation_iteration_20261030.json";write(report,path)
    write(dict(client_date="2026-10-30",status="completed_rejected",active_jobs=[],selected=None,
               final_audit=str(path.relative_to(ROOT)),no_restarts_or_automatic_quantization_after_rejection=True),
          ROOT/"experiments/reranker4b_resume_20261030.json")
    print(json.dumps(dict(artifact=str(path),selected=None,source_counts=report["verified_source_counts"],
                         actual_best_public=report["actual_best_public"],new_trainable_models=4,
                         dedicated4b_correct=result["candidate_correct"],actual_submission=False),indent=2),flush=True)


if __name__=="__main__":main()
