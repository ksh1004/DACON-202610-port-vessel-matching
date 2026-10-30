"""Collect the rejected context trial and verify the unchanged release."""
import datetime
import json

import joblib
import lightgbm as lgb

from src.common import ROOT
from tools.context_density_experiment import TAG, FOLDER, PROTOCOL, check, sha


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    protocol=check();audit=read(f"experiments/{TAG}_audit.json")
    selection=read(f"experiments/{TAG}_selection.json")
    assert selection["selected"] is None and audit["selected"] is None
    assert audit==read(f"experiments/{TAG}_screen_locked.json")
    runtime=read(f"experiments/{TAG}_runtime_checks.json")
    assert runtime["status"]=="PASS" and sum(c["actual_queries"] for c in runtime["checks"])==6
    training=[];model_hashes={}
    for fold in protocol["screen_folds"]:
        record=json.loads((FOLDER/f"fold{fold}_training.json").read_text())
        assert record["fingerprint"]==sha(PROTOCOL) and record["outer_vessel_overlap"]==0
        assert record["minimum_supported_distinct_vessels"]>=10
        codec=joblib.load(FOLDER/f"fold{fold}_codec.joblib")
        assert set(codec.__dict__)=={"maps","support_counts"}
        for name in ("conditional","permuted","marginal"):
            path=FOLDER/f"fold{fold}_{name}.txt"
            assert lgb.Booster(model_file=str(path)).num_trees()==200
            model_hashes[str(path.relative_to(ROOT))]=sha(path)
        path=FOLDER/f"fold{fold}_codec.joblib";model_hashes[str(path.relative_to(ROOT))]=sha(path)
        training.append(record)
    frozen=read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name,h in frozen["source_hashes"].items():assert sha(ROOT/"src"/name)==h,name
    release=read("experiments/release_v12_encoder.json");archive=ROOT/release["zip"]
    assert archive.stat().st_size==release["bytes"] and sha(archive)==release["sha256"]
    snapshot=read("experiments/observation_redesign_sources_20261029.json")
    for name,h in snapshot["artifacts_sha256"].items():assert sha(ROOT/name)==h,name
    limits=read(f"experiments/{TAG}_limits.json")
    assert limits["protocol_sha256"]==sha(PROTOCOL)
    assert limits["source_sha256"]==sha(ROOT/"tools/audit_context_limits.py")
    for s,counts in limits["error_reasons"].items():
        assert sum(counts.values())==limits["baseline_actual_hangul_error_rows"][s]
    result=dict(client_date="2026-10-29",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                selected=None,protocol_fingerprint=sha(PROTOCOL),source_count=len(protocol["source_hashes"]),
                training=training,model_files_sha256=model_hashes,audit=audit,runtime_checks=runtime,
                post_result_diagnostic={k:v for k,v in limits.items() if k!="cases"},
                frozen_v12_source_modules_verified=len(frozen["source_hashes"]),v12_zip_sha256_verified=release["sha256"],
                research_snapshot_preserved=True,actual_best_public=.9065951735,actual_server_seconds=591,
                actual_submissions=9,target_public=.97,new_final_model=False,new_zip=False,actual_competition_submission=False,
                fresh_candidate_validation=False,full_five_fold_candidate_validation=False,callsign_or_identity_lookup=False,
                no_other_candidate_or_coefficient_search_after_rejection=True,
                limitations=["Only repeated exploratory synthetic CV folds1/4, with historical reference retrieval mapper exception.",
                             "Context-model diagnostics use labels and cannot predict Public performance or bound other methods.",
                             "Runtime checks cover saved metadata experts and frozen anchors, not a new end-to-end submission ZIP."])
    path=ROOT/"experiments/context_iteration_20261029.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(artifact=str(path),selected=None,trained_models=6,source_count=result["source_count"],
                         baseline_preserved=True,actual_public_unchanged=result["actual_best_public"]),indent=2),flush=True)


if __name__=="__main__":main()
