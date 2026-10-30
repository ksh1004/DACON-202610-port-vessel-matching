"""Collect input-audit evidence and rejected explicit-format BGE models."""
import datetime
import json

import joblib
import numpy as np
import torch

from src.common import ROOT
from src.encoder_explicit import MODES
from tools.explicit_input_experiment import TAG,FOLDER,PROTOCOL,CONTROL,check,fingerprint,model_folder
from tools.calibration_experiment import sha,write


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    p=check();audit=read(f"experiments/{TAG}_audit.json");selection=read(f"experiments/{TAG}_selection.json")
    assert audit["selected"] is selection["selected"] is None and selection["stage"]=="screen"
    assert audit==read(f"experiments/{TAG}_screen_locked.json")
    assert all(not v["screen_gate_pass"] for v in audit["contenders"].values())
    for mode,item in audit["contenders"].items():
        assert item["screen_checks"]["bootstrap_lower_positive"]==(item["bootstrap_95_vs_v12"][0]>0)
    runtime=read(f"experiments/{TAG}_runtime_checks.json")
    assert runtime["status"]=="PASS" and len(runtime["checks"])==4
    process={kind:read(f"experiments/{TAG}_{kind}.json") for kind in ("process","runtime_process")}
    assert all(v["returncode"]==0 for v in process.values())
    inputs=read("experiments/token_budget_20261030.json")
    assert inputs["script_sha256"]==sha(ROOT/"tools/audit_token_budget.py")
    for file,h in inputs["source_hashes"].items():assert sha(ROOT/file)==h,file
    total=sum(v["pairs"] for v in inputs["reports"].values())
    assert total==15756
    assert all(v["qwen_truncated_pairs"]==v["bge_truncated_pairs"]==0 for v in inputs["reports"].values())
    training=[];checkpoint_checks=[];model_hashes={};lengths={}
    for fold in (1,4):
        data=joblib.load(FOLDER/f"fold{fold}_data.joblib")
        olddata=joblib.load(ROOT/f"experiments/artifacts/encoder_list20_v1/fold{fold}_data.joblib")
        oldlength=[len(t) for group in olddata["groups"][8] for t in group]
        lengths[str(fold)]=dict(legacy=dict(min=int(min(oldlength)),median=float(np.median(oldlength)),max=int(max(oldlength))),
                               new=data["report"]["token_lengths"])
        assert data["report"]["outer_vessel_overlap"]==0 and len(data["selected"])==1200
        control=json.loads((CONTROL/f"fold{fold}_training.json").read_text())
        for mode in MODES:
            record=json.loads((FOLDER/f"fold{fold}_{mode}_training.json").read_text())
            assert record["fingerprint"]==fingerprint() and record["steps"]==600
            assert record["finite_losses_and_gradients"] and record["initial_trainable_sha256"]==control["initial_trainable_sha256"]
            assert record["token_sha256"]==data["token_sha256"][mode]
            training.append(record)
            for file in model_folder(fold,mode).glob("*"):
                if file.is_file():model_hashes[str(file.relative_to(ROOT))]=sha(file)
            for slot in (0,1):
                folder=FOLDER/mode/"width8"/f"fold{fold}_checkpoint{slot}"
                ready=json.loads((folder/"ready.json").read_text())
                state=torch.load(folder/"state.pt",map_location="cpu",weights_only=False)
                assert state["fingerprint"]==ready["fingerprint"]==fingerprint()
                assert state["completed"]==ready["step"] and state["width"]==8 and state["fold"]==fold
                assert state["initial_hash"]==record["initial_trainable_sha256"]
                assert len(state["losses"])==state["completed"]
                for entry in state["optimizer"]["state"].values():
                    for value in entry.values():
                        if torch.is_tensor(value):assert torch.isfinite(value).all()
                checkpoint_checks.append(dict(fold=fold,mode=mode,slot=slot,step=ready["step"],cpu_optimizer_and_rng_state_read=True,
                                              optimizer_values_finite=True,gpu_resume_replay_tested=False))
                del state
        del data,olddata
    frozen=read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name,h in frozen["source_hashes"].items():assert sha(ROOT/"src"/name)==h,name
    release=read("experiments/release_v12_encoder.json");archive=ROOT/release["zip"]
    assert archive.stat().st_size==release["bytes"] and sha(archive)==release["sha256"]
    for file,h in read("experiments/observation_redesign_sources_20261029.json")["artifacts_sha256"].items():
        assert sha(ROOT/file)==h,file
    result=dict(client_date="2026-10-30",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=None,
                protocol_fingerprint=fingerprint(),selection=selection,audit=audit,runtime_checks=runtime,processes=process,
                input_only_audit=inputs,paired_training_input_lengths=lengths,training=training,checkpoint_checks=checkpoint_checks,
                saved_model_files_sha256=model_hashes,source_count=len(p["source_hashes"]),
                reused_known_control=True,new_control_fit=False,initial_screen_criteria_fixed_before_new_fit=True,
                frozen_v12_source_modules_verified=len(frozen["source_hashes"]),v12_zip_sha256_verified=release["sha256"],
                old_research_snapshot_preserved=True,actual_best_public=.9065951735,actual_server_seconds=591,
                actual_submissions=9,target_public=.97,new_final_model=False,new_zip=False,actual_competition_submission=False,
                callsign_or_identity_lookup=False,fresh_candidate_validation=False,full_five_fold_candidate_validation=False,
                caveats=["No truncation found in the inspected pairs; the new representation is not a proven repair of missing input.",
                         "Reused synthetic CV, known earlier control and historical reference retrieval mapper exception remain.",
                         "Upfront smaller-gain screen adds both-fold consistency, zero clean loss and positive bootstrap lower bound; does not relax Public0.97 target.",
                         "Checkpoint CPU validation is not a completed GPU resume/retraining reproducibility test.",
                         "Saved-model query checks do not constitute an end-to-end new submission ZIP execution."])
    path=ROOT/"experiments/explicit_input_iteration_20261030.json";write(result,path)
    write(dict(client_date="2026-10-30",status="completed_rejected",active_jobs=[],selected=None,
               final_audit=str(path.relative_to(ROOT)),do_not_restart_completed_frozen_screens=True),ROOT/"experiments/explicit_input_resume_20261030.json")
    print(json.dumps(dict(artifact=str(path),selected=None,neural_models=4,optimizer_updates=sum(t["steps"] for t in training),
                         source_count=result["source_count"],input_pairs_audited=total,baseline_v12_preserved=True),indent=2),flush=True)


if __name__=="__main__":main()
