"""Finish rejected calibration/list-width screens and preserve the v12 release."""
import datetime
import json

import joblib
import torch

from src.common import ROOT
from tools import calibration_experiment as calibration
from tools import list20_experiment as lists


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    cp=calibration.check();lp=lists.check()
    reports={};runtime={};selections={}
    for tag in (calibration.TAG,lists.TAG):
        reports[tag]=read(f"experiments/{tag}_audit.json")
        selections[tag]=read(f"experiments/{tag}_selection.json")
        assert reports[tag]["selected"] is None and selections[tag]["selected"] is None
        assert reports[tag]==read(f"experiments/{tag}_screen_locked.json")
        runtime[tag]=read(f"experiments/{tag}_runtime_checks.json")
        assert runtime[tag]["status"]=="PASS"
    fits=[]
    for fold in (0,2):
        record=read(f"experiments/artifacts/{calibration.TAG}/fold{fold}_calibrators.json")
        assert record["fingerprint"]==calibration.sha(calibration.PROTOCOL)
        assert record["training"]["outer_vessel_overlap"]==0
        assert all(v["converged"] for v in record["models"].values())
        fits.append(record)
    training=[];checkpoint_checks=[];model_hashes={};score_protocols=[]
    for fold in (1,4):
        initial=[]
        for width in lists.ARMS:
            out=lists.arm_folder(width)
            record=json.loads((out/f"fold{fold}_training.json").read_text())
            assert record["fingerprint"]==lists.fingerprint() and record["steps"]==600
            assert record["finite_losses_and_gradients"]
            initial.append(record["initial_trainable_sha256"]);training.append(record)
            raw=joblib.load(out/f"fold{fold}_epoch1_scores.joblib")["protocol"]
            assert raw["list20_fingerprint"]==lists.fingerprint() and raw["list20_width"]==width
            score_protocols.append(dict(fold=fold,width=width,
                raw_inherited_experiment=raw.get("experiment"),
                raw_inherited_training_token_groups_exactly_legacy=raw.get("training_token_groups_exactly_legacy"),
                canonical=dict(experiment=lists.TAG,training_groups=1200,candidates_per_training_group=width,
                               original8_token_prefix_exact=True,whole_legacy2400_group_dataset_replayed=False,
                               list20_fingerprint=lists.fingerprint()),
                metadata_note="The score saver retained two descriptors from the earlier preparation dataset. Canonical fields correct reporting only; raw score files, actual tokens, trained models and decisions are unchanged."))
            for file in (out/f"fold{fold}_merged").glob("*"):
                if file.is_file():model_hashes[str(file.relative_to(ROOT))]=calibration.sha(file)
            for slot in (0,1):
                path=out/f"fold{fold}_checkpoint{slot}"
                ready=json.loads((path/"ready.json").read_text())
                state=torch.load(path/"state.pt",map_location="cpu",weights_only=False)
                assert ready["fingerprint"]==state["fingerprint"]==lists.fingerprint()
                assert ready["step"]==state["completed"] and state["width"]==width and state["fold"]==fold
                assert state["initial_hash"]==record["initial_trainable_sha256"]
                for entry in state["optimizer"]["state"].values():
                    for value in entry.values():
                        if torch.is_tensor(value):assert torch.isfinite(value).all()
                assert len(state["losses"])==state["completed"] and len(state["trainable_names"])>0
                checkpoint_checks.append(dict(fold=fold,width=width,slot=slot,step=ready["step"],
                                              cpu_state_read_and_finite=True,gpu_resume_replay_tested=False))
                del state
        assert initial[0]==initial[1]
    transfer=read(f"experiments/{calibration.TAG}_transfer.json")
    assert transfer["protocol_sha256"]==calibration.sha(calibration.PROTOCOL)
    assert transfer["source_sha256"]==calibration.sha(ROOT/"tools/audit_calibration_transfer.py")
    frozen=read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name,h in frozen["source_hashes"].items():assert calibration.sha(ROOT/"src"/name)==h,name
    release=read("experiments/release_v12_encoder.json");archive=ROOT/release["zip"]
    assert archive.stat().st_size==release["bytes"] and calibration.sha(archive)==release["sha256"]
    for name,h in read("experiments/observation_redesign_sources_20261029.json")["artifacts_sha256"].items():
        assert calibration.sha(ROOT/name)==h,name
    log_bytes=(ROOT/"experiments/encoder_list20_v1_run.log").read_bytes()
    log=log_bytes.decode("utf-16" if log_bytes.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig")
    assert "Traceback (most recent call last)" not in log
    result=dict(client_date="2026-10-30",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                selected=None,selections=selections,audits=reports,runtime_checks=runtime,
                calibration_fits=fits,neural_training=training,checkpoint_checks=checkpoint_checks,
                actual_saved_model_files_sha256=model_hashes,post_result_calibration_diagnostic=transfer,
                score_protocol_reporting=score_protocols,
                main_launcher=dict(tool_reported_exit_code=1,python_exit_code_not_separately_captured=True,
                                   native_command_warning_present="NativeCommandError" in log,
                                   all_training_score_and_selection_artifacts_complete=True,
                                   independent_saved_model_runtime_process_exit_code=0,
                                   note="The PowerShell launcher recorded the fast-tokenizer padding advisory as a native-command warning. Do not claim the main Python exit code was measured as zero; independent checks and all required artifacts confirm completion."),
                verified_source_counts=dict(calibration=len(cp["source_hashes"]),list20=len(lp["source_hashes"])),
                frozen_v12_source_modules_verified=len(frozen["source_hashes"]),v12_zip_sha256_verified=release["sha256"],
                old_research_snapshot_preserved=True,actual_best_public=.9065951735,actual_server_seconds=591,
                actual_submissions=9,target_public=.97,new_final_model=False,new_zip=False,actual_competition_submission=False,
                callsign_or_identity_lookup=False,fresh_candidate_validation=False,full_five_fold_candidate_validation=False,
                caveats=["Two separate fixed experimental families, with reused synthetic CV and historical base retrieval mapper exception.",
                         "Lower conditional log loss and label-aware choice oracles do not establish improved competition accuracy.",
                         "List20 uses2.5x training pairs relative to its8-control, at equal queries/updates.",
                         "CPU checkpoint-state validation is not a completed GPU resume/retrain reproducibility test.",
                         "New saved-model checks are not a new end-to-end submission ZIP run."])
    path=ROOT/"experiments/improvement_iteration_20261030.json"
    calibration.write(result,path)
    calibration.write(dict(client_date="2026-10-30",status="completed_rejected",active_jobs=[],
                           final_audit=str(path.relative_to(ROOT)),selected=None,actual_submission=False,
                           do_not_restart_completed_frozen_screens=True),ROOT/"experiments/improvement_resume_20261030.json")
    print(json.dumps(dict(artifact=str(path),selected=None,calibrator_fits=4,new_neural_models=4,
                         optimizer_updates=sum(t["steps"] for t in training),source_counts=result["verified_source_counts"],
                         baseline_v12_preserved=True),indent=2),flush=True)


if __name__=="__main__":main()
