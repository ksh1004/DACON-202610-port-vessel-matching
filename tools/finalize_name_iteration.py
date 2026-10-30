"""Collect rejected byte models and the independent forward-teacher diagnostic."""
import datetime
import hashlib
import json

from src.common import ROOT
from tools import byte_names_experiment as base
from tools import byte_names_length_experiment as length


def read(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    primary=base.check_lock();normalised=length.check()
    for f,h in read("experiments/byte_names_length_v1_execution_lock.json").items():assert base.sha(ROOT/f)==h,f
    teacher_protocol=read("experiments/forward_kanana_v1_probe_protocol.json")
    for f,h in teacher_protocol["source_hashes"].items():assert base.sha(ROOT/f)==h,f
    selections={t:read(f"experiments/{t}_selection.json") for t in ("byte_names_v1","byte_names_length_v1")}
    for t,d in selections.items():
        assert d["selected"] is None and d["stage"]=="screen"
        if "screen_qualified" in d:assert not d["screen_qualified"]
    reports={t:read(f"experiments/{t}_audit.json") for t in selections}
    for tag,report in reports.items():
        assert report==read(f"experiments/{tag}_screen_locked.json")
        assert report["complete_folds"]==[1,4] and report["selected"] is None
    runtime={t:read(f"experiments/{t}_runtime_checks.json") for t in selections}
    for r in runtime.values():assert r["status"]=="PASS" and len(r["checks"])==2
    training=[]
    for fold in (1,4):
        for phase in ("pilot","full"):
            r=read(f"experiments/artifacts/byte_names_v1/fold{fold}_{phase}/training.json")
            assert r["fingerprint"]==base.fingerprint() and r["all_losses_and_gradients_finite"]
            training.append(r)
    teacher=read("experiments/forward_kanana_v1_probe.json")
    assert teacher["rows"]==48 and teacher["distinct_vessels"]==48 and not teacher["admitted"]
    assert teacher["exact_pair_order_and_single_checks"]
    # Preserve the raw prototype record. Its parent-class metadata retained
    # reverse-direction labels; canonicalise only the reporting description here.
    actual_protocol=dict(teacher["scorer_protocol"])
    inherited={k:actual_protocol[k] for k in ("prompt_sha256","input_fields","score")}
    actual_protocol.update(prompt_sha256=hashlib.sha256(actual_protocol["prompt"].encode()).hexdigest(),
                           input_fields=["observed_korean_name","candidate_latin_fragment"],
                           score="Latin target tokens plus assistant-turn end log probability")
    frozen=read("experiments/artifacts/ranker_v12_encoder/encoder_views.json")
    for name,h in frozen["source_hashes"].items():assert base.sha(ROOT/"src"/name)==h,name
    release=read("experiments/release_v12_encoder.json");archive=ROOT/release["zip"]
    assert archive.stat().st_size==release["bytes"] and base.sha(archive)==release["sha256"]
    assert base.sha(base.MODEL_DIR/"pytorch_model.bin")==primary["model"]["weights_sha256"]
    result={"client_date":"2026-10-29","completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selected":None,"decisions":selections,"audits":reports,"runtime_checks":runtime,"training":training,
            "forward_teacher_probe":teacher,"canonical_forward_scorer_protocol":actual_protocol,
            "inherited_parent_metadata_retained_in_raw_probe":inherited,
            "metadata_note":"Parent protocol descriptors were inherited from the reverse scorer; actual forward prompt text, examples, source, and terminator were already recorded. Canonical fields above correct reporting only; no tokens, scores, cases or decisions changed.",
            "verified_source_counts":{"primary":len(primary["source_hashes"]),"length":len(normalised["source_hashes"]),
                                      "forward_probe":len(teacher_protocol["source_hashes"])},
            "frozen_v12_source_modules_verified":len(frozen["source_hashes"]),"v12_zip_sha256_verified":release["sha256"],
            "generic_byt5_weights_sha256_verified":primary["model"]["weights_sha256"],
            "actual_best_public":.9065951735,"actual_server_seconds":591,"actual_submissions":9,
            "target_public":.97,"new_final_model":False,"new_zip":False,"actual_competition_submission":False,
            "callsign_or_identity_lookup":False,"fresh_candidate_validation":False,"full_five_fold_candidate_validation":False,
            "caveats":["Reused synthetic CV; historical outer retrieval mapper exception remains.",
                       "Weak exact fragment labels and small training support do not establish correct pronunciations.",
                       "Length correction registered after common training began but before outer performance inspection; original recipe was preserved.",
                       "48vessel teacher diagnostic is not a competition accuracy estimate.",
                       "Scorer checks are not a new end-to-end submission ZIP execution."]}
    path=ROOT/"experiments/name_iteration_20261029.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"artifact":str(path),"selected":None,"trained_phases":len(training),
                      "source_counts":result["verified_source_counts"],"frozen_v12_preserved":True},indent=2))


if __name__=="__main__":main()
