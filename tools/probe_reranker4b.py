"""A fixed48-case admission probe for a larger dedicated pretrained reranker."""
import json
import time

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from src.common import ROOT,load_train,load_catalog
from src.llm_cache import BeamCache,CACHE_PATH
from src.reranker4b_score import MODEL,REVISION,Reranker4B,encode
from src.expert_calibration import probability
from tools.blend_matching import lightweight_lab
from tools.calibration_experiment import sha,write,dump

TAG="dedicated_reranker4b_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_probe_protocol.json"


def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True);path=FOLDER/"probe_cases.joblib"
    if path.exists():return joblib.load(path)
    old=joblib.load(ROOT/"experiments/artifacts/forward_kanana_v1/probe_cases.joblib")
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib");lab=lightweight_lab(archived["identity"])
    frame=lab.sets["testlike"].df;rows=dict(zip(frame.sample_id,frame.to_dict("records")))
    positions={sid:i for i,sid in enumerate(archived["identity"]["sets"]["testlike"]["sample_ids"])}
    catalog=load_catalog().to_dict("records");beams=BeamCache(allow_generate=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="left")
    cases=[]
    for g in old:
        row=rows[g["sample_id"]];assert row["vessel_name"]==g["name"]
        assert int(g["candidates"][np.argmax(g["anchor"])])==int(archived["ensemble"]["testlike"][positions[g["sample_id"]]])
        choices=[catalog[int(j)] for j in g["candidates"]]
        hints=beams([row["vessel_name"]])[0];tokens=encode(tokenizer,row,choices,hints)
        assert len(tokens)==20
        if len(cases)<3:
            assert encode(tokenizer,dict(row,sample_id="ignored",target_vessel_id="ignored",call_sign="ignored"),
                          [dict(c,vessel_id="ignored",call_sign="ignored") for c in choices],hints)==tokens
        cases.append(dict(g,row=row,beams=hints,tokens=tokens))
    assert len(cases)==48 and len({g["truth"] for g in cases})==48
    assert sum(g["candidates"][np.argmax(g["anchor"])]==g["truth"] for g in cases)==17
    dump(cases,path);return cases


def numeric_check(model,tokens):
    scores=model.score_tokens(tokens)
    np.testing.assert_array_equal(model.score_tokens(tokens[::-1])[::-1],scores)
    for j in (0,len(tokens)//2,len(tokens)-1):np.testing.assert_array_equal(model.score_tokens([tokens[j]]),scores[j:j+1])
    return scores


def register(model,preflight):
    import datetime
    assert not PROTOCOL.exists();cases=prepare()
    files=["src/reranker4b_score.py","tools/probe_reranker4b.py","tools/download_reranker4b.py",
           "src/neural_pair.py","src/pair_evidence.py","src/expert_calibration.py","src/ensemble.py",
           "src/common.py","src/llm_cache.py",str(CACHE_PATH.relative_to(ROOT)),"src/experiment.py","src/augment.py",
           "src/validate.py","src/llm_translit.py","tools/blend_matching.py","tools/calibration_experiment.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv","experiments/encoder_bge_phonetic_predictions.joblib",
           "experiments/artifacts/forward_kanana_v1/probe_cases.joblib","experiments/forward_kanana_v1_probe_protocol.json",
           str((FOLDER/"probe_cases.joblib").relative_to(ROOT))]
    meta=json.loads((MODEL/"codex_revision.json").read_text())
    write(dict(tag=TAG,client_date="2026-10-30",registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               registered_before_new_probe_pair_scores_and_accuracy=True,
               pre_registration_numeric_checks="Official nonprobe fold3 rows and official catalog inputs without accuracy labels; fixed-shape order/single tests only",
               model=dict(repo=meta["repo"],revision=meta["revision"],license=meta["license"],dtype="BF16",weight_bytes=meta["weight_bytes"],
                          files_sha256=meta["files_sha256"]),
               scope="Larger dedicated generic pretrained reranker; not the earlier4B chat model or a retrained0.6B. No new ship data or model training.",
               cases="Exactly same48 difficult distinct-vessel cases as forward_kanana_v1; original top20 and actualv12 anchor; old fewshot exclusions and no baseline correctness filtering retained. Reused known diagnostic, not unseen Public data.",
               input="Untruncated existing v11 evidence/PREFIX/SUFFIX with query-only cached Latin hints, official metadata and current candidate; yes/no next-token logit difference",
               scoring=dict(batch_size=model.batch_size,pad_length=256,padding="left",position_ids="attention-mask cumulative, padding0",attention="sdpa",dtype="BF16",TF32=False,
                            numeric_policy="Prefer batch4; on numerical order/single failure only, retry batch1 before reading task correctness. Same weights/prompt/mix. Exact numerical checks required.",
                            fixed_logits_to_keep=1,truncation=False),
               fixed_mix=dict(actual_v12_anchor=.7,new_reranker_probability=.3),
               admission_gate=dict(net_correct_gain_min=3,lost_correct_rows_max=1,same48_required=True,exact_numeric_checks_required=True),
               deployment="BF16 weights8.04GB cannot be appended to current6.98GB ZIP under10GB. If admitted, require separate4bit numerical/performance/package/runtime validation before full release. No quantization library installed or validated yet.",
               preflight=preflight,source_hashes={f:sha(ROOT/f) for f in files},actual_competition_submission=False,callsign_or_identity_lookup=False,
               no_prompt_weight_temperature_or_case_search_after_results=True,
               research_sources=["https://huggingface.co/Qwen/Qwen3-Reranker-4B","https://huggingface.co/docs/bitsandbytes/main/en/installation"]),PROTOCOL)


def main():
    output=ROOT/"experiments"/f"{TAG}_probe.json";assert not output.exists(),"Completed fixed probe must not be resampled"
    cases=prepare();meta=json.loads((MODEL/"codex_revision.json").read_text());assert meta["revision"]==REVISION
    for name,h in meta["files_sha256"].items():assert sha(MODEL/name)==h,name
    model=Reranker4B(batch_size=4)
    train=load_train();folds=__import__("src.validate",fromlist=["load_folds"]).load_folds(train)
    unit_rows=train.loc[folds==3].head(3).to_dict("records");catalog=load_catalog().head(3).to_dict("records")
    unit=[t for row in unit_rows for t in encode(model.tokenizer,row,catalog,())]
    try:numeric_check(model,unit)
    except AssertionError:
        model.batch_size=1;numeric_check(model,unit)
    preflight=dict(status="PASS",official_nonprobe_pairs=len(unit),selected_batch=model.batch_size,exact_order_single=True,
                   maximum_probe_tokens=max(len(t) for g in cases for t in g["tokens"]))
    if not PROTOCOL.exists():register(model,preflight)
    p=json.loads(PROTOCOL.read_text())
    for f,h in p["source_hashes"].items():assert sha(ROOT/f)==h,f
    model.batch_size=p["scoring"]["batch_size"]
    cache=FOLDER/"probe_scores.joblib";scored=joblib.load(cache) if cache.exists() else dict(fingerprint=sha(PROTOCOL),batch=model.batch_size,scores=[])
    assert scored["fingerprint"]==sha(PROTOCOL);model.batch_size=scored["batch"]
    start=time.perf_counter()
    def complete():
        for i in range(len(scored["scores"]),len(cases)):
            scored["scores"].append(model.score_tokens(cases[i]["tokens"]))
            if (i+1)%8==0 or i+1==len(cases):
                dump(scored,cache);print(f"Dedicated4B probe {i+1}/48 seconds{time.perf_counter()-start:.1f}",flush=True)
    complete()
    try:
        for i in (0,23,47):
            np.testing.assert_array_equal(numeric_check(model,cases[i]["tokens"]),scored["scores"][i])
    except AssertionError:
        assert model.batch_size==4
        model.batch_size=1;numeric_check(model,unit)
        scored=dict(fingerprint=sha(PROTOCOL),batch=1,scores=[]);complete()
        for i in (0,23,47):np.testing.assert_array_equal(numeric_check(model,cases[i]["tokens"]),scored["scores"][i])
    old=[];new=[];details=[]
    for g,raw in zip(cases,scored["scores"]):
        before=int(g["candidates"][np.argmax(g["anchor"])])
        after=int(g["candidates"][np.argmax(.7*g["anchor"]+.3*probability(raw))])
        old.append(before==g["truth"]);new.append(after==g["truth"])
        details.append(dict(sample_id=g["sample_id"],before=before,after=after,truth=int(g["truth"])))
    old,new=np.array(old),np.array(new);gain=int((~old&new).sum());lost=int((old&~new).sum());net=gain-lost
    report=dict(tag=TAG,rows=48,distinct_vessels=48,reference_correct=int(old.sum()),candidate_correct=int(new.sum()),
                gained=gain,lost=lost,net_gain=net,admitted=net>=3 and lost<=1,seconds=time.perf_counter()-start,
                actual_batch=scored["batch"],exact_order_single_checks=True,details=details,
                max_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),model_revision=REVISION,
                actual_competition_submission=False,new_model_training=False,diagnostic_not_competition_score=True)
    write(report,output);print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2),flush=True)


if __name__=="__main__":main()
