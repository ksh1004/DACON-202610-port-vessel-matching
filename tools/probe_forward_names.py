"""A separate frozen-LM admission test on48distinct difficult vessels."""
import hashlib
import json
import time

import joblib
import numpy as np
import torch

from src.byte_name_model import fragments,probability
from src.common import ROOT,load_catalog,load_train
from src.forward_name_likelihood import ForwardNameScorer
from src.llm_translit import FEW_SHOT
from tools import byte_names_experiment as shared

TAG="forward_kanana_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_probe_protocol.json"


def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True)
    path=FOLDER/"probe_cases.joblib"
    if path.exists():return joblib.load(path)
    train,catalog=load_train(),load_catalog()
    ids=set(train.loc[train.vessel_name.isin({q for q,c in FEW_SHOT}),"target_vessel_id"])
    excluded=set(np.flatnonzero(catalog.vessel_id.isin(ids)))
    shared.FOLDER=FOLDER
    pool=[]
    for fold in (0,2):
        data=shared.evaluation(fold)
        for g in data["sets"]["testlike"]:
            if g["active"] and g["truth"] in g["candidates"] and g["truth"] not in excluded and g["sample_id"].endswith(("_drop_word","_both")):
                pool.append({**g,"fold":fold})
    selected=[];seen=set()
    for i in np.random.default_rng(20261029).permutation(len(pool)):
        g=pool[int(i)]
        if g["truth"] in seen:continue
        selected.append(g);seen.add(g["truth"])
        if len(selected)==48:break
    assert len(selected)==48
    shared.dump(selected,path)
    return selected


def register():
    assert not PROTOCOL.exists()
    cases=prepare()
    files=["src/forward_name_likelihood.py","tools/probe_forward_names.py","src/reverse_name_likelihood.py",
           "src/llm_translit.py","src/byte_name_model.py","tools/byte_names_experiment.py","experiments/byte_names_v1_protocol.json",
           "train.csv","vessel_catalog.csv",str((FOLDER/"probe_cases.joblib").relative_to(ROOT))]
    import datetime
    p={"tag":TAG,"registered_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"registered_before_new_teacher_scores":True,
       "scope":"Separate pretrained forward-conditional teacher; no new training or weight download; not a retuned ByT5 score",
       "probe_vessels":48,"folds":[0,2],"few_shot_vessels_excluded":True,
       "sampling":"One seeded random eligible drop_word/both case per distinct truth vessel; actual v12 residual0.3 confidence region and truth in original top20; no baseline correctness filter; no truth injection",
       "fixed_mix":{"actual_v12_top20":.7,"new_name_probability":.3},
       "name_score":"Whole name0.5, other distinct contiguous fragments uniform0.5, raw full target+turn-end logP; no length/weight/prompt search",
       "admission_gate":{"net_correct_gain_min":3,"same48cases_required":True},
       "source_hashes":{f:shared.sha(ROOT/f) for f in files},"actual_competition_submission":False,"lookup_used":False,
       "caveat":"Difficult48vessel diagnostic, not official metric or Public prediction; repeated known vessels. Probe excludes fixed fewshot vessels. Full twofold and strict guard required if admitted."}
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(f"Registered {PROTOCOL}",flush=True)


def main():
    if not PROTOCOL.exists():register()
    p=json.loads(PROTOCOL.read_text())
    for f,h in p["source_hashes"].items():assert shared.sha(ROOT/f)==h,f
    out=ROOT/"experiments"/f"{TAG}_probe.json"
    assert not out.exists(),"Do not resample a completed teacher test"
    cases=prepare();torch.set_num_threads(4)
    model=ForwardNameScorer()
    pairs=sorted({(g["name"],span) for g in cases for name in g["names"] for span,prior in fragments(name)})
    assert all(len(model.prefix_ids(q))+len(model.target_ids(s))<=model.pad_length for q,s in pairs)
    first=model.score_pairs(pairs[:17]);np.testing.assert_array_equal(first,model.score_pairs(pairs[:17][::-1])[::-1])
    for i in (0,8,16):np.testing.assert_array_equal(model.score_pairs([pairs[i]]),first[i:i+1])
    cache={};start=time.perf_counter()
    for offset in range(0,len(pairs),256):
        part=pairs[offset:offset+256]
        cache.update(zip(part,model.score_pairs(part)))
        if offset%1024==0 or offset+256>=len(pairs):
            shared.dump({"scores":cache,"protocol":model.protocol()},FOLDER/"probe_pairs.joblib")
            print(f"Forward probe pairs{min(offset+256,len(pairs))}/{len(pairs)} seconds{time.perf_counter()-start:.1f}",flush=True)
    old=[];new=[];details=[]
    for g in cases:
        scores=np.array([np.logaddexp.reduce([cache[(g["name"],s)]+np.log(prior) for s,prior in fragments(name)]) for name in g["names"]])
        before=int(g["candidates"][np.argmax(g["anchor"])])
        after=int(g["candidates"][np.argmax(.7*g["anchor"]+.3*probability(scores))])
        old.append(before==g["truth"]);new.append(after==g["truth"])
        details.append({"sample_id":g["sample_id"],"before":before,"after":after,"truth":g["truth"]})
    old,new=np.array(old),np.array(new);gain=int(new.sum()-old.sum())
    report={"tag":TAG,"rows":48,"distinct_vessels":48,"reference_correct":int(old.sum()),"candidate_correct":int(new.sum()),
            "gained":int((~old&new).sum()),"lost":int((old&~new).sum()),"net_gain":gain,
            "admitted":gain>=p["admission_gate"]["net_correct_gain_min"],"seconds":time.perf_counter()-start,
            "scorer_protocol":model.protocol(),"exact_pair_order_and_single_checks":True,
            "details":details,"actual_competition_submission":False,"diagnostic_not_competition_score":True}
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("details","scorer_protocol")},indent=2),flush=True)


if __name__=="__main__":main()
