"""Conditional six-feature reverse-name addition to the frozen nested head."""
import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT
from src.nested_features import NESTED_NAMES,probability
from src.ensemble import normalize
from src.train_ranker import PARAMS
from tools import nested_combiner_experiment as base
from tools import reverse_ranker_experiment as reverse

TAG="nested_combiner_reverse_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"


def register():
    assert not PROTOCOL.exists()
    p=json.loads(base.PROTOCOL.read_text())
    p.update({"tag":TAG,"conditional_execution":"Only after both105-feature nested contenders fail screen",
              "head_features":"raw78 +27 nested expert statistics +6 frozen complete-name Kanana likelihood features,111total",
              "matched_control":"Previously frozen105-feature nested head, same out-of-sample groups/weights and anchor",
              "reverse_recipe":"Exact frozen scorer/source and fixed batch32/pad128/maxTarget32 from reverse_kanana_v1; name strings only, no fitted official-label network"})
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")


def training(outer):
    groups=[];xs=[]
    for inner in (0,1):
        d=base.prepare_inner(outer,inner)
        folder=base.FOLDER/f"outer{outer}_inner{inner}"
        q=joblib.load(folder/"meta_qwen_scores.joblib")["scores"]
        b=joblib.load(folder/"meta_bge_scores.joblib")["scores"]
        groups.extend(d["meta"])
        xs.extend(base.neural_features(g,qq,bb) for g,qq,bb in zip(d["meta"],q,b))
    return groups,xs


def all_groups(outers):
    result=[]
    for outer in outers:
        groups,_=training(outer)
        result.extend(groups)
        result.extend(g for rows in base.prepare_outer(outer)["sets"].values() for g in rows if g["actual_hangul"])
    return result


def score(outers):
    import torch
    from src.reverse_name_likelihood import ReverseNameScorer
    groups=all_groups(outers)
    pairs={(g["name"],n) for g in groups for n in g["names"]}
    path=FOLDER/"pair_scores.joblib"
    cache=joblib.load(path) if path.exists() else joblib.load(ROOT/"experiments/artifacts/reverse_kanana_partial_v1/pair_scores.joblib")
    torch.set_num_threads(4)
    model=ReverseNameScorer()
    assert cache["protocol"]==model.protocol()
    missing=sorted(pairs-cache["scores"].keys())
    lengths_q={q:len(model.target_ids(q)) for q in {q for q,c in missing}}
    lengths_c={c:len(model.prefix_ids(c)) for c in {c for q,c in missing}}
    assert all(lengths_q[q]+lengths_c[c]<=model.pad_length for q,c in missing)
    start=time.perf_counter()
    print(f"Nested reverse missing{len(missing)}, total requested{len(pairs)}",flush=True)
    for offset in range(0,len(missing),256):
        chunk=missing[offset:offset+256]
        cache["scores"].update(zip(chunk,model.score_many_pairs(chunk)))
        if offset%2048==0 or offset+256>=len(missing):
            base.dump(cache,path)
            print(f"Nested reverse {min(offset+256,len(missing))}/{len(missing)} seconds{time.perf_counter()-start:.1f}",flush=True)
    cache["requested_pairs"]=len(pairs)
    cache["last_nested_reverse_seconds"]=time.perf_counter()-start
    base.dump(cache,path)
    return cache


def fit(outers,cache):
    for outer in outers:
        groups,xx=training(outer)
        x=np.vstack([np.column_stack((a,reverse.features(g,cache["scores"])[:,78:])) for g,a in zip(groups,xx)])
        y=np.concatenate([g["candidates"]==g["truth"] for g in groups]).astype(int)
        weights=np.concatenate([np.full(len(g["candidates"]),g["weight"]) for g in groups])
        model=lgb.train(PARAMS,lgb.Dataset(x,label=y,weight=weights,group=[len(g["candidates"]) for g in groups],
                                          feature_name=NESTED_NAMES+reverse.EXTRA),num_boost_round=150)
        model.save_model(str(FOLDER/f"fold{outer}_sets_head.txt"))
        control=joblib.load(base.FOLDER/f"fold{outer}_results.joblib")
        data=base.prepare_outer(outer)
        q=iter(joblib.load(base.FOLDER/f"outer{outer}_qwen_scores.joblib")["scores"])
        b=iter(joblib.load(base.FOLDER/f"outer{outer}_bge_scores.joblib")["scores"])
        result={"fold":outer,"sets":{},"controls":control["sets"],"fingerprint":base.signature(),"meta_groups":len(groups)}
        for s,rows in data["sets"].items():
            result["sets"][s]=[]
            for g in rows:
                qq,bb=next(q),next(b)
                values=None;pred={"anchor50":None,"direct":None}
                if g["actual_hangul"]:
                    x=np.column_stack((base.neural_features(g,qq,bb),reverse.features(g,cache["scores"])[:,78:]))
                    values=model.predict(x,num_threads=4)
                    bp=g["base"]/g["base"].sum()
                    anchor=.75*bp+.125*normalize(qq,"prob").astype(float)+.125*normalize(bb,"prob").astype(float)
                    pred={"direct":int(g["candidates"][np.argmax(values)]),
                          "anchor50":int(g["candidates"][np.argmax(.5*anchor+.5*probability(values))])}
                result["sets"][s].append({"sample_id":g["sample_id"],"truth":g["truth"],"predictions":pred,"scores":values,
                                           "truth_in_top20":bool(g["truth"] in g["candidates"])})
        base.dump(result,FOLDER/f"fold{outer}_results.joblib")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");a=p.parse_args()
    if a.register:register()
    else:
        assert json.loads((ROOT/"experiments/nested_combiner_v1_audit.json").read_text())["selected"] is None
        FOLDER.mkdir(parents=True,exist_ok=True)
        cache=score([0,2]);fit([0,2],cache)
        from tools.audit_reverse_ranker import audit
        audit(FOLDER,PROTOCOL)
