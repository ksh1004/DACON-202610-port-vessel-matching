"""Conditional ten-feature official-training name-token prior ablation."""
import argparse
import datetime
import hashlib
import json

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT,load_catalog
from src.ensemble import normalize
from src.name_token_prior import NameTokenPrior,TOKEN_PRIOR_NAMES
from src.nested_features import NESTED_NAMES,probability
from src.train_ranker import PARAMS
from tools import nested_combiner_experiment as base
from tools.nested_reverse_extension import training

TAG="nested_combiner_token_prior_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"


def register():
    assert not PROTOCOL.exists()
    p=json.loads(base.PROTOCOL.read_text())
    p.update({"tag":TAG,"registered_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "conditional_execution":"Only if105-feature primary and111-feature reverse screens both have no qualifying contender",
              "head_features":"raw78+27 nested expert stats+10 first/last alphabetic-name-token priors,115total",
              "token_recipe":"Each fitting vessel contributes once, modal query flag/type, minimum3distinct vessels/token; catalog-normalized population prior and flag/type/joint log lift with pseudocount5; unknown tokens/categories neutral; no vessel/callsign mapping stored",
              "fitting_scope":"Inner-training official rows including English names for meta features; outer-training official rows for outer evaluation; no outer0/2 validation labels in feasibility analysis",
              "matched_control":"Previously frozen105-feature nested head, same groups/weights and anchor; no reverse features",
              "feature_source_sha256":hashlib.sha256((ROOT/"src/name_token_prior.py").read_bytes()).hexdigest(),
              "runner_source_sha256":hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()})
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")


def check():
    import pandas as pd
    catalog=pd.DataFrame({"vessel_id":["a","b","c","d"],"vessel_name":["Fleet A","Fleet B","Fleet C","Rare D"]})
    train=pd.DataFrame({"target_vessel_id":["a","a","b","c","d"],"flag_name":["X"]*5,"vessel_type":["Cargo"]*5})
    prior=NameTokenPrior().fit(train,catalog)
    assert prior.population==4 and prior.counts[0]["FLEET"][0]==3 and "RARE" not in prior.counts[0]
    row={"flag_name":"X","vessel_type":"Cargo","sample_id":"ignored","call_sign":"ignored"}
    x=prior.features(row,["Fleet A","Rare D"])
    assert x[0,0]>0 and np.all(x[1]==0)
    np.testing.assert_array_equal(prior.features(row,["Rare D","Fleet A"]),x[::-1])
    np.testing.assert_array_equal(prior.features({**row,"call_sign":"changed","target_vessel_id":"changed"},["Fleet A","Rare D"]),x)
    assert not hasattr(prior,"vessel_id") and not hasattr(prior,"callsign_map")
    print(json.dumps({"token_prior_checks":"PASS","deduplicated_vessel_support":True,"minimum3_enforced":True,
                      "permutation_exact":True,"id_inputs_ignored":True}),flush=True)


def fit():
    catalog=load_catalog()
    for outer in (0,2):
        train,folds,split,*_=base.frames(outer)
        groups,xs=training(outer)
        augmented=[];position=0
        for inner in (0,1):
            data=base.prepare_inner(outer,inner)
            prior=NameTokenPrior().fit(train.loc[(folds!=outer)&(split!=inner)],catalog)
            for g in data["meta"]:
                augmented.append(np.column_stack((xs[position],prior.features(g["row"],g["names"]))))
                position+=1
        assert position==len(groups)
        x=np.vstack(augmented)
        y=np.concatenate([g["candidates"]==g["truth"] for g in groups]).astype(int)
        weights=np.concatenate([np.full(len(g["candidates"]),g["weight"]) for g in groups])
        model=lgb.train(PARAMS,lgb.Dataset(x,label=y,weight=weights,group=[len(g["candidates"]) for g in groups],
                                          feature_name=NESTED_NAMES+TOKEN_PRIOR_NAMES),num_boost_round=150)
        model.save_model(str(FOLDER/f"fold{outer}_sets_head.txt"))
        prior=NameTokenPrior().fit(train.loc[folds!=outer],catalog)
        base.dump(prior,FOLDER/f"fold{outer}_token_prior.joblib")
        data=base.prepare_outer(outer)
        q=iter(joblib.load(base.FOLDER/f"outer{outer}_qwen_scores.joblib")["scores"])
        b=iter(joblib.load(base.FOLDER/f"outer{outer}_bge_scores.joblib")["scores"])
        control=joblib.load(base.FOLDER/f"fold{outer}_results.joblib")
        result={"fold":outer,"sets":{},"controls":control["sets"],"meta_groups":len(groups),"fingerprint":base.signature()}
        for s,rows in data["sets"].items():
            result["sets"][s]=[]
            for g in rows:
                qq,bb=next(q),next(b)
                values=None;pred={"anchor50":None,"direct":None}
                if g["actual_hangul"]:
                    x=np.column_stack((base.neural_features(g,qq,bb),prior.features(g["row"],g["names"])))
                    values=model.predict(x,num_threads=4)
                    bp=g["base"]/g["base"].sum()
                    anchor=.75*bp+.125*normalize(qq,"prob").astype(float)+.125*normalize(bb,"prob").astype(float)
                    pred={"direct":int(g["candidates"][np.argmax(values)]),
                          "anchor50":int(g["candidates"][np.argmax(.5*anchor+.5*probability(values))])}
                result["sets"][s].append({"sample_id":g["sample_id"],"truth":g["truth"],"predictions":pred,"scores":values,
                                           "truth_in_top20":bool(g["truth"] in g["candidates"])})
        base.dump(result,FOLDER/f"fold{outer}_results.joblib")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.register:register()
    elif a.check:check()
    else:
        for tag in ("nested_combiner_v1","nested_combiner_reverse_v1"):
            assert json.loads((ROOT/"experiments"/f"{tag}_audit.json").read_text())["selected"] is None
        protocol=json.loads(PROTOCOL.read_text())
        assert protocol["feature_source_sha256"]==hashlib.sha256((ROOT/"src/name_token_prior.py").read_bytes()).hexdigest()
        assert protocol["runner_source_sha256"]==hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
        FOLDER.mkdir(parents=True,exist_ok=True);fit()
        from tools.audit_reverse_ranker import audit
        audit(FOLDER,PROTOCOL)
