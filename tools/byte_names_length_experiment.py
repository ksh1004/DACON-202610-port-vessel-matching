"""Conditional, prospectively fixed length normalisation of shared byte models."""
import argparse
import datetime
import json

import joblib
import numpy as np

from src.byte_name_length import aggregate, length_scale
from src.byte_name_model import MODEL_DIR, probability
from src.common import ROOT
from tools import byte_names_experiment as base
from tools.prepare_byte_names import prepare

TAG="byte_names_length_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"


def register():
    assert not PROTOCOL.exists()
    p=json.loads(base.PROTOCOL.read_text())
    p.update(tag=TAG,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             registered_before_any_new_training_or_outer_model_scores=False,
             registered_before_outer_performance_inspection=True,
             common_model_training_and_pair_scoring_already_active=True,
             common_primary_protocol_sha256=base.fingerprint(),
             conditional_execution="Only if raw sequence primary fails its initial two-fold screen; no fallback after a qualifying primary fails fresh/full",
             score="Each fragment logP divided by its own UTF8byte+EOS length, multiplied by median target length in that outer-training full data, then same0.5/0.5fragment marginal; compatibility score, not proper sequence probability",
             matched_control="Identical length normalisation and fixed residual0.3 applied to the same public untuned ByT5 checkpoint; no alternate coefficient",
             registration_reason="Unequal Latin target lengths and small-data inner tokenNLL motivated a single training-derived length correction; no outer performance or predictions inspected; primary unchanged")
    p["contenders"]=["residual30"]
    p["source_hashes"]={f:base.sha(ROOT/f) for f in ["src/byte_name_length.py","tools/byte_names_length_experiment.py",
                         "experiments/byte_names_v1_protocol.json","src/byte_name_model.py","tools/byte_names_experiment.py"]}
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(f"Registered conditional length recipe: {PROTOCOL}",flush=True)


def check():
    base.check_lock();p=json.loads(PROTOCOL.read_text())
    assert p["common_primary_protocol_sha256"]==base.fingerprint()
    for path,expected in p["source_hashes"].items():assert base.sha(ROOT/path)==expected,path
    return p


def run(fold):
    check();d=prepare(fold)
    pilot=base.train_phase(fold,"pilot",d,12)
    base.train_phase(fold,"full",d,pilot["selected_epoch"])
    data=base.evaluation(fold)
    trained=base.score_cache(fold,"trained",base.FOLDER/f"fold{fold}_full/model",data)
    control=base.score_cache(fold,"untrained",MODEL_DIR,data)
    scale=length_scale(d["full"])
    result={"fold":fold,"sets":{},"controls":{},"common_fingerprint":base.fingerprint(),"scale":scale}
    for source,groups in data["sets"].items():
        for recipe in ("sets","controls"):result[recipe][source]=[]
        for g in groups:
            for recipe,cache in (("sets",trained),("controls",control)):
                prediction=None;scores=None
                if g["actual_hangul"]:
                    mixed=g["anchor"]
                    if g["active"]:
                        scores=aggregate(g["name"],g["names"],cache,scale)
                        mixed=(1-base.WEIGHT)*mixed+base.WEIGHT*probability(scores)
                    prediction=int(g["candidates"][np.argmax(mixed)])
                result[recipe][source].append({"sample_id":g["sample_id"],"truth":g["truth"],
                      "predictions":{"residual30":prediction},"scores":scores,"truth_in_top20":bool(g["truth"] in g["candidates"])})
    base.dump(result,FOLDER/f"fold{fold}_results.joblib")
    print(json.dumps({"fold":fold,"training_median_target_length":scale}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");p.add_argument("--folds",type=int,nargs="+",default=[1,4]);a=p.parse_args()
    if a.register:return register()
    protocol=check()
    primary=json.loads((ROOT/"experiments/byte_names_v1_selection.json").read_text())
    assert primary["stage"]=="screen" and primary["selected"] is None,"No fallback after primary screen qualification"
    for fold in a.folds:run(fold)
    from tools.audit_reverse_ranker import audit
    report=audit(FOLDER,PROTOCOL)
    (ROOT/"experiments"/f"{TAG}_screen_locked.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    result={"selected":None,"screen_qualified":report["selected"] is not None,"stage":"screen",
            "candidate":report["selected"],"report":report,"actual_competition_submission":False}
    (ROOT/"experiments"/f"{TAG}_selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"screen_qualified":result["screen_qualified"]}),flush=True)


if __name__=="__main__":main()
