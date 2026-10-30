"""Predeclared raw78 baseline as a candidate for the data/weighting change."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import joblib

from src.common import ROOT
from tools import nested_combiner_experiment as base

TAG="nested_raw78_fallback_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"


def register():
    assert not PROTOCOL.exists()
    p=json.loads(base.PROTOCOL.read_text())
    p.update({"tag":TAG,"registered_before_training_and_new_validation":False,
              "registered_before_any_head_training_or_new_head_validation":True,
              "registered_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "conditional_execution":"If both105-feature learned-neural contenders fail; before expensive reverse extension",
              "head_features":"raw78 only, identical nested meta groups/weights as the original matched control; data/weight/learner effect, not new neural-feature effect",
              "matched_control":"Fixed v12 reference; not an identical-data feature ablation, since the candidate is that original baseline itself",
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    p["screen_gate"]["gain_vs_matched_control_min"]=.003
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")


def audit():
    assert json.loads((ROOT/"experiments/nested_combiner_v1_audit.json").read_text())["selected"] is None
    p=json.loads(PROTOCOL.read_text())
    assert p["source_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    FOLDER.mkdir(parents=True,exist_ok=True)
    for outer in (0,2):
        source=joblib.load(base.FOLDER/f"fold{outer}_results.joblib")
        reference={s:[{**r,"scores":None,"predictions":{m:None for m in p["contenders"]}} for r in rows]
                   for s,rows in source["controls"].items()}
        result={"fold":outer,"sets":source["controls"],"controls":reference,
                "meta_groups":source["meta_groups"],"fingerprint":source["fingerprint"]}
        base.dump(result,FOLDER/f"fold{outer}_results.joblib")
        shutil.copyfile(base.FOLDER/f"fold{outer}_controls_head.txt",FOLDER/f"fold{outer}_sets_head.txt")
    from tools.audit_reverse_ranker import audit as run
    return run(FOLDER,PROTOCOL)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");a=p.parse_args()
    register() if a.register else audit()
