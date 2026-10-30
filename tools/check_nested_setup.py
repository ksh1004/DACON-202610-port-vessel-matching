"""Leakage, ID exclusion and expert-feature invariance before head fitting."""
import json

import numpy as np
from transformers import AutoTokenizer

from src.common import ROOT,load_catalog
from src.nested_features import extend,stats,probability
from tools import nested_combiner_experiment as module


def main():
    rng=np.random.default_rng(20261029)
    raw=rng.normal(size=(20,78)).astype(np.float32)
    scores=[rng.normal(size=20) for _ in range(3)]
    x=extend(raw,*scores)
    perm=rng.permutation(20)
    np.testing.assert_array_equal(extend(raw[perm],*(s[perm] for s in scores)),x[perm])
    shifted=extend(raw,*(s+8. for s in scores))
    np.testing.assert_allclose(shifted,x,rtol=0,atol=1e-6)
    uniform=stats(np.zeros(20))
    np.testing.assert_allclose(uniform[:,0],.05)
    np.testing.assert_array_equal(uniform[:,2],np.ones(20))
    np.testing.assert_allclose(uniform[:,4],np.log(20))
    np.testing.assert_array_equal(uniform[:,-1],np.zeros(20))
    catalog=load_catalog().to_dict("records")
    reports=[]
    for outer in (0,2):
        train,folds,split,original,aug,population,outer_ids=module.frames(outer)
        for inner in (0,1):
            fit_ids=set(train.loc[(folds!=outer)&(split!=inner),"target_vessel_id"])
            held_ids=set(train.loc[(folds!=outer)&(split==inner),"target_vessel_id"])
            assert not fit_ids&held_ids and not outer_ids&(fit_ids|held_ids)
            path=module.FOLDER/f"outer{outer}_inner{inner}/data.joblib"
            if not path.exists():continue
            d=module.prepare_inner(outer,inner)
            assert all(g["row"]["target_vessel_id"] in fit_ids for g in d["training"])
            assert all(g["row"]["target_vessel_id"] in held_ids and g["truth"] in g["candidates"] for g in d["meta"])
            g=d["training"][0]
            changed={**g,"row":{**g["row"],"sample_id":"UNUSED_RECORD_MARKER","call_sign":"UNUSED_CALLSIGN_MARKER",
                                  "target_vessel_id":"UNUSED_LABEL_MARKER"}}
            for kind in ("qwen","bge"):
                tok=AutoTokenizer.from_pretrained(module.MODELS[kind],local_files_only=True)
                assert module.encode(kind,tok,g,catalog)==module.encode(kind,tok,changed,catalog)
            reports.append({"outer":outer,"inner":inner,"overlap":0,"nn_groups":len(d["training"]),"meta_groups":len(d["meta"])})
    report={"status":"PASS","feature_candidate_permutation_exact":True,"expert_shift_invariance":True,
            "uniform_confidence_statistics":True,"input_ids_and_callsign_excluded":True,"prepared_splits":reports}
    (ROOT/"experiments/nested_combiner_setup_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
