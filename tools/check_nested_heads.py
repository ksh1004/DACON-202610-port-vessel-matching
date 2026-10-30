"""Saved-head and row/candidate independence checks with frozen expert scores."""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT
from tools import nested_combiner_experiment as base
from tools import reverse_ranker_experiment as reverse


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",default="nested_combiner_v1");args=p.parse_args()
    folder=ROOT/"experiments/artifacts"/args.tag
    inverse=joblib.load(folder/"pair_scores.joblib")["scores"] if args.tag=="nested_combiner_reverse_v1" else None
    reports=[]
    for outer in (0,2):
        data=base.prepare_outer(outer)
        q=joblib.load(base.FOLDER/f"outer{outer}_qwen_scores.joblib")["scores"]
        b=joblib.load(base.FOLDER/f"outer{outer}_bge_scores.joblib")["scores"]
        prior=joblib.load(folder/f"fold{outer}_token_prior.joblib") if args.tag=="nested_combiner_token_prior_v1" else None
        saved=joblib.load(folder/f"fold{outer}_results.joblib")

        def features(g,qq,bb,width):
            x=base.neural_features(g,qq,bb)
            if width==111:x=np.column_stack((x,reverse.features(g,inverse)[:,78:]))
            elif width==115:x=np.column_stack((x,prior.features(g["row"],g["names"])))
            return x[:,:width]

        for recipe in ("sets","controls") if args.tag=="nested_combiner_v1" else ("sets",):
            model=lgb.Booster(model_file=str(folder/f"fold{outer}_{recipe}_head.txt"))
            width=model.num_feature()
            matrices=[];expected=[];checks=0;index=0
            for s,groups in data["sets"].items():
                for g,row in zip(groups,saved[recipe][s]):
                    qq,bb=q[index],b[index];index+=1
                    assert g["sample_id"]==row["sample_id"] and g["truth"]==row["truth"]
                    if not g["actual_hangul"]:continue
                    x=features(g,qq,bb,width)
                    matrices.append(x);expected.append(row["scores"])
                    if checks<32:
                        perm=np.random.default_rng(20261029+checks).permutation(len(g["candidates"]))
                        changed={**g,"raw":g["raw"][perm],"proxy":g["proxy"][perm],
                                 "candidates":g["candidates"][perm],"names":[g["names"][i] for i in perm]}
                        np.testing.assert_array_equal(features(changed,qq[perm],bb[perm],width),x[perm])
                        checks+=1
            x=np.vstack(matrices);expected=np.concatenate(expected)
            actual=model.predict(x,num_threads=4)
            np.testing.assert_array_equal(actual,expected)
            perm=np.random.default_rng(20261029).permutation(len(x))
            np.testing.assert_array_equal(model.predict(x[perm],num_threads=4),actual[perm])
            np.testing.assert_array_equal(model.predict(x[:1],num_threads=1),actual[:1])
            reports.append({"outer":outer,"recipe":recipe,"features":width,"candidate_rows":len(x),
                            "exact_save_reload":True,"exact_row_shuffle_and_single":True,"exact_candidate_permutation_groups":checks})
    report={"status":"PASS","tag":args.tag,"checks":reports,
            "scope":"Cached expert scores and head trees; not an end-to-end submission/package test"}
    (ROOT/"experiments"/f"{args.tag}_head_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
