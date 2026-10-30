"""Actual saved-model replay and query independence for the 8/20 comparison."""
import gc
import json

import joblib
import numpy as np

from src.common import ROOT,load_catalog
from tools.list20_experiment import TAG,FOLDER,ARMS,arm_folder,prepare,check,fingerprint
from tools.calibration_experiment import write


def runtime():
    import torch
    from src.encoder_phonetics import EncoderPhoneticScorer,encode_pairs
    check();catalog=load_catalog().to_dict("records");checks=[]
    for fold in (1,4):
        data=prepare(fold)
        for width in ARMS:
            scores=joblib.load(arm_folder(width)/f"fold{fold}_epoch1_scores.joblib")
            active=[(s,i,g) for s,rows in scores["sets"].items() for i,g in enumerate(rows) if g["neural_scores"] is not None]
            selected=[active[i] for i in (0,len(active)//2,len(active)-1)]
            model=EncoderPhoneticScorer(arm_folder(width)/f"fold{fold}_merged");previous=[]
            for s,i,g in selected:
                row=data["evaluation"][s][i]["row"];c=g["candidates"][g["keep"]]
                choices=[catalog[int(j)] for j in c]
                # Evaluation token groups already include the frozen query-local beams.
                # Load that same immutable global beam file read-only; never generate.
                from src.llm_cache import BeamCache
                beams=BeamCache(allow_generate=False)([row["vessel_name"]])[0]
                result=model.score(row,choices,beams)
                np.testing.assert_array_equal(result,g["neural_scores"])
                np.testing.assert_array_equal(model.score(row,choices[::-1],beams),result[::-1])
                changed=dict(row,sample_id="ignored",target_vessel_id="ignored",call_sign="ignored")
                altered=[dict(v,vessel_id="ignored",call_sign="ignored") for v in choices]
                assert encode_pairs(model.tokenizer,changed,altered,beams)==encode_pairs(model.tokenizer,row,choices,beams)
                previous.append((row,choices,beams,result))
            for row,choices,beams,result in reversed(previous):np.testing.assert_array_equal(model.score(row,choices,beams),result)
            row,choices,beams,result=previous[1]
            np.testing.assert_array_equal(model.score(row,choices,beams),result)
            checks.append(dict(fold=fold,width=width,actual_queries=3,saved_scores_exact=True,
                               query_reverse_subset_single_exact=True,candidate_order_exact=True,
                               identity_fields_ignored_in_tokens=True))
            del model;gc.collect();torch.cuda.empty_cache()
    report=dict(status="PASS",fingerprint=fingerprint(),checks=checks,
                scope="Four actual saved BGE models, frozen candidates and row-local beams. No new submission ZIP executed.")
    write(report,ROOT/"experiments"/f"{TAG}_runtime_checks.json");print(json.dumps(report),flush=True)


if __name__=="__main__":runtime()
