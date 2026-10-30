"""Replay actual new-format BGE models and their final v12 combinations."""
import gc
import json

import joblib
import numpy as np

from src.common import ROOT,load_catalog
from src.encoder_explicit import MODES,encode_pairs,ExplicitScorer
from src.llm_cache import BeamCache
from src.pair_views import mix_views
from src.expert_calibration import positive_probability
from tools.explicit_input_experiment import TAG,FOLDER,PROTOCOL,prepare,check,fingerprint,model_folder
from tools.calibration_experiment import write


def main():
    import torch
    check();catalog=load_catalog().to_dict("records");beams=BeamCache(allow_generate=False);reports=[]
    for fold in (1,4):
        data=prepare(fold);saved=joblib.load(FOLDER/f"fold{fold}_results.joblib")
        old=joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
        qwen=joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
        for mode in MODES:
            scores=joblib.load(FOLDER/f"fold{fold}_{mode}_scores.joblib")["sets"]
            active=[(s,i,g) for s,rows in old["sets"].items() for i,g in enumerate(rows) if scores[s][i] is not None]
            selected=[active[j] for j in (0,len(active)//2,len(active)-1)]
            model=ExplicitScorer(model_folder(fold,mode),mode);previous=[]
            for s,i,g in selected:
                row=data["evaluation"][s][i]["row"];choices=[catalog[int(j)] for j in g["candidates"][g["keep"]]]
                hints=beams([row["vessel_name"]])[0];raw=model.score(row,choices,hints)
                np.testing.assert_array_equal(raw,scores[s][i])
                np.testing.assert_array_equal(model.score(row,choices[::-1],hints),raw[::-1])
                altered=dict(row,sample_id="ignored",target_vessel_id="ignored",call_sign="ignored")
                altchoices=[dict(c,vessel_id="ignored",call_sign="ignored") for c in choices]
                assert encode_pairs(model.tokenizer,altered,altchoices,hints,mode)==encode_pairs(model.tokenizer,row,choices,hints,mode)
                bp=positive_probability(g["base_scores"][g["keep"]])
                final=mix_views(bp,[qwen["sets"][s][i]["neural_scores"],raw],[1,1])
                assert int(g["candidates"][g["keep"]][np.argmax(final)])==saved["sets"][s][i]["predictions"][mode]
                previous.append((row,choices,hints,raw))
            for row,choices,hints,raw in reversed(previous):np.testing.assert_array_equal(model.score(row,choices,hints),raw)
            row,choices,hints,raw=previous[1];np.testing.assert_array_equal(model.score(row,choices,hints),raw)
            reports.append(dict(fold=fold,mode=mode,actual_queries=3,saved_scores_exact=True,final_predictions_equal=True,
                                query_reverse_subset_single_exact=True,candidate_order_exact=True,identity_inputs_ignored=True))
            del model;gc.collect();torch.cuda.empty_cache()
    report=dict(status="PASS",fingerprint=fingerprint(),checks=reports,
                scope="Four actual saved BGE models with new formatting, frozen candidates/beams and oldQwen/base; not a full new ZIP execution")
    write(report,ROOT/"experiments"/f"{TAG}_runtime_checks.json");print(json.dumps(report),flush=True)


if __name__=="__main__":main()
