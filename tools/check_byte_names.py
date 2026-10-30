"""Setup and actual saved-model checks for the byte-name experiment."""
import argparse
import hashlib
import json

import joblib
import numpy as np

from src.byte_name_model import ByteNameScorer, fragments, normalise, probability
from src.common import ROOT
from tools.byte_names_experiment import WEIGHT, check_lock, evaluation, fingerprint
from tools.prepare_byte_names import FOLDER, TAG, prepare


def setup():
    protocol = check_lock()
    assert json.loads((ROOT/"experiments/byte_names_v1_preflight.json").read_text())["status"] == "PASS"
    for name in ("LUNA", "SS LUNA", "A A A"):
        group = fragments(name)
        assert abs(sum(p for s, p in group)-1)<1e-12
        assert len(group)==len({s for s,p in group})
        assert abs(float(np.logaddexp.reduce([np.log(p) for s,p in group])))<1e-12
    checks = []
    for fold in protocol["screen_folds"]:
        d = prepare(fold)
        record = {k: v for k, v in d.items() if k != "data_sha256"}
        assert hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest() == d["data_sha256"]
        report = d["report"]
        fit, valid, outer = (set(report[k]) for k in ("fit_vessels","valid_vessels","outer_vessels"))
        assert not fit & valid and not outer & (fit|valid)
        assert not {(r["query"],r["target"]) for r in d["fit"]} & {(r["query"],r["target"]) for r in d["valid"]}
        assert len(d["full"])>=500 and len(d["fit"])>=400 and len(d["valid"])>=100
        assert all(set(r)=={"query","target","weight"} for r in d["full"])
        ev = evaluation(fold); assert ev["fingerprint"] == fingerprint()
        count = skipped = 0
        for rows in ev["sets"].values():
            for g in rows:
                count += g["active"]
                if g["actual_hangul"] and not g["active"] and len(g["anchor"])>1:
                    top = np.argsort(g["anchor"])[-2:]
                    assert (1-WEIGHT)*g["anchor"][top[-1]] > (1-WEIGHT)*g["anchor"][top[-2]]+WEIGHT
                    skipped += 1
        checks.append({"fold":fold,"pairs":len(d["full"]),"inner_fit":len(d["fit"]),"inner_valid":len(d["valid"]),
                       "vessel_overlap":0,"inner_pair_overlap":0,"old_v12_winners_all_reproduced":True,
                       "byte_active_queries":int(count),"proven_winner_preserving_skips":int(skipped)})
    result = {"status":"PASS","fingerprint":fingerprint(),"only_name_strings_in_neural_examples":True,
              "fragment_prior_normalised":True,"folds":checks}
    (ROOT/"experiments"/f"{TAG}_setup_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


def runtime(folds):
    check_lock(); reports=[]
    for fold in folds:
        data=evaluation(fold)
        saved=joblib.load(FOLDER/f"fold{fold}_results.joblib")
        active=[(s,i,g) for s,rows in data["sets"].items() for i,g in enumerate(rows) if g["active"]]
        chosen=[active[i] for i in (0,len(active)//2,len(active)-1)]
        model=ByteNameScorer(FOLDER/f"fold{fold}_full/model")
        original=[]
        for s,i,g in chosen:
            scores=model.score(g["name"],g["names"])
            np.testing.assert_array_equal(scores,saved["sets"][s][i]["scores"])
            perm=np.arange(len(g["names"]))[::-1]
            np.testing.assert_array_equal(model.score(g["name"],[g["names"][int(j)] for j in perm]),scores[perm])
            values=(1-WEIGHT)*g["anchor"]+WEIGHT*probability(scores)
            assert int(g["candidates"][np.argmax(values)])==saved["sets"][s][i]["predictions"]["residual30"]
            original.append((g,scores))
        for g,scores in reversed(original): np.testing.assert_array_equal(model.score(g["name"],g["names"]),scores)
        g,scores=original[1];np.testing.assert_array_equal(model.score(g["name"],g["names"]),scores)
        reports.append({"fold":fold,"actual_queries":3,"saved_scores_exact":True,"final_predictions_equal":True,
                        "row_reverse_subset_single_exact":True,"candidate_permutation_exact":True})
        del model
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
    result={"status":"PASS","fingerprint":fingerprint(),"checks":reports,
            "scope":"Actual saved byte model plus frozen per-query anchor/candidates; not a complete new submission ZIP"}
    (ROOT/"experiments"/f"{TAG}_runtime_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--runtime",action="store_true");p.add_argument("--folds",nargs="+",type=int,default=[1,4]);a=p.parse_args()
    runtime(a.folds) if a.runtime else setup()
