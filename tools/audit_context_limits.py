"""Post-result diagnostic, never an achieved score or a selected configuration."""
import json

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.context_density import CLIP, TEMPER
from src.experiment import TESTLIKE_MIX
from src.validate import load_folds
from tools.blend_matching import lightweight_lab
from tools.context_density_experiment import FOLDER, TAG, PROTOCOL, check, evaluation, load_models, sha


def main():
    check()
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    lab=lightweight_lab(archived["identity"]);reference=archived["ensemble"]
    oracle={s:v.copy() for s,v in reference.items()}
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in reference}
    cases=[];counts={s:{} for s in reference};totals={s:0 for s in reference}
    for fold in (1,4):
        data=evaluation(fold);codec,models=load_models(fold)
        result=joblib.load(FOLDER/f"fold{fold}_results.joblib")
        for s,rows in data["sets"].items():
            for index,g in enumerate(rows):
                if not g["actual_hangul"]:continue
                truth=g["truth"];winner=int(np.argmax(g["anchor"]));cands=g["candidates"]
                if int(cands[winner])==truth:continue
                totals[s]+=1;correct=np.flatnonzero(cands==truth)
                record=dict(fold=fold,source=s,sample_id=g["sample_id"],
                            baseline_name=g["choices"][winner]["vessel_name"],observed_name=g["row"]["vessel_name"])
                if not len(correct):
                    reason="truth_outside_frozen_top20"
                else:
                    target=int(correct[0]);x=codec.features(g["row"],g["choices"])
                    same=np.all(x==x[target],axis=1)
                    # A metadata function must assign the same score to identical profiles.
                    # Give the truth profile its maximum allowed multiplier, all others minimum.
                    ideal=g["anchor"]*np.exp(np.where(same,TEMPER*CLIP,-TEMPER*CLIP))
                    predicted=int(cands[np.argmax(ideal)])
                    if int(cands[winner]) in cands[same]:
                        reason="baseline_winner_and_truth_identical_encoded_metadata"
                    elif int(cands[np.argmax(np.where(same,g["anchor"],-1))])!=truth:
                        reason="another_candidate_beats_truth_with_identical_metadata"
                    elif predicted!=truth:
                        reason="fixed_correction_bound_prevents_recovery"
                    else:
                        reason="recoverable_by_label_aware_metadata_oracle"
                    if predicted==truth:
                        oracle[s][positions[s][g["sample_id"]]]=truth
                    raw=result["sets"][s][index]["scores"]
                    actual=result["sets"][s][index]["predictions"]["association"]
                    record.update(truth_name=g["choices"][target]["vessel_name"],
                                  actual_candidate_correct=actual==truth,
                                  identical_truth_profile_candidates=int(same.sum()),
                                  baseline_truth_probability=float(g["anchor"][target]),
                                  baseline_winner_probability=float(g["anchor"][winner]),
                                  learned_logit_lift_truth=float(raw["conditional"][target]-raw["marginal"][target]),
                                  learned_logit_lift_winner=float(raw["conditional"][winner]-raw["marginal"][winner]))
                record["reason"]=reason;cases.append(record)
                counts[s][reason]=counts[s].get(reason,0)+1
    folds=load_folds(lab.ctx.train);scope=np.isin(folds,[1,4]);h=lab.ctx.hangul
    truth={"oof":lab.ctx.truth,**{s:d.truth for s,d in lab.sets.items()}}
    def metrics(pred):
        correct={s:pred[s]==truth[s] for s in pred}
        parts={"clean":float(correct["oof"][scope&h].mean())}
        english=float(correct["oof"][scope&~h].mean())
        frame=lab.sets["testlike"].df
        for s in ("gt_missing","drop_word","both"):
            mask=frame.scenario.eq(s).to_numpy() & scope[frame.origin.to_numpy()]
            parts[s]=float(correct["testlike"][mask].mean())
        return dict(testlike=.7*sum(TESTLIKE_MIX[s]*v for s,v in parts.items())+.3*english,scenarios=parts)
    report=dict(tag=TAG,diagnostic_registered_before_performance=False,
                role="Post-result explanation only; label-aware optimistic oracle is not a learned score, Public forecast, or a bound for other models",
                folds=[1,4],v12=metrics(reference),
                ideal_truth_profile_maximum_allowed_correction=metrics(oracle),
                baseline_actual_hangul_error_rows=totals,error_reasons=counts,
                cases=cases,source_sha256=sha(__import__("pathlib").Path(__file__)),protocol_sha256=sha(PROTOCOL),
                caveat="Each query independently receives impossible knowledge of its true profile. Original correct rows are preserved by an oracle. Only frozen top20, coarse encoded profiles, temper0.5 and cliplog4 are analyzed. This is a diagnostic ceiling for this fixed mechanism, not an information-theoretic or competition ceiling.")
    (ROOT/"experiments"/f"{TAG}_limits.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"},indent=2),flush=True)


if __name__=="__main__":main()
