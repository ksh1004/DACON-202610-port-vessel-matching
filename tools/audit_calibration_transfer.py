"""Post-result loss transfer and label-aware expert-choice diagnostics."""
import json
from collections import defaultdict

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.experiment import TESTLIKE_MIX
from src.expert_calibration import MODES, L2, corrected_logits, features
from src.validate import load_folds
from tools.blend_matching import lightweight_lab
from tools.calibration_experiment import TAG, FOLDER, PROTOCOL, check, evaluation, sha, write


def nll(z,target):
    z=np.asarray(z,dtype=float);v=z-z.max()
    return float(np.log(np.sort(np.exp(v)).sum())-v[target])


def main():
    check();archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    lab=lightweight_lab(archived["identity"]);reference=archived["ensemble"]
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in reference}
    oracle={k:{s:p.copy() for s,p in reference.items()} for k in (1,3,5)}
    losses=defaultdict(list);counts=defaultdict(lambda:defaultdict(int));training=[]
    for fold in (0,2):
        data=evaluation(fold)
        models=json.loads((FOLDER/f"fold{fold}_calibrators.json").read_text())
        for mode,m in models["models"].items():
            training.append(dict(fold=fold,mode=mode,objective_before=m["objective_before"],
                                 objective_after=m["objective_after"],
                                 nll_after_without_regularizer=m["objective_after"]-.5*L2*np.dot(m["coefficients"],m["coefficients"])))
        for s,rows in data["sets"].items():
            for g in rows:
                if not g["actual_hangul"]:continue
                correct=g["candidates"][np.argmax(g["anchor"])]==g["truth"]
                counts[s]["baseline_error_rows"]+=int(not correct)
                if not g["active"]:
                    counts[s]["baseline_errors_on_old_pruned_rows"]+=int(not correct)
                    continue
                counts[s]["active_rows"]+=1
                target=np.flatnonzero(g["candidates"]==g["truth"])
                if not len(target):
                    counts[s]["active_truth_outside_top20"]+=1
                    continue
                target=int(target[0]);index=positions[s][g["sample_id"]]
                counts[s]["active_truth_in_top20"]+=1
                for k in oracle:
                    available=correct or any(target in np.argsort(-p,kind="stable")[:k] for p in g["experts"])
                    if available:oracle[k][s][index]=g["truth"]
                    counts[s][f"baseline_error_recoverable_expert_top{k}"]+=int(not correct and available)
                weight=TESTLIKE_MIX[g["row"]["scenario"]] if s=="testlike" else 1.
                before=nll(np.log(np.maximum(g["anchor"],1e-15)),target)
                for mode in MODES:
                    x=features(g["anchor"],g["experts"],g["row"],mode)
                    z=corrected_logits(g["anchor"],x,models["models"][mode]["coefficients"])
                    losses[(s,mode)].append((before,nll(z,target),weight))
    outer_losses={}
    for (source,mode),values in losses.items():
        a=np.array(values);w=a[:,2]/a[:,2].sum()
        outer_losses.setdefault(source,{})[mode]=dict(rows=len(a),reference_nll=float(w@a[:,0]),candidate_nll=float(w@a[:,1]),
                                                    nll_delta=float(w@(a[:,1]-a[:,0])))
    scope=np.isin(load_folds(lab.ctx.train),[0,2]);h=lab.ctx.hangul
    truth={"oof":lab.ctx.truth,**{s:d.truth for s,d in lab.sets.items()}}
    def metrics(predictions):
        ok={s:predictions[s]==truth[s] for s in predictions}
        parts={"clean":float(ok["oof"][scope&h].mean())}
        frame=lab.sets["testlike"].df
        for s in ("gt_missing","drop_word","both"):
            mask=frame.scenario.eq(s).to_numpy()&scope[frame.origin.to_numpy()]
            parts[s]=float(ok["testlike"][mask].mean())
        return dict(testlike=.7*sum(TESTLIKE_MIX[s]*v for s,v in parts.items())+.3*float(ok["oof"][scope&~h].mean()),scenarios=parts)
    report=dict(tag=TAG,client_date="2026-10-30",post_result_diagnostic=True,source_sha256=sha(__import__("pathlib").Path(__file__)),
                protocol_sha256=sha(PROTOCOL),reference=metrics(reference),
                label_aware_expert_choice_oracles={f"union_top{k}":metrics(p) for k,p in oracle.items()},
                counts=dict(counts),training_objective=training,outer_conditional_nll=outer_losses,
                scope="Only original v12 neural-active rows can be changed; oracle may keep actual v12 winner or choose any top-k base/Qwen/BGE candidate using true labels. Other rows unchanged.",
                caveat="Oracles are not learned scores, Public forecasts or bounds for more general reranking. NLL excludes inactive or truth-outside-top20 rows, so is conditional and cannot be read as full-population loss. Fitting uses a half-training single proxy and neural models; outer evaluation uses the full-training v8 ensemble and neural models. A distribution shift is a plausible limitation, not a proven unique cause.")
    write(report,ROOT/"experiments"/f"{TAG}_transfer.json");print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
