"""Freeze, fit, and screen 3/6-parameter calibration on nested held-out scores."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import time

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.ensemble import normalize
from src.expert_calibration import (MODES, L2, LIMIT, LOG_RATIO_CLIP, positive_probability,
                                   probability, missing_size, features, corrected_logits, objective, fit)
from src.pair_views import mix_views
from tools.blend_matching import lightweight_lab

TAG = "expert_calibration_v1"
FOLDER = ROOT/"experiments/artifacts"/TAG
PROTOCOL = ROOT/"experiments"/f"{TAG}_protocol.json"
NESTED = ROOT/"experiments/artifacts/nested_combiner_v1"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):h.update(block)
    return h.hexdigest()


def write(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def dump(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    joblib.dump(value,temporary,compress=3);temporary.replace(path)


def check():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for file,h in p["source_hashes"].items():assert sha(ROOT/file)==h,file
    return p


def training(outer):
    records=[];report=[]
    outer_ids=None
    for inner in (0,1):
        folder=NESTED/f"outer{outer}_inner{inner}"
        data=joblib.load(folder/"data.joblib")
        prep=data["report"]
        fit_ids,meta_ids,held_ids=(set(prep[k]) for k in ("fit_vessels","meta_vessels","outer_validation_vessels"))
        assert not fit_ids&meta_ids and not held_ids&(fit_ids|meta_ids)
        outer_ids=held_ids
        scores={k:joblib.load(folder/f"meta_{k}_scores.joblib")["scores"] for k in ("qwen","bge")}
        assert len(scores["qwen"])==len(scores["bge"])==len(data["meta"])
        active=0
        for g,q,b in zip(data["meta"],scores["qwen"],scores["bge"]):
            assert g["row"]["target_vessel_id"] in meta_ids
            p0=probability(g["proxy"])
            top=np.sort(p0)
            if len(top)<2 or top[-1]-top[-2] > 1/3+1e-6:continue
            experts=[p0,probability(q),probability(b)]
            anchor=positive_probability(.75*experts[0]+.125*experts[1]+.125*experts[2])
            target=np.flatnonzero(g["candidates"]==g["truth"])
            assert len(target)==1 and len(anchor)==20
            records.append(dict(row=g["row"],experts=experts,anchor=anchor,target=int(target[0]),weight=g["weight"]))
            active+=1
        report.append(dict(inner=inner,meta_groups=len(data["meta"]),active_training_groups=active,
                           fit_vessels=len(fit_ids),meta_vessels=len(meta_ids),outer_vessels=len(held_ids),overlap=0))
    assert all(r["row"]["target_vessel_id"] not in outer_ids for r in records)
    totals=defaultdict(float)
    for g in records:totals[g["row"]["target_vessel_id"]]+=g["weight"]
    weights=np.array([g["weight"]/totals[g["row"]["target_vessel_id"]] for g in records])
    report=dict(splits=report,groups=len(records),vessels=len(totals),
                missing_size_groups=int(sum(missing_size(g["row"]) for g in records)),
                fitting_weight_sum=float(weights.sum()),outer_vessel_overlap=0)
    return records,weights,report


def preflight():
    rng=np.random.default_rng(20261030)
    x=rng.normal(size=(7,20,6));off=rng.normal(size=(7,20));targets=np.arange(7);w=np.full(7,1/7)
    beta=np.linspace(-.12,.19,6);value,gradient=objective(beta,off,x,targets,w)
    numerical=np.array([(objective(beta+np.eye(6)[j]*1e-6,off,x,targets,w)[0]-
                         objective(beta-np.eye(6)[j]*1e-6,off,x,targets,w)[0])/2e-6 for j in range(6)])
    assert np.max(np.abs(numerical-gradient))<1e-8
    anchor=probability(rng.normal(size=20));experts=[probability(rng.normal(size=20)) for _ in range(3)]
    row={"gross_tonnage":1200,"international_gross_tonnage":None}
    for mode in MODES:
        v=features(anchor,experts,row,mode)
        coeff=np.linspace(-.2,.2,v.shape[1]);z=corrected_logits(anchor,v,coeff)
        np.testing.assert_array_equal(corrected_logits(anchor[::-1],v[::-1],coeff),z[::-1])
        for j in (0,8,19):np.testing.assert_array_equal(corrected_logits(anchor[j:j+1],v[j:j+1],coeff),z[j:j+1])
        np.testing.assert_array_equal(features(anchor,experts,dict(row,target_vessel_id="ignore",call_sign="ignore",sample_id="ignore",vessel_name="ignore"),mode),v)
        assert np.argmax(corrected_logits(anchor,v,np.zeros(v.shape[1])))==np.argmax(anchor)
    preparation={}
    for outer in (0,2):
        _,_,preparation[str(outer)]=training(outer)
    result=dict(status="PASS",maximum_gradient_error=float(np.max(np.abs(numerical-gradient))),
                zero_correction_winner_exact=True,pointwise_and_candidate_order_exact=True,
                calibrator_identity_inputs_ignored=True,training_only_preparation=preparation,
                new_optimizer_fit_or_outer_performance=False)
    write(result,ROOT/"experiments"/f"{TAG}_preflight.json");print(json.dumps(result,indent=2),flush=True)


def register():
    assert not PROTOCOL.exists()
    assert json.loads((ROOT/"experiments"/f"{TAG}_preflight.json").read_text())["status"]=="PASS"
    files=["src/expert_calibration.py","tools/calibration_experiment.py","tools/audit_reverse_ranker.py",
           "tools/blend_matching.py","src/common.py","src/ensemble.py","src/pair_views.py","src/expert_calibration.py",
           "src/experiment.py","src/augment.py","src/validate.py","src/llm_translit.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv","experiments/encoder_bge_phonetic_predictions.joblib",
           f"experiments/{TAG}_preflight.json"]
    for k in range(5):
        for tag in ("encoder_bge_phonetic","neural_evidence_v1"):
            files.append(f"experiments/artifacts/{tag}/fold{k}_epoch1_scores.joblib")
    for outer in (0,2):
        for inner in (0,1):
            for file in ("data.joblib","meta_qwen_scores.joblib","meta_bge_scores.joblib"):
                files.append(f"experiments/artifacts/nested_combiner_v1/outer{outer}_inner{inner}/{file}")
    p=dict(tag=TAG,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           client_date="2026-10-30",registered_before_new_calibrator_fit_and_outer_performance=True,
           contenders=list(MODES),reference="v12",reference_public=.9065951735,
           matched_control="Unchanged actual v12, exactly zero correction. pooled3 and conditional6 are both prospectively selectable in that fixed priority order conditional6 then pooled3; no later coefficient search.",
           training="Existing genuinely inner-vessel-held-out Qwen/BGE/proxy distributions from nested outer0/2; metadata models and rankers fitted on inner training only. Retain uncertain proxy groups with top2 probability gap<=1/3+1e-6. Weight scenario within each retained vessel then equal total weight per fitting vessel. No resplitting old outer neural predictions to train the calibrator.",
           formula="log(actual_v12_anchor)+sum beta_j*clip(log(p_expert)-log(anchor),-8,8). Pooled3: three coefficients. Conditional6: disjoint coefficients according to both GT/IGT missing versus any positive tonnage.",
           optimization=dict(method="L-BFGS-B",objective="weighted candidate cross entropy +0.5*L2*sum(beta^2)",
                             l2=L2,bounds=[-LIMIT,LIMIT],start="all zero",maxiter=200,ftol=1e-12,gtol=1e-8),
           inference="Only actual-Han original-v12 neural-active rows change; preserve old bound-pruned fallback and all original candidates. No additional neural calls or new pruning bound. Zero correction must reproduce all actual v12 winners.",
           screen_folds=[0,2],screen_gate=dict(testlike_gain_min=.003,minimum_fold_delta=-.002,hard_hangul_delta_min=-.002,
                drop_word_delta_min=-.002,clean_lost_rows_max=1,gain_vs_matched_control_min=.001),
           fresh_guard=dict(fold=3,seed=2026103001,minimum_mean_accuracy_delta=-.002,training_only_retrieval=True,
                            few_shot_vessels_excluded=True,previously_used_vessels_not_pristine=True,
                            fitting="Must create fold3-specific inner heldout experts; cannot fit on outer0/2 meta scores that trained on fold3"),
           full_gate=dict(all_5_folds_required=True,testlike_gain_min=.005,positive_folds_min=4,hard_hangul_delta_min=-.001,
                          drop_word_delta_min=-.002,oof_delta_min=-.0006,gain_vs_matched_control_min=.001),
           source_hashes={f:sha(ROOT/f) for f in dict.fromkeys(files)},actual_competition_submission=False,
           callsign_or_identity_lookup=False,no_new_neural_training_for_initial_screen=True,
           caveat="Reused exploratory synthetic CV with historical v8 retrieval mapper exception. Single proxy anchor for fitting differs from actual v8 ensemble at inference. Log-loss improvement does not prove accuracy gain. No Public0.97 guarantee.",
           research_sources=["https://proceedings.mlr.press/v70/guo17a.html",
                             "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html"])
    write(p,PROTOCOL);print("Registered "+str(PROTOCOL),flush=True)


def evaluation(fold):
    path=FOLDER/f"fold{fold}_evaluation.joblib"
    if path.exists():
        result=joblib.load(path);assert result["fingerprint"]==sha(PROTOCOL);return result
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib");lab=lightweight_lab(archived["identity"])
    frames={"oof":lab.ctx.train,**{s:d.df for s,d in lab.sets.items()}}
    records={s:dict(zip(f.sample_id,f.to_dict("records"))) for s,f in frames.items()}
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in frames}
    old=joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    qwen=joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
    sets={}
    for s,rows in old["sets"].items():
        sets[s]=[]
        for i,g in enumerate(rows):
            sid=old["sample_ids"][s][i];row=records[s][sid];c=g["candidates"][g["keep"]]
            bp=positive_probability(g["base_scores"][g["keep"]]);anchor=bp;experts=None
            active=bool(is_hangul(row["vessel_name"]) and not g["bound_pruned"])
            if active:
                q,b=qwen["sets"][s][i]["neural_scores"],g["neural_scores"]
                anchor=positive_probability(mix_views(bp,[q,b],[1,1]))
                experts=[bp,probability(q),probability(b)]
            assert int(c[np.argmax(anchor)])==int(archived["ensemble"][s][positions[s][sid]])
            sets[s].append(dict(sample_id=sid,row=row,candidates=c,anchor=anchor,experts=experts,
                                active=active,actual_hangul=bool(is_hangul(row["vessel_name"])),truth=int(g["truth"])))
    result=dict(sets=sets,fingerprint=sha(PROTOCOL));dump(result,path);return result


def fit_fold(fold):
    check();groups,weights,report=training(fold)
    offsets=np.log(np.maximum(np.vstack([g["anchor"] for g in groups]),1e-15))
    targets=np.array([g["target"] for g in groups]);start=time.perf_counter();models={}
    for mode in MODES:
        x=np.stack([features(g["anchor"],g["experts"],g["row"],mode) for g in groups])
        models[mode]=fit(offsets,x,targets,weights)
    value=dict(fold=fold,training=report,models=models,seconds=time.perf_counter()-start,fingerprint=sha(PROTOCOL))
    write(value,FOLDER/f"fold{fold}_calibrators.json");print(json.dumps(value,indent=2),flush=True)
    return value


def results(fold,model):
    data=evaluation(fold);result=dict(fold=fold,sets={},controls={},fingerprint=sha(PROTOCOL))
    for s,rows in data["sets"].items():
        result["sets"][s]=[];result["controls"][s]=[]
        for g in rows:
            predictions={m:None for m in MODES};baseline=dict(predictions);scores=None
            if g["actual_hangul"]:
                baseline={m:int(g["candidates"][np.argmax(g["anchor"])]) for m in MODES}
                predictions=dict(baseline);scores={}
                if g["active"]:
                    for m in MODES:
                        x=features(g["anchor"],g["experts"],g["row"],m)
                        scores[m]=corrected_logits(g["anchor"],x,model["models"][m]["coefficients"])
                        predictions[m]=int(g["candidates"][np.argmax(scores[m])])
            common=dict(sample_id=g["sample_id"],truth=g["truth"],truth_in_top20=bool(g["truth"] in g["candidates"]))
            result["sets"][s].append(dict(common,predictions=predictions,scores=scores))
            result["controls"][s].append(dict(common,predictions=baseline,scores=None))
    dump(result,FOLDER/f"fold{fold}_results.joblib")


def runtime():
    check();reports=[]
    for fold in (0,2):
        data=evaluation(fold);model=json.loads((FOLDER/f"fold{fold}_calibrators.json").read_text())
        saved=joblib.load(FOLDER/f"fold{fold}_results.joblib")
        active=[(s,i,g) for s,rows in data["sets"].items() for i,g in enumerate(rows) if g["active"]]
        selected=[active[i] for i in (0,len(active)//2,len(active)-1)];previous=[]
        for s,i,g in selected:
            for m in MODES:
                x=features(g["anchor"],g["experts"],g["row"],m);beta=model["models"][m]["coefficients"]
                z=corrected_logits(g["anchor"],x,beta)
                np.testing.assert_array_equal(z,saved["sets"][s][i]["scores"][m])
                xr=features(g["anchor"][::-1],[p[::-1] for p in g["experts"]],g["row"],m)
                np.testing.assert_array_equal(corrected_logits(g["anchor"][::-1],xr,beta),z[::-1])
                for j in (0,len(z)-1):
                    np.testing.assert_array_equal(corrected_logits(g["anchor"][j:j+1],x[j:j+1],beta),z[j:j+1])
                np.testing.assert_array_equal(features(g["anchor"],g["experts"],dict(g["row"],sample_id="ignored",call_sign="ignored",target_vessel_id="ignored"),m),x)
                assert int(g["candidates"][np.argmax(z)])==saved["sets"][s][i]["predictions"][m]
                previous.append((g,m,z))
        for g,m,z in reversed(previous):
            np.testing.assert_array_equal(corrected_logits(g["anchor"],features(g["anchor"],g["experts"],g["row"],m),model["models"][m]["coefficients"]),z)
        reports.append(dict(fold=fold,actual_queries=3,calibrators=2,saved_scores_and_predictions_exact=True,
                            query_order_and_subset_exact=True,candidate_order_exact=True,pointwise_arithmetic_exact=True))
    report=dict(status="PASS",checks=reports,fingerprint=sha(PROTOCOL),
                scope="Reloaded calibrator coefficients plus frozen expert distributions, not a new full neural pipeline/ZIP")
    write(report,ROOT/"experiments"/f"{TAG}_runtime_checks.json");print(json.dumps(report),flush=True)


def run():
    p=check()
    for fold in p["screen_folds"]:
        if not (FOLDER/f"fold{fold}_calibrators.json").exists():model=fit_fold(fold)
        else:
            model=json.loads((FOLDER/f"fold{fold}_calibrators.json").read_text());assert model["fingerprint"]==sha(PROTOCOL)
        results(fold,model)
    runtime()
    from tools.audit_reverse_ranker import audit
    report=audit(FOLDER,PROTOCOL)
    write(report,ROOT/"experiments"/f"{TAG}_screen_locked.json")
    write(dict(selected=report["selected"],stage="screen",fingerprint=sha(PROTOCOL),actual_competition_submission=False),
          ROOT/"experiments"/f"{TAG}_selection.json")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preflight",action="store_true");p.add_argument("--register",action="store_true");a=p.parse_args()
    if a.preflight:preflight()
    elif a.register:register()
    else:run()
