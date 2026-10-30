"""Prospectively fixed coarse metadata association against actual v12."""
import argparse
import datetime
import hashlib
import json
import time

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT, is_hangul, load_train, load_catalog
from src.context_density import (ContextCodec, FIELDS, QUERY_FIELDS, PARAMS, ROUNDS,
                                TEMPER, CLIP, fitting_examples, combine)
from src.validate import load_folds
from src.pair_views import mix_views
from src.byte_name_model import normalise
from tools.blend_matching import lightweight_lab

TAG = "context_density_v1"
FOLDER = ROOT/"experiments/artifacts"/TAG
PROTOCOL = ROOT/"experiments"/f"{TAG}_protocol.json"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def dump(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    joblib.dump(value,temp,compress=3);temp.replace(path)


def check():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for f,h in p["source_hashes"].items():assert sha(ROOT/f)==h,f
    return p


def register():
    assert not PROTOCOL.exists()
    files=["src/context_density.py","tools/context_density_experiment.py","tools/audit_reverse_ranker.py",
           "tools/blend_matching.py","src/common.py","src/validate.py","src/experiment.py","src/augment.py",
           "src/pair_views.py","src/byte_name_model.py","src/ensemble.py","src/llm_translit.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv",
           "experiments/encoder_bge_phonetic_predictions.joblib","experiments/context_density_v1_preflight.json"]
    for fold in range(5):
        for tag in ("encoder_bge_phonetic","neural_evidence_v1"):
            files.append(f"experiments/artifacts/{tag}/fold{fold}_epoch1_scores.joblib")
    p=dict(tag=TAG,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           registered_before_classifier_training_and_outer_performance=True,
           pre_registration_preparation="Training-only codec/support, example-weight and permutation checks completed; no classifier fit or new outer predictions",
           software={"lightgbm":lgb.__version__,"numpy":np.__version__},reference="v12", reference_public=.9065951735,
           fitting="Outer-training official original rows only, Han and English. Unique metadata contexts, one total positive/noise mass per vessel; 16 uniform official catalog negatives; paired observed/missing size. No artificial Korean labels.",
           features=FIELDS,query="Flag/type, three-year bucket, coarse GT-or-IGT bucket; no names, IDs, callsigns, exact tonnage or train-test lookup",
           support="Category or fixed numeric bucket needs >=10 distinct outer-training vessels; other values map to unknown0. No stored vessel profiles/identity map.",
           learner={"params":PARAMS,"rounds":ROUNDS},
           density_interpretation="data-vs-uniform-catalog classifier logits; conditional minus candidate-only logits estimates association P(c|q)/P(c), not calibrated ship posterior",
           contenders=["association"], matched_control="Identical model/data/weights, query context family deranged between fitting vessels, minus same candidate-only classifier; same fixed posterior correction. Not selectable.",
           combination={"formula":"log(actual v12 anchor)+temper*clip(conditional_logit-marginal_logit,-clip,clip)",
                        "temper":TEMPER,"clip":CLIP,"maximum_per_candidate_multiplier":2.,"minimum_per_candidate_multiplier":.5,
                        "candidates":"Frozen actual base top20, actual v12 old-pruned fallback preserved", "runtime":"All Han rows scored; no old or new neural pruning applied to metadata adjustment"},
           screen_folds=[1,4], screen_gate=dict(testlike_gain_min=.003,minimum_fold_delta=-.002,hard_hangul_delta_min=-.002,
                     drop_word_delta_min=-.002,clean_lost_rows_max=1,gain_vs_matched_control_min=.001),
           fresh_guard=dict(fold=3,seed=2026102901,minimum_mean_accuracy_delta=-.002,training_only_retrieval=True,
                            few_shot_vessels_excluded=True,previously_used_vessels_not_pristine=True),
           full_gate=dict(all_5_folds_required=True,testlike_gain_min=.005,positive_folds_min=4,hard_hangul_delta_min=-.001,
                          drop_word_delta_min=-.002,oof_delta_min=-.0006,gain_vs_matched_control_min=.001),
           no_hyperparameter_or_coefficient_search_after_results=True,no_automatic_other_method_after_failure=True,
           source_hashes={f:sha(ROOT/f) for f in files},actual_competition_submission=False,lookup_used=False,
           caveat="Repeated exploratory synthetic CV, not confirmed test distribution. Historical v8 outer retrieval mapper exception remains in reference comparison. Metadata likelihood may duplicate existing base evidence; fixed small trial tests net benefit. No guarantee of Public0.97.",
           research_sources=["https://www.jmlr.org/papers/v13/gutmann12a.html","https://dacon.io/competitions/official/236771/overview/rules"])
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print("Registered "+str(PROTOCOL),flush=True)


def frames():
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    lab=lightweight_lab(archived["identity"])
    dfs={"oof":lab.ctx.train,**{s:d.df for s,d in lab.sets.items()}}
    return archived,dfs


def evaluation(fold):
    path=FOLDER/f"fold{fold}_evaluation.joblib"
    if path.exists():
        result=joblib.load(path);assert result["fingerprint"]==sha(PROTOCOL);return result
    archived,dfs=frames()
    records={s:dict(zip(df.sample_id,df.to_dict("records"))) for s,df in dfs.items()}
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in dfs}
    b=joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    q=joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
    catalog=load_catalog().to_dict("records");sets={}
    for s,rows in b["sets"].items():
        sets[s]=[]
        for i,g in enumerate(rows):
            sid=b["sample_ids"][s][i];row=records[s][sid];keep=g["keep"]
            candidates=g["candidates"][keep];bp=normalise(g["base_scores"][keep])
            anchor=bp
            if is_hangul(row["vessel_name"]) and not g["bound_pruned"]:
                anchor=normalise(mix_views(bp,[q["sets"][s][i]["neural_scores"],g["neural_scores"]],[1,1]))
            assert int(candidates[np.argmax(anchor)])==int(archived["ensemble"][s][positions[s][sid]])
            sets[s].append(dict(sample_id=sid,row=row,candidates=candidates,anchor=anchor,truth=int(g["truth"]),
                                choices=[catalog[int(c)] for c in candidates],actual_hangul=bool(is_hangul(row["vessel_name"]))))
    result=dict(sets=sets,fingerprint=sha(PROTOCOL));dump(result,path);return result


def fit(fold):
    check();path=FOLDER/f"fold{fold}_training.json"
    if path.exists():
        d=json.loads(path.read_text());assert d["fingerprint"]==sha(PROTOCOL);return
    start=time.perf_counter();train,catalog=load_train(),load_catalog();folds=load_folds(train)
    fitting=train.loc[folds!=fold]
    assert not set(fitting.target_vessel_id)&set(train.loc[folds==fold,"target_vessel_id"])
    codec=ContextCodec().fit(fitting,catalog);x,shuffled,y,w,report=fitting_examples(fitting,catalog,codec)
    FOLDER.mkdir(parents=True,exist_ok=True)
    dump(codec,FOLDER/f"fold{fold}_codec.joblib")
    assert report["permutation_fixed_vessel_donors"]==0
    for name,data,names in (("conditional",x,FIELDS),("permuted",shuffled,FIELDS),
                             ("marginal",x[:,len(QUERY_FIELDS):],FIELDS[len(QUERY_FIELDS):])):
        model=lgb.train(PARAMS,lgb.Dataset(data,label=y,weight=w,feature_name=names,
                         categorical_feature=list(range(data.shape[1]))),num_boost_round=ROUNDS)
        model.save_model(str(FOLDER/f"fold{fold}_{name}.txt"))
        assert np.isfinite(model.predict(data[:256],raw_score=True,num_threads=1)).all()
    report.update(fold=fold,fingerprint=sha(PROTOCOL),models=3,rounds_each=ROUNDS,
                  elapsed_seconds=time.perf_counter()-start,outer_vessel_overlap=0,
                  minimum_supported_distinct_vessels=min(n for c in codec.support_counts for n in c.values()),
                  categories_by_feature={f:len(m) for f,m in zip(FIELDS,codec.maps)},
                  inference_contains_identity_map=False)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


def load_models(fold):
    codec=joblib.load(FOLDER/f"fold{fold}_codec.joblib")
    models={m:lgb.Booster(model_file=str(FOLDER/f"fold{fold}_{m}.txt")) for m in ("conditional","permuted","marginal")}
    return codec,models


def score(codec,models,row,choices):
    x=codec.features(row,choices)
    return {m:model.predict(x if m!="marginal" else x[:,len(QUERY_FIELDS):],raw_score=True,num_threads=1)
            for m,model in models.items()}


def results(fold):
    check();data=evaluation(fold);codec,models=load_models(fold)
    # Batch pointwise tree scoring is identical to row-wise; no normalization across queries.
    all_groups=[g for rows in data["sets"].values() for g in rows if g["actual_hangul"]]
    x=np.vstack([codec.features(g["row"],g["choices"]) for g in all_groups])
    scores={m:model.predict(x if m!="marginal" else x[:,len(QUERY_FIELDS):],raw_score=True,num_threads=1)
            for m,model in models.items()}
    result=dict(fold=fold,sets={},controls={},fingerprint=sha(PROTOCOL));position=0
    for s,rows in data["sets"].items():
        result["sets"][s]=[];result["controls"][s]=[]
        for g in rows:
            raw=None
            if g["actual_hangul"]:
                end=position+len(g["candidates"]);raw={m:v[position:end] for m,v in scores.items()};position=end
            for recipe,m in (("sets","conditional"),("controls","permuted")):
                values=None if raw is None else combine(g["anchor"],raw[m],raw["marginal"])
                prediction=None if values is None else int(g["candidates"][np.argmax(values)])
                result[recipe][s].append(dict(sample_id=g["sample_id"],truth=g["truth"],
                      predictions={"association":prediction},scores=raw,truth_in_top20=bool(g["truth"] in g["candidates"])))
    assert position==len(x);dump(result,FOLDER/f"fold{fold}_results.joblib")


def runtime(folds):
    check();reports=[]
    for fold in folds:
        data=evaluation(fold);saved=joblib.load(FOLDER/f"fold{fold}_results.joblib");codec,models=load_models(fold)
        groups=[(s,i,g) for s,rows in data["sets"].items() for i,g in enumerate(rows) if g["actual_hangul"]]
        chosen=[groups[i] for i in (0,len(groups)//2,len(groups)-1)];original=[]
        for s,i,g in chosen:
            raw=score(codec,models,g["row"],g["choices"])
            for m in raw:np.testing.assert_array_equal(raw[m],saved["sets"][s][i]["scores"][m])
            rev=score(codec,models,g["row"],g["choices"][::-1])
            for m in raw:np.testing.assert_array_equal(rev[m],raw[m][::-1])
            for j in (0,len(g["choices"])-1):
                single=score(codec,models,g["row"],[g["choices"][j]])
                for m in raw:np.testing.assert_array_equal(single[m],raw[m][j:j+1])
            changed=dict(g["row"],target_vessel_id="changed",call_sign="changed",sample_id="changed",vessel_name="changed")
            np.testing.assert_array_equal(codec.features(changed,g["choices"]),codec.features(g["row"],g["choices"]))
            values=combine(g["anchor"],raw["conditional"],raw["marginal"])
            assert int(g["candidates"][np.argmax(values)])==saved["sets"][s][i]["predictions"]["association"]
            np.testing.assert_array_equal(combine(g["anchor"][::-1],rev["conditional"],rev["marginal"]),values[::-1])
            original.append((g,raw))
        for g,raw in reversed(original):
            again=score(codec,models,g["row"],g["choices"])
            for m in raw:np.testing.assert_array_equal(again[m],raw[m])
        reports.append(dict(fold=fold,actual_queries=3,saved_scores_exact=True,final_predictions_equal=True,
                      row_reverse_subset_single_exact=True,candidate_permutation_exact=True,identity_and_names_ignored=True))
    report=dict(status="PASS",fingerprint=sha(PROTOCOL),checks=reports,
                scope="Actual reloaded metadata models plus frozen query anchor/candidates; not a complete new submission ZIP")
    (ROOT/"experiments"/f"{TAG}_runtime_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


def run():
    protocol=check()
    for fold in protocol["screen_folds"]:fit(fold);results(fold)
    runtime(protocol["screen_folds"])
    from tools.audit_reverse_ranker import audit
    report=audit(FOLDER,PROTOCOL)
    selection=dict(selected=report["selected"],stage="screen",fingerprint=sha(PROTOCOL),
                   actual_competition_submission=False,no_coefficient_search_after_results=True)
    (ROOT/"experiments"/f"{TAG}_screen_locked.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (ROOT/"experiments"/f"{TAG}_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    if report["selected"]:print("SCREEN QUALIFIED: continue with frozen fresh/full gates; no release yet",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--register",action="store_true");p.add_argument("--runtime",action="store_true");a=p.parse_args()
    if a.register:register()
    elif a.runtime:runtime([1,4])
    else:run()
