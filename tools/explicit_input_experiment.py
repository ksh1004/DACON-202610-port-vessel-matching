"""Controlled new BGE input formats, reusing the immutable1200-query8-control."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.common import ROOT,load_catalog,is_hangul,load_train
from src.encoder_explicit import MODES,MAX_LENGTH,encode_pairs,ExplicitScorer,gap,number
from src.expert_calibration import positive_probability
from src.llm_cache import BeamCache,CACHE_PATH
from src.pair_views import mix_views
from src.validate import load_folds
from tools.calibration_experiment import sha,write,dump

TAG="encoder_explicit_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"
CONTROL=ROOT/"experiments/artifacts/encoder_list20_v1/width8"


def fingerprint():return sha(PROTOCOL)


def check():
    p=json.loads(PROTOCOL.read_text())
    for f,h in p["source_hashes"].items():assert sha(ROOT/f)==h,f
    return p


def model_folder(fold,mode):return FOLDER/mode/"width8"/f"fold{fold}_merged"


def prepare(fold):
    from transformers import AutoTokenizer
    path=FOLDER/f"fold{fold}_data.joblib"
    if path.exists():return joblib.load(path)
    original=joblib.load(ROOT/f"experiments/artifacts/encoder_list20_v1/fold{fold}_data.joblib")
    selected=original["selected"];assert len(selected)==1200
    catalog=load_catalog().to_dict("records")
    tokenizer=AutoTokenizer.from_pretrained("D:/KSH/hf_models/BAAI__bge-reranker-v2-m3",local_files_only=True,padding_side="right")
    groups={m:[] for m in MODES};lengths={m:[] for m in MODES}
    train=load_train();folds=load_folds(train)
    assert not set(train.loc[folds==fold,"target_vessel_id"])&{g["row"]["target_vessel_id"] for g in selected}
    for i,g in enumerate(selected):
        choices=[catalog[int(j)] for j in g["candidates"][:8]]
        assert choices[0]["vessel_id"]==g["row"]["target_vessel_id"]
        for mode in MODES:
            tokens=encode_pairs(tokenizer,g["row"],choices,g["beams"],mode)
            groups[mode].append(tokens);lengths[mode].extend(map(len,tokens))
            if i in (0,600,1199):
                altered=dict(g["row"],sample_id="ignored",call_sign="ignored",target_vessel_id="ignored")
                assert encode_pairs(tokenizer,altered,[dict(c,vessel_id="ignored",call_sign="ignored") for c in choices],g["beams"],mode)==tokens
                assert encode_pairs(tokenizer,g["row"],choices[::-1],g["beams"],mode)==tokens[::-1]
    report=dict(fold=fold,groups=1200,candidates=8,distinct_vessels=original["report"]["distinct_fitting_vessels"],
                outer_vessel_overlap=0,query_and_candidate_ids_exactly_control=True,
                token_lengths={m:dict(min=int(min(v)),median=float(np.median(v)),max=int(max(v))) for m,v in lengths.items()},
                silent_truncation=False,identity_fields_ignored=True,candidate_order_tokens_exact=True)
    d=dict(groups=groups,selected=selected,evaluation=original["evaluation"],report=report,
           token_sha256={m:hashlib.sha256(json.dumps(groups[m],separators=(",",":")).encode()).hexdigest() for m in MODES})
    dump(d,path);write(report,FOLDER/f"fold{fold}_preparation.json");print(json.dumps(report),flush=True);return d


def register():
    import datetime
    assert not PROTOCOL.exists()
    audit=json.loads((ROOT/"experiments/token_budget_20261030.json").read_text())
    assert all(r["qwen_truncated_pairs"]==r["bge_truncated_pairs"]==0 for r in audit["reports"].values())
    assert gap(None,100)=="unknown" and gap(100,100)=="same" and gap(100,101)=="within 1 percent"
    assert number(1200)==number(1200.) and gap(0,100)=="unknown"
    files=["src/encoder_explicit.py","tools/explicit_input_experiment.py","tools/check_explicit_input.py",
           "tools/audit_token_budget.py","experiments/token_budget_20261030.json",
           "src/common.py","src/encoder_pair.py","src/encoder_phonetics.py","src/learned_alignment.py","src/hangul.py",
           "src/retrieval.py","src/neural_pair.py","src/pair_views.py","src/ensemble.py","src/expert_calibration.py",
           "src/validate.py","src/augment.py","src/experiment.py","src/features.py","src/llm_cache.py",str(CACHE_PATH.relative_to(ROOT)),
           "src/llm_translit.py","tools/list20_experiment.py","tools/calibration_experiment.py","tools/audit_reverse_ranker.py","tools/blend_matching.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv","experiments/encoder_bge_phonetic_predictions.joblib"]
    for fold in (1,4):
        prepare(fold)
        for file in (FOLDER/f"fold{fold}_data.joblib",FOLDER/f"fold{fold}_preparation.json",
                     ROOT/f"experiments/artifacts/encoder_list20_v1/fold{fold}_data.joblib",
                     CONTROL/f"fold{fold}_epoch1_scores.joblib",CONTROL/f"fold{fold}_training.json"):
            files.append(str(file.relative_to(ROOT)))
    for fold in range(5):
        for tag in ("encoder_bge_phonetic","neural_evidence_v1"):
            files.append(f"experiments/artifacts/{tag}/fold{fold}_epoch1_scores.joblib")
    write(dict(tag=TAG,client_date="2026-10-30",registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               registered_before_new_representation_model_training_and_scores=True,contenders=list(MODES),
               reference="v12",reference_public=.9065951735,
               matched_control="Previously reported original JSON/phonetic8-way BGE control from encoder_list20_v1; exact same1200queries, candidates, seed, public starting weights, microbatch1/accum2 and600updates. Control performance known before this registration, not a new prospective control fit.",
               primary="characters: short field labels + explicit GT/IGT comparison buckets + vowel-preserving Korean romanized spelling and Latin character spelling; keep names/consonant keys/hints and observations",
               secondary="compact: same shorter field serialization and numeric comparison buckets, without the added spelling lines. Both prospectively selectable, characters then compact. Report added-spelling comparison descriptively.",
               linguistic_rules="Only deterministic INITIAL/MEDIAL/FINAL tables via learned_alignment.observed; never call LearnedAlignment.fit/load or use any fitted alignment matrix. No external names, facts, labels or category lookup.",
               number_representation="Positive values, exact integer/float canonicalization, full repr for nonintegral floats; nonpositive/nonfinite/missing become unknown. Relative GT/IGT gap buckets same/<=1%/<=5%/<=20%/different/unknown, from current row and candidate only.",
               max_length_guard=MAX_LENGTH,truncation=False,
               training=dict(groups=1200,candidates=8,updates=600,seed=42,microbatch=1,accumulation=2,
                             optimizer="AdamW",learning_rate=1e-4,weight_decay=.01,clip=1.,lora_rank=8,lora_alpha=16,
                             lora_dropout=.05,dtype="BF16",starting_point="Public original BGE-reranker-v2-m3",
                             trainer="Reuse immutable list20_experiment.train with private folders/protocol; only input tokens change"),
               inference="Same actual v12 base75%+oldQwen12.5%+newBGE12.5%, original20 candidates and neural pruning. Document-side numeric comparisons use only current query/candidate.",
               incremental_screen_rationale="Prospectively test small consistent gains rather than only large ones: minimum gain0.001, both folds nonnegative, no original-correct loss and vessel-bootstrap95% lower bound>0. These criteria are fixed before any new-format fit/accuracy and do not relax the Public0.97 target. Reused CV means the interval is exploratory, not a selection-adjusted guarantee.",
               screen_folds=[1,4],screen_gate=dict(testlike_gain_min=.001,minimum_fold_delta=0.,hard_hangul_delta_min=-.001,
                    drop_word_delta_min=-.001,clean_lost_rows_max=0,gain_vs_matched_control_min=.001,bootstrap_lower_strictly_above=0.),
               fresh_guard=dict(fold=3,seed=2026103002,minimum_mean_accuracy_delta=-.002,training_only_retrieval=True,few_shot_vessels_excluded=True),
               full_gate=dict(all_5_folds_required=True,testlike_gain_min=.002,positive_folds_min=4,hard_hangul_delta_min=-.001,
                              drop_word_delta_min=-.001,oof_delta_min=0.,gain_vs_matched_control_min=.001,bootstrap_lower_strictly_above=0.),
               source_hashes={f:sha(ROOT/f) for f in files},actual_competition_submission=False,callsign_or_identity_lookup=False,
               no_weight_or_checkpoint_search_after_results=True,
               caveat="Repeated exploratory synthetic CV and historical reference retrieval mapper exception. Reused known8-control. Compact is a joint serialization/numeric-evidence intervention, not an isolated one-factor compression ablation. Variable token counts differ between modes; equal queries and updates, not equal token compute. BGE12.5% replacement alone is not a proof of a Public0.97 path.",
               research_sources=["https://arxiv.org/abs/2004.00584","https://arxiv.org/abs/2305.14685"]),PROTOCOL)
    print("Registered "+str(PROTOCOL),flush=True)


def train_mode(fold,mode,data):
    from tools import list20_experiment as helper
    check();saved=(helper.FOLDER,helper.PROTOCOL)
    try:
        helper.FOLDER,helper.PROTOCOL=FOLDER/mode,PROTOCOL
        d=dict(groups={8:data["groups"][mode]},token_sha256={"8":data["token_sha256"][mode]})
        print(f"EXPLICIT mode={mode} fold={fold} train start",flush=True)
        path=helper.train(fold,8,d)
    finally:helper.FOLDER,helper.PROTOCOL=saved
    report=json.loads((FOLDER/mode/"width8"/f"fold{fold}_training.json").read_text())
    control=json.loads((CONTROL/f"fold{fold}_training.json").read_text())
    assert report["initial_trainable_sha256"]==control["initial_trainable_sha256"]
    write(dict(report,experiment=TAG,mode=mode),FOLDER/f"fold{fold}_{mode}_training.json")
    return path


def evaluate(fold,data):
    import torch
    check();catalog=load_catalog().to_dict("records");beams=BeamCache(allow_generate=False)
    old=joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    qwen=joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in old["sets"]}
    for mode in MODES:
        path=FOLDER/f"fold{fold}_{mode}_scores.joblib"
        if path.exists():assert joblib.load(path)["fingerprint"]==fingerprint();continue
        model=ExplicitScorer(model_folder(fold,mode),mode);scored={}
        for s,rows in old["sets"].items():
            scored[s]=[]
            for i,g in enumerate(rows):
                row=data["evaluation"][s][i]["row"];raw=None
                if is_hangul(row["vessel_name"]) and not g["bound_pruned"]:
                    choices=[catalog[int(j)] for j in g["candidates"][g["keep"]]]
                    raw=model.score(row,choices,beams([row["vessel_name"]])[0])
                    assert np.isfinite(raw).all()
                scored[s].append(raw)
            print(f"EXPLICIT fold{fold} {mode} scored {s}",flush=True)
        dump(dict(sets=scored,fingerprint=fingerprint(),mode=mode),path)
        del model;gc.collect();torch.cuda.empty_cache()
    new={m:joblib.load(FOLDER/f"fold{fold}_{m}_scores.joblib")["sets"] for m in MODES}
    control=joblib.load(CONTROL/f"fold{fold}_epoch1_scores.joblib")
    result=dict(fold=fold,sets={},controls={},fingerprint=fingerprint())
    for s,rows in old["sets"].items():
        result["sets"][s]=[];result["controls"][s]=[]
        for i,g in enumerate(rows):
            row=data["evaluation"][s][i]["row"];sid=row["sample_id"];c=g["candidates"][g["keep"]]
            assert control["sample_ids"][s][i]==sid
            np.testing.assert_array_equal(control["sets"][s][i]["candidates"][control["sets"][s][i]["keep"]],c)
            bp=positive_probability(g["base_scores"][g["keep"]]);anchor=bp
            active=is_hangul(row["vessel_name"]) and not g["bound_pruned"]
            if active:anchor=mix_views(bp,[qwen["sets"][s][i]["neural_scores"],g["neural_scores"]],[1,1])
            assert int(c[np.argmax(anchor)])==int(archived["ensemble"][s][positions[s][sid]])
            pred={m:None for m in MODES};ctrl=dict(pred)
            if is_hangul(row["vessel_name"]):
                for mode in MODES:
                    value=anchor;control_value=anchor
                    if active:
                        value=mix_views(bp,[qwen["sets"][s][i]["neural_scores"],new[mode][s][i]],[1,1])
                        control_value=mix_views(bp,[qwen["sets"][s][i]["neural_scores"],control["sets"][s][i]["neural_scores"]],[1,1])
                    pred[mode]=int(c[np.argmax(value)]);ctrl[mode]=int(c[np.argmax(control_value)])
            common=dict(sample_id=sid,truth=int(g["truth"]),truth_in_top20=bool(g["truth"] in c))
            result["sets"][s].append(dict(common,predictions=pred,scores={m:new[m][s][i] for m in MODES}))
            result["controls"][s].append(dict(common,predictions=ctrl,scores=None))
    dump(result,FOLDER/f"fold{fold}_results.joblib")


def run():
    p=check();data={k:prepare(k) for k in p["screen_folds"]}
    for fold in p["screen_folds"]:
        for mode in MODES:train_mode(fold,mode,data[fold])
    for fold in p["screen_folds"]:evaluate(fold,data[fold])
    from tools.audit_reverse_ranker import audit
    report=audit(FOLDER,PROTOCOL)
    write(report,ROOT/"experiments"/f"{TAG}_base_audit.json")
    for mode,item in report["contenders"].items():
        item["screen_checks"]["bootstrap_lower_positive"]=item["bootstrap_95_vs_v12"][0]>p["screen_gate"]["bootstrap_lower_strictly_above"]
        item["screen_gate_pass"]=bool(all(item["screen_checks"].values()))
    report["selected"]=next((m for m in MODES if report["contenders"][m]["screen_gate_pass"]),None)
    report["selection_note"]="Includes prospectively frozen positive-bootstrap-lower-bound requirement in addition to the common auditor checks"
    write(report,ROOT/"experiments"/f"{TAG}_audit.json")
    write(report,ROOT/"experiments"/f"{TAG}_screen_locked.json")
    write(dict(selected=report["selected"],stage="screen",fingerprint=fingerprint(),actual_competition_submission=False),
          ROOT/"experiments"/f"{TAG}_selection.json")
    print(json.dumps({"final_screen_selected":report["selected"],"positive_bootstrap_required":True}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--register",action="store_true");a=p.parse_args()
    if a.prepare:
        for k in (1,4):prepare(k)
    elif a.register:register()
    else:run()
