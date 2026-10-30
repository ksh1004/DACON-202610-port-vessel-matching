"""Paired BGE training: identical queries, 8 versus 20 listwise candidates."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT, load_train, load_catalog, is_hangul
from src.ensemble import normalize
from src.features import FEATURE_NAMES
from src.validate import load_folds
from src.expert_calibration import positive_probability
from tools.calibration_experiment import sha, write, dump

TAG="encoder_list20_v1"
FOLDER=ROOT/"experiments/artifacts"/TAG
PROTOCOL=ROOT/"experiments"/f"{TAG}_protocol.json"
MODEL=Path("D:/KSH/hf_models/BAAI__bge-reranker-v2-m3")
ARMS=(8,20)
GROUPS=1200


def fingerprint():return sha(PROTOCOL)


def check():
    p=json.loads(PROTOCOL.read_text())
    for file,h in p["source_hashes"].items():assert sha(ROOT/file)==h,file
    return p


def arm_folder(width):return FOLDER/f"width{width}"


def prepare(fold):
    from transformers import AutoTokenizer
    from src.encoder_phonetics import encode_pairs
    path=FOLDER/f"fold{fold}_data.joblib"
    if path.exists():return joblib.load(path)
    start=time.perf_counter()
    original=joblib.load(ROOT/f"experiments/artifacts/encoder_ambiguity_v1/fold{fold}_data.joblib")
    matrix=joblib.load(ROOT/"experiments/cache"/original["protocol"]["train_cache"])
    lookup={s:i for i,s in enumerate(matrix["sample_ids"])}
    selections=original["selected"][:GROUPS]
    train,catalog=load_train(),load_catalog();folds=load_folds(train)
    held=set(train.loc[folds==fold,"target_vessel_id"])
    assert not held & {g["row"]["target_vessel_id"] for g in selections}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="right")
    miner=lgb.Booster(model_file=str(ROOT/f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt"))
    blocks=[matrix["blocks"][lookup[g["row"]["sample_id"]]][:,:len(FEATURE_NAMES)] for g in selections]
    ends=np.cumsum([0]+[len(b) for b in blocks])
    score=miner.predict(np.vstack(blocks),num_threads=4)
    groups={8:[],20:[]};selected=[];cap=0
    for pos,g in enumerate(selections):
        i=lookup[g["row"]["sample_id"]];c=matrix["candidates"][i]
        assert int(c[np.flatnonzero(c==matrix["truth"][i])[0]])==int(g["candidates"][0])
        chosen=list(map(int,g["candidates"]))
        assert len(chosen)==8 and len(set(chosen))==8
        for j in np.argsort(-score[ends[pos]:ends[pos+1]],kind="stable"):
            if int(c[j]) not in chosen:chosen.append(int(c[j]))
            if len(chosen)==20:break
        assert len(chosen)==len(set(chosen))==20
        assert catalog.iloc[chosen[0]].vessel_id==g["row"]["target_vessel_id"]
        choices=catalog.iloc[chosen].to_dict("records")
        tokens=encode_pairs(tokenizer,g["row"],choices,g["beams"])
        assert tokens[:8]==original["groups"][pos]
        altered=dict(g["row"],sample_id="ignored",call_sign="ignored",target_vessel_id="ignored")
        if pos in (0,GROUPS//2,GROUPS-1):assert encode_pairs(tokenizer,altered,choices,g["beams"])==tokens
        groups[8].append(tokens[:8]);groups[20].append(tokens)
        cap+=sum(len(t)==256 for t in tokens)
        selected.append(dict(row=g["row"],candidates=np.array(chosen),beams=g["beams"]))
    report=dict(fold=fold,groups=GROUPS,widths=list(ARMS),pairs8=GROUPS*8,pairs20=GROUPS*20,
                distinct_fitting_vessels=len({g["row"]["target_vessel_id"] for g in selected}),
                outer_vessel_overlap=0,original8_token_prefix_exact=True,all_candidates_unique=True,
                positive_at_index0=True,identity_inputs_ignored=True,token_pairs_at256_cap=cap,
                strict_matrix=original["protocol"]["train_cache"],seconds=time.perf_counter()-start)
    result=dict(groups=groups,selected=selected,evaluation=original["evaluation"],report=report,
                protocol=original["protocol"],token_sha256={str(w):hashlib.sha256(json.dumps(groups[w],separators=(",",":")).encode()).hexdigest() for w in ARMS})
    dump(result,path);write(report,FOLDER/f"fold{fold}_preparation.json")
    print(json.dumps(report),flush=True);return result


def register():
    import datetime
    assert not PROTOCOL.exists()
    from src.llm_cache import CACHE_PATH
    files=["tools/list20_experiment.py","tools/check_list20.py","tools/calibration_experiment.py","src/expert_calibration.py",
           "src/common.py","src/encoder_phonetic_experiment.py","src/encoder_phonetics.py","src/encoder_pair.py",
           "src/neural_pair.py","src/pair_views.py","src/ensemble.py","src/augment.py","src/experiment.py",
           "src/features.py","src/validate.py","src/llm_translit.py","src/llm_cache.py",str(CACHE_PATH.relative_to(ROOT)),"tools/audit_reverse_ranker.py","tools/blend_matching.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv","experiments/encoder_bge_phonetic_predictions.joblib"]
    for fold in (1,4):
        d=prepare(fold)
        for name in (f"experiments/artifacts/{TAG}/fold{fold}_data.joblib",
                     f"experiments/artifacts/{TAG}/fold{fold}_preparation.json",
                     f"experiments/artifacts/encoder_ambiguity_v1/fold{fold}_data.joblib",
                     f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt",
                     f"experiments/cache/{d['report']['strict_matrix']}"):
            files.append(name)
    for fold in range(5):
        for tag in ("encoder_bge_phonetic","neural_evidence_v1"):
            files.append(f"experiments/artifacts/{tag}/fold{fold}_epoch1_scores.joblib")
    write(dict(tag=TAG,client_date="2026-10-30",registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               registered_before_new_neural_training_or_outer_accuracy=True,reference="v12",reference_public=.9065951735,
               contender="list20",contenders=["list20"],
               matched_control="New BGE trained on exactly the same1200 queries, initial weights, query order, optimizer updates, microbatch and effective batch, but original8-candidate groups; fixed v12 composition with this control replacing BGE. Not selectable.",
               preparation="First1200 legacy seed42-selected training groups; retain original positive+5LGB+2lexical candidates exactly as8-prefix. Add12 highest train-only LGB candidates not already in group. IDs used only for official joins and vessel split. No new names or external data.",
               model=dict(repo="BAAI/bge-reranker-v2-m3",revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",license="Apache-2.0"),
               training=dict(groups=GROUPS,epochs=1,updates=600,seed=42,microbatch_groups=1,accumulation=2,
                             learning_rate=1e-4,weight_decay=.01,gradient_clip=1.,lora_rank=8,lora_alpha=16,lora_dropout=.05,
                             trainable="query/key/value LoRA and classifier",loss="cross entropy over8 or20 candidates; gold0",
                             starting_point="Public original weights for both arms, not continued v12 training",dtype="BF16",
                             optimizer="AdamW",checkpoint="Two rolling adapter+optimizer+all RNG slots every200updates; ready markers written last"),
               inference="Actual frozen v12: base75%+oldQwen12.5%+newBGE12.5%, same20 candidates, old neural pruning. No new inference architecture/calls.",
               screen_folds=[1,4],screen_gate=dict(testlike_gain_min=.003,minimum_fold_delta=-.002,hard_hangul_delta_min=-.002,
                    drop_word_delta_min=-.002,clean_lost_rows_max=1,gain_vs_matched_control_min=.001),
               fresh_guard=dict(fold=3,seed=2026103001,minimum_mean_accuracy_delta=-.002,training_only_retrieval=True,
                                few_shot_vessels_excluded=True,previously_used_vessels_not_pristine=True),
               full_gate=dict(all_5_folds_required=True,testlike_gain_min=.005,positive_folds_min=4,hard_hangul_delta_min=-.001,
                              drop_word_delta_min=-.002,oof_delta_min=-.0006,gain_vs_matched_control_min=.001),
               source_hashes={f:sha(ROOT/f) for f in files},actual_competition_submission=False,callsign_or_identity_lookup=False,
               no_coefficient_or_checkpoint_search_after_results=True,
               caveat="Repeated exploratory synthetic CV, historical base retrieval mapper exception. Candidate20 sees2.5x as many pairs as control8 at equal query/update count; this is part of the intervention, not equal-compute training. Original8 grouping was already listwise CE; the change is group width, not first introduction of listwise learning. Changing group width may worsen ambiguous-negative pressure.",
               research_sources=["https://arxiv.org/abs/2210.10634"]),PROTOCOL)
    print("Registered "+str(PROTOCOL),flush=True)


def seed():
    import torch
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)


def train(fold,width,data):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    check();out=arm_folder(width);out.mkdir(parents=True,exist_ok=True)
    report=out/f"fold{fold}_training.json";merged=out/f"fold{fold}_merged"
    if report.exists():
        assert json.loads(report.read_text())["fingerprint"]==fingerprint();return merged
    seed();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="right")
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16).to("cuda")
    model.config.use_cache=False
    checkpoints=[]
    for slot in (0,1):
        path=out/f"fold{fold}_checkpoint{slot}"
        if (path/"ready.json").exists():
            ready=json.loads((path/"ready.json").read_text());assert ready["fingerprint"]==fingerprint()
            checkpoints.append((ready["step"],path))
    completed=0;losses=[]
    if checkpoints:
        _,cp=max(checkpoints,key=lambda pair:pair[0]);state=torch.load(cp/"state.pt",map_location="cuda",weights_only=False)
        assert state["fingerprint"]==fingerprint() and state["width"]==width and state["fold"]==fold
        model=PeftModel.from_pretrained(model,cp,is_trainable=True)
        completed=state["completed"];initial_hash=state["initial_hash"];losses=state["losses"]
    else:
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=["query","key","value"],modules_to_save=["classifier"],task_type="SEQ_CLS",bias="none"))
        digest=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:digest.update(name.encode());digest.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
        initial_hash=digest.hexdigest()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if checkpoints:
        assert state["trainable_names"]==names;optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"]);del state
    order=np.random.default_rng(42).permutation(GROUPS);start=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
    for offset in range(completed*2,GROUPS,2):
        value=0.
        for index in order[offset:offset+2]:
            enc=tokenizer.pad({"input_ids":data["groups"][width][int(index)]},padding=True,pad_to_multiple_of=8,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(**enc).logits.reshape(1,width).float()
                loss=functional.cross_entropy(logits,torch.zeros(1,dtype=torch.long,device="cuda"))/2
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite listwise loss")
            loss.backward();value+=float(loss.item())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite listwise gradient")
        optimizer.step();optimizer.zero_grad(set_to_none=True);step=offset//2+1;losses.append(value)
        if step%100==0:print(f"list20 fold{fold} width{width} {step}/600 loss{np.mean(losses[-100:]):.4f} seconds{time.perf_counter()-start:.1f}",flush=True)
        if step%200==0:
            cp=out/f"fold{fold}_checkpoint{(step//200)%2}";cp.mkdir(parents=True,exist_ok=True)
            (cp/"ready.json").unlink(missing_ok=True);model.save_pretrained(cp)
            torch.save(dict(fingerprint=fingerprint(),width=width,fold=fold,completed=step,initial_hash=initial_hash,
                            losses=losses,trainable_names=names,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),cp/"state.pt")
            write(dict(step=step,fingerprint=fingerprint()),cp/"ready.json")
    model.save_pretrained(out/f"fold{fold}_adapter");model=model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged);tokenizer.save_pretrained(merged)
    write(dict(fingerprint=fingerprint(),fold=fold,width=width,groups=GROUPS,steps=600,
               pairs=GROUPS*width,initial_trainable_sha256=initial_hash,token_sha256=data["token_sha256"][str(width)],
               finite_losses_and_gradients=True,resumed_after_step=completed,seconds_this_segment=time.perf_counter()-start,
               mean_training_loss=float(np.mean(losses)),max_training_loss=float(max(losses))),report)
    del model,optimizer,enc,logits,loss;gc.collect();torch.cuda.empty_cache();return merged


def score(fold,width,data):
    import torch
    from src.encoder_phonetics import EncoderPhoneticScorer
    from src.encoder_phonetic_experiment import evaluate
    check();out=arm_folder(width);path=out/f"fold{fold}_epoch1_scores.joblib"
    if path.exists():
        assert joblib.load(path)["protocol"]["list20_fingerprint"]==fingerprint();return
    scorer=EncoderPhoneticScorer(out/f"fold{fold}_merged")
    runtime=dict(data,protocol=dict(data["protocol"],training_groups=GROUPS,list20_width=width,list20_fingerprint=fingerprint(),
                                   groups_per_microbatch=1,accumulation=2,updates=600,
                                   negative_mode="original8 prefix" if width==8 else "original8 plus12 distinct train-only LGB candidates"))
    evaluate(scorer,runtime,out,fold);del scorer;gc.collect();torch.cuda.empty_cache()


def results(fold,data):
    from src.pair_views import mix_views
    old=joblib.load(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    q=joblib.load(ROOT/f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
    fresh={w:joblib.load(arm_folder(w)/f"fold{fold}_epoch1_scores.joblib") for w in ARMS}
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in old["sets"]}
    output=dict(fold=fold,sets={},controls={},fingerprint=fingerprint())
    for s,rows in old["sets"].items():
        output["sets"][s]=[];output["controls"][s]=[]
        for i,g in enumerate(rows):
            row=data["evaluation"][s][i]["row"];sid=row["sample_id"];keep=g["keep"];c=g["candidates"][keep]
            bp=positive_probability(g["base_scores"][keep]);anchor=bp;active=is_hangul(row["vessel_name"]) and not g["bound_pruned"]
            if active:anchor=mix_views(bp,[q["sets"][s][i]["neural_scores"],g["neural_scores"]],[1,1])
            assert int(c[np.argmax(anchor)])==int(archived["ensemble"][s][positions[s][sid]])
            for recipe,w in (("sets",20),("controls",8)):
                prediction=None;scores=None
                if is_hangul(row["vessel_name"]):
                    value=anchor
                    if active:
                        scores=fresh[w]["sets"][s][i]["neural_scores"]
                        np.testing.assert_array_equal(fresh[w]["sets"][s][i]["candidates"][fresh[w]["sets"][s][i]["keep"]],c)
                        value=mix_views(bp,[q["sets"][s][i]["neural_scores"],scores],[1,1])
                    prediction=int(c[np.argmax(value)])
                output[recipe][s].append(dict(sample_id=sid,truth=int(g["truth"]),predictions=dict(list20=prediction),scores=scores,
                                               truth_in_top20=bool(g["truth"] in c)))
    dump(output,FOLDER/f"fold{fold}_results.joblib")


def run():
    p=check();data={fold:prepare(fold) for fold in p["screen_folds"]}
    for fold in p["screen_folds"]:
        for width in ARMS:train(fold,width,data[fold])
        reports=[json.loads((arm_folder(w)/f"fold{fold}_training.json").read_text()) for w in ARMS]
        assert reports[0]["initial_trainable_sha256"]==reports[1]["initial_trainable_sha256"]
    for fold in p["screen_folds"]:
        for width in ARMS:score(fold,width,data[fold])
        results(fold,data[fold])
    from tools.audit_reverse_ranker import audit
    report=audit(FOLDER,PROTOCOL);write(report,ROOT/"experiments"/f"{TAG}_screen_locked.json")
    write(dict(selected=report["selected"],stage="screen",fingerprint=fingerprint(),actual_competition_submission=False),
          ROOT/"experiments"/f"{TAG}_selection.json")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--register",action="store_true");a=p.parse_args()
    if a.prepare:
        for fold in (1,4):prepare(fold)
    elif a.register:register()
    else:run()
