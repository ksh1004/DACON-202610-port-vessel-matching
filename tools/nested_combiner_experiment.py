"""Two-level vessel-held-out training for the final Qwen/BGE decision model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.augment import augment
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.experiment import TESTLIKE_MIX, build_testlike
from src.features import FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.ensemble import normalize
from src.llm_cache import BeamCache
from src.nested_features import NESTED_NAMES, extend, probability
from src.retrieval import CatalogRetriever
from src.train_ranker import PARAMS
from src.validate import load_folds
from tools import reverse_ranker_experiment as prior

TAG = "nested_combiner_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG
PROTOCOL = ROOT / "experiments" / f"{TAG}_protocol.json"
MODELS = {"qwen": Path("D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B"),
          "bge": Path("D:/KSH/hf_models/BAAI__bge-reranker-v2-m3")}


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    joblib.dump(value, tmp)
    tmp.replace(path)


def signature():
    h = hashlib.sha256(PROTOCOL.read_bytes())
    for fn in (frames,prepare_inner,encode):
        h.update(inspect.getsource(fn).encode())
    for p in [ROOT/"src/nested_features.py", ROOT/"src/features.py", ROOT/"src/retrieval.py",
              ROOT/"src/augment.py", ROOT/"src/neural_pair.py", ROOT/"src/pair_evidence.py",
              ROOT/"src/encoder_phonetics.py", ROOT/"train.csv", ROOT/"vessel_catalog.csv"]:
        h.update(p.read_bytes())
    return h.hexdigest()


def frames(outer):
    train = load_train()
    folds = load_folds(train)
    han = train.vessel_name.map(is_hangul).to_numpy()
    outer_ids = set(train.loc[folds == outer, "target_vessel_id"])
    assert not outer_ids & set(train.loc[folds != outer, "target_vessel_id"])
    selected = np.flatnonzero(folds != outer)
    split = np.full(len(train), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=20261029)
    for k, (_, valid) in enumerate(sgkf.split(train.iloc[selected], han[selected],
                                             groups=train.target_vessel_id.iloc[selected])):
        split[selected[valid]] = k
    original = train.copy()
    original["origin"] = np.arange(len(train))
    original["meta_scenario"] = "clean"
    original["meta_weight"] = .44
    aug = augment(train, 4, 1, seed=7)
    aug["meta_scenario"], aug["meta_weight"] = "augmentation", .10
    tl = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=han))
    tl["meta_scenario"] = tl.scenario
    tl["meta_weight"] = tl.scenario.map(TESTLIKE_MIX)
    population = pd.concat([original.loc[han], aug.loc[han[aug.origin.to_numpy()]], tl], ignore_index=True)
    population = population.loc[folds[population.origin.to_numpy()] != outer].reset_index(drop=True)
    return train, folds, split, original, aug, population, outer_ids


def prepare_inner(outer, inner):
    folder = FOLDER/f"outer{outer}_inner{inner}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"data.joblib"
    fp = signature()
    if path.exists():
        value = joblib.load(path, mmap_mode="r")
        assert value["fingerprint"] == fp
        return value
    train, folds, split, original, aug, population, outer_ids = frames(outer)
    han = train.vessel_name.map(is_hangul).to_numpy()
    fit = (folds != outer) & (split != inner)
    held = (folds != outer) & (split == inner)
    tr = pd.concat([original.loc[fit & han], aug.loc[fit[aug.origin.to_numpy()] & han[aug.origin.to_numpy()]]], ignore_index=True)
    meta = population.loc[held[population.origin.to_numpy()] & population.vessel_name.map(is_hangul)].reset_index(drop=True)
    fit_ids, held_ids = set(train.loc[fit,"target_vessel_id"]), set(train.loc[held,"target_vessel_id"])
    assert not fit_ids & held_ids and not outer_ids & (fit_ids|held_ids)
    catalog = load_catalog()
    r = CatalogRetriever(catalog)
    builder = FeatureBuilder(r, *fit_mappers(train.loc[fit], catalog))
    r.category_score = builder.category_score
    beams = BeamCache(allow_generate=False)
    truth = train.target_vessel_id.map(r.id_to_index).to_numpy()
    start = time.perf_counter()

    def matrix(df, name):
        cache = folder/f"{name}_matrix.joblib"
        if cache.exists():
            result = joblib.load(cache, mmap_mode="r")
            assert result["fingerprint"] == fp and result["sample_ids"] == df.sample_id.tolist()
            return result
        bb = beams(df.vessel_name.tolist())

        def chunk(offset):
            part = df.iloc[offset:offset+48]
            signals = r.generate(part, llm_beams=bb[offset:offset+48])
            return [(sig["cand"], builder.build(row, sig)) for row, sig in zip(part.to_dict("records"), signals)]

        blocks, candidates = [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for number, part in enumerate(pool.map(chunk, range(0,len(df),48)), 1):
                candidates.extend(c for c,b in part)
                blocks.extend(b for c,b in part)
                if number % 20 == 0:
                    print(f"nested {outer}/{inner} {name}: {min(number*48,len(df))}/{len(df)}", flush=True)
        result = {"blocks": blocks, "candidates": candidates, "truth": truth[df.origin.to_numpy()],
                  "sample_ids": df.sample_id.tolist(), "fingerprint": fp}
        dump(result,cache)
        return result

    tm = matrix(tr, "train")
    keep = [i for i,(c,t) in enumerate(zip(tm["candidates"],tm["truth"])) if t in c]
    x = np.vstack([tm["blocks"][i] for i in keep])
    y = np.concatenate([(tm["candidates"][i] == tm["truth"][i]).astype(int) for i in keep])
    proxy = lgb.train(PARAMS,lgb.Dataset(x,label=y,group=[len(tm["candidates"][i]) for i in keep],
                                       feature_name=FEATURE_NAMES),num_boost_round=150)
    proxy.save_model(str(folder/"proxy.txt"))
    del x,y
    gc.collect()
    nn_groups = []
    columns = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max","key_lev","best_gap")]
    for i in np.random.default_rng(42).permutation(len(tr)):
        c,b,t = tm["candidates"][i], tm["blocks"][i], tm["truth"][i]
        positive = np.flatnonzero(c == t)
        if not len(positive) or len(c)<8:
            continue
        score = proxy.predict(b,num_threads=4)
        strength = np.maximum(b[:,columns[0]],b[:,columns[1]])-.2*b[:,columns[2]]
        learned = [int(j) for j in np.argsort(-score,kind="stable") if j != positive[0]][:5]
        lexical = [int(j) for j in np.argsort(-strength,kind="stable") if j != positive[0] and j not in learned][:2]
        row = tr.iloc[i].to_dict()
        nn_groups.append({"row":row,"candidates":c[[int(positive[0])]+learned+lexical].copy(),
                          "beams":beams([row["vessel_name"]])[0]})
        if len(nn_groups)==2400:
            break
    del tm
    gc.collect()
    mm = matrix(meta,"meta")
    meta_groups = []
    absent = 0
    for i,row in enumerate(meta.to_dict("records")):
        c,b,t = mm["candidates"][i],mm["blocks"][i],mm["truth"][i]
        score = proxy.predict(b,num_threads=4)
        chosen = np.argsort(-score,kind="stable")[:20]
        if t not in c[chosen]:
            absent += 1
            continue
        meta_groups.append({"row":row,"name":row["vessel_name"],"names":catalog.vessel_name.iloc[c[chosen]].astype(str).tolist(),
                            "sample_id":row["sample_id"],"candidates":c[chosen].copy(),"truth":int(t),
                            "raw":b[chosen].copy(),"proxy":score[chosen],"beams":beams([row["vessel_name"]])[0],
                            "weight":float(row["meta_weight"])})
    del mm
    report = {"outer":outer,"inner":inner,"mapper_training_rows":int(fit.sum()),
              "nn_groups":len(nn_groups),"meta_groups":len(meta_groups),"meta_truth_absent_skipped":absent,
              "fit_vessels":sorted(fit_ids),"meta_vessels":sorted(held_ids),"outer_validation_vessels":sorted(outer_ids),
              "overlap":0,"seconds":time.perf_counter()-start,"fingerprint":fp}
    value = {"training":nn_groups,"meta":meta_groups,"report":report,"fingerprint":fp}
    dump(value,path)
    (folder/"preparation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Prepared nested {outer}/{inner}: NN{len(nn_groups)} meta{len(meta_groups)}",flush=True)
    return value


def encode(kind, tokenizer, item, catalog):
    if kind == "qwen":
        from src.pair_evidence import encode_pairs
    else:
        from src.encoder_phonetics import encode_pairs
    return encode_pairs(tokenizer,item["row"],[catalog[int(j)] for j in item["candidates"]],item["beams"])


def seed():
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def train_inner(outer, inner, kind, data):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,PeftModel,get_peft_model
    from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer
    folder = FOLDER/f"outer{outer}_inner{inner}"/kind
    merged = folder/"merged"
    training_fp=hashlib.sha256((data["fingerprint"]+inspect.getsource(train_inner)+inspect.getsource(seed)).encode()).hexdigest()
    if (folder/"training.json").exists():
        result = json.loads((folder/"training.json").read_text())
        assert result["fingerprint"]==data["fingerprint"]
        assert result["training_fingerprint"]==training_fp
        return merged
    folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[kind],local_files_only=True,
                                             padding_side="left" if kind=="qwen" else "right")
    catalog = load_catalog().to_dict("records")
    groups = [encode(kind,tokenizer,g,catalog) for g in data["training"]]
    assert all(len(g)==8 for g in groups)
    cls = AutoModelForCausalLM if kind=="qwen" else AutoModelForSequenceClassification
    seed()
    model = cls.from_pretrained(MODELS[kind],local_files_only=True,dtype=torch.bfloat16).to("cuda")
    model.config.use_cache=False
    checkpoints = sorted(p for p in folder.glob("checkpoint_*/state.pt") if (p.parent/"ready.json").exists())
    completed=0
    if checkpoints:
        state = torch.load(checkpoints[-1],map_location="cuda",weights_only=False)
        assert state["fingerprint"]==data["fingerprint"]
        assert state["training_fingerprint"]==training_fp
        model = PeftModel.from_pretrained(model,checkpoints[-1].parent,is_trainable=True)
        completed=state["completed"]
    else:
        seed()
        config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                          target_modules=["q_proj","k_proj","v_proj","o_proj"] if kind=="qwen" else ["query","key","value"],
                          modules_to_save=None if kind=="qwen" else ["classifier"],
                          task_type="CAUSAL_LM" if kind=="qwen" else "SEQ_CLS",bias="none")
        model=get_peft_model(model,config)
    if kind=="qwen":
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if checkpoints:
        assert state["trainable_names"]==trainable
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        del state
    yes,no=[tokenizer.convert_tokens_to_ids(w) for w in ("yes","no")] if kind=="qwen" else (None,None)
    order=np.random.default_rng(42).permutation(len(groups))
    start=time.perf_counter();losses=[]
    model.train()
    for offset in range(completed*2,len(order),2):
        selected=order[offset:offset+2]
        tokens=[t for i in selected for t in groups[i]]
        enc=tokenizer.pad({"input_ids":tokens},padding=True,pad_to_multiple_of=8,return_tensors="pt").to("cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if kind=="qwen":
                logits=model(**enc,use_cache=False,logits_to_keep=1).logits[:,-1].float()
                scores=logits[:,yes]-logits[:,no]
            else:
                scores=model(**enc).logits.reshape(-1).float()
            loss=F.cross_entropy(scores.reshape(len(selected),8),torch.zeros(len(selected),dtype=torch.long,device="cuda"))
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite nested loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        step=offset//2+1
        losses.append(float(loss.item()))
        if step%100==0:
            print(f"nested {outer}/{inner} {kind} step{step} loss{np.mean(losses[-100:]):.4f} seconds{time.perf_counter()-start:.1f}",flush=True)
        if step%200==0:
            cp=folder/f"checkpoint_{step:06d}"
            model.save_pretrained(cp)
            state={"completed":step,"fingerprint":data["fingerprint"],"training_fingerprint":training_fp,"optimizer":optimizer.state_dict(),
                   "trainable_names":trainable,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                   "numpy_rng":np.random.get_state(),"python_rng":random.getstate()}
            torch.save(state,cp/"state.pt")
            (cp/"ready.json").write_text(json.dumps({"step":step}),encoding="utf-8")
            del state
    seconds=time.perf_counter()-start
    model.save_pretrained(folder/"adapter")
    model=model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged);tokenizer.save_pretrained(merged)
    result={"fingerprint":data["fingerprint"],"training_fingerprint":training_fp,"kind":kind,"groups":len(groups),"steps":(len(groups)+1)//2,
            "seconds_this_segment":seconds,"resumed_after_step":completed,"finite_loss":True,
            "at_token_cap":sum(len(t)==256 for g in groups for t in g),"mean_loss_this_segment":float(np.mean(losses)) if losses else None}
    (folder/"training.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    enc=scores=loss=None
    del model,optimizer,groups
    gc.collect();torch.cuda.empty_cache()
    return merged


def scorer(kind, model_dir):
    if kind=="qwen":
        from src.pair_views import PairViewScorer
        return PairViewScorer(model_dir,"evidence")
    from src.encoder_phonetics import EncoderPhoneticScorer
    return EncoderPhoneticScorer(model_dir)


def score_groups(kind, model_dir, groups, path, fp):
    import torch
    torch.set_num_threads(4)
    digest=hashlib.sha256((fp+str(model_dir)).encode())
    digest.update((model_dir/"config.json").read_bytes())
    for p in sorted(model_dir.glob("*.safetensors")):
        digest.update(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    fp=digest.hexdigest()
    data=joblib.load(path) if path.exists() else {"fingerprint":fp,"scores":[]}
    assert data["fingerprint"]==fp
    if len(data["scores"])==len(groups):return data["scores"]
    model=scorer(kind,model_dir)
    catalog=load_catalog().to_dict("records")
    memo={}
    start=time.perf_counter()
    for i in range(len(data["scores"]),len(groups)):
        g=groups[i]
        if not is_hangul(g["row"]["vessel_name"]):
            values=None
        else:
            # Memoization includes the complete ordered row/candidate token group.
            tokens=encode(kind,model.tokenizer,g,catalog)
            key=tuple(tuple(t) for t in tokens)
            if key not in memo:
                memo[key]=model.score(g["row"],[catalog[int(j)] for j in g["candidates"]],g["beams"])
            values=memo[key]
            assert np.isfinite(values).all() and len(values)==len(g["candidates"])
        data["scores"].append(values)
        if (i+1)%200==0 or i+1==len(groups):
            dump(data,path)
            print(f"{path.stem} {i+1}/{len(groups)} seconds{time.perf_counter()-start:.1f}",flush=True)
    del model
    gc.collect();torch.cuda.empty_cache()
    return data["scores"]


def run_inner(outer,inner):
    data=prepare_inner(outer,inner)
    folder=FOLDER/f"outer{outer}_inner{inner}"
    for kind in ("bge","qwen"):
        merged=train_inner(outer,inner,kind,data)
        score_groups(kind,merged,data["meta"],folder/f"meta_{kind}_scores.joblib",data["fingerprint"])


def prepare_outer(outer):
    path=FOLDER/f"outer{outer}_data.joblib"
    fp=signature()
    if path.exists():
        data=joblib.load(path)
        assert data["fingerprint"]==fp
        return data
    base=prior.prepare(outer)
    _,catalog,_,frames_df,_=prior.frames_and_training(outer)
    beams=BeamCache(allow_generate=False)
    miner=lgb.Booster(model_file=str(ROOT/f"experiments/artifacts/strict_han_v1/base_fold{outer}.txt"))
    sets={}
    for s,groups in base["evaluation"].items():
        records={r["sample_id"]:r for r in frames_df[s].to_dict("records")}
        sets[s]=[{**g,"row":records[g["sample_id"]],"proxy":miner.predict(g["raw"],num_threads=4),
                  "beams":beams([g["name"]])[0]} for g in groups]
    data={"sets":sets,"fingerprint":fp}
    dump(data,path)
    return data


def outer_qwen(outer):
    original=ROOT/f"experiments/artifacts/neural_evidence_v1/fold{outer}_runtime_merged"
    if original.exists():return original
    target=FOLDER/f"outer{outer}_qwen_merged"
    if (target/"config.json").exists():return target
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model=AutoModelForCausalLM.from_pretrained(MODELS["qwen"],local_files_only=True,dtype=torch.bfloat16).to("cuda")
    model=PeftModel.from_pretrained(model,ROOT/f"experiments/artifacts/neural_evidence_v1/fold{outer}_adapter").merge_and_unload().eval()
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(MODELS["qwen"],local_files_only=True).save_pretrained(target)
    del model;gc.collect();torch.cuda.empty_cache()
    return target


def run_outer(outer):
    data=prepare_outer(outer)
    groups=[g for v in data["sets"].values() for g in v]
    precision={}
    for kind in ("bge","qwen"):
        model_dir=ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{outer}_merged" if kind=="bge" else outer_qwen(outer)
        scores=score_groups(kind,model_dir,groups,FOLDER/f"outer{outer}_{kind}_scores.joblib",data["fingerprint"])
        old=joblib.load(ROOT/f"experiments/artifacts/{'encoder_bge_phonetic' if kind=='bge' else 'neural_evidence_v1'}/fold{outer}_epoch1_scores.joblib")
        prior_rows=[r for v in old["sets"].values() for r in v]
        assert len(prior_rows)==len(scores)
        differences=[]
        for g,new,previous in zip(groups,scores,prior_rows):
            np.testing.assert_array_equal(g["candidates"],previous["candidates"][previous["keep"]])
            if previous["neural_scores"] is not None and g["actual_hangul"]:
                assert new.shape==previous["neural_scores"].shape and np.isfinite(new).all()
                differences.append(float(np.max(np.abs(new-previous["neural_scores"]))))
        precision[kind]={"active_rows":len(differences),"exact_rows":sum(d==0 for d in differences),
                         "maximum_abs_logit_delta":max(differences,default=0.)}
        print(f"Outer{outer} {kind}: archive precision {precision[kind]}",flush=True)
    archive=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    q=iter(joblib.load(FOLDER/f"outer{outer}_qwen_scores.joblib")["scores"])
    b=iter(joblib.load(FOLDER/f"outer{outer}_bge_scores.joblib")["scores"])
    for s,rows in data["sets"].items():
        position={sid:i for i,sid in enumerate(archive["identity"]["sets"][s]["sample_ids"])}
        for g in rows:
            qq,bb=next(q),next(b)
            if not g["actual_hangul"]:continue
            bp=g["base"]/g["base"].sum()
            anchor=.75*bp+.125*normalize(qq,"prob").astype(float)+.125*normalize(bb,"prob").astype(float)
            assert g["candidates"][np.argmax(anchor)]==archive["ensemble"][s][position[g["sample_id"]]], "v12 winner changed"
    precision["all_v12_winners_exact"]=True
    (FOLDER/f"outer{outer}_precision.json").write_text(json.dumps(precision,indent=2),encoding="utf-8")
    return data


def neural_features(g,q,b):
    return extend(g["raw"],g["proxy"],q,b)


def fit_heads(outers):
    archived=joblib.load(ROOT/"experiments/encoder_bge_phonetic_predictions.joblib")
    positions={s:{sid:i for i,sid in enumerate(archived["identity"]["sets"][s]["sample_ids"])} for s in archived["ensemble"]}
    for outer in outers:
        groups=[];xs=[]
        for inner in (0,1):
            data=prepare_inner(outer,inner)
            folder=FOLDER/f"outer{outer}_inner{inner}"
            q=joblib.load(folder/"meta_qwen_scores.joblib")["scores"]
            b=joblib.load(folder/"meta_bge_scores.joblib")["scores"]
            assert len(q)==len(b)==len(data["meta"])
            groups.extend(data["meta"])
            xs.extend(neural_features(g,qq,bb) for g,qq,bb in zip(data["meta"],q,b))
        x=np.vstack(xs)
        y=np.concatenate([g["candidates"]==g["truth"] for g in groups]).astype(int)
        weight=np.concatenate([np.full(len(g["candidates"]),g["weight"]) for g in groups])
        size=[len(g["candidates"]) for g in groups]
        assert y.sum()==len(groups)
        outer_data=prepare_outer(outer)
        q=iter(joblib.load(FOLDER/f"outer{outer}_qwen_scores.joblib")["scores"])
        b=iter(joblib.load(FOLDER/f"outer{outer}_bge_scores.joblib")["scores"])
        evaluated={}
        for s,rows in outer_data["sets"].items():
            evaluated[s]=[]
            for g in rows:
                qq,bb=next(q),next(b)
                anchor=None;xx=None
                if g["actual_hangul"]:
                    bp=g["base"]/g["base"].sum()
                    anchor=.75*bp+.125*normalize(qq,"prob").astype(float)+.125*normalize(bb,"prob").astype(float)
                    ref=archived["ensemble"][s][positions[s][g["sample_id"]]]
                    assert g["candidates"][np.argmax(anchor)]==ref, "Exact v12 reconstruction failed"
                    xx=neural_features(g,qq,bb)
                evaluated[s].append((g,anchor,xx))
        result={"fold":outer,"sets":{},"controls":{},"fingerprint":signature(),"meta_groups":len(groups)}
        for recipe,width in (("sets",105),("controls",78)):
            model=lgb.train(PARAMS,lgb.Dataset(x[:,:width],label=y,group=size,weight=weight,
                                               feature_name=NESTED_NAMES[:width]),num_boost_round=150)
            model.save_model(str(FOLDER/f"fold{outer}_{recipe}_head.txt"))
            for s,rows in evaluated.items():
                result[recipe][s]=[]
                for g,anchor,xx in rows:
                    scores=None;pred={"anchor50":None,"direct":None}
                    if xx is not None:
                        scores=model.predict(xx[:,:width],num_threads=4)
                        pred={"direct":int(g["candidates"][np.argmax(scores)]),
                              "anchor50":int(g["candidates"][np.argmax(.5*anchor+.5*probability(scores))])}
                    result[recipe][s].append({"sample_id":g["sample_id"],"truth":g["truth"],"predictions":pred,
                                              "scores":scores,"truth_in_top20":bool(g["truth"] in g["candidates"])})
        dump(result,FOLDER/f"fold{outer}_results.joblib")
        print(f"Nested head fold{outer} trained on{len(groups)} out-of-sample groups",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--folds",nargs="+",type=int,default=[0,2])
    p.add_argument("--inner",nargs="+",type=int,default=[0,1])
    p.add_argument("--stage",choices=["prepare","inner","outer","fit","all"],default="all")
    args=p.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True)
    for outer in args.folds:
        if args.stage in ("prepare","inner","all"):
            for inner in args.inner:
                prepare_inner(outer,inner) if args.stage=="prepare" else run_inner(outer,inner)
        if args.stage in ("outer","all"):run_outer(outer)
    if args.stage in ("fit","all"):
        fit_heads(args.folds)
        from tools.audit_reverse_ranker import audit
        audit(FOLDER,PROTOCOL)


if __name__=="__main__":
    main()
