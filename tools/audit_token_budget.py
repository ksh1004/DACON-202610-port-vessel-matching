"""Inspect real neural inputs without fitting models or using error labels."""
from collections import Counter
import hashlib
import json

import joblib
import numpy as np
from transformers import AutoTokenizer

from src.common import ROOT,load_catalog,is_hangul
from src.encoder_phonetics import text as bge_text
from src.neural_pair import PREFIX,SUFFIX,record_text
from src.pair_evidence import INSTRUCTION
from src.llm_cache import BeamCache


def main():
    qwen=AutoTokenizer.from_pretrained("D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B",local_files_only=True,padding_side="left")
    bge=AutoTokenizer.from_pretrained("D:/KSH/hf_models/BAAI__bge-reranker-v2-m3",local_files_only=True,padding_side="right")
    prefix=len(qwen.encode(PREFIX,add_special_tokens=False));suffix=len(qwen.encode(SUFFIX,add_special_tokens=False))
    budget=256-prefix-suffix;catalog=load_catalog().to_dict("records");beams=BeamCache(allow_generate=False)
    reports={};sources={}
    for fold in (1,4):
        path=ROOT/f"experiments/artifacts/encoder_ambiguity_v1/fold{fold}_data.joblib"
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=joblib.load(path);groups=[]
        for g in data["selected"][:256]:groups.append(("training_sample",g["row"],g["candidates"],g["beams"]))
        for s,rows in data["evaluation"].items():
            for g in rows:
                ref=g["reference"];row=g["row"]
                if is_hangul(row["vessel_name"]) and ref["neural_scores"] is not None:
                    groups.append((s,row,ref["candidates"][ref["keep"]],beams([row["vessel_name"]])[0]))
        for source in dict.fromkeys(g[0] for g in groups):
            chosen=[g for g in groups if g[0]==source];texts=[];queries=[];docs=[];field_ends=[]
            for _,row,candidates,bb in chosen:
                query=json.loads(record_text(row));query["possible_latin_spellings"]=list(dict.fromkeys(str(b) for b in bb))[:4]
                query=json.dumps(query,ensure_ascii=False,separators=(",",":"))
                for index in candidates:
                    candidate=catalog[int(index)];document=record_text(candidate,catalog=True)
                    body=f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
                    start=len(body)-len(document);obj=json.loads(document);ends={}
                    for key,value in obj.items():
                        marker=json.dumps(key)+":";field_start=document.index(marker)+len(marker)
                        ends[key]=start+field_start+len(json.dumps(value,ensure_ascii=False,separators=(",",":")))
                    texts.append(body);field_ends.append(ends)
                    queries.append(bge_text(row,beams=bb));docs.append(bge_text(candidate,catalog=True))
            qlengths=[];blengths=[];qcropped=0;lost=Counter()
            for offset in range(0,len(texts),256):
                encoded=qwen(texts[offset:offset+256],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
                for local,(tokens,spans) in enumerate(zip(encoded["input_ids"],encoded["offset_mapping"])):
                    qlengths.append(prefix+suffix+len(tokens))
                    if len(tokens)>budget:
                        qcropped+=1;cut=spans[budget-1][1]
                        for key,end in field_ends[offset+local].items():lost[key]+=int(cut<end)
                encoded_b=bge(queries[offset:offset+256],docs[offset:offset+256],truncation=False)
                blengths.extend(map(len,encoded_b["input_ids"]))
            def summary(values):
                return dict(min=int(min(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=int(max(values)))
            reports[f"fold{fold}_{source}"]=dict(groups=len(chosen),pairs=len(texts),
                qwen_full_lengths=summary(qlengths),qwen_truncated_pairs=qcropped,
                qwen_candidate_field_incomplete=dict(lost),bge_full_lengths=summary(blengths),
                bge_truncated_pairs=int(sum(v>256 for v in blengths)))
            print(json.dumps({f"fold{fold}_{source}":reports[f"fold{fold}_{source}"]}),flush=True)
    report=dict(client_date="2026-10-30",role="Input-only diagnostic; no new model, accuracy or Public claim",
                max_length=256,qwen_prefix_tokens=prefix,qwen_suffix_tokens=suffix,qwen_body_budget=budget,
                query_groups="First256 seeded training groups per fold plus all previously neural-active heldout groups; no error filtering",
                reports=reports,source_hashes=sources,
                script_sha256=hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest())
    (ROOT/"experiments/token_budget_20261030.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
