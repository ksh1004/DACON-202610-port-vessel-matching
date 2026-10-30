"""Fixed-shape generic Qwen3-Reranker-4B scoring of official record pairs."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

from src.neural_pair import PREFIX,SUFFIX,record_text
from src.pair_evidence import INSTRUCTION

MODEL=Path("D:/KSH/hf_models/Qwen__Qwen3-Reranker-4B")
REVISION="22e683669bc0f0bd69640a1354a6d0aebcfeede5"
PAD_LENGTH=256


def encode(tokenizer,row,candidates,beams=()):
    query=json.loads(record_text(row))
    query["possible_latin_spellings"]=list(dict.fromkeys(str(b) for b in beams))[:4]
    query=json.dumps(query,ensure_ascii=False,separators=(",",":"))
    prefix=tokenizer.encode(PREFIX,add_special_tokens=False);suffix=tokenizer.encode(SUFFIX,add_special_tokens=False)
    result=[]
    for candidate in candidates:
        body=f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {record_text(candidate,catalog=True)}"
        tokens=prefix+tokenizer.encode(body,add_special_tokens=False)+suffix
        if len(tokens)>PAD_LENGTH:raise ValueError("Dedicated4B probe input exceeds256; no silent truncation")
        result.append(tokens)
    return result


class Reranker4B:
    def __init__(self,batch_size=4):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="left")
        assert self.tokenizer.pad_token_id is not None
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
                    dtype=torch.bfloat16,device_map={"":"cuda"},attn_implementation="sdpa",low_cpu_mem_usage=True).eval()
        self.model.config.use_cache=False
        self.yes,self.no=(self.tokenizer.convert_tokens_to_ids(t) for t in ("yes","no"))

    @torch.inference_mode()
    def score_tokens(self,pairs):
        values=[]
        for offset in range(0,len(pairs),self.batch_size):
            chunk=pairs[offset:offset+self.batch_size];n=len(chunk)
            while len(chunk)<self.batch_size:chunk=chunk+[chunk[-1]]
            encoded=self.tokenizer.pad({"input_ids":chunk},padding="max_length",max_length=PAD_LENGTH,return_tensors="pt").to("cuda")
            positions=encoded["attention_mask"].long().cumsum(-1)-1
            positions.masked_fill_(encoded["attention_mask"]==0,0)
            logits=self.model(**encoded,position_ids=positions,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            values.extend((logits[:n,self.yes]-logits[:n,self.no]).cpu().numpy().tolist())
        return np.asarray(values,dtype=np.float32)

    def score(self,row,candidates,beams=()):
        return self.score_tokens(encode(self.tokenizer,row,candidates,beams))
