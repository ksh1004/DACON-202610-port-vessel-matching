"""Explicit pair evidence and optional vowel-preserving character spelling.

Uses only the current record, its candidate, query-local Latin hints and fixed
linguistic rules. No learned alignment matrix, category map or ID lookup.
"""
import re
import unicodedata

import numpy as np
import torch

from src.common import normalize_name
from src.encoder_pair import EncoderPairScorer
from src.hangul import english_key
from src.retrieval import query_key
from src.learned_alignment import observed

MODES=("characters", "compact")
MAX_LENGTH=384


def positive(value):
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if np.isfinite(value) and value>0 else None


def number(value):
    value=positive(value)
    if value is None:return "unknown"
    return str(int(value)) if value.is_integer() else repr(value)


def label(value):
    return value.strip() if isinstance(value,str) and value.strip() else "unknown"


def gap(query,candidate):
    q,c=positive(query),positive(candidate)
    if q is None or c is None:return "unknown"
    difference=abs(q-c)/max(q,c,1.)
    if abs(q-c)<=1e-6:return "same"
    if difference<=.01:return "within 1 percent"
    if difference<=.05:return "within 5 percent"
    if difference<=.2:return "within 20 percent"
    return "different"


def characters(name,catalog=False):
    if catalog:
        words=re.findall(r"[a-z0-9]+",unicodedata.normalize("NFKD",str(name)).lower())
    else:
        words=[observed(w) for w in normalize_name(name).split()]
    return " | ".join(" ".join(w) for w in words if w)


def query_text(row,beams,mode):
    assert mode in MODES
    name=normalize_name(row.get("vessel_name"))
    key=" | ".join(" ".join(w) for w in query_key(row.get("vessel_name")).split())
    hints=list(dict.fromkeys(normalize_name(b) for b in beams if normalize_name(b)))[:4]
    lines=["Port record",f"Name: {name}",f"Consonants: {key}",
           f"GT: {number(row.get('gross_tonnage'))}",f"IGT: {number(row.get('international_gross_tonnage'))}",
           f"Type: {label(row.get('vessel_type'))}",f"Flag: {label(row.get('flag_name'))}",
           "Possible Latin names: "+" / ".join(hints)]
    if mode=="characters":lines.insert(3,"Spelling letters: "+characters(row.get("vessel_name")))
    return "\n".join(lines)


def document_text(row,candidate,mode):
    name=normalize_name(candidate.get("vessel_name"))
    key=" | ".join(" ".join(w) for w in english_key(candidate.get("vessel_name")).split())
    lines=["Catalog record",f"Name: {name}",f"Consonants: {key}",
           f"GT: {number(candidate.get('gross_tonnage'))}",f"Type: {label(candidate.get('vessel_type'))}",
           f"Flag: {label(candidate.get('flag_name'))}",
           "Port GT comparison: "+gap(row.get("gross_tonnage"),candidate.get("gross_tonnage")),
           "Port IGT comparison: "+gap(row.get("international_gross_tonnage"),candidate.get("gross_tonnage"))]
    if mode=="characters":lines.insert(3,"Spelling letters: "+characters(candidate.get("vessel_name"),True))
    return "\n".join(lines)


def encode_pairs(tokenizer,row,candidates,beams,mode):
    tokens=tokenizer([query_text(row,beams,mode)]*len(candidates),
                     [document_text(row,c,mode) for c in candidates],padding=False,truncation=False)["input_ids"]
    if any(len(t)>MAX_LENGTH for t in tokens):
        raise ValueError("Explicit representation exceeds frozen384-token ceiling; no silent truncation")
    return tokens


class ExplicitScorer(EncoderPairScorer):
    def __init__(self,model_dir,mode,device="cuda"):
        super().__init__(model_dir,device)
        assert mode in MODES;self.mode=mode

    @torch.inference_mode()
    def score(self,row,catalog_rows,beams=()):
        tokens=encode_pairs(self.tokenizer,row,catalog_rows,beams,self.mode)
        enc=self.tokenizer.pad({"input_ids":tokens},padding=True,pad_to_multiple_of=8,return_tensors="pt").to(self.device)
        return self.model(**enc).logits.reshape(-1).float().cpu().numpy()
