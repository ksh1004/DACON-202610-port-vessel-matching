"""Query-local structured pairs for a standard multilingual cross encoder."""
from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.neural_pair import record_text


def query_text(row, beams):
    values = json.loads(record_text(row))
    values["possible_latin_spellings"] = list(dict.fromkeys(str(b) for b in beams))[:4]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def encode_pairs(tokenizer, row, catalog_rows, beams):
    query = query_text(row, beams)
    documents = [record_text(c, catalog=True) for c in catalog_rows]
    return tokenizer([query]*len(documents), documents, padding=False, truncation=True,
                     max_length=256)["input_ids"]


class EncoderPairScorer:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="right")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16).to(device).eval()

    @torch.inference_mode()
    def score(self, row, catalog_rows, beams=()):
        if not catalog_rows:
            return np.empty(0, dtype=np.float32)
        tokens = encode_pairs(self.tokenizer, row, catalog_rows, beams)
        enc = self.tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt").to(self.device)
        return self.model(**enc).logits.reshape(-1).float().cpu().numpy()
