"""Explicit consonant-key alignment evidence for a multilingual encoder."""
import json

import numpy as np
import torch

from src.encoder_pair import EncoderPairScorer
from src.hangul import english_key
from src.neural_pair import record_text
from src.retrieval import query_key


def text(row, catalog=False, beams=()):
    values = json.loads(record_text(row, catalog=catalog))
    name = values.pop("vessel_name")
    key = english_key(name) if catalog else query_key(name)
    # Separate letters so the pretrained tokenizer can align individual
    # consonants, preserving the original word boundaries as bars.
    spelled = " | ".join(" ".join(word) for word in key.split())
    output = {"vessel_name": name, "phonetic_key": spelled, **values}
    if not catalog:
        output["possible_latin_spellings"] = list(dict.fromkeys(str(b) for b in beams))[:4]
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def encode_pairs(tokenizer, row, catalog_rows, beams):
    query = text(row, beams=beams)
    documents = [text(candidate, catalog=True) for candidate in catalog_rows]
    return tokenizer([query]*len(documents), documents, padding=False,
                     truncation=True, max_length=256)["input_ids"]


class EncoderPhoneticScorer(EncoderPairScorer):
    @torch.inference_mode()
    def score(self, row, catalog_rows, beams=()):
        if not catalog_rows:
            return np.empty(0, dtype=np.float32)
        tokens = encode_pairs(self.tokenizer, row, catalog_rows, beams)
        enc = self.tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt").to(self.device)
        return self.model(**enc).logits.reshape(-1).float().cpu().numpy()
