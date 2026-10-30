"""Offline-only pair scoring shared with the supervised reranker protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import is_hangul
from src.ensemble import Ensemble, normalize
from src.features import FEATURE_NAMES


PREFIX = ('<|im_start|>system\nDetermine whether the two provided records identify the same ship. '
          'Answer only yes or no.<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = ("Match a port record to a ship catalog entry. Names can have Korean transliteration, "
               "misspellings or omitted words. Tonnage can be missing or inaccurate. "
               "Compare name, numbers, tonnage, type and flag; missing fields do not indicate a mismatch.")


def record_text(row, catalog=False):
    keys = ["vessel_name", "gross_tonnage", "vessel_type", "flag_name"]
    if not catalog:
        keys += ["international_gross_tonnage"]
    values = {}
    for k in keys:
        value = row.get(k)
        if pd.isna(value):
            value = None
        elif not catalog and k in ("gross_tonnage", "international_gross_tonnage"):
            try:
                value = float(value)
                value = value if math.isfinite(value) else None
                if value == 0:
                    value = 0.0
            except (TypeError, ValueError):
                value = None
        values[k] = value
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def encode_pairs(tokenizer, row, catalog_rows):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    query = record_text(row)
    texts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {record_text(c, catalog=True)}"
             for c in catalog_rows]
    ids = tokenizer(texts, add_special_tokens=False, truncation=True,
                    max_length=256 - len(prefix) - len(suffix)).input_ids
    return [prefix + seq + suffix for seq in ids]


class PairScorer:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                         dtype=torch.bfloat16).to(device).eval()
        self.model.config.use_cache = False
        self.yes, self.no = [self.tokenizer.convert_tokens_to_ids(w) for w in ("yes", "no")]

    @torch.inference_mode()
    def score(self, row, catalog_rows):
        if not catalog_rows:
            return np.empty(0, dtype=np.float32)
        tokens = encode_pairs(self.tokenizer, row, catalog_rows)
        enc = self.tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt").to(self.device)
        logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return (logits[:, self.yes] - logits[:, self.no]).cpu().numpy()


class NeuralEnsemble:
    def __init__(self, baseline, scorer, retriever, top_k=20, weight=.25):
        self.baseline, self.scorer, self.r = baseline, scorer, retriever
        self.top_k, self.weight = int(top_k), float(weight)

    @classmethod
    def load(cls, folder: Path, retriever):
        spec = json.loads((folder / "neural.json").read_text(encoding="utf-8"))
        if spec["feature_names"] != FEATURE_NAMES or not 0 < spec["weight"] < .5:
            raise ValueError("Neural ensemble schema/weight mismatch")
        for filename, expected in spec.get("source_hashes", {}).items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Neural inference implementation mismatch: {filename}")
        # Eager loading fixes GPU model state before transliteration begins.
        return cls(Ensemble.load(folder), PairScorer(folder / spec["model_dir"]), retriever,
                   spec["top_k"], spec["weight"])

    def score_records(self, rows, signals, blocks):
        baseline = self.baseline.score_groups(blocks)
        output = []
        threshold = self.weight / (1 - self.weight)
        for row, sig, base in zip(rows, signals, baseline):
            if not is_hangul(row.get("vessel_name")):
                output.append(base)
                continue
            keep = np.argsort(-base, kind="stable")[:self.top_k]
            if len(keep) < 2:
                output.append(base)
                continue
            bp = base[keep] / base[keep].sum()
            if bp[0] - bp[1] > threshold + 1e-6:
                output.append(base)
                continue
            candidates = [self.r.catalog.iloc[i].to_dict() for i in sig["cand"][keep]]
            score = self.scorer.score(row, candidates)
            mixed = (1 - self.weight) * bp + self.weight * normalize(score, "prob")
            final = np.zeros(len(base), dtype=np.float64)
            final[keep] = mixed
            output.append(final)
        return output
