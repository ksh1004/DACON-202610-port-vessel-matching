"""Offline inference for one or two fixed views of a candidate pair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.common import is_hangul
from src.ensemble import Ensemble, normalize
from src.features import FEATURE_NAMES
from src.neural_pair import PairScorer
from src.pair_evidence import encode_pairs


def mix_views(base_probability, neural_scores, view_weights, weight=.25):
    mixed = (1-weight)*np.asarray(base_probability, dtype=np.float64)
    total = sum(view_weights)
    for score, view_weight in zip(neural_scores, view_weights):
        mixed += weight*view_weight/total*normalize(score, "prob").astype(np.float64)
    return mixed


class PairViewScorer(PairScorer):
    def __init__(self, model_dir, recipe):
        if recipe not in {"raw", "evidence"}:
            raise ValueError(f"Unknown pair view: {recipe}")
        super().__init__(model_dir)
        self.recipe = recipe

    @torch.inference_mode()
    def score(self, row, catalog_rows, beams=()):
        if self.recipe == "raw":
            return super().score(row, catalog_rows)
        if not catalog_rows:
            return np.empty(0, dtype=np.float32)
        tokens = encode_pairs(self.tokenizer, row, catalog_rows, beams)
        enc = self.tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8,
                                  return_tensors="pt").to(self.device)
        logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return (logits[:, self.yes]-logits[:, self.no]).cpu().numpy()


class PairViewsEnsemble:
    def __init__(self, baseline, views, retriever, top_k=20, weight=.25):
        self.baseline, self.views, self.r = baseline, views, retriever
        self.top_k, self.weight = top_k, weight

    @classmethod
    def load(cls, folder: Path, retriever):
        spec = json.loads((folder / "pair_views.json").read_text(encoding="utf-8"))
        if spec["feature_names"] != FEATURE_NAMES or not 0 < spec["weight"] < .5:
            raise ValueError("Pair-view ensemble schema/weight mismatch")
        if spec["top_k"] != 20 or not spec["views"] or any(v["weight"] <= 0 for v in spec["views"]):
            raise ValueError("Invalid pair-view configuration")
        for filename, expected in spec["source_hashes"].items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Pair-view inference source mismatch: {filename}")
        views = [(PairViewScorer(folder / v["dir"], v["recipe"]), v["weight"]) for v in spec["views"]]
        return cls(Ensemble.load(folder), views, retriever, spec["top_k"], spec["weight"])

    def score_records(self, rows, signals, blocks):
        baseline = self.baseline.score_groups(blocks)
        output = []
        for row, sig, base in zip(rows, signals, baseline):
            if not is_hangul(row.get("vessel_name")):
                output.append(base)
                continue
            keep = np.argsort(-base, kind="stable")[:self.top_k]
            if len(keep) < 2:
                output.append(base)
                continue
            bp = base[keep]/base[keep].sum()
            if bp[0]-bp[1] > self.weight/(1-self.weight)+1e-6:
                output.append(base)
                continue
            candidates = [self.r.catalog.iloc[i].to_dict() for i in sig["cand"][keep]]
            scores = [scorer.score(row, candidates, sig.get("llm_beams", ())) for scorer, _ in self.views]
            mixed = mix_views(bp, scores, [w for _, w in self.views], self.weight)
            final = np.zeros(len(base), dtype=np.float64)
            final[keep] = mixed
            output.append(final)
        return output
