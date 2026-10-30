"""Fixed Qwen/encoder probability views; all computation is query-local."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.common import is_hangul
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetics import EncoderPhoneticScorer
from src.ensemble import Ensemble
from src.features import FEATURE_NAMES
from src.pair_views import PairViewScorer, mix_views


class ViewScorer:
    def __init__(self, directory, kind, recipe):
        self.name = f"{kind}:{recipe}"
        if kind == "qwen":
            self.scorer = PairViewScorer(directory, recipe)
        elif kind == "encoder" and recipe in {"plain", "phonetic"}:
            cls = EncoderPhoneticScorer if recipe == "phonetic" else EncoderPairScorer
            self.scorer = cls(directory)
        else:
            raise ValueError(f"Unknown scorer view: {self.name}")

    def score(self, row, candidates, beams):
        return self.scorer.score(row, candidates, beams)


class EncoderViewsEnsemble:
    def __init__(self, baseline, views, retriever, weight=.25, top_k=20):
        self.baseline, self.views, self.r = baseline, views, retriever
        self.weight, self.top_k = weight, top_k

    @classmethod
    def load(cls, folder: Path, retriever):
        spec = json.loads((folder / "encoder_views.json").read_text(encoding="utf-8"))
        if spec["feature_names"] != FEATURE_NAMES or spec["top_k"] != 20 or not 0 < spec["weight"] < .5:
            raise ValueError("Encoder-view schema/weight mismatch")
        if not spec["views"] or any(view["weight"] <= 0 for view in spec["views"]):
            raise ValueError("Invalid encoder-view weights")
        for name, expected in spec["source_hashes"].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Encoder-view source mismatch: {name}")
        baseline = Ensemble.load(folder)
        views = [(ViewScorer(folder / v["dir"], v["kind"], v["recipe"]), v["weight"]) for v in spec["views"]]
        return cls(baseline, views, retriever, spec["weight"], spec["top_k"])

    def score_records(self, rows, signals, blocks):
        output = []
        for row, sig, base in zip(rows, signals, self.baseline.score_groups(blocks)):
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
            full = np.zeros(len(base), dtype=np.float64)
            full[keep] = mixed
            output.append(full)
        return output
