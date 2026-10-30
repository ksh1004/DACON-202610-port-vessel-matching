"""Inference for a feature-rich LightGBM -> top20 CatBoost ranker."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRanker

from src.ensemble import Ensemble, normalize
from src.features import FEATURE_NAMES
from src.match_features import EXTRA_FEATURE_NAMES, ExtraFeatures


class MatchingRanker:
    def __init__(self, primary, second, extra, top_k=20, mode="cat_only", base_ensemble=None, blend_weight=2.0):
        self.primary, self.second, self.extra = primary, second, extra
        self.top_k, self.mode = int(top_k), mode
        self.base_ensemble, self.blend_weight = base_ensemble, float(blend_weight)

    @classmethod
    def load(cls, folder: Path, retriever):
        spec = json.loads((folder / "matching.json").read_text(encoding="utf-8"))
        if spec["feature_names"] != FEATURE_NAMES + EXTRA_FEATURE_NAMES:
            raise ValueError("Matching ranker feature version mismatch")
        for filename, expected in spec.get("source_hashes", {}).items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Matching feature implementation mismatch: {filename}")
        primary = lgb.Booster(model_file=str(folder / spec["primary"]))
        second = CatBoostRanker().load_model(str(folder / spec["second"]))
        if primary.num_feature() != len(spec["feature_names"]) or len(second.feature_names_) != len(spec["feature_names"]):
            raise ValueError("Matching model feature count mismatch")
        if spec["mode"] not in ("cat_only", "blend") or not 1 <= spec["top_k"] <= 100:
            raise ValueError("Invalid matching ranker settings")
        base = Ensemble.load(folder) if spec.get("base_ensemble") else None
        return cls(primary, second, ExtraFeatures(retriever), spec["top_k"], spec["mode"],
                   base, spec.get("blend_weight", 2.0))

    def extend(self, row, signal, base):
        return np.column_stack([base, self.extra.build(row, signal["cand"], base, signal.get("llm_beams", []))])

    def score_groups(self, blocks):
        if not blocks:
            return []
        if not any(len(b) for b in blocks):
            return [np.empty(0, dtype=np.float32) for _ in blocks]
        bounds = np.cumsum([0] + [len(b) for b in blocks])
        # Match the OOF score precision used to choose the top20 candidates.
        raw = np.asarray(self.primary.predict(np.vstack(blocks), num_threads=6), dtype=np.float32)
        primary = [raw[bounds[i]:bounds[i + 1]] for i in range(len(blocks))]
        picks = [np.argsort(-sc, kind="stable")[:self.top_k] for sc in primary]
        sizes = [len(p) for p in picks]
        if not sum(sizes):
            return [np.empty(0, dtype=np.float32) for _ in blocks]
        selected = np.vstack([b[p] for b, p in zip(blocks, picks) if len(p)])
        second_raw = self.second.predict(selected, thread_count=6)
        output, offset = [], 0
        for sc, pick in zip(primary, picks):
            new = second_raw[offset:offset + len(pick)]
            offset += len(pick)
            final = np.full(len(sc), -1000.0, dtype=np.float32)
            if len(pick):
                if self.mode == "cat_only":
                    final[pick] = new
                else:
                    probability = .5 * normalize(sc[pick], "prob") + .5 * normalize(new, "prob")
                    final[pick] = np.log(np.maximum(probability, 1e-30))
            output.append(final)
        if self.base_ensemble is not None:
            base = self.base_ensemble.score_groups([b[:, :len(FEATURE_NAMES)] for b in blocks])
            output = [old + self.blend_weight * normalize(new, "prob") for old, new in zip(base, output)]
        return output
