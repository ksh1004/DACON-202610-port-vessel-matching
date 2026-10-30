"""Additional pair features using only a query and the official catalog.

Kept separate from the 78-feature baseline so frozen models/caches stay usable.
No fitted vessel identities, record IDs, call signs, or other query rows are used.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from src.common import compact_name, normalize_name
from src.features import FEATURE_NAMES
from src.retrieval import query_key


EXTRA_FEATURE_NAMES = [
    "query_word_idf_cover", "query_key_idf_cover", "cand_key_idf_cover",
    "key_idf_fuzzy_cover", "key_order_alignment", "key_extra_idf",
    "beam_idf_cover", "cand_beam_idf_cover", "beam_idf_fuzzy_cover",
    "beam_order_alignment", "beam_extra_idf",
    "first_beam_ratio", "first_beam_token_set", "first_beam_exact",
    "beam_ratio_mean", "beam_ratio_std", "beam_support_85", "beam_exact_count", "first_beam_advantage",
    "tonnage_rank_similar_names", "tonnage_ties_similar_names", "tonnage_gap_similar_names",
    "similar_name_count", "name_strength_gap", "query_arrival_year", "query_tonnage_disagreement",
    "query_key_word_count", "query_beam_word_count",
]


class ExtraFeatures:
    def __init__(self, retriever):
        self.r = retriever
        self.name_tokens = [tuple(n.split()) for n in retriever.names]
        self.key_tokens = [tuple(n.split()) for n in retriever.keys]
        self.idf = {}
        for which, values in (("name", self.name_tokens), ("key", self.key_tokens)):
            counts = Counter(token for tokens in values for token in set(tokens))
            self.idf[which] = {t: float(np.log((len(values) + 1) / (n + 1)) + 1) for t, n in counts.items()}
        self.unknown_idf = float(np.log(len(self.name_tokens) + 1) + 1)
        self.col = {name: FEATURE_NAMES.index(name) for name in
                    ("llm_ratio_max", "key_lev", "best_gap", "q_gt_missing")}

    @lru_cache(maxsize=200_000)
    def token_stats(self, query: tuple[str, ...], cand: tuple[str, ...], which: str):
        if not query or not cand:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        lookup = self.idf[which]
        qw = np.array([lookup.get(t, self.unknown_idf) for t in query], dtype=float)
        cw = np.array([lookup.get(t, self.unknown_idf) for t in cand], dtype=float)
        qs, cs = set(query), set(cand)
        cover_q = sum(w for t, w in zip(query, qw) if t in cs) / qw.sum()
        cover_c = sum(w for t, w in zip(cand, cw) if t in qs) / cw.sum()
        sim = process.cdist(query, cand, scorer=Levenshtein.normalized_similarity,
                            workers=1, dtype=np.float32)
        fuzzy = float(np.dot(qw, sim.max(axis=1)) / qw.sum())
        # Weighted monotonic alignment cannot match two query tokens to the same
        # candidate token, unlike independent maximum token similarities.
        previous = np.zeros(len(cand) + 1)
        for i, weight in enumerate(qw):
            current = np.zeros(len(cand) + 1)
            for j in range(len(cand)):
                current[j + 1] = max(previous[j + 1], current[j], previous[j] + weight * sim[i, j])
            previous = current
        order = float(previous[-1] / qw.sum())
        extra = float(sum(w for t, w in zip(cand, cw) if t not in qs))
        return float(cover_q), float(cover_c), fuzzy, order, extra

    def build(self, row, cand, base, beams):
        n = len(cand)
        if not n:
            return np.empty((0, len(EXTRA_FEATURE_NAMES)), dtype=np.float32)
        qname = tuple(normalize_name(row.get("vessel_name")).split())
        qkey = tuple(query_key(row.get("vessel_name")).split())
        word = np.array([self.token_stats(qname, self.name_tokens[i], "name") for i in cand])
        key = np.array([self.token_stats(qkey, self.key_tokens[i], "key") for i in cand])
        bnames = [normalize_name(b) for b in beams]
        cnames = [self.r.names[i] for i in cand]
        beam_stats = np.zeros((n, 5))
        beam_values = np.zeros((n, 8))
        if bnames:
            ratio = process.cdist(bnames, cnames, scorer=fuzz.ratio, workers=1, dtype=np.float32) / 100
            first_tset = process.cdist(bnames[:1], cnames, scorer=fuzz.token_set_ratio,
                                       workers=1, dtype=np.float32)[0] / 100
            exact = np.array([[compact_name(b) == self.r.compact[i] for i in cand] for b in bnames])
            all_stats = np.array([[self.token_stats(tuple(b.split()), self.name_tokens[i], "name")
                                   for i in cand] for b in bnames])
            best = (all_stats[:, :, 0] + all_stats[:, :, 1] * .001).argmax(axis=0)
            beam_stats = all_stats[best, np.arange(n)]
            competitor = ratio[1:].max(axis=0) if len(bnames) > 1 else ratio[0]
            beam_values = np.column_stack([ratio[0], first_tset, exact[0], ratio.mean(axis=0),
                                            ratio.std(axis=0), (ratio >= .85).sum(axis=0),
                                            exact.sum(axis=0), ratio[0] - competitor])
        strength = np.maximum(base[:, self.col["llm_ratio_max"]], base[:, self.col["key_lev"]])
        similar = strength >= strength.max() - .10
        gap = base[:, self.col["best_gap"]]
        refs = gap[similar]
        tonnage_rank = np.searchsorted(np.sort(refs), gap, side="left").astype(float)
        ties = np.array([(refs == value).sum() for value in gap], dtype=float)
        relative = gap - refs.min()
        try:
            year = float(row.get("arrival_year"))
        except (TypeError, ValueError):
            year = np.nan
        try:
            gt, igt = float(row.get("gross_tonnage")), float(row.get("international_gross_tonnage"))
            disagree = abs(np.log(gt / igt)) if gt > 0 and igt > 0 else np.nan
        except (TypeError, ValueError):
            disagree = np.nan
        out = np.column_stack([
            word[:, 0], key, beam_stats, beam_values,
            tonnage_rank, ties, relative, np.full(n, similar.sum()), strength.max() - strength,
            np.full(n, year), np.full(n, disagree), np.full(n, len(qkey)),
            np.full(n, len(bnames[0].split()) if bnames else 0),
        ]).astype(np.float32)
        if out.shape != (n, len(EXTRA_FEATURE_NAMES)):
            raise ValueError(f"Extra feature schema mismatch: {out.shape}")
        return out
