"""Coarse metadata association, learned without names or identity features.

Binary data-versus-catalog classification estimates a density ratio. Subtracting
the candidate-only model removes the population prior already present in v12.
This is an association score, not a calibrated competition posterior.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

QUERY_FIELDS = ["q_flag", "q_type", "q_year", "q_size"]
CANDIDATE_FIELDS = ["c_flag", "c_type", "c_unit", "c_gt", "c_dwt", "c_size", "c_ratio"]
FIELDS = QUERY_FIELDS + CANDIDATE_FIELDS
MASS_EDGES = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 300000])
RATIO_EDGES = np.array([.25, .5, 1., 1.5, 2., 3., 5.])
MIN_SUPPORT = 10
PARAMS = dict(objective="binary", learning_rate=.05, num_leaves=15, max_depth=4,
              min_data_in_leaf=100, min_sum_hessian_in_leaf=10., lambda_l2=10.,
              max_cat_to_onehot=4, cat_smooth=20., cat_l2=10., max_cat_threshold=32,
              feature_fraction=1., bagging_fraction=1., verbosity=-1, num_threads=6,
              deterministic=True, force_col_wise=True, seed=20261029)
ROUNDS = 200
TEMPER = .5
CLIP = float(np.log(4.))


def category(value):
    return str(value).strip() if pd.notna(value) and str(value).strip() else "MISSING"


def bucket(value, edges=MASS_EDGES):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "MISSING"
    return str(int(np.searchsorted(edges, value, side="right"))) if np.isfinite(value) and value > 0 else "MISSING"


def query_profile(row, missing_size=False):
    year = row.get("arrival_year")
    try:
        year = str(int(np.floor((float(year)-2015)/3))) if np.isfinite(float(year)) else "MISSING"
    except (ValueError, TypeError):
        year = "MISSING"
    size = bucket(row.get("gross_tonnage"))
    if size == "MISSING": size = bucket(row.get("international_gross_tonnage"))
    return (category(row.get("flag_name")), category(row.get("vessel_type")), year,
            "MISSING" if missing_size else size)


def candidate_profile(row):
    try:
        ratio = float(row.get("deadweight_tonnage"))/float(row.get("gross_tonnage"))
    except (ValueError, TypeError, ZeroDivisionError):
        ratio = np.nan
    return (category(row.get("flag_name")), category(row.get("vessel_type")), category(row.get("unit")),
            bucket(row.get("gross_tonnage")), bucket(row.get("deadweight_tonnage")),
            bucket(row.get("size")), bucket(ratio, RATIO_EDGES))


class ContextCodec:
    def fit(self, train, catalog):
        lookup = catalog.set_index("vessel_id").to_dict("index")
        support = [defaultdict(set) for _ in FIELDS]
        for row in train.to_dict("records"):
            vid = row["target_vessel_id"]
            vals = query_profile(row) + candidate_profile(lookup[vid])
            for j, v in enumerate(vals): support[j][v].add(vid)
        self.maps = [{v: i+1 for i, v in enumerate(sorted(v for v, ids in s.items() if len(ids)>=MIN_SUPPORT))}
                     for s in support]
        self.support_counts = [{v: len(s[v]) for v in m} for s, m in zip(support, self.maps)]
        assert all(n>=MIN_SUPPORT for counts in self.support_counts for n in counts.values())
        # Only broad category/bucket dictionaries survive. No identity map.
        return self

    def encode(self, query_profiles, candidate_profiles):
        return np.asarray([[m.get(v, 0) for m, v in zip(self.maps, q+c)]
                           for q, c in zip(query_profiles, candidate_profiles)], dtype=np.float32)

    def features(self, row, candidates):
        return self.encode([query_profile(row)]*len(candidates), [candidate_profile(c) for c in candidates])


def fitting_examples(train, catalog, codec):
    """Each distinct fitting vessel has positive mass 1 and catalog noise mass 1."""
    lookup = dict(zip(catalog.vessel_id, range(len(catalog))))
    profiles = [candidate_profile(c) for c in catalog.to_dict("records")]
    contexts = defaultdict(set)
    for row in train.to_dict("records"):
        contexts[row["target_vessel_id"]].add(query_profile(row))
    rng = np.random.default_rng(20261029)
    qs, cs, labels, weights, vessel_groups = [], [], [], [], []
    collisions = 0
    # Randomly permute a vessel's whole context family for the matched control.
    vessels = sorted(contexts)
    # A true derangement avoids an accidental unchanged context family.
    order = rng.permutation(len(vessels))
    donors = np.empty(len(vessels), dtype=int)
    donors[order] = np.roll(order, 1)
    permuted_queries = []
    for vpos, vid in enumerate(vessels):
        original = sorted(contexts[vid])
        donor = sorted(contexts[vessels[int(donors[vpos])]])
        for pos, q in enumerate(original):
            for missing in (False, True):
                query = q[:3]+("MISSING" if missing else q[3],)
                dq = donor[pos%len(donor)]
                control_query = dq[:3]+("MISSING" if missing else dq[3],)
                noise = rng.integers(len(catalog), size=16)
                choices = np.r_[lookup[vid], noise]
                collisions += int(np.sum(noise==lookup[vid]))
                for j, choice in enumerate(choices):
                    qs.append(query); permuted_queries.append(control_query); cs.append(profiles[int(choice)])
                    labels.append(int(j==0)); vessel_groups.append(vpos)
                    weights.append((1. if j==0 else 1/16)/(2*len(original)))
    x = codec.encode(qs, cs)
    shuffled = codec.encode(permuted_queries, cs)
    y, w, groups = np.asarray(labels), np.asarray(weights), np.asarray(vessel_groups)
    for value in (0, 1):
        np.testing.assert_allclose(np.bincount(groups[y==value], weights=w[y==value]), 1., atol=1e-12)
    return x, shuffled, y, w, dict(vessels=len(vessels), pairs=len(y),
            contexts=sum(map(len, contexts.values())), positive_weight=float(w[y==1].sum()),
            negative_weight=float(w[y==0].sum()), noise_truth_collisions=collisions,
            permutation_fixed_vessel_donors=int(np.sum(donors==np.arange(len(vessels)))))


def combine(anchor, conditional_logits, marginal_logits):
    adjustment = TEMPER*np.clip(np.asarray(conditional_logits)-np.asarray(marginal_logits), -CLIP, CLIP)
    logits = np.log(np.maximum(anchor, np.finfo(float).tiny))+adjustment
    values = np.exp(logits-logits.max())
    return values/np.sum(np.sort(values))
