"""Check pair-cache invariance on real official-data strings, without labels."""
import argparse
import json

import joblib
import numpy as np
import torch

from src.common import ROOT
from src.reverse_name_likelihood import ReverseNameScorer
from tools.reverse_ranker_experiment import FOLDER


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    folder = ROOT / "experiments/artifacts/reverse_kanana_partial_v1" if args.partial else FOLDER
    cache = joblib.load(folder / "pair_scores.joblib")
    keys = sorted(cache["scores"])
    rng = np.random.default_rng(20261028)
    chosen = [keys[int(i)] for i in rng.choice(len(keys), size=96, replace=False)]
    torch.set_num_threads(4)
    scorer = ReverseNameScorer()
    assert scorer.protocol() == cache["protocol"]
    expected = np.array([cache["scores"][key] for key in chosen])
    actual = scorer.score_many_pairs(chosen)
    np.testing.assert_array_equal(actual, expected)
    reversed_values = scorer.score_many_pairs(chosen[::-1])[::-1]
    np.testing.assert_array_equal(reversed_values, expected)
    for pair, value in zip(chosen[:3], expected[:3]):
        np.testing.assert_array_equal(scorer.score_many_pairs([pair])[0], value)
    # Whole-query API uses canonical candidate strings and deduplication.
    query = chosen[0][0]
    names = [c for q, c in keys if q == query][:20]
    names += names[:2]
    result = scorer.score_details(query, names)
    np.testing.assert_array_equal(result["mean_logp"], np.array([cache["scores"][(query,c)][1] for c in names]))
    report = {"status": "PASS", "partial_cache": args.partial, "random_official_pairs": len(chosen),
              "exact_cached_vs_fresh": True, "exact_reverse": True, "exact_singles": 3,
              "exact_query_api_with_duplicates": True, "query_api_candidates": len(names),
              "protocol": scorer.protocol(),
              "scope": "Scorer/pair cache on current GPU and frozen settings; not a submission/package execution"}
    (ROOT / "experiments" / f"{folder.name}_real_pair_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
