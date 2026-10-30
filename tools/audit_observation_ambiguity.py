"""Descriptive ambiguity audit; no new model, predictions, or performance claim."""
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.common import ROOT, compact_name, is_hangul, load_catalog, normalize_name
from src.llm_cache import BeamCache
from tools.blend_matching import lightweight_lab


@lru_cache(maxsize=None)
def spans(name):
    words = normalize_name(name).split()
    return frozenset(compact_name(" ".join(words[i:j]))
                     for i in range(len(words)) for j in range(i + 1, len(words) + 1))


def positive(value):
    try:
        value = float(value)
        return bool(np.isfinite(value) and value > 0)
    except (TypeError, ValueError):
        return False


def same_metadata(row, left, right):
    # This deliberately tests equality of the two candidate explanations, not
    # whether either matches the record's possibly inaccurate metadata.
    for key in ("flag_name", "vessel_type"):
        if pd.notna(row.get(key)):
            if pd.isna(left.get(key)) or pd.isna(right.get(key)) or left[key] != right[key]:
                return False
    if positive(row.get("gross_tonnage")) or positive(row.get("international_gross_tonnage")):
        if not positive(left.get("gross_tonnage")) or not positive(right.get("gross_tonnage")):
            return False
        if float(left["gross_tonnage"]) != float(right["gross_tonnage"]):
            return False
    return True


def main():
    archived = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archived["identity"])
    catalog = load_catalog().to_dict("records")
    beams = BeamCache(allow_generate=False)
    by_name = {}

    def beam_keys(name):
        if name not in by_name:
            values = beams([name])[0] if is_hangul(name) else [name]
            by_name[name] = frozenset(compact_name(v) for v in values if compact_name(v))
        return by_name[name]

    def shared(row, a, b):
        return beam_keys(row["vessel_name"]) & spans(a["vessel_name"]) & spans(b["vessel_name"])

    evaluation = defaultdict(Counter)
    cases = []
    for source, rows, truth in (
        ("oof", lab.ctx.rows, lab.ctx.truth),
        ("testlike", lab.sets["testlike"].df.to_dict("records"), lab.sets["testlike"].truth),
    ):
        predictions = archived["ensemble"][source]
        for i, (row, target, chosen) in enumerate(zip(rows, truth, predictions)):
            if not is_hangul(row["vessel_name"]):
                continue
            scenario = "clean" if source == "oof" else row["scenario"]
            count = evaluation[scenario]
            count["rows"] += 1
            if int(target) == int(chosen):
                continue
            count["v12_errors"] += 1
            left, right = catalog[int(target)], catalog[int(chosen)]
            common = shared(row, left, right)
            metadata_tie = same_metadata(row, left, right)
            count["identical_available_candidate_metadata"] += int(metadata_tie)
            count["same_exact_latin_span_from_cached_beams"] += int(bool(common))
            count["both_conditions"] += int(bool(common) and metadata_tie)
            if common and metadata_tie and len(cases) < 32:
                cases.append({"scenario": scenario, "sample_id": row["sample_id"],
                              "query": row["vessel_name"], "shared_beam_spans": sorted(common),
                              "truth": left["vessel_name"], "chosen": right["vessel_name"],
                              "tonnage_available": positive(row.get("gross_tonnage")) or positive(row.get("international_gross_tonnage")),
                              "truth_gt": left["gross_tonnage"], "chosen_gt": right["gross_tonnage"]})
    training = []
    for outer in (0, 2):
        for inner in (0, 1):
            path = ROOT / f"experiments/artifacts/nested_combiner_v1/outer{outer}_inner{inner}/data.joblib"
            data = joblib.load(path, mmap_mode="r")
            counts = Counter()
            for group in data["training"]:
                row = group["row"]
                counts["groups"] += 1
                if not is_hangul(row["vessel_name"]):
                    continue
                counts["actual_hangul_groups"] += 1
                candidates = group["candidates"]
                # The frozen training preparation stores the one positive first.
                left = catalog[int(candidates[0])]
                assert left["vessel_id"] == row["target_vessel_id"]
                overlaps = metadata_ties = both = 0
                for negative in candidates[1:]:
                    right = catalog[int(negative)]
                    overlap = bool(shared(row, left, right))
                    tie = same_metadata(row, left, right)
                    overlaps += overlap
                    metadata_ties += tie
                    both += overlap and tie
                counts["groups_with_shared_exact_span"] += int(overlaps > 0)
                counts["groups_with_same_metadata_negative"] += int(metadata_ties > 0)
                counts["groups_with_both_conditions_negative"] += int(both > 0)
                counts["negative_pairs_with_both_conditions"] += both
                counts["no_tonnage_groups_with_both"] += int(both > 0 and not positive(row.get("gross_tonnage")) and not positive(row.get("international_gross_tonnage")))
            training.append({"outer": outer, "inner": inner, **dict(counts)})
    totals = Counter()
    for record in training:
        totals.update({k: v for k, v in record.items() if k not in ("outer", "inner")})
    partitions = {}
    for scenario, counts in evaluation.items():
        both = counts["both_conditions"]
        partitions[scenario] = {
            "shared_span_and_metadata_tie": both,
            "shared_span_only": counts["same_exact_latin_span_from_cached_beams"] - both,
            "metadata_tie_only": counts["identical_available_candidate_metadata"] - both,
            "neither": counts["v12_errors"] - counts["same_exact_latin_span_from_cached_beams"]
                       - counts["identical_available_candidate_metadata"] + both}
        assert all(n >= 0 for n in partitions[scenario].values())
        assert sum(partitions[scenario].values()) == counts["v12_errors"]
    assert totals["groups"] == 9254 and totals["groups_with_both_conditions_negative"] == 384
    result = {
        "diagnostic_only": True, "uses_ground_truth_to_describe_errors": True,
        "new_model_or_score_evaluated": False,
        "definition": "Truth and selected/negative candidates share an exact contiguous Latin span present in a cached query-only transliteration beam, and have identical catalog metadata on fields available in the query.",
        "evaluation": dict(evaluation), "error_partitions": partitions,
        "training": training, "training_total": dict(totals), "examples": cases,
        "checks": {"error_partitions_reconcile": True, "training_positive_ids_verified": True,
                   "actual_hangul_scope": True, "no_new_model_predictions": True},
        "limits": ["This is a visible-evidence tie under one deliberately conservative diagnostic, not an information-theoretic ceiling.",
                   "Cached Latin beams can be incorrect or ambiguous; nonmatching spans do not imply disambiguation is possible.",
                   "Arrival year and learned population/name priors may still contain information; catalog has no arrival-year field.",
                   "Different vessel IDs remain true negatives for identity; this audit does not relabel them as the same entity.",
                   "Nested inner training data is not the exact original v12 training population; it diagnoses the repeated training recipe.",
                   "No test rows, call_sign lookup, or same-vessel identity lookup are used."],
        "source_sha256": hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(),
    }
    path = ROOT / "experiments/observation_ambiguity_20261029.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evaluation": result["evaluation"], "training": training}, indent=2), flush=True)


if __name__ == "__main__":
    main()
