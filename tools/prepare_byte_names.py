"""Training-only exact fragment supervision from official pairs and frozen beams."""
import hashlib
import json
import re
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.augment import augment
from src.byte_name_model import fragments
from src.common import ROOT, compact_name, is_hangul, load_catalog, load_train, normalize_name
from src.llm_cache import BeamCache
from src.validate import load_folds

TAG = "byte_names_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG
PROTOCOL = ROOT / "experiments" / f"{TAG}_protocol.json"


def annotate(query, canonical, beams):
    keys = {compact_name(v) for v in beams}
    keys.discard("")
    choices = [s for s, _ in fragments(canonical) if compact_name(s) in keys]
    if len(choices) != 1:
        return None
    target = choices[0]
    if re.findall(r"\d+", normalize_name(query)) != re.findall(r"\d+", target):
        return None
    return normalize_name(query), target


def prepare(fold):
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / f"fold{fold}_data.joblib"
    if path.exists(): return joblib.load(path)
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    han = train.vessel_name.map(is_hangul).to_numpy()
    eligible = han & (folds != fold)
    ids = train.loc[eligible, "target_vessel_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261029)
    fit_pos, val_pos = next(splitter.split(ids, groups=ids))
    fit_ids, val_ids = set(ids[fit_pos]), set(ids[val_pos])
    outside = set(train.loc[folds == fold, "target_vessel_id"])
    assert not fit_ids & val_ids and not outside & (fit_ids | val_ids)
    aug = augment(train, 4, 1, seed=7)
    original = train.loc[eligible].copy(); original["origin"] = original.index; original["source"] = "original"
    augmented = aug.loc[eligible[aug.origin.to_numpy()]].copy(); augmented["source"] = "augmentation"
    frame = pd.concat([original, augmented], ignore_index=True)
    canonical = catalog.set_index("vessel_id").vessel_name
    beams = BeamCache(allow_generate=False)
    accepted = {"fit": set(), "valid": set()}
    examples, stats, accepted_ids = [], Counter(), set()
    for row in frame.to_dict("records"):
        if not is_hangul(row["vessel_name"]): continue
        stats["eligible_"+row["source"]] += 1
        name = canonical[row["target_vessel_id"]]
        pair = annotate(row["vessel_name"], name, beams([row["vessel_name"]])[0])
        if pair is None: continue
        stats["accepted_"+row["source"]] += 1
        accepted_ids.add(row["target_vessel_id"])
        split = "fit" if row["target_vessel_id"] in fit_ids else "valid"
        accepted[split].add(pair)
        if len(examples) < 30:
            examples.append({"query": row["vessel_name"], "canonical": name, "target_fragment": pair[1], "source": row["source"]})
    all_pairs = accepted["fit"] | accepted["valid"]
    overlap = accepted["fit"] & accepted["valid"]
    accepted["fit"] -= overlap
    assert not accepted["fit"] & accepted["valid"]

    def weighted(pairs):
        counts = Counter(q for q, c in pairs)
        return [{"query": q, "target": c, "weight": 1/counts[q]} for q, c in sorted(pairs)]

    report = {"fold": fold, "counts": dict(stats), "accepted_training_vessels": len(accepted_ids),
              "full_unique_pairs": len(all_pairs), "fit_unique_pairs": len(accepted["fit"]),
              "valid_unique_pairs": len(accepted["valid"]), "purged_fit_exact_pairs": len(overlap),
              "fit_vessels": sorted(fit_ids), "valid_vessels": sorted(val_ids), "outer_vessels": sorted(outside),
              "vessel_overlap": 0, "exact_pair_overlap_after_purge": 0, "synthetic_korean_labels": 0,
              "examples": examples}
    result = {"fit": weighted(accepted["fit"]), "valid": weighted(accepted["valid"]),
              "full": weighted(all_pairs), "report": report}
    result["data_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    joblib.dump(result, path, compress=3)
    (FOLDER / f"fold{fold}_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("fit_vessels", "valid_vessels", "outer_vessels", "examples")}), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--folds", type=int, nargs="+", default=[1, 4])
    for fold in p.parse_args().folds: prepare(fold)
