"""Read-only cache/identity and pair-budget inspection for reverse-name scoring.

Only writes experiments/reverse_data_inspection.json. Does not load a language
model, train a learner, or inspect any new-model predictions.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import augment
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.experiment import build_hard, build_testlike
from src.features import CATEGORY_COLUMNS, FEATURE_NAMES
from src.llm_cache import CACHE_PATH
from src.retrieval import RETRIEVAL_VERSION
from src.validate import load_folds


def main() -> None:
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    hangul = train.vessel_name.map(is_hangul).to_numpy()
    aug = augment(train, 4, 1, seed=7)
    ctx = SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=hangul)
    frames = {"oof": train, "hard": build_hard(ctx), "testlike": build_testlike(ctx)}
    positions = {s: {v: i for i, v in enumerate(df.sample_id)} for s, df in frames.items()}
    catalog_names = catalog.vessel_name.fillna("").astype(str).tolist()
    refs = {k: joblib.load(ROOT / f"experiments/artifacts/encoder_bge_phonetic/fold{k}_epoch1_scores.joblib")
            for k in (0, 2)}
    report = {
        "purpose": "Cache identities and inference budgets only; no new model performance inspected",
        "features": {"names_count": len(FEATURE_NAMES), "category_columns": list(CATEGORY_COLUMNS)},
        "folds": {}, "historical_validation": {},
    }
    train_pair_sets = {cap: set() for cap in (240, 600, 2400, -1)}
    valid_pair_set, pilot_pair_set = set(), set()
    for fold in (0, 2):
        trdf = pd.concat([train.loc[hangul & (folds != fold)],
                         aug.loc[hangul[aug.origin.to_numpy()] & (folds[aug.origin.to_numpy()] != fold)]],
                        ignore_index=True)
        path = max((ROOT / "experiments/cache").glob(f"strict_han_f{fold}_train_*.joblib"),
                   key=lambda p: p.stat().st_mtime)
        matrix = joblib.load(path, mmap_mode="r")
        assert matrix["sample_ids"] == trdf.sample_id.tolist()
        assert not set(trdf.target_vessel_id) & set(train.loc[folds == fold, "target_vessel_id"])
        miner_path = ROOT / f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt"
        miner = lgb.Booster(model_file=str(miner_path))
        names = trdf.vessel_name.fillna("").astype(str).tolist()
        eligible = [int(i) for i in np.random.default_rng(42).permutation(len(trdf))
                    if matrix["truth"][i] in matrix["candidates"][i] and len(matrix["candidates"][i]) >= 8]
        stat = {
            "train_cache": str(path.relative_to(ROOT)), "file_bytes": path.stat().st_size,
            "groups": len(trdf), "eligible_positive_present_groups": len(eligible),
            "positive_absent_groups": int(sum(t not in c for t, c in zip(matrix["truth"], matrix["candidates"]))),
            "actual_nonhangul_train_groups": int(sum(not is_hangul(v) for v in names)),
            "all_candidate_pairs": int(sum(len(c) for c in matrix["candidates"])),
            "block_shape_example": list(matrix["blocks"][0].shape),
            "block_dtype": str(matrix["blocks"][0].dtype),
            "mmap_block": isinstance(matrix["blocks"][0], np.memmap),
            "sample_identity_verified": True, "heldout_vessel_overlap": 0,
            "caps": {}, "validation": {},
        }
        selected_pairs = []
        for i in eligible:
            c, block, truth = matrix["candidates"][i], matrix["blocks"][i], matrix["truth"][i]
            positive = int(np.flatnonzero(c == truth)[0])
            score = miner.predict(block[:, :len(FEATURE_NAMES)], num_threads=4)
            negatives = [int(j) for j in np.argsort(-score, kind="stable") if j != positive][:19]
            selected = [positive] + negatives
            selected_pairs.append([(names[i], catalog_names[int(j)]) for j in c[selected]])
        for cap in train_pair_sets:
            use = selected_pairs if cap < 0 else selected_pairs[:cap]
            exact = {pair for group in use for pair in group}
            stat["caps"][str(cap)] = {
                "groups": len(use), "pairs": sum(map(len, use)), "unique_name_pairs": len(exact),
                "actual_hangul_unique_pairs": sum(is_hangul(q) for q, _ in exact),
            }
            train_pair_sets[cap].update(exact)
        for s, entries in refs[fold]["sets"].items():
            ids = refs[fold]["sample_ids"][s]
            assert len(ids) == len(entries)
            pair_set = set()
            total = actual_han = pruned_han = 0
            selected_pilot = 0
            for sample_id, old in zip(ids, entries):
                row = frames[s].iloc[positions[s][sample_id]]
                if not is_hangul(row.vessel_name):
                    continue
                actual_han += 1
                pruned_han += bool(old.get("bound_pruned", False))
                choices = old["candidates"][old["keep"]]
                pairs = {(str(row.vessel_name), catalog_names[int(j)]) for j in choices}
                pair_set.update(pairs)
                total += len(choices)
                if selected_pilot < 32:
                    pilot_pair_set.update(pairs)
                    selected_pilot += 1
            valid_pair_set.update(pair_set)
            stat["validation"][s] = {
                "all_groups": len(entries), "actual_hangul_groups": actual_han,
                "previously_pruned_hangul_groups": pruned_han,
                "pairs": total, "unique_name_pairs": len(pair_set),
            }
        report["folds"][str(fold)] = stat
        del matrix, miner, selected_pairs
        gc.collect()
        print(f"fold {fold} train/evaluation pair inspection complete", flush=True)
    tag = hashlib.md5((",".join(FEATURE_NAMES) + CACHE_PATH.name + RETRIEVAL_VERSION).encode()).hexdigest()[:8]
    prefixes = {"oof": "train", "hard": "hard_s2026", "testlike": "testlike_s3031"}
    for s, prefix in prefixes.items():
        path = ROOT / f"experiments/cache/{prefix}_features_{tag}.joblib"
        cache = joblib.load(path, mmap_mode="r")
        stat = {
            "path": str(path.relative_to(ROOT)), "file_bytes": path.stat().st_size,
            "groups": len(cache["cands"]), "frame_groups": len(frames[s]),
            "block_shape_example": list(cache["blocks"][0].shape),
            "block_dtype": str(cache["blocks"][0].dtype),
            "mmap_block": isinstance(cache["blocks"][0], np.memmap),
            "folds": {},
        }
        assert len(cache["cands"]) == len(frames[s])
        for fold, ref in refs.items():
            ids = ref["sample_ids"][s]
            identical = [np.array_equal(cache["cands"][positions[s][sid]], old["candidates"])
                         for sid, old in zip(ids, ref["sets"][s])]
            stat["folds"][str(fold)] = {"compared": len(ids), "exact_candidate_matches": sum(identical)}
            assert all(identical), (path, fold, sum(identical), len(ids))
        report["historical_validation"][s] = stat
        del cache
        gc.collect()
        print(f"{s} historical feature candidate identity verified", flush=True)
    report["pair_budgets_across_folds_0_2"] = {
        "full_evaluation_unique_pairs": len(valid_pair_set),
        "pilot_first32_actual_hangul_rows_per_set_per_fold_unique_pairs": len(pilot_pair_set),
        "train_caps": {
            str(cap): {"unique_train_pairs": len(pairs),
                       "unique_train_and_full_evaluation_pairs": len(pairs | valid_pair_set),
                       "unique_train_and_pilot_evaluation_pairs": len(pairs | pilot_pair_set)}
            for cap, pairs in train_pair_sets.items()
        },
    }
    report["recipe"] = {
        "training_groups": "First 2400 eligible RNG(42) permutation groups from each strict train cache; same eligibility/order as original BGE",
        "training_candidates": "Positive plus first19 stable strict-base-miner negatives, feature block first78 columns",
        "validation_candidates": "Exact archived v12 candidates[keep], every actual Hangul row including previously pruned",
        "validation_features": "Read one historical78 cache at a time with mmap_mode=r; map sample_id to original reconstructed frame; copy blocks[row][keep]; refresh CATEGORY_COLUMNS with fit_mappers(train[fold != heldout], catalog)",
        "selection": "Predeclare direct ranker plus fixed 50:50 v12/new score blend; paired78-only ablation on identical groups; screen folds0,2 once, fresh qualification fold3 before full5fold acceptance",
        "leakage_caveat": "Archived v12 retrieval used historical fulltrain category mapper; new feature category columns must be outer-train only; no claim of strict heldout retrieval",
        "score_inputs": "Raw78 and frozen pretrained name likelihood only; no fitted-neural or fitted-base scores as learned inputs",
    }
    out = ROOT / "experiments/reverse_data_inspection.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["pair_budgets_across_folds_0_2"], indent=2), flush=True)


if __name__ == "__main__":
    main()
