"""Private, prospectively frozen held-out noise diagnostic for the loss trial."""
import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import drop_word, truncate
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.ensemble import normalize
from src.features import FeatureBuilder, fit_mappers
from src.llm_cache import BeamCache
from src.llm_translit import FEW_SHOT
from src.retrieval import CatalogRetriever
from src.validate import load_folds
from tools.ambiguity_encoder_experiment import FOLDER, TAG, check_lock, dump, fingerprint


def make_rows(fold):
    train = load_train(); folds = load_folds(train)
    shots = set(train.loc[train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"])
    frame = train.loc[(folds == fold) & train.vessel_name.map(is_hangul) & ~train.target_vessel_id.isin(shots)]
    rows = []
    for original in frame.to_dict("records"):
        stable = int(hashlib.sha256(original["sample_id"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(2026102901 + stable)
        dropped = drop_word(original["vessel_name"], rng)
        short = truncate(original["vessel_name"])
        for scenario in ("gt_only_missing", "igt_only_missing", "gt15pct_conflict", "word_drop_gt", "truncated_gt", "word_drop_missing", "truncated_missing"):
            row = dict(original)
            if scenario == "gt_only_missing": row["gross_tonnage"] = np.nan
            elif scenario == "igt_only_missing": row["international_gross_tonnage"] = np.nan
            elif scenario == "gt15pct_conflict": row["gross_tonnage"] *= float(rng.choice([.85, 1.15]))
            else:
                row["vessel_name"] = dropped if scenario.startswith("word") else short
                if scenario.endswith("missing"):
                    row["gross_tonnage"] = row["international_gross_tonnage"] = np.nan
            row["sample_id"] += "_ambiguity_guard_" + scenario
            row["scenario"] = scenario
            row["name_changed"] = row["vessel_name"] != original["vessel_name"]
            rows.append(row)
    return train, folds, pd.DataFrame(rows)


def prepare(fold):
    check_lock()
    path = FOLDER / f"fresh_fold{fold}.joblib"
    if path.exists():
        result = joblib.load(path)
        assert result["experiment_fingerprint"] == fingerprint()
        return result
    train, folds, frame = make_rows(fold)
    private = FOLDER / "private_beams.json"
    beams = BeamCache(path=private)
    frozen = BeamCache(allow_generate=False)
    for name in frame.vessel_name.unique():
        if name not in beams.data and name in frozen.data: beams.data[name] = frozen.data[name]
    for name in frame.vessel_name.unique():
        if is_hangul(name): beams([name])
    beams.save()
    bb = beams(frame.vessel_name.tolist())
    beams._model = None
    import torch
    gc.collect(); torch.cuda.empty_cache()
    catalog = load_catalog()
    retriever = CatalogRetriever(catalog)
    builder = FeatureBuilder(retriever, *fit_mappers(train.loc[folds != fold], catalog))
    retriever.category_score = builder.category_score
    model = lgb.Booster(model_file=str(ROOT / f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt"))

    def chunk(offset):
        part = frame.iloc[offset:offset+48]
        signals = retriever.generate(part, llm_beams=bb[offset:offset+48])
        blocks = [builder.build(r, s) for r, s in zip(part.to_dict("records"), signals)]
        ends = np.cumsum([0]+[len(b) for b in blocks])
        scores = model.predict(np.vstack(blocks), num_threads=1)
        return [{"candidates": s["cand"], "base": normalize(scores[ends[j]:ends[j+1]], "prob")} for j, s in enumerate(signals)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        pairs = [p for part in pool.map(chunk, range(0, len(frame), 48)) for p in part]
    overlap = bool(set(train.loc[folds != fold, "target_vessel_id"]) & set(frame.target_vessel_id))
    assert not overlap
    result = {"records": frame.to_dict("records"), "beams": bb, "pairs": pairs,
              "truth": frame.target_vessel_id.map(retriever.id_to_index).to_numpy(), "fold": fold,
              "source": hashlib.sha256((fingerprint()+frame.to_csv(index=False)+json.dumps(bb, ensure_ascii=False)).encode()).hexdigest(),
              "few_shot_vessels_excluded": True, "training_valid_vessels_overlap": 0, "experiment_fingerprint": fingerprint()}
    dump(result, path)
    return result


def main():
    from tools import fresh_encoder_extension as guard
    guard.prepare = prepare
    report = guard.run(f"{TAG}_targeted", check_lock()["fresh_guard"]["fold"])
    report["noise_seed"] = 2026102901
    report["previously_used_vessels_not_pristine"] = True
    report["private_data"] = str(FOLDER / "fresh_fold3.joblib")
    (ROOT / "experiments" / f"{TAG}_fresh.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
