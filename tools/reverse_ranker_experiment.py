"""Frozen reverse-name likelihood plus a direct LambdaRank decision model.

New modules only: the submitted v12 source and artifacts remain frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import augment
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.ensemble import normalize
from src.experiment import build_hard, build_testlike
from src.features import CATEGORY_COLUMNS, FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.train_ranker import PARAMS
from src.validate import load_folds

PROTOCOL = ROOT / "experiments/reverse_ranker_protocol.json"
FOLDER = ROOT / "experiments/artifacts/reverse_kanana_v1"
EXTRA = ["reverse_sum_logp", "reverse_mean_logp", "reverse_eos_logp",
         "reverse_gap_to_best", "reverse_dense_rank", "reverse_target_tokens"]


def frames_and_training(fold):
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    h = train.vessel_name.map(is_hangul).to_numpy()
    ctx = SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=h)
    frames = {"oof": train, "hard": build_hard(ctx), "testlike": build_testlike(ctx)}
    aug = augment(train, 4, 1, seed=7)
    origins = aug.origin.to_numpy()
    trdf = pd.concat([train.loc[h & (folds != fold)],
                      aug.loc[h[origins] & (folds[origins] != fold)]], ignore_index=True)
    assert not set(trdf.target_vessel_id) & set(train.loc[folds == fold, "target_vessel_id"])
    return train, catalog, folds, frames, trdf


def prepare(fold):
    FOLDER.mkdir(parents=True, exist_ok=True)
    path = FOLDER / f"fold{fold}_data.joblib"
    protocol_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if path.exists():
        data = joblib.load(path)
        assert data["protocol_sha256"] == protocol_sha
        return data
    train, catalog, folds, frames, trdf = frames_and_training(fold)
    catnames = catalog.vessel_name.fillna("").astype(str).to_numpy()
    matrix_path = max((ROOT / "experiments/cache").glob(f"strict_han_f{fold}_train_*.joblib"),
                      key=lambda p: p.stat().st_mtime)
    matrix = joblib.load(matrix_path, mmap_mode="r")
    assert matrix["sample_ids"] == trdf.sample_id.tolist()
    miner_path = ROOT / f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt"
    miner = lgb.Booster(model_file=str(miner_path))
    groups = []
    for i in np.random.default_rng(42).permutation(len(trdf)):
        c, b, truth = matrix["candidates"][i], matrix["blocks"][i], matrix["truth"][i]
        positive = np.flatnonzero(c == truth)
        if not len(positive) or len(c) < 8:
            continue
        scores = miner.predict(b[:, :78], num_threads=4)
        neg = [int(j) for j in np.argsort(-scores, kind="stable") if j != positive[0]][:19]
        selected = np.array([int(positive[0])]+neg)
        selected = selected[np.argsort(c[selected], kind="stable")]
        row = trdf.iloc[i]
        groups.append({"sample_id": row.sample_id, "name": str(row.vessel_name),
                       "candidates": c[selected].copy(), "names": catnames[c[selected]].tolist(),
                       "raw": np.asarray(b[selected, :78], dtype=np.float32), "truth": int(truth)})
        if len(groups) == 2400:
            break
    del matrix, miner
    gc.collect()
    builder = FeatureBuilder.__new__(FeatureBuilder)
    builder.type_mapper, builder.flag_mapper = fit_mappers(train.loc[folds != fold], catalog)
    builder.cat_type = catalog.vessel_type.astype(str).to_numpy()
    builder.cat_flag = catalog.flag_name.astype(str).to_numpy()
    ref = joblib.load(ROOT / f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
    inspection = json.loads((ROOT / "experiments/reverse_data_inspection.json").read_text())
    evaluation = {}
    for s, frame in frames.items():
        cache_path = ROOT / inspection["historical_validation"][s]["path"]
        cache = joblib.load(cache_path, mmap_mode="r")
        positions = {sid: i for i, sid in enumerate(frame.sample_id)}
        evaluation[s] = []
        for sid, old in zip(ref["sample_ids"][s], ref["sets"][s]):
            i = positions[sid]
            row = frame.iloc[i].to_dict()
            np.testing.assert_array_equal(cache["cands"][i], old["candidates"])
            c = old["candidates"][old["keep"]]
            raw = cache["blocks"][i][old["keep"], :78].copy()
            for col, value in zip(CATEGORY_COLUMNS, builder.category(row, c)):
                raw[:, col] = value
            assert np.sum(c == old["truth"]) <= 1
            evaluation[s].append({"sample_id": sid, "name": str(row["vessel_name"]),
                                  "candidates": c, "names": catnames[c].tolist(), "raw": raw,
                                  "truth": int(old["truth"]), "base": old["base_scores"][old["keep"]],
                                  "actual_hangul": bool(is_hangul(row["vessel_name"]))})
        del cache
        gc.collect()
    data = {"fold": fold, "train": groups, "evaluation": evaluation,
            "protocol_sha256": protocol_sha, "train_cache": matrix_path.name,
            "miner_sha256": hashlib.sha256(miner_path.read_bytes()).hexdigest()}
    joblib.dump(data, path, compress=3)
    print(f"Prepared fold{fold}: {len(groups)} groups, evaluation {sum(map(len,evaluation.values()))}", flush=True)
    return data


def all_pairs(data):
    return {(g["name"], n) for d in data for g in d["train"]+
            [r for rows in d["evaluation"].values() for r in rows if r["actual_hangul"]]
            for n in g["names"]}


def score_pairs(data):
    import torch
    from src.reverse_name_likelihood import ReverseNameScorer
    torch.set_num_threads(4)
    scorer = ReverseNameScorer()
    protocol = scorer.protocol()
    path = FOLDER / "pair_scores.joblib"
    cache = joblib.load(path) if path.exists() else {"protocol": protocol, "scores": {}}
    assert cache["protocol"] == protocol, "Scorer protocol changed; use a new experiment/cache"
    missing = sorted(all_pairs(data)-cache["scores"].keys())
    print(f"Reverse scoring: {len(missing)} missing, {len(cache['scores'])} cached", flush=True)
    # Preflight every target/prefix length before computing any validation scores.
    qs = {q: len(scorer.target_ids(q)) for q, _ in missing}
    cs = {c: len(scorer.prefix_ids(c)) for _, c in missing}
    assert all(qs[q]+cs[c] <= scorer.pad_length for q, c in missing), "No truncation allowed"
    start = time.perf_counter()
    for offset in range(0, len(missing), 256):
        chunk = missing[offset:offset+256]
        values = scorer.score_many_pairs(chunk)
        cache["scores"].update(zip(chunk, values))
        if offset % 2048 == 0 or offset+256 >= len(missing):
            joblib.dump(cache, path, compress=3)
            print(f"Reverse pairs {min(offset+256,len(missing))}/{len(missing)}, seconds {time.perf_counter()-start:.1f}", flush=True)
    cache["last_run_seconds"] = time.perf_counter()-start
    cache["unique_pairs"] = len(cache["scores"])
    joblib.dump(cache, path, compress=3)
    return cache


def features(group, scores):
    v = np.array([scores[(group["name"], n)] for n in group["names"]])
    mean = v[:, 1]
    # Dense ranks give identical values for tied likelihoods regardless of label/order.
    rank = np.searchsorted(np.unique(-mean), -mean)+1
    extra = np.column_stack((v[:, 0], mean, v[:, 2], mean-mean.max(), rank, v[:, 3]))
    return np.column_stack((group["raw"], extra)).astype(np.float32)


def train_and_evaluate(data, cache):
    for d in data:
        fold = d["fold"]
        train = d["train"]
        x = np.vstack([features(g, cache["scores"]) for g in train])
        y = np.concatenate([g["candidates"] == g["truth"] for g in train]).astype(int)
        sizes = [len(g["candidates"]) for g in train]
        assert len(train) == 2400 and y.sum() == 2400
        stored = {"fold": fold, "protocol_sha256": d["protocol_sha256"], "sets": {}, "controls": {}}
        for recipe, width in (("reverse", 84), ("raw78", 78)):
            ds = lgb.Dataset(x[:, :width], label=y, group=sizes,
                             feature_name=(FEATURE_NAMES+EXTRA)[:width], free_raw_data=True)
            model = lgb.train(PARAMS, ds, num_boost_round=150)
            model.save_model(str(FOLDER / f"fold{fold}_{recipe}.txt"))
            results = {}
            for s, groups in d["evaluation"].items():
                results[s] = []
                for g in groups:
                    if not g["actual_hangul"]:
                        pred = {"direct": None, "base25_head75": None}
                        scores = None
                    else:
                        xx = features(g, cache["scores"])[:, :width]
                        scores = model.predict(xx, num_threads=4)
                        bp = g["base"]/g["base"].sum()
                        pred = {"direct": int(g["candidates"][np.argmax(scores)]),
                                "base25_head75": int(g["candidates"][np.argmax(.25*bp+.75*normalize(scores,"prob"))])}
                    results[s].append({"sample_id": g["sample_id"], "truth": g["truth"],
                                       "predictions": pred, "scores": scores,
                                       "truth_in_top20": bool(g["truth"] in g["candidates"])})
            stored["sets" if recipe == "reverse" else "controls"] = results
        joblib.dump(stored, FOLDER / f"fold{fold}_results.joblib", compress=3)
        print(f"Saved fold{fold} head/control predictions", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 2])
    p.add_argument("--stage", choices=["prepare", "score", "train", "all"], default="all")
    args = p.parse_args()
    data = [prepare(k) for k in args.folds]
    if args.stage == "prepare":
        print(f"Unique pairs: {len(all_pairs(data))}", flush=True)
        return
    cache = score_pairs(data) if args.stage in ("score", "all") else joblib.load(FOLDER / "pair_scores.joblib")
    if args.stage in ("train", "all"):
        train_and_evaluate(data, cache)


if __name__ == "__main__":
    main()
