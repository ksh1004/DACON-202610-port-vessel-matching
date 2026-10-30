"""Independent fold retrieval and Korean-only pair-ranker experiments.

Each fold's category mappers, candidate generation and rank features use only
that fold's training vessels. English labels can train category translations,
but ranker training uses Korean-origin records/augmentations only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import augment
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.experiment import TESTLIKE_MIX, build_testlike, _predict_blocks, _stack
from src.features import FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.llm_cache import BeamCache
from src.match_features import EXTRA_FEATURE_NAMES, ExtraFeatures
from src.retrieval import CatalogRetriever, RETRIEVAL_VERSION
from src.train_ranker import PARAMS
from src.validate import load_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[4])
    parser.add_argument("--learners", nargs="+", choices=["base", "rich", "cat"], default=["base", "rich"])
    parser.add_argument("--tag", default="strict_han_v1")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    train, catalog = load_train(), load_catalog()
    hangul = train.vessel_name.map(is_hangul).to_numpy()
    folds = load_folds(train)
    r = CatalogRetriever(catalog)
    truth = train.target_vessel_id.map(r.id_to_index).to_numpy()
    beams = BeamCache(allow_generate=False)
    aug = augment(train, 4, 1, seed=7)
    aug = aug.loc[hangul[aug.origin.to_numpy()]].reset_index(drop=True)
    # Fixed testlike transform, independent of fold and training augmentation.
    from types import SimpleNamespace
    tl = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=hangul))
    hard = augment(train, 2, 1, seed=2026)
    hard = hard.loc[hangul[hard.origin.to_numpy()]].reset_index(drop=True)
    hard["sample_id"] = hard.sample_id + "_hard"
    result_path = ROOT / "experiments" / f"{args.tag}.json"
    results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"folds": {}, "retrieval": "training-fold-only category mappers", "ranker_population": "Korean origin only"}
    for k in args.folds:
        start = time.perf_counter()
        tr = np.flatnonzero(folds != k)
        va = np.flatnonzero((folds == k) & hangul)
        builder = FeatureBuilder(r, *fit_mappers(train.iloc[tr], catalog))
        r.category_score = builder.category_score
        extra = ExtraFeatures(r)
        original = train.loc[(folds != k) & hangul].copy()
        original["origin"] = np.flatnonzero((folds != k) & hangul)
        train_df = pd.concat([original, aug.loc[folds[aug.origin.to_numpy()] != k]], ignore_index=True)
        valid_df = train.iloc[va].copy()
        valid_df["origin"] = va
        datasets = {"train": train_df, "oof": valid_df,
                    "hard": hard.loc[folds[hard.origin.to_numpy()] == k],
                    "testlike": tl.loc[folds[tl.origin.to_numpy()] == k]}
        matrices = {}
        signature = hashlib.sha256((RETRIEVAL_VERSION + ",".join(FEATURE_NAMES + EXTRA_FEATURE_NAMES)).encode())
        for module in ("match_features.py", "features.py", "retrieval.py", "strict_hangul_experiment.py"):
            signature.update((ROOT / "src" / module).read_bytes())
        signature.update(train.iloc[tr].to_csv(index=False).encode())
        signature.update(catalog.to_csv(index=False).encode())
        source_key = signature.hexdigest()[:12]
        for s, df in datasets.items():
            cache = ROOT / "experiments" / "cache" / f"strict_han_f{k}_{s}_{source_key}.joblib"
            if cache.exists():
                matrices[s] = joblib.load(cache)
                continue
            row_list = df.to_dict("records")
            bb = beams(df.vessel_name.tolist())
            def build_chunk(start):
                part = df.iloc[start:start + 48]
                part_beams = bb[start:start + 48]
                signals = r.generate(part, llm_beams=part_beams)
                block_list = []
                for row, sig, beam in zip(part.to_dict("records"), signals, part_beams):
                    base = builder.build(row, sig)
                    block_list.append(np.column_stack([base, extra.build(row, sig["cand"], base, beam)]))
                return block_list, [sig["cand"] for sig in signals]

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                chunks = list(executor.map(build_chunk, range(0, len(df), 48)))
            blocks = [b for chunk, _ in chunks for b in chunk]
            cands = [c for _, chunk in chunks for c in chunk]
            matrices[s] = {"blocks": blocks, "candidates": cands,
                           "truth": truth[df.origin.to_numpy()], "sample_ids": df.sample_id.tolist()}
            joblib.dump(matrices[s], cache)
            del chunks, bb
            extra.token_stats.cache_clear()
            print(f"strict fold {k}: {s} {len(df)} rows ready ({(time.perf_counter() - start)/60:.1f}m)", flush=True)
        labels = [(c == t).astype(np.float32) for c, t in zip(matrices["train"]["candidates"], matrices["train"]["truth"])]
        fold_results = {}
        for learner in args.learners:
            names = FEATURE_NAMES if learner == "base" else FEATURE_NAMES + EXTRA_FEATURE_NAMES
            data = [b[:, :len(names)] for b in matrices["train"]["blocks"]]
            xs, ys, group = _stack(data, labels)
            if learner == "cat":
                from catboost import CatBoostRanker, Pool
                model = CatBoostRanker(loss_function="YetiRank", iterations=400, learning_rate=.1, depth=6,
                                       random_seed=42, thread_count=8, verbose=False, metric_period=1000)
                model.fit(Pool(xs, ys, group_id=np.repeat(np.arange(len(group)), group)))
                predict = lambda b: model.predict(b)
            else:
                model = lgb.train(PARAMS, lgb.Dataset(xs, ys, group=group, feature_name=names), num_boost_round=150)
                predict = lambda b: model.predict(b, num_threads=6)
            del xs, ys, group, data
            gc.collect()
            scores, correct, recalls = {}, {}, {}
            for s in ("oof", "hard", "testlike"):
                m = matrices[s]
                scores[s] = _predict_blocks(predict, [b[:, :len(names)] for b in m["blocks"]])
                pred = np.array([c[int(np.argmax(sc))] if len(c) else -1 for c, sc in zip(m["candidates"], scores[s])])
                correct[s] = pred == m["truth"]
                recalls[s] = float(np.mean([t in c for c, t in zip(m["candidates"], m["truth"])]))
            parts = {"clean": float(correct["oof"].mean())}
            for scenario in ("gt_missing", "drop_word", "both"):
                parts[scenario] = float(correct["testlike"][datasets["testlike"].scenario.eq(scenario).to_numpy()].mean())
            metrics = {"testlike_hangul": sum(TESTLIKE_MIX[s] * parts[s] for s in parts),
                       "scenarios": parts, "hard_hangul": float(correct["hard"].mean()), "recall": recalls}
            fold_results[learner] = metrics
            out = ROOT / "experiments" / "artifacts" / args.tag
            out.mkdir(parents=True, exist_ok=True)
            model.save_model(str(out / f"{learner}_fold{k}.{'cbm' if learner == 'cat' else 'txt'}"))
            joblib.dump({"scores": scores, "sets": {s: {key: value for key, value in matrices[s].items() if key != "blocks"}
                                                      for s in ("oof", "hard", "testlike")},
                         "source": source_key, "metrics": metrics}, out / f"{learner}_fold{k}_scores.joblib", compress=3)
            print(json.dumps({"fold": k, "learner": learner, **metrics}, ensure_ascii=False), flush=True)
            del model
            gc.collect()
        results["folds"][str(k)] = {**results["folds"].get(str(k), {}), **fold_results}
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (ROOT / "experiments" / "artifacts" / args.tag / f"fold{k}.ready").write_text(source_key, encoding="utf-8")
        del matrices, datasets, builder, extra
        gc.collect()


if __name__ == "__main__":
    main()
