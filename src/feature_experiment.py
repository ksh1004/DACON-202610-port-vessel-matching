"""Train richer pair rankers on the existing fixed candidate sets.

This first ablation shares historical candidate retrieval (including its documented
full-train category-sort caveat). Strict fold retrieval is a separate promotion check.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT
from src.ensemble import normalize
from src.experiment import Lab, RESULTS, SCORES, _predict_blocks, _stack
from src.features import FEATURE_NAMES
from src.match_features import EXTRA_FEATURE_NAMES, ExtraFeatures
from src.train_ranker import PARAMS


def extend(lab):
    ctx = lab.ctx
    extra = ExtraFeatures(ctx.retriever)
    sets = {"oof": (ctx.rows, ctx.cands, ctx.blocks),
            "aug": (ctx.aug_rows, ctx.aug_cands, ctx.aug_blocks),
            **{s: (es.rows, es.cands, es.blocks) for s, es in lab.sets.items()}}
    identity = hashlib.sha256(Path(__file__).with_name("match_features.py").read_bytes())
    identity.update(json.dumps(ctx.beams.data, sort_keys=True, ensure_ascii=False).encode())
    identity.update(ctx.catalog.to_csv(index=False).encode())
    for s, (rows, cands, _) in sets.items():
        identity.update(s.encode())
        for row, cand in zip(rows, cands):
            identity.update(str(row["sample_id"]).encode())
            identity.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            identity.update(np.asarray(cand, dtype=np.int64).tobytes())
    digest = identity.hexdigest()
    cache = ROOT / "experiments" / "cache" / f"extra_match_{digest[:12]}.joblib"
    if cache.exists():
        added = joblib.load(cache)
    else:
        added = {}
        for s, (rows, cands, blocks) in sets.items():
            start = time.perf_counter()
            added[s] = []
            for i, (row, cand, block) in enumerate(zip(rows, cands, blocks)):
                beams = ctx.beams([row["vessel_name"]])[0]
                added[s].append(extra.build(row, cand, block, beams))
                if (i + 1) % 2000 == 0:
                    print(f"extra {s}: {i + 1}/{len(rows)} ({time.perf_counter() - start:.0f}s)", flush=True)
            extra.token_stats.cache_clear()
        joblib.dump(added, cache)
    for s, (_, cands, blocks) in sets.items():
        if len(blocks) != len(added[s]):
            raise ValueError(f"Cached extra row count mismatch: {s}")
        for i, (cand, value) in enumerate(zip(cands, added[s])):
            if value.shape != (len(cand), len(EXTRA_FEATURE_NAMES)):
                raise ValueError(f"Cached extra candidate count mismatch: {s}/{i}")
            blocks[i] = np.column_stack([blocks[i], value])
    # Scores keep their original format for src.blend; a companion file stores
    # IDs, order and an input/source fingerprint for unambiguous reuse.
    meta = {"fingerprint": digest, "feature_names": FEATURE_NAMES + EXTRA_FEATURE_NAMES,
            "retrieval": "historical fixed candidates; full-train category-sort caveat",
            "sets": {s: {"sample_ids": [r["sample_id"] for r in rows], "candidates": cands}
                     for s, (rows, cands, _) in sets.items() if s != "aug"}}
    del added, sets
    gc.collect()
    return meta


def fit_lgb(X, y, names, rounds=150, hangul_weight=1.0):
    xs, ys, group = _stack(X, y)
    weights = np.where(xs[:, FEATURE_NAMES.index("q_hangul")] > 0, hangul_weight, 1.0)
    model = lgb.train(PARAMS, lgb.Dataset(xs, ys, group=group, feature_name=names,
                                        weight=weights if hangul_weight != 1 else None), num_boost_round=rounds)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["rich150", "hard20"],
                        choices=["base150", "rich150", "rich300", "hard20", "hardcat"])
    parser.add_argument("--tag", default="match_v1")
    parser.add_argument("--hangul-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.hangul_weight) or args.hangul_weight <= 0:
        parser.error("--hangul-weight must be finite and positive")
    lab = Lab()
    meta = extend(lab)
    all_names = FEATURE_NAMES + EXTRA_FEATURE_NAMES
    for kind in args.models:
        start = time.perf_counter()
        output = {s: [None] * size for s, size in lab.sizes().items()}
        auxiliary = {s: [None] * size for s, size in lab.sizes().items()} if kind == "hardcat" else None
        fold_metrics = []
        for k, X, y, evals in lab.folds():
            base_only = kind == "base150"
            names = FEATURE_NAMES if base_only else all_names
            fit_x = [b[:, :len(FEATURE_NAMES)] for b in X] if base_only else X
            primary = fit_lgb(fit_x, y, names, rounds=300 if kind == "rich300" else 150,
                              hangul_weight=args.hangul_weight)
            second = None
            if kind in ("hard20", "hardcat"):
                # Mine with this outer fold's training-only model. Do not use
                # stored OOF scores for training rows, as those models may have
                # seen the current outer validation fold.
                train_scores = _predict_blocks(lambda b: primary.predict(b, num_threads=6), X)
                pick = [np.unique(np.r_[np.argsort(-sc, kind="stable")[:20], np.flatnonzero(t)])
                        for sc, t in zip(train_scores, y)]
                selected_x, selected_y = [b[p] for b, p in zip(X, pick)], [t[p] for t, p in zip(y, pick)]
                if kind == "hardcat":
                    from catboost import CatBoostRanker, Pool
                    xs, ys, group = _stack(selected_x, selected_y)
                    second = CatBoostRanker(loss_function="YetiRank", iterations=400, depth=6, learning_rate=.1,
                                           random_seed=42, thread_count=8, verbose=False, metric_period=1000)
                    group_weights = np.where(xs[:, FEATURE_NAMES.index("q_hangul")] > 0, args.hangul_weight, 1.0)
                    second.fit(Pool(xs, ys, group_id=np.repeat(np.arange(len(group)), group),
                                    group_weight=group_weights))
                    del xs, ys, group
                else:
                    second = fit_lgb(selected_x, selected_y, all_names, hangul_weight=args.hangul_weight)
                del selected_x, selected_y
                del train_scores, pick
            for s, (idx, blocks) in evals.items():
                data = [b[:, :len(FEATURE_NAMES)] for b in blocks] if base_only else blocks
                raw = _predict_blocks(lambda b: primary.predict(b, num_threads=6), data)
                if second is not None:
                    rerank = _predict_blocks(lambda b: second.predict(b, thread_count=6) if kind == "hardcat"
                                               else second.predict(b, num_threads=6), data)
                    result = []
                    for j, (sc, new) in enumerate(zip(raw, rerank)):
                        keep = np.argsort(-sc, kind="stable")[:20]
                        final = np.full(len(sc), -1000.0, dtype=np.float32)
                        probability = .5 * normalize(sc[keep], "prob") + .5 * normalize(new[keep], "prob")
                        final[keep] = np.log(np.maximum(probability, 1e-30))
                        result.append(final)
                        if auxiliary is not None:
                            single = np.full(len(sc), -1000.0, dtype=np.float32)
                            single[keep] = new[keep]
                            auxiliary[s][idx[j]] = single
                    raw = result
                for i, sc in zip(idx, raw):
                    output[s][i] = sc
            # Keep a reproducible partial checkpoint for long runs.
            name = f"{kind}_{args.tag}"
            folder = ROOT / "experiments" / "artifacts" / name
            folder.mkdir(parents=True, exist_ok=True)
            primary.save_model(str(folder / f"fold{k}.txt"))
            if second is not None:
                second.save_model(str(folder / f"fold{k}_second.{'cbm' if kind == 'hardcat' else 'txt'}"))
            truth = lab.ctx.truth
            idx = evals["oof"][0]
            acc = float(np.mean([lab.ctx.cands[i][int(np.argmax(output["oof"][i]))] == truth[i] for i in idx]))
            fold_metrics.append({"fold": int(k), "unweighted_oof_accuracy": acc})
            print(f"{name}: fold {k} finished, raw accuracy {acc:.5f}, elapsed {(time.perf_counter() - start)/60:.1f}m", flush=True)
            del primary, second, X, y, evals, fit_x
            gc.collect()
        name = f"{kind}_{args.tag}"
        joblib.dump(output, SCORES / f"{name}.joblib")
        joblib.dump(meta, SCORES / f"{name}.identity.joblib", compress=3)
        result = {"name": name, **lab.evaluate(output), "minutes": round((time.perf_counter() - start) / 60, 1),
                  "n_features": len(FEATURE_NAMES) if kind == "base150" else len(all_names),
                  "hangul_weight": args.hangul_weight,
                  "folds": fold_metrics, "fingerprint": meta["fingerprint"]}
        with RESULTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if auxiliary is not None:
            aux_name = f"{kind}_only_{args.tag}"
            joblib.dump(auxiliary, SCORES / f"{aux_name}.joblib")
            joblib.dump(meta, SCORES / f"{aux_name}.identity.joblib", compress=3)
            aux_result = {"name": aux_name, **lab.evaluate(auxiliary), "fingerprint": meta["fingerprint"]}
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(aux_result, ensure_ascii=False) + "\n")
            print(json.dumps(aux_result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
