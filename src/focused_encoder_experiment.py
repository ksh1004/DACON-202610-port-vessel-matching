"""Fixed-budget augmentation/positive-restoration controls against frozen v12."""
import argparse
import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.augment import augment
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.encoder_extended_experiment import seed_training, fit
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetic_experiment import MODEL_DIR, prepare as baseline_prepare, evaluate
from src.encoder_phonetics import encode_pairs
from src.experiment import _predict_blocks, _stack
from src.features import FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.llm_cache import BeamCache
from src.neural_pair import record_text
from src.retrieval import CatalogRetriever
from src.validate import load_folds


def is_target(row, original):
    if not (pd.isna(row.get("gross_tonnage")) and pd.isna(row.get("international_gross_tonnage"))):
        return False
    words = original["vessel_name"].split()
    return len(words) > 1 and any(" ".join(words[:j]+words[j+1:]) == row["vessel_name"] for j in range(len(words)))


def identity(row):
    # Labels are used for sampling/deduplication only, never model input.
    return (row["target_vessel_id"], record_text(row))


def training_source(fold, tokenizer):
    train, catalog = load_train(), load_catalog()
    han = train.vessel_name.map(is_hangul).to_numpy()
    folds = load_folds(train)
    eligible_origin = han if fold is None else han & (folds != fold)
    if fold is not None:
        baseline = baseline_prepare(fold, 2400, tokenizer)
        original = train.loc[eligible_origin].copy()
        original["origin"] = np.flatnonzero(eligible_origin)
        aug = augment(train, 4, 1, seed=7)
        frame = pd.concat([original, aug.loc[eligible_origin[aug.origin.to_numpy()]]], ignore_index=True)
        path = max((ROOT / "experiments/cache").glob(f"strict_han_f{fold}_train_*.joblib"), key=lambda p: p.stat().st_mtime)
        matrix = joblib.load(path)
        assert matrix["sample_ids"] == frame.sample_id.tolist()
        miner_path = ROOT / "experiments/artifacts/strict_han_v1" / f"base_fold{fold}.txt"
        miner = lgb.Booster(model_file=str(miner_path))
        source = {"matrix": path.name, "miner_sha256": hashlib.sha256(miner_path.read_bytes()).hexdigest()}
    else:
        from src.train_ranker import Context, PARAMS
        spec = json.loads((ROOT / "experiments/artifacts/ranker_v12_encoder/encoder_views.json").read_text(encoding="utf-8"))
        fp = spec["training_group_fingerprint"]
        baseline = joblib.load(ROOT / "experiments/cache" / f"final_encoder_groups_{fp[:12]}.joblib")
        assert baseline["fingerprint"] == fp and len(baseline["groups"]) == 2400
        baseline["evaluation"] = {}
        ctx = Context(4, 1, seed=7, use_llm=True, allow_llm_generate=False)
        items = [(dict(row, origin=i), c, b, t) for i, (row, c, b, t) in enumerate(zip(ctx.rows, ctx.cands, ctx.blocks, ctx.truth)) if ctx.hangul[i]]
        items += [(row, c, b, ctx.truth[o]) for row, c, b, o in zip(ctx.aug_rows, ctx.aug_cands, ctx.aug_blocks, ctx.aug_origin) if ctx.hangul[o]]
        frame = pd.DataFrame([row for row, _, _, _ in items])
        matrix = {"candidates": [c for _, c, _, _ in items], "blocks": [b for _, _, b, _ in items],
                  "truth": np.array([t for _, _, _, t in items])}
        xs, ys, groups = _stack(matrix["blocks"], [(c == t).astype(np.float32) for c, t in zip(matrix["candidates"], matrix["truth"])])
        miner = lgb.train({**PARAMS, "num_threads": 6}, lgb.Dataset(xs, ys, group=groups, feature_name=FEATURE_NAMES), num_boost_round=150)
        source = {"matrix": "full official original+seed7 augmentation", "miner_sha256": hashlib.sha256(miner.model_to_string().encode()).hexdigest()}
        del ctx, items, xs, ys, groups
        gc.collect()
    return train, catalog, folds, eligible_origin, frame, matrix, miner, baseline, source


def prepare(fold, mode, tokenizer):
    assert mode in {"focused", "restore"}
    stamp = "full" if fold is None else str(fold)
    digest = hashlib.sha256(f"{stamp}:{mode}:2400:target480:seed20261028".encode())
    for name in ("src/focused_encoder_experiment.py", "src/encoder_phonetic_experiment.py", "src/encoder_phonetics.py",
                 "src/features.py", "src/retrieval.py", "src/augment.py", "train.csv", "vessel_catalog.csv"):
        digest.update((ROOT / name).read_bytes())
    for name in ("tokenizer.json", "codex_revision.json"):
        digest.update((MODEL_DIR / name).read_bytes())
    if fold is None:
        digest.update((ROOT / "experiments/artifacts/ranker_v12_encoder/encoder_views.json").read_bytes())
    else:
        digest.update((ROOT / "experiments/artifacts/strict_han_v1" / f"base_fold{fold}.txt").read_bytes())
        for tag in ("encoder_bge_phonetic", "neural_evidence_v1"):
            digest.update((ROOT / "experiments/artifacts" / tag / f"fold{fold}_epoch1_scores.joblib").read_bytes())
    path = ROOT / "experiments/cache" / f"focused_groups_{stamp}_{mode}_{digest.hexdigest()[:12]}.joblib"
    if path.exists():
        result = joblib.load(path)
        assert result["fingerprint"] == digest.hexdigest()
        beams = BeamCache(allow_generate=False)
        actual = {n: beams([n])[0] for n in result["used_new_names"]}
        assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == result["new_beams_sha256"]
        return result
    train, catalog, folds, eligible, frame, matrix, miner, baseline, source = training_source(fold, tokenizer)
    rows = frame.to_dict("records")
    catrows = catalog.to_dict("records")
    chosen = []
    absent = []
    for i in np.random.default_rng(42).permutation(len(rows)):
        c, target = matrix["candidates"][i], matrix["truth"][i]
        if target not in c:
            absent.append(int(i))
        elif len(c) >= 8 and len(chosen) < 2400:
            chosen.append(int(i))
    assert len(chosen) == len(baseline["groups"]) == 2400
    beams = BeamCache(allow_generate=False)
    for position, i in enumerate(chosen):
        tokens = encode_pairs(tokenizer, rows[i], [catrows[matrix["truth"][i]]], beams([rows[i]["vessel_name"]])[0])[0]
        assert tokens == baseline["groups"][position][0], f"Baseline positive token mismatch at {position}"
    original_rows = train.to_dict("records")
    old_target = [is_target(rows[i], original_rows[int(rows[i]["origin"])]) for i in chosen]
    ids = {identity(rows[i]) for i in chosen}
    pool = []
    seen = set(ids)
    if mode == "focused":
        for origin in np.flatnonzero(eligible):
            original = original_rows[origin]
            words = original["vessel_name"].split()
            for j in range(len(words)) if len(words) > 1 else ():
                name = " ".join(words[:j]+words[j+1:])
                if not is_hangul(name):
                    continue
                row = dict(original, vessel_name=name, gross_tonnage=np.nan, international_gross_tonnage=np.nan,
                           origin=int(origin), sample_id=f"focus_{origin}_{j}")
                key = identity(row)
                if key not in seen:
                    pool.append(row)
                    seen.add(key)
        need = 480-sum(old_target)
        assert need > 0
    else:
        for i in absent:
            if identity(rows[i]) not in seen and len(matrix["candidates"][i]) >= 7:
                pool.append(dict(rows[i], source_index=i))
                seen.add(identity(rows[i]))
        need = min(len(pool), 240)
        assert need > 0
    if fold is not None:
        assert not set(row["target_vessel_id"] for row in pool) & set(train.loc[folds == fold, "target_vessel_id"])
    order = np.random.default_rng(20261028).permutation(len(pool))
    pool = [pool[int(i)] for i in order]
    print(f"{mode} fold {stamp}: baseline target {sum(old_target)}, fresh unique pool {len(pool)}, replacements {need}", flush=True)
    cols = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]

    def encode_group(row, c, block, scores, target):
        hit = np.flatnonzero(c == target)
        strength = np.maximum(block[:, cols[0]], block[:, cols[1]])-.2*block[:, cols[2]]
        hard = [int(j) for j in np.argsort(-scores, kind="stable") if c[j] != target][:5]
        lexical = [int(j) for j in np.argsort(-strength, kind="stable") if c[j] != target and j not in hard][:2]
        assert len(hard)+len(lexical) == 7
        documents = [catrows[target]]+[catrows[c[j]] for j in hard+lexical]
        return encode_pairs(tokenizer, row, documents, beams([row["vessel_name"]])[0])

    new_groups, selected = [], []
    if mode == "restore":
        for row in pool[:need]:
            i = row["source_index"]
            c, block, target = matrix["candidates"][i], matrix["blocks"][i][:, :len(FEATURE_NAMES)], int(matrix["truth"][i])
            assert target not in c
            scores = miner.predict(block, num_threads=4)
            new_groups.append(encode_group(row, c, block, scores, target))
            selected.append(row)
    else:
        del matrix
        gc.collect()
        # The frozen generator is query-local. Complete before resetting training RNG.
        beams = BeamCache()
        for name in dict.fromkeys(row["vessel_name"] for row in pool):
            beams([name])
        beams.save()
        beams._model = None
        gc.collect()
        torch.cuda.empty_cache()
        r = CatalogRetriever(catalog)
        mapping_train = train if fold is None else train.loc[folds != fold]
        builder = FeatureBuilder(r, *fit_mappers(mapping_train, catalog))
        r.category_score = builder.category_score

        def chunk(start):
            part = pool[start:start+40]
            bb = beams([row["vessel_name"] for row in part])
            signals = r.generate(pd.DataFrame(part), llm_beams=bb)
            output = []
            for row, sig in zip(part, signals):
                c = sig["cand"]
                target = r.id_to_index[row["target_vessel_id"]]
                if target not in c or len(c) < 8:
                    continue  # A1 retains the original positive-present criterion.
                block = builder.build(row, sig)
                output.append((row, c, block, miner.predict(block, num_threads=1), target))
            return output

        with ThreadPoolExecutor(max_workers=6) as executor:
            chunks = list(executor.map(chunk, range(0, len(pool), 40)))
        candidates = [item for part in chunks for item in part]
        print(f"{mode} fold {stamp}: {len(candidates)} eligible fresh cases", flush=True)
        assert len(candidates) >= need, "Not enough unique fresh cases; do not fill with repeated examples"
        for row, c, block, scores, target in candidates[:need]:
            new_groups.append(encode_group(row, c, block, scores, target))
            selected.append(row)
        del candidates, chunks, builder, r
        gc.collect()
    replace_pool = [i for i, flag in enumerate(old_target) if not flag] if mode == "focused" else list(range(2400))
    positions = np.random.default_rng(20261028).permutation(replace_pool)[:need]
    groups = list(baseline["groups"])
    selected_rows = [rows[i] for i in chosen]
    for position, tokens, row in zip(positions, new_groups, selected):
        groups[int(position)] = tokens
        selected_rows[int(position)] = row
    final_target = sum(is_target(row, original_rows[int(row["origin"])]) for row in selected_rows)
    assert len(groups) == 2400
    if mode == "focused":
        assert final_target == 480
    unchanged = set(range(2400))-set(map(int, positions))
    assert all(groups[i] == baseline["groups"][i] for i in unchanged)
    overlap = 0 if fold is None else len(set(row["target_vessel_id"] for row in selected_rows) & set(train.loc[folds == fold, "target_vessel_id"]))
    assert overlap == 0
    names = sorted({row["vessel_name"] for row in selected})
    bb = {n: beams([n])[0] for n in names}
    result = {"groups": groups, "evaluation": baseline["evaluation"], "fingerprint": digest.hexdigest(),
              "used_new_names": names, "new_beams_sha256": hashlib.sha256(json.dumps(bb, sort_keys=True).encode()).hexdigest(),
              "protocol": {"fold": fold, "mode": mode, "input_recipe": "phonetic", "training_groups": 2400,
                           "epochs": 1, "seed": 42, "max_tokens": 256, "reference": "v12", "vessel_overlap": overlap,
                           "old_target_groups": sum(old_target), "final_target_groups": final_target,
                           "replacement_groups": need, "new_unique_cases": len({identity(row) for row in selected}),
                           "training_at_cap": sum(len(tokens) == 256 for group in groups for tokens in group),
                           "baseline_fingerprint": baseline["fingerprint"], "source": source,
                           "trainer_sha256": hashlib.sha256((ROOT / "src/focused_encoder_experiment.py").read_bytes()).hexdigest(),
                           "caveat": "Training retrieval/mappers strictly outer-fold only. Historical v8 validation candidate mapper exception remains."},
              "selected_rows": selected_rows, "replaced_positions": list(map(int, positions))}
    joblib.dump(result, path, compress=3)
    print(f"Saved {path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=["focused", "restore"], default="focused")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, args.mode, tokenizer)
    print(json.dumps(data["protocol"]), flush=True)
    if args.prepare_only:
        return
    tag = "encoder_focused20_v1" if args.mode == "focused" else "encoder_restore_v1"
    folder = ROOT / "experiments/artifacts" / tag
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{args.fold}_epoch1_scores.joblib").exists():
        raise FileExistsError("Preserve completed focused trial")
    seed_training()
    training = fit(data["groups"], tokenizer, folder / f"fold{args.fold}_merged", folder / f"fold{args.fold}_adapter")
    scorer = EncoderPairScorer(folder / f"fold{args.fold}_merged")
    metrics = evaluate(scorer, data, folder, args.fold)
    (ROOT / "experiments" / f"{tag}_fold{args.fold}.json").write_text(
        json.dumps({"protocol": data["protocol"], "training": training, "diagnostic_vs_v11": metrics}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
