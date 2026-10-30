"""Fresh held-out perturbations with independently fitted retrieval and mapper.

This diagnostic compares v10 and a new pair scorer on the same strict LGB
baseline. Its scenario mean is not the official metric or historical testlike.
"""
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

from src.augment import keep_first, keep_last
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.ensemble import normalize
from src.features import FeatureBuilder, fit_mappers
from src.llm_cache import BeamCache
from src.llm_translit import FEW_SHOT
from src.retrieval import CatalogRetriever
from src.validate import load_folds
from src.pair_views import mix_views


def make_rows(fold):
    train = load_train()
    folds = load_folds(train)
    shot_ids = train.loc[train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"]
    selected = train.loc[(folds == fold) & train.vessel_name.map(is_hangul)
                         & ~train.target_vessel_id.isin(shot_ids)]
    rng = np.random.default_rng(20261028)
    rows = []
    for original in selected.to_dict("records"):
        for scenario in ("gt_only_missing", "igt_only_missing", "conflicting_gt",
                         "first_with_gt", "last_with_gt", "first_both_missing", "last_both_missing"):
            row = dict(original)
            if scenario == "gt_only_missing":
                row["gross_tonnage"] = np.nan
            elif scenario == "igt_only_missing":
                row["international_gross_tonnage"] = np.nan
            elif scenario == "conflicting_gt":
                row["gross_tonnage"] *= float(rng.choice([.5, 1.5]))
            elif scenario.startswith("first"):
                row["vessel_name"] = keep_first(row["vessel_name"])
            elif scenario.startswith("last"):
                row["vessel_name"] = keep_last(row["vessel_name"])
            if scenario.endswith("both_missing"):
                row["gross_tonnage"] = row["international_gross_tonnage"] = np.nan
            row["sample_id"] += "_fresh_" + scenario
            row["scenario"] = scenario
            row["name_changed"] = row["vessel_name"] != original["vessel_name"]
            rows.append(row)
    return train, folds, pd.DataFrame(rows)


def prepare(fold):
    train, folds, frame = make_rows(fold)
    beams = BeamCache()
    names = list(dict.fromkeys(frame.vessel_name))
    missing = [name for name in names if is_hangul(name) and name not in beams.data]
    print(f"Fresh fold {fold}: {len(frame)} cases, {len(missing)} missing transliterations", flush=True)
    for name in missing:
        beams([name])
    beams.save()
    bb = beams(frame.vessel_name.tolist())
    beams._model = None
    gc.collect()
    torch.cuda.empty_cache()
    source = hashlib.sha256(frame.to_csv(index=False).encode())
    for file in [ROOT / "vessel_catalog.csv", ROOT / "train.csv", ROOT / "tools/pair_fresh_stress.py"] + [
        ROOT / "src" / name for name in ("features.py", "retrieval.py", "common.py", "hangul.py")]:
        source.update(file.read_bytes())
    source.update(json.dumps(bb, ensure_ascii=False).encode())
    miner_path = ROOT / "experiments/artifacts/strict_han_v1" / f"base_fold{fold}.txt"
    source.update(miner_path.read_bytes())
    path = ROOT / "experiments/cache" / f"fresh_pairs_f{fold}_{source.hexdigest()[:12]}.joblib"
    if not path.exists():
        catalog = load_catalog()
        r = CatalogRetriever(catalog)
        builder = FeatureBuilder(r, *fit_mappers(train.loc[folds != fold], catalog))
        r.category_score = builder.category_score
        model = lgb.Booster(model_file=str(miner_path))

        def chunk(start):
            part = frame.iloc[start:start+48]
            sigs = r.generate(part, llm_beams=bb[start:start+48])
            blocks = [builder.build(row, sig) for row, sig in zip(part.to_dict("records"), sigs)]
            sizes = np.cumsum([0] + [len(b) for b in blocks])
            raw = model.predict(np.vstack(blocks), num_threads=1)
            return [{"candidates": sig["cand"], "base": normalize(raw[sizes[i]:sizes[i+1]], "prob")}
                    for i, sig in enumerate(sigs)]

        with ThreadPoolExecutor(max_workers=6) as executor:
            pieces = list(executor.map(chunk, range(0, len(frame), 48)))
        records = frame.to_dict("records")
        data = {"records": records, "beams": bb, "pairs": [item for part in pieces for item in part],
                "truth": frame.target_vessel_id.map(r.id_to_index).to_numpy(), "fold": fold,
                "source": source.hexdigest(), "few_shot_vessels_excluded": True,
                "training_valid_vessels_overlap": int(bool(set(train.loc[folds != fold, "target_vessel_id"])
                                                             & set(frame.target_vessel_id)))}
        assert data["training_valid_vessels_overlap"] == 0
        joblib.dump(data, path, compress=3)
    pointer = ROOT / "experiments" / f"fresh_pairs_f{fold}.json"
    pointer.write_text(json.dumps({"path": str(path), "source": source.hexdigest()}, indent=2), encoding="utf-8")
    print(f"Prepared fresh candidates: {path}", flush=True)
    return joblib.load(path)


def evaluate(data, tag):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.neural_pair import encode_pairs as raw_encode
    from src.pair_evidence import encode_pairs as evidence_encode
    model_dir = "D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    yes, no = [tokenizer.convert_tokens_to_ids(w) for w in ("yes", "no")]
    catalog = load_catalog().to_dict("records")
    predictions = {"baseline": np.array([row["candidates"][int(np.argmax(row["base"]))] for row in data["pairs"]])}
    distributions = {}
    torch.set_num_threads(4)
    for label, folder, recipe in (("v10", "neural_v8_cv", "raw"), ("candidate", tag, "evidence")):
        base = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, ROOT / "experiments/artifacts" / folder / f"fold{data['fold']}_adapter").merge_and_unload().eval()
        model.config.use_cache = False
        pred = predictions["baseline"].copy()
        probabilities = [None] * len(pred)
        calls = 0
        with torch.inference_mode():
            for i, (row, pair, beams) in enumerate(zip(data["records"], data["pairs"], data["beams"])):
                if not is_hangul(row["vessel_name"]):
                    continue
                keep = np.argsort(-pair["base"], kind="stable")[:20]
                bp = pair["base"][keep]
                bp = bp/bp.sum()
                if len(keep) < 2 or bp[0]-bp[1] > 1/3+1e-6:
                    continue
                c = [catalog[j] for j in pair["candidates"][keep]]
                tokens = evidence_encode(tokenizer, row, c, beams) if recipe == "evidence" else raw_encode(tokenizer, row, c)
                enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
                logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                score = (logits[:, yes]-logits[:, no]).cpu().numpy()
                mixed = .75*bp+.25*normalize(score, "prob")
                probabilities[i] = (keep, score)
                pred[i] = pair["candidates"][keep[int(np.argmax(mixed))]]
                calls += 1
        predictions[label] = pred
        distributions[label] = probabilities
        print(f"Fresh {label}: neural calls {calls}", flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    predictions["ensemble"] = predictions["baseline"].copy()
    for i, (old, new) in enumerate(zip(distributions["v10"], distributions["candidate"])):
        assert (old is None) == (new is None)
        if old is not None:
            np.testing.assert_array_equal(old[0], new[0])
            bp = data["pairs"][i]["base"][old[0]]
            average = mix_views(bp/bp.sum(), [old[1], new[1]], [1, 1])
            predictions["ensemble"][i] = data["pairs"][i]["candidates"][old[0][int(np.argmax(average))]]
    df = pd.DataFrame(data["records"])
    correct = {s: p == data["truth"] for s, p in predictions.items()}
    parts = {}
    for scenario in df.scenario.unique():
        pick = df.scenario.eq(scenario).to_numpy()
        changed = pick & df.name_changed.to_numpy()
        parts[scenario] = {"rows": int(pick.sum()), "accuracy": {s: float(ok[pick].mean()) for s, ok in correct.items()},
                           "name_changed_rows": int(changed.sum()),
                           "changed_only_accuracy": {s: float(ok[changed].mean()) for s, ok in correct.items()} if changed.any() else None}
    delta = correct["candidate"].astype(float)-correct["v10"]
    _, group = np.unique(df.target_vessel_id, return_inverse=True)
    n = int(group.max())+1
    draws = np.random.default_rng(20261028).multinomial(n, np.full(n, 1/n), size=2000)
    sums = np.bincount(group, weights=delta, minlength=n)
    counts = np.bincount(group, minlength=n)
    boot = (draws@sums)/(draws@counts)
    report = {"fold": data["fold"], "tag": tag, "rows": len(df), "source": data["source"],
              "scenario_accuracy": parts, "mean_accuracy": {s: float(ok.mean()) for s, ok in correct.items()},
              "candidate_delta_vs_v10": float(delta.mean()), "bootstrap_95_vs_v10": np.quantile(boot, [.025, .975]).tolist(),
              "gains_vs_v10": int(np.sum(delta > 0)), "losses_vs_v10": int(np.sum(delta < 0)),
              "ensemble_delta_vs_v10": float((correct["ensemble"].astype(float)-correct["v10"]).mean()),
              "caveat": "Diagnostic, equal scenario mix; strict Korean-only LGB baseline, not official metric/testlike/Public."}
    stem = ROOT / "experiments" / f"fresh_pairs_f{data['fold']}_{tag}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": predictions, "sample_ids": df.sample_id.tolist(), "source": data["source"]},
                stem.with_suffix(".joblib"), compress=3)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--tag", default="neural_evidence_v1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data = prepare(args.fold)
    if not args.prepare_only:
        evaluate(data, args.tag)


if __name__ == "__main__":
    main()
