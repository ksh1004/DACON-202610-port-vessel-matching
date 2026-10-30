"""Golden check for the new 40% pruning boundary and unchanged model logits."""
import argparse
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from src.common import ROOT, load_train, load_catalog, is_hangul
from src.encoder_views import ViewScorer, EncoderViewsEnsemble
from src.experiment import build_testlike
from src.llm_cache import BeamCache
from src.pair_views import mix_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fold = args.fold
    saved = joblib.load(ROOT / "experiments/artifacts/encoder_confidence40_v1" / f"fold{fold}_epoch1_scores.joblib")
    train, catalog = load_train(), load_catalog()
    frame = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=train.vessel_name.map(is_hangul).to_numpy()))
    lookup = {r["sample_id"]: r for r in frame.to_dict("records")}
    selected = []
    for mode in ("pruned", "previously_active", "newly_active"):
        for key, item in zip(saved["sample_ids"]["testlike"], saved["sets"]["testlike"]):
            if not is_hangul(lookup[key]["vessel_name"]):
                continue
            valid = item["bound_pruned"] if mode == "pruned" else (
                not item["bound_pruned"] and item["prior_bound_pruned"] == (mode == "newly_active"))
            if valid:
                selected.append((lookup[key], item))
                break
        else:
            raise AssertionError(f"No golden case for {mode}")
    views = [(ViewScorer(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_runtime_merged", "qwen", "evidence"), 1),
             (ViewScorer(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_merged", "encoder", "phonetic"), 1)]
    beams = BeamCache(allow_generate=False)
    signals = []
    for row, item in selected:
        bb = beams([row["vessel_name"]])[0]
        signals.append({"cand": item["candidates"], "llm_beams": bb})
        if item["bound_pruned"]:
            continue
        candidates = [catalog.iloc[i].to_dict() for i in item["candidates"][item["keep"]]]
        for scorer, _ in views:
            actual = scorer.score(row, candidates, bb)
            expected = item["qwen_scores"] if scorer.name.startswith("qwen:") else item["neural_scores"]
            np.testing.assert_array_equal(actual, expected)

    class Baseline:
        def score_groups(self, blocks):
            return [selected[int(b[0, 0])][1]["base_scores"] for b in blocks]

    runtime = EncoderViewsEnsemble(Baseline(), views, SimpleNamespace(catalog=catalog), weight=.4)
    rows = [row for row, _ in selected]
    blocks = [np.array([[i]], dtype=float) for i in range(len(rows))]
    values = runtime.score_records(rows, signals, blocks)
    for i, (_, item) in enumerate(selected):
        winner = int(np.argmax(item["base_scores"]))
        if not item["bound_pruned"]:
            keep = item["keep"]
            bp = item["base_scores"][keep]
            winner = int(keep[np.argmax(mix_views(bp/bp.sum(), [item["qwen_scores"], item["neural_scores"]], [1, 1], .4))])
        assert int(np.argmax(values[i])) == winner
    order = [2, 0, 1]
    shuffled = runtime.score_records([rows[i] for i in order], [signals[i] for i in order], [blocks[i] for i in order])
    for j, i in enumerate(order):
        np.testing.assert_array_equal(values[i], shuffled[j])
        np.testing.assert_array_equal(values[i], runtime.score_records([rows[i]], [signals[i]], [blocks[i]])[0])
    print("PASS: weight40 old/new pruning cases, saved logits, deployed mixing, shuffle and singles", flush=True)


if __name__ == "__main__":
    main()
