"""Check merged two-view inference against held-out logits and row order."""
import argparse
import gc
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import ROOT, load_train, load_catalog, is_hangul
from src.experiment import build_testlike
from src.llm_cache import BeamCache
from src.pair_views import PairViewScorer, PairViewsEnsemble, mix_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--tag", default="neural_evidence_v1")
    parser.add_argument("--variant", choices=["evidence_only", "equal_neural_ensemble"], default="equal_neural_ensemble")
    args = parser.parse_args()
    data = joblib.load(ROOT / "experiments/artifacts" / args.tag / f"fold{args.fold}_epoch1_scores.joblib")
    old_label = "merged" if args.fold == 4 else "epoch1"
    old = joblib.load(ROOT / "experiments/artifacts/neural_v8_cv" / f"fold{args.fold}_{old_label}_scores.joblib")
    train, catalog = load_train(), load_catalog()
    frame = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=train.vessel_name.map(is_hangul).to_numpy()))
    rows = {row["sample_id"]: row for row in frame.to_dict("records")}
    selected = []
    for pruned, count in ((True, 1), (False, 2)):
        for sample, item, reference in zip(data["sample_ids"]["testlike"], data["sets"]["testlike"], old["sets"]["testlike"]):
            if item["bound_pruned"] == pruned and is_hangul(rows[sample]["vessel_name"]):
                selected.append((rows[sample], item, reference))
                count -= 1
                if not count:
                    break
        assert count == 0
    beams = BeamCache(allow_generate=False)
    model_dir = "D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B"
    recipes = [("neural_v8_cv", "raw"), (args.tag, "evidence")] if args.variant == "equal_neural_ensemble" else [(args.tag, "evidence")]
    for tag, recipe in recipes:
        merged = ROOT / "experiments/artifacts" / tag / f"fold{args.fold}_runtime_merged"
        if not (merged / "config.json").exists():
            base = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda")
            model = PeftModel.from_pretrained(base, ROOT / "experiments/artifacts" / tag / f"fold{args.fold}_adapter").merge_and_unload().eval()
            model.save_pretrained(merged)
            AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left").save_pretrained(merged)
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
    views = [(PairViewScorer(ROOT / "experiments/artifacts" / tag / f"fold{args.fold}_runtime_merged", recipe), 1)
             for tag, recipe in recipes]
    for row, item, reference in selected:
        if item["bound_pruned"]:
            continue
        c = [catalog.iloc[i].to_dict() for i in item["candidates"][item["keep"]]]
        for scorer, _ in views:
            expected = reference["neural_scores"] if scorer.recipe == "raw" else item["neural_scores"]
            np.testing.assert_array_equal(scorer.score(row, c, beams([row["vessel_name"]])[0]), expected)

    class Baseline:
        def score_groups(self, blocks):
            return [selected[int(b[0, 0])][1]["base_scores"] for b in blocks]

    model = PairViewsEnsemble(Baseline(), views, SimpleNamespace(catalog=catalog))
    rr = [row for row, _, _ in selected]
    ss = [{"cand": item["candidates"], "llm_beams": beams([row["vessel_name"]])[0]} for row, item, _ in selected]
    blocks = [np.array([[i]], dtype=float) for i in range(len(selected))]
    result = model.score_records(rr, ss, blocks)
    for i, (_, item, reference) in enumerate(selected):
        expected = int(np.argmax(item["base_scores"]))
        if not item["bound_pruned"]:
            keep = item["keep"]
            bp = item["base_scores"][keep]
            scores = [reference["neural_scores"], item["neural_scores"]] if len(views) == 2 else [item["neural_scores"]]
            expected = int(keep[int(np.argmax(mix_views(bp/bp.sum(), scores, [1]*len(scores))))])
        assert int(np.argmax(result[i])) == expected
    order = [2, 0, 1]
    shuffled = model.score_records([rr[i] for i in order], [ss[i] for i in order], [blocks[i] for i in order])
    for pos, i in enumerate(order):
        np.testing.assert_array_equal(result[i], shuffled[pos])
        np.testing.assert_array_equal(result[i], model.score_records([rr[i]], [ss[i]], [blocks[i]])[0])
    print(f"PASS: {args.variant}; merged/reloaded logits equal OOF, mixing agrees, grouped/shuffled/single scores equal", flush=True)


if __name__ == "__main__":
    main()
