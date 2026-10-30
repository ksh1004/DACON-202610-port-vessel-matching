"""Check serialized inference against held-out neural scores and row permutations."""
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import ROOT, load_train, load_catalog, is_hangul
from src.ensemble import normalize
from src.experiment import build_testlike
from src.neural_pair import PairScorer, NeuralEnsemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    folder = ROOT / "experiments" / "artifacts" / "neural_v8_cv"
    merged = folder / f"fold{args.fold}_merged"
    base_dir = "D:/KSH/hf_models/Qwen__Qwen3-Reranker-0.6B"
    if not (merged / "config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, folder / f"fold{args.fold}_adapter").merge_and_unload().eval()
        model.save_pretrained(merged)
        AutoTokenizer.from_pretrained(base_dir, local_files_only=True, padding_side="left").save_pretrained(merged)
        del base, model
        gc.collect()
        torch.cuda.empty_cache()
    data = joblib.load(folder / f"fold{args.fold}_epoch1_scores.joblib")
    train, catalog = load_train(), load_catalog()
    tl = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=train.vessel_name.map(is_hangul).to_numpy()))
    rows = {r["sample_id"]: r for r in tl.to_dict("records")}
    selected = []
    for pruned, count in ((True, 1), (False, 2)):
        for sample, item in zip(data["sample_ids"]["testlike"], data["sets"]["testlike"]):
            if item["bound_pruned"] == pruned and is_hangul(rows[sample]["vessel_name"]):
                selected.append((rows[sample], item))
                count -= 1
                if not count:
                    break
        assert count == 0
    scorer = PairScorer(merged)
    for row, item in selected:
        if item["bound_pruned"]:
            continue
        choices = [catalog.iloc[i].to_dict() for i in item["candidates"][item["keep"]]]
        actual = scorer.score(row, choices)
        np.testing.assert_array_equal(actual, item["neural_scores"])
    class Baseline:
        def score_groups(self, blocks):
            return [selected[int(b[0, 0])][1]["base_scores"] for b in blocks]
    ensemble = NeuralEnsemble(Baseline(), scorer, SimpleNamespace(catalog=catalog))
    rr = [r for r, _ in selected]
    ss = [{"cand": item["candidates"]} for _, item in selected]
    blocks = [np.array([[i]], dtype=float) for i in range(len(selected))]
    output = ensemble.score_records(rr, ss, blocks)
    for j, (_, item) in enumerate(selected):
        if item["bound_pruned"]:
            expected = int(np.argmax(item["base_scores"]))
        else:
            keep = item["keep"]
            p = item["base_scores"][keep]
            mixed = .75*p/p.sum()+.25*normalize(item["neural_scores"], "prob")
            expected = int(keep[int(np.argmax(mixed))])
        assert int(np.argmax(output[j])) == expected
    permutation = [2, 0, 1]
    again = ensemble.score_records([rr[i] for i in permutation], [ss[i] for i in permutation], [blocks[i] for i in permutation])
    for pos, i in enumerate(permutation):
        np.testing.assert_array_equal(output[i], again[pos])
        np.testing.assert_array_equal(output[i], ensemble.score_records([rr[i]], [ss[i]], [blocks[i]])[0])
    print("PASS: merged/serialized neural logits exactly match OOF; production blending matches; grouping/permutation/single-row scores identical", flush=True)


if __name__ == "__main__":
    main()
