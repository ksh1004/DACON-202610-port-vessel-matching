"""Check encoder/Qwen deployed scoring against saved held-out logits."""
import argparse
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np

from src.common import ROOT, load_train, load_catalog, is_hangul
from src.encoder_views import ViewScorer, EncoderViewsEnsemble
from src.experiment import build_testlike
from src.llm_cache import BeamCache
from src.pair_views import mix_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--tag", default="encoder_seed_probability_v1")
    parser.add_argument("--variant", choices=["encoder_only", "equal_ensemble"], default="equal_ensemble")
    args = parser.parse_args()
    folder = ROOT / "experiments/artifacts" / args.tag
    new = joblib.load(folder / f"fold{args.fold}_epoch1_scores.joblib")
    old = joblib.load(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{args.fold}_epoch1_scores.joblib")
    prior = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{args.fold}_epoch1_scores.joblib")
    prior_by_sample = dict(zip(prior["sample_ids"]["testlike"], prior["sets"]["testlike"]))
    qwen_dir = ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{args.fold}_runtime_merged"
    if args.variant == "equal_ensemble" and not (qwen_dir / "config.json").exists():
        subprocess.run([sys.executable, "-u", "-m", "tools.check_pair_views_runtime", "--fold", str(args.fold),
                        "--variant", "evidence_only"], check=True, cwd=ROOT)
    train, catalog = load_train(), load_catalog()
    frame = build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=train.vessel_name.map(is_hangul).to_numpy()))
    lookup = {row["sample_id"]: row for row in frame.to_dict("records")}
    assert new["sample_ids"]["testlike"] == old["sample_ids"]["testlike"]
    selected = []
    for prune, count in ((True, 1), (False, 2)):
        for sample, row, reference in zip(new["sample_ids"]["testlike"], new["sets"]["testlike"], old["sets"]["testlike"]):
            if row["bound_pruned"] == prune and is_hangul(lookup[sample]["vessel_name"]):
                selected.append((lookup[sample], row, reference)); count -= 1
                if not count:
                    break
        assert count == 0
    views = [(ViewScorer(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{args.fold}_merged", "encoder", "phonetic"), 1),
             (ViewScorer(ROOT / "experiments/artifacts/encoder_seed_average_v1" / f"fold{args.fold}_seed2026_merged", "encoder", "phonetic"), 1)]
    if args.variant == "equal_ensemble":
        views.insert(0, (ViewScorer(qwen_dir, "qwen", "evidence"), 2))
    beams = BeamCache(allow_generate=False)
    for row, item, reference in selected:
        if item["bound_pruned"]:
            continue
        choices = [catalog.iloc[j].to_dict() for j in item["candidates"][item["keep"]]]
        for view_index, (scorer, _) in enumerate(views):
            actual = scorer.score(row, choices, beams([row["vessel_name"]])[0])
            expected = reference["neural_scores"] if view_index == 0 else (prior_by_sample[row["sample_id"]]["neural_scores"] if view_index == 1 else item["neural_scores"])
            np.testing.assert_array_equal(actual, expected)

    class Baseline:
        def score_groups(self, blocks):
            return [selected[int(block[0, 0])][1]["base_scores"] for block in blocks]

    model = EncoderViewsEnsemble(Baseline(), views, SimpleNamespace(catalog=catalog))
    rr = [row for row, _, _ in selected]
    ss = [{"cand": item["candidates"], "llm_beams": beams([row["vessel_name"]])[0]} for row, item, _ in selected]
    blocks = [np.array([[i]], dtype=float) for i in range(len(selected))]
    result = model.score_records(rr, ss, blocks)
    for i, (record, item, reference) in enumerate(selected):
        expected = int(np.argmax(item["base_scores"]))
        if not item["bound_pruned"]:
            keep = item["keep"]; bp = item["base_scores"][keep]
            scores = [reference["neural_scores"], prior_by_sample[record["sample_id"]]["neural_scores"], item["neural_scores"]]
            expected = int(keep[int(np.argmax(mix_views(bp/bp.sum(), scores, [2, 1, 1])))])
        assert int(np.argmax(result[i])) == expected
    order = [2, 0, 1]
    shuffled = model.score_records([rr[i] for i in order], [ss[i] for i in order], [blocks[i] for i in order])
    for pos, i in enumerate(order):
        np.testing.assert_array_equal(result[i], shuffled[pos])
        np.testing.assert_array_equal(result[i], model.score_records([rr[i]], [ss[i]], [blocks[i]])[0])
    print(f"PASS: {args.variant}; encoder/Qwen logits match saved OOF, deployed mixing agrees, grouped/shuffled/single scores identical", flush=True)


if __name__ == "__main__":
    main()
