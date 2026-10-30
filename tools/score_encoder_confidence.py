"""Single predeclared 40% neural mix using the unchanged v12 fold models."""
import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from src.augment import augment
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.encoder_phonetics import EncoderPhoneticScorer
from src.experiment import build_testlike
from src.llm_cache import BeamCache
from src.pair_views import PairViewScorer


def score(fold):
    torch.set_num_threads(4)
    tag = "encoder_confidence40_v1"
    output = ROOT / "experiments/artifacts" / tag
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"fold{fold}_epoch1_scores.joblib"
    if path.exists():
        raise FileExistsError(path)
    qwen_dir = ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_runtime_merged"
    if not (qwen_dir / "config.json").exists():
        subprocess.run([sys.executable, "-u", "-m", "tools.check_pair_views_runtime",
                        "--fold", str(fold), "--variant", "evidence_only"], check=True, cwd=ROOT)
    old = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_epoch1_scores.joblib")
    qr = joblib.load(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_epoch1_scores.joblib")
    train = load_train()
    han = train.vessel_name.map(is_hangul).to_numpy()
    hard = augment(train, 2, 1, seed=2026)
    hard["sample_id"] += "_hard"
    frames = {"oof": train, "hard": hard,
              "testlike": build_testlike(SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=han))}
    lookup = {s: {r["sample_id"]: r for r in frame.to_dict("records")} for s, frame in frames.items()}
    cat = load_catalog().to_dict("records")
    beams = BeamCache(allow_generate=False)
    qwen = PairViewScorer(qwen_dir, "evidence")
    encoder = EncoderPhoneticScorer(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_merged")
    sets = {}
    added = 0
    for s, rows in old["sets"].items():
        assert old["sample_ids"][s] == qr["sample_ids"][s]
        result = []
        for key, prior, qs in zip(old["sample_ids"][s], rows, qr["sets"][s]):
            row = lookup[s][key]
            keep = prior["keep"]
            bp = prior["base_scores"][keep]
            bp = bp/bp.sum()
            prune = len(keep) < 2 or bp[0]-bp[1] > 2/3+1e-6
            qscore, nscore = qs["neural_scores"], prior["neural_scores"]
            if is_hangul(row["vessel_name"]) and not prune and prior["bound_pruned"]:
                choices = [cat[i] for i in prior["candidates"][keep]]
                bb = beams([row["vessel_name"]])[0]
                qscore = qwen.score(row, choices, bb)
                nscore = encoder.score(row, choices, bb)
                added += 1
                if added % 50 == 0:
                    print(f"Weight40 fold {fold}: {added} additional rows", flush=True)
            result.append({**prior, "bound_pruned": prune,
                           "neural_scores": nscore, "qwen_scores": qscore,
                           "prior_bound_pruned": prior["bound_pruned"]})
        sets[s] = result
    result = {"sets": sets, "sample_ids": old["sample_ids"],
              "protocol": {**old["protocol"], "reference": "v12", "fusion_weight": .4,
                           "training_change": False, "additional_scored_rows": added}}
    joblib.dump(result, path, compress=3)
    print(f"Saved {path}; {added} additional rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    score(parser.parse_args().fold)
