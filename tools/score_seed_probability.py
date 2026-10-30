"""Score the already trained seed2026 encoder for the fixed probability mean."""
import argparse
import json

from transformers import AutoTokenizer

from src.common import ROOT
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetic_experiment import MODEL_DIR, prepare, evaluate
from src.encoder_seed_average import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    seed_all(2026)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, 2400, tokenizer)
    data["protocol"] = {**data["protocol"], "training_seed": 2026, "data_selection_seed": 42,
                        "reference": "v12", "probability_weights": [2, 1, 1]}
    folder = ROOT / "experiments/artifacts/encoder_seed_probability_v1"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{args.fold}_epoch1_scores.joblib").exists():
        raise FileExistsError("Preserve completed seed probability scores")
    scorer = EncoderPairScorer(ROOT / "experiments/artifacts/encoder_seed_average_v1" / f"fold{args.fold}_seed2026_merged")
    metrics = evaluate(scorer, data, folder, args.fold)
    print(json.dumps({"note": "raw-seed diagnostics; use probability-mean auditor for decisions", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
