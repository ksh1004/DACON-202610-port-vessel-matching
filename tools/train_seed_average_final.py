"""Fit exactly the second seed on the original full-data v12 groups."""
import argparse
import hashlib
import json
import shutil

import joblib
from transformers import AutoTokenizer

from src.common import ROOT
from src.encoder_phonetic_experiment import MODEL_DIR
from src.encoder_seed_average import fit_seed, average_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/artifacts/ranker_v13_followup")
    parser.add_argument("--variant", choices=["seed_average", "seed_probability"], default="seed_average")
    args = parser.parse_args()
    original = ROOT / "experiments/artifacts/ranker_v12_encoder"
    spec = json.loads((original / "encoder_views.json").read_text(encoding="utf-8"))
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == expected, name
    for name, expected in spec["data_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    fingerprint = spec["training_group_fingerprint"]
    cache = ROOT / "experiments/cache" / f"final_encoder_groups_{fingerprint[:12]}.joblib"
    training = joblib.load(cache)
    assert training["fingerprint"] == fingerprint
    assert len(training["groups"]) == spec["training_groups"] == 2400
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    raw = out / "encoder_seed2026"
    result = fit_seed(training["groups"], tokenizer, raw, out / "training_adapter")
    if args.variant == "seed_average":
        average = average_models(original / "encoder", raw, out / "encoder")
        views = [{"dir": "qwen", "kind": "qwen", "recipe": "evidence", "weight": 1.0},
                 {"dir": "encoder", "kind": "encoder", "recipe": "phonetic", "weight": 1.0}]
    else:
        average = {"probability_weights": [2, 1, 1], "seeds": [42, 2026]}
        shutil.copytree(original / "encoder", out / "encoder_seed42", dirs_exist_ok=True)
        views = [{"dir": "qwen", "kind": "qwen", "recipe": "evidence", "weight": 2.0},
                 {"dir": "encoder_seed42", "kind": "encoder", "recipe": "phonetic", "weight": 1.0},
                 {"dir": "encoder_seed2026", "kind": "encoder", "recipe": "phonetic", "weight": 1.0}]
    for directory in ([out / "encoder"] if args.variant == "seed_average" else [raw]):
        for name in ("LICENSE", "MODEL_CARD.md"):
            shutil.copyfile(original / "encoder" / name, directory / name)
        (directory / "NOTICE").write_text(
            "BAAI/bge-reranker-v2-m3, Apache-2.0.\n"
            "Modified: LoRA/classifier training on the same 2400 official DACON groups as v12.\n"
            f"Variant: {args.variant}; seeds 42 and 2026.\n", encoding="utf-8")
    ensemble = json.loads((original / "ensemble.json").read_text(encoding="utf-8"))
    for name in ["ensemble.json", "mappers.joblib"]+[m["file"] for m in ensemble["members"]]:
        shutil.copyfile(original / name, out / name)
    shutil.copytree(original / "qwen", out / "qwen", dirs_exist_ok=True)
    spec.update(views=views, variant=args.variant, seeds=[42, 2026], average=average,
                second_seed_training=result, weight=.25, reference="v12",
                training_code_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in ("src/encoder_seed_average.py", "tools/train_seed_average_final.py")})
    (out / "encoder_views.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved {args.variant}: {out}", flush=True)


if __name__ == "__main__":
    main()
