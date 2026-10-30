"""Export a fixed-budget data-recipe candidate while preserving all v12 assets."""
import argparse
import hashlib
import json
import shutil

from transformers import AutoTokenizer

from src.common import ROOT
from src.encoder_extended_experiment import seed_training, fit
from src.encoder_phonetic_experiment import MODEL_DIR
from src.focused_encoder_experiment import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["focused", "restore"], required=True)
    parser.add_argument("--out", default="experiments/artifacts/ranker_v13_data")
    args = parser.parse_args()
    source = ROOT / "experiments/artifacts/ranker_v12_encoder"
    spec = json.loads((source / "encoder_views.json").read_text(encoding="utf-8"))
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == expected, name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(None, args.mode, tokenizer)
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_training()
    training = fit(data["groups"], tokenizer, out / "encoder", out / "training_adapter")
    for name in ("LICENSE", "MODEL_CARD.md"):
        shutil.copyfile(source / "encoder" / name, out / "encoder" / name)
    (out / "encoder/NOTICE").write_text(
        "BAAI/bge-reranker-v2-m3, Apache-2.0.\n"
        f"Modified: merged LoRA/classifier training on official data, recipe {args.mode}.\n"
        "Fixed 2400 groups, one epoch; row-local names, metadata, phonetic keys, generated Latin spellings.\n",
        encoding="utf-8")
    ensemble = json.loads((source / "ensemble.json").read_text(encoding="utf-8"))
    for name in ["ensemble.json", "mappers.joblib"]+[m["file"] for m in ensemble["members"]]:
        shutil.copyfile(source / name, out / name)
    shutil.copytree(source / "qwen", out / "qwen", dirs_exist_ok=True)
    spec.update(variant=args.mode, reference="v12", data_recipe=data["protocol"],
                training_group_fingerprint=data["fingerprint"], training_groups=2400,
                training_minutes=training["training_seconds"]/60, epochs=1, seed=42,
                training_code_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                     ("src/focused_encoder_experiment.py", "src/encoder_extended_experiment.py", "tools/train_focused_final.py")})
    (out / "encoder_views.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "protocol": data["protocol"], "training": training}, indent=2), flush=True)


if __name__ == "__main__":
    main()
