"""Export the selected catalog-adapted encoder with the frozen v12 runtime."""
import hashlib
import json
import shutil

import joblib
from transformers import AutoTokenizer

from src.common import ROOT
from src.catalog_warm_experiment import FOLDER, check_protocol, fit_from
from src.encoder_phonetic_experiment import MODEL_DIR


def main():
    check_protocol()
    selected = json.loads((ROOT / "experiments/catalog_warm_selection.json").read_text(encoding="utf-8"))["selected"]
    assert selected and selected["passed"] and all(selected["checks"].values())
    source = ROOT / "experiments/artifacts/ranker_v12_encoder"
    out = ROOT / "experiments/artifacts/ranker_v13_catalog"
    if out.exists():
        raise FileExistsError("Preserve existing final model")
    spec = json.loads((source / "encoder_views.json").read_text(encoding="utf-8"))
    for name, expected in spec["source_hashes"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == expected, name
    data = joblib.load(ROOT / "experiments/cache/final_encoder_groups_e0c97ee269b9.joblib")
    assert data["fingerprint"] == "e0c97ee269b9b4dee49484b0ce715b6403ed9e3101fe8645caf41bc3472136a8"
    assert len(data["groups"]) == 2400
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    out.mkdir(parents=True)
    training = fit_from(data["groups"], tokenizer, FOLDER / "warm_merged", out / "encoder",
                        out / "training_adapter", 1e-4)
    for name in ("LICENSE", "MODEL_CARD.md"):
        shutil.copyfile(source / "encoder" / name, out / "encoder" / name)
    (out / "encoder/NOTICE").write_text(
        "BAAI/bge-reranker-v2-m3, Apache-2.0.\n"
        "Modified: merged LoRA/classifier on synthetic phonetic transformations of novel official catalog names,\n"
        "then the original 2400 real v12 groups. No runtime alias or identity lookup.\n", encoding="utf-8")
    ensemble = json.loads((source / "ensemble.json").read_text(encoding="utf-8"))
    for name in ["ensemble.json", "mappers.joblib"]+[m["file"] for m in ensemble["members"]]:
        shutil.copyfile(source / name, out / name)
    shutil.copytree(source / "qwen", out / "qwen")
    spec.update(variant="catalog_warmup", reference="v12", training_group_fingerprint=data["fingerprint"],
                training_groups=2400, training_minutes=training["training_seconds"]/60, epochs=1, seed=42,
                warmup=json.loads((ROOT / "experiments/catalog_warm_training.json").read_text(encoding="utf-8")),
                training_code_hashes={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                    ("src/catalog_warm_experiment.py", "tools/prepare_catalog_warmup.py", "tools/train_catalog_final.py")})
    (out / "encoder_views.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "training": training}, indent=2), flush=True)


if __name__ == "__main__":
    main()
