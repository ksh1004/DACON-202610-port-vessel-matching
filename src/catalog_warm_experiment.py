"""Catalog-name adaptation, followed by the unchanged real v12 ranking groups."""
import argparse
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import ROOT
from src.encoder_extended_experiment import seed_training
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetic_experiment import MODEL_DIR, prepare, evaluate
from tools.prepare_catalog_warmup import prepare as prepare_warmup


TAG = "encoder_catalog_warm_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG


def fit_from(groups, tokenizer, initial, merged, adapter, lr):
    seed_training()
    model = AutoModelForSequenceClassification.from_pretrained(
        initial, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05, target_modules=["query", "key", "value"],
        modules_to_save=["classifier"], task_type="SEQ_CLS", bias="none"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=.01)
    ordering = np.random.default_rng(42).permutation(len(groups))
    losses, start = [], time.perf_counter()
    model.train()
    for step, offset in enumerate(range(0, len(ordering), 2), 1):
        ids = ordering[offset:offset+2]
        tokens = [t for i in ids for t in groups[i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True,
                            pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(ids), 8).float()
            loss = F.cross_entropy(scores, torch.zeros(len(ids), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite catalog adaptation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"catalog adaptation {step}/{(len(groups)+1)//2}, "
                  f"loss {np.mean(losses[-100:]):.4f}, elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            torch.cuda.empty_cache()
    model.save_pretrained(adapter)
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    del model, optimizer, enc, scores, loss
    gc.collect()
    torch.cuda.empty_cache()
    return {"training_seconds": time.perf_counter()-start, "training_groups": len(groups),
            "updates": len(losses), "learning_rate": lr,
            "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
            "precision": "BF16 merge, save and reload", "initial_model": str(initial)}


def check_protocol():
    path = ROOT / "experiments/catalog_warm_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in protocol["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["warmup", "fold"], required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    registered = check_protocol()
    FOLDER.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    warm = FOLDER / "warm_merged"
    if args.stage == "warmup":
        report_path = ROOT / "experiments/catalog_warm_training.json"
        if warm.exists() or report_path.exists():
            raise FileExistsError("Preserve the completed synthetic warmup")
        data = prepare_warmup(tokenizer)
        training = fit_from(data["groups"], tokenizer, MODEL_DIR, warm, FOLDER / "warm_adapter", 5e-5)
        report = {"training": training, "data_fingerprint": data["fingerprint"], "data": data["protocol"]}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        k = args.fold
        assert k in range(5) and warm.exists()
        if (FOLDER / f"fold{k}_epoch1_scores.joblib").exists():
            raise FileExistsError("Preserve completed fold")
        data = prepare(k, 2400, tokenizer)
        data["protocol"] = {**data["protocol"], "tag": TAG, "initialization": "novel catalog name-only warmup",
                            "real_data_fingerprint": data["fingerprint"], "registered": registered,
                            "warm_training_sha256": hashlib.sha256((ROOT / "experiments/catalog_warm_training.json").read_bytes()).hexdigest()}
        merged = FOLDER / f"fold{k}_merged"
        training = fit_from(data["groups"], tokenizer, warm, merged, FOLDER / f"fold{k}_adapter", 1e-4)
        scorer = EncoderPairScorer(merged)
        metrics = evaluate(scorer, data, FOLDER, k)
        report = {"protocol": data["protocol"], "training": training, "diagnostic_vs_v11": metrics}
        (ROOT / "experiments" / f"{TAG}_fold{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
