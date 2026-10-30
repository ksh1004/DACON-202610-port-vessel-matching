"""Controlled extension: use all eligible training groups at the v12 settings."""
import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import ROOT
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetic_experiment import MODEL_DIR, prepare, evaluate


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)


def fit(groups, tokenizer, merged, adapter):
    """One epoch, same architecture/optimizer/order recipe as v12."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["query", "key", "value"], modules_to_save=["classifier"],
        task_type="SEQ_CLS", bias="none"))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    ordering = np.random.default_rng(42).permutation(len(groups))
    start = time.perf_counter()
    losses = []
    model.train()
    for step, offset in enumerate(range(0, len(ordering), 2), 1):
        ids = ordering[offset:offset+2]
        tokens = [tokens for i in ids for tokens in groups[i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True,
                            pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(ids), 8).float()
            loss = F.cross_entropy(scores, torch.zeros(len(ids), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite extended encoder loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"extended encoder step {step}/{(len(groups)+1)//2}, "
                  f"loss {np.mean(losses[-100:]):.4f}, "
                  f"elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            torch.cuda.empty_cache()
    model.save_pretrained(adapter)
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    del model, optimizer, enc, scores, loss
    gc.collect()
    torch.cuda.empty_cache()
    return {"training_seconds": time.perf_counter()-start,
            "training_groups": len(groups), "updates": len(losses),
            "loss_first100": float(np.mean(losses[:100])),
            "loss_last100": float(np.mean(losses[-100:])),
            "precision": "merged BF16, saved then reloaded before evaluation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--tag", default="encoder_bge_full_v1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    seed_training()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, 100000, tokenizer)
    data["protocol"] = {**data["protocol"], "reference": "v12",
                        "training_recipe": "all eligible original plus four seed7 augmentations, one epoch",
                        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(data["protocol"]), flush=True)
    if args.prepare_only:
        return
    folder = ROOT / "experiments/artifacts" / args.tag
    score = folder / f"fold{args.fold}_epoch1_scores.joblib"
    if score.exists():
        raise FileExistsError(f"Preserve completed experiment: {score}")
    folder.mkdir(parents=True, exist_ok=True)
    merged = folder / f"fold{args.fold}_merged"
    training = fit(data["groups"], tokenizer, merged, folder / f"fold{args.fold}_adapter")
    scorer = EncoderPairScorer(merged)
    metrics = evaluate(scorer, data, folder, args.fold)
    report = {"protocol": data["protocol"], "training": training,
              "diagnostic_metrics_vs_v11": metrics,
              "note": "Decision must use audit_encoder_extension against v12, not these v11 diagnostics."}
    (ROOT / "experiments" / f"{args.tag}_fold{args.fold}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
