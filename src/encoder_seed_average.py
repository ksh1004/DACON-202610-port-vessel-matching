"""One fixed extra training seed, averaged with the frozen v12 encoder weights."""
import argparse
import gc
import hashlib
import json
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import ROOT
from src.encoder_pair import EncoderPairScorer
from src.encoder_phonetic_experiment import MODEL_DIR, prepare, evaluate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def fit_seed(groups, tokenizer, merged, adapter):
    seed_all(2026)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=["query", "key", "value"], modules_to_save=["classifier"],
                                            task_type="SEQ_CLS", bias="none"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    order = np.random.default_rng(2026).permutation(len(groups))
    start = time.perf_counter()
    losses = []
    model.train()
    for step, offset in enumerate(range(0, len(order), 2), 1):
        selected = order[offset:offset+2]
        tokens = [part for i in selected for part in groups[i]]
        enc = tokenizer.pad({"input_ids": tokens}, padding=True, pad_to_multiple_of=8, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**enc).logits.reshape(len(selected), 8).float()
            loss = F.cross_entropy(scores, torch.zeros(len(selected), dtype=torch.long, device="cuda"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite seed2026 loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if step % 100 == 0:
            print(f"seed2026 step {step}, loss {np.mean(losses[-100:]):.4f}, elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
            torch.cuda.empty_cache()
    model.save_pretrained(adapter)
    model = model.merge_and_unload().to(dtype=torch.bfloat16).eval()
    model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    del model, optimizer, enc, scores, loss
    gc.collect()
    torch.cuda.empty_cache()
    return {"seed": 2026, "training_groups": len(groups), "epochs": 1,
            "training_seconds": time.perf_counter()-start, "updates": len(losses),
            "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:]))}


def average_models(first, second, output):
    """FP32 arithmetic mean of aligned merged weights, exported in original BF16."""
    a = load_file(str(first / "model.safetensors"), device="cpu")
    b = load_file(str(second / "model.safetensors"), device="cpu")
    assert a.keys() == b.keys()
    result = {}
    different = 0
    for name, left in a.items():
        right = b[name]
        assert left.shape == right.shape and left.dtype == right.dtype
        if left.is_floating_point():
            value = ((left.float()+right.float())*.5).to(left.dtype).contiguous()
        else:
            assert torch.equal(left, right)
            value = left.contiguous()
        assert torch.isfinite(value).all(), name
        different += int(not torch.equal(left, value))
        result[name] = value
    assert different > 0
    output.mkdir(parents=True, exist_ok=True)
    for path in first.iterdir():
        if path.is_file() and path.name != "model.safetensors":
            shutil.copyfile(path, output / path.name)
    save_file(result, str(output / "model.safetensors"), metadata={"format": "pt"})
    del a, b, result
    gc.collect()
    return {"weight_average": [.5, .5], "seeds": [42, 2026], "changed_tensors_vs_seed42": different,
            "averaging_precision": "FP32 then original BF16"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--tag", default="encoder_seed_average_v1")
    args = parser.parse_args()
    seed_all(2026)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, 2400, tokenizer)
    data["protocol"] = {**data["protocol"], "reference": "v12", "training_seed": 2026,
                        "data_selection_seed": 42, "weight_average": [.5, .5],
                        "trainer_sha256": hashlib.sha256((ROOT / "src/encoder_seed_average.py").read_bytes()).hexdigest()}
    folder = ROOT / "experiments/artifacts" / args.tag
    if (folder / f"fold{args.fold}_epoch1_scores.joblib").exists():
        raise FileExistsError("Preserve completed seed experiment")
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"fold{args.fold}_seed2026_merged"
    training = fit_seed(data["groups"], tokenizer, raw, folder / f"fold{args.fold}_adapter")
    merged = folder / f"fold{args.fold}_merged"
    average = average_models(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{args.fold}_merged", raw, merged)
    print(json.dumps(average), flush=True)
    scorer = EncoderPairScorer(merged)
    metrics = evaluate(scorer, data, folder, args.fold)
    (ROOT / "experiments" / f"{args.tag}_fold{args.fold}.json").write_text(
        json.dumps({"protocol": data["protocol"], "training": training, "average": average,
                    "diagnostic_metrics_vs_v11": metrics}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
