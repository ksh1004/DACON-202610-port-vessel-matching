"""Kanana LoRA 파인튜닝: 한글 선박명(노이즈 포함) → 카탈로그 영문명.

공식 train 데이터만 사용한다. 학습 후 LoRA 를 원본 가중치에 병합해 저장하므로 추론 서버에는 peft 가 필요 없다.
OOF 평가를 위해 fold 의 검증 선박을 빼고 학습할 수 있다.

    python -m src.llm_finetune --holdout-fold 0 --out D:/KSH/hf_models/kanana_lora_f0 --eval
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.augment import augment
from src.common import compact_name, is_hangul, load_catalog, load_train, normalize_name
from src.llm_translit import SYSTEM_PROMPT
from src.validate import load_folds

BASE_MODEL = Path("D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505")


def build_pairs(train: pd.DataFrame, catalog: pd.DataFrame, keep_rows: np.ndarray) -> list[tuple[str, str]]:
    """keep_rows 위치의 train 행과, 그 행에서 만든 증강 행으로 (한글, 영문) 쌍을 만든다."""
    names = catalog.set_index("vessel_id")["vessel_name"]
    pairs = []
    for pos in keep_rows:
        row = train.iloc[pos]
        if is_hangul(row["vessel_name"]):
            pairs.append((row["vessel_name"], names[row["target_vessel_id"]]))
    aug = augment(train, 4, 1, 7)
    aug = aug[aug["origin"].isin(keep_rows)]
    for name, origin in zip(aug["vessel_name"], aug["origin"]):
        original = train["vessel_name"].iloc[origin]
        if not is_hangul(name):
            continue
        # 단어가 생략된 행은 없는 단어를 지어내도록 가르치게 되므로 제외
        if len(normalize_name(name).split()) < len(normalize_name(original).split()):
            continue
        pairs.append((name, names[train["target_vessel_id"].iloc[origin]]))
    return list(dict.fromkeys(pairs))


def plain_messages(name: str, answer: str | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": name}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def encode(tokenizer, ko: str, en: str) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(plain_messages(ko), tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(plain_messages(ko, en), tokenize=True, add_generation_prompt=False)
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template prefix mismatch")
    labels = [-100] * len(prompt) + full[len(prompt):]
    return full, labels


def train_lora(pairs: list[tuple[str, str]], out_dir: Path, epochs: int = 3, lr: float = 2e-4,
               batch_size: int = 16, rank: int = 16, seed: int = 42) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, local_files_only=True).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    data = [encode(tokenizer, ko, en) for ko, en in pairs]
    steps = math.ceil(len(data) / batch_size) * epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.05), steps)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.train()
    t0, step = time.perf_counter(), 0
    for epoch in range(epochs):
        order = np.random.permutation(len(data))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [data[i] for i in order[start : start + batch_size]]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in batch]).cuda()
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch]).cuda()
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch]).cuda()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            step += 1
        print(f"epoch {epoch + 1}/{epochs}  loss {total / math.ceil(len(order) / batch_size):.4f}  "
              f"({time.perf_counter() - t0:.0f}초)", flush=True)

    merged = model.merge_and_unload()
    merged.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="1GB")  # 저장 시 RAM 사용량 제한
    tokenizer.save_pretrained(out_dir)
    for extra in ("generation_config.json", "LICENSE"):
        if (BASE_MODEL / extra).exists():
            shutil.copy(BASE_MODEL / extra, out_dir / extra)
    (out_dir / "translit.json").write_text(json.dumps({"prompt": "plain", "base": BASE_MODEL.name,
                                                       "pairs": len(pairs), "epochs": epochs, "rank": rank}),
                                           encoding="utf-8")


def evaluate(model_dir: Path, names: list[str], truth_ids: list[str]) -> dict:
    from src.llm_translit import Transliterator
    from src.retrieval import CatalogRetriever

    retriever = CatalogRetriever(load_catalog())
    truth = np.array([retriever.id_to_index[t] for t in truth_ids])
    model = Transliterator(model_dir)
    t0 = time.perf_counter()
    beams = model.generate(names, n_best=4, batch_size=1)
    ms = (time.perf_counter() - t0) / len(names) * 1000
    truth_compact = [retriever.compact[t] for t in truth]
    exact1 = np.mean([bool(b) and compact_name(b[0]) == tc for b, tc in zip(beams, truth_compact)])
    exact_any = np.mean([any(compact_name(x) == tc for x in b) for b, tc in zip(beams, truth_compact)])
    first, _ = retriever.search_tfidf_word(pd.Series([b[0] if b else "" for b in beams]), 10)
    r1 = np.mean([len(c) > 0 and c[0] == t for c, t in zip(first, truth)])
    flat = [(i, x) for i, b in enumerate(beams) for x in b]
    tops, _ = retriever.search_tfidf_word(pd.Series([x for _, x in flat]), 10)
    hit = np.zeros(len(names), dtype=bool)
    for (i, _), c in zip(flat, tops):
        hit[i] |= truth[i] in c
    return {"model": Path(model_dir).name, "n": len(names), "ms_per_name": round(ms, 1),
            "exact_top1": round(exact1, 4), "exact_any": round(exact_any, 4),
            "search_R@1": round(r1, 4), "search_R@10": round(hit.mean(), 4)}


def eval_names(train: pd.DataFrame, fold_mask: np.ndarray, noisy: bool) -> tuple[list[str], list[str]]:
    """평가용 한글 이름: 검증 fold 선박의 원본 이름 또는 노이즈를 입힌 이름 (hard 세트와 같은 방식)."""
    rows = train[fold_mask & train["vessel_name"].map(is_hangul).to_numpy()]
    if not noisy:
        rows = rows.drop_duplicates("call_sign")
        return rows["vessel_name"].tolist(), rows["target_vessel_id"].tolist()
    from src.experiment import HARD_SEED
    hard = augment(rows.reset_index(drop=True), 1, 0, HARD_SEED)
    hard = hard[hard["vessel_name"].map(is_hangul)]
    return hard["vessel_name"].tolist(), rows["target_vessel_id"].iloc[hard["origin"]].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-fold", type=int, default=-1, help="이 fold 의 선박을 학습에서 제외 (-1 = 전체 학습)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--eval", action="store_true", help="holdout fold 에서 few-shot 원본 모델과 비교")
    args = parser.parse_args()

    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    keep = np.flatnonzero(folds != args.holdout_fold)
    if not list(args.out.glob("*.safetensors")):
        pairs = build_pairs(train, catalog, keep)
        print(f"학습 쌍 {len(pairs)}개 (holdout fold {args.holdout_fold})", flush=True)
        train_lora(pairs, args.out, epochs=args.epochs, rank=args.rank)
        torch.cuda.empty_cache()
    if args.eval and args.holdout_fold >= 0:
        mask = folds == args.holdout_fold
        for noisy in (False, True):
            names, truth = eval_names(train, mask, noisy)
            for model_dir in (args.out, BASE_MODEL):
                result = evaluate(model_dir, names, truth)
                result["set"] = "noisy" if noisy else "clean"
                print(json.dumps(result, ensure_ascii=False), flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
