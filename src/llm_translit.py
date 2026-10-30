"""LLM 음역: 한글 선박명 → 영문 원표기 후보(N-best) 생성.

공개 범용 사전학습 가중치만 사용한다 (대회 규칙). 한 행씩 독립적으로 생성할 수 있다.

    python -m src.llm_translit --model D:/KSH/hf_models/Qwen__Qwen3-1.7B     # 벤치마크
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SYSTEM_PROMPT = (
    "You restore the original Latin-alphabet spelling of ship names that were written in Korean Hangul. "
    "The names come from many languages (English, Korean, Chinese pinyin, Japanese, Greek, Dutch ...). "
    "Korean particles such as '제N' or a trailing '호' mean 'No. N' or nothing. "
    "Answer with the ship name only, in English letters."
)

# 평가에서 제외되는 few-shot 예시 (train 한글 선박에서 다양한 유형을 고름)
FEW_SHOT = [
    ("모닝 메나드", "Morning Menad"),
    ("케이엠티씨 하이퐁", "KMTC Haiphong"),
    ("흥아 사라", "Heung-A Sarah"),
    ("바오루이링", "Bao Rui Ling"),
    ("카시마 마루 10", "Kashima Maru No. 10"),
    ("제2흥아 파이오니어호", "No.2 Heung-A Pioneer"),
    ("엠에스샤론", "MS Sharon"),
    ("헤로익에이스", "Heroic Ace"),
    ("거영썬5", "Keoyoung Sun 5"),
    ("돌피즌그라츠", "Dolfijngracht"),
    ("샨동 푸 유안", "Shandong Fu Yuan"),
    ("프런트 디스커버리", "Front Discovery"),
]


def _clean(text: str) -> str:
    text = text.strip().split("\n")[0]
    text = re.sub(r"^(answer|ship name)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip(" \"'`*.")


class Transliterator:
    def __init__(self, model_dir: str | Path, device: str = "cuda", dtype=torch.bfloat16) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_dir = str(model_dir)
        config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        self.seq2seq = bool(getattr(config, "is_encoder_decoder", False))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        cls = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        self.model = cls.from_pretrained(self.model_dir, torch_dtype=dtype, local_files_only=True).to(device).eval()
        self.device = device
        if not self.seq2seq:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        # 파인튜닝한 모델(translit.json: prompt=plain)은 few-shot 없이 학습 때와 같은 짧은 프롬프트를 쓴다
        marker = Path(self.model_dir) / "translit.json"
        self.plain = marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("prompt") == "plain"
        # 채팅 템플릿 문자열에 BOS 가 이미 있으므로, 학습 때(tokenize=True)와 같게 특수 토큰을 다시 붙이지 않는다
        self.add_special_tokens = not self.plain

    def _prompt(self, name: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ko, en in ([] if self.plain else FEW_SHOT):
            messages += [{"role": "user", "content": ko}, {"role": "assistant", "content": en}]
        messages.append({"role": "user", "content": name})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    @torch.inference_mode()
    def generate(self, names: list[str], n_best: int = 4, batch_size: int = 16, max_new_tokens: int = 24) -> list[list[str]]:
        out: list[list[str]] = []
        for start in range(0, len(names), batch_size):
            chunk = names[start : start + batch_size]
            texts = chunk if self.seq2seq else [self._prompt(n) for n in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 add_special_tokens=self.add_special_tokens or self.seq2seq).to(self.device)
            enc.pop("token_type_ids", None)  # 일부 토크나이저(Midm)가 내보내지만 생성 모델은 받지 않음
            gen = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, num_beams=n_best, num_return_sequences=n_best,
                do_sample=False, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id,
            )
            if not self.seq2seq:
                gen = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(chunk)):
                beams = [_clean(t) for t in decoded[i * n_best : (i + 1) * n_best]]
                out.append(list(dict.fromkeys(b for b in beams if b)))  # 순서 유지 중복 제거
        return out


def benchmark(model_dir: str, n_best: int = 4, batch_size: int = 16) -> dict:
    from src.common import compact_name, is_hangul, load_catalog, load_train
    from src.retrieval import CatalogRetriever, query_key

    train, catalog = load_train(), load_catalog()
    retriever = CatalogRetriever(catalog)
    hangul = train[train["vessel_name"].map(is_hangul)].drop_duplicates("call_sign")
    shot_names = {ko for ko, _ in FEW_SHOT}
    hangul = hangul[~hangul["vessel_name"].isin(shot_names)].reset_index(drop=True)
    truth = hangul["target_vessel_id"].map(retriever.id_to_index).to_numpy()
    truth_compact = [retriever.compact[t] for t in truth]

    t0 = time.perf_counter()
    model = Transliterator(model_dir)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    beams = model.generate(hangul["vessel_name"].tolist(), n_best=n_best, batch_size=batch_size)
    gen_s = time.perf_counter() - t0

    exact1 = np.mean([bool(b) and compact_name(b[0]) == tc for b, tc in zip(beams, truth_compact)])
    exact_any = np.mean([any(compact_name(x) == tc for x in b) for b, tc in zip(beams, truth_compact)])
    # 생성된 영문으로 카탈로그 검색 (이름만, 톤수 없이): 1번 빔의 1위 / 모든 빔의 상위 10위
    first = pd.Series([b[0] if b else "" for b in beams])
    top_first, _ = retriever.search_tfidf_word(first, 10)
    r1 = np.mean([len(c) > 0 and c[0] == t for c, t in zip(top_first, truth)])
    flat = [(i, x) for i, b in enumerate(beams) for x in b]
    tops, _ = retriever.search_tfidf_word(pd.Series([x for _, x in flat]), 10)
    hit10 = np.zeros(len(beams), dtype=bool)
    for (i, _), c in zip(flat, tops):
        hit10[i] |= truth[i] in c
    # 비교 기준: 발음 키 검색 (규칙 기반)
    keys = pd.Series([query_key(n).replace(" ", "") for n in hangul["vessel_name"]])
    key_top, _ = retriever._tfidf_topk(retriever.tfidf_key, retriever.matrix_key, keys.tolist(), 10)
    key_r1 = np.mean([len(c) > 0 and c[0] == t for c, t in zip(key_top, truth)])
    key_r10 = np.mean([t in c for c, t in zip(key_top, truth)])
    # 보완성: 발음 키 상위 10위와 LLM 상위 10위를 합쳤을 때
    union_r10 = np.mean([hit10[i] or truth[i] in key_top[i] for i in range(len(truth))])
    # LLM 출력을 다시 발음 키로 바꿔 검색 (철자는 틀려도 발음이 맞는 경우)
    from src.hangul import english_key
    llm_keys = [english_key(b[0]).replace(" ", "") if b else "" for b in beams]
    llm_key_top, _ = retriever._tfidf_topk(retriever.tfidf_key, retriever.matrix_key, llm_keys, 10)
    llm_key_r1 = np.mean([len(c) > 0 and c[0] == t for c, t in zip(llm_key_top, truth)])

    examples = [f"{n} -> {b[:2]} (정답 {retriever.catalog['vessel_name'][t]})"
                for n, b, t in list(zip(hangul["vessel_name"], beams, truth))[:12]]
    return {
        "model": Path(model_dir).name, "n_vessels": len(hangul), "load_s": round(load_s, 1),
        "ms_per_name": round(gen_s / len(hangul) * 1000, 1), "exact_top1": round(exact1, 4),
        "exact_any": round(exact_any, 4), "search_R@1": round(r1, 4), "search_R@10_any_beam": round(hit10.mean(), 4),
        "key_baseline_R@1": round(key_r1, 4), "key_baseline_R@10": round(key_r10, 4),
        "union_R@10": round(union_r10, 4), "llm_output_key_R@1": round(llm_key_r1, 4),
        "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-best", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = benchmark(args.model, args.n_best, args.batch_size)
    out = Path(__file__).resolve().parents[1] / "experiments" / "llm_benchmark.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
