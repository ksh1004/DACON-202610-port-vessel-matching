"""LLM 역방향 음역: 카탈로그 영문 선박명 → 한글 표기 (few-shot, 공개 Kanana 원본 가중치).

test 한글 이름의 오타(음절 삭제·중복·순서 바뀜)는 한글 공간에서 생기므로, 카탈로그 이름을
미리 한글로 바꿔 두면 한글끼리 직접 비교할 수 있다. 카탈로그에만 의존하므로 행 독립성과 무관하다.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

SYSTEM_PROMPT = (
    "You write ship names in Korean Hangul the way Korean port records do. "
    "The names come from many languages (English, Korean romanization, Chinese pinyin, Japanese, Greek, Dutch ...). "
    "'No. N' becomes '제N' or a number. Answer with the Hangul name only."
)

FEW_SHOT = [
    ("Morning Menad", "모닝 메나드"),
    ("KMTC Haiphong", "케이엠티씨 하이퐁"),
    ("Heung-A Sarah", "흥아 사라"),
    ("Bao Rui Ling", "바오루이링"),
    ("Kashima Maru No. 10", "카시마 마루 10"),
    ("No.2 Heung-A Pioneer", "제2흥아 파이오니어호"),
    ("MS Sharon", "엠에스샤론"),
    ("Heroic Ace", "헤로익에이스"),
    ("Keoyoung Sun 5", "거영썬5"),
    ("Dolfijngracht", "돌피즌그라츠"),
    ("Shandong Fu Yuan", "샨동 푸 유안"),
    ("Front Discovery", "프런트 디스커버리"),
]


class KoTransliterator:
    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16,
                                                          local_files_only=True).to(device).eval()
        self.device = device

    def _prompt(self, name: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for en, ko in FEW_SHOT:
            msgs += [{"role": "user", "content": en}, {"role": "assistant", "content": ko}]
        msgs.append({"role": "user", "content": name})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, names: list[str], n_best: int = 1, batch_size: int = 64, max_new_tokens: int = 24) -> list[list[str]]:
        out = []
        for s in range(0, len(names), batch_size):
            chunk = names[s:s + batch_size]
            enc = self.tok([self._prompt(n) for n in chunk], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            enc.pop("token_type_ids", None)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=n_best,
                                      num_return_sequences=n_best, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            dec = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i in range(len(chunk)):
                beams = [re.sub(r"\s+", " ", t.strip().split("\n")[0]).strip(" \"'`*.") for t in dec[i * n_best:(i + 1) * n_best]]
                out.append(list(dict.fromkeys(b for b in beams if b)))
        return out
