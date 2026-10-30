"""Symphonym 학생 모델(문자 단위 음성 임베딩)의 최소 재구현: 한글·라틴 이름 → 128차원 단위 벡터.

원 모델: Gadd, "Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching" (arXiv 2601.06932),
가중치 docuracy/symphonym-v8 (CC BY 4.0), 원 코드 BSD 3-Clause (University of Pittsburgh).
내려받은 코드는 실행하지 않고, 구조를 같은 모양으로 다시 작성해 가중치(safetensors)와 어휘(JSON)만 읽는다.
이 대회에 필요한 라틴·한글 입력만 다룬다 (그 밖의 문자는 OTHER 로 처리).

입력 하나씩 독립으로 임베딩할 수 있으므로 행 독립 원칙에 맞는다.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, UNK_ID, SPACE_ID = 0, 1, 2

# 한글 음절 → 자모 (초성·중성·종성), 원 토크나이저와 같은 표
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
_LATIN = [(0x0000, 0x024F), (0x0250, 0x02AF), (0x1D00, 0x1D7F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F),
          (0xA720, 0xA7FF), (0xAB30, 0xAB6F)]
_HANGUL = [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xD7B0, 0xD7FF)]


def _in(ranges, cp: int) -> bool:
    return any(a <= cp <= b for a, b in ranges)


def detect_script(text: str) -> str:
    """알파벳 문자만 세어 가장 많은 문자 체계 (동점이면 먼저 나온 쪽)."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        script = "HANGUL" if _in(_HANGUL, cp) else "LATIN" if _in(_LATIN, cp) else "OTHER"
        counts[script] = counts.get(script, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "OTHER"


def _decompose(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code <= 0xD7A3 - 0xAC00:
            out += [_CHO[code // 588], _JUNG[(code % 588) // 28], _JONG[code % 28]]
        else:
            out.append(ch)
    return "".join(out)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim, self.scale = heads, dim // heads, math.sqrt(dim // heads)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, mask):
        b, length, dim = x.shape
        split = lambda t: t.view(b, length, self.heads, self.head_dim).transpose(1, 2)  # noqa: E731
        scores = split(self.q_proj(x)) @ split(self.k_proj(x)).transpose(-2, -1) / self.scale
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        out = (F.softmax(scores, dim=-1) @ split(self.v_proj(x))).transpose(1, 2).contiguous().view(b, length, dim)
        return self.out_proj(out)


class _Encoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h = cfg["hidden_dim"]
        self.char_embed = nn.Embedding(cfg["vocab_size"], cfg["char_embed_dim"], padding_idx=0)
        self.script_embed = nn.Embedding(cfg["num_scripts"], cfg["script_embed_dim"])
        self.lang_embed = nn.Embedding(cfg["num_langs"], cfg["lang_embed_dim"], padding_idx=0)
        self.length_embed = nn.Embedding(cfg["num_length_buckets"], cfg["length_embed_dim"])
        self.num_length_buckets = cfg["num_length_buckets"]
        in_dim = cfg["char_embed_dim"] + cfg["script_embed_dim"] + cfg["lang_embed_dim"] + cfg["length_embed_dim"]
        self.input_proj, self.input_norm = nn.Linear(in_dim, h), nn.LayerNorm(h)
        self.bilstm = nn.LSTM(h, h, num_layers=cfg["num_layers"], batch_first=True, bidirectional=True)
        self.self_attention = _SelfAttention(2 * h, cfg["num_attention_heads"])
        self.pooling = nn.Module()
        self.pooling.attention = nn.Sequential(nn.Linear(2 * h, h), nn.Tanh(), nn.Identity(), nn.Linear(h, 1))
        self.output_proj = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Identity(), nn.Linear(h, cfg["embed_dim"]),
                                         nn.LayerNorm(cfg["embed_dim"]))

    def forward(self, chars, script, lang, lengths):
        b, length = chars.shape
        mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
        bucket = ((lengths - 1) // 2).clamp(0, self.num_length_buckets - 1)
        x = torch.cat([self.char_embed(chars),
                       self.script_embed(script).unsqueeze(1).expand(-1, length, -1),
                       self.lang_embed(lang).unsqueeze(1).expand(-1, length, -1),
                       self.length_embed(bucket).unsqueeze(1).expand(-1, length, -1)], dim=-1)
        x = self.input_norm(self.input_proj(x))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=length)
        out = self.self_attention(out, mask) + out
        weights = F.softmax(self.pooling.attention(out).squeeze(-1).masked_fill(~mask, float("-inf")), dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return F.normalize(self.output_proj(pooled), p=2, dim=-1)


class Symphonym:
    def __init__(self, model_dir: str | Path) -> None:
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        read = lambda name, key: (lambda d: d.get(key, d))(  # noqa: E731
            json.loads((model_dir / "vocab" / name).read_text(encoding="utf-8")))
        self.char_to_id = read("char_vocab.json", "char_to_id")
        self.lang_to_id = read("lang_vocab.json", "lang_to_id")
        self.script_to_id = read("script_vocab.json", "script_to_id")
        self.model = _Encoder(cfg)
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"))
        self.model.eval()
        vocab = self.model.char_embed.num_embeddings
        self.char_to_id = {k: (v if 0 <= v < vocab else UNK_ID) for k, v in self.char_to_id.items()}

    def _ids(self, text: str) -> tuple[list[int], int]:
        script = detect_script(text)
        text = _decompose(text) if script == "HANGUL" else unicodedata.normalize("NFC", text)
        ids = [SPACE_ID if ch == " " else self.char_to_id.get(ch, UNK_ID) for ch in text if ch == " " or ch.strip()]
        sid = self.script_to_id.get(script, self.script_to_id.get("OTHER", 0))
        return (ids[:50] or [UNK_ID]), sid

    @torch.inference_mode()
    def embed(self, texts: list[str], lang: str, batch: int = 1) -> np.ndarray:
        """이름 목록 → (N, 128). batch=1 이면 이름마다 따로 계산 (질의용: 다른 행의 영향 없음)."""
        lid = self.lang_to_id.get(lang, 0)
        out = []
        for start in range(0, len(texts), batch):
            encoded = [self._ids(t) for t in texts[start : start + batch]]
            lengths = torch.tensor([len(ids) for ids, _ in encoded])
            chars = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long)
            for i, (ids, _) in enumerate(encoded):
                chars[i, : len(ids)] = torch.tensor(ids)
            script = torch.tensor([sid for _, sid in encoded])
            out.append(self.model(chars, script, torch.full((len(encoded),), lid), lengths).numpy())
        return np.vstack(out).astype(np.float32)
