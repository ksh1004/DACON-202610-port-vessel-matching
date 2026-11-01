"""학습 데이터 증강: test 에서 있을 법한 노이즈(정보 누락, 약칭, 오타, 톤수 오차)를 입힌 복사본.

대회 설명의 노이즈 유형 — 한글·영문 표기 차이, 철자 변형, 약칭, 톤수 오차, 정보 누락 — 을 흉내 낸다.
원본 행의 정답을 그대로 쓰고, 원본과 같은 fold 에만 들어가므로 검증 누수가 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import is_hangul

# 비슷하게 들리거나 자주 헷갈리는 중성 (ㅐ↔ㅔ, ㅓ↔ㅗ …) — 중성 인덱스 기준
_VOWEL_SWAP = {1: 5, 5: 1, 4: 8, 8: 4, 6: 12, 12: 6, 18: 13, 13: 18, 20: 16, 16: 20, 0: 4, 2: 6}


def _letters(name: str) -> list[int]:
    return [i for i, ch in enumerate(name) if ch.isalpha()]


def typo_vowel(name: str, rng: np.random.Generator) -> str:
    pos = [i for i, ch in enumerate(name) if "가" <= ch <= "힣"]
    if not pos:
        return name
    i = int(rng.choice(pos))
    code = ord(name[i]) - 0xAC00
    initial, medial, final = code // 588, (code % 588) // 28, code % 28
    medial = _VOWEL_SWAP.get(medial, medial)
    return name[:i] + chr(0xAC00 + initial * 588 + medial * 28 + final) + name[i + 1 :]


def typo_drop(name: str, rng: np.random.Generator) -> str:
    pos = _letters(name)
    if len(pos) < 4:
        return name
    i = int(rng.choice(pos[1:]))
    return name[:i] + name[i + 1 :]


def typo_double(name: str, rng: np.random.Generator) -> str:
    pos = _letters(name)
    if not pos:
        return name
    i = int(rng.choice(pos))
    return name[: i + 1] + name[i] + name[i + 1 :]


def typo_swap(name: str, rng: np.random.Generator) -> str:
    pos = [i for i in _letters(name) if i + 1 < len(name) and name[i + 1].isalpha()]
    if not pos:
        return name
    i = int(rng.choice(pos))
    return name[:i] + name[i + 1] + name[i] + name[i + 2 :]


def drop_word(name: str, rng: np.random.Generator) -> str:
    words = name.split()
    if len(words) < 2:
        return name
    del words[int(rng.integers(len(words)))]
    return " ".join(words)


def no_space(name: str, rng: np.random.Generator) -> str:
    return name.replace(" ", "")


def keep_first(name: str, rng: np.random.Generator | None = None) -> str:
    return name.split()[0] if name.split() else name


def keep_last(name: str, rng: np.random.Generator | None = None) -> str:
    return name.split()[-1] if name.split() else name


def truncate(name: str, rng: np.random.Generator | None = None) -> str:
    """앞에서부터 글자 60% 만 남김 (띄어쓰기는 세지 않음, 최소 2글자)."""
    keep = max(2, int(round(len(name.replace(" ", "")) * 0.6)))
    out, count = [], 0
    for ch in name:
        if count >= keep:
            break
        out.append(ch)
        count += ch != " "
    return "".join(out).strip()


# 약칭 방식은 test 에서 확인할 수 없으므로 여러 방식을 섞는다 (무작위 단어 생략이 절반)
ABBREV_STYLES = [(drop_word, 0.5), (keep_first, 0.15), (keep_last, 0.15), (truncate, 0.2)]


NAME_NOISE_HANGUL = [typo_vowel, typo_drop, typo_double, no_space]
NAME_NOISE_ENGLISH = [typo_drop, typo_double, typo_swap]
DROP_WORD_PROB = 0.3  # 약칭은 스트레스 테스트에서 가장 약한 유형이라 따로 높은 확률로 넣는다


# 노이즈 강도. 'testmix' 는 제출 기록으로 역추정한 test 한글 구성(약칭 42%, 톤수 결측 33%)에 맞추고,
# train 한글 이름에는 오타가 없으므로 한글 오타를 줄인다 (영문 행은 기본과 같음)
PROFILES = {
    "default": {"drop": DROP_WORD_PROB, "typos": [0, 1, 1, 2], "gt_missing": 0.4},
    "testmix": {"drop": 0.42, "typos": [0, 0, 0, 1], "gt_missing": 0.33},
}


def perturb_row(row: dict, rng: np.random.Generator, style_rng: np.random.Generator | None = None,
                profile: str = "default") -> dict:
    """style_rng 가 있으면 약칭 방식을 ABBREV_STYLES 에서 고른다.

    방식 선택은 별도 난수열로 하고 drop_word 의 난수 소비는 그대로 두어, 나머지 노이즈는 style_rng 없이 만든 행과 같다.
    """
    row = dict(row)
    original = row["vessel_name"]
    hangul = is_hangul(original)
    noises = NAME_NOISE_HANGUL if hangul else NAME_NOISE_ENGLISH
    prof = PROFILES[profile if hangul else "default"]
    style = None
    if rng.random() < prof["drop"]:
        style = drop_word
        if style_rng is not None:
            fns, probs = zip(*ABBREV_STYLES)
            style = fns[int(style_rng.choice(len(fns), p=probs))]
        row["vessel_name"] = drop_word(original, rng)
    # 철자 노이즈 0~2개
    typos = rng.choice(noises, size=int(rng.choice(prof["typos"])), replace=False)
    for fn in typos:
        row["vessel_name"] = fn(row["vessel_name"], rng) or row["vessel_name"]
    if style is not None and style is not drop_word:  # 다른 약칭 방식: 같은 오타 종류를 별도 난수열로 다시 입힘
        name = style(original)
        for fn in typos:
            name = fn(name, style_rng) or name
        row["vessel_name"] = name
    u = rng.random()
    if u < prof["gt_missing"]:  # 톤수 결측
        row["gross_tonnage"] = np.nan
        row["international_gross_tonnage"] = np.nan
    elif u < 0.7:  # 톤수 오차
        scale = rng.uniform(0.95, 1.05)
        row["gross_tonnage"] = row["gross_tonnage"] * scale
        row["international_gross_tonnage"] = row["international_gross_tonnage"] * scale
    if rng.random() < 0.3:
        row["flag_name"] = np.nan
    return row


def organizer_edit(name: str, rng: np.random.Generator) -> str:
    """주최 측 오타 1회: 공백이 아닌 글자(음절) 하나를 삭제 35% / 중복 35% / 인접 순서 바꿈 30%.

    train 영문 이름에 남은 주최 측 노이즈와 같은 종류다 (tools/make_organizer_noise.py 와 같은 규칙).
    """
    pos = [i for i, ch in enumerate(name) if not ch.isspace()]
    if len(pos) < 2:
        return name
    op = rng.choice(["delete", "dup", "swap"], p=[0.35, 0.35, 0.30])
    if op == "swap":
        pairs = [i for i in pos if i + 1 < len(name) and not name[i + 1].isspace()]
        if pairs:
            i = int(rng.choice(pairs))
            return name[:i] + name[i + 1] + name[i] + name[i + 2:]
        op = "dup"
    i = int(rng.choice(pos))
    if op == "delete":
        return name[:i] + name[i + 1:]
    return name[:i + 1] + name[i] + name[i + 1:]


def organizer_augment(train: pd.DataFrame, copies: int, seed: int, gt_missing: float = 0.5,
                      flag_missing: float = 0.17, two_edit: float = 0.3) -> pd.DataFrame:
    """한글 행만 주최 측 방식 노이즈 복사본을 만든다. `origin` 은 원본 train 위치, `nogt` 는 톤수 결측 여부."""
    rng = np.random.default_rng(seed)
    out = []
    for pos, row in enumerate(train.to_dict("records")):
        if not is_hangul(row["vessel_name"]):
            continue
        for c in range(copies):
            new = dict(row)
            name = new["vessel_name"]
            for _ in range(2 if rng.random() < two_edit else 1):
                name = organizer_edit(name, rng)
            new["vessel_name"] = name
            new["nogt"] = bool(rng.random() < gt_missing)
            if new["nogt"]:
                new["gross_tonnage"] = new["international_gross_tonnage"] = np.nan
            if rng.random() < flag_missing:
                new["flag_name"] = np.nan
            new["sample_id"] = f"{row['sample_id']}_org{seed}_{c}"
            new["origin"] = pos
            out.append(new)
    return pd.DataFrame(out)


def augment(train: pd.DataFrame, copies_hangul: int = 4, copies_english: int = 1, seed: int = 7,
            abbrev_mix: bool = False, profile: str = "default") -> pd.DataFrame:
    """증강 행을 만든다. `origin` 열은 원본 train 의 위치 인덱스.

    abbrev_mix=False 는 예전과 같은 행을 만든다 (hard 세트 고정용).
    """
    rng = np.random.default_rng(seed)
    style_rng = np.random.default_rng(seed + 100_000) if abbrev_mix else None
    out = []
    for pos, row in enumerate(train.to_dict("records")):
        copies = copies_hangul if is_hangul(row["vessel_name"]) else copies_english
        for c in range(copies):
            new = perturb_row(row, rng, style_rng, profile)
            new["sample_id"] = f"{row['sample_id']}_aug{c}"
            new["origin"] = pos
            out.append(new)
    return pd.DataFrame(out)
