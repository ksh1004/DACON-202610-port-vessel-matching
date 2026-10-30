"""공용 유틸: 데이터 로딩, 이름 정규화, 한글 판별, 대회 지표."""

from __future__ import annotations

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANGUL_WEIGHT = 0.7


def load_train() -> pd.DataFrame:
    return pd.read_csv(ROOT / "train.csv", encoding="utf-8-sig")


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(ROOT / "vessel_catalog.csv", encoding="utf-8-sig")


def is_hangul(value: object) -> bool:
    """대회 기준: 선박명에 한글 음절이 하나라도 있으면 한글 기록."""
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def normalize_name(value: object) -> str:
    """NFKC + 대문자 + 영숫자/한글과 단어 경계(공백 1칸)만 남긴다."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    chars = [ch if ch.isalnum() else " " for ch in value]
    return " ".join("".join(chars).split())


def compact_name(value: object) -> str:
    """띄어쓰기까지 모두 제거한 이름 (붙여쓰기·띄어쓰기 차이 흡수)."""
    return normalize_name(value).replace(" ", "")


def weighted_accuracy(names: pd.Series, correct: np.ndarray) -> dict[str, float]:
    """대회 지표: 0.7 * 한글 정확도 + 0.3 * 비한글 정확도."""
    correct = np.asarray(correct, dtype=bool)
    hangul = names.map(is_hangul).to_numpy(dtype=bool)
    h = float(correct[hangul].mean()) if hangul.any() else float("nan")
    n = float(correct[~hangul].mean()) if (~hangul).any() else float("nan")
    return {
        "score": HANGUL_WEIGHT * h + (1 - HANGUL_WEIGHT) * n,
        "hangul": h,
        "non_hangul": n,
        "n_hangul": int(hangul.sum()),
        "n_non_hangul": int((~hangul).sum()),
    }
