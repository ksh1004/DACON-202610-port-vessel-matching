"""검증 체계: 선박 단위 fold 분할과 후보 recall 평가.

    python -m src.validate          # experiments/folds.csv 생성
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.common import ROOT, is_hangul, load_train

N_FOLDS = 5
SEED = 42
FOLDS_PATH = ROOT / "experiments" / "folds.csv"


def make_folds(train: pd.DataFrame) -> pd.DataFrame:
    """같은 선박(call_sign)은 한 fold 에만, 한글 비율은 fold 마다 비슷하게."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(train), -1, dtype=int)
    y = train["vessel_name"].map(is_hangul).astype(int)
    for k, (_, valid) in enumerate(splitter.split(train, y, groups=train["call_sign"])):
        fold[valid] = k
    return pd.DataFrame({"sample_id": train["sample_id"], "fold": fold})


def load_folds(train: pd.DataFrame) -> np.ndarray:
    folds = pd.read_csv(FOLDS_PATH)
    return train[["sample_id"]].merge(folds, on="sample_id", how="left")["fold"].to_numpy()


def recall_at_k(
    true_index: np.ndarray, candidates: list[np.ndarray], names: pd.Series, ks=(1, 10, 50, 100, 200)
) -> pd.DataFrame:
    """후보 목록 안에서 정답 순위를 구해 한글/비한글별 recall@k 표를 만든다."""
    rank = np.array([
        int(np.flatnonzero(c == t)[0]) if (c == t).any() else 10**9
        for t, c in zip(true_index, candidates)
    ])
    hangul = names.map(is_hangul).to_numpy(dtype=bool)
    rows = {}
    for label, mask in (("hangul", hangul), ("non_hangul", ~hangul), ("all", np.ones_like(hangul))):
        rows[label] = {f"R@{k}": float((rank[mask] < k).mean()) for k in ks}
    return pd.DataFrame(rows).T


def main() -> None:
    train = load_train()
    folds = make_folds(train)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False)
    summary = train.assign(fold=folds["fold"], hangul=train["vessel_name"].map(is_hangul))
    print(summary.groupby("fold").agg(rows=("sample_id", "size"), vessels=("call_sign", "nunique"),
                                      hangul_ratio=("hangul", "mean")))
    leak = summary.groupby("call_sign")["fold"].nunique().max()
    print(f"선박당 fold 수 최댓값: {leak} (1이어야 누수 없음)")


if __name__ == "__main__":
    main()
