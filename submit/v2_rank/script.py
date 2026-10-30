"""DACON 추론 진입점: TF-IDF 후보 검색 + LightGBM LambdaRank 재순위.

모든 예측은 test 의 각 행과 공식 카탈로그만으로 계산한다 (행 독립).
"""

from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"
sys.path.insert(0, str(MODEL_DIR))
from src.common import is_hangul  # noqa: E402
from src.features import FEATURE_NAMES, FeatureBuilder  # noqa: E402
from src.retrieval import CatalogRetriever  # noqa: E402

HANGUL_MODE = "ranker"  # "ranker": 모든 행을 재순위 모델로 / "baseline": 한글 행은 공식 베이스라인으로
CHUNK = 512


def predict_ranker(test: pd.DataFrame, retriever: CatalogRetriever) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "mappers.joblib")
    if list(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model and feature code versions do not match")
    builder = FeatureBuilder(retriever, bundle["type"], bundle["flag"])
    ranker = lgb.Booster(model_file=str(MODEL_DIR / "ranker.txt"))

    out = np.empty(len(test), dtype=object)
    for start in range(0, len(test), CHUNK):
        part = test.iloc[start : start + CHUNK]
        signals = retriever.generate(part)
        blocks, owners = [], []
        for j, (row, sig) in enumerate(zip(part.to_dict("records"), signals)):
            if len(sig["cand"]) == 0:  # 이름·톤수 모두 단서가 없는 행
                out[start + j] = retriever.ids[0]
                continue
            blocks.append(builder.build(row, sig))
            owners.append((start + j, sig["cand"]))
        if not blocks:
            continue
        scores = ranker.predict(np.vstack(blocks), num_threads=6)
        offset = 0
        for (i, cand), block in zip(owners, blocks):
            local = scores[offset : offset + len(block)]
            out[i] = retriever.ids[cand[int(np.argmax(local))]]
            offset += len(block)
    return out


def predict_baseline(test: pd.DataFrame, catalog: pd.DataFrame) -> list[str]:
    from baseline_lib import CatalogIndex, predict_ids

    bundle = joblib.load(MODEL_DIR / "rf.pkl")
    return predict_ids(bundle["model"], test, CatalogIndex(catalog))


def run(
    test_path: str | Path = "data/test.csv",
    catalog_path: str | Path = "data/vessel_catalog.csv",
    output_path: str | Path = "output/submission.csv",
) -> Path:
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")
    test = test.reset_index(drop=True)
    retriever = CatalogRetriever(catalog)

    predictions = np.empty(len(test), dtype=object)
    hangul = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    use_ranker = ~hangul if HANGUL_MODE == "baseline" else np.ones(len(test), dtype=bool)
    if use_ranker.any():
        predictions[use_ranker] = predict_ranker(test[use_ranker].reset_index(drop=True), retriever)
    if (~use_ranker).any():
        predictions[~use_ranker] = predict_baseline(test[~use_ranker].reset_index(drop=True), catalog)

    valid_ids = set(retriever.ids)
    if any(p not in valid_ids for p in predictions):
        raise ValueError("Invalid catalog predictions")
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
