"""DACON 추론 진입점: TF-IDF 후보 검색 + LightGBM LambdaRank 재순위.

모든 예측은 test 의 각 행과 공식 카탈로그만으로 계산한다 (행 독립).
"""

import os
from pathlib import Path
import sys

# 추론 서버는 인터넷이 차단되어 있으므로 Hugging Face 가 네트워크에 접근하지 않도록 강제한다
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
LLM_DIR = MODEL_DIR / "llm"  # 있으면 한글 선박명의 영문 음역 후보를 LLM 으로 생성
CHUNK = 512


class LLMBeams:
    """한글 이름 → 영문 음역 N-best. 이름 하나씩(배치 1) 생성해 다른 행의 영향을 받지 않는다.

    같은 이름은 항상 같은 결과이므로 실행 중 메모이제이션은 예측을 바꾸지 않는다.
    """

    def __init__(self) -> None:
        from src.llm_cache import N_BEST
        from src.llm_translit import Transliterator

        self.model = Transliterator(LLM_DIR)
        self.n_best = N_BEST
        self.memo: dict[str, list[str]] = {}

    def __call__(self, names: list[str]) -> list[list[str]]:
        out = []
        for name in names:
            if not is_hangul(name):
                out.append([])
                continue
            if name not in self.memo:
                self.memo[name] = self.model.generate([name], n_best=self.n_best, batch_size=1)[0]
            out.append(self.memo[name])
        return out


FUZZY_K = 50  # 한글 질의마다 편집 거리 검색에서 더할 후보 수 (발음 키 / LLM 음역 각각)


def _fuzzy_top(score: np.ndarray, k: int) -> np.ndarray:
    cols = np.argpartition(-score, kth=min(k, len(score) - 1))[:k]
    cols = cols[np.lexsort((cols, -score[cols]))]
    return cols[score[cols] > 0]


def add_fuzzy_candidates(retriever: CatalogRetriever) -> None:
    """한글 질의의 후보에 편집 거리(Indel) 기반 검색 결과를 더한다.

    오타(음절 탈락·순서 바뀜 등)가 있고 톤수가 없으면 TF-IDF 검색이 정답을 놓친다. 발음 키와 LLM 음역을
    카탈로그 전체와 편집 거리로 비교해 상위 후보를 추가한다. 그 행의 입력만 쓰므로 행 독립이다.
    새 후보는 원래 검색기들의 상위 목록 밖이므로 순위 신호는 999(목록 밖), 유사도는 같은 방식으로 다시 계산한다.
    """
    from rapidfuzz import process
    from rapidfuzz.distance import Indel
    from src.common import compact_name, normalize_name
    from src.retrieval import query_key

    original = retriever.generate
    r = retriever

    def generate(queries, *args, llm_beams=None, **kwargs):
        out = original(queries, *args, llm_beams=llm_beams, **kwargs)
        names = queries["vessel_name"].tolist()
        for i, sig in enumerate(out):
            if not is_hangul(names[i]) or len(sig["cand"]) == 0:
                continue
            beams = list(sig.get("llm_beams") or [])
            pools = []
            if sig["query_key"]:
                pools.append(_fuzzy_top(process.cdist([sig["query_key"]], r.keys_compact,
                                                      scorer=Indel.normalized_similarity, workers=1)[0], FUZZY_K))
            if beams:
                pools.append(_fuzzy_top(process.cdist([compact_name(b) for b in beams], r.compact,
                                                      scorer=Indel.normalized_similarity, workers=1).max(axis=0), FUZZY_K))
            if not pools:
                continue
            new = pd.unique(np.concatenate(pools)).astype(int)
            new = new[~np.isin(new, sig["cand"])]
            if len(new) == 0:
                continue
            name = names[i]
            key_spaced = query_key(name)
            sim = lambda vec, mat, q: (vec.transform(q) @ mat[new].T).toarray()  # noqa: E731
            add = {
                "sim_word": sim(r.tfidf_word, r.matrix_word, [normalize_name(name)])[0],
                "sim_compact": sim(r.tfidf_compact, r.matrix_compact, [compact_name(name)])[0],
                "sim_key": sim(r.tfidf_key, r.matrix_key, [key_spaced.replace(" ", "")])[0],
                "sim_key_word": sim(r.tfidf_key_word, r.matrix_key_word, [key_spaced])[0],
                "sim_llm": (sim(r.tfidf_word, r.matrix_word, [normalize_name(b) for b in beams]).max(axis=0)
                            if beams else np.zeros(len(new))),
                "is_exact": np.zeros(len(new)),
            }
            for field in ("rank_word", "rank_compact", "rank_key", "rank_key_word", "rank_gt", "rank_tok", "rank_llm"):
                add[field] = np.full(len(new), 999.0)
            for field, values in add.items():
                base = sig[field]
                sig[field] = np.concatenate([base, np.asarray(values, dtype=base.dtype)])
            sig["cand"] = np.concatenate([sig["cand"], new])
        return out

    retriever.generate = generate


def predict_ranker(test: pd.DataFrame, retriever: CatalogRetriever) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "mappers.joblib")
    if list(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model and feature code versions do not match")
    builder = FeatureBuilder(retriever, bundle["type"], bundle["flag"])
    retriever.category_score = builder.category_score
    matching = None
    neural = None
    if (MODEL_DIR / "encoder_views.json").exists():
        from src.encoder_views import EncoderViewsEnsemble
        neural = EncoderViewsEnsemble.load(MODEL_DIR, retriever)
    elif (MODEL_DIR / "pair_views.json").exists():
        from src.pair_views import PairViewsEnsemble
        neural = PairViewsEnsemble.load(MODEL_DIR, retriever)
    elif (MODEL_DIR / "neural.json").exists():
        from src.neural_pair import NeuralEnsemble
        neural = NeuralEnsemble.load(MODEL_DIR, retriever)
    elif (MODEL_DIR / "matching.json").exists():
        from src.matching_ranker import MatchingRanker
        matching = MatchingRanker.load(MODEL_DIR, retriever)
        score_groups = matching.score_groups
    elif (MODEL_DIR / "ensemble.json").exists():  # 여러 학습기의 그룹 내 정규화 점수 가중 합
        from src.ensemble import Ensemble
        score_groups = Ensemble.load(MODEL_DIR).score_groups
    else:
        ranker = lgb.Booster(model_file=str(MODEL_DIR / "ranker.txt"))

        def score_groups(blocks):
            scores = ranker.predict(np.vstack(blocks), num_threads=6)
            bounds = np.cumsum([0] + [len(b) for b in blocks])
            return [scores[bounds[g] : bounds[g + 1]] for g in range(len(blocks))]
    beams_fn = LLMBeams() if LLM_DIR.is_dir() else None

    # test 선박은 train 선박과 겹치지 않는다고 보고, train 에 나온 선박은 후보에서 뺀다 (train.csv 에서 만든 고정 목록)
    exclude = np.zeros(len(retriever.ids), dtype=bool)
    exclude_path = MODEL_DIR / "train_vessels.json"
    if exclude_path.exists():
        import json
        for vessel in json.loads(exclude_path.read_text(encoding="utf-8")):
            if vessel in retriever.id_to_index:
                exclude[retriever.id_to_index[vessel]] = True
    current_cands: list = []
    if neural is not None:
        # 신경망이 상위 후보를 고르기 전에 기본 확률에서 train 선박을 0 으로 만든다
        base_score_groups = neural.baseline.score_groups

        def masked_score_groups(blocks):
            out_groups = []
            for base, cand in zip(base_score_groups(blocks), current_cands):
                bad = exclude[np.asarray(cand)]
                if bad.any() and not bad.all():
                    base = np.where(bad, 0.0, base)
                    base = base / base.sum()
                out_groups.append(base)
            return out_groups

        neural.baseline.score_groups = masked_score_groups

    out = np.empty(len(test), dtype=object)
    for start in range(0, len(test), CHUNK):
        part = test.iloc[start : start + CHUNK]
        beams = beams_fn(part["vessel_name"].tolist()) if beams_fn is not None else None
        signals = retriever.generate(part, llm_beams=beams)
        blocks, owners, block_rows, block_signals = [], [], [], []
        for j, (row, sig) in enumerate(zip(part.to_dict("records"), signals)):
            if len(sig["cand"]) == 0:  # 이름·톤수 모두 단서가 없는 행
                out[start + j] = retriever.ids[0]
                continue
            block = builder.build(row, sig)
            blocks.append(matching.extend(row, sig, block) if matching is not None else block)
            owners.append((start + j, sig["cand"]))
            block_rows.append(row)
            block_signals.append(sig)
        if not blocks:
            continue
        current_cands[:] = [cand for _, cand in owners]
        local_scores = neural.score_records(block_rows, block_signals, blocks) if neural is not None else score_groups(blocks)
        for (i, cand), local in zip(owners, local_scores):
            local = np.asarray(local, dtype=np.float64)
            bad = exclude[np.asarray(cand)]
            if bad.any() and not bad.all():
                local = np.where(bad, -np.inf, local)
            out[i] = retriever.ids[cand[int(np.argmax(local))]]
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
    test = pd.read_csv(test_path, encoding="utf-8-sig", dtype={
        name: str for name in ("sample_id", "vessel_name", "call_sign", "vessel_type", "flag_name")})
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")
    test = test.reset_index(drop=True)
    retriever = CatalogRetriever(catalog)
    add_fuzzy_candidates(retriever)

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
