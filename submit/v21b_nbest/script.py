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


FUZZY_K = 500  # 한글 질의마다 편집 거리 검색에서 더할 후보 수 (발음 키 / LLM 음역 각각)

# v20: 한글 공간 비교. 카탈로그 영문 이름을 미리 LLM(같은 Kanana, few-shot)으로 한글 음역해 둔 표를 쓴다.
# test 한글 이름의 오타(음절 삭제·중복·순서 바뀜)는 한글 공간에서 생기므로 한글끼리 직접 비교한다.
# 표는 카탈로그만으로 만든 것이라 행 독립성과 무관하다.
KO_TABLE = MODEL_DIR / "catalog_ko.json"
EXCLUDE_PATH = MODEL_DIR / "train_vessels.json"  # train 에 나온 선박 목록 (후보에서 제외)
KO_W = 6.0  # 기본 순위 확률(로그)에 더할 한글 유사도 가중치 (v21b: n-best 음역과 함께 6, 노이즈 764행 v20 대비 +6/−2)
KO_K = 50  # 한글 공간 편집 거리 검색에서 더할 후보 수
KO_SCORER = "indel"
# v21: 카탈로그 이름마다 빔 음역(n-best)을 여러 개 두고, 탐욕 음역과 함께 가장 가까운 것과의 유사도를 쓴다.
KO_NBEST = MODEL_DIR / "catalog_ko_nbest.json"
# v21: 영문(비한글) 질의에도 카탈로그 이름과의 Indel 편집 거리 상위 후보를 더한다 (주최 측 오타: 글자 삭제·중복·순서 바뀜)
EN_K = 50  # 영문 노이즈 세트: 글자 오타 800행 +10/−0, 단어 빠짐·잘림 400행 +8/−5 (200은 오타 +12/−2)


def _ko_norm(name) -> str:
    s = "".join(str(name).split())
    return s[:-1] if len(s) > 2 and s.endswith("호") else s


def load_ko_names(retriever: CatalogRetriever) -> list[list[str]]:
    """카탈로그 순서대로 한글 음역 목록 (탐욕 음역 + 빔 음역, 중복·빈 문자열 제거)."""
    import json
    table = json.loads(KO_TABLE.read_text(encoding="utf-8")) if KO_TABLE.exists() else {}
    nbest = json.loads(KO_NBEST.read_text(encoding="utf-8")) if KO_NBEST.exists() else {}
    out = []
    for n in retriever.catalog["vessel_name"].astype(str):
        variants = [_ko_norm(table.get(n, ""))] + [_ko_norm(v) for v in nbest.get(n, ())]
        out.append(list(dict.fromkeys(v for v in variants if v)))
    return out


def _flat_ko(ko_names: list[list[str]]):
    flat = [v for vs in ko_names for v in vs]
    owner = np.repeat(np.arange(len(ko_names)), [len(vs) for vs in ko_names])
    return flat, owner


def ko_similarity(query: str, cand, ko_names: list[str]) -> np.ndarray:
    from rapidfuzz.distance import OSA, Indel
    scorer = Indel if KO_SCORER == "indel" else OSA
    q = _ko_norm(query)
    return np.array([max([scorer.normalized_similarity(q, v) for v in ko_names[c]] or [0.0]) for c in cand])


def _fuzzy_top(score: np.ndarray, k: int) -> np.ndarray:
    cols = np.argpartition(-score, kth=min(k, len(score) - 1))[:k]
    cols = cols[np.lexsort((cols, -score[cols]))]
    return cols[score[cols] > 0]


def _typo_reversals(name: str) -> list[str]:
    """주최 측 오타(인접 글자 순서 바뀜, 글자 중복)를 한 번 되돌린 이름 후보들."""
    out = []
    for i in range(len(name) - 1):
        a, b = name[i], name[i + 1]
        if a.isspace() or b.isspace():
            continue
        if a == b:
            out.append(name[:i] + name[i + 1:])
        else:
            out.append(name[:i] + b + a + name[i + 2:])
    return list(dict.fromkeys(out))


def add_fuzzy_candidates(retriever: CatalogRetriever, ko_names: list[str] | None = None) -> None:
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
    if ko_names is not None:
        ko_flat, ko_owner = _flat_ko(ko_names)

    def generate(queries, *args, llm_beams=None, **kwargs):
        out = original(queries, *args, llm_beams=llm_beams, **kwargs)
        names = queries["vessel_name"].tolist()
        for i, sig in enumerate(out):
            if len(sig["cand"]) == 0:
                continue
            pools = []
            if not is_hangul(names[i]):
                qc = compact_name(names[i])
                if EN_K > 0 and qc:
                    new = _fuzzy_top(process.cdist([qc], r.compact, scorer=Indel.normalized_similarity, workers=1)[0], EN_K)
                    new = new[~np.isin(new, sig["cand"])]
                    if len(new):
                        _append(r, sig, names[i], new, [])
                continue
            beams = list(sig.get("llm_beams") or [])
            if sig["query_key"]:
                pools.append(_fuzzy_top(process.cdist([sig["query_key"]], r.keys_compact,
                                                      scorer=Indel.normalized_similarity, workers=1)[0], FUZZY_K))
            if beams:
                pools.append(_fuzzy_top(process.cdist([compact_name(b) for b in beams], r.compact,
                                                      scorer=Indel.normalized_similarity, workers=1).max(axis=0), FUZZY_K))
            if ko_names is not None and KO_K > 0:
                flat_sim = process.cdist([_ko_norm(names[i])], ko_flat, scorer=Indel.normalized_similarity, workers=1)[0]
                per = np.zeros(len(ko_names), dtype=np.float64)
                np.maximum.at(per, ko_owner, flat_sim)
                pools.append(_fuzzy_top(per, KO_K))
            if not pools:
                continue
            for variant in _typo_reversals(names[i]):
                vkey = query_key(variant).replace(" ", "")
                if vkey and vkey != sig["query_key"]:
                    hits = r.exact_key.get(vkey, [])
                    if hits:
                        pools.append(r._closest_tonnage(hits, queries.iloc[i].to_dict(), 10))
            new = pd.unique(np.concatenate(pools)).astype(int)
            new = new[~np.isin(new, sig["cand"])]
            if len(new) == 0:
                continue
            _append(r, sig, names[i], new, beams)
        return out

    def _append(r, sig, name, new, beams):
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

    retriever.generate = generate


def predict_ranker(test: pd.DataFrame, retriever: CatalogRetriever, ko_names: list[str] | None = None) -> np.ndarray:
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
    exclude_path = EXCLUDE_PATH
    if exclude_path.exists():
        import json
        for vessel in json.loads(exclude_path.read_text(encoding="utf-8")):
            if vessel in retriever.id_to_index:
                exclude[retriever.id_to_index[vessel]] = True
    current_cands: list = []
    current_names: list = []
    if neural is not None:
        # 신경망이 상위 후보를 고르기 전에 기본 확률에서 train 선박을 0 으로 만든다
        base_score_groups = neural.baseline.score_groups

        def masked_score_groups(blocks):
            out_groups = []
            for base, cand, name in zip(base_score_groups(blocks), current_cands, current_names):
                bad = exclude[np.asarray(cand)]
                if bad.any() and not bad.all():
                    base = np.where(bad, 0.0, base)
                    base = base / base.sum()
                if ko_names is not None and KO_W > 0 and is_hangul(name):
                    with np.errstate(divide="ignore"):
                        z = np.log(base) + KO_W * ko_similarity(name, cand, ko_names)
                    z = z - z[np.isfinite(z)].max()
                    base = np.exp(z)
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
        current_names[:] = [row["vessel_name"] for row in block_rows]
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
    ko_names = load_ko_names(retriever) if KO_TABLE.exists() else None
    add_fuzzy_candidates(retriever, ko_names)

    predictions = np.empty(len(test), dtype=object)
    hangul = test["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    use_ranker = ~hangul if HANGUL_MODE == "baseline" else np.ones(len(test), dtype=bool)
    if use_ranker.any():
        predictions[use_ranker] = predict_ranker(test[use_ranker].reset_index(drop=True), retriever, ko_names)
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
