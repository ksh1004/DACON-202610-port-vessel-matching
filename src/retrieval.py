"""1단계 후보 생성: 공식 카탈로그만 색인하고, 질의는 한 행씩 독립적으로 검색한다.

    python -m src.retrieval          # train 전체에 대한 검색기별 recall@k 리포트
"""

from __future__ import annotations

from collections import defaultdict
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common import compact_name, is_hangul, load_catalog, load_train, normalize_name
from src.hangul import english_key, hangul_key


def query_key(name: object) -> str:
    """질의 선박명의 발음 키 (한글이면 한글 규칙, 아니면 영문 규칙)."""
    return hangul_key(name) if is_hangul(name) else english_key(name)


_EMPTY = np.empty(0, dtype=int)
# 실험용 스위치: VM_TOKEN_SEARCH=0 이면 단어 포함 검색을 끈다 (v6 와 같은 후보, 비교 실험용)
TOKEN_SEARCH = os.environ.get("VM_TOKEN_SEARCH", "1") == "1"
RETRIEVAL_VERSION = "tok2" if TOKEN_SEARCH else "notok"  # 후보 생성 방식이 바뀌면 올린다 (특징 캐시 키에 들어감)


def _postings(token_lists: list[list[str]]) -> dict[str, np.ndarray]:
    """단어 → 그 단어를 가진 카탈로그 행 번호 (오름차순, 중복 없음)."""
    index: dict[str, list[int]] = defaultdict(list)
    for i, tokens in enumerate(token_lists):
        for token in dict.fromkeys(tokens):
            index[token].append(i)
    return {t: np.asarray(v, dtype=int) for t, v in index.items()}


def _positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


class CatalogRetriever:
    """카탈로그 색인. 모든 검색 메서드는 질의 행마다 독립적인 결과를 낸다."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.ids = self.catalog["vessel_id"].astype(str).to_numpy()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.names = self.catalog["vessel_name"].map(normalize_name).tolist()
        self.compact = self.catalog["vessel_name"].map(compact_name).tolist()

        # 단어 경계를 살린 문자 n-gram: 단어 단위 오타에 강함
        self.tfidf_word = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.matrix_word = self.tfidf_word.fit_transform(self.names)
        # 띄어쓰기를 지운 문자 n-gram: 붙여쓰기·띄어쓰기 차이에 강함
        self.tfidf_compact = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.matrix_compact = self.tfidf_compact.fit_transform(self.compact)

        self.exact: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.compact):
            if name:
                self.exact[name].append(i)

        # 발음 키: 한글 질의를 영문 카탈로그와 같은 자음 골격 공간에서 검색
        self.keys = self.catalog["vessel_name"].map(english_key).tolist()
        self.keys_compact = [k.replace(" ", "") for k in self.keys]
        self.tfidf_key = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True,
                                         lowercase=False, dtype=np.float32)
        self.matrix_key = self.tfidf_key.fit_transform(self.keys_compact)
        # 단어 경계를 살린 발음 키: 약칭(단어 생략)된 질의도 남은 단어로 찾기
        self.tfidf_key_word = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True,
                                              lowercase=False, dtype=np.float32)
        self.matrix_key_word = self.tfidf_key_word.fit_transform(self.keys)
        self.exact_key: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(self.keys_compact):
            if key:
                self.exact_key[key].append(i)

        # 단어 역색인: 약칭(단어 생략) 질의는 '남은 단어를 모두 포함하는 선박 중 톤수가 가까운 순'으로 찾는다
        self.name_postings = _postings([n.split() for n in self.names])
        self.key_postings = _postings([k.split() for k in self.keys])
        self.postings = {"name": self.name_postings, "key": self.key_postings}
        self._vocab = {w: list(p) for w, p in self.postings.items()}
        self._fuzzy_memo: dict[tuple[str, str], np.ndarray] = {}  # 단어 → 포스팅 (입력만의 함수라 행 독립)
        # (질의 행, 카탈로그 행 번호들) → 선종·국적 일치 점수. 톤수가 없을 때 단어 포함 후보의 순서를 정한다
        self.category_score = None

        gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(gt) & (gt > 0))
        order = valid[np.argsort(gt[valid], kind="mergesort")]
        self.gt = gt
        self.gt_sorted_index = order
        self.gt_sorted_value = gt[order]

    # ---- 개별 검색기 -------------------------------------------------------
    def search_exact(self, names: pd.Series) -> list[np.ndarray]:
        return [np.asarray(self.exact.get(compact_name(n), []), dtype=int) for n in names]

    def _tfidf_topk(self, vectorizer, matrix, texts: list[str], k: int, batch: int = 512):
        out_idx, out_score = [], []
        for start in range(0, len(texts), batch):
            q = vectorizer.transform(texts[start : start + batch])
            sim = (q @ matrix.T).toarray()
            top = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
            for row, cols in zip(sim, top):
                # 동점은 카탈로그 행 번호로 정렬해 결정적으로 만든다
                order = np.lexsort((cols, -row[cols]))
                cols = cols[order]
                keep = row[cols] > 0
                out_idx.append(cols[keep])
                out_score.append(row[cols][keep])
        return out_idx, out_score

    def search_tfidf_word(self, names: pd.Series, k: int = 100):
        return self._tfidf_topk(self.tfidf_word, self.matrix_word, [normalize_name(n) for n in names], k)

    def search_tfidf_compact(self, names: pd.Series, k: int = 100):
        return self._tfidf_topk(self.tfidf_compact, self.matrix_compact, [compact_name(n) for n in names], k)

    def search_tonnage(self, tonnages: pd.Series, k: int = 50) -> list[np.ndarray]:
        out = []
        for value in tonnages:
            value = _positive(value)
            if value is None:
                out.append(np.empty(0, dtype=int))
                continue
            pos = int(np.searchsorted(self.gt_sorted_value, value))
            lo, hi = max(0, pos - k), min(len(self.gt_sorted_value), pos + k)
            window = np.arange(lo, hi)
            gap = np.abs(self.gt_sorted_value[window] - value)
            order = np.lexsort((self.gt_sorted_index[window], gap))[:k]
            out.append(self.gt_sorted_index[window[order]])
        return out

    def _token_postings(self, token: str, which: str) -> np.ndarray:
        """단어의 포스팅. 4글자 이상이면 편집거리 1 이내의 카탈로그 단어까지 포함 (오타 + 약칭 동시 대응)."""
        postings = self.postings[which]
        if len(token) < 4:
            return postings.get(token, _EMPTY)
        memo_key = (which, token)
        if memo_key not in self._fuzzy_memo:
            hits = process.extract(token, self._vocab[which], scorer=Levenshtein.distance, score_cutoff=1, limit=50)
            arrays = [postings[h[0]] for h in hits]
            self._fuzzy_memo[memo_key] = np.unique(np.concatenate(arrays)) if arrays else _EMPTY
        return self._fuzzy_memo[memo_key]

    def search_tokens(self, queries: list[tuple[list[str], str]], row: dict, k: int) -> tuple[np.ndarray, int]:
        """(단어 목록, 색인 'name'|'key') 마다 그 단어를 모두 포함하는 선박을 모아 톤수가 가까운 순으로 k 개씩.

        톤수가 없으면 선종·국적 일치 점수가 높은 순, 같으면 이름이 짧은 순 (생략된 단어가 적은 선박 우선).
        두 번째 값은 포함 선박 수의 최솟값 (모호함의 정도).
        """
        refs = [v for v in (_positive(row.get("gross_tonnage")), _positive(row.get("international_gross_tonnage"))) if v]
        gaps, picks, sizes = [], [], []
        for tokens, which in queries:
            if not tokens:
                continue
            pool = self._token_postings(tokens[0], which)
            for token in tokens[1:]:
                if len(pool) == 0:
                    break
                pool = np.intersect1d(pool, self._token_postings(token, which), assume_unique=True)
            if len(pool) == 0:
                continue
            if refs:
                gap = np.min([np.abs(self.gt[pool] - r) / r for r in refs], axis=0)
                gap = np.where(np.isnan(gap), np.inf, gap)
            else:  # 간격 대신 (범주 점수, 이름 길이) 순서의 순번을 쓴다
                length = np.array([len(self.compact[i]) for i in pool], dtype=float)
                if self.category_score is not None:
                    order = np.lexsort((pool, length, -self.category_score(row, pool)))
                    gap = np.empty(len(pool), dtype=float)
                    gap[order] = np.arange(len(pool), dtype=float)
                else:
                    gap = length
            top = np.lexsort((pool, gap))[:k]
            gaps.append(gap[top])
            picks.append(pool[top])
            sizes.append(len(pool))
        if not picks:
            return _EMPTY, 0
        gap, idx = np.concatenate(gaps), np.concatenate(picks)
        return pd.unique(idx[np.lexsort((idx, gap))]).astype(int), min(sizes)

    # ---- 재순위용: 후보 + 검색 신호 ------------------------------------------
    def generate(
        self, queries: pd.DataFrame, k_word: int = 100, k_compact: int = 100, k_gt: int = 30,
        k_key: int = 100, k_key_exact: int = 50, batch: int = 512,
        llm_beams: list[list[str]] | None = None, k_llm: int = 50, k_tok: int = 30,
    ) -> list[dict[str, np.ndarray]]:
        """질의 행마다 후보 인덱스와 검색 신호(TF-IDF 코사인, 검색기별 순위)를 돌려준다.

        TF-IDF 유사도는 질의 행 벡터와 카탈로그 행렬의 곱이므로 다른 질의 행의 영향을 받지 않는다.
        llm_beams 가 주어지면 행마다 LLM 음역 후보(영문)로도 검색한다 (그 행의 이름에서만 생성된 값).
        """
        names = queries["vessel_name"].tolist()
        records = queries.to_dict("records")
        exact = self.search_exact(queries["vessel_name"])
        gt = self.search_tonnage(queries["gross_tonnage"], k_gt)
        igt = self.search_tonnage(queries["international_gross_tonnage"], k_gt)
        out: list[dict[str, np.ndarray]] = []
        for start in range(0, len(names), batch):
            chunk = names[start : start + batch]
            sim_w = (self.tfidf_word.transform([normalize_name(n) for n in chunk]) @ self.matrix_word.T).toarray()
            sim_c = (self.tfidf_compact.transform([compact_name(n) for n in chunk]) @ self.matrix_compact.T).toarray()
            keys_spaced = [query_key(n) for n in chunk]
            keys = [k.replace(" ", "") for k in keys_spaced]
            sim_k = (self.tfidf_key.transform(keys) @ self.matrix_key.T).toarray()
            sim_kw = (self.tfidf_key_word.transform(keys_spaced) @ self.matrix_key_word.T).toarray()
            for j in range(len(chunk)):
                i = start + j
                top_w = self._sorted_top(sim_w[j], k_word)
                top_c = self._sorted_top(sim_c[j], k_compact)
                top_k = top_kw = top_llm = np.empty(0, dtype=int)
                sim_llm = None
                beams = llm_beams[i] if llm_beams is not None else []
                parts = [exact[i], top_w, top_c]
                if is_hangul(chunk[j]):  # 발음 키 단어 + LLM 음역 단어
                    tok_queries = [(keys_spaced[j].split(), "key")] + [(normalize_name(b).split(), "name") for b in beams]
                else:
                    tok_queries = [(normalize_name(chunk[j]).split(), "name")]
                top_tok, tok_pool = self.search_tokens(tok_queries, records[i], k_tok) if TOKEN_SEARCH else (_EMPTY, 0)
                parts.append(top_tok)
                if is_hangul(chunk[j]):  # 한글 질의만 발음 키로 후보 확장 (영문은 이미 recall 99.9%)
                    key_exact = self._closest_tonnage(self.exact_key.get(keys[j], []), records[i], k_key_exact)
                    top_k = self._sorted_top(sim_k[j], k_key)
                    top_kw = self._sorted_top(sim_kw[j], k_key)
                    parts += [key_exact, top_k, top_kw]
                    if beams:  # LLM 음역 후보들로 영문 검색, 후보별 최고 유사도
                        llm_exact = [k for b in beams for k in self.exact.get(compact_name(b), [])]
                        sim_llm = (self.tfidf_word.transform([normalize_name(b) for b in beams])
                                   @ self.matrix_word.T).toarray().max(axis=0)
                        top_llm = self._sorted_top(sim_llm, k_llm)
                        parts += [np.asarray(llm_exact, dtype=int), top_llm]
                parts += [gt[i], igt[i]]
                cand = pd.unique(np.concatenate(parts)).astype(int)
                out.append({
                    "sim_llm": sim_llm[cand] if sim_llm is not None else np.zeros(len(cand), dtype=np.float32),
                    "rank_llm": self._rank_of(cand, top_llm),
                    "llm_beams": beams,
                    "cand": cand,
                    "sim_word": sim_w[j][cand],
                    "sim_compact": sim_c[j][cand],
                    "sim_key": sim_k[j][cand],
                    "sim_key_word": sim_kw[j][cand],
                    "rank_key_word": self._rank_of(cand, top_kw),
                    "rank_word": self._rank_of(cand, top_w),
                    "rank_compact": self._rank_of(cand, top_c),
                    "rank_key": self._rank_of(cand, top_k),
                    "rank_gt": np.minimum(self._rank_of(cand, gt[i]), self._rank_of(cand, igt[i])),
                    "rank_tok": self._rank_of(cand, top_tok),
                    "tok_pool": float(np.log1p(tok_pool)),
                    "is_exact": np.isin(cand, exact[i]).astype(np.float32),
                    "query_key": keys[j],
                })
        return out

    def _closest_tonnage(self, indices: list[int], row: dict, k: int) -> np.ndarray:
        """같은 발음 키를 가진 선박이 많을 때 톤수가 가까운 순으로 k 개만 남긴다."""
        indices = np.asarray(indices, dtype=int)
        if len(indices) <= k:
            return indices
        refs = [v for v in (_positive(row.get("gross_tonnage")), _positive(row.get("international_gross_tonnage"))) if v]
        if not refs:
            return indices[:k]
        gap = np.min([np.abs(self.gt[indices] - r) for r in refs], axis=0)
        gap = np.where(np.isnan(gap), np.inf, gap)
        return indices[np.lexsort((indices, gap))[:k]]

    @staticmethod
    def _sorted_top(row: np.ndarray, k: int) -> np.ndarray:
        cols = np.argpartition(-row, kth=min(k, len(row) - 1))[:k]
        cols = cols[np.lexsort((cols, -row[cols]))]
        return cols[row[cols] > 0]

    @staticmethod
    def _rank_of(cand: np.ndarray, ranked: np.ndarray, missing: int = 999) -> np.ndarray:
        position = {c: r for r, c in enumerate(ranked.tolist())}
        return np.array([position.get(c, missing) for c in cand.tolist()], dtype=np.float32)

    # ---- 합집합 --------------------------------------------------------------
    def candidates(self, queries: pd.DataFrame, k_word: int = 100, k_compact: int = 100, k_gt: int = 30):
        """검색기 결과를 정해진 순서로 합쳐 중복을 제거한다 (질의 행마다 독립)."""
        exact = self.search_exact(queries["vessel_name"])
        word, _ = self.search_tfidf_word(queries["vessel_name"], k_word)
        comp, _ = self.search_tfidf_compact(queries["vessel_name"], k_compact)
        gt = self.search_tonnage(queries["gross_tonnage"], k_gt)
        igt = self.search_tonnage(queries["international_gross_tonnage"], k_gt)
        merged = []
        for parts in zip(exact, word, comp, gt, igt):
            merged.append(pd.unique(np.concatenate(parts)).astype(int))
        return merged


def main() -> None:
    from src.validate import recall_at_k

    train, catalog = load_train(), load_catalog()
    t0 = time.perf_counter()
    retriever = CatalogRetriever(catalog)
    print(f"색인 생성 {time.perf_counter() - t0:.1f}초, 어휘 크기 word={len(retriever.tfidf_word.vocabulary_)}, "
          f"compact={len(retriever.tfidf_compact.vocabulary_)}")
    truth = train["target_vessel_id"].map(retriever.id_to_index).to_numpy()
    names = train["vessel_name"]

    results = {}
    t0 = time.perf_counter()
    results["exact"] = retriever.search_exact(names)
    results["tfidf_word"] = retriever.search_tfidf_word(names, 200)[0]
    results["tfidf_compact"] = retriever.search_tfidf_compact(names, 200)[0]
    results["tonnage_gt"] = retriever.search_tonnage(train["gross_tonnage"], 200)
    print(f"개별 검색 {time.perf_counter() - t0:.1f}초 ({len(train)}행)")
    t0 = time.perf_counter()
    results["union"] = retriever.candidates(train)
    elapsed = time.perf_counter() - t0
    sizes = np.array([len(c) for c in results["union"]])
    print(f"합집합 검색 {elapsed:.1f}초 ({elapsed / len(train) * 1000:.2f} ms/행), 후보 수 평균 {sizes.mean():.0f} / 최대 {sizes.max()}")

    pd.set_option("display.width", 200)
    for name, cands in results.items():
        print(f"\n== {name}")
        print(recall_at_k(truth, cands, names).round(4))


if __name__ == "__main__":
    main()
