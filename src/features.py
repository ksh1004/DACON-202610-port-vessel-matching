"""2단계 재순위 특징: 질의 1행과 그 후보들 사이의 쌍(pair) 특징.

모든 특징은 (질의 행, 후보, 공식 데이터에서 학습한 매핑)만으로 계산한다.
같은 질의의 후보끼리 비교하는 특징(그룹 내 순위 등)은 행 독립 원칙에 맞는다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

from src.common import compact_name, is_hangul, normalize_name
from src.hangul import english_key
from src.retrieval import CatalogRetriever, query_key

FEATURE_NAMES = [
    # 이름
    "is_exact", "sim_word", "sim_compact", "rank_word", "rank_compact",
    "fz_ratio", "fz_partial", "fz_token_sort", "fz_token_set", "fz_wratio",
    "jaro_winkler", "lev_norm", "dl_norm", "prefix_ratio", "suffix_ratio",
    "len_query", "len_cand", "len_diff", "words_diff", "same_sorted_chars",
    "digits_state", "name_dup_count",
    # 발음 키 (한글 ↔ 영문 자음 골격)
    "sim_key", "rank_key", "key_exact", "key_lev", "key_jw", "key_tset",
    "key_rank_in_group", "key_lev_gap_to_best", "key_len_diff",
    "sim_key_word", "rank_key_word", "key_tok_cover_q", "key_tok_cover_c",
    "name_tok_cover_q", "name_tok_cover_c", "name_tok_fuzzy_cover_q",
    # LLM 음역 (한글 질의만, 없으면 0)
    "has_llm", "sim_llm", "rank_llm", "llm_exact_any", "llm_ratio_max", "llm_jw_max", "llm_tset_max",
    "llm_key_lev_max", "llm_best_beam", "llm_ratio_rank_in_group", "llm_ratio_gap_to_best",
    # 단어 포함 + 톤수 근접 검색 (약칭 대응): 검색 순위, 질의 단어를 모두 포함하는 선박 수(로그)
    "rank_tok", "tok_pool",
    # 약칭 대응: LLM 음역 단어가 후보에 포함되는 비율, 질의(또는 음역)가 후보 이름의 앞부분인지
    "llm_tok_cover_q", "llm_tok_cover_c", "prefix_match",
    # 그룹 내 상대값
    "sim_word_gap_to_best", "sim_compact_gap_to_best", "dl_rank_in_group", "group_size",
    # 톤수
    "gt_gap", "igt_gap", "best_gap", "log_ratio", "gt_exact", "q_gt_missing", "c_gt_missing", "rank_gt",
    "c_dwt_gt_ratio",
    # 범주
    "type_logp", "type_seen", "flag_logp", "flag_seen", "q_flag_missing",
    # 후보 자체의 사전 정보: 입항 선박에 흔한 선종·국적인지, 크기와 단위 (톤수가 없을 때 대신 쓰는 단서)
    "type_prior", "flag_prior", "c_log_gt", "c_log_size", "c_unit",
    # 질의
    "q_hangul",
]

_DIGITS = re.compile(r"\d+")


def _positive(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > 0 else np.nan


def _rel_gap(query: float, cand: np.ndarray) -> np.ndarray:
    gap = np.abs(cand - query) / np.maximum(np.maximum(cand, query), 1.0)
    return np.where(np.isnan(gap), 2.0, np.minimum(gap, 2.0))


class CategoryMapper:
    """한글 범주(선종/국적) → 카탈로그 영문 범주의 조건부 확률 P(영문 | 한글).

    train 정답 쌍에서 학습한다. fold 학습 시에는 학습 fold 로만 적합해 누수를 막는다.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.counts: dict[str, Counter] = {}
        self.vocab_size = 1

    def fit(self, query_values: pd.Series, catalog_values: pd.Series,
            catalog_all: pd.Series | None = None) -> "CategoryMapper":
        counts: dict[str, Counter] = defaultdict(Counter)
        for q, c in zip(query_values, catalog_values):
            if isinstance(q, str) and isinstance(c, str):
                counts[q][c] += 1
        self.counts = dict(counts)
        self.vocab_size = max(1, len({c for counter in counts.values() for c in counter}))
        # 사전확률 비: 정답(입항 선박)에서의 빈도 / 카탈로그 전체에서의 빈도 (질의 값과 무관한 후보 자체의 특징)
        self.prior: dict[str, float] = {}
        if catalog_all is not None:
            target = Counter(c for c in catalog_values if isinstance(c, str))
            base = Counter(catalog_all.astype(str))
            n_t, n_b, a = sum(target.values()), sum(base.values()), 5.0
            # 관측 정답 수 / 기대 정답 수 (카탈로그 비율 기준), 가짜 개수 a 로 증거가 적은 값은 0 근처
            self.prior = {c: float(np.log((target.get(c, 0) + a) / (n_t * base[c] / n_b + a))) for c in base}
        return self

    def log_prior(self, cand_values: np.ndarray) -> np.ndarray:
        prior = getattr(self, "prior", {})
        return np.array([prior.get(c, 0.0) for c in cand_values], dtype=np.float32)

    def log_prob(self, query_value: object, cand_values: np.ndarray) -> tuple[np.ndarray, float]:
        counter = self.counts.get(query_value) if isinstance(query_value, str) else None
        if not counter:
            return np.zeros(len(cand_values), dtype=np.float32), 0.0
        total = sum(counter.values()) + self.alpha * self.vocab_size
        probs = np.array([(counter.get(c, 0) + self.alpha) / total for c in cand_values], dtype=np.float32)
        return np.log(probs), 1.0


class FeatureBuilder:
    def __init__(self, retriever: CatalogRetriever, type_mapper: CategoryMapper, flag_mapper: CategoryMapper):
        self.r = retriever
        catalog = retriever.catalog
        self.cat_type = catalog["vessel_type"].astype(str).to_numpy()
        self.cat_flag = catalog["flag_name"].astype(str).to_numpy()
        self.cat_gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        dwt = pd.to_numeric(catalog["deadweight_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.cat_dwt_ratio = np.where(self.cat_gt > 0, dwt / self.cat_gt, np.nan)
        self.cat_log_gt = np.log1p(self.cat_gt).astype(np.float32)
        self.cat_log_size = np.log1p(pd.to_numeric(catalog["size"], errors="coerce").to_numpy(dtype=float)).astype(np.float32)
        units = sorted(catalog["unit"].astype(str).unique())  # 카탈로그만으로 정해지는 고정 코드
        self.cat_unit = catalog["unit"].astype(str).map({u: i for i, u in enumerate(units)}).to_numpy(dtype=np.float32)
        self.cat_sorted = ["".join(sorted(n)) for n in retriever.compact]
        self.cat_digits = [" ".join(_DIGITS.findall(n)) for n in retriever.names]
        dup = Counter(retriever.compact)
        self.cat_dup = np.array([dup[n] for n in retriever.compact], dtype=np.float32)
        self.type_mapper = type_mapper
        self.flag_mapper = flag_mapper

    def build(self, row: dict, signals: dict[str, np.ndarray]) -> np.ndarray:
        cand = signals["cand"]
        n = len(cand)
        if n == 0:  # 이름·톤수 단서가 모두 없어 후보가 없는 질의
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        q_name = normalize_name(row.get("vessel_name"))
        q_comp = compact_name(row.get("vessel_name"))
        c_names = [self.r.names[i] for i in cand]
        c_comp = [self.r.compact[i] for i in cand]

        def cd(scorer, query, choices):
            return process.cdist([query], choices, scorer=scorer, workers=1, dtype=np.float32)[0]

        fz_ratio = cd(fuzz.ratio, q_name, c_names) / 100
        fz_partial = cd(fuzz.partial_ratio, q_name, c_names) / 100
        fz_tsort = cd(fuzz.token_sort_ratio, q_name, c_names) / 100
        fz_tset = cd(fuzz.token_set_ratio, q_name, c_names) / 100
        fz_w = cd(fuzz.WRatio, q_name, c_names) / 100
        jw = cd(JaroWinkler.normalized_similarity, q_comp, c_comp)
        lev = cd(Levenshtein.normalized_similarity, q_comp, c_comp)
        dl = cd(DamerauLevenshtein.normalized_similarity, q_comp, c_comp)

        len_q = len(q_comp)
        len_c = np.array([len(c) for c in c_comp], dtype=np.float32)
        prefix = np.array([_common_prefix(q_comp, c) for c in c_comp], dtype=np.float32)
        suffix = np.array([_common_prefix(q_comp[::-1], c[::-1]) for c in c_comp], dtype=np.float32)
        denom = np.maximum(np.maximum(len_c, len_q), 1)
        q_words = len(q_name.split())
        words_diff = np.array([abs(len(c.split()) - q_words) for c in c_names], dtype=np.float32)
        q_sorted = "".join(sorted(q_comp))
        same_sorted = np.array([q_sorted == self.cat_sorted[i] for i in cand], dtype=np.float32)
        q_digits = " ".join(_DIGITS.findall(q_name))
        digits_state = np.array([
            -1.0 if not q_digits and not self.cat_digits[i] else float(q_digits == self.cat_digits[i])
            for i in cand
        ], dtype=np.float32)

        gt = _positive(row.get("gross_tonnage"))
        igt = _positive(row.get("international_gross_tonnage"))
        c_gt = self.cat_gt[cand]
        gt_gap = _rel_gap(gt, c_gt)
        igt_gap = _rel_gap(igt, c_gt)
        best_gap = np.minimum(gt_gap, igt_gap)
        ref = gt if np.isfinite(gt) else igt
        log_ratio = np.abs(np.log(np.where(c_gt > 0, c_gt, np.nan) / ref)) if np.isfinite(ref) else np.full(n, np.nan)
        gt_exact = ((c_gt == gt) | (c_gt == igt)).astype(np.float32)

        type_logp, type_seen, flag_logp, flag_seen, type_prior, flag_prior = self.category(row, cand)

        sim_w, sim_c = signals["sim_word"], signals["sim_compact"]
        dl_rank = (-dl).argsort(kind="stable").argsort(kind="stable").astype(np.float32)

        q_key = signals["query_key"]
        c_keys = [self.r.keys_compact[i] for i in cand]
        key_lev = cd(Levenshtein.normalized_similarity, q_key, c_keys)
        key_jw = cd(JaroWinkler.normalized_similarity, q_key, c_keys)
        key_tset = cd(fuzz.token_set_ratio, query_key(row.get("vessel_name")), [self.r.keys[i] for i in cand]) / 100
        key_exact = np.array([bool(q_key) and q_key == k for k in c_keys], dtype=np.float32)
        key_rank = (-key_lev).argsort(kind="stable").argsort(kind="stable").astype(np.float32)
        key_len_diff = np.abs(np.array([len(k) for k in c_keys], dtype=np.float32) - len(q_key))
        # 단어 단위 포함률: 약칭(단어 생략)이면 질의 단어는 모두 후보에 있지만 후보 단어 일부만 질의에 있음
        cover_q, cover_c = _token_cover(set(query_key(row.get("vessel_name")).split()),
                                        [self.r.keys[i] for i in cand])
        # 영문 이름 단어 포함률 (오타 허용 버전 포함): 'TORM' 처럼 단어가 생략된 질의 대응
        q_words = q_name.split()
        name_cover_q, name_cover_c = _token_cover(set(q_words), c_names)
        fuzzy_cover_q = np.zeros(n, dtype=np.float32)
        if q_words:
            for j, c in enumerate(c_names):
                c_words = c.split()
                if c_words:
                    best = process.cdist(q_words, c_words, scorer=Levenshtein.normalized_similarity,
                                         workers=1, dtype=np.float32).max(axis=1)
                    fuzzy_cover_q[j] = float((best >= 0.75).mean())

        llm = _llm_features(signals, c_names, c_comp, c_keys, n)

        cols = [
            signals["is_exact"], sim_w, sim_c, signals["rank_word"], signals["rank_compact"],
            fz_ratio, fz_partial, fz_tsort, fz_tset, fz_w,
            jw, lev, dl, prefix / denom, suffix / denom,
            np.full(n, len_q), len_c, np.abs(len_c - len_q), words_diff, same_sorted,
            digits_state, self.cat_dup[cand],
            signals["sim_key"], signals["rank_key"], key_exact, key_lev, key_jw, key_tset,
            key_rank, key_lev.max() - key_lev, key_len_diff,
            signals["sim_key_word"], signals["rank_key_word"], cover_q, cover_c,
            name_cover_q, name_cover_c, fuzzy_cover_q,
            *llm,
            signals["rank_tok"], np.full(n, signals["tok_pool"]),
            *_llm_token_cover(signals, c_names, n), _prefix_match(q_comp, q_key, signals, c_comp, c_keys),
            sim_w.max() - sim_w, sim_c.max() - sim_c, dl_rank, np.full(n, n),
            gt_gap, igt_gap, best_gap, log_ratio, gt_exact,
            np.full(n, float(np.isnan(gt) and np.isnan(igt))), np.isnan(c_gt).astype(np.float32), signals["rank_gt"],
            self.cat_dwt_ratio[cand],
            type_logp, np.full(n, type_seen), flag_logp, np.full(n, flag_seen),
            np.full(n, float(not isinstance(row.get("flag_name"), str))),
            type_prior, flag_prior, self.cat_log_gt[cand], self.cat_log_size[cand], self.cat_unit[cand],
            np.full(n, float(is_hangul(row.get("vessel_name")))),
        ]
        return np.column_stack(cols).astype(np.float32)

    def category_score(self, row: dict, cand: np.ndarray) -> np.ndarray:
        """검색기용 선종·국적 일치 점수 (로그 확률의 합)."""
        type_logp, _, flag_logp, _, _, _ = self.category(row, cand)
        return type_logp + flag_logp

    def category(self, row: dict, cand: np.ndarray):
        """선종·국적 매핑 특징 (fold 마다 매퍼를 바꿔 다시 계산할 수 있도록 분리)."""
        type_logp, type_seen = self.type_mapper.log_prob(row.get("vessel_type"), self.cat_type[cand])
        flag_logp, flag_seen = self.flag_mapper.log_prob(row.get("flag_name"), self.cat_flag[cand])
        type_prior = self.type_mapper.log_prior(self.cat_type[cand])
        flag_prior = self.flag_mapper.log_prior(self.cat_flag[cand])
        return type_logp, type_seen, flag_logp, flag_seen, type_prior, flag_prior


# fold 마다 학습 fold 의 매퍼로 다시 계산하는 열 (FeatureBuilder.category 의 반환 순서)
CATEGORY_COLUMNS = [FEATURE_NAMES.index(c) for c in
                    ("type_logp", "type_seen", "flag_logp", "flag_seen", "type_prior", "flag_prior")]


def _llm_features(signals: dict, c_names: list[str], c_comp: list[str], c_keys: list[str], n: int) -> list:
    """LLM 이 생성한 영문 후보(빔)들과 카탈로그 후보 이름의 유사도. 빔별 최댓값을 쓴다."""
    beams = signals.get("llm_beams") or []
    sim_llm = signals.get("sim_llm", np.zeros(n, dtype=np.float32))
    rank_llm = signals.get("rank_llm", np.full(n, 999, dtype=np.float32))
    if not beams:
        zeros = np.zeros(n, dtype=np.float32)
        return [zeros, sim_llm, rank_llm, zeros, zeros, zeros, zeros, zeros,
                np.full(n, 9, dtype=np.float32), np.full(n, 999, dtype=np.float32), zeros]
    b_names = [normalize_name(b) for b in beams]
    b_comp = [compact_name(b) for b in beams]
    b_keys = [english_key(b).replace(" ", "") for b in beams]

    def cd(scorer, queries, choices):
        return process.cdist(queries, choices, scorer=scorer, workers=1, dtype=np.float32)

    ratio = cd(fuzz.ratio, b_names, c_names) / 100
    jw = cd(JaroWinkler.normalized_similarity, b_comp, c_comp)
    tset = cd(fuzz.token_set_ratio, b_names, c_names) / 100
    key_lev = cd(Levenshtein.normalized_similarity, b_keys, c_keys)
    exact = np.array([[bc == c for c in c_comp] for bc in b_comp], dtype=np.float32)
    ratio_max = ratio.max(axis=0)
    rank = (-ratio_max).argsort(kind="stable").argsort(kind="stable").astype(np.float32)
    return [np.ones(n, dtype=np.float32), sim_llm, rank_llm, exact.max(axis=0), ratio_max, jw.max(axis=0),
            tset.max(axis=0), key_lev.max(axis=0), ratio.argmax(axis=0).astype(np.float32), rank,
            ratio_max.max() - ratio_max]


def _llm_token_cover(signals: dict, c_names: list[str], n: int) -> list[np.ndarray]:
    """LLM 빔 단어 중 후보에 있는 비율(빔별 최댓값)과 후보 단어 중 빔에 있는 비율(같은 빔 기준)."""
    beams = signals.get("llm_beams") or []
    best_q, best_c = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for beam in beams:
        cover_q, cover_c = _token_cover(set(normalize_name(beam).split()), c_names)
        better = (cover_q > best_q) | ((cover_q == best_q) & (cover_c > best_c))
        best_q = np.where(better, cover_q, best_q)
        best_c = np.where(better, cover_c, best_c)
    return [best_q, best_c]


def _prefix_match(q_comp: str, q_key: str, signals: dict, c_comp: list[str], c_keys: list[str]) -> np.ndarray:
    """질의가 후보 이름의 앞부분인지 (뒷부분이 잘린 약칭). 한글은 발음 키와 LLM 음역으로 판단."""
    out = np.zeros(len(c_comp), dtype=np.float32)
    if is_hangul(q_comp):
        if len(q_key) >= 2:
            out = np.maximum(out, [float(k.startswith(q_key) and k != q_key) for k in c_keys])
        for beam in signals.get("llm_beams") or []:
            b = compact_name(beam)
            if len(b) >= 3:
                out = np.maximum(out, [float(c.startswith(b) and c != b) for c in c_comp])
    elif len(q_comp) >= 3:
        out = np.array([float(c.startswith(q_comp) and c != q_comp) for c in c_comp], dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _token_cover(q_tokens: set[str], cand_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """질의 단어 중 후보에 있는 비율, 후보 단어 중 질의에 있는 비율."""
    cover_q = np.zeros(len(cand_texts), dtype=np.float32)
    cover_c = np.zeros(len(cand_texts), dtype=np.float32)
    if not q_tokens:
        return cover_q, cover_c
    for j, text in enumerate(cand_texts):
        c_tokens = set(text.split())
        if c_tokens:
            common = len(q_tokens & c_tokens)
            cover_q[j] = common / len(q_tokens)
            cover_c[j] = common / len(c_tokens)
    return cover_q, cover_c


def _common_prefix(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def fit_mappers(train: pd.DataFrame, catalog: pd.DataFrame) -> tuple[CategoryMapper, CategoryMapper]:
    target = catalog.set_index("vessel_id").loc[train["target_vessel_id"]]
    # 사전확률은 선박 단위로 센다 (여러 번 입항한 선박이 과대 반영되지 않게)
    type_mapper = CategoryMapper().fit(train["vessel_type"], target["vessel_type"].reset_index(drop=True))
    flag_mapper = CategoryMapper().fit(train["flag_name"], target["flag_name"].reset_index(drop=True))
    vessels = catalog.set_index("vessel_id").loc[train["target_vessel_id"].unique()]
    type_mapper.prior = CategoryMapper().fit(vessels["vessel_type"], vessels["vessel_type"], catalog["vessel_type"]).prior
    flag_mapper.prior = CategoryMapper().fit(vessels["flag_name"], vessels["flag_name"], catalog["flag_name"]).prior
    return type_mapper, flag_mapper
