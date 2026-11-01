"""한글 질의 후보에 편집 거리(Indel) 퍼지 검색 후보와 오타 되돌림 후보를 더한다.

제출 패키지 v17/v18 의 script.py 에 있는 add_fuzzy_candidates 와 같은 로직이다 (K 만 인자로 받음).
학습 행렬에도 같은 후보를 넣어, 순위 모델이 퍼지 후보(검색 순위 999)를 학습 때 보게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import compact_name, is_hangul, normalize_name
from src.retrieval import CatalogRetriever, query_key


def _fuzzy_top(score: np.ndarray, k: int) -> np.ndarray:
    cols = np.argpartition(-score, kth=min(k, len(score) - 1))[:k]
    cols = cols[np.lexsort((cols, -score[cols]))]
    return cols[score[cols] > 0]


def typo_reversals(name: str) -> list[str]:
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


def add_fuzzy_candidates(retriever: CatalogRetriever, k: int) -> None:
    from rapidfuzz import process
    from rapidfuzz.distance import Indel

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
                                                      scorer=Indel.normalized_similarity, workers=1)[0], k))
            if beams:
                pools.append(_fuzzy_top(process.cdist([compact_name(b) for b in beams], r.compact,
                                                      scorer=Indel.normalized_similarity, workers=1).max(axis=0), k))
            if not pools:
                continue
            for variant in typo_reversals(names[i]):
                vkey = query_key(variant).replace(" ", "")
                if vkey and vkey != sig["query_key"]:
                    hits = r.exact_key.get(vkey, [])
                    if hits:
                        pools.append(r._closest_tonnage(hits, queries.iloc[i].to_dict(), 10))
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
