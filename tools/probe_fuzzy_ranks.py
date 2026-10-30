"""Where does the truth rank, per retriever, for stress rows whose truth was missed?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein, Indel

PKG = Path(r"D:/KSH/pjt/DACON202610/submit/v14_exclude")
sys.path.insert(0, str(PKG / "model"))
sys.path.insert(0, str(PKG))
import script  # noqa: E402
from src.common import compact_name, normalize_name  # noqa: E402
from src.retrieval import CatalogRetriever, query_key  # noqa: E402

W = Path(sys.argv[1])
test = pd.read_csv(W / "data/test.csv", encoding="utf-8-sig", dtype=str)
scored = pd.read_csv(W / "scored.csv", encoding="utf-8-sig")
test = test.merge(scored[["sample_id", "variant", "target_vessel_id_x", "ok"]], on="sample_id")
catalog = pd.read_csv(W / "data/vessel_catalog.csv", encoding="utf-8-sig")
r = CatalogRetriever(catalog)
rows = test[~test.ok].reset_index(drop=True)
beams = script.LLMBeams()(rows["vessel_name"].tolist())


def rank(sim, t):
    return int((sim > sim[t]).sum()) + 1


out = []
for name, tid, b in zip(rows.vessel_name, rows.target_vessel_id_x, beams):
    t = r.id_to_index[tid]
    key_sp = query_key(name)
    key = key_sp.replace(" ", "")
    sim_k = (r.tfidf_key.transform([key]) @ r.matrix_key.T).toarray()[0]
    sim_kw = (r.tfidf_key_word.transform([key_sp]) @ r.matrix_key_word.T).toarray()[0]
    sim_w = (r.tfidf_word.transform([normalize_name(name)]) @ r.matrix_word.T).toarray()[0]
    sim_llm = (r.tfidf_word.transform([normalize_name(x) for x in b]) @ r.matrix_word.T).toarray().max(axis=0) if b else np.zeros(len(r.ids))
    lev_k = process.cdist([key], r.keys_compact, scorer=Indel.normalized_similarity, workers=-1)[0]
    lev_llm = (process.cdist([compact_name(x) for x in b], r.compact, scorer=Indel.normalized_similarity, workers=-1).max(axis=0)
               if b else np.zeros(len(r.ids)))
    out.append({"name": name, "truth": catalog.vessel_name[t], "beams": "|".join(b[:2]),
                "key": rank(sim_k, t), "key_word": rank(sim_kw, t), "llm": rank(sim_llm, t),
                "indel_key": rank(lev_k, t), "indel_llm": rank(lev_llm, t)})
df = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(df.to_string())
for col in ["key", "key_word", "llm", "indel_key", "indel_llm"]:
    print(col, {k: int((df[col] <= k).sum()) for k in (20, 50, 100, 200, 500)})
best = df[["key", "key_word", "llm", "indel_key", "indel_llm"]].min(axis=1)
print("best-of", {k: int((best <= k).sum()) for k in (20, 50, 100, 200, 500)}, "of", len(df))
