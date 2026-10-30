"""For stress rows (tonnage missing), is the truth inside v14's candidate set?"""
import sys
from pathlib import Path

import pandas as pd

PKG = Path(r"D:/KSH/pjt/DACON202610/submit/v14_exclude")
sys.path.insert(0, str(PKG / "model"))
sys.path.insert(0, str(PKG))
import script  # noqa: E402
from src.retrieval import CatalogRetriever  # noqa: E402

W = Path(sys.argv[1])
test = pd.read_csv(W / "data/test.csv", encoding="utf-8-sig", dtype={"vessel_name": str, "flag_name": str, "vessel_type": str, "call_sign": str})
scored = pd.read_csv(W / "scored.csv", encoding="utf-8-sig")
test = test.merge(scored[["sample_id", "variant", "target_vessel_id_x", "ok"]], on="sample_id")
catalog = pd.read_csv(W / "data/vessel_catalog.csv", encoding="utf-8-sig")
r = CatalogRetriever(catalog)
import joblib
from src.features import FeatureBuilder
bundle = joblib.load(PKG / "model/mappers.joblib")
builder = FeatureBuilder(r, bundle["type"], bundle["flag"])
r.category_score = builder.category_score
beams = script.LLMBeams()
rows = test.reset_index(drop=True)
sigs = r.generate(rows, llm_beams=beams(rows["vessel_name"].tolist()))
rows["in_cand"] = [r.id_to_index[t] in set(s["cand"].tolist()) for t, s in zip(rows.target_vessel_id_x, sigs)]
rows["n_cand"] = [len(s["cand"]) for s in sigs]
print(rows.groupby("variant")[["ok", "in_cand"]].mean().round(3).to_string())
print("failures with truth in candidates:", int((~rows.ok & rows.in_cand).sum()), "/", int((~rows.ok).sum()))
print("median n_cand", rows.n_cand.median())
