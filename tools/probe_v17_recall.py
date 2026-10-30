"""v17 failures on the organizer-noise set: is the truth inside v17's candidate set (with fuzzy wrapper)?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(r"D:/KSH/pjt/DACON202610/submit/v17_fuzzy200")
sys.path.insert(0, str(PKG / "model"))
sys.path.insert(0, str(PKG))
import script  # noqa: E402
from src.retrieval import CatalogRetriever  # noqa: E402

W = Path(sys.argv[1])
test = pd.read_csv(W / "data/test.csv", encoding="utf-8-sig", dtype={"vessel_name": str, "flag_name": str, "vessel_type": str, "call_sign": str})
truth = pd.read_csv(W / "truth.csv", encoding="utf-8-sig")
pred = pd.read_csv(W / "V17/output/submission.csv").rename(columns={"target_vessel_id": "pred"})
test = test.merge(truth[["sample_id", "nogt", "target_vessel_id"]], on="sample_id").merge(pred, on="sample_id")
catalog = pd.read_csv(W / "data/vessel_catalog.csv", encoding="utf-8-sig")
r = CatalogRetriever(catalog)
script.add_fuzzy_candidates(r)
import joblib
from src.features import FeatureBuilder
bundle = joblib.load(PKG / "model/mappers.joblib")
builder = FeatureBuilder(r, bundle["type"], bundle["flag"])
r.category_score = builder.category_score
rows = test[test.pred != test.target_vessel_id].reset_index(drop=True)
beams = script.LLMBeams()(rows["vessel_name"].tolist())
sigs = r.generate(rows, llm_beams=beams)
rows["in_cand"] = [r.id_to_index[t] in set(s["cand"].tolist()) for t, s in zip(rows.target_vessel_id, sigs)]
rows["n_cand"] = [len(s["cand"]) for s in sigs]
rows["beam0"] = [b[0] if b else "" for b in beams]
rows["truth_name"] = [catalog.vessel_name[r.id_to_index[t]] for t in rows.target_vessel_id]
rows["pred_name"] = [catalog.vessel_name[r.id_to_index[t]] for t in rows.pred]
print(rows.groupby("nogt").in_cand.agg(["mean", "size"]))
print("median n_cand", rows.n_cand.median())
pd.set_option("display.width", 250)
print(rows[["vessel_name", "nogt", "in_cand", "beam0", "truth_name", "pred_name"]].to_string())
