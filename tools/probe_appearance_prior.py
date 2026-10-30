"""Read-only probe: catalog-level 'appearance prior' on top of train-vessel exclusion.

For each vessel-grouped fold k, a LightGBM classifier learns P(catalog vessel appears in records)
from catalog attributes only. Positives = vessels of training folds (all rows / hangul rows);
everything else, including fold-k vessels, is negative — the same situation as the real test.
Candidates of fold-k eval rows get  log p_v8 + a * logit_all + b * logit_hangul(hangul queries).

    python -m tools.probe_appearance_prior
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blend import blend
from src.common import is_hangul
from src.experiment import SCORES, Lab

MEMBERS = {"cat_yeti400_prior": 3, "cat_yeti400_prior_a8": 3, "cat_yeti400_prior_a9": 3,
           "lgb_base_prior": 1, "lgb_base_prior_a8": 1}


def catalog_features(cat: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=cat.index)
    for col in ("vessel_type", "flag_name", "unit"):
        f[col] = cat[col].astype("category")
    for col in ("size", "deadweight_tonnage", "gross_tonnage"):
        v = pd.to_numeric(cat[col], errors="coerce")
        f["log_" + col] = np.log1p(v)
    f["dwt_gt"] = f["log_deadweight_tonnage"] - f["log_gross_tonnage"]
    name = cat["vessel_name"].astype(str)
    f["name_len"] = name.str.len()
    f["n_words"] = name.str.split().str.len()
    f["has_digit"] = name.str.contains(r"\d").astype(int)
    f["pos"] = np.arange(len(cat)) / len(cat)
    return f


def fit_logit(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    model.fit(X, y)
    return model.predict(X, raw_score=True)


def main() -> None:
    lab = Lab()
    ctx = lab.ctx
    cat = ctx.retriever.catalog
    X = catalog_features(cat)
    hangul_row = ctx.train["vessel_name"].map(is_hangul).to_numpy()
    scores = blend([joblib.load(SCORES / f"{m}.joblib") for m in MEMBERS], list(MEMBERS.values()), "prob")

    folds = sorted(set(ctx.folds))
    excl, logit_all, logit_han = {}, {}, {}
    for k in folds:
        tr = ctx.folds != k
        seen = np.zeros(len(cat), dtype=bool)
        seen[ctx.truth[tr]] = True
        han = np.zeros(len(cat), dtype=bool)
        han[ctx.truth[tr & hangul_row]] = True
        excl[k] = seen
        logit_all[k] = fit_logit(X, seen.astype(int))
        logit_han[k] = fit_logit(X, han.astype(int))
        print(f"fold {k}: pos {seen.sum()} / hangul pos {han.sum()}", flush=True)

    def combined(a: float, b: float) -> dict:
        out = {}
        for s, rows in scores.items():
            origin = np.arange(len(ctx.train)) if s == "oof" else lab.sets[s].origin
            queries = ctx.train["vessel_name"] if s == "oof" else lab.sets[s].df["vessel_name"]
            res = []
            for i, (cand, sc) in enumerate(zip(lab.cands(s), rows)):
                if sc is None or len(sc) == 0:
                    res.append(sc)
                    continue
                k = ctx.folds[origin[i]]
                cand = np.asarray(cand)
                z = np.log(np.asarray(sc, dtype=np.float64) + 1e-9) + a * logit_all[k][cand]
                if b and (hangul_row[origin[i]] or is_hangul(queries.iloc[i])):
                    z = z + b * logit_han[k][cand]
                bad = excl[k][cand]
                if bad.any() and not bad.all():
                    z = np.where(bad, -np.inf, z)
                res.append(z)
            out[s] = res
        return out

    results = []
    for a in (0.0, 0.1, 0.2, 0.3, 0.5):
        for b in (0.0, 0.1, 0.2, 0.3, 0.5):
            r = lab.evaluate(combined(a, b))
            results.append({"a": a, "b": b, **r})
            print(json.dumps(results[-1]), flush=True)
    with open("experiments/appearance_prior_probe.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
