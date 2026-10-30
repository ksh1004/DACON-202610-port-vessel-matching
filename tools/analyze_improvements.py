"""Audit saved OOF scores without training, test data, or model changes.

Run from the project root: python -m tools.analyze_improvements
The bootstrap is paired and clustered by target vessel; intervals are exploratory
because these validation sets have already been used for model selection.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.blend import blend
from src.common import ROOT
from src.experiment import Lab, SCORES, TESTLIKE_MIX
from src.features import FEATURE_NAMES
from src.llm_translit import FEW_SHOT


def main() -> None:
    lab = Lab()
    ctx = lab.ctx
    names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "lgb_base_prior", "lgb_base_prior_a8"]
    raw = [joblib.load(SCORES / f"{name}.joblib") for name in names]
    # Saved score files have no candidate IDs. Shape checks are necessary, but
    # matching candidate order additionally relies on their shared cache lineage.
    for name, scores in zip(names, raw):
        if set(scores) != set(lab.sizes()):
            raise ValueError(f"Evaluation sets differ: {name}")
        for s, size in lab.sizes().items():
            if len(scores[s]) != size or any(sc is None or len(sc) != len(c)
                                            for sc, c in zip(scores[s], lab.cands(s))):
                raise ValueError(f"Score/candidate mismatch: {name}/{s}")

    specs = {"v7": ([0, 2, 3], [3, 1, 1]), "cats_equal": ([0, 1], [1, 1]),
             "four_equal": ([0, 1, 2, 3], [1, 1, 1, 1]),
             "four_2211": ([0, 1, 2, 3], [2, 2, 1, 1]),
             "cat_a8": ([1], [1]), "a8_311": ([1, 2, 3], [3, 1, 1])}
    score_sets = {name: blend([raw[i] for i in idx], weights, "prob")
                  for name, (idx, weights) in specs.items()}
    rows = {"oof": ctx.train, **{s: es.df for s, es in lab.sets.items()}}
    missing = {s: (~(pd.to_numeric(df["gross_tonnage"], errors="coerce") > 0)
                   & ~(pd.to_numeric(df["international_gross_tonnage"], errors="coerce") > 0)).to_numpy()
               for s, df in rows.items()}
    score_sets["missing_2211_else_v7"] = {
        s: [new if flag else old for flag, new, old in zip(missing[s], score_sets["four_2211"][s],
                                                          score_sets["v7"][s])]
        for s in rows}
    predictions = {name: {s: np.array([c[int(np.argmax(sc))] if len(c) else -1
                                     for c, sc in zip(lab.cands(s), scores[s])])
                          for s in rows} for name, scores in score_sets.items()}
    blocks = {"oof": ctx.blocks, **{s: es.blocks for s, es in lab.sets.items()}}
    exact_col = FEATURE_NAMES.index("llm_exact_any")
    override = {s: pred.copy() for s, pred in predictions["v7"].items()}
    for s in rows:
        for i, (c, block) in enumerate(zip(lab.cands(s), blocks[s])):
            hits = np.flatnonzero(block[:, exact_col] > 0)
            if missing[s][i] and len(hits) == 1:
                override[s][i] = c[hits[0]]
    predictions["unique_llm_exact_missing"] = override
    truth = {"oof": ctx.truth, **{s: es.truth for s, es in lab.sets.items()}}
    correct = {name: {s: p == truth[s] for s, p in preds.items()} for name, preds in predictions.items()}

    def per_origin(ok):
        out = ok["oof"].astype(float)
        out[ctx.hangul] *= TESTLIKE_MIX["clean"]
        tl = lab.sets["testlike"]
        for sc, weight in TESTLIKE_MIX.items():
            if sc == "clean":
                continue
            mask = (tl.df["scenario"] == sc).to_numpy()
            np.add.at(out, tl.origin[mask], weight * ok["testlike"][mask])
        return out

    def weighted(values, mask):
        return float(0.7 * values[mask & ctx.hangul].mean()
                     + 0.3 * values[mask & ~ctx.hangul].mean())

    contributions = {name: per_origin(ok) for name, ok in correct.items()}
    all_rows = np.ones(len(ctx.train), dtype=bool)
    shot_ids = ctx.train.loc[ctx.train["vessel_name"].isin({ko for ko, _ in FEW_SHOT}),
                             "target_vessel_id"].unique()
    no_shots = ~ctx.train["target_vessel_id"].isin(shot_ids).to_numpy()
    groups, group_index = np.unique(ctx.truth, return_inverse=True)
    n_groups = len(groups)
    rng = np.random.default_rng(20261026)
    draws = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=2000)
    h_count = np.bincount(group_index, weights=ctx.hangul, minlength=n_groups)
    n_count = np.bincount(group_index, weights=~ctx.hangul, minlength=n_groups)
    h_den, n_den = draws @ h_count, draws @ n_count
    summary = {}
    for name, values in contributions.items():
        delta = values - contributions["v7"]
        h_delta = np.bincount(group_index, weights=delta * ctx.hangul, minlength=n_groups)
        n_delta = np.bincount(group_index, weights=delta * ~ctx.hangul, minlength=n_groups)
        boot = 0.7 * (draws @ h_delta) / h_den + 0.3 * (draws @ n_delta) / n_den
        metrics = lab.evaluate(score_sets[name]) if name in score_sets else None
        if metrics is not None and abs(metrics["testlike"] - weighted(values, all_rows)) > 0.000051:
            raise AssertionError(f"Per-origin metric reconstruction failed: {name}")
        summary[name] = {"metrics": metrics, "testlike_exact": weighted(values, all_rows),
                         "delta_vs_v7": weighted(delta, all_rows),
                         "delta_cluster_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                         "testlike_excluding_shot_vessels": weighted(values, no_shots),
                         "delta_by_fold": {str(k): weighted(delta, ctx.folds == k)
                                           for k in sorted(set(ctx.folds))}}

    diagnostics, errors = {}, []
    sets = [("clean", "oof", ctx.hangul)] + [
        (sc, "testlike", (lab.sets["testlike"].df["scenario"] == sc).to_numpy())
        for sc in ("gt_missing", "drop_word", "both")]
    for scenario, s, mask in sets:
        ranks = []
        origin = np.arange(len(ctx.train)) if s == "oof" else lab.sets[s].origin
        for i in np.flatnonzero(mask):
            c, sc = lab.cands(s)[i], score_sets["v7"][s][i]
            order = np.argsort(-sc, kind="stable")
            hit = np.flatnonzero(c[order] == truth[s][i])
            rank = int(hit[0]) + 1 if len(hit) else 10**9
            ranks.append(rank)
            if rank != 1:
                p = predictions["v7"][s][i]
                errors.append({"scenario": scenario, "sample_id": rows[s].iloc[i]["sample_id"],
                               "query": rows[s].iloc[i]["vessel_name"],
                               "truth": ctx.retriever.names[truth[s][i]],
                               "prediction": ctx.retriever.names[p] if p >= 0 else "",
                               "truth_rank": rank if len(hit) else None,
                               "fold": int(ctx.folds[origin[i]]),
                               "cat_a8_correct": bool(correct["cat_a8"][s][i])})
        ranks = np.array(ranks)
        base_ok = correct["v7"][s][mask]
        a8_ok = correct["cat_a8"][s][mask]
        model_union = np.any(np.stack([correct[m][s][mask] for m in specs]), axis=0)
        diagnostics[scenario] = {"n": int(mask.sum()), "accuracy": float(base_ok.mean()),
                                 "candidate_recall": float((ranks < 10**9).mean()),
                                 "top2": float((ranks <= 2).mean()), "top5": float((ranks <= 5).mean()),
                                 "retrieval_misses": int((ranks == 10**9).sum()),
                                 "rerank_errors": int(((ranks > 1) & (ranks < 10**9)).sum()),
                                 "a8_fixes": int((~base_ok & a8_ok).sum()),
                                 "a8_breaks": int((base_ok & ~a8_ok).sum()),
                                 "saved_model_selection_oracle": float(model_union.mean())}
        if scenario in ("drop_word", "both"):
            original_names = ctx.train["vessel_name"].to_numpy()[origin[mask]]
            changed = rows[s].loc[mask, "vessel_name"].to_numpy() != original_names
            diagnostics[scenario]["actually_changed"] = {
                "n": int(changed.sum()), "accuracy": float(base_ok[changed].mean()),
                "cat_a8_accuracy": float(a8_ok[changed].mean()),
                "top5": float((ranks[changed] <= 5).mean())}
            diagnostics[scenario]["unchanged"] = {
                "n": int((~changed).sum()), "accuracy": float(base_ok[~changed].mean())}

    collisions = {}
    for scenario, s, mask in sets:
        frame = rows[s].loc[mask].copy()
        frame["truth"] = truth[s][mask]
        fields = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type"]
        for year in (False, True):
            keys = fields + (["arrival_year"] if year else [])
            counts = frame.groupby(keys + ["truth"], dropna=False).size()
            maximum = counts.groupby(level=list(range(len(keys))), dropna=False).max().sum()
            collisions[f"{scenario}{'_with_year' if year else ''}"] = float(maximum / len(frame))

    result = {"bootstrap_replicates": 2000, "bootstrap_unit": "target_vessel_id",
              "score_alignment": "shared-cache lineage plus set/length checks; saved files lack IDs",
              "excluded_shot_vessels": len(shot_ids), "excluded_shot_hangul_rows": int((~no_shots & ctx.hangul).sum()),
              "catalog_columns": ctx.catalog.columns.tolist(),
              "target_vessels_spanning_call_sign_folds": int(
                  ctx.train.assign(fold=ctx.folds).groupby("target_vessel_id")["fold"].nunique().gt(1).sum()),
              "summary": summary, "diagnostics": diagnostics,
              "max_testlike_gain_from_fixing_all_hangul_retrieval_misses": float(sum(
                  0.7 * TESTLIKE_MIX[sc] * (1 - item["candidate_recall"])
                  for sc, item in diagnostics.items())),
              "empirical_collision_ceiling_on_this_sample_only": collisions}
    out = ROOT / "experiments" / "improvement_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(errors).to_csv(ROOT / "experiments" / "improvement_errors.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
