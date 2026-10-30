"""Label-aware diagnostics only: fixed-mixture reachability and exact-input collisions."""
import collections
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.common import ROOT, is_hangul, load_catalog
from src.ensemble import normalize
from src.experiment import TESTLIKE_MIX
from tools.blend_matching import lightweight_lab


def main():
    archive = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archive["identity"])
    frames = {"oof": lab.ctx.train, "testlike": lab.sets["testlike"].df}
    truth = {"oof": lab.ctx.truth, "testlike": lab.sets["testlike"].truth}
    records = {s: f.to_dict("records") for s, f in frames.items()}
    positions = {s: {v: i for i, v in enumerate(f.sample_id)} for s, f in frames.items()}
    reference = {s: archive["ensemble"][s] == truth[s] for s in frames}
    weights = [0., .125, .25, .4, .5, .75, 1.]
    possible = {str(w): {s: a.copy() for s, a in reference.items()} for w in weights}
    encoder_only = {s: a.copy() for s, a in reference.items()}
    seen = {s: np.zeros(len(f), dtype=bool) for s, f in frames.items()}
    failures = collections.Counter()
    for fold in range(5):
        enc = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_epoch1_scores.joblib")
        qwen = joblib.load(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_epoch1_scores.joblib")
        for s in frames:
            assert enc["sample_ids"][s] == qwen["sample_ids"][s]
            for sid, row, qr in zip(enc["sample_ids"][s], enc["sets"][s], qwen["sets"][s]):
                i = positions[s][sid]
                assert not seen[s][i]
                seen[s][i] = True
                assert row["truth"] == truth[s][i]
                for field in ("candidates", "keep", "base_scores"):
                    np.testing.assert_array_equal(row[field], qr[field])
                c, keep, base = row["candidates"], row["keep"], row["base_scores"]
                bcorrect = c[int(np.argmax(base))] == row["truth"]
                if not is_hangul(records[s][i]["vessel_name"]):
                    assert bcorrect == reference[s][i]
                    continue
                bp = base[keep]/base[keep].sum()
                loc = np.flatnonzero(c[keep] == row["truth"])
                for w in weights:
                    possible[str(w)][s][i] = False
                    if len(loc):
                        best = (1-w)*bp.copy()
                        best[loc[0]] += w
                        possible[str(w)][s][i] = int(np.argmax(best)) == int(loc[0])
                encoder_only[s][i] = bcorrect
                if not row["bound_pruned"]:
                    encoder_only[s][i] = False
                    if len(loc):
                        score = .75*bp+.125*normalize(qr["neural_scores"], "prob")
                        score[loc[0]] += .125
                        encoder_only[s][i] = int(np.argmax(score)) == int(loc[0])
                assert encoder_only[s][i] or not reference[s][i]
                assert possible["0.25"][s][i] or not reference[s][i]
                if len(loc) and not possible["0.25"][s][i]:
                    failures["top20_true_but_25pct_cannot_win"] += 1
                    failures["pruned" if row["bound_pruned"] else "active"] += 1
    np.testing.assert_array_equal(seen["oof"], lab.ctx.hangul)
    assert seen["testlike"].all()
    for s in frames:
        assert np.all(~encoder_only[s] | possible["0.25"][s])
        for lower, upper in zip(weights[:-1], weights[1:]):
            assert np.all(~possible[str(lower)][s] | possible[str(upper)][s])
    h = lab.ctx.hangul

    def metrics(values):
        parts = {"clean": float(values["oof"][h].mean())}
        for scenario in ("gt_missing", "drop_word", "both"):
            mask = frames["testlike"].scenario.eq(scenario).to_numpy()
            parts[scenario] = float(values["testlike"][mask].mean())
        non_han = float(values["oof"][~h].mean())
        return {"testlike": .7*sum(TESTLIKE_MIX[s]*parts[s] for s in parts)+.3*non_han,
                "scenarios": parts, "non_hangul": non_han}

    reference_metrics = metrics(reference)
    assert abs(reference_metrics["testlike"]-.9118708558226221) < 1e-12
    assert abs(metrics(possible["0.0"])["testlike"]-.9049590170316901) < 1e-12
    columns = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "vessel_type", "flag_name", "arrival_year"]
    groups = collections.defaultdict(lambda: collections.defaultdict(float))
    group_counts = collections.Counter()
    for s, rows in records.items():
        for i, row in enumerate(rows):
            if s == "oof":
                weight = .7*TESTLIKE_MIX["clean"]/h.sum() if h[i] else .3/(~h).sum()
            else:
                weight = .7*TESTLIKE_MIX[row["scenario"]]/h.sum()
            key = tuple(None if pd.isna(row.get(c)) else row.get(c) for c in columns)
            groups[key][int(truth[s][i])] += weight
            group_counts[key] += 1
    assert abs(sum(sum(g.values()) for g in groups.values())-1) < 1e-10
    collisions = sorted([(sum(g.values())-max(g.values()), k, g) for k, g in groups.items() if len(g)>1], reverse=True, key=lambda x:x[0])
    catalog = load_catalog()
    result = {"diagnostic_only_uses_labels": True, "reference_v12": reference_metrics,
              "arbitrary_reranker_oracle_by_neural_weight": {w: metrics(v) for w,v in possible.items()},
              "replace_only_bge_12_5pct_oracle": metrics(encoder_only),
              "reachability_failures": dict(failures),
              "exact_input_collision": {"columns": columns, "ambiguous_inputs": len(collisions),
                   "empirical_weighted_ceiling": 1-sum(r[0] for r in collisions),
                   "examples": [{"weighted_loss": loss, "input": dict(zip(columns,key)),
                                 "labels": [{"name": catalog.vessel_name.iloc[j], "weight": v} for j,v in labels.items()],
                                 "rows": group_counts[key]} for loss,key,labels in collisions[:8]]},
              "source_sha256": hashlib.sha256((ROOT / "tools/audit_v12_structural_limits.py").read_bytes()).hexdigest(),
              "data_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              ("train.csv", "vessel_catalog.csv", "experiments/encoder_bge_phonetic_predictions.joblib")},
              "checks": {"v12_and_v8_reproduced": True, "candidate_order_identical": True,
                         "v12_dominated_by_encoder_oracle": True, "oracle_monotonicity": True},
              "caveats": ["These are label-aware local diagnostics, not achievable scores or Public forecasts.",
                          "The weight oracles assume an arbitrary perfect distribution over fixed top20; changed weights require revised pruning.",
                          "Other neural scores stay frozen only for the BGE-only oracle; old pruning remains there.",
                          "Collision bound concerns exact observations in this reused synthetic evaluation only."]}
    (ROOT / "experiments/v12_structural_limits.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("exact_input_collision",)}, indent=2), flush=True)
    print(json.dumps({k:v for k,v in result["exact_input_collision"].items() if k != "examples"}), flush=True)


if __name__ == "__main__":
    main()
