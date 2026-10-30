"""Compare a fixed 40% neural mixture with v12 on a strict held-out diagnostic."""
import argparse
import gc
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from src.common import ROOT, is_hangul, load_catalog
from src.encoder_phonetics import EncoderPhoneticScorer
from src.pair_views import PairViewScorer, mix_views
from tools.pair_fresh_stress import prepare


def run(tag, fold):
    torch.set_num_threads(4)
    qwen_dir = ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_runtime_merged"
    if not (qwen_dir / "config.json").exists():
        subprocess.run([sys.executable, "-u", "-m", "tools.check_pair_views_runtime",
                        "--fold", str(fold), "--variant", "evidence_only"], check=True, cwd=ROOT)
    data = prepare(fold)
    catalog = load_catalog().to_dict("records")
    qwen = PairViewScorer(qwen_dir, "evidence")
    old = EncoderPhoneticScorer(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_merged")
    base = np.array([p["candidates"][int(np.argmax(p["base"]))] for p in data["pairs"]])
    predictions = {"reference_v12": base.copy(), "candidate": base.copy()}
    distributions = []
    calls = 0
    for i, (row, pair, beams) in enumerate(zip(data["records"], data["pairs"], data["beams"])):
        record = None
        if is_hangul(row["vessel_name"]):
            keep = np.argsort(-pair["base"], kind="stable")[:20]
            bp = pair["base"][keep]/pair["base"][keep].sum()
            if len(keep) > 1 and bp[0]-bp[1] <= 2/3+1e-6:
                choices = [catalog[j] for j in pair["candidates"][keep]]
                qs = qwen.score(row, choices, beams)
                os = old.score(row, choices, beams)
                for label, weight in (("reference_v12", .25), ("candidate", .4)):
                    if bp[0]-bp[1] > weight/(1-weight)+1e-6:
                        continue
                    prediction = mix_views(bp, [qs, os], [1, 1], weight)
                    predictions[label][i] = pair["candidates"][keep[int(np.argmax(prediction))]]
                record = {"keep": keep, "qwen": qs, "old": os}
                calls += 1
                if calls % 100 == 0:
                    print(f"Fresh extension: {calls} uncertain rows", flush=True)
        distributions.append(record)
    del qwen, old
    gc.collect()
    torch.cuda.empty_cache()
    frame = pd.DataFrame(data["records"])
    correct = {k: p == data["truth"] for k, p in predictions.items()}
    scenarios = {}
    for scenario in frame.scenario.unique():
        pick = frame.scenario.eq(scenario).to_numpy()
        changed = pick & frame.name_changed.to_numpy()
        scenarios[scenario] = {"rows": int(pick.sum()),
                               "accuracy": {k: float(v[pick].mean()) for k, v in correct.items()},
                               "changed_rows": int(changed.sum()),
                               "changed_only": {k: float(v[changed].mean()) for k, v in correct.items()} if changed.any() else None}
    delta = correct["candidate"].astype(float)-correct["reference_v12"]
    _, group = np.unique(frame.target_vessel_id, return_inverse=True)
    count = int(group.max())+1
    draws = np.random.default_rng(20261028).multinomial(count, np.full(count, 1/count), size=2000)
    boot = (draws@np.bincount(group, weights=delta, minlength=count))/(draws@np.bincount(group, minlength=count))
    report = {"tag": tag, "fold": fold, "rows": len(frame), "neural_rows": calls,
              "source": data["source"], "mean_accuracy": {k: float(v.mean()) for k, v in correct.items()},
              "delta_vs_v12": float(delta.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
              "gained": int((delta > 0).sum()), "lost": int((delta < 0).sum()), "scenarios": scenarios,
              "caveat": "Diagnostic equal scenario mix, not official/testlike metric. Train-only retrieval; few-shot vessels excluded."}
    stem = ROOT / "experiments" / f"fresh_extension_f{fold}_{tag}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": predictions, "distributions": distributions,
                 "sample_ids": frame.sample_id.tolist(), "source": data["source"]},
                stem.with_suffix(".joblib"), compress=3)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--tag", default="encoder_confidence40_v1")
    args = parser.parse_args()
    run(args.tag, args.fold)
