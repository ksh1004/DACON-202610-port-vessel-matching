"""Compare v11 and BGE on new strict held-out perturbations."""
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
from src.encoder_pair import EncoderPairScorer
from src.ensemble import normalize
from src.pair_views import PairViewScorer, mix_views
from tools.pair_fresh_stress import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--tag", default="encoder_bge_v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference_dir = ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{args.fold}_runtime_merged"
    if not (reference_dir / "config.json").exists():
        subprocess.run([sys.executable, "-u", "-m", "tools.check_pair_views_runtime", "--fold", str(args.fold),
                        "--variant", "evidence_only"], check=True, cwd=ROOT)
    data = prepare(args.fold)
    catalog = load_catalog().to_dict("records")
    reference = PairViewScorer(reference_dir, "evidence")
    protocol = joblib.load(ROOT / "experiments/artifacts" / args.tag / f"fold{args.fold}_epoch1_scores.joblib")["protocol"]
    if protocol.get("input_recipe") == "phonetic":
        from src.encoder_phonetics import EncoderPhoneticScorer
        scorer_class = EncoderPhoneticScorer
    else:
        scorer_class = EncoderPairScorer
    encoder = scorer_class(ROOT / "experiments/artifacts" / args.tag / f"fold{args.fold}_merged")
    predictions = {"baseline": np.array([p["candidates"][int(np.argmax(p["base"]))] for p in data["pairs"]])}
    for method in ("reference_v11", "candidate", "ensemble"):
        predictions[method] = predictions["baseline"].copy()
    calls = 0
    for i, (row, pair, beams) in enumerate(zip(data["records"], data["pairs"], data["beams"])):
        if not is_hangul(row["vessel_name"]):
            continue
        keep = np.argsort(-pair["base"], kind="stable")[:20]
        bp = pair["base"][keep]/pair["base"][keep].sum()
        if len(keep) < 2 or bp[0]-bp[1] > 1/3+1e-6:
            continue
        choices = [catalog[j] for j in pair["candidates"][keep]]
        old = reference.score(row, choices, beams)
        new = encoder.score(row, choices, beams)
        parts = {"reference_v11": .75*bp+.25*normalize(old, "prob"),
                 "candidate": .75*bp+.25*normalize(new, "prob"),
                 "ensemble": mix_views(bp, [old, new], [1, 1])}
        for method, values in parts.items():
            predictions[method][i] = pair["candidates"][keep[int(np.argmax(values))]]
        calls += 1
        if calls % 100 == 0:
            print(f"Fresh encoder: {calls} uncertain rows scored", flush=True)
    del reference, encoder
    gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(data["records"])
    correct = {k: p == data["truth"] for k, p in predictions.items()}
    scenarios = {}
    for scenario in frame.scenario.unique():
        pick = frame.scenario.eq(scenario).to_numpy()
        changed = pick & frame.name_changed.to_numpy()
        scenarios[scenario] = {"rows": int(pick.sum()), "accuracy": {m: float(ok[pick].mean()) for m, ok in correct.items()},
                               "changed_rows": int(changed.sum()),
                               "changed_only": {m: float(ok[changed].mean()) for m, ok in correct.items()} if changed.any() else None}
    _, group = np.unique(frame.target_vessel_id, return_inverse=True)
    n = int(group.max())+1
    draws = np.random.default_rng(20261029).multinomial(n, np.full(n, 1/n), size=2000)
    count = np.bincount(group, minlength=n)
    comparisons = {}
    for method in ("candidate", "ensemble"):
        difference = correct[method].astype(float)-correct["reference_v11"]
        boot = (draws@np.bincount(group, weights=difference, minlength=n))/(draws@count)
        comparisons[method] = {"delta": float(difference.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
                               "gains": int(np.sum(difference > 0)), "losses": int(np.sum(difference < 0))}
    report = {"fold": args.fold, "tag": args.tag, "rows": len(frame), "neural_rows": calls,
              "source": data["source"], "mean_accuracy": {m: float(ok.mean()) for m, ok in correct.items()},
              "scenarios": scenarios, "comparisons": comparisons,
              "candidate_delta_vs_v11": comparisons["candidate"]["delta"],
              "ensemble_delta_vs_v11": comparisons["ensemble"]["delta"],
              "caveat": "Diagnostic equal scenario mix with train-fold-only retrieval and LGB; not official score or historical testlike."}
    stem = ROOT / "experiments" / f"fresh_pairs_f{args.fold}_{args.tag}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"predictions": predictions, "sample_ids": frame.sample_id.tolist(), "source": data["source"]},
                stem.with_suffix(".joblib"), compress=3)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
