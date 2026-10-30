"""Predeclared omission marginal: no unpenalized best-substring selection."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import time

import joblib
import lightgbm as lgb
import numpy as np
from scipy.special import logsumexp

from src.common import ROOT
from src.ensemble import normalize
from src.features import FEATURE_NAMES
from src.train_ranker import PARAMS
from tools import reverse_ranker_experiment as base

TAG = "reverse_kanana_partial_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG
PROTOCOL = ROOT / "experiments" / f"{TAG}_protocol.json"
EXTRA = ["partial_sum_logp", "partial_mean_logp", "partial_sum_with_eos",
         "partial_gap_to_best", "partial_dense_rank", "partial_full_posterior"]


def variants(name):
    words = name.split()
    spans = []
    for i in range(len(words)):
        for j in range(i+1, len(words)+1):
            if i == 0 and j == len(words):
                continue
            spans.append((" ".join(words[i:j]), len(words)-(j-i)))
    if not spans:
        return [(name, 1.0)]
    # Fixed full-name prior0.5; remaining0.5 is normalized over omission paths.
    unnormalized = np.exp(-np.array([n for _, n in spans], dtype=float))
    weights = .5*unnormalized/unnormalized.sum()
    merged = {name: .5}
    for (s, _), w in zip(spans, weights):
        merged[s] = merged.get(s, 0.0)+float(w)
    return sorted(merged.items())


def partial_pairs(data):
    return {(q, span) for q, c in base.all_pairs(data) for span, _ in variants(c)}


def register():
    assert not PROTOCOL.exists(), "Never silently overwrite a preregistration"
    protocol = json.loads(base.PROTOCOL.read_text())
    protocol.update({"tag": TAG, "registered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "conditional_execution": "Only if complete-name screen has no qualifying contender",
                     "partial_recipe": "All nonempty contiguous word spans; full-name prior0.5; proper spans collectively0.5 weighted exp(-omitted_words), normalized within candidate",
                     "new_features": base.EXTRA+EXTRA,
                     "matched_control": "same groups/raw78 with complete-name reverse features; tests incremental omission marginal",
                     "partial_score": "logsumexp(log prior + observed-token sum logp); separately with EOS; no max and no validation parameter search"})
    PROTOCOL.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Registered {PROTOCOL}", flush=True)


def score(data):
    import torch
    from src.reverse_name_likelihood import ReverseNameScorer
    torch.set_num_threads(4)
    scorer = ReverseNameScorer()
    original = joblib.load(base.FOLDER / "pair_scores.joblib")
    assert original["protocol"] == scorer.protocol()
    path = FOLDER / "pair_scores.joblib"
    cache = joblib.load(path) if path.exists() else original
    assert cache["protocol"] == scorer.protocol()
    missing = sorted(partial_pairs(data)-cache["scores"].keys())
    print(f"Partial pairs: missing {len(missing)}, cached {len(cache['scores'])}", flush=True)
    start = time.perf_counter()
    for offset in range(0, len(missing), 256):
        chunk = missing[offset:offset+256]
        cache["scores"].update(zip(chunk, scorer.score_many_pairs(chunk)))
        if offset % 2048 == 0 or offset+256 >= len(missing):
            joblib.dump(cache, path, compress=3)
            print(f"Partial {min(offset+256,len(missing))}/{len(missing)}, seconds {time.perf_counter()-start:.1f}", flush=True)
    cache["last_partial_seconds"] = time.perf_counter()-start
    joblib.dump(cache, path, compress=3)
    return cache


def features(g, scores):
    values = []
    for name in g["names"]:
        v = variants(name)
        lp = np.array([scores[(g["name"], n)] for n, _ in v])
        prior = np.log([w for _, w in v])
        total = float(logsumexp(prior+lp[:, 0]))
        eos = float(logsumexp(prior+lp[:, 0]+lp[:, 2]))
        full = next(i for i, (n, _) in enumerate(v) if n == name)
        posterior = float(np.exp(prior[full]+lp[full, 0]-total))
        values.append((total, total/lp[0, 3], eos, posterior))
    v = np.array(values)
    rank = np.searchsorted(np.unique(-v[:, 1]), -v[:, 1])+1
    extra = np.column_stack((v[:, :3], v[:, 1]-v[:, 1].max(), rank, v[:, 3]))
    return np.column_stack((base.features(g, scores), extra)).astype(np.float32)


def train_and_evaluate(data, cache):
    # The control is exactly the previously frozen complete-name result.
    for d in data:
        fold = d["fold"]
        control = joblib.load(base.FOLDER / f"fold{fold}_results.joblib")
        train = d["train"]
        x = np.vstack([features(g, cache["scores"]) for g in train])
        y = np.concatenate([g["candidates"] == g["truth"] for g in train]).astype(int)
        model = lgb.train(PARAMS, lgb.Dataset(x, label=y, group=[len(g["candidates"]) for g in train],
                                             feature_name=FEATURE_NAMES+base.EXTRA+EXTRA), num_boost_round=150)
        model.save_model(str(FOLDER / f"fold{fold}_partial.txt"))
        stored = {"fold": fold, "sets": {}, "controls": control["sets"],
                  "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
        for s, groups in d["evaluation"].items():
            stored["sets"][s] = []
            for g in groups:
                scores = None
                pred = {"direct": None, "base25_head75": None}
                if g["actual_hangul"]:
                    scores = model.predict(features(g, cache["scores"]), num_threads=4)
                    bp = g["base"]/g["base"].sum()
                    pred = {"direct": int(g["candidates"][np.argmax(scores)]),
                            "base25_head75": int(g["candidates"][np.argmax(.25*bp+.75*normalize(scores,"prob"))])}
                stored["sets"][s].append({"sample_id": g["sample_id"], "truth": g["truth"],
                                          "predictions": pred, "scores": scores,
                                          "truth_in_top20": bool(g["truth"] in g["candidates"])})
        joblib.dump(stored, FOLDER / f"fold{fold}_results.joblib", compress=3)
        print(f"Saved partial fold{fold}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["register", "budget", "all"], default="all")
    args = p.parse_args()
    if args.stage == "register":
        register()
        return
    data = [base.prepare(k) for k in (0, 2)]
    if args.stage == "budget":
        pairs = partial_pairs(data)
        full = base.all_pairs(data)
        print(json.dumps({"full_unique": len(full), "with_partial_unique": len(pairs|full), "additional_pairs": len(pairs-full)}), flush=True)
        return
    original_audit = json.loads((ROOT / "experiments/reverse_kanana_v1_audit.json").read_text())
    assert original_audit["selected"] is None, "Do not search alternatives after a qualifying full-name result"
    FOLDER.mkdir(parents=True, exist_ok=True)
    assert PROTOCOL.exists()
    cache = score(data)
    train_and_evaluate(data, cache)
    from tools.audit_reverse_ranker import audit
    audit(FOLDER, PROTOCOL)


if __name__ == "__main__":
    main()
