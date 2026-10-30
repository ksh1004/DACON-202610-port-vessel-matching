"""Audit the 0.97 target and optionally probe listwise LLM matching.

Uses official training/catalog rows and existing held-out ranker scores only.
No identity lookup, test access, model training, or submission is performed.
The small probe is a feasibility screen, not a new validation score.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.blend import blend
from src.common import ROOT
from src.experiment import Lab, SCORES, TESTLIKE_MIX
from src.llm_translit import FEW_SHOT


SYSTEM = (
    "Match one port-arrival record to one of the listed ship catalog entries. "
    "Use only the fields provided, never outside vessel facts. "
    "The record name may be a Korean transliteration, misspelling, or abbreviated form of the English name. "
    "Compare the pronunciation, remaining words, numbers, tonnage, vessel type and flag together. "
    "Tonnage can be missing or slightly inaccurate; flag can differ. "
    "Missing fields are not evidence of disagreement. "
    "Return only the single option letter for the most likely matching ship."
)


def fields(row, catalog=False):
    keys = ["vessel_name", "gross_tonnage", "flag_name", "vessel_type"]
    if not catalog:
        keys += ["international_gross_tonnage", "arrival_year"]
    return {k: None if pd.isna(row.get(k)) else row.get(k) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model", default="D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505")
    parser.add_argument("--per-scenario", type=int, default=20)
    parser.add_argument("--tag", default="kanana")
    args = parser.parse_args()
    if args.per_scenario < 1:
        parser.error("--per-scenario must be positive")
    lab = Lab()
    ctx = lab.ctx
    member_names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9",
                    "lgb_base_prior", "lgb_base_prior_a8"]
    raw = [joblib.load(SCORES / f"{name}.joblib") for name in member_names]
    for name, scores in zip(member_names, raw):
        for s, size in lab.sizes().items():
            assert len(scores[s]) == size, (name, s)
            assert all(sc is not None and len(sc) == len(c) for sc, c in zip(scores[s], lab.cands(s))), (name, s)
    configs = {"v7": ([0, 3, 4], [3, 1, 1]), "v8": ([0, 1, 2, 3, 4], [3, 3, 3, 1, 1])}
    scores = {name: blend([raw[i] for i in idx], weights, "prob")
              for name, (idx, weights) in configs.items()}
    del raw
    tl = lab.sets["testlike"]
    shot_vessels = set(ctx.train.loc[ctx.train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"])
    report = {"date": "2026-10-27", "target": 0.97,
              "test_rows": "not disclosed by official data page", "public_fraction": 0.5,
              "required_hangul_if_non_hangul_0_9981": (0.97 - 0.3 * 0.9981) / 0.7,
              "fraction_of_public_v7_errors_to_fix": (0.97 - 0.8998) / (1 - 0.8998),
              "oracles": {}, "note": "Oracles use validation truth and are unattainable selection rules, not predictions."}
    for name, parts in scores.items():
        ranks = {}
        for s in ("oof", "testlike"):
            truth = ctx.truth if s == "oof" else tl.truth
            rr = []
            for c, sc, target in zip(lab.cands(s), parts[s], truth):
                hit = np.flatnonzero(c[np.argsort(-sc, kind="stable")] == target)
                rr.append(int(hit[0]) + 1 if len(hit) else 10**9)
            ranks[s] = np.array(rr)
        non = float((ranks["oof"][~ctx.hangul] == 1).mean())
        rr = {}
        for k in (1, 2, 5, 10, 20, 50, 10**9 - 1):
            acc = {"clean": float((ranks["oof"][ctx.hangul] <= k).mean())}
            for scenario in ("gt_missing", "drop_word", "both"):
                mask = tl.df.scenario.eq(scenario).to_numpy()
                acc[scenario] = float((ranks["testlike"][mask] <= k).mean())
            rr[str(k) if k < 10**8 else "all_candidates"] = {
                "testlike": 0.7 * sum(TESTLIKE_MIX[s] * acc[s] for s in TESTLIKE_MIX) + 0.3 * non,
                "hangul_scenarios": acc}
        report["oracles"][name] = rr
        report.setdefault("metrics", {})[name] = lab.evaluate(parts)
    audit_path = ROOT / "experiments" / "target97_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.probe:
        return

    # Choose rows without looking at predictions/correctness. One record per
    # target vessel in each stratum; few-shot vessels are excluded entirely.
    cases = []
    for scenario in ("gt_missing", "drop_word", "both"):
        idx = np.flatnonzero(tl.df.scenario.eq(scenario).to_numpy())
        rng = np.random.default_rng(20261027)
        used = set()
        for i in rng.permutation(idx):
            origin = tl.origin[i]
            vessel = ctx.train.iloc[origin].target_vessel_id
            if vessel in shot_vessels or vessel in used:
                continue
            if scenario != "gt_missing" and tl.rows[i]["vessel_name"] == ctx.rows[origin]["vessel_name"]:
                continue
            c, sc = tl.cands[i], scores["v7"]["testlike"][i]
            top = np.argsort(-sc, kind="stable")[:10]
            # Stable catalog order conceals the baseline rank from the LLM.
            top = top[np.argsort(c[top], kind="stable")]
            options = c[top]
            base = sc[top].astype(float)
            base /= base.sum()
            cases.append({"scenario": scenario, "sample_id": tl.rows[i]["sample_id"],
                          "record": fields(tl.rows[i]),
                          "options": [fields(ctx.catalog.iloc[j].to_dict(), catalog=True) for j in options],
                          "option_ids": ctx.retriever.ids[options].tolist(),
                          "truth_id": ctx.retriever.ids[tl.truth[i]],
                          "baseline_probs": base.tolist()})
            used.add(vessel)
            if len(used) >= args.per_scenario:
                break
    del lab, ctx, tl, scores
    gc.collect()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    letter_ids = [tokenizer.encode(chr(65 + j), add_special_tokens=False) for j in range(10)]
    if any(len(ids) != 1 for ids in letter_ids):
        raise ValueError("Option letters must each be a single token")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                torch_dtype=torch.bfloat16).to("cuda").eval()
    start = time.perf_counter()
    for n, case in enumerate(cases, 1):
        prompt = "Record: " + json.dumps(case["record"], ensure_ascii=False) + "\nCandidates:\n"
        prompt += "\n".join(chr(65 + j) + ": " + json.dumps(option, ensure_ascii=False)
                            for j, option in enumerate(case["options"]))
        prompt += "\nBest option letter:"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        enc.pop("token_type_ids", None)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        ids = [v[0] for v in letter_ids[:len(case["options"])]]
        llm_probs = torch.softmax(logits[ids], dim=0).cpu().numpy()
        base = np.array(case["baseline_probs"])
        case["llm_probs"] = llm_probs.tolist()
        for method, values in (("baseline", base), ("llm", llm_probs), ("blend_25pct_llm", 0.75 * base + 0.25 * llm_probs)):
            case[method + "_prediction"] = case["option_ids"][int(np.argmax(values))]
        if n % 10 == 0:
            print(f"Probe {n}/{len(cases)} ({time.perf_counter() - start:.1f}s)", flush=True)
    elapsed = time.perf_counter() - start
    summary = {"model": args.model, "n": len(cases), "top_k": 10,
               "sample_seed": 20261027, "batch_size": 1, "elapsed_seconds": elapsed,
               "seconds_per_row": elapsed / len(cases),
               "method": "single-token option scoring; catalog-ordered options; no baseline rank shown",
               "warning": "Small stratified screen, not testlike/public/CV or an independence certificate.",
               "scenarios": {}}
    for scenario in ("gt_missing", "drop_word", "both"):
        part = [c for c in cases if c["scenario"] == scenario]
        summary["scenarios"][scenario] = {"n": len(part), "candidate_recall": float(np.mean(
            [c["truth_id"] in c["option_ids"] for c in part]))}
        for method in ("baseline", "llm", "blend_25pct_llm"):
            summary["scenarios"][scenario][method] = float(np.mean(
                [c[method + "_prediction"] == c["truth_id"] for c in part]))
    (ROOT / "experiments" / f"candidate_probe_{args.tag}.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
