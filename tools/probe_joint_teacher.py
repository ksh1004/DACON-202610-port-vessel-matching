"""Conditional frozen4B joint-candidate probe; never a release/public score.

Uses60 distinct vessels on new-method probe folds1,4. No training, external
ship facts, callsigns, or real vessel IDs enter the model prompt.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

from src.common import ROOT, is_hangul, load_catalog, load_train
from src.llm_cache import BeamCache
from src.llm_translit import FEW_SHOT
from tools.reverse_ranker_experiment import frames_and_training

TAG = "joint_qwen4b_probe_v1"
FOLDER = ROOT / "experiments/artifacts" / TAG
PROTOCOL = ROOT / "experiments" / f"{TAG}_protocol.json"
MODEL_DIR = Path("D:/KSH/hf_models/Qwen__Qwen3-4B-Instruct-2507")
FIELDS = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type"]
SYSTEM = (
    "Match one noisy port-arrival record to a ship in the supplied candidate list. "
    "The Korean name is a phonetic rendering and may omit name words or contain typos. "
    "Latin restoration hints are uncertain, not additional candidates. Compare pronunciation, "
    "numbers, flag, ship type and tonnage jointly. Korean and English categories may express "
    "the same meaning. Both arrival GT and IGT are possible estimates of catalog GT; "
    "missing values are unknown, not mismatches. Use only the supplied record and candidates. "
    "Reply with exactly one uppercase candidate letter, with no explanation."
)


def clean(value):
    if value is None or (not isinstance(value, (str, list, dict)) and pd.isna(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def prompt(item, order):
    lines = ["Arrival: "+json.dumps(item["observed"], ensure_ascii=False, separators=(",", ":")),
             "Latin restoration hints: "+json.dumps(item["beams"], ensure_ascii=False), "Candidates:"]
    for letter, i in zip("ABCDEFGHIJKLMNOPQRST", order):
        lines.append(letter+": "+json.dumps(item["candidate_records"][i], ensure_ascii=False, separators=(",", ":")))
    lines.append("Return the matching candidate letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def prepare():
    from transformers import AutoTokenizer
    assert not PROTOCOL.exists(), "Never overwrite a preregistration silently"
    FOLDER.mkdir(parents=True, exist_ok=True)
    train, catalog = load_train(), load_catalog()
    shots = set(train.loc[train.vessel_name.isin({ko for ko, _ in FEW_SHOT}), "target_vessel_id"])
    archived = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    positions = {sid: i for i, sid in enumerate(archived["identity"]["sets"]["testlike"]["sample_ids"])}
    beams = BeamCache(allow_generate=False)
    items, counts = [], {}
    for fold in (1, 4):
        _, _, _, frames, _ = frames_and_training(fold)
        frame = frames["testlike"]
        rows = {r["sample_id"]: r for r in frame.to_dict("records")}
        ref = joblib.load(ROOT / f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
        choices = []
        for sid, old in zip(ref["sample_ids"]["testlike"], ref["sets"]["testlike"]):
            row = rows[sid]
            if row["scenario"] not in ("drop_word", "both") or not is_hangul(row["vessel_name"]):
                continue
            if row["target_vessel_id"] in shots or old["bound_pruned"]:
                continue
            choices.append((sid, row, old))
        seen = set()
        selected = []
        for i in np.random.default_rng(20261028+fold).permutation(len(choices)):
            sid, row, old = choices[i]
            if row["target_vessel_id"] in seen:
                continue
            seen.add(row["target_vessel_id"])
            c = old["candidates"][old["keep"]]
            c = np.sort(c)
            records = [{k: clean(catalog.iloc[int(j)][k]) for k in
                        ("vessel_name", "gross_tonnage", "deadweight_tonnage", "flag_name", "vessel_type")} for j in c]
            selected.append({"sample_id": sid, "fold": fold, "scenario": row["scenario"],
                             "observed": {k: clean(row[k]) for k in FIELDS},
                             "beams": beams([row["vessel_name"]])[0],
                             "candidate_records": records, "candidates": c,
                             "truth": int(old["truth"]), "v12": int(archived["ensemble"]["testlike"][positions[sid]]),
                             "vessel_id_for_dedup_only": row["target_vessel_id"]})
            if len(selected) == 30:
                break
        assert len(selected) == 30, f"fold{fold}: only {len(selected)} eligible unique vessels"
        items.extend(selected)
        counts[str(fold)] = {"eligible_rows": len(choices), "selected_vessels": len(selected)}
    assert len({i["vessel_id_for_dedup_only"] for i in items}) == 60
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    label_ids = [tokenizer.encode(c, add_special_tokens=False) for c in "ABCDEFGHIJKLMNOPQRST"]
    assert all(len(t) == 1 for t in label_ids)
    lengths = [len(tokenizer.apply_chat_template(prompt(i, order), tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False))
               for i in items for order in (list(range(len(i["candidates"]))), list(range(len(i["candidates"])-1, -1, -1)))]
    assert max(lengths) <= 2048, "No silent truncation"
    protocol = {"tag": TAG, "registered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "conditional_execution": "Only if reverse full-name and omission-marginal screens both fail",
                "model": str(MODEL_DIR), "model_config_sha256": hashlib.sha256((MODEL_DIR/"config.json").read_bytes()).hexdigest(),
                "model_index_sha256": hashlib.sha256((MODEL_DIR/"model.safetensors.index.json").read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256((MODEL_DIR/"tokenizer.json").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "system_prompt": SYSTEM, "few_shots": 0, "folds": [1, 4], "rows": 60,
                "unique_vessels": 60, "selection": "30 per fold RNG20261028+fold, actual Hangul drop/both and previously active v12, excluding fewshot vessels; first row per vessel; no truth-based eligibility",
                "counts": counts, "maximum_input_tokens": max(lengths), "max_tokens": 2048,
                "candidate_orders": "canonical catalog-index ascending and descending; average letter-conditional probabilities",
                "inference": "BF16 SDPA, batch1, final-token logits, letters A-T each one token; no generation or fine-tuning",
                "qualification": {"minimum_accuracy_gain_vs_v12": .05, "minimum_correct_rows_gain": 3},
                "actual_competition_submission": False,
                "caveat": "Small selected hard-case diagnostic, reused CV data; not official-weighted score, not full CV, no student/distillation or release"}
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(items, FOLDER/"items.joblib", compress=3)
    print(json.dumps({"prepared": True, "counts": counts, "maximum_input_tokens": max(lengths)}), flush=True)


def evaluate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for tag in ("reverse_kanana_v1", "reverse_kanana_partial_v1"):
        result = json.loads((ROOT/"experiments"/f"{tag}_audit.json").read_text())
        assert result["selected"] is None
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    items = joblib.load(FOLDER/"items.joblib")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()
    model.config.use_cache = False
    labels = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHIJKLMNOPQRST"]
    stored = []
    start = time.perf_counter()

    @torch.inference_mode()
    def score(item):
        n = len(item["candidates"])
        probabilities, masses = [], []
        for order in (list(range(n)), list(range(n-1, -1, -1))):
            ids = tokenizer.apply_chat_template(prompt(item, order), tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
            assert len(ids) <= 2048
            x = torch.tensor([ids], device="cuda")
            logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            scores = logits[labels[:n]].cpu().numpy()
            p = softmax(scores.astype(np.float64))
            canonical = np.empty(n, dtype=float)
            canonical[order] = p
            probabilities.append(canonical)
            masses.append(float(torch.exp(torch.logsumexp(logits[labels[:n]], 0)-torch.logsumexp(logits, 0)).item()))
        return np.mean(probabilities, axis=0), masses

    for i, item in enumerate(items):
        p, masses = score(item)
        stored.append({"sample_id": item["sample_id"], "fold": item["fold"], "truth": item["truth"],
                       "v12": item["v12"], "teacher": int(item["candidates"][np.argmax(p)]),
                       "probabilities": p, "letter_probability_mass": masses})
        if (i+1) % 10 == 0:
            print(f"Joint4B probe {i+1}/{len(items)}, seconds {time.perf_counter()-start:.1f}", flush=True)
    repeat, _ = score(items[0])
    np.testing.assert_array_equal(repeat, stored[0]["probabilities"])
    old_ok = np.array([r["v12"] == r["truth"] for r in stored])
    new_ok = np.array([r["teacher"] == r["truth"] for r in stored])
    gain = int(new_ok.sum()-old_ok.sum())
    report = {"tag": TAG, "rows": len(items), "unique_vessels": 60,
              "reference_v12_accuracy": float(old_ok.mean()), "teacher_accuracy": float(new_ok.mean()),
              "correct_rows_gain": gain, "gained": int(np.sum(~old_ok & new_ok)), "lost": int(np.sum(old_ok & ~new_ok)),
              "qualifies_for_larger_teacher_test": bool(gain >= 3 and new_ok.mean()-old_ok.mean() >= .05-1e-12),
              "accuracy_by_fold": {str(k): {"v12": float(old_ok[np.array([r["fold"] == k for r in stored])].mean()),
                                             "teacher": float(new_ok[np.array([r["fold"] == k for r in stored])].mean())} for k in (1, 4)},
              "seconds": time.perf_counter()-start, "exact_repeat": True,
              "minimum_letter_probability_mass": float(min(m for r in stored for m in r["letter_probability_mass"])),
              "actual_competition_submission": False, "caveat": protocol["caveat"]}
    joblib.dump(stored, FOLDER/"predictions.joblib", compress=3)
    (ROOT/"experiments"/f"{TAG}_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else evaluate()
