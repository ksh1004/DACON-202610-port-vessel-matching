"""Actual checkpoint restore or saved-scorer row/candidate invariance checks."""
import argparse
import gc
import json

import joblib
import numpy as np
import torch

from src.common import ROOT, is_hangul, load_catalog
from tools.ambiguity_encoder_experiment import ARMS, FOLDER, TAG, check_lock, fingerprint, folder


def checkpoint():
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification
    from src.encoder_phonetic_experiment import MODEL_DIR

    check_lock()
    cp = folder("none") / "fold1_checkpoints/step000200"
    assert (cp / "ready.json").exists()
    state = torch.load(cp / "state.pt", map_location="cpu", weights_only=False)
    assert state["fingerprint"] == fingerprint()
    torch.set_num_threads(4)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, cp, is_trainable=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names == state["trainable_names"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    optimizer.load_state_dict(state["optimizer"])
    saved = load_file(str(cp / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    assert set(saved) == set(loaded)
    for name in saved:
        torch.testing.assert_close(saved[name], loaded[name], rtol=0, atol=0)
    for item in optimizer.state.values():
        for value in item.values():
            if torch.is_tensor(value): assert torch.isfinite(value).all()
    assert state["completed"] == 200 and state["torch_rng"].dtype == torch.uint8
    assert len(state["cuda_rng"]) == 1 and state["cuda_rng"][0].dtype == torch.uint8
    report = {"status": "PASS", "step": 200, "trainable_parameters": len(names),
              "adapter_restore_exact": True, "optimizer_restore_finite": True, "random_states_present": True,
              "scope": "Actual checkpoint load on CPU; not a second full training or exact CUDA continuation comparison"}
    (ROOT / "experiments" / f"{TAG}_checkpoint_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def runtime(folds):
    from src.encoder_phonetics import EncoderPhoneticScorer
    from src.llm_cache import BeamCache

    check_lock(); torch.set_num_threads(4)
    catalog = load_catalog().to_dict("records")
    beams = BeamCache(allow_generate=False)
    reports = []
    for fold in folds:
        data = joblib.load(FOLDER / f"fold{fold}_data.joblib")
        eligible = [(s, i, g) for s, rows in data["evaluation"].items() for i, g in enumerate(rows)
                    if is_hangul(g["row"]["vessel_name"]) and not g["reference"]["bound_pruned"]]
        chosen = [eligible[i] for i in (0, len(eligible)//2, len(eligible)-1)]
        for arm in ARMS:
            scores = joblib.load(folder(arm) / f"fold{fold}_epoch1_scores.joblib")
            scorer = EncoderPhoneticScorer(folder(arm) / f"fold{fold}_merged")
            expected = []
            for s, i, g in chosen:
                row, ref = g["row"], g["reference"]
                candidate_rows = [catalog[int(j)] for j in ref["candidates"][ref["keep"]]]
                bb = beams([row["vessel_name"]])[0]
                saved = scores["sets"][s][i]["neural_scores"]
                fresh = scorer.score(row, candidate_rows, bb)
                np.testing.assert_array_equal(fresh, saved)
                perm = np.arange(len(candidate_rows))[::-1]
                np.testing.assert_array_equal(scorer.score(row, [candidate_rows[int(j)] for j in perm], bb), fresh[perm])
                changed = {**row, "sample_id": "ignored", "call_sign": "ignored", "target_vessel_id": "ignored"}
                np.testing.assert_array_equal(scorer.score(changed, candidate_rows, bb), fresh)
                expected.append((row, candidate_rows, bb, fresh))
            for row, candidate_rows, bb, fresh in reversed(expected):
                np.testing.assert_array_equal(scorer.score(row, candidate_rows, bb), fresh)
            row, candidate_rows, bb, fresh = expected[1]
            np.testing.assert_array_equal(scorer.score(row, candidate_rows, bb), fresh)
            reports.append({"fold": fold, "arm": arm, "actual_queries": len(expected),
                            "save_reload_exact": True, "row_reverse_subset_single_exact": True,
                            "candidate_permutation_exact": True, "id_and_callsign_ignored": True})
            del scorer
            gc.collect(); torch.cuda.empty_cache()
    report = {"status": "PASS", "fingerprint": fingerprint(), "checks": reports,
              "scope": "Actual saved BGE scorer and immutable per-query candidate lists; not a complete submission ZIP execution"}
    (ROOT / "experiments" / f"{TAG}_runtime_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--folds", nargs="+", type=int, default=[1, 4]); args = p.parse_args()
    checkpoint() if args.checkpoint else runtime(args.folds)
