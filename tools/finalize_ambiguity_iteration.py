"""Verify and collect loss-trial evidence without changing its decisions."""
import datetime
import json

import joblib
import numpy as np

from src.common import ROOT, is_hangul
from src.pair_views import mix_views
from tools.ambiguity_encoder_experiment import ARMS, FOLDER, TAG, check_lock, fingerprint, folder, sha


def main():
    protocol = check_lock()
    decision = json.loads((ROOT / "experiments" / f"{TAG}_selection.json").read_text())
    runtime = json.loads((ROOT / "experiments" / f"{TAG}_runtime_checks.json").read_text())
    assert runtime["status"] == "PASS" and runtime["fingerprint"] == fingerprint()
    setup = json.loads((ROOT / "experiments" / f"{TAG}_setup_checks.json").read_text())
    assert setup["status"] == "PASS" and setup["fingerprint"] == fingerprint()
    checkpoint = json.loads((ROOT / "experiments" / f"{TAG}_checkpoint_checks.json").read_text())
    assert checkpoint["status"] == "PASS"
    training = []
    replay = {}
    for fold in protocol["screen_folds"]:
        data = joblib.load(FOLDER / f"fold{fold}_data.joblib")
        initial, tokens = set(), set()
        for arm in ARMS:
            record = json.loads((folder(arm) / f"fold{fold}_training.json").read_text())
            assert record["fingerprint"] == fingerprint() and record["finite_loss"]
            initial.add(record["initial_trainable_sha256"]); tokens.add(record["token_sha256"])
            training.append(record)
        assert len(initial) == len(tokens) == 1
        new = joblib.load(folder("none") / f"fold{fold}_epoch1_scores.joblib")
        old = joblib.load(ROOT / f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_epoch1_scores.joblib")
        qwen = joblib.load(ROOT / f"experiments/artifacts/neural_evidence_v1/fold{fold}_epoch1_scores.joblib")
        report = {"rows": 0, "active_hangul": 0, "bitwise_logits_equal": 0,
                  "v12_winners_preserved": 0, "max_logit_delta": 0.}
        for source, rows in new["sets"].items():
            for i, (n, o, g) in enumerate(zip(rows, old["sets"][source], data["evaluation"][source])):
                report["rows"] += 1
                if not is_hangul(g["row"]["vessel_name"]) or n["bound_pruned"]:
                    report["v12_winners_preserved"] += 1
                    continue
                report["active_hangul"] += 1
                report["bitwise_logits_equal"] += int(np.array_equal(n["neural_scores"], o["neural_scores"]))
                report["max_logit_delta"] = max(report["max_logit_delta"], float(np.max(np.abs(n["neural_scores"]-o["neural_scores"]))))
                keep = n["keep"]; bp = n["base_scores"][keep]; bp = bp/bp.sum()
                qs = qwen["sets"][source][i]["neural_scores"]
                winners = [n["candidates"][keep[np.argmax(mix_views(bp, [qs, score], [1, 1]))]]
                           for score in (n["neural_scores"], o["neural_scores"])]
                report["v12_winners_preserved"] += int(winners[0] == winners[1])
        replay[str(fold)] = report
    frozen = json.loads((ROOT / "experiments/artifacts/ranker_v12_encoder/encoder_views.json").read_text())
    for name, expected in frozen["source_hashes"].items():
        assert sha(ROOT / "src" / name) == expected, name
    release = json.loads((ROOT / "experiments/release_v12_encoder.json").read_text())
    archive = ROOT / release["zip"]
    assert archive.stat().st_size == release["bytes"] and sha(archive) == release["sha256"]
    result = {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "client_date": "2026-10-29", "tag": TAG, "decision": decision,
              "protocol_fingerprint": fingerprint(), "source_and_input_files_verified": len(protocol["source_hashes"]),
              "frozen_v12_modules_verified": len(frozen["source_hashes"]), "v12_zip_sha256_verified": release["sha256"],
              "screen_training_records": training, "retrained_unmasked_vs_historical": replay,
              "runtime_checks": runtime, "checkpoint_checks": checkpoint,
              "actual_public": .9065951735, "actual_server_seconds": 591, "actual_submissions": 9,
              "actual_competition_submission": False, "callsign_or_identity_lookup": False,
              "notes": ["Retrained unmasked control is not an exact replay of historical v12 weights or predictions.",
                        "All new arms shared verified initialization and token groups; the historical replay difference remains an uncertainty.",
                        "No coefficient, mask-rule or candidate search was performed after validation.",
                        "Runtime checks exercise actual saved scorers, not a new full submission ZIP."]}
    path = ROOT / "experiments" / f"{TAG}_final.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(path), "selected": decision["selected"], "stage": decision["stage"],
                      "frozen_v12_preserved": True, "new_screen_neural_models": len(training)}, indent=2))


if __name__ == "__main__": main()
