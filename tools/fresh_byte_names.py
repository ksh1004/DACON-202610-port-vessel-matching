"""Same strict baseline and a new byte-name expert on fixed held-out noise."""
import gc
import json

import numpy as np
import pandas as pd
import torch

from src.byte_name_model import ByteNameScorer, normalise, probability
from src.common import ROOT, is_hangul, load_catalog
from src.encoder_phonetics import EncoderPhoneticScorer
from src.pair_views import PairViewScorer, mix_views
from tools.byte_names_experiment import WEIGHT, check_lock, dump, fingerprint
from tools.prepare_byte_names import FOLDER, TAG


def run():
    protocol = check_lock(); fold = protocol["fresh_guard"]["fold"]
    # Reuse the frozen noise/retrieval implementation in a new private folder.
    # No previous experiment artifacts or mutable global beam cache are changed.
    from tools import fresh_ambiguity_guard as helper
    helper.FOLDER, helper.check_lock, helper.fingerprint = FOLDER, check_lock, fingerprint
    data = helper.prepare(fold)
    catalog = load_catalog().to_dict("records")
    from tools.nested_combiner_experiment import outer_qwen
    qwen = PairViewScorer(outer_qwen(fold), "evidence")
    bge = EncoderPhoneticScorer(ROOT/f"experiments/artifacts/encoder_bge_phonetic/fold{fold}_merged")
    byte = ByteNameScorer(FOLDER/f"fold{fold}_full/model")
    predictions = {k: np.array([p["candidates"][np.argmax(p["base"])] for p in data["pairs"]])
                   for k in ("reference_v12", "candidate")}
    calls = 0
    for i, (row, pair, beams) in enumerate(zip(data["records"], data["pairs"], data["beams"])):
        if not is_hangul(row["vessel_name"]): continue
        keep = np.argsort(-pair["base"], kind="stable")[:20]
        bp = normalise(pair["base"][keep]); choices = [catalog[int(j)] for j in pair["candidates"][keep]]
        anchor = bp
        if len(keep)>1 and bp[0]-bp[1] <= 1/3+1e-6:
            anchor = normalise(mix_views(bp, [qwen.score(row, choices, beams), bge.score(row, choices, beams)], [1, 1]))
        winner = int(pair["candidates"][keep[np.argmax(anchor)]])
        predictions["reference_v12"][i] = predictions["candidate"][i] = winner
        if len(keep)>1 and np.sort(anchor)[-1]-np.sort(anchor)[-2] <= WEIGHT/(1-WEIGHT)+1e-6:
            scores = byte.score(row["vessel_name"], [c["vessel_name"] for c in choices])
            predictions["candidate"][i] = pair["candidates"][keep[np.argmax((1-WEIGHT)*anchor+WEIGHT*probability(scores))]]
            calls += 1
            if calls%50 == 0: print(f"Fresh byte rows{calls}", flush=True)
    correct = {k: p == data["truth"] for k, p in predictions.items()}
    delta = correct["candidate"].astype(float)-correct["reference_v12"]
    frame = pd.DataFrame(data["records"])
    scenarios = {s: {k: float(ok[frame.scenario.eq(s)].mean()) for k, ok in correct.items()} for s in frame.scenario.unique()}
    report = {"tag": TAG, "fold": fold, "rows": len(frame), "byte_calls": calls,
              "mean_accuracy": {k: float(v.mean()) for k, v in correct.items()}, "delta_vs_v12": float(delta.mean()),
              "gained": int((delta>0).sum()), "lost": int((delta<0).sum()), "scenarios": scenarios,
              "source": data["source"], "fingerprint": fingerprint(),
              "caveat": "Equal seven-scenario diagnostic on previously used vessels; strict training-only LGB baseline/retrieval, not official CV or Public performance."}
    (ROOT/"experiments"/f"{TAG}_fresh.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dump({"predictions": predictions, "source": data["source"]}, FOLDER/"fresh_predictions.joblib")
    print(json.dumps(report, indent=2), flush=True)
    del qwen, bge, byte; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__": run()
