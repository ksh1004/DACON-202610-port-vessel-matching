"""Screen constrained-name likelihood on the fixed 60-row pilot sample."""
import json
import time

import numpy as np

from src.closed_name_score import ClosedNameScorer
from src.common import ROOT
from src.ensemble import normalize
from src.llm_cache import MODEL_DIR


def main():
    source = json.loads((ROOT / "experiments" / "candidate_probe_kanana.json").read_text(encoding="utf-8"))
    cases = source["cases"]
    scorer = ClosedNameScorer(MODEL_DIR)
    start = time.perf_counter()
    first = None
    for i, case in enumerate(cases):
        values = scorer.score(case["record"]["vessel_name"], [o["vessel_name"] for o in case["options"]])
        case["closed_logprob"] = values.tolist()
        if first is None:
            first = values.copy()
        for method, scores in (("closed_mean", values[:, 0]), ("closed_sum", values[:, 1]),
                               ("closed_blend25", .75 * np.array(case["baseline_probs"]) + .25 * normalize(values[:, 0], "prob"))):
            case[method + "_prediction"] = case["option_ids"][int(np.argmax(scores))]
        if (i + 1) % 10 == 0:
            print(f"closed score {i + 1}/{len(cases)} {time.perf_counter() - start:.1f}s", flush=True)
    again = scorer.score(cases[0]["record"]["vessel_name"], [o["vessel_name"] for o in cases[0]["options"]])
    np.testing.assert_array_equal(first, again)
    summary = {"n": len(cases), "elapsed_seconds": time.perf_counter() - start,
               "repeat_after_other_rows": "identical", "scenarios": {}}
    for scenario in ("gt_missing", "drop_word", "both"):
        part = [c for c in cases if c["scenario"] == scenario]
        summary["scenarios"][scenario] = {method: sum(c[method + "_prediction"] == c["truth_id"] for c in part)
                                           for method in ("baseline", "closed_mean", "closed_sum", "closed_blend25")}
    (ROOT / "experiments" / "closed_score_probe.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
