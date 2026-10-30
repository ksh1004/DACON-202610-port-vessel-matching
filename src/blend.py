"""저장된 학습기별 후보 점수(experiments/scores/*.joblib)를 조합해 블렌딩을 오프라인으로 평가한다.

점수는 질의 1행의 후보 그룹 안에서만 정규화하므로 (z 점수 / 확률 / 순위) 행 독립 원칙에 맞는다.

    python -m src.blend lgb_base+lgb_bin "lgb_base*2+lgb_bin" --modes z prob rank
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from src.ensemble import normalize
from src.experiment import RESULTS, SCORES, Lab


def blend(score_sets: list[dict], weights: list[float], mode: str) -> dict:
    out = {}
    for s in score_sets[0]:
        rows = []
        for parts in zip(*(ss[s] for ss in score_sets)):
            if parts[0] is None or len(parts[0]) == 0:
                rows.append(parts[0])
                continue
            rows.append(sum(w * normalize(p, mode) for w, p in zip(weights, parts)))
        out[s] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("combos", nargs="+", help="'a+b+c' 형식, 가중치는 'a*2+b'")
    parser.add_argument("--modes", nargs="*", default=["z"], choices=["z", "prob", "rank"])
    args = parser.parse_args()
    lab = Lab()
    cache: dict[str, dict] = {}
    for combo in args.combos:
        models, weights = [], []
        for part in combo.split("+"):
            name, _, w = part.partition("*")
            models.append(name)
            weights.append(float(w) if w else 1.0)
        for m in models:
            if m not in cache:
                cache[m] = joblib.load(SCORES / f"{m}.joblib")
        for mode in args.modes:
            result = {"name": f"blend:{combo}:{mode}", **lab.evaluate(blend([cache[m] for m in models], weights, mode))}
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{result['name']:60s} TL {result['testlike']:.4f} h {result['testlike_hangul']:.4f} "
                  f"gt {result['tl_gt']:.4f} dw {result['tl_dw']:.4f} both {result['tl_both']:.4f} | "
                  f"oof {result['oof']:.4f} hard {result['hard']:.4f}", flush=True)


if __name__ == "__main__":
    main()
