"""스트레스 테스트: 검증 fold 의 질의에 test 에서 있을 법한 변형을 가하고 정확도 하락을 잰다.

Public(test)과 CV 의 차이 원인을 제출 없이 추정하고, 증강의 효과를 확인하기 위한 도구.

    python -m src.stress                                  # 증강 없는 모델
    python -m src.stress --aug-hangul 4 --aug-english 1   # 증강한 모델
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.augment import drop_word, keep_first, keep_last, no_space, truncate, typo_double, typo_drop, typo_vowel
from src.train_ranker import Context


def _names(fn):
    return lambda df, rng: df.assign(vessel_name=[fn(n, rng) for n in df["vessel_name"]])


SCENARIOS = {
    "original": lambda df, rng: df,
    "gt_missing": lambda df, rng: df.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan),
    "gt_noise_5pct": lambda df, rng: df.assign(
        gross_tonnage=df["gross_tonnage"] * rng.uniform(0.95, 1.05, len(df)),
        international_gross_tonnage=df["international_gross_tonnage"] * rng.uniform(0.95, 1.05, len(df))),
    "flag_missing": lambda df, rng: df.assign(flag_name=np.nan),
    "no_space": _names(no_space),
    "typo_vowel": _names(typo_vowel),
    "typo_drop": _names(typo_drop),
    "typo_double": _names(typo_double),
    "drop_word": _names(drop_word),
}
SCENARIOS["gt_missing+drop_word"] = lambda df, rng: SCENARIOS["drop_word"](SCENARIOS["gt_missing"](df, rng), rng)


# 약칭 방식이 무엇인지 모르므로 여러 방식을 따로 잰다
SCENARIOS["keep_first"] = _names(keep_first)
SCENARIOS["keep_last"] = _names(keep_last)
SCENARIOS["truncate"] = _names(truncate)
SCENARIOS["gt_missing+keep_last"] = lambda df, rng: SCENARIOS["keep_last"](SCENARIOS["gt_missing"](df, rng), rng)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["hangul", "non_hangul", "all"], default="hangul")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--aug-hangul", type=int, default=0)
    parser.add_argument("--aug-english", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="대상 행을 이 수만큼만 표본 추출 (속도용, 0=전체)")
    parser.add_argument("--llm", action="store_true", help="LLM 음역 사용 (변형된 이름은 즉석 생성 후 캐시)")
    args = parser.parse_args()

    ctx = Context(args.aug_hangul, args.aug_english, use_llm=args.llm, allow_llm_generate=args.llm)
    target = {"hangul": ctx.hangul, "non_hangul": ~ctx.hangul, "all": np.ones_like(ctx.hangul)}[args.target].copy()
    if args.max_rows and target.sum() > args.max_rows:
        keep = np.random.default_rng(0).choice(np.flatnonzero(target), size=args.max_rows, replace=False)
        target[:] = False
        target[keep] = True
    correct = {name: np.zeros(len(ctx.train), dtype=bool) for name in SCENARIOS}
    recalled = {name: np.zeros(len(ctx.train), dtype=bool) for name in SCENARIOS}

    for k in sorted(set(ctx.folds)):
        tr_idx = np.flatnonzero(ctx.folds != k)
        va_idx = np.flatnonzero((ctx.folds == k) & target)
        builder = ctx.fold_builder(tr_idx)
        model, _ = ctx.fit(tr_idx, args.rounds, builder)
        valid = ctx.train.iloc[va_idx].reset_index(drop=True)
        for name, fn in SCENARIOS.items():
            changed = fn(valid.copy(), np.random.default_rng(1000 + k))
            beams = ctx.beams(changed["vessel_name"].tolist()) if ctx.beams is not None else None
            signals = ctx.retriever.generate(changed, llm_beams=beams)
            for j, (row, sig) in enumerate(zip(changed.to_dict("records"), signals)):
                i = va_idx[j]
                recalled[name][i] = ctx.truth[i] in sig["cand"]
                if len(sig["cand"]):
                    scores = model.predict(builder.build(row, sig), num_threads=6)
                    correct[name][i] = sig["cand"][int(np.argmax(scores))] == ctx.truth[i]
        if ctx.beams is not None:
            ctx.beams.save()
        print(f"fold {k} 완료", flush=True)

    report = pd.DataFrame({
        "recall": {n: recalled[n][target].mean() for n in SCENARIOS},
        "accuracy": {n: correct[n][target].mean() for n in SCENARIOS},
    })
    report["drop_vs_original"] = report["accuracy"] - report.loc["original", "accuracy"]
    print(f"\n== 대상: {args.target} ({target.sum()}행), 증강 한글 {args.aug_hangul} / 영문 {args.aug_english}")
    print(report.round(4))


if __name__ == "__main__":
    main()
