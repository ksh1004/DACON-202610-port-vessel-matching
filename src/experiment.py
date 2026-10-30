"""모델·학습법 비교 실험 도구: 원본 OOF 와 '어려운 검증 세트(hard)' 점수를 함께 잰다.

hard 세트 = 각 검증 행에 노이즈(톤수 결측·오차, 오타, 약칭, 국적 결측)를 입힌 고정 복사본.
CV 가 포화(0.995)된 상황에서 test 의 노이즈에 대한 강건성을 비교하기 위한 지표다.
증강과 다른 시드로 만들고, 원본과 같은 fold 로만 평가해 누수가 없다.

    python -m src.experiment --build-hard          # hard 세트 특징·LLM 음역 캐시 생성 (최초 1회)
    python -m src.experiment --learners lgb_base lgb_big cat_yeti
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import augment, drop_word
from src.common import ROOT, is_hangul, weighted_accuracy
from src.features import FEATURE_NAMES
from src.llm_cache import BeamCache
from src.train_ranker import PARAMS, Context, build_matrix, refresh_category

HARD_SEED = 2026
HARD_ENGLISH_SAMPLE = 3000
TESTLIKE_SEED = 3031
# 제출 기록으로 역추정한 test 한글 구성 (submissions.md): 원본, 톤수 결측, 단어 생략, 둘 다
TESTLIKE_MIX = {"clean": 0.44, "gt_missing": 0.14, "drop_word": 0.23, "both": 0.19}
RESULTS = ROOT / "experiments" / "experiment_results.jsonl"
SCORES = ROOT / "experiments" / "scores"  # 학습기별 후보 점수 (오프라인 블렌딩용)


def build_hard(ctx: Context) -> pd.DataFrame:
    hard = augment(ctx.train, copies_hangul=2, copies_english=1, seed=HARD_SEED)
    hard_hangul = hard["vessel_name"].map(is_hangul) | ctx.hangul[hard["origin"].to_numpy()]
    english = hard[~hard_hangul].sample(HARD_ENGLISH_SAMPLE, random_state=HARD_SEED)
    hard = pd.concat([hard[hard_hangul], english]).sort_index().reset_index(drop=True)
    hard["sample_id"] = hard["sample_id"] + "_hard"
    return hard


def build_testlike(ctx: Context) -> pd.DataFrame:
    """train 한글 행마다 톤수 결측 / 단어 생략 / 둘 다 복사본 (원본 시나리오는 OOF 한글 그대로 사용).

    단어 생략은 행마다 고정 시드로 한 번 정해 두 시나리오가 같은 이름을 쓴다. 오타는 넣지 않는다 (train 한글에 오타가 없음).
    """
    out = []
    for pos in np.flatnonzero(ctx.hangul):
        row = dict(ctx.rows[pos])
        dropped = drop_word(row["vessel_name"], np.random.default_rng(TESTLIKE_SEED + int(pos)))
        for scenario in ("gt_missing", "drop_word", "both"):
            new = dict(row, origin=int(pos), scenario=scenario, sample_id=f"{row['sample_id']}_{scenario}")
            if scenario != "gt_missing":
                new["vessel_name"] = dropped
            if scenario != "drop_word":
                new["gross_tonnage"] = new["international_gross_tonnage"] = np.nan
            out.append(new)
    return pd.DataFrame(out)


class EvalSet:
    def __init__(self, name: str, df: pd.DataFrame, ctx: Context) -> None:
        data = build_matrix(df, ctx.retriever, ctx.builder, name=name, beams_fn=ctx.beams)
        self.df = df
        self.rows = df.to_dict("records")
        self.origin = df["origin"].to_numpy()
        self.cands, self.blocks = data["cands"], data["blocks"]
        self.truth = ctx.truth[self.origin]
        # 점수 계산용 한글 여부는 원본 기준 (약칭으로 이름이 바뀌어도 한글 기록은 한글 기록)
        self.names = ctx.train["vessel_name"].iloc[self.origin].reset_index(drop=True)


class Lab:
    def __init__(self, generate_llm: bool = False) -> None:
        # VM_AUG_SEED: 증강 시드를 바꾼 학습기(증강 배깅 앙상블 재료). 검증 세트는 시드와 무관하게 같다
        self.ctx = Context(4, 1, seed=int(os.environ.get("VM_AUG_SEED", "7")), use_llm=True,
                           allow_llm_generate=generate_llm)
        self.sets = {
            "hard": EvalSet(f"hard_s{HARD_SEED}", build_hard(self.ctx), self.ctx),
            "testlike": EvalSet(f"testlike_s{TESTLIKE_SEED}", build_testlike(self.ctx), self.ctx),
        }
        if generate_llm:
            self.ctx.beams.save()

    def folds(self):
        """fold 마다 (학습 블록, 라벨, {세트 이름: (행 번호, 블록)}) — 'oof' 는 원본 검증 fold."""
        ctx = self.ctx
        for k in sorted(set(ctx.folds)):
            tr_idx, va_idx = np.flatnonzero(ctx.folds != k), np.flatnonzero(ctx.folds == k)
            builder = ctx.fold_builder(tr_idx)
            blocks = refresh_category(ctx.blocks, ctx.cands, ctx.rows, builder)
            X = [blocks[i] for i in tr_idx]
            y = [ctx.labels[i] for i in tr_idx]
            pick = np.flatnonzero(np.isin(ctx.aug_origin, tr_idx))
            X += refresh_category([ctx.aug_blocks[i] for i in pick], [ctx.aug_cands[i] for i in pick],
                                  [ctx.aug_rows[i] for i in pick], builder)
            y += [ctx.aug_labels[i] for i in pick]
            evals = {"oof": (va_idx, [blocks[i] for i in va_idx])}
            for name, es in self.sets.items():
                idx = np.flatnonzero(np.isin(es.origin, va_idx))
                evals[name] = (idx, refresh_category([es.blocks[i] for i in idx], [es.cands[i] for i in idx],
                                                     [es.rows[i] for i in idx], builder))
            yield k, X, y, evals

    def run(self, name: str, fit_fn, save_scores: bool = True) -> dict:
        """fit_fn(X_blocks, y_blocks) → predict_fn(행렬) → 행별 점수. 세트별 후보 점수를 저장해 블렌딩에 쓴다."""
        t0 = time.perf_counter()
        scores = {s: [None] * n for s, n in self.sizes().items()}
        for k, X, y, evals in self.folds():
            predict = fit_fn(X, y)
            for s, (idx, blocks) in evals.items():
                for i, sc in zip(idx, _predict_blocks(predict, blocks)):
                    scores[s][i] = sc
        if save_scores:
            SCORES.mkdir(parents=True, exist_ok=True)
            joblib.dump(scores, SCORES / f"{name}.joblib")
        result = {"name": name, **self.evaluate(scores), "minutes": round((time.perf_counter() - t0) / 60, 1)}
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return {"result": result, "scores": scores}

    def sizes(self) -> dict[str, int]:
        return {"oof": len(self.ctx.train), **{s: len(es.df) for s, es in self.sets.items()}}

    def cands(self, s: str):
        return self.ctx.cands if s == "oof" else self.sets[s].cands

    def evaluate(self, scores: dict) -> dict:
        """세트별 후보 점수 → 지표. testlike = 0.7 × (test 구성 가중 한글) + 0.3 × OOF 비한글."""
        ctx = self.ctx
        pred = {s: np.array([c[int(np.argmax(sc))] if sc is not None and len(sc) else -1
                             for c, sc in zip(self.cands(s), scores[s])]) for s in scores}
        orig = weighted_accuracy(ctx.train["vessel_name"], pred["oof"] == ctx.truth)
        hard_set = self.sets["hard"]
        hard = weighted_accuracy(hard_set.names, pred["hard"] == hard_set.truth)
        tl = self.sets["testlike"]
        tl_ok = pred["testlike"] == tl.truth
        parts = {"clean": orig["hangul"]}
        for sc in ("gt_missing", "drop_word", "both"):
            parts[sc] = float(tl_ok[(tl.df["scenario"] == sc).to_numpy()].mean())
        tl_hangul = sum(TESTLIKE_MIX[k] * v for k, v in parts.items())
        r = lambda v: round(float(v), 4)  # noqa: E731
        return {"testlike": r(0.7 * tl_hangul + 0.3 * orig["non_hangul"]), "testlike_hangul": r(tl_hangul),
                "tl_gt": r(parts["gt_missing"]), "tl_dw": r(parts["drop_word"]), "tl_both": r(parts["both"]),
                "oof": r(orig["score"]), "oof_hangul": r(orig["hangul"]), "oof_non_hangul": r(orig["non_hangul"]),
                "hard": r(hard["score"]), "hard_hangul": r(hard["hangul"]), "hard_non_hangul": r(hard["non_hangul"])}


def _predict_blocks(predict, blocks) -> list:
    """블록 목록을 한 번에 예측해 블록별 점수 배열로 나눈다 (빈 블록은 빈 배열)."""
    out = [np.empty(0, dtype=np.float32) for _ in blocks]
    nonempty = [i for i, b in enumerate(blocks) if len(b)]
    if not nonempty:
        return out
    scores = predict(np.vstack([blocks[i] for i in nonempty]))
    offset = 0
    for i in nonempty:
        size = len(blocks[i])
        out[i] = np.asarray(scores[offset : offset + size], dtype=np.float32)
        offset += size
    return out


def _stack(X, y):
    keep = [i for i, t in enumerate(y) if t.any()]
    return (np.vstack([X[i] for i in keep]), np.concatenate([y[i] for i in keep]),
            np.array([len(y[i]) for i in keep]))


# ---- 학습기 후보 -------------------------------------------------------------
Q_HANGUL = FEATURE_NAMES.index("q_hangul")


def lgb_learner(rounds: int = 150, hangul_weight: float = 1.0, **overrides):
    """hangul_weight: 한글 질의 그룹의 가중치 (지표에서 한글 비중이 70%)."""
    def fit(X, y):
        Xs, ys, group = _stack(X, y)
        weight = np.where(Xs[:, Q_HANGUL] > 0, hangul_weight, 1.0) if hangul_weight != 1.0 else None
        model = lgb.train({**PARAMS, **overrides},
                          lgb.Dataset(Xs, ys, group=group, weight=weight, feature_name=FEATURE_NAMES),
                          num_boost_round=rounds)
        return lambda B: model.predict(B, num_threads=6)
    return fit


def lgb_binary(rounds: int = 300, **overrides):
    """점별(pointwise) 이진 분류: 순위 학습과 오차 양상이 달라 블렌딩 재료로 쓴다."""
    params = {**PARAMS, "objective": "binary", "metric": "binary_logloss", **overrides}
    params.pop("eval_at", None)
    params.pop("lambdarank_truncation_level", None)

    def fit(X, y):
        Xs, ys, _ = _stack(X, y)
        model = lgb.train(params, lgb.Dataset(Xs, ys, feature_name=FEATURE_NAMES), num_boost_round=rounds)
        return lambda B: model.predict(B, num_threads=6, raw_score=True)
    return fit


def cat_learner(loss: str = "YetiRank", iterations: int = 600, depth: int = 6, lr: float = 0.08, task_type: str = "CPU",
                seed: int = 42):
    def fit(X, y):
        from catboost import CatBoostRanker, Pool
        Xs, ys, group = _stack(X, y)
        group_id = np.repeat(np.arange(len(group)), group)
        model = CatBoostRanker(loss_function=loss, iterations=iterations, depth=depth, learning_rate=lr,
                               random_seed=seed, thread_count=int(os.environ.get("VM_THREADS", "12")),
                               verbose=False, task_type=task_type, metric_period=1000)  # 학습 지표 계산 생략
        model.fit(Pool(Xs, ys, group_id=group_id))
        return lambda B: model.predict(B)
    return fit


def blend_learner(*fits, weights=None):
    """학습기 여러 개의 점수를 질의 그룹 구분 없이 표준화해 평균 (행별 점수 순위만 쓰므로 선형 결합으로 충분)."""
    def fit(X, y):
        predictors = [f(X, y) for f in fits]
        w = weights or [1.0] * len(predictors)

        def predict(B):
            total = 0
            for wi, p in zip(w, predictors):
                s = p(B)
                total = total + wi * (s - s.mean()) / (s.std() + 1e-9)
            return total
        return predict
    return fit


LEARNERS = {
    "lgb_wh3": lambda: lgb_learner(150, hangul_weight=3.0),
    "lgb_s1": lambda: lgb_learner(150, seed=1, feature_fraction=0.7),
    "lgb_s2": lambda: lgb_learner(150, seed=2, feature_fraction=0.7),
    "lgb_bin": lambda: lgb_binary(300),
    "cat_yeti_gpu": lambda: cat_learner("YetiRank", task_type="GPU"),
    "cat_yeti400": lambda: cat_learner("YetiRank", iterations=400, lr=0.1),
    "cat_s2": lambda: cat_learner("YetiRank", iterations=400, lr=0.1, seed=2),
    # GPU (같은 데이터 크기에서 CPU 보다 약 8배 빠름)
    "catg_400": lambda: cat_learner("YetiRank", iterations=400, lr=0.1, task_type="GPU"),
    "catg_1000": lambda: cat_learner("YetiRank", iterations=1000, lr=0.05, task_type="GPU"),
    "catg_d8": lambda: cat_learner("YetiRank", iterations=600, lr=0.08, depth=8, task_type="GPU"),
    "catg_2000": lambda: cat_learner("YetiRank", iterations=2000, lr=0.03, task_type="GPU"),
    "cat_d8": lambda: cat_learner("YetiRank", iterations=400, lr=0.1, depth=8),
    "lgb_tok": lambda: lgb_learner(150),
    "lgb_tok_mix": lambda: lgb_learner(150),
    "lgb_tok_feat": lambda: lgb_learner(150),
    "lgb_base": lambda: lgb_learner(150),
    "lgb_300": lambda: lgb_learner(300),
    "lgb_big": lambda: lgb_learner(300, num_leaves=127, min_data_in_leaf=30, learning_rate=0.05),
    "lgb_small": lambda: lgb_learner(300, num_leaves=31, min_data_in_leaf=100, learning_rate=0.05),
    "lgb_xendcg": lambda: lgb_learner(150, objective="rank_xendcg"),
    "lgb_trunc50": lambda: lgb_learner(150, lambdarank_truncation_level=50),
    "cat_yeti": lambda: cat_learner("YetiRank"),
    "cat_pairlogit": lambda: cat_learner("PairLogit"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-hard", action="store_true")
    parser.add_argument("--learners", nargs="*", default=["lgb_base"])
    parser.add_argument("--tag", default="", help="결과·점수 파일 이름 뒤에 붙임 (특징 구성이 다른 실행 구분)")
    args = parser.parse_args()
    lab = Lab(generate_llm=args.build_hard)
    print({s: n for s, n in lab.sizes().items()}, flush=True)
    for name in args.learners:
        lab.run(name + args.tag, LEARNERS[name]())


if __name__ == "__main__":
    main()
