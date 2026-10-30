"""LightGBM LambdaRank 재순위 모델: 선박 단위 5-fold OOF 평가 후 전체 train 으로 최종 학습.

    python -m src.train_ranker --out experiments/artifacts/ranker_v4 --aug-hangul 4 --aug-english 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import DROP_WORD_PROB, augment
from src.common import ROOT, is_hangul, load_catalog, load_train, weighted_accuracy
from src.llm_cache import BeamCache
from src.features import CATEGORY_COLUMNS, FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.retrieval import RETRIEVAL_VERSION, CatalogRetriever
from src.validate import load_folds

CACHE = ROOT / "experiments" / "cache"
# 증강의 약칭 방식: 무작위 단어 생략만(0, 기본) / 여러 방식 혼합(1). 혼합은 hard 0.837 → 0.832 로 이득 없음
ABBREV_MIX = os.environ.get("VM_ABBREV_MIX", "0") == "1"
# 증강 노이즈 강도 (src.augment.PROFILES): default / testmix
AUG_PROFILE = os.environ.get("VM_AUG_PROFILE", "default")
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [1],
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 6,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 42,
    "verbose": -1,
}


def build_matrix(df: pd.DataFrame, retriever: CatalogRetriever, builder: FeatureBuilder, name: str = "train",
                 beams_fn=None):
    """질의들의 후보와 특징을 만들고 캐시한다 (선종·국적 열은 fold 마다 다시 계산).

    beams_fn: 이름 목록 → LLM 음역 N-best 목록 (None 이면 LLM 신호 없이)
    """
    llm_tag = getattr(beams_fn, "path", None)
    key = ",".join(FEATURE_NAMES) + str(llm_tag and Path(llm_tag).name) + RETRIEVAL_VERSION
    tag = hashlib.md5(key.encode()).hexdigest()[:8]
    cache = CACHE / f"{name}_features_{tag}.joblib"  # 특징 목록, LLM 캐시, 검색 방식이 바뀌면 새로 만든다
    if cache.exists():
        return joblib.load(cache)
    t0 = time.perf_counter()
    beams = beams_fn(df["vessel_name"].tolist()) if beams_fn is not None else None
    signals = retriever.generate(df, llm_beams=beams)
    rows = df.to_dict("records")
    blocks = [builder.build(row, sig) for row, sig in zip(rows, signals)]
    print(f"[{name}] 후보·특징 생성 {time.perf_counter() - t0:.1f}초 ({len(df)}행)")
    data = {"cands": [s["cand"] for s in signals], "blocks": blocks}
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, cache)
    return data


def refresh_category(blocks, cands, rows, builder: FeatureBuilder) -> list[np.ndarray]:
    out = []
    for block, cand, row in zip(blocks, cands, rows):
        block = block.copy()
        for col, value in zip(CATEGORY_COLUMNS, builder.category(row, cand)):
            block[:, col] = value
        out.append(block)
    return out


def to_dataset(blocks, labels, weights=None):
    """정답이 후보에 있는 질의만 학습에 쓴다 (정답이 없으면 순위를 배울 수 없음)."""
    keep = [i for i, y in enumerate(labels) if y.any()]
    X = np.vstack([blocks[i] for i in keep])
    y = np.concatenate([labels[i] for i in keep])
    group = [len(labels[i]) for i in keep]
    w = None if weights is None else np.concatenate([np.full(len(labels[i]), weights[i]) for i in keep])
    return lgb.Dataset(X, y, group=group, weight=w, feature_name=FEATURE_NAMES, free_raw_data=False)


def top1(model: lgb.Booster, blocks, cands) -> np.ndarray:
    sizes = [len(c) for c in cands]
    scores = model.predict(np.vstack(blocks), num_threads=PARAMS["num_threads"])
    out, offset = [], 0
    for cand, size in zip(cands, sizes):
        local = scores[offset : offset + size]
        out.append(cand[int(np.argmax(local))])  # 동점이면 앞선 후보 (검색 순서) → 결정적
        offset += size
    return np.array(out)


class Context:
    """원본 train 과 증강 행의 후보·특징을 한 번만 만들어 두고 fold 학습에 재사용한다."""

    def __init__(self, aug_hangul: int = 0, aug_english: int = 0, seed: int = 7, use_llm: bool = False,
                 allow_llm_generate: bool = False) -> None:
        self.train, self.catalog = load_train(), load_catalog()
        self.beams = BeamCache(allow_generate=allow_llm_generate) if use_llm else None
        self.folds = load_folds(self.train)
        self.retriever = CatalogRetriever(self.catalog)
        self.builder = FeatureBuilder(self.retriever, *fit_mappers(self.train, self.catalog))
        # 검색 단계의 선종·국적 순서는 전체 train 매퍼로 고정 (fold 마다 후보를 다시 만들지 않기 위해).
        # 선종 19종 → 카탈로그 선종 대응표라 검증 선박 1/5 의 영향은 미미하다
        self.retriever.category_score = self.builder.category_score
        self.rows = self.train.to_dict("records")
        self.truth = self.train["target_vessel_id"].map(self.retriever.id_to_index).to_numpy()
        self.hangul = self.train["vessel_name"].map(is_hangul).to_numpy()
        data = build_matrix(self.train, self.retriever, self.builder, beams_fn=self.beams)
        self.cands, self.blocks = data["cands"], data["blocks"]
        self.labels = [(c == t).astype(np.float32) for c, t in zip(self.cands, self.truth)]

        self.aug = (augment(self.train, aug_hangul, aug_english, seed, abbrev_mix=ABBREV_MIX, profile=AUG_PROFILE)
                    if (aug_hangul or aug_english) else None)
        if self.aug is not None:
            aug_data = build_matrix(self.aug, self.retriever, self.builder,
                                    name=f"aug_h{aug_hangul}_e{aug_english}_s{seed}_d{DROP_WORD_PROB}"
                                         + ("_mix" if ABBREV_MIX else "")
                                         + ("" if AUG_PROFILE == "default" else f"_{AUG_PROFILE}"),
                                    beams_fn=self.beams)
            self.aug_rows = self.aug.to_dict("records")
            self.aug_origin = self.aug["origin"].to_numpy()
            self.aug_cands, self.aug_blocks = aug_data["cands"], aug_data["blocks"]
            aug_truth = self.truth[self.aug_origin]
            self.aug_labels = [(c == t).astype(np.float32) for c, t in zip(self.aug_cands, aug_truth)]

    def fold_builder(self, train_idx: np.ndarray) -> FeatureBuilder:
        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.__dict__.update(self.builder.__dict__)
        builder.type_mapper, builder.flag_mapper = fit_mappers(self.train.iloc[train_idx], self.catalog)
        return builder

    def fit(self, train_idx: np.ndarray, rounds: int, builder: FeatureBuilder):
        """원본 train_idx 행 + 원본이 train_idx 에 속한 증강 행으로 학습. 원본 행의 특징을 함께 돌려준다."""
        blocks = refresh_category(self.blocks, self.cands, self.rows, builder)
        X = [blocks[i] for i in train_idx]
        y = [self.labels[i] for i in train_idx]
        if self.aug is not None:
            pick = np.flatnonzero(np.isin(self.aug_origin, train_idx))
            aug_blocks = refresh_category([self.aug_blocks[i] for i in pick], [self.aug_cands[i] for i in pick],
                                          [self.aug_rows[i] for i in pick], builder)
            X += aug_blocks
            y += [self.aug_labels[i] for i in pick]
        model = lgb.train(PARAMS, to_dataset(X, y), num_boost_round=rounds)
        return model, blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "artifacts" / "ranker")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--aug-hangul", type=int, default=0, help="한글 행당 증강 복사본 수")
    parser.add_argument("--aug-english", type=int, default=0, help="비한글 행당 증강 복사본 수")
    parser.add_argument("--no-final", action="store_true", help="OOF 평가만 하고 최종 모델은 만들지 않음")
    parser.add_argument("--llm", action="store_true", help="LLM 음역 신호 사용 (src.llm_cache 로 미리 생성한 캐시 필요)")
    args = parser.parse_args()

    ctx = Context(args.aug_hangul, args.aug_english, use_llm=args.llm)
    recall = np.array([y.any() for y in ctx.labels])
    print(f"후보 recall: 비한글 {recall[~ctx.hangul].mean():.4f}, 한글 {recall[ctx.hangul].mean():.4f}")
    if ctx.aug is not None:
        aug_recall = np.mean([y.any() for y in ctx.aug_labels])
        print(f"증강 행 {len(ctx.aug)}개, 증강 행 후보 recall {aug_recall:.4f}")

    oof = np.full(len(ctx.train), -1)
    for k in sorted(set(ctx.folds)):
        tr_idx, va_idx = np.flatnonzero(ctx.folds != k), np.flatnonzero(ctx.folds == k)
        # early stopping 은 검증 fold 로 반복 수를 고르므로 OOF 가 부풀려진다 → 반복 수 고정
        model, blocks = ctx.fit(tr_idx, args.rounds, ctx.fold_builder(tr_idx))
        oof[va_idx] = top1(model, [blocks[i] for i in va_idx], [ctx.cands[i] for i in va_idx])
        fold_score = weighted_accuracy(ctx.train["vessel_name"].iloc[va_idx], oof[va_idx] == ctx.truth[va_idx])
        print(f"fold {k}: 점수 {fold_score['score']:.4f}  "
              f"한글 {fold_score['hangul']:.4f}  비한글 {fold_score['non_hangul']:.4f}", flush=True)

    result = weighted_accuracy(ctx.train["vessel_name"], oof == ctx.truth)
    print(f"OOF 전체: 점수 {result['score']:.4f}  한글 {result['hangul']:.4f}  비한글 {result['non_hangul']:.4f}")
    if args.llm:  # few-shot 예시로 쓴 선박은 LLM 이 정답을 본 셈이므로 빼고도 확인
        from src.llm_translit import FEW_SHOT
        clean = ~ctx.train["vessel_name"].isin({ko for ko, _ in FEW_SHOT}).to_numpy()
        r = weighted_accuracy(ctx.train["vessel_name"][clean], (oof == ctx.truth)[clean])
        print(f"OOF (few-shot 선박 제외): 점수 {r['score']:.4f}  한글 {r['hangul']:.4f}  비한글 {r['non_hangul']:.4f}")
    if args.no_final:
        return

    model, _ = ctx.fit(np.arange(len(ctx.train)), args.rounds, ctx.builder)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.out / "ranker.txt"))
    joblib.dump({"type": ctx.builder.type_mapper, "flag": ctx.builder.flag_mapper, "feature_names": FEATURE_NAMES},
                args.out / "mappers.joblib")
    importance = pd.Series(model.feature_importance("gain"), index=FEATURE_NAMES).sort_values(ascending=False)
    summary = {"oof": result, "n_rounds": args.rounds, "aug_hangul": args.aug_hangul,
               "aug_english": args.aug_english, "top_features": importance.head(15).round(1).to_dict()}
    (ROOT / "experiments" / f"{args.out.name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": ctx.train["sample_id"], "pred_index": oof, "truth_index": ctx.truth}).to_csv(
        ROOT / "experiments" / f"oof_{args.out.name}.csv", index=False)
    print(f"저장: {args.out}  (반복 {args.rounds})")
    print("중요 특징 (gain):", importance.head(10).round(0).to_dict())


if __name__ == "__main__":
    main()
