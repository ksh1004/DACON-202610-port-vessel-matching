"""순위 모델 재학습 실험: 주최 측 방식 한글 증강 + 학습 후보에 퍼지 후보 포함.

v17/v18 은 퍼지 후보(검색 순위 999)를 추론 때만 더하고, 순위 모델은 그런 후보를 본 적 없이 학습됐다.
여기서는 학습 행렬에도 같은 퍼지 후보를 넣고, test 한글 노이즈(음절 삭제·중복·순서 바뀜, 톤수 결측)를
흉내 낸 증강을 더해 선박 단위 5-fold 로 비교한다. 평가는 항상 추론과 같은 퍼지 500 후보로 하고,
학습 fold 선박은 후보에서 뺀다 (test 에서 train 선박을 빼는 것과 같은 상황).

    python -m tools.retrain_organizer beams      # 증강·평가 이름의 LLM 음역 생성 (GPU)
    python -m tools.retrain_organizer build      # 후보·특징 행렬 캐시
    python -m tools.retrain_organizer cv V0 V1 V2 V3
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.augment import augment, organizer_augment
from src.common import ROOT, is_hangul, load_catalog, load_train
from src.features import CATEGORY_COLUMNS, FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.fuzzy_candidates import add_fuzzy_candidates
from src.llm_cache import BeamCache
from src.retrieval import CatalogRetriever
from src.train_ranker import PARAMS
from src.validate import load_folds

PKG_LLM = ROOT.parent / "submit_v18" / "model" / "llm"  # 패키지와 같은 few-shot Kanana (미세조정 없음)
CACHE = ROOT / "experiments" / "cache" / "retrain_org"
OUT = ROOT / "experiments" / "retrain_organizer.json"
EVAL_K = 500  # v18 추론과 같은 퍼지 후보 수
ORG_COPIES, ORG_SEED = 4, 11
EVAL_COPIES, EVAL_SEED = 2, 2027

# 변형: (학습 세트 목록, 학습 fold 의 다른 train 선박을 학습 그룹에서 뺄지)
VARIANTS = {
    "V0": (["train@0", "aug@0"], False),            # 현재 방식 (퍼지 없이 학습)
    "V1": (["train@500", "aug@500"], False),        # 학습 후보에 퍼지 포함
    "V2": (["train@500", "aug@500", "org@500"], False),  # + 주최 측 방식 한글 증강
    "V3": (["train@500", "aug@500", "org@500"], True),   # + 학습 그룹에서 다른 train 선박 제외
}


def frames(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": train.assign(origin=np.arange(len(train))),
        "aug": augment(train, 4, 1, 7),  # 현재 패키지 학습과 같은 증강 (시드 7)
        "org": organizer_augment(train, ORG_COPIES, ORG_SEED),
        "evorg": organizer_augment(train, EVAL_COPIES, EVAL_SEED),
    }


def beams_cache(allow_generate: bool) -> BeamCache:
    return BeamCache(model_dir=PKG_LLM, allow_generate=allow_generate)


def cmd_beams() -> None:
    train = load_train()
    names = list(dict.fromkeys(n for df in frames(train).values() for n in df["vessel_name"] if is_hangul(n)))
    cache = beams_cache(True)
    todo = [n for n in names if n not in cache.data]
    print(f"한글 이름 {len(names)}개 중 생성 필요 {len(todo)}개", flush=True)
    t0 = time.perf_counter()
    for i, name in enumerate(todo, 1):
        cache([name])
        if i % 200 == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"{i}/{len(todo)}  {rate * 1000:.0f} ms/이름, 남은 {rate * (len(todo) - i) / 60:.1f}분", flush=True)
    cache.save()
    print("완료", flush=True)


def setup():
    train, catalog = load_train(), load_catalog()
    retrievers = {}
    for k in (0, EVAL_K):
        r = CatalogRetriever(catalog)
        builder = FeatureBuilder(r, *fit_mappers(train, catalog))
        r.category_score = builder.category_score
        if k:
            add_fuzzy_candidates(r, k)
        retrievers[k] = (r, builder)
    return train, catalog, retrievers


def cmd_build() -> None:
    train, _, retrievers = setup()
    beams = beams_cache(False)
    dfs = frames(train)
    CACHE.mkdir(parents=True, exist_ok=True)
    for key in ("train@0", "aug@0", "train@500", "aug@500", "org@500", "evorg@500"):
        path = CACHE / f"{key.replace('@', '_k')}.joblib"
        if path.exists():
            continue
        name, k = key.split("@")
        df = dfs[name]
        r, builder = retrievers[int(k)]
        t0 = time.perf_counter()
        cands, blocks = [], []
        for start in range(0, len(df), 2000):
            part = df.iloc[start:start + 2000]
            sig = r.generate(part, llm_beams=beams(part["vessel_name"].tolist()))
            for row, s in zip(part.to_dict("records"), sig):
                cands.append(s["cand"].astype(np.int32))
                blocks.append(builder.build(row, s))
        rows = sum(len(b) for b in blocks)
        print(f"{key}: {len(df)}질의, 후보 {rows:,}행 ({rows * len(FEATURE_NAMES) * 4 / 2**30:.2f}GB), "
              f"{time.perf_counter() - t0:.0f}초", flush=True)
        joblib.dump({"cands": cands, "blocks": blocks, "origin": df["origin"].to_numpy(),
                     "rows": df.drop(columns=[c for c in ("origin",) if c in df]).to_dict("records"),
                     "nogt": df["nogt"].to_numpy() if "nogt" in df else None}, path)
        del cands, blocks
        gc.collect()


def load(key: str) -> dict:
    return joblib.load(CACHE / f"{key.replace('@', '_k')}.joblib")


def stack(sets: list[dict], pick: list[np.ndarray], builder: FeatureBuilder, truth_idx: np.ndarray,
          drop: np.ndarray | None):
    """학습 행렬을 한 번에 채운다 (선종·국적 열은 fold 매퍼로 다시 계산). drop: 학습 그룹에서 뺄 카탈로그 선박."""
    parts = []
    for data, idx in zip(sets, pick):
        for i in idx:
            cand = data["cands"][i]
            y = (cand == truth_idx[data["origin"][i]])
            if not y.any():
                continue  # 정답이 후보에 없으면 순위를 배울 수 없음
            keep = None
            if drop is not None:
                keep = ~drop[cand] | y
                if keep.all():
                    keep = None
            parts.append((data, i, keep, y))
    n = sum(len(p[3]) if p[2] is None else int(p[2].sum()) for p in parts)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    Y = np.empty(n, dtype=np.float32)
    group = np.empty(len(parts), dtype=np.int32)
    offset = 0
    for g, (data, i, keep, y) in enumerate(parts):
        block, cand = data["blocks"][i], data["cands"][i]
        if keep is not None:
            block, cand, y = block[keep], cand[keep], y[keep]
        m = len(block)
        X[offset:offset + m] = block
        for col, value in zip(CATEGORY_COLUMNS, builder.category(data["rows"][i], cand)):
            X[offset:offset + m, col] = value
        Y[offset:offset + m] = y
        group[g] = m
        offset += m
    return X, Y, group


def predict(model: lgb.Booster, data: dict, idx: np.ndarray, builder: FeatureBuilder, banned: np.ndarray):
    """검증 질의마다 top-1 카탈로그 인덱스. banned(학습 fold 선박)는 후보에서 뺀다."""
    out = np.full(len(idx), -1)
    blocks = []
    for i in idx:
        block = data["blocks"][i].copy()
        for col, value in zip(CATEGORY_COLUMNS, builder.category(data["rows"][i], data["cands"][i])):
            block[:, col] = value
        blocks.append(block)
    sizes = [len(b) for b in blocks]
    nonempty = [b for b in blocks if len(b)]
    scores = model.predict(np.vstack(nonempty), num_threads=PARAMS["num_threads"]) if nonempty else []
    offset = 0
    for j, (i, size) in enumerate(zip(idx, sizes)):
        if size == 0:
            continue
        local = np.asarray(scores[offset:offset + size], dtype=np.float64)
        offset += size
        cand = data["cands"][i]
        bad = banned[cand]
        if bad.any() and not bad.all():
            local = np.where(bad, -np.inf, local)
        out[j] = cand[int(np.argmax(local))]
    return out


def cmd_cv(variants: list[str], rounds: int, threads: int) -> None:
    train, catalog, retrievers = setup()
    r, _ = retrievers[EVAL_K]
    truth_idx = train["target_vessel_id"].map(r.id_to_index).to_numpy()
    folds = load_folds(train)
    hangul = train["vessel_name"].map(is_hangul).to_numpy()
    params = {**PARAMS, "num_threads": threads}
    PARAMS["num_threads"] = threads
    ev_train, ev_org = load("train@500"), load("evorg@500")
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in variants:
        keys, filtered = VARIANTS[name]
        sets = [ev_train if key == "train@500" else load(key) for key in keys]
        pred_tr = np.full(len(train), -1)
        pred_org = np.full(len(ev_org["cands"]), -1)
        t0 = time.perf_counter()
        for k in sorted(set(folds)):
            tr_rows = np.flatnonzero(folds != k)
            builder = FeatureBuilder.__new__(FeatureBuilder)
            builder.__dict__.update(retrievers[EVAL_K][1].__dict__)
            builder.type_mapper, builder.flag_mapper = fit_mappers(train.iloc[tr_rows], catalog)
            banned = np.zeros(len(r.ids), dtype=bool)
            banned[truth_idx[tr_rows]] = True
            pick = [np.flatnonzero(np.isin(d["origin"], tr_rows)) for d in sets]
            X, Y, group = stack(sets, pick, builder, truth_idx, banned if filtered else None)
            ds = lgb.Dataset(X, Y, group=group, feature_name=FEATURE_NAMES, free_raw_data=True)
            model = lgb.train(params, ds, num_boost_round=rounds)
            n_rows = len(Y)
            del X, Y, group, ds
            gc.collect()
            va = np.flatnonzero(folds == k)
            pred_tr[va] = predict(model, ev_train, va, builder, banned)
            va_org = np.flatnonzero(np.isin(ev_org["origin"], va))
            pred_org[va_org] = predict(model, ev_org, va_org, builder, banned)
            print(f"{name} fold {k}: 학습 {n_rows:,}행, 누적 {(time.perf_counter() - t0) / 60:.1f}분", flush=True)
        ok_tr = pred_tr == truth_idx
        ok_org = pred_org == truth_idx[ev_org["origin"]]
        nogt = ev_org["nogt"].astype(bool)
        res = {
            "oof_hangul": round(float(ok_tr[hangul].mean()), 4),
            "oof_non_hangul": round(float(ok_tr[~hangul].mean()), 4),
            "org": round(float(ok_org.mean()), 4),
            "org_gt": round(float(ok_org[~nogt].mean()), 4),
            "org_nogt": round(float(ok_org[nogt].mean()), 4),
            "org_correct": int(ok_org.sum()), "org_n": int(len(ok_org)),
            "minutes": round((time.perf_counter() - t0) / 60, 1), "rounds": rounds,
        }
        results[name] = res
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        joblib.dump({"pred_tr": pred_tr, "pred_org": pred_org}, CACHE / f"pred_{name}.joblib")
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
        del sets
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["beams", "build", "cv"])
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.cmd == "beams":
        cmd_beams()
    elif args.cmd == "build":
        cmd_build()
    else:
        cmd_cv(args.variants, args.rounds, args.threads)


if __name__ == "__main__":
    main()
