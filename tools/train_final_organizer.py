"""retrain_organizer 의 학습 방식으로 전체 train 에 최종 기본 앙상블을 학습하고 제출 패키지를 만든다.

구성은 v18 과 같다: CatBoost YetiRank 400 ×3 (가중 3) + LightGBM LambdaRank 150 ×2 (가중 1), 그룹 내 softmax.
v18 은 증강 시드 7·8·9 로 멤버를 다양화했지만, 여기서는 같은 데이터에 모델 시드만 바꾼다.
신경망 재판별(Qwen·BGE)과 script.py 는 v18 그대로 쓴다.

    python -m tools.train_final_organizer V2 --pkg ../submit_v19 --work E:/dacon_tmp/v19
    (중단 후 이어서: 같은 명령에 --reuse)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT, load_catalog, load_train
from src.features import FEATURE_NAMES, fit_mappers
from src.train_ranker import PARAMS
from tools.retrain_organizer import VARIANTS, load, setup, stack

V18 = ROOT.parent / "submit_v18"
MEMBERS = [  # (이름, 종류, 시드, 가중치) — v18 ensemble.json 과 같은 순서·가중치
    ("cat_yeti400", "cat", 42, 3.0),
    ("cat_yeti400@8", "cat", 8, 3.0),
    ("cat_yeti400@9", "cat", 9, 3.0),
    ("lgb_base", "lgb", 42, 1.0),
    ("lgb_base@8", "lgb", 8, 1.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("--pkg", type=Path, required=True, help="만들 패키지 폴더 (v18 을 복사해 기본 앙상블만 교체)")
    parser.add_argument("--work", type=Path, required=True, help="모델 파일을 쓸 영문 경로 (LightGBM 은 한글 경로 불가)")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--reuse", action="store_true", help="--work 에 이미 있는 멤버 파일은 다시 학습하지 않는다")
    args = parser.parse_args()

    train, catalog, retrievers = setup()
    r, builder = retrievers[500]
    truth_idx = train["target_vessel_id"].map(r.id_to_index).to_numpy()
    keys, filtered = VARIANTS[args.variant]
    sets = [load(k) for k in keys]
    banned = np.zeros(len(r.ids), dtype=bool)
    banned[truth_idx] = True
    X, Y, group = stack(sets, [np.arange(len(d["cands"])) for d in sets], builder, truth_idx,
                        banned if filtered else None)
    del sets
    print(f"학습 행렬 {X.shape}, 그룹 {len(group)}", flush=True)

    args.work.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (name, kind, seed, weight) in enumerate(MEMBERS):
        t0 = time.perf_counter()
        base = name.partition("@")[0]
        file = f"m{i}_{base}{'' if seed == 42 else f'_s{seed}'}.{'cbm' if kind == 'cat' else 'txt'}"
        if args.reuse and (args.work / file).exists():
            print(f"{name}: 기존 파일 재사용", flush=True)
        elif kind == "cat":
            from catboost import CatBoostRanker, Pool
            model = CatBoostRanker(loss_function="YetiRank", iterations=400, learning_rate=0.1, depth=6,
                                   random_seed=seed, thread_count=args.threads, verbose=False)
            model.fit(Pool(X, Y, group_id=np.repeat(np.arange(len(group)), group)))
        else:
            params = {**PARAMS, "seed": seed, "num_threads": args.threads}
            model = lgb.train(params, lgb.Dataset(X, Y, group=group, feature_name=FEATURE_NAMES),
                              num_boost_round=150)
        if not (args.reuse and (args.work / file).exists()):
            model.save_model(str(args.work / file))
        entries.append({"name": name, "kind": kind, "file": file, "weight": weight})
        print(f"{name}: {(time.perf_counter() - t0) / 60:.1f}분", flush=True)

    (args.work / "ensemble.json").write_text(json.dumps({"mode": "prob", "members": entries}, indent=2),
                                             encoding="utf-8")
    joblib.dump({"type": builder.type_mapper, "flag": builder.flag_mapper, "feature_names": FEATURE_NAMES},
                args.work / "mappers.joblib")

    # 패키지: v18 복사 → 기본 앙상블 파일만 교체
    pkg = args.pkg.resolve()
    if not pkg.exists():
        shutil.copytree(V18, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    for old in (V18 / "model").glob("m[0-9]_*"):
        (pkg / "model" / old.name).unlink(missing_ok=True)
    for f in [e["file"] for e in entries] + ["ensemble.json", "mappers.joblib"]:
        shutil.copy(args.work / f, pkg / "model" / f)
    print(f"패키지: {pkg}", flush=True)


if __name__ == "__main__":
    main()
