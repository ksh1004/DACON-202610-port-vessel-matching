"""재순위 앙상블: 여러 학습기의 후보 점수를 질의 그룹 안에서 정규화해 가중 합한다.

정규화는 질의 1행의 후보끼리만 하므로 행 독립 원칙에 맞는다. 추론 스크립트와 실험 도구가 같은 코드를 쓴다.

    python -m src.ensemble --members lgb_base lgb_small lgb_bin --mode prob --out experiments/artifacts/ranker_v7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 학습기 정의 (experiment.py 의 같은 이름 학습기와 같은 설정)
SPECS: dict[str, dict] = {
    "lgb_base": {"kind": "lgb", "rounds": 150, "params": {}},
    "lgb_s1": {"kind": "lgb", "rounds": 150, "params": {"seed": 1, "feature_fraction": 0.7}},
    "lgb_s2": {"kind": "lgb", "rounds": 150, "params": {"seed": 2, "feature_fraction": 0.7}},
    "lgb_small": {"kind": "lgb", "rounds": 300, "params": {"num_leaves": 31, "min_data_in_leaf": 100}},
    "lgb_big": {"kind": "lgb", "rounds": 300, "params": {"num_leaves": 127, "min_data_in_leaf": 30}},
    "lgb_bin": {"kind": "lgb_bin", "rounds": 300, "params": {}},
    "cat_yeti400": {"kind": "cat", "rounds": 400, "params": {"learning_rate": 0.1, "depth": 6}},
    "cat_s2": {"kind": "cat", "rounds": 400, "params": {"learning_rate": 0.1, "depth": 6, "random_seed": 2}},
    "cat_d8": {"kind": "cat", "rounds": 400, "params": {"learning_rate": 0.1, "depth": 8}},
}


def normalize(score: np.ndarray, mode: str) -> np.ndarray:
    """한 질의의 후보 점수 → 그룹 내 정규화 점수."""
    if len(score) == 0:
        return score
    if mode == "z":
        return (score - score.mean()) / (score.std() + 1e-6)
    if mode == "prob":  # 그룹 내 softmax
        e = np.exp(score - score.max())
        return e / e.sum()
    if mode == "rank":  # 1등 = 1, 꼴찌 = 0
        order = (-score).argsort(kind="stable").argsort(kind="stable")
        return 1.0 - order / max(len(score) - 1, 1)
    raise ValueError(mode)


def fit_member(name: str, X: np.ndarray, y: np.ndarray, group: np.ndarray, feature_names: list[str]):
    import lightgbm as lgb
    from src.train_ranker import PARAMS

    spec = SPECS[name]
    if spec["kind"] == "lgb":
        return lgb.train({**PARAMS, **spec["params"]}, lgb.Dataset(X, y, group=group, feature_name=feature_names),
                         num_boost_round=spec["rounds"])
    if spec["kind"] == "lgb_bin":  # 점별 이진 분류: 순위 학습과 오차 양상이 달라 블렌딩 재료로 씀
        params = {k: v for k, v in PARAMS.items() if k not in ("eval_at", "lambdarank_truncation_level")}
        params.update(objective="binary", metric="binary_logloss", **spec["params"])
        return lgb.train(params, lgb.Dataset(X, y, feature_name=feature_names), num_boost_round=spec["rounds"])
    if spec["kind"] == "cat":
        from catboost import CatBoostRanker, Pool
        params = {"random_seed": 42, **spec["params"]}
        model = CatBoostRanker(loss_function="YetiRank", iterations=spec["rounds"], thread_count=12, verbose=False,
                               **params)
        model.fit(Pool(X, y, group_id=np.repeat(np.arange(len(group)), group)))
        return model
    raise ValueError(spec["kind"])


def member_score(kind: str, model, X: np.ndarray) -> np.ndarray:
    if kind == "lgb":
        return model.predict(X, num_threads=6)
    if kind == "lgb_bin":
        return model.predict(X, num_threads=6, raw_score=True)
    return model.predict(X)


class Ensemble:
    """ensemble.json + 학습기 파일들. score_groups(blocks) → 블록(질의)별 앙상블 점수."""

    def __init__(self, members: list[tuple[str, object, float]], mode: str) -> None:
        self.members, self.mode = members, mode

    @classmethod
    def load(cls, folder: Path) -> "Ensemble":
        import lightgbm as lgb

        spec = json.loads((folder / "ensemble.json").read_text(encoding="utf-8"))
        members = []
        for m in spec["members"]:
            if m["kind"] == "cat":
                from catboost import CatBoostRanker
                model = CatBoostRanker()
                model.load_model(str(folder / m["file"]))
            else:
                model = lgb.Booster(model_file=str(folder / m["file"]))
            members.append((m["kind"], model, float(m["weight"])))
        return cls(members, spec["mode"])

    def score_groups(self, blocks: list[np.ndarray]) -> list[np.ndarray]:
        sizes = [len(b) for b in blocks]
        X = np.vstack(blocks)
        bounds = np.cumsum([0] + sizes)
        total = [np.zeros(s, dtype=np.float64) for s in sizes]
        for kind, model, weight in self.members:
            raw = member_score(kind, model, X)
            for g in range(len(sizes)):
                total[g] += weight * normalize(raw[bounds[g] : bounds[g + 1]], self.mode)
        return total


def main() -> None:
    import joblib

    import gc

    from src.common import ROOT, load_catalog, load_train
    from src.features import FEATURE_NAMES, fit_mappers
    from src.train_ranker import Context

    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--mode", default="prob", choices=["z", "prob", "rank"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    weights = args.weights or [1.0] * len(args.members)

    contexts: dict[int, tuple] = {}

    def data(seed: int):
        """증강 시드별 전체 학습 행렬 (멤버 이름이 '@시드' 로 끝나면 그 시드의 증강으로 학습).

        시드 하나의 데이터가 수 GB 이므로 다른 시드를 불러오기 전에 이전 시드를 메모리에서 지운다.
        """
        if seed not in contexts:
            contexts.clear()
            gc.collect()
            ctx = Context(4, 1, seed=seed, use_llm=True)
            X_blocks, y_blocks = ctx.blocks + ctx.aug_blocks, ctx.labels + ctx.aug_labels
            keep = [i for i, t in enumerate(y_blocks) if t.any()]
            contexts[seed] = (np.vstack([X_blocks[i] for i in keep]), np.concatenate([y_blocks[i] for i in keep]),
                              np.array([len(y_blocks[i]) for i in keep]))
            del ctx, X_blocks, y_blocks
        return contexts[seed]

    args.out.mkdir(parents=True, exist_ok=True)
    members = [(i, m, w, int(m.partition("@")[2] or 7)) for i, (m, w) in enumerate(zip(args.members, weights))]
    entries = {}
    for i, member, w, seed in sorted(members, key=lambda t: (t[3], t[0])):  # 같은 시드끼리 연달아 학습
        name = member.partition("@")[0]
        X, y, group = data(seed)
        model = fit_member(name, X, y, group, FEATURE_NAMES)
        kind = SPECS[name]["kind"]
        file = f"m{i}_{name}{'' if seed == 7 else f'_s{seed}'}.{'cbm' if kind == 'cat' else 'txt'}"
        model.save_model(str(args.out / file))
        entries[i] = {"name": member, "kind": kind, "file": file, "weight": w}
        print(f"학습 완료: {member}", flush=True)
    (args.out / "ensemble.json").write_text(
        json.dumps({"mode": args.mode, "members": [entries[i] for i in sorted(entries)]}, indent=2), encoding="utf-8")
    # 선종·국적 매퍼는 원본 train 만으로 정해진다 (증강 시드와 무관)
    type_mapper, flag_mapper = fit_mappers(load_train(), load_catalog())
    joblib.dump({"type": type_mapper, "flag": flag_mapper, "feature_names": FEATURE_NAMES}, args.out / "mappers.joblib")
    print(f"저장: {args.out}  ({ROOT.name})")


if __name__ == "__main__":
    main()
