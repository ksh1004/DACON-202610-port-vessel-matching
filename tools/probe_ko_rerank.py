"""한글 공간 비교(카탈로그 한글 음역 ↔ 질의 한글)로 재순위했을 때의 효과를 선박 단위 5-fold 로 잰다.

    python -m tools.probe_ko_rerank scores   # V0(v18 방식) LightGBM fold 점수 저장
    python -m tools.probe_ko_rerank eval     # 한글 유사도 가중치별 정확도
"""

from __future__ import annotations

import gc
import json
import sys

import joblib
import lightgbm as lgb
import numpy as np

from src.common import ROOT, is_hangul
from src.features import FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.train_ranker import PARAMS
from tools.retrain_organizer import CACHE, CATEGORY_COLUMNS, VARIANTS, load, load_folds, setup, stack

KO = ROOT / "experiments" / "cache" / "catalog_ko.json"
SCORES = CACHE / "scores_V0.joblib"
KINDS = ["indel", "osa", "jamo", "jamo_osa", "mix"]
NBEST: dict = {}  # 카탈로그 위치 → 빔 음역 목록 (eval nb 일 때)


def fold_scores(model, data, idx, builder, banned):
    out = []
    blocks = []
    for i in idx:
        block = data["blocks"][i].copy()
        for col, value in zip(CATEGORY_COLUMNS, builder.category(data["rows"][i], data["cands"][i])):
            block[:, col] = value
        blocks.append(block)
    sizes = [len(b) for b in blocks]
    nonempty = [b for b in blocks if len(b)]
    scores = model.predict(np.vstack(nonempty), num_threads=20) if nonempty else []
    offset = 0
    for i, size in zip(idx, sizes):
        local = np.asarray(scores[offset:offset + size], dtype=np.float64)
        offset += size
        bad = banned[data["cands"][i]]
        if bad.any() and not bad.all():
            local = np.where(bad, -np.inf, local)
        out.append(local)
    return out


def cmd_scores(variant: str = "V0") -> None:
    train, catalog, retrievers = setup()
    r, base_builder = retrievers[500]
    truth_idx = train["target_vessel_id"].map(r.id_to_index).to_numpy()
    folds = load_folds(train)
    hangul = train["vessel_name"].map(is_hangul).to_numpy()
    ev_train, ev_org = load("train@500"), load("evorg@500")
    keys, filtered = VARIANTS[variant]
    sets = [ev_train if k == "train@500" else load(k) for k in keys]
    s_tr, s_org = {}, {}
    for k in sorted(set(folds)):
        tr_rows = np.flatnonzero(folds != k)
        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.__dict__.update(base_builder.__dict__)
        builder.type_mapper, builder.flag_mapper = fit_mappers(train.iloc[tr_rows], catalog)
        banned = np.zeros(len(r.ids), dtype=bool)
        banned[truth_idx[tr_rows]] = True
        pick = [np.flatnonzero(np.isin(d["origin"], tr_rows)) for d in sets]
        X, Y, group = stack(sets, pick, builder, truth_idx, banned if filtered else None)
        model = lgb.train({**PARAMS, "num_threads": 20}, lgb.Dataset(X, Y, group=group, feature_name=FEATURE_NAMES),
                          num_boost_round=150)
        del X, Y, group
        gc.collect()
        va = np.flatnonzero((folds == k) & hangul)
        s_tr.update(zip(va.tolist(), fold_scores(model, ev_train, va, builder, banned)))
        va_org = np.flatnonzero(np.isin(ev_org["origin"], np.flatnonzero(folds == k)))
        s_org.update(zip(va_org.tolist(), fold_scores(model, ev_org, va_org, builder, banned)))
        print(f"fold {k} 완료", flush=True)
    joblib.dump({"tr": s_tr, "org": s_org}, CACHE / f"scores_{variant}.joblib")


def ko_norm(name: str) -> str:
    s = "".join(str(name).split())
    return s[:-1] if len(s) > 2 and s.endswith("호") else s


def jamo(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out.append(chr(0x1100 + o // 588))
            out.append(chr(0x1161 + (o % 588) // 28))
            if o % 28:
                out.append(chr(0x11A7 + o % 28))
        else:
            out.append(ch)
    return "".join(out)


def hangul_sims(query: str, cand: np.ndarray, ko_names: list[str]) -> dict[str, np.ndarray]:
    from rapidfuzz.distance import OSA, Indel
    q = ko_norm(query)
    qj = jamo(q)
    ks = [ko_names[c] for c in cand]
    indel = np.array([Indel.normalized_similarity(q, k) if k else 0.0 for k in ks])
    osa = np.array([OSA.normalized_similarity(q, k) if k else 0.0 for k in ks])
    jind = np.array([Indel.normalized_similarity(qj, jamo(k)) if k else 0.0 for k in ks])
    josa = np.array([OSA.normalized_similarity(qj, jamo(k)) if k else 0.0 for k in ks])
    out = {"indel": indel, "osa": osa, "jamo": jind, "jamo_osa": josa, "mix": (indel + jind) / 2}
    if NBEST:
        ks_n = [NBEST.get(c, ()) for c in cand]
        nb = np.array([max([Indel.normalized_similarity(q, v) for v in vs] or [0.0]) for vs in ks_n])
        out["nb_max"] = np.maximum(indel, nb)  # 탐욕 음역과 빔 음역 중 가장 가까운 것
        out["nb_beam"] = np.where([bool(v) for v in ks_n], nb, indel)  # 빔 음역만 (없으면 탐욕)
        out["nb_avg"] = (indel + out["nb_max"]) / 2
    return out


def cmd_eval(variant: str = "V0", nb: str = "") -> None:
    from src.common import load_catalog, load_train
    train, catalog = load_train(), load_catalog()
    id2i = {v: i for i, v in enumerate(catalog.vessel_id)}
    truth_idx = train["target_vessel_id"].map(id2i).to_numpy()
    ko = json.loads(KO.read_text(encoding="utf-8"))
    ko_names = [ko_norm(ko.get(str(n), "")) for n in catalog.vessel_name]
    print(f"음역된 카탈로그 비율 {np.mean([bool(k) for k in ko_names]):.3f}", flush=True)
    if nb:
        table = json.loads((ROOT / "experiments" / "cache" / "catalog_ko_nbest.json").read_text(encoding="utf-8"))
        NBEST.update({i: [ko_norm(v) for v in table[n]] for i, n in enumerate(catalog.vessel_name.astype(str)) if n in table})
        KINDS[:] = ["indel", "nb_max", "nb_beam", "nb_avg"]
        print(f"빔 음역 있는 카탈로그 {len(NBEST)}개", flush=True)
    sc = joblib.load(CACHE / f"scores_{variant}.joblib")
    ev_train, ev_org = load("train@500"), load("evorg@500")
    sets = {}
    for tag, data, scores, origin in (("clean", ev_train, sc["tr"], None), ("org", ev_org, sc["org"], ev_org["origin"])):
        rows = []
        for i, local in scores.items():
            cand = data["cands"][i]
            if len(cand) == 0:
                continue
            t = truth_idx[i if origin is None else origin[i]]
            name = data["rows"][i]["vessel_name"]
            finite = np.isfinite(local)
            lp = np.where(finite, local - np.logaddexp.reduce(local[finite]), -np.inf)
            rows.append((lp, hangul_sims(name, cand, ko_names), cand == t,
                         bool(data["nogt"][i]) if data.get("nogt") is not None else False))
        sets[tag] = rows
    res = {}
    for kind in KINDS:
        for w in (0, 1, 2, 3, 4, 6, 8, 12, 16):
            accs = {}
            for tag, rows in sets.items():
                ok = [bool(y[np.argmax(lp + w * h[kind])]) for lp, h, y, _ in rows]
                accs[tag] = round(float(np.mean(ok)), 4)
                if tag == "org":
                    ng = np.array([n for *_, n in rows])
                    accs["org_nogt"] = round(float(np.mean(np.array(ok)[ng])), 4)
                    accs["org_gt"] = round(float(np.mean(np.array(ok)[~ng])), 4)
            res[f"{kind}_w{w}"] = accs
            print(kind, w, accs, flush=True)
    (ROOT / "experiments" / f"ko_rerank_{variant}{'_nb' if nb else ''}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"scores": lambda: cmd_scores(*sys.argv[2:]), "eval": lambda: cmd_eval(*sys.argv[2:])}[sys.argv[1]]()
