"""카탈로그 이름의 한글 음역을 여러 개(빔 n-best) 만들어 캐시에 저장한다 (이어서 실행 가능).

    python -m tools.catalog_ko_nbest order   # 생성 순서: CV 탐침 후보 → 한글 질의에 자주 나오는 국적·선종 순
    python -m tools.catalog_ko_nbest run     # 순서대로 생성, 1024개마다 저장
"""

from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np

from src.common import ROOT, is_hangul, load_catalog, load_train

OUT = ROOT / "experiments" / "cache" / "catalog_ko_nbest.json"
ORDER = ROOT / "experiments" / "cache" / "catalog_ko_nbest_order.json"
N_BEST, BATCH = 4, int(__import__("os").environ.get("KO_BATCH", "4"))


def cmd_order(top: int = 10) -> None:
    from tools.probe_ko_rerank import CACHE, KO, hangul_sims, ko_norm
    from tools.retrain_organizer import load
    catalog, train = load_catalog(), load_train()
    names = catalog.vessel_name.astype(str).tolist()
    id2i = {v: i for i, v in enumerate(catalog.vessel_id)}
    ko = json.loads(KO.read_text(encoding="utf-8"))
    ko_names = [ko_norm(ko.get(n, "")) for n in names]
    sc = joblib.load(CACHE / "scores_V0.joblib")
    probe = set()
    for data, scores in ((load("train@500"), sc["tr"]), (load("evorg@500"), sc["org"])):
        for i, local in scores.items():
            cand = data["cands"][i]
            if len(cand) == 0:
                continue
            finite = np.isfinite(local)
            lp = np.where(finite, local - np.logaddexp.reduce(local[finite]), -np.inf)
            z = lp + 4 * hangul_sims(data["rows"][i]["vessel_name"], cand, ko_names)["indel"]
            probe.update(int(c) for c in cand[np.argsort(-z)[:top]])
    truth = train.loc[train.vessel_name.map(is_hangul), "target_vessel_id"].map(id2i)
    probe.update(int(t) for t in truth)
    # 나머지: 한글 질의 정답에서 많이 나오는 국적·선종일수록 먼저
    h = catalog.iloc[truth.unique()]
    r_flag = (h.flag_name.value_counts(normalize=True) / catalog.flag_name.value_counts(normalize=True)).fillna(0)
    r_type = (h.vessel_type.value_counts(normalize=True) / catalog.vessel_type.value_counts(normalize=True)).fillna(0)
    prior = catalog.flag_name.map(r_flag).fillna(0).to_numpy() + 0.01 + catalog.vessel_type.map(r_type).fillna(0).to_numpy()
    first = [names[i] for i in sorted(probe)]
    rest = [names[i] for i in np.argsort(-prior, kind="stable")]
    order = list(dict.fromkeys(first + rest))
    ORDER.write_text(json.dumps({"probe": len(dict.fromkeys(first)), "order": order}, ensure_ascii=False), encoding="utf-8")
    print(f"탐침 후보 이름 {len(dict.fromkeys(first))}개, 전체 {len(order)}개", flush=True)


def cmd_run() -> None:
    from src.llm_ko import KoTransliterator
    order = json.loads(ORDER.read_text(encoding="utf-8"))["order"]
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [n for n in order if n not in data]
    print(f"생성 필요 {len(todo)}개 (완료 {len(data)}개)", flush=True)
    model = KoTransliterator(ROOT.parent / "submit_v18" / "model" / "llm")
    t0 = time.perf_counter()
    step = 256
    for s in range(0, len(todo), step):
        chunk = todo[s:s + step]
        for name, out in zip(chunk, model.generate(chunk, n_best=N_BEST, batch_size=BATCH)):
            data[name] = out
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OUT)
        done = s + len(chunk)
        rate = (time.perf_counter() - t0) / done
        print(f"{len(data)} 완료 ({done}/{len(todo)})  {rate * 1000:.0f} ms/이름, 남은 {rate * (len(todo) - done) / 60:.0f}분", flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    {"order": lambda: cmd_order(*map(int, sys.argv[2:])), "run": cmd_run}[sys.argv[1]]()
