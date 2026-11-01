"""카탈로그 영문 이름 전체를 한글로 음역해 캐시에 저장한다 (src/llm_ko.py, 이어서 실행 가능).

    python -m tools.catalog_ko
"""

from __future__ import annotations

import json
import time

from src.common import ROOT, is_hangul, load_catalog, load_train
from src.llm_ko import KoTransliterator

OUT = ROOT / "experiments" / "cache" / "catalog_ko.json"


def main() -> None:
    catalog, train = load_catalog(), load_train()
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    id2name = dict(zip(catalog.vessel_id, catalog.vessel_name.astype(str)))
    first = [id2name[t] for t in train.loc[train.vessel_name.map(is_hangul), "target_vessel_id"].unique()]
    rest = sorted(set(id2name.values()) - set(first), key=len)
    todo = [n for n in dict.fromkeys(first + rest) if n not in data]
    print(f"이름 {len(set(id2name.values()))}개 중 생성 필요 {len(todo)}개", flush=True)
    model = KoTransliterator(ROOT.parent / "submit_v18" / "model" / "llm")
    t0 = time.perf_counter()
    step = 1024
    for s in range(0, len(todo), step):
        chunk = todo[s:s + step]
        for name, out in zip(chunk, model.generate(chunk, batch_size=32)):
            data[name] = out[0] if out else ""
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OUT)
        done = s + len(chunk)
        rate = (time.perf_counter() - t0) / done
        print(f"{done}/{len(todo)}  {rate * 1000:.0f} ms/이름, 남은 {rate * (len(todo) - done) / 60:.0f}분", flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
