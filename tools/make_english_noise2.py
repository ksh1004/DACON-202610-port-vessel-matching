"""영문 단어 단위 손상 세트: 여러 단어 이름에서 단어 하나 빠짐(50%) 또는 끝이 잘림(50%). 톤수 결측 50%.

    python tools/make_english_noise2.py <데이터 폴더> <출력 폴더> [N=400]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
t = pd.read_csv(f"{root}/train.csv", encoding="utf-8-sig", dtype=str)
e = t[~t.vessel_name.fillna("").str.contains("[가-힣]") & (t.vessel_name.fillna("").str.split().str.len() >= 2)]
rows = e.groupby("target_vessel_id").sample(1, random_state=3).sample(n, random_state=4)
rng = np.random.default_rng(2028)
out_rows = []
for r in rows.to_dict("records"):
    v = dict(r)
    words = v["vessel_name"].split()
    if rng.random() < 0.5:
        words.pop(int(rng.integers(len(words))))
        v["vessel_name"] = " ".join(words)
        v["kind"] = "drop"
    else:
        name = v["vessel_name"]
        v["vessel_name"] = name[:max(4, int(len(name) * rng.uniform(0.5, 0.85)))].rstrip()
        v["kind"] = "trunc"
    v["nogt"] = rng.random() < 0.5
    if v["nogt"]:
        v["gross_tonnage"] = v["international_gross_tonnage"] = None
    out_rows.append(v)
df = pd.DataFrame(out_rows)
df["sample_id"] = [f"TEST_{i:05d}" for i in range(len(df))]
(out / "data").mkdir(parents=True, exist_ok=True)
df.drop(columns=["target_vessel_id", "nogt", "kind"]).to_csv(out / "data/test.csv", index=False, encoding="utf-8-sig")
df[["sample_id", "vessel_name", "nogt", "kind", "target_vessel_id"]].to_csv(out / "truth.csv", index=False, encoding="utf-8-sig")
print(len(df), df.nogt.mean(), df.kind.value_counts().to_dict())
print(df.vessel_name.head(10).tolist())
