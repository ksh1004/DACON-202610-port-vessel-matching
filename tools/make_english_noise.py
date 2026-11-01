"""영문(비한글) 이름 노이즈 세트: 한글 노이즈 세트(make_organizer_noise.py)와 같은 규칙을 영문 train 이름에 적용한다.

행마다 글자 편집 1회(70%) 또는 2회(30%), 편집 = 공백이 아닌 글자 하나 삭제 / 중복 / 이웃과 순서 바꿈.
톤수 결측 50%, 국적 결측 17%. 영문 선박 N척 x 1행.

    python tools/make_english_noise.py <데이터 폴더> <출력 폴더> [N=800]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 800
t = pd.read_csv(f"{root}/train.csv", encoding="utf-8-sig", dtype=str)
e = t[~t.vessel_name.fillna("").str.contains("[가-힣]") & t.vessel_name.notna()]
rows = e.groupby("target_vessel_id").sample(1, random_state=1).sample(n, random_state=2)
rng = np.random.default_rng(2027)


def edit(name: str) -> str:
    pos = [i for i, ch in enumerate(name) if not ch.isspace()]
    if len(pos) < 2:
        return name
    op = rng.choice(["delete", "dup", "swap"], p=[0.35, 0.35, 0.30])
    if op == "swap":
        pairs = [i for i in pos if i + 1 < len(name) and not name[i + 1].isspace() and name[i] != name[i + 1]]
        if pairs:
            i = int(rng.choice(pairs))
            return name[:i] + name[i + 1] + name[i] + name[i + 2:]
        op = "dup"
    i = int(rng.choice(pos))
    if op == "delete":
        return name[:i] + name[i + 1:]
    return name[:i + 1] + name[i] + name[i + 1:]


out_rows = []
for r in rows.to_dict("records"):
    v = dict(r)
    name = v["vessel_name"]
    for _ in range(1 if rng.random() < 0.7 else 2):
        name = edit(name)
    v["vessel_name"] = name
    v["nogt"] = rng.random() < 0.5
    if v["nogt"]:
        v["gross_tonnage"] = v["international_gross_tonnage"] = None
    if rng.random() < 0.17:
        v["flag_name"] = None
    out_rows.append(v)
df = pd.DataFrame(out_rows)
df["sample_id"] = [f"TEST_{i:05d}" for i in range(len(df))]
(out / "data").mkdir(parents=True, exist_ok=True)
df.drop(columns=["target_vessel_id", "nogt"]).to_csv(out / "data/test.csv", index=False, encoding="utf-8-sig")
df[["sample_id", "vessel_name", "nogt", "target_vessel_id"]].to_csv(out / "truth.csv", index=False, encoding="utf-8-sig")
print(len(df), df.nogt.mean())
print(df.vessel_name.head(12).tolist())
