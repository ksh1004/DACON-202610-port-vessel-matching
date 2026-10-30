"""Hangul stress set that mimics the organizer's noise seen on English train names.

Per row: 1 edit (70%) or 2 edits (30%), each edit = delete / duplicate / adjacent swap of one
non-space character (syllable). Tonnage missing 50%, flag missing 17%. 382 vessels x 2 copies.
"""
import sys

import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
t = pd.read_csv(f"{root}/train.csv", encoding="utf-8-sig", dtype=str)
h = t[t.vessel_name.str.contains("[가-힣]")]
rows = h.groupby("target_vessel_id").sample(1, random_state=1)
rng = np.random.default_rng(2026)


def edit(name: str) -> str:
    pos = [i for i, ch in enumerate(name) if not ch.isspace()]
    if len(pos) < 2:
        return name
    op = rng.choice(["delete", "dup", "swap"], p=[0.35, 0.35, 0.30])
    if op == "swap":
        pairs = [i for i in pos if i + 1 < len(name) and not name[i + 1].isspace()]
        if pairs:
            i = int(rng.choice(pairs))
            return name[:i] + name[i + 1] + name[i] + name[i + 2:]
        op = "dup"
    i = int(rng.choice(pos))
    if op == "delete":
        return name[:i] + name[i + 1:]
    return name[:i + 1] + name[i] + name[i + 1:]


out_rows = []
for copy in range(2):
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
df.drop(columns=["target_vessel_id", "nogt"]).to_csv(f"{out}/data/test.csv", index=False, encoding="utf-8-sig")
df[["sample_id", "vessel_name", "nogt", "target_vessel_id"]].to_csv(f"{out}/truth.csv", index=False, encoding="utf-8-sig")
print(len(df), df.nogt.mean())
print(df.vessel_name.head(12).tolist())
