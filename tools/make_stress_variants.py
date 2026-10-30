"""Build a hangul stress set of 'solvable' variants from train hangul rows.

One row per vessel (200 vessels), each variant changes one thing.
Note: v12 was trained on all train rows, so absolute accuracy is optimistic;
we only look for variants where accuracy collapses relative to 'orig'.
"""
import random
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
t = pd.read_csv(f"{root}/train.csv", encoding="utf-8-sig", dtype=str)
c = pd.read_csv(f"{root}/vessel_catalog.csv", encoding="utf-8-sig", dtype=str)
h = t[t.vessel_name.str.contains("[가-힣]")]
rows = h.groupby("target_vessel_id").sample(1, random_state=0).sample(200, random_state=0)
en_name = dict(zip(c.vessel_id, c.vessel_name))
rng = random.Random(0)

VOW = {1: 5, 5: 1, 4: 8, 8: 4, 18: 13, 13: 18, 6: 12, 12: 6, 0: 4, 2: 6}
CHO = {0: 15, 15: 0, 3: 16, 16: 3, 7: 17, 17: 7, 9: 10, 10: 9, 12: 14, 14: 12}


def syls(s):
    return [i for i, ch in enumerate(s) if "가" <= ch <= "힣"]


def compose(cho, jung, jong):
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def parts(ch):
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def jamo_typo(s):
    idx = syls(s)
    if not idx:
        return s
    i = rng.choice(idx)
    cho, jung, jong = parts(s[i])
    kind = rng.choice(["v", "c", "f"])
    if kind == "v" and jung in VOW:
        jung = VOW[jung]
    elif kind == "c" and cho in CHO:
        cho = CHO[cho]
    else:
        jong = 0 if jong else 4  # drop or add ㄴ batchim
    return s[:i] + compose(cho, jung, jong) + s[i + 1:]


def syl_swap(s):
    idx = [i for i in syls(s) if i + 1 < len(s) and "가" <= s[i + 1] <= "힣"]
    if not idx:
        return s
    i = rng.choice(idx)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def syl_dup(s):
    idx = syls(s)
    if not idx:
        return s
    i = rng.choice(idx)
    return s[:i] + s[i] + s[i:]


def syl_drop(s):
    idx = syls(s)
    if len(idx) < 3:
        return s
    i = rng.choice(idx)
    return s[:i] + s[i + 1:]


def mixed_en(s, vid):
    """first word -> catalog English first word (upper)."""
    en = str(en_name.get(vid, "")).split()
    words = s.split()
    if len(words) < 2 or not en:
        return s
    return " ".join([en[0].upper()] + words[1:])


def extra_space(s):
    return " ".join(ch for ch in s.replace(" ", ""))


variants = {
    "orig": lambda r: r,
    "nospace": lambda r: {**r, "vessel_name": r["vessel_name"].replace(" ", "")},
    "space_each": lambda r: {**r, "vessel_name": extra_space(r["vessel_name"])},
    "jamo_typo": lambda r: {**r, "vessel_name": jamo_typo(r["vessel_name"])},
    "syl_swap": lambda r: {**r, "vessel_name": syl_swap(r["vessel_name"])},
    "syl_dup": lambda r: {**r, "vessel_name": syl_dup(r["vessel_name"])},
    "syl_drop": lambda r: {**r, "vessel_name": syl_drop(r["vessel_name"])},
    "add_ho": lambda r: {**r, "vessel_name": r["vessel_name"].replace(" ", "") + "호"},
    "mixed_en": lambda r: {**r, "vessel_name": mixed_en(r["vessel_name"], r["target_vessel_id"])},
    "gt_missing": lambda r: {**r, "gross_tonnage": None, "international_gross_tonnage": None},
    "gt_noise10": lambda r: {**r, "gross_tonnage": str(round(float(r["gross_tonnage"]) * rng.choice([0.9, 1.1]))),
                             "international_gross_tonnage": str(round(float(r["international_gross_tonnage"]) * rng.choice([0.9, 1.1])))},
    "meta_missing": lambda r: {**r, "flag_name": None},
}
out_rows = []
for name, fn in variants.items():
    for r in rows.to_dict("records"):
        v = fn(dict(r))
        v["variant"] = name
        out_rows.append(v)
df = pd.DataFrame(out_rows)
df["sample_id"] = [f"TEST_{i:05d}" for i in range(len(df))]
df.drop(columns=["variant", "target_vessel_id"]).to_csv(f"{out}/data/test.csv", index=False, encoding="utf-8-sig")
df[["sample_id", "variant", "vessel_name", "target_vessel_id"]].to_csv(f"{out}/truth.csv", index=False, encoding="utf-8-sig")
print(df.groupby("variant").vessel_name.apply(lambda s: s.iloc[:3].tolist()))
