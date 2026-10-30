"""Evaluate fixed matching-feature blends without loading gigabytes of features."""
import json
from types import SimpleNamespace

import joblib

from src.blend import blend
from src.common import ROOT, load_train, load_catalog
from src.experiment import Lab, build_hard, build_testlike, SCORES, RESULTS


def lightweight_lab(identity):
    train, catalog = load_train(), load_catalog()
    from src.common import is_hangul
    ctx = SimpleNamespace(train=train, rows=train.to_dict("records"), hangul=train.vessel_name.map(is_hangul).to_numpy())
    id_to_index = dict(zip(catalog.vessel_id, range(len(catalog))))
    ctx.truth = train.target_vessel_id.map(id_to_index).to_numpy()
    ctx.cands = identity["sets"]["oof"]["candidates"]
    assert train.sample_id.tolist() == identity["sets"]["oof"]["sample_ids"]
    lab = Lab.__new__(Lab)
    lab.ctx = ctx
    lab.sets = {}
    for s, df in (("hard", build_hard(ctx)), ("testlike", build_testlike(ctx))):
        assert df.sample_id.tolist() == identity["sets"][s]["sample_ids"]
        origin = df.origin.to_numpy()
        lab.sets[s] = SimpleNamespace(df=df, truth=ctx.truth[origin],
                                     names=train.vessel_name.iloc[origin].reset_index(drop=True),
                                     cands=identity["sets"][s]["candidates"])
    return lab


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("new", nargs="?", default="rich150_match_v1")
    args = parser.parse_args()
    identity = joblib.load(SCORES / f"{args.new}.identity.joblib")
    lab = lightweight_lab(identity)
    names = ["cat_yeti400_prior", "cat_yeti400_prior_a8", "cat_yeti400_prior_a9",
             "lgb_base_prior", "lgb_base_prior_a8", args.new]
    scores = [joblib.load(SCORES / f"{name}.joblib") for name in names]
    for name, parts in zip(names, scores):
        for s in lab.sizes():
            assert len(parts[s]) == len(lab.cands(s)), (name, s)
            assert all(len(sc) == len(c) for sc, c in zip(parts[s], lab.cands(s))), (name, s)
    plans = {"v7_reference": ([0, 3, 4], [3, 1, 1]),
             "v8_reference": ([0, 1, 2, 3, 4], [3, 3, 3, 1, 1]),
             "v7_add": ([0, 3, 4, 5], [3, 1, 1, 1]),
             "v7_replace": ([0, 4, 5], [3, 1, 1]),
             "v8_add": ([0, 1, 2, 3, 4, 5], [3, 3, 3, 1, 1, 2]),
             "v8_replace": ([0, 1, 2, 5], [3, 3, 3, 2])}
    output = []
    for name, (idx, weights) in plans.items():
        result = {"name": f"matching_blend:{args.new}:{name}",
                  **lab.evaluate(blend([scores[i] for i in idx], weights, "prob"))}
        if name == "v7_reference":
            assert result["testlike"] == .902
        if name == "v8_reference":
            assert result["testlike"] == .905
        output.append(result)
        with RESULTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result), flush=True)
    (ROOT / "experiments" / f"blends_{args.new}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
