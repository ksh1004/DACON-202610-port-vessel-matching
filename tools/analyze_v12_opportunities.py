"""Read-only v12 error budget and candidate-stage diagnosis, without new fitting."""
import argparse
import collections
import gc
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.augment import augment
from src.common import ROOT, is_hangul, load_train, load_catalog
from src.experiment import TESTLIKE_MIX
from src.pair_views import mix_views
from src.validate import load_folds
from tools.blend_matching import lightweight_lab


def training_profile(train, h, folds, fold):
    aug = augment(train, 4, 1, seed=7)
    trdf = pd.concat([train.loc[h & (folds != fold)],
                      aug.loc[h[aug.origin.to_numpy()] & (folds[aug.origin.to_numpy()] != fold)]], ignore_index=True)
    source = max((ROOT / "experiments/cache").glob(f"strict_han_f{fold}_train_*.joblib"), key=lambda p: p.stat().st_mtime)
    matrix = joblib.load(source)
    assert matrix["sample_ids"] == trdf.sample_id.tolist()
    eligibility = collections.Counter()
    selected = []
    for i in np.random.default_rng(42).permutation(len(trdf)):
        c, target = matrix["candidates"][i], matrix["truth"][i]
        eligibility["rows"] += 1
        if target not in c:
            eligibility["positive_absent"] += 1
            continue
        if len(c) < 8:
            eligibility["fewer_than_8_candidates"] += 1
            continue
        eligibility["eligible"] += 1
        if len(selected) < 2400:
            selected.append(int(i))
    del matrix
    gc.collect()
    profile = collections.Counter()
    for i in selected:
        row = trdf.iloc[i]
        original = row if pd.isna(row.get("origin")) else train.iloc[int(row["origin"])]
        changed = row.vessel_name != original.vessel_name
        words = original.vessel_name.split()
        qwords = row.vessel_name.split()
        exact_drop = any(" ".join(words[:j]+words[j+1:]) == row.vessel_name for j in range(len(words))) if len(words) > 1 else False
        missing = pd.isna(row.gross_tonnage) and pd.isna(row.international_gross_tonnage)
        profile["rows"] += 1
        profile["name_unchanged"] += int(not changed)
        profile["both_tonnages_missing"] += int(missing)
        profile["exact_word_drop_no_other_text_changes"] += int(exact_drop)
        profile["exact_word_drop_and_both_tonnages_missing"] += int(exact_drop and missing)
        profile["text_changed_other_than_exact_word_drop"] += int(changed and not exact_drop)
    counts = trdf.iloc[selected].target_vessel_id.value_counts()
    return {"fold": fold, "source": source.name, "eligibility": dict(eligibility),
                                           "selected_profile": dict(profile), "selected_unique_vessels": int(len(counts)),
                                           "max_groups_one_vessel": int(counts.max()), "top10_vessel_group_share": float(counts.head(10).sum()/len(selected))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-folds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    archive = joblib.load(ROOT / "experiments/encoder_bge_phonetic_predictions.joblib")
    lab = lightweight_lab(archive["identity"])
    train, catalog = lab.ctx.train, load_catalog()
    h, folds = lab.ctx.hangul, load_folds(train)
    truth = {"oof": lab.ctx.truth, **{s: x.truth for s, x in lab.sets.items()}}
    prediction = archive["ensemble"]
    ids = {s: {key: i for i, key in enumerate(archive["identity"]["sets"][s]["sample_ids"])} for s in prediction}
    records = {"oof": train.to_dict("records"), **{s: e.df.to_dict("records") for s, e in lab.sets.items()}}
    observed = {s: np.zeros(len(values), dtype=bool) for s, values in prediction.items()}
    budget = {}
    for scenario in ("clean", "gt_missing", "drop_word", "both"):
        if scenario == "clean":
            errors = int(np.sum(h & (prediction["oof"] != truth["oof"])))
        else:
            mask = lab.sets["testlike"].df.scenario.eq(scenario).to_numpy()
            errors = int(np.sum(mask & (prediction["testlike"] != truth["testlike"])))
        budget[scenario] = {"rows": int(h.sum()), "errors": errors,
                            "weighted_loss": .7*TESTLIKE_MIX[scenario]*errors/int(h.sum())}
    en_error = int(np.sum(~h & (prediction["oof"] != truth["oof"])))
    budget["non_hangul"] = {"rows": int((~h).sum()), "errors": en_error,
                            "weighted_loss": .3*en_error/int((~h).sum())}
    score = 1-sum(item["weighted_loss"] for item in budget.values())
    assert abs(score-.9118708558226221) < 1e-12
    for item in budget.values():
        item["share_of_weighted_error"] = item["weighted_loss"]/(1-score)

    stages = collections.defaultdict(collections.Counter)
    ranks = collections.defaultdict(collections.Counter)
    view_oracles = collections.defaultdict(collections.Counter)
    wrong_metadata = collections.defaultdict(collections.Counter)
    examples = []
    for fold in range(5):
        data = joblib.load(ROOT / "experiments/artifacts/encoder_bge_phonetic" / f"fold{fold}_epoch1_scores.joblib")
        qdata = joblib.load(ROOT / "experiments/artifacts/neural_evidence_v1" / f"fold{fold}_epoch1_scores.joblib")
        for s in ("oof", "testlike"):
            assert data["sample_ids"][s] == qdata["sample_ids"][s]
            for key, row, qr in zip(data["sample_ids"][s], data["sets"][s], qdata["sets"][s]):
                i = ids[s][key]
                assert not observed[s][i]
                observed[s][i] = True
                record = records[s][i]
                scenario = "clean" if s == "oof" else record["scenario"]
                c, base, keep, target = row["candidates"], row["base_scores"], row["keep"], row["truth"]
                assert target == truth[s][i]
                loc = np.flatnonzero(c == target)
                base_order = np.argsort(-base, kind="stable")
                rank = int(np.flatnonzero(base_order == loc[0])[0])+1 if len(loc) else None
                stages[scenario]["rows"] += 1
                for k in (2, 5, 10, 20, 50):
                    stages[scenario][f"truth_in_base_top{k}"] += int(rank is not None and rank <= k)
                stages[scenario]["truth_in_all_candidates"] += int(rank is not None)
                candidates = {"base": int(c[np.argmax(base)])}
                if is_hangul(record["vessel_name"]) and not row["bound_pruned"]:
                    bp = base[keep]/base[keep].sum()
                    values = mix_views(bp, [qr["neural_scores"], row["neural_scores"]], [1, 1])
                    winner = int(c[keep[np.argmax(values)]])
                    qwinner = int(c[keep[np.argmax(qr["neural_scores"])]])
                    ewinner = int(c[keep[np.argmax(row["neural_scores"])]])
                    candidates.update(qwen_raw=qwinner, encoder_raw=ewinner)
                    final_order = keep[np.argsort(-values, kind="stable")]
                    if qwinner == ewinner:
                        view_oracles[scenario]["active_agree"] += 1
                        view_oracles[scenario]["active_agree_wrong"] += int(qwinner != target)
                    view_oracles[scenario]["active_rows"] += 1
                else:
                    winner = candidates["base"]
                    final_order = base_order
                assert winner == prediction[s][i]
                candidates.update(v11=int(archive["reference"][s][i]),
                                  bge25=int(archive["predictions"][s][i]), v12=winner)
                view_oracles[scenario]["top1_union_hits"] += int(target in candidates.values())
                for k in (2, 5, 10, 20):
                    ranks[scenario][f"v12_top{k}_hits"] += int(target in c[final_order[:k]])
                if winner == target:
                    stages[scenario]["correct"] += 1
                    continue
                if not is_hangul(record["vessel_name"]):
                    category = "routed_non_hangul"
                elif rank is None:
                    category = "absent_from_retrieval"
                elif rank > 20:
                    category = "outside_neural_top20"
                elif row["bound_pruned"]:
                    category = "top20_present_but_pruned"
                else:
                    category = "top20_active_selection_error"
                stages[scenario][category] += 1
                true, wrong = catalog.iloc[target], catalog.iloc[winner]
                if true.gross_tonnage == wrong.gross_tonnage:
                    wrong_metadata[scenario]["same_catalog_tonnage"] += 1
                    if true.vessel_type == wrong.vessel_type and true.flag_name == wrong.flag_name:
                        wrong_metadata[scenario]["same_catalog_tonnage_type_flag"] += 1
                if len(examples) < 16 and category in ("outside_neural_top20", "top20_present_but_pruned"):
                    examples.append({"scenario": scenario, "stage": category, "query_name": record["vessel_name"],
                                     "truth_name": true.vessel_name, "chosen_name": wrong.vessel_name,
                                     "truth_base_rank": rank})
    np.testing.assert_array_equal(observed["oof"], h)
    assert observed["testlike"].all()
    partitions = ("correct", "routed_non_hangul", "absent_from_retrieval", "outside_neural_top20", "top20_present_but_pruned", "top20_active_selection_error")
    for counts in stages.values():
        assert sum(counts[k] for k in partitions) == counts["rows"]

    weighted_stages = {}
    for stage in partitions[1:]:
        weighted_stages[stage] = sum(.7*TESTLIKE_MIX[s]*stages[s][stage]/int(h.sum()) for s in TESTLIKE_MIX)
    english_acc = 1-en_error/int((~h).sum())
    top1_union_oracle = .3*english_acc+sum(.7*TESTLIKE_MIX[s]*view_oracles[s]["top1_union_hits"]/int(h.sum()) for s in TESTLIKE_MIX)
    result = {"reference": "v12", "public": .9065951735, "local_testlike": score,
              "scenario_mix": TESTLIKE_MIX, "error_budget": budget, "stage_counts": dict(stages),
              "weighted_error_by_stage": weighted_stages, "v12_topk": dict(ranks),
              "view_diagnostics": dict(view_oracles), "archived_top1_switch_oracle": top1_union_oracle,
              "wrong_candidate_metadata": dict(wrong_metadata), "examples": examples,
              "counterfactual_local_if_both_perfect_others_fixed": score+budget["both"]["weighted_loss"],
              "counterfactual_local_if_all_other_hangul_perfect_both_fixed": .3*english_acc+.7*(1-TESTLIKE_MIX["both"]*budget["both"]["errors"]/int(h.sum())),
              "caveat": "Historical reused synthetic CV and shared v8 retrieval mapper. Oracles/counterfactuals use labels and are not learned scores/Public projections. Catalog metadata equality is not a proof of indistinguishability."}

    result["training_fold_diagnostics"] = {str(fold): training_profile(train, h, folds, fold) for fold in args.training_folds}
    for name, item in budget.items():
        assert 0 <= item["weighted_loss"] <= 1
    changed_profile = {}
    frame = lab.sets["testlike"].df
    for scenario in ("gt_missing", "drop_word", "both"):
        pick = frame.scenario.eq(scenario).to_numpy()
        part = frame.loc[pick]
        originals = train.vessel_name.iloc[part.origin.to_numpy()].to_numpy()
        changed = part.vessel_name.to_numpy() != originals
        correct = prediction["testlike"][pick] == truth["testlike"][pick]
        changed_profile[scenario] = {"rows": len(part), "actually_changed_names": int(changed.sum()),
                                     "changed_name_errors": int(np.sum(changed & ~correct)),
                                     "unchanged_name_errors": int(np.sum(~changed & ~correct)),
                                     "weighted_mass_changed": TESTLIKE_MIX[scenario]*float(changed.mean())}
    result["actual_name_change_by_scenario"] = changed_profile
    result["v12_topk_oracles"] = {str(k): .3*english_acc+sum(.7*TESTLIKE_MIX[s]*ranks[s][f"v12_top{k}_hits"]/int(h.sum()) for s in TESTLIKE_MIX) for k in (2, 5, 10, 20)}
    result["analysis_source_sha256"] = hashlib.sha256((ROOT / "tools/analyze_v12_opportunities.py").read_bytes()).hexdigest()
    path = ROOT / "experiments/v12_opportunities_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
