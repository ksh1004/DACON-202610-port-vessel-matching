"""Out-of-sample test: does retraining the ranker on organizer-style hangul noise help?

Evaluation set: every train hangul row with organizer-style noise (1-2 syllable delete/dup/swap,
tonnage missing 50%, flag missing 17%). Candidates use the v17 retrieval (fuzzy K + typo reversal)
and exclude training-fold vessels, exactly like the real test. For each vessel-grouped fold a
LightGBM ranker is trained without that fold's vessels and scored on that fold only.

  (a) current training data (cached train + default augmentation)
  (c) (a) + 4 organizer-noise copies per hangul row, built with the fuzzy retrieval

    python -m tools.oof_organizer_noise
"""
import json
import sys

import numpy as np
import pandas as pd

from src.common import is_hangul
from src.experiment import _predict_blocks, lgb_learner
from src.llm_cache import BeamCache
from src.retrieval import query_key
from src.train_ranker import Context, build_matrix, refresh_category

FUZZY_K = 200


def _fuzzy_top(score, k):
    cols = np.argpartition(-score, kth=min(k, len(score) - 1))[:k]
    cols = cols[np.lexsort((cols, -score[cols]))]
    return cols[score[cols] > 0]


def _typo_reversals(name):
    out = []
    for i in range(len(name) - 1):
        a, b = name[i], name[i + 1]
        if a.isspace() or b.isspace():
            continue
        out.append(name[:i] + name[i + 1:] if a == b else name[:i] + b + a + name[i + 2:])
    return list(dict.fromkeys(out))


def add_fuzzy_candidates(r):
    """Same logic as submit/v17_fuzzy200/script.py (kept in sync by hand)."""
    from rapidfuzz import process
    from rapidfuzz.distance import Indel
    from src.common import compact_name, normalize_name

    original = r.generate

    def generate(queries, *args, llm_beams=None, **kwargs):
        out = original(queries, *args, llm_beams=llm_beams, **kwargs)
        names = queries["vessel_name"].tolist()
        for i, sig in enumerate(out):
            if not is_hangul(names[i]) or len(sig["cand"]) == 0:
                continue
            beams = list(sig.get("llm_beams") or [])
            pools = []
            if sig["query_key"]:
                pools.append(_fuzzy_top(process.cdist([sig["query_key"]], r.keys_compact,
                                                      scorer=Indel.normalized_similarity, workers=1)[0], FUZZY_K))
            if beams:
                pools.append(_fuzzy_top(process.cdist([compact_name(b) for b in beams], r.compact,
                                                      scorer=Indel.normalized_similarity, workers=1).max(axis=0), FUZZY_K))
            for variant in _typo_reversals(names[i]):
                vkey = query_key(variant).replace(" ", "")
                if vkey and vkey != sig["query_key"]:
                    hits = r.exact_key.get(vkey, [])
                    if hits:
                        pools.append(r._closest_tonnage(hits, queries.iloc[i].to_dict(), 10))
            if not pools:
                continue
            new = pd.unique(np.concatenate(pools)).astype(int)
            new = new[~np.isin(new, sig["cand"])]
            if len(new) == 0:
                continue
            name = names[i]
            key_spaced = query_key(name)
            sim = lambda vec, mat, q: (vec.transform(q) @ mat[new].T).toarray()  # noqa: E731
            add = {
                "sim_word": sim(r.tfidf_word, r.matrix_word, [normalize_name(name)])[0],
                "sim_compact": sim(r.tfidf_compact, r.matrix_compact, [compact_name(name)])[0],
                "sim_key": sim(r.tfidf_key, r.matrix_key, [key_spaced.replace(" ", "")])[0],
                "sim_key_word": sim(r.tfidf_key_word, r.matrix_key_word, [key_spaced])[0],
                "sim_llm": (sim(r.tfidf_word, r.matrix_word, [normalize_name(b) for b in beams]).max(axis=0)
                            if beams else np.zeros(len(new))),
                "is_exact": np.zeros(len(new)),
            }
            for field in ("rank_word", "rank_compact", "rank_key", "rank_key_word", "rank_gt", "rank_tok", "rank_llm"):
                add[field] = np.full(len(new), 999.0)
            for field, values in add.items():
                sig[field] = np.concatenate([sig[field], np.asarray(values, dtype=sig[field].dtype)])
            sig["cand"] = np.concatenate([sig["cand"], new])
        return out

    r.generate = generate


def organizer_noise(train: pd.DataFrame, copies: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def edit(name):
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
        return name[:i] + name[i + 1:] if op == "delete" else name[:i + 1] + name[i] + name[i + 1:]

    out = []
    for pos, row in enumerate(train.to_dict("records")):
        if not is_hangul(row["vessel_name"]):
            continue
        for c in range(copies):
            v = dict(row, origin=pos, sample_id=f"{row['sample_id']}_org{seed}_{c}")
            name = v["vessel_name"]
            for _ in range(1 if rng.random() < 0.7 else 2):
                name = edit(name)
            v["vessel_name"] = name
            v["nogt"] = bool(rng.random() < 0.5)
            if v["nogt"]:
                v["gross_tonnage"] = v["international_gross_tonnage"] = np.nan
            if rng.random() < 0.17:
                v["flag_name"] = np.nan
            out.append(v)
    return pd.DataFrame(out)


def main() -> None:
    ctx = Context(4, 1, seed=7, use_llm=True)  # production training data (cached, no fuzzy)
    add_fuzzy_candidates(ctx.retriever)
    beams = BeamCache(allow_generate=True)

    ev = organizer_noise(ctx.train, 1, seed=31337)
    ev_data = build_matrix(ev, ctx.retriever, ctx.builder, name=f"orgnoise_eval_fz{FUZZY_K}", beams_fn=beams)
    beams.save()
    print("eval built", len(ev), flush=True)
    aug = organizer_noise(ctx.train, 4, seed=4242)
    aug_data = build_matrix(aug, ctx.retriever, ctx.builder, name=f"orgnoise_aug4_fz{FUZZY_K}", beams_fn=beams)
    beams.save()
    print("aug built", len(aug), flush=True)

    ev_rows, aug_rows = ev.to_dict("records"), aug.to_dict("records")
    ev_origin, aug_origin = ev["origin"].to_numpy(), aug["origin"].to_numpy()
    ev_truth, aug_truth = ctx.truth[ev_origin], ctx.truth[aug_origin]
    aug_labels = [(c == t).astype(np.float32) for c, t in zip(aug_data["cands"], aug_truth)]

    results = {m: np.zeros(len(ev), dtype=bool) for m in ("a", "c")}
    for k in sorted(set(ctx.folds)):
        tr_idx = np.flatnonzero(ctx.folds != k)
        builder = ctx.fold_builder(tr_idx)
        blocks = refresh_category(ctx.blocks, ctx.cands, ctx.rows, builder)
        X = [blocks[i] for i in tr_idx]
        y = [ctx.labels[i] for i in tr_idx]
        pick = np.flatnonzero(np.isin(ctx.aug_origin, tr_idx))
        X += refresh_category([ctx.aug_blocks[i] for i in pick], [ctx.aug_cands[i] for i in pick],
                              [ctx.aug_rows[i] for i in pick], builder)
        y += [ctx.aug_labels[i] for i in pick]
        apick = np.flatnonzero(np.isin(aug_origin, tr_idx))
        Xo = refresh_category([aug_data["blocks"][i] for i in apick], [aug_data["cands"][i] for i in apick],
                              [aug_rows[i] for i in apick], builder)
        yo = [aug_labels[i] for i in apick]

        seen = np.zeros(len(ctx.retriever.ids), dtype=bool)
        seen[ctx.truth[tr_idx]] = True
        eidx = np.flatnonzero(np.isin(ev_origin, np.flatnonzero(ctx.folds == k)))
        eblocks = refresh_category([ev_data["blocks"][i] for i in eidx], [ev_data["cands"][i] for i in eidx],
                                   [ev_rows[i] for i in eidx], builder)
        for m, (XX, yy) in {"a": (X, y), "c": (X + Xo, y + yo)}.items():
            predict = lgb_learner(150)(XX, yy)
            for i, sc in zip(eidx, _predict_blocks(predict, eblocks)):
                cand = np.asarray(ev_data["cands"][i])
                if len(sc) == 0:
                    continue
                bad = seen[cand]
                sc = np.where(bad, -np.inf, sc) if bad.any() and not bad.all() else sc
                results[m][i] = cand[int(np.argmax(sc))] == ev_truth[i]
        print(f"fold {k}: a {results['a'][eidx].mean():.4f} c {results['c'][eidx].mean():.4f}", flush=True)

    recall = np.array([t in set(np.asarray(c).tolist()) for c, t in zip(ev_data["cands"], ev_truth)])
    nogt = ev["nogt"].to_numpy()
    summary = {"rows": len(ev), "recall": float(recall.mean()),
               **{f"{m}_{part}": float(v[mask].mean()) for m, v in results.items()
                  for part, mask in (("all", np.ones(len(ev), bool)), ("gt", ~nogt), ("nogt", nogt))},
               "c_vs_a_gain": int((results["c"] & ~results["a"]).sum()),
               "c_vs_a_loss": int((results["a"] & ~results["c"]).sum())}
    print(json.dumps(summary, indent=1))
    with open("experiments/oof_organizer_noise.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
