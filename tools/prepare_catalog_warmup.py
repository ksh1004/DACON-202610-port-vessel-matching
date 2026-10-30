"""Freeze a name-only synthetic corpus from novel official catalog entities."""
import hashlib
import json
import re
from collections import Counter, defaultdict

import joblib
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from src.common import ROOT, compact_name, load_train, load_catalog
from src.encoder_phonetic_experiment import MODEL_DIR
from src.encoder_phonetics import encode_pairs, english_key, query_key


FOLDER = ROOT / "experiments/catalog_synthetic_v1"
SIZE = 1200


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_corpus():
    path = FOLDER / "warm_corpus.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["builder_sha256"] == digest(ROOT / "tools/prepare_catalog_warmup.py")
        return value
    raw = [json.loads(x) for x in (FOLDER / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw) >= 1536, "Generate the fixed first 1536 names before freezing"
    raw = raw[:1536]
    train, catalog = load_train(), load_catalog()
    excluded = set(train.target_vessel_id)
    known = {compact_name(n) for n in catalog.loc[catalog.vessel_id.isin(excluded), "vessel_name"]}
    valid, owners = [], defaultdict(set)
    for row in raw:
        assert row["vessel_id"] not in excluded and compact_name(row["latin_name"]) not in known
        values = []
        for name in row["hangul_names"]:
            score = Levenshtein.normalized_similarity(
                query_key(name).replace(" ", ""), english_key(row["latin_name"]).replace(" ", ""))
            if re.search(r"[A-Za-z]", name) or score < .9:
                continue
            if re.findall(r"\d+", name) != re.findall(r"\d+", row["latin_name"]):
                continue
            item = {**{k: row[k] for k in ("catalog_index", "vessel_id", "latin_name")},
                    "hangul_name": name, "consonant_similarity": score}
            values.append(item)
            owners[compact_name(name)].add(compact_name(row["latin_name"]))
        valid.append(values)
    # Give distinct catalog names priority over second beam variants.
    selected, seen = [], set()
    for beam in range(2):
        for values in valid:
            if len(values) <= beam:
                continue
            row = values[beam]
            key = (row["vessel_id"], compact_name(row["hangul_name"]))
            if len(owners[key[1]]) != 1 or key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) == SIZE:
                break
        if len(selected) == SIZE:
            break
    assert len(selected) == SIZE
    count = len({r["vessel_id"] for r in selected})
    assert count >= 600
    value = {"records": selected, "groups": SIZE, "distinct_catalog_entities": count,
             "source_limit": 1536, "minimum_consonant_similarity": .9,
             "all_official_train_ids_and_full_names_excluded": True,
             "query_and_document_metadata": "all null", "latin_beams": [],
             "word_deletion": False, "weak_labels_not_verified_pronunciations": True,
             "builder_sha256": digest(ROOT / "tools/prepare_catalog_warmup.py"),
             "generator_manifest_sha256": digest(FOLDER / "manifest.json"),
             "generated_prefix_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
             "data_hashes": {n: digest(ROOT / n) for n in ("train.csv", "vessel_catalog.csv")}}
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return value


def nearest(matrix, vector, limit=100):
    scores = (matrix @ vector.T).tocoo()
    order = np.argsort(-scores.data, kind="stable")[:limit]
    return scores.row[order].tolist()


def prepare(tokenizer):
    corpus = freeze_corpus()
    fp = hashlib.sha256()
    for path in (FOLDER / "warm_corpus.json", ROOT / "tools/prepare_catalog_warmup.py",
                 ROOT / "src/encoder_phonetics.py", ROOT / "src/neural_pair.py",
                 MODEL_DIR / "tokenizer.json", MODEL_DIR / "codex_revision.json"):
        fp.update(path.read_bytes())
    cache = FOLDER / f"warm_groups_{fp.hexdigest()[:12]}.joblib"
    if cache.exists():
        data = joblib.load(cache)
        assert data["fingerprint"] == fp.hexdigest()
        return data
    catalog = load_catalog()
    keys = catalog.vessel_name.map(compact_name).tolist()
    phonetics = catalog.vessel_name.map(english_key).str.replace(" ", "", regex=False).tolist()
    lexical = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), dtype=np.float32)
    lm = lexical.fit_transform(keys)
    phonetic = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), dtype=np.float32)
    pm = phonetic.fit_transform(phonetics)
    groups, chosen, hard_counts = [], [], Counter()
    for number, row in enumerate(corpus["records"]):
        i = row["catalog_index"]
        assert catalog.iloc[i].vessel_id == row["vessel_id"]
        negatives, used = [], {keys[i]}
        sources = [nearest(lm, lexical.transform([keys[i]])),
                   nearest(pm, phonetic.transform([query_key(row["hangul_name"]).replace(" ", "")]))]
        for label, source, limit in (("canonical", sources[0], 5), ("phonetic", sources[1], 7),
                                     ("fallback", sources[0], 7)):
            before = len(negatives)
            for j in source:
                if keys[j] in used:
                    continue
                negatives.append(j)
                used.add(keys[j])
                if len(negatives) == limit:
                    break
            hard_counts[label] += len(negatives)-before
            if len(negatives) == 7:
                break
        assert len(negatives) == 7
        indices = [i]+negatives
        query = {"vessel_name": row["hangul_name"]}
        documents = [{"vessel_name": catalog.vessel_name.iloc[j]} for j in indices]
        groups.append(encode_pairs(tokenizer, query, documents, []))
        chosen.append(indices)
        if (number+1) % 200 == 0:
            print(f"Prepared name-only synthetic groups {number+1}/{SIZE}", flush=True)
    data = {"groups": groups, "candidates": chosen, "records": corpus["records"],
            "fingerprint": fp.hexdigest(), "protocol": {k: v for k, v in corpus.items() if k != "records"},
            "negative_counts": dict(hard_counts),
            "at_token_cap": sum(len(t) == 256 for g in groups for t in g)}
    assert data["at_token_cap"] == 0
    joblib.dump(data, cache, compress=3)
    (FOLDER / "warm_preparation.json").write_text(json.dumps({k: v for k, v in data.items()
        if k not in ("groups", "records", "candidates")}, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(tokenizer)
    print(json.dumps({"groups": len(data["groups"]), "fingerprint": data["fingerprint"],
                      "distinct_names": data["protocol"]["distinct_catalog_entities"]}), flush=True)
