"""Candidate-pair input with query-local transliteration evidence only."""
from __future__ import annotations

import json

from src.neural_pair import PREFIX, SUFFIX, record_text


INSTRUCTION = (
    "Match the ship records. Latin spellings are uncertain transliterations. "
    "Names may omit words; missing fields are unknown. Compare names, numbers and tonnage."
)


def encode_pairs(tokenizer, row, catalog_rows, beams):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    query = json.loads(record_text(row))
    # No catalog spellings or labels enter this list: the existing Kanana
    # model generates these solely from the current query name.
    query["possible_latin_spellings"] = list(dict.fromkeys(str(b) for b in beams))[:4]
    query = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    texts = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {record_text(c, catalog=True)}"
             for c in catalog_rows]
    tokens = tokenizer(texts, add_special_tokens=False, truncation=True,
                       max_length=256-len(prefix)-len(suffix)).input_ids
    return [prefix + part + suffix for part in tokens]
