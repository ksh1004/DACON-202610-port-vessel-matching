"""Offline phonetic variants of official catalog names; excludes all train entities."""
import argparse
import hashlib
import json
import re
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import ROOT, compact_name, is_hangul, load_train, load_catalog
from src.llm_cache import MODEL_DIR


PROMPT = (
    "Convert the supplied Latin-alphabet ship name phonetically into Korean Hangul. "
    "Render each original word and keep spaces between words. Preserve all digits exactly. "
    "Keep the complete name. Do not translate the meaning, complete another name, or add ship facts. "
    "Output one Korean rendering only, without explanations or quotes."
)
TYPES = {"Bulk", "Container", "PCC", "GCargo", "Chem & Oil", "Chemical Parcel", "Chemical Unknown", "LPG", "LNG"}


def sample_catalog():
    train, cat = load_train(), load_catalog()
    excluded = set(train.target_vessel_id)
    known_names = {compact_name(n) for n in cat.loc[cat.vessel_id.isin(excluded), "vessel_name"]}
    pool = cat.loc[~cat.vessel_id.isin(excluded) & cat.vessel_type.isin(TYPES)]
    counter, names, chosen = Counter(), set(), []
    for i in np.random.default_rng(20261028).permutation(pool.index.to_numpy()):
        row = cat.loc[int(i)]
        name = str(row.vessel_name)
        key = compact_name(name)
        words = re.findall(r"[A-Za-z0-9]+", name.lower())
        if key in known_names or key in names or len(key) < 4 or len(name) > 55 or not words:
            continue
        if counter[words[0]] >= 8:
            continue
        chosen.append({"catalog_index": int(i), "vessel_id": row.vessel_id, "latin_name": name})
        names.add(key)
        counter[words[0]] += 1
        if len(chosen) == 3000:
            break
    assert len(chosen) == 3000
    assert not {r["vessel_id"] for r in chosen} & excluded
    return chosen


def clean(value):
    value = value.strip().split("\n")[0].strip(' "\'`*')
    value = re.sub(r"^(?:Korean|Hangul|Answer)\s*[:：]\s*", "", value, flags=re.I)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261028)
    selected = sample_catalog()
    folder = ROOT / "experiments/catalog_synthetic_v1"
    folder.mkdir(parents=True, exist_ok=True)
    config = {"prompt": PROMPT, "generator": str(MODEL_DIR), "beams": 2, "batch": args.batch,
              "max_new_tokens": 40, "seed": 20261028, "records": selected,
              "all_official_train_vessels_and_names_excluded": True,
              "source_sha256": hashlib.sha256((ROOT / "tools/generate_catalog_queries.py").read_bytes()).hexdigest(),
              "data_hashes": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ("train.csv", "vessel_catalog.csv")},
              "model_file_bytes": (MODEL_DIR / "model.safetensors").stat().st_size,
              "model_config_sha256": hashlib.sha256((MODEL_DIR / "config.json").read_bytes()).hexdigest()}
    config_path = folder / "manifest.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    path = folder / "queries.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    assert all(row["vessel_id"] == selected[i]["vessel_id"] for i, row in enumerate(existing))
    todo = selected[len(existing):min(args.limit, 3000)]
    print(f"Catalog synthesis: {len(existing)} already generated, {len(todo)} new", flush=True)
    if not todo:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    with path.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start+args.batch]
            texts = [tokenizer.apply_chat_template([{"role": "system", "content": PROMPT}, {"role": "user", "content": row["latin_name"]}],
                                                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
            inputs = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
            inputs.pop("token_type_ids", None)
            output = model.generate(**inputs, max_new_tokens=40, num_beams=2, num_return_sequences=2,
                                    do_sample=False, early_stopping=True, pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, row in enumerate(batch):
                values = list(dict.fromkeys(clean(v) for v in decoded[i*2:i*2+2]))
                valid = [v for v in values if is_hangul(v) and re.findall(r"\d+", v) == re.findall(r"\d+", row["latin_name"])]
                stream.write(json.dumps({**row, "raw": values, "hangul_names": valid}, ensure_ascii=False)+"\n")
            stream.flush()
            print(f"Generated {len(existing)+min(start+args.batch,len(todo))}/{min(args.limit,3000)} names", flush=True)


if __name__ == "__main__":
    main()
