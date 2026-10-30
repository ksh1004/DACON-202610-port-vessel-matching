"""Recompute training groups serially to audit threaded retrieval and label use."""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.common import ROOT, load_train, load_catalog
from src.encoder_phonetic_experiment import MODEL_DIR
from src.encoder_phonetics import encode_pairs
from src.features import FEATURE_NAMES, FeatureBuilder, fit_mappers
from src.focused_encoder_experiment import prepare
from src.llm_cache import BeamCache
from src.retrieval import CatalogRetriever
from src.validate import load_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", choices=["focused", "restore"], default="focused")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    data = prepare(args.fold, args.mode, tokenizer)
    train, catalog = load_train(), load_catalog()
    folds = load_folds(train)
    r = CatalogRetriever(catalog)
    builder = FeatureBuilder(r, *fit_mappers(train.loc[folds != args.fold], catalog))
    r.category_score = builder.category_score
    miner = lgb.Booster(model_file=str(ROOT / "experiments/artifacts/strict_han_v1" / f"base_fold{args.fold}.txt"))
    beams = BeamCache(allow_generate=False)
    cols = [FEATURE_NAMES.index(n) for n in ("llm_ratio_max", "key_lev", "best_gap")]
    positions = data["replaced_positions"]
    pick = [positions[0], positions[len(positions)//2], positions[-1]]
    for position in pick:
        row = data["selected_rows"][position]
        bb = beams([row["vessel_name"]])[0]
        signal = r.generate(pd.DataFrame([row]), llm_beams=[bb])[0]
        c = signal["cand"]
        block = builder.build(row, signal)
        target = r.id_to_index[row["target_vessel_id"]]
        assert (target in c) == (args.mode == "focused")
        scores = miner.predict(block, num_threads=1)
        strength = np.maximum(block[:, cols[0]], block[:, cols[1]])-.2*block[:, cols[2]]
        hard = [int(j) for j in np.argsort(-scores, kind="stable") if c[j] != target][:5]
        lexical = [int(j) for j in np.argsort(-strength, kind="stable") if c[j] != target and j not in hard][:2]
        choices = [catalog.iloc[target].to_dict()]+[catalog.iloc[c[j]].to_dict() for j in hard+lexical]
        actual = encode_pairs(tokenizer, row, choices, bb)
        assert actual == data["groups"][position], position
        changed = dict(row, sample_id="changed", call_sign="changed", target_vessel_id="not-a-model-input")
        assert encode_pairs(tokenizer, changed, choices, bb) == actual
    print(f"PASS: {args.mode} fold{args.fold}; 3 serial groups equal cached groups; candidate criteria and ID/callsign input invariance", flush=True)


if __name__ == "__main__":
    main()
