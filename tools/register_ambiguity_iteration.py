"""Lock the single loss recipe and controls before any new training or scores."""
import datetime
import json

from src.common import ROOT
from src.encoder_phonetic_experiment import MODEL_DIR
from tools.ambiguity_encoder_experiment import PROTOCOL, TAG, sha


def main():
    assert not PROTOCOL.exists(), "Do not change a registered experiment"
    sources = ["src/ambiguity_loss.py", "tools/ambiguity_encoder_experiment.py", "tools/audit_observation_ambiguity.py",
               "tools/audit_ambiguity_iteration.py", "tools/check_ambiguity_setup.py", "tools/run_ambiguity_iteration.py",
               "tools/register_ambiguity_iteration.py", "tools/fresh_ambiguity_guard.py", "tools/fresh_encoder_extension.py",
               "tools/audit_encoder_extension.py", "src/focused_encoder_experiment.py", "src/encoder_phonetic_experiment.py",
               "src/encoder_extended_experiment.py", "src/encoder_phonetics.py", "src/encoder_pair.py", "src/pair_views.py",
               "src/pair_evidence.py", "src/neural_pair.py", "src/features.py", "src/retrieval.py", "src/augment.py",
               "train.csv", "vessel_catalog.csv", "experiments/folds.csv"]
    from src.llm_cache import CACHE_PATH
    sources.append(str(CACHE_PATH.relative_to(ROOT)))
    for fold in range(5):
        sources.append(f"experiments/artifacts/strict_han_v1/base_fold{fold}.txt")
    protocol = {
        "tag": TAG, "registered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "registered_before_new_training_or_scores": True, "reference": "v12", "reference_public": .9065951735,
        "reference_server_seconds": 591, "training_groups": 2400, "epochs": 1, "seed": 42,
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05, "learning_rate": 1e-4,
        "groups_per_batch": 2, "group_size": 8, "max_tokens": 256,
        "arms": ["none", "targeted", "random"], "only_selectable_arm": "targeted",
        "negative_weight": .25, "loss": "cross_entropy(scores + log(negative_weights), positive0); positive_weight1, unselected_negative_weight1, selected_negative_weight0.25",
        "target_rule": "Actual Hangul; a query-only cached Latin beam exactly matches a contiguous word span in both positive and negative catalog names; same candidate flag/type when query has that field; same positive catalogGT when either queryGT orIGT is positive. Other vessel IDs stay negatives; training pressure only is relaxed.",
        "random_control": "Same number per identical query, uniform choice among all7negatives without replacement; independent numpy RNG seed20261029+10000*fold+groupPosition; overlap with targeted masks allowed.",
        "training_inputs": "Exact all19200 tokenized pairs from the original v12 phonetic groups; all three arms share initialization, ordering, shape and updates. Reset seed immediately before model initialization.",
        "fixed_mix": {"v8": .75, "qwen_v11": .125, "new_encoder": .125},
        "screen_folds": [1, 4], "screen_gate": {"testlike_gain_min": .003, "minimum_fold_delta": -.002,
            "hard_hangul_delta_min": -.002, "drop_word_delta_min": -.002, "clean_lost_rows_max": 1,
            "gain_vs_unmasked_min": .001, "gain_vs_random_min": .001},
        "fresh_guard": {"fold": 3, "minimum_mean_accuracy_delta": -.002, "seed": 2026102901,
            "scenarios": ["gt_only_missing", "igt_only_missing", "gt15pct_conflict", "word_drop_gt", "truncated_gt", "word_drop_missing", "truncated_missing"],
            "few_shot_vessels_excluded": True, "training_only_retrieval": True, "previously_used_vessels_not_pristine": True},
        "full_gate": {"all_5_folds_required": True, "testlike_gain_min": .005, "hard_hangul_delta_min": -.001,
            "drop_word_delta_min": -.002, "oof_delta_min": -.0006, "positive_folds_min": 4,
            "gain_vs_unmasked_min": .001, "gain_vs_random_min": .001},
        "model": {"path": str(MODEL_DIR), "revision": json.loads((MODEL_DIR/"codex_revision.json").read_text()),
                  "config_sha256": sha(MODEL_DIR/"config.json"), "tokenizer_sha256": sha(MODEL_DIR/"tokenizer.json")},
        "source_hashes": {p: sha(ROOT/p) for p in sources},
        "no_hyperparameter_or_mask_rule_search": True, "random_or_unmasked_not_fallback_candidates": True,
        "no_automatic_byt5_or_context_prior_after_failure": True, "callsign_or_identity_lookup": False,
        "actual_competition_submission": False,
        "caveat": "Repeated exploratory validation; historical v8 outer retrieval mapper exception. Training retrieval/miner strictly vessel-held-out. Existing public forward-LLM fewshots; report no-shot-vessel sensitivity. BGE12.5% replacement is a causal loss pilot, not a direct route to Public0.97."}
    PROTOCOL.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Registered {PROTOCOL}", flush=True)


if __name__ == "__main__": main()
