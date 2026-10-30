"""Prospective registration of a single byte transduction/residual candidate."""
import datetime
import json

from src.byte_name_model import MODEL_DIR
from src.common import ROOT
from tools.byte_names_experiment import sha
from tools.prepare_byte_names import FOLDER, PROTOCOL, TAG


def main():
    assert not PROTOCOL.exists()
    files=["src/byte_name_model.py","tools/prepare_byte_names.py","tools/byte_names_experiment.py",
           "tools/check_byte_names.py","tools/run_byte_names.py","tools/fresh_byte_names.py","tools/register_byte_names.py",
           "tools/audit_reverse_ranker.py","tools/blend_matching.py","tools/fresh_ambiguity_guard.py",
           "tools/ambiguity_encoder_experiment.py","tools/audit_observation_ambiguity.py",
           "src/common.py","src/features.py","src/retrieval.py","src/augment.py","src/pair_views.py",
           "src/encoder_phonetics.py","src/encoder_pair.py","src/llm_cache.py","src/llm_translit.py",
           "train.csv","vessel_catalog.csv","experiments/folds.csv","experiments/encoder_bge_phonetic_predictions.joblib",
           "experiments/byte_names_v1_preflight.json"]
    from src.llm_cache import CACHE_PATH
    files.append(str(CACHE_PATH.relative_to(ROOT)))
    for fold in range(5):
        for tag in ("encoder_bge_phonetic","neural_evidence_v1"):
            files.append(f"experiments/artifacts/{tag}/fold{fold}_epoch1_scores.joblib")
    for fold in (1,4): files.append(str((FOLDER/f"fold{fold}_data.joblib").relative_to(ROOT)))
    p={"tag":TAG,"registered_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "registered_before_any_new_training_or_outer_model_scores":True,"reference":"v12","reference_public":.9065951735,
       "model":{"repo":"google/byt5-small","revision":"68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
                "license":"Apache-2.0","parameters":299637760,"weights_sha256":sha(MODEL_DIR/"pytorch_model.bin")},
       "annotation":"Official original Han rows +four global seed7 augmentations; unique exact compact Latin-beam match to one contiguous canonical-name fragment; identical digit groups; exclude ambiguous/no exact match. No synthetic Korean labels. Deduplicate query/fragment pairs, equal query weight across alternate spellings.",
       "inner_validation":"GroupShuffleSplit20% of outer-training Han vessels, seed20261029; purge exact query/fragment pairs shared with inner validation from fit. Select minimum validation tokenNLL among epochs1/3/6/12; refit all outer-training pairs from public checkpoint for selected epoch count. No outer-score checkpoint choice.",
       "training":{"all_parameters_trainable":True,"epochs_horizon":12,"epoch_checks":[1,3,6,12],"batch":16,"seed":42,
                   "optimizer":"Adafactor constant baseLR0.001, relative_stepFalse, scale_parameterFalse, warmup_initFalse, weight_decay0.01",
                   "lr_schedule":"linear warmup10% then linear decay over12epoch horizon, including refits stopping earlier",
                   "global_gradient_clip":1.,"parameter_dtype":"float32","forward_autocast":"bfloat16","attention":"eager","gradient_checkpointing":True},
       "scoring":{"direction":"P(Latin fragment including EOS | observed Korean name)","prefix":"latin: ","dtype":"float32",
                  "TF32":False,"attention":"eager","batch":16,"source_minimum":128,"source_bucket_step":64,
                  "target_minimum":96,"target_bucket_step":32,"last_batch":"repeat last pair to16","truncation":False,
                  "name_marginal":"full canonical name0.5, remaining distinct contiguous fragments uniform0.5; single-word prior1",
                  "runtime":"Each query only; no cross-row pair cache. Offline cache only after exact companion/slot checks and weight hashes."},
       "contenders":["residual30"],"matched_control":"Same fixed0.7*v12_actual_top20 +0.3*name probability, public untuned ByT5 with no task training; not selectable",
       "fixed_residual_weight":.3,"anchor":"Actual v12 distribution restricted and normalised over its original base top20; on its original pruned rows use actual base fallback, not invented neural logits",
       "byte_pruning":"Anchor top2 gap greater than3/7+1e-6 provably preserves winner; never apply old1/3 bound to the new expert",
       "screen_folds":[1,4],"screen_gate":{"testlike_gain_min":.003,"minimum_fold_delta":-.002,"hard_hangul_delta_min":-.002,
                     "drop_word_delta_min":-.002,"clean_lost_rows_max":1,"gain_vs_matched_control_min":.001},
       "fresh_guard":{"fold":3,"minimum_mean_accuracy_delta":-.002,"seed":2026102901,"few_shot_vessels_excluded":True,
                      "training_only_retrieval":True,"previously_used_vessels_not_pristine":True},
       "full_gate":{"all_5_folds_required":True,"testlike_gain_min":.005,"positive_folds_min":4,"hard_hangul_delta_min":-.001,
                    "drop_word_delta_min":-.002,"oof_delta_min":-.0006,"gain_vs_matched_control_min":.001},
       "source_hashes":{f:sha(ROOT/f) for f in files},"lookup_used":False,"actual_competition_submission":False,
       "no_coefficient_search_after_results":True,"no_automatic_context_prior_after_failure":True,
       "caveat":"Repeated exploratory synthetic CV and historical v8 outer retrieval mapper exception. Forward frozen beams have original fewshots; report no-shot sensitivity. Weak fragment supervision can be wrong despite exact spelling. Label-aware oracle used for pretraining architecture analysis, not performance forecast."}
    PROTOCOL.write_text(json.dumps(p,indent=2),encoding="utf-8")
    print(f"Registered {PROTOCOL}",flush=True)


if __name__=="__main__":main()
