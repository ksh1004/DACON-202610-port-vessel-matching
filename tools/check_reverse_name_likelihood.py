"""Independent likelihood, permutation and batching checks on synthetic names."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.reverse_name_likelihood import ReverseNameScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pad-length", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    load_start = time.perf_counter()
    scorer = ReverseNameScorer(batch_size=args.batch_size, pad_length=args.pad_length)
    load_seconds = time.perf_counter()-load_start
    # Constructed names, not selected from held-out outcomes.
    query = "블루 오션 7"
    names = ["Blue Ocean 7", "Blue Ocean 8", "Red Ocean 7", "Blue River 7",
             "Ocean Blue 7", "Ocean 7", "Blue Ocean", "Blue Ocean 7"]
    start = time.perf_counter()
    result = scorer.score_details(query, names)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    repeated = scorer.score_details(query, names)
    reversed_result = scorer.score_details(query, names[::-1])
    for field in ("sum_logp", "mean_logp", "eos_logp", "sum_with_eos", "mean_with_eos"):
        np.testing.assert_array_equal(result[field], repeated[field])
        np.testing.assert_array_equal(result[field], reversed_result[field][::-1])
    assert result["mean_logp"][0] == result["mean_logp"][-1]

    # Full logits plus independently shifted cross entropy, using identical
    # encoded shape to separate a correctness check from BF16 kernel effects.
    canonical = sorted(set(names))
    enc, target, actual = scorer.encode(query, canonical[:scorer.batch_size])
    with torch.inference_mode():
        logits = scorer.model(**enc, use_cache=False, logits_to_keep=0).logits
        labels = enc["input_ids"].clone()
        labels[:, :-len(target)] = -100
        token_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                     labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
        manual = -token_loss.reshape(labels.shape[0], -1)[:, -len(target):].sum(-1).cpu().numpy()
    by_name = dict(zip(names, result["sum_logp"]))
    optimized = np.array([by_name[n] for n in canonical[:actual]])
    # A full-vocabulary projection over all sequence positions may select a
    # different GEMM kernel than logits_to_keep, even at the same input shape.
    manual_max_delta = float(np.max(np.abs(manual[:actual]-optimized)))
    assert manual_max_delta < .10, (manual[:actual], optimized)
    # Verify target-only gather formula exactly with the same truncated logits.
    with torch.inference_mode():
        logits = scorer.model(**enc, use_cache=False, logits_to_keep=scorer.max_target_tokens+1).logits[:, -len(target)-1:].float()
        target_tensor = torch.tensor(target, device=scorer.device)
        manual_target = F.log_softmax(logits[:, :-1], dim=-1).gather(
            -1, target_tensor[None, :, None].expand(logits.shape[0], -1, 1)).squeeze(-1)
        independent = manual_target[:actual].cpu().numpy().sum(axis=-1, dtype=np.float64)
    np.testing.assert_allclose(independent, optimized, rtol=0, atol=2e-5)

    # Evaluate each candidate with the same fixed batch shape, then with batch1.
    singles_fixed = np.array([scorer.score(query, [n])[0] for n in names])
    fixed_batch_single_delta = float(np.max(np.abs(singles_fixed-result["mean_logp"])))
    np.testing.assert_array_equal(singles_fixed, result["mean_logp"])
    pairs = [(query, name) for name in names]
    pair_values = scorer.score_many_pairs(pairs)
    shuffled = scorer.score_many_pairs(pairs[::-1])[::-1]
    np.testing.assert_array_equal(pair_values, shuffled)
    companions = [("레드 리버", "Red River")]+pairs[:6]+[("그린", "Green")]
    shifted = scorer.score_many_pairs(companions)
    np.testing.assert_array_equal(pair_values[:6], shifted[1:7])
    original_batch = scorer.batch_size
    scorer.batch_size = 1
    single_start = time.perf_counter()
    singles_batch1 = np.array([scorer.score(query, [n])[0] for n in names])
    torch.cuda.synchronize()
    batch1_seconds = time.perf_counter()-single_start
    scorer.batch_size = original_batch
    batch1_delta = float(np.max(np.abs(singles_batch1-result["mean_logp"])))

    try:
        scorer.encode(query*200, [names[0]])
        raise AssertionError("Expected explicit no-truncation failure")
    except ValueError as error:
        assert "no truncation performed" in str(error)
    report = {"status": "PASS", "protocol": scorer.protocol(), "load_seconds": load_seconds,
              "unique_candidates": len(set(names)), "batch_call_seconds": seconds,
              "batch1_eight_calls_seconds": batch1_seconds,
              "target_tokens": result["target_tokens"],
              "scores": dict(zip(names, result["mean_logp"].tolist())),
              "exact_repeat": True, "exact_candidate_reverse": True,
              "exact_pair_slot_companion_invariance": True,
              "full_forward_shifted_loss_max_sum_delta": manual_max_delta,
              "fixed_batch_single_candidate_max_mean_delta": fixed_batch_single_delta,
              "batch1_max_mean_delta": batch1_delta,
              "top1_batch_equals_single": names[int(np.argmax(result["mean_logp"]))] == names[int(np.argmax(singles_batch1))],
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "caveat": "Different batch protocols are not interchangeable; exact per-pair caching requires its own invariance evidence."}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
