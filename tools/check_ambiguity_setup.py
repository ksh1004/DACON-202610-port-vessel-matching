"""Check weighted CE mathematics and the unchanged training inputs."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.ambiguity_loss import ranking_loss
from src.common import ROOT, load_catalog
from src.encoder_phonetic_experiment import MODEL_DIR
from src.encoder_phonetics import encode_pairs
from tools.ambiguity_encoder_experiment import TAG, check_lock, fingerprint, masks, prepare


def main():
    protocol = check_lock()
    scores = torch.tensor([[.1, .4, -.5, .8], [.2, -.1, .3, -.4]], requires_grad=True)
    zero = np.zeros(tuple(scores.shape), dtype=bool)
    original = F.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long))
    unmasked = ranking_loss(scores, zero)
    torch.testing.assert_close(unmasked, original, rtol=0, atol=0)
    plain_gradient = torch.autograd.grad(original, scores)[0]
    torch.testing.assert_close(torch.autograd.grad(unmasked, scores)[0], plain_gradient, rtol=0, atol=0)
    mask = np.array([[False, True, False, False], [False, False, True, False]])
    weight = torch.where(torch.tensor(mask), .25, 1.)
    expected = (-scores[:, 0] + torch.log((torch.exp(scores) * weight).sum(axis=1))).mean()
    changed = ranking_loss(scores, mask)
    torch.testing.assert_close(changed, expected, rtol=1e-6, atol=1e-7)
    gradient = torch.autograd.grad(changed, scores)[0]
    assert gradient[0, 1] < plain_gradient[0, 1]
    rejected = False
    try: ranking_loss(scores, np.ones(tuple(scores.shape), dtype=bool))
    except ValueError: rejected = True
    assert rejected
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, padding_side="right")
    catalog = load_catalog().to_dict("records")
    checks = []
    for fold in protocol["screen_folds"]:
        data = prepare(fold)
        assert len(data["groups"]) == 2400
        np.testing.assert_array_equal(data["masks"]["targeted"].sum(1), data["masks"]["random"].sum(1))
        assert not data["masks"]["none"].any()
        for position in (0, 1199, 2399):
            g = data["selected"][position]
            choices = [catalog[int(j)] for j in g["candidates"]]
            row = {**g["row"], "sample_id": "ignored", "call_sign": "ignored", "target_vessel_id": "ignored"}
            assert encode_pairs(tokenizer, row, choices, g["beams"]) == data["groups"][position]
            actual = masks(row, choices, g["beams"], fold, position)
            np.testing.assert_array_equal(actual["targeted"], data["masks"]["targeted"][position])
            perm = np.r_[0, np.arange(7, 0, -1)]
            moved = masks(row, [choices[int(j)] for j in perm], g["beams"], fold, position)
            np.testing.assert_array_equal(moved["targeted"], actual["targeted"][perm])
        checks.append({"fold": fold, "groups": 2400, "all_tokens_equal_legacy": True,
                       "per_query_relaxed_counts_equal": True, "id_and_callsign_ignored": True,
                       "positive_preserved": True, "target_mask_permutation_exact": True})
    result = {"status": "PASS", "fingerprint": fingerprint(), "unmasked_loss_and_gradient_exact": True,
              "manual_weighted_loss_equal": True, "positive_mask_rejected": True, "folds": checks}
    (ROOT / "experiments" / f"{TAG}_setup_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
