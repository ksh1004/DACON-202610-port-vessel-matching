"""Training-only relaxation of selected negatives; positive labels stay intact."""
import math


def ranking_loss(scores, relaxed, weight=0.25):
    import torch
    import torch.nn.functional as F

    if not 0 < weight <= 1:
        raise ValueError("Negative weight must be in (0,1]")
    mask = torch.as_tensor(relaxed, dtype=torch.bool, device=scores.device)
    if mask.shape != scores.shape or bool(mask[:, 0].any()):
        raise ValueError("Mask must match scores and preserve the first positive")
    adjustment = torch.where(mask, math.log(weight), 0.0)
    return F.cross_entropy(scores + adjustment,
                           torch.zeros(len(scores), dtype=torch.long, device=scores.device))
