"""Row-local expert statistics for a nested, supervised final selector."""
import numpy as np

from src.features import FEATURE_NAMES

STAT_NAMES = ["prob", "log_prob", "dense_rank", "logit_gap", "entropy", "top_prob", "top_prob_gap"]
NESTED_NAMES = FEATURE_NAMES+[f"{e}_{s}" for e in ("proxy", "qwen", "bge") for s in STAT_NAMES]+[
    "proxy_mix_prob", "proxy_mix_dense_rank", "proxy_mix_gap",
    "agree_proxy_qwen", "agree_proxy_bge", "agree_qwen_bge"]


def probability(score):
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all() or not len(score):
        raise ValueError("Expert scores must be finite and nonempty")
    x = np.exp(score-score.max())
    # Fixed reduction order also makes candidate permutation checks exact.
    return x/np.sort(x).sum(dtype=np.float64)


def dense_rank(score):
    return np.searchsorted(np.unique(-np.asarray(score)), -np.asarray(score))+1


def stats(score):
    score = np.asarray(score, dtype=np.float64)
    gap = score-score.max()
    z = np.sort(np.exp(gap)).sum(dtype=np.float64)
    logp = gap-np.log(z)
    p = np.exp(gap)/z
    top = np.sort(p)
    entropy = -np.sort(p*logp).sum(dtype=np.float64)
    return np.column_stack((p, logp, dense_rank(score), gap,
                            np.full(len(p), entropy), np.full(len(p), top[-1]),
                            np.full(len(p), top[-1]-top[-2] if len(p)>1 else 1.0)))


def extend(raw, proxy, qwen, bge):
    arrays = [np.asarray(x) for x in (proxy, qwen, bge)]
    if raw.shape != (len(proxy), len(FEATURE_NAMES)) or any(x.shape != (len(proxy),) for x in arrays):
        raise ValueError("Candidate/feature identity mismatch")
    p = [probability(x) for x in arrays]
    mixed = .75*p[0]+.125*p[1]+.125*p[2]
    # Ties are agreement only if the entire tied-top set is identical.
    winners = [x == x.max() for x in arrays]
    agree = [np.full(len(proxy), float(np.array_equal(winners[i], winners[j])))
             for i,j in ((0,1),(0,2),(1,2))]
    extra = np.column_stack((mixed, dense_rank(mixed), mixed-mixed.max(), *agree))
    result = np.column_stack((raw, *(stats(x) for x in arrays), extra)).astype(np.float32)
    assert result.shape[1] == len(NESTED_NAMES) == 105
    return result
