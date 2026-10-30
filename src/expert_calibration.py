"""Low-dimensional, regularized log-odds correction around the frozen v12.

Inputs are three row-local expert distributions and a tonnage-availability bit.
No new ship features or identity maps are fitted. Zero coefficients reproduce
the anchor. This is not scalar temperature scaling and does not promise an
accuracy improvement merely because fitting log loss decreases.
"""
import numpy as np
from scipy.optimize import minimize

L2 = .05
LIMIT = .5
LOG_RATIO_CLIP = 8.
MODES = ("conditional6", "pooled3")


def positive_probability(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all() and np.all(values >= 0) and values.max() > 0
    return values / np.sort(values).sum(dtype=np.float64)


def probability(logits):
    logits = np.asarray(logits, dtype=np.float64)
    assert np.isfinite(logits).all()
    return positive_probability(np.exp(logits-logits.max()))


def missing_size(row):
    for field in ("gross_tonnage", "international_gross_tonnage"):
        try:
            value = float(row.get(field))
        except (ValueError, TypeError):
            continue
        if np.isfinite(value) and value > 0:
            return False
    return True


def features(anchor, experts, row, mode):
    anchor = positive_probability(anchor)
    x = np.column_stack([np.log(np.maximum(positive_probability(p), 1e-15))
                         - np.log(np.maximum(anchor, 1e-15)) for p in experts])
    x = np.clip(x, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    if mode == "conditional6":
        padded = np.zeros((len(anchor), 6), dtype=np.float64)
        start = 3*int(missing_size(row))
        padded[:, start:start+3] = x
        x = padded
    else:
        assert mode == "pooled3"
    return x


def corrected_logits(anchor, x, coefficients):
    # Fixed per-candidate arithmetic, also for a one-candidate subset of x.
    result = np.log(np.maximum(np.asarray(anchor, dtype=np.float64), 1e-15))
    for j, value in enumerate(coefficients):
        result = result + x[:, j]*value
    return result


def objective(beta, offsets, x, targets, weights):
    z = offsets + np.einsum("ncd,d->nc", x, beta, optimize=False)
    zmax = z.max(axis=1)
    e = np.exp(z-zmax[:, None]); sums = e.sum(axis=1)
    p = e/sums[:, None]
    loss = np.sum(weights*(zmax+np.log(sums)-z[np.arange(len(z)), targets]))
    p[np.arange(len(z)), targets] -= 1
    grad = np.einsum("n,nc,ncd->d", weights, p, x, optimize=False)
    return float(loss+.5*L2*np.dot(beta, beta)), grad+L2*beta


def fit(offsets, x, targets, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights/weights.sum()
    zero = np.zeros(x.shape[-1], dtype=np.float64)
    before = objective(zero, offsets, x, targets, weights)[0]
    solution = minimize(objective, zero, args=(offsets, x, targets, weights), jac=True,
                        method="L-BFGS-B", bounds=[(-LIMIT, LIMIT)]*len(zero),
                        options={"maxiter":200, "ftol":1e-12, "gtol":1e-8})
    assert solution.success, solution.message
    assert np.isfinite(solution.x).all() and np.isfinite(solution.fun)
    assert solution.fun <= before+1e-12
    return dict(coefficients=solution.x.tolist(), objective_before=float(before),
                objective_after=float(solution.fun), iterations=int(solution.nit),
                converged=bool(solution.success), projected_optimizer_message=str(solution.message))
