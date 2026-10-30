"""Small train-only character channel with vowel-preserving Korean spelling."""
import re
import unicodedata
from functools import lru_cache

import numpy as np

from src.hangul import normalize_hangul_name


INITIAL = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
MEDIAL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
FINAL = ["", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lk", "lm", "lb", "ls", "lt", "lp", "lh", "m", "p", "ps", "t", "t", "ng", "t", "t", "k", "t", "p", "h"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
INDEX = {c: i for i, c in enumerate(ALPHABET)}
EPS = len(ALPHABET)


def observed(name):
    result = []
    for ch in normalize_hangul_name(name).lower():
        code = ord(ch)-0xac00
        if 0 <= code < 11172:
            result.append(INITIAL[code//588]+MEDIAL[(code%588)//28]+FINAL[code%28])
        elif ch in INDEX:
            result.append(ch)
    return "".join(result)


def canonical(name):
    name = unicodedata.normalize("NFKD", name).lower()
    words = re.findall(r"[a-z0-9]+", name)
    ends = np.cumsum([len(w) for w in words]).tolist()
    return "".join(words), [0]+ends


def initial_costs():
    costs = np.full((EPS+1, EPS+1), 2.5, dtype=float)
    for i, a in enumerate(ALPHABET):
        for j, b in enumerate(ALPHABET):
            if a == b:
                costs[i, j] = .1
            elif a in "aeiouy" and b in "aeiouy":
                costs[i, j] = 1.5
            elif any(a in group and b in group for group in ("gkqc", "bpvf", "dt", "lr", "szjc")):
                costs[i, j] = 1.0
            if a.isdigit() != b.isdigit() or a.isdigit() and a != b:
                costs[i, j] = 9.
    costs[EPS, :] = 1.2
    costs[:, EPS] = 1.2
    for i, ch in enumerate(ALPHABET):
        if ch.isdigit():
            costs[EPS, i] = costs[i, EPS] = 6.
    costs[EPS, EPS] = 100.
    return costs


def align(query, document, boundaries, costs, trace=False):
    """Consume the whole observation; skip complete unobserved edge words."""
    x = [INDEX[c] for c in query]
    y = [INDEX[c] for c in document]
    n, m = len(x), len(y)
    dp = np.full((n+1, m+1), np.inf)
    back = np.zeros((n+1, m+1), dtype=np.int8) if trace else None
    dp[0, 0] = 0
    for j in range(1, m+1):
        dp[0, j] = 0 if j in boundaries else dp[0, j-1]+costs[y[j-1], EPS]
        if trace:
            back[0, j] = 2
    for i in range(1, n+1):
        dp[i, 0] = dp[i-1, 0]+costs[EPS, x[i-1]]
        if trace:
            back[i, 0] = 1
        for j in range(1, m+1):
            a = dp[i-1, j-1]+costs[y[j-1], x[i-1]]
            b = dp[i-1, j]+costs[EPS, x[i-1]]
            c = dp[i, j-1]+costs[y[j-1], EPS]
            if a <= b and a <= c:
                dp[i, j] = a
            elif b <= c:
                dp[i, j] = b
                if trace:
                    back[i, j] = 1
            else:
                dp[i, j] = c
                if trace:
                    back[i, j] = 2
    end = min(boundaries[1:] or [m], key=lambda j: dp[n, j])
    if not trace:
        return float(dp[n, end]/max(n, 1))
    counts = np.zeros_like(costs)
    i, j = n, end
    # Unobserved prefixes are deliberately absent from the learned channel.
    while i > 0:
        action = back[i, j]
        if action == 0:
            counts[y[j-1], x[i-1]] += 1
            i -= 1
            j -= 1
        elif action == 1:
            counts[EPS, x[i-1]] += 1
            i -= 1
        else:
            counts[y[j-1], EPS] += 1
            j -= 1
    return counts


class LearnedAlignment:
    def __init__(self, costs=None):
        self.costs = initial_costs() if costs is None else np.asarray(costs)

    def fit(self, pairs, iterations=4):
        prepared = [(observed(a), *canonical(b)) for a, b in pairs]
        prior = np.exp(-initial_costs())
        prior = 10*prior/prior.sum(axis=1, keepdims=True)
        for step in range(iterations):
            counts = prior.copy()
            for query, document, boundaries in prepared:
                if query and document:
                    counts += align(query, document, boundaries, self.costs, trace=True)
            self.costs = -np.log(counts/counts.sum(axis=1, keepdims=True))
            # The epsilon source is an insertion process, with a fixed .1 mass.
            self.costs[EPS, :] += -np.log(.1)
            self.costs[EPS, EPS] = 100.
            print(f"Learned alignment iteration {step+1}/{iterations}, pairs {len(prepared)}", flush=True)
        return self

    @lru_cache(maxsize=200000)
    def score(self, hangul_name, latin_name):
        query, (document, boundaries) = observed(hangul_name), canonical(latin_name)
        if not query or not document:
            return -100.
        return -10*align(query, document, boundaries, self.costs)
