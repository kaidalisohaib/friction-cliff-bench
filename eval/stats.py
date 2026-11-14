"""Statistics for the eval harness (EVAL_PROTOCOL.md §3).

- Wilson 95% interval for a binary rate.
- Hierarchical bootstrap for paired differences: resample training seeds (outer), then scenarios (inner),
  keeping pairs intact (method A and method B on the same scenario instance).
- TOST equivalence check at a margin (H1).
- Holm–Bonferroni over the confirmatory family.
Pure NumPy; deterministic given the rng seed.
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def hier_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, rng=None,
                        pair_seeds: bool = False) -> np.ndarray:
    """Bootstrap distribution of mean(a) - mean(b) for paired arrays of shape (seeds, scenarios).

    a[i, j], b[i, j]: metric of method A / B, training seed i, scenario j (same scenario instance). If the two
    methods have different seed counts, pass them separately with b of shape (seeds_b, scenarios); seeds are
    resampled independently per method, scenarios jointly (that keeps the pairing).
    pair_seeds=True: seed i of A and seed i of B share a training input (e.g. two students of the same teacher
    seed), so seeds are resampled jointly too. Needs equal seed counts.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.atleast_2d(np.asarray(a, float))
    b = np.atleast_2d(np.asarray(b, float))
    assert a.shape[1] == b.shape[1], "paired: same scenarios"
    n_sc = a.shape[1]
    if pair_seeds:
        assert a.shape[0] == b.shape[0], "pair_seeds: same seed count"
    out = np.empty(n_boot)
    for t in range(n_boot):
        ia = rng.integers(0, a.shape[0], a.shape[0])
        ib = ia if pair_seeds else rng.integers(0, b.shape[0], b.shape[0])
        js = rng.integers(0, n_sc, n_sc)
        out[t] = a[ia][:, js].mean() - b[ib][:, js].mean()
    return out


def ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(dist, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def p_two_sided(dist: np.ndarray) -> float:
    """Bootstrap p-value for H0: difference = 0 (fraction of the distribution on the other side of 0, x2)."""
    d = np.asarray(dist)
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def tost(dist: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests on a bootstrap distribution of a difference: equivalent if the (1 - 2 alpha)
    interval lies inside (-margin, margin). p = max of the two one-sided p-values."""
    d = np.asarray(dist)
    lo, hi = np.quantile(d, [alpha, 1 - alpha])
    p_low = float(np.mean(d <= -margin))  # H0: diff <= -margin
    p_high = float(np.mean(d >= margin))  # H0: diff >= margin
    return {"equivalent": bool(lo > -margin and hi < margin), "ci90": (float(lo), float(hi)), "p": max(p_low, p_high)}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm–Bonferroni step-down: returns {name: rejected}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for i, (name, p) in enumerate(items):
        if stop or p > alpha / (m - i):
            stop = True
            out[name] = False
        else:
            out[name] = True
    return out
