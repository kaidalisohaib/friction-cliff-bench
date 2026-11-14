"""Eval statistics (EVAL_PROTOCOL.md §3): checked against known values and by simulation."""

import numpy as np

from eval.stats import ci, hier_bootstrap_diff, holm, p_two_sided, tost, wilson


def test_wilson_matches_textbook_values_and_protocol_half_widths():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3  # standard Wilson 95% for 50/100
    lo, hi = wilson(500, 1000)
    assert abs((hi - lo) / 2 - 0.031) < 0.001  # EVAL_PROTOCOL: +-3.1 pp at p = 0.5, n = 1000
    lo, hi = wilson(100, 1000)
    assert abs((hi - lo) / 2 - 0.019) < 0.001  # +-1.9 pp at p = 0.1
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0


def test_wilson_coverage_by_simulation():
    rng = np.random.default_rng(1)
    p, n, hits = 0.12, 200, 0
    for _ in range(4000):
        lo, hi = wilson(int(rng.binomial(n, p)), n)
        hits += lo <= p <= hi
    assert 0.93 < hits / 4000 < 0.97


def test_paired_bootstrap_ci_covers_the_true_difference_and_uses_pairing():
    rng = np.random.default_rng(2)
    n_sc = 400
    difficulty = rng.uniform(0, 1, n_sc)  # shared per-scenario effect: pairing should cancel it
    a = (rng.uniform(0, 1, (5, n_sc)) < 0.2 + 0.6 * difficulty).astype(float)
    b = (rng.uniform(0, 1, (5, n_sc)) < 0.15 + 0.6 * difficulty).astype(float)
    d = hier_bootstrap_diff(a, b, n_boot=2000, rng=np.random.default_rng(3))
    lo, hi = ci(d)
    assert lo < 0.05 < hi
    # the paired interval is narrower than the same hierarchical bootstrap with scenarios resampled
    # independently per method (which ignores that both methods saw the same scenario instances)
    rng2 = np.random.default_rng(4)
    unpaired = []
    for _ in range(2000):
        ia, ib = rng2.integers(0, 5, 5), rng2.integers(0, 5, 5)
        unpaired.append(a[ia][:, rng2.integers(0, n_sc, n_sc)].mean() - b[ib][:, rng2.integers(0, n_sc, n_sc)].mean())
    assert (hi - lo) < (np.quantile(unpaired, 0.975) - np.quantile(unpaired, 0.025))


def test_p_value_tost_and_holm():
    rng = np.random.default_rng(5)
    assert p_two_sided(rng.normal(1.0, 0.2, 20000)) < 1e-3
    assert p_two_sided(rng.normal(0.0, 0.2, 20000)) > 0.5
    assert tost(rng.normal(0.0, 0.01, 20000), 0.05)["equivalent"]
    assert not tost(rng.normal(0.04, 0.01, 20000), 0.05)["equivalent"]
    assert holm({"H1": 0.01, "H2": 0.04, "H3": 0.02}) == {"H1": True, "H2": True, "H3": True}
    assert holm({"H1": 0.01, "H2": 0.04, "H3": 0.026}) == {"H1": True, "H3": False, "H2": False}


def test_hier_bootstrap_diff_pair_seeds_removes_shared_seed_variance():
    import numpy as np
    from eval.stats import ci, hier_bootstrap_diff
    rng = np.random.default_rng(0)
    seed_eff = np.array([0.0, 0.03, 0.15, 0.05, 0.7])[:, None]  # large shared seed effect
    b = np.clip(seed_eff + rng.normal(0, 0.01, (5, 400)), 0, 1)
    a = b + 0.01  # A is B plus a constant
    ind = ci(hier_bootstrap_diff(a, b, 2000, np.random.default_rng(1)))
    par = ci(hier_bootstrap_diff(a, b, 2000, np.random.default_rng(1), pair_seeds=True))
    assert par[1] - par[0] < 0.1 * (ind[1] - ind[0])
    assert par[0] - 1e-9 <= 0.01 <= par[1] + 1e-9
