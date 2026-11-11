"""Recovery latency (prereg V4) on hand-made traces."""

import numpy as np

from eval.latency import recovery_latency


def _tr(e, r, td=0.2):
    return {"sid": np.array(["a"]), "e": np.array([e], np.float32), "r": np.array([r], np.float32),
            "t_drop": np.array([td])}


def test_latency_counts_from_the_drop_until_the_band_is_held_half_a_second():
    n = 200
    eo, ro = np.zeros(n), np.zeros(n)
    e = np.zeros(n)
    e[10:60] = 0.5  # out of band from step 10 to 59, back in band at step 60 (1.2 s)
    lat, cens = recovery_latency(_tr(e, ro.copy()), _tr(eo, ro))
    # drop at 0.2 s = end of step 10 (k0 = 9); band regained at k = 60 and held 25 steps -> latency (60 - 9) dt
    assert not cens[0] and abs(lat[0] - 51 * 0.02) < 1e-9


def test_never_recovering_is_censored():
    n = 100
    e = np.full(n, 0.5)
    lat, cens = recovery_latency(_tr(e, np.zeros(n)), _tr(np.zeros(n), np.zeros(n)))
    assert cens[0] and lat[0] > 1.5


def test_short_dip_into_the_band_does_not_count():
    n = 200
    e = np.full(n, 0.5)
    e[30:40] = 0.0  # 0.2 s in band: shorter than the 0.5 s hold
    e[100:] = 0.0
    lat, cens = recovery_latency(_tr(e, np.zeros(n)), _tr(np.zeros(n), np.zeros(n)))
    assert not cens[0] and abs(lat[0] - (100 - 9) * 0.02) < 1e-9
