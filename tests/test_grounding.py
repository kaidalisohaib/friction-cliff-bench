"""The understeer fit of scripts/grounding_comma.py recovers a known steering ratio and gradient."""

import math

import numpy as np

from scripts.grounding_comma import fit, sim_fit
from sim.params import G, VehicleParams


def test_fit_recovers_ratio_and_gradient_from_synthetic_segments():
    rng = np.random.default_rng(0)
    L, SR, K = 2.8, 14.0, math.radians(2.0) / G  # 2 deg/g
    segs = []
    for _ in range(50):
        v = rng.uniform(10, 30, 200)
        ay = rng.uniform(-3, 3, 200)
        roll = rng.uniform(-0.02, 0.02, 200)
        swa = SR * (L * ay / v**2 + K * ay) + 0.3 * G * roll + 0.01 + rng.normal(0, 0.002, 200)
        segs.append(np.stack([v, ay, roll, swa], 1))
    f = fit(segs, L)
    assert abs(f["SR_eff"] - SR) < 0.1
    assert abs(f["K_deg_per_g"] - 2.0) < 0.05


def test_simulator_fit_gives_unit_ratio():
    assert abs(sim_fit(VehicleParams())["SR_eff"] - 1.0) < 0.01
