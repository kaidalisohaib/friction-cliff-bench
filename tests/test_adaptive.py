"""M-B bank scoring identifies the true (surface, mu, delay) without noise; M-L plumbing runs."""

import numpy as np
import pytest
import torch

from control.adaptive import BankMPPI, LearnedMPPI, bank_models
from control.controllers import ORACLE_OR
from control.estimator import Estimator
from control.mppi import MPPIConfig
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams
from sim.warp_sim import ScenarioSim

TRACKS = T.all_tracks()
CFG = MPPIConfig(K=8, H=5, sigma_steer=0.05, sigma_long=0.3, pd_sample=True)


def _drive(surface, mu1, delay_extra, steps=25, W=20):
    sc = S.Scenario("b", 0, 15, "straight", 0.0, "straight", surface, 1e9, 0.0, 15.0, 0.95, mu1, 0, 1.0)
    p = VehicleParams().with_(delay_steps=1 + delay_extra)
    sim = ScenarioSim([sc], p, TRACKS)
    sim.reset_at_start([p])
    sim.step(np.zeros((1, 2)))  # onto the patch (drop at s = 1 m, front axle at 1.25 m)
    snap = sim.snapshot()
    mb = BankMPPI(sim, [sc], CFG, 0, snap, window=W, noise_scale=0.0)
    mb.on_reset()
    t = np.arange(steps) * 0.02
    for k in range(steps):
        a = np.array([[0.25 * np.sin(2 * np.pi * 1.0 * t[k]), 0.4 * np.sin(2 * np.pi * 0.7 * t[k])]], np.float32)
        sim.step(a)
        mb.feat.step()
        mb.on_step()
    mb.update_model()
    return mb.models[int(mb.sel[0])]


@pytest.mark.parametrize("surface,mu1,dx", [("ice", 0.1, 0), ("snow", 0.25, 1), ("wet", 0.7, 2)])
def test_bank_identifies_true_model_without_noise(surface, mu1, dx):
    sname, mu, d = _drive(surface, mu1, dx)
    assert sname == surface and abs(mu - mu1) < 1e-6 and d == dx, (sname, mu, d)


def test_bank_size():
    models, vecs = bank_models()
    assert len(models) == 45 and vecs.shape == (45, 8)


def test_learned_mppi_runs():
    sc = S.Scenario("l", 0, 15, "straight", 0.0, "straight", "ice", 1e9, 0.0, 15.0, 0.95, 0.1, 0, 30.0)
    sim = ScenarioSim([sc, sc], VehicleParams(), TRACKS)
    sim.reset_at_start()
    snap = sim.snapshot()
    ml = LearnedMPPI(sim, [sc, sc], CFG, 0, snap, Estimator(5))
    for k in range(3):
        a = ml(sim, k)
        sim.step(a)
    assert np.isfinite(sim.get_state()).all()
