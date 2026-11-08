"""Observation model (sim/observe.py): noise-free channels equal the truth; noise has the documented statistics."""

import numpy as np

from scenarios import sampler as S
from scenarios import track as T
from sim.observe import NOISE, OBS_NAMES, Observer
from sim.params import CONTROL_DT, VehicleParams
from sim.warp_sim import ScenarioSim

P = VehicleParams()
TRACKS = T.all_tracks()


def _sim(n=1):
    tr = TRACKS[2]
    sc = S.Scenario("o", 0, 15, "R100-apex", 100.0, "apex", "wet", 30.0, 0.0, 15.0, 0.95, 0.7, 2, tr.s_apex)
    sim = ScenarioSim([sc] * n, P, TRACKS)
    sim.reset_at_start()
    return sim, tr


def test_noise_free_observation_matches_truth():
    sim, tr = _sim()
    ob = Observer(sim, seeds=np.array([1]), noise_scale=0.0)
    for _ in range(5):
        sim.step(np.array([[0.0, 0.2]]))
    ob.reset()
    x_prev = sim.get_state()[0]
    sim.step(np.array([[0.1, 0.2]]))
    o = ob.observe().numpy()[0]
    x = sim.get_state()[0]
    fr = sim.get_frenet()[0]
    get = dict(zip(OBS_NAMES, o))
    assert np.isclose(get["r"], x[5], atol=1e-6) and np.isclose(get["ax"], x[12], atol=1e-5)
    assert np.isclose(get["ay"], (x[4] - x_prev[4]) / CONTROL_DT + x[5] * x[3], atol=1e-3)
    assert np.isclose(get["e_y"], fr[1], atol=1e-6)
    assert np.isclose(get["u_s_prev"], 0.1) and np.isclose(get["u_l_prev"], 0.2)
    assert np.isclose(get["speed"], np.hypot(x[3], x[4]), atol=1e-5)
    assert np.isclose(get["vy"], x[4], atol=1e-6)
    i = min(int(fr[0] / T.DS), len(tr.s) - 1)
    assert np.isclose(get["k0"], tr.kappa[i], atol=2e-4)  # projection hint vs floor(s/DS): same table cell


def test_noise_statistics_and_constant_bias():
    n = 2000
    sim, _ = _sim(n)
    sim.step(np.zeros((n, 2)))
    ob = Observer(sim, seeds=np.arange(n))
    ob.reset()
    clean = Observer(sim, seeds=np.arange(n), noise_scale=0.0)
    clean.reset()
    o1 = ob.observe().numpy().copy()  # CPU arrays: .numpy() is a view of the same buffer
    c1 = clean.observe().numpy().copy()
    o2 = ob.observe().numpy().copy()  # same state again: only the white noise differs
    c2 = clean.observe().numpy().copy()
    for k, (s, b) in NOISE.items():
        j = OBS_NAMES.index(k)
        if k == "ay":
            continue  # ay depends on the previous state, which observe() advances
        tot = (o1 - c1)[:, j]
        assert abs(tot.std() - np.hypot(s, b)) < 0.1 * np.hypot(s, b) + 1e-9, k
        white = ((o1 - c1) - (o2 - c2))[:, j] / np.sqrt(2)
        assert abs(white.std() - s) < 0.1 * s + 1e-9, k
