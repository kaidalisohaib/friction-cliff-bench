"""Tracks, Frenet projection, friction field, sampler, and the scenario-driven simulator."""

import json
from dataclasses import replace

import numpy as np
import pytest

from scenarios import sampler as S
from scenarios import track as T
from scenarios.friction import INF_PATCH, blend_weight, surface_at
from sim.params import DRY, ICE, VehicleParams
from sim.reference import IVX, IX, IY, ReferenceSim
from sim.warp_sim import ScenarioSim, WarpSim

P = VehicleParams()
TRACKS = T.all_tracks()


@pytest.mark.parametrize("tr", TRACKS, ids=lambda t: t.name)
def test_track_geometry(tr):
    assert np.all(np.diff(tr.s) > 0)
    assert np.allclose(np.hypot(np.diff(tr.x), np.diff(tr.y)), T.DS, atol=1e-6)  # arc-length parametrized
    assert np.max(np.abs(np.diff(tr.kappa))) < 1e-3  # continuous curvature (no jumps)
    if tr.R > 0:
        assert tr.theta[-1] == pytest.approx(np.pi / 2, abs=1e-6)
        k = (tr.s > tr.s_arc_start + 1) & (tr.s < tr.s_arc_end - 1)
        cx, cy = tr.x[k] - np.sin(tr.theta[k]) * tr.R, tr.y[k] + np.cos(tr.theta[k]) * tr.R
        assert np.ptp(cx) < 1e-6 and np.ptp(cy) < 1e-6  # the arc really is a circle of radius R


def test_projection_roundtrip():
    tr = TRACKS[2]
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = rng.integers(10, len(tr.s) - 10)
        e = rng.uniform(-3, 3)
        px = tr.x[i] - e * np.sin(tr.theta[i])
        py = tr.y[i] + e * np.cos(tr.theta[i])
        s, e_hat, _ = T.project(tr, px, py)
        # polyline (chord) projection error for an offset point ~ |e| * kappa * DS / 2 <= 4 mm here
        assert s == pytest.approx(tr.s[i], abs=5e-3) and e_hat == pytest.approx(e, abs=5e-3)


def test_blend_weight():
    s = np.array([69.9, 70.0, 71.0, 72.0, 81.9, 82.0, 83.0, 84.5, 200.0])
    np.testing.assert_allclose(blend_weight(s, 70.0, 2.0, 10.0), [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    np.testing.assert_allclose(blend_weight(s, 70.0, 0.0, 10.0), [0, 1, 1, 1, 0, 0, 0, 0, 0])  # patch ends at 80 m
    assert blend_weight(1e5, 70.0, 5.0, INF_PATCH) == 1.0
    v = surface_at(np.array([0.0, 100.0]), 70.0, 0.0, 10.0, 0.9, "ice", 0.12)
    assert v[0, 0] == pytest.approx(0.9) and v[1, 0] == pytest.approx(0.9)  # dry before and after


def test_sampler_matches_prereg_spec():
    assert S.N_CELLS == 1872 and S.TEST_SEED == 4019369836 and S.TUNING_SEED == 4210175585
    pool = S.test_pool()
    assert len(pool) == 4000 and len({s.sid for s in pool}) == 4000
    assert S.to_jsonl(pool) == S.to_jsonl(S.test_pool())  # deterministic
    for sc in pool[:500]:
        assert abs(sc.v0 - sc.v_stratum) <= 1.0 and 0.85 <= sc.mu0 <= 1.0
        lo, hi = {"wet": (0.5, 0.8), "snow": (0.2, 0.35), "ice": (0.08, 0.15)}[sc.surface]
        assert lo <= sc.mu1 <= hi
    header = json.loads(S.to_jsonl(pool).splitlines()[0])["header"]
    assert header["spec"] == "v1"
    tun = {(s.v0, s.mu0) for s in S.tuning_split()}
    assert not tun & {(s.v0, s.mu0) for s in pool}  # tuning never overlaps the test pool


def _straight_scenario(**kw):
    base = S.Scenario("x", 0, 20.0, "straight", 0.0, "straight", "ice", 30.0, 0.0, 20.0, 0.95, 0.1, 0, T.LEAD_IN)
    return replace(base, **kw)


def test_field_front_axle_meets_patch_first():
    """Coast along the straight at 20 m/s over an ice step: the front axle switches when its own s reaches the
    drop, the rear ~L/v = 0.135 s later (the 'free preview' studied in E4)."""
    sc = _straight_scenario()
    sim = ScenarioSim([sc], P, TRACKS)
    sim.reset_at_start()
    t_front = t_rear = None
    for k in range(250):
        sim.step([0.0, 0.0])
        mu_f, mu_r = sim.get_axle_mu()[0]
        if t_front is None and mu_f < 0.5:
            t_front = (k + 1) * 0.02
            x_front = sim.get_state()[0, IX] + P.lf
        if t_rear is None and mu_r < 0.5:
            t_rear = (k + 1) * 0.02
    assert x_front == pytest.approx(T.LEAD_IN, abs=0.5)  # within one control step of travel
    assert t_rear - t_front == pytest.approx(P.L / 20.0, abs=0.025)


def test_field_sim_equals_plain_sim_without_patch():
    """With mu1 = mu0 = dry the friction field is uniform, so ScenarioSim must equal WarpSim exactly."""
    sc = _straight_scenario(surface="ice", mu1=0.95, mu0=0.95, patch_len=0.0)
    sim_f = ScenarioSim([sc], P, TRACKS)
    sim_f.reset_at_start()
    sim_p = WarpSim(1, P)
    sim_p.reset(ReferenceSim.steady_state(sc.v0, P))
    sim_p.set_surfaces(DRY.with_mu(0.95), DRY.with_mu(0.95))
    for k in range(100):
        a = [0.1 * np.sin(k * 0.1), 0.2]
        sim_f.step(a)
        sim_p.step(a)
    np.testing.assert_array_equal(sim_f.get_state(), sim_p.get_state())


def test_frenet_tracks_the_car_on_a_curve():
    """Simple path follower on the R=100 track on dry at 10 m/s: Frenet s advances like the speed and the
    lateral error stays small (checks projection, hints and the sign convention of e)."""
    sc = S.make_scenario(S.GEOMETRIES.index(("R100-apex", 100.0, "apex")) * 0, np.random.default_rng(0), "c")
    sc = replace(sc, geometry="R100-apex", R=100.0, track_id=2, v0=10.0, mu0=0.95, mu1=0.95, surface="wet",
                 s_drop=TRACKS[2].s_apex)
    sim = ScenarioSim([sc], P, TRACKS)
    sim.reset_at_start()
    tr = TRACKS[2]
    max_e = 0.0
    for k in range(2000):  # 40 s, ~400 m: through the whole curve
        s_, e = sim.get_frenet()[0]
        x = sim.get_state()[0]
        i = min(int(s_ / T.DS), len(tr.s) - 1)
        e_psi = np.angle(np.exp(1j * (x[2] - tr.theta[i])))
        delta = P.L * tr.kappa[i] - 0.15 * e - 0.8 * e_psi
        sim.step([np.clip(delta / P.delta_max, -1, 1), 0.3 * (10.0 - x[IVX])])
        if k > 10:
            max_e = max(max_e, abs(e))
    s_end = sim.get_frenet()[0, 0]
    assert s_end == pytest.approx(400.0, rel=0.03) and max_e < 0.3
