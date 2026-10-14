"""Feasibility certificate: soundness checks (FEASIBILITY.md §2, gate S1 preview) and the G1 classic cell."""

import numpy as np
import pytest

from feasibility.certificate import _propagate, accel_bounds, certify_arc, certify_scenario
from feasibility.pointmass import survives
from scenarios import sampler as S
from scenarios import track as T
from sim.params import G, PHYSICS_DT, VehicleParams
from sim.warp_sim import ScenarioSim

P = VehicleParams()
W = 1.75 - P.width / 2
TRACKS = T.all_tracks()


def test_g1_classic_cell_is_certified_infeasible():
    """15 m/s arriving on an R = 50 m arc onto endless ice (mu 0.1): needs 4.5 m/s^2, has ~1.1."""
    r = certify_arc(15.0, 50.0, 0.95, 0.1, 0.0, np.inf, 1e9)
    assert r.certified and r.t_cross < r.t_window


@pytest.mark.parametrize("R", [50.0, 100.0, 200.0, 400.0])
@pytest.mark.parametrize("mu1", [0.08, 0.15, 0.3, 0.6])
def test_never_certifies_when_steady_circling_is_possible(R, mu1):
    """If v^2 / (R - w) <= mu1 g the point mass can simply circle on the patch: must not be certified."""
    v = np.sqrt(mu1 * G * (R - W))
    for Lp in (10.0, 100.0, np.inf):
        assert not certify_arc(v, R, 0.95, mu1, 0.0, Lp, 1e9).certified


def test_certification_is_monotone_in_speed():
    for R, mu1 in [(50.0, 0.1), (100.0, 0.3), (200.0, 0.1)]:
        flags = [certify_arc(v, R, 0.95, mu1, 0.0, np.inf, 1e9).certified for v in np.arange(5, 40, 0.5)]
        first = flags.index(True) if True in flags else len(flags)
        assert all(flags[first:]), (R, mu1)


def _random_pointmass(v_a, R, mu0, mu1, lt, e0, edot0, rng, t_end, dt=PHYSICS_DT):
    """Point mass with random accelerations at the full budget (A0 until the rear axle could be on the patch by
    actual progress, A1 after), semi-implicit Euler like the simulator. Returns (t, rho) until it leaves."""
    A0, A1 = accel_bounds(P, mu0, mu1, v_a + 8.0)
    r0 = R - e0
    vt = np.sqrt(v_a ** 2 - edot0 ** 2)
    pos = np.array([r0, 0.0])
    vel = np.array([-edot0, vt])
    prog, t = 0.0, 0.0
    ts, rhos = [], []
    phi = rng.uniform(0, 2 * np.pi)
    while t < t_end:
        A = A0 if prog < P.L + lt else A1
        if rng.random() < 0.02:
            phi = rng.uniform(0, 2 * np.pi)
        r = np.linalg.norm(pos)
        er = pos / r
        et = np.array([-er[1], er[0]])
        mag = A * rng.choice([1.0, rng.random()])
        vel = vel + dt * mag * (np.cos(phi) * er + np.sin(phi) * et)
        pos = pos + dt * vel
        prog += dt * (vel @ et) * R / r
        t += dt
        rho = np.linalg.norm(pos) - R
        ts.append(t)
        rhos.append(rho)
        if abs(rho) > W:
            break
    return np.array(ts), np.array(rhos)


@pytest.mark.parametrize("case", [(15.0, 50.0, 0.1, 0.0, 0.0, 0.0), (20.0, 100.0, 0.15, 2.0, 0.3, 0.2),
                                  (25.0, 200.0, 0.3, 5.0, 0.5, 0.5), (12.0, 50.0, 0.6, 0.0, 0.2, 0.0)])
def test_lower_bound_holds_for_random_pointmass_trajectories(case):
    """Monte Carlo check of the core inequality: rho(t) >= rho_lb(t) for any admissible point-mass trajectory,
    as long as it is inside the corridor (the derivation assumes that). Uses the certificate's own bound with
    the rear-entry time bound it computes."""
    v_a, R, mu1, lt, e0, edot0 = case
    mu0 = 0.95
    rng = np.random.default_rng(0)
    A0, A1 = accel_bounds(P, mu0, mu1, v_a + 8.0)
    res = certify_arc(v_a, R, mu0, mu1, lt, np.inf, 1e9, e0=e0, edot0=edot0)
    T0 = res.t_rear_max
    t = np.arange(0.0, 6.0, 0.001)
    H0 = (R - e0) * np.sqrt(v_a ** 2 - edot0 ** 2) - (R + W) * A0 * PHYSICS_DT
    b = _propagate(t, (A0, A1), T0, H0, -e0, -edot0 - A0 * PHYSICS_DT, R, W)
    for _ in range(40):
        ts, rho = _random_pointmass(v_a, R, mu0, mu1, lt, e0, edot0, rng, t_end=6.0)
        inside = np.abs(rho) <= W
        lb = np.interp(ts, t, b["rho"])
        assert np.all(rho[inside] >= lb[inside] - 1e-6)


@pytest.mark.parametrize("R", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("mu1", [0.1, 0.3])
@pytest.mark.parametrize("Lp", [30.0, np.inf])
def test_pointmass_search_never_survives_a_certified_case(R, mu1, Lp):
    for v in np.arange(8.0, 34.0, 2.0):
        if certify_arc(v, R, 0.95, mu1, 0.0, Lp, 1e9).certified:
            assert not survives(v, R, 0.95, mu1, 0.0, Lp), (v, R, mu1, Lp)


def _footprint_out(sim_state, frenet_e, track, s):
    """Max distance of any footprint corner beyond the lane edge (approx: corners projected via Frenet)."""
    x = sim_state
    c, sn = np.cos(x[2]), np.sin(x[2])
    worst = -np.inf
    for lon in (P.foot_front, -P.foot_rear):
        for lat in (P.width / 2, -P.width / 2):
            px = x[0] + lon * c - lat * sn
            py = x[1] + lon * sn + lat * c
            _, e, _ = T.project(track, px, py)
            worst = max(worst, abs(e) - 1.75)
    return worst


@pytest.mark.parametrize("track_id,v0", [(1, 17.0), (2, 19.0)], ids=["R50_17ms", "R100_19ms"])
def test_full_car_cannot_stay_in_lane_when_certified(track_id, v0):
    """S1 preview on the real simulator: apex drop onto endless ice (mu 0.1) at a certified arrival speed,
    several strong heuristic controllers (track the centreline; additionally brake at 0.5 / 1.0 on arrival).
    Each must be inside the lane when it reaches the patch (so the test is not passed vacuously) and none may
    keep the footprint inside afterwards."""
    tr = TRACKS[track_id]
    sc = S.Scenario("c", 0, v0, "apex", tr.R, "apex", "ice", 1e9, 0.0, v0, 0.95, 0.1, track_id, tr.s_apex)
    assert certify_scenario(sc, v0, TRACKS).certified
    brakes = [0.0, -0.5, -1.0]
    sims = ScenarioSim([sc] * len(brakes), P, TRACKS)
    sims.reset_at_start()
    departed = np.zeros(len(brakes), bool)
    in_lane_at_arrival = np.zeros(len(brakes), bool)
    for k in range(int(14.0 / 0.02)):
        fr = sims.get_frenet()
        X = sims.get_state()
        acts = []
        for j, br in enumerate(brakes):
            s_, e = fr[j]
            i = min(int(s_ / T.DS), len(tr.s) - 1)
            e_psi = np.angle(np.exp(1j * (X[j, 2] - tr.theta[i])))
            v_ref = v0
            delta = P.L * tr.kappa[i] - 0.15 * e - 0.8 * e_psi - 0.1 * (X[j, 5] - X[j, 3] * tr.kappa[i])
            u_long = br if s_ > sc.s_drop - P.lf else 0.3 * (v_ref - X[j, 3])
            acts.append([np.clip(delta / P.delta_max, -1, 1), u_long])
        sims.step(np.array(acts))
        fr = sims.get_frenet()
        X = sims.get_state()
        for j in range(len(brakes)):
            out = _footprint_out(X[j], fr[j, 1], tr, fr[j, 0])
            if not in_lane_at_arrival[j] and fr[j, 0] + P.lf >= sc.s_drop:
                in_lane_at_arrival[j] = out <= 0
                assert in_lane_at_arrival[j], "controller already left the lane before the patch"
            if fr[j, 0] + P.lf > sc.s_drop and out > 0:
                departed[j] = True
    assert in_lane_at_arrival.all() and departed.all()
