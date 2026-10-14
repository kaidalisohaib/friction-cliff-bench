"""Simulator validation suite V1, V2, V5, V6, V7 (DESIGN.md §5), run on the Warp simulator.

Each test compares against a hand calculation stated in its docstring.
"""

import numpy as np
import pytest

from sim.params import DRY, G, ICE, SNOW, WET, VehicleParams
from sim.steady import solve_turn, steady_turn_state
from sim.reference import IDELTA, IPSI, IR, IVX, IVY, IX, IY, NX, ReferenceSim
from sim.warp_sim import WarpSim, surface_vec

P = VehicleParams()
DT = 0.02


def cornering_stiffness(p: VehicleParams, s=DRY):
    L = p.L
    Fzf, Fzr = p.m * G * p.lr / L, p.m * G * p.lf / L
    return s.B * p.B_scale_f * s.C * s.mu * Fzf, s.B * p.B_scale_r * s.C * s.mu * Fzr


# --- V1 ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b_scale_r", [1.0, 1.25], ids=["neutral", "understeer"])
def test_v1_linear_regime_steady_yaw_rate(b_scale_r):
    """r_ss = v*delta / (L + K_us v^2), K_us = (m/L)(lr/Cf - lf/Cr), within 2% at small delta.
    Speed is held by a P controller (small drive force, negligible combined-slip effect)."""
    p = P.with_(B_scale_r=b_scale_r)
    Cf, Cr = cornering_stiffness(p)
    K_us = p.m / p.L * (p.lr / Cf - p.lf / Cr)
    speeds = np.array([10.0, 15.0, 20.0, 25.0])
    delta = 0.01
    n = len(speeds)
    sim = WarpSim(n, p)
    x0 = np.stack([ReferenceSim.steady_state(v, p) for v in speeds])
    sim.reset(x0)
    sim.set_surfaces(DRY, DRY)
    rs = []
    for k in range(400):  # 8 s
        x = sim.get_state()
        u_long = np.clip(0.5 * (speeds - x[:, IVX]), -1, 1)
        sim.step(np.stack([np.full(n, delta / p.delta_max), u_long], 1))
        if k >= 350:
            rs.append(sim.get_state()[:, IR])
    x = sim.get_state()
    r_meas = np.mean(rs, 0)
    d = x[:, IDELTA]
    r_pred = x[:, IVX] * d / (p.L + K_us * x[:, IVX] ** 2)
    np.testing.assert_allclose(r_meas, r_pred, rtol=0.02)
    if b_scale_r > 1.0:
        assert K_us > 0 and np.all(r_meas < x[:, IVX] * d / p.L)  # understeer: less yaw than kinematic


# --- V2 ------------------------------------------------------------------------------------------------
def circle_controller(x, R, v_target, integ, u_ff=None):
    """Track a left circle of radius R centred at (0, R): steer/throttle feedforward (kinematic L/R by
    default, or the solved equilibrium u_ff) + PD on errors + I."""
    dx, dy = x[:, IX], x[:, IY] - R
    dist = np.hypot(dx, dy)
    e_out = dist - R  # > 0: outside the circle
    theta = np.arctan2(dy, dx)
    psi_ref = theta + np.pi / 2
    course = x[:, IPSI] + np.arctan2(x[:, IVY], x[:, IVX])  # velocity direction, not heading (ice: |beta| ~ 10 deg)
    e_psi = np.angle(np.exp(1j * (course - psi_ref)))
    integ += DT * e_out
    e_r = x[:, IR] - x[:, IVX] / R
    d_ff, a_ff = (P.L / R, 0.0) if u_ff is None else (u_ff[0] * P.delta_max, u_ff[1])
    delta = d_ff + 0.05 * e_out + 0.01 * integ - 0.5 * e_psi - 0.15 * e_r
    u_long = np.clip(a_ff + 0.3 * (v_target - x[:, IVX]), -1, 1)
    return np.stack([np.clip(delta / P.delta_max, -1, 1), u_long], 1), e_out


def max_steady_ratio(surface, R, p):
    """Largest v / sqrt(mu g R) with a steady-turn equilibrium (bisection on sim.steady.solve_turn)."""
    v_lim = np.sqrt(surface.mu * G * R)
    lo, hi = 0.3, 1.2
    assert solve_turn(lo * v_lim, R, p, surface).ok
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if solve_turn(mid * v_lim, R, p, surface).ok else (lo, mid)
    return lo


@pytest.mark.parametrize("surface,R", [(DRY, 50.0), (WET, 80.0), (SNOW, 200.0), (ICE, 100.0)],
                         ids=["dry", "wet", "snow", "ice"])
def test_v2a_no_steady_turn_above_point_mass_limit(surface, R):
    """Soundness property used by the feasibility certificate: no steady turn exists above sqrt(mu g R)
    (with or without resistances), because |F_axle| <= mu Fz and Fzf + Fzr = m g."""
    for p in (P, P.with_(f_rr=0.0, CdA=0.0)):
        assert max_steady_ratio(surface, R, p) <= 1.0


@pytest.mark.parametrize("surface,R", [(DRY, 50.0), (WET, 80.0)], ids=["dry", "wet"])
def test_v2b_limit_within_3pct_on_dry_and_wet(surface, R):
    """V2: max steady speed within 3% of sqrt(mu g R). Below it only because holding speed needs drive
    force against 'cornering drag' (Fy * sin(alpha)), which uses part of the rear friction budget."""
    assert max_steady_ratio(surface, R, P) >= 0.97


def test_v2c_low_mu_limit_is_reported(capsys):
    """Not a gate yet: with the DESIGN 'typical' coefficients (B=4-5, C=2) the *lateral* peak sits at a
    17-21 deg slip angle, so cornering drag is large and the RWD limit falls well below sqrt(mu g R) on
    snow/ice. The surface shapes are re-sourced at the G1 parameter freeze (D13); re-gate then."""
    rows = [(s.name, max_steady_ratio(s, R, P)) for s, R in ((SNOW, 200.0), (ICE, 100.0))]
    with capsys.disabled():
        print("\nV2c max steady v/sqrt(mu g R):", ", ".join(f"{n} {r:.3f}" for n, r in rows))
    assert all(0.5 < r <= 1.0 for _, r in rows)


@pytest.mark.parametrize("ratio,held", [(0.95, True), (1.03, False)])
def test_v2d_closed_loop_on_dry_circle(ratio, held):
    """The simulator itself (not just the algebraic equilibrium) holds a dry R = 50 m circle at 0.95 of
    the limit for 20 s, and cannot at 1.03. Start from the solved equilibrium (pre-filled delay buffer)."""
    R = 50.0
    v = ratio * np.sqrt(DRY.mu * G * R)
    x0, u0, _ = steady_turn_state(min(v, 0.95 * np.sqrt(DRY.mu * G * R)), R, P, DRY)
    x0[IVX] = v
    sim = WarpSim(1, P)
    sim.reset(x0, u0=u0)
    sim.set_surfaces(DRY, DRY)
    integ = np.zeros(1)
    max_err = 0.0
    for k in range(1000):
        u, e = circle_controller(sim.get_state(), R, np.array([v]), integ, u_ff=u0)
        sim.step(u)
        max_err = max(max_err, abs(e[0]))
    x = sim.get_state()[0]
    ok = max_err < 1.0 and abs(x[IVX] - v) < 0.05 * v
    assert ok == held, (max_err, x[IVX] / v)


# --- V5 ------------------------------------------------------------------------------------------------
def steady_turn(v=15.0, R=100.0, steps=600):
    """Settle into a steady dry left turn with the circle controller; return the state and the action."""
    sim = WarpSim(1, P)
    sim.reset(ReferenceSim.steady_state(v, P, 1 / R), u0=[P.L / R / P.delta_max, 0.0])
    sim.set_surfaces(DRY, DRY)
    integ = np.zeros(1)
    for _ in range(steps):
        u, _ = circle_controller(sim.get_state(), R, np.array([v]), integ)
        sim.step(u)
    return sim, u


@pytest.mark.parametrize("which", ["rear", "front"])
def test_v5_mu_step_transients(which):
    """Steady dry turn (15 m/s, R = 100 m, a_y = 2.25 m/s^2), steering frozen, then one axle -> ice.
    Hand calc of the initial yaw acceleration. Dry axle forces: Fyf = m a_y lr/L = 1933 N,
    Fyr = m a_y lf/L = 1667 N; loads Fz(f,r) = 8429, 7267 N; dry slip angles alpha = Fy/(B C mu Fz) ~ 0.012 rad.
    At the *same* slip angle the iced axle is on the linear part of the ice curve (D11: shape changes too):
      F_ice = (B C mu Fz)_ice * alpha = 4*2*0.1*Fz*0.012 ~ 70-80 N  (far below the 0.1*Fz ice peak)
      rear ice:  (lf*1933 - lr*70)/Iz  = +0.86 rad/s^2 -> oversteer, beta opposite in sign to r
      front ice: (lf*81 - lr*1667)/Iz  = -0.86 rad/s^2 -> understeer, yaw rate collapses"""
    sim, u = steady_turn()
    x_before = sim.get_state()[0]
    r0 = x_before[IR]
    ice = surface_vec(ICE)
    dry = surface_vec(DRY)
    sim.set_surfaces(dry if which == "rear" else ice, ice if which == "rear" else dry)
    sim.step(u)
    sim.step(u)
    r_2 = sim.get_state()[0, IR]
    yaw_acc = (r_2 - r0) / (2 * DT)
    expected = 0.86 if which == "rear" else -0.86
    assert yaw_acc == pytest.approx(expected, rel=0.15)
    for _ in range(48):  # 1 s total
        sim.step(u)
    x = sim.get_state()[0]
    beta = np.arctan2(x[IVY], x[IVX])
    if which == "rear":
        assert x[IR] > r0 and np.sign(beta) == -np.sign(x[IR]) and abs(beta) > np.radians(5)
    else:
        assert abs(x[IR]) < 0.5 * abs(r0) and abs(beta) < np.radians(3)


# --- V6 ------------------------------------------------------------------------------------------------
def abuse_run(n_env, n_steps, device, seed=0):
    """Random actions incl. full-lock steering, wheel lock, reverse, deep slides on random per-axle
    surfaces, random model switches and delays. Returns the number of non-finite states seen."""
    rng = np.random.default_rng(seed)
    plist = [P.with_(sigma_relax=float(rng.choice([0.0, 0.3, 0.6])), combined_alpha=float(rng.choice([0, 0.5, 1])),
                     delay_steps=int(rng.integers(0, 9)), drive_front_frac=float(rng.choice([0.0, 0.5, 1.0])))
             for _ in range(n_env)]
    sim = WarpSim(n_env, plist, device=device)
    x0 = np.zeros((n_env, NX))
    x0[:, IVX] = rng.uniform(-5, 30, n_env)
    x0[:, IVY] = rng.uniform(-3, 3, n_env)
    x0[:, IR] = rng.uniform(-1, 1, n_env)
    x0[:, 6] = x0[:, IVX] / P.R_w * rng.uniform(0, 2, n_env)
    x0[:, 7] = x0[:, IVX] / P.R_w * rng.uniform(0, 2, n_env)
    sim.reset(x0)
    surfs = np.stack([surface_vec(s) for s in (DRY, ICE, SNOW)])
    bad = 0
    for k in range(n_steps):
        if k % 25 == 0:
            sim.set_surfaces(surfs[rng.integers(0, 3, n_env)], surfs[rng.integers(0, 3, n_env)])
        a = rng.uniform(-1, 1, (n_env, 2))
        a[:, 0] = np.where(rng.random(n_env) < 0.3, np.sign(a[:, 0]), a[:, 0])  # full lock
        a[:, 1] = np.where(rng.random(n_env) < 0.2, -1.0, a[:, 1])  # full brake -> wheel lock
        sim.step(a)
        if k % 50 == 49 or k == n_steps - 1:
            bad += int((~np.isfinite(sim.get_state())).any(axis=1).sum())
    return bad


def test_v6_robustness_cpu_short():
    assert abuse_run(n_env=2048, n_steps=300, device="cpu") == 0  # 6e5 env-steps


@pytest.mark.gpu
@pytest.mark.slow
def test_v6_robustness_gpu_1e8():
    """Full V6: 1e8 random-action env-steps (65,536 envs x 1,526 steps)."""
    assert abuse_run(n_env=65536, n_steps=1526, device="cuda:0", seed=1) == 0


# --- V7 ------------------------------------------------------------------------------------------------
def scripted_run(device, n=64, steps=100, seed=3):
    rng = np.random.default_rng(seed)
    sim = WarpSim(n, P, device=device)
    x0 = np.stack([ReferenceSim.steady_state(v, P) for v in rng.uniform(10, 25, n)])
    sim.reset(x0)
    sim.set_surfaces(DRY, DRY)
    t = np.arange(steps) * DT
    amp = rng.uniform(0.02, 0.2, n)
    traj = []
    for k in range(steps):
        a = np.stack([amp * np.sin(2 * np.pi * 0.5 * t[k]), np.full(n, 0.1)], 1)
        sim.step(a)
        traj.append(sim.get_state())
    return np.array(traj)


@pytest.mark.gpu
def test_v7_gpu_replay_is_bit_identical():
    a = scripted_run("cuda:0")
    b = scripted_run("cuda:0")
    assert np.array_equal(a, b)


@pytest.mark.gpu
def test_v7_cpu_gpu_parity():
    """CPU vs GPU within 1e-4 relative over 2 s (GPU uses fused multiply-add, so not bit-identical)."""
    c = scripted_run("cpu")
    g = scripted_run("cuda:0")
    for i in (IX, IY, IVX, IVY, IR):
        scale = np.maximum(np.abs(c[..., i]).max(), 1.0)
        assert np.abs(c[..., i] - g[..., i]).max() / scale < 1e-4, i
