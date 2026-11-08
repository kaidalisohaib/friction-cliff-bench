"""Batched reference MPPI in Warp (BASELINES.md §3). E cars (scenario runs) in parallel, K samples each.

One MPPI iteration per 20 ms control step:
  1. sample      V[e,k,h] = clip(U[e,h] + sigma * n_h, -1, 1), n an AR(1) sequence along the horizon (rho = 0:
                 white noise); sample k = 0 is noise-free (V = U)
  2. roll out    each (e, k) simulates the car H control steps ahead with the *simulator's own* substep function
                 (20 x 1 ms), starting from the true current state, delay buffer and track hints, and sums the
                 shared objective (DESIGN.md §4, prereg V1) along the way
  3. weight      w_k = exp(-(S_k - min_k S_k) / lambda) / sum
  4. update      U[e,h] = sum_k w_k V[e,k,h]   (the information-theoretic MPPI update on the sampled controls)
                 (steps 1-4 repeat n_iter times per control step, each sampling around the improved plan)
  5. act         apply U[e,0]; warm start = shift U by one step (repeat the last entry)

What the rollout model knows about friction is the *information level* of the controller:
  MODE_FIXED : per-car (front, rear) surfaces held constant over the horizon, supplied by the caller each step.
               O-R gets the true current per-axle surfaces; M-N gets the nominal dry surface; M-B/M-L (P3) get their
               selected/estimated surfaces.
  MODE_FIELD : rollouts see the true friction field along the track (projected per axle per substep), i.e. the
               clairvoyant oracle O-C.
Delay: the rollout replays the car's real action buffer, so the actions already "in the pipe" are applied
exactly; new samples take effect after the model's delay (model params per car, so mismatch is possible).

Footprint corners in the cost use a small-angle Frenet approximation (corner offset = e + lon sin(e_psi) +
lat cos(e_psi) - kappa lon^2 / 2). That is the *planner's* model of the lane; evaluation uses exact
projections (`control/episode.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from sim.params import CONTROL_DT, N_SUBSTEPS, VehicleParams
from sim.warp_sim import Params, ScenarioSim, field_surface, make_params, project_local, substep, vec8, vec13

MODE_FIXED, MODE_FIELD = 0, 1
_HALF_LANE = wp.constant(wp.float32(1.75))


@dataclass(frozen=True)
class CostWeights:
    """Shared objective, prereg V1 (never tuned)."""
    w_lane: float = 100.0
    w_ey: float = 1.0
    w_v: float = 2.0
    w_du: float = 0.1
    w_prog: float = 0.1
    C_fail: float = 2000.0
    severe: float = 1.0  # termination: footprint beyond lane edge + 1.0 m
    beta_term: float = np.deg2rad(30.0)  # termination: |beta| > 30 deg
    spin_term: float = np.pi / 2  # termination: |psi - psi_path| > 90 deg


@wp.struct
class CostW:
    w_lane: wp.float32
    w_ey: wp.float32
    w_v: wp.float32
    w_du: wp.float32
    w_prog: wp.float32
    C_fail: wp.float32
    severe: wp.float32
    beta_term: wp.float32
    spin_term: wp.float32


def make_cost(c: CostWeights) -> CostW:
    s = CostW()  # every field set explicitly (wp.struct defaults are ignored)
    for k in ("w_lane", "w_ey", "w_v", "w_du", "w_prog", "C_fail", "severe", "beta_term", "spin_term"):
        setattr(s, k, float(getattr(c, k)))
    return s


@dataclass(frozen=True)
class MPPIConfig:
    K: int = 1024  # samples per car
    H: int = 100  # horizon [control steps]
    sigma_steer: float = 0.3  # noise std on normalized steering
    sigma_long: float = 0.4  # noise std on normalized throttle/brake
    lam: float = 10.0  # temperature
    rho: float = 0.0  # AR(1) noise correlation per control step (0 = white noise)
    n_iter: int = 1  # MPPI iterations per control step (each re-samples around the improved plan)
    ess_target: float = 0.0  # > 0: choose lambda per step so the effective sample size is ~ this (lam ignored)
    pd_sample: bool = False  # sample k = 1 is the closed-loop path follower (ancillary controller sample)
    mode: int = MODE_FIXED


@wp.func
def wrap(a: wp.float32) -> wp.float32:
    return wp.atan2(wp.sin(a), wp.cos(a))


@wp.func
def step_cost(x: vec13, s: wp.float32, e: wp.float32, idx: int, tid: int, s_prev: wp.float32,
              th: wp.array2d(dtype=wp.float32), kap: wp.array2d(dtype=wp.float32), p: Params, v_ref: wp.float32,
              du2: wp.float32, cw: CostW) -> wp.vec2:
    """(running cost, terminated flag) for one control step."""
    epsi = wrap(x[2] - th[tid, idx])
    k = kap[tid, idx]
    sp = wp.sin(epsi)
    cp = wp.cos(epsi)
    worst = wp.float32(0.0)
    for a in range(2):
        lon = p.foot_front
        if a == 1:
            lon = -p.foot_rear
        for b in range(2):
            lat = wp.float32(0.5) * p.width
            if b == 1:
                lat = -lat
            ec = e + lon * sp + lat * cp - wp.float32(0.5) * k * lon * lon
            worst = wp.max(worst, wp.abs(ec))
    viol = wp.max(worst - _HALF_LANE, wp.float32(0.0))
    dv = (x[3] - v_ref) / v_ref
    c = cw.w_lane * viol * viol + cw.w_ey * e * e + cw.w_v * dv * dv + cw.w_du * du2 - cw.w_prog * (s - s_prev)
    term = wp.float32(0.0)
    speed = wp.sqrt(x[3] * x[3] + x[4] * x[4])
    beta = wp.float32(0.0)
    if speed > wp.float32(1.0):
        beta = wp.atan2(x[4], wp.abs(x[3]))
    if viol > cw.severe or wp.abs(beta) > cw.beta_term or wp.abs(epsi) > cw.spin_term:
        c = c + cw.C_fail
        term = wp.float32(1.0)
    return wp.vec2(c, term)


@wp.func
def pd_steer(x: vec13, q: wp.vec3, tid: int, th: wp.array2d(dtype=wp.float32), kap: wp.array2d(dtype=wp.float32),
             p: Params) -> wp.float32:
    """Same law as control/episode.py prefix_action: curvature feedforward + PD on e, e_psi + yaw-rate damping."""
    idx = int(q[2])
    k = kap[tid, idx]
    epsi = wrap(x[2] - th[tid, idx])
    delta = (p.lf + p.lr) * k - wp.float32(0.15) * q[1] - wp.float32(0.8) * epsi - wp.float32(0.1) * (x[5] - x[3] * k)
    return wp.clamp(delta / p.delta_max, wp.float32(-1.0), wp.float32(1.0))


@wp.func
def pd_long(x: vec13, q: wp.vec3, tid: int, kap: wp.array2d(dtype=wp.float32), npts: int, ds: wp.float32,
            v_ref: wp.float32, mu: wp.float32, mode: int, pr: vec8, base: wp.array(dtype=vec8)) -> wp.float32:
    """P on speed towards a target speed.
    MODE_FIXED: min(v_ref, sqrt(0.7 mu g / |kappa 0.5 s ahead|)), mu = the friction the planner assumes.
    MODE_FIELD (preview, O-C only): min over points s' up to 80 m ahead of sqrt(v_allow(s')^2 + 2 a_b (s' - s)),
    v_allow = sqrt(0.7 mu(s') g / |kappa(s')|), a_b = 0.5 mu_now g: brake in time for low-friction curves ahead."""
    g = wp.float32(9.81)
    i0 = int(q[2])
    ia = wp.clamp(i0 + int(wp.max(x[3], wp.float32(1.0)) * wp.float32(0.5) / ds), 0, npts - 1)
    ka = wp.max(wp.abs(kap[tid, ia]), wp.float32(1.0e-9))
    vt = wp.min(v_ref, wp.sqrt(wp.float32(0.7) * mu * g / ka))
    if mode == 1:
        ab = wp.float32(0.5) * mu * g
        for j in range(40):
            dsj = wp.float32(j) * wp.float32(2.0)
            ij = wp.clamp(i0 + int(dsj / ds), 0, npts - 1)
            mu_j = field_surface(q[0] + dsj, pr, base)[0]
            kj = wp.max(wp.abs(kap[tid, ij]), wp.float32(1.0e-9))
            va = wp.sqrt(wp.float32(0.7) * mu_j * g / kj)
            vt = wp.min(vt, wp.sqrt(va * va + wp.float32(2.0) * ab * dsj))
    return wp.clamp(wp.float32(0.3) * (vt - x[3]), wp.float32(-1.0), wp.float32(1.0))


@wp.kernel
def sample_kernel(U: wp.array3d(dtype=wp.float32), V: wp.array4d(dtype=wp.float32), sig: wp.vec2, rho: wp.float32,
                  seeds: wp.array(dtype=int), t: int):
    """AR(1) noise along the horizon: n_h = rho n_{h-1} + sqrt(1 - rho^2) xi_h, xi ~ N(0, 1) (stationary
    variance 1, correlation time ~ dt / (1 - rho)). rho = 0 is classic white-noise MPPI."""
    e, k = wp.tid()
    H = U.shape[1]
    # per-car seed (not the batch index), so a run's noise does not depend on which batch it is in
    st = wp.rand_init(seeds[e] + t * 7919, k)
    q = wp.sqrt(wp.float32(1.0) - rho * rho)
    n0 = wp.randn(st)
    n1 = wp.randn(st)
    for h in range(H):
        if h > 0:
            n0 = rho * n0 + q * wp.randn(st)
            n1 = rho * n1 + q * wp.randn(st)
        a0 = sig[0] * n0
        a1 = sig[1] * n1
        if k == 0:
            a0 = wp.float32(0.0)
            a1 = wp.float32(0.0)
        V[e, k, h, 0] = wp.clamp(U[e, h, 0] + a0, wp.float32(-1.0), wp.float32(1.0))
        V[e, k, h, 1] = wp.clamp(U[e, h, 1] + a1, wp.float32(-1.0), wp.float32(1.0))


@wp.kernel
def rollout_kernel(mparams: wp.array(dtype=Params), state: wp.array(dtype=vec13), buf: wp.array3d(dtype=wp.float32),
                   hints: wp.array2d(dtype=int), frenet: wp.array2d(dtype=wp.float32), mode: int,
                   sf_fix: wp.array(dtype=vec8), sr_fix: wp.array(dtype=vec8), prof: wp.array(dtype=vec8),
                   base: wp.array(dtype=vec8), track_id: wp.array(dtype=int), track_x: wp.array2d(dtype=wp.float32),
                   track_y: wp.array2d(dtype=wp.float32), track_th: wp.array2d(dtype=wp.float32),
                   track_k: wp.array2d(dtype=wp.float32), track_n: wp.array(dtype=int), ds: wp.float32,
                   v_ref: wp.array(dtype=wp.float32), V: wp.array4d(dtype=wp.float32), cw: CostW, nsub: int,
                   dt: wp.float32, pd_sample: int, costs: wp.array2d(dtype=wp.float32), xend: wp.array2d(dtype=vec13)):
    e, k = wp.tid()
    H = V.shape[2]
    p = mparams[e]
    x = state[e]
    tid = track_id[e]
    npts = track_n[tid]
    pr = prof[e]
    hc = hints[e, 0]
    hf = hints[e, 1]
    hr = hints[e, 2]
    s_prev = frenet[e, 0]
    sf = sf_fix[e]
    sr = sr_fix[e]
    sf[4] = sf[4] * p.B_scale_f
    sr[4] = sr[4] * p.B_scale_r
    d = p.delay_steps
    u_prev_s = buf[e, 0, 0]
    u_prev_l = buf[e, 0, 1]
    total = wp.float32(0.0)
    is_pd = pd_sample == 1 and k == 1
    q = wp.vec3(s_prev, frenet[e, 1], wp.float32(hc))
    for h in range(H):
        if is_pd:
            # ancillary sample: the closed-loop path follower, acting on this rollout's own state; its actions are
            # written into V so the MPPI average can move towards them (BASELINES §3, D20)
            V[e, k, h, 0] = pd_steer(x, q, tid, track_th, track_k, p)
            V[e, k, h, 1] = pd_long(x, q, tid, track_k, npts, ds, v_ref[e], sf[0], mode, pr, base)
        us = wp.float32(0.0)
        ul = wp.float32(0.0)
        if h >= d:
            us = V[e, k, h - d, 0]
            ul = V[e, k, h - d, 1]
        else:
            us = buf[e, d - 1 - h, 0]
            ul = buf[e, d - 1 - h, 1]
        for _j in range(nsub):
            if mode == 1:
                c = wp.cos(x[2])
                s_ = wp.sin(x[2])
                qf = project_local(track_x, track_y, tid, npts, ds, x[0] + p.lf * c, x[1] + p.lf * s_, hf)
                qr = project_local(track_x, track_y, tid, npts, ds, x[0] - p.lr * c, x[1] - p.lr * s_, hr)
                hf = int(qf[2])
                hr = int(qr[2])
                sf = field_surface(qf[0], pr, base)
                sr = field_surface(qr[0], pr, base)
                sf[4] = sf[4] * p.B_scale_f
                sr[4] = sr[4] * p.B_scale_r
            x = substep(x, us, ul, p, sf, sr, dt)
        q = project_local(track_x, track_y, tid, npts, ds, x[0], x[1], hc)
        hc = int(q[2])
        vs = V[e, k, h, 0]
        vl = V[e, k, h, 1]
        du2 = (vs - u_prev_s) * (vs - u_prev_s) + (vl - u_prev_l) * (vl - u_prev_l)
        u_prev_s = vs
        u_prev_l = vl
        ct = step_cost(x, q[0], q[1], hc, tid, s_prev, track_th, track_k, p, v_ref[e], du2, cw)
        s_prev = q[0]
        total = total + ct[0]
        if ct[1] > wp.float32(0.0):
            break
    costs[e, k] = total
    xend[e, k] = x


@wp.func
def ess_at(costs: wp.array2d(dtype=wp.float32), e: int, cmin: wp.float32, lam: wp.float32) -> wp.float32:
    K = costs.shape[1]
    z = wp.float32(0.0)
    z2 = wp.float32(0.0)
    for k in range(K):
        wk = wp.exp(-(costs[e, k] - cmin) / lam)
        z = z + wk
        z2 = z2 + wk * wk
    return z * z / z2


@wp.kernel
def weights_kernel(costs: wp.array2d(dtype=wp.float32), lam: wp.float32, ess_target: wp.float32,
                   w: wp.array2d(dtype=wp.float32), info: wp.array2d(dtype=wp.float32)):
    """w = softmin(costs / lambda). If ess_target > 0, lambda is chosen per car and step (bisection in log space)
    so that the effective sample size 1/sum(w^2) is close to ess_target: the update then always averages over a
    controlled number of good samples, whatever the cost scale."""
    e = wp.tid()
    K = costs.shape[1]
    cmin = costs[e, 0]
    for k in range(K):
        cmin = wp.min(cmin, costs[e, k])
    lam_e = lam
    if ess_target > wp.float32(0.0):
        lo = wp.float32(-6.0)  # log10 lambda
        hi = wp.float32(5.0)
        for _it in range(24):
            mid = wp.float32(0.5) * (lo + hi)
            if ess_at(costs, e, cmin, wp.pow(wp.float32(10.0), mid)) < ess_target:
                lo = mid
            else:
                hi = mid
        lam_e = wp.pow(wp.float32(10.0), wp.float32(0.5) * (lo + hi))
    z = wp.float32(0.0)
    for k in range(K):
        wk = wp.exp(-(costs[e, k] - cmin) / lam_e)
        w[e, k] = wk
        z = z + wk
    s2 = wp.float32(0.0)
    for k in range(K):
        wk = w[e, k] / z
        w[e, k] = wk
        s2 = s2 + wk * wk
    info[e, 0] = cmin
    info[e, 1] = wp.float32(1.0) / s2  # effective sample size
    info[e, 2] = lam_e


@wp.kernel
def update_kernel(w: wp.array2d(dtype=wp.float32), V: wp.array4d(dtype=wp.float32), Unew: wp.array3d(dtype=wp.float32)):
    """Unew[e, h] = sum_k w[e, k] V[e, k, h]."""
    e, h = wp.tid()
    K = V.shape[1]
    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    for k in range(K):
        a0 = a0 + w[e, k] * V[e, k, h, 0]
        a1 = a1 + w[e, k] * V[e, k, h, 1]
    Unew[e, h, 0] = a0
    Unew[e, h, 1] = a1


@wp.kernel
def shift_kernel(Unew: wp.array3d(dtype=wp.float32), U: wp.array3d(dtype=wp.float32), act: wp.array2d(dtype=wp.float32)):
    """act = Unew[:, 0]; warm start U[h] = Unew[h + 1] (last entry repeated)."""
    e, h = wp.tid()
    H = U.shape[1]
    j = wp.min(h + 1, H - 1)
    U[e, h, 0] = Unew[e, j, 0]
    U[e, h, 1] = Unew[e, j, 1]
    if h == 0:
        act[e, 0] = Unew[e, 0, 0]
        act[e, 1] = Unew[e, 0, 1]


class MPPI:
    """MPPI for every car of a ScenarioSim (same device, same order). The rollouts read the sim's current state,
    action buffer and track hints directly, so call `act()` right before `sim.step()`."""

    def __init__(self, sim: ScenarioSim, cfg: MPPIConfig, seed: int | np.ndarray = 0,
                 model_params: VehicleParams | list[VehicleParams] | None = None,
                 cost: CostWeights = CostWeights(), v_ref: np.ndarray | None = None):
        self.sim, self.cfg = sim, cfg
        n, d = sim.n, sim.device
        seeds = np.broadcast_to(np.asarray(seed, np.int64), (n,)) % (2 ** 30)
        self.seeds = wp.array(seeds.astype(np.int32), dtype=int, device=d)
        mp = model_params if model_params is not None else VehicleParams()
        plist = mp if isinstance(mp, list) else [mp] * n
        self.mparams = wp.array([make_params(p) for p in plist], dtype=Params, device=d)
        self.cw = make_cost(cost)
        vr = np.array([sc.v0 for sc in sim.scenarios], np.float32) if v_ref is None else np.asarray(v_ref, np.float32)
        self.v_ref = wp.array(vr, dtype=wp.float32, device=d)
        self.U = wp.zeros((n, cfg.H, 2), dtype=wp.float32, device=d)
        self.V = wp.zeros((n, cfg.K, cfg.H, 2), dtype=wp.float32, device=d)
        self.costs = wp.zeros((n, cfg.K), dtype=wp.float32, device=d)
        self.xend = wp.zeros((n, cfg.K), dtype=vec13, device=d)  # rollout end states (tests)
        self.w = wp.zeros((n, cfg.K), dtype=wp.float32, device=d)
        self.info = wp.zeros((n, 3), dtype=wp.float32, device=d)
        self.action = wp.zeros((n, 2), dtype=wp.float32, device=d)
        self.Unew = wp.zeros((n, cfg.H, 2), dtype=wp.float32, device=d)
        self.sf_fix = wp.zeros(n, dtype=vec8, device=d)
        self.sr_fix = wp.zeros(n, dtype=vec8, device=d)
        self.t = 0

    def reset(self, u_init: np.ndarray | None = None) -> None:
        """Initial plan: u_init (n, 2) repeated over the horizon (e.g. the last applied action), else zeros."""
        u = np.zeros((self.sim.n, 2), np.float32) if u_init is None else np.asarray(u_init, np.float32)
        self.U.assign(wp.array(np.repeat(u[:, None, :], self.cfg.H, axis=1), dtype=wp.float32, device=self.sim.device))
        self.t = 0

    def set_surfaces(self, sf: np.ndarray, sr: np.ndarray) -> None:
        """MODE_FIXED: (n, 8) unscaled surfaces the rollouts assume for the front and rear axle."""
        d = self.sim.device
        self.sf_fix.assign(wp.array(np.ascontiguousarray(sf, np.float32), dtype=vec8, device=d))
        self.sr_fix.assign(wp.array(np.ascontiguousarray(sr, np.float32), dtype=vec8, device=d))

    def act(self) -> wp.array:
        """Run one MPPI iteration; returns the (n, 2) action array on the device (also kept in self.action)."""
        s, c, d = self.sim, self.cfg, self.sim.device
        n = s.n
        for it in range(c.n_iter):
            wp.launch(sample_kernel, dim=(n, c.K), inputs=[self.U, self.V, wp.vec2(c.sigma_steer, c.sigma_long),
                                                           c.rho, self.seeds, self.t * 16 + it], device=d)
            wp.launch(rollout_kernel, dim=(n, c.K),
                      inputs=[self.mparams, s.state, s.buf, s.hints, s.frenet, c.mode, self.sf_fix, self.sr_fix,
                              s.prof, s.base, s.track_id, s.track_x, s.track_y, s.track_theta, s.track_kappa,
                              s.track_n, s.ds, self.v_ref, self.V, self.cw, N_SUBSTEPS, CONTROL_DT / N_SUBSTEPS,
                              int(c.pd_sample), self.costs, self.xend], device=d)
            wp.launch(weights_kernel, dim=n, inputs=[self.costs, c.lam, c.ess_target, self.w, self.info], device=d)
            wp.launch(update_kernel, dim=(n, c.H), inputs=[self.w, self.V, self.Unew], device=d)
            if it < c.n_iter - 1:
                wp.copy(self.U, self.Unew)
        wp.launch(shift_kernel, dim=(n, c.H), inputs=[self.Unew, self.U, self.action], device=d)
        self.t += 1
        return self.action
