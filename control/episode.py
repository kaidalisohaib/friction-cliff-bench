"""Episodes: prefix driving, controller handover, exact metrics, per-run records (EVAL_PROTOCOL.md §1).

Episode protocol (DESIGN.md §3.2; handover per D19 / prereg A4.1):
  1. Prefix. Every scenario starts at s = 0 in straight-line steady state at v0 (sampler header). A fixed,
     deterministic path follower (`prefix_action`) drives it on dry road until the front axle is T_HANDOVER
     seconds (at the current speed) before the friction drop, or immediately if that point is already passed at
     s = 0. It slows for curves with a curvature speed profile. Its end state ("handover state") is identical for
     every method, so all methods are compared from the same paired state, and nothing informative is lost:
     before handover the road is dry and nobody knows (or, with T_HANDOVER >= the oracle horizon, can see)
     the drop.
  2. Controller. The method under test drives from the handover state.
  3. End: patch exit + 3 s (rear axle past the end ramp), or drop + 8 s for an endless patch, or termination
     (footprint > lane edge + 1 m, |beta| > 30 deg, spin), or a cap of drop + 20 s.
Metrics are computed from *exact* footprint-corner projections (`metrics_kernel`), every control step.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import warp as wp

from scenarios import track as T
from scenarios.sampler import A_LAT_FRAC
from sim.params import CONTROL_DT, G, VehicleParams
from sim.warp_sim import Params, ScenarioSim, make_params, project_local, vec13

T_HANDOVER = 4.0  # [s] before the drop (>= the oracle horizon, D19)
T_AFTER_EXIT = 3.0
T_AFTER_DROP_INF = 8.0
T_CAP_AFTER_DROP = 20.0
T_CAP_PREFIX = 120.0
BETA_LOC = np.deg2rad(20.0)  # loss of control (success criterion)
_WIN_PTS = 8


@wp.kernel
def metrics_kernel(params: wp.array(dtype=Params), state: wp.array(dtype=vec13), hints: wp.array2d(dtype=int),
                   track_id: wp.array(dtype=int), track_x: wp.array2d(dtype=wp.float32),
                   track_y: wp.array2d(dtype=wp.float32), track_th: wp.array2d(dtype=wp.float32),
                   track_n: wp.array(dtype=int), ds: wp.float32, out: wp.array2d(dtype=wp.float32)):
    """Per car: s, e, e_psi, max |corner e| (exact projection of the 4 footprint corners), beta, e_dot, speed."""
    i = wp.tid()
    p = params[i]
    x = state[i]
    tid = track_id[i]
    npts = track_n[tid]
    hc = hints[i, 0]
    q = project_local(track_x, track_y, tid, npts, ds, x[0], x[1], hc)
    c = wp.cos(x[2])
    sn = wp.sin(x[2])
    worst = wp.float32(0.0)
    for a in range(2):
        lon = p.foot_front
        if a == 1:
            lon = -p.foot_rear
        h0 = int(q[2]) + int(lon / ds)
        for b in range(2):
            lat = wp.float32(0.5) * p.width
            if b == 1:
                lat = -lat
            px = x[0] + lon * c - lat * sn
            py = x[1] + lon * sn + lat * c
            qc = project_local(track_x, track_y, tid, npts, ds, px, py, wp.clamp(h0, 0, npts - 1))
            worst = wp.max(worst, wp.abs(qc[1]))
    th = track_th[tid, int(q[2])]
    epsi = wp.atan2(wp.sin(x[2] - th), wp.cos(x[2] - th))
    speed = wp.sqrt(x[3] * x[3] + x[4] * x[4])
    beta = wp.float32(0.0)
    if speed > wp.float32(1.0):
        beta = wp.atan2(x[4], wp.abs(x[3]))
    out[i, 0] = q[0]
    out[i, 1] = q[1]
    out[i, 2] = epsi
    out[i, 3] = worst
    out[i, 4] = beta
    out[i, 5] = x[3] * wp.sin(epsi) + x[4] * wp.cos(epsi)  # lateral velocity relative to the path
    out[i, 6] = speed


def speed_profile(track: T.Track, v0: float, mu0: float, a_lat_frac=A_LAT_FRAC, a_brake=0.4 * G) -> np.ndarray:
    """Max speed along the track: v <= sqrt(a_lat / |kappa|) with a_lat = 0.7 mu0 g, braking at <= a_brake
    (backward pass), capped at v0."""
    k = np.abs(track.kappa)
    v = np.minimum(v0, np.sqrt(a_lat_frac * mu0 * G / np.maximum(k, 1e-9)))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brake * T.DS))
    return v


_PREFIX_CACHE: dict = {}


def _prefix_tables(scs, tracks, profiles):
    key = (id(profiles), len(profiles), id(scs))
    if key not in _PREFIX_CACHE:
        if len(_PREFIX_CACHE) > 64:
            _PREFIX_CACHE.clear()
        npt = np.array([len(t.s) for t in tracks])
        th = np.zeros((len(tracks), npt.max()))
        ka = np.zeros((len(tracks), npt.max()))
        for k, t in enumerate(tracks):
            th[k, : npt[k]], ka[k, : npt[k]] = t.theta, t.kappa
        tid = np.array([sc.track_id for sc in scs])
        lp = np.array([len(pr) for pr in profiles])
        pr = np.zeros((len(profiles), lp.max()))
        for j, v in enumerate(profiles):
            pr[j, : len(v)] = v
        _PREFIX_CACHE[key] = (tid, npt[tid], th, ka, pr, lp, profiles, scs)
    return _PREFIX_CACHE[key][:6]


def prefix_action(fr, X, scs, tracks, profiles, p: VehicleParams):
    """Deterministic path follower (curvature feedforward + PD on e, e_psi, yaw-rate damping; P on speed).
    Vectorized over cars; identical to `prefix_action_loop` (tested)."""
    tid, n_tr, th, ka, pr, lp = _prefix_tables(scs, tracks, profiles)
    s_, e = fr[:, 0], fr[:, 1]
    i = np.minimum(np.trunc(s_ / T.DS).astype(np.int64), n_tr - 1)
    i_w = np.where(i < 0, i + n_tr, i)  # python negative indexing, as in the loop version
    thi, kai = th[tid, i_w], ka[tid, i_w]
    e_psi = np.angle(np.exp(1j * (X[:, 2] - thi)))
    delta = p.L * kai - 0.15 * e - 0.8 * e_psi - 0.1 * (X[:, 5] - X[:, 3] * kai)
    i_ahead = np.minimum(i + np.trunc(np.maximum(X[:, 3], 1.0) * 0.5 / T.DS).astype(np.int64), n_tr - 1)
    i_ahead = np.where(i_ahead < 0, i_ahead + lp, i_ahead)
    v_t = pr[np.arange(len(tid)), i_ahead]
    return np.stack([np.clip(delta / p.delta_max, -1, 1), np.clip(0.3 * (v_t - X[:, 3]), -1, 1)], 1)


def prefix_action_loop(fr, X, scs, tracks, profiles, p: VehicleParams):
    """Reference (scalar) implementation of `prefix_action`."""
    acts = np.zeros((len(scs), 2))
    for j, sc in enumerate(scs):
        tr = tracks[sc.track_id]
        s_, e = fr[j]
        i = min(int(s_ / T.DS), len(tr.s) - 1)
        e_psi = np.angle(np.exp(1j * (X[j, 2] - tr.theta[i])))
        delta = p.L * tr.kappa[i] - 0.15 * e - 0.8 * e_psi - 0.1 * (X[j, 5] - X[j, 3] * tr.kappa[i])
        i_ahead = min(i + int(max(X[j, 3], 1.0) * 0.5 / T.DS), len(tr.s) - 1)  # look 0.5 s ahead
        v_t = profiles[j][i_ahead]
        acts[j] = [np.clip(delta / p.delta_max, -1, 1), np.clip(0.3 * (v_t - X[j, 3]), -1, 1)]
    return acts


def run_prefix(scs, tracks, p: VehicleParams = VehicleParams(), device="cpu", a_lat_frac: float = A_LAT_FRAC):
    """Drive every scenario to its handover state. Returns (snapshot dict with one row per scenario, info)."""
    sim = ScenarioSim(scs, p, tracks, device)
    sim.reset_at_start()
    profiles = [speed_profile(tracks[sc.track_id], sc.v0, sc.mu0, a_lat_frac) for sc in scs]
    n = len(scs)
    done = np.zeros(n, bool)
    snap = None
    t_prefix = np.zeros(n)
    s_drop = np.array([sc.s_drop for sc in scs])
    # the state at s = 0 counts as a handover candidate (the sim must be stepped once to fill hints/surfaces,
    # so the zero-length case hands over after one step of the follower)
    for k in range(int(T_CAP_PREFIX / CONTROL_DT)):
        sim.step(prefix_action(sim.get_frenet(), sim.get_state(), scs, tracks, profiles, p))
        fr, X = sim.get_frenet(), sim.get_state()
        ready = (~done) & ((s_drop - (fr[:, 0] + p.lf)) <= T_HANDOVER * np.maximum(X[:, 3], 0.1))
        if ready.any():
            cur = sim.snapshot()
            if snap is None:
                snap = {kk: np.zeros_like(vv) for kk, vv in cur.items()}
            for kk in snap:
                snap[kk][ready] = cur[kk][ready]
            t_prefix[ready] = (k + 1) * CONTROL_DT
            done |= ready
        if done.all():
            break
    if not done.all():
        raise RuntimeError(f"prefix did not reach handover for {np.count_nonzero(~done)} scenarios")
    return snap, {"t_prefix": t_prefix}


@dataclass
class RunRecord:
    sid: str
    method: str
    seed: int
    success: bool
    departed: bool  # any footprint corner beyond the lane edge
    loss_of_control: bool  # |beta| > 20 deg or spin
    terminated: str  # "" | "severe" | "beta" | "spin" | "cap"
    min_clearance: float  # min over the episode of (1.75 - max corner |e|) [m]; < 0 = departed
    max_abs_ey: float
    time_out_of_lane: float
    t_drop: float | None  # time (from handover) when the front axle reached the drop
    arrival_v: float | None
    arrival_e: float | None
    arrival_edot: float | None
    duration: float
    progress: float
    extra: dict = field(default_factory=dict)


def select_rows(snap: dict, idx) -> dict:
    return {k: v[idx] for k, v in snap.items()}


def run_episodes(scs, snap, controller, method: str, seed: int, tracks, p: VehicleParams = VehicleParams(),
                 device="cpu", sim: ScenarioSim | None = None, keep_trace=False):
    """Run one controller on a batch of scenarios from their handover states.

    controller(sim, k) -> (n, 2) actions (numpy or wp.array); it may read anything its information level allows.
    Returns a list of RunRecord in scenario order (and the traces if keep_trace).
    """
    n = len(scs)
    if sim is None:
        sim = ScenarioSim(scs, p, tracks, device)
    sim.load(snap)
    mbuf = wp.zeros((n, 7), dtype=wp.float32, device=device)
    s_drop = np.array([sc.s_drop for sc in scs])
    s_exit = np.array([sc.s_drop + sc.transition + sc.patch_len for sc in scs])
    inf_patch = np.array([sc.patch_len >= 1e8 for sc in scs])
    t_drop = np.full(n, np.nan)
    arr = np.full((n, 3), np.nan)
    t_exit = np.full(n, np.nan)
    active = np.ones(n, bool)
    term = np.array([""] * n, dtype=object)
    min_clear = np.full(n, np.inf)
    max_ey = np.zeros(n)
    t_out = np.zeros(n)
    loc = np.zeros(n, bool)
    duration = np.zeros(n)
    s_start = snap["frenet"][:, 0].astype(float)
    s_last = s_start.copy()
    traces = [] if keep_trace else None
    k = 0
    while active.any():
        acts = controller(sim, k)
        sim.step(acts)
        k += 1
        t = k * CONTROL_DT
        wp.launch(metrics_kernel, dim=n, inputs=[sim.params, sim.state, sim.hints, sim.track_id, sim.track_x,
                                                 sim.track_y, sim.track_theta, sim.track_n, sim.ds, mbuf],
                  device=device)
        m = mbuf.numpy().astype(float)
        if keep_trace:
            traces.append(np.concatenate([m, sim.get_state()[:, [3, 4, 5]]], axis=1))
        a = active
        s_, e, epsi, worst, beta, edot = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5]
        clear = 1.75 - worst
        min_clear[a] = np.minimum(min_clear[a], clear[a])
        max_ey[a] = np.maximum(max_ey[a], np.abs(e[a]))
        t_out[a] += CONTROL_DT * (clear[a] < 0)
        loc[a] |= (np.abs(beta[a]) > BETA_LOC) | (np.abs(epsi[a]) > np.pi / 2)
        s_last[a] = s_[a]
        duration[a] = t
        arrived = a & np.isnan(t_drop) & (s_ + p.lf >= s_drop)
        t_drop[arrived] = t
        arr[arrived] = np.stack([m[arrived, 6], np.abs(e[arrived]), np.abs(edot[arrived])], axis=1)
        exited = a & np.isnan(t_exit) & ~inf_patch & (s_ - p.lr >= s_exit)
        t_exit[exited] = t
        # terminations
        sev = a & (worst - 1.75 > 1.0)
        bet = a & ~sev & (np.abs(beta) > np.deg2rad(30.0))
        spn = a & ~sev & ~bet & (np.abs(epsi) > np.pi / 2)
        term[sev], term[bet], term[spn] = "severe", "beta", "spin"
        end = sev | bet | spn
        end |= a & ~np.isnan(t_exit) & (t >= t_exit + T_AFTER_EXIT)
        end |= a & inf_patch & ~np.isnan(t_drop) & (t >= t_drop + T_AFTER_DROP_INF)
        cap = a & ~end & ((~np.isnan(t_drop) & (t >= t_drop + T_CAP_AFTER_DROP)) | (t >= T_CAP_PREFIX))
        term[cap] = "cap"
        active = a & ~(end | cap)
    recs = []
    for j, sc in enumerate(scs):
        departed = bool(min_clear[j] < 0)
        recs.append(RunRecord(
            sid=sc.sid, method=method, seed=seed, success=bool(not departed and not loc[j] and term[j] in ("", "cap")
                                                                and not np.isnan(t_drop[j])),
            departed=departed, loss_of_control=bool(loc[j]), terminated=str(term[j]),
            min_clearance=float(min_clear[j]), max_abs_ey=float(max_ey[j]), time_out_of_lane=float(t_out[j]),
            t_drop=None if np.isnan(t_drop[j]) else float(t_drop[j]),
            arrival_v=None if np.isnan(arr[j, 0]) else float(arr[j, 0]),
            arrival_e=None if np.isnan(arr[j, 1]) else float(arr[j, 1]),
            arrival_edot=None if np.isnan(arr[j, 2]) else float(arr[j, 2]),
            duration=float(duration[j]), progress=float(s_last[j] - s_start[j])))
    return (recs, np.stack(traces, axis=1) if keep_trace else None) if keep_trace else recs


def record_dict(r: RunRecord) -> dict:
    return asdict(r)
