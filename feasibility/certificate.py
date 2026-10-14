"""Analytic infeasibility certificate on constant-radius arcs (FEASIBILITY.md §2; soundness argument there).

Question: a car arrives at a friction drop on an arc of radius R (front axle at the start of the transition) with
speed v_a, lateral offset |e| <= e0 and lateral velocity |de/dt| <= edot0. Can *any* controller keep every
footprint corner inside the lane until the low-friction window ends? If the function below returns
certified=True, the answer is provably no (for every trajectory of the simulator, sampled every 20 ms).

Point-mass relaxation (why it is sound, in brief):
  |F_axle| <= mu_axle * Fz_axle (tire bound + clip after relaxation), Fzf + Fzr = m g (ax clamp), rolling
  resistance <= f_rr m g, drag <= 0.5 rho CdA v^2  =>  |a_CoG| <= A(t) = mu_max(t) g + f_rr g + drag_max / m.
  In polar coordinates around the arc centre (r, theta), with h = r^2 theta_dot (angular momentum per mass):
      r_ddot = h^2 / r^3 + a_r,      dh/dt = r a_theta.
  While the CoG is inside the CoG corridor r in [R - w, R + w] (w = 1.75 - width/2: the largest lateral CoG
  deviation with no footprint point beyond the lane edge):
      |dh/dt| <= (R + w) A  =>  h(t) >= h_lb(t) = h0 - (R + w) int_0^t A,
      r_ddot >= h_lb(t)_+^2 / (R + w)^3 - A(t) =: g(t).
  Integrating twice gives rho(t) = r - R >= rho_lb(t). If rho_lb exceeds w at a time when the car is still
  provably on the arc and on the patch, the assumption "stays inside" is contradicted.
Every bound below is computed in closed form (piecewise polynomials) and evaluated on a time grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.params import G, PHYSICS_DT, RHO_AIR, VehicleParams

CHECK_PERIOD = 0.02  # lane departure is evaluated every control step
MARGIN = 1e-3  # [m] on top of w; also absorbs float rounding
GRID_DT = 0.002
T_MAX = 8.0
DRAG_HEADROOM = 8.0  # [m/s] assumed max speed gain inside the horizon for the drag bound (verified a posteriori)


@dataclass
class CertResult:
    certified: bool
    reason: str
    t_cross: float | None = None  # first time rho_lb > w (with the persistence conditions)
    t_window: float | None = None  # lower bound on the end of the usable window
    t_rear_max: float | None = None  # upper bound on when the rear axle is fully on the patch


def accel_bounds(p: VehicleParams, mu0: float, mu1: float, v_ub: float) -> tuple[float, float]:
    extra = p.f_rr * G + 0.5 * RHO_AIR * p.CdA * v_ub * v_ub / p.m
    return mu0 * G + extra, mu1 * G + extra


def _propagate(t, A_of_piece, T_switch, H0, rho0, rhod0, R, w):
    """Closed-form lower bounds at times t (sorted) for piecewise-constant A (A0 on [0, T_switch), A1 after).
    Returns rho_lb, rhod_lb, g, h_lb, int_A, int_h (all arrays)."""
    c3 = (R + w) ** 3
    out = {k: np.zeros_like(t) for k in ("rho", "rhod", "g", "h", "intA", "inth")}
    # piece boundaries: [0, T_switch), [T_switch, inf); each split where h_lb hits 0
    state = dict(t=0.0, rho=rho0, rhod=rhod0, H=H0, intA=0.0, inth=0.0)
    pieces = [(0.0, T_switch, A_of_piece[0]), (T_switch, np.inf, A_of_piece[1])]
    for (ta, tb, A) in pieces:
        k = (R + w) * A
        sub = []
        if state["H"] > 0:
            uz = state["H"] / k if k > 0 else np.inf
            sub.append((ta, min(tb, ta + uz), True))
            if ta + uz < tb:
                sub.append((ta + uz, tb, False))
        else:
            sub.append((ta, tb, False))
        for (sa, sb, hpos) in sub:
            if sa >= sb:
                continue
            H = state["H"] if hpos else 0.0
            m = (t >= sa) & (t < sb)
            u = t[m] - sa
            if hpos:
                gu = (H - k * u) ** 2 / c3 - A
                I1 = (H * H * u - H * k * u ** 2 + k * k * u ** 3 / 3) / c3 - A * u
                I2 = (H * H * u ** 2 / 2 - H * k * u ** 3 / 3 + k * k * u ** 4 / 12) / c3 - A * u ** 2 / 2
                hu = H - k * u
                Ih = H * u - k * u ** 2 / 2
            else:
                gu = -A * np.ones_like(u)
                I1 = -A * u
                I2 = -A * u ** 2 / 2
                hu = np.zeros_like(u)
                Ih = np.zeros_like(u)
            out["rho"][m] = state["rho"] + state["rhod"] * u + I2
            out["rhod"][m] = state["rhod"] + I1
            out["g"][m] = gu
            out["h"][m] = hu
            out["intA"][m] = state["intA"] + A * u
            out["inth"][m] = state["inth"] + Ih
            # advance the state to the end of the sub-piece
            if np.isfinite(sb):
                D = sb - sa
                if hpos:
                    I1D = (H * H * D - H * k * D ** 2 + k * k * D ** 3 / 3) / c3 - A * D
                    I2D = (H * H * D ** 2 / 2 - H * k * D ** 3 / 3 + k * k * D ** 4 / 12) / c3 - A * D ** 2 / 2
                    state["inth"] += H * D - k * D ** 2 / 2
                    state["H"] = max(H - k * D, 0.0)
                else:
                    I1D, I2D = -A * D, -A * D ** 2 / 2
                    state["H"] = 0.0
                state["rho"] += state["rhod"] * D + I2D
                state["rhod"] += I1D
                state["intA"] += A * D
    return out


def certify_arc(v_a: float, R: float, mu0: float, mu1: float, lt: float, Lp: float, prog_arc: float,
                p: VehicleParams = VehicleParams(), e0: float = 0.0, edot0: float = 0.0) -> CertResult:
    """Certify infeasibility of an arc friction drop from the arrival state (see module docstring).

    prog_arc: how far (arc length along the centreline) the CoG may advance from arrival before it can have
              left the constant-radius arc (callers compute it from the track; see `certify_scenario`).
    Lp: patch length at full mu1 (np.inf for an endless patch); lt: ramp length (0 = step).
    """
    w = 1.75 - p.width / 2
    if mu1 >= mu0:
        return CertResult(False, "no friction drop")
    if v_a <= edot0:
        return CertResult(False, "arrival speed too low")
    v_ub = v_a + DRAG_HEADROOM
    A0, A1 = accel_bounds(p, mu0, mu1, v_ub)
    t = np.arange(0.0, T_MAX, GRID_DT)

    # discretization conservatism (velocity impulses every PHYSICS_DT): one extra step of worst-case change
    H0 = (R - e0) * math.sqrt(v_a * v_a - edot0 * edot0) - (R + w) * A0 * PHYSICS_DT
    rho0, rhod0 = -e0, -edot0 - A0 * PHYSICS_DT

    # 1) upper bound on when the rear axle is past the entry ramp: needs CoG progress >= L + lt (+ margin),
    #    and progress >= R/(R+w)^2 * int h_lb  (theta_dot = h / r^2 >= h_lb / (R+w)^2). Use A0 throughout.
    P_rear = p.L + lt + 0.1
    b0 = _propagate(t, (A0, A0), np.inf, H0, rho0, rhod0, R, w)
    prog_lb0 = R / (R + w) ** 2 * b0["inth"]
    idx = np.nonzero(prog_lb0 >= P_rear)[0]
    if len(idx) == 0:
        return CertResult(False, "rear axle not provably on the patch within the horizon")
    T0 = t[idx[0]]

    # 2) lower bound on the end of the window: front axle reaches the exit ramp, or the CoG leaves the arc.
    #    progress <= R/(R-w)^2 * int h_ub, h_ub = (R+w)(v_a + int A).
    b = _propagate(t, (A0, A1), T0, H0, rho0, rhod0, R, w)
    int_intA = np.concatenate([[0.0], np.cumsum(0.5 * (b["intA"][1:] + b["intA"][:-1]) * GRID_DT)])
    int_intA = int_intA + b["intA"] * GRID_DT  # trapezoid of an increasing function + one-step slack: an upper bound
    prog_ub = R * (R + w) / (R - w) ** 2 * (v_a * t + int_intA)
    P_end = (lt + Lp - 0.1) if np.isfinite(Lp) else np.inf
    P_win = min(P_end, prog_arc - 0.1)
    over = np.nonzero(prog_ub >= P_win)[0]
    T_win = t[over[0] - 1] if len(over) else t[-1]
    if over.size and over[0] == 0:
        return CertResult(False, "window empty", t_window=0.0, t_rear_max=T0)

    # 3) find the first grid time with rho_lb > w + MARGIN that persists until the next lane check
    ok = (b["rho"] > w + MARGIN) & (b["rhod"] >= 0)
    n_chk = int(round(CHECK_PERIOD / GRID_DT))
    for j in np.nonzero(ok)[0]:
        tj = t[j]
        if tj + CHECK_PERIOD > T_win or j + n_chk >= len(t):
            break
        # g >= 0 on [tj, tj + CHECK_PERIOD]: inside a constant-A piece g decreases (h_lb decreases), so its minimum
        # is at the right end or just before the A-switch at T0
        g_end = b["g"][j + n_chk]
        g_pre = (b["h"][j] ** 2 / (R + w) ** 3 - A0) if tj < T0 <= tj + CHECK_PERIOD else np.inf
        if min(g_end, g_pre) < 0:
            continue
        # validity: speed stays above the kinematic-blend threshold, and below the drag headroom
        v_lb = v_a - b["intA"][j + n_chk]
        v_gain = b["intA"][j + n_chk]
        if v_lb < p.v_hi or v_gain > DRAG_HEADROOM:
            return CertResult(False, "validity (speed bounds) not met", t_window=T_win, t_rear_max=T0)
        return CertResult(True, "certified", t_cross=float(tj), t_window=float(T_win), t_rear_max=float(T0))
    return CertResult(False, "no provable departure inside the window", t_window=float(T_win), t_rear_max=float(T0))


def certify_scenario(sc, v_a: float, tracks, p: VehicleParams = VehicleParams(), e0: float = 0.0,
                     edot0: float = 0.0) -> CertResult:
    """Apply the certificate to a sampled scenario. Only drops where the CoG is already on the arc at arrival
    (apex, exit) are covered in v1; straights are never infeasible for a point mass; entry drops -> not covered."""
    if sc.R == 0.0:
        return CertResult(False, "straight: braking in a straight line is always feasible for the point mass")
    tr = tracks[sc.track_id]
    s_cog0_min = sc.s_drop - p.lf  # CoG is at least lf behind the front axle (|cos| <= 1)
    if s_cog0_min - 0.1 < tr.s_arc_start:
        return CertResult(False, "not covered: CoG not on the arc at arrival (entry drop)")
    prog_arc = tr.s_arc_end - (sc.s_drop - p.lf + 0.05)
    Lp = np.inf if sc.patch_len >= 1e8 else sc.patch_len
    return certify_arc(v_a, sc.R, sc.mu0, sc.mu1, sc.transition, Lp, prog_arc, p, e0, edot0)
