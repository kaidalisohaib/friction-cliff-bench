"""Scalar NumPy reference of the vehicle dynamics (DESIGN.md §2.2), float64.

Slow and readable on purpose: it is the "answer key" the Warp simulator is checked against (V8, D12).
Written from DESIGN.md, not from the Warp code. One function = one physics substep; `control_step`
runs the 20 substeps of a 20 ms control period.

State layout (shared with the Warp simulator), index -> meaning:
  0 X, 1 Y [m] world position of the CoG       2 psi [rad] yaw
  3 vx, 4 vy [m/s] body-frame velocity            5 r [rad/s] yaw rate
  6 wf, 7 wr [rad/s] lumped wheel speeds         8 Fyf_lag, 9 Fyr_lag [N] relaxed lateral forces
  10 delta [rad] road-wheel angle                11 a_act [-] lagged longitudinal command in [-1, 1]
  12 ax_prev [m/s^2] last inertial longitudinal acceleration (for load transfer)
Action: (u_steer, u_long) in [-1, 1]^2; u_long > 0 drives, < 0 brakes.
"""

from __future__ import annotations

import math

import numpy as np

from dataclasses import replace

from sim.params import BUF_LEN, CONTROL_DT, G, N_SUBSTEPS, RHO_AIR, Surface, VehicleParams
from sim.tire import ALPHA_MAX, ONE_PLUS_KAPPA_MIN, S_EPS

NX = 13
(IX, IY, IPSI, IVX, IVY, IR, IWF, IWR, IFYF, IFYR, IDELTA, IAACT, IAX) = range(NX)

FD_DOMEGA = 0.05  # finite-difference step for dFx/domega [rad/s]


# --- scalar tire (same equations as sim/tire.py, scalar math for speed) -----------------------------
def _mf(x: float, B: float, C: float, D: float, E: float) -> float:
    bx = B * x
    return D * math.sin(C * math.atan(bx - E * (bx - math.atan(bx))))


def tire(kappa: float, alpha: float, Fz: float, s: Surface, combined_alpha: float) -> tuple[float, float]:
    a = min(max(alpha, -ALPHA_MAX), ALPHA_MAX)
    opk = max(1.0 + kappa, ONE_PLUS_KAPPA_MIN)
    sx, sy = kappa / opk, math.tan(a) / opk
    D = s.mu * Fz
    sn = math.hypot(sx, sy)
    if sn > S_EPS:
        fx = _mf(sn, s.Bx, s.Cx, D, s.Ex) * sx / sn
        fy = _mf(sn, s.By, s.Cy, D, s.Ey) * sy / sn
    else:
        fx, fy = 0.0, 0.0
    if combined_alpha != 0.0:
        fx_e = _mf(sx, s.Bx, s.Cx, D, s.Ex)
        ratio = fx_e / D if D > 0 else 0.0
        fy_e = _mf(math.tan(a), s.By, s.Cy, D, s.Ey) * math.sqrt(min(max(1.0 - ratio * ratio, 0.0), 1.0))
        w = combined_alpha
        fx, fy = (1 - w) * fx + w * fx_e, (1 - w) * fy + w * fy_e
    return fx, fy


def scaled_surfaces(sf: Surface, sr: Surface, p: VehicleParams) -> tuple[Surface, Surface]:
    """Apply the per-axle cornering-stiffness scales (lateral B only)."""
    return replace(sf, By=sf.By * p.B_scale_f), replace(sr, By=sr.By * p.B_scale_r)


def slip_ratio(R: float, omega: float, vx_wheel: float, eps_v: float) -> float:
    return (R * omega - vx_wheel) / max(abs(R * omega), abs(vx_wheel), eps_v)


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def clip_circle(fx: float, fy: float, fmax: float) -> tuple[float, float]:
    n = math.hypot(fx, fy)
    if n > fmax:
        k = fmax / n
        return fx * k, fy * k
    return fx, fy


def _wheel_update(omega, T_drive, T_brake, kappa_fn, Fx, Fz, s, alpha, p: VehicleParams, dt):
    """Linearly implicit Euler for J*domega = T_drive - R*Fx(omega) - b*omega, then a brake that cannot
    reverse the wheel. Implicit because R^2*dFx/d(R*omega)/J makes the plain update unstable at low speed."""
    Fx_h, _ = tire(kappa_fn(omega + FD_DOMEGA), alpha, Fz, s, p.combined_alpha)
    dFx = (Fx_h - Fx) / FD_DOMEGA
    k = max((p.R_w * dFx + p.b_w) / p.Jw, 0.0)
    denom = 1.0 + dt * k
    om = omega + dt * (T_drive - p.R_w * Fx - p.b_w * omega) / p.Jw / denom
    db = dt * T_brake / p.Jw / denom
    if abs(om) <= db:
        return 0.0
    return om - math.copysign(db, om)


def substep(x: np.ndarray, u_steer: float, u_long: float, p: VehicleParams, sf: Surface, sr: Surface,
            dt: float) -> np.ndarray:
    X, Y, psi, vx, vy, r, wf, wr, fyf_l, fyr_l, delta, a_act, ax_prev = (float(v) for v in x)
    L = p.lf + p.lr

    # actuators: steering lag + rate limit + saturation; longitudinal first-order lag
    d_target = min(max(u_steer, -1.0), 1.0) * p.delta_max
    d_dot = min(max((d_target - delta) / p.tau_delta, -p.delta_rate), p.delta_rate)
    delta = min(max(delta + dt * d_dot, -p.delta_max), p.delta_max)
    a_act = a_act + dt / p.tau_T * (min(max(u_long, -1.0), 1.0) - a_act)

    T_drive = max(a_act, 0.0) * p.drive_force_max * p.R_w
    T_brake = max(-a_act, 0.0) * p.brake_force_max * p.R_w
    Tdf, Tdr = T_drive * p.drive_front_frac, T_drive * (1.0 - p.drive_front_frac)
    Tbf, Tbr = T_brake * p.brake_front_frac, T_brake * (1.0 - p.brake_front_frac)

    # quasi-static longitudinal load transfer; clamping ax (not Fz) keeps Fzf + Fzr = m*g exactly
    ax = min(max(ax_prev, -0.95 * G * p.lf / p.h), 0.95 * G * p.lr / p.h)
    Fzf = p.m * (G * p.lr - ax * p.h) / L
    Fzr = p.m * (G * p.lf + ax * p.h) / L

    # hub velocities in each wheel frame (front is steered)
    c, s = math.cos(delta), math.sin(delta)
    vyf_body = vy + p.lf * r
    vxf, vyf = vx * c + vyf_body * s, -vx * s + vyf_body * c
    vxr, vyr = vx, vy - p.lr * r
    alpha_f = -math.atan2(vyf, abs(vxf))  # = delta - atan2(vy + lf r, |vx|) for vx > 0
    alpha_r = -math.atan2(vyr, abs(vxr))

    kf = lambda om: slip_ratio(p.R_w, om, vxf, p.eps_v)  # noqa: E731
    kr = lambda om: slip_ratio(p.R_w, om, vxr, p.eps_v)  # noqa: E731
    Fxf, Fyf = tire(kf(wf), alpha_f, Fzf, sf, p.combined_alpha)
    Fxr, Fyr = tire(kr(wr), alpha_r, Fzr, sr, p.combined_alpha)

    wf = _wheel_update(wf, Tdf, Tbf, kf, Fxf, Fzf, sf, alpha_f, p, dt)
    wr = _wheel_update(wr, Tdr, Tbr, kr, Fxr, Fzr, sr, alpha_r, p, dt)

    # tire relaxation (H2 axis 2); sigma = 0 means instantaneous lateral force
    if p.sigma_relax > 0.0:
        fyf_l += min(dt * abs(vxf) / p.sigma_relax, 1.0) * (Fyf - fyf_l)
        fyr_l += min(dt * abs(vxr) / p.sigma_relax, 1.0) * (Fyr - fyr_l)
    else:
        fyf_l, fyr_l = Fyf, Fyr
    # applied force never exceeds the friction circle, even with a lagged Fy (needed by the certificate)
    Fxf_a, Fyf_a = clip_circle(Fxf, fyf_l, sf.mu * Fzf)
    Fxr_a, Fyr_a = clip_circle(Fxr, fyr_l, sr.mu * Fzr)

    # planar rigid body (body frame)
    Fx_fb, Fy_fb = Fxf_a * c - Fyf_a * s, Fxf_a * s + Fyf_a * c
    F_rr = -p.f_rr * (Fzf + Fzr) * math.tanh(vx / 0.1)
    F_aero = -0.5 * RHO_AIR * p.CdA * vx * abs(vx)
    SFx = Fx_fb + Fxr_a + F_rr + F_aero
    SFy = Fy_fb + Fyr_a
    Mz = p.lf * Fy_fb - p.lr * Fyr_a

    dvx = SFx / p.m + r * vy
    dvy_dyn = SFy / p.m - r * vx
    dr_dyn = Mz / p.Iz
    # low-speed blend to a kinematic bicycle on the *speed* |v| (not |vx|: a car sliding sideways is not slow);
    # the weight is exactly 1 above v_hi, which the feasibility certificate relies on
    wgt = smoothstep((math.hypot(vx, vy) - p.v_lo) / (p.v_hi - p.v_lo))
    r_kin = vx * math.tan(delta) / L
    vy_kin = p.lr * r_kin
    dvy = wgt * dvy_dyn + (1.0 - wgt) * (vy_kin - vy) / p.tau_kin
    dr = wgt * dr_dyn + (1.0 - wgt) * (r_kin - r) / p.tau_kin

    # semi-implicit Euler: velocities first, then pose with the new velocities
    vx, vy, r = vx + dt * dvx, vy + dt * dvy, r + dt * dr
    psi = psi + dt * r
    X = X + dt * (vx * math.cos(psi) - vy * math.sin(psi))
    Y = Y + dt * (vx * math.sin(psi) + vy * math.cos(psi))

    return np.array([X, Y, psi, vx, vy, r, wf, wr, fyf_l, fyr_l, delta, a_act, SFx / p.m])


class ReferenceSim:
    """One car, float64, with the pure action delay buffer."""

    def __init__(self, p: VehicleParams, x0: np.ndarray | None = None, u0=None):
        self.p = p
        self.x = np.zeros(NX) if x0 is None else np.asarray(x0, float).copy()
        self.buf = np.zeros((BUF_LEN, 2))
        if u0 is not None:
            self.buf[:] = np.asarray(u0, float)

    @staticmethod
    def steady_state(v0: float, p: VehicleParams, curvature: float = 0.0) -> np.ndarray:
        """Free-rolling state at speed v0; with curvature k, a kinematic steady turn (r = v k, delta = L k)."""
        x = np.zeros(NX)
        x[IVX] = v0
        x[IWF] = x[IWR] = v0 / p.R_w
        x[IR] = v0 * curvature
        x[IDELTA] = (p.lf + p.lr) * curvature
        return x

    def step(self, action, sf: Surface, sr: Surface) -> np.ndarray:
        # per-axle cornering-stiffness scaling acts on the lateral B (substep() takes the effective surfaces)
        sf, sr = scaled_surfaces(sf, sr, self.p)
        self.buf[1:] = self.buf[:-1]
        self.buf[0] = action
        u = self.buf[self.p.delay_steps]
        dt = CONTROL_DT / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            self.x = substep(self.x, u[0], u[1], self.p, sf, sr, dt)
        return self.x
