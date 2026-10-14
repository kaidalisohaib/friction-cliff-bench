"""Batched GPU/CPU vehicle simulator in NVIDIA Warp (DESIGN.md §2). One thread per car.

Implements exactly the equations of sim/reference.py (the NumPy answer key), in float32.
Verified against warp-lang 1.18.0. Known trap: wp.struct field defaults are ignored (fields read 0),
so every field is set explicitly in `make_params` and checked by tests/test_warp_sim.py.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import warp as wp

from sim.params import BUF_LEN, CONTROL_DT, N_SUBSTEPS, Surface, VehicleParams
from sim.reference import FD_DOMEGA, NX
from sim.tire import ALPHA_MAX, ONE_PLUS_KAPPA_MIN, S_EPS

wp.config.quiet = True

vec13 = wp.types.vector(length=NX, dtype=wp.float32)

_G = wp.constant(wp.float32(9.81))
_RHO = wp.constant(wp.float32(1.225))
_ALPHA_MAX = wp.constant(wp.float32(ALPHA_MAX))
_OPK_MIN = wp.constant(wp.float32(ONE_PLUS_KAPPA_MIN))
_S_EPS = wp.constant(wp.float32(S_EPS))
_FD = wp.constant(wp.float32(FD_DOMEGA))
_BUF = wp.constant(BUF_LEN)


@wp.struct
class Params:
    m: wp.float32
    Iz: wp.float32
    lf: wp.float32
    lr: wp.float32
    h: wp.float32
    R_w: wp.float32
    Jw: wp.float32
    b_w: wp.float32
    foot_front: wp.float32
    foot_rear: wp.float32
    width: wp.float32
    delta_max: wp.float32
    delta_rate: wp.float32
    tau_delta: wp.float32
    tau_T: wp.float32
    drive_force_max: wp.float32
    brake_force_max: wp.float32
    brake_front_frac: wp.float32
    drive_front_frac: wp.float32
    delay_steps: wp.int32
    B_scale_f: wp.float32
    B_scale_r: wp.float32
    f_rr: wp.float32
    CdA: wp.float32
    sigma_relax: wp.float32
    combined_alpha: wp.float32
    v_lo: wp.float32
    v_hi: wp.float32
    tau_kin: wp.float32
    eps_v: wp.float32


def make_params(p: VehicleParams) -> Params:
    """Explicitly copy *every* dataclass field into the Warp struct (never rely on struct defaults)."""
    s = Params()
    for f in fields(VehicleParams):
        setattr(s, f.name, getattr(p, f.name))
    return s


# --- tire ---------------------------------------------------------------------------------------------
@wp.func
def mf(x: wp.float32, B: wp.float32, C: wp.float32, D: wp.float32, E: wp.float32) -> wp.float32:
    bx = B * x
    return D * wp.sin(C * wp.atan(bx - E * (bx - wp.atan(bx))))


@wp.func
def tire(kappa: wp.float32, alpha: wp.float32, Fz: wp.float32, B: wp.float32, C: wp.float32,
         mu: wp.float32, E: wp.float32, ca: wp.float32) -> wp.vec2:
    a = wp.clamp(alpha, -_ALPHA_MAX, _ALPHA_MAX)
    opk = wp.max(wp.float32(1.0) + kappa, _OPK_MIN)
    sx = kappa / opk
    sy = wp.tan(a) / opk
    D = mu * Fz
    sn = wp.sqrt(sx * sx + sy * sy)
    fx = wp.float32(0.0)
    fy = wp.float32(0.0)
    if sn > _S_EPS:
        F = mf(sn, B, C, D, E)
        fx = F * sx / sn
        fy = F * sy / sn
    if ca != wp.float32(0.0):
        fx_e = mf(sx, B, C, D, E)
        ratio = wp.float32(0.0)
        if D > wp.float32(0.0):
            ratio = fx_e / D
        fy_e = mf(wp.tan(a), B, C, D, E) * wp.sqrt(wp.clamp(wp.float32(1.0) - ratio * ratio, wp.float32(0.0), wp.float32(1.0)))
        fx = (wp.float32(1.0) - ca) * fx + ca * fx_e
        fy = (wp.float32(1.0) - ca) * fy + ca * fy_e
    return wp.vec2(fx, fy)


@wp.func
def slip_ratio(R: wp.float32, omega: wp.float32, vxw: wp.float32, eps_v: wp.float32) -> wp.float32:
    return (R * omega - vxw) / wp.max(wp.max(wp.abs(R * omega), wp.abs(vxw)), eps_v)


@wp.func
def smoothstep(t: wp.float32) -> wp.float32:
    u = wp.clamp(t, wp.float32(0.0), wp.float32(1.0))
    return u * u * (wp.float32(3.0) - wp.float32(2.0) * u)


@wp.func
def clip_circle(f: wp.vec2, fmax: wp.float32) -> wp.vec2:
    n = wp.length(f)
    if n > fmax:
        return f * (fmax / n)
    return f


@wp.func
def wheel_update(omega: wp.float32, T_drive: wp.float32, T_brake: wp.float32, Fx: wp.float32,
                 vxw: wp.float32, alpha: wp.float32, Fz: wp.float32, B: wp.float32, C: wp.float32,
                 mu: wp.float32, E: wp.float32, p: Params, dt: wp.float32) -> wp.float32:
    Fx_h = tire(slip_ratio(p.R_w, omega + _FD, vxw, p.eps_v), alpha, Fz, B, C, mu, E, p.combined_alpha)[0]
    dFx = (Fx_h - Fx) / _FD
    k = wp.max((p.R_w * dFx + p.b_w) / p.Jw, wp.float32(0.0))
    denom = wp.float32(1.0) + dt * k
    om = omega + dt * (T_drive - p.R_w * Fx - p.b_w * omega) / p.Jw / denom
    db = dt * T_brake / p.Jw / denom
    if wp.abs(om) <= db:
        return wp.float32(0.0)
    if om > wp.float32(0.0):
        return om - db
    return om + db


@wp.func
def substep(x: vec13, u_steer: wp.float32, u_long: wp.float32, p: Params, sf: wp.vec4, sr: wp.vec4,
            dt: wp.float32) -> vec13:
    # sf, sr = (B, C, mu, E) effective per axle (B already scaled)
    vx = x[3]
    vy = x[4]
    r = x[5]
    wf = x[6]
    wr = x[7]
    fyf_l = x[8]
    fyr_l = x[9]
    delta = x[10]
    a_act = x[11]
    L = p.lf + p.lr
    one = wp.float32(1.0)
    zero = wp.float32(0.0)

    d_target = wp.clamp(u_steer, -one, one) * p.delta_max
    d_dot = wp.clamp((d_target - delta) / p.tau_delta, -p.delta_rate, p.delta_rate)
    delta = wp.clamp(delta + dt * d_dot, -p.delta_max, p.delta_max)
    a_act = a_act + dt / p.tau_T * (wp.clamp(u_long, -one, one) - a_act)

    T_drive = wp.max(a_act, zero) * p.drive_force_max * p.R_w
    T_brake = wp.max(-a_act, zero) * p.brake_force_max * p.R_w
    Tdf = T_drive * p.drive_front_frac
    Tdr = T_drive * (one - p.drive_front_frac)
    Tbf = T_brake * p.brake_front_frac
    Tbr = T_brake * (one - p.brake_front_frac)

    ax = wp.clamp(x[12], -wp.float32(0.95) * _G * p.lf / p.h, wp.float32(0.95) * _G * p.lr / p.h)
    Fzf = p.m * (_G * p.lr - ax * p.h) / L
    Fzr = p.m * (_G * p.lf + ax * p.h) / L

    c = wp.cos(delta)
    s = wp.sin(delta)
    vyf_body = vy + p.lf * r
    vxf = vx * c + vyf_body * s
    vyf = -vx * s + vyf_body * c
    vxr = vx
    vyr = vy - p.lr * r
    alpha_f = -wp.atan2(vyf, wp.abs(vxf))
    alpha_r = -wp.atan2(vyr, wp.abs(vxr))

    Ff = tire(slip_ratio(p.R_w, wf, vxf, p.eps_v), alpha_f, Fzf, sf[0], sf[1], sf[2], sf[3], p.combined_alpha)
    Fr = tire(slip_ratio(p.R_w, wr, vxr, p.eps_v), alpha_r, Fzr, sr[0], sr[1], sr[2], sr[3], p.combined_alpha)

    wf = wheel_update(wf, Tdf, Tbf, Ff[0], vxf, alpha_f, Fzf, sf[0], sf[1], sf[2], sf[3], p, dt)
    wr = wheel_update(wr, Tdr, Tbr, Fr[0], vxr, alpha_r, Fzr, sr[0], sr[1], sr[2], sr[3], p, dt)

    if p.sigma_relax > zero:
        fyf_l = fyf_l + wp.min(dt * wp.abs(vxf) / p.sigma_relax, one) * (Ff[1] - fyf_l)
        fyr_l = fyr_l + wp.min(dt * wp.abs(vxr) / p.sigma_relax, one) * (Fr[1] - fyr_l)
    else:
        fyf_l = Ff[1]
        fyr_l = Fr[1]
    Faf = clip_circle(wp.vec2(Ff[0], fyf_l), sf[2] * Fzf)
    Far = clip_circle(wp.vec2(Fr[0], fyr_l), sr[2] * Fzr)

    Fx_fb = Faf[0] * c - Faf[1] * s
    Fy_fb = Faf[0] * s + Faf[1] * c
    F_rr = -p.f_rr * (Fzf + Fzr) * wp.tanh(vx / wp.float32(0.1))
    F_aero = -wp.float32(0.5) * _RHO * p.CdA * vx * wp.abs(vx)
    SFx = Fx_fb + Far[0] + F_rr + F_aero
    SFy = Fy_fb + Far[1]
    Mz = p.lf * Fy_fb - p.lr * Far[1]

    dvx = SFx / p.m + r * vy
    dvy_dyn = SFy / p.m - r * vx
    dr_dyn = Mz / p.Iz
    wgt = smoothstep((wp.abs(vx) - p.v_lo) / (p.v_hi - p.v_lo))
    r_kin = vx * wp.tan(delta) / L
    vy_kin = p.lr * r_kin
    dvy = wgt * dvy_dyn + (one - wgt) * (vy_kin - vy) / p.tau_kin
    dr = wgt * dr_dyn + (one - wgt) * (r_kin - r) / p.tau_kin

    vx = vx + dt * dvx
    vy = vy + dt * dvy
    r = r + dt * dr
    psi = x[2] + dt * r
    X = x[0] + dt * (vx * wp.cos(psi) - vy * wp.sin(psi))
    Y = x[1] + dt * (vx * wp.sin(psi) + vy * wp.cos(psi))

    out = vec13()
    out[0] = X
    out[1] = Y
    out[2] = psi
    out[3] = vx
    out[4] = vy
    out[5] = r
    out[6] = wf
    out[7] = wr
    out[8] = fyf_l
    out[9] = fyr_l
    out[10] = delta
    out[11] = a_act
    out[12] = SFx / p.m
    return out


@wp.kernel
def step_kernel(params: wp.array(dtype=Params), state: wp.array(dtype=vec13), buf: wp.array3d(dtype=wp.float32),
                actions: wp.array2d(dtype=wp.float32), surf_f: wp.array(dtype=wp.vec4),
                surf_r: wp.array(dtype=wp.vec4), nsub: int, dt: wp.float32):
    i = wp.tid()
    p = params[i]
    # action delay buffer: index j holds the action from j control steps ago
    for jj in range(_BUF - 1):
        j = _BUF - 1 - jj
        buf[i, j, 0] = buf[i, j - 1, 0]
        buf[i, j, 1] = buf[i, j - 1, 1]
    buf[i, 0, 0] = actions[i, 0]
    buf[i, 0, 1] = actions[i, 1]
    u_s = buf[i, p.delay_steps, 0]
    u_l = buf[i, p.delay_steps, 1]
    sf = surf_f[i]
    sr = surf_r[i]
    sf = wp.vec4(sf[0] * p.B_scale_f, sf[1], sf[2], sf[3])
    sr = wp.vec4(sr[0] * p.B_scale_r, sr[1], sr[2], sr[3])
    x = state[i]
    for _k in range(nsub):
        x = substep(x, u_s, u_l, p, sf, sr, dt)
    state[i] = x


def surface_vec(s: Surface) -> np.ndarray:
    return np.array([s.B, s.C, s.mu, s.E], dtype=np.float32)


class WarpSim:
    """n cars stepped in parallel. Per-car parameters and per-axle surfaces; numpy in, numpy/wp out."""

    def __init__(self, n: int, params: VehicleParams | list[VehicleParams], device: str = "cpu"):
        self.n, self.device = n, device
        plist = params if isinstance(params, list) else [params] * n
        assert len(plist) == n
        self.params = wp.array([make_params(p) for p in plist], dtype=Params, device=device)
        self.state = wp.zeros(n, dtype=vec13, device=device)
        self.buf = wp.zeros((n, BUF_LEN, 2), dtype=wp.float32, device=device)
        self.actions = wp.zeros((n, 2), dtype=wp.float32, device=device)
        self.surf_f = wp.zeros(n, dtype=wp.vec4, device=device)
        self.surf_r = wp.zeros(n, dtype=wp.vec4, device=device)

    def reset(self, x0: np.ndarray, u0=None) -> None:
        """Set the state; u0 (shape (2,) or (n, 2)) pre-fills the action delay buffer so a steady-state
        start is not disturbed by stale zero actions."""
        x0 = np.broadcast_to(np.asarray(x0, np.float32), (self.n, NX))
        self.state.assign(wp.array(np.ascontiguousarray(x0), dtype=vec13, device=self.device))
        b = np.zeros((self.n, BUF_LEN, 2), np.float32)
        if u0 is not None:
            b[:] = np.broadcast_to(np.asarray(u0, np.float32), (self.n, 2))[:, None, :]
        self.buf.assign(wp.array(b, dtype=wp.float32, device=self.device))

    def set_surfaces(self, sf, sr) -> None:
        """sf, sr: Surface, or arrays of shape (n, 4) = (B, C, mu, E)."""
        for arr, s in ((self.surf_f, sf), (self.surf_r, sr)):
            v = surface_vec(s) if isinstance(s, Surface) else np.asarray(s, np.float32)
            arr.assign(wp.array(np.ascontiguousarray(np.broadcast_to(v, (self.n, 4))), dtype=wp.vec4, device=self.device))

    def step(self, actions) -> None:
        if isinstance(actions, wp.array):
            self.actions = actions
        else:
            a = np.ascontiguousarray(np.broadcast_to(np.asarray(actions, np.float32), (self.n, 2)))
            self.actions.assign(wp.array(a, dtype=wp.float32, device=self.device))
        wp.launch(step_kernel, dim=self.n,
                  inputs=[self.params, self.state, self.buf, self.actions, self.surf_f, self.surf_r,
                          N_SUBSTEPS, CONTROL_DT / N_SUBSTEPS],
                  device=self.device)

    def get_state(self) -> np.ndarray:
        return self.state.numpy().reshape(self.n, NX).astype(np.float64)
