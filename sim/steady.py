"""Steady-state cornering equilibrium of the vehicle model (for V2 and for scenario initial conditions).

Unknowns: lateral velocity vy, road-wheel angle delta, rear slip ratio kappa_r (drive force that holds
speed). Axle loads include the longitudinal transfer from ax = -r vy (centripetal acceleration seen in the
body frame when there is sideslip). Equations (body frame, constant speed vx on radius R, r = vx/R,
front free-rolling):
  dvx = 0:  Fxr - Fyf sin(delta) + F_rr + F_aero + m r vy = 0
  dvy = 0:  Fyf cos(delta) + Fyr - m r vx = 0
  dr  = 0:  lf Fyf cos(delta) - lr Fyr = 0
Solved with a damped Newton iteration (numerical Jacobian). No equilibrium exists above the limit speed.
Uses the NumPy reference tire, so it follows exactly the same force model as the simulators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sim.params import G, RHO_AIR, Surface, VehicleParams
from sim.reference import IAACT, IDELTA, IPSI, IR, IVX, IVY, IWF, IWR, NX, tire


@dataclass
class Equilibrium:
    ok: bool
    vy: float
    delta: float
    kappa_r: float
    Fxr: float
    residual: float


def _residual(z, vx, R, p: VehicleParams, sf: Surface, sr: Surface):
    vy, delta, kappa_r = z
    r = vx / R
    L = p.lf + p.lr
    # constant speed, but with sideslip the centripetal acceleration has a body-x component: ax = -r vy
    # (same quasi-static load transfer and ax clamp as the simulator)
    ax = min(max(-r * vy, -0.95 * G * p.lf / p.h), 0.95 * G * p.lr / p.h)
    Fzf, Fzr = p.m * (G * p.lr - ax * p.h) / L, p.m * (G * p.lf + ax * p.h) / L
    alpha_f = delta - math.atan2(vy + p.lf * r, abs(vx))
    alpha_r = -math.atan2(vy - p.lr * r, abs(vx))
    _, Fyf = tire(0.0, alpha_f, Fzf, sf, p.combined_alpha)
    Fxr, Fyr = tire(kappa_r, alpha_r, Fzr, sr, p.combined_alpha)
    F_res = -p.f_rr * p.m * G * math.tanh(vx / 0.1) - 0.5 * RHO_AIR * p.CdA * vx * abs(vx)
    return np.array([
        (Fxr - Fyf * math.sin(delta) + F_res + p.m * r * vy) / (p.m * G),
        (Fyf * math.cos(delta) + Fyr - p.m * r * vx) / (p.m * G),
        (p.lf * Fyf * math.cos(delta) - p.lr * Fyr) / (p.m * G * L),
    ]), Fxr


def solve_turn(vx: float, R: float, p: VehicleParams, sf: Surface, sr: Surface | None = None,
               tol: float = 1e-10, iters: int = 200) -> Equilibrium:
    """Find the steady turn at speed vx on radius R (left turn for R > 0). Surfaces get the per-axle B scale."""
    sr = sf if sr is None else sr
    sf = replace(sf, B=sf.B * p.B_scale_f)
    sr = replace(sr, B=sr.B * p.B_scale_r)
    L = p.lf + p.lr
    z = np.array([0.0, L / R, 0.0])
    # start from the linear-tire solution to land in the right (pre-peak) basin
    for _ in range(iters):
        f, _ = _residual(z, vx, R, p, sf, sr)
        if np.max(np.abs(f)) < tol:
            break
        J = np.zeros((3, 3))
        for j, h in enumerate((1e-6, 1e-7, 1e-7)):
            dz = np.zeros(3)
            dz[j] = h
            J[:, j] = (_residual(z + dz, vx, R, p, sf, sr)[0] - f) / h
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            break
        # damping: never move the slip angles by more than ~0.05 rad per iteration
        scale = min(1.0, 0.05 / max(abs(step[1]), abs(step[0]) / max(vx, 1.0), 1e-12), 0.05 / max(abs(step[2]), 1e-12))
        z = z + scale * step
    f, Fxr = _residual(z, vx, R, p, sf, sr)
    res = float(np.max(np.abs(f)))
    ok = res < 1e-8 and abs(z[1]) < p.delta_max
    return Equilibrium(ok, float(z[0]), float(z[1]), float(z[2]), float(Fxr), res)


def steady_turn_state(vx: float, R: float, p: VehicleParams, sf: Surface, sr: Surface | None = None):
    """Full simulator state + the action that holds it (for the action-buffer pre-fill)."""
    eq = solve_turn(vx, R, p, sf, sr)
    if not eq.ok:
        raise ValueError(f"no steady turn at vx={vx} on R={R} (residual {eq.residual:.2e})")
    x = np.zeros(NX)
    x[IVX], x[IVY], x[IR], x[IDELTA] = vx, eq.vy, vx / R, eq.delta
    # heading = path tangent (taken as 0) minus the sideslip angle, so the *velocity* is tangent to the path
    x[IPSI] = -math.atan2(eq.vy, vx)
    x[IWF] = vx / p.R_w
    x[IWR] = vx * (1.0 + eq.kappa_r) / p.R_w  # kappa = (R w - v)/v for R w > v
    a = eq.Fxr / p.drive_force_max if eq.Fxr >= 0 else eq.Fxr / p.brake_force_max
    x[IAACT] = a
    x[12] = -(vx / R) * eq.vy  # inertial ax, so the first substep sees the right load transfer
    return x, np.array([eq.delta / p.delta_max, a]), eq
