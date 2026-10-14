"""Magic Formula tire with combined slip (DESIGN.md §2.2), NumPy reference implementation.

Per lumped axle. Conventions:
- kappa: longitudinal slip ratio (>0 driving, <0 braking), alpha: slip angle [rad], positive alpha gives
  positive (leftward) lateral force.
- Separate longitudinal and lateral pure-slip curves F_x0, F_y0 (shapes Bx,Cx,Ex / By,Cy,Ey), peak D = mu*Fz.
- Nominal combined slip: Pacejka's normalized "theoretical slip" vector
      sx = kappa / (1 + kappa),  sy = tan(alpha) / (1 + kappa),  s = |(sx, sy)|,
      Fx = (sx/s) F_x0(s),  Fy = (sy/s) F_y0(s).
  Then |F|^2 = (sx/s)^2 F_x0^2 + (sy/s)^2 F_y0^2 <= D^2, so |F| <= mu*Fz always (gate S4), and pure slip
  reduces exactly to F_x0(sx) and F_y0(tan alpha).
- H2 mismatch variant (prereg V6 axis 1): longitudinal-priority friction ellipse,
      Fx = F_x0(sx_pure),  Fy = F_y0(tan alpha) * sqrt(1 - (Fx/D)^2),
  blended convexly with weight combined_alpha (a convex blend of two vectors in the friction circle stays in it).

The Warp version in sim/warp_sim.py implements exactly these equations.
"""

from __future__ import annotations

import numpy as np

ALPHA_MAX = np.pi / 2 - 0.01  # clamp before tan()
ONE_PLUS_KAPPA_MIN = 0.01  # keeps sx, sy finite when the wheel is locked (kappa -> -1)
S_EPS = 1e-9


def magic_formula(x, B, C, D, E):
    """Pacejka Magic Formula, D sin(C atan(Bx - E(Bx - atan(Bx)))). Odd in x."""
    bx = B * x
    return D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))


def _slip_vector(kappa, alpha):
    a = np.clip(alpha, -ALPHA_MAX, ALPHA_MAX)
    opk = np.maximum(1.0 + kappa, ONE_PLUS_KAPPA_MIN)
    return kappa / opk, np.tan(a) / opk


def slip_vector_forces(kappa, alpha, Fz, mu, Bx, Cx, Ex, By, Cy, Ey):
    sx, sy = _slip_vector(kappa, alpha)
    s = np.hypot(sx, sy)
    D = mu * Fz
    inv = np.where(s > S_EPS, 1.0 / np.maximum(s, S_EPS), 0.0)
    return magic_formula(s, Bx, Cx, D, Ex) * sx * inv, magic_formula(s, By, Cy, D, Ey) * sy * inv


def friction_ellipse_forces(kappa, alpha, Fz, mu, Bx, Cx, Ex, By, Cy, Ey):
    D = mu * Fz
    a = np.clip(alpha, -ALPHA_MAX, ALPHA_MAX)
    sx_pure = kappa / np.maximum(1.0 + kappa, ONE_PLUS_KAPPA_MIN)
    fx = magic_formula(sx_pure, Bx, Cx, D, Ex)
    fy0 = magic_formula(np.tan(a), By, Cy, D, Ey)
    ratio = np.where(D > 0, fx / np.maximum(D, 1e-12), 0.0)
    return fx, fy0 * np.sqrt(np.clip(1.0 - ratio * ratio, 0.0, 1.0))


def tire_forces(kappa, alpha, Fz, mu, Bx, Cx, Ex, By, Cy, Ey, combined_alpha=0.0):
    """Axle forces in the wheel frame: (Fx, Fy)."""
    kappa, alpha, Fz = np.asarray(kappa, float), np.asarray(alpha, float), np.asarray(Fz, float)
    args = (mu, Bx, Cx, Ex, By, Cy, Ey)
    fx_sv, fy_sv = slip_vector_forces(kappa, alpha, Fz, *args)
    if combined_alpha == 0.0:
        return fx_sv, fy_sv
    fx_fe, fy_fe = friction_ellipse_forces(kappa, alpha, Fz, *args)
    w = combined_alpha
    return (1 - w) * fx_sv + w * fx_fe, (1 - w) * fy_sv + w * fy_fe


def surface_args(s, mu=None):
    """(mu, Bx, Cx, Ex, By, Cy, Ey) of a Surface, optionally with a different mu."""
    return (s.mu if mu is None else mu, s.Bx, s.Cx, s.Ex, s.By, s.Cy, s.Ey)


def clip_to_circle(fx, fy, fmax):
    """Scale (fx, fy) down to radius fmax if it lies outside (used after tire relaxation)."""
    n = np.hypot(fx, fy)
    scale = np.where(n > fmax, fmax / np.maximum(n, 1e-12), 1.0)
    return fx * scale, fy * scale
