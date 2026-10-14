"""Magic Formula tire with combined slip (DESIGN.md §2.2), NumPy reference implementation.

Per lumped axle. Conventions:
- kappa: longitudinal slip ratio (>0 driving, <0 braking), alpha: slip angle [rad], positive alpha gives
  positive (leftward) lateral force.
- Peak force D = mu * Fz (so |F| <= mu * Fz always, gate S4).
- Nominal combined slip: Pacejka's normalized "theoretical slip" vector
  sx = kappa / (1 + kappa), sy = tan(alpha) / (1 + kappa), force magnitude MF(|s|) along s.
- H2 mismatch variant (prereg V6 axis 1): longitudinal-priority friction ellipse,
  Fx = MF(sx_pure), Fy = MF(tan alpha) * sqrt(1 - (Fx/D)^2), blended convexly with weight combined_alpha.

The Warp version in sim/warp_sim.py must implement exactly these equations.
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


def slip_vector_forces(kappa, alpha, Fz, mu, B, C, E):
    sx, sy = _slip_vector(kappa, alpha)
    s = np.hypot(sx, sy)
    F = magic_formula(s, B, C, mu * Fz, E)
    inv = np.where(s > S_EPS, 1.0 / np.maximum(s, S_EPS), 0.0)
    return F * sx * inv, F * sy * inv


def friction_ellipse_forces(kappa, alpha, Fz, mu, B, C, E):
    D = mu * Fz
    a = np.clip(alpha, -ALPHA_MAX, ALPHA_MAX)
    sx_pure = kappa / np.maximum(1.0 + kappa, ONE_PLUS_KAPPA_MIN)
    fx = magic_formula(sx_pure, B, C, D, E)
    fy0 = magic_formula(np.tan(a), B, C, D, E)
    ratio = np.where(D > 0, fx / np.maximum(D, 1e-12), 0.0)
    fy = fy0 * np.sqrt(np.clip(1.0 - ratio * ratio, 0.0, 1.0))
    return fx, fy


def tire_forces(kappa, alpha, Fz, mu, B, C, E, combined_alpha=0.0):
    """Axle forces in the wheel frame: (Fx, Fy)."""
    kappa, alpha, Fz = np.asarray(kappa, float), np.asarray(alpha, float), np.asarray(Fz, float)
    fx_sv, fy_sv = slip_vector_forces(kappa, alpha, Fz, mu, B, C, E)
    if combined_alpha == 0.0:
        return fx_sv, fy_sv
    fx_fe, fy_fe = friction_ellipse_forces(kappa, alpha, Fz, mu, B, C, E)
    w = combined_alpha
    return (1 - w) * fx_sv + w * fx_fe, (1 - w) * fy_sv + w * fy_fe


def clip_to_circle(fx, fy, fmax):
    """Scale (fx, fy) down to radius fmax if it lies outside (used after tire relaxation)."""
    n = np.hypot(fx, fy)
    scale = np.where(n > fmax, fmax / np.maximum(n, 1e-12), 1.0)
    return fx * scale, fy * scale
