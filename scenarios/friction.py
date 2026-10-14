"""Friction field along the path (DESIGN.md §3.2): dry before the drop, an optional linear ramp, the patch
surface over the patch length, a ramp back, then dry again. The surface *shape* is blended with mu.

Blend weight w(s) in [0, 1] (0 = the dry base surface, 1 = the patch surface):
    s <  s_drop                         : 0
    s_drop <= s < s_drop + lt           : (s - s_drop)/lt            (lt = 0: a step)
    s_drop + lt <= s < s_end            : 1                          (s_end = s_drop + lt + Lp)
    s_end <= s < s_end + lt             : 1 - (s - s_end)/lt
    after                               : 0                          (Lp = inf: stays 1)
Surface parameters = (1 - w) * dry + w * patch, entry by entry. Every blended MF curve still peaks at
D = mu_blend * Fz, so |F| <= mu Fz holds everywhere (the certificate only needs that).
"""

from __future__ import annotations

import numpy as np

from sim.params import DRY, SURFACES
from sim.warp_sim import surface_vec

INF_PATCH = 1.0e9


def blend_weight(s, s_drop: float, lt: float, Lp: float):
    s = np.asarray(s, float)
    s_end = s_drop + lt + Lp
    if lt > 0:
        up = np.clip((s - s_drop) / lt, 0.0, 1.0)
        down = np.clip(1.0 - (s - s_end) / lt, 0.0, 1.0)
    else:
        up = (s >= s_drop).astype(float)
        down = (s < s_end).astype(float)
    return np.minimum(up, down)


def surface_at(s, s_drop: float, lt: float, Lp: float, mu0: float, patch: str, mu1: float) -> np.ndarray:
    """(..., 8) surface vectors (mu, Bx, Cx, Ex, By, Cy, Ey, 0) along s."""
    w = blend_weight(s, s_drop, lt, Lp)[..., None]
    a = surface_vec(DRY, mu0).astype(float)
    b = surface_vec(SURFACES[patch], mu1).astype(float)
    return (1 - w) * a + w * b
