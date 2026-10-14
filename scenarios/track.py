"""Reference paths (lane centrelines) and Frenet projection (DESIGN.md §3.1).

A track is a curvature profile k(s) built from primitives: lead-in straight, clothoid (linear curvature
ramp), constant-radius arc, clothoid out, exit straight. Curvature is continuous (no jumps). All left turns
(the problem is mirror-symmetric). Tracks are tabulated at DS = 0.25 m; every scenario references one of a
handful of shared tables (one per geometry), so the GPU holds only a few small arrays.

Frenet coordinates of a point: s = arc length of its projection on the centreline, e = signed lateral
offset (left of the direction of travel is positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DS = 0.25  # table resolution [m]
LANE_WIDTH = 3.5
LEAD_IN = 70.0  # straight before anything happens: >= 2 s at 25 m/s + margin (v0 strata max 25 +/- 1 m/s)
ARC_ANGLE = np.pi / 2  # every curve turns 90 degrees
EXIT_LEN = 350.0  # straight after the curve (episode can run 8 s past the drop at up to 26 m/s)
STRAIGHT_LEN = 600.0
RADII = (50.0, 100.0, 200.0, 400.0)


def clothoid_length(R: float) -> float:
    """Transition length: R/3 clipped to [15, 80] m (rate of change of curvature ~ 3/R^2 per metre)."""
    return float(np.clip(R / 3.0, 15.0, 80.0))


@dataclass(frozen=True)
class Track:
    name: str
    R: float  # 0 for the straight
    s: np.ndarray  # (N,)
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray  # heading of the centreline
    kappa: np.ndarray  # curvature
    # landmarks (arc length) for drop locations
    s_entry: float  # start of the clothoid in (curvature starts to rise)
    s_arc_start: float
    s_apex: float  # middle of the arc
    s_arc_end: float  # end of the arc ("exit")

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def curvature_at(self, s):
        return np.interp(s, self.s, self.kappa)


def _curvature_profile(s, R, Lc, La):
    if R == 0.0:
        return np.zeros_like(s)
    k0 = 1.0 / R
    s1 = LEAD_IN
    s2, s3, s4 = s1 + Lc, s1 + Lc + La, s1 + 2 * Lc + La
    k = np.zeros_like(s)
    k = np.where((s >= s1) & (s < s2), k0 * (s - s1) / Lc, k)
    k = np.where((s >= s2) & (s < s3), k0, k)
    k = np.where((s >= s3) & (s < s4), k0 * (1 - (s - s3) / Lc), k)
    return k


def build_track(R: float) -> Track:
    """Integrate heading and position on a fine grid (exact heading for piecewise-linear curvature)."""
    if R == 0.0:
        total, Lc, La = STRAIGHT_LEN, 0.0, 0.0
    else:
        Lc = clothoid_length(R)
        # a 90-degree turn: the two clothoids contribute Lc/R of heading in total; the arc takes the rest
        La = R * (ARC_ANGLE - Lc / R)
        total = LEAD_IN + 2 * Lc + La + EXIT_LEN
    fine = 0.01
    sf = np.arange(0.0, total + fine / 2, fine)
    kf = _curvature_profile(sf, R, Lc, La)
    th = np.concatenate([[0.0], np.cumsum(0.5 * (kf[1:] + kf[:-1]) * fine)])
    # positions: midpoint rule on heading
    thm = 0.5 * (th[1:] + th[:-1])
    xf = np.concatenate([[0.0], np.cumsum(np.cos(thm) * fine)])
    yf = np.concatenate([[0.0], np.cumsum(np.sin(thm) * fine)])
    step = int(round(DS / fine))
    sl = slice(None, None, step)
    s = sf[sl]
    if R == 0.0:
        return Track("straight", 0.0, s, xf[sl], yf[sl], th[sl], kf[sl], LEAD_IN, LEAD_IN, LEAD_IN, LEAD_IN)
    s1 = LEAD_IN
    return Track(f"R{int(R)}", R, s, xf[sl], yf[sl], th[sl], kf[sl],
                 s_entry=s1, s_arc_start=s1 + Lc, s_apex=s1 + Lc + La / 2, s_arc_end=s1 + Lc + La)


def all_tracks() -> list[Track]:
    """Track table order = track_id used by scenarios: 0 straight, 1..4 the radii."""
    return [build_track(0.0)] + [build_track(R) for R in RADII]


def project(track: Track, x: float, y: float, hint: int | None = None, window: int = 40):
    """Frenet (s, e, idx) of a point. Brute force if no hint, else a local search around the hint."""
    if hint is None:
        d2 = (track.x - x) ** 2 + (track.y - y) ** 2
        j = int(np.argmin(d2))
    else:
        lo, hi = max(hint - window, 0), min(hint + window + 1, len(track.s))
        d2 = (track.x[lo:hi] - x) ** 2 + (track.y[lo:hi] - y) ** 2
        j = lo + int(np.argmin(d2))
    best = None
    for a in (j - 1, j):
        if a < 0 or a + 1 >= len(track.s):
            continue
        tx, ty = track.x[a + 1] - track.x[a], track.y[a + 1] - track.y[a]
        seg2 = tx * tx + ty * ty
        t = np.clip(((x - track.x[a]) * tx + (y - track.y[a]) * ty) / seg2, 0.0, 1.0)
        px, py = track.x[a] + t * tx, track.y[a] + t * ty
        d = (x - px) ** 2 + (y - py) ** 2
        if best is None or d < best[0]:
            seg = np.sqrt(seg2)
            e = (tx * (y - track.y[a]) - ty * (x - track.x[a])) / seg  # cross(t, p - a): left positive
            best = (d, track.s[a] + t * (track.s[a + 1] - track.s[a]), e, j)
    return best[1], best[2], best[3]
