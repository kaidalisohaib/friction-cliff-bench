"""Brute-force point-mass search (FEASIBILITY.md §2): tries hard to *stay inside* the CoG corridor under the same
acceleration bound the certificate uses. Used (a) to check the certificate is never contradicted (a point-mass
success in a certified case would be a bug) and (b) to measure how conservative the certificate is.

Strategy family ("turn first, brake with what is left"): radial acceleration that tracks the centreline circle,
  a_r* = -(v_theta^2 / r) - kp*rho - kd*rho_dot  (clipped to the budget),  a_theta = -sqrt(A^2 - a_r^2)
swept over gains. 1 ms semi-implicit Euler in Cartesian coordinates around the arc centre.
"""

from __future__ import annotations

import numpy as np

from feasibility.certificate import accel_bounds
from sim.params import VehicleParams

GAINS = [(kp, kd) for kp in (0.0, 0.5, 2.0, 8.0) for kd in (0.0, 1.0, 3.0)]


def survives(v_a, R, mu0, mu1, lt, Lp, p: VehicleParams = VehicleParams(), t_max=12.0, dt=1e-3):
    """True if some strategy keeps the CoG in [R - w, R + w] until the patch window ends (or the point mass has
    slowed enough to circle forever on the patch). Arrival: on the centreline, tangential, speed v_a."""
    w = 1.75 - p.width / 2
    A0, A1 = accel_bounds(p, mu0, mu1, v_a + 8.0)
    for kp, kd in GAINS:
        pos = np.array([R, 0.0])  # centre at the origin, moving counter-clockwise
        vel = np.array([0.0, v_a])
        prog = 0.0
        ok = True
        t = 0.0
        while t < t_max:
            r = np.linalg.norm(pos)
            er = pos / r
            et = np.array([-er[1], er[0]])
            vr, vt = vel @ er, vel @ et
            # rear axle on the patch after progress L + lt; window ends at front axle exit (progress lt + Lp)
            A = A0 if prog < p.L + lt else A1
            if prog >= lt + Lp:
                break
            if vt * vt / r <= 0.98 * A1 and abs(r - R) < w and prog >= p.L + lt:
                break  # can circle forever on the patch
            ar = np.clip(-(vt * vt / r) - kp * (r - R) - kd * vr, -A, A)
            at = -np.sqrt(max(A * A - ar * ar, 0.0)) if vt > 0.5 else 0.0
            vel = vel + dt * (ar * er + at * et)
            pos = pos + dt * vel
            prog += dt * (vel @ et) * R / r
            t += dt
            if abs(np.linalg.norm(pos) - R) > w:
                ok = False
                break
        if ok:
            return True
    return False


def threshold(R, mu0, mu1, lt, Lp, lo=3.0, hi=40.0, tol=0.1):
    """Smallest arrival speed at which no strategy in the family survives (bisection)."""
    if survives(hi, R, mu0, mu1, lt, Lp):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if survives(mid, R, mu0, mu1, lt, Lp) else (lo, mid)
    return hi
