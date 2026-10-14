"""Fit Magic Formula shapes (B, C, E) per surface to the Burckhardt mu-slip curves (P1 parameter sourcing, D13).

Burckhardt (1993), "Fahrwerktechnik: Radschlupf-Regelsysteme", Vogel, Wuerzburg; reproduced e.g. in
Kiencke & Nielsen, "Automotive Control Systems" (2005), and in many ABS papers:
    mu(s) = c1 (1 - exp(-c2 s)) - c3 s,   s = braking slip in [0, 1].
Only the *shape* is taken from Burckhardt (curve normalized by its own peak). The peak mu D stays in our
frozen ranges (prereg V7). Our tire evaluates MF on the theoretical slip sigma = s / (1 - s) for braking,
so the fit is done in that variable.

Run:  uv run python scripts/fit_surfaces.py
"""

import numpy as np
from scipy.optimize import least_squares

BURCKHARDT = {  # c1, c2, c3
    "dry": (1.2801, 23.99, 0.52),
    "wet": (0.857, 33.822, 0.347),
    "snow": (0.1946, 94.129, 0.0646),
    "ice": (0.05, 306.39, 0.0),
}


def burckhardt(s, c1, c2, c3):
    return c1 * (1 - np.exp(-c2 * s)) - c3 * s


def mf_norm(sigma, B, C, E):
    bx = B * sigma
    return np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))


def fit(name):
    c = BURCKHARDT[name]
    s = np.concatenate([np.linspace(1e-4, 0.3, 600), np.linspace(0.3, 0.95, 200)])
    mu = burckhardt(s, *c)
    target = mu / mu.max()
    sigma = s / (1 - s)
    w = np.where(s < 0.3, 3.0, 1.0)  # the pre-peak / peak region matters most for control

    def res(z):
        B, C, E = z
        return w * (mf_norm(sigma, B, C, E) - target)

    best = None
    for B0 in (5, 10, 20, 40, 80, 160):
        r = least_squares(res, [B0, 1.6, 0.5], bounds=([0.5, 1.0, -5.0], [1000, 2.0, 1.0]))
        if best is None or r.cost < best.cost:
            best = r
    B, C, E = best.x
    err = np.max(np.abs(mf_norm(sigma, B, C, E) - target))
    # where does the fitted curve peak, in slip ratio (longitudinal) and in slip angle (lateral, sigma = tan a)?
    sg = np.linspace(1e-5, 3, 300001)
    f = mf_norm(sg, B, C, E)
    sig_pk = sg[np.argmax(f)]
    s_peak_burck = s[np.argmax(mu)]
    return dict(B=B, C=C, E=E, max_abs_err=err, s_peak_fit=sig_pk / (1 + sig_pk), s_peak_burck=s_peak_burck,
                alpha_peak_deg=np.degrees(np.arctan(sig_pk)), slide_ratio=mf_norm(1e3, B, C, E),
                BC=B * C)


if __name__ == "__main__":
    print(f"{'surface':6s} {'B':>8s} {'C':>6s} {'E':>7s} {'max|err|':>9s} {'s_pk fit':>9s} {'s_pk Bur':>9s} "
          f"{'a_pk deg':>9s} {'F_slide/D':>9s} {'BC':>7s}")
    for n in BURCKHARDT:
        r = fit(n)
        print(f"{n:6s} {r['B']:8.3f} {r['C']:6.3f} {r['E']:7.3f} {r['max_abs_err']:9.4f} {r['s_peak_fit']:9.4f} "
              f"{r['s_peak_burck']:9.4f} {r['alpha_peak_deg']:9.2f} {r['slide_ratio']:9.3f} {r['BC']:7.2f}")
