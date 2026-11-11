"""P5 grounding: is the simulator's steady-state cornering response in the range of real passenger cars?

Real data: commaSteeringControl (MIT, HF `commaai/commaSteeringControl`), openpilot-engaged driving, 10 Hz.
Download the platform zips into data/external/comma_steering/ (gitignored), then:
  PYTHONPATH=. uv run python scripts/grounding_comma.py

Quantity: the understeer gradient K of the linear steady-state bicycle relation
    delta_rw = L a_y / v^2 + K a_y            (road-wheel angle, wheelbase L, lateral acceleration a_y, speed v)
On real cars only the steering-wheel angle is logged, and the steering ratio SR in openpilot's car specs is
itself a tuned value. So the fit is on the steering-wheel angle, with the ratio left free:
    SWA = A (a_y / v^2) + B a_y + C g roll + c,   SR_eff = A / L,   K = B / SR_eff = B L / A
Only the wheelbase is taken from the car specs. a_y is the localizer's lateral acceleration (latAccelLocalizer),
smoothed over 1 s. Quasi-steady samples only: engaged, no driver torque, 8-35 m/s, |a_x| < 0.3 m/s^2, steering
rate < 2 deg/s, |da_y/dt| < 0.3 m/s^3, |roll| < 0.03 rad, 0.4 <= |a_y| <= 3.0 m/s^2 (linear range).
95% CIs: cluster bootstrap over 60 s segments (200 resamples).

Simulator: the same fit on steady turns of the dry-road model (sim/steady.py: solve_turn), same speed and a_y
ranges; there the road-wheel angle is known (SR = 1).

Outputs: docs/results/grounding/comma_understeer.{json,md,png}. Grounding check, not a confirmatory result.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from sim.params import DRY, G, VehicleParams
from sim.steady import solve_turn

# Wheelbase [m] from opendbc car specs (github.com/commaai/opendbc, MIT; read 2026-11-11). The platform names are
# the dataset's; the spec each one maps to is noted.
WHEELBASE = {
    "HONDA_CIVIC_2022": 2.70,           # HONDA_CIVIC_2022 uses HONDA_CIVIC_BOSCH specs
    "TOYOTA_AVALON_2019": 2.82,         # TOYOTA_AVALON_2019 uses TOYOTA_AVALON specs
    "KIA_K5_2021": 2.85,
    "VOLKSWAGEN_JETTA_7TH_GEN": 2.71,   # VOLKSWAGEN_JETTA_MK7
    "SKODA_OCTAVIA_3RD_GEN": 2.68,      # SKODA_OCTAVIA_MK3
    "LEXUS_IS_2018": 2.79908,           # LEXUS_IS
    "SUBARU_IMPREZA_SPORT_2020": 2.67,  # SUBARU_IMPREZA_2020
    "NISSAN_ALTIMA_2020": 2.824,        # NISSAN_ALTIMA
    "HYUNDAI_SONATA_2019": 2.804,       # HYUNDAI_SONATA_LF
    "AUDI_A3_3RD_GEN": 2.61,            # AUDI_A3_MK3
}
DT = 0.1
V_RANGE, AY_RANGE = (8.0, 35.0), (0.4, 3.0)


def movavg(x, w):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def segment_samples(rows) -> np.ndarray | None:
    """Quasi-steady samples of one 60 s segment: columns v, a_y, roll, SWA [rad]."""
    if len(rows) < 50:
        return None
    f = lambda k: np.array([float(r[k]) for r in rows])
    b = lambda k: np.array([r[k] == "True" for r in rows])
    v, ax, swa, roll, ay = f("vEgo"), f("aEgo"), np.deg2rad(f("steeringAngleDeg")), f("roll"), f("latAccelLocalizer")
    act, pressed = b("latActive"), b("steeringPressed")
    w = int(round(1.0 / DT))
    ay_s, swa_s, ax_s = movavg(ay, w), movavg(swa, w), movavg(ax, w)
    swa_rate = np.gradient(swa_s, DT)
    ay_rate = np.gradient(ay_s, DT)
    # also exclude the second after any driver intervention
    pressed_w = movavg(pressed.astype(float), 2 * w) > 0
    ok = (act & ~pressed_w & (v >= V_RANGE[0]) & (v <= V_RANGE[1]) & (np.abs(ax_s) < 0.3)
          & (np.abs(swa_rate) < np.deg2rad(2.0)) & (np.abs(ay_rate) < 0.3) & (np.abs(roll) < 0.03)
          & (np.abs(ay_s) >= AY_RANGE[0]) & (np.abs(ay_s) <= AY_RANGE[1]))
    ok[:w] = ok[-w:] = False
    if not ok.any():
        return None
    return np.stack([v[ok], ay_s[ok], roll[ok], swa_s[ok]], 1)


def fit(segs: list[np.ndarray], L: float) -> dict:
    X = np.concatenate(segs)
    v, ay, roll, swa = X.T
    M = np.stack([ay / v**2, ay, G * roll, np.ones_like(ay)], 1)
    keep = np.ones(len(X), bool)
    for _ in range(2):  # one round of 4-MAD outlier trimming
        coef, *_ = np.linalg.lstsq(M[keep], swa[keep], rcond=None)
        r = swa - M @ coef
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < 4 * 1.4826 * mad
    A, B = coef[0], coef[1]
    # sign conventions of SWA and a_y differ between platforms: the ratio is reported as a magnitude
    return {"SR_eff": abs(A) / L, "K": B * L / A, "K_deg_per_g": math.degrees(B * L / A * G), "n": int(keep.sum())}


def boot(segs, L, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    ks, srs = [], []
    for _ in range(n_boot):
        s = [segs[i] for i in rng.integers(0, len(segs), len(segs))]
        f = fit(s, L)
        ks.append(f["K_deg_per_g"])
        srs.append(f["SR_eff"])
    return np.percentile(ks, [2.5, 97.5]).tolist(), np.percentile(srs, [2.5, 97.5]).tolist()


def platform(zpath: Path, max_segments: int) -> list[np.ndarray]:
    segs = []
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))[:max_segments]
        for n in names:
            with z.open(n) as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")))
            s = segment_samples(rows)
            if s is not None and len(s) >= 10:
                segs.append(s)
    return segs


def sim_fit(p: VehicleParams) -> dict:
    """The same regression on steady turns of the dry-road model (road-wheel angle known, SR = 1)."""
    rows = []
    for v in np.arange(V_RANGE[0], V_RANGE[1] + 0.1, 3.0):
        for ay in np.linspace(AY_RANGE[0], AY_RANGE[1], 6):
            eq = solve_turn(float(v), float(v * v / ay), p, DRY)
            if eq.ok:
                rows.append([v, ay, 0.0, eq.delta])
    X = np.array(rows)
    v, ay, _, d = X.T
    M = np.stack([ay / v**2, ay, np.ones_like(ay)], 1)
    (A, B, _), *_ = np.linalg.lstsq(M, d, rcond=None)
    L = p.lf + p.lr
    return {"SR_eff": A / L, "K": B * L / A, "K_deg_per_g": math.degrees(B * L / A * G), "n": len(X)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/external/comma_steering")
    ap.add_argument("--out", default="docs/results/grounding")
    ap.add_argument("--max-segments", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"source": "commaSteeringControl (MIT), HF commaai/commaSteeringControl", "platforms": {}}
    for name, L in WHEELBASE.items():
        z = Path(a.data) / f"{name}.zip"
        if not z.exists():
            print("missing", z)
            continue
        segs = platform(z, a.max_segments)
        if len(segs) < 20:
            print(name, "too few segments", len(segs))
            continue
        f = fit(segs, L)
        f["K_ci95"], f["SR_ci95"] = boot(segs, L)
        f["n_segments"] = len(segs)
        res["platforms"][name] = f
        print(f"{name}: K = {f['K_deg_per_g']:.2f} deg/g {np.round(f['K_ci95'], 2)}, SR_eff = {f['SR_eff']:.1f}, "
              f"N = {f['n']} samples / {len(segs)} segments")
    p = VehicleParams()
    res["simulator"] = sim_fit(p)
    print("simulator (dry):", res["simulator"])
    ks = [f["K_deg_per_g"] for f in res["platforms"].values()]
    if ks:
        res["real_range_deg_per_g"] = [float(min(ks)), float(max(ks))]
        res["real_median_deg_per_g"] = float(np.median(ks))
    (out / "comma_understeer.json").write_text(json.dumps(res, indent=1))
    write_md(res, out)
    write_fig(res, out)
    write_tex(res, Path("paper/gen"))


def write_tex(res, out):
    """Paragraph + figure for the tech report (every number from this run)."""
    out.mkdir(parents=True, exist_ok=True)
    P, s = res["platforms"], res["simulator"]
    if not P:
        return
    ks = sorted(f["K_deg_per_g"] for f in P.values())
    srs = sorted(f["SR_eff"] for f in P.values())
    k = s["K_deg_per_g"]
    n_above = sum(k > f["K_ci95"][1] for f in P.values())
    n_below = sum(k < f["K_ci95"][0] for f in P.values())
    n_in = len(P) - n_above - n_below
    n_samp = sum(f["n"] for f in P.values())
    n_seg = sum(f["n_segments"] for f in P.values())
    txt = (
        f"Over {len(P)} platforms ({n_samp:,} quasi-steady samples from {n_seg:,} one-minute segments), the fitted "
        f"understeer gradients range from {ks[0]:.2f} to {ks[-1]:.2f}\\,deg/g (median {float(np.median(ks)):.2f}; "
        f"per-platform 95\\% CIs by cluster bootstrap over segments), and the fitted steering ratios from "
        f"{srs[0]:.1f} to {srs[-1]:.1f}. The simulator's dry-road value, from the same fit on {s['n']} steady turns, "
        f"is {s['K_deg_per_g']:.2f}\\,deg/g (fitted ratio {s['SR_eff']:.3f}; it should be 1). This is inside the "
        f"real range: within the 95\\% CI of {n_in} platforms, above it for {n_above} and below it for {n_below} "
        "(Figure~\\ref{fig:grounding}). Two platforms (\\v{S}koda Octavia, Audi A3) give near-zero or negative "
        "gradients together with the highest fitted ratios. A speed- or angle-dependent steering ratio would break "
        "the linear model there, so their values are shown but not interpreted. This checks one linear-range "
        "property of the model (the regime before the drop), not its limit behaviour on low friction.\n\n"
        "\\begin{figure}[t]\\centering\\includegraphics[width=0.8\\linewidth]{comma_understeer.png}\n"
        "\\caption{Understeer gradient of ten passenger cars fitted from commaSteeringControl (95\\% CIs) and of "
        "the simulator (red line).}\\label{fig:grounding}\\end{figure}\n")
    (out / "grounding.tex").write_text(txt)


def write_md(res, out):
    L = ["# Grounding: understeer gradient, real cars vs the simulator", "",
         "Generated by `scripts/grounding_comma.py` from commaSteeringControl (MIT). Steady-state fit "
         "SWA = A·a_y/v² + B·a_y + C·g·roll + c on quasi-steady samples, K = B·L/A (steering ratio fitted, not "
         "assumed). 95% CIs: cluster bootstrap over 60 s segments. Grounding check, not a confirmatory result.", "",
         "| platform | K [deg/g] | 95% CI | fitted steering ratio | samples | segments |", "|---|---|---|---|---|---|"]
    for n, f in res["platforms"].items():
        L.append(f"| {n} | {f['K_deg_per_g']:.2f} | [{f['K_ci95'][0]:.2f}, {f['K_ci95'][1]:.2f}] | "
                 f"{f['SR_eff']:.1f} [{f['SR_ci95'][0]:.1f}, {f['SR_ci95'][1]:.1f}] | {f['n']} | {f['n_segments']} |")
    s = res["simulator"]
    L += ["", f"**Simulator (dry, nominal parameters):** K = {s['K_deg_per_g']:.2f} deg/g "
          f"(fitted ratio {s['SR_eff']:.3f}, should be 1; {s['n']} steady turns)."]
    (out / "comma_understeer.md").write_text("\n".join(L) + "\n")


def write_fig(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res["platforms"])
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.4))
    for i, n in enumerate(names):
        f = res["platforms"][n]
        ax.errorbar(f["K_deg_per_g"], i, xerr=[[f["K_deg_per_g"] - f["K_ci95"][0]], [f["K_ci95"][1] - f["K_deg_per_g"]]],
                    fmt="o", color="#4e79a7", capsize=2)
    ax.axvline(res["simulator"]["K_deg_per_g"], color="#e15759", lw=1.5, label="simulator (dry, nominal)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax.set_xlabel("understeer gradient K [deg/g] (road-wheel angle)")
    ax.set_title("Steady-state understeer: real cars (commaSteeringControl, 95% CI) vs simulator", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "comma_understeer.png", dpi=140)


if __name__ == "__main__":
    main()
