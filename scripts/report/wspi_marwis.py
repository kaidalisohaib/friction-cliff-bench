"""Friction statistics from MARWIS road-sensor logs (MCity-WSPI, CC-BY-4.0) against the simulator's mu ranges.

MARWIS is an optical road-weather sensor: its "friction" is a grip proxy on a 0-0.82 scale (0.82 = dry), not a tire
peak-friction measurement. So the comparison is made on the ratio to dry (surface grip / dry grip), which is what the
scenario sampler's mu ranges encode relative to the dry range.

Inputs (downloaded separately into their own directory, see docs/DATA_AND_LICENSES.md):
  <root>/mcity_wspi/csv/<day>/marwis_data*.csv   text road-condition labels (dry / snow covered / snow/ice covered)
  <root>/marwis_kalamazoo_2022-23/*.csv          numeric Lufft condition codes (meaning not documented in the
                                                 dataset; used only as an unlabelled overall distribution)
Each file is one drive. Samples within a drive are strongly correlated, so CIs come from a cluster bootstrap over
drives (2,000 resamples).

Usage: uv run python -I scripts/report/wspi_marwis.py data_ext/wspi_marwis/raw [--out docs/results/grounding]
       [--tex paper/gen/marwis.tex]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

MU_RANGES = {"dry": (0.85, 1.0), "wet": (0.5, 0.8), "snow": (0.20, 0.35), "ice": (0.08, 0.15)}  # sim/params.py
CLASS = {"dry": "dry", "snow covered": "snow", "snow/ice covered": "snow/ice"}


def load_mcity(root: Path) -> list[dict]:
    drives = []
    for f in sorted(root.glob("mcity_wspi/csv/*/marwis*.csv")):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        t = np.array([datetime.fromisoformat(r["timestamp_local"]).timestamp() for r in rows])
        drives.append({"file": f"{f.parent.name}/{f.name}", "t": t,
                       "cls": np.array([CLASS.get(r["road_condition"], r["road_condition"]) for r in rows]),
                       "mu": np.array([float(r["friction"]) for r in rows])})
    return drives


def load_kalamazoo(root: Path) -> list[np.ndarray]:
    out = []
    for f in sorted(root.glob("marwis_kalamazoo_2022-23/*.csv")):
        v = []
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            try:
                v.append(float(r["Friction []"]))
            except (KeyError, ValueError):
                pass
        if v:
            out.append(np.array(v))
    return out


def cluster_boot(groups: list[np.ndarray], stat, n_boot=2000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(stat(np.concatenate(groups)))
    bs = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize(root: Path) -> dict:
    drives = load_mcity(root)
    res = {"source": "MCity-WSPI MARWIS logs (CC-BY-4.0)", "classes": {}}
    dry = [d["mu"][d["cls"] == "dry"] for d in drives if (d["cls"] == "dry").any()]
    dry_med = float(np.median(np.concatenate(dry))) if dry else 0.82
    for c in ("dry", "snow", "snow/ice"):
        g = [d["mu"][d["cls"] == c] for d in drives if (d["cls"] == c).any()]
        if not g:
            continue
        allv = np.concatenate(g)
        med = cluster_boot(g, np.median)
        res["classes"][c] = {"n_samples": int(allv.size), "n_drives": len(g),
                             "median": med, "p05": float(np.quantile(allv, 0.05)),
                             "p95": float(np.quantile(allv, 0.95)),
                             "ratio_to_dry_median": [x / dry_med for x in med]}
    res["dry_reference"] = dry_med
    # update interval: time between changes of the reported value (the sensor's effective rate)
    dts = np.concatenate([np.diff(d["t"][np.r_[True, np.diff(d["mu"]) != 0]]) for d in drives])
    res["value_change_interval_s_median"] = float(np.median(dts[dts > 0]))
    res["log_interval_s_median"] = float(np.median(np.concatenate([np.diff(d["t"]) for d in drives])))
    kz = load_kalamazoo(root)
    if kz:
        allk = np.concatenate(kz)
        res["kalamazoo"] = {"n_samples": int(allk.size), "n_drives": len(kz),
                            "quantiles": {q: float(np.quantile(allk, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)},
                            "min": float(allk.min()),
                            "frac_below_0.4": cluster_boot(kz, lambda v: float(np.mean(v < 0.4)))}
    res["sim_mu_ranges"] = MU_RANGES
    res["sim_ratio_to_dry"] = {k: [lo / MU_RANGES["dry"][1], hi / MU_RANGES["dry"][0]] for k, (lo, hi) in
                               MU_RANGES.items()}
    return res


def write_md(res: dict, path: Path) -> None:
    L = ["# MARWIS friction proxy vs simulator mu ranges (harness output of `scripts/report/wspi_marwis.py`)", "",
         "MARWIS reports an optical grip proxy (0.82 = dry), not tire peak friction; compare ratios to dry. Medians "
         "with 95% cluster-bootstrap CIs over drives.", "",
         "| class | samples | drives | median [95% CI] | 5–95% | ratio to dry median [95% CI] |",
         "|---|---|---|---|---|---|"]
    for c, r in res["classes"].items():
        m, ra = r["median"], r["ratio_to_dry_median"]
        L.append(f"| {c} | {r['n_samples']} | {r['n_drives']} | {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}] | "
                 f"{r['p05']:.3f}–{r['p95']:.3f} | {ra[0]:.2f} [{ra[1]:.2f}, {ra[2]:.2f}] |")
    L += ["", "Simulator (sampled mu ranges and their ratio to dry): " + "; ".join(
        f"{k} {lo:.2f}–{hi:.2f} (ratio {res['sim_ratio_to_dry'][k][0]:.2f}–{res['sim_ratio_to_dry'][k][1]:.2f})"
        for k, (lo, hi) in res["sim_mu_ranges"].items()) + ".",
          "", f"Logged every {res['log_interval_s_median']:.2f} s (median); the reported value changes every "
              f"{res['value_change_interval_s_median']:.1f} s (median time between changes)."]
    if "kalamazoo" in res:
        k = res["kalamazoo"]
        f = k["frac_below_0.4"]
        L += ["", f"Kalamazoo 2022–23 winter drives (unlabelled; {k['n_samples']} samples, {k['n_drives']} drives): "
                  "quantiles " + ", ".join(f"{q}: {v:.2f}" for q, v in k["quantiles"].items()) + f", min {k['min']:.2f}"
              + f"; fraction below 0.4: {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}]."]
    path.write_text("\n".join(L) + "\n")


def write_fig(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = 0
    labels = []
    for k, (lo, hi) in res["sim_ratio_to_dry"].items():
        ax.plot([lo, min(hi, 1.0)], [y, y], lw=8, color="#E15759", alpha=0.6, solid_capstyle="butt")
        labels.append(f"sim {k}")
        y += 1
    for c, r in res["classes"].items():
        ra = r["ratio_to_dry_median"]
        p05, p95 = r["p05"] / res["dry_reference"], r["p95"] / res["dry_reference"]
        ax.plot([p05, p95], [y, y], lw=2, color="#4C72B0")
        ax.errorbar([ra[0]], [y], xerr=[[ra[0] - ra[1]], [ra[2] - ra[0]]], fmt="o", color="#4C72B0", capsize=3)
        labels.append(f"MARWIS {c} (n={r['n_samples']})")
        y += 1
    ax.set_yticks(range(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("friction relative to dry")
    ax.set_xlim(0, 1.05)
    ax.set_title("Simulator mu ranges (red) vs MARWIS grip proxy (blue: median, 95% CI; line: 5-95%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_tex(res: dict, path: Path) -> None:
    parts = []
    for c, r in res["classes"].items():
        ra = r["ratio_to_dry_median"]
        parts.append(f"{c} {ra[0]:.2f} [{ra[1]:.2f}, {ra[2]:.2f}] ($N={r['n_samples']:,}$ samples, {r['n_drives']} "
                     f"drive{'s' if r['n_drives'] != 1 else ''})")
    sim = res["sim_ratio_to_dry"]
    txt = ("Road-weather sensor logs from winter drives at Mcity (MARWIS, part of MCity-WSPI~\\citep{wspi}, CC-BY-4.0) give an "
           "optical grip proxy rather than tire friction, so we compare ratios to dry. Median grip relative to dry "
           "(95\\% cluster-bootstrap CI over drives): " + "; ".join(parts) + ". The simulator's snow range spans "
           f"{sim['snow'][0]:.2f}--{sim['snow'][1]:.2f} of dry and its ice range "
           f"{sim['ice'][0]:.2f}--{sim['ice'][1]:.2f}: no Mcity reading is that low, so our ice is harsher than "
           "these logs"
           + (f" (the lowest 1\\% of the unlabelled Kalamazoo winter readings, $N={res['kalamazoo']['n_samples']:,}$, "
              f"are below {res['kalamazoo']['quantiles'][0.01] / res['dry_reference']:.2f} of dry)"
              if "kalamazoo" in res else "")
           + ". The reported value changes only every "
           f"{res['value_change_interval_s_median']:.0f}\\,s or so (median) and averages over a road patch, so the "
           "sensor cannot resolve how sharp a transition is at the tire; the ramp lengths of the scenarios are not "
           "checked against it.")
    path.write_text(txt + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="docs/results/grounding")
    ap.add_argument("--tex", default="paper/gen/marwis.tex")
    a = ap.parse_args()
    res = summarize(Path(a.root))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "marwis.json").write_text(json.dumps(res, indent=1))
    write_md(res, out / "marwis.md")
    write_fig(res, out / "marwis.png")
    if a.tex:
        Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
        write_tex(res, Path(a.tex))
    print((out / "marwis.md").read_text())


if __name__ == "__main__":
    main()
