"""G1 figure: analytic feasibility map (FEASIBILITY.md §2.1) over arrival speed x radius x post-drop mu x patch length.

Analytic setting: the arc continues for the whole patch (no track end), arrival on the centreline at the given
speed, dry mu0 = 0.95 before the drop, step transition. Each (speed, radius) cell is one of:
  - steady circling possible: v^2 / (R - w) <= mu1 g, so the point mass can circle on the patch (never certified);
  - certified infeasible: the certificate proves every trajectory leaves the lane;
  - undetermined by the certificate: needs the oracles (P2).
The dashed tick marks where the brute-force point-mass search stops surviving (the truth lies between that tick
and the certified region). Not a benchmark result: an analysis of the certificate itself.

Run:  PYTHONPATH=. uv run python scripts/report/fig_feasibility_map.py      (writes docs/figures/feasibility_map.png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from feasibility.certificate import certify_arc  # noqa: E402
from feasibility.pointmass import threshold  # noqa: E402
from sim.params import G, VehicleParams  # noqa: E402

P = VehicleParams()
W = 1.75 - P.width / 2
MU0 = 0.95
SURFS = [("ice", 0.10), ("snow", 0.30), ("wet", 0.65)]
PATCHES = [10.0, 30.0, 100.0, np.inf]
RADII = [50.0, 100.0, 200.0, 400.0]
SPEEDS = np.arange(5.0, 30.01, 0.25)
COL = {"circle": "#2a78d6", "cert": "#eb6834", "undet": "#1baf7a"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def classify(v, R, mu1, Lp):
    if v * v / (R - W) <= mu1 * G:
        return "circle"
    return "cert" if certify_arc(v, R, MU0, mu1, 0.0, Lp, 1e9).certified else "undet"


def main():
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SURFS), len(PATCHES), figsize=(13, 7.5), sharex=True, sharey=True,
                             facecolor=SURF)
    for i, (sname, mu1) in enumerate(SURFS):
        for j, Lp in enumerate(PATCHES):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            for k, R in enumerate(RADII):
                cls = [classify(v, R, mu1, Lp) for v in SPEEDS]
                # draw runs of equal class as bars with a 2px surface gap
                start = 0
                for n in range(1, len(SPEEDS) + 1):
                    if n == len(SPEEDS) or cls[n] != cls[start]:
                        x0, x1 = SPEEDS[start] - 0.125, SPEEDS[n - 1] + 0.125
                        ax.barh(k, x1 - x0, left=x0, height=0.62, color=COL[cls[start]], edgecolor=SURF, linewidth=2)
                        start = n
                vt = threshold(R, MU0, mu1, 0.0, Lp, lo=3.0, hi=34.0, tol=0.25)
                if vt is not None and vt <= SPEEDS[-1]:
                    ax.plot([vt, vt], [k - 0.42, k + 0.42], color=INK, lw=1.5, ls=(0, (2, 1.5)))
            if sname == "ice" and np.isinf(Lp):
                ax.scatter([15.0], [0], s=90, facecolor="none", edgecolor=INK, linewidth=1.8, zorder=5)
                ax.annotate("15 m/s, R 50 m, ice", (15.0, 0), (17.0, -0.85), color=INK, fontsize=8,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            ax.set_yticks(range(len(RADII)), [f"R {int(r)}" for r in RADII])
            ax.set_xlim(5, 30)
            ax.set_ylim(-1.1, len(RADII) - 0.4)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
            if i == 0:
                ax.set_title(f"patch {'endless' if np.isinf(Lp) else f'{int(Lp)} m'}", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sname} (mu1 {mu1:.2f})", color=INK, fontsize=10)
            if i == len(SURFS) - 1:
                ax.set_xlabel("arrival speed [m/s]", color=INK2, fontsize=9)
    handles = [Patch(color=COL["circle"], label="steady circling possible (point mass)"),
               Patch(color=COL["undet"], label="undetermined by the certificate (needs oracles)"),
               Patch(color=COL["cert"], label="certified infeasible"),
               plt.Line2D([], [], color=INK, ls=(0, (2, 1.5)), lw=1.5, label="best point-mass strategy found fails above")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.suptitle("Analytic feasibility map: arc continues through the patch, arrival on the centreline, dry mu0 0.95, "
                 "step transition", y=0.995, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "feasibility_map.png", dpi=160, facecolor=SURF)
    print("wrote", out / "feasibility_map.png")


if __name__ == "__main__":
    main()
