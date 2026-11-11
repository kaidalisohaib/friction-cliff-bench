"""Regenerate every confirmatory figure and table from the raw run logs (gate G4).

  PYTHONPATH=. uv run python scripts/make_figures.py --root runs/eval_test --labels runs/label_test_T0/labels.jsonl \
      --split splits/test_v1_T0.jsonl --out docs/results

Writes <out>/results.json (every number, eval/analysis.py), <out>/results.md (tables), and figures:
  fig_departure_by_class.png  departure rate per method and feasibility class (bootstrap 95% CIs)
  fig_regime_map.png          departure rate per method over post-drop surface x effective speed (reactive-feasible)
  fig_h1.png                  paired H1 difference with its 95% CI against the +-5 pp margin
  fig_h2.png                  regret vs O-R over each mismatch axis (when the H2 sets exist)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval import analysis as A  # noqa: E402
from eval.runner import h2_set_names, load_labels  # noqa: E402
from scenarios import sampler as S  # noqa: E402

METHODS = ["O-R", "O-C", "M-N", "M-B", "M-L", "P-T", "P-TP", "P-DR", "P-S", "P-SP"]
COLORS = {"O-R": "#555555", "O-C": "#999999", "M-N": "#b07aa1", "M-B": "#4e79a7", "M-L": "#76b7b2",
          "P-T": "#f28e2b", "P-TP": "#ffbe7d", "P-DR": "#e15759", "P-S": "#59a14f", "P-SP": "#8cd17d"}
CLASSES = ["reactive-feasible", "adaptation-required", "preview-required", "certified-infeasible", "undetermined"]


def fig_by_class(res, out):
    cls = [c for c in CLASSES if res["classes"].get(c, {}).get("n_scenarios", 0) > 0]
    ms = [m for m in METHODS if any(m in res["classes"][c] for c in cls)]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        xs, ys, lo, hi = [], [], [], []
        for j, c in enumerate(cls):
            r = res["classes"][c].get(m)
            if r is None:
                continue
            d = r["departed"]
            xs.append(j + (i - (len(ms) - 1) / 2) * w)
            ys.append(d["rate"])
            lo.append(d["rate"] - d["boot95"][0])
            hi.append(d["boot95"][1] - d["rate"])
        ax.bar(xs, ys, width=w, color=COLORS.get(m), label=m, yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.7})
    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels([f"{c}\n(N = {res['classes'][c]['n_scenarios']})" for c in cls], fontsize=8)
    ax.set_ylabel("lane-departure rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Departure by feasibility class (pooled seeds; hierarchical-bootstrap 95% CIs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_departure_by_class.png", dpi=140)


def fig_regime_map(root, labels, split_files, out):
    main = A.load_set(root, "main", METHODS)
    scs = {sc.sid: sc for f in split_files for sc in S.load_jsonl(f)[1]}
    sids = main["O-R"]["sids"]
    rf = [j for j, s in enumerate(sids) if labels[s]["label"] == "reactive-feasible"]
    surf = ["wet", "snow", "ice"]
    bands = [(0, 12), (12, 17), (17, 22), (22, 99)]
    ms = [m for m in METHODS if m in main]
    fig, axs = plt.subplots(1, len(ms), figsize=(2.1 * len(ms), 2.9), sharey=True)
    for ax, m in zip(np.atleast_1d(axs), ms):
        dep = A.mat(main, m, "departed").mean(axis=0)
        grid = np.full((len(surf), len(bands)), np.nan)
        cnt = np.zeros_like(grid)
        for j in rf:
            sc = scs[sids[j]]
            ve = S.v_eff(sc)
            b = next(i for i, (lo, hi) in enumerate(bands) if lo < ve <= hi or (i == 0 and ve <= hi))
            si = surf.index(sc.surface)
            grid[si, b] = np.nansum([grid[si, b], dep[j]])
            cnt[si, b] += 1
        grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
        ax.imshow(grid, vmin=0, vmax=max(0.3, np.nanmax(grid) if np.isfinite(grid).any() else 0.3), cmap="Reds")
        for (i, k), v in np.ndenumerate(grid):
            if np.isfinite(v):
                ax.text(k, i, f"{v:.2f}\n{int(cnt[i, k])}", ha="center", va="center", fontsize=6)
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(["≤12", "12–17", "17–22", ">22"], fontsize=6)
        ax.set_yticks(range(len(surf)))
        ax.set_yticklabels(surf, fontsize=7)
    fig.suptitle("Reactive-feasible departure rate by post-drop surface × v_eff [m/s] (cell: rate, N scenarios)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_regime_map.png", dpi=140)


def fig_h1(res, out):
    rows = [("all reactive-feasible", res["hypotheses"].get("H1")),
            ("adaptation-required", res.get("H1_adaptation_required"))]
    rows = [(n, h) for n, h in rows if h]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 1.2 + 0.5 * len(rows)))
    for i, (n, h) in enumerate(rows):
        e = h["diff_best_minus_PS"]
        ax.errorbar(e["estimate"], i, xerr=[[e["estimate"] - e["ci95"][0]], [e["ci95"][1] - e["estimate"]]],
                    fmt="o", color="black", capsize=3)
        ax.text(e["ci95"][1] + 0.003, i, f"N = {h['n_scenarios']}", va="center", fontsize=7)
    for x in (-A.MARGIN, A.MARGIN):
        ax.axvline(x, color="#e15759", ls="--", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.6)
    lo = min(-2 * A.MARGIN, *(h["diff_best_minus_PS"]["ci95"][0] for _, h in rows))
    hi = max(2 * A.MARGIN, *(h["diff_best_minus_PS"]["ci95"][1] for _, h in rows))
    ax.set_xlim(lo - 0.02, hi + 0.08)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=8)
    best = rows[0][1]["better_mppi"]
    ax.set_xlabel(f"departure({best}) − departure(P-S)   (> 0: the RL student departs less)")
    ax.set_title("H1: paired difference, 95% CI; dashed = ±5 pp margin", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_h1.png", dpi=140)


def fig_h2(res, out):
    h2 = res["hypotheses"].get("H2") or ({"axes": res["H2_partial"]} if res.get("H2_partial") else None)
    if not h2:
        return
    axes = h2["axes"]
    labels = {"slip": ["0", "0.25", "0.5", "1"], "relax": ["0", "0.2", "0.4", "0.6"], "delay": ["+0", "+60", "+100", "+140"]}
    names = {"slip": "combined-slip blend α", "relax": "relaxation length [m]", "delay": "extra delay [ms]"}
    fig, axs = plt.subplots(1, len(axes), figsize=(3.6 * len(axes), 3), sharey=True)
    for ax, (axis, a) in zip(np.atleast_1d(axs), axes.items()):
        for m, ys in a["regret_by_level"].items():
            if ys:
                ax.plot(range(len(ys)), ys, "o-", color=COLORS.get(m, "black"), label=m)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels[axis])
        ax.set_xlabel(names[axis])
        ax.set_title(f"{axis}: pass = {a['pass']}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("regret vs O-R (departure-rate difference)")
    np.atleast_1d(axs)[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"H2: regret under model mismatch (N = {next(iter(axes.values()))['n_scenarios']} scenarios per level)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_h2.png", dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated label files")
    ap.add_argument("--split", required=True, help="comma-separated split files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Friction-Cliff-Bench: confirmatory results")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.labels.split(","))
    h2n = [n for n in h2_set_names() if (root / n).exists()]
    res = A.analyze(root, labels, METHODS, h2n or None)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text(A.markdown(res, a.title))
    fig_by_class(res, out)
    fig_regime_map(root, labels, a.split.split(","), out)
    fig_h1(res, out)
    fig_h2(res, out)
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
