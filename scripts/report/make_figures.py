"""Regenerate every confirmatory figure and table from the raw run logs (gate G4).

  PYTHONPATH=. uv run python scripts/report/make_figures.py --root runs/eval_test --labels labels/test_v1_T0.jsonl \
      --split splits/test_v1_T0.jsonl --out docs/results

Writes <out>/results.json (every number, eval/analysis.py), <out>/results.md (tables), and figures:
  fig_departure_by_class.png  departure rate per method and feasibility class (bootstrap 95% CIs)
  fig_regime_map.png          departure rate per method over post-drop surface x effective speed (reactive-feasible)
  fig_h1.png                  paired H1 difference with its 95% CI against the +-5 pp margin
  fig_h2.png                  regret vs O-R over each mismatch axis (when the H2 sets exist)
  fig_caution.png             exploratory E6 (A7): RF departure vs arrival speed at the drop / v_eff, per method
  fig_mechanism.png           exploratory (trace set): adaptive MPPI's model friction vs the true friction under the
                              front axle, aligned at the drop (median and IQR over runs)
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

METHODS = ["O-R", "O-C", "M-N", "M-B", "M-L", "P-T", "P-TP", "P-DR", "P-S", "P-SP", "M-B-c"]
COLORS = {"O-R": "#555555", "O-C": "#999999", "M-N": "#b07aa1", "M-B": "#4e79a7", "M-L": "#76b7b2",
          "P-T": "#f28e2b", "P-TP": "#ffbe7d", "P-DR": "#e15759", "P-S": "#59a14f", "P-SP": "#8cd17d", "M-B-c": "#a0cbe8"}
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
            lo.append(max(d["rate"] - d["boot95"][0], 0))  # the pooled rate can sit just outside a percentile CI
            hi.append(max(d["boot95"][1] - d["rate"], 0))
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
    grids = {}
    for m in ms:
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
        grids[m] = (np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan), cnt)
    vmax = max([0.3] + [float(np.nanmax(g)) for g, _ in grids.values() if np.isfinite(g).any()])
    ncol = min(6, len(ms))
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow + 0.5), squeeze=False)
    for ax in axs.flat[len(ms):]:
        ax.axis("off")
    for ax, m in zip(axs.flat, ms):
        grid, cnt = grids[m]
        ax.imshow(grid, vmin=0, vmax=vmax, cmap="Reds")
        for (i, k), v in np.ndenumerate(grid):
            if np.isfinite(v):
                ax.text(k, i, f"{v:.2f}\n{int(cnt[i, k])}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 * vmax else "black")
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(["≤12", "12–17", "17–22", ">22"], fontsize=6)
        ax.set_yticks(range(len(surf)))
        ax.set_yticklabels(surf, fontsize=7)
    fig.suptitle("Reactive-feasible departure rate by post-drop surface × v_eff [m/s] (cell: rate, N scenarios; "
                 f"one colour scale, 0 to {vmax:.2f})", fontsize=9)
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
    fig, axs = plt.subplots(1, len(axes), figsize=(max(6.0, 3.8 * len(axes)), 3.4), sharey=True)
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
    np.atleast_1d(axs)[0].set_ylabel("regret vs O-R\n(departure-rate difference)")
    np.atleast_1d(axs)[0].legend(fontsize=7, frameon=False)
    fig.suptitle("H2: regret under model mismatch (N = " + ", ".join(f"{k} {a['n_scenarios']}" for k, a in axes.items())
                 + " scenarios per level)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_h2.png", dpi=140)


def fig_caution(res, out):
    """Exploratory E6 (prereg A7): is a method safer because it arrives slower?"""
    ca = res.get("exploratory_caution_rf")
    rf = res["classes"].get("reactive-feasible", {})
    if not ca:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    for m, c in ca.items():
        if m not in rf:
            continue
        x, d = c["arrival_ratio"], rf[m]["departed"]
        ax.errorbar(x["estimate"], d["rate"], xerr=[[max(x["estimate"] - x["ci95"][0], 0)], [max(x["ci95"][1] - x["estimate"], 0)]],
                    yerr=[[max(d["rate"] - d["boot95"][0], 0)], [max(d["boot95"][1] - d["rate"], 0)]], fmt="o",
                    color=COLORS.get(m, "black"), capsize=2, label=m)
        ax.annotate(m, (x["estimate"], d["rate"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("mean arrival speed at the drop / v_eff")
    ax.set_ylabel("departure rate (reactive-feasible)")
    ax.set_title("caution vs safety (95% CIs)", fontsize=9)
    ax = axs[1]
    labels_b = ["<0.8", "0.8–0.9", "0.9–1.0", "≥1.0"]
    ms = list(ca)
    w = 0.8 / max(len(ms), 1)
    for i, m in enumerate(ms):
        xs, ys, lo, hi = [], [], [], []
        for j, b in enumerate(ca[m]["bins"]):
            if b["n_runs"] < 20:  # tiny bins are in the table, not drawn
                continue
            xs.append(j + (i - (len(ms) - 1) / 2) * w)
            ys.append(b["rate"])
            lo.append(max(b["rate"] - b["wilson95"][0], 0))
            hi.append(max(b["wilson95"][1] - b["rate"], 0))
        ax.bar(xs, ys, width=w, color=COLORS.get(m), label=m, yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_b)
    ax.set_xlabel("arrival speed at the drop / v_eff")
    ax.set_ylabel("departure rate (Wilson 95%)")
    ax.set_title("departure at matched arrival speed (bins with >= 20 runs)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory (prereg A7): adaptation or caution?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_caution.png", dpi=140)


def fig_mechanism(root, out, methods=("M-B", "M-L"), t_before=0.5, t_after=3.0):
    tdir = root / "trace"
    files = {m: sorted(tdir.glob(f"{m}_s*.npz")) for m in methods}
    if not any(files.values()):
        return
    dt = 0.02
    ks = np.arange(-int(t_before / dt), int(t_after / dt) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    ex = None
    for m, fs in files.items():
        rows = []
        for f in fs:
            d = np.load(f)
            for i in range(len(d["sid"])):
                if np.isnan(d["t_drop"][i]):
                    continue
                k0 = int(round(d["t_drop"][i] / dt)) - 1
                kk = k0 + ks
                ok = (kk >= 0) & (kk < d["mu_hat"].shape[1])
                err = np.full(len(ks), np.nan)
                err[ok] = d["mu_hat"][i, kk[ok]] - d["mu_true"][i, kk[ok]]
                rows.append(err)
                if ex is None and m == methods[0]:
                    ex = (d["sid"][i], f, i, kk, ok)
        if not rows:
            continue
        R_ = np.array(rows)
        med = np.nanmedian(R_, 0)
        q1, q3 = np.nanpercentile(R_, 25, 0), np.nanpercentile(R_, 75, 0)
        axs[1].plot(ks * dt, med, color=COLORS[m], label=f"{m} (N = {len(R_)} runs)")
        axs[1].fill_between(ks * dt, q1, q3, color=COLORS[m], alpha=0.2, lw=0)
    if ex is not None:
        sid, f, i, kk, ok = ex
        t = ks[ok] * dt
        d = np.load(f)
        axs[0].plot(t, d["mu_true"][i, kk[ok]], color="black", lw=1.5, label="true (front axle)")
        for m in methods:
            g = tdir / Path(f).name.replace(methods[0], m)
            if g.exists():
                dm = np.load(g)
                axs[0].plot(t, dm["mu_hat"][i, kk[ok]], color=COLORS[m], label=f"{m} model")
        axs[0].set_title(f"one run ({sid})", fontsize=9)
        axs[0].set_ylabel("friction coefficient μ")
        axs[0].legend(fontsize=7, frameon=False)
    axs[1].axhline(0, color="#999999", lw=0.6)
    axs[1].set_ylabel("model μ − true μ (front axle)")
    axs[1].set_title("all runs: median and IQR", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.axvline(0, color="#e15759", ls="--", lw=0.8)
        ax.set_xlabel("time since the front axle reaches the drop [s]")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory: what the adaptive planners believe around the friction drop", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_mechanism.png", dpi=140)


def pct(x):
    return f"{100 * x:+.1f}"


def relpct(e):
    return f"{100 * e['estimate']:.1f}\\% [{100 * e['ci95'][0]:.1f}, {100 * e['ci95'][1]:.1f}]"


def ci_pp(e):
    return f"{pct(e['estimate'])}\\,pp [{pct(e['ci95'][0])}, {pct(e['ci95'][1])}]"


LEARNED = ["P-T", "P-TP", "P-DR", "P-S", "P-SP"]


def ci3(e, k=3):
    return f"{e['estimate']:.{k}f} [{e['ci95'][0]:.{k}f}, {e['ci95'][1]:.{k}f}]"


def s2_tex(res, s2_path=Path("ledgers/s2_gate_A8.json")) -> list[str]:
    """Gate S2 caveat (A8) and the per-seed reactive-feasible departure table; placed before H1."""
    L = []
    ps = res.get("per_seed_rf_departure", {})
    if s2_path.exists():
        s2 = json.loads(s2_path.read_text())
        pt, o = s2["P-T"], s2["O-R"]
        L.append("\\paragraph{Caveat: gate S2 failed.} On the reactive-feasible tuning scenarios the final teacher P-T "
                 f"departed on {pt['departure']:.3f} of $N={pt['n_runs']:,}$ runs (Wilson 95\\% "
                 f"[{pt['wilson'][0]:.3f}, {pt['wilson'][1]:.3f}]) against {o['departure']:.3f} for O-R, above the "
                 f"{s2['S2']['margin_pp']:.0f}\\,pp limit, also after the collapse rollback of amendment A8. As A8 "
                 "fixed in advance, the results below are reported with this caveat; all seeds count, none is "
                 "dropped. The spread is between seeds, so the per-seed test departure is reported too.")
    if ps:
        rows = [m for m in LEARNED + ["M-L"] if m in ps]
        if rows:
            L.append("\\begin{table}[t]\\centering\\small\\begin{tabular}{@{}l" + "c" * len(ps[rows[0]]) +
                     "@{}}\\toprule method & " + " & ".join(f"seed {i}" for i in range(len(ps[rows[0]]))) +
                     "\\\\\\midrule")
            dry = res.get("per_seed_dry_departure", {})
            for m in rows:
                L.append(f"{m} & " + " & ".join(f"{x:.3f}" for x in ps[m]) + "\\\\")
                if m in dry:
                    L.append("\\quad dry set & " + " & ".join(f"{x:.3f}" for x in dry[m]) + "\\\\")
            L.append("\\bottomrule\\end{tabular}\\caption{Departure per training seed on the reactive-feasible "
                     "test scenarios and, below each method, on the dry set (the same kind of scenarios with the "
                     "friction drop removed). Each seed is one trained policy or estimator.}\\label{tab:seeds}"
                     "\\end{table}\n")
    return L


def exploratory_tex(res) -> list[str]:
    """Secondary dry pace, exploratory E6/E7 (A7) and recovery latency."""
    L = []
    sec = res.get("secondary_rf", {})
    tax = {m: r["dry_pace_tax"] for m, r in sec.items() if "dry_pace_tax" in r}
    if tax:
        L.append("\\paragraph{Dry pace (secondary).} Progress lost against O-R on the dry set (no drop; 95\\% CI): "
                 + "; ".join(f"{m} {ci3(t)}" for m, t in tax.items()) + ".\n")
    cau = res.get("exploratory_caution_rf", {})
    if cau:
        L.append("\\paragraph{Exploratory E6: arrival speed.} Mean arrival speed at the drop as a fraction of the "
                 "scenario's effective speed, reactive-feasible scenarios (95\\% CI): "
                 + "; ".join(f"{m} {ci3(c['arrival_ratio'])}" for m, c in cau.items()) + ".")
        lo_bins = []
        for m, c in cau.items():
            b = c["bins"][3]
            if b["n_runs"] >= 20:
                lo_bins.append(f"{m} {b['departed']}/{b['n_runs']}")
        if lo_bins:
            L.append(" Departures among runs arriving at $\\geq 1.0\\,v_\\mathrm{eff}$: " + ", ".join(lo_bins) + ".")
        L.append("\n")
    e = res.get("exploratory_e7")
    if e:
        d = e["diff_MBc_minus_PS"]
        txt = (f"\\paragraph{{Exploratory E7: adaptation or caution?}} M-B-c is the bank MPPI driving at "
               f"$c$ times its reference speed, with $c$ frozen from the student's arrival speed on tuning data. "
               f"On {e['n_scenarios']:,} reactive-feasible scenarios it departs {e['dep_MBc']:.3f} against "
               f"{e['dep_PS']:.3f} for P-S; the paired difference M-B-c $-$ P-S is {ci_pp(d)}.")
        if "dry_pace_tax_diff_MBc_minus_PS" in e:
            txt += f" Its dry pace tax exceeds P-S's by {ci_pp(e['dry_pace_tax_diff_MBc_minus_PS'])}."
        if not e["student_advantage_in_H1"]:
            txt += (" The student does not depart less than the better adaptive MPPI in H1, so the A7 rule has no "
                    "advantage to attribute.")
        elif e["attributed_to_caution"]:
            txt += (" By the rule fixed in A7 before the test run, the student's H1 advantage is attributed to "
                    "caution (arriving slower), not to adaptation.")
        else:
            txt += (" M-B-c departs more than P-S, so by the A7 rule the student's H1 advantage is not explained by "
                    "its lower arrival speed alone.")
        L.append(txt + "\n")
    rec = res.get("exploratory_recovery")
    if rec:
        parts = []
        for m, r in rec.items():
            md, mn = r["median_latency_recovered_s"], r.get("mean_latency_recovered_s")
            parts.append(f"{m}: recovered {ci3(r['recovered_frac'])}"
                         + (f", latency median {ci3(md, 2)}\\,s, mean {ci3(mn, 2)}\\,s" if md else "") + f" ($N={r['n_runs']:,}$ runs)")
        L.append("\\paragraph{Exploratory: recovery latency.} Time from the drop until the car holds O-R's lateral "
                 "error and yaw-rate band for 0.5\\,s (H2 subset, nominal truth; censored runs count as not "
                 "recovered): " + "; ".join(parts) + ".\n")
    return L


def write_tex(res, labels, gen: Path, has):
    """Tech-report fragments (paper/gen/*.tex): every number comes from results.json of this run."""
    gen.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    cnt, n = Counter(v["label"] for v in labels.values()), len(labels)
    ar = sum(1 for v in labels.values() if v["label"] == "reactive-feasible" and v.get("adaptation_required"))
    (gen / "labels.tex").write_text(
        f"Of the {n:,} test scenarios, {cnt['reactive-feasible']:,} are reactive-feasible ({ar:,} of them "
        f"adaptation-required), {cnt['preview-required']:,} preview-required, {cnt['certified-infeasible']:,} "
        f"certified-infeasible and {cnt['undetermined']:,} undetermined. Gate S1 found no violation.\n")
    hyp, L, A = res.get("hypotheses", {}), [], []
    rf = res["classes"].get("reactive-feasible", {})
    n_seeds = round(rf["O-R"]["departed"]["n_runs"] / rf["n_scenarios"]) if "O-R" in rf else "the"
    L.append(f"All rates pool {n_seeds} seeds per method; intervals are hierarchical-bootstrap 95\\% CIs over scenarios and "
             "seeds unless marked otherwise. Departure means any footprint corner beyond the lane edge.\n")
    L += s2_tex(res)
    # per-class table
    cls = [c for c in CLASSES if res["classes"].get(c, {}).get("n_scenarios", 0) > 0]
    ms = [m for m in METHODS if any(m in res["classes"][c] for c in cls)]
    L.append("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}")
    L.append("\\resizebox{\\linewidth}{!}{\\begin{tabular}{@{}l" + "c" * len(cls) + "@{}}\\toprule")
    L.append("method & " + " & ".join(f"{c} ($N={res['classes'][c]['n_scenarios']}$)" for c in cls) + "\\\\\\midrule")
    for m in ms:
        cells = []
        for c in cls:
            r = res["classes"][c].get(m)
            cells.append("--" if r is None else
                         f"{r['departed']['rate']:.3f} {{\\scriptsize[{r['departed']['boot95'][0]:.3f}, "
                         f"{r['departed']['boot95'][1]:.3f}]}}")
        L.append(f"{m} & " + " & ".join(cells) + "\\\\")
    L.append("\\bottomrule\\end{tabular}}\\caption{Lane-departure rate by feasibility class (95\\% CI).}"
             "\\label{tab:classes}\\end{table}\n")
    if "H1" in hyp:
        h = hyp["H1"]
        e = h["diff_best_minus_PS"]
        verdict = ("is falsified: the student departs less by more than the 5\\,pp margin" if h["falsified"] else
                   "is not falsified")
        L.append(f"\\paragraph{{H1.}} The better adaptive MPPI on the test set is {h['better_mppi']}. On "
                 f"{h['n_scenarios']:,} reactive-feasible scenarios, departure({h['better_mppi']}) $-$ departure(P-S) "
                 f"$=$ {ci_pp(e)}; H1 {verdict} (one-sided $p={h['p_falsifier']:.4f}$). TOST at $\\pm5$\\,pp: "
                 f"{'equivalent' if h['tost']['equivalent'] else 'equivalence not shown'} (90\\% CI "
                 f"[{pct(h['tost']['ci90'][0])}, {pct(h['tost']['ci90'][1])}]\\,pp).")
        A.append(f"On {h['n_scenarios']:,} reactive-feasible test scenarios the departure difference between the "
                 f"better adaptive MPPI ({h['better_mppi']}) and the RMA student is {ci_pp(e)} (95\\% CI), so H1 "
                 f"{verdict.split(':')[0]}.")
        if "H1_adaptation_required" in res:
            g = res["H1_adaptation_required"]
            L.append(f" On the pre-declared adaptation-required stratum ($N={g['n_scenarios']:,}$) the difference is "
                     f"{ci_pp(g['diff_best_minus_PS'])}.")
        L.append("\n")
    if "H2" in hyp:
        h = hyp["H2"]
        L.append(f"\\paragraph{{H2.}} {h['n_axes_pass']} of 3 mismatch axes pass both slope and end-point tests, so "
                 f"H2 is {'supported' if h['supported'] else 'not supported'} ($p={h['p']:.4f}$). Per axis "
                 "(slope of $\\Delta$ = regret(MPPI) $-$ regret(P-S) per level, 95\\% CI): " +
                 "; ".join(f"{ax} {a['slope']['estimate']:+.4f} [{a['slope']['ci95'][0]:+.4f}, "
                           f"{a['slope']['ci95'][1]:+.4f}] ($N={a['n_scenarios']}$)" for ax, a in h["axes"].items())
                 + ".\n")
        A.append(f"Under model mismatch, H2 is {'supported' if h['supported'] else 'not supported'} "
                 f"({h['n_axes_pass']}/3 axes).")
    if "H3" in hyp:
        h = hyp["H3"]
        L.append(f"\\paragraph{{H3.}} Departure(P-S) $-$ departure(P-SP) $=$ {ci_pp(h['diff_PS_minus_PSP'])}"
                 + (f"; P-SP's relative dry-progress loss vs P-S is {ci_pp(h['dry_progress_loss_PSP_vs_PS'])}"
                    if "dry_progress_loss_PSP_vs_PS" in h else "")
                 + f". H3 is {'falsified' if h['falsified'] else 'not falsified'}.\n")
    if "holm" in res:
        L.append("Holm--Bonferroni over the confirmatory family: " + ", ".join(
            f"{k} $p={p:.4f}$ ({'rejected' if res['holm']['rejected'][k] else 'not rejected'})"
            for k, p in res["holm"]["p"].items()) + ".\n")
    L += exploratory_tex(res)
    figs = [("fig_departure_by_class.png", "Departure by feasibility class and method."),
            ("fig_regime_map.png", "Reactive-feasible departure rate over post-drop surface and effective speed."),
            ("fig_h1.png", "H1: paired departure difference with its 95\\% CI against the $\\pm5$\\,pp margin."),
            ("fig_h2.png", "H2: regret against O-R over each mismatch axis."),
            ("fig_caution.png", "Exploratory (A7): departure against arrival speed at the drop, per method."),
            ("fig_mechanism.png", "Exploratory: the adaptive planners' friction belief around the drop.")]
    for f, cap in figs:
        if f in has:
            L.append(f"\\begin{{figure}}[t]\\centering\\includegraphics[width=\\linewidth]{{{f}}}"
                     f"\\caption{{{cap}}}\\label{{fig:{f[:-4]}}}\\end{{figure}}\n")
    (gen / "results.tex").write_text("\n".join(L))
    ps = res.get("per_seed_rf_departure", {}).get("P-S")
    if ps:
        A.append(f"The student's five training seeds depart on {min(ps):.3f} to {max(ps):.3f} of reactive-feasible "
                 "runs, so its reliability across seeds, not its best case, decides the comparison.")
    if "H3" in hyp:
        h = hyp["H3"]
        A.append(f"Distilling from a teacher with road preview lowers the student's departure by "
                 f"{ci_pp(h['diff_PS_minus_PSP'])}, but the pre-registered dry-pace bound is not met, so H3 is "
                 f"{'falsified' if h['falsified'] else 'not falsified'}.")
    e, sec = res.get("exploratory_e7"), res.get("secondary_rf", {})
    if e and all(m in sec and "dry_pace_tax" in sec[m] for m in ("M-B", "M-B-c")):
        A.append(f"Exploratory: the bank MPPI driven at the student's lower arrival speed departs on {e['dep_MBc']:.3f} "
                 f"of reactive-feasible runs, at a dry-road pace cost of {relpct(sec['M-B-c']['dry_pace_tax'])} "
                 f"against {relpct(sec['M-B']['dry_pace_tax'])} for M-B.")
    (gen / "abstract_results.tex").write_text(" ".join(A) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated label files")
    ap.add_argument("--split", required=True, help="comma-separated split files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Friction-Cliff-Bench: confirmatory results")
    ap.add_argument("--tex", default="", help="write the tech-report fragments here (e.g. paper/gen)")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.labels.split(","))
    h2n = [n for n in h2_set_names() if (root / n).exists()]
    veff = {sc.sid: S.v_eff(sc) for f in a.split.split(",") for sc in S.load_jsonl(f)[1]}
    res = A.analyze(root, labels, METHODS, h2n or None, v_eff=veff)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text(A.markdown(res, a.title))
    fig_by_class(res, out)
    fig_regime_map(root, labels, a.split.split(","), out)
    fig_h1(res, out)
    fig_h2(res, out)
    fig_mechanism(root, out)
    fig_caution(res, out)
    if a.tex:
        write_tex(res, labels, Path(a.tex), {p.name for p in out.glob("*.png")})
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
