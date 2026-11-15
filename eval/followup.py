"""Analysis of the RL follow-up (prereg amendment A9, D36) on test pool v2 (the pre-declared top-up blocks T1–T3).

Inputs: run records (scripts/evaluate/eval_v2.py) under <root>/{main,dry}/<method>_s<seed>.jsonl, every method on the same
reactive-feasible v2 scenarios in the same order. Departure is the H1 metric; dry pace tax = 1 − mean dry progress
/ mean dry progress of O-R (as in eval/analysis.py). Intervals: hierarchical bootstrap (scenarios jointly, seeds per
method), 95%.

  H4 (non-inferior to the bank MPPI at matched pace; the one confirmatory test of A9):
       departure(P-SP2) − departure(M-Bc) < +2 pp. Supported if the upper 95% bound < 0.02; p = bootstrap mass at
       or above 0.02. Superiority (upper bound < 0) is reported as secondary. Pace match: |dry pace tax(P-SP2) − dry
       pace tax(M-Bc)| <= 3 pp (point estimate); otherwise H4 is reported as "pace not matched" and not interpreted.
  H5 (seed reliability; a pre-registered rule, no p-value): gate S2 (prereg V5: departure within O-R + 5 pp, point
       estimate) applied to every seed separately on the v2 RF scenarios. Holds if all 10 P-SP2 seeds pass.
       Reported for every learned method.
Exploratory (pre-declared): P-S − P-S2 (the checkpoint-selection fix on the current-friction student), P-S − P-SP2
(whole recipe), P-SP − P-SP2 (the fix on the preview-teacher student), the pre-declared top-5-by-validation subset of
P-SP2, per-seed tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import stats as St
from eval.analysis import boot_means, load_set, mat, summary

N_BOOT = 10_000
NI_MARGIN = 0.02  # H4 non-inferiority margin (absolute departure rate)
PACE_TOL = 0.03  # pace-match tolerance on the dry pace tax (absolute)
S2_MARGIN = 0.05  # gate S2 (prereg V5): departure within O-R + 5 pp
SEEDS = {"O-R": range(5), "M-B": range(5), "M-Bc": range(5), "P-S": range(10), "P-SP": range(5),
         "P-S2": range(10), "P-SP2": range(10)}


def load(root: Path, methods=tuple(SEEDS)) -> tuple[dict, dict]:
    main, dry = {}, {}
    for m in methods:
        main |= load_set(root, "main", [m], tuple(SEEDS[m]))
        dry |= load_set(root, "dry", [m], tuple(SEEDS[m]))
    return main, dry


def pace(dry: dict, m: str, n_boot: int = 4000, seed: int = 3) -> dict | None:
    if m not in dry or "O-R" not in dry:
        return None
    pm, po = mat(dry, m, "progress"), mat(dry, "O-R", "progress")
    bm = boot_means({"m": pm, "o": po}, n_boot=n_boot, seed=seed)
    return summary(1.0 - bm["m"] / bm["o"], float(1.0 - np.nanmean(pm) / np.nanmean(po)))


def pace_diff(dry: dict, a: str, b: str, n_boot: int = 4000, seed: int = 4) -> dict | None:
    if not all(m in dry for m in (a, b, "O-R")):
        return None
    pa, pb, po = (mat(dry, m, "progress") for m in (a, b, "O-R"))
    bm = boot_means({"a": pa, "b": pb, "o": po}, n_boot=n_boot, seed=seed)
    d = (bm["b"] - bm["a"]) / bm["o"]  # tax(a) - tax(b) = (prog_b - prog_a) / prog_o
    return summary(d, float((np.nanmean(pb) - np.nanmean(pa)) / np.nanmean(po)))


def diff(main: dict, a: str, b: str, n_boot: int = N_BOOT, seed: int = 1, cols=None) -> tuple[np.ndarray, dict]:
    x, y = mat(main, a, "departed", cols), mat(main, b, "departed", cols)
    d = St.hier_bootstrap_diff(x, y, n_boot, np.random.default_rng(seed))
    return d, summary(d, float(np.nanmean(x) - np.nanmean(y)))


def analyze(root: Path, top5: list[int] | None = None, n_boot: int = N_BOOT) -> dict:
    main, dry = load(root)
    res = {"n_scenarios": len(next(iter(main.values()))["sids"]) if main else 0, "methods": {}}
    for m in SEEDS:
        if m not in main:
            continue
        a = mat(main, m, "departed")
        bm = boot_means({"x": a}, n_boot=4000, seed=11)["x"]
        res["methods"][m] = {"departure": summary(bm, float(np.nanmean(a))),
                             "per_seed": [float(x) for x in np.nanmean(a, axis=1)],
                             "n_runs": int(np.isfinite(a).sum()), "dry_pace_tax": pace(dry, m)}
    if "P-SP2" in main and "M-Bc" in main:
        d, s = diff(main, "P-SP2", "M-Bc", n_boot, 2)
        pdiff = pace_diff(dry, "P-SP2", "M-Bc")
        matched = pdiff is not None and abs(pdiff["estimate"]) <= PACE_TOL
        res["H4"] = {"diff_PSP2_minus_MBc": s, "margin": NI_MARGIN, "pace_tax_diff_PSP2_minus_MBc": pdiff,
                     "pace_matched": bool(matched),
                     "non_inferior": bool(matched and s["ci95"][1] < NI_MARGIN),
                     "superior": bool(matched and s["ci95"][1] < 0), "p": float(np.mean(d >= NI_MARGIN))}
    if "O-R" in res["methods"]:
        lim = res["methods"]["O-R"]["departure"]["estimate"] + S2_MARGIN
        gate = {m: {"limit": lim, "pass_per_seed": [bool(x <= lim) for x in r["per_seed"]]}
                for m, r in res["methods"].items() if m.startswith("P-")}
        for g in gate.values():
            g["n_pass"], g["n_seeds"] = sum(g["pass_per_seed"]), len(g["pass_per_seed"])
        res["H5"] = {"gate": gate, "holds": bool("P-SP2" in gate and gate["P-SP2"]["n_pass"] == gate["P-SP2"]["n_seeds"])}
    ex = {}
    for a_, b_ in (("P-S", "P-S2"), ("P-S", "P-SP2"), ("P-SP", "P-SP2")):
        if a_ in main and b_ in main:
            ex[f"{a_}_minus_{b_}"] = diff(main, a_, b_, 4000, 5)[1]
    if top5 and "P-SP2" in main:
        sub = {"P-SP2": {"sids": main["P-SP2"]["sids"], "recs": [main["P-SP2"]["recs"][i] for i in top5]}}
        a = mat(sub, "P-SP2", "departed")
        ex["P-SP2_top5"] = {"seeds": top5, "departure": summary(boot_means({"x": a}, 4000, 12)["x"],
                                                                float(np.nanmean(a)))}
        if "M-Bc" in main:
            d = St.hier_bootstrap_diff(a, mat(main, "M-Bc", "departed"), 4000, np.random.default_rng(13))
            ex["P-SP2_top5_minus_MBc"] = summary(d, float(np.nanmean(a) - np.nanmean(mat(main, "M-Bc", "departed"))))
        if "P-SP2" in dry and "M-Bc" in dry:  # POST HOC (added 2026-11-14 after the v2 results): is it pace-matched?
            sd = {"P-SP2_top5": {"sids": dry["P-SP2"]["sids"], "recs": [dry["P-SP2"]["recs"][i] for i in top5]},
                  "M-Bc": dry["M-Bc"], "O-R": dry["O-R"]}
            ex["post_hoc_P-SP2_top5_pace_tax"] = pace(sd, "P-SP2_top5")
            ex["post_hoc_P-SP2_top5_minus_MBc_pace_tax"] = pace_diff(sd, "P-SP2_top5", "M-Bc")
    res["exploratory"] = ex
    return res


def pp(e: dict) -> str:
    return f"{100 * e['estimate']:+.1f} pp [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"


def markdown(res: dict) -> str:
    L = ["# RL follow-up on test pool v2 (prereg A9; harness output of `eval/followup.py`)", "",
         f"Reactive-feasible v2 scenarios: N = {res['n_scenarios']:,}; seeds per method as listed. 95% hierarchical-"
         "bootstrap CIs.", "", "| method | seeds | departure [95% CI] | per seed | dry pace tax [95% CI] |",
         "|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        d, pt = r["departure"], r["dry_pace_tax"]
        L.append(f"| {m} | {len(r['per_seed'])} | {d['estimate']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | "
                 f"{', '.join(f'{x:.3f}' for x in r['per_seed'])} | "
                 + (f"{100 * pt['estimate']:.1f}% [{100 * pt['ci95'][0]:.1f}, {100 * pt['ci95'][1]:.1f}]" if pt else "—")
                 + " |")
    if "H4" in res:
        h = res["H4"]
        pd_ = h["pace_tax_diff_PSP2_minus_MBc"]
        L += ["", f"**H4** (non-inferior to M-B at matched pace, margin {100 * h['margin']:.0f} pp): departure(P-SP2) − "
                  f"departure(M-Bc) = {pp(h['diff_PSP2_minus_MBc'])}; dry pace tax difference "
                  + (pp(pd_) if pd_ else "—") + "; "
                  + ("pace not matched (not interpreted)" if not h["pace_matched"] else
                     ("non-inferior" if h["non_inferior"] else "non-inferiority not shown")
                     + ("; superior" if h["superior"] else "; superiority not shown"))
                  + (f" (p = {h['p']:.4f})." if h["p"] > 0 else f" (bootstrap p < 1/{N_BOOT:,}).")]
    if "H5" in res:
        g = res["H5"]["gate"]
        L += ["", f"**H5** (seed reliability: gate S2 per seed, departure ≤ O-R + 5 pp = "
                  f"{next(iter(g.values()))['limit']:.3f}): "
                  + "; ".join(f"{m} {v['n_pass']}/{v['n_seeds']} seeds pass" for m, v in g.items())
                  + f". {'Holds' if res['H5']['holds'] else 'Does not hold'} for P-SP2."]
    ex = res.get("exploratory", {})
    if ex:
        L += ["", "## Exploratory (pre-declared in A9)"]
        if "P-S_minus_P-S2" in ex:
            L.append(f"- The checkpoint-selection fix on the current-friction student: departure(P-S) − departure(P-S2) = "
                     f"{pp(ex['P-S_minus_P-S2'])}.")
        if "P-S_minus_P-SP2" in ex:
            L.append(f"- Whole recipe (preview teacher + validation selection): departure(P-S) − departure(P-SP2) = "
                     f"{pp(ex['P-S_minus_P-SP2'])}.")
        if "P-SP_minus_P-SP2" in ex:
            L.append(f"- The fix on the preview-teacher student: departure(P-SP) − departure(P-SP2) = "
                     f"{pp(ex['P-SP_minus_P-SP2'])}.")
        if "P-SP2_top5" in ex:
            t = ex["P-SP2_top5"]
            L.append(f"- Top-5-by-validation P-SP2 seeds {t['seeds']}: departure {t['departure']['estimate']:.4f} "
                     f"[{t['departure']['ci95'][0]:.4f}, {t['departure']['ci95'][1]:.4f}]"
                     + (f"; minus M-Bc {pp(ex['P-SP2_top5_minus_MBc'])}" if "P-SP2_top5_minus_MBc" in ex else "")
                     + ".")
        if ex.get("post_hoc_P-SP2_top5_pace_tax"):
            t, dt = ex["post_hoc_P-SP2_top5_pace_tax"], ex["post_hoc_P-SP2_top5_minus_MBc_pace_tax"]
            L.append(f"- *Post hoc* (added after the v2 results, not pre-declared): the top-5 subset's dry pace tax is "
                     f"{100 * t['estimate']:.1f}% [{100 * t['ci95'][0]:.1f}, {100 * t['ci95'][1]:.1f}], "
                     f"{pp(dt)} relative to M-Bc, so the top-5 comparison above is "
                     + ("within" if abs(dt["estimate"]) <= PACE_TOL else "outside")
                     + " the A9 pace tolerance (3 pp).")
    return "\n".join(L) + "\n"


def tex(res: dict) -> str:
    """Report fragment (paper/gen/followup.tex; the section header and method text are in paper/main.tex); every
    number from `res`."""
    M = res["methods"]

    def dep(m):
        d = M[m]["departure"]
        return f"{d['estimate']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"

    def tax(m):
        t = M[m]["dry_pace_tax"]
        return f"{100 * t['estimate']:.1f}\\%" if t else "--"

    rows = "\n".join(f"{m} & {len(M[m]['per_seed'])} & {dep(m)} & {tax(m)} & "
                      f"{min(M[m]['per_seed']):.3f}--{max(M[m]['per_seed']):.3f} \\\\" for m in M)
    P = [f"\\textbf{{Results.}} Test pool v2 (the pre-declared top-up blocks, never used before): $N={str(format(res['n_scenarios'], ',')).replace(',', '{,}')}$ reactive-feasible "
         "scenarios; departure with 95\\% hierarchical-bootstrap CIs.",
         "\\begin{table}[t]\\centering\\small\\begin{tabular}{lcccc}\\toprule method & seeds & departure [95\\% CI] & "
         "dry pace tax & per-seed range \\\\\\midrule", rows, "\\bottomrule\\end{tabular}"
         "\\caption{Test v2. P-S2/P-SP2: checkpoints selected by deterministic validation; M-Bc: bank MPPI at the "
         "pace of P-SP2.}\\label{tab:followup}\\end{table}"]
    if "H4" in res:
        h = res["H4"]
        pdf = h["pace_tax_diff_PSP2_minus_MBc"]
        P.append((f"\\textbf{{H4 (non-inferiority at matched pace).}} departure(P-SP2) $-$ departure(M-Bc) $=$ "
                 f"{pp(h['diff_PSP2_minus_MBc'])}, margin {100 * h['margin']:.0f}\\,pp; dry pace tax difference "
                 + (pp(pdf) if pdf else "--") + ". "
                 + ("The pace match failed, so H4 is not interpreted." if not h["pace_matched"] else
                    ("Non-inferiority is shown" if h["non_inferior"] else "Non-inferiority is not shown")
                    + (" and the learned policy is also superior." if h["superior"] else
                       "; superiority is not shown.")))
                 .replace(" pp", "\\,pp"))
    if "H5" in res:
        g = res["H5"]["gate"]
        P.append("\\textbf{H5 (seed reliability).} Gate S2 per seed (departure $\\le$ O-R $+\\,5$\\,pp): "
                 + "; ".join(f"{m} {v['n_pass']}/{v['n_seeds']}" for m, v in g.items()) + ". "
                 + ("H5 holds." if res["H5"]["holds"] else "H5 does not hold."))
    ex = res.get("exploratory", {})
    parts = []
    for k, lab in (("P-S_minus_P-S2", "P-S $-$ P-S2"), ("P-S_minus_P-SP2", "P-S $-$ P-SP2"),
                   ("P-SP_minus_P-SP2", "P-SP $-$ P-SP2")):
        if k in ex:
            parts.append(f"{lab}: {pp(ex[k])}".replace(" pp", "\\,pp"))
    if "P-SP2_top5_minus_MBc" in ex:
        t = ex["P-SP2_top5"]
        part = (f"the P-SP2 seeds of the five best teachers by validation success depart "
                f"{t['departure']['estimate']:.4f} [{t['departure']['ci95'][0]:.4f}, {t['departure']['ci95'][1]:.4f}], "
                f"{pp(ex['P-SP2_top5_minus_MBc'])} against M-Bc")
        if ex.get("post_hoc_P-SP2_top5_minus_MBc_pace_tax"):
            part += (f", but they drive slower (post hoc, not pre-declared: dry pace tax "
                     f"{pp(ex['post_hoc_P-SP2_top5_minus_MBc_pace_tax'])} relative to M-Bc)")
        parts.append(part.replace(" pp", "\\,pp"))
    if parts:
        P.append("\\textbf{Exploratory.} " + "; ".join(parts) + ".")
    return "\n\n".join(P) + "\n"


def _h4_phrase(res: dict) -> str:
    h = res["H4"]
    d = pp(h["diff_PSP2_minus_MBc"])
    if not h["pace_matched"]:
        return f"the pace match with the slowed bank MPPI failed, so H4 is not interpreted (departure difference {d})"
    if h["non_inferior"]:
        out = (f"at matched pace its lane-departure rate is non-inferior to the bank MPPI's (difference {d}, margin "
               f"{100 * h['margin']:.0f} pp)")
        return out + ("" if not h["superior"] else " and lower")
    return f"at matched pace non-inferiority to the bank MPPI is not shown (difference {d})"


def _h5_phrase(res: dict) -> str:
    g = res["H5"]["gate"].get("P-SP2")
    if g is None:
        return ""
    return (f"all {g['n_seeds']} training seeds pass the per-seed reliability gate" if res["H5"]["holds"] else
            f"{g['n_pass']} of {g['n_seeds']} training seeds pass the per-seed reliability gate")


def abstract_tex(res: dict) -> str:
    """One abstract sentence (paper/gen/abstract_followup.tex); empty unless H4 and H5 were computed."""
    if "H4" not in res or "H5" not in res:
        return ""
    s = (f"A pre-registered follow-up repairs the learned policy's training (checkpoints selected on deterministic "
         f"held-out validation) and re-tests it on {res['n_scenarios']:,} fresh reactive-feasible scenarios: "
         f"{_h4_phrase(res)}, and {_h5_phrase(res)}.")
    return s.replace("%", "\\%").replace(" pp", "\\,pp") + "\n"


def readme_lines(res: dict) -> list[str]:
    """README follow-up block (scripts/report/readme_results.py); every number from `res`."""
    if "H4" not in res:
        return []
    M = res["methods"]
    L = ["", f"**Follow-up (pre-registered amendment A9, fresh test pool v2, N = {res['n_scenarios']:,} "
             "reactive-feasible scenarios).** The learned policy's training was repaired (checkpoints selected on "
             "deterministic held-out validation; diagnosis in the technical report, Section 10) "
             "and re-tested. Full tables: [`docs/results/followup.md`](docs/results/followup.md).", "",
         "| method | seeds | departure [95% CI] | per-seed range | dry pace tax [95% CI] |", "|---|---|---|---|---|"]
    for m, r in M.items():
        d, t = r["departure"], r["dry_pace_tax"]
        L.append(f"| {m} | {len(r['per_seed'])} | {d['estimate']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] | "
                 f"{min(r['per_seed']):.3f}–{max(r['per_seed']):.3f} | "
                 + (f"{100 * t['estimate']:.1f}% [{100 * t['ci95'][0]:.1f}, {100 * t['ci95'][1]:.1f}]" if t else "—")
                 + " |")
    L += ["", f"- **H4** (P-SP2 against M-Bc, the bank MPPI slowed to P-SP2's pace): {_h4_phrase(res)}."]
    if "H5" in res:
        L.append(f"- **H5** (P-SP2): {_h5_phrase(res)} (" + "; ".join(f"{m} {v['n_pass']}/{v['n_seeds']}"
                                                         for m, v in res["H5"]["gate"].items()) + ").")
    return L


# Figure palette (validated with the dataviz checker: lightness, chroma, all-pairs CVD and contrast pass); families,
# with marker shape as a second encoding.
FAMILY = {"O-R": ("#555555", "D"), "M-B": ("#2a6fbb", "s"), "M-Bc": ("#2a6fbb", "s"), "P-S": ("#3a9a5b", "^"),
          "P-SP": ("#3a9a5b", "^"), "P-S2": ("#d9650b", "o"), "P-SP2": ("#d9650b", "o")}

# Direct-label offsets (points): the matched-pace pairs sit on top of each other by design.
LABEL_AT = {"M-Bc": (6, -10, "left"), "P-SP2": (6, 5, "left"), "P-SP": (-6, 5, "right"), "P-S2": (6, 5, "left")}


def figure(res: dict, path: Path) -> None:
    """docs/results/fig_followup.png: (a) departure per training seed against gate S2; (b) departure vs dry pace tax."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = res["methods"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axs[0]
    order = [m for m in ("P-S", "P-S2", "P-SP", "P-SP2", "M-B", "M-Bc") if m in M]
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        c, mk = FAMILY[m]
        y = np.array(M[m]["per_seed"])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=36, color=c, marker=mk, edgecolor="white", lw=1, zorder=3)
    if "H5" in res:
        lim = next(iter(res["H5"]["gate"].values()))["limit"]
        ax.axhline(lim, color="#777777", ls="--", lw=1)
        ax.text(0.99, lim, "gate S2: O-R + 5 pp ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="#444444")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{m}\n({len(M[m]['per_seed'])} seeds)" for m in order], fontsize=8)
    ax.set_ylabel("departure rate per training seed")
    ax.set_title("(a) one dot per training seed (test v2, reactive-feasible)", fontsize=9)
    ax = axs[1]
    for m, r in M.items():
        t = r["dry_pace_tax"]
        x = 0.0 if m == "O-R" else (100 * t["estimate"] if t else None)
        if x is None:
            continue
        d = r["departure"]
        xe = [[0], [0]] if not t else [[100 * (t["estimate"] - t["ci95"][0])], [100 * (t["ci95"][1] - t["estimate"])]]
        c, mk = FAMILY[m]
        ax.errorbar(x, d["estimate"], xerr=np.maximum(xe, 0),
                    yerr=np.maximum([[d["estimate"] - d["ci95"][0]], [d["ci95"][1] - d["estimate"]]], 0),
                    fmt=mk, color=c, ms=7, capsize=2, lw=1, mec="white", mew=1)
        dx, dy, ha = LABEL_AT.get(m, (5, 4, "left"))
        ax.annotate(m, (x, d["estimate"]), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7,
                    color="#333333")
    ax.set_xlabel("dry-road pace tax vs O-R [%] (lower = faster)")
    ax.set_ylabel("departure rate (95% CI)")
    ax.set_title("(b) safety against speed", fontsize=9)
    from matplotlib.lines import Line2D
    fams = [("oracle", "O-R"), ("bank MPPI", "M-B"), ("RL, A8 recipe", "P-S"), ("RL, repaired (A9)", "P-SP2")]
    ax.legend([Line2D([], [], color=FAMILY[k][0], marker=FAMILY[k][1], ls="", ms=6) for _, k in fams],
              [n for n, _ in fams], fontsize=7, frameon=False, loc="upper left")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/eval_v2")
    ap.add_argument("--out", default="docs/results/followup")
    a = ap.parse_args()
    top5 = json.loads(Path("ledgers/A9_frozen.json").read_text())["P-SP2_top5_by_validation"]
    res = analyze(Path(a.root), top5)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    Path(a.out + ".md").write_text(markdown(res))
    Path("paper/gen").mkdir(parents=True, exist_ok=True)
    Path("paper/gen/followup.tex").write_text(tex(res))
    Path("paper/gen/abstract_followup.tex").write_text(abstract_tex(res))
    figure(res, Path(a.out).parent / "fig_followup.png")
    print(markdown(res))


if __name__ == "__main__":
    main()
