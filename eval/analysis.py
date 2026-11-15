"""Confirmatory analysis (prereg H1-H3 as specified in V6 and amendment A5; EVAL_PROTOCOL §3-4).

Reads the run logs of eval/runner.py and writes one JSON with every number plus Markdown tables. Figures:
scripts/report/make_figures.py. Nothing here chooses anything after looking at test outcomes, except "the better of
M-B / M-L" (lower pooled reactive-feasible departure), which is the conservative choice for the H1 falsifier
(prereg A5.3).

Conventions:
- A metric matrix is (seeds, scenarios), aligned to the set's scenario order; methods are paired by scenario.
- Bootstrap = hierarchical (seeds per method independently, scenarios jointly), 10,000 resamples, seed 0.
- p-values are bootstrap p-values in the direction of the tested claim (1.0 if the estimate points the other way).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import stats as St

N_BOOT = 10_000
MARGIN = 0.05  # H1 margin (5 pp)
DRY_LOSS = 0.02  # H3: "losing < 2% dry progress"


# --- loading ----------------------------------------------------------------------------------------------------
def load_set(root: Path, set_name: str, methods, seeds=(0, 1, 2, 3, 4)) -> dict:
    """{method: {"sids": [...], "recs": [[rec per scenario] per seed]}} for the methods whose logs exist."""
    out = {}
    for m in methods:
        runs = []
        for s in seeds:
            p = root / set_name / f"{m}_s{s}.jsonl"
            if p.exists():
                runs.append([json.loads(line) for line in open(p)])
        if runs:
            sids = [r["sid"] for r in runs[0]]
            assert all([r["sid"] for r in rr] == sids for rr in runs), (set_name, m)
            out[m] = {"sids": sids, "recs": runs}
    return out


def mat(d: dict, m: str, key: str, cols=None) -> np.ndarray:
    """(seeds, scenarios) matrix of a record field (bools -> 0/1, None -> nan); cols = scenario index subset."""
    a = np.array([[np.nan if r[key] is None else float(r[key]) for r in rr] for rr in d[m]["recs"]])
    return a if cols is None else a[:, cols]


# --- bootstrap helpers -------------------------------------------------------------------------------------------
def boot_means(mats: dict[str, np.ndarray], n_boot=N_BOOT, seed=0) -> dict[str, np.ndarray]:
    """Joint hierarchical bootstrap of the mean of several paired (seeds, scenarios) matrices: one scenario
    resample per replicate shared by all methods, seeds resampled per method. Returns {name: (n_boot,)}."""
    rng = np.random.default_rng(seed)
    n_sc = next(iter(mats.values())).shape[1]
    assert all(v.shape[1] == n_sc for v in mats.values())
    out = {k: np.empty(n_boot) for k in mats}
    for t in range(n_boot):
        js = rng.integers(0, n_sc, n_sc)
        for k, v in mats.items():
            ii = rng.integers(0, v.shape[0], v.shape[0])
            out[k][t] = np.nanmean(v[ii][:, js])
    return out


def p_dir(dist: np.ndarray, positive: bool = True) -> float:
    """Two-sided bootstrap p for 'difference = 0', counted only if the estimate points in the claimed direction."""
    med = float(np.median(dist))
    if (med > 0) != positive or med == 0:
        return 1.0
    return St.p_two_sided(dist)


def summary(dist: np.ndarray, point: float) -> dict:
    lo, hi = St.ci(dist)
    return {"estimate": float(point), "ci95": [lo, hi]}


# --- tables ------------------------------------------------------------------------------------------------------
def class_rates(main: dict, labels: dict, methods) -> dict:
    """Departure / loss-of-control rate per feasibility class and method (pooled runs), with Wilson 95% over runs
    and a hierarchical-bootstrap 95% CI over scenarios and seeds."""
    out = {}
    sids = next(iter(main.values()))["sids"]
    classes = ["reactive-feasible", "preview-required", "certified-infeasible", "undetermined", "adaptation-required"]
    for c in classes:
        if c == "adaptation-required":
            cols = [j for j, s in enumerate(sids) if labels[s].get("adaptation_required")]
        else:
            cols = [j for j, s in enumerate(sids) if labels[s]["label"] == c]
        out[c] = {"n_scenarios": len(cols)}
        if not cols:
            continue
        for m in methods:
            if m not in main:
                continue
            row = {}
            for key in ("departed", "loss_of_control", "success"):
                a = mat(main, m, key, cols)
                k, n = int(np.nansum(a)), int(np.isfinite(a).sum())
                bd = boot_means({"x": a}, n_boot=2000)["x"]
                row[key] = {"rate": k / n, "k": k, "n_runs": n, "wilson95": list(St.wilson(k, n)),
                            "boot95": list(St.ci(bd))}
            out[c][m] = row
    return out


# --- hypotheses --------------------------------------------------------------------------------------------------
def better_mppi(main: dict, rf_cols) -> str:
    cand = [m for m in ("M-B", "M-L") if m in main]
    return min(cand, key=lambda m: float(np.nanmean(mat(main, m, "departed", rf_cols))))


def h1(main: dict, cols, best: str) -> dict:
    a, b = mat(main, best, "departed", cols), mat(main, "P-S", "departed", cols)
    d = St.hier_bootstrap_diff(a, b, N_BOOT, np.random.default_rng(1))
    point = float(np.nanmean(a) - np.nanmean(b))
    p_fals = float(np.mean(d <= MARGIN))  # H0 of the falsifier: (best MPPI - P-S) <= +5 pp
    lo, hi = St.ci(d)
    return {"better_mppi": best, "n_scenarios": len(cols), "diff_best_minus_PS": summary(d, point),
            "falsified": bool(lo > MARGIN), "p_falsifier": p_fals, "tost": St.tost(d, MARGIN),
            "dep_best": float(np.nanmean(a)), "dep_PS": float(np.nanmean(b))}


def h2_axis(levels: list[dict], best: str, n_boot=N_BOOT, seed=2) -> dict:
    """levels: [main restricted to the H2 subset (level 0), level 1, 2, 3] each {method: data}. Regret of a method
    = its departure minus O-R's on the same scenarios; delta = regret(best MPPI) - regret(P-S)."""
    rng = np.random.default_rng(seed)
    n_sc = len(levels[0]["O-R"]["sids"])
    dep = [{m: mat(L, m, "departed") for m in ("O-R", best, "P-S", "P-DR") if m in L} for L in levels]
    point_delta, point_reg = [], {m: [] for m in (best, "P-S", "P-DR")}
    for D in dep:
        r = {m: np.nanmean(D[m]) - np.nanmean(D["O-R"]) for m in point_reg if m in D}
        for m, v in r.items():
            point_reg[m].append(float(v))
        point_delta.append(float(r[best] - r["P-S"]))
    x = np.arange(len(levels)) - (len(levels) - 1) / 2
    slope_pt = float(np.dot(x, point_delta) / np.dot(x, x))
    slopes, last = np.empty(n_boot), np.empty(n_boot)
    for t in range(n_boot):
        js = rng.integers(0, n_sc, n_sc)
        dl = []
        for D in dep:
            mean = {}
            for m in ("O-R", best, "P-S"):
                ii = rng.integers(0, D[m].shape[0], D[m].shape[0])
                mean[m] = np.nanmean(D[m][ii][:, js])
            dl.append((mean[best] - mean["O-R"]) - (mean["P-S"] - mean["O-R"]))
        dl = np.array(dl)
        slopes[t] = np.dot(x, dl) / np.dot(x, x)
        last[t] = dl[-1]
    pa, pb = p_dir(slopes, True), p_dir(last, True)
    a_ok = St.ci(slopes)[0] > 0
    b_ok = St.ci(last)[0] > 0
    return {"regret_by_level": point_reg, "delta_by_level": point_delta, "slope": summary(slopes, slope_pt),
            "delta_last": summary(last, point_delta[-1]), "pass_a": bool(a_ok), "pass_b": bool(b_ok),
            "pass": bool(a_ok and b_ok), "p_a": pa, "p_b": pb, "p_axis": max(pa, pb), "n_scenarios": n_sc}


def h3(main: dict, cols, dry: dict) -> dict:
    a, b = mat(main, "P-S", "departed", cols), mat(main, "P-SP", "departed", cols)
    d = St.hier_bootstrap_diff(a, b, N_BOOT, np.random.default_rng(3))  # > 0: P-SP departs less
    point = float(np.nanmean(a) - np.nanmean(b))
    out = {"n_scenarios": len(cols), "dep_PS": float(np.nanmean(a)), "dep_PSP": float(np.nanmean(b)),
           "diff_PS_minus_PSP": summary(d, point)}
    pa = p_dir(d, True)
    if dry and "P-S" in dry and "P-SP" in dry:
        ps, psp = mat(dry, "P-S", "progress"), mat(dry, "P-SP", "progress")
        bm = boot_means({"ps": ps, "psp": psp}, seed=4)
        loss = 1.0 - bm["psp"] / bm["ps"]
        loss_pt = float(1.0 - np.nanmean(psp) / np.nanmean(ps))
        pb = float(np.mean(loss >= DRY_LOSS))  # H0: P-SP loses >= 2% dry progress
        out["dry_progress_loss_PSP_vs_PS"] = summary(loss, loss_pt)
        out["p_dry"] = pb
        fals = St.ci(d)[0] > 0 and St.ci(loss)[1] < DRY_LOSS
    else:
        pb, fals = 1.0, False
    out.update({"falsified": bool(fals), "p_falsifier": max(pa, pb)})
    return out


def secondary(main: dict, cols, dry: dict, methods) -> dict:
    """Regret vs O-R (paired) on reactive-feasible scenarios, and the dry pace tax (progress deficit vs O-R on the
    dry set). Bootstrap 95% CIs."""
    out = {}
    for m in methods:
        if m not in main or m == "O-R":
            continue
        row = {}
        for key in ("max_abs_ey", "time_out_of_lane", "progress", "loss_of_control"):
            a, b = mat(main, m, key, cols), mat(main, "O-R", key, cols)
            d = St.hier_bootstrap_diff(a, b, 2000, np.random.default_rng(5))
            row[f"regret_{key}"] = summary(d, float(np.nanmean(a) - np.nanmean(b)))
        if dry and m in dry and "O-R" in dry:
            pm, po = mat(dry, m, "progress"), mat(dry, "O-R", "progress")
            bm = boot_means({"m": pm, "o": po}, n_boot=2000, seed=6)
            tax = 1.0 - bm["m"] / bm["o"]
            row["dry_pace_tax"] = summary(tax, float(1.0 - np.nanmean(pm) / np.nanmean(po)))
            dd = mat(dry, m, "departed")
            row["dry_departure"] = {"rate": float(np.nanmean(dd)), "n_runs": int(np.isfinite(dd).sum())}
        out[m] = row
    return out


ARRIVAL_BINS = (0.0, 0.8, 0.9, 1.0, np.inf)  # arrival speed / v_eff (prereg A7)


def caution(main: dict, cols, v_eff: np.ndarray, methods) -> dict:
    """Exploratory E6 (prereg A7): is a method safer because it arrives slower? Per method on the given scenarios:
    the mean arrival-speed ratio (front axle at the drop, / v_eff) with a bootstrap 95% CI, and the departure
    rate within arrival-ratio bins (pooled runs, Wilson 95%). Runs that never reach the drop have no ratio."""
    out = {}
    for m in methods:
        if m not in main:
            continue
        av = mat(main, m, "arrival_v", cols) / v_eff[None, :]
        dep = mat(main, m, "departed", cols)
        bm = boot_means({"r": av}, n_boot=2000, seed=7)["r"]
        bins = []
        for lo, hi in zip(ARRIVAL_BINS[:-1], ARRIVAL_BINS[1:]):
            sel = np.isfinite(av) & (av >= lo) & (av < hi)
            k, n = int(np.nansum(dep[sel])), int(sel.sum())
            bins.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "n_runs": n, "departed": k,
                         "rate": k / n if n else None, "wilson95": list(St.wilson(k, n)) if n else None})
        out[m] = {"arrival_ratio": summary(bm, float(np.nanmean(av))), "bins": bins,
                  "n_no_arrival": int((~np.isfinite(av)).sum())}
    return out


def e7(main: dict, cols, dry: dict, h1_res: dict | None = None) -> dict:
    """Exploratory E7 (prereg A7): speed-matched bank MPPI (M-B-c) against the student P-S, paired.

    - departure difference M-B-c - P-S on the given (reactive-feasible) scenarios, hierarchical bootstrap;
    - dry pace tax difference (tax(M-B-c) - tax(P-S), both against O-R on the dry set), joint bootstrap.
    Interpretation fixed by A7: if M-B-c's departure is not higher than P-S's (95% CI of the difference includes 0
    or favours M-B-c), the student's H1 advantage is attributed to caution, not adaptation. The rule only has
    something to explain if the student departs less than the better adaptive MPPI (point estimate of H1 > 0)."""
    a, b = mat(main, "M-B-c", "departed", cols), mat(main, "P-S", "departed", cols)
    d = St.hier_bootstrap_diff(a, b, N_BOOT, np.random.default_rng(8))
    point = float(np.nanmean(a) - np.nanmean(b))
    lo, hi = St.ci(d)
    out = {"n_scenarios": len(cols), "dep_MBc": float(np.nanmean(a)), "dep_PS": float(np.nanmean(b)),
           "diff_MBc_minus_PS": summary(d, point), "mbc_not_higher": bool(lo <= 0)}
    adv = h1_res is not None and h1_res["diff_best_minus_PS"]["estimate"] > 0
    out["student_advantage_in_H1"] = bool(adv)
    out["attributed_to_caution"] = bool(adv and lo <= 0)
    if dry and all(m in dry for m in ("M-B-c", "P-S", "O-R")):
        pm, ps, po = (mat(dry, m, "progress") for m in ("M-B-c", "P-S", "O-R"))
        bm = boot_means({"m": pm, "s": ps, "o": po}, n_boot=2000, seed=9)
        dt = (1.0 - bm["m"] / bm["o"]) - (1.0 - bm["s"] / bm["o"])
        out["dry_pace_tax_diff_MBc_minus_PS"] = summary(
            dt, float((np.nanmean(ps) - np.nanmean(pm)) / np.nanmean(po)))
    return out


def recovery(root: Path, methods, v_eff: dict | None = None, seeds=(0, 1, 2, 3, 4), wheelbase: float = 2.7,
             n_boot: int = 2000) -> dict:
    """Exploratory recovery latency (V4, A5.9) from the trace pass, per method against O-R (same seed index):
    the fraction of runs that recover (enter and hold the O-R band) and the median latency of the recovered runs,
    each with a hierarchical-bootstrap 95% CI; censored runs are counted as not recovered. E4: Spearman correlation
    of the recovered runs' latency with the axle transit time L / v_eff."""
    from scipy.stats import spearmanr

    from eval.latency import recovery_latency
    tr = root / "trace"
    out = {}
    for m in methods:
        if m == "O-R":
            continue
        L, C, sids = [], [], None
        for s in seeds:
            fm, fo = tr / f"{m}_s{s}.npz", tr / f"O-R_s{s}.npz"
            if not (fm.exists() and fo.exists()):
                continue
            dm, do = dict(np.load(fm)), dict(np.load(fo))
            lat, cens = recovery_latency(dm, do)
            L.append(lat)
            C.append(cens | ~np.isfinite(lat))
            sids = list(dm["sid"])
        if not L:
            continue
        lat, cens = np.array(L), np.array(C)
        rec = (~cens).astype(float)
        lat_r = np.where(cens, np.nan, lat)
        rng = np.random.default_rng(10)
        bf, bm, bmean = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
        for t in range(n_boot):
            ii = rng.integers(0, lat.shape[0], lat.shape[0])
            js = rng.integers(0, lat.shape[1], lat.shape[1])
            bf[t] = rec[ii][:, js].mean()
            sub = lat_r[ii][:, js]
            bm[t] = np.nanmedian(sub) if np.isfinite(sub).any() else np.nan
            bmean[t] = np.nanmean(sub) if np.isfinite(sub).any() else np.nan
        row = {"n_scenarios": int(lat.shape[1]), "n_runs": int(lat.size),
               "recovered_frac": summary(bf, float(rec.mean())),
               "median_latency_recovered_s": (summary(bm[np.isfinite(bm)], float(np.nanmedian(lat_r)))
                                              if np.isfinite(lat_r).any() else None),
               "mean_latency_recovered_s": (summary(bmean[np.isfinite(bmean)], float(np.nanmean(lat_r)))
                                            if np.isfinite(lat_r).any() else None)}
        if v_eff is not None:
            transit = np.array([wheelbase / v_eff[s] for s in sids])[None, :].repeat(lat.shape[0], 0)
            ok = ~cens
            if ok.sum() >= 10:
                rho = spearmanr(lat[ok], transit[ok])
                row["e4_spearman_latency_vs_transit"] = {"rho": float(rho.statistic), "p": float(rho.pvalue),
                                                         "n_runs": int(ok.sum())}
        out[m] = row
    return out


def analyze(root: Path, labels: dict, methods, h2_names: list[str] | None = None,
            v_eff: dict | None = None) -> dict:
    main = load_set(root, "main", methods)
    sids = main["O-R"]["sids"]
    rf_cols = [j for j, s in enumerate(sids) if labels[s]["label"] == "reactive-feasible"]
    ar_cols = [j for j, s in enumerate(sids) if labels[s]["label"] == "reactive-feasible"
               and labels[s].get("adaptation_required")]
    dry = load_set(root, "dry", methods)
    res = {"n_main": len(sids), "classes": class_rates(main, labels, methods)}
    best = better_mppi(main, rf_cols)
    hyp = {}
    if "P-S" in main:
        hyp["H1"] = h1(main, rf_cols, best)
        if ar_cols:
            res["H1_adaptation_required"] = h1(main, ar_cols, best)  # pre-declared secondary (A4.5)
    if "P-SP" in main and "P-S" in main:
        hyp["H3"] = h3(main, rf_cols, dry)
    if h2_names:
        axes = {}
        h2m = ("O-R", best, "P-S", "P-DR")
        for axis in sorted({n.split("-")[1] for n in h2_names}):
            names = sorted([n for n in h2_names if n.split("-")[1] == axis], key=lambda n: float(n.split("-")[2]))
            lv = [load_set(root, n, h2m) for n in names]
            if not all("O-R" in L and best in L and "P-S" in L for L in lv):
                continue
            sub_sids = lv[0]["O-R"]["sids"]  # the H2 subset minus the axis' A6 exclusions
            assert all(L[m]["sids"] == sub_sids for L in lv for m in L), f"H2 {axis}: levels differ in scenarios"
            pos = {s: j for j, s in enumerate(sids)}
            cols = [pos[s] for s in sub_sids]
            lv0 = {m: {"sids": sub_sids, "recs": [[rr[c] for c in cols] for rr in main[m]["recs"]]}
                   for m in h2m if m in main}
            axes[axis] = h2_axis([lv0] + lv, best)
        if len(axes) == 3:
            ps = sorted(a["p_axis"] for a in axes.values())
            hyp["H2"] = {"axes": axes, "n_axes_pass": sum(a["pass"] for a in axes.values()),
                         "supported": sum(a["pass"] for a in axes.values()) >= 2, "p": min(1.0, 2 * ps[1])}
        else:
            res["H2_partial"] = axes
    if hyp:
        pv = {k: (v["p"] if k == "H2" else v["p_falsifier"]) for k, v in hyp.items()}
        res["holm"] = {"p": pv, "rejected": St.holm(pv)}
    res["hypotheses"] = hyp
    res["secondary_rf"] = secondary(main, rf_cols, dry, methods)
    # per-seed reactive-feasible departure (all seeds reported, never best-of; A8: S2 failed on seed instability)
    res["per_seed_rf_departure"] = {m: [float(x) for x in np.nanmean(mat(main, m, "departed", rf_cols), axis=1)]
                                    for m in methods if m in main}
    if dry:  # diagnostic: the same per-seed rate on the dry set (no drop): lane keeping without any friction change
        res["per_seed_dry_departure"] = {m: [float(x) for x in np.nanmean(mat(dry, m, "departed"), axis=1)]
                                         for m in methods if m in dry}
    if v_eff is not None:  # exploratory E6 (A7)
        ve = np.array([v_eff[sids[j]] for j in rf_cols])
        res["exploratory_caution_rf"] = caution(main, rf_cols, ve, methods)
    if "M-B-c" in main and "P-S" in main:  # exploratory E7 (A7)
        res["exploratory_e7"] = e7(main, rf_cols, dry, hyp.get("H1"))
    if (root / "trace").exists():  # exploratory recovery latency (V4, A5.9) and E4
        rl = recovery(root, methods, v_eff)
        if rl:
            res["exploratory_recovery"] = rl
    return res


def markdown(res: dict, title: str) -> str:
    L = [f"# {title}", "", "Harness output (`eval/analysis.py`). Rates pooled over seeds; Wilson 95% over runs and "
         "hierarchical-bootstrap 95% over scenarios and seeds.", ""]
    for c, row in res["classes"].items():
        L += [f"## {c} (N = {row['n_scenarios']} scenarios)", ""]
        if row["n_scenarios"] == 0:
            continue
        L += ["| method | departure | Wilson 95% | bootstrap 95% | loss of control | runs |", "|---|---|---|---|---|---|"]
        for m, r in row.items():
            if m == "n_scenarios":
                continue
            d, lc = r["departed"], r["loss_of_control"]
            L.append(f"| {m} | {d['rate']:.3f} | [{d['wilson95'][0]:.3f}, {d['wilson95'][1]:.3f}] | "
                     f"[{d['boot95'][0]:.3f}, {d['boot95'][1]:.3f}] | {lc['rate']:.3f} | {d['n_runs']} |")
        L.append("")
    hyp = res.get("hypotheses", {})
    if "H1" in hyp:
        h = hyp["H1"]
        e = h["diff_best_minus_PS"]
        L += ["## H1", "", f"Better adaptive MPPI: {h['better_mppi']}. Departure difference ({h['better_mppi']} − P-S) on "
              f"{h['n_scenarios']} reactive-feasible scenarios: {e['estimate']:+.4f}, 95% CI [{e['ci95'][0]:+.4f}, "
              f"{e['ci95'][1]:+.4f}]. Falsifier (CI entirely above +0.05): **{'met' if h['falsified'] else 'not met'}**; "
              f"one-sided p = {h['p_falsifier']:.4f}. TOST ±5 pp: {'equivalent' if h['tost']['equivalent'] else 'not shown equivalent'} "
              f"(90% CI [{h['tost']['ci90'][0]:+.4f}, {h['tost']['ci90'][1]:+.4f}]).", ""]
    if "H1_adaptation_required" in res:
        h = res["H1_adaptation_required"]
        e = h["diff_best_minus_PS"]
        L += ["### H1 on the adaptation-required stratum (pre-declared secondary, A4.5)", "",
              f"N = {h['n_scenarios']}: {e['estimate']:+.4f}, 95% CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]; "
              f"TOST: {'equivalent' if h['tost']['equivalent'] else 'not shown equivalent'}.", ""]
    if "H2" in hyp:
        h = hyp["H2"]
        L += ["## H2", "", f"Axes passing: {h['n_axes_pass']}/3 → **{'supported' if h['supported'] else 'not supported'}** "
              f"(p = {h['p']:.4f}).", "", "| axis | delta by level 0..3 | slope [95% CI] | delta at max [95% CI] | pass |",
              "|---|---|---|---|---|"]
        for ax, a in h["axes"].items():
            L.append(f"| {ax} | {', '.join(f'{v:+.3f}' for v in a['delta_by_level'])} | {a['slope']['estimate']:+.4f} "
                     f"[{a['slope']['ci95'][0]:+.4f}, {a['slope']['ci95'][1]:+.4f}] | {a['delta_last']['estimate']:+.3f} "
                     f"[{a['delta_last']['ci95'][0]:+.3f}, {a['delta_last']['ci95'][1]:+.3f}] | {a['pass']} |")
        L.append("")
    if "H3" in hyp:
        h = hyp["H3"]
        e = h["diff_PS_minus_PSP"]
        L += ["## H3", "", f"Departure P-S − P-SP: {e['estimate']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]."]
        if "dry_progress_loss_PSP_vs_PS" in h:
            q = h["dry_progress_loss_PSP_vs_PS"]
            L.append(f"P-SP dry progress loss vs P-S: {q['estimate']:+.4f} [{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}].")
        L += [f"Falsifier met: **{h['falsified']}**.", ""]
    if "holm" in res:
        L += ["## Holm–Bonferroni over H1–H3 (α = 0.05)", "", "| hypothesis | p | rejected |", "|---|---|---|"]
        for k, p in res["holm"]["p"].items():
            L.append(f"| {k} | {p:.4f} | {res['holm']['rejected'][k]} |")
        L.append("")
    if res.get("secondary_rf"):
        L += ["## Secondary (reactive-feasible; regret = method − O-R, paired)", "",
              "| method | Δ max abs e_y [m] | Δ time out of lane [s] | Δ progress [m] | dry pace tax |", "|---|---|---|---|---|"]
        for m, r in res["secondary_rf"].items():
            def f(k, nd=3):
                if k not in r:
                    return "—"
                return f"{r[k]['estimate']:+.{nd}f} [{r[k]['ci95'][0]:+.{nd}f}, {r[k]['ci95'][1]:+.{nd}f}]"
            L.append(f"| {m} | {f('regret_max_abs_ey')} | {f('regret_time_out_of_lane')} | {f('regret_progress', 1)} | "
                     f"{f('dry_pace_tax', 4)} |")
    if res.get("per_seed_rf_departure"):
        L += ["", "## Per-seed departure (reactive-feasible; seed 0..4)", "", "| method | per seed |", "|---|---|"]
        for m, v in res["per_seed_rf_departure"].items():
            L.append(f"| {m} | {', '.join(f'{x:.3f}' for x in v)} |")
    if res.get("per_seed_dry_departure"):
        L += ["", "## Per-seed departure on the dry set (no friction drop; diagnostic)", "", "| method | per seed |",
              "|---|---|"]
        for m, v in res["per_seed_dry_departure"].items():
            L.append(f"| {m} | {', '.join(f'{x:.3f}' for x in v)} |")
    if res.get("exploratory_caution_rf"):
        L += ["", "## Exploratory E6 (A7): arrival speed at the drop (reactive-feasible)", "",
              "| method | arrival / v_eff [95% CI] | departure by arrival bin <0.8 / 0.8–0.9 / 0.9–1.0 / ≥1.0 (k/n) |",
              "|---|---|---|"]
        for m, c in res["exploratory_caution_rf"].items():
            a = c["arrival_ratio"]
            bins = " / ".join(f"{b['departed']}/{b['n_runs']}" for b in c["bins"])
            L.append(f"| {m} | {a['estimate']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {bins} |")
    if res.get("exploratory_e7"):
        e = res["exploratory_e7"]
        d = e["diff_MBc_minus_PS"]
        L += ["", "## Exploratory E7 (A7): speed-matched bank MPPI M-B-c vs P-S (reactive-feasible, paired)", "",
              f"- N = {e['n_scenarios']} scenarios; departure M-B-c {e['dep_MBc']:.3f}, P-S {e['dep_PS']:.3f}; "
              f"difference {d['estimate']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]."]
        if "dry_pace_tax_diff_MBc_minus_PS" in e:
            t = e["dry_pace_tax_diff_MBc_minus_PS"]
            L.append(f"- Dry pace tax difference (M-B-c - P-S): {t['estimate']:+.4f} "
                     f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}].")
        L.append(f"- Student advantage in H1 (point estimate): {e['student_advantage_in_H1']}; M-B-c not higher "
                 f"than P-S: {e['mbc_not_higher']}; A7 rule -> attributed to caution: {e['attributed_to_caution']}.")
    if res.get("exploratory_recovery"):
        L += ["", "## Exploratory: recovery latency vs O-R (V4; H2 subset, nominal truth)", "",
              "| method | N runs | recovered fraction [95% CI] | median / mean latency of recovered runs [s, 95% CI] | "
              "E4 Spearman(latency, L/v_eff) |", "|---|---|---|---|---|"]
        for m, r in res["exploratory_recovery"].items():
            f, md, mn = r["recovered_frac"], r["median_latency_recovered_s"], r.get("mean_latency_recovered_s")
            e4 = r.get("e4_spearman_latency_vs_transit")
            L.append(f"| {m} | {r['n_runs']} | {f['estimate']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}] | "
                     + (f"{md['estimate']:.2f} [{md['ci95'][0]:.2f}, {md['ci95'][1]:.2f}] / "
                        f"{mn['estimate']:.2f} [{mn['ci95'][0]:.2f}, {mn['ci95'][1]:.2f}]" if md else "--") + " | "
                     + (f"{e4['rho']:+.3f} (p = {e4['p']:.3g}, n = {e4['n_runs']})" if e4 else "--") + " |")
    return "\n".join(L) + "\n"
