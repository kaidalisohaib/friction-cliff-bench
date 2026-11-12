"""Fill the README results block from harness output (no hand-typed numbers).

Replaces the text between `<!-- RESULTS:BEGIN -->` and `<!-- RESULTS:END -->` in README.md with a summary built
from docs/results/results.json (eval/analysis.py via scripts/make_figures.py), the S2 record and the compute
benchmark JSON. Every rate comes with its N and a 95% CI.

Usage: PYTHONPATH=. uv run python scripts/readme_results.py [--results docs/results/results.json] [--readme README.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
ORDER = ["O-R", "O-C", "M-N", "M-B", "M-L", "M-B-c", "P-T", "P-TP", "P-DR", "P-S", "P-SP"]
DESC = {"O-R": "oracle, knows current friction", "O-C": "oracle, sees the road ahead",
        "M-N": "MPPI, assumes dry road", "M-B": "adaptive MPPI, model bank", "M-L": "adaptive MPPI, learned estimator",
        "M-B-c": "M-B at the student's arrival speed (exploratory, A7)", "P-T": "PPO teacher (sees friction)",
        "P-TP": "PPO teacher with road preview", "P-DR": "domain-randomized PPO", "P-S": "RMA student (history only)",
        "P-SP": "student of the preview teacher"}


def pp(e: dict) -> str:
    return f"{100 * e['estimate']:+.1f} pp [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"


def render(res: dict, s2: dict | None, comp: dict | None) -> str:
    L = []
    rf = res["classes"]["reactive-feasible"]
    hyp = res.get("hypotheses", {})
    n_seeds = round(rf["O-R"]["departed"]["n_runs"] / rf["n_scenarios"])
    L.append(f"Confirmatory test split: {res['n_main']:,} scenarios, {rf['n_scenarios']:,} of them reactive-feasible, "
             f"{n_seeds} seeds per method. Intervals are 95% hierarchical-bootstrap CIs over scenarios and seeds. "
             "Full tables: [`docs/results/results.md`](docs/results/results.md).")
    L.append("")
    if s2:
        pt = s2["P-T"]
        L.append(f"> **Caveat (gate S2 failed, amendment A8).** The final PPO teacher departed on {pt['departure']:.3f} "
                 f"of N = {pt['n_runs']:,} reactive-feasible *tuning* runs (Wilson 95% [{pt['wilson'][0]:.3f}, "
                 f"{pt['wilson'][1]:.3f}]) vs {s2['O-R']['departure']:.3f} for the oracle. The cause is "
                 "seed-to-seed instability, so per-seed results are reported and no seed was dropped.")
        L.append("")
    L.append("| method | | departure on reactive-feasible scenarios [95% CI] |")
    L.append("|---|---|---|")
    for m in ORDER:
        if m in rf:
            d = rf[m]["departed"]
            L.append(f"| {m} | {DESC[m]} | {d['rate']:.3f} [{d['boot95'][0]:.3f}, {d['boot95'][1]:.3f}] |")
    L.append("")
    if "H1" in hyp:
        h = hyp["H1"]
        L.append(f"- **H1** (prediction: the RMA student does *not* depart less than the better adaptive MPPI by more than 5 pp): departure("
                 f"{h['better_mppi']}) − departure(P-S) = {pp(h['diff_best_minus_PS'])}; "
                 f"{'falsified' if h['falsified'] else 'not falsified'}. TOST ±5 pp: "
                 f"{'equivalent' if h['tost']['equivalent'] else 'equivalence not shown'}.")
    if "H2" in hyp:
        h = hyp["H2"]
        L.append(f"- **H2** (prediction: under model mismatch the adaptive MPPI's regret grows faster than the student's, and the curves cross): "
                 f"{h['n_axes_pass']} of 3 axes pass; {'supported' if h['supported'] else 'not supported'}.")
    if "H3" in hyp:
        h = hyp["H3"]
        L.append(f"- **H3** (prediction: the student of the current-friction teacher matches or beats the preview-teacher student): departure(P-S) − "
                 f"departure(P-SP) = {pp(h['diff_PS_minus_PSP'])}; {'falsified' if h['falsified'] else 'not falsified'}.")
    if "holm" in res:
        L.append("- Holm over {" + ", ".join(res["holm"]["p"]) + "}: " + ", ".join(
            f"{k} {'rejected' if v else 'not rejected'}" for k, v in res["holm"]["rejected"].items()) + ".")
    e = res.get("exploratory_e7")
    if e:
        verdict = ("no student advantage to explain" if not e["student_advantage_in_H1"] else
                   "the student's advantage is attributed to **caution** (arriving slower), not adaptation"
                   if e["attributed_to_caution"] else
                   "the student's advantage is **not** explained by its lower arrival speed alone")
        L.append(f"- **Adaptation or caution?** (exploratory, rule fixed in A7 before the test run) The bank MPPI "
                 f"driven at the student's arrival speed departs {e['dep_MBc']:.3f} vs P-S {e['dep_PS']:.3f}; "
                 f"difference {pp(e['diff_MBc_minus_PS'])}, so {verdict}.")
    if comp:
        lat = {(r["mode"], r["K"], r["H"], r["backend"]): r for r in comp["latency_1car"]}
        w, c = lat[("fixed", 1024, 150, "warp")], lat[("fixed", 1024, 150, "cpp")]
        b = {(r["mode"], r["backend"]): r for r in comp["batched"]}
        bw, bc = b[("fixed", "warp")], b[("fixed", "cpp")]
        L.append(f"- **Compute** ({comp['device']}, median of N = {comp['reps']}): one MPPI control step "
                 f"(K = 1,024 samples, H = 150) takes {c['median'] * 1e3:.1f} ms in C++/CUDA vs "
                 f"{w['median'] * 1e3:.1f} ms in Warp. For 64 cars at once: {bc['median'] * 1e3:.1f} vs "
                 f"{bw['median'] * 1e3:.1f} ms. Details: "
                 "[`docs/results/compute_benchmarks.md`](docs/results/compute_benchmarks.md).")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/results/results.json")
    ap.add_argument("--s2", default="ledgers/s2_gate_A8.json")
    ap.add_argument("--compute", default="docs/results/compute_benchmarks.json")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    s2 = json.loads(Path(a.s2).read_text()) if Path(a.s2).exists() else None
    comp = json.loads(Path(a.compute).read_text()) if Path(a.compute).exists() else None
    readme = Path(a.readme)
    txt = readme.read_text()
    i, j = txt.index(BEGIN) + len(BEGIN), txt.index(END)
    readme.write_text(txt[:i] + "\n" + render(res, s2, comp) + "\n" + txt[j:])
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
