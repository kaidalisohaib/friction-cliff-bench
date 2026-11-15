"""Write paper/gen/compute.tex from docs/results/compute_benchmarks.json (output of scripts/bench/bench_compute.py).

Every number comes from the harness JSON: median of N timed repetitions and the [2.5, 97.5] percentile range.
Usage: python scripts/report/compute_tex.py [--json docs/results/compute_benchmarks.json] [--out paper/gen/compute.tex]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ms(r: dict) -> str:
    return f"{r['median'] * 1e3:.1f} [{r['lo'] * 1e3:.1f}, {r['hi'] * 1e3:.1f}]"


def sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def render(d: dict, H: int = 150) -> str:
    lat = {(r["mode"], r["K"], r["H"], r["backend"]): r for r in d["latency_1car"]}
    Ks = sorted({r["K"] for r in d["latency_1car"]})
    rows = []
    for mode in ("fixed", "field"):
        for K in Ks:
            w, c = lat[(mode, K, H, "warp")], lat[(mode, K, H, "cpp")]
            rows.append(f"{mode} & {K:,} & {ms(w)} & {ms(c)} & {w['median'] / c['median']:.2f} \\\\")
    b = {(r["mode"], r["backend"]): r for r in d["batched"]}
    E, Kb, Hb = d["batched"][0]["E"], d["batched"][0]["K"], d["batched"][0]["H"]
    bat = "; ".join(
        f"{mode} mode {ms(b[(mode, 'warp')])}\\,ms (Warp) vs {ms(b[(mode, 'cpp')])}\\,ms (C++/CUDA)"
        for mode in ("fixed", "field"))
    sim = ", ".join(f"${sci(r['ctrl_steps_per_s']['median'])}$ control steps/s with {r['cars']:,} cars"
                    for r in d["sim"])
    shared = "the GPU was shared" if d["gpu_shared"] else "the GPU was otherwise idle"
    return "\n".join([
        f"Table~\\ref{{tab:compute}} gives the latency of one MPPI control step for one car (median of N = {d['reps']} "
        f"timed repetitions after a warm-up, [2.5, 97.5] percentile range; {d['device']}, Warp {d['warp']}, {shared}). "
        f"The latency hardly grows with $K$ because each sample's {H}-step rollout runs sequentially in one thread, "
        "so the horizon, not the sample count, sets the floor. "
        f"Batched over {E} cars ($K = {Kb:,}$, $H = {Hb}$), one control step takes {bat}. "
        f"The simulator alone runs at {sim} (fixed surfaces). These are engineering measurements, not evaluation results.",
        "",
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{MPPI latency per control step, one car, $H = {H}$ [ms]; median [2.5, 97.5 percentile], "
        f"N = {d['reps']}. The control period is 20\\,ms.}}\\label{{tab:compute}}",
        "\\begin{tabular}{lrccc}\\toprule",
        "mode & $K$ & Warp & C++/CUDA & speed-up \\\\\\midrule",
        *rows,
        "\\bottomrule\\end{tabular}\\end{table}",
        "",
    ])


def policy_sentence(p: dict) -> str:
    """Learned-policy latency (docs/results/compute_policy.json, scripts/bench/bench_policy.py)."""
    c = {(r["device"], r["threads"], r["cars"]): r for r in p["cells"]}
    one = c[("cpu", 1, 1)]
    gpu = c[("cuda", None, 64)]
    shared = "GPU shared" if p["gpu_shared"] else "GPU otherwise idle"
    ms = lambda r: f"{r['median'] * 1e3:.3f} [{r['lo'] * 1e3:.3f}, {r['hi'] * 1e3:.3f}]"  # sub-millisecond
    return (f"For comparison, one forward pass of the learned student (GRU over $W = {p['W']}$ feature vectors and an "
            f"MLP actor; feature computation excluded) takes {ms(one)}\\,ms for one car on a single CPU thread and "
            f"{ms(gpu)}\\,ms for 64 cars on the GPU (median of N = {p['reps']}, {shared}).\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/results/compute_benchmarks.json")
    ap.add_argument("--policy", default="docs/results/compute_policy.json", help="scripts/bench/bench_policy.py output")
    ap.add_argument("--out", default="paper/gen/compute.tex")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = render(json.loads(Path(a.json).read_text()))
    if Path(a.policy).exists():
        txt += policy_sentence(json.loads(Path(a.policy).read_text()))
    out.write_text(txt)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
