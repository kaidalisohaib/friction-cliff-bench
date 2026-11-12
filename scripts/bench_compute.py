"""Compute benchmarks with repeats and intervals (engineering measurements, not eval results).

1. MPPI latency per control step, one car (the real-time case), Warp vs C++/CUDA backend, over
   samples K in {256, 1024, 4096, 16384} x horizon H in {50, 100, 150} (BASELINES.md §5), fixed-surface mode
   (O-R / M-N / M-B / M-L) and friction-field mode (O-C).
2. Batched MPPI throughput (64 cars, K = 1024, H = 150: the oracle setting used in labelling and evaluation).
3. Simulator throughput (control steps/s, 20 physics substeps each) at 65,536 and 262,144 cars.

Each cell: `--reps` timed repetitions after a warm-up; reported as median and the 2.5-97.5 percentile range of the
repetitions, with N. Refuses to run while another process computes on the GPU (shared-GPU timings are not
comparable) unless --force. Writes docs/results/compute_benchmarks.{json,md}.

PYTHONPATH=. uv run python scripts/bench_compute.py [--reps 30] [--force]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

from control.mppi import MODE_FIELD, MODE_FIXED, MPPI, MPPIConfig
from control.mppi_cpp import CppMPPI
from labelling.label import git_sha
from scenarios import sampler as S
from scenarios import track as T
from sim.params import DRY, VehicleParams
from sim.reference import ReferenceSim
from sim.warp_sim import ScenarioSim, WarpSim

OUT = Path("docs/results")


def other_gpu_jobs() -> list[str]:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    import os
    return [line for line in q if line and int(line.split(",")[0]) != os.getpid()]


def summarize(ts: np.ndarray) -> dict:
    return {"median": float(np.median(ts)), "lo": float(np.quantile(ts, 0.025)),
            "hi": float(np.quantile(ts, 0.975)), "n": int(len(ts))}


def time_calls(fn, reps: int, dev: str) -> np.ndarray:
    fn()
    wp.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def mppi_cell(cls, E, K, H, mode, reps, dev, pool, tracks, P) -> dict:
    sim = ScenarioSim(pool[:E], P, tracks, dev)
    sim.reset_at_start()
    sim.step(np.zeros((E, 2)))
    m = cls(sim, MPPIConfig(K=K, H=H, mode=mode, pd_sample=True), seed=np.arange(E))
    m.reset()
    m.set_surfaces(sim.axle_sf.numpy(), sim.axle_sr.numpy())
    return summarize(time_calls(m.act, reps, dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = a.device
    busy = other_gpu_jobs()
    if busy and not a.force:
        raise SystemExit(f"GPU busy ({busy}); rerun when idle or pass --force (timings then not comparable)")
    P = VehicleParams()
    tracks = T.all_tracks()
    pool = S.tuning_split()
    res = {"git": git_sha(), "device": wp.get_device(dev).name, "warp": wp.__version__, "host": platform.node(),
           "reps": a.reps, "gpu_shared": bool(busy), "latency_1car": [], "batched": [], "sim": []}
    for mode, mname in ((MODE_FIXED, "fixed"), (MODE_FIELD, "field")):
        for K in (256, 1024, 4096, 16384):
            for H in (50, 100, 150):
                for cls, bname in ((MPPI, "warp"), (CppMPPI, "cpp")):
                    s = mppi_cell(cls, 1, K, H, mode, a.reps, dev, pool, tracks, P)
                    res["latency_1car"].append({"mode": mname, "K": K, "H": H, "backend": bname, **s})
                    print(f"1 car {mname:5s} K={K:5d} H={H:3d} {bname:4s}: median {s['median'] * 1e3:7.2f} ms "
                          f"[{s['lo'] * 1e3:.2f}, {s['hi'] * 1e3:.2f}]", flush=True)
    for mode, mname in ((MODE_FIXED, "fixed"), (MODE_FIELD, "field")):
        for cls, bname in ((MPPI, "warp"), (CppMPPI, "cpp")):
            s = mppi_cell(cls, 64, 1024, 150, mode, a.reps, dev, pool, tracks, P)
            rate = 64 * 1024 * 150 / np.array([s["median"], s["hi"], s["lo"]])
            res["batched"].append({"mode": mname, "E": 64, "K": 1024, "H": 150, "backend": bname, **s,
                                   "rollout_steps_per_s": {"median": rate[0], "lo": rate[1], "hi": rate[2]}})
            print(f"64 cars {mname} {bname}: {s['median'] * 1e3:.1f} ms, {rate[0]:.3e} rollout ctrl-steps/s", flush=True)
    rng = np.random.default_rng(0)
    x0 = ReferenceSim.steady_state(15.0, P)
    for n in (65536, 262144):
        sim = WarpSim(n, P, dev)
        sim.reset(x0)
        sim.set_surfaces(DRY, DRY)
        acts = wp.array(rng.uniform(-1, 1, (n, 2)).astype(np.float32), dtype=wp.float32, device=dev)
        ts = time_calls(lambda: [sim.step(acts) for _ in range(10)], a.reps, dev) / 10
        s = summarize(ts)
        rate = n / np.array([s["median"], s["hi"], s["lo"]])
        res["sim"].append({**s, "cars": n, "ctrl_steps_per_s": {"median": rate[0], "lo": rate[1], "hi": rate[2]}})
        print(f"sim n={n}: {rate[0]:.3e} control steps/s [{rate[1]:.3e}, {rate[2]:.3e}]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compute_benchmarks.json").write_text(json.dumps(res, indent=1))
    L = [f"# Compute benchmarks ({res['device']}, git {res['git']})", "",
         f"Harness output of `scripts/bench_compute.py`: median of N = {a.reps} timed repetitions after a warm-up, "
         "[2.5, 97.5] percentile range in brackets. Engineering measurements, not evaluation results."
         + (" **GPU was shared during this run.**" if busy else " GPU idle otherwise."), "",
         "## MPPI latency per control step, one car [ms]", "",
         "| mode | K | H | Warp | C++/CUDA | speed-up |", "|---|---|---|---|---|---|"]
    lat = {(r["mode"], r["K"], r["H"], r["backend"]): r for r in res["latency_1car"]}
    for mode in ("fixed", "field"):
        for K in (256, 1024, 4096, 16384):
            for H in (50, 100, 150):
                w, c = lat[(mode, K, H, "warp")], lat[(mode, K, H, "cpp")]
                L.append(f"| {mode} | {K} | {H} | {w['median'] * 1e3:.2f} [{w['lo'] * 1e3:.2f}, {w['hi'] * 1e3:.2f}] | "
                         f"{c['median'] * 1e3:.2f} [{c['lo'] * 1e3:.2f}, {c['hi'] * 1e3:.2f}] | "
                         f"{w['median'] / c['median']:.2f}x |")
    L += ["", "## Batched MPPI, 64 cars x K = 1024 x H = 150", "",
          "| mode | backend | ms per control step | rollout control steps/s |", "|---|---|---|---|"]
    for r in res["batched"]:
        q = r["rollout_steps_per_s"]
        L.append(f"| {r['mode']} | {r['backend']} | {r['median'] * 1e3:.1f} [{r['lo'] * 1e3:.1f}, {r['hi'] * 1e3:.1f}] | "
                 f"{q['median']:.3e} [{q['lo']:.3e}, {q['hi']:.3e}] |")
    L += ["", "## Simulator (Warp), fixed surfaces", "", "| cars | control steps/s |", "|---|---|"]
    for r in res["sim"]:
        q = r["ctrl_steps_per_s"]
        L.append(f"| {r['cars']} | {q['median']:.3e} [{q['lo']:.3e}, {q['hi']:.3e}] |")
    (OUT / "compute_benchmarks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
