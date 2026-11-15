"""Throughput of the Warp simulator (logged in LOG.md; never a claimed result, DESIGN.md §5).

Measures control steps per second (each = 20 physics substeps) for the plain kernel (fixed surfaces) and the
friction-field kernel (per-axle projection every substep), at several batch sizes.

Run:  PYTHONPATH=. uv run python scripts/bench/bench_throughput.py [--device cuda:0]
"""

import argparse
import time

import numpy as np
import warp as wp

from scenarios import sampler as S
from scenarios import track as T
from sim.params import DRY, VehicleParams
from sim.reference import ReferenceSim
from sim.warp_sim import ScenarioSim, WarpSim


def bench(sim, n, steps, rng):
    acts = wp.array(rng.uniform(-1, 1, (n, 2)).astype(np.float32), dtype=wp.float32, device=sim.device)
    sim.step(acts)
    wp.synchronize_device(sim.device)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step(acts)
    wp.synchronize_device(sim.device)
    return n * steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sizes", default="4096,65536,262144")
    a = ap.parse_args()
    p = VehicleParams()
    rng = np.random.default_rng(0)
    x0 = ReferenceSim.steady_state(15.0, p)
    tracks = T.all_tracks()
    pool = S.tuning_split()
    for n in [int(s) for s in a.sizes.split(",")]:
        sim = WarpSim(n, p, a.device)
        sim.reset(x0)
        sim.set_surfaces(DRY, DRY)
        r1 = bench(sim, n, 50, rng)
        fs = ScenarioSim([pool[i % len(pool)] for i in range(n)], p, tracks, a.device)
        fs.reset_at_start()
        r2 = bench(fs, n, 50, rng)
        print(f"n={n:7d}  plain {r1:.3e} ctrl-steps/s ({20 * r1:.2e} substeps/s)   field {r2:.3e} ctrl-steps/s")


if __name__ == "__main__":
    main()
