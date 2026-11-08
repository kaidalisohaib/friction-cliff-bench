"""MPPI iteration cost on the GPU (logged, not a claimed result). Rollout control-steps per second for
E cars x K samples x H horizon, fixed-surface (O-R / M-N) and friction-field (O-C) modes.

Run:  PYTHONPATH=. uv run python scripts/bench_mppi.py
"""

import time

import numpy as np
import warp as wp

from control.mppi import MODE_FIELD, MODE_FIXED, MPPI, MPPIConfig
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams
from sim.warp_sim import ScenarioSim


def main(dev="cuda:0"):
    tracks = T.all_tracks()
    pool = S.tuning_split()
    P = VehicleParams()
    for E, K, H in [(8, 4096, 150), (32, 4096, 150), (8, 16384, 150), (32, 16384, 150), (64, 1024, 100)]:
        sim = ScenarioSim(pool[:E], P, tracks, dev)
        sim.reset_at_start()
        sim.step(np.zeros((E, 2)))
        for mode, name in ((MODE_FIXED, "fixed"), (MODE_FIELD, "field")):
            m = MPPI(sim, MPPIConfig(K=K, H=H, mode=mode))
            m.reset()
            m.set_surfaces(sim.axle_sf.numpy(), sim.axle_sr.numpy())
            m.act()
            wp.synchronize_device(dev)
            n_it = 5
            t0 = time.perf_counter()
            for _ in range(n_it):
                m.act()
            wp.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / n_it
            print(f"E={E:3d} K={K:6d} H={H:4d} {name}: {dt * 1e3:8.1f} ms/iter  "
                  f"{E * K * H / dt:.2e} rollout ctrl-steps/s  ({E * K * H * 2 * 4 / 1e6:.0f} MB samples)")


if __name__ == "__main__":
    main()
