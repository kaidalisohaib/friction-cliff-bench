"""Non-privileged observations (BASELINES.md §2 item 4), shared by every adaptive method (M-B, M-L, P-S, P-SP, P-DR).

Per car and control step, a float32 vector (layout OBS_NAMES):
  IMU  ax, ay [m/s^2], r [rad/s]   with white noise + a per-episode constant bias
  wheel speeds wf, wr [rad/s]      white noise
  steering angle delta [rad]       white noise
  last commanded action (u_s, u_l) exact (the controller knows what it sent)
  path-relative e_y [m], e_psi [rad], speed [m/s] (localization), white noise
  geometry preview: centreline curvature at s + {0, 10, 20, 30, 40} m (map knowledge, no friction)
  lateral velocity vy [m/s]        white noise 0.1 m/s (a sideslip estimate, as from GNSS/INS fusion; D24:
                                   MPPI methods start rollouts from the true state, so the policies get vy too)
Friction never appears. ay is the body-frame lateral acceleration over the last control step,
(vy - vy_prev)/dt + r vx; ax is the simulator's inertial longitudinal acceleration.

Noise levels (engineering choice for a consumer-grade automotive MEMS IMU + production wheel-speed sensors +
lane-level localization; documented in PARAMETERS.md §7): see NOISE. The prereg V4 recovery band (0.05 rad/s,
0.15 m) is 10x / 3x these noise levels and is computed on the true state anyway.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from sim.params import CONTROL_DT
from sim.warp_sim import ScenarioSim, vec13

OBS_NAMES = ["ax", "ay", "r", "wf", "wr", "delta", "u_s_prev", "u_l_prev", "e_y", "e_psi", "speed",
             "k0", "k10", "k20", "k30", "k40", "vy"]
N_OBS = len(OBS_NAMES)
PREVIEW = (0.0, 10.0, 20.0, 30.0, 40.0)
# (white-noise std, bias std) per noisy channel
NOISE = {"ax": (0.05, 0.05), "ay": (0.05, 0.05), "r": (0.005, 0.002), "wf": (0.1, 0.0), "wr": (0.1, 0.0),
         "delta": (0.001, 0.0), "e_y": (0.05, 0.0), "e_psi": (0.005, 0.0), "speed": (0.05, 0.0),
         "vy": (0.1, 0.0)}

obs_vec = wp.types.vector(length=N_OBS, dtype=wp.float32)


@wp.kernel
def observe_kernel(state: wp.array(dtype=vec13), prev: wp.array(dtype=vec13), buf: wp.array3d(dtype=wp.float32),
                   frenet: wp.array2d(dtype=wp.float32), hints: wp.array2d(dtype=int), track_id: wp.array(dtype=int),
                   track_th: wp.array2d(dtype=wp.float32), track_k: wp.array2d(dtype=wp.float32),
                   track_n: wp.array(dtype=int), ds: wp.float32, sig: obs_vec, bias: wp.array2d(dtype=wp.float32),
                   seeds: wp.array(dtype=int), t: int, dt: wp.float32, out: wp.array(dtype=obs_vec)):
    i = wp.tid()
    x = state[i]
    xp = prev[i]
    tid = track_id[i]
    npts = track_n[tid]
    idx = hints[i, 0]
    st = wp.rand_init(seeds[i] + t * 7919, 0)
    o = obs_vec()
    o[0] = x[12] + bias[i, 0]
    o[1] = (x[4] - xp[4]) / dt + x[5] * x[3] + bias[i, 1]
    o[2] = x[5] + bias[i, 2]
    o[3] = x[6]
    o[4] = x[7]
    o[5] = x[10]
    o[6] = buf[i, 0, 0]
    o[7] = buf[i, 0, 1]
    o[8] = frenet[i, 1]
    th = track_th[tid, idx]
    o[9] = wp.atan2(wp.sin(x[2] - th), wp.cos(x[2] - th))
    o[10] = wp.sqrt(x[3] * x[3] + x[4] * x[4])
    for j in range(5):
        jj = wp.clamp(idx + int(wp.float32(j) * wp.float32(10.0) / ds), 0, npts - 1)
        o[11 + j] = track_k[tid, jj]
    o[16] = x[4]
    for c in range(N_OBS):
        if sig[c] > wp.float32(0.0):
            o[c] = o[c] + sig[c] * wp.randn(st)
    out[i] = o


class Observer:
    """Observation generator for a ScenarioSim. Call `reset()` at episode start and `observe()` after each step."""

    def __init__(self, sim: ScenarioSim, seeds: np.ndarray, noise: dict | None = None, noise_scale: float = 1.0):
        self.sim, d = sim, sim.device
        nz = NOISE if noise is None else noise
        sig = np.zeros(N_OBS, np.float32)
        bstd = np.zeros(N_OBS, np.float32)
        for k, (s, b) in nz.items():
            sig[OBS_NAMES.index(k)] = s * noise_scale
            bstd[OBS_NAMES.index(k)] = b * noise_scale
        self.sig = obs_vec(*sig.tolist())
        self.bstd = bstd[:3]
        self.seeds_np = np.asarray(seeds, np.int64) % (2 ** 30)
        self.seeds = wp.array(self.seeds_np.astype(np.int32), dtype=int, device=d)
        self.bias = wp.zeros((sim.n, 3), dtype=wp.float32, device=d)
        self.prev = wp.zeros(sim.n, dtype=vec13, device=d)
        self.out = wp.zeros(sim.n, dtype=obs_vec, device=d)
        self.t = 0

    def reset(self) -> None:
        """New per-episode IMU biases (deterministic in the per-car seed) and the 'previous state' for ay."""
        b = np.stack([np.random.default_rng(int(s)).normal(0.0, 1.0, 3) for s in self.seeds_np]) * self.bstd
        self.bias.assign(wp.array(b.astype(np.float32), dtype=wp.float32, device=self.sim.device))
        wp.copy(self.prev, self.sim.state)
        self.t = 0

    def observe(self) -> wp.array:
        s = self.sim
        wp.launch(observe_kernel, dim=s.n,
                  inputs=[s.state, self.prev, s.buf, s.frenet, s.hints, s.track_id, s.track_theta, s.track_kappa,
                          s.track_n, s.ds, self.sig, self.bias, self.seeds, self.t, CONTROL_DT, self.out],
                  device=s.device)
        wp.copy(self.prev, s.state)
        self.t += 1
        return self.out
