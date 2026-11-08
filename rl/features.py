"""Per-step inputs of the learned methods. One code path for training (`rl/env.py`) and evaluation
(`rl/policy.py`), so a policy sees exactly the same features in both.

  f_t   (N_FEAT) : the non-privileged observation (sim/observe.py, with noise) scaled to O(1), plus the target speed
                   v_ref and (speed - v_ref). This is everything P-S, P-DR, P-SP, M-L see at step t.
  hist  (W, N_FEAT): the last W feature vectors (oldest first). At handover the history is filled with the
                   handover features (the car has been driving steadily on dry road).
  priv  (N_PRIV) : privileged, teacher only (BASELINES §4): true current per-axle mu, per-axle surface one-hot,
                   normalized DR parameters.
  prev  (N_PREV) : mu ahead of the front axle at +0/10/20/30 m (the P-SP preview teacher only, H3).
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from sim.observe import N_OBS, OBS_NAMES, obs_vec, observe_kernel, NOISE
from sim.params import CONTROL_DT, VehicleParams

_SCALE = {"ax": 5.0, "ay": 5.0, "r": 0.5, "wf": 60.0, "wr": 60.0, "delta": 0.3, "u_s_prev": 1.0, "u_l_prev": 1.0,
          "e_y": 1.0, "e_psi": 0.2, "speed": 20.0, "k0": 0.02, "k10": 0.02, "k20": 0.02, "k30": 0.02, "k40": 0.02,
          "vy": 1.0}
OBS_SCALE = np.array([_SCALE[k] for k in OBS_NAMES], np.float32)
N_FEAT = N_OBS + 2
N_PRIV = 2 + 8 + 6
PREVIEW_D = (0.0, 10.0, 20.0, 30.0)
N_PREV = len(PREVIEW_D)


def blend_weight_t(s: torch.Tensor, s_drop, lt, lp) -> torch.Tensor:
    """Torch version of scenarios.friction.blend_weight (vectorized over cars)."""
    s_end = s_drop + lt + lp
    ramp = lt > 0
    lt_safe = torch.where(ramp, lt, torch.ones_like(lt))
    up_r = ((s - s_drop) / lt_safe).clamp(0, 1)
    down_r = (1 - (s - s_end) / lt_safe).clamp(0, 1)
    up_s = (s >= s_drop).float()
    down_s = (s < s_end).float()
    return torch.minimum(torch.where(ramp, up_r, up_s), torch.where(ramp, down_r, down_s))


def share_stream(device: str) -> None:
    """Make Warp launch on torch's current CUDA stream, so zero-copy tensors and Warp kernels are ordered."""
    if str(device).startswith("cuda"):
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream(wp.device_to_torch(device))), device=device)


class Features:
    """Feature builder bound to a ScenarioSim. Per-car scenario data come from `prof` (n, 8: s_drop, lt, Lp, mu0,
    mu1, surface index, -, -), `v_ref` (n,) and `dr_vec` (n, 6) torch tensors whose rows the caller may overwrite."""

    def __init__(self, sim, hist_len: int, seeds: np.ndarray, noise_scale: float = 1.0, p: VehicleParams = VehicleParams()):
        self.sim, self.n, self.W = sim, sim.n, max(int(hist_len), 1)
        d = sim.device
        share_stream(d)
        self.dev = wp.device_to_torch(d)
        sig = np.zeros(N_OBS, np.float32)
        self.bstd = np.zeros(3, np.float32)
        for k, (s, b) in NOISE.items():
            sig[OBS_NAMES.index(k)] = s * noise_scale
            if OBS_NAMES.index(k) < 3:
                self.bstd[OBS_NAMES.index(k)] = b * noise_scale
        self.sig = obs_vec(*sig.tolist())
        self.seeds = wp.array((np.asarray(seeds, np.int64) % (2 ** 30)).astype(np.int32), dtype=int, device=d)
        self.bias = wp.zeros((self.n, 3), dtype=wp.float32, device=d)
        self.prev = wp.zeros(self.n, dtype=sim.state.dtype, device=d)
        self.out = wp.zeros(self.n, dtype=obs_vec, device=d)
        self.t = 0
        self.lf = p.lf
        self.scale = torch.tensor(OBS_SCALE, device=self.dev)
        self.hist = torch.zeros(self.n, self.W, N_FEAT, device=self.dev)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(np.asarray(seeds).sum()) % (2 ** 31))
        self.prof = torch.zeros(self.n, 8, device=self.dev)
        self.v_ref = torch.ones(self.n, device=self.dev)
        self.dr_vec = torch.zeros(self.n, 6, device=self.dev)

    # --- raw observation ------------------------------------------------------------------------------
    def _observe(self) -> torch.Tensor:
        s = self.sim
        wp.launch(observe_kernel, dim=s.n,
                  inputs=[s.state, self.prev, s.buf, s.frenet, s.hints, s.track_id, s.track_theta, s.track_kappa,
                          s.track_n, s.ds, self.sig, self.bias, self.seeds, self.t, CONTROL_DT, self.out],
                  device=s.device)
        self.t += 1
        return wp.to_torch(self.out).view(self.n, N_OBS).clone()

    def _feat(self, o: torch.Tensor) -> torch.Tensor:
        sp = o[:, OBS_NAMES.index("speed")]
        return torch.cat([o / self.scale, (self.v_ref / 20.0)[:, None], ((sp - self.v_ref) / 5.0)[:, None]], 1)

    def reset_rows(self, mask: torch.Tensor) -> None:
        """Rows whose sim state was just (re)loaded: new IMU bias, history filled with the current features."""
        if not bool(mask.any()):
            return
        b = torch.randn(self.n, 3, device=self.dev, generator=self.gen) * torch.tensor(self.bstd, device=self.dev)
        bias = wp.to_torch(self.bias)
        bias[mask] = b[mask]
        st, pv = wp.to_torch(self.sim.state), wp.to_torch(self.prev)
        pv[mask] = st[mask]
        o = self._observe()
        f = self._feat(o)
        self.hist[mask] = f[mask][:, None, :].expand(-1, self.W, -1)

    def step(self) -> torch.Tensor:
        """After sim.step(): new features appended to the history. Returns f_t (n, N_FEAT)."""
        o = self._observe()
        wp.copy(self.prev, self.sim.state)
        f = self._feat(o)
        self.hist = torch.cat([self.hist[:, 1:], f[:, None, :]], 1)
        return f

    @property
    def f(self) -> torch.Tensor:
        return self.hist[:, -1]

    # --- privileged ---------------------------------------------------------------------------------------
    def priv(self) -> torch.Tensor:
        mu = wp.to_torch(self.sim.axle_mu).view(self.n, 2)
        mu0, mu1 = self.prof[:, 3:4], self.prof[:, 4:5]
        w = ((mu0 - mu) / (mu0 - mu1).clamp(min=1e-3)).clamp(0, 1)
        post = torch.nn.functional.one_hot(self.prof[:, 5].long(), 4).float()
        dry = torch.nn.functional.one_hot(torch.zeros(self.n, dtype=torch.long, device=self.dev), 4).float()
        sf = torch.where(w[:, :1] > 0.5, post, dry)
        sr = torch.where(w[:, 1:] > 0.5, post, dry)
        return torch.cat([mu, sf, sr, self.dr_vec], 1)

    def preview(self) -> torch.Tensor:
        s = wp.to_torch(self.sim.frenet).view(self.n, 2)[:, 0] + self.lf
        d = torch.tensor(PREVIEW_D, device=self.dev)
        sq = s[:, None] + d[None, :]
        pr = self.prof
        w = blend_weight_t(sq, pr[:, 0:1], pr[:, 1:2], pr[:, 2:3])
        return pr[:, 3:4] * (1 - w) + pr[:, 4:5] * w
