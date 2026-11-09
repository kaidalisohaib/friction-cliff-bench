"""M-L friction/delay estimator (BASELINES.md §3): a GRU on the last W feature vectors (rl/features.py, the same
non-privileged inputs as the policies) predicting per-axle mu, per-axle surface class and extra actuator delay.

Trained offline, supervised, on the training library (DR on) with a behavior driver: the shared path-follower law
(curvature feedforward + PD, the same gains as the prefix driver) on the true state, with a random per-episode
speed multiplier U[0.85, 1.15] and Ornstein–Uhlenbeck action noise, so the data include sliding near and
beyond the limit. The estimator's training counts toward M-L's tuning budget (prereg V2).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

from rl.env import FrictionEnv
from rl.features import N_FEAT
from scenarios import track as T
from sim.params import G, VehicleParams


class Estimator(nn.Module):
    def __init__(self, hist_len: int, hidden: int = 64):
        super().__init__()
        self.hist_len = hist_len
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2 + 8 + 3))

    def forward(self, hist):
        _, h = self.gru(hist)
        o = self.head(h[-1])
        return torch.sigmoid(o[:, :2]) * 1.1, o[:, 2:6], o[:, 6:10], o[:, 10:13]

    def predict(self, hist):
        mu, lf, lr, ld = self(hist)
        return mu, lf.argmax(1), lr.argmax(1), ld.argmax(1)


class BehaviorDriver:
    """Vectorized path follower on the true state (data collection only)."""

    def __init__(self, env: FrictionEnv, seed: int = 0):
        self.env, self.n = env, env.n
        s = env.sim
        self.th = wp.to_torch(s.track_theta)
        self.kap = wp.to_torch(s.track_kappa)
        self.npts = wp.to_torch(s.track_n)
        self.g = torch.Generator(device=env.dev)
        self.g.manual_seed(seed)
        self.mult = torch.ones(self.n, device=env.dev)
        self.ou = torch.zeros(self.n, 2, device=env.dev)
        self.p = VehicleParams()
        self.resample(torch.ones(self.n, dtype=torch.bool, device=env.dev))

    def resample(self, mask):
        u = torch.rand(self.n, device=self.env.dev, generator=self.g)
        self.mult = torch.where(mask, 0.85 + 0.30 * u, self.mult)
        self.ou[mask] = 0

    def act(self):
        a = self.law()
        self.ou = 0.9 * self.ou + torch.randn(self.n, 2, device=self.env.dev, generator=self.g) * torch.tensor(
            [0.02, 0.08], device=self.env.dev)
        return (a + self.ou).clamp(-1, 1)

    def law(self):
        """Noise-free path-follower action on the true state (with this episode's speed multiplier)."""
        s, env, p = self.env.sim, self.env, self.p
        x = wp.to_torch(s.state).view(self.n, 13)
        fr = wp.to_torch(s.frenet).view(self.n, 2)
        tid = wp.to_torch(s.track_id).long()
        i = wp.to_torch(s.hints)[:, 0].long()
        npts = self.npts[tid].long()
        k = self.kap[tid, i]
        e_psi = torch.atan2(torch.sin(x[:, 2] - self.th[tid, i]), torch.cos(x[:, 2] - self.th[tid, i]))
        delta = p.L * k - 0.15 * fr[:, 1] - 0.8 * e_psi - 0.1 * (x[:, 5] - x[:, 3] * k)
        ia = torch.minimum(i + (x[:, 3].clamp(min=1.0) * 1.0 / T.DS).long(), npts - 1)
        ka = self.kap[tid, ia].abs().clamp(min=1e-9)
        mu0 = env.feat.prof[:, 3]
        vt = torch.minimum(env.feat.v_ref * self.mult, torch.sqrt(0.7 * mu0 * G / ka) * self.mult)
        return torch.stack([(delta / p.delta_max).clamp(-1, 1), (0.3 * (vt - x[:, 3])).clamp(-1, 1)], 1)


def train_estimator(lib: dict, hist_len: int, steps: int = 3000, n_envs: int = 4096, lr: float = 1e-3,
                    hidden: int = 64, seed: int = 0, device: str = "cuda:0", log=print) -> tuple[Estimator, dict]:
    torch.manual_seed(seed)
    env = FrictionEnv(lib, n_envs, hist_len, device, seed=seed + 2000)
    drv = BehaviorDriver(env, seed)
    est = Estimator(hist_len, hidden).to(env.dev)
    opt = torch.optim.Adam(est.parameters(), lr=lr)
    t0 = time.time()
    hist = {"it": [], "loss": [], "mu_mae": [], "acc_f": [], "acc_d": []}
    for it in range(steps):
        with torch.no_grad():
            _, done, _, info = env.step(drv.act())
            if info:
                drv.resample(done)
            pv = env.feat.priv()
            h = env.feat.hist.clone()
        mu_t = pv[:, :2]
        cf_t, cr_t = pv[:, 2:6].argmax(1), pv[:, 6:10].argmax(1)
        d_t = (env.feat.dr_vec[:, -1] + 1).round().long().clamp(0, 2)
        mu, lf, lr_, ld = est(h)
        loss = F.mse_loss(mu, mu_t) / 0.01 + F.cross_entropy(lf, cf_t) + F.cross_entropy(lr_, cr_t) + F.cross_entropy(ld, d_t)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(est.parameters(), 1.0)
        opt.step()
        if it % 200 == 0 or it == steps - 1:
            with torch.no_grad():
                mae = (mu - mu_t).abs().mean().item()
                af = (lf.argmax(1) == cf_t).float().mean().item()
                ad = (ld.argmax(1) == d_t).float().mean().item()
            for kk, vv in zip(hist, (it, float(loss), mae, af, ad)):
                hist[kk].append(vv)
            log(f"[estimator W={hist_len}] it {it}/{steps} loss {float(loss):.3f} mu-MAE {mae:.3f} "
                f"surf-acc {af:.3f} delay-acc {ad:.3f} {time.time() - t0:.0f}s")
    hist["wall_s"] = time.time() - t0
    return est, hist
