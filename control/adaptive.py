"""Adaptive MPPI baselines (BASELINES.md §1, §3): M-B (model bank, LLA-MPC style) and M-L (learned estimator).

Both are the same Warp MPPI as the oracles (`control/mppi.py`, MODE_FIXED) with the same settings budget. They
differ only in where the rollout model's friction (and extra actuator delay) comes from:
  O-R  true current per-axle surfaces;  M-N  nominal dry;
  M-B  argmin over a bank of candidate models of the one-step prediction error over the last W control steps
       (LLA-MPC Eq. 7; prereg V8: argmin, not softmin);
  M-L  a GRU estimator on the last W feature vectors (rl/features.py), trained offline under DR (control/estimator.py).
Information (D24): rollouts start from the true vehicle state (perfect state estimation, as in LLA-MPC and the
oracles); friction and delay must be inferred from the noisy observation history, exactly as for the policies,
which observe the same channels including a noisy lateral velocity.

M-B bank (45 models): one surface for both axles x friction level x extra delay {0, 1, 2} steps (0–40 ms, D14):
  dry mu {0.85, 0.925, 1.0}, wet {0.5, 0.6, 0.7, 0.8}, snow {0.2, 0.25, 0.3, 0.35}, ice {0.08, 0.1, 0.12, 0.15}
  (each with that surface's Magic Formula shape), spanning the DR/test friction ranges on a grid finer than
  the gaps between surfaces. Size justification: LLA-MPC uses ~20k bicycle variants because it also identifies
  geometric/inertial parameters; here only (surface, mu, delay) are unknown, so a 45-model grid covers the
  hypothesis space; finer grids are an exploratory option (`bank_models(mu_points=...)`).
Prediction error: the bank model steps the *true* state x_t one control step (20 x 1 ms substeps) with the action
actually applied under the candidate delay, and is compared with the next *noisy observation* on yaw rate,
wheel speeds, accelerations, speed and lateral velocity, each divided by a scale ~ 2-4x its noise std.
Before any pair is available (right after handover), the nominal dry model is used.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from control.controllers import MU_NOMINAL_DRY, run_seed
from control.mppi import MPPIConfig
from control.mppi_cpp import make_mppi
from rl.features import Features, share_stream
from sim.observe import N_OBS, OBS_NAMES
from sim.params import CONTROL_DT, N_SUBSTEPS, SURFACES, VehicleParams
from sim.warp_sim import Params, make_params, substep, surface_vec, vec8, vec13

SURF_NAMES = ["dry", "wet", "snow", "ice"]
BANK_MU = {"dry": (0.85, 0.925, 1.0), "wet": (0.5, 0.6, 0.7, 0.8), "snow": (0.2, 0.25, 0.3, 0.35),
           "ice": (0.08, 0.1, 0.12, 0.15)}
BANK_DELAYS = (0, 1, 2)
# scored channels and their scales (obs index, scale)
_CH = [("r", 0.02), ("wf", 0.3), ("wr", 0.3), ("ax", 0.3), ("ay", 0.3), ("speed", 0.15), ("vy", 0.3)]
SCORE_IDX = [OBS_NAMES.index(c) for c, _ in _CH]
SCORE_SCALE = [s for _, s in _CH]


def bank_models(mu_points: dict = BANK_MU, delays=BANK_DELAYS):
    """List of (surface name, mu, extra delay) and the matching (M, 8) surface vectors."""
    models, vecs = [], []
    for sname in SURF_NAMES:
        for mu in mu_points[sname]:
            for d in delays:
                models.append((sname, float(mu), int(d)))
                vecs.append(surface_vec(SURFACES[sname], mu))
    return models, np.stack(vecs).astype(np.float32)


@wp.kernel
def bank_score_kernel(params: wp.array(dtype=Params), h_state: wp.array2d(dtype=vec13),
                      h_buf: wp.array4d(dtype=wp.float32), h_obs: wp.array3d(dtype=wp.float32),
                      n_valid: wp.array(dtype=int), cand_sf: wp.array(dtype=vec8), cand_delay: wp.array(dtype=int),
                      ch: wp.array(dtype=int), scale: wp.array(dtype=wp.float32), nsub: int, dt: wp.float32,
                      cdt: wp.float32, score: wp.array2d(dtype=wp.float32)):
    """h_state[e, j]: state after step j (j = 0 oldest); h_buf[e, j]: action buffer after step j; h_obs[e, j]:
    observation after step j. Pairs (j, j + 1) for the last n_valid[e] pairs (W pairs max)."""
    e, c = wp.tid()
    p = params[e]
    d = p.delay_steps + cand_delay[c]
    sf = cand_sf[c]
    sr = cand_sf[c]
    sf[4] = sf[4] * p.B_scale_f
    sr[4] = sr[4] * p.B_scale_r
    W1 = h_state.shape[1]
    tot = wp.float32(0.0)
    for jj in range(W1 - 1):
        j = W1 - 2 - jj
        if jj < n_valid[e]:
            x0 = h_state[e, j]
            us = h_buf[e, j + 1, d, 0]
            ul = h_buf[e, j + 1, d, 1]
            x = x0
            for _k in range(nsub):
                x = substep(x, us, ul, p, sf, sr, dt)
            for q in range(ch.shape[0]):
                k = ch[q]
                pred = wp.float32(0.0)
                if k == 0:
                    pred = x[12]
                elif k == 1:
                    pred = (x[4] - x0[4]) / cdt + x[5] * x[3]
                elif k == 2:
                    pred = x[5]
                elif k == 3:
                    pred = x[6]
                elif k == 4:
                    pred = x[7]
                elif k == 10:
                    pred = wp.sqrt(x[3] * x[3] + x[4] * x[4])
                elif k == 16:
                    pred = x[4]
                err = (pred - h_obs[e, j + 1, k]) / scale[q]
                tot = tot + err * err
    score[e, c] = tot


@wp.kernel
def set_model_kernel(base: wp.array(dtype=Params), extra: wp.array(dtype=int), out: wp.array(dtype=Params)):
    i = wp.tid()
    p = base[i]
    p.delay_steps = base[i].delay_steps + extra[i]
    out[i] = p


class _AdaptiveBase:
    """Shared plumbing: an MPPI on the sim's cars, a feature/observation pipeline, model updates per step."""

    def __init__(self, sim, scs, cfg: MPPIConfig, seed: int, snap: dict, method: str, hist_len: int,
                 model_params: VehicleParams = VehicleParams(), noise_scale: float = 1.0):
        self.sim, self.n = sim, sim.n
        share_stream(sim.device)
        self.mppi = make_mppi(sim, cfg, seed=np.array([run_seed(sc.sid, method, seed) for sc in scs]),
                         model_params=model_params)
        self.mppi.reset(snap["buf"][:, 0, :])
        self.base_params = wp.clone(self.mppi.mparams)
        self.feat = Features(sim, hist_len, seeds=np.array([run_seed(sc.sid, method + "-obs", seed) for sc in scs]),
                             noise_scale=noise_scale)
        self.dev = self.feat.dev
        from scenarios import sampler as S
        self.feat.v_ref[:] = torch.tensor([S.v_eff(sc) for sc in scs], dtype=torch.float32, device=self.dev)
        self.surf_base = torch.tensor(np.stack([surface_vec(SURFACES[k]) for k in SURF_NAMES]), device=self.dev)
        dry = surface_vec(SURFACES["dry"], MU_NOMINAL_DRY)
        self.set_model(np.broadcast_to(dry, (self.n, 8)), np.broadcast_to(dry, (self.n, 8)), np.zeros(self.n, np.int32))
        self.sel = None  # last selection (diagnostics)

    def set_model(self, sf, sr, extra_delay) -> None:
        sf_t = torch.as_tensor(np.array(sf), dtype=torch.float32, device=self.dev) if not torch.is_tensor(sf) else sf
        sr_t = torch.as_tensor(np.array(sr), dtype=torch.float32, device=self.dev) if not torch.is_tensor(sr) else sr
        ex = (torch.as_tensor(np.array(extra_delay), device=self.dev) if not torch.is_tensor(extra_delay)
              else extra_delay).int().contiguous()
        wp.to_torch(self.mppi.sf_fix).view(self.n, 8)[:] = sf_t
        wp.to_torch(self.mppi.sr_fix).view(self.n, 8)[:] = sr_t
        self._ex = ex
        wp.launch(set_model_kernel, dim=self.n, inputs=[self.base_params, wp.from_torch(ex), self.mppi.mparams],
                  device=self.sim.device)

    def __call__(self, s, k):
        if k == 0:
            self.feat.reset_rows(torch.ones(self.n, dtype=torch.bool, device=self.dev))
            self.on_reset()
        else:
            self.feat.step()
            self.on_step()
        self.update_model()
        return self.mppi.act()

    def on_reset(self):
        pass

    def on_step(self):
        pass

    def update_model(self):
        raise NotImplementedError


class BankMPPI(_AdaptiveBase):
    """M-B."""

    def __init__(self, sim, scs, cfg: MPPIConfig, seed: int, snap: dict, window: int, noise_scale: float = 1.0,
                 mu_points: dict = BANK_MU):
        self.W = int(window)
        super().__init__(sim, scs, cfg, seed, snap, "M-B", 1, noise_scale=noise_scale)
        self.models, vecs = bank_models(mu_points)
        d = sim.device
        self.cand_sf = wp.array(vecs, dtype=vec8, device=d)
        self.cand_delay = wp.array(np.array([m[2] for m in self.models], np.int32), dtype=int, device=d)
        self.cand_vec = torch.tensor(vecs, device=self.dev)
        self.cand_ex = torch.tensor([m[2] for m in self.models], device=self.dev, dtype=torch.int32)
        self.ch = wp.array(np.array(SCORE_IDX, np.int32), dtype=int, device=d)
        self.scale = wp.array(np.array(SCORE_SCALE, np.float32), dtype=wp.float32, device=d)
        n, W1 = self.n, self.W + 1
        self.h_state = torch.zeros(n, W1, 13, device=self.dev)
        self.h_buf = torch.zeros(n, W1, sim.buf.shape[1], 2, device=self.dev)
        self.h_obs = torch.zeros(n, W1, N_OBS, device=self.dev)
        self.n_valid = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.score = wp.zeros((n, len(self.models)), dtype=wp.float32, device=d)

    def _push(self):
        st = wp.to_torch(self.sim.state).view(self.n, 13)
        bf = wp.to_torch(self.sim.buf)
        ob = wp.to_torch(self.feat.out).view(self.n, N_OBS)
        self.h_state = torch.cat([self.h_state[:, 1:], st[:, None].clone()], 1)
        self.h_buf = torch.cat([self.h_buf[:, 1:], bf[:, None].clone()], 1)
        self.h_obs = torch.cat([self.h_obs[:, 1:], ob[:, None].clone()], 1)

    def on_reset(self):
        self.n_valid.zero_()
        self._push()

    def on_step(self):
        self._push()
        self.n_valid = (self.n_valid + 1).clamp(max=self.W)

    def update_model(self):
        if int(self.n_valid.max()) == 0:
            return
        wp.launch(bank_score_kernel, dim=(self.n, len(self.models)),
                  inputs=[self.base_params, wp.from_torch(self.h_state.contiguous(), dtype=vec13),
                          wp.from_torch(self.h_buf.contiguous()), wp.from_torch(self.h_obs.contiguous()),
                          wp.from_torch(self.n_valid), self.cand_sf, self.cand_delay, self.ch, self.scale, N_SUBSTEPS,
                          CONTROL_DT / N_SUBSTEPS, CONTROL_DT, self.score], device=self.sim.device)
        sc = wp.to_torch(self.score)
        best = sc.argmin(1)
        valid = self.n_valid > 0
        sf = wp.to_torch(self.mppi.sf_fix).view(self.n, 8).clone()
        new = self.cand_vec[best]
        sf[valid] = new[valid]
        ex = self._ex.clone()
        ex[valid] = self.cand_ex[best][valid]
        self.sel = best
        self.set_model(sf, sf, ex)


class LearnedMPPI(_AdaptiveBase):
    """M-L: per-axle mu, surface class and extra delay from the GRU estimator (control/estimator.py)."""

    def __init__(self, sim, scs, cfg: MPPIConfig, seed: int, snap: dict, estimator, noise_scale: float = 1.0):
        self.est = estimator
        super().__init__(sim, scs, cfg, seed, snap, "M-L", estimator.hist_len, noise_scale=noise_scale)
        self.est.to(self.dev).eval()

    def update_model(self):
        with torch.no_grad():
            mu, cf, cr, dl = self.est.predict(self.feat.hist)
        sf = self.surf_base[cf].clone()
        sr = self.surf_base[cr].clone()
        sf[:, 0], sr[:, 0] = mu[:, 0], mu[:, 1]
        self.sel = (mu, cf, cr, dl)
        self.set_model(sf, sr, dl)
