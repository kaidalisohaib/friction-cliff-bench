"""Batched training environment for the learned methods (BASELINES.md §4). Everything stays on the GPU.

Episode = the evaluation episode (control/episode.py): start at a handover state from the training library,
drive until patch exit + 3 s / drop + 8 s (endless patch) / termination / drop + 20 s cap. Terminations use the
exact footprint metrics (metrics_kernel), like evaluation.

Reward = -(per-step objective of DESIGN §4 / prereg V1), with the exact footprint-corner lane violation and
-C_fail on termination (severe departure, |beta| > 30 deg, spin), times `reward_scale` (a PPO hyperparameter;
scaling is not shaping). No other shaping terms. Done environments reset immediately to a random library entry.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from control.episode import BETA_LOC, T_AFTER_DROP_INF, T_AFTER_EXIT, T_CAP_AFTER_DROP, metrics_kernel
from control.mppi import CostWeights
from rl.features import Features
from rl.library import params_from_row
from scenarios import sampler as S
from scenarios import track as T
from sim.params import BUF_LEN, CONTROL_DT, VehicleParams
from sim.warp_sim import Params, ScenarioSim, make_params, vec8, vec13

T_CAP_EP = 60.0  # [s] hard cap on an episode from handover (never reached by the rules above in practice)


@wp.kernel
def load_rows_kernel(mask: wp.array(dtype=int), idx: wp.array(dtype=int),
                     l_state: wp.array(dtype=vec13), l_buf: wp.array3d(dtype=wp.float32),
                     l_hints: wp.array2d(dtype=int), l_frenet: wp.array2d(dtype=wp.float32),
                     l_sf: wp.array(dtype=vec8), l_sr: wp.array(dtype=vec8), l_prof: wp.array(dtype=vec8),
                     l_tid: wp.array(dtype=int), l_params: wp.array(dtype=Params),
                     state: wp.array(dtype=vec13), buf: wp.array3d(dtype=wp.float32), hints: wp.array2d(dtype=int),
                     frenet: wp.array2d(dtype=wp.float32), sf: wp.array(dtype=vec8), sr: wp.array(dtype=vec8),
                     prof: wp.array(dtype=vec8), tid: wp.array(dtype=int), params: wp.array(dtype=Params)):
    i = wp.tid()
    if mask[i] == 0:
        return
    j = idx[i]
    state[i] = l_state[j]
    for b in range(buf.shape[1]):
        buf[i, b, 0] = l_buf[j, b, 0]
        buf[i, b, 1] = l_buf[j, b, 1]
    for c in range(3):
        hints[i, c] = l_hints[j, c]
    frenet[i, 0] = l_frenet[j, 0]
    frenet[i, 1] = l_frenet[j, 1]
    sf[i] = l_sf[j]
    sr[i] = l_sr[j]
    prof[i] = l_prof[j]
    tid[i] = l_tid[j]
    params[i] = l_params[j]


class FrictionEnv:
    def __init__(self, lib: dict, n_envs: int, hist_len: int, device: str = "cuda:0", seed: int = 0,
                 reward_scale: float = 0.01, noise_scale: float = 1.0, sequential: bool = False):
        """sequential=True: env i runs library entries i, i + n, i + 2n, ... once each (evaluation); otherwise
        random entries with replacement (training)."""
        self.n, self.device = n_envs, device
        self.dev = wp.device_to_torch(device)
        self.N = len(lib["track_id"])
        self.rng = np.random.default_rng(seed)
        self.sequential = sequential
        self.reward_scale = reward_scale
        self.cw = CostWeights()
        tracks = T.all_tracks()
        dummy = S.Scenario("d", 0, 10, "straight", 0.0, "straight", "wet", 10.0, 0.0, 10.0, 0.9, 0.6, 0, T.LEAD_IN)
        self.sim = ScenarioSim([dummy] * n_envs, VehicleParams(), tracks, device)
        self.p = VehicleParams()
        L = {}
        L["state"] = wp.array(lib["state"].astype(np.float32), dtype=vec13, device=device)
        L["buf"] = wp.array(lib["buf"].astype(np.float32), dtype=wp.float32, device=device)
        L["hints"] = wp.array(lib["hints"].astype(np.int32), dtype=int, device=device)
        L["frenet"] = wp.array(lib["frenet"].astype(np.float32), dtype=wp.float32, device=device)
        L["sf"] = wp.array(lib["axle_sf"].astype(np.float32), dtype=vec8, device=device)
        L["sr"] = wp.array(lib["axle_sr"].astype(np.float32), dtype=vec8, device=device)
        L["prof"] = wp.array(lib["prof"].astype(np.float32), dtype=vec8, device=device)
        L["tid"] = wp.array(lib["track_id"].astype(np.int32), dtype=int, device=device)
        L["params"] = wp.array([make_params(params_from_row(r)) for r in lib["dr_params"]], dtype=Params, device=device)
        self.L = L
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.dev)
        self.l_prof, self.l_vref, self.l_dr = t(lib["prof"]), t(lib["v_ref"]), t(lib["dr_vec"])
        self.l_delay = torch.as_tensor(lib["dr_params"][:, 5].astype(np.int64), device=self.dev)
        self.feat = Features(self.sim, hist_len, seeds=np.arange(n_envs) + 7 * seed + 1, noise_scale=noise_scale)
        self.mbuf = wp.zeros((n_envs, 7), dtype=wp.float32, device=device)
        z = lambda: torch.zeros(n_envs, device=self.dev)
        self.k, self.t_drop, self.t_exit, self.s_prev = z(), z(), z(), z()
        self.min_clear, self.loc, self.ret = z(), z(), z()
        self.u_prev = torch.zeros(n_envs, 2, device=self.dev)
        self.lib_idx = torch.zeros(n_envs, dtype=torch.long, device=self.dev)
        self.next_seq = n_envs
        self.active = torch.ones(n_envs, dtype=torch.bool, device=self.dev)
        self._reset(torch.ones(n_envs, dtype=torch.bool, device=self.dev), first=True)

    # ------------------------------------------------------------------------------------------------------
    def _pick(self, m: int, first: bool) -> torch.Tensor:
        if not self.sequential:
            return torch.as_tensor(self.rng.integers(0, self.N, m), device=self.dev)
        if first:
            return torch.arange(m, device=self.dev)
        ids = torch.arange(self.next_seq, self.next_seq + m, device=self.dev)
        self.next_seq += m
        return ids

    def _reset(self, mask: torch.Tensor, first: bool = False) -> None:
        m = int(mask.sum())
        if m == 0:
            return
        idx = self._pick(m, first)
        valid = idx < self.N
        rows = torch.nonzero(mask).squeeze(1)
        if self.sequential:
            self.active[rows[~valid]] = False
            rows, idx = rows[valid], idx[valid]
            mask = torch.zeros_like(mask)
            mask[rows] = True
            if len(rows) == 0:
                return
        self.lib_idx[rows] = idx
        full = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        full[rows] = idx
        s, L = self.sim, self.L
        wp.launch(load_rows_kernel, dim=self.n,
                  inputs=[wp.from_torch(mask.int()), wp.from_torch(full.int()), L["state"], L["buf"], L["hints"],
                          L["frenet"], L["sf"], L["sr"], L["prof"], L["tid"], L["params"], s.state, s.buf, s.hints,
                          s.frenet, s.axle_sf, s.axle_sr, s.prof, s.track_id, s.params], device=self.device)
        self.feat.prof[rows] = self.l_prof[idx]
        self.feat.v_ref[rows] = self.l_vref[idx]
        self.feat.dr_vec[rows] = self.l_dr[idx]
        # axle mu = first entry of the axle surfaces (for the teacher's privileged input before the first step)
        wp.to_torch(s.axle_mu).view(self.n, 2)[rows] = torch.stack(
            [wp.to_torch(s.axle_sf).view(self.n, 8)[rows, 0], wp.to_torch(s.axle_sr).view(self.n, 8)[rows, 0]], 1)
        self.k[rows] = 0
        self.t_drop[rows] = float("nan")
        self.t_exit[rows] = float("nan")
        self.s_prev[rows] = wp.to_torch(s.frenet).view(self.n, 2)[rows, 0]
        self.min_clear[rows] = float("inf")
        self.loc[rows] = 0
        self.ret[rows] = 0
        self.u_prev[rows] = wp.to_torch(s.buf).view(self.n, BUF_LEN, 2)[rows, 0]
        self.feat.reset_rows(mask)

    # ------------------------------------------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action (n, 2) in [-1, 1] (clipped here). Returns (reward, done, terminated, info) torch tensors;
        `info` holds per-episode results for the rows that finished this step."""
        a = action.clamp(-1, 1).contiguous().float()
        s = self.sim
        s.step(wp.from_torch(a))
        self.feat.step()
        wp.launch(metrics_kernel, dim=self.n, inputs=[s.params, s.state, s.hints, s.track_id, s.track_x, s.track_y,
                                                      s.track_theta, s.track_n, s.ds, self.mbuf], device=self.device)
        m = wp.to_torch(self.mbuf)
        sp, e, epsi, worst, beta = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4]
        vx = wp.to_torch(s.state).view(self.n, 13)[:, 3]
        cw, p = self.cw, self.p
        self.k += 1
        t = self.k * CONTROL_DT
        viol = (worst - 1.75).clamp(min=0)
        dv = (vx - self.feat.v_ref) / self.feat.v_ref
        du2 = ((a - self.u_prev) ** 2).sum(1)
        cost = cw.w_lane * viol ** 2 + cw.w_ey * e ** 2 + cw.w_v * dv ** 2 + cw.w_du * du2 - cw.w_prog * (sp - self.s_prev)
        self.s_prev = sp
        self.u_prev = a
        self.min_clear = torch.minimum(self.min_clear, 1.75 - worst)
        self.loc = torch.maximum(self.loc, ((beta.abs() > BETA_LOC) | (epsi.abs() > np.pi / 2)).float())
        pr = self.feat.prof
        s_drop, lt, lp = pr[:, 0], pr[:, 1], pr[:, 2]
        inf_patch = lp >= 1e8
        arrived = torch.isnan(self.t_drop) & (sp + p.lf >= s_drop)
        self.t_drop = torch.where(arrived, t, self.t_drop)
        exited = torch.isnan(self.t_exit) & ~inf_patch & (sp - p.lr >= s_drop + lt + lp)
        self.t_exit = torch.where(exited, t, self.t_exit)
        term = (worst - 1.75 > cw.severe) | (beta.abs() > cw.beta_term) | (epsi.abs() > cw.spin_term)
        end = term | (~torch.isnan(self.t_exit) & (t >= self.t_exit + T_AFTER_EXIT))
        end |= inf_patch & ~torch.isnan(self.t_drop) & (t >= self.t_drop + T_AFTER_DROP_INF)
        end |= (~torch.isnan(self.t_drop) & (t >= self.t_drop + T_CAP_AFTER_DROP)) | (t >= T_CAP_EP)
        cost = cost + cw.C_fail * term.float()
        rew = -cost * self.reward_scale
        self.ret += -cost
        end &= self.active
        info = {}
        if bool(end.any()):
            succ = (self.min_clear >= 0) & (self.loc == 0) & ~term & ~torch.isnan(self.t_drop)
            info = {"rows": torch.nonzero(end).squeeze(1), "success": succ[end], "departed": (self.min_clear < 0)[end],
                    "loc": (self.loc > 0)[end] | term[end], "return": self.ret[end], "lib_idx": self.lib_idx[end],
                    "min_clear": self.min_clear[end], "progress": None,
                    "arrived": ~torch.isnan(self.t_drop[end]), "len_s": t[end], "term": term[end]}
            self._reset(end)
        return rew, end, term & end, info
