"""Learned policies as evaluation controllers (`control/episode.py: run_episodes`), with the same feature
pipeline as training (rl/features.py). Deterministic: the actor's mean action."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from control.controllers import run_seed
from rl.features import Features
from rl.library import SURF_IDX
from scenarios import sampler as S


def policy_controller(sim, scs, model, kind: str, hist_len: int, seed: int = 0, noise_scale: float = 1.0):
    """kind: "teacher" | "teacher_preview" (privileged, P-T) | "dr" (P-DR) | "student" (P-S, P-SP)."""
    feat = Features(sim, hist_len, seeds=np.array([run_seed(sc.sid, "policy", seed) for sc in scs]),
                    noise_scale=noise_scale)
    dev = feat.dev
    feat.prof[:] = torch.tensor([[sc.s_drop, sc.transition, sc.patch_len, sc.mu0, sc.mu1, SURF_IDX[sc.surface], 0, 0]
                                 for sc in scs], dtype=torch.float32, device=dev)
    feat.v_ref[:] = torch.tensor([S.v_eff(sc) for sc in scs], dtype=torch.float32, device=dev)
    feat.dr_vec[:] = 0.0
    feat.dr_vec[:, -1] = -1.0  # nominal parameters, no extra delay
    model.eval()

    def ctrl(s, k):
        if k == 0:
            # handover: axle mu from the loaded axle surfaces (the sim fills axle_mu on its first step)
            wp.to_torch(s.axle_mu).view(s.n, 2)[:] = torch.stack(
                [wp.to_torch(s.axle_sf).view(s.n, 8)[:, 0], wp.to_torch(s.axle_sr).view(s.n, 8)[:, 0]], 1)
            feat.reset_rows(torch.ones(s.n, dtype=torch.bool, device=dev))
        else:
            feat.step()
        with torch.no_grad():
            if kind == "student":
                a = model.mean(feat.f, feat.hist)
            else:
                a = model.mean(feat.f, feat.priv(), feat.preview())
        ctrl.last = a.clamp(-1, 1).contiguous()  # keep the tensor alive while the sim holds a view of it
        return wp.from_torch(ctrl.last)

    ctrl.feat = feat
    return ctrl
