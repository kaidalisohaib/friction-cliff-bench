"""PPO / distillation / estimator training loops run end to end on a tiny CPU env and produce finite outputs."""

import numpy as np
import torch

from control.episode import run_prefix
from control.estimator import train_estimator
from eval.methods import run_controller
from rl import library as Lb
from rl.policy import policy_controller
from rl.ppo import DistillConfig, PPOConfig, distill, train_ppo
from scenarios import sampler as S
from scenarios import track as T

TRACKS = T.all_tracks()


def _lib(n=8):
    _, scs = S.load_jsonl("splits/tuning_v1.jsonl")
    scs = scs[:n]
    snap, _ = run_prefix(scs, TRACKS, device="cpu")
    return scs, snap, Lb.from_scenarios(scs, snap)


def test_training_loops_run_and_policies_evaluate():
    scs, snap, lib = _lib()
    for kind in ("teacher", "teacher_preview", "dr"):
        ac, hist = train_ppo(kind, lib, PPOConfig(n_envs=8, horizon=8, total_steps=8 * 8 * 3, epochs=1, bc_steps=3),
                             device="cpu", log=lambda s: None)
        assert all(torch.isfinite(p).all() for p in ac.parameters())
    teacher, _ = train_ppo("teacher", lib, PPOConfig(n_envs=8, horizon=8, total_steps=64, epochs=1, bc_steps=3), device="cpu",
                           log=lambda s: None)
    st, h = distill(teacher, lib, DistillConfig(hist_len=5, n_envs=8, total_steps=8 * 8 * 2, batch_steps=2),
                    device="cpu", log=lambda s: None)
    assert np.isfinite(h["loss"]).all()
    recs = run_controller(scs[:3], {k: v[:3] for k, v in snap.items()},
                          lambda sim, sub, ss, seed: policy_controller(sim, sub, st, "student", 5, seed), "P-S", 0,
                          device="cpu")
    assert len(recs) == 3


def test_estimator_training_runs():
    _, _, lib = _lib(4)
    est, h = train_estimator(lib, 5, steps=3, n_envs=4, device="cpu", log=lambda s: None)
    assert np.isfinite(h["loss"]).all()
