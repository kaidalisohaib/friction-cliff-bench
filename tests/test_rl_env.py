"""RL environment (rl/env.py): episodes, outcomes and features match the evaluation episode runner."""

import numpy as np
import torch
import warp as wp

from control.episode import prefix_action, run_episodes, run_prefix, speed_profile
from rl import library as Lb
from rl.env import FrictionEnv
from rl.features import N_FEAT, N_PREV, N_PRIV
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams

TRACKS = T.all_tracks()
P = VehicleParams()


def _setup(n=6):
    _, scs = S.load_jsonl("splits/tuning_v1.jsonl")
    scs = scs[:n]
    snap, _ = run_prefix(scs, TRACKS, P, "cpu")
    return scs, snap


def test_env_matches_episode_runner_under_the_path_follower():
    scs, snap = _setup()
    prof = [speed_profile(TRACKS[s.track_id], s.v0, s.mu0) for s in scs]
    ref = run_episodes(scs, snap, lambda sim, k: prefix_action(sim.get_frenet(), sim.get_state(), scs, TRACKS, prof, P),
                       "W-PD", 0, TRACKS, P, "cpu")
    lib = Lb.from_scenarios(scs, snap)
    env = FrictionEnv(lib, len(scs), hist_len=5, device="cpu", sequential=True)
    out = {}
    for _ in range(5000):
        st, fr = env.sim.get_state(), env.sim.get_frenet()
        a = torch.as_tensor(prefix_action(fr, st, scs, TRACKS, prof, P), dtype=torch.float32)
        _, done, _, info = env.step(a)
        if info:
            for r, sc, mc in zip(info["rows"].tolist(), info["success"].tolist(), info["min_clear"].tolist()):
                out.setdefault(r, (sc, mc))
        if not bool(env.active.any()):
            break
    assert len(out) == len(scs)
    for j, r in enumerate(ref):
        assert out[j][0] == r.success, (j, out[j], r)
        assert abs(out[j][1] - r.min_clearance) < 1e-4


def test_features_and_privileged_shapes_and_values():
    scs, snap = _setup(3)
    env = FrictionEnv(Lb.from_scenarios(scs, snap), 3, hist_len=7, device="cpu", sequential=True)
    f = env.feat
    assert f.hist.shape == (3, 7, N_FEAT) and torch.isfinite(f.hist).all()
    assert torch.allclose(f.hist[:, 0], f.hist[:, -1])  # handover: history filled with the handover features
    env.step(torch.zeros(3, 2))
    pv = f.priv()
    assert pv.shape == (3, N_PRIV)
    mu = wp.to_torch(env.sim.axle_mu).view(3, 2)
    assert torch.allclose(pv[:, :2], mu)
    assert f.preview().shape == (3, N_PREV)
    # dry before the drop: the preview at +0 m equals mu0
    assert torch.allclose(f.preview()[:, 0], f.prof[:, 3])


def test_dr_draws_within_ranges():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p, v = Lb.dr_params(rng)
        assert abs(p.m / P.m - 1) <= 0.15 + 1e-9 and abs(p.h / P.h - 1) <= 0.10 + 1e-9
        assert P.delay_steps <= p.delay_steps <= P.delay_steps + 2
        assert np.all(np.abs(v) <= 1.0)


def test_vectorized_prefix_action_equals_loop():
    from control.episode import prefix_action_loop
    _, scs = S.load_jsonl("splits/tuning_v1.jsonl")
    scs = scs[:50]
    prof = [speed_profile(TRACKS[s.track_id], s.v0, s.mu0) for s in scs]
    rng = np.random.default_rng(0)
    for _ in range(20):
        fr = np.stack([rng.uniform(-1.0, 700.0, 50), rng.normal(0, 0.5, 50)], 1)
        X = rng.normal(0, 1, (50, 13))
        X[:, 3] = rng.uniform(-2, 27, 50)
        a = prefix_action(fr, X, scs, TRACKS, prof, P)
        b = prefix_action_loop(fr, X, scs, TRACKS, prof, P)
        assert np.array_equal(a, b)
