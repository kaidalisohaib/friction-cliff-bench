"""PPO, the RMA teacher/student pair, and the DR policy (BASELINES.md §4). Own minimal implementation (D26).

Variants (`kind`):
  "teacher"          P-T : actor(f_t, z),  z = enc(priv)               priv = current per-axle mu, surfaces, DR params
  "teacher_preview"  teacher of P-SP: z = enc(priv, mu preview 0–30 m)
  "dr"               P-DR: actor(f_t)                                  no history, no privileged input
Student (RMA phase 2, P-S / P-SP): z_hat = hist_enc(last W feature vectors), regressed onto the teacher's z with
the teacher's actor frozen; data from on-policy rollouts of the student itself (DAgger-style).
Critic (training only, asymmetric): sees f_t, priv and the preview for every variant.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.env import FrictionEnv
from rl.features import N_FEAT, N_PREV, N_PRIV


def mlp(i, hs, o, act=nn.ELU):
    layers, d = [], i
    for h in hs:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class HistEncoder(nn.Module):
    """GRU over the last W feature vectors -> z_hat."""

    def __init__(self, z_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.out = nn.Linear(hidden, z_dim)

    def forward(self, hist):
        _, h = self.gru(hist)
        return self.out(h[-1])


class ActorCritic(nn.Module):
    def __init__(self, kind: str, z_dim: int = 8, width: int = 256, init_std: float = 0.3):
        super().__init__()
        self.kind, self.z_dim = kind, z_dim
        if kind == "dr":
            a_in = N_FEAT
        else:
            p_in = N_PRIV + (N_PREV if kind == "teacher_preview" else 0)
            self.enc = mlp(p_in, [64, 64], z_dim)
            a_in = N_FEAT + z_dim
        self.actor = mlp(a_in, [width, width // 2], 2)
        self.critic = mlp(N_FEAT + N_PRIV + N_PREV, [width, width // 2], 1)
        self.log_std = nn.Parameter(torch.full((2,), float(np.log(init_std))))

    def z(self, priv, prev):
        x = torch.cat([priv, prev], 1) if self.kind == "teacher_preview" else priv
        return self.enc(x)

    def mean(self, f, priv, prev, z=None):
        if self.kind == "dr":
            return self.actor(f)
        if z is None:
            z = self.z(priv, prev)
        return self.actor(torch.cat([f, z], 1))

    def value(self, f, priv, prev):
        return self.critic(torch.cat([f, priv, prev], 1)).squeeze(1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    horizon: int = 32  # rollout steps per env per iteration
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad: float = 1.0
    reward_scale: float = 0.01
    width: int = 256
    init_std: float = 0.3
    n_envs: int = 4096
    total_steps: int = 50_000_000
    hist_len: int = 1  # only the student uses history; the env keeps at least 1


def _inputs(env: FrictionEnv):
    f = env.feat
    return f.f.clone(), f.priv().clone(), f.preview().clone()


def train_ppo(kind: str, lib: dict, cfg: PPOConfig, seed: int = 0, device: str = "cuda:0", log=print,
              time_budget_s: float | None = None) -> tuple[ActorCritic, dict]:
    torch.manual_seed(seed)
    env = FrictionEnv(lib, cfg.n_envs, cfg.hist_len, device, seed=seed, reward_scale=cfg.reward_scale)
    dev = env.dev
    ac = ActorCritic(kind, width=cfg.width, init_std=cfg.init_std).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    n, T = cfg.n_envs, cfg.horizon
    iters = max(1, cfg.total_steps // (n * T))
    hist = {"iter": [], "succ": [], "ret": [], "steps": []}
    win_s, win_r = [], []
    t0 = time.time()
    f, pv, pr = _inputs(env)
    for it in range(iters):
        buf = {k: [] for k in ("f", "pv", "pr", "a", "logp", "v", "r", "d")}
        with torch.no_grad():
            for _ in range(T):
                mu = ac.mean(f, pv, pr)
                std = ac.log_std.exp()
                a = mu + std * torch.randn_like(mu)
                logp = (-0.5 * ((a - mu) / std) ** 2 - ac.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
                v = ac.value(f, pv, pr)
                r, done, _, info = env.step(a)
                for k, x in zip(("f", "pv", "pr", "a", "logp", "v", "r", "d"), (f, pv, pr, a, logp, v, r, done.float())):
                    buf[k].append(x)
                if info:
                    win_s += info["success"].float().tolist()
                    win_r += info["return"].tolist()
                f, pv, pr = _inputs(env)
            v_last = ac.value(f, pv, pr)
        B = {k: torch.stack(v) for k, v in buf.items()}
        adv = torch.zeros_like(B["r"])
        last = torch.zeros(n, device=dev)
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else B["v"][t + 1]
            nonterm = 1.0 - B["d"][t]
            delta = B["r"][t] + cfg.gamma * nv * nonterm - B["v"][t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[t] = last
        ret = adv + B["v"]
        flat = {k: v.reshape(T * n, *v.shape[2:]) for k, v in B.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        N = T * n
        mb = N // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(cfg.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                mu = ac.mean(flat["f"][j], flat["pv"][j], flat["pr"][j])
                logp = (-0.5 * ((flat["a"][j] - mu) / ac.log_std.exp()) ** 2 - ac.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
                ratio = (logp - flat["logp"][j]).exp()
                ad = flat["adv"][j]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vl = ((ac.value(flat["f"][j], flat["pv"][j], flat["pr"][j]) - flat["ret"][j]) ** 2).mean()
                ent = (ac.log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad)
                opt.step()
        if it % 10 == 0 or it == iters - 1:
            s = float(np.mean(win_s)) if win_s else float("nan")
            rr = float(np.mean(win_r)) if win_r else float("nan")
            hist["iter"].append(it), hist["succ"].append(s), hist["ret"].append(rr), hist["steps"].append((it + 1) * n * T)
            log(f"[{kind}] it {it}/{iters} steps {(it + 1) * n * T / 1e6:.1f}M train-success {s:.3f} return {rr:.1f} "
                f"std {ac.log_std.exp().tolist()} {time.time() - t0:.0f}s")
            win_s, win_r = [], []
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f"[{kind}] time budget reached at it {it}")
            break
    hist["wall_s"] = time.time() - t0
    return ac, hist


@dataclass
class DistillConfig:
    hist_len: int = 25
    lr: float = 1e-3
    hidden: int = 64
    n_envs: int = 4096
    total_steps: int = 10_000_000
    batch_steps: int = 8  # env steps accumulated per gradient step
    act_noise: float = 0.0  # exploration noise on the student's actions while collecting


class Student(nn.Module):
    """P-S / P-SP at test time: frozen teacher actor + history encoder."""

    def __init__(self, teacher: ActorCritic, hidden: int = 64):
        super().__init__()
        self.teacher = teacher
        self.henc = HistEncoder(teacher.z_dim, hidden)
        for p_ in self.teacher.parameters():
            p_.requires_grad_(False)

    def mean(self, f, hist):
        return self.teacher.actor(torch.cat([f, self.henc(hist)], 1))


def distill(teacher: ActorCritic, lib: dict, cfg: DistillConfig, seed: int = 0, device: str = "cuda:0", log=print,
            time_budget_s: float | None = None) -> tuple[Student, dict]:
    torch.manual_seed(seed)
    env = FrictionEnv(lib, cfg.n_envs, cfg.hist_len, device, seed=seed + 1000)
    dev = env.dev
    st = Student(teacher, cfg.hidden).to(dev)
    opt = torch.optim.Adam(st.henc.parameters(), lr=cfg.lr)
    iters = max(1, cfg.total_steps // (cfg.n_envs * cfg.batch_steps))
    hist = {"iter": [], "loss": [], "succ": []}
    win_s = []
    t0 = time.time()
    for it in range(iters):
        loss = 0.0
        for _ in range(cfg.batch_steps):
            f, pv, pr = _inputs(env)
            h = env.feat.hist.clone()
            with torch.no_grad():
                z = teacher.z(pv, pr)
            zh = st.henc(h)
            loss = loss + ((zh - z) ** 2).mean()
            with torch.no_grad():
                a = teacher.actor(torch.cat([f, zh.detach()], 1))
                if cfg.act_noise > 0:
                    a = a + cfg.act_noise * torch.randn_like(a)
                _, _, _, info = env.step(a)
            if info:
                win_s += info["success"].float().tolist()
        loss = loss / cfg.batch_steps
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(st.henc.parameters(), 1.0)
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            s = float(np.mean(win_s)) if win_s else float("nan")
            hist["iter"].append(it), hist["loss"].append(float(loss)), hist["succ"].append(s)
            log(f"[student] it {it}/{iters} loss {float(loss):.4f} train-success {s:.3f} {time.time() - t0:.0f}s")
            win_s = []
        if time_budget_s and time.time() - t0 > time_budget_s:
            break
    hist["wall_s"] = time.time() - t0
    return st, hist


def save_policy(path, model: nn.Module, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
