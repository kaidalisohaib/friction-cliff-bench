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
    def __init__(self, kind: str, z_dim: int = 8, width: int = 256, init_std: float = 0.1, init_std_long: float = 0.3):
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
        self.log_std = nn.Parameter(torch.tensor([float(np.log(init_std)), float(np.log(init_std_long))]))

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
    lr: float = 3e-5
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    horizon: int = 64  # rollout steps per env per iteration
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad: float = 1.0
    reward_scale: float = 0.01
    width: int = 256
    init_std: float = 0.1  # steering (normalized); 0.1 x 0.55 rad = 3 deg
    init_std_long: float = 0.15
    n_envs: int = 8192
    total_steps: int = 200_000_000
    hist_len: int = 1  # only the student uses history; the env keeps at least 1
    critic_warmup: int = 30  # PPO iterations with the actor frozen (value function catches up with the BC actor)
    target_kl: float = 0.005
    lr_critic: float = 3e-4  # stop an update's epochs early when the approximate KL exceeds this
    bc_reg: float = 0.0  # weight of a BC term (actor mean vs the path follower) in the PPO loss, decaying linearly
    bc_reg_end: float = 0.5  # ... to 0 at this fraction of training (D27)
    bc_steps: int = 2000  # behavior-cloning warm start of the actor on the shared path follower (D27); 0 = off
    # collapse rollback (D33, prereg A8): when the windowed training success stays below rollback_frac x the best so
    # far for rollback_patience consecutive log points, restore the best checkpoint (actor, critic, std) and halve
    # the actor learning rate; at most rollback_max times per run. A run that never collapses is unaffected.
    rollback_frac: float = 0.5
    rollback_patience: int = 3
    rollback_max: int = 3
    # follow-up options (D35, prereg A9); the defaults reproduce the A8 recipe exactly
    norm_reward: bool = True  # divide rewards by the running std of the discounted return
    snapshot_every: int = 0  # keep a copy of the weights every this many iterations in hist["snapshots"] (0 = off)
    bound_coef: float = 0.0  # weight of a penalty on the actor mean outside [-1, 1] (mean of relu(|mu| - 1)^2)
    val_every: int = 0  # deterministic (mean-action) validation on held-out scenarios every this many iterations
    select_on_val: bool = False  # keep the checkpoint with the best validation success instead of training success
    std_final_frac: float = 1.0  # < 1: exploration std is not learned but annealed linearly to this fraction


def should_rollback(recent: list[float], best: float, frac: float, patience: int) -> bool:
    """True when the last `patience` windowed success values are all below frac x best (D33)."""
    return best > 0 and len(recent) >= patience and all(x < frac * best for x in recent[-patience:])


def _inputs(env: FrictionEnv):
    f = env.feat
    return f.f.clone(), f.priv().clone(), f.preview().clone()


def bc_warm_start(ac: "ActorCritic", env: FrictionEnv, steps: int, seed: int, log=print) -> None:
    """D27: regress the actor mean onto the noise-free shared path-follower action (curvature feedforward + PD,
    the prefix driver's law), on states visited by that driver with action noise and a random per-episode speed
    multiplier (control/estimator.py: BehaviorDriver). Same procedure and data distribution for every PPO variant.
    The friction-blind follower has nothing to say about friction, so this only removes the exploration problem
    of learning to drive a curve from scratch; PPO then learns everything that depends on friction. The env is
    reset afterwards."""
    from control.estimator import BehaviorDriver
    drv = BehaviorDriver(env, seed + 77)
    opt = torch.optim.Adam(ac.actor.parameters() if ac.kind == "dr" else
                           list(ac.actor.parameters()) + list(ac.enc.parameters()), lr=1e-3)
    t0 = time.time()
    for it in range(steps):
        f, pv, pr = _inputs(env)
        with torch.no_grad():
            target = drv.law()
            act = (target + drv.ou).clamp(-1, 1)
        loss = ((ac.mean(f, pv, pr) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            drv.ou = 0.9 * drv.ou + torch.randn_like(drv.ou) * torch.tensor([0.02, 0.08], device=env.dev)
            _, done, _, info = env.step(act)
            if info:
                drv.resample(done)
        if it % 500 == 0 or it == steps - 1:
            log(f"[bc] it {it}/{steps} mse {float(loss.detach()):.5f} {time.time() - t0:.0f}s")
    env._reset(torch.ones(env.n, dtype=torch.bool, device=env.dev))


@torch.no_grad()
def deterministic_eval(ac: "ActorCritic", env: FrictionEnv) -> dict:
    """Run the mean policy once on every entry of a sequential env (FrictionEnv(..., sequential=True) with
    n_envs = library size); returns success / departure / termination rates and the mean episode length."""
    env.feat.t = 0  # same observation-noise realisation on every call, so checkpoints are compared like for like
    env.feat.gen.manual_seed(env.feat.gen.initial_seed())
    env.next_seq = env.n
    env.active[:] = True
    env._reset(torch.ones(env.n, dtype=torch.bool, device=env.dev), first=True)
    res = {k: [] for k in ("success", "departed", "term", "len_s", "return")}
    f, pv, pr = _inputs(env)
    for _ in range(int(80.0 / 0.02)):
        _, _, _, info = env.step(ac.mean(f, pv, pr))
        if info:
            for k in res:
                res[k] += info[k].float().tolist()
        if not bool(env.active.any()):
            break
        f, pv, pr = _inputs(env)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in res.items()} | {"n": len(res["success"])}


def train_ppo(kind: str, lib: dict, cfg: PPOConfig, seed: int = 0, device: str = "cuda:0", log=print,
              time_budget_s: float | None = None, env=None, val_lib: dict | None = None) -> tuple[ActorCritic, dict]:
    torch.manual_seed(seed)
    if env is None:
        env = FrictionEnv(lib, cfg.n_envs, cfg.hist_len, device, seed=seed, reward_scale=cfg.reward_scale)
    dev = env.dev
    ac = ActorCritic(kind, width=cfg.width, init_std=cfg.init_std, init_std_long=cfg.init_std_long).to(dev)
    drv = None
    if cfg.bc_steps > 0:
        bc_warm_start(ac, env, cfg.bc_steps, seed, log)
    if cfg.bc_reg > 0:
        from control.estimator import BehaviorDriver
        drv = BehaviorDriver(env, seed + 99)
        drv.mult[:] = 1.0
    anneal = cfg.std_final_frac < 1.0
    log_std0 = ac.log_std.detach().clone()
    if anneal:
        ac.log_std.requires_grad_(False)
    a_params = ([] if anneal else [ac.log_std]) + list(ac.actor.parameters()) + \
        (list(ac.enc.parameters()) if ac.kind != "dr" else [])
    venv = None
    if cfg.val_every and val_lib is not None:
        venv = FrictionEnv(val_lib, len(val_lib["track_id"]), cfg.hist_len, device, seed=seed + 5000,
                           reward_scale=cfg.reward_scale, sequential=True)
    vbest = (-1.0, None, -1)
    opt = torch.optim.Adam(a_params, lr=cfg.lr)
    opt_c = torch.optim.Adam(ac.critic.parameters(), lr=cfg.lr_critic)
    n, T = cfg.n_envs, cfg.horizon
    iters = max(1, cfg.total_steps // (n * T))
    hist = {"iter": [], "succ": [], "ret": [], "steps": [], "val": []}
    disc = torch.zeros(n, device=dev)  # running discounted return per env (reward normalization)
    rstat = [0.0, 1.0, 1e-4]  # mean, var, count of the discounted returns
    best = (-1.0, None, -1)
    win_s, win_r, win_d, upd = [], [], {}, {"kl": [], "clip": [], "sat": [], "ev": []}
    lr_scale, n_rollback, since_best = 1.0, 0, []
    hist["diag"] = []
    hist["rollbacks"] = []
    t0 = time.time()
    f, pv, pr = _inputs(env)
    for it in range(iters):
        if anneal:  # fixed for the whole iteration, so the stored log-probs match the update's policy
            frac = 1.0 + (cfg.std_final_frac - 1.0) * it / max(iters - 1, 1)
            ac.log_std.data = log_std0 + float(np.log(frac))
        buf = {k: [] for k in ("f", "pv", "pr", "a", "logp", "v", "r", "d", "bc")}
        with torch.no_grad():
            for _ in range(T):
                mu = ac.mean(f, pv, pr)
                std = ac.log_std.exp()
                a = mu + std * torch.randn_like(mu)
                logp = (-0.5 * ((a - mu) / std) ** 2 - ac.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
                v = ac.value(f, pv, pr)
                buf["bc"].append(drv.law() if drv is not None else mu)
                r, done, _, info = env.step(a)
                disc = disc * cfg.gamma + r
                bm, bv, bc = float(disc.mean()), float(disc.var()), disc.numel()
                m0, v0, c0 = rstat
                tot = c0 + bc
                dlt = bm - m0
                rstat = [m0 + dlt * bc / tot, (v0 * c0 + bv * bc + dlt ** 2 * c0 * bc / tot) / tot, tot]
                disc = torch.where(done, torch.zeros_like(disc), disc)
                if cfg.norm_reward:
                    r = r / float(np.sqrt(rstat[1]) + 1e-8)
                if drv is not None and info:
                    drv.mult[done] = 1.0
                for k, x in zip(("f", "pv", "pr", "a", "logp", "v", "r", "d"), (f, pv, pr, a, logp, v, r, done.float())):
                    buf[k].append(x)
                if info:
                    win_s += info["success"].float().tolist()
                    win_r += info["return"].tolist()
                    for k in ("departed", "arrived", "term", "len_s"):
                        win_d.setdefault(k, []).extend(info[k].float().tolist())
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
        upd["ev"].append(float(1 - (ret - B["v"]).var() / (ret.var() + 1e-8)))
        flat = {k: v.reshape(T * n, *v.shape[2:]) for k, v in B.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        N = T * n
        mb = N // cfg.minibatches
        warm = it < cfg.critic_warmup
        stop = False
        for g in opt.param_groups:
            g["lr"] = lr_scale * cfg.lr * max(0.0, 1.0 - max(it - cfg.critic_warmup, 0) / max(iters - cfg.critic_warmup, 1))
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(N, device=dev)
            for i in range(cfg.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                mu = ac.mean(flat["f"][j], flat["pv"][j], flat["pr"][j])
                logp = (-0.5 * ((flat["a"][j] - mu) / ac.log_std.exp()) ** 2 - ac.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
                lr_ = logp - flat["logp"][j]
                ratio = lr_.exp()
                kl = float(((ratio - 1) - lr_).mean().detach())
                upd["kl"].append(kl)
                upd["clip"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                upd["sat"].append(float((mu.detach().abs() > 1).float().mean()))
                upd.setdefault("sat_long", []).append(float((mu.detach()[:, 1].abs() > 1).float().mean()))
                upd.setdefault("mu_long", []).append(float(mu.detach()[:, 1].mean()))
                if not warm and kl > cfg.target_kl:
                    stop = True
                    break
                ad = flat["adv"][j]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vl = ((ac.value(flat["f"][j], flat["pv"][j], flat["pr"][j]) - flat["ret"][j]) ** 2).mean()
                ent = (ac.log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
                loss = cfg.vf_coef * vl if warm else pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                if cfg.bound_coef > 0 and not warm:
                    loss = loss + cfg.bound_coef * ((mu.abs() - 1).clamp(min=0) ** 2).sum(1).mean()
                if drv is not None and not warm:
                    frac = max(0.0, 1.0 - it / max(cfg.bc_reg_end * iters, 1.0))
                    loss = loss + cfg.bc_reg * frac * ((mu - flat["bc"][j]) ** 2).sum(1).mean()
                opt.zero_grad()
                opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(a_params, cfg.max_grad)
                nn.utils.clip_grad_norm_(ac.critic.parameters(), cfg.max_grad)
                if not warm:
                    opt.step()
                opt_c.step()
        if it % 10 == 0 or it == iters - 1:
            s = float(np.mean(win_s)) if win_s else float("nan")
            rr = float(np.mean(win_r)) if win_r else float("nan")
            hist["iter"].append(it), hist["succ"].append(s), hist["ret"].append(rr), hist["steps"].append((it + 1) * n * T)
            if venv is not None and not warm and it % cfg.val_every == 0:
                vr = deterministic_eval(ac, venv)
                hist["val"].append({"it": it, **vr})
                win_d.setdefault("val_succ", []).append(vr["success"])
                if vr["success"] >= vbest[0]:
                    vbest = (vr["success"], {k: v.detach().clone() for k, v in ac.state_dict().items()}, it)
            dg = {k: float(np.mean(v)) for k, v in win_d.items() if v}
            dg.update({k: float(np.mean(v)) for k, v in upd.items() if v})
            dg.update({"it": it, "succ": s, "ret": rr, "rstd": float(np.sqrt(rstat[1]))})
            hist["diag"].append(dg)
            log(f"[{kind}] it {it}/{iters} steps {(it + 1) * n * T / 1e6:.1f}M train-success {s:.3f} return {rr:.1f} "
                f"std {[round(x, 4) for x in ac.log_std.exp().tolist()]} {time.time() - t0:.0f}s | "
                + " ".join(f"{k} {v:.3g}" for k, v in dg.items() if k not in ("it", "succ", "ret")))
            win_d, upd = {}, {"kl": [], "clip": [], "sat": [], "ev": []}
            if not warm and len(win_s) >= 500 and s > best[0]:
                best = (s, {k: v.detach().clone() for k, v in ac.state_dict().items()}, it)
                since_best = []
            elif not warm and len(win_s) >= 500:
                since_best.append(s)
                if (n_rollback < cfg.rollback_max and best[1] is not None
                        and should_rollback(since_best, best[0], cfg.rollback_frac, cfg.rollback_patience)):
                    ac.load_state_dict(best[1])
                    lr_scale *= 0.5
                    n_rollback += 1
                    since_best = []
                    opt = torch.optim.Adam(a_params, lr=cfg.lr)  # fresh moments after the rollback
                    opt_c = torch.optim.Adam(ac.critic.parameters(), lr=cfg.lr_critic)
                    hist["rollbacks"].append(it)
                    log(f"[{kind}] collapse rollback {n_rollback} at it {it} to it {best[2]} "
                        f"(success {s:.3f} < {cfg.rollback_frac} x {best[0]:.3f}); actor lr x{lr_scale}")
            win_s, win_r = [], []
        if cfg.snapshot_every and it % cfg.snapshot_every == 0:
            hist.setdefault("snapshots", {})[it] = {k: v.detach().cpu().clone() for k, v in ac.state_dict().items()}
        if time_budget_s and time.time() - t0 > time_budget_s:
            log(f"[{kind}] time budget reached at it {it}")
            break
    if cfg.select_on_val and venv is not None:
        vr = deterministic_eval(ac, venv)  # the final weights are a candidate too
        hist["val"].append({"it": iters, **vr})
        if vr["success"] >= vbest[0]:
            vbest = (vr["success"], None, iters)
        if vbest[1] is not None:
            ac.load_state_dict(vbest[1])
        log(f"[{kind}] kept checkpoint of it {vbest[2]} (deterministic validation success {vbest[0]:.3f})")
        best = (best[0], None, vbest[2])
    elif best[1] is not None:
        ac.load_state_dict(best[1])  # best training-window checkpoint (training episodes only)
        log(f"[{kind}] kept checkpoint of it {best[2]} (train-success {best[0]:.3f})")
    hist["best_succ"], hist["best_it"] = best[0], best[2]
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
    target: str = "latent"  # "latent": regress the teacher's z (RMA); "action": match the teacher's actions (E3)


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
    assert cfg.target in ("latent", "action"), cfg.target
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
            if cfg.target == "action":  # E3: DAgger action distillation through the frozen teacher head
                with torch.no_grad():
                    a_t = teacher.actor(torch.cat([f, z], 1))
                loss = loss + ((teacher.actor(torch.cat([f, zh], 1)) - a_t) ** 2).mean()
            else:
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
            hist["iter"].append(it), hist["loss"].append(float(loss.detach())), hist["succ"].append(s)
            log(f"[student] it {it}/{iters} loss {float(loss.detach()):.4f} train-success {s:.3f} {time.time() - t0:.0f}s")
            win_s = []
        if time_budget_s and time.time() - t0 > time_budget_s:
            break
    hist["wall_s"] = time.time() - t0
    return st, hist


def save_policy(path, model: nn.Module, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
