"""Tune one method on the tuning split within the prereg V2 budget (20 Optuna TPE trials / 15 GPU-h; eval/tuning.py).

Methods and trial definitions (one trial = one sampled configuration evaluated on the reactive-feasible
tuning scenarios of both tuning blocks, noise/training seed 0):
  M-B   MPPI noise/temperature/correlation + window W               (K = 1024, H = 150 fixed = oracle compute)
  M-L   the same + estimator width; each trial trains its own estimator (counted in the budget)
  P-T   PPO teacher (current per-axle mu): lr, init std, reward scale, gamma, width         [P-S budget, 14 trials]
  P-S   RMA student of the best P-T: W, encoder width, lr                                    [P-S budget, 6 trials]
  P-TP  preview teacher (H3): same space as P-T                                               [P-SP budget, 14 trials]
  P-SP  student of the best P-TP                                                              [P-SP budget, 6 trials]
  P-DR  PPO, no history, no privileged input: same space as P-T                               [20 trials]
GPU-seconds are wall-clock seconds of the trial on the (shared) GPU, which over-counts: conservative for the cap.
Ledger: runs/tuning/ledger.jsonl (one line per trial); best configs: runs/tuning/best_<method>.json.

PYTHONPATH=. uv run python scripts/train/tune_method.py --method M-B [--trials 20] [--gpu-hours 15]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from control.adaptive import BankMPPI, LearnedMPPI
from control.controllers import ORACLE_OR
from control.estimator import train_estimator
from eval.methods import handover, objective, run_controller, tuning_sets
from eval.tuning import tune
from labelling.label import git_sha
from rl import library as Lb
from rl.policy import policy_controller
from rl.ppo import DistillConfig, PPOConfig, distill, save_policy, train_ppo

OUT = Path("runs/tuning")
LIB = Path("runs/rl/lib_train_v1.npz")
W_SET = (5, 12, 25, 50)  # allowed history windows [control steps] = {0.1, 0.25, 0.5, 1.0} s (prereg E1 set)
PPO_STEPS = 200_000_000
DISTILL_STEPS = 15_000_000
SPLIT_BUDGET = {"P-T": 14, "P-S": 6, "P-TP": 14, "P-SP": 6}


def mppi_space(trial):
    return {"sigma_steer": trial.suggest_float("sigma_steer", 0.02, 0.2, log=True),
            "sigma_long": trial.suggest_float("sigma_long", 0.1, 0.6),
            "lam": trial.suggest_float("lam", 1.0, 100.0, log=True),
            "rho": trial.suggest_categorical("rho", [0.0, 0.5, 0.9]),
            "W": trial.suggest_categorical("W", list(W_SET))}


def ppo_space(trial):
    return {"lr": trial.suggest_float("lr", 1e-5, 1e-4, log=True),
            "init_std": trial.suggest_float("init_std", 0.03, 0.15, log=True),
            "init_std_long": trial.suggest_float("init_std_long", 0.05, 0.3, log=True),
            "reward_scale": trial.suggest_float("reward_scale", 0.003, 0.03, log=True),
            "gamma": trial.suggest_categorical("gamma", [0.99, 0.995, 0.998]),
            "width": trial.suggest_categorical("width", [128, 256])}


def student_space(trial):
    return {"hist_len": trial.suggest_categorical("hist_len", list(W_SET)),
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["M-B", "M-L", "P-T", "P-S", "P-TP", "P-SP", "P-DR"])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = a.method
    n_trials = a.trials or SPLIT_BUDGET.get(m, 20)
    gpu_h = a.gpu_hours or (15.0 * SPLIT_BUDGET[m] / 20 if m in SPLIT_BUDGET else 15.0)
    scs, rf = tuning_sets()
    rf_set = set(rf)
    ev_scs = [s for s in scs if s.sid in rf_set]
    snap = handover(ev_scs, device=a.device, cache=OUT / "handover_tuning_rf.npz")
    lib = Lb.load(LIB) if m.startswith("P-") or m == "M-L" else None
    log_f = open(OUT / f"log_{m}.txt", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    def evaluate(factory, name):
        recs = run_controller(ev_scs, snap, factory, name, 0, a.device, a.batch)
        return objective(recs, [s.sid for s in ev_scs]), recs

    def obj(params):
        t0 = time.time()
        tag = f"{m}_{int(t0)}"
        if m in ("M-B", "M-L"):
            cfg = replace(ORACLE_OR, sigma_steer=params["sigma_steer"], sigma_long=params["sigma_long"],
                          lam=params["lam"], rho=params["rho"])
            if m == "M-B":
                fac = lambda sim, sub, ss, seed: BankMPPI(sim, sub, cfg, seed, ss, params["W"])
            else:
                est, _ = train_estimator(lib, params["W"], hidden=params.get("hidden", 64), device=a.device, log=log)
                torch.save({"state_dict": est.state_dict(), "W": params["W"], "hidden": params.get("hidden", 64)},
                           OUT / f"{tag}_estimator.pt")
                fac = lambda sim, sub, ss, seed: LearnedMPPI(sim, sub, cfg, seed, ss, est)
            val, _ = evaluate(fac, m)
        elif m in ("P-T", "P-TP", "P-DR"):
            kind = {"P-T": "teacher", "P-TP": "teacher_preview", "P-DR": "dr"}[m]
            cfg = PPOConfig(lr=params["lr"], init_std=params["init_std"], init_std_long=params["init_std_long"],
                            reward_scale=params["reward_scale"],
                            gamma=params["gamma"], width=params["width"], total_steps=PPO_STEPS)
            ac, hist = train_ppo(kind, lib, cfg, seed=0, device=a.device, log=log)
            save_policy(OUT / f"{tag}.pt", ac, {"kind": kind, "cfg": asdict(cfg), "params": params})
            val, _ = evaluate(lambda sim, sub, ss, seed: policy_controller(sim, sub, ac, kind, 1, seed), m)
        else:  # students
            tname = {"P-S": "P-T", "P-SP": "P-TP"}[m]
            best = json.loads((OUT / f"best_{tname}.json").read_text())
            ck = torch.load(best["checkpoint"], weights_only=False)
            from rl.ppo import ActorCritic
            teacher = ActorCritic(ck["meta"]["kind"], width=ck["meta"]["cfg"]["width"]).to("cuda")
            teacher.load_state_dict(ck["state_dict"])
            dcfg = DistillConfig(hist_len=params["hist_len"], hidden=params["hidden"], lr=params["lr"],
                                 total_steps=DISTILL_STEPS)
            st, hist = distill(teacher, lib, dcfg, seed=0, device=a.device, log=log)
            save_policy(OUT / f"{tag}.pt", st, {"kind": "student", "teacher": best["checkpoint"],
                                                 "cfg": asdict(dcfg), "params": params})
            val, _ = evaluate(lambda sim, sub, ss, seed: policy_controller(sim, sub, st, "student",
                                                                         params["hist_len"], seed), m)
            tag = tag
        log(f"[{m}] trial params {params} -> objective {val:.4f} ({time.time() - t0:.0f}s)")
        obj.last_tag = tag
        return val, time.time() - t0

    space = {"M-B": mppi_space, "P-T": ppo_space, "P-TP": ppo_space, "P-DR": ppo_space,
             "P-S": student_space, "P-SP": student_space}.get(m)
    if m == "M-L":
        def space(trial):
            d = mppi_space(trial)
            d["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128])
            return d

    def obj_tagged(params):
        v, g = obj(params)
        return v, g, {"tag": obj.last_tag}

    res = tune(m, obj_tagged, space, OUT / "ledger.jsonl", n_trials=n_trials, gpu_hours=gpu_h, seed=0, git=git_sha())
    led = [json.loads(x) for x in (OUT / "ledger.jsonl").read_text().splitlines()]
    best_tag = next((r.get("tag") for r in led if r["method"] == m and r["value"] == res.best_value), None)
    rec = {"method": m, "best_params": res.best_params, "best_value": res.best_value, "n_trials": res.n_trials,
           "gpu_hours": res.gpu_hours, "stopped_by": res.stopped_by, "tag": best_tag,
           "n_tuning_rf": len(ev_scs), "git": git_sha()}
    if m.startswith("P-"):
        rec["checkpoint"] = str(OUT / f"{best_tag}.pt")
    if m == "M-L":
        rec["checkpoint"] = str(OUT / f"{best_tag}_estimator.pt")
    (OUT / f"best_{m}.json").write_text(json.dumps(rec, indent=1))
    log(json.dumps(rec))


if __name__ == "__main__":
    main()
