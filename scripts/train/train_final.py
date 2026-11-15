"""Final training runs of the learned methods at their tuned configurations (EVAL_PROTOCOL §2: >= 5 training seeds;
all seeds kept and reported, never best-of). Not part of the tuning budget (the configuration is fixed).

  P-T, P-TP, P-DR : PPO with the best tuned config, seeds 0..4
  P-S, P-SP       : student of the same-seed teacher, best tuned student config
  M-L             : estimator with the best tuned config, seeds 0..4
Checkpoints: runs/final/<method>_s<seed>.pt ; ledger lines: runs/final/ledger.jsonl.

PYTHONPATH=. uv run python scripts/train/train_final.py --methods P-T,P-S,P-DR,P-TP,P-SP,M-L --seeds 0,1,2,3,4
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from control.estimator import train_estimator
from labelling.label import git_sha
from rl import library as Lb
from rl.ppo import DistillConfig, PPOConfig, distill, save_policy, train_ppo

TUNE = Path("runs/tuning")
OUT = Path("runs/final")
KIND = {"P-T": "teacher", "P-TP": "teacher_preview", "P-DR": "dr"}
TEACHER_OF = {"P-S": "P-T", "P-SP": "P-TP"}


def best(m):
    return json.loads((TUNE / f"best_{m}.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="P-T,P-S,P-DR,P-TP,P-SP,M-L")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lib = Lb.load(Path("runs/rl/lib_train_v1.npz"))
    from scripts.train.tune_method import DISTILL_STEPS, PPO_STEPS
    for m in a.methods.split(","):
        for seed in map(int, a.seeds.split(",")):
            path = OUT / f"{m}_s{seed}.pt"
            if path.exists():
                continue
            t0 = time.time()
            bp = best(m)["best_params"]
            if m in KIND:
                cfg = PPOConfig(lr=bp["lr"], init_std=bp["init_std"], init_std_long=bp["init_std_long"], reward_scale=bp["reward_scale"],
                                gamma=bp["gamma"], width=bp["width"], total_steps=PPO_STEPS)
                model, hist = train_ppo(KIND[m], lib, cfg, seed=seed, device=a.device)
                meta = {"kind": KIND[m], "cfg": asdict(cfg), "params": bp, "seed": seed}
            elif m in TEACHER_OF:
                tpath = OUT / f"{TEACHER_OF[m]}_s{seed}.pt"
                ck = torch.load(tpath, weights_only=False)
                from rl.ppo import ActorCritic
                teacher = ActorCritic(ck["meta"]["kind"], width=ck["meta"]["cfg"]["width"]).to("cuda")
                teacher.load_state_dict(ck["state_dict"])
                dcfg = DistillConfig(hist_len=bp["hist_len"], hidden=bp["hidden"], lr=bp["lr"], total_steps=DISTILL_STEPS)
                model, hist = distill(teacher, lib, dcfg, seed=seed, device=a.device)
                meta = {"kind": "student", "teacher": str(tpath), "cfg": asdict(dcfg), "params": bp, "seed": seed}
            elif m == "M-L":
                model, hist = train_estimator(lib, bp["W"], hidden=bp["hidden"], seed=seed, device=a.device)
                meta = {"kind": "estimator", "W": bp["W"], "hidden": bp["hidden"], "params": bp, "seed": seed}
            else:
                raise SystemExit(m)
            save_policy(path, model, meta)
            with open(OUT / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"method": m, "seed": seed, "gpu_s": time.time() - t0, "git": git_sha(),
                                    "final_train_success": (hist.get("succ") or [None])[-1]}) + "\n")
            print(f"{m} seed {seed} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
