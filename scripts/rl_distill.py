"""Distil a history-only student from a follow-up teacher checkpoint (scripts/rl_diag.py) with the tuned student
configuration of the matching method (P-S for "teacher", P-SP for "teacher_preview"). Training library only.

PYTHONPATH=. uv run python scripts/rl_distill.py runs/rl_fix/val_s0.pt [--seed S]   -> runs/rl_fix/stu_<teacher stem>.pt
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from rl import library as Lb
from rl.ppo import ActorCritic, DistillConfig, distill, save_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from scripts.tune_method import DISTILL_STEPS
    tp = Path(a.teacher)
    ck = torch.load(tp, weights_only=False)
    meta = ck["meta"]
    m = {"teacher": "P-S", "teacher_preview": "P-SP"}[meta["kind"]]
    bp = json.loads(Path(f"runs/tuning/best_{m}.json").read_text())["best_params"]
    teacher = ActorCritic(meta["kind"], width=meta["cfg"]["width"]).to(a.device)
    teacher.load_state_dict(ck["state_dict"])
    dcfg = DistillConfig(hist_len=bp["hist_len"], hidden=bp["hidden"], lr=bp["lr"], total_steps=DISTILL_STEPS)
    seed = int(meta.get("seed", 0))
    t0 = time.time()
    st, hist = distill(teacher, Lb.load(Path("runs/rl/lib_train_v1.npz")), dcfg, seed=seed, device=a.device,
                       log=lambda s: print(s, flush=True))
    out = tp.with_name(f"stu_{tp.stem}.pt")
    save_policy(out, st, {"kind": "student", "teacher": str(tp), "cfg": asdict(dcfg), "params": bp, "seed": seed})
    print(f"done {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
