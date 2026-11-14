"""Diagnostic PPO training runs for the RL follow-up (D35). Training library only (never a test
scenario); nothing here is a result.

Trains one PPO variant at the tuned configuration (runs/tuning/best_<method>.json), optionally overridden by
`--set key=value` (PPOConfig fields), and writes runs/rl_fix/<tag>_s<seed>.{pt,json}. The JSON holds the per-log-point
diagnostics (success, departure, arrival, termination, episode length, KL, clip fraction, saturation, explained variance).

PYTHONPATH=. uv run python scripts/rl_diag.py --method P-T --seed 4 --tag base [--set gamma=0.998 --set lr=2e-5]
"""

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

from rl import library as Lb
from rl.ppo import PPOConfig, save_policy, train_ppo

KIND = {"P-T": "teacher", "P-TP": "teacher_preview", "P-DR": "dr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="P-T")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/rl_fix")
    a = ap.parse_args()
    from scripts.tune_method import PPO_STEPS
    bp = json.loads(Path(f"runs/tuning/best_{a.method}.json").read_text())["best_params"]
    cfg = PPOConfig(lr=bp["lr"], init_std=bp["init_std"], init_std_long=bp["init_std_long"],
                    reward_scale=bp["reward_scale"], gamma=bp["gamma"], width=bp["width"], total_steps=PPO_STEPS)
    types = {f.name: f.type for f in fields(PPOConfig)}
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in types, k
        setattr(cfg, k, type(getattr(cfg, k))(v) if not isinstance(getattr(cfg, k), bool) else v == "1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lib = Lb.load(Path("runs/rl/lib_train_v1.npz"))
    model, hist = train_ppo(KIND[a.method], lib, cfg, seed=a.seed, device=a.device,
                            log=lambda s: print(s, flush=True))
    stem = out / f"{a.tag}_s{a.seed}"
    snaps = hist.pop("snapshots", None)
    if snaps:
        import torch
        torch.save(snaps, stem.with_name(stem.name + "_snaps.pt"))
    save_policy(stem.with_suffix(".pt"), model, {"kind": KIND[a.method], "cfg": asdict(cfg), "seed": a.seed,
                                                 "method": a.method, "tag": a.tag})
    stem.with_suffix(".json").write_text(json.dumps({"cfg": asdict(cfg), "method": a.method, "seed": a.seed,
                                                     "wall_s": time.time() - t0, "hist": hist}, default=float))
    print(f"done {stem} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
