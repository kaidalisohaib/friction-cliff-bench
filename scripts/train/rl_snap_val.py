"""Score every weight snapshot of a follow-up PPO run on the held-out validation library (deterministic policy,
fixed observation noise) to compare checkpoint-selection rules post hoc. Validation library only.

PYTHONPATH=. uv run python scripts/train/rl_snap_val.py runs/rl_fix/valsnap_s0_snaps.pt [...]
  -> runs/rl_fix/<stem>_snapval.json  {iteration: {success, departed, term, len_s, return, n}}
"""

import json
import sys
from pathlib import Path

import torch

from rl import library as Lb
from rl.env import FrictionEnv
from rl.ppo import ActorCritic, deterministic_eval


def main():
    val = Lb.load(Path("runs/rl/lib_val_v1.npz"))
    for src in map(Path, sys.argv[1:]):
        stem = src.name.replace("_snaps.pt", "")
        out = src.with_name(stem + "_snapval.json")
        if out.exists():
            continue
        meta = json.loads(src.with_name(stem + ".json").read_text())
        cfg = meta["cfg"]
        kind = {"P-T": "teacher", "P-TP": "teacher_preview", "P-DR": "dr"}[meta["method"]]
        env = FrictionEnv(val, len(val["track_id"]), 1, "cuda:0", seed=meta["seed"] + 5000,
                          reward_scale=cfg["reward_scale"], sequential=True)
        ac = ActorCritic(kind, width=cfg["width"]).cuda()
        res = {}
        for it, sd in sorted(torch.load(src, weights_only=False).items()):
            ac.load_state_dict(sd)
            res[int(it)] = deterministic_eval(ac, env)
        out.write_text(json.dumps(res))
        best_s = max(res, key=lambda k: res[k]["success"])
        best_r = max(res, key=lambda k: res[k]["return"])
        print(f"{stem}: best success it {best_s} ({res[best_s]['success']:.3f}); best return it {best_r} "
              f"({res[best_r]['return']:.0f}, success {res[best_r]['success']:.3f})", flush=True)


if __name__ == "__main__":
    main()
