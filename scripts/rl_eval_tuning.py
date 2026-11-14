"""Score controllers on the TUNING split for the RL follow-up (D35). Never reads the test split.

Sets: "main" = the reactive-feasible tuning scenarios (prereg A4 labels); "dry" = the same scenarios with the drop
removed (mu1 = mu0, dry surface; like eval/runner.py's dry set). Per controller and set the records are written to
runs/rl_fix/eval/<name>/<set>.jsonl; the summary line (departure on both sets, dry progress relative to O-R = pace
tax, mean arrival-speed ratio) is appended to runs/rl_fix/eval/summary.jsonl. Engineering numbers, not results.

Controllers:
  --ckpt PATH[,PATH...]     PPO teacher/DR checkpoints (scripts/rl_diag.py) or students (meta "teacher" path)
  --mppi O-R,M-B,M-B@0.8    reference MPPIs; "M-B@c" = bank MPPI with v_ref scaled by c (as M-B-c)

MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/rl_eval_tuning.py --mppi O-R,M-B --ckpt runs/rl_fix/base_s4.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from eval.methods import handover, run_controller, tuning_sets
from scenarios import sampler as S
from sim.params import VehicleParams

OUT = Path("runs/rl_fix/eval")


def sets():
    scs, rf = tuning_sets()
    rf = set(rf)
    main = [s for s in scs if s.sid in rf]
    return {"main": main, "dry": [replace(s, surface="dry", mu1=s.mu0) for s in main]}


def controller(spec: str, seed: int):
    from eval.runner import factory, load_policy
    from rl.policy import policy_controller
    if spec.endswith(".pt"):
        model, kind, W = load_policy(Path(spec))
        return lambda sim, sub, ss, sd: policy_controller(sim, sub, model, kind, W, sd)
    if "@" in spec:
        from control.adaptive import BankMPPI
        from eval.runner import best, mppi_cfg
        c = float(spec.split("@")[1])
        cfg, W = mppi_cfg("M-B"), best("M-B")["best_params"]["W"]
        return lambda sim, sub, ss, sd: BankMPPI(sim, sub, cfg, sd, ss, W, v_ref_scale=c)
    return factory(spec, seed, VehicleParams())


def name_of(spec: str) -> str:
    return Path(spec).stem if spec.endswith(".pt") else spec


def run(spec: str, seed: int, S_: dict, snaps: dict) -> dict:
    name = name_of(spec)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    recs = {}
    for k, scs in S_.items():
        f = d / f"{k}.jsonl"
        if f.exists() and sum(1 for _ in open(f)) == len(scs):
            recs[k] = [json.loads(x) for x in open(f)]
            continue
        r = run_controller(scs, snaps[k], controller(spec, seed), name, seed)
        recs[k] = [asdict(r[s.sid]) for s in scs]
        f.write_text("".join(json.dumps(x) + "\n" for x in recs[k]))
    return recs


def summarize(name: str, recs: dict, ref: dict | None, veff: dict) -> dict:
    m, dr = recs["main"], recs["dry"]
    out = {"name": name, "n": len(m), "dep": float(np.mean([r["departed"] for r in m])),
           "fail": float(np.mean([not r["success"] for r in m])),
           "dep_dry": float(np.mean([r["departed"] for r in dr])),
           "prog_dry": float(np.mean([r["progress"] for r in dr])),
           "arr_ratio": float(np.nanmean([r["arrival_v"] / veff[r["sid"]] for r in m if r["arrival_v"] is not None]))}
    if ref is not None:
        out["pace_tax"] = 1.0 - out["prog_dry"] / float(np.mean([r["progress"] for r in ref["dry"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--mppi", default="")
    ap.add_argument("--seed", type=int, default=0, help="observation-noise seed for MPPI references")
    a = ap.parse_args()
    S_ = sets()
    snaps = {k: handover(v, cache=OUT / f"handover_{k}.npz") for k, v in S_.items()}
    veff = {s.sid: S.v_eff(s) for s in S_["main"]}
    specs = [x for x in a.mppi.split(",") if x] + [x for x in a.ckpt.split(",") if x]
    ref = run("O-R", a.seed, S_, snaps)
    for spec in specs:
        t0 = time.time()
        seed = a.seed
        if spec.endswith(".pt"):
            import torch
            seed = int(torch.load(spec, weights_only=False)["meta"].get("seed", 0))
        recs = run(spec, seed, S_, snaps)
        s = summarize(name_of(spec), recs, ref, veff)
        s["wall_s"] = time.time() - t0
        with open(OUT / "summary.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
