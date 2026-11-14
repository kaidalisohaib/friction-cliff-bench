"""Evaluation of the RL follow-up (prereg A9) on test pool v2 = the pre-declared top-up blocks T1–T3 (A2 hashes),
reactive-feasible scenarios only (frozen A4 oracle labels in runs/label_test_T{1,2,3}/labels.jsonl).

Sets: main (nominal truth) and dry (drop removed: mu1 = mu0, dry surface) for the pace tax. Methods (A9):
  O-R, M-B (5 seeds), M-Bc (bank MPPI at v_ref x c', c' frozen in ledgers/A9_frozen.json; 5 seeds),
  P-S, P-SP (confirmatory A8 checkpoints, runs/final; 5 seeds),
  P-S2, P-SP2 (follow-up students, runs/rl_fix/stu_val_s*.pt / stu_tpval_s*.pt; 10 seeds).
Refuses to run unless every checkpoint matches its SHA-256 in ledgers/A9_frozen.json. Resumable: complete
(method, seed) files are skipped. Batch 64 (D34); run with MPPI_BACKEND=cpp.

MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/eval_v2.py --sets main,dry [--methods P-SP2,M-Bc]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from eval.methods import handover, run_controller
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams

OUT = Path("runs/eval_v2")
LEDGER = Path("ledgers/A9_frozen.json")
BLOCKS = ("T1", "T2", "T3")
SEEDS = {"O-R": 5, "M-B": 5, "M-Bc": 5, "P-S": 10, "P-SP": 5, "P-S2": 10, "P-SP2": 10}
# P-S = the A8 recipe: seeds 0-4 are the confirmatory checkpoints; seeds 5-9 were trained by the same code and
# configuration during the follow-up (training is deterministic per seed: seeds 0-4 reproduce bit for bit).
CKPT = {"P-S": lambda s: f"runs/final/P-S_s{s}.pt" if s < 5 else f"runs/rl_fix/stu_base_s{s}.pt",
        "P-SP": lambda s: f"runs/final/P-SP_s{s}.pt", "P-S2": lambda s: f"runs/rl_fix/stu_val_s{s}.pt",
        "P-SP2": lambda s: f"runs/rl_fix/stu_tpval_s{s}.pt"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def v2_rf() -> list:
    scs = []
    for b in BLOCKS:
        _, ss = S.load_jsonl(f"splits/test_v1_{b}.jsonl")
        lab = {d["sid"]: d["label"] for d in map(json.loads, open(f"runs/label_test_{b}/labels.jsonl"))}
        scs += [s for s in ss if lab[s.sid] == "reactive-feasible"]
    return scs


def check_frozen(led: dict) -> None:
    for m, pat in CKPT.items():
        for s in range(SEEDS[m]):
            p = pat(s)
            assert led["sha256"][p] == sha256(Path(p)), f"{p} differs from the A9 ledger"
            meta = __import__("torch").load(p, weights_only=False)["meta"]
            if meta.get("teacher"):
                t = meta["teacher"]
                assert led["sha256"][t] == sha256(Path(t)), f"teacher {t} differs from the A9 ledger"


def factory(m: str, seed: int, led: dict):
    from eval.runner import factory as base_factory, load_policy
    from rl.policy import policy_controller
    if m in ("O-R", "M-B"):
        return base_factory(m, seed, VehicleParams())
    if m == "M-Bc":
        from control.adaptive import BankMPPI
        from eval.runner import best, mppi_cfg
        cfg, W, c = mppi_cfg("M-B"), best("M-B")["best_params"]["W"], float(led["c_prime"])
        return lambda sim, sub, ss, sd: BankMPPI(sim, sub, cfg, sd, ss, W, v_ref_scale=c)
    model, kind, W = load_policy(Path(CKPT[m](seed)))
    return lambda sim, sub, ss, sd: policy_controller(sim, sub, model, kind, W, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="main,dry")
    ap.add_argument("--methods", default=",".join(SEEDS))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    led = json.loads(LEDGER.read_text())
    check_frozen(led)
    rf = v2_rf()
    tracks = T.all_tracks()
    for set_name in a.sets.split(","):
        scs = rf if set_name == "main" else [replace(s, surface="dry", mu1=s.mu0) for s in rf]
        out = OUT / set_name
        out.mkdir(parents=True, exist_ok=True)
        snap = handover(scs, tracks, cache=out / "handover.npz")
        for m in a.methods.split(","):
            for seed in range(SEEDS[m]):
                path = out / f"{m}_s{seed}.jsonl"
                if path.exists() and sum(1 for _ in open(path)) == len(scs):
                    continue
                t0 = time.time()
                recs = run_controller(scs, snap, factory(m, seed, led), m, seed, batch=a.batch, tracks=tracks)
                tmp = path.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps({**asdict(recs[s.sid]), "extra": {"set": set_name}}) + "\n"
                                       for s in scs))
                tmp.rename(path)
                print(f"[v2 {set_name}] {m} seed {seed}: {len(scs)} runs in {time.time() - t0:.0f}s "
                      f"(departure {np.mean([recs[s.sid].departed for s in scs]):.4f}; engineering log)", flush=True)


if __name__ == "__main__":
    main()
