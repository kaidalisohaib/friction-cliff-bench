"""Per-control-step latency of a learned history-only policy (student: GRU over the last W feature vectors + MLP
actor), for comparison with the MPPI latency table of scripts/bench_compute.py. Engineering measurement.

Cells: device in {cuda, cpu (1 thread), cpu (all threads)} x cars in {1, 64}; each cell `--reps` timed calls after a
warm-up, reported as median and the 2.5-97.5 percentile range with N. Refuses while another process computes on
the GPU unless --force. Writes docs/results/compute_policy.{json,md}.

PYTHONPATH=. uv run python scripts/bench_policy.py runs/rl_fix/stu_val_s0.pt [--reps 300] [--force]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from labelling.label import git_sha
from rl.features import N_FEAT
from scripts.bench_compute import other_gpu_jobs, summarize

OUT = Path("docs/results")


def cell(model, n: int, W: int, dev: str, reps: int) -> dict:
    f = torch.randn(n, N_FEAT, device=dev)
    h = torch.randn(n, W, N_FEAT, device=dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    with torch.no_grad():
        for _ in range(20):
            model.mean(f, h)
        sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model.mean(f, h)
            sync()
            ts.append(time.perf_counter() - t0)
    return summarize(np.array(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    busy = other_gpu_jobs()
    if busy and not a.force:
        raise SystemExit(f"GPU busy ({busy}); rerun when idle or pass --force")
    from eval.runner import load_policy
    model, kind, W = load_policy(Path(a.ckpt))
    assert kind == "student", kind
    model.eval()
    res = {"ckpt": a.ckpt, "W": W, "git": git_sha(), "reps": a.reps, "gpu_shared": bool(busy),
           "device": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(), "cells": []}
    nthreads = torch.get_num_threads()
    for dev, threads in (("cuda", None), ("cpu", 1), ("cpu", nthreads)):
        m = model.to(dev)
        if threads:
            torch.set_num_threads(threads)
        for n in (1, 64):
            s = cell(m, n, W, dev, a.reps)
            res["cells"].append({"device": dev, "threads": threads, "cars": n, **s})
            print(f"{dev} threads={threads} cars={n}: median {s['median'] * 1e3:.3f} ms "
                  f"[{s['lo'] * 1e3:.3f}, {s['hi'] * 1e3:.3f}] (N={s['n']})", flush=True)
    torch.set_num_threads(nthreads)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compute_policy.json").write_text(json.dumps(res, indent=1))
    L = ["# Learned-policy latency per control step (harness output of `scripts/bench_policy.py`)", "",
         f"Student (GRU over W = {W} feature vectors + MLP actor), `{a.ckpt}`. Median and 2.5–97.5% range of "
         f"N = {a.reps} timed calls; GPU shared during the run: {res['gpu_shared']}.", "",
         "| device | threads | cars | median [ms] | 2.5–97.5% [ms] |", "|---|---|---|---|---|"]
    for c in res["cells"]:
        L.append(f"| {c['device']} | {c['threads'] or '—'} | {c['cars']} | {c['median'] * 1e3:.3f} | "
                 f"{c['lo'] * 1e3:.3f}–{c['hi'] * 1e3:.3f} |")
    L += ["", "All-thread CPU cells contend with any other CPU load on the machine (here: "
          + ("another project's GPU job was running" if res["gpu_shared"] else "none") + "); the single-thread cells "
          "are the deployment-relevant ones."]
    (OUT / "compute_policy.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
