"""Oracle settings pilot on the tuning split (never the test split). Oracles are frozen before labelling
(FEASIBILITY.md §3); this grid documents how the frozen values were chosen. Logged in LOG.md.

Run:  PYTHONPATH=. uv run python scripts/oracle_pilot.py --limit 40 --seeds 1 --out runs/oracle_pilot
"""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from control.mppi import MPPIConfig
from labelling.label import label_split
from scenarios import sampler as S
from scenarios import track as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--H", type=int, default=150)
    ap.add_argument("--grid", default="0.05,0.1|0.2,0.4|1,10|0,0.9")
    ap.add_argument("--pd-sample", action="store_true")
    ap.add_argument("--out", default="runs/oracle_pilot")
    a = ap.parse_args()
    _, scs = S.load_jsonl("splits/tuning_v1.jsonl")
    scs = scs[: a.limit]
    tracks = T.all_tracks()
    axes = [[float(v) for v in g.split(",")] for g in a.grid.split("|")]
    rows = []
    for ss, sl, lam, rho in itertools.product(*axes):
        cfg = MPPIConfig(K=a.K, H=a.H, sigma_steer=ss, sigma_long=sl, lam=lam, rho=rho, pd_sample=a.pd_sample)
        tag = f"ss{ss}_sl{sl}_lam{lam}_rho{rho}_K{a.K}_H{a.H}" + ("_pd" if a.pd_sample else "")
        labels, s1, meta = label_split(scs, tracks, Path(a.out) / tag, cfg, cfg, a.seeds, tag=tag, log=lambda m: None)
        c = Counter(lab["label"] for lab in labels)
        clear = [lab["or_best_clearance"] for lab in labels if lab["label"] == "reactive-feasible"]
        c20 = Counter(lab["label_d20"] for lab in labels)
        row = {"tag": tag, **c, "d20": dict(c20), "s1": len(s1), "median_clear_rf": float(np.median(clear)) if clear else None,
               "gpu_s": round(meta["gpu_s_or"] + meta["gpu_s_oc"])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (Path(a.out) / "summary.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
