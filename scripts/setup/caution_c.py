"""Exploratory E7 (prereg A7): the speed scale c for M-B-c, frozen before any test evaluation.

c = mean over the five final P-S seeds and the reactive-feasible TUNING scenarios of (arrival speed of the front
axle at the drop) / v_eff. Runs that never reach the drop are excluded. Input: the tuning-split main-set runs of
the final P-S checkpoints:
  MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/evaluate/eval_split.py --split tuning --sets main --methods P-S \\
      --out runs/eval_tuning_final
  PYTHONPATH=. uv run python scripts/setup/caution_c.py
Writes runs/final/caution_c.json (read by eval/runner.py for M-B-c) and a committed copy in ledgers/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.methods import tuning_sets
from labelling.label import git_sha
from scenarios import sampler as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/eval_tuning_final/main")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    scs, rf = tuning_sets()
    veff = {sc.sid: S.v_eff(sc) for sc in scs}
    rf = set(rf)
    ratios, per_seed = [], {}
    for s in a.seeds.split(","):
        rs = [json.loads(line) for line in open(Path(a.root) / f"P-S_s{s}.jsonl")]
        r = [d["arrival_v"] / veff[d["sid"]] for d in rs if d["sid"] in rf and d["arrival_v"] is not None]
        per_seed[s] = float(np.mean(r))
        ratios += r
    c = float(np.mean(ratios))
    rec = {"c": c, "per_seed": per_seed, "n_runs": len(ratios), "source": a.root, "split": "tuning (RF)",
           "git": git_sha(), "rule": "prereg A7 (E7)"}
    for p in (Path("runs/final/caution_c.json"), Path("ledgers/caution_c.json")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
