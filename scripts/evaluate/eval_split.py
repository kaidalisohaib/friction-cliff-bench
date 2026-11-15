"""Run the evaluation sets (eval/runner.py) on a labelled split.

Confirmatory (test split; needs prereg A5 in the docs, the final checkpoints in runs/final/, the C++ backend):
  MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/evaluate/eval_split.py --split test --sets main,h2,dry

Dry run of the whole pipeline on the tuning split (engineering only, never reported as results):
  MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/evaluate/eval_split.py --split tuning --sets main --out runs/eval_tuning

Sets: main | h2 (all 9 mismatch levels) | h2-<axis>-<level> | dry | trace (exploratory recovery latency). Methods default to every method that a set
needs (main and dry: all; H2: O-R, M-B, M-L, P-S, P-DR, the methods in the H2 test).
"""

import argparse
import json
from pathlib import Path

from control.mppi_cpp import use_cpp
from eval import runner as R

SPLITS = {
    "test": (["splits/test_v1_T0.jsonl"], ["labels/test_v1_T0.jsonl"]),
    "tuning": (["splits/tuning_v1.jsonl", "splits/tuning_v1_b2.jsonl"],
               ["labels/tuning_v1.jsonl", "labels/tuning_v1_b2.jsonl"]),
}
ALL = ["O-R", "O-C", "M-N", "M-B", "M-L", "P-T", "P-TP", "P-DR", "P-S", "P-SP", "M-B-c"]  # M-B-c: E7 (A7)
H2_METHODS = ["O-R", "M-B", "M-L", "P-S", "P-DR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--sets", default="main")
    ap.add_argument("--methods", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="")
    ap.add_argument("--final-dir", default="", help="checkpoint dir (default runs/final)")
    ap.add_argument("--labels", default="", help="override label files (comma-separated)")
    ap.add_argument("--splits", default="", help="override split files (comma-separated), e.g. with top-up blocks")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N scenarios only (tuning dry runs)")
    a = ap.parse_args()
    splits, labels = SPLITS[a.split]
    if a.splits:
        splits = a.splits.split(",")
    if a.labels:
        labels = a.labels.split(",")
    if a.split == "test":
        prereg = Path("docs/PREREGISTRATION.md").read_text()
        if not all(f"### A{i}" in prereg for i in (5, 6, 7, 8)):
            raise SystemExit("test-split evaluation needs prereg amendments A5-A8 first")
        s2 = Path("runs/final/s2_gate.json")
        if not s2.exists():
            raise SystemExit("run scripts/evaluate/s2_gate.py on the final P-T seeds first (A5.2)")
        if not json.loads(s2.read_text())["S2"]["pass"]:
            print("NOTE: S2 failed on the final P-T seeds; proceeding under prereg A8 (reported with that caveat)")
        if not use_cpp():
            raise SystemExit("confirmatory runs use the C++ MPPI backend (D29): set MPPI_BACKEND=cpp")
        if a.final_dir or a.limit:
            raise SystemExit("the test split is evaluated in full, only with runs/final checkpoints")
    if a.final_dir:
        R.FINAL = Path(a.final_dir)
    want = a.methods.split(",") if a.methods else ALL
    if a.split == "test" and "M-B-c" in want and not (R.FINAL / "caution_c.json").exists():
        raise SystemExit("M-B-c needs c frozen from tuning data first (scripts/setup/caution_c.py, prereg A7)")
    out = Path(a.out or f"runs/eval_{a.split}")
    seeds = [int(s) for s in a.seeds.split(",")]
    sets = []
    for s in a.sets.split(","):
        sets += R.h2_set_names() if s == "h2" else [s]
    for s in sets:
        methods = a.methods.split(",") if a.methods else (H2_METHODS if s.startswith("h2-") else ALL)
        if s == "trace":  # exploratory recovery-latency pass (A5.9)
            R.run_trace_set(methods, seeds, splits, labels, out, a.device, a.batch, limit=a.limit)
        else:
            R.run_set(s, methods, seeds, splits, labels, out, a.device, a.batch, limit=a.limit)


if __name__ == "__main__":
    main()
