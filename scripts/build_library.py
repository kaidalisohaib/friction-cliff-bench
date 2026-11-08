"""Build the training-scenario library (rl/library.py): handover states + DR draws, saved to runs/ (gitignored).

PYTHONPATH=. uv run python scripts/build_library.py --n 16384 --out runs/rl/lib_train_v1.npz
"""
import argparse
import time
from pathlib import Path

from rl import library as Lb
from scenarios import sampler as S

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16384)
ap.add_argument("--out", default="runs/rl/lib_train_v1.npz")
ap.add_argument("--no-dr", action="store_true")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
t0 = time.time()
lib = Lb.build_library(a.n, S.TRAIN_SEED, "R", dr=not a.no_dr, device=a.device)
Lb.save(lib, Path(a.out))
print(f"library {a.n} entries in {time.time() - t0:.0f}s -> {a.out}")
