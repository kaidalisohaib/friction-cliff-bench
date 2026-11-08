"""Run the oracles on a split file and write records, labels and S1 results (labelling/label.py).

The test pool is only labelled with oracle settings frozen by a dated prereg amendment (V7 compute note, D18);
this script refuses `test_*` files unless --allow-test is given.

Run:  PYTHONPATH=. uv run python scripts/label_split.py splits/tuning_v1.jsonl --out runs/label_tuning [--K 4096 --H 150]
"""

import argparse
from collections import Counter
from pathlib import Path

from control.mppi import MPPIConfig
from labelling.label import label_split
from scenarios import sampler as S
from scenarios import track as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--H", type=int, default=150)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--sig-steer", type=float, default=0.3)
    ap.add_argument("--sig-long", type=float, default=0.4)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--ess", type=float, default=0.0)
    ap.add_argument("--n-iter", type=int, default=1)
    ap.add_argument("--pd-sample", action="store_true")
    ap.add_argument("--oc-rho", type=float, default=None, help="O-C AR(1) rho (default: same as --rho)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if Path(a.split).name.startswith("test") and not a.allow_test:
        raise SystemExit("refusing to label a test file without frozen oracle settings (--allow-test)")
    _, scs = S.load_jsonl(a.split)
    if a.limit:
        scs = scs[: a.limit]
    cfg = MPPIConfig(K=a.K, H=a.H, sigma_steer=a.sig_steer, sigma_long=a.sig_long, lam=a.lam, rho=a.rho, ess_target=a.ess, n_iter=a.n_iter, pd_sample=a.pd_sample)
    cfg_oc = cfg if a.oc_rho is None else MPPIConfig(**{**cfg.__dict__, "rho": a.oc_rho})
    labels, s1, meta = label_split(scs, T.all_tracks(), Path(a.out), cfg, cfg_oc, a.seeds, device=a.device,
                                   batch=a.batch, tag=a.tag)
    print("A3 labels:", Counter(lab["label"] for lab in labels))
    print("D20 labels:", Counter(lab["label_d20"] for lab in labels))
    print("S1 violations:", len(s1), "| O-R GPU s:", round(meta["gpu_s_or"]), "| O-C GPU s:", round(meta["gpu_s_oc"]))


if __name__ == "__main__":
    main()
