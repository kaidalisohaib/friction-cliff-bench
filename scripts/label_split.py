"""Run the oracles on a split file and write records, labels and S1 results (labelling/label.py).

Default: the frozen oracle settings of prereg A4 (`control/controllers.py: ORACLE_OR, ORACLE_OC`, 5 seeds).
`--explore` unlocks the settings flags for pilots on the tuning split; it is refused for `test_*` files, so the
test pool can only ever be labelled with the frozen settings.

Run:  PYTHONPATH=. uv run python scripts/label_split.py splits/test_v1_T0.jsonl --out runs/label_test_T0
      PYTHONPATH=. uv run python scripts/label_split.py splits/tuning_v1.jsonl --out runs/x --explore --K 4096
"""

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

from control.controllers import ORACLE_OC, ORACLE_OR
from labelling.label import label_split
from scenarios import sampler as S
from scenarios import track as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--explore", action="store_true", help="allow the settings flags below (never on test files)")
    ap.add_argument("--K", type=int)
    ap.add_argument("--H", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--sig-steer", type=float)
    ap.add_argument("--sig-long", type=float)
    ap.add_argument("--rho", type=float)
    ap.add_argument("--oc-rho", type=float)
    ap.add_argument("--n-iter", type=int)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated scenario ids (explore only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    over = {k: getattr(a, k2) for k, k2 in [("K", "K"), ("H", "H"), ("lam", "lam"), ("sigma_steer", "sig_steer"),
                                            ("sigma_long", "sig_long"), ("n_iter", "n_iter")] if getattr(a, k2) is not None}
    changed = bool(over) or a.rho is not None or a.oc_rho is not None or a.seeds != 5 or a.limit or a.only
    if Path(a.split).name.startswith("test") and (a.explore or changed):
        raise SystemExit("test files are labelled only with the frozen prereg-A4 settings (no flags)")
    if changed and not a.explore:
        raise SystemExit("settings flags need --explore")
    cfg = replace(ORACLE_OR, **over, **({"rho": a.rho} if a.rho is not None else {}))
    cfg_oc = replace(ORACLE_OC, **over, **({"rho": a.oc_rho} if a.oc_rho is not None else {}))
    _, scs = S.load_jsonl(a.split)
    if a.only:
        keep = set(a.only.split(","))
        scs = [s for s in scs if s.sid in keep]
    if a.limit:
        scs = scs[: a.limit]
    labels, s1, meta = label_split(scs, T.all_tracks(), Path(a.out), cfg, cfg_oc, a.seeds, device=a.device,
                                   batch=a.batch, tag=a.tag or ("frozen-A4" if not changed else "explore"))
    print("labels:", Counter(lab["label"] for lab in labels))
    print("O-R-only labels:", Counter(lab["label_or_only"] for lab in labels))
    print("adaptation-required:", sum(lab["adaptation_required"] for lab in labels))
    print("S1 violations:", len(s1), "| O-R GPU s:", round(meta["gpu_s_or"]), "| O-C GPU s:", round(meta["gpu_s_oc"]))


if __name__ == "__main__":
    main()
