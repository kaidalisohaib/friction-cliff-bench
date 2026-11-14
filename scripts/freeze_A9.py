"""Freeze everything prereg amendment A9 needs before test pool v2 is labelled or evaluated -> ledgers/A9_frozen.json.

- SHA-256 of every follow-up checkpoint (P-S2, P-SP2 students and their teachers), the A8 checkpoints P-S/P-SP and
  their teachers, the validation library and the dev split + labels.
- c' for M-Bc: linear interpolation of c against the dry pace tax over every M-B@c run on the dev split
  (runs/rl_fix/eval_dev/summary.jsonl), evaluated at the mean dev pace tax of the 10 P-SP2 students. Dev split only.
- P-SP2_top5_by_validation: the 5 P-SP2 seeds whose teacher had the highest deterministic validation success
  (validation library only; the exploratory subset of A9).

PYTHONPATH=. uv run python scripts/freeze_A9.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from labelling.label import git_sha


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    summ = [json.loads(x) for x in open("runs/rl_fix/eval_dev/summary.jsonl")]
    last = {d["name"]: d for d in summ}  # latest line per controller
    mb = sorted((float(n.split("@")[1]) if "@" in n else 1.0, d["pace_tax"]) for n, d in last.items()
                if n == "M-B" or n.startswith("M-B@"))
    psp2 = [last[f"stu_tpval_s{s}"]["pace_tax"] for s in range(10)]
    target = float(np.mean(psp2))
    cs, taxes = np.array([c for c, _ in mb]), np.array([t for _, t in mb])
    order = np.argsort(taxes)
    c_prime = float(np.interp(target, taxes[order], cs[order]))
    val = {}
    for s in range(10):
        h = json.loads(Path(f"runs/rl_fix/tpval_s{s}.json").read_text())["hist"]["val"]
        val[s] = max(v["success"] for v in h)
    top5 = sorted(val, key=lambda s: -val[s])[:5]
    files = ([f"runs/rl_fix/stu_tpval_s{s}.pt" for s in range(10)] + [f"runs/rl_fix/tpval_s{s}.pt" for s in range(10)]
             + [f"runs/rl_fix/stu_val_s{s}.pt" for s in range(10)] + [f"runs/rl_fix/val_s{s}.pt" for s in range(10)]
             + [f"runs/rl_fix/stu_base_s{s}.pt" for s in range(5, 10)] + [f"runs/rl_fix/base_s{s}.pt" for s in range(5, 10)]
             + [f"runs/final/{m}_s{s}.pt" for m in ("P-S", "P-SP", "P-T", "P-TP") for s in range(5)]
             + ["runs/rl/lib_val_v1.npz", "runs/rl/lib_train_v1.npz", "splits/dev_v1.jsonl",
                "runs/label_dev/labels.jsonl"])
    led = {"amendment": "A9", "git": git_sha(), "c_prime": c_prime,
           "c_prime_rule": "interp of c vs dev dry pace tax over M-B@c runs, at mean dev pace tax of P-SP2 (10 seeds)",
           "mb_dev_points": [{"c": c, "pace_tax": t} for c, t in mb], "psp2_dev_pace_tax_mean": target,
           "P-SP2_top5_by_validation": top5, "teacher_val_success": val,
           "sha256": {f: sha(f) for f in files}}
    Path("ledgers").mkdir(exist_ok=True)
    Path("ledgers/A9_frozen.json").write_text(json.dumps(led, indent=1))
    print(json.dumps({k: v for k, v in led.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
