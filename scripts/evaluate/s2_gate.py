"""Gate S2 (prereg V5): the PPO teacher P-T's lane-departure rate is within O-R's + 5 pp (point estimate, pooled over
>= 5 training seeds) on >= 200 reactive-feasible tuning scenarios. O-R's rate is pooled over its 5 labelling
seeds (the labelling records of both tuning blocks). Also reports the other final learned methods (tuning split
only; engineering, not results).

PYTHONPATH=. uv run python scripts/evaluate/s2_gate.py [--methods P-T,P-S,P-DR,P-TP,P-SP]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval.methods import handover, run_controller, tuning_sets, write_records
from eval.stats import wilson
from rl.policy import policy_controller
from rl.ppo import ActorCritic, Student

FINAL = Path("runs/final")
LABEL_RUNS = [Path("runs/label_tuning_veff/records.jsonl"), Path("runs/label_tuning_b2/records.jsonl")]


def load_model(path):
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    if meta["kind"] == "student":
        tk = torch.load(meta["teacher"], weights_only=False)
        t = ActorCritic(tk["meta"]["kind"], width=tk["meta"]["cfg"]["width"])
        m = Student(t, meta["cfg"]["hidden"])
        m.load_state_dict(ck["state_dict"])
        return m.cuda(), "student", meta["cfg"]["hist_len"]
    m = ActorCritic(meta["kind"], width=meta["cfg"]["width"])
    m.load_state_dict(ck["state_dict"])
    return m.cuda(), meta["kind"], 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="P-T")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    scs, rf = tuning_sets()
    rf = set(rf)
    ev = [s for s in scs if s.sid in rf]
    snap = handover(ev, cache=Path("runs/tuning/handover_tuning_rf.npz"))
    o_r = [json.loads(line) for p in LABEL_RUNS for line in open(p)]
    o_r = [r for r in o_r if r["method"] == "O-R" and r["sid"] in rf]
    dep_or = float(np.mean([r["departed"] for r in o_r]))
    out = {"n_rf_tuning": len(ev), "O-R": {"departure": dep_or, "n_runs": len(o_r),
                                           "wilson": wilson(int(sum(r["departed"] for r in o_r)), len(o_r))}}
    for m in a.methods.split(","):
        deps, n = 0, 0
        for seed in map(int, a.seeds.split(",")):
            model, kind, W = load_model(FINAL / f"{m}_s{seed}.pt")
            recs = run_controller(ev, snap, lambda sim, sub, ss, sd: policy_controller(sim, sub, model, kind, W, sd),
                                  m, seed)
            write_records(FINAL / f"eval_tuning_{m}_s{seed}.jsonl", recs)
            deps += sum(r.departed for r in recs.values())
            n += len(recs)
        out[m] = {"departure": deps / n, "n_runs": n, "wilson": wilson(deps, n)}
    if "P-T" in out:
        out["S2"] = {"margin_pp": 5.0, "pass": bool(out["P-T"]["departure"] <= dep_or + 0.05 and len(ev) >= 200)}
    (FINAL / "s2_gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
