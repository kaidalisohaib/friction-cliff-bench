"""S1 probe set (D23, prereg A4.6): make gate S1 non-vacuous.

S1 (prereg A3): no run that *arrives* in a state the certificate certifies may stay in the lane. On the labelled
splits almost no run arrives in a certified state, so S1 is rarely exercised there. This probe builds scenarios
that are not part of any split (own seed), where cars arrive fast at apex drops onto snow/ice, so many runs
arrive in certified states:
  - curve tracks R in {50, 100} m, apex drop, snow or ice, patch 100 m or endless, step or 2 m ramp;
  - v0 = sqrt(f * mu0 * g * R) with f ~ U[0.70, 0.88] (below the dry limit), and the shared dry driver uses
    a_lat_frac = 0.9 so it does not slow below v0 before the drop.
Controllers that cannot see the drop (W-PD, O-R, M-N) arrive at that speed. Every run is certified at its *actual*
arrival state. Pass criterion: at least MIN_CERTIFIED runs arrive certified, and none of them succeeds.

Usage: PYTHONPATH=. uv run python scripts/setup/s1_probe.py [--n 96] [--seeds 2] [--out runs/s1_probe]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from control.controllers import ORACLE_OR
from control.episode import run_prefix
from feasibility.certificate import certify_scenario
from labelling.label import run_method, run_pd_witness
from scenarios import sampler as S
from scenarios import track as T
from sim.params import MU_RANGES, VehicleParams

PROBE_SEED = S._seed("friction-cliff-bench s1 probe v1")
PROBE_FRAC = 0.9
MIN_CERTIFIED = 30


def probe_scenarios(n: int) -> list[S.Scenario]:
    rng = np.random.default_rng(PROBE_SEED)
    tracks = T.all_tracks()
    out = []
    for i in range(n):
        R = float(rng.choice([50.0, 100.0]))
        tid = T.RADII.index(R) + 1
        surf = str(rng.choice(["snow", "ice"]))
        lp = float(rng.choice([100.0, 1e9]))
        lt = float(rng.choice([0.0, 2.0]))
        mu0 = float(rng.uniform(*MU_RANGES["dry"]))
        mu1 = float(rng.uniform(*MU_RANGES[surf]))
        v0 = float(np.sqrt(rng.uniform(0.70, 0.88) * mu0 * 9.81 * R))
        out.append(S.Scenario(f"P-{i:05d}", -1, round(v0), f"R{int(R)}-apex", R, "apex", surf, lp, lt, v0, mu0, mu1,
                              tid, tracks[tid].s_apex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="runs/s1_probe")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p, tracks = VehicleParams(), T.all_tracks()
    scs = probe_scenarios(a.n)
    t0 = time.time()
    snap, _ = run_prefix(scs, tracks, p, a.device, a_lat_frac=PROBE_FRAC)
    vref = np.array([sc.v0 for sc in scs], np.float32)
    runs = list(run_pd_witness(scs, snap, tracks, p, a.device, 32, a_lat_frac=PROBE_FRAC).values())
    for m in ("O-R", "M-N"):
        res = run_method(scs, snap, m, list(range(a.seeds)), ORACLE_OR, tracks, p, a.device, 32, v_ref=vref)
        runs += [r for rr in res.values() for r in rr]
    sc_by = {sc.sid: sc for sc in scs}
    tally, viol = Counter(), []
    with open(out / "records.jsonl", "w") as f:
        for r in runs:
            cert = (r.arrival_v is not None and certify_scenario(sc_by[r.sid], r.arrival_v, tracks, p,
                                                                  e0=r.arrival_e, edot0=r.arrival_edot).certified)
            tally[(r.method, "certified" if cert else "not-certified", "success" if r.success else "fail")] += 1
            if cert and r.success:
                viol.append(asdict(r))
            f.write(json.dumps({**asdict(r), "arrival_certified": bool(cert)}) + "\n")
    n_cert = sum(v for k, v in tally.items() if k[1] == "certified")
    verdict = "PASS" if (n_cert >= MIN_CERTIFIED and not viol) else "FAIL"
    summary = {"verdict": verdict, "n_runs": len(runs), "n_certified_arrivals": n_cert, "violations": viol,
               "tally": {"|".join(k): v for k, v in sorted(tally.items())}, "wall_s": time.time() - t0,
               "probe_seed": PROBE_SEED, "n_scenarios": a.n, "seeds": a.seeds}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
