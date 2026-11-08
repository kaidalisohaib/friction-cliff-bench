"""Run any controller (oracles, M-N, M-B, M-L, policies) on a scenario set from the shared handover states, and
the tuning objective (prereg V2: tuning only ever reads the tuning split).

A controller *factory* builds the per-batch controller: factory(sim, scenarios, snapshot, seed) -> controller(sim, k).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from control.episode import run_episodes, run_prefix
from labelling.label import batches
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams
from sim.warp_sim import ScenarioSim


def handover(scs, tracks=None, p=VehicleParams(), device="cuda:0", cache: Path | None = None) -> dict:
    """Handover snapshot for a scenario list (cached on disk by scenario ids)."""
    if cache is not None and cache.exists():
        z = dict(np.load(cache, allow_pickle=False))
        if list(z.pop("sid")) == [sc.sid for sc in scs]:
            return z
    snap, _ = run_prefix(scs, tracks or T.all_tracks(), p, device)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, sid=np.array([sc.sid for sc in scs]), **snap)
    return snap


def run_controller(scs, snap, factory, method: str, seed: int, device="cuda:0", batch=64, tracks=None,
                   p: VehicleParams = VehicleParams()) -> dict:
    tracks = tracks or T.all_tracks()
    out = {}
    for idx in batches(scs, batch):
        sub = [scs[i] for i in idx]
        ssnap = {k: v[idx] for k, v in snap.items()}
        sim = ScenarioSim(sub, p, tracks, device)
        sim.load(ssnap)
        ctrl = factory(sim, sub, ssnap, seed)
        for r in run_episodes(sub, ssnap, ctrl, method, seed, tracks, p, device, sim=sim):
            out[r.sid] = r
    return out


def objective(recs: dict, sids) -> float:
    """Tuning objective on the reactive-feasible tuning scenarios: failure rate (departure, loss of control or
    termination; the H1 metric family) plus a 1e-2 tie-break on mean time out of lane [s]."""
    rs = [recs[s] for s in sids]
    fail = np.mean([not r.success for r in rs])
    return float(fail + 1e-2 * np.mean([r.time_out_of_lane for r in rs]))


def tuning_sets(label_files=("labels/tuning_v1.jsonl", "labels/tuning_v1_b2.jsonl"),
                split_files=("splits/tuning_v1.jsonl", "splits/tuning_v1_b2.jsonl"), which="label"):
    """All tuning scenarios and the ids of the reactive-feasible ones (prereg A4 label)."""
    scs, rf = [], []
    for sf, lf in zip(split_files, label_files):
        _, ss = S.load_jsonl(sf)
        scs += ss
        if Path(lf).exists():
            rf += [d["sid"] for d in map(json.loads, open(lf)) if d[which] == "reactive-feasible"]
    return scs, rf


def write_records(path: Path, recs: dict, extra: dict | None = None) -> None:
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs.values():
            f.write(json.dumps({**asdict(r), **(extra or {})}) + "\n")
