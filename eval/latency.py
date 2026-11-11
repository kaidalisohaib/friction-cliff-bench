"""Recovery latency (prereg V4; exploratory pass per A5.9) and the E4 axle-transit check.

Recovery latency of a run: time from the front axle reaching the drop until |e_y - e_y^O-R| <= 0.15 m and
|r - r^O-R| <= 0.05 rad/s hold for 0.5 s without interruption, comparing with O-R's trace on the same scenario
instance and the same seed index, time-aligned from the common handover state. Censored (reported as such) if the
band is never held for 0.5 s before either episode ends.

Traces: runs/eval_<split>/trace/<method>_s<seed>.npz with e_y and r per control step (NaN after the episode end),
plus, for the mechanism figure (exploratory), the true front-axle friction under the car (`mu_true`) and the
front-axle friction in the planner's model (`mu_hat`, adaptive MPPI only; NaN otherwise), sampled when the
controller is called at step k (the model it plans the action of step k with).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

from control.episode import run_episodes
from labelling.label import batches
from scenarios import track as T
from sim.params import CONTROL_DT
from sim.warp_sim import ScenarioSim

BAND_E, BAND_R, HOLD_S = 0.15, 0.05, 0.5  # V4


def run_traces(scs, snap, factory, method: str, seed: int, truth, device="cuda:0", batch=64):
    tracks = T.all_tracks()
    n = len(scs)
    E, Rr, MT, MH, t_drop = [None] * n, [None] * n, [None] * n, [None] * n, np.full(n, np.nan)
    for idx in batches(scs, batch):
        sub = [scs[i] for i in idx]
        ss = {k: v[idx] for k, v in snap.items()}
        sim = ScenarioSim(sub, truth, tracks, device)
        sim.load(ss)
        ctrl, mt, mh = factory(sim, sub, ss, seed), [], []

        def logged(sim_, k, ctrl=ctrl, mt=mt, mh=mh):
            a = ctrl(sim_, k)
            mt.append(sim_.get_axle_mu()[:, 0] if k > 0 else np.full(len(sub), np.nan))
            mp = getattr(ctrl, "mppi", None)
            if hasattr(ctrl, "sel") and mp is not None:  # adaptive MPPI: the model it plans with
                mh.append(wp.to_torch(mp.sf_fix).view(len(sub), 8)[:, 0].cpu().numpy().astype(np.float64))
            else:
                mh.append(np.full(len(sub), np.nan))
            return a

        recs, tr = run_episodes(sub, ss, logged, method, seed, tracks, truth, device, sim=sim, keep_trace=True)
        mt, mh = np.stack(mt, 1), np.stack(mh, 1)
        for j, i in enumerate(idx):
            L = int(round(recs[j].duration / CONTROL_DT))
            E[i] = tr[j, :L, 1].astype(np.float32)   # e_y
            Rr[i] = tr[j, :L, 9].astype(np.float32)  # yaw rate
            MT[i], MH[i] = mt[j, :L].astype(np.float32), mh[j, :L].astype(np.float32)
            t_drop[i] = np.nan if recs[j].t_drop is None else recs[j].t_drop
    L = max(len(e) for e in E)
    pad = lambda xs: np.stack([np.pad(x, (0, L - len(x)), constant_values=np.nan) for x in xs])
    return {"sid": np.array([sc.sid for sc in scs]), "e": pad(E), "r": pad(Rr), "mu_true": pad(MT),
            "mu_hat": pad(MH), "t_drop": t_drop}


def save_traces(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def recovery_latency(m: dict, o: dict) -> tuple[np.ndarray, np.ndarray]:
    """(latency [s], censored flag) per scenario for method traces m against O-R traces o (same order)."""
    assert (m["sid"] == o["sid"]).all()
    hold = int(round(HOLD_S / CONTROL_DT))
    n = len(m["sid"])
    lat, cens = np.full(n, np.nan), np.ones(n, bool)
    for i in range(n):
        td = m["t_drop"][i]
        if np.isnan(td):
            continue
        k0 = int(round(td / CONTROL_DT)) - 1
        e, r, eo, ro = m["e"][i], m["r"][i], o["e"][i], o["r"][i]
        L = max(len(e), len(eo))  # trace files of different methods are padded to different lengths
        e, r, eo, ro = (np.pad(x, (0, L - len(x)), constant_values=np.nan) for x in (e, r, eo, ro))
        ok = (np.abs(e - eo) <= BAND_E) & (np.abs(r - ro) <= BAND_R)  # NaN (either ended) -> False
        run = 0
        for k in range(max(k0, 0), len(ok)):
            run = run + 1 if ok[k] else 0
            if run >= hold:
                lat[i] = (k - hold + 1 - k0) * CONTROL_DT
                cens[i] = False
                break
        if cens[i]:
            last = np.where(np.isfinite(e) & np.isfinite(eo))[0]
            lat[i] = ((last[-1] if len(last) else k0) - k0) * CONTROL_DT
    return lat, cens
