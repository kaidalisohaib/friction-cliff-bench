"""Feasibility labelling (FEASIBILITY.md §1–4, prereg A3) and the oracle-run pipeline.

Pipeline for a split file:
  1. handover states: `run_prefix` for every scenario (deterministic, shared by every method)
  2. W-PD (the prefix path follower continuing, no friction information) once, and O-R with N seeds
  3. O-C only where every O-R seed failed (the label of an O-R success does not depend on O-C), until the first
     O-C success (one witness is enough) or N seeds
  4. labels by precedence: reactive-feasible (O-R or W-PD, A4.3) > preview-required > certified-infeasible
     (certificate at v_eff, A4.2) > undetermined; flags certified-reactive-infeasible, adaptation-required
     (reactive-feasible and W-PD fails, A4.5), marginal (best O-R seed's clearance < 0.30 m, prereg V3)
  5. gate S1 per run at the run's actual arrival state (prereg A3)
Records are JSON lines with the config hash and git SHA (EVAL_PROTOCOL.md §4).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from control.controllers import mppi_controller
from control.episode import (T_AFTER_DROP_INF, T_AFTER_EXIT, T_HANDOVER, prefix_action, run_episodes, run_prefix,
                             speed_profile)
from control.mppi import MPPIConfig
from feasibility.certificate import certify_scenario
from scenarios.sampler import A_LAT_FRAC, v_eff
from sim.params import VehicleParams, fingerprint
from sim.warp_sim import ScenarioSim

MARGINAL_BAND = 0.30  # prereg V3


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def est_duration(sc) -> float:
    after = T_AFTER_DROP_INF if sc.patch_len >= 1e8 else (sc.patch_len + sc.transition) / max(sc.v0, 1.0) + T_AFTER_EXIT
    return T_HANDOVER + after


def batches(scs, size):
    """Group scenarios of similar expected duration (a batch runs until its longest episode ends)."""
    order = sorted(range(len(scs)), key=lambda i: est_duration(scs[i]))
    return [order[i:i + size] for i in range(0, len(order), size)]


def run_method(scs, snap, method, seeds, cfg: MPPIConfig, tracks, p, device, batch, log=None, v_ref=None):
    """All (scenario, seed) runs of one method. Returns {sid: [RunRecord per seed]}."""
    out = {sc.sid: [] for sc in scs}
    for seed in seeds:
        for idx in batches(scs, batch):
            sub = [scs[i] for i in idx]
            ssnap = {k: v[idx] for k, v in snap.items()}
            sim = ScenarioSim(sub, p, tracks, device)
            vr = None if v_ref is None else np.asarray(v_ref)[idx]
            recs = run_episodes(sub, ssnap, mppi_controller(sim, method, cfg, seed, ssnap, vr), method, seed, tracks, p,
                                device, sim=sim)
            for r in recs:
                out[r.sid].append(r)
            if log:
                log(f"{method} seed {seed}: {len(sub)} runs")
    return out


def run_pd_witness(scs, snap, tracks, p, device, batch, a_lat_frac=A_LAT_FRAC):
    """W-PD: the deterministic prefix path follower keeps driving after handover (no friction information at all).
    A success is a witness of feasibility without preview (D20, proposed)."""
    out = {}
    for idx in batches(scs, batch):
        sub = [scs[i] for i in idx]
        ssnap = {k: v[idx] for k, v in snap.items()}
        prof = [speed_profile(tracks[s.track_id], s.v0, s.mu0, a_lat_frac) for s in sub]

        def ctrl(sim, k, sub=sub, prof=prof):
            return prefix_action(sim.get_frenet(), sim.get_state(), sub, tracks, prof, p)

        for r in run_episodes(sub, ssnap, ctrl, "W-PD", 0, tracks, p, device):
            out[r.sid] = r
    return out


def s1_check(sc, rec, tracks, p) -> dict | None:
    """Prereg A3: certificate at the run's actual arrival state. Returns a violation dict, or None."""
    if rec.arrival_v is None or not rec.success:
        return None
    cert = certify_scenario(sc, rec.arrival_v, tracks, p, e0=rec.arrival_e, edot0=rec.arrival_edot)
    if cert.certified:
        return {"sid": sc.sid, "method": rec.method, "seed": rec.seed, "arrival_v": rec.arrival_v,
                "arrival_e": rec.arrival_e, "arrival_edot": rec.arrival_edot}
    return None


def _label(reactive_ok: bool, oc_ok: bool, certified: bool) -> str:
    if reactive_ok:
        return "reactive-feasible"
    if oc_ok:
        return "preview-required"
    return "certified-infeasible" if certified else "undetermined"


def assign_label(sc, or_recs, oc_recs, tracks, p, pd_rec=None) -> dict:
    """`label` (prereg A4.3, D20): reactive witnesses are O-R (>= 1 of N seeds) or W-PD, the friction-blind path
    follower; preview witness O-C; certificate at the nominal arrival state (speed v_eff, A4.2).
    `label_or_only`: the A3 rule (O-R is the only reactive witness), logged for comparison."""
    ve = v_eff(sc)
    cert = certify_scenario(sc, ve, tracks, p)  # nominal arrival state at v_eff (prereg A4.2)
    or_ok = [r for r in or_recs if r.success]
    oc_ok = [r for r in oc_recs if r.success]
    pd_ok = bool(pd_rec is not None and pd_rec.success)
    label = _label(bool(or_ok) or pd_ok, bool(oc_ok), cert.certified)
    label_or = _label(bool(or_ok), bool(oc_ok), cert.certified)
    best_clear = max((r.min_clearance for r in or_recs), default=None)  # prereg V3: best O-R seed
    return {"sid": sc.sid, "v_eff": ve, "label": label, "label_or_only": label_or, "pd_success": pd_ok,
            "adaptation_required": bool(label == "reactive-feasible" and not pd_ok),
            "certified_nominal": bool(cert.certified), "cert_reason": cert.reason,
            "certified_reactive_infeasible": bool(label == "preview-required" and cert.certified),
            "marginal": bool(label == "reactive-feasible" and best_clear is not None and best_clear < MARGINAL_BAND),
            "or_best_clearance": best_clear, "n_or_success": len(or_ok), "n_or": len(or_recs),
            "n_oc_success": len(oc_ok), "n_oc": len(oc_recs)}


def label_split(scs, tracks, out_dir: Path, cfg_or: MPPIConfig, cfg_oc: MPPIConfig, n_seeds=5,
                p: VehicleParams = VehicleParams(), device="cuda:0", batch=32, tag="", log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git_sha(), "params_fingerprint": fingerprint(), "O-R": asdict(cfg_or), "O-C": asdict(cfg_oc),
            "n_seeds": n_seeds, "t_handover": T_HANDOVER, "n_scenarios": len(scs), "tag": tag}
    meta["config_hash"] = config_hash({k: v for k, v in meta.items() if k not in ("git", "tag")})
    t0 = time.time()
    snap, info = run_prefix(scs, tracks, p, device)
    log(f"prefix done in {time.time() - t0:.0f}s")
    seeds = list(range(n_seeds))
    pd_runs = run_pd_witness(scs, snap, tracks, p, device, batch)
    t1 = time.time()
    or_runs = run_method(scs, snap, "O-R", seeds, cfg_or, tracks, p, device, batch, log)
    meta["gpu_s_or"] = time.time() - t1
    # O-C is only needed where O-R has no success (so that both `label` and `label_or_only` are complete)
    need_oc = [i for i, sc in enumerate(scs) if not any(r.success for r in or_runs[sc.sid])]
    oc_runs = {sc.sid: [] for sc in scs}
    t2 = time.time()
    remaining = list(need_oc)
    for seed in seeds:  # until the first O-C success per scenario
        if not remaining:
            break
        sub = [scs[i] for i in remaining]
        ssnap = {k: v[remaining] for k, v in snap.items()}
        res = run_method(sub, ssnap, "O-C", [seed], cfg_oc, tracks, p, device, batch, log)
        for sid, rr in res.items():
            oc_runs[sid].extend(rr)
        remaining = [i for i in remaining if not any(r.success for r in oc_runs[scs[i].sid])]
    meta["gpu_s_oc"] = time.time() - t2
    labels, s1 = [], []
    with open(out_dir / "records.jsonl", "w") as f:
        for sc in scs:
            for r in or_runs[sc.sid] + oc_runs[sc.sid] + [pd_runs[sc.sid]]:
                f.write(json.dumps({**asdict(r), "config_hash": meta["config_hash"], "git": meta["git"]}) + "\n")
                v = s1_check(sc, r, tracks, p)
                if v:
                    s1.append(v)
            labels.append({**assign_label(sc, or_runs[sc.sid], oc_runs[sc.sid], tracks, p, pd_runs[sc.sid]),
                           "t_prefix": float(info["t_prefix"][scs.index(sc)])})
    with open(out_dir / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    meta["s1_violations"] = s1
    meta["wall_s"] = time.time() - t0
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return labels, s1, meta
