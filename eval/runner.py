"""Confirmatory evaluation runner (EVAL_PROTOCOL.md, prereg A5): every method on identical scenario instances.

Scenario sets (all built from frozen split files; see `scenario_set`):
  main               the labelled test pool (every feasibility class), nominal truth model
  h2-<axis>-<level>  the pre-declared H2 subset with one truth-model mismatch (prereg V6):
                     axis "slip" alpha in {0.25, 0.5, 1}, "relax" sigma in {0.2, 0.4, 0.6} m,
                     "delay" +{3, 5, 7} control steps (+60/100/140 ms)
  dry                the H2 subset with the friction drop removed (mu1 = mu0, dry surface): dry pace

Methods and runs per scenario:
  O-R, O-C, M-N, M-B   MPPI, 5 noise seeds (O-R/O-C use the true model, also under mismatch)
  M-L                  estimator of training seed s + MPPI noise seed s, s = 0..4
  P-T, P-TP, P-DR, P-S, P-SP   policy of training seed s, observation-noise seed s, s = 0..4
One JSONL file per (set, method, seed): runs/eval/<set>/<method>_s<seed>.jsonl. Existing complete files are
skipped, so the run is resumable. MPPI methods use the backend selected by MPPI_BACKEND (cpp for confirmatory
runs, D29).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from control.controllers import ORACLE_OC, ORACLE_OR, mppi_controller
from control.mppi_cpp import use_cpp
from eval.methods import handover, run_controller
from labelling.label import git_sha
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams

TUNE = Path("runs/tuning")
FINAL = Path("runs/final")
MPPI_METHODS = ("O-R", "O-C", "M-N", "M-B", "M-L")
POLICY_KIND = {"P-T": "teacher", "P-TP": "teacher_preview", "P-DR": "dr", "P-S": "student", "P-SP": "student"}
H2_LEVELS = {"slip": (0.25, 0.5, 1.0), "relax": (0.2, 0.4, 0.6), "delay": (3, 5, 7)}
H2_SUBSET_N = 500
H2_SUBSET_SEED = 2027  # prereg A5


def truth_params(set_name: str) -> VehicleParams:
    p = VehicleParams()
    if not set_name.startswith("h2-"):
        return p
    _, axis, level = set_name.split("-")
    if axis == "slip":
        return p.with_(combined_alpha=float(level))
    if axis == "relax":
        return p.with_(sigma_relax=float(level))
    if axis == "delay":
        return p.with_(delay_steps=p.delay_steps + int(level))
    raise ValueError(set_name)


def h2_set_names() -> list[str]:
    return [f"h2-{ax}-{lv}" for ax, lvs in H2_LEVELS.items() for lv in lvs]


def load_labels(label_files) -> dict:
    out = {}
    for lf in label_files:
        for line in open(lf):
            d = json.loads(line)
            out[d["sid"]] = d
    return out


def h2_subset(labels: dict, scs: list) -> list:
    """Pre-declared H2/dry subset: H2_SUBSET_N reactive-feasible scenarios drawn without replacement with
    H2_SUBSET_SEED from the reactive-feasible test scenarios in split-file order (prereg A5)."""
    rf = [sc for sc in scs if labels[sc.sid]["label"] == "reactive-feasible"]
    idx = np.sort(np.random.default_rng(H2_SUBSET_SEED).choice(len(rf), min(H2_SUBSET_N, len(rf)), replace=False))
    return [rf[i] for i in idx]


def scenario_set(set_name: str, split_files, label_files, limit: int = 0) -> list:
    scs = [sc for f in split_files for sc in S.load_jsonl(f)[1]]
    labels = load_labels(label_files)
    scs = [sc for sc in scs if sc.sid in labels]
    if limit:  # pipeline dry runs on the tuning split only (refused for test by scripts/eval_split.py)
        scs = scs[:limit]
    if set_name == "main":
        return scs
    sub = h2_subset(labels, scs)
    if set_name == "dry":
        return [replace(sc, surface="dry", mu1=sc.mu0) for sc in sub]
    return sub


def axis_exclusions(axis: str, sub: list, tracks, device, out_root: Path) -> list[str]:
    """Prereg A6: H2-subset scenarios in which the shared dry driver cannot reach the handover state under some
    level of this mismatch axis (the driver itself is destabilised, e.g. by +140 ms delay). They are dropped from
    every level of the axis, level 0 included, for every method, and their number is reported. Decided from the
    prefix drive only, before any method runs (cached in <out>/h2_excluded_<axis>.json)."""
    from control.episode import run_prefix
    f = out_root / f"h2_excluded_{axis}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d["subset"] == [sc.sid for sc in sub]:
            return d["sids"]
    bad = set()
    for lv in H2_LEVELS[axis]:
        _, info = run_prefix(sub, tracks, truth_params(f"h2-{axis}-{lv}"), device, strict=False)
        bad |= {sc.sid for sc, ok in zip(sub, info["done"]) if not ok}
    out_root.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"axis": axis, "rule": "prereg A6", "subset": [sc.sid for sc in sub],
                             "sids": sorted(bad)}, indent=1))
    return sorted(bad)


def best(m: str) -> dict:
    return json.loads((TUNE / f"best_{m}.json").read_text())


def mppi_cfg(m: str):
    from control.mppi import MPPIConfig
    bp = best("M-B" if m in ("M-B", "M-N") else m)["best_params"]
    return MPPIConfig(**{**ORACLE_OR.__dict__, "sigma_steer": bp["sigma_steer"], "sigma_long": bp["sigma_long"],
                         "lam": bp["lam"], "rho": bp["rho"]})


def load_policy(path: Path):
    from rl.ppo import ActorCritic, Student
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    if meta["kind"] == "student":
        tk = torch.load(meta["teacher"], weights_only=False)
        t = ActorCritic(tk["meta"]["kind"], width=tk["meta"]["cfg"]["width"])
        model = Student(t, meta["cfg"]["hidden"])
        model.load_state_dict(ck["state_dict"])
        return model.cuda(), "student", meta["cfg"]["hist_len"]
    model = ActorCritic(meta["kind"], width=meta["cfg"]["width"])
    model.load_state_dict(ck["state_dict"])
    return model.cuda(), meta["kind"], 1


def load_estimator(path: Path):
    from control.estimator import Estimator
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    est = Estimator(meta["W"], meta["hidden"])
    est.load_state_dict(ck["state_dict"])
    return est


def factory(method: str, seed: int, truth: VehicleParams):
    """controller factory (sim, scenarios, snapshot, seed) -> controller, for run_controller."""
    from control.adaptive import BankMPPI, LearnedMPPI
    from rl.policy import policy_controller
    if method in ("O-R", "O-C"):
        cfg = ORACLE_OR if method == "O-R" else ORACLE_OC
        return lambda sim, sub, ss, sd: mppi_controller(sim, method, cfg, sd, ss, model_params=truth)
    if method == "M-N":
        cfg = mppi_cfg("M-N")
        return lambda sim, sub, ss, sd: mppi_controller(sim, "M-N", cfg, sd, ss)
    if method == "M-B":
        cfg, W = mppi_cfg("M-B"), best("M-B")["best_params"]["W"]
        return lambda sim, sub, ss, sd: BankMPPI(sim, sub, cfg, sd, ss, W)
    if method == "M-L":
        cfg = mppi_cfg("M-L")
        est = load_estimator(FINAL / f"M-L_s{seed}.pt")
        return lambda sim, sub, ss, sd: LearnedMPPI(sim, sub, cfg, sd, ss, est)
    if method in POLICY_KIND:
        model, kind, W = load_policy(FINAL / f"{method}_s{seed}.pt")
        return lambda sim, sub, ss, sd: policy_controller(sim, sub, model, kind, W, sd)
    raise ValueError(method)


def run_set(set_name: str, methods: list[str], seeds: list[int], split_files, label_files, out_root: Path,
            device="cuda:0", batch=64, log=print, limit: int = 0) -> None:
    scs = scenario_set(set_name, split_files, label_files, limit)
    truth = truth_params(set_name)
    out = out_root / set_name
    out.mkdir(parents=True, exist_ok=True)
    tracks = T.all_tracks()
    excluded = []
    if set_name.startswith("h2-"):
        excluded = axis_exclusions(set_name.split("-")[1], scs, tracks, device, out_root)
        scs = [sc for sc in scs if sc.sid not in set(excluded)]
    snap = handover(scs, tracks, truth, device, cache=out / "handover.npz")
    meta = {"set": set_name, "n": len(scs), "truth": asdict(truth), "git": git_sha(),
            "mppi_backend": "cpp" if use_cpp() else "warp", "excluded_A6": excluded}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for m in methods:
        for seed in seeds:
            path = out / f"{m}_s{seed}.jsonl"
            if path.exists() and sum(1 for _ in open(path)) == len(scs):
                continue
            t0 = time.time()
            recs = run_controller(scs, snap, factory(m, seed, truth), m, seed, device, batch, tracks, truth)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for sc in scs:
                    r = asdict(recs[sc.sid])
                    r["extra"] = {"set": set_name, "gpu_s_batch": None}
                    f.write(json.dumps(r) + "\n")
            tmp.rename(path)
            dep = np.mean([recs[sc.sid].departed for sc in scs])
            log(f"[{set_name}] {m} seed {seed}: {len(scs)} runs in {time.time() - t0:.0f}s "
                f"(departure {dep:.3f}; engineering log, not a result)")
            with open(out / "timing.jsonl", "a") as f:
                f.write(json.dumps({"method": m, "seed": seed, "wall_s": time.time() - t0, "n": len(scs),
                                    "backend": meta["mppi_backend"]}) + "\n")


def run_trace_set(methods: list[str], seeds: list[int], split_files, label_files, out_root: Path, device="cuda:0",
                  batch=64, log=print, limit: int = 0) -> None:
    """Exploratory recovery-latency pass (A5.9): the H2 subset, nominal truth, with per-step traces."""
    from eval.latency import run_traces, save_traces
    labels = load_labels(label_files)
    scs = [sc for f in split_files for sc in S.load_jsonl(f)[1] if sc.sid in labels]
    if limit:
        scs = scs[:limit]
    scs = h2_subset(labels, scs)
    truth = VehicleParams()
    out = out_root / "trace"
    tracks = T.all_tracks()
    snap = handover(scs, tracks, truth, device, cache=out / "handover.npz")
    for m in methods:
        for seed in seeds:
            path = out / f"{m}_s{seed}.npz"
            if path.exists():
                continue
            t0 = time.time()
            save_traces(path, run_traces(scs, snap, factory(m, seed, truth), m, seed, truth, device, batch))
            log(f"[trace] {m} seed {seed}: {len(scs)} runs in {time.time() - t0:.0f}s")
