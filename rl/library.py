"""Training-scenario library for the learned methods (BASELINES.md §4).

Training scenarios come from the tuning sampler spec with the training seed ("friction-cliff-bench train v1";
never the test seed). Each one is driven by the shared dry driver to its handover state (D19), exactly like the
evaluation episodes, and stored on disk. A library entry also carries one domain-randomization (DR) draw:
  mass ±15%, yaw inertia ±15%, CoG height ±10%, lateral tire stiffness scale (B, both axles independently) ±15%,
  extra pure actuator delay 0–2 control steps (0–40 ms, D14).
(BASELINES §4 lists "tire shape (B, C, E) ±15%"; the simulator exposes only the per-axle B scale as a per-car
parameter, so C and E are not randomized: D25.) The prefix runs with nominal parameters; the DR parameters
apply from handover, when the friction drop is still >= 4 s away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from control.episode import run_prefix
from scenarios import sampler as S
from scenarios import track as T
from sim.params import VehicleParams

DR_RANGES = {"m": 0.15, "Iz": 0.15, "h": 0.10, "B_scale_f": 0.15, "B_scale_r": 0.15}
DR_DELAY_EXTRA = (0, 2)  # inclusive range of extra control steps
SURF_IDX = {"dry": 0, "wet": 1, "snow": 2, "ice": 3}


def dr_params(rng: np.random.Generator, base: VehicleParams = VehicleParams()) -> tuple[VehicleParams, np.ndarray]:
    """One DR draw. Returns the params and the normalized DR vector (each entry in [-1, 1]) for privileged inputs."""
    u = rng.uniform(-1.0, 1.0, len(DR_RANGES))
    kw = {k: getattr(base, k) * (1.0 + r * uu) for (k, r), uu in zip(DR_RANGES.items(), u)}
    extra = int(rng.integers(DR_DELAY_EXTRA[0], DR_DELAY_EXTRA[1] + 1))
    kw["delay_steps"] = base.delay_steps + extra
    vec = np.concatenate([u, [extra - 1.0]]).astype(np.float32)
    return base.with_(**kw), vec


def build_library(n: int, seed: int = S.TRAIN_SEED, prefix: str = "R", dr: bool = True, device: str = "cuda:0",
                  batch: int = 1024, dr_seed: int = 0) -> dict:
    scs = S.sample(n, seed, prefix)
    tracks = T.all_tracks()
    snaps = []
    for i in range(0, n, batch):
        snap, _ = run_prefix(scs[i:i + batch], tracks, VehicleParams(), device)
        snaps.append(snap)
    lib = {k: np.concatenate([s[k] for s in snaps]) for k in snaps[0]}
    rng = np.random.default_rng(dr_seed)
    dvec = np.zeros((n, len(DR_RANGES) + 1), np.float32)
    plist = []
    for j in range(n):
        if dr:
            pp, dvec[j] = dr_params(rng)
        else:
            pp = VehicleParams()
        plist.append(pp)
    lib["dr_vec"] = dvec
    lib["dr_params"] = np.array([[pp.m, pp.Iz, pp.h, pp.B_scale_f, pp.B_scale_r, pp.delay_steps] for pp in plist],
                                np.float32)
    lib["prof"] = np.array([[sc.s_drop, sc.transition, sc.patch_len, sc.mu0, sc.mu1, SURF_IDX[sc.surface], 0, 0]
                            for sc in scs], np.float32)
    lib["track_id"] = np.array([sc.track_id for sc in scs], np.int32)
    lib["v_ref"] = np.array([S.v_eff(sc) for sc in scs], np.float32)
    lib["sid"] = np.array([sc.sid for sc in scs])
    return lib


def params_from_row(row: np.ndarray) -> VehicleParams:
    m, Iz, h, bf, br, d = row.tolist()
    return VehicleParams().with_(m=m, Iz=Iz, h=h, B_scale_f=bf, B_scale_r=br, delay_steps=int(round(d)))


def save(lib: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **lib)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def from_scenarios(scs, snap: dict) -> dict:
    """A nominal-parameter 'library' from evaluation scenarios and their handover snapshot (for evaluation)."""
    lib = dict(snap)
    p = VehicleParams()
    n = len(scs)
    lib["dr_vec"] = np.zeros((n, len(DR_RANGES) + 1), np.float32)
    lib["dr_vec"][:, -1] = -1.0  # extra delay 0
    lib["dr_params"] = np.tile(np.array([p.m, p.Iz, p.h, p.B_scale_f, p.B_scale_r, p.delay_steps], np.float32), (n, 1))
    lib["prof"] = np.array([[sc.s_drop, sc.transition, sc.patch_len, sc.mu0, sc.mu1, SURF_IDX[sc.surface], 0, 0]
                            for sc in scs], np.float32)
    lib["track_id"] = np.array([sc.track_id for sc in scs], np.int32)
    lib["v_ref"] = np.array([S.v_eff(sc) for sc in scs], np.float32)
    lib["sid"] = np.array([sc.sid for sc in scs])
    return lib
