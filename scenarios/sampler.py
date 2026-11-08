"""Scenario sampler, prereg V7 ("test-split sampler spec v1"), plus the tuning/training samplers.

Cells (full factorial, 1,872): v0 stratum {10, 15, 20, 25} m/s x geometry {straight; R in {50, 100, 200, 400} m
x drop at {entry, apex, exit}} (13) x post-drop surface {wet, snow, ice} x patch length {10, 30, 100, inf} m
x transition {step, ramp 2 m, ramp 5 m}. Within a cell: v0 = stratum + U[-1, 1]; mu0 ~ U[0.85, 1.0] (dry shape);
mu1 ~ U[surface range] with that surface's shape. Cells are drawn uniformly at random with replacement.

Implementation choices left open by the prereg (fixed here, written into every file header):
- tracks: shared tables from scenarios/track.py (lead-in 70 m, clothoids R/3 clipped to [15, 80] m, 90-degree arcs);
- drop locations: entry = start of the clothoid in, apex = middle of the arc, exit = end of the arc;
  straight = 70 m after the start;
- the episode starts at s = 0 in straight-line steady state at v0 (so >= 2 s before every drop);
- random stream: numpy.random.default_rng(seed) (PCG64), one draw sequence per scenario in a fixed order.
The output is canonical JSON lines (sorted keys, floats with repr); its SHA-256 identifies the set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from scenarios import track as T
from scenarios.friction import INF_PATCH
from sim.params import MU_RANGES

SPEC_VERSION = "v1"
V_STRATA = (10.0, 15.0, 20.0, 25.0)
GEOMETRIES = [("straight", 0.0, "straight")] + [(f"R{int(R)}-{loc}", R, loc) for R in T.RADII for loc in ("entry", "apex", "exit")]
SURF_POST = ("wet", "snow", "ice")
PATCH_LENGTHS = (10.0, 30.0, 100.0, float("inf"))
TRANSITIONS = (0.0, 2.0, 5.0)
N_CELLS = len(V_STRATA) * len(GEOMETRIES) * len(SURF_POST) * len(PATCH_LENGTHS) * len(TRANSITIONS)


def _seed(text: str) -> int:
    """Nothing-up-my-sleeve seeds: first 8 hex digits of sha256(text)."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


TEST_SEED = _seed("friction-cliff-bench test split v1")  # 4019369836 (prereg V7)
TUNING_SEED = _seed("friction-cliff-bench tuning split v1")  # 4210175585
TRAIN_SEED = _seed("friction-cliff-bench train v1")

_TRACKS = None


def tracks():
    global _TRACKS
    if _TRACKS is None:
        _TRACKS = T.all_tracks()
    return _TRACKS


@dataclass(frozen=True)
class Scenario:
    sid: str
    cell: int
    v_stratum: float
    geometry: str  # "straight" | "R50-entry" | ...
    R: float
    drop_loc: str
    surface: str
    patch_len: float  # INF_PATCH for an endless patch
    transition: float  # ramp length lt [m], 0 = step
    v0: float
    mu0: float
    mu1: float
    track_id: int
    s_drop: float


def decode_cell(c: int):
    c, it = divmod(c, len(TRANSITIONS))
    c, ip = divmod(c, len(PATCH_LENGTHS))
    c, isurf = divmod(c, len(SURF_POST))
    iv, ig = divmod(c, len(GEOMETRIES))
    return iv, ig, isurf, ip, it


def make_scenario(cell: int, rng: np.random.Generator, sid: str) -> Scenario:
    iv, ig, isurf, ip, it = decode_cell(cell)
    geo, R, loc = GEOMETRIES[ig]
    surf = SURF_POST[isurf]
    # fixed draw order: v0, mu0, mu1
    v0 = V_STRATA[iv] + rng.uniform(-1.0, 1.0)
    mu0 = rng.uniform(*MU_RANGES["dry"])
    mu1 = rng.uniform(*MU_RANGES[surf])
    track_id = 0 if R == 0.0 else 1 + T.RADII.index(R)
    tr = tracks()[track_id]
    s_drop = {"straight": T.LEAD_IN, "entry": tr.s_entry, "apex": tr.s_apex, "exit": tr.s_arc_end}[loc]
    Lp = PATCH_LENGTHS[ip]
    return Scenario(sid, cell, V_STRATA[iv], geo, R, loc, surf, INF_PATCH if np.isinf(Lp) else Lp,
                    TRANSITIONS[it], float(v0), float(mu0), float(mu1), track_id, float(s_drop))


def sample(n: int, seed: int, prefix: str) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cell = int(rng.integers(N_CELLS))
        out.append(make_scenario(cell, rng, f"{prefix}{i:05d}"))
    return out


def test_pool(block: int = 0) -> list[Scenario]:
    """Block 0 = the 4,000-scenario pool; blocks 1..3 = pre-declared top-ups of 1,000 (seed + block)."""
    if block == 0:
        return sample(4000, TEST_SEED, "T0-")
    assert 1 <= block <= 3
    return sample(1000, TEST_SEED + block, f"T{block}-")


def tuning_split() -> list[Scenario]:
    return sample(200, TUNING_SEED, "U-")


def header() -> dict:
    return {"spec": SPEC_VERSION, "n_cells": N_CELLS, "lead_in": T.LEAD_IN, "arc_angle_deg": 90.0,
            "clothoid": "R/3 clipped [15, 80] m", "ds": T.DS, "lane_width": T.LANE_WIDTH,
            "drop_locations": "entry=clothoid start, apex=arc middle, exit=arc end, straight=lead-in end",
            "rng": "numpy.random.default_rng(seed) PCG64; per scenario: integers(N_CELLS), uniform v0, mu0, mu1",
            "numpy": np.__version__, "inf_patch": INF_PATCH}


def to_jsonl(scenarios: list[Scenario]) -> str:
    lines = [json.dumps({"header": header()}, sort_keys=True)]
    lines += [json.dumps(asdict(s), sort_keys=True) for s in scenarios]
    return "\n".join(lines) + "\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_jsonl(path) -> tuple[dict, list[Scenario]]:
    """Read a committed split file (the authority, prereg A2): (header, scenarios)."""
    with open(path) as f:
        lines = f.read().splitlines()
    head = json.loads(lines[0])["header"]
    return head, [Scenario(**json.loads(line)) for line in lines[1:]]
