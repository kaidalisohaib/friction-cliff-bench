"""Controller factories with their information levels (BASELINES.md §1). Each returns controller(sim, k) -> actions.

  O-R  reactive oracle : MPPI, rollouts assume the true *current* per-axle surfaces, held over the horizon
  O-C  clairvoyant     : MPPI, rollouts see the true friction field ahead
  M-N  nominal         : MPPI, rollouts assume the nominal dry surface (mu = middle of the dry range)
"""

from __future__ import annotations

import hashlib

import numpy as np
import warp as wp

from control.mppi import MODE_FIELD, MODE_FIXED, MPPIConfig
from control.mppi_cpp import make_mppi
from sim.params import DRY, MU_RANGES, VehicleParams
from sim.warp_sim import ScenarioSim, surface_vec

MU_NOMINAL_DRY = 0.5 * sum(MU_RANGES["dry"])

# Oracle settings chosen on the tuning split (D21, PROPOSED; frozen by prereg A4 once approved)
ORACLE_OR = MPPIConfig(K=1024, H=150, sigma_steer=0.05, sigma_long=0.3, lam=10.0, rho=0.0, pd_sample=True)
ORACLE_OC = MPPIConfig(K=1024, H=150, sigma_steer=0.05, sigma_long=0.3, lam=10.0, rho=0.9, pd_sample=True,
                       mode=MODE_FIELD)


def run_seed(sid: str, method: str, seed: int) -> int:
    """Noise seed of one (scenario, method, seed) run: independent of batching and of the other methods."""
    return int(hashlib.sha256(f"{sid}|{method}|{seed}".encode()).hexdigest()[:7], 16)


def mppi_controller(sim: ScenarioSim, method: str, cfg: MPPIConfig, seed: int, snap: dict, v_ref=None,
                    model_params: VehicleParams | None = None):
    """Build an MPPI controller for `method` in {"O-R", "O-C", "M-N"} on the cars of `sim`; `seed` is the run
    seed index (0..N-1), mapped to a per-scenario noise seed by run_seed()."""
    mode = MODE_FIELD if method == "O-C" else MODE_FIXED
    c = MPPIConfig(**{**cfg.__dict__, "mode": mode})
    m = make_mppi(sim, c, seed=np.array([run_seed(sc.sid, method, seed) for sc in sim.scenarios]), v_ref=v_ref,
                  model_params=model_params)
    m.reset(snap["buf"][:, 0, :])
    if method == "M-N":
        dry = np.broadcast_to(surface_vec(DRY, MU_NOMINAL_DRY), (sim.n, 8))
        m.set_surfaces(dry, dry)

    def ctrl(s, k):
        if method == "O-R":
            wp.copy(m.sf_fix, s.axle_sf)
            wp.copy(m.sr_fix, s.axle_sr)
        return m.act()

    ctrl.mppi = m
    return ctrl
