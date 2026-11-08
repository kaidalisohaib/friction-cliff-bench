"""Tuning harness with an enforced budget and a ledger (prereg V2, BASELINES.md §2 item 5).

Every tunable method gets N = 20 Optuna (TPE) trials and G = 15 GPU-hours on the tuning split, whichever binds
first. A trial = one evaluation of a sampled configuration (for PPO: one training run). The harness stops
before starting a trial that would break either limit (using the running mean trial time to predict it),
and appends one ledger line per trial to a JSONL file that `LOG.md`'s tuning ledger summarizes:
method, trial index, params, objective, GPU-seconds, wall time, git SHA. Human-hours are logged by hand.

The objective function is method-specific and must only ever read the tuning split.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna

N_TRIALS = 20  # prereg V2
GPU_HOURS = 15.0  # prereg V2


@dataclass
class TuneResult:
    best_params: dict
    best_value: float
    n_trials: int
    gpu_hours: float
    stopped_by: str  # "trials" | "gpu_hours"


def tune(method: str, objective: Callable[[dict], tuple[float, float]], space: Callable[[optuna.Trial], dict],
         ledger: Path, n_trials: int = N_TRIALS, gpu_hours: float = GPU_HOURS, seed: int = 0,
         git: str = "unknown") -> TuneResult:
    """objective(params) -> (value to minimize, GPU-seconds used). space(trial) -> params dict."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    used = 0.0
    done = 0
    stopped_by = "trials"
    while done < n_trials:
        if done > 0 and used + used / done > gpu_hours * 3600.0:
            stopped_by = "gpu_hours"
            break
        trial = study.ask()
        params = space(trial)
        t0 = time.time()
        value, gpu_s = objective(params)
        study.tell(trial, value)
        used += gpu_s
        done += 1
        with open(ledger, "a") as f:
            f.write(json.dumps({"method": method, "trial": trial.number, "params": params, "value": value,
                                "gpu_s": gpu_s, "wall_s": time.time() - t0, "git": git}) + "\n")
    return TuneResult(study.best_params, float(study.best_value), done, used / 3600.0, stopped_by)
