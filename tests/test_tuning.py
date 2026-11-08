"""Tuning harness: the budget binds, the ledger records every trial, TPE finds a simple optimum."""

import json

from eval.tuning import tune


def _space(trial):
    return {"x": trial.suggest_float("x", -5.0, 5.0)}


def test_trial_budget_and_ledger(tmp_path):
    led = tmp_path / "ledger.jsonl"
    res = tune("toy", lambda p: ((p["x"] - 1.3) ** 2, 1.0), _space, led, n_trials=20)
    lines = [json.loads(line) for line in led.read_text().splitlines()]
    assert res.n_trials == 20 and len(lines) == 20 and res.stopped_by == "trials"
    assert abs(res.best_params["x"] - 1.3) < 0.5
    assert all(set(r) >= {"method", "trial", "params", "value", "gpu_s"} for r in lines)


def test_gpu_hour_budget_binds_first(tmp_path):
    # 1 GPU-hour per trial, 3.5 h cap: the 4th trial would exceed it, so 3 run
    res = tune("toy", lambda p: (p["x"] ** 2, 3600.0), _space, tmp_path / "l.jsonl", n_trials=20, gpu_hours=3.5)
    assert res.n_trials == 3 and res.stopped_by == "gpu_hours"
