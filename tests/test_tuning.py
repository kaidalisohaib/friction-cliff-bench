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


def test_tuning_resumes_from_ledger(tmp_path):
    import optuna
    from eval.tuning import tune
    space = lambda t: {"x": t.suggest_float("x", -1.0, 1.0), "c": t.suggest_categorical("c", [1, 2])}
    calls = []
    obj = lambda p: (calls.append(p) or (p["x"] ** 2, 1.0))
    led = tmp_path / "ledger.jsonl"
    r1 = tune("m", obj, space, led, n_trials=3)
    assert r1.n_trials == 3 and len(calls) == 3
    r2 = tune("m", obj, space, led, n_trials=5)
    assert r2.n_trials == 5 and len(calls) == 5  # only 2 new evaluations
    assert r2.best_value <= r1.best_value
