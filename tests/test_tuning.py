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


def test_resume_does_not_redraw_an_evaluated_config(tmp_path):
    # regression (2026-11-11): a resumed study restarted the sampler RNG and re-ran the first configuration
    space = lambda t: {"x": t.suggest_float("x", -1.0, 1.0)}
    calls = []
    obj = lambda p: (calls.append(p) or (p["x"] ** 2, 1.0))
    led = tmp_path / "ledger.jsonl"
    tune("m", obj, space, led, n_trials=1)
    tune("m", obj, space, led, n_trials=4)
    xs = [c["x"] for c in calls]
    assert len(xs) == 4 and len(set(xs)) == 4


def test_duplicate_config_costs_gpu_hours_but_not_a_trial(tmp_path):
    led = tmp_path / "ledger.jsonl"
    rec = {"method": "m", "trial": 0, "params": {"x": 0.5}, "value": 0.25, "gpu_s": 3600.0}
    led.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "trial": 1}) + "\n")
    calls = []
    res = tune("m", lambda p: (calls.append(p) or (p["x"] ** 2, 3600.0)),
               lambda t: {"x": t.suggest_float("x", -1.0, 1.0)}, led, n_trials=3, gpu_hours=10.0)
    assert len(calls) == 2 and res.n_trials == 3  # 1 distinct replayed config + 2 new ones
    assert abs(res.gpu_hours - 4.0) < 1e-9  # both replayed lines count toward the cap
