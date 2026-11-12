"""Confirmatory analysis on synthetic run logs with known effects (eval/analysis.py)."""

import json

import numpy as np

from eval import analysis as A


def _write(root, set_name, method, dep_prob, n=300, seeds=5, progress=100.0, rng=None, sids=None):
    rng = rng or np.random.default_rng(0)
    d = root / set_name
    d.mkdir(parents=True, exist_ok=True)
    sids = sids or [f"s{j}" for j in range(n)]
    p = np.broadcast_to(np.asarray(dep_prob, float), (len(sids),))
    for s in range(seeds):
        with open(d / f"{method}_s{s}.jsonl", "w") as f:
            for j, sid in enumerate(sids):
                dep = bool(rng.random() < p[j])
                f.write(json.dumps({"sid": sid, "departed": dep, "loss_of_control": False, "success": not dep,
                                    "max_abs_ey": 0.5 + dep, "time_out_of_lane": 0.1 * dep,
                                    "progress": progress * (1 - 0.01 * rng.random())}) + "\n")


def _labels(n=300):
    return {f"s{j}": {"label": "reactive-feasible", "adaptation_required": j % 3 == 0} for j in range(n)}


def test_h1_falsified_only_by_a_large_rl_win(tmp_path):
    for case, ps, falsified in (("win", 0.02, True), ("tie", 0.20, False)):
        root = tmp_path / case
        rng = np.random.default_rng(1)
        n = 1000  # the hierarchical bootstrap (seeds, then scenarios) is conservative: TOST needs some N
        for m, p in (("O-R", 0.05), ("M-B", 0.20), ("M-L", 0.30), ("P-S", ps)):
            _write(root, "main", m, p, n=n, rng=rng)
        res = A.analyze(root, _labels(n), ["O-R", "M-B", "M-L", "P-S"])
        h = res["hypotheses"]["H1"]
        assert h["better_mppi"] == "M-B"
        assert h["falsified"] is falsified, h
        if case == "tie":
            assert h["tost"]["equivalent"]


def test_h2_detects_mppi_regret_growing_faster(tmp_path):
    rng = np.random.default_rng(2)
    labels = _labels()
    _write(tmp_path, "main", "O-R", 0.02, rng=rng)
    _write(tmp_path, "main", "M-B", 0.05, rng=rng)
    _write(tmp_path, "main", "P-S", 0.06, rng=rng)
    names = []
    for axis in ("slip", "relax", "delay"):
        for i, lv in enumerate((1, 2, 3)):
            n = f"h2-{axis}-{lv}"
            names.append(n)
            grow = axis != "relax"  # two of three axes have the effect
            _write(tmp_path, n, "O-R", 0.02, rng=rng)
            _write(tmp_path, n, "M-B", 0.05 + (0.12 * (i + 1) if grow else 0.0), rng=rng)
            _write(tmp_path, n, "P-S", 0.06 + 0.01 * (i + 1), rng=rng)
    res = A.analyze(tmp_path, labels, ["O-R", "M-B", "P-S"], names)
    h2 = res["hypotheses"]["H2"]
    assert h2["axes"]["slip"]["pass"] and h2["axes"]["delay"]["pass"] and not h2["axes"]["relax"]["pass"]
    assert h2["supported"] and h2["p"] < 0.05
    assert "holm" in res and set(res["holm"]["p"]) == {"H1", "H2"}


def test_h3_needs_lower_departure_and_no_dry_pace_loss(tmp_path):
    rng = np.random.default_rng(3)
    _write(tmp_path, "main", "O-R", 0.02, rng=rng)
    _write(tmp_path, "main", "M-B", 0.10, rng=rng)
    _write(tmp_path, "main", "P-S", 0.15, rng=rng)
    _write(tmp_path, "main", "P-SP", 0.03, rng=rng)
    for m, prog in (("O-R", 100.0), ("P-S", 100.0), ("P-SP", 99.9)):
        _write(tmp_path, "dry", m, 0.0, rng=rng, progress=prog)
    res = A.analyze(tmp_path, _labels(), ["O-R", "M-B", "P-S", "P-SP"])
    assert res["hypotheses"]["H3"]["falsified"]
    # same departure gain but P-SP 5% slower on dry road: not falsified
    _write(tmp_path, "dry", "P-SP", 0.0, rng=rng, progress=95.0)
    res = A.analyze(tmp_path, _labels(), ["O-R", "M-B", "P-S", "P-SP"])
    assert not res["hypotheses"]["H3"]["falsified"]
    assert res["secondary_rf"]["P-SP"]["dry_pace_tax"]["estimate"] > 0.03


def test_markdown_renders(tmp_path):
    rng = np.random.default_rng(4)
    for m, p in (("O-R", 0.02), ("M-B", 0.1), ("P-S", 0.1)):
        _write(tmp_path, "main", m, p, rng=rng)
    res = A.analyze(tmp_path, _labels(), ["O-R", "M-B", "P-S"])
    md = A.markdown(res, "synthetic")
    assert "## H1" in md and "reactive-feasible" in md


def test_caution_bins_split_departures_by_arrival_speed():
    # method "A" arrives slow and never departs; "B" arrives at v_eff and departs in half of the runs
    n_sc, ve = 40, np.full(40, 20.0)

    def runs(av, dep):
        return {"sids": [f"s{j}" for j in range(n_sc)],
                "recs": [[{"arrival_v": av, "departed": bool(dep and j % 2)} for j in range(n_sc)] for _ in range(3)]}

    main = {"A": runs(14.0, False), "B": runs(20.0, True)}
    out = A.caution(main, list(range(n_sc)), ve, ["A", "B"])
    assert abs(out["A"]["arrival_ratio"]["estimate"] - 0.7) < 1e-9
    assert out["A"]["bins"][0]["n_runs"] == 120 and out["A"]["bins"][0]["departed"] == 0
    b = out["B"]["bins"][3]  # ratio 1.0 falls in [1.0, inf)
    assert b["n_runs"] == 120 and b["departed"] == 60


def test_e7_caution_rule(tmp_path):
    # P-S departs 2%, M-B (better MPPI) 20%: a student advantage in H1. M-B-c either matches P-S (caution) or
    # stays at 20% (the slower speed does not help: the advantage is not attributed to caution).
    labels = _labels()
    for case, p_mbc, caution in (("slow-helps", 0.02, True), ("slow-no-help", 0.20, False)):
        root = tmp_path / case
        rng = np.random.default_rng(1)
        for m, p in (("O-R", 0.02), ("M-B", 0.20), ("M-L", 0.30), ("P-S", 0.02), ("M-B-c", p_mbc)):
            _write(root, "main", m, p, rng=rng)
        res = A.analyze(root, labels, ["O-R", "M-B", "M-L", "P-S", "M-B-c"])
        e = res["exploratory_e7"]
        assert e["student_advantage_in_H1"]
        assert e["attributed_to_caution"] is caution, (case, e)
    # no student advantage -> nothing to attribute, whatever M-B-c does
    root = tmp_path / "no-adv"
    rng = np.random.default_rng(2)
    for m, p in (("O-R", 0.02), ("M-B", 0.02), ("M-L", 0.30), ("P-S", 0.20), ("M-B-c", 0.02)):
        _write(root, "main", m, p, rng=rng)
    e = A.analyze(root, labels, ["O-R", "M-B", "M-L", "P-S", "M-B-c"])["exploratory_e7"]
    assert not e["student_advantage_in_H1"] and not e["attributed_to_caution"]


def test_e7_dry_pace_tax_difference(tmp_path):
    labels = _labels(100)
    rng = np.random.default_rng(3)
    for m in ("O-R", "M-B", "P-S", "M-B-c"):
        _write(tmp_path, "main", m, 0.05, n=100, rng=rng)
    for m, prog in (("O-R", 100.0), ("M-B", 100.0), ("P-S", 90.0), ("M-B-c", 80.0)):
        _write(tmp_path, "dry", m, 0.0, n=100, progress=prog, rng=rng)
    e = A.analyze(tmp_path, labels, ["O-R", "M-B", "P-S", "M-B-c"])["exploratory_e7"]
    t = e["dry_pace_tax_diff_MBc_minus_PS"]  # tax 0.2 - 0.1 = +0.1 (up to the 1% progress jitter)
    assert abs(t["estimate"] - 0.1) < 0.01 and t["ci95"][0] > 0.08


def test_recovery_summary(tmp_path):
    # O-R holds e = 0; method "M-B" leaves the band for 1 s after the drop then returns; "M-L" never returns.
    from sim.params import CONTROL_DT
    tr = tmp_path / "trace"
    tr.mkdir()
    n, T = 30, 400
    sid = np.array([f"s{j}" for j in range(n)])
    t_drop = np.full(n, 1.0)
    k0 = int(round(1.0 / CONTROL_DT)) - 1
    zero = np.zeros((n, T))
    late = zero.copy()
    late[:, k0:k0 + int(1.0 / CONTROL_DT)] = 1.0
    never = zero.copy()
    never[:, k0:] = 1.0
    for s in range(2):
        for m, e in (("O-R", zero), ("M-B", late), ("M-L", never)):
            np.savez(tr / f"{m}_s{s}.npz", sid=sid, e=e, r=zero, t_drop=t_drop)
    v_eff = {f"s{j}": 10.0 + j for j in range(n)}
    out = A.recovery(tmp_path, ["O-R", "M-B", "M-L"], v_eff, seeds=(0, 1), n_boot=200)
    assert out["M-B"]["recovered_frac"]["estimate"] == 1.0
    assert abs(out["M-B"]["median_latency_recovered_s"]["estimate"] - 1.0) < 1e-6
    assert out["M-L"]["recovered_frac"]["estimate"] == 0.0
    assert out["M-L"]["median_latency_recovered_s"] is None
    assert "M-L" in A.markdown({"classes": {}, "exploratory_recovery": out}, "t")
    assert out["M-B"]["n_runs"] == 2 * n
