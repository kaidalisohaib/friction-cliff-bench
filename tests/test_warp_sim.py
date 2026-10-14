"""Warp simulator: struct-default trap, and V8 cross-implementation check against the NumPy reference."""

from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from sim.params import DRY, ICE, SNOW, WET, VehicleParams
from sim.reference import IR, IVX, IVY, NX, ReferenceSim
from sim.warp_sim import Params, WarpSim, make_params

P = VehicleParams()


def test_every_param_field_reaches_the_struct():
    """Guards the known Warp trap: wp.struct field defaults are ignored and silently read 0."""
    struct_fields = set(Params.vars.keys())
    dc_fields = {f.name for f in fields(VehicleParams)}
    assert struct_fields == dc_fields, f"mismatch: {struct_fields ^ dc_fields}"
    p = P.with_(m=1234.5, delay_steps=3, sigma_relax=0.4, combined_alpha=0.25, B_scale_r=1.1)
    s = make_params(p)
    for f in fields(VehicleParams):
        assert getattr(s, f.name) == pytest.approx(getattr(p, f.name), rel=1e-6), f.name
    # and inside a kernel, via an array of structs
    arr = wp.array([s], dtype=Params, device="cpu").numpy()
    for f in fields(VehicleParams):
        assert arr[f.name][0] == pytest.approx(getattr(p, f.name), rel=1e-6), f.name


def run_pair(p, x0, actions, surfaces):
    """Run the reference and Warp (CPU) on the same action/surface sequence; return both trajectories."""
    ref = ReferenceSim(p, x0)
    ws = WarpSim(1, p, device="cpu")
    ws.reset(x0)
    tr, tw = [], []
    for a, (sf, sr) in zip(actions, surfaces):
        ref.step(a, sf, sr)
        ws.set_surfaces(sf, sr)
        ws.step(a)
        tr.append(ref.x.copy())
        tw.append(ws.get_state()[0])
    return np.array(tr), np.array(tw)


def maneuvers():
    v = 20.0
    x0 = ReferenceSim.steady_state(v, P)
    n = 100  # 2 s
    t = np.arange(n) * 0.02
    sine = np.stack([0.15 * np.sin(2 * np.pi * 0.5 * t), 0.1 * np.ones(n)], 1)
    yield "sine_steer_dry", P, x0, sine, [(DRY, DRY)] * n
    yield "brake_wet", P, x0, np.tile([0.0, -0.7], (n, 1)), [(WET, WET)] * n
    yield "throttle_snow", P, ReferenceSim.steady_state(10.0, P), np.tile([0.0, 0.6], (n, 1)), [(SNOW, SNOW)] * n
    yield "relax_and_ellipse", P.with_(sigma_relax=0.4, combined_alpha=0.5), x0, sine, [(DRY, DRY)] * n
    yield "delay_4_steps", P.with_(delay_steps=4), x0, sine, [(DRY, DRY)] * n
    yield "fwd_understeer_scaled", P.with_(drive_front_frac=1.0, B_scale_r=1.2), x0, sine, [(DRY, DRY)] * n
    yield "front_ice_patch", P, x0, np.tile([0.1, 0.0], (n, 1)), [(DRY, DRY)] * 30 + [(ICE, DRY)] * 70


@pytest.mark.parametrize("name,p,x0,actions,surfaces", list(maneuvers()), ids=[m[0] for m in maneuvers()])
def test_v8_warp_matches_reference(name, p, x0, actions, surfaces):
    tr, tw = run_pair(p, x0, actions, surfaces)
    assert np.all(np.isfinite(tw))
    # position within 2 cm and velocities within 1e-3 relative over 2 s (float32 vs float64)
    pos_err = np.max(np.hypot(tr[:, 0] - tw[:, 0], tr[:, 1] - tw[:, 1]))
    assert pos_err < 0.02, pos_err
    for i in (IVX, IVY, IR):
        scale = max(np.max(np.abs(tr[:, i])), 0.1)
        assert np.max(np.abs(tr[:, i] - tw[:, i])) / scale < 1e-3, (i, np.max(np.abs(tr[:, i] - tw[:, i])))


def test_v8_error_report(capsys):
    """Not a gate: prints the worst errors over all maneuvers so they can be documented (DESIGN V8)."""
    rows = []
    for name, p, x0, actions, surfaces in maneuvers():
        tr, tw = run_pair(p, x0, actions, surfaces)
        rows.append((name, np.max(np.hypot(tr[:, 0] - tw[:, 0], tr[:, 1] - tw[:, 1])),
                     np.max(np.abs(tr[:, IVX] - tw[:, IVX])), np.max(np.abs(tr[:, IR] - tw[:, IR]))))
    with capsys.disabled():
        print("\nV8 Warp(float32, CPU) vs NumPy reference(float64), 2 s:")
        for r in rows:
            print(f"  {r[0]:24s} max|dpos| {r[1]:.2e} m  max|dvx| {r[2]:.2e} m/s  max|dr| {r[3]:.2e} rad/s")
    assert len(rows) == 7 and NX == 13
