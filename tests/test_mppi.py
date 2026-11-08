"""Reference MPPI (BASELINES.md §3): rollout model = truth simulator, update math, closed-loop sanity."""

import numpy as np
import pytest
import warp as wp

from control.controllers import ORACLE_OR, mppi_controller
from control.episode import run_episodes, run_prefix
from control.mppi import MODE_FIELD, MODE_FIXED, MPPI, MPPIConfig, shift_kernel, update_kernel, weights_kernel
from scenarios import sampler as S
from scenarios import track as T
from sim.params import DRY, ICE, VehicleParams
from sim.warp_sim import ScenarioSim, surface_vec

P = VehicleParams()
TRACKS = T.all_tracks()


def _sc(sid, track_id, loc, surface, Lp, v0, mu1, lt=0.0):
    tr = TRACKS[track_id]
    s_drop = {"apex": tr.s_apex, "straight": T.LEAD_IN, "entry": tr.s_entry, "exit": tr.s_arc_end}[loc]
    return S.Scenario(sid, 0, v0, tr.name, tr.R, loc, surface, Lp, lt, v0, 0.95, mu1, track_id, s_drop)


@pytest.mark.parametrize("mode", [MODE_FIELD, MODE_FIXED])
def test_noise_free_rollout_reproduces_the_truth_simulator(mode):
    """The k = 0 rollout of a plan must end exactly where the truth simulator ends when fed the same plan open
    loop (same substep code, same delay-buffer replay, same friction field). Checked across an ice patch."""
    scs = [_sc("a", 0, "straight", "ice", 30.0, 15.0, 0.1, lt=2.0), _sc("b", 1, "apex", "snow", 1e9, 12.0, 0.3)]
    H = 60
    truth = ScenarioSim(scs, P, TRACKS)
    truth.reset_at_start()
    # coast to ~1 s before the straight's drop, so the 1.2 s horizon crosses onto the patch
    for _ in range(int(4.2 / 0.02)):
        truth.step(np.array([[0.0, 0.0], [0.0, 0.0]]))
    snap = truth.snapshot()
    m = MPPI(truth, MPPIConfig(K=2, H=H, mode=mode), seed=1)
    tt = np.arange(H) * 0.02
    plan = np.stack([np.stack([0.05 * np.sin(2 * tt), -0.2 + 0.1 * np.cos(tt)], 1)] * 2).astype(np.float32)
    m.U.assign(wp.array(plan, dtype=wp.float32))
    if mode == MODE_FIXED:
        m.set_surfaces(snap["axle_sf"], snap["axle_sr"])
    m.act()
    xend = m.xend.numpy()[:, 0, :]
    replay = ScenarioSim(scs, P, TRACKS)
    replay.load(snap)
    if mode == MODE_FIXED:  # fixed surfaces: compare with a sim whose field is the constant snapshot surface
        from sim.warp_sim import WarpSim
        replay = WarpSim(2, P)
        replay.reset(snap["state"])
        replay.buf.assign(wp.array(snap["buf"], dtype=wp.float32))
        replay.set_surfaces(snap["axle_sf"], snap["axle_sr"])
    for h in range(H):
        replay.step(plan[:, h, :])
    np.testing.assert_allclose(xend, replay.state.numpy(), rtol=0, atol=1e-5)
    if mode == MODE_FIELD:  # the plan must actually have crossed onto the patch for this to mean anything
        assert replay.get_axle_mu()[0, 1] < 0.2


def test_weights_and_update_match_numpy():
    rng = np.random.default_rng(0)
    n, K, H, lam = 2, 64, 5, 3.0
    costs = rng.uniform(0, 20, (n, K)).astype(np.float32)
    V = rng.uniform(-1, 1, (n, K, H, 2)).astype(np.float32)
    w = wp.zeros((n, K), dtype=wp.float32)
    info = wp.zeros((n, 3), dtype=wp.float32)
    U = wp.zeros((n, H, 2), dtype=wp.float32)
    Unew = wp.zeros((n, H, 2), dtype=wp.float32)
    act = wp.zeros((n, 2), dtype=wp.float32)
    wp.launch(weights_kernel, dim=n, inputs=[wp.array(costs), lam, 0.0, w, info])
    wp.launch(update_kernel, dim=(n, H), inputs=[w, wp.array(V), Unew])
    wp.launch(shift_kernel, dim=(n, H), inputs=[Unew, U, act])
    wn = np.exp(-(costs - costs.min(1, keepdims=True)) / lam)
    wn /= wn.sum(1, keepdims=True)
    new = np.einsum("nk,nkhc->nhc", wn, V)
    np.testing.assert_allclose(w.numpy(), wn, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(act.numpy(), new[:, 0], atol=1e-5)
    np.testing.assert_allclose(U.numpy()[:, :-1], new[:, 1:], atol=1e-5)  # warm start = shifted plan
    np.testing.assert_allclose(U.numpy()[:, -1], new[:, -1], atol=1e-5)
    np.testing.assert_allclose(info.numpy()[:, 1], 1 / (wn ** 2).sum(1), rtol=1e-4)


def test_lower_temperature_concentrates_weight():
    costs = np.linspace(0, 10, 32, dtype=np.float32)[None]
    ess = []
    for lam in (0.5, 5.0, 50.0):
        w = wp.zeros((1, 32), dtype=wp.float32)
        info = wp.zeros((1, 3), dtype=wp.float32)
        wp.launch(weights_kernel, dim=1, inputs=[wp.array(costs), lam, 0.0, w, info])
        ess.append(info.numpy()[0, 1])
    assert ess[0] < ess[1] < ess[2] <= 32.0 + 1e-3


@pytest.mark.gpu
@pytest.mark.slow
def test_closed_loop_oracles_on_easy_and_classic_cases():
    """O-R keeps the car in lane on an easy wet patch and on a straight ice patch; nobody survives the
    certified classic case from the reactive information level; same seed -> identical records."""
    scs = [_sc("easy", 1, "apex", "wet", 30.0, 12.0, 0.8), _sc("classic", 1, "apex", "ice", 1e9, 15.0, 0.1),
           _sc("straight", 0, "straight", "ice", 30.0, 20.0, 0.1)]
    snap, _ = run_prefix(scs, TRACKS, P, "cuda:0")
    out = {}
    for meth in ("O-R", "M-N"):
        sim = ScenarioSim(scs, P, TRACKS, "cuda:0")
        recs = run_episodes(scs, snap, mppi_controller(sim, meth, ORACLE_OR, 0, snap), meth, 0,
                            TRACKS, P, "cuda:0", sim=sim)
        out[meth] = {r.sid: r for r in recs}
    assert out["O-R"]["easy"].success and out["O-R"]["straight"].success
    assert not out["O-R"]["classic"].success and not out["M-N"]["classic"].success
    sim = ScenarioSim(scs, P, TRACKS, "cuda:0")
    again = run_episodes(scs, snap, mppi_controller(sim, "O-R", ORACLE_OR, 0, snap), "O-R", 0,
                         TRACKS, P, "cuda:0", sim=sim)
    assert [r.min_clearance for r in again] == [out["O-R"][s.sid].min_clearance for s in scs]


def test_run_does_not_depend_on_batch_composition():
    """Same scenario, method and seed -> identical record whether it runs alone or batched with others."""
    a = _sc("a", 0, "straight", "snow", 10.0, 12.0, 0.3)
    b = _sc("b", 1, "apex", "wet", 10.0, 11.0, 0.7)
    cfg = MPPIConfig(K=16, H=10)
    out = []
    for scs in ([a], [b, a]):
        snap, _ = run_prefix(scs, TRACKS, P)
        sim = ScenarioSim(scs, P, TRACKS)
        recs = run_episodes(scs, snap, mppi_controller(sim, "O-R", cfg, 3, snap), "O-R", 3, TRACKS, P, sim=sim)
        out.append([r for r in recs if r.sid == "a"][0])
    assert out[0].min_clearance == out[1].min_clearance and out[0].duration == out[1].duration


def test_ar1_noise_has_unit_variance_and_the_requested_correlation():
    from control.mppi import sample_kernel
    n, K, H, rho = 1, 4000, 40, 0.9
    U = wp.zeros((n, H, 2), dtype=wp.float32)
    V = wp.zeros((n, K, H, 2), dtype=wp.float32)
    wp.launch(sample_kernel, dim=(n, K), inputs=[U, V, wp.vec2(0.1, 0.1), rho, wp.array([7], dtype=int), 0])
    x = V.numpy()[0, 1:, :, 0] / 0.1  # sample 0 is noise-free; 0.1 std keeps clipping negligible
    assert np.all(V.numpy()[0, 0] == 0)
    assert abs(x.std() - 1.0) < 0.03
    lag1 = np.mean(x[:, 1:] * x[:, :-1]) / np.mean(x * x)
    assert abs(lag1 - rho) < 0.02


def test_ess_targeted_temperature_hits_the_target_at_any_cost_scale():
    rng = np.random.default_rng(0)
    for scale in (0.01, 1.0, 1000.0):
        costs = (rng.exponential(1.0, (3, 512)) * scale).astype(np.float32)
        w = wp.zeros((3, 512), dtype=wp.float32)
        info = wp.zeros((3, 3), dtype=wp.float32)
        wp.launch(weights_kernel, dim=3, inputs=[wp.array(costs), 1.0, 32.0, w, info])
        ess = info.numpy()[:, 1]
        assert np.all(np.abs(ess - 32.0) < 1.0), (scale, ess)
        np.testing.assert_allclose(w.numpy().sum(1), 1.0, rtol=1e-4)


def test_pd_sample_records_the_actions_it_applied():
    """Ancillary PD sample (k = 1): replaying its recorded actions open loop in the truth simulator must end at
    the same state (the delay buffer means each recorded action is applied d steps later, as for any sample)."""
    scs = [_sc("a", 2, "apex", "snow", 1e9, 14.0, 0.3)]
    snap, _ = run_prefix(scs, TRACKS, P)  # handover state on the curve, 4 s before the drop
    truth = ScenarioSim(scs, P, TRACKS)
    truth.load(snap)
    H = 50
    m = MPPI(truth, MPPIConfig(K=2, H=H, mode=MODE_FIELD, pd_sample=True), seed=0)
    m.act()
    rec = m.V.numpy()[0, 1]
    xend = m.xend.numpy()[0, 1]
    replay = ScenarioSim(scs, P, TRACKS)
    replay.load(snap)
    for h in range(H):
        replay.step(rec[h][None])
    np.testing.assert_allclose(xend, replay.state.numpy()[0], atol=1e-5)
    assert np.abs(rec[:, 0]).max() > 0.01  # it actually steered (the track curves here)
    assert m.costs.numpy()[0, 1] < 100.0  # and stayed on the road (no termination penalty)
