"""C++17/CUDA MPPI (mppi_cpp/, control/mppi_cpp.py) against the Warp reference MPPI.

S3 (BASELINES.md §5): with the same seeds the two backends draw the same samples, and the control actions agree
within a tolerance on 1,000 random states. The port reproduces Warp's RNG, so "fixed noise" holds by construction
(and is checked: the sampled controls must be bit-identical). The remaining differences come from compiler
rounding (nvcc vs Warp's NVRTC build): |action| <= 1e-4 on the first step (identical inputs), 2e-3 on the second
(inputs already differ by rounding); DECISIONS D29.
"""

from dataclasses import fields

import numpy as np
import pytest
import warp as wp

from control import mppi_cpp as C
from sim.params import VehicleParams

pytestmark = pytest.mark.skipif(not C.available(), reason="mppi_cpp not built (see control/mppi_cpp.py)")


def test_args_struct_matches_the_c_layout():
    # 10 int32 + 7 float32 + CostW (9 float32) = 104 bytes, then 25 pointers
    assert C.Args.cw.offset == 68 and C.Args.mparams.offset == 104
    assert C.ctypes.sizeof(C.Args) == 104 + 25 * 8


def test_rng_matches_warp():
    @wp.kernel
    def draw(seeds: wp.array(dtype=int), offs: wp.array(dtype=int), out: wp.array2d(dtype=float)):
        i = wp.tid()
        st = wp.rand_init(seeds[i], offs[i])
        for j in range(out.shape[1]):
            out[i, j] = wp.randn(st)

    rng = np.random.default_rng(1)
    seeds = rng.integers(-2**31, 2**31 - 1, 64).astype(np.int32)
    offs = rng.integers(0, 16384, 64).astype(np.int32)
    out = wp.zeros((64, 32), dtype=float, device="cpu")
    wp.launch(draw, dim=64, inputs=[wp.array(seeds, dtype=int, device="cpu"), wp.array(offs, dtype=int, device="cpu"),
                                    out], device="cpu")
    ref = out.numpy()
    got = np.zeros(32, np.float32)
    for i in range(64):
        C.lib().fcb_randn_host(int(seeds[i]), int(offs[i]), 32, got.ctypes.data)
        np.testing.assert_allclose(got, ref[i], rtol=0, atol=2e-6)


def test_params_struct_is_read_in_place():
    from sim.warp_sim import Params, make_params
    p = VehicleParams().with_(m=1234.5, delay_steps=3, sigma_relax=0.4, combined_alpha=0.25, B_scale_r=1.1)
    arr = wp.array([make_params(p)], dtype=Params, device="cpu")
    out = np.zeros(30, np.float32)
    C.lib().fcb_params_echo_host(arr.ptr, out.ctypes.data)
    for i, f in enumerate(fields(VehicleParams)):
        assert out[i] == pytest.approx(getattr(p, f.name), rel=1e-6), f.name


def _random_states(n: int, device: str, seed: int = 0):
    """n random planner states: handover states of tuning scenarios (sampled with replacement), perturbed in
    sideslip, yaw rate, steering, actuator state, delay buffer, assumed friction and warm-start plan."""
    from control.episode import run_prefix
    from eval.methods import tuning_sets
    from scenarios import track as T
    from sim.warp_sim import ScenarioSim

    scs, _ = tuning_sets()
    rng = np.random.default_rng(seed)
    base = [scs[i] for i in rng.choice(len(scs), 64, replace=False)]
    tracks = T.all_tracks()
    snap0, _ = run_prefix(base, tracks, VehicleParams(), device)
    idx = rng.integers(0, len(base), n)
    sub = [base[i] for i in idx]
    sn = {k: v[idx].copy() for k, v in snap0.items()}
    sn["state"][:, 4] += rng.uniform(-0.5, 0.5, n)
    sn["state"][:, 5] += rng.uniform(-0.1, 0.1, n)
    sn["state"][:, 10] += rng.uniform(-0.05, 0.05, n)
    sn["state"][:, 11] = rng.uniform(-0.5, 0.5, n)
    sn["buf"] = rng.uniform(-0.5, 0.5, sn["buf"].shape).astype(np.float32)
    mu = rng.uniform(0.08, 1.0, n).astype(np.float32)
    sf, sr = sn["axle_sf"].copy(), sn["axle_sr"].copy()
    sf[:, 0] = mu
    sr[:, 0] = mu

    def make(cls, cfg):
        sim = ScenarioSim(sub, VehicleParams(), tracks, device)
        sim.load(sn)
        m = cls(sim, cfg, seed=np.arange(n) * 7 + 3)
        U0 = np.random.default_rng(seed + 1).uniform(-0.5, 0.5, (n, cfg.H, 2)).astype(np.float32)
        m.U.assign(wp.array(U0, dtype=wp.float32, device=device))
        m.set_surfaces(sf, sr)
        m.t = 5
        return m

    return make


def _s3(n: int, device: str, K: int | None = None, H: int | None = None):
    from control.controllers import ORACLE_OC, ORACLE_OR
    from control.mppi import MPPI, MPPIConfig

    make = _random_states(n, device)
    for cfg in (ORACLE_OR, ORACLE_OC):
        cfg = MPPIConfig(**{**cfg.__dict__, **({"K": K} if K else {}), **({"H": H} if H else {})})
        mw, mc = make(MPPI, cfg), make(C.CppMPPI, cfg)
        for step in range(2):  # two consecutive control steps (warm start + step counter in the RNG stream)
            mw.act()
            mc.act()
            wp.synchronize()
            # step 0 is the S3 check proper: identical inputs and noise. The random samples (k >= 2) are
            # bit-identical and the action agrees to 1e-4. Step 1 starts from plans that already differ by
            # rounding (~1e-6), which rollouts near a termination boundary can amplify: 2e-3 (D29).
            if step == 0:
                np.testing.assert_array_equal(mw.V.numpy()[:, 2:], mc.V.numpy()[:, 2:])
            tol = 1e-4 if step == 0 else 2e-3
            np.testing.assert_allclose(mc.action.numpy(), mw.action.numpy(), rtol=0, atol=tol)
            np.testing.assert_allclose(mc.U.numpy(), mw.U.numpy(), rtol=0, atol=10 * tol)
        assert mc.t == mw.t == 7


def test_s3_cpu_backend_small():
    _s3(8, "cpu", K=32, H=40)


@pytest.mark.gpu
def test_s3_cuda_200_states():
    _s3(200, "cuda:0")


@pytest.mark.slow
@pytest.mark.gpu
def test_s3_cuda_1000_states():
    _s3(1000, "cuda:0")
