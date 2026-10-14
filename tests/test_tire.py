"""V3 (tire function) and V4/S4 (friction bound) for the NumPy tire model."""

import numpy as np
import pytest

from sim.params import SURFACES
from sim.tire import ALPHA_MAX, clip_to_circle, magic_formula, surface_args, tire_forces

SURF = list(SURFACES.values())
FZ = 7800.0  # roughly a static axle load [N]


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_pure_longitudinal_equals_mf_of_theoretical_slip(s):
    kappa = np.linspace(-0.95, 1.0, 2001)
    fx, fy = tire_forces(kappa, np.zeros_like(kappa), FZ, *surface_args(s))
    sx = kappa / (1.0 + kappa)
    expected = np.sign(sx) * magic_formula(np.abs(sx), s.Bx, s.Cx, s.mu * FZ, s.Ex)
    np.testing.assert_allclose(fx, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(fy, 0.0)


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_pure_lateral_equals_mf_of_tan_alpha(s):
    alpha = np.linspace(-1.5, 1.5, 2001)
    fx, fy = tire_forces(np.zeros_like(alpha), alpha, FZ, *surface_args(s))
    ty = np.tan(alpha)
    expected = np.sign(ty) * magic_formula(np.abs(ty), s.By, s.Cy, s.mu * FZ, s.Ey)
    np.testing.assert_allclose(fy, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(fx, 0.0, atol=1e-9)


@pytest.mark.parametrize("ca", [0.0, 0.5, 1.0])
def test_v3_every_combined_variant_reduces_to_pure_slip(ca):
    s = SURFACES["dry"]
    kappa = np.linspace(-0.9, 0.9, 101)
    alpha = np.linspace(-1.2, 1.2, 101)
    fx0, _ = tire_forces(kappa, 0 * kappa, FZ, *surface_args(s), 0.0)
    fxc, fyc = tire_forces(kappa, 0 * kappa, FZ, *surface_args(s), ca)
    np.testing.assert_allclose(fxc, fx0, atol=1e-6)
    np.testing.assert_allclose(fyc, 0.0, atol=1e-6)
    _, fy0 = tire_forces(0 * alpha, alpha, FZ, *surface_args(s), 0.0)
    fxc, fyc = tire_forces(0 * alpha, alpha, FZ, *surface_args(s), ca)
    np.testing.assert_allclose(fyc, fy0, atol=1e-6)
    np.testing.assert_allclose(fxc, 0.0, atol=1e-6)


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_small_slip_stiffness_is_BCD(s):
    d = 1e-6
    _, fy = tire_forces(0.0, d, FZ, *surface_args(s))
    fx, _ = tire_forces(d, 0.0, FZ, *surface_args(s))
    assert fy / d == pytest.approx(s.By * s.Cy * s.mu * FZ, rel=1e-4)
    assert fx / d == pytest.approx(s.Bx * s.Cx * s.mu * FZ, rel=1e-4)


@pytest.mark.parametrize("s", [x for x in SURF if x.Cy > 1.0], ids=lambda s: s.name)
def test_peak_force_equals_mu_fz(s):
    """D = mu*Fz must be *reached* in both directions (V2 relies on this). Snow's lateral curve (C = 1)
    only approaches mu*Fz asymptotically by design (CRREL), so it is excluded laterally."""
    alpha = np.linspace(0, 1.5, 200001)
    _, fy = tire_forces(0 * alpha, alpha, FZ, *surface_args(s))
    assert fy.max() == pytest.approx(s.mu * FZ, rel=1e-6)
    kappa = np.linspace(-0.99, 0, 200001)
    fx, _ = tire_forces(kappa, 0 * kappa, FZ, *surface_args(s))
    assert -fx.min() == pytest.approx(s.mu * FZ, rel=1e-6)


@pytest.mark.parametrize("name,peak_deg", [("dry", 7.0), ("wet", 6.0), ("ice", 3.0)])
def test_lateral_peak_angles_match_sources(name, peak_deg):
    """D15 sources: dry 7 deg, wet 6 deg (typical passenger tire), ice 3 deg (CRREL: 2-4 deg)."""
    s = SURFACES[name]
    alpha = np.radians(np.linspace(0.01, 30, 30000))
    _, fy = tire_forces(0 * alpha, alpha, FZ, *surface_args(s))
    assert np.degrees(alpha[np.argmax(fy)]) == pytest.approx(peak_deg, abs=0.05)


def test_ice_falls_off_and_snow_keeps_rising():
    """CRREL shapes: ice drops markedly after its peak; snow approaches its maximum up to large angles."""
    a = np.radians([3.0, 15.0, 40.0])
    _, fy_ice = tire_forces(0 * a, a, FZ, *surface_args(SURFACES["ice"]))
    _, fy_snow = tire_forces(0 * a, a, FZ, *surface_args(SURFACES["snow"]))
    assert fy_ice[2] < 0.75 * fy_ice[0]
    assert fy_snow[0] < fy_snow[1] < fy_snow[2]


def test_lateral_force_is_odd_in_alpha():
    s = SURFACES["snow"]
    k = np.linspace(-0.5, 0.5, 51)[:, None]
    a = np.linspace(0.01, 1.4, 51)[None, :]
    fxp, fyp = tire_forces(k, a, FZ, *surface_args(s))
    fxn, fyn = tire_forces(k, -a, FZ, *surface_args(s))
    np.testing.assert_allclose(fyn, -fyp, atol=1e-9)
    np.testing.assert_allclose(fxn, fxp, atol=1e-9)


@pytest.mark.parametrize("ca", [0.0, 0.25, 0.5, 1.0])
def test_v4_friction_bound_dense_grid(ca):
    """S4: |F| <= mu*Fz on a dense (kappa, alpha, Fz, mu) grid for every surface and combined-slip variant."""
    kappa = np.concatenate([np.linspace(-1.0, 1.0, 81), [-0.999, -0.99, 0.999]])
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 81)
    fz = np.linspace(0.05, 1.6, 9) * FZ
    K, A, Z = np.meshgrid(kappa, alpha, fz, indexing="ij")
    for s in SURF:
        for mu in np.linspace(0.05, 1.0, 6):
            fx, fy = tire_forces(K, A, Z, *surface_args(s, mu), ca)
            assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))
            assert np.all(np.hypot(fx, fy) <= mu * Z * (1 + 1e-9))


def test_alpha_is_clamped_before_tan():
    s = SURFACES["dry"]
    fx, fy = tire_forces(0.0, np.pi / 2, FZ, *surface_args(s))
    assert np.isfinite(fy) and ALPHA_MAX < np.pi / 2


def test_clip_to_circle():
    fx, fy = clip_to_circle(np.array([3.0, 0.3]), np.array([4.0, 0.4]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(np.hypot(fx, fy), [1.0, 0.5])
    np.testing.assert_allclose(fx / fy, [0.75, 0.75])
