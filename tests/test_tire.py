"""V3 (tire function) and V4/S4 (friction bound) for the NumPy tire model."""

import numpy as np
import pytest

from sim.params import SURFACES
from sim.tire import ALPHA_MAX, clip_to_circle, magic_formula, tire_forces

SURF = list(SURFACES.values())
FZ = 7800.0  # roughly a static axle load [N]


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_pure_longitudinal_equals_mf_of_theoretical_slip(s):
    kappa = np.linspace(-0.95, 1.0, 2001)
    fx, fy = tire_forces(kappa, np.zeros_like(kappa), FZ, s.mu, s.B, s.C, s.E, 0.0)
    sx = kappa / (1.0 + kappa)
    expected = np.sign(sx) * magic_formula(np.abs(sx), s.B, s.C, s.mu * FZ, s.E)
    np.testing.assert_allclose(fx, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(fy, 0.0)


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_pure_lateral_equals_mf_of_tan_alpha(s):
    alpha = np.linspace(-1.5, 1.5, 2001)
    fx, fy = tire_forces(np.zeros_like(alpha), alpha, FZ, s.mu, s.B, s.C, s.E, 0.0)
    ty = np.tan(alpha)
    expected = np.sign(ty) * magic_formula(np.abs(ty), s.B, s.C, s.mu * FZ, s.E)
    np.testing.assert_allclose(fy, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(fx, 0.0, atol=1e-9)


@pytest.mark.parametrize("ca", [0.0, 0.5, 1.0])
def test_v3_every_combined_variant_reduces_to_pure_slip(ca):
    s = SURFACES["dry"]
    kappa = np.linspace(-0.9, 0.9, 101)
    alpha = np.linspace(-1.2, 1.2, 101)
    fx0, _ = tire_forces(kappa, 0 * kappa, FZ, s.mu, s.B, s.C, s.E, 0.0)
    fxc, fyc = tire_forces(kappa, 0 * kappa, FZ, s.mu, s.B, s.C, s.E, ca)
    np.testing.assert_allclose(fxc, fx0, atol=1e-6)
    np.testing.assert_allclose(fyc, 0.0, atol=1e-6)
    _, fy0 = tire_forces(0 * alpha, alpha, FZ, s.mu, s.B, s.C, s.E, 0.0)
    fxc, fyc = tire_forces(0 * alpha, alpha, FZ, s.mu, s.B, s.C, s.E, ca)
    np.testing.assert_allclose(fyc, fy0, atol=1e-6)
    np.testing.assert_allclose(fxc, 0.0, atol=1e-6)


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_v3_small_slip_stiffness_is_BCD(s):
    d = 1e-6
    _, fy = tire_forces(0.0, d, FZ, s.mu, s.B, s.C, s.E, 0.0)
    assert fy / d == pytest.approx(s.B * s.C * s.mu * FZ, rel=1e-4)


@pytest.mark.parametrize("s", SURF, ids=lambda s: s.name)
def test_peak_force_equals_mu_fz(s):
    """D = mu*Fz must be *reached*, not just bounded (V2 relies on this)."""
    alpha = np.linspace(0, 1.5, 200001)
    _, fy = tire_forces(0 * alpha, alpha, FZ, s.mu, s.B, s.C, s.E, 0.0)
    assert fy.max() == pytest.approx(s.mu * FZ, rel=1e-6)


def test_lateral_force_is_odd_in_alpha():
    s = SURFACES["snow"]
    k = np.linspace(-0.5, 0.5, 51)[:, None]
    a = np.linspace(0.01, 1.4, 51)[None, :]
    fxp, fyp = tire_forces(k, a, FZ, s.mu, s.B, s.C, s.E, 0.0)
    fxn, fyn = tire_forces(k, -a, FZ, s.mu, s.B, s.C, s.E, 0.0)
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
            fx, fy = tire_forces(K, A, Z, mu, s.B, s.C, s.E, ca)
            assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fy))
            assert np.all(np.hypot(fx, fy) <= mu * Z * (1 + 1e-9))


def test_alpha_is_clamped_before_tan():
    s = SURFACES["dry"]
    fx, fy = tire_forces(0.0, np.pi / 2, FZ, s.mu, s.B, s.C, s.E, 0.0)
    assert np.isfinite(fy) and ALPHA_MAX < np.pi / 2


def test_clip_to_circle():
    fx, fy = clip_to_circle(np.array([3.0, 0.3]), np.array([4.0, 0.4]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(np.hypot(fx, fy), [1.0, 0.5])
    np.testing.assert_allclose(fx / fy, [0.75, 0.75])
