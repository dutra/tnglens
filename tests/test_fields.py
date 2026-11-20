"""The field solver against an analytic Gaussian lens."""
import numpy as np
from scipy.special import exp1

from tnglens.fields import deflection_jacobian, isolated_fields, multiscale_fields
from tnglens.projection import pixel_centers


def gaussian(x, y, k0, s, cx=0.0, cy=0.0):
    """kappa, alpha, psi and shear of kappa = k0 exp(-r^2 / 2 s^2) (exact)."""
    dx, dy = x - cx, y - cy
    r2 = dx * dx + dy * dy
    u = r2 / (2 * s * s)
    kappa = k0 * np.exp(-u)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(u > 1e-12, 2 * k0 * s * s * (-np.expm1(-u)) / r2, k0)  # mean kappa inside r
        psi = k0 * s * s * np.where(u > 1e-12, exp1(u) + np.log(u), -np.euler_gamma)
        c2, s2 = np.where(r2 > 0, (dx * dx - dy * dy) / r2, 1.0), np.where(r2 > 0, 2 * dx * dy / r2, 0.0)
    return dict(kappa=kappa, alpha_x=mean * dx, alpha_y=mean * dy, potential=psi,
                gamma1=(kappa - mean) * c2, gamma2=(kappa - mean) * s2)


def test_isolated_fields_match_gaussian_lens():
    n, width, s = 256, 80.0, 4.0
    x = pixel_centers(width, n)
    X, Y = np.meshgrid(x, x)
    exact = gaussian(X, Y, 0.8, s, cx=3.0, cy=-5.0)
    out = isolated_fields(exact["kappa"], x[1] - x[0], threads=2,
                          names=("potential", "alpha_x", "alpha_y", "gamma1", "gamma2"))
    inner = (np.abs(X) < 30) & (np.abs(Y) < 30)
    scale = np.abs(exact["alpha_x"]).max()
    for name in ("alpha_x", "alpha_y"):
        assert np.abs(out[name] - exact[name])[inner].max() < 2e-3 * scale
    dpsi = (out["potential"] - exact["potential"])[inner]
    assert np.ptp(dpsi) < 2e-3 * np.ptp(exact["potential"][inner])  # equal up to a constant
    for name in ("gamma1", "gamma2"):
        assert np.abs(out[name] - exact[name])[inner].max() < 5e-3 * 0.8


def test_multiscale_adds_exterior_mass_and_keeps_core_kappa():
    """Core mass plus a clump outside the core, seen only by the coarse environment grid."""
    xf = pixel_centers(40.0, 128)
    xe = pixel_centers(120.0, 96)
    lens = lambda X, Y: [gaussian(X, Y, 0.6, 2.5, 3.0, -2.0), gaussian(X, Y, 0.5, 4.0, 35.0, 10.0)]
    Xf, Yf = np.meshgrid(xf, xf)
    Xe, Ye = np.meshgrid(xe, xe)
    fine_parts, env_parts = lens(Xf, Yf), lens(Xe, Ye)
    kf = sum(p["kappa"] for p in fine_parts)
    ke = sum(p["kappa"] for p in env_parts)
    fields, info = multiscale_fields(kf, ke, xf, xe, threads=2)
    assert np.array_equal(fields["kappa"], kf)
    lo, hi = info["trusted_interior"]
    assert np.isclose(info["guard"], 2 * (xe[1] - xe[0]))
    inner = (Xf > lo) & (Xf < hi) & (Yf > lo) & (Yf < hi)
    for name in ("alpha_x", "alpha_y"):
        exact = sum(p[name] for p in fine_parts)
        scale = np.abs(exact).max()
        assert np.abs(fields[name] - exact)[inner].max() < 5e-3 * scale
    # Without the environment the exterior clump's pull (a few percent of alpha) is missing.
    alone = isolated_fields(kf, xf[1] - xf[0], threads=2)
    exterior = fine_parts[1]["alpha_x"]
    assert np.abs(alone["alpha_x"] - sum(p["alpha_x"] for p in fine_parts))[inner].max() > 0.5 * np.abs(exterior[inner]).max()


def test_deflection_jacobian_of_gaussian_reproduces_kappa_and_shear():
    n, width, s = 256, 80.0, 4.0
    x = pixel_centers(width, n)
    X, Y = np.meshgrid(x, x)
    exact = gaussian(X, Y, 0.8, s)
    axx, axy, ayx, ayy = deflection_jacobian(exact["alpha_x"], exact["alpha_y"], x, x)
    inner = (np.abs(X) < 35) & (np.abs(Y) < 35)
    assert np.abs(0.5 * (axx + ayy) - exact["kappa"])[inner].max() < 2e-3
    assert np.abs(0.5 * (axx - ayy) - exact["gamma1"])[inner].max() < 2e-3
    assert np.abs(0.5 * (axy + ayx) - exact["gamma2"])[inner].max() < 2e-3
    # The central-difference form of the old pipeline agrees to O(step^2).
    cd = deflection_jacobian(exact["alpha_x"], exact["alpha_y"], x, x, step=0.02)
    for a, b in zip((axx, axy, ayx, ayy), cd):
        assert np.abs(a - b)[inner].max() < 1e-4
