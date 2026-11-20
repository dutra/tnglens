import h5py
import numpy as np
import pytest
from scipy import ndimage

from tnglens import cosmology, lensmap
from tnglens.lensmap import LensMap, Spline, analytic_lens, map_extent, read_meta, require, write_synthetic_lens


@pytest.fixture(scope="module")
def lensfile(tmp_path_factory):
    return write_synthetic_lens(tmp_path_factory.mktemp("lens") / "lens.h5")


def test_values_and_derivatives_match_the_analytic_lens(lensfile):
    L = LensMap.from_file(lensfile)
    f = analytic_lens()
    rng = np.random.default_rng(3)
    p = rng.uniform(-50, 50, (4000, 2))
    v = f(p[:, 0], p[:, 1])
    a = L.alpha(p)
    assert np.max(np.abs(a[:, 0] - v["ax"])) < 2e-4
    assert np.max(np.abs(a[:, 1] - v["ay"])) < 2e-4
    axx, axy, ayx, ayy, a2 = L.jacobian(p)
    assert np.array_equal(a, a2)
    for got, want in ((axx, v["pxx"]), (ayy, v["pyy"]), (axy, v["pxy"]), (ayx, v["pxy"])):
        assert np.max(np.abs(got - want)) < 2e-3
    psi = L.potential(p)
    d = psi - v["psi"]
    assert np.ptp(d) < 2e-3            # equal up to a constant (here zero)
    k, g1, g2, det = L.hessian(p, 0.7)
    kt = 0.35 * (v["pxx"] + v["pyy"])
    assert np.max(np.abs(k - kt)) < 2e-3
    dt = (1 - 0.7 * v["pxx"]) * (1 - 0.7 * v["pyy"]) - 0.49 * v["pxy"] ** 2
    assert np.max(np.abs(det - dt)) < 5e-3


def test_spline_equals_map_coordinates_and_its_gradient_is_exact():
    rng = np.random.default_rng(0)
    a = ndimage.gaussian_filter(rng.normal(size=(80, 90)), 3)
    x = np.arange(90) * 0.5 - 20
    y = np.arange(80) * 0.25 - 10
    sp = Spline.fit(x, y, {"f": a})
    # the whole grid, edge cells included
    p = np.column_stack([rng.uniform(x[0], x[-1], 500), rng.uniform(y[0], y[-1], 500)])
    p[:20, 0], p[20:40, 0], p[40:60, 1] = x[0], x[-1], y[0]
    uv = [(p[:, 1] - y[0]) / 0.25, (p[:, 0] - x[0]) / 0.5]
    want = ndimage.map_coordinates(sp.coef["f"], uv, order=3, prefilter=False, mode="mirror")
    val, gx, gy = sp(p, ("f",), deriv=True)["f"]
    assert np.max(np.abs(val - want)) < 1e-12
    assert np.max(np.abs(val - ndimage.map_coordinates(a, uv, order=3, mode="mirror"))) < 1e-12
    h = 1e-5
    fx = (sp(p + [h, 0], ("f",))["f"] - sp(p - [h, 0], ("f",))["f"]) / (2 * h)
    fy = (sp(p + [0, h], ("f",))["f"] - sp(p - [0, h], ("f",))["f"]) / (2 * h)
    assert np.max(np.abs(gx - fx)) < 1e-6 and np.max(np.abs(gy - fy)) < 1e-6
    # interpolates the samples
    node = sp(np.array([[x[10], y[20]]]), ("f",))["f"][0]
    assert abs(node - a[20, 10]) < 1e-12


def test_spline_interpolates_the_map_edges(lensfile):
    """Prefilter and evaluation share the boundary: every pixel, first and last rows and columns included."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(30, 40)) + np.linspace(0, 5, 40)          # a gradient makes a boundary error visible
    x, y = np.arange(40) * 0.3 - 6, np.arange(30) * 0.3 - 4
    sp = Spline.fit(x, y, {"f": a})
    X, Y = np.meshgrid(x, y)
    v = sp(np.column_stack([X.ravel(), Y.ravel()]), ("f",))["f"].reshape(a.shape)
    assert np.max(np.abs(v - a)) < 1e-12
    # the lens map at its first and last pixel columns and rows equals the stored values
    L = LensMap.from_file(lensfile)
    with h5py.File(lensfile, "r") as f:
        ax = f["alpha_x"][:].astype(np.float64)
    for col in (0, 1, -2, -1):
        pts = np.column_stack([np.full(len(L.y), L.x[col]), L.y])
        assert np.max(np.abs(L.alpha(pts)[:, 0] - ax[:, col])) < 1e-9
    pts = np.column_stack([L.x, np.full(len(L.x), L.y[0])])
    assert np.max(np.abs(L.alpha(pts)[:, 0] - ax[0])) < 1e-9


def test_axes_agree_with_the_projection():
    projection = pytest.importorskip("tnglens.projection")
    assert lensmap.AXES == projection.AXES


def test_map_extent(lensfile):
    half, dx = map_extent(lensfile)
    assert dx == pytest.approx(0.25) and half == pytest.approx(64.0 - 0.125)


def test_cropped_spline_equals_full(lensfile):
    full = LensMap.from_file(lensfile)
    crop = LensMap.from_file(lensfile, half=30.0)
    assert crop.spline.coef["alpha_x"].shape[0] < full.spline.coef["alpha_x"].shape[0]
    p = np.random.default_rng(1).uniform(-30, 30, (1000, 2))
    assert np.max(np.abs(full.alpha(p) - crop.alpha(p))) < 1e-9
    with pytest.raises(ValueError):
        LensMap.from_file(lensfile, half=80.0)


def test_metadata_overrides_aliases_and_cosmology(tmp_path):
    p = write_synthetic_lens(tmp_path / "l.h5", n=64, attrs=dict(H0=70.0, Om0=0.3))
    with h5py.File(p, "a") as f:          # an older file: zoom instead of zoom_index, no axis
        del f.attrs["zoom_index"], f.attrs["axis"]
        f.attrs["zoom"] = 14
    m = read_meta(p)
    assert m["zoom_index"] == 14 and "axis" not in m
    with pytest.raises(KeyError):
        require(m, ("axis",), "test")
    m = read_meta(p, {"axis": "zx", "H0": 67.74})
    assert m["axis"] == "zx" and m["H0"] == 67.74
    with pytest.raises(ValueError):
        read_meta(p, {"axis": "xz"})
    L = LensMap.from_file(p, {"H0": 70.0, "Om0": 0.3})
    c = cosmology.make({"H0": 70.0, "Om0": 0.3})
    assert L.efficiency(2.0) == pytest.approx(cosmology.efficiency(c, 0.3, 2.0))
    assert L.kpc_per_arcsec == pytest.approx(cosmology.kpc_per_arcsec(c, 0.3))
