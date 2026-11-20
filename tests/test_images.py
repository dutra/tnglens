import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from tnglens.images import ImageFinder, classify, image_table, observable, separations
from tnglens.lensmap import LensMap, analytic_lens, write_synthetic_lens

LENS = analytic_lens()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = write_synthetic_lens(tmp_path_factory.mktemp("lens") / "lens.h5")
    L = LensMap.from_file(path, half=50.0)
    return L, ImageFinder(L, 45.0, stride=2)


def analytic_images(beta, s, half=45.0, step=0.4):
    """Every solution of the analytic lens equation: Newton from a dense grid of starts."""
    g = np.arange(-half, half + step, step)
    th = np.array(np.meshgrid(g, g)).reshape(2, -1).T.astype(float)
    for _ in range(60):
        v = LENS(th[:, 0], th[:, 1])
        f = th - s * np.column_stack([v["ax"], v["ay"]]) - beta
        a11, a22, a12 = 1 - s * v["pxx"], 1 - s * v["pyy"], -s * v["pxy"]
        det = a11 * a22 - a12 * a12
        dx = (a22 * f[:, 0] - a12 * f[:, 1]) / det
        dy = (-a12 * f[:, 0] + a11 * f[:, 1]) / det
        n = np.maximum(np.hypot(dx, dy), 1e-300)
        k = np.minimum(1, 1.0 / n)
        th -= np.column_stack([k * dx, k * dy])
    v = LENS(th[:, 0], th[:, 1])
    f = th - s * np.column_stack([v["ax"], v["ay"]]) - beta
    ok = (np.hypot(*f.T) < 1e-10) & (np.max(np.abs(th), axis=1) <= half)
    roots = []
    for p in th[ok]:
        if not any(np.hypot(*(p - r)) < 1e-4 for r in roots):
            roots.append(p)
    return np.array(roots)


def analytic_props(theta, beta, s):
    v = LENS(theta[:, 0], theta[:, 1])
    det = (1 - s * v["pxx"]) * (1 - s * v["pyy"]) - s * s * v["pxy"] ** 2
    tau = 0.5 * np.sum((theta - beta) ** 2, axis=1) / s - v["psi"]
    return 1 / det, tau


SOURCES = [((0.6, -0.4), 0.7), ((1.5, 2.0), 0.8), ((-3.0, 0.5), 0.6), ((0.1, 0.05), 0.9), ((4.0, 3.0), 0.75),
           ((-0.9, -2.2), 0.65), ((12.0, 0.0), 0.7)]


@pytest.mark.parametrize("beta,s", SOURCES)
def test_images_match_the_analytic_lens(setup, beta, s):
    L, F = setup
    beta = np.array(beta)
    th = F.find(beta, s)
    want = analytic_images(beta, s)
    assert len(th) == len(want), (th, want)
    assert len(th) % 2 == 1                       # non-singular lens: odd number
    d = np.linalg.norm(want[:, None] - th[None], axis=2)
    j = d.argmin(1)
    assert len(set(j)) == len(want)
    assert d.min(1).max() < 2e-3                  # spline of float32 maps vs exact
    tab = image_table(L, th, beta, s)
    assert tab["resid"].max() < 1e-7              # solves the splined lens equation
    mu_true, tau_true = analytic_props(want, beta, s)
    mu = tab["mu"][j]
    assert np.array_equal(np.sign(mu), np.sign(mu_true))
    assert np.array_equal(tab["parity"][j], np.sign(mu_true).astype(int))
    assert np.allclose(mu, mu_true, rtol=2e-2, atol=1e-3)
    assert np.sum(np.abs(mu)) == pytest.approx(np.sum(np.abs(mu_true)), rel=1e-2)
    # time delays against the analytic Fermat potential, relative to the first image
    dt_true = (tau_true - tau_true.min()) * L.time_delay_days()
    assert np.allclose(tab["dt_days"][j], dt_true, rtol=1e-3, atol=2e-3 * L.time_delay_days())
    assert tab["dt_days"].min() == 0.0
    # first-arriving image is a minimum (positive parity, both eigenvalues of A positive)
    first = np.argmin(tab["dt_days"])
    assert tab["parity"][first] == 1 and 1 - tab["kappa"][first] > 0


def test_sum_of_signed_magnification_far_source(setup):
    """A source far from the caustics has one image with mu close to the analytic value."""
    L, F = setup
    beta, s = np.array([30.0, -20.0]), 0.7
    th = F.find(beta, s)
    assert len(th) == 1
    mu_true, _ = analytic_props(th, beta, s)
    assert image_table(L, th, beta, s)["mu"][0] == pytest.approx(mu_true[0], rel=1e-3)


def test_find_is_deterministic(setup):
    L, F = setup
    a = F.find(np.array([0.6, -0.4]), 0.7)
    b = F.find(np.array([0.6, -0.4]), 0.7)
    assert np.array_equal(a, b)


def test_observable_and_separations_and_classes():
    tab = dict(theta=np.array([[0.5, 0.0], [10.0, 0.0], [0.0, -20.0], [5.0, 5.0]]),
               mu=np.array([3.0, 2.0, -0.1, -4.0]))
    obs = observable(tab, (0.0, 0.0), r_exclude=3.0, mu_min=0.25)
    assert obs.tolist() == [False, True, False, True]
    assert separations(tab["theta"][obs]) == pytest.approx((np.hypot(5, 5), np.hypot(5, 5)))
    assert separations(tab["theta"][:1]) == (np.inf, 0.0)
    th = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 30.0]])
    assert classify(th, np.array([1, -1, 1])) == "wide"
    th = np.array([[0.0, 0.0], [0.5, 0.0], [30.0, 0.0]])
    assert classify(th, np.array([1, -1, 1])) == "fold"
    assert classify(th[:1], np.array([1])) == "single"


def test_search_square_limit(setup):
    L, _ = setup
    with pytest.raises(ValueError):
        ImageFinder(L, 60.0)


def test_singular_isothermal_sphere_closed_form(tmp_path):
    """Positions, magnifications and the time delay of an SIS against its closed form.

    psi = b |theta| (D_ds/D_s = 1): for a source at beta < s b the images lie on the source's axis at radii
    s b + beta (minimum, mu = 1 + s b / beta) and s b - beta (saddle, mu = 1 - s b / beta), and the delay is
    dt = 2 b beta (1 + z_l) D_l / c.  T is computed here from astropy, not from tnglens.cosmology.
    """
    zl, H0, Om0, b = 0.3, 70.0, 0.3, 15.0
    sis = analytic_lens(b=b, rc=0.0, e=0.0, gamma_ext=(0.0, 0.0))
    path = write_synthetic_lens(tmp_path / "sis.h5", lens=sis, n=512, half=64.0, zl=zl, H0=H0, Om0=Om0)
    L = LensMap.from_file(path, half=50.0)
    F = ImageFinder(L, 45.0, stride=2)
    T = ((1 + zl) * FlatLambdaCDM(H0=H0, Om0=Om0).angular_diameter_distance(zl) / const.c).to(u.day).value \
        * (np.pi / 180 / 3600) ** 2                                  # days per arcsec^2
    s = 0.7
    for beta_r, phi in ((3.0, 0.4), (1.5, 2.0), (5.0, -1.0)):
        e = np.array([np.cos(phi), np.sin(phi)])
        beta = beta_r * e
        th = F.find(beta, s)
        tab = image_table(L, th, beta, s)
        want = {"min": ((s * b + beta_r) * e, 1 + s * b / beta_r), "saddle": (-(s * b - beta_r) * e, 1 - s * b / beta_r)}
        idx = {}
        for k, (pos, mu) in want.items():
            d = np.hypot(*(th - pos).T)
            i = int(np.argmin(d))
            idx[k] = i
            assert d[i] < 2e-3, (k, th, pos)
            assert tab["mu"][i] == pytest.approx(mu, rel=5e-3)
            assert tab["parity"][i] == np.sign(mu)
        # images away from the unresolved centre are just these two
        assert sum(np.hypot(*th.T) > 2.0) == 2
        dt = tab["dt_days"][idx["saddle"]] - tab["dt_days"][idx["min"]]
        assert dt == pytest.approx(2 * b * beta_r * T, rel=1e-3)
        assert dt > 0
