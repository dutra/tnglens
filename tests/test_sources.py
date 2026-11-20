import numpy as np
import pytest

from tnglens.images import ImageFinder, image_table
from tnglens.lensmap import LensMap, write_synthetic_lens
from tnglens.sources import SourceSampler, check_redshifts, draw_redshift, foreground_fraction

from test_images import analytic_images, analytic_props


@pytest.fixture(scope="module")
def finder(tmp_path_factory):
    path = write_synthetic_lens(tmp_path_factory.mktemp("lens") / "lens.h5")
    L = LensMap.from_file(path, half=50.0)
    return ImageFinder(L, 45.0, stride=2)


def test_draws_are_deterministic_and_independent_of_order(finder):
    a = SourceSampler(finder, {}, seed=7)
    b = SourceSampler(finder, {}, seed=7)
    fwd = [a.draw(i) for i in range(20)]
    rev = [b.draw(i) for i in reversed(range(20))][::-1]
    for x, y in zip(fwd, rev):
        assert (x.kind, x.z) == (y.kind, y.z) and np.array_equal(x.beta, y.beta)
    c = SourceSampler(finder, {}, seed=8).draw(0)
    assert not np.array_equal(c.beta, fwd[0].beta)


def test_redshift_specs():
    rng = np.random.default_rng(0)
    spec = dict(edges=[1.0, 2.0, 4.0], weights=[0.0, 1.0])
    z = [draw_redshift(rng, spec) for _ in range(200)]
    assert min(z) >= 2.0 and max(z) <= 4.0 and all(round(v, 4) == v for v in z)
    z = {draw_redshift(rng, dict(values=[1.5, 3.0])) for _ in range(100)}
    assert z == {1.5, 3.0}
    with pytest.raises(ValueError):
        draw_redshift(rng, dict(edges=[1, 2, 3], weights=[1.0]))


def test_image_plane_seeds_give_multiple_images(finder):
    S = SourceSampler(finder, dict(seed_kinds=dict(negative=1.0), redshift=dict(values=[2.0]),
                                   seed_radius=[1.0, 40.0]), seed=1)
    n = []
    for i in range(15):
        c = S.draw(i)
        assert c.kind == "negative" and c.seed is not None
        th = finder.find(c.beta, c.s)
        assert np.min(np.hypot(*(th - c.seed).T)) < 1e-6      # the seed is one of the images
        n.append(len(th))
    assert min(n) >= 3


def test_critical_seeds_high_magnification_match_analytic(finder):
    L = finder.lens
    S = SourceSampler(finder, dict(seed_kinds=dict(critical=1.0), critical_det=[0.01], redshift=dict(values=[2.0]),
                                   seed_radius=[1.0, 40.0]), seed=2)
    big = 0
    for i in range(6):
        c = S.draw(i)
        th = finder.find(c.beta, c.s)
        want = analytic_images(c.beta, c.s)
        assert len(th) == len(want)
        d = np.linalg.norm(want[:, None] - th[None], axis=2)
        j = d.argmin(1)
        mu = image_table(L, th, c.beta, c.s)["mu"][j]
        mu_true, _ = analytic_props(want, c.beta, c.s)
        assert np.array_equal(np.sign(mu), np.sign(mu_true))
        assert np.allclose(mu, mu_true, rtol=0.1)
        big += np.abs(mu).max() > 20
    assert big >= 3


def test_source_kind_is_uniform_in_a_box(finder):
    S = SourceSampler(finder, dict(seed_kinds=dict(source=1.0), redshift=dict(values=[2.0])), seed=3)
    b = np.array([S.draw(i).beta for i in range(200)])
    s = S.lens.efficiency(2.0)
    bx, by = finder.source_plane(s)
    neg = finder.det(s) < 0
    assert b[:, 0].min() >= bx[neg].min() and b[:, 0].max() <= bx[neg].max()
    assert b[:, 1].min() >= by[neg].min() and b[:, 1].max() <= by[neg].max()
    assert S.draw(0).seed is None


def test_unknown_kind(finder):
    with pytest.raises(ValueError):
        SourceSampler(finder, dict(seed_kinds=dict(blob=1.0)), seed=0)


def test_foreground_sources_have_their_own_outcome(finder):
    S = SourceSampler(finder, dict(redshift=dict(values=[0.2, 0.3, 2.0])), seed=4)      # lens at z = 0.3
    out = [S.draw(i) for i in range(60)]
    fg = [c for c in out if c.outcome == "foreground"]
    assert fg and all(c.z <= 0.3 and c.beta is None and c.kind for c in fg)
    assert all(c.outcome == "ok" and c.z == 2.0 and c.beta is not None for c in out if c not in fg)


def test_no_seed_outcome(finder):
    S = SourceSampler(finder, dict(seed_kinds=dict(negative=1.0), redshift=dict(values=[2.0]),
                                   seed_radius=[44.0, 44.5]), seed=1)          # no det A < 0 node out there
    c = S.draw(0)
    assert c.outcome == "no_seed" and c.beta is None and c.z == 2.0


def test_foreground_fraction():
    assert foreground_fraction(dict(values=[0.2, 2.0], weights=[1, 3]), 0.3) == pytest.approx(0.25)
    assert foreground_fraction(dict(edges=[0.0, 1.0, 2.0], weights=[1, 1]), 0.5) == pytest.approx(0.25)
    assert check_redshifts(dict(edges=[1.0, 2.0], weights=[1]), 0.3) == 0.0
    with pytest.raises(ValueError, match="in front of the lens"):
        check_redshifts(dict(edges=[0.1, 0.3], weights=[1]), 0.3)
    with pytest.raises(ValueError):
        check_redshifts(dict(values=[0.3]), 0.3)
