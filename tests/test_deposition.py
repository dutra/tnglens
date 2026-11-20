import numpy as np

from tnglens.deposition import deposit_small


def test_mass_is_conserved_for_contained_kernels():
    rng = np.random.default_rng(0)
    n, width = 128, 64.0
    x, y = rng.uniform(-25, 25, (2, 5000))
    h = rng.uniform(0.05, 4.0, 5000)          # CIC and stencil kernels (dx = 0.5)
    m = rng.uniform(1, 3, 5000)
    image = np.zeros((n, n))
    deposit_small(x, y, h, m, width, image)
    assert np.isclose(image.sum() * (width / n) ** 2, m.sum(), rtol=1e-12)


def test_cic_conserves_first_moment():
    n, width = 64, 64.0
    x, y, m = np.array([3.3]), np.array([-7.8]), np.array([2.0])
    image = np.zeros((n, n))
    deposit_small(x, y, np.array([0.5]), m, width, image)
    c = (np.arange(n) + 0.5) - width / 2
    total = image.sum()
    assert np.isclose((image.sum(axis=0) * c).sum() / total, 3.3)
    assert np.isclose((image.sum(axis=1) * c).sum() / total, -7.8)


def test_stencil_is_normalised_before_clipping():
    # A kernel centred on the edge loses the part outside the image, nothing more.
    n, width = 64, 64.0
    image = np.zeros((n, n))
    deposit_small(np.array([32.0]), np.array([0.0]), np.array([3.0]), np.array([1.0]), width, image)
    deposited = image.sum() * (width / n) ** 2
    assert 0.4 < deposited < 0.6
