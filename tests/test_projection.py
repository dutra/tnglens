import h5py
import numpy as np
import pytest

from tnglens.projection import Grid, MassLedgerError, make_grids, project

SETTINGS = {"depth_half_r200": 2.0, "support_floor_pkpc": 0.5, "target_cells_per_h": 4, "gas_support": "neighbors"}


def write_species(tmp_path, pt, pos, m, h):
    pf, sf = tmp_path / f"p{pt}.h5", tmp_path / f"s{pt}.h5"
    with h5py.File(pf, "w") as f:
        f["coordinates"] = pos.astype(np.float32)
        f["masses"] = m.astype(np.float32)
    with h5py.File(sf, "w") as f:
        f["neighbor_distance"] = h.astype(np.float32)
    return (pt, pf, sf)


def test_mass_ledger_closes_for_all_deposit_paths(tmp_path):
    rng = np.random.default_rng(3)
    n = 20000
    pos = rng.normal(0, 15, (n, 3))
    h = rng.uniform(0.05, 20.0, n)            # CIC, stencil and sphviewer2 kernels on both grids
    m = rng.uniform(1e6, 2e6, n)
    grids = (Grid("fine", 200.0, 256), Grid("environment", 400.0, 128))
    sigma, ledger = project([write_species(tmp_path, 1, pos, m, h)], "xy", grids, SETTINGS, R200c_pkpc=100.0,
                            box_pkpc=1e4, threads=2, chunk=7000)
    (entry,) = ledger
    for g in grids:
        row = entry["grids"][g.name]
        assert row["sphviewer_particles"] > 0 and row["stencil_particles"] > 0 and row["cic_particles"] > 0
        assert row["particles"] == n and np.isclose(row["selected_mass"], m.astype(np.float32).sum(), rtol=1e-6)
        assert row["fully_contained_mass"] == row["selected_mass"]  # every kernel inside the grid
        assert abs(row["deposited_mass"] / row["selected_mass"] - 1) < 1e-4
        assert row["mass_check_pass"]
        assert np.isclose(sigma[g.name].sum() * g.dx ** 2, row["deposited_mass"])


def test_depth_cut_and_axis_orientation(tmp_path):
    # One particle at simulation (x, y, z) = (10, -20, 30); for axis yz the map shows (y, z) = (-20, 30).
    pos = np.array([[10.0, -20.0, 30.0], [500.0, 0.0, 0.0]])  # second one: outside the slab along x
    spec = write_species(tmp_path, 4, pos, np.array([1e9, 1e9]), np.array([3.0, 3.0]))
    g = Grid("fine", 128.0, 128)
    sigma, ledger = project([spec], "yz", (g,), SETTINGS, R200c_pkpc=100.0, box_pkpc=1e4, threads=1)
    iy, ix = np.unravel_index(np.argmax(sigma["fine"]), sigma["fine"].shape)
    assert abs(g.centers[ix] - (-20.0)) < 1.0 and abs(g.centers[iy] - 30.0) < 1.0
    assert ledger[0]["grids"]["fine"]["particles"] == 1
    assert np.isclose(ledger[0]["slab_mass"], 1e9) and np.isclose(ledger[0]["fof_mass"], 2e9)


def test_environment_grid_is_at_least_one_and_a_half_cores():
    p = {"core_width_arcsec": 400.0, "core_resolution": 4096, "environment_width_r200": 4.0,
         "environment_resolution": 2048}
    fine, env = make_grids(p, R200c_pkpc=100.0, kpc_per_arcsec=5.0)
    assert fine.width == 2000.0 and env.width == 3000.0
    fine, env = make_grids(p, R200c_pkpc=2000.0, kpc_per_arcsec=5.0)
    assert env.width == 8000.0


@pytest.mark.parametrize("axis, expect", [("xy", (10.0, -20.0)), ("yz", (-20.0, 30.0)), ("zx", (30.0, 10.0))])
def test_axis_orientation_through_sphviewer2(tmp_path, axis, expect):
    # h = 12 pixels >= target_cells_per_h (4): rendered by sphviewer2, not the stencil.
    pos = np.array([[10.0, -20.0, 30.0]])
    spec = write_species(tmp_path, 1, pos, np.array([1e9]), np.array([12.0]))
    g = Grid("fine", 128.0, 128)
    sigma, ledger = project([spec], axis, (g,), SETTINGS, R200c_pkpc=100.0, box_pkpc=1e4, threads=1)
    row = ledger[0]["grids"]["fine"]
    assert row["sphviewer_particles"] == 1 and row["stencil_particles"] == 0 and row["cic_particles"] == 0
    image = sigma["fine"]
    total = image.sum()
    cx = (image.sum(axis=0) * g.centers).sum() / total
    cy = (image.sum(axis=1) * g.centers).sum() / total
    assert abs(cx - expect[0]) < 0.5 and abs(cy - expect[1]) < 0.5
    iy, ix = np.unravel_index(np.argmax(image), image.shape)
    assert abs(g.centers[ix] - expect[0]) <= 1.0 and abs(g.centers[iy] - expect[1]) <= 1.0
    assert abs(total * g.dx ** 2 / 1e9 - 1) < 5e-3  # sphviewer2 conserves a single kernel to ~0.3%


def test_mass_ledger_failure_fails_the_projection(tmp_path, monkeypatch):
    import tnglens.projection as projection
    real = projection.deposit_small
    monkeypatch.setattr(projection, "deposit_small", lambda x, y, h, m, *a, **k: real(x, y, h, 2 * m, *a, **k))
    pos = np.random.default_rng(1).normal(0, 10, (500, 3))
    spec = write_species(tmp_path, 4, pos, np.full(500, 1e8), np.full(500, 1.0))  # stencil/CIC only
    with pytest.raises(MassLedgerError, match="type 4 fine grid: deposited mass"):
        project([spec], "xy", (Grid("fine", 128.0, 128),), SETTINGS, R200c_pkpc=100.0, box_pkpc=1e4, threads=1)
