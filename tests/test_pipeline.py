"""Halo lookup, particle reading, and the lens-file contract on a miniature simulation."""
import json
import os
import sys
import types

import h5py
import numpy as np
import pytest
import yaml
from scipy.interpolate import RectBivariateSpline

from synthetic import BOX, H, SNAP, make_simulation, small_config
from tnglens import __version__, cli, config, cosmology, halos, pipeline
from tnglens.cache import FileLock, to_json
from tnglens.maps import lens_config
from tnglens.particles import neighbor_distance, read_species

CONTRACT_MAPS = ("kappa", "alpha_x", "alpha_y", "potential", "gamma1", "gamma2", "Sigma")
CONTRACT_ATTRS = (
    "lens_redshift", "snapshot", "zoom_index", "halo_id", "axis", "H0", "Om0", "cosmology_source",
    "kpc_per_arcsec", "sigma_critical_inf_Msun_pkpc2", "M200c_Msun", "R200c_pkpc", "halo_position_ckpc_h",
    "box_ckpc_h", "h", "scale_factor", "group_catalog_base", "particle_types", "particle_scope",
    "source_redshifts", "trusted_half_width_arcsec", "support_floor_pkpc", "mass_ledger_json", "config_json",
    "tnglens_version", "description", "coordinates", "gamma_convention", "consistency",
    "kappa_vs_div_alpha_median", "kappa_vs_div_alpha_p99")


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    return make_simulation(root)


def test_inventory_and_lookup(sim):
    assert halos.inventory(sim) == {SNAP: [0, 1]}
    # The files have tracers (PartType3: ParentID/TracerID, no Coordinates), as in TNG; --check skips them.
    snapfile = halos.snapshot_path(sim, SNAP, 0)
    with h5py.File(snapfile) as f:
        assert "Coordinates" not in f["PartType3"] and f["Header"].attrs["NumPart_ThisFile"][3] > 0
    assert halos.particle_file_problems(snapfile) == []
    assert halos.particle_file_problems(snapfile, types=(0, 1, 2, 3, 4, 5)) == ["PartType3 missing"]
    assert halos.inventory(sim, check=True) == {SNAP: [0, 1]}
    primary = halos.find_halo(sim, SNAP, zoom=1)
    assert primary.halo_id == 3 and primary.primary and primary.name("xy") == "tngc_z001_s050_xy"
    other = halos.find_halo(sim, SNAP, halo_id=1)
    assert other.zoom == 0 and not other.primary and other.name("yz") == "tngc_z000_s050_yz_h1"
    assert np.isclose(primary.M200c_Msun, 2e4 * 1e10 / H)


def test_reader_units_and_periodic_wrap(sim, tmp_path):
    halo = halos.find_halo(sim, SNAP, zoom=0)
    dest = tmp_path / "type1.h5"
    assert read_species(halo, 1, dest, chunk=3000)
    assert not read_species(halo, 1, dest)  # cached
    with h5py.File(dest) as f:
        pos, m = f["coordinates"][:], f["masses"][:]
        assert np.allclose(m, 1e9, rtol=1e-6)
        assert np.isclose(f.attrs["total_mass_Msun"], 1e9 * len(m))
    # The halo straddles the box corner; relative coordinates must not jump by a box length.
    assert np.abs(pos).max() < 2000 and abs(np.median(pos[:, 1])) < 20
    # Low-resolution DM: MassTable[2] = 0, so the per-particle Masses are used.
    assert halo.mass_table[2] == 0
    dest = tmp_path / "type2.h5"
    assert read_species(halo, 2, dest)
    with h5py.File(halo.snapshot_file) as f:
        raw = f["PartType2/Masses"][:].astype(np.float64)
    with h5py.File(dest) as f:
        m = f["masses"][:]
        assert np.ptp(m) > 0 and np.allclose(m, raw * 1e10 / H, rtol=1e-6)
        assert np.isclose(f.attrs["total_mass_Msun"], raw.sum() * 1e10 / H)


def test_neighbor_distance_is_exact_and_periodic():
    rng = np.random.default_rng(5)
    box = 10.0
    pos = rng.uniform(0, box, (400, 3))
    pos[:50] = rng.uniform(0, 0.5, (50, 3))  # a clump at the corner, wrapped neighbours matter
    k = 8
    got = neighbor_distance(pos - 3.0, k, box, threads=1, batch=97)  # shifted: wrapping handled inside
    d = np.abs(pos[:, None, :] - pos[None, :, :])
    d = np.minimum(d, box - d)
    dist = np.sqrt((d ** 2).sum(-1))
    expect = np.sort(dist, axis=1)[:, k]
    assert np.allclose(got, expect)


def _statuses(summary, name):
    stages = {**summary["stages"], **summary["products"][name]["stages"]}
    return [stages[k]["status"] for k in pipeline.STAGES if k in stages]


def _fake_catalog(monkeypatch, calls):
    fake = types.ModuleType("tnglens.catalog")
    fake.build_catalog = lambda lens_path, cfg, outdir: calls.append((lens_path, cfg, outdir)) or {"n": 0}
    fake.resolve_config = lambda cfg: dict(cfg)
    monkeypatch.setitem(sys.modules, "tnglens.catalog", fake)


def spline_jacobian(ax, ay, x, y):
    """d alpha / d (x, y) from scipy's interpolating bicubic spline, independent of tnglens.fields."""
    out = []
    for a in (ax, ay):
        spl = RectBivariateSpline(y, x, np.asarray(a, dtype=np.float64), kx=3, ky=3, s=0)  # first axis is y
        out += [spl(y, x, dy=1), spl(y, x, dx=1)]
    return out  # axx, axy, ayx, ayy


def test_lens_file_contract_and_caching(sim, tmp_path, monkeypatch):
    calls = []
    _fake_catalog(monkeypatch, calls)
    cfg = config.validate(small_config(sim, tmp_path, catalog={"seed": 7, "anything": {"x": 1}}))
    (summary,) = pipeline.run(cfg)
    product = summary["products"]["tngc_z000_s050_yz"]
    assert "error" not in product, product.get("error")
    path = tmp_path / "out" / "tngc_z000_s050_yz" / "lens.h5"
    assert product["lens_file"] == str(path)
    assert calls == [(str(path), {"seed": 7, "anything": {"x": 1}}, str(path.parent))]

    with h5py.File(path) as f:
        for name in CONTRACT_MAPS + ("mu_zs1", "mu_zs2.5"):
            assert f[name].dtype == np.float32 and f[name].shape == (128, 128) and f[name].compression
        x, y = f["x"][:], f["y"][:]
        assert x.dtype == np.float64 and np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(x, -x[::-1])
        a = {k: v for k, v in f.attrs.items()}
        missing = [k for k in CONTRACT_ATTRS if k not in a]
        assert not missing, missing
        kappa, sigma = f["kappa"][:].astype(float), f["Sigma"][:].astype(float)
        ax, ay = f["alpha_x"][:], f["alpha_y"][:]
        g1, g2 = f["gamma1"][:].astype(float), f["gamma2"][:].astype(float)
        mu = f["mu_zs2.5"][:].astype(float)
        eff = f["mu_zs2.5"].attrs["efficiency"]

    assert a["axis"] == "yz" and a["zoom_index"] == 0 and a["halo_id"] == 0 and a["snapshot"] == SNAP
    assert a["cosmology_source"] == "native" and np.isclose(a["H0"], 70.0) and np.isclose(a["Om0"], 0.3)
    assert list(a["particle_types"]) == [0, 1, 2, 4, 5] and list(a["source_redshifts"]) == [1.0, 2.5]
    assert np.isclose(a["box_ckpc_h"], BOX)
    cosmo = cosmology.make({"H0": 70.0, "Om0": 0.3})
    assert np.isclose(a["sigma_critical_inf_Msun_pkpc2"], cosmology.sigma_critical_inf(cosmo, a["lens_redshift"]))
    assert np.isclose(eff, cosmology.efficiency(cosmo, a["lens_redshift"], 2.5))
    # config_json is the part of the config the lens file depends on (no catalog, threads, paths)
    assert json.loads(a["config_json"]) == json.loads(to_json(lens_config(cfg)))
    (proj,) = (tmp_path / "cache" / "projections" / "tngc_z000_s050_yz").glob("sigma-*.h5")
    assert proj.name == f"sigma-{a['projection_signature'][:16]}.h5"
    ledger = json.loads(a["mass_ledger_json"])
    assert [e["part_type"] for e in ledger] == [0, 1, 2, 4, 5]
    assert all(row["mass_check_pass"] for e in ledger for row in e["grids"].values())
    assert ledger[2]["grids"]["environment"]["selected_mass"] == 0  # low-res DM is far outside the grids

    # kappa is the deposited density.
    assert np.allclose(kappa, sigma / a["sigma_critical_inf_Msun_pkpc2"], rtol=1e-6, atol=0)
    # Shear and mu are the derivatives of the stored deflection, checked against scipy's own
    # interpolating bicubic spline (same knots, different boundary condition, so compare the interior).
    axx, axy, ayx, ayy = spline_jacobian(ax, ay, x, y)
    inner = slice(16, -16)
    scale = np.abs(axx).max()
    assert np.allclose(g1[inner, inner], (0.5 * (axx - ayy))[inner, inner], rtol=0, atol=1e-5 * scale)
    assert np.allclose(g2[inner, inner], (0.5 * (axy + ayx))[inner, inner], rtol=0, atol=1e-5 * scale)
    det = (1 - eff * axx) * (1 - eff * ayy) - eff ** 2 * axy * ayx
    assert np.allclose((1 / mu)[inner, inner], det[inner, inner], rtol=1e-5, atol=1e-5 * scale)
    trusted = np.abs(x) < a["trusted_half_width_arcsec"]
    residual = np.abs(kappa - 0.5 * (axx + ayy))[np.ix_(trusted, trusted)]
    assert np.isclose(a["kappa_vs_div_alpha_median"], np.median(residual), rtol=1e-3, atol=1e-7)
    assert a["kappa_vs_div_alpha_p99"] < 0.05 * kappa.max()

    # Mass in the core equals the particle mass inside the core footprint (all kernels contained).
    assert np.isclose(sigma.sum() * (a["kpc_per_arcsec"] * (x[1] - x[0])) ** 2,
                      sum(e["grids"]["fine"]["deposited_mass"] for e in ledger), rtol=1e-5)

    # stages.json records every stage, particles and smoothing included.
    recorded = json.loads((path.parent / "stages.json").read_text())["stages"]
    assert set(recorded) == set(pipeline.STAGES)
    assert all({"status", "seconds", "peak_rss_gib"} <= set(v) for v in recorded.values())
    assert recorded["particles"]["status"] == "done" and recorded["smoothing"]["status"] == "done"

    # Second run reuses every product and does not even open the particle caches.
    (again,) = pipeline.run(cfg)
    assert _statuses(again, "tngc_z000_s050_yz") == ["not needed", "not needed", "cached", "cached", "cached"]
    (forced,) = pipeline.run(cfg, force="maps")
    assert _statuses(forced, "tngc_z000_s050_yz") == ["not needed", "not needed", "cached", "done", "done"]
    assert len(calls) == 2
    (forced,) = pipeline.run(cfg, force="projection")
    assert _statuses(forced, "tngc_z000_s050_yz") == ["cached", "cached", "done", "done", "done"]

    # A rebuilt catalog first clears the previous catalog outputs, and nothing else.
    outdir = path.parent
    (outdir / "truth").mkdir(exist_ok=True)
    (outdir / "truth" / "stale.csv").write_text("old")
    (outdir / "observed").mkdir(exist_ok=True)
    (outdir / "README.md").write_text("old")
    (outdir / "notes.txt").write_text("mine")
    changed = config.validate(small_config(sim, tmp_path, catalog={"seed": 8}))
    (rebuilt,) = pipeline.run(changed)
    assert _statuses(rebuilt, "tngc_z000_s050_yz")[-2:] == ["cached", "done"]
    assert not (outdir / "truth").exists() and not (outdir / "observed").exists()
    assert not (outdir / "README.md").exists()
    assert (outdir / "notes.txt").read_text() == "mine" and path.exists()
    assert json.loads((outdir / "catalog.done.json").read_text())["tnglens_version"] == __version__


def test_config_cosmology_changes_only_angles(sim, tmp_path):
    base = small_config(sim, tmp_path)
    cfg = config.validate({**base, "cosmology": {"H0": 50.0, "Om0": 0.5}, "output_root": str(tmp_path / "o2")})
    (summary,) = pipeline.run(cfg)
    with h5py.File(summary["products"]["tngc_z000_s050_yz"]["lens_file"]) as f:
        assert f.attrs["cosmology_source"] == "config" and f.attrs["H0"] == 50.0
        cosmo = cosmology.make({"H0": 50.0, "Om0": 0.5})
        assert np.isclose(f.attrs["kpc_per_arcsec"], cosmology.kpc_per_arcsec(cosmo, f.attrs["lens_redshift"]))
        assert np.isclose(f.attrs["M200c_Msun"], 2e4 * 1e10 / H)  # masses unchanged


def test_only_filter_and_parallel_workers(sim, tmp_path):
    cfg = config.validate(small_config(sim, tmp_path, workers=2,
                                       halos=[{"zoom": [0, 1], "snapshot": SNAP, "axes": ["xy", "zx"]}]))
    with pytest.raises(config.ConfigError, match="matches no product"):
        pipeline.run(cfg, only=["nothing*"])
    summaries = pipeline.run(cfg, only=["tngc_z00?_s050_zx"])
    names = sorted(n for s in summaries for n in s["products"])
    assert names == ["tngc_z000_s050_zx", "tngc_z001_s050_zx"]
    assert all("error" not in p for s in summaries for p in s["products"].values())


def test_source_in_front_of_lens_is_rejected(sim, tmp_path):
    cfg = config.validate(small_config(sim, tmp_path, source_redshifts=[0.2, 2.0]))
    with pytest.raises(ValueError, match="not behind the lens"):
        pipeline.run(cfg)


def _lens_datasets(path):
    with h5py.File(path) as f:
        return {k: f[k][:] for k in f}


def test_default_with_low_res_dm_equals_old_default_when_none_is_in_the_box(sim, tmp_path):
    old = config.validate(small_config(sim, tmp_path, particle_types=[0, 1, 4, 5], output_root=str(tmp_path / "old")))
    new = config.validate(small_config(sim, tmp_path, output_root=str(tmp_path / "new")))
    assert new["particle_types"] == [0, 1, 2, 4, 5]
    (a,) = pipeline.run(old)
    (b,) = pipeline.run(new)
    assert b["stages"]["particles"]["built_types"] == [2]  # only the new type is read
    da = _lens_datasets(a["products"]["tngc_z000_s050_yz"]["lens_file"])
    db = _lens_datasets(b["products"]["tngc_z000_s050_yz"]["lens_file"])
    assert da.keys() == db.keys()
    for k in da:
        assert np.array_equal(da[k], db[k]), k


def test_low_res_dm_inside_the_box_is_deposited(tmp_path):
    base = make_simulation(tmp_path / "sim", lowres_offset_pkpc=(150.0, -100.0, 50.0))
    cfg = config.validate(small_config(base, tmp_path))
    (summary,) = pipeline.run(cfg)
    product = summary["products"]["tngc_z000_s050_yz"]
    assert "error" not in product, product.get("error")
    with h5py.File(product["lens_file"]) as f:
        ledger = json.loads(f.attrs["mass_ledger_json"])
    (lowres,) = [e for e in ledger if e["part_type"] == 2]
    row = lowres["grids"]["environment"]
    assert row["selected_mass"] > 0.3 * lowres["fof_mass"] and row["mass_check_pass"]
    assert row["deposited_mass"] >= row["fully_contained_mass"] > 0


def test_projection_cache_is_per_signature(sim, tmp_path):
    """Alternating cosmologies keep both projections; the maps stage checks the file it opens."""
    native = config.validate(small_config(sim, tmp_path, output_root=str(tmp_path / "native")))
    other = config.validate(small_config(sim, tmp_path, cosmology={"H0": 50.0, "Om0": 0.5},
                                         output_root=str(tmp_path / "h50")))
    for cfg in (native, other):
        pipeline.run(cfg)
    files = sorted((tmp_path / "cache" / "projections" / "tngc_z000_s050_yz").glob("sigma-*.h5"))
    assert len(files) == 2
    for cfg in (native, other):
        (again,) = pipeline.run(cfg, force="maps")
        assert _statuses(again, "tngc_z000_s050_yz")[2:4] == ["cached", "done"]
    with h5py.File(files[0], "r+") as f:
        sig = f.attrs["signature"]
    with pytest.raises(RuntimeError, match="expected"):
        pipeline._read_projection(files[0], "0" * 64)
    assert pipeline._read_projection(files[0], sig)[0].shape == (128, 128)


def test_keep_particles_false_removes_only_caches_it_built(sim, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cfg = config.validate(small_config(sim, tmp_path, keep_particles=False))
    (s,) = pipeline.run(cfg)
    assert "error" not in s["products"]["tngc_z000_s050_yz"]
    assert not list((cache / "particles").rglob("*.h5")) and not list((cache / "smoothing").rglob("*.h5"))
    # Fully cached rerun: nothing upstream is needed, so the missing particle caches do not matter.
    (s,) = pipeline.run(cfg)
    assert _statuses(s, "tngc_z000_s050_yz") == ["not needed", "not needed", "cached", "cached"]
    assert not (cache / "particles").exists() or not list((cache / "particles").rglob("*.h5"))

    # Caches that existed before the run are kept.
    keep = config.validate(small_config(sim, tmp_path))
    pipeline.run(keep, force="particles")
    before = sorted(p.name for p in (cache / "particles").rglob("*.h5"))
    assert before == [f"type{t}.h5" for t in (0, 1, 2, 4, 5)]
    (s,) = pipeline.run(cfg, force="projection")
    assert _statuses(s, "tngc_z000_s050_yz")[:3] == ["cached", "cached", "done"]
    assert sorted(p.name for p in (cache / "particles").rglob("*.h5")) == before

    # Caches built by this run but in use by another run are kept.  (Another run reading them holds
    # the lock shared, so the non-blocking exclusive attempt fails; simulated here, since a shared lock
    # held from the start would make this run wait to build them.)
    for p in list((cache / "particles").rglob("*.h5")) + list((cache / "smoothing").rglob("*.h5")):
        p.unlink()
    monkeypatch.setattr(FileLock, "try_exclusive", lambda self: False)
    (s,) = pipeline.run(cfg, force="projection")
    assert s["stages"]["particles"]["status"] == "done"
    assert len(list((cache / "particles").rglob("*.h5"))) == 5


def test_file_lock_shared_blocks_exclusive(tmp_path):
    a, b = FileLock(tmp_path / "x.lock"), FileLock(tmp_path / "x.lock")
    a.shared()
    b.shared()                      # shared locks coexist
    c = FileLock(tmp_path / "x.lock")
    assert not c.try_exclusive()    # ... and keep anyone from deleting
    a.release()
    b.release()
    assert c.try_exclusive()
    assert not a._fd and not FileLock(tmp_path / "x.lock").try_exclusive()
    c.release()


def _write_yaml(tmp_path, cfg):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_missing_halo_does_not_stop_others_and_only_skips_it(sim, tmp_path, capsys):
    raw = small_config(sim, tmp_path, halos=[{"zoom": 0, "snapshot": SNAP, "axes": ["yz"]},
                                             {"zoom": 7, "snapshot": SNAP, "axes": ["yz"]}])
    cfg = config.validate(raw)
    # --only that does not select the missing halo never looks it up.
    (s,) = pipeline.run(cfg, only=["tngc_z000_*"])
    assert s["halo"] == "tngc_z000_s050" and "error" not in s["products"]["tngc_z000_s050_yz"]
    # Without --only, the lookup failure is reported and the other halo still runs.
    results = pipeline.run(cfg)
    assert [r["halo"] for r in results] == ["snapshot 50 zoom 7", "tngc_z000_s050"]
    assert results[0]["lookup"] and "out of range" in results[0]["error"]
    assert pipeline.failed_names(results) == ["snapshot 50 zoom 7"]
    path = _write_yaml(tmp_path, raw)
    assert cli.main(["run", str(path)]) == 2
    out = capsys.readouterr().out
    assert "FAILED snapshot 50 zoom 7: lookup failed" in out and "summary: 1 product(s) done, 1 failed" in out
    assert "Traceback" not in out


def test_cli_user_errors_are_one_line_with_exit_2(sim, tmp_path, capsys):
    path = _write_yaml(tmp_path, small_config(sim, tmp_path))
    assert cli.main(["run", str(path), "--only", "nothing"]) == 2
    err = capsys.readouterr().err
    assert err.strip().startswith("tnglens: error:") and "matches no product" in err and "Traceback" not in err
    bad = _write_yaml(tmp_path, {**small_config(sim, tmp_path), "catalog": {"n_sytems": 3}})
    assert cli.main(["run", str(bad)]) == 2
    assert "unknown catalog key" in capsys.readouterr().err
    assert cli.main(["run", str(tmp_path / "missing.yaml")]) == 2
    with pytest.raises(SystemExit):
        cli.main(["run", str(path), "--workers", "0"])
    assert "must be >= 1" in capsys.readouterr().err


def test_one_failing_halo_in_a_parallel_run(tmp_path, capsys):
    base = make_simulation(tmp_path / "sim")
    with h5py.File(halos.snapshot_path(base, SNAP, 1), "r+") as f:
        del f["PartType4/Masses"]  # zoom 1 fails while reading its particles
    raw = small_config(base, tmp_path, workers=2, halos=[{"zoom": [0, 1], "snapshot": SNAP, "axes": ["yz"]}])
    results = pipeline.run(config.validate(raw))
    assert [r["halo"] for r in results] == ["tngc_z000_s050", "tngc_z001_s050"]
    assert "error" not in results[0] and "error" not in results[0]["products"]["tngc_z000_s050_yz"]
    assert "Masses" in results[1]["error"]
    assert pipeline.failed_names(results) == ["tngc_z001_s050"]
    assert cli.main(["run", str(_write_yaml(tmp_path, raw))]) == 1
    assert "summary: 1 product(s) done, 1 failed: tngc_z001_s050" in capsys.readouterr().out


def test_stale_partials_of_dead_processes_are_removed(sim, tmp_path):
    import subprocess
    from tnglens import cache
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    d = tmp_path / "cache" / "projections" / "tngc_z000_s050_yz"
    d.mkdir(parents=True)
    stale = [d / f"sigma-x.h5.partial-{cache._HOST}-{dead.pid}", d / f"sigma-y.h5.partial-{dead.pid}"]
    kept = [d / f"sigma-z.h5.partial-{cache._HOST}-{os.getpid()}",      # alive (this process)
            d / f"sigma-w.h5.partial-otherhost-{dead.pid}"]               # another host: unknown
    for p in stale + kept:
        p.write_text("")
    (s,) = pipeline.run(config.validate(small_config(sim, tmp_path)))
    assert "error" not in s["products"]["tngc_z000_s050_yz"]
    assert not any(p.exists() for p in stale) and all(p.exists() for p in kept)
