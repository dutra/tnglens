import csv
import filecmp
import os
import re
import warnings

import h5py

import numpy as np
import pytest
import yaml

from tnglens.catalog import build_catalog, main, resolve_config
from tnglens.lensmap import analytic_lens, write_synthetic_lens

from test_members import make_catalog

# kappaccino's allowed keys (kappaccino/io/image_catalog.py and member_catalog.py)
TOP_KEYS = {"format", "defaults", "systems", "notes"}
SYSTEM_KEYS = {"z", "z_err", "reliability", "knots", "images", "notes"}
IMAGE_KEYS = {"ra", "dec", "shape", "mag", "mag_err", "bands", "prior", "notes", "parity"}
MEMBER_TOP = {"band", "kinematics_sources", "members", "notes"}
MEMBER_KEYS = {"ra", "dec", "mag", "shape", "z", "role", "treatment", "keep", "p_member", "kinematics", "notes"}
CFG = dict(n_systems=6, seed=11, finder_stride=2, quotas=None, knots=dict(n_systems=0),
           sources=dict(seed_kinds=dict(negative=0.45, critical=0.55)))    # the plain catalog: first n_systems
LENS = analytic_lens()


@pytest.fixture(scope="module")
def lens_with_members(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    meta = make_catalog(root, "yz")
    attrs = {k: v for k, v in meta.items() if k not in ("H0", "Om0", "lens_redshift")}
    return write_synthetic_lens(root / "lens.h5", attrs=attrs)


@pytest.fixture(scope="module")
def built(lens_with_members, tmp_path_factory):
    out = tmp_path_factory.mktemp("cat")
    return build_catalog(lens_with_members, CFG, out), out


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def test_files_and_yaml_structure(built):
    paths, out = built
    for k in ("images_truth_yaml", "images_truth_csv", "sources_truth_csv", "members_truth_csv", "images_yaml",
              "members_yaml", "readme", "selection_yaml", "candidates_csv"):
        assert (out / paths[k].split(str(out) + "/")[1]).exists()
    assert paths["images_yaml"].endswith("observed/images.yaml")
    assert paths["images_truth_yaml"].endswith("truth/images_truth.yaml")
    for name in ("images_yaml", "images_truth_yaml"):
        doc = yaml.safe_load(open(paths[name]))
        assert set(doc) == TOP_KEYS and doc["format"] == "lenscluster-images/1"
        assert len(doc["systems"]) == CFG["n_systems"]
        for sname, sysd in doc["systems"].items():
            assert isinstance(sname, str) and re.fullmatch(r"[A-Z0-9]+", sname)
            assert set(sysd) <= SYSTEM_KEYS and sysd["z"] > 0
            assert len(sysd["images"]) >= (2 if name == "images_yaml" else 3)
            for label, im in sysd["images"].items():
                assert re.fullmatch(r"[a-z]+", label) and set(im) <= IMAGE_KEYS
                assert im["parity"] in (-1, 1) and abs(im["ra"]) < 0.02 and abs(im["dec"]) < 0.02
    mem = yaml.safe_load(open(paths["members_yaml"]))
    assert set(mem) <= MEMBER_TOP and mem["band"] == "F125W"
    roles = [m.get("role", "member") for m in mem["members"].values()]
    assert roles.count("bcg") == 1 and len(roles) == 3
    for m in mem["members"].values():
        assert set(m) <= MEMBER_KEYS and set(m["mag"]) == {"F125W"}


def test_truth_obeys_the_analytic_lens(built):
    paths, _ = built
    rows = read_csv(paths["images_truth_csv"])
    th = np.array([[float(r["x_true"]), float(r["y_true"])] for r in rows])
    beta = np.array([[float(r["beta_x"]), float(r["beta_y"])] for r in rows])
    s = np.array([float(r["efficiency"]) for r in rows])
    v = LENS(th[:, 0], th[:, 1])
    resid = th - s[:, None] * np.column_stack([v["ax"], v["ay"]]) - beta
    assert np.max(np.hypot(*resid.T)) < 2e-3
    det = (1 - s * v["pxx"]) * (1 - s * v["pyy"]) - (s * v["pxy"]) ** 2
    assert all(int(r["parity"]) == np.sign(d) for r, d in zip(rows, det))
    # observability: |mu| > 0.25 and > 3" from the BCG at the origin; obs positions only for those
    for r in rows:
        mu, rad = abs(float(r["mu"])), np.hypot(float(r["x_true"]), float(r["y_true"]))
        assert (r["observable"] == "True") == (mu > 0.25 and rad > 3.0)
        assert (r["x_obs"] == "nan") == (r["observable"] == "False")
    # time delays: zero for the first image of each system, positive otherwise
    by = {}
    for r in rows:
        by.setdefault(r["system"], []).append(float(r["dt_days"]))
    assert all(min(v) == 0.0 and sum(x == 0.0 for x in v) == 1 for v in by.values())
    # observed catalog = observable images, noisy at about sigma_pos
    obs = yaml.safe_load(open(paths["images_yaml"]))["systems"]
    d = []
    for r in rows:
        if r["observable"] == "True":
            im = obs[r["system"]]["images"][r["image"]]
            x, y = -im["ra"] * 3600, im["dec"] * 3600
            assert (x, y) == pytest.approx((float(r["x_obs"]), float(r["y_obs"])), abs=1e-5)
            d += [x - float(r["x_true"]), y - float(r["y_true"])]
    assert len(d) == 2 * sum(len(s["images"]) for s in obs.values())
    assert 0.015 < np.std(d) < 0.05


def test_selection_counts(built):
    paths, _ = built
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    c = sel["status_counts"]
    assert c["accepted"] == CFG["n_systems"] == sel["n_systems"]
    assert sum(c.values()) == sel["n_candidates"] == len(read_csv(paths["candidates_csv"]))
    src = read_csv(paths["sources_truth_csv"])
    assert len(src) == CFG["n_systems"]
    assert all(int(r["n_observable"]) >= 2 and float(r["min_sep"]) >= 0.3 for r in src)
    assert sum(sel["observable_multiplicity"].values()) == CFG["n_systems"]
    assert [float(r["z_true"]) for r in src] == sorted(float(r["z_true"]) for r in src)
    assert sel["members"]["central_id"] == 300 and "workers" not in sel["config"]
    readme = open(paths["readme"]).read()
    assert "not an unbiased source population" in readme and "seed 11" in readme


def test_deterministic_and_independent_of_workers_and_batch(lens_with_members, built, tmp_path):
    paths, out = built
    build_catalog(lens_with_members, dict(CFG, workers=2, batch=5), tmp_path / "b")
    for sub in ("truth", "observed"):
        cmp = filecmp.dircmp(out / sub, tmp_path / "b" / sub)
        assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    assert filecmp.cmp(out / "README.md", tmp_path / "b" / "README.md", shallow=False)
    build_catalog(lens_with_members, dict(CFG, seed=12), tmp_path / "c")
    assert not filecmp.cmp(out / "truth/images_truth.csv", tmp_path / "c/truth/images_truth.csv", shallow=False)


def test_without_group_catalog_and_cli(tmp_path):
    lens = write_synthetic_lens(tmp_path / "lens.h5")          # group_catalog_base is empty
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({"catalog": dict(n_systems=3, seed=5,
                                                                         noise=dict(sigma_pos=0.0))}))
    assert main([str(lens), "--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "o"),
                 "--override", "axis=zx"]) == 0
    mem = yaml.safe_load(open(tmp_path / "o/observed/members.yaml"))
    assert mem["members"] == {}
    sel = yaml.safe_load(open(tmp_path / "o/truth/selection.yaml"))
    assert "member table skipped" in " ".join(sel["notes"]) and sel["config"]["overrides"] == {"axis": "zx"}
    rows = read_csv(tmp_path / "o/truth/images_truth.csv")
    assert all(r["x_obs"] == r["x_true"] for r in rows if r["observable"] == "True")


def test_csv_keeps_significant_digits(built):
    paths, _ = built
    rows = read_csv(paths["images_truth_csv"])
    res = [float(r["residual_arcsec"]) for r in rows]
    assert max(res) < 1e-7 and any(v > 0 for v in res)           # %.6f printed them all as 0.000000
    for r in rows:                                               # 9 significant digits, whatever the size
        for k in ("mu", "kappa", "x_true", "dt_days", "residual_arcsec"):
            assert r[k] == f"{float(r[k]):.9g}"
    small = [r["mu"] for r in rows if abs(float(r["mu"])) < 0.1]
    assert small and all(len(re.sub(r"e.*|[-.]", "", m).lstrip("0")) >= 6 for m in small)


def test_bcg_core_images_are_flagged(built):
    paths, _ = built
    rows = read_csv(paths["images_truth_csv"])
    n_in = {}
    for r in rows:
        inside = np.hypot(float(r["x_true"]), float(r["y_true"])) <= 3.0       # BCG at the origin
        assert (r["inside_bcg_radius"] == "True") == inside
        assert not (inside and r["observable"] == "True")
        n_in[r["system"]] = n_in.get(r["system"], 0) + inside
    assert sum(n_in.values()) > 0                                # the cored analytic lens has central images
    for r in read_csv(paths["sources_truth_csv"]):
        assert int(r["n_images_outside_bcg"]) == int(r["n_images"]) - n_in.get(r["system"], 0)
    assert "inside_bcg_radius" in open(paths["readme"]).read()


MINSEP = 2.5
NOISY = dict(n_systems=8, seed=11, observability=dict(min_separation=MINSEP),
             noise=dict(sigma_pos=0.5, photoz_fraction=1.0, photoz_sigma=1.0, photoz_zmin=0.0),
             knots=dict(n_systems=0), sources=dict(seed_kinds=dict(negative=0.45, critical=0.55)))


@pytest.fixture(scope="module")
def noisy(lens_with_members, tmp_path_factory):
    out = tmp_path_factory.mktemp("noisy")
    return build_catalog(lens_with_members, NOISY, out), out


def test_min_separation_holds_for_the_observed_positions(noisy):
    paths, _ = noisy
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    assert sel["status_counts"]["close_pair_observed"] > 0       # 0.5" noise moves a pair below the 2.5" cut
    for sysd in yaml.safe_load(open(paths["images_yaml"]))["systems"].values():
        p = np.array([[-im["ra"] * 3600, im["dec"] * 3600] for im in sysd["images"].values()])
        d = np.linalg.norm(p[:, None] - p[None], axis=2)[np.triu_indices(len(p), 1)]
        assert d.min() >= MINSEP - 1e-6
    for r in read_csv(paths["sources_truth_csv"]):
        assert float(r["min_sep"]) >= MINSEP and float(r["min_sep_observed"]) >= MINSEP
    status = {r["index"]: r["status"] for r in read_csv(paths["candidates_csv"])}
    assert sum(v == "close_pair_observed" for v in status.values()) == sel["status_counts"]["close_pair_observed"]


def test_photometric_redshifts_stay_behind_the_lens_and_report_their_own_error(noisy):
    paths, _ = noisy
    for r in read_csv(paths["sources_truth_csv"]):
        zc, ze = float(r["z_catalog"]), float(r["z_err"])
        assert zc >= 0.3 + 0.01                                  # lens at z = 0.3; photoz_zmin = 0 and sigma = 1
        assert ze == pytest.approx(round(1.0 * (1 + zc), 3))     # from the catalogued z, not the true one
    obs = yaml.safe_load(open(paths["images_yaml"]))["systems"]
    assert all(s["z"] > 0.3 and s["z_err"] == pytest.approx(round(1 + s["z"], 3)) for s in obs.values())


def test_noise_is_a_function_of_seed_and_candidate(lens_with_members, noisy, tmp_path):
    """The noisy cut cannot depend on the workers or on which candidates were kept before."""
    paths, out = noisy
    build_catalog(lens_with_members, dict(NOISY, n_systems=3, workers=2, batch=3), tmp_path / "w")
    a = {r["candidate_index"]: r for r in read_csv(paths["sources_truth_csv"])}
    b = read_csv(tmp_path / "w/truth/sources_truth.csv")
    for r in b:
        for k in ("z_catalog", "z_err", "min_sep_observed", "source_mag"):
            assert r[k] == a[r["candidate_index"]][k]


def test_empty_result_writes_an_empty_mapping(lens_with_members, tmp_path):
    cfg = dict(n_systems=2, max_candidates=4, seed=3, observability=dict(min_images=40))
    paths = build_catalog(lens_with_members, cfg, tmp_path / "e")
    for name in ("images_yaml", "images_truth_yaml"):
        doc = yaml.safe_load(open(paths[name]))
        assert doc["systems"] == {}
    assert "no system accepted" in " ".join(yaml.safe_load(open(paths["selection_yaml"]))["notes"])
    # kappaccino's loader takes the empty mapping as a mapping and then refuses an empty catalog as such
    ic = pytest.importorskip("kappaccino.io.image_catalog")
    with pytest.raises(ic.CatalogError, match="no systems"):
        ic.load_image_catalog(paths["images_yaml"])


def test_foreground_sources(lens_with_members, tmp_path):
    bad = dict(CFG, sources=dict(redshift=dict(values=[0.1, 0.3])))       # lens at z = 0.3
    with pytest.raises(ValueError, match="in front of the lens"):
        build_catalog(lens_with_members, bad, tmp_path / "f0")
    part = dict(CFG, n_systems=2, sources=dict(redshift=dict(values=[0.2, 2.0])))
    with pytest.warns(UserWarning, match="50.0% of the sources.redshift"):
        paths = build_catalog(lens_with_members, part, tmp_path / "f1")
    rows = read_csv(paths["candidates_csv"])
    fg = [r for r in rows if r["status"] == "foreground"]
    assert fg and all(float(r["z"]) == 0.2 and r["kind"] for r in fg)
    assert all(float(r["z"]) == 2.0 for r in rows if r["status"] != "foreground")
    assert yaml.safe_load(open(paths["selection_yaml"]))["status_counts"]["foreground"] == len(fg)


def test_missing_lens_attributes_and_members(lens_with_members, tmp_path):
    p = write_synthetic_lens(tmp_path / "old.h5", n=128)
    with h5py.File(p, "a") as f:
        del f.attrs["H0"], f.attrs["lens_redshift"]
    with pytest.raises(KeyError, match="pass them as overrides"):
        build_catalog(p, dict(CFG, n_systems=1), tmp_path / "o1")
    # members.enabled: true without a group catalog: name the attribute, or the absolute missing path
    p2 = write_synthetic_lens(tmp_path / "nogroups.h5", n=128)
    with pytest.raises(KeyError, match="group_catalog_base"):
        build_catalog(p2, dict(CFG, n_systems=1, members=dict(enabled=True)), tmp_path / "o2")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path / "nowhere" / "groups_099"))):
            build_catalog(p2, dict(CFG, n_systems=1, members=dict(enabled=True),
                                   overrides=dict(group_catalog_base="nowhere")), tmp_path / "o3")
    finally:
        os.chdir(cwd)


def test_search_half_auto_without_trusted_width(tmp_path):
    p = write_synthetic_lens(tmp_path / "small.h5", n=256, half=32.0)
    with h5py.File(p, "a") as f:
        del f.attrs["trusted_half_width_arcsec"]
    paths = build_catalog(p, dict(n_systems=2, seed=4), tmp_path / "s")
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    extent = 32.0 - 0.125                                        # outermost pixel centre
    assert sel["search_half"] == pytest.approx(extent - (16 + 3 * 2) * 0.25)
    assert any("no trusted half-width" in n for n in sel["notes"]) and sel["n_systems"] == 2


def test_outputs_do_not_depend_on_the_lens_path_spelling(lens_with_members, built, tmp_path):
    paths, out = built
    other = os.path.relpath(os.path.join(os.path.dirname(lens_with_members), ".", "lens.h5"))
    build_catalog(other, CFG, tmp_path / "p")
    for sub in ("truth", "observed"):
        cmp = filecmp.dircmp(out / sub, tmp_path / "p" / sub)
        assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    assert filecmp.cmp(out / "README.md", tmp_path / "p" / "README.md", shallow=False)
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    assert sel["lens_file"] == "lens.h5" and sel["lens_file_attributes"]["zoom_index"] == 2
    assert str(lens_with_members) not in open(paths["readme"]).read()


def test_config_validation():
    with pytest.raises(ValueError, match="n_sytems"):
        resolve_config(dict(n_sytems=3))
    with pytest.raises(ValueError, match="sigma"):
        resolve_config(dict(noise=dict(sigma=0.1)))
    for bad, key in ((dict(n_systems=0), "n_systems"), (dict(seed=1.5), "seed"),
                     (dict(search_half="big"), "search_half"), (dict(observability=dict(mu_min="x")), "mu_min"),
                     (dict(sources=dict(seed_kinds=dict(blob=1.0))), "seed_kinds"),
                     (dict(sources=dict(redshift=dict(edges=[2, 1], weights=[1]))), "edges"),
                     (dict(sources=dict(redshift=dict(edges=[1, 2, 3], weights=[1]))), "weights"),
                     (dict(noise=dict(photoz_fraction=2)), "photoz_fraction"),
                     (dict(members=dict(enabled="yes")), "members.enabled"), (dict(noise=0.1), "noise"),
                     (dict(overrides=[1]), "overrides")):
        with pytest.raises(ValueError, match=key):
            resolve_config(bad)
    c = resolve_config(dict(observability=dict(mu_min=1.0)))
    assert c["observability"]["mu_min"] == 1.0 and c["observability"]["bcg_radius"] == 3.0
    assert resolve_config(c) == c


def test_kappaccino_reads_the_catalogs(built):
    ic = pytest.importorskip("kappaccino.io.image_catalog")
    mc = pytest.importorskip("kappaccino.io.member_catalog")
    paths, _ = built
    assert len(ic.load_image_catalog(paths["images_yaml"])) > 0
    assert len(ic.load_image_catalog(paths["images_truth_yaml"])) > 0
    assert len(mc.load_member_catalog(paths["members_yaml"], (3, 0.0, 0.0))) == 3
