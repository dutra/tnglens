"""Class quotas, knots and images tied to satellites (the parts of the tngc014 mock)."""
import csv
import filecmp

import h5py
import numpy as np
import pytest
import yaml

from tnglens import cosmology
from tnglens.catalog import Quotas, build_catalog, resolve_config
from tnglens.images import ImageFinder, image_table, observable
from tnglens.lensmap import LensMap, analytic_lens, write_synthetic_lens
from tnglens.satellites import MinusLens, Patch, Satellites, finder_without, particle_reader

MACRO = analytic_lens()
QUOTAS = dict(wide=3, intermediate=2, cusp=1, fold=3)
CFG = dict(seed=5, finder_stride=2, quotas=QUOTAS, max_candidates=600, knots=dict(n_systems=2),
           sources=dict(seed_kinds=dict(negative=0.45, critical=0.55)))


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- quotas
def record(cls, min_sep=5.0, n_obs=3, wide_draw=0.0, sub=None):
    return dict(cls=cls, min_sep=min_sep, n_obs=n_obs, wide_draw=wide_draw, cf=[dict(sub=sub)] if sub is not None else [])


def test_quotas_fill_by_class_fold_bin_and_satellite():
    sel = resolve_config({})["selection"]
    q = Quotas(dict(wide=1, fold=4, subhalo=3), sel)
    assert q.wanted == dict(wide=1, intermediate=0, cusp=0, subhalo=3, other=0, fold=6)     # ceil(4 / 3) per bin
    assert q.offer(record("cusp")) == "no_quota" and q.offer(record("other")) == "no_quota"
    assert q.offer(record("wide", n_obs=2, wide_draw=0.9)) == "thinned"
    assert q.offer(record("wide", n_obs=2, wide_draw=0.1)) == "accepted"
    q.take(record("wide"))
    assert q.offer(record("wide")) == "quota_full"
    for sep, expect in ((0.5, "accepted"), (0.6, "accepted"), (0.65, "quota_full"), (1.0, "accepted"), (2.9, "accepted"),
                        (3.0, "no_quota"), (0.2, "no_quota")):
        r = record("fold", min_sep=sep)
        assert q.offer(r) == expect, sep
        if expect == "accepted":
            q.take(r)
    for sub, expect in ((7, "accepted"), (7, "accepted"), (7, "quota_full"), (8, "accepted"), (9, "quota_full")):
        r = record("subhalo", sub=sub)
        assert q.offer(r) == expect
        if expect == "accepted":
            q.take(r)
    assert not q.full() and q.missing() == dict(fold=2)
    assert Quotas(dict(wide=2), sel).offer(record("subhalo", sub=1)) == "no_quota"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("quota")
    lens = write_synthetic_lens(root / "lens.h5")
    return lens, build_catalog(lens, CFG, root / "a"), root


def test_catalog_with_quotas_and_knots(built):
    lens, paths, root = built
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    src = read_csv(paths["sources_truth_csv"])
    assert sel["quotas"] == dict(wide=3, intermediate=2, cusp=1, subhalo=0, other=0, fold=3)
    counts = sel["systems_by_class"]
    assert sum(counts.values()) == len(src) == sel["n_systems"] and all(counts[k] <= sel["quotas"][k] for k in counts)
    assert set(counts) <= set(QUOTAS) and counts["wide"] == 3
    folds = [float(r["min_sep"]) for r in src if r["cls"] == "fold"]
    assert all(sum(a <= f < b for f in folds) <= 1 for a, b in ((0.3, 0.7), (0.7, 1.5), (1.5, 3.0)))
    c = sel["status_counts"]
    assert c["accepted"] == len(src) and c["quota_full"] + c["no_quota"] > 0
    assert sum(c.values()) == sel["n_candidates"] == len(read_csv(paths["candidates_csv"]))
    missing = {k: sel["quotas"][k] - counts.get(k, 0) for k in QUOTAS if sel["quotas"][k] > counts.get(k, 0)}
    assert bool(missing) == any("quotas not filled" in n for n in sel["notes"])
    assert "kept by class quota" in open(paths["readme"]).read()


def test_knots_are_images_of_nearby_sources(built):
    lens, paths, _ = built
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    assert sel["n_systems_with_knots"] == 2
    rows = read_csv(paths["images_truth_csv"])
    src = {r["system"]: r for r in read_csv(paths["sources_truth_csv"])}
    knotted = sorted({r["system"] for r in rows if r["knot"]})
    assert len(knotted) == 2 and all(int(src[s]["n_knots"]) in (2, 3) for s in knotted)
    assert all(int(r["n_knots"]) == 0 for s, r in src.items() if s not in knotted)
    for name in knotted:
        mine = [r for r in rows if r["system"] == name]
        base = {r["image"]: r for r in mine if r["knot"] == "A"}
        for r in mine:
            s, x, y = float(r["efficiency"]), float(r["x_true"]), float(r["y_true"])
            v = MACRO(np.array([x]), np.array([y]))
            # the knot's image maps back to the knot's own source in the analytic lens
            assert abs(x - s * v["ax"][0] - float(r["beta_x"])) < 2e-3 and abs(y - s * v["ay"][0] - float(r["beta_y"])) < 2e-3
            a = base[r["image"]]
            assert r["parity"] == a["parity"]
            if r["knot"] != "A":
                off = np.hypot(float(r["beta_x"]) - float(a["beta_x"]), float(r["beta_y"]) - float(a["beta_y"]))
                assert 0.03 <= off <= 0.15 and abs(float(r["source_mag"]) - float(a["source_mag"]) - 1.0) < 1e-6
                if r["observable"] == "True":
                    assert 0.08 < np.hypot(x - float(a["x_true"]), y - float(a["y_true"])) < 4.0
    obs = yaml.safe_load(open(paths["images_yaml"]))["systems"]
    for name, system in obs.items():
        if name in knotted:
            names = system["knots"]
            assert names[0] == "A" and all(set(im["knots"]) == set(names) and "ra" not in im for im in system["images"].values())
        else:
            assert "knots" not in system and all("ra" in im for im in system["images"].values())


def test_quota_catalog_is_deterministic_and_kappaccino_reads_it(built, tmp_path):
    lens, paths, root = built
    build_catalog(lens, dict(CFG, workers=2, batch=7), tmp_path / "b")
    for sub in ("truth", "observed"):
        cmp = filecmp.dircmp(root / "a" / sub, tmp_path / "b" / sub)
        assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    ic = pytest.importorskip("kappaccino.io.image_catalog")
    assert len(ic.load_image_catalog(paths["images_yaml"])) > 0
    assert len(ic.load_image_catalog(paths["images_truth_yaml"])) > 0


def test_new_sections_are_validated():
    for bad, key in ((dict(quotas=dict(arcs=3)), "quotas"), (dict(quotas=dict(wide=-1)), "quotas.wide"),
                     (dict(quotas=dict(wide=0)), "quotas"), (dict(knots=dict(extra=[0, 2])), "knots.extra"),
                     (dict(knots=dict(n=3)), "knots"), (dict(selection=dict(fold_bins=[1.0])), "fold_bins"),
                     (dict(subhalos=dict(enabled=False), quotas=dict(subhalo=2)), "subhalos.enabled"),
                     (dict(subhalos=dict(patch_keep=50.0)), "patch_keep")):
        with pytest.raises(ValueError, match=key):
            resolve_config(bad)
    c = resolve_config({})
    assert c["quotas"] == dict(wide=22, intermediate=14, cusp=8, fold=15, subhalo=10) and c["n_systems"] is None
    assert c["knots"]["n_systems"] == 12 and c["sources"]["seed_kinds"]["subhalo"] == 0.35
    assert resolve_config(dict(n_systems=5))["quotas"] is None            # a number of systems: no quotas
    with pytest.raises(ValueError, match="n_systems"):
        resolve_config(dict(n_systems=5, quotas=dict(wide=2)))
    with pytest.raises(ValueError, match="n_systems"):
        resolve_config(dict(quotas=None))


# ----------------------------------------------------------------------------- satellites
def blob(n=40000, sigma=0.8, seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, (n, 2))


def test_patch_is_a_point_mass_far_away_and_its_derivatives_are_consistent():
    sig, kpa = 3.0e9, 5.0
    te2 = 1.3 ** 2
    offsets = blob()
    masses = np.full(len(offsets), te2 * np.pi * sig * kpa ** 2 / len(offsets))
    p = Patch((10.0, -4.0), offsets, masses, sig, kpa, half=20.0, pixels=500, keep=12.0)
    assert abs(p.theta_e2 / te2 - 1) < 1e-3 and p.mass_fraction > 0.999
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    for r in (6.0, 11.0, 25.0):                      # inside the splined grid and outside it
        pts = np.column_stack([10.0 + r * np.cos(ang), -4.0 + r * np.sin(ang)])
        a = p.alpha(pts)
        radial = (a[:, 0] * np.cos(ang) + a[:, 1] * np.sin(ang))
        assert np.allclose(radial, te2 / r, rtol=5e-3), r
    pts = np.column_stack([10.0 + np.array([0.3, 1.5, -2.0, 5.0, 14.0]), -4.0 + np.array([0.2, -1.0, 2.5, 3.0, -9.0])])
    a, axx, axy, ayx, ayy = p.fields(pts)
    h = 1e-4
    fx = (p.alpha(pts + [h, 0]) - p.alpha(pts - [h, 0])) / (2 * h)
    fy = (p.alpha(pts + [0, h]) - p.alpha(pts - [0, h])) / (2 * h)
    assert np.allclose(axx, fx[:, 0], atol=1e-5) and np.allclose(ayx, fx[:, 1], atol=1e-5)
    assert np.allclose(axy, fy[:, 0], atol=1e-5) and np.allclose(ayy, fy[:, 1], atol=1e-5)
    # the convergence at the centre of a Gaussian of sigma^2 = 0.8^2 + smoothing^2
    k0 = 0.5 * (p.fields(np.array([[10.0, -4.0]]))[1] + p.fields(np.array([[10.0, -4.0]]))[4])[0]
    assert abs(k0 / (te2 / (2 * (0.8 ** 2 + (2 * 40.0 / 500) ** 2))) - 1) < 0.03
    xs, ys = np.linspace(-5, 30, 36), np.linspace(-20, 10, 31)
    X, Y = np.meshgrid(xs, ys)
    mesh = p.on_mesh(xs, ys)
    ref = p.fields(np.column_stack([X.ravel(), Y.ravel()]))
    for got, want in zip(mesh, (ref[0][:, 0], ref[0][:, 1], ref[1], ref[2], ref[3], ref[4])):
        assert np.allclose(got.ravel(), want, atol=1e-12)


SAT_XY, SAT_TE = (13.0, 4.0), 1.2


@pytest.fixture(scope="module")
def satellite_lens(tmp_path_factory):
    """The analytic macro lens plus one satellite (a Gaussian blob of particles) at SAT_XY, and its Satellites."""
    root = tmp_path_factory.mktemp("sat")
    cosmo = cosmology.make({"H0": 70.0, "Om0": 0.3})
    sig, kpa = cosmology.sigma_critical_inf(cosmo, 0.3), cosmology.kpc_per_arcsec(cosmo, 0.3)
    offsets = blob(sigma=0.5)
    mass = SAT_TE ** 2 * np.pi * sig * kpa ** 2
    masses = np.full(len(offsets), mass / len(offsets))
    truth = Patch(SAT_XY, offsets, masses, sig, kpa)

    def lens(X, Y):
        v = MACRO(X, Y)
        a = truth.alpha(np.column_stack([np.ravel(X), np.ravel(Y)]))
        v["ax"], v["ay"] = v["ax"] + a[:, 0].reshape(np.shape(X)), v["ay"] + a[:, 1].reshape(np.shape(X))
        return v

    path = write_synthetic_lens(root / "lens.h5", lens=lens, n=1024)
    lm = LensMap.from_file(path, half=55.0)
    finder = ImageFinder(lm, 55.0, 2)
    table = dict(x=np.array([0.0, SAT_XY[0], -30.0]), y=np.array([0.0, SAT_XY[1], 20.0]), los=np.zeros(3),
                 mlens=np.array([1e14, mass, mass / 1e3]), soff=700, central=0)
    reader = lambda k: (offsets, masses) if k == 1 else (np.zeros((1, 2)), np.array([table["mlens"][k]]))
    return lm, finder, Satellites(lm, finder, table, reader, dict(min_mass=mass / 10), radius=60.0)


def test_lens_without_the_satellite_is_the_macro_lens(satellite_lens):
    lm, finder, sats = satellite_lens
    assert sats.tested.tolist() == [1] and sats.global_id(1) == 701
    without = finder_without(finder, sats.patch(1))
    pts = np.array([[12.0, 4.5], [14.5, 3.0], [-20.0, 9.0], [30.0, -25.0]])
    v = MACRO(pts[:, 0], pts[:, 1])
    assert np.allclose(without.lens.alpha(pts), np.column_stack([v["ax"], v["ay"]]), atol=2e-3)
    assert isinstance(without.lens, MinusLens) and np.allclose(without.ax, MACRO(*finder.mesh)["ax"], atol=5e-3)
    k, g1, g2, det = without.lens.hessian(pts, 0.7)
    assert np.allclose(k, 0.7 * 0.5 * (v["pxx"] + v["pyy"]), atol=3e-3)
    mask = sats.seed_mask(*finder.mesh)
    X, Y = finder.mesh
    assert mask.any() and np.all(np.hypot(X[mask] - SAT_XY[0], Y[mask] - SAT_XY[1]) < 4.0)


def test_counterfactual_ties_the_images_at_the_satellite(satellite_lens):
    lm, finder, sats = satellite_lens
    s = 0.75
    # a source with an image right next to the satellite, and one whose images are all far from it
    theta0 = np.array([[SAT_XY[0] + 0.9, SAT_XY[1] + 0.3]])
    near = (theta0 - s * lm.alpha(theta0))[0]
    theta1 = np.array([[-12.0, -9.0]])
    far = (theta1 - s * lm.alpha(theta1))[0]
    out = {}
    for name, beta in (("near", near), ("far", far)):
        theta = finder.find(beta, s)
        tab = image_table(lm, theta, beta, s)
        obs = observable(tab)
        out[name] = (tab, obs, sats.counterfactual(s, tab, obs))
    tab, obs, cf = out["near"]
    assert len(cf) == 1 and cf[0]["sub"] == 1 and cf[0]["id"] == 701
    assert abs(cf[0]["theta_e"] - SAT_TE * np.sqrt(s)) < 0.02 and cf[0]["radius"] == pytest.approx(4 * cf[0]["theta_e"])
    d = np.hypot(tab["theta"][:, 0] - SAT_XY[0], tab["theta"][:, 1] - SAT_XY[1])
    assert cf[0]["tagged"] and all(d[i] < cf[0]["radius"] and obs[i] for i in cf[0]["tagged"])
    assert all(sh > 0.5 for sh, dist in zip(cf[0]["shift"], cf[0]["dist"]) if dist < cf[0]["radius"])
    assert out["far"][2] == []
    assert Satellites(lm, finder, sats.table, sats.reader, dict(min_mass=1e30), radius=60.0).counterfactual(s, tab, obs) == []


def test_particle_reader_units_axes_and_offsets(tmp_path):
    h, a, box, kpa = 0.7, 0.8, 1000.0, 5.0
    centre = np.array([990.0, 500.0, 500.0])              # the halo sits near the box edge in x
    rng = np.random.default_rng(0)
    stars = rng.uniform(-3, 3, (9, 3))                     # ckpc/h about the satellite
    dm = rng.uniform(-3, 3, (6, 3))
    sat = np.array([20.0, -10.0, 4.0])                     # satellite, ckpc/h from the halo centre
    path = tmp_path / "snapshot.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["MassTable"] = np.array([0, 0.02, 0, 0, 0, 0.0])
        f["PartType1/Coordinates"] = np.concatenate([np.zeros((4, 3)), (centre + sat + dm) % box])
        f["PartType4/Coordinates"] = np.concatenate([np.zeros((2, 3)), (centre + sat + stars) % box, np.zeros((3, 3))])
        f["PartType4/Masses"] = np.concatenate([np.zeros(2), np.full(9, 0.001), np.zeros(3)])
    slen, offset = np.zeros((1, 6), int), np.zeros((1, 6), int)
    slen[0, 1], offset[0, 1], slen[0, 4], offset[0, 4] = 6, 4, 9, 2
    meta = dict(particle_types=[0, 1, 4, 5], halo_position_ckpc_h=centre.tolist(), box_ckpc_h=box, h=h, scale_factor=a,
                axis="zx")
    table = dict(slen=slen, offset=offset, x=np.array([sat[2] * a / h / kpa]), y=np.array([sat[0] * a / h / kpa]))
    offsets, masses = particle_reader(meta, table, kpa, path)(0)
    assert len(offsets) == 15 and np.allclose(masses[:6], 0.02e10 / h) and np.allclose(masses[6:], 0.001e10 / h)
    want = np.concatenate([dm, stars])[:, [2, 0]] * a / h / kpa          # zx: map x = sim z, map y = sim x
    assert np.allclose(offsets, want, atol=1e-9)
    with pytest.raises(FileNotFoundError, match="snapshot chunk"):
        particle_reader(meta, table, kpa, tmp_path / "missing.hdf5")


# ----------------------------------------------------------------------------- end to end, from the simulation files
@pytest.fixture(scope="module")
def tied_catalog(tmp_path_factory):
    """A lens file made of the macro lens plus the dark matter of every satellite of a synthetic Subfind chunk."""
    from test_members import KPA, SNAP, ZOOM, make_catalog, map_to_sim
    from tnglens import members
    root = tmp_path_factory.mktemp("tied")
    meta = make_catalog(root, "yz")
    snapshot = root / "output" / f"snapdir_{SNAP:03d}" / f"snapshot_{SNAP:03d}.{ZOOM}.hdf5"
    table = members.load_subhalos(meta, KPA)
    cosmo = cosmology.make({"H0": 70.0, "Om0": 0.3})
    sig = cosmology.sigma_critical_inf(cosmo, meta["lens_redshift"])
    particle_mass = 1.2 ** 2 * np.pi * sig * KPA ** 2 / 1000          # 1000 particles: theta_E = 1.2" at s = 1
    rng = np.random.default_rng(4)
    blocks = []
    for k in range(len(table["x"])):                                   # 1000 dark-matter particles per subhalo
        d = rng.normal(0.0, 0.5 * KPA, (1000, 2))
        blocks.append(np.array([map_to_sim(table["x"][k] * KPA + a, table["y"][k] * KPA + b, table["los"][k], "yz")
                                for a, b in d]))
    with h5py.File(snapshot, "r+") as f:
        f["PartType4/Masses"][:] = 1e-12                                # the stars weigh nothing here
        f["PartType1/Coordinates"] = np.concatenate(blocks)
        f.create_group("Header").attrs["MassTable"] = np.array([0, particle_mass * meta["h"] / 1e10, 0, 0, 0, 0.0])
    reader = particle_reader(meta, table, KPA, snapshot)
    patches = [Patch((table["x"][k], table["y"][k]), *reader(k), sig, KPA) for k in range(1, len(table["x"]))]

    def lens(X, Y):
        v = MACRO(X, Y)
        pts = np.column_stack([np.ravel(X), np.ravel(Y)])
        for p in patches:
            a = p.alpha(pts)
            v["ax"], v["ay"] = v["ax"] + a[:, 0].reshape(np.shape(X)), v["ay"] + a[:, 1].reshape(np.shape(X))
        return v

    attrs = {k: v for k, v in meta.items() if k not in ("H0", "Om0", "lens_redshift")}
    path = write_synthetic_lens(root / "lens.h5", lens=lens, n=1024, attrs=attrs)
    cfg = dict(seed=3, max_candidates=300, quotas=dict(wide=2, intermediate=1, subhalo=3),
               sources=dict(seed_kinds=dict(negative=0.3, subhalo=0.7)),
               subhalos=dict(min_mass=1e10, seed_min_mass=1e10))
    return path, cfg, build_catalog(path, cfg, root / "out"), root


def test_systems_tied_to_satellites_from_the_simulation_files(tied_catalog):
    path, cfg, paths, root = tied_catalog
    sel = yaml.safe_load(open(paths["selection_yaml"]))
    src = read_csv(paths["sources_truth_csv"])
    tied = [r for r in src if r["cls"] == "subhalo"]
    assert 1 <= len(tied) <= 3 and sel["n_systems_tied_to_satellites"] == len(tied) == sel["systems_by_class"]["subhalo"]
    assert [r["system"] for r in tied] == [f"S{i + 1}" for i in range(len(tied))]
    assert all(r["subhalos"] and r["geometry"] != "subhalo" for r in tied)
    assert all(not r["subhalos"] and r["system"].isdigit() for r in src if r["cls"] != "subhalo")
    assert sum(r["subhalos"].split(";")[0] == sid for r in tied for sid in ("301",)) <= 2      # max_per_satellite
    assert any(r["seed_kind"] == "subhalo" for r in src)
    rows = read_csv(paths["images_truth_csv"])
    satellites = {"301": (20.0, -10.0), "302": (-35.0, 25.0), "304": (70.0, 70.0)}
    for r in rows:
        for sid in filter(None, r["subhalo"].split(";")):
            assert r["system"].startswith("S") and r["observable"] == "True"
            x0, y0 = satellites[sid]
            assert np.hypot(float(r["x_true"]) - x0, float(r["y_true"]) - y0) < 6.0
    assert any(r["subhalo"] for r in rows)
    obs = yaml.safe_load(open(paths["images_yaml"]))["systems"]
    for name, system in obs.items():
        noted = [im for im in system["images"].values() if "subhalo" in str(im.get("notes", ""))]
        assert bool(noted) == name.startswith("S") == ("subhalo 30" in system["notes"])
    assert "Systems S<n> have images tied to a satellite" in open(paths["images_yaml"]).readline() + open(paths["images_yaml"]).read()
    assert "is tied to it" in open(paths["readme"]).read()


def test_tied_catalog_does_not_depend_on_workers(tied_catalog, tmp_path):
    path, cfg, paths, root = tied_catalog
    build_catalog(path, dict(cfg, workers=2, batch=9), tmp_path / "b")
    for sub in ("truth", "observed"):
        cmp = filecmp.dircmp(root / "out" / sub, tmp_path / "b" / sub)
        assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
