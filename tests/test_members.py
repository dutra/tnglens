"""The member table from a small synthetic Subfind chunk in the TNG-Cluster layout."""
import h5py
import numpy as np
import pytest

from tnglens import cosmology
from tnglens.members import available, chunk_offsets, group_file, member_table

H, A, BOX, ZL = 0.7, 0.8, 680000.0, 0.3
CENTRE = np.array([100.0, 5000.0, 5000.0])          # ckpc/h, near the box edge in x
COSMO = cosmology.make({"H0": 70.0, "Om0": 0.3})
KPA = cosmology.kpc_per_arcsec(COSMO, ZL)
ZOOM, SNAP = 2, 99
GOFF, SOFF = 500, 300                                # global IDs of the chunk's first group and subhalo


def map_to_sim(mx, my, los, axis):
    """Physical kpc offsets in the map frame -> simulation ckpc/h positions (wrapped).

    The projection convention, written out here rather than taken from the package:
    xy: map x = sim x, map y = sim y, LOS = sim z; yz: map x = sim y, map y = sim z, LOS = sim x;
    zx: map x = sim z, map y = sim x, LOS = sim y.
    """
    d = {"xy": (mx, my, los), "yz": (los, mx, my), "zx": (my, los, mx)}[axis]
    return (CENTRE + np.array(d) * H / A) % BOX


def make_catalog(root, axis, with_offsets=True, with_snapshot=True):
    rng = np.random.default_rng(0)
    base = root / "output"
    (base / f"groups_{SNAP:03d}").mkdir(parents=True)
    # earlier chunks only need headers for the fallback offsets
    for i, (ng, ns) in enumerate([(200, 100), (300, 200)]):
        with h5py.File(base / f"groups_{SNAP:03d}" / f"fof_subhalo_tab_{SNAP:03d}.{i}.hdf5", "w") as f:
            f.create_group("Header").attrs.update(Ngroups_ThisFile=ng, Nsubgroups_ThisFile=ns)
    if with_offsets:
        (root / "postprocessing" / "offsets").mkdir(parents=True)
        with h5py.File(root / "postprocessing" / "offsets" / f"offsets_{SNAP:03d}.hdf5", "w") as f:
            f["FileOffsets/Group"] = np.array([0, 200, GOFF, 900])
            f["FileOffsets/Subhalo"] = np.array([0, 100, SOFF, 700])
    # subhalos: (map x arcsec, map y arcsec, los pkpc, group, n_star, M_z, mstar)
    subs = [(0.0, 0.0, 0.0, 0, 3000, -24.0, 1e12),        # central of the halo
            (20.0, -10.0, 300.0, 0, 5000, -21.0, 1e11),   # elongated satellite
            (-35.0, 25.0, -100.0, 0, 0, 0.0, 0.0),        # dark subhalo
            (-60.0, -5.0, 9000.0, 0, 100, -20.0, 1e10),   # too far along the line of sight
            (70.0, 70.0, 50.0, 1, 200, -19.0, 1e10),      # other FoF group, in the field
            (200.0, 0.0, 0.0, 1, 100, -22.0, 1e11)]       # outside the radius
    n = len(subs)
    slen = np.zeros((n, 6), np.int64)
    slen[:, 4] = [s[4] for s in subs]
    slen[:, 1] = 1000
    pos = np.array([map_to_sim(s[0] * KPA, s[1] * KPA, s[2], axis) for s in subs])
    grnr = np.array([s[3] for s in subs]) + GOFF
    fuzz = np.array([[0, 0, 0, 0, 7, 0], [0, 0, 0, 0, 3, 0]])
    glen = np.array([slen[:4].sum(0), slen[4:].sum(0)]) + fuzz
    phot = np.full((n, 8), 99.0)
    phot[:, 7] = [s[5] for s in subs]
    mtype = np.zeros((n, 6))
    mtype[:, 4] = [s[6] * H / 1e10 for s in subs]
    mtype[:, 1] = 5.0
    with h5py.File(base / f"groups_{SNAP:03d}" / f"fof_subhalo_tab_{SNAP:03d}.{ZOOM}.hdf5", "w") as f:
        f.create_group("Header").attrs.update(Ngroups_ThisFile=2, Nsubgroups_ThisFile=n)
        f["Group/GroupLenType"] = glen
        f["Group/GroupFirstSub"] = np.array([SOFF, SOFF + 4])
        S = f.create_group("Subhalo")
        S["SubhaloLenType"], S["SubhaloGrNr"], S["SubhaloPos"] = slen, grnr, pos
        S["SubhaloMass"] = mtype.sum(1)
        S["SubhaloMassType"] = mtype
        S["SubhaloStellarPhotometrics"] = phot
        S["SubhaloVelDisp"] = np.arange(n) * 10.0 + 100
        S["SubhaloVmax"] = np.arange(n) * 10.0 + 200
        S["SubhaloVmaxRad"] = np.full(n, 10.0)
        hm = np.zeros((n, 6))
        hm[:, 4] = 30.0                      # ckpc/h: 2 rhalf = 68.6 pkpc, wider than the satellite's stars
        S["SubhaloHalfmassRadType"], S["SubhaloHalfmassRad"] = hm, np.full(n, 50.0)
        S["SubhaloMassInRad"], S["SubhaloMassInRadType"] = mtype.sum(1) / 2, mtype / 2
    if with_snapshot:
        # star particles in chunk order: group 0 subhalos, group 0 fuzz, group 1 subhalos, group 1 fuzz
        blocks = []
        for k, s in enumerate(subs):
            m = s[4]
            if k == 1:     # elongated: sigma 2 and 1 pkpc, major axis at 30 deg from map +x
                u, v = rng.normal(0, 2.0, m), rng.normal(0, 1.0, m)
                c, sn = np.cos(np.radians(30)), np.sin(np.radians(30))
                dx, dy = c * u - sn * v, sn * u + c * v
            else:
                dx, dy = rng.normal(0, 1.0, m), rng.normal(0, 1.0, m)
            pts = np.array([map_to_sim(s[0] * KPA + a, s[1] * KPA + b, s[2], axis) for a, b in zip(dx, dy)])
            blocks.append(pts.reshape(-1, 3))
            if k == 3:
                blocks.append(np.tile(CENTRE, (7, 1)))
        blocks.append(np.tile(CENTRE, (3, 1)))
        coords = np.concatenate(blocks)
        (base / f"snapdir_{SNAP:03d}").mkdir()
        with h5py.File(base / f"snapdir_{SNAP:03d}" / f"snapshot_{SNAP:03d}.{ZOOM}.hdf5", "w") as f:
            f["PartType4/Coordinates"] = coords.astype(np.float32)
            f["PartType4/Masses"] = np.ones(len(coords), np.float32)
    return dict(lens_redshift=ZL, snapshot=SNAP, zoom_index=ZOOM, halo_id=GOFF, axis=axis,
                halo_position_ckpc_h=CENTRE.tolist(), box_ckpc_h=BOX, h=H, scale_factor=A,
                group_catalog_base=str(base), R200c_pkpc=2000.0, particle_types=[0, 1, 4, 5], H0=70.0, Om0=0.3)


@pytest.mark.parametrize("axis", ["yz", "zx", "xy"])
def test_member_table(tmp_path, axis):
    meta = make_catalog(tmp_path, axis)
    assert available(meta)
    rows, info = member_table(meta, COSMO, KPA)
    ids = [r["id"] for r in rows]
    assert ids == [str(SOFF), str(SOFF + 1), str(SOFF + 4)]       # brightest first; cuts applied
    bcg, sat, other = rows
    assert bcg["role"] == "bcg" and sat["role"] == "member" and info["central_id"] == SOFF
    assert info["bcg_xy"] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert (sat["x"], sat["y"]) == pytest.approx((20.0, -10.0), abs=1e-4)
    assert sat["los_pkpc"] == pytest.approx(300.0, abs=1e-3)
    assert other["fof_group"] == GOFF + 1
    dm = COSMO.distmod(ZL).value - 2.5 * np.log10(1 + ZL)
    assert sat["mag"] == pytest.approx(-21.0 + dm)
    assert sat["mstar"] == pytest.approx(1e11)
    assert sat["mlens"] == pytest.approx(1e11 + 5e10 / H)
    a, b, th = sat["shape"]
    assert a == pytest.approx(2.0 / KPA, rel=0.05) and b == pytest.approx(1.0 / KPA, rel=0.05)
    assert th == pytest.approx(30.0, abs=2.0)
    assert bcg["shape"] is not None and bcg["n_star_shape"] > 2000
    assert other["shape"] is not None                              # 200 stars of the second group


def test_shapes_skipped_without_particles_and_offsets_fallback(tmp_path):
    meta = make_catalog(tmp_path, "yz", with_offsets=False, with_snapshot=False)
    groups = tmp_path / "output" / f"groups_{SNAP:03d}" / f"fof_subhalo_tab_{SNAP:03d}.{ZOOM}.hdf5"
    assert chunk_offsets(groups, tmp_path / "nope.hdf5", ZOOM)[:2] == (GOFF, SOFF)
    rows, info = member_table(meta, COSMO, KPA)
    assert len(rows) == 3 and all(r["shape"] is None for r in rows)
    assert info["shapes"].startswith("skipped") and info["offsets"] == "chunk headers"
    rows, info = member_table(meta, COSMO, KPA, dict(shapes="none", mag_limit=-20.0 + 41.0))
    assert info["shapes"] == "disabled"


def test_unavailable(tmp_path):
    assert not available(dict(group_catalog_base="", snapshot=1, zoom_index=0))
    assert not available(dict(group_catalog_base=str(tmp_path), snapshot=1, zoom_index=0))


def test_missing_group_catalog_is_a_clear_error(tmp_path):
    meta = make_catalog(tmp_path, "yz")
    assert group_file(meta, {}).is_absolute()
    with pytest.raises(KeyError, match="group_catalog_base"):
        member_table(dict(meta, group_catalog_base=""), COSMO, KPA)
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "elsewhere")):
        member_table(dict(meta, group_catalog_base=str(tmp_path / "elsewhere")), COSMO, KPA)
    with pytest.raises(FileNotFoundError, match="members.group_file"):
        member_table(meta, COSMO, KPA, dict(group_file=str(tmp_path / "x.hdf5")))
    with pytest.raises(FileNotFoundError, match="snapshot chunk"):
        member_table(meta, COSMO, KPA, dict(shapes="snapshot", snapshot_file=str(tmp_path / "s.hdf5")))
