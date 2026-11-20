"""Cluster members and subhalos in the field, from the Subfind group catalog of the zoom.

TNG-Cluster stores each zoom in its own file chunk: the group catalog ``groups_SSS/fof_subhalo_tab_SSS.Z.hdf5``
and the snapshot ``snapdir_SSS/snapshot_SSS.Z.hdf5`` with Z the zoom index, both ordered by FoF group and,
within a group, by subhalo.  Global IDs are local indices plus the chunk offsets, read from
``postprocessing/offsets/offsets_SSS.hdf5`` next to the output directory or, if that is missing, summed from
the headers of the earlier chunks.

Positions are put in the frame of the lens map: relative to the halo centre ``halo_position_ckpc_h`` with the
periodic wrap, in physical kpc, then map x and y are the simulation axes named by ``axis`` ("yz": map x = sim y,
map y = sim z, line of sight = sim x), divided by ``kpc_per_arcsec``.

Magnitudes: rest-frame ``SubhaloStellarPhotometrics`` band (default 7, SDSS z) + distance modulus
- 2.5 log10(1 + z_lens): the observed AB magnitude at (1 + z_lens) times the rest wavelength, without dust
(SDSS z at z_lens ~ 0.38 lands near F125W).

Stellar shapes are mass-weighted second moments of the star particles within twice the stellar half-mass radius,
[a, b, theta] in arcsec and degrees counter-clockwise from +x; they need the snapshot chunk (or a particle cache)
and are left empty when it is not available.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .lensmap import AXES, require

DEFAULTS = dict(
    enabled="auto",            # true, false, or auto (only if the group catalog is found)
    mag_limit=24.0,            # members brighter than this
    radius=120.0,              # projected radius about the halo centre [arcsec]
    los_r200c=2.0,             # |line-of-sight offset| < this * R200c
    band_index=7,              # column of SubhaloStellarPhotometrics (U B V K g r i z)
    band_name="F125W",         # name the magnitude is given under in members.yaml
    shapes="auto",             # "auto" (snapshot chunk if present), "snapshot", a directory with type4.h5, or none
    shape_min_particles=50,
    group_file=None,           # explicit paths, else derived from group_catalog_base
    offsets_file=None,
    snapshot_file=None,
)

COLUMNS = ("id", "index", "x", "y", "los_pkpc", "mag", "role", "fof_group", "mtot", "mstar", "m_in_2rhalf",
           "veldisp", "vmax", "rvmax_pkpc", "rhalf_star_pkpc", "rhalf_pkpc", "mstar_in_2rhalf", "mlens",
           "shape_a", "shape_b", "shape_theta", "n_star_shape")


def catalog_paths(meta, cfg):
    base = Path(meta.get("group_catalog_base") or "")
    snap, zoom = meta.get("snapshot"), meta.get("zoom_index")
    groups = cfg.get("group_file") or base / f"groups_{snap:03d}" / f"fof_subhalo_tab_{snap:03d}.{zoom}.hdf5"
    offsets = cfg.get("offsets_file") or base.parent / "postprocessing" / "offsets" / f"offsets_{snap:03d}.hdf5"
    snapshot = cfg.get("snapshot_file") or base / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.{zoom}.hdf5"
    return Path(groups), Path(offsets), Path(snapshot)


def group_file(meta, cfg) -> Path:
    """The zoom's group-catalog file, or a clear error naming what is missing (attribute, override or path)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    if not cfg.get("group_file"):
        if not meta.get("group_catalog_base"):
            raise KeyError("the member table needs the Subfind group catalog, but the lens file has no "
                           "group_catalog_base attribute; pass it as an override (catalog.overrides."
                           "group_catalog_base), or set catalog.members.group_file, or members.enabled: auto/false")
        require(meta, ("snapshot", "zoom_index"), "the group-catalog path")
    groups = Path(cfg["group_file"]) if cfg.get("group_file") else catalog_paths(meta, cfg)[0]
    groups = groups.expanduser().absolute()
    if not groups.exists():
        how = ("catalog.members.group_file" if cfg.get("group_file") else
               f"group_catalog_base={meta.get('group_catalog_base')!r}, snapshot {meta.get('snapshot')}, "
               f"zoom_index {meta.get('zoom_index')}")
        raise FileNotFoundError(f"the member table needs the group catalog {groups}, which does not exist "
                                f"(from {how}); fix the override or the path, or set members.enabled: auto/false")
    return groups


def chunk_offsets(groups: Path, offsets: Path, zoom: int):
    """Global ID of the first group and first subhalo of chunk ``zoom``."""
    if offsets.exists():
        with h5py.File(offsets, "r") as f:
            return int(f["FileOffsets/Group"][zoom]), int(f["FileOffsets/Subhalo"][zoom]), str(offsets)
    goff = soff = 0
    for i in range(zoom):
        p = groups.with_name(groups.name.replace(f".{zoom}.hdf5", f".{i}.hdf5"))
        with h5py.File(p, "r") as f:
            goff += int(f["Header"].attrs["Ngroups_ThisFile"])
            soff += int(f["Header"].attrs["Nsubgroups_ThisFile"])
    return goff, soff, "chunk headers"


def load_subhalos(meta: dict, kpc_per_arcsec: float, cfg: dict | None = None) -> dict:
    """Every subhalo of the zoom's group-catalog chunk, in the map frame; arrays indexed by local subhalo index."""
    cfg = {**DEFAULTS, **(cfg or {})}
    require(meta, ("snapshot", "zoom_index", "axis", "halo_position_ckpc_h", "box_ckpc_h", "h", "scale_factor"),
            "the member table")
    groups = group_file(meta, cfg)
    _, offsets, _ = catalog_paths(meta, cfg)
    h, a, box = meta["h"], meta["scale_factor"], meta["box_ckpc_h"]
    goff, soff, off_src = chunk_offsets(groups, offsets, meta["zoom_index"])
    with h5py.File(groups, "r") as f:
        Gr, S = f["Group"], f["Subhalo"]
        glen = Gr["GroupLenType"][:].astype(np.int64)
        gfirst = Gr["GroupFirstSub"][:].astype(np.int64)
        slen = S["SubhaloLenType"][:].astype(np.int64)
        grnr = S["SubhaloGrNr"][:].astype(np.int64) - goff
        pos = S["SubhaloPos"][:].astype(np.float64) - np.asarray(meta["halo_position_ckpc_h"])
        mtype = S["SubhaloMassType"][:].astype(np.float64) * 1e10 / h
        cols = dict(mtot=S["SubhaloMass"][:].astype(np.float64) * 1e10 / h, mstar=mtype[:, 4],
                    phot=S["SubhaloStellarPhotometrics"][:, int(cfg["band_index"])].astype(np.float64),
                    veldisp=S["SubhaloVelDisp"][:].astype(np.float64), vmax=S["SubhaloVmax"][:].astype(np.float64),
                    rvmax=S["SubhaloVmaxRad"][:].astype(np.float64) * a / h,
                    rhalf_star=S["SubhaloHalfmassRadType"][:, 4].astype(np.float64) * a / h,
                    rhalf=S["SubhaloHalfmassRad"][:].astype(np.float64) * a / h,
                    m_in_2rhalf=S["SubhaloMassInRad"][:].astype(np.float64) * 1e10 / h,
                    mstar_in_2rhalf=S["SubhaloMassInRadType"][:, 4].astype(np.float64) * 1e10 / h)
    nsub = len(slen)
    ngroups = len(glen)
    halo_local = int(meta["halo_id"]) - goff if meta.get("halo_id") is not None else 0
    if not 0 <= halo_local < ngroups:
        raise ValueError(f"halo_id {meta.get('halo_id')} is not in chunk {meta['zoom_index']} "
                         f"(groups {goff} .. {goff + ngroups - 1})")
    gfirst_local = np.where(gfirst >= 0, gfirst - soff, -1)
    central = int(gfirst_local[halo_local])
    pos = ((pos + box / 2) % box - box / 2) * a / h          # physical kpc about the halo centre
    ix, iy, il = AXES[meta["axis"]]
    ptypes = meta.get("particle_types")
    mlens = mtype[:, list(ptypes)].sum(axis=1) if ptypes is not None else np.full(nsub, np.nan)
    # particle offsets of each subhalo within the chunk, per type
    group_off = np.vstack([np.zeros(6, np.int64), np.cumsum(glen, axis=0)[:-1]])
    cum = np.vstack([np.zeros(6, np.int64), np.cumsum(slen, axis=0)[:-1]])
    offset = group_off[grnr] + cum - cum[gfirst_local[grnr]]
    return dict(x=pos[:, ix] / kpc_per_arcsec, y=pos[:, iy] / kpc_per_arcsec, los=pos[:, il], pos=pos,
                offset=offset, slen=slen, soff=soff, goff=goff, grnr=grnr + goff, central=central,
                mlens=mlens, offsets_source=off_src, group_file=str(groups), **cols)


def _star_reader(meta, cfg):
    """A function k -> (map-frame offsets [arcsec] of subhalo k's stars from its centre, masses), or None."""
    mode = cfg.get("shapes", "auto")
    if mode in (None, False, "none", "off"):
        return None, "disabled"
    _, _, snapshot = catalog_paths(meta, cfg)
    h, a, box = meta["h"], meta["scale_factor"], meta["box_ckpc_h"]
    centre = np.asarray(meta["halo_position_ckpc_h"])
    ix, iy, _ = AXES[meta["axis"]]
    if mode in ("auto", "snapshot"):
        if not snapshot.exists():
            if mode == "snapshot":
                raise FileNotFoundError(f"members.shapes is snapshot, but the snapshot chunk {snapshot.absolute()} "
                                        "does not exist (set members.snapshot_file, or shapes: auto/none)")
            return None, f"skipped: {snapshot} not found"

        def read(o, n):
            with h5py.File(snapshot, "r") as f:
                c = f["PartType4/Coordinates"][o:o + n].astype(np.float64) - centre
                m = f["PartType4/Masses"][o:o + n].astype(np.float64)
            return ((c + box / 2) % box - box / 2) * a / h, m
        return read, str(snapshot)
    cache = Path(mode) / "type4.h5"         # coordinates already in pkpc about the halo centre
    if not cache.exists():
        return None, f"skipped: {cache} not found"

    def read(o, n):
        with h5py.File(cache, "r") as f:
            return f["coordinates"][o:o + n].astype(np.float64), f["masses"][o:o + n].astype(np.float64)
    return read, str(cache)


def stellar_shape(d, m, r_max, min_particles=50):
    """[a, b, theta] of the mass-weighted second moments of offsets ``d`` (N, 2) within ``r_max``."""
    keep = np.hypot(d[:, 0], d[:, 1]) < r_max
    if keep.sum() <= min_particles:
        return None, int(keep.sum())
    cov = np.cov(d[keep].T, aweights=m[keep])
    val, vec = np.linalg.eigh(cov)
    return [float(np.sqrt(val[1])), float(np.sqrt(val[0])),
            float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180)], int(keep.sum())


def member_table(meta: dict, cosmo, kpc_per_arcsec: float, cfg: dict | None = None):
    """Members in the field, brightest first: a list of dicts with ``COLUMNS`` (and ``shape``), and an info dict."""
    cfg = {**DEFAULTS, **(cfg or {})}
    require(meta, ("lens_redshift", "R200c_pkpc"), "the member table")
    S = load_subhalos(meta, kpc_per_arcsec, cfg)
    zl = meta["lens_redshift"]
    dm = float(cosmo.distmod(zl).value) - 2.5 * np.log10(1 + zl)
    mag = S["phot"] + dm
    infield = (np.hypot(S["x"], S["y"]) < cfg["radius"]) & (np.abs(S["los"]) < cfg["los_r200c"] * meta["R200c_pkpc"])
    sel = np.flatnonzero(infield & (mag < cfg["mag_limit"]) & (S["mstar"] > 0))
    sel = sel[np.lexsort((sel, mag[sel]))]
    reader, shape_src = _star_reader(meta, cfg)
    ix, iy, _ = AXES[meta["axis"]]
    rows = []
    for k in sel:
        shape, nstar = None, 0
        n, o = int(S["slen"][k, 4]), int(S["offset"][k, 4])
        if reader is not None and n > 0:
            p, m = reader(o, n)
            d = np.column_stack([p[:, ix], p[:, iy]]) / kpc_per_arcsec - np.array([S["x"][k], S["y"][k]])
            shape, nstar = stellar_shape(d, m, 2 * S["rhalf_star"][k] / kpc_per_arcsec, cfg["shape_min_particles"])
        rows.append(dict(
            id=str(int(k + S["soff"])), index=int(k), x=float(S["x"][k]), y=float(S["y"][k]),
            los_pkpc=float(S["los"][k]), mag=float(mag[k]), role="bcg" if k == S["central"] else "member",
            fof_group=int(S["grnr"][k]), mtot=float(S["mtot"][k]), mstar=float(S["mstar"][k]),
            m_in_2rhalf=float(S["m_in_2rhalf"][k]), veldisp=float(S["veldisp"][k]), vmax=float(S["vmax"][k]),
            rvmax_pkpc=float(S["rvmax"][k]), rhalf_star_pkpc=float(S["rhalf_star"][k]),
            rhalf_pkpc=float(S["rhalf"][k]), mstar_in_2rhalf=float(S["mstar_in_2rhalf"][k]),
            mlens=float(S["mlens"][k]),
            shape_a=shape[0] if shape else np.nan, shape_b=shape[1] if shape else np.nan,
            shape_theta=shape[2] if shape else np.nan, n_star_shape=nstar, shape=shape))
    central = S["central"]
    info = dict(n_members=len(rows), central_id=int(central + S["soff"]), central_index=int(central),
                bcg_xy=[float(S["x"][central]), float(S["y"][central])], distance_modulus=dm,
                shapes=shape_src, offsets=S["offsets_source"], group_file=S["group_file"],
                n_subhalos_chunk=int(len(S["x"])), bcg_in_table=any(r["role"] == "bcg" for r in rows))
    return rows, info


def available(meta: dict, cfg: dict | None = None) -> bool:
    """Whether the group catalog of this lens file can be found."""
    cfg = {**DEFAULTS, **(cfg or {})}
    try:
        groups, _, _ = catalog_paths(meta, cfg)
    except (TypeError, ValueError):
        return False
    return bool(meta.get("group_catalog_base") or cfg.get("group_file")) and groups.exists()
