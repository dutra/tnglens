"""A miniature TNG-Cluster directory tree for tests (no simulation files needed)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

H = 0.7
SNAP = 50
REDSHIFT = 0.3
BOX = 20000.0       # ckpc/h
N_ZOOMS = 2
# Halo of zoom 0 sits close to a box corner so its particles straddle the periodic seam.
CENTERS = [np.array([150.0, BOX - 80.0, 9000.0]), np.array([12000.0, 8000.0, 4000.0])]
R200_CKPC_H = 300.0
M200_1E10_H = 2.0e4
DM_MASS = 1e9 * H / 1e10   # MassTable entry: 1e9 Msun
TRACER_MASS = 1e7 * H / 1e10  # MassTable[3]; tracers have no Coordinates or Masses (as in TNG)
N_TRACERS = 500
LOWRES_SIGMA_PKPC = 150.0


def _blob(rng, n, sigma_pkpc, center, a):
    """Gaussian blob in ckpc/h around ``center`` (periodic wrap into the box)."""
    sigma = sigma_pkpc * H / a
    pos = center + rng.normal(0, sigma, (n, 3))
    return np.mod(pos, BOX)


def make_simulation(root: Path, seed: int = 1, n_dm: int = 20000, n_gas: int = 8000, n_star: int = 3000,
                    n_lowres: int = 400, lowres_offset_pkpc=(2500.0, 2500.0, 2500.0)) -> Path:
    """Write output/groups_050, output/snapdir_050 and postprocessing/offsets; return the output dir.

    Besides gas (0), dark matter (1, fixed mass), stars (4) and black holes (5), every zoom has
    low-resolution dark matter (2, variable masses, MassTable[2] = 0) in a blob centred
    ``lowres_offset_pkpc`` from the halo (by default far outside every grid and slab), and
    tracers (3: ParentID and TracerID only, like TNG).
    """
    rng = np.random.default_rng(seed)
    rng_lowres = np.random.default_rng(seed + 1000)  # separate stream: the other types do not change
    a = 1 / (1 + REDSHIFT)
    base = root / "output"
    (base / f"groups_{SNAP:03d}").mkdir(parents=True)
    (base / f"snapdir_{SNAP:03d}").mkdir(parents=True)
    (root / "postprocessing" / "offsets").mkdir(parents=True)
    ngroups = [3, 2]
    counts = []
    for zoom in range(N_ZOOMS):
        center = CENTERS[zoom]
        parts = {
            0: _blob(rng, n_gas, 120.0, center, a),
            1: _blob(rng, n_dm, 150.0, center, a),
            4: _blob(rng, n_star, 30.0, center, a),
            5: np.mod(center + rng.normal(0, 1.0, (2, 3)), BOX),
        }
        offset = np.asarray(lowres_offset_pkpc, dtype=np.float64) * H / a
        parts[2] = _blob(rng_lowres, n_lowres, LOWRES_SIGMA_PKPC, center + offset, a)
        lowres_mass = rng_lowres.uniform(5e9, 5e10, n_lowres) * H / 1e10
        num = np.zeros(6, dtype=np.int64)
        for pt, pos in parts.items():
            num[pt] = len(pos)
        num[3] = N_TRACERS
        counts.append(num)
        with h5py.File(base / f"snapdir_{SNAP:03d}" / f"snapshot_{SNAP:03d}.{zoom}.hdf5", "w") as f:
            hdr = f.create_group("Header").attrs
            hdr.update(NumPart_ThisFile=num.astype(np.int32), MassTable=np.array([0, DM_MASS, 0, TRACER_MASS, 0, 0.0]),
                       HubbleParam=H, Time=a, Redshift=1 / a - 1, BoxSize=BOX, Omega0=0.3, OmegaLambda=0.7)
            for pt, pos in parts.items():
                g = f.create_group(f"PartType{pt}")
                g["Coordinates"] = pos.astype(np.float32)
                if pt == 2:
                    g["Masses"] = lowres_mass.astype(np.float32)
                elif pt != 1:
                    g["Masses"] = np.full(len(pos), {0: 2e8, 4: 5e8, 5: 1e9}[pt] * H / 1e10, dtype=np.float32)
                if pt == 0:
                    g["Density"] = np.full(len(pos), 1e-4, dtype=np.float32)
            tracers = f.create_group("PartType3")
            tracers["ParentID"] = np.arange(N_TRACERS, dtype=np.uint64)
            tracers["TracerID"] = np.arange(N_TRACERS, dtype=np.uint64) + 10**9
        with h5py.File(base / f"groups_{SNAP:03d}" / f"fof_subhalo_tab_{SNAP:03d}.{zoom}.hdf5", "w") as f:
            f.create_group("Header").attrs.update(Ngroups_ThisFile=ngroups[zoom], NumFiles=N_ZOOMS,
                                                  Redshift=1 / a - 1)
            g = f.create_group("Group")
            pos = np.vstack([center, center + 500.0, center - 700.0])[: ngroups[zoom]]
            g["GroupPos"] = np.mod(pos, BOX).astype(np.float32)
            g["Group_M_Crit200"] = np.array([M200_1E10_H, 10.0, 5.0][: ngroups[zoom]], dtype=np.float32)
            g["Group_R_Crit200"] = np.array([R200_CKPC_H, 50.0, 40.0][: ngroups[zoom]], dtype=np.float32)
            g["GroupContaminationFracByMass"] = np.zeros(ngroups[zoom], dtype=np.float32)
            g["GroupPrimaryZoomTarget"] = np.array([1, 0, 0][: ngroups[zoom]], dtype=np.int32)
    with h5py.File(root / "postprocessing" / "offsets" / f"offsets_{SNAP:03d}.hdf5", "w") as f:
        f["FileOffsets/Group"] = np.concatenate([[0], np.cumsum(ngroups)[:-1]]).astype(np.int64)
        f["OriginalZooms/HaloIDs"] = np.arange(N_ZOOMS, dtype=np.int32)
        f["OriginalZooms/GroupsTotalLengthByType"] = np.array(counts, dtype=np.int32)
    return base


def small_config(base: Path, tmp: Path, **overrides) -> dict:
    cfg = {
        "simulation": {"base": str(base)},
        "cache_root": str(tmp / "cache"),
        "output_root": str(tmp / "out"),
        "halos": [{"zoom": 0, "snapshot": SNAP, "axes": ["yz"]}],
        "source_redshifts": [1.0, 2.5],
        "projection": {"neighbors": 16, "support_floor_pkpc": 2.0, "core_width_arcsec": 120.0,
                       "core_resolution": 128, "environment_width_r200": 4.0, "environment_resolution": 128,
                       "depth_half_r200": 2.0, "target_cells_per_h": 4},
        "threads": 2,
        "workers": 1,
        "read_chunk": 5000,
    }
    cfg.update(overrides)
    return cfg
