"""Reading particles of one zoom and computing their smoothing lengths.

Stage ``particles`` writes one file per particle type with, for every FoF particle of the
original zoom (the whole ``snapshot_NNN.<zoom>.hdf5`` chunk; no outer fuzz):

* ``coordinates`` [pkpc], float32, relative to the halo centre (periodic minimum image);
* ``masses`` [Msun], float32;
* ``cell_support`` [pkpc] (gas only): ``2 (3 m / (4 pi rho))^(1/3)``, twice the radius of the
  sphere with the Voronoi cell's volume.

Stage ``smoothing`` writes ``neighbor_distance`` [pkpc], float32: the exact distance to the
k-th nearest other particle of the same type, in the full periodic box, over all FoF
particles of the zoom.  Black holes get 0 (the support floor applies).  The support floor is
applied later, at projection time, so a floor change never reruns the tree search.
"""
from __future__ import annotations

import logging

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .cache import atomic_h5, is_complete, signature
from .halos import Halo

log = logging.getLogger("tnglens")

READER_VERSION = 1
SMOOTHING_VERSION = 1
H5 = dict(compression="lzf", shuffle=True)


def relative_coordinates(coords, center, box, physical_factor):
    """Minimum-image offsets from ``center`` (both in ckpc/h), converted by ``physical_factor``."""
    out = np.asarray(coords, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    out -= box * np.rint(out / box)
    out *= physical_factor
    return out


def particle_signature(halo: Halo, pt: int) -> str:
    return signature({"reader": READER_VERSION, "file": halo.snapshot_file, "part_type": pt,
                      "center_ckpc_h": halo.position_ckpc_h, "n": halo.num_part[pt],
                      "h": halo.h, "a": halo.scale_factor, "box": halo.box_ckpc_h})


def smoothing_signature(halo: Halo, pt: int, neighbors: int) -> str:
    return signature({"smoothing": SMOOTHING_VERSION, "particles": particle_signature(halo, pt),
                      "neighbors": neighbors})


def read_species(halo: Halo, pt: int, dest, chunk: int = 2_000_000, force=False) -> bool:
    """Write the particle cache of type ``pt``; return False if a valid one already existed."""
    sig = particle_signature(halo, pt)
    if not force and is_complete(dest, sig):
        return False
    n = int(halo.num_part[pt])
    fixed_mass = halo.mass_table[pt]
    factor = halo.scale_factor / halo.h
    total_mass = 0.0
    with h5py.File(halo.snapshot_file, "r") as src, atomic_h5(dest, sig) as out:
        group = src[f"PartType{pt}"] if n else None
        rows = min(max(n, 1), 65536)
        coords = out.create_dataset("coordinates", (n, 3), dtype="f4", chunks=(rows, 3), **H5)
        masses = out.create_dataset("masses", (n,), dtype="f4", chunks=(rows,), **H5)
        cells = out.create_dataset("cell_support", (n,), dtype="f4", chunks=(rows,), **H5) if pt == 0 else None
        for begin in range(0, n, chunk):
            end = min(begin + chunk, n)
            pos = relative_coordinates(group["Coordinates"][begin:end], halo.position_ckpc_h,
                                       halo.box_ckpc_h, factor)
            raw = np.full(end - begin, fixed_mass) if fixed_mass else group["Masses"][begin:end].astype(np.float64)
            coords[begin:end] = pos
            masses[begin:end] = raw * 1e10 / halo.h
            total_mass += float(np.sum(raw, dtype=np.float64)) * 1e10 / halo.h
            if cells is not None:
                density = group["Density"][begin:end].astype(np.float64)
                cells[begin:end] = 2 * (3 * raw / density / (4 * np.pi)) ** (1 / 3) * factor
        out.attrs.update(part_type=pt, n_particles=n, total_mass_Msun=total_mass, source=halo.snapshot_file,
                         halo_id=halo.halo_id, units="coordinates pkpc relative to halo centre; masses Msun")
    log.info("%s: read type %d, %d particles, %.4g Msun", halo.key(), pt, n, total_mass)
    return True


def neighbor_distance(pos, neighbors: int, box: float, threads: int = 8, batch: int = 65536) -> np.ndarray:
    """Exact distance to the ``neighbors``-th nearest other point, periodic in a box of side ``box``.

    Only that one order is queried (``k=[k+1]``, the first neighbour being the point itself), so
    no N x k table is allocated.  Duplicate positions are counted as separate neighbours.
    """
    pos = np.array(pos, dtype=np.float64)  # copy: wrapped into [0, box)
    if len(pos) < 2:
        raise ValueError("at least two points are required")
    np.remainder(pos, box, out=pos)
    np.minimum(pos, np.nextafter(box, 0.0), out=pos)  # remainder can round to exactly box
    k = min(int(neighbors), len(pos) - 1)
    tree = cKDTree(pos, leafsize=32, compact_nodes=True, balanced_tree=True, boxsize=box)
    out = np.empty(len(pos), dtype=np.float64)
    for begin in range(0, len(pos), batch):
        distance, _ = tree.query(pos[begin:begin + batch], k=[k + 1], workers=threads)
        out[begin:begin + batch] = distance[:, 0]
    return out


def smooth_species(halo: Halo, pt: int, particle_file, dest, neighbors: int, threads: int, force=False) -> bool:
    """Write the neighbour-distance cache of type ``pt``; False if a valid one already existed."""
    sig = smoothing_signature(halo, pt, neighbors)
    if not force and is_complete(dest, sig):
        return False
    with h5py.File(particle_file, "r") as f:
        n = len(f["masses"])
        if pt == 5 or n < 2:
            distance = np.zeros(n)
            policy = "black holes: zero (support floor applies)" if pt == 5 else "fewer than two particles: zero"
        else:
            distance = neighbor_distance(f["coordinates"][:], neighbors, halo.physical_box_pkpc, threads)
            policy = f"distance to neighbour {min(neighbors, n - 1)} of the same type, periodic, all FoF particles"
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError(f"{halo.key()}: invalid neighbour distances for type {pt}")
    with atomic_h5(dest, sig) as out:
        out.create_dataset("neighbor_distance", data=distance.astype(np.float32), **H5)
        out.attrs.update(part_type=pt, n_particles=n, neighbors=neighbors, policy=policy)
        if n:
            out.attrs["quantiles_pkpc_0_10_50_90_100"] = np.quantile(distance, [0, 0.1, 0.5, 0.9, 1])
    log.info("%s: smoothing type %d done (%d particles)", halo.key(), pt, n)
    return True
