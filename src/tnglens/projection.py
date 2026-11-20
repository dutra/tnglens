"""Projecting particles onto the core and environment surface-density grids.

Two square grids centred on the halo, in physical kpc on the lens plane:

* ``fine`` (the core): ``core_width_arcsec`` wide at ``core_resolution`` pixels;
* ``environment``: ``max(environment_width_r200 * R200c, 1.5 * core width)`` wide at
  ``environment_resolution`` pixels.

A particle is used if its centre lies within ``depth_half_r200 * R200c`` of the halo along the
line of sight (its projected kernel is not clipped at the slab faces) and its kernel overlaps
the grid.  Its support is ``h = max(neighbour distance, support_floor_pkpc)`` (gas may use the
cell size instead).  Kernels with ``h >= target_cells_per_h`` pixels are rendered by
sphviewer2; smaller ones by :func:`tnglens.deposition.deposit_small`.

The mass ledger records, per type and grid, the selected mass, the mass of kernels entirely
inside the grid, and the deposited mass; the deposit must lie between the two (1% tolerance),
otherwise :func:`project` raises :class:`MassLedgerError` and no projection is written.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numpy as np
import sphviewer2

from .deposition import CIC_LIMIT, deposit_small

log = logging.getLogger("tnglens")

AXES = {"xy": (0, 1, 2), "yz": (1, 2, 0), "zx": (2, 0, 1)}  # map x, map y, line of sight
MASS_TOLERANCE = 0.01


class MassLedgerError(RuntimeError):
    """The deposited mass of some type and grid is outside [fully contained, selected] mass."""


@dataclass(frozen=True)
class Grid:
    name: str
    width: float  # pkpc
    n: int

    @property
    def dx(self) -> float:
        return self.width / self.n

    @property
    def centers(self) -> np.ndarray:
        return pixel_centers(self.width, self.n)


def pixel_centers(width: float, n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) * width / n - width / 2


def make_grids(settings: dict, R200c_pkpc: float, kpc_per_arcsec: float) -> tuple[Grid, Grid]:
    fine = settings["core_width_arcsec"] * kpc_per_arcsec
    environment = max(settings["environment_width_r200"] * R200c_pkpc, 1.5 * fine)
    return (Grid("fine", fine, settings["core_resolution"]),
            Grid("environment", environment, settings["environment_resolution"]))


def _empty_ledger():
    return dict(particles=0, selected_mass=0.0, fully_contained_mass=0.0, deposited_mass=0.0,
                sphviewer_particles=0, stencil_particles=0, cic_particles=0)


def project(species, axis: str, grids, settings: dict, R200c_pkpc: float, box_pkpc: float,
            threads: int = 8, chunk: int = 2_000_000):
    """Surface density [Msun/pkpc^2] on each grid, and the mass ledger.

    ``species`` is a list of ``(part_type, particle_file, smoothing_file)``.  Returns
    ``({grid.name: image[y, x] float64}, ledger)`` where ledger is a list with one entry per type.
    """
    ix, iy, iz = AXES[axis]
    depth = settings["depth_half_r200"] * R200c_pkpc
    floor = settings["support_floor_pkpc"]
    quality = settings["target_cells_per_h"]
    totals = {g.name: np.zeros((g.n, g.n)) for g in grids}
    ledger = []
    failures = []
    for pt, particle_file, smoothing_file in species:
        projectors = {g.name: sphviewer2.Projector(sphviewer2.Camera(
            Lbox=box_pkpc, extent=g.width, xc=0.0, yc=0.0, zc=0.0, periodic=False,
            target_cells_per_h=quality, r_max=int(np.log2(g.n)))) for g in grids}
        small = {g.name: np.zeros((g.n, g.n)) for g in grids}
        entry = {"part_type": pt, "fof_mass": 0.0, "slab_mass": 0.0, "grids": {g.name: _empty_ledger() for g in grids}}
        use_cells = pt == 0 and settings["gas_support"] == "cells"
        with h5py.File(particle_file, "r") as fp, h5py.File(smoothing_file, "r") as fs:
            n = len(fp["masses"])
            support = fp["cell_support"] if use_cells else fs["neighbor_distance"]
            for begin in range(0, n, chunk):
                m = fp["masses"][begin:begin + chunk].astype(np.float64)
                entry["fof_mass"] += float(m.sum())
                pos = fp["coordinates"][begin:begin + chunk].astype(np.float64)
                h = np.maximum(support[begin:begin + chunk].astype(np.float64), floor)
                in_slab = np.abs(pos[:, iz]) <= depth
                entry["slab_mass"] += float(m[in_slab].sum())
                for g in grids:
                    _deposit(pos, m, h, in_slab, (ix, iy, iz), g, quality, projectors[g.name], small[g.name],
                             entry["grids"][g.name], threads)
        for g in grids:
            image = projectors[g.name].collapse() + small[g.name]
            totals[g.name] += image
            row = entry["grids"][g.name]
            row["deposited_mass"] = float(image.sum(dtype=np.float64) * g.dx ** 2)
            row["mass_check_pass"] = bool(
                row["fully_contained_mass"] * (1 - MASS_TOLERANCE) <= row["deposited_mass"]
                <= row["selected_mass"] * (1 + MASS_TOLERANCE) + 1e-10)
            if not row["mass_check_pass"]:
                failures.append(f"type {pt} {g.name} grid: deposited mass {row['deposited_mass']:.6g} Msun outside "
                                f"[{row['fully_contained_mass']:.6g}, {row['selected_mass']:.6g}] "
                                f"(fully contained, selected; tolerance {MASS_TOLERANCE:g})")
        del projectors, small
        ledger.append(entry)
        log.info("projected type %d along %s", pt, axis)
    if failures:
        raise MassLedgerError("mass ledger check failed along " + axis + ": " + "; ".join(failures))
    return totals, ledger


def _deposit(pos, m, h, in_slab, axes, grid: Grid, quality, projector, small_image, row, threads):
    ix, iy, iz = axes
    half, dx = grid.width / 2, grid.dx
    reach = np.maximum(h, dx)
    sel = in_slab & (np.abs(pos[:, ix]) < half + reach) & (np.abs(pos[:, iy]) < half + reach)
    if not sel.any():
        return
    x, y, z, hs, ms = pos[sel, ix], pos[sel, iy], pos[sel, iz], h[sel], m[sel]
    resolved = hs >= quality * dx
    if resolved.any():
        projector.project(sphviewer2.Particles(x[resolved], y[resolved], z[resolved], hs[resolved], ms[resolved]),
                          num_threads=threads)
    if (~resolved).any():
        deposit_small(x[~resolved], y[~resolved], hs[~resolved], ms[~resolved], grid.width, small_image)
    inside = (np.abs(x) + reach[sel] < half) & (np.abs(y) + reach[sel] < half)
    row["particles"] += int(sel.sum())
    row["selected_mass"] += float(ms.sum())
    row["fully_contained_mass"] += float(ms[inside].sum())
    row["sphviewer_particles"] += int(resolved.sum())
    row["stencil_particles"] += int(np.count_nonzero(~resolved & (hs >= CIC_LIMIT * dx)))
    row["cic_particles"] += int(np.count_nonzero(hs < CIC_LIMIT * dx))
