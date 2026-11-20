"""Mass-conserving deposit of kernels that are small compared with the pixel size.

sphviewer2 handles kernels resolved by at least ``target_cells_per_h`` pixels.  Smaller ones
go through :func:`deposit_small`:

* ``h >= 1.5`` pixels: the same projected kernel shape, ``(1 - r^2/h^2)^(3/2)``, evaluated at
  pixel centres and normalised over the *full* stencil before clipping at the image edge, so the
  particle's mass is exact up to the part that falls outside the image;
* ``h < 1.5`` pixels: cloud-in-cell, which conserves mass and the first moments.

This is a vectorised NumPy port of ``deposit_small`` in the old ``cosmic_telescope`` pipeline
(numba there); the arithmetic is the same, only the summation order differs.
"""
from __future__ import annotations

import numpy as np

CIC_LIMIT = 1.5  # kernels with support below this many pixels use cloud-in-cell


def deposit_small(x, y, h, m, width: float, image: np.ndarray, max_entries: int = 1 << 22) -> None:
    """Add particles to ``image`` (surface density, [y, x], square, centred on 0, side ``width``).

    ``x, y, h`` share the length unit of ``width``; ``m`` is mass.  ``image`` is float64 and is
    modified in place.  ``max_entries`` bounds the temporary stencil arrays.
    """
    n = image.shape[0]
    if image.shape != (n, n) or image.dtype != np.float64 or not image.flags.c_contiguous:
        raise ValueError("image must be a square, C-contiguous float64 array")
    if len(x) == 0:
        return
    dx = width / n
    u = (np.asarray(x, dtype=np.float64) + width / 2) / dx - 0.5  # pixel-index coordinates
    v = (np.asarray(y, dtype=np.float64) + width / 2) / dx - 0.5
    hp = np.asarray(h, dtype=np.float64) / dx
    density = np.asarray(m, dtype=np.float64) / (dx * dx)
    flat = image.reshape(-1)

    cic = hp < CIC_LIMIT
    if cic.any():
        _cic(u[cic], v[cic], density[cic], n, flat)
    rest = ~cic
    if rest.any():
        u, v, hp, density = u[rest], v[rest], hp[rest], density[rest]
        half = np.ceil(hp).astype(np.int64)
        for k in np.unique(half):
            sel = half == k
            _stencil(u[sel], v[sel], hp[sel], density[sel], int(k), n, flat, max_entries)


def _cic(u, v, density, n, flat):
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - iu, v - iv
    for ox, wx in ((0, 1 - fu), (1, fu)):
        for oy, wy in ((0, 1 - fv), (1, fv)):
            j, i = iu + ox, iv + oy
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            np.add.at(flat, i[ok] * n + j[ok], (density * wx * wy)[ok])


def _stencil(u, v, hp, density, k, n, flat, max_entries):
    """Kernels with ceil(h) == k pixels: a (2k+2)^2 box around floor(u), floor(v) covers them."""
    offsets = np.arange(-k, k + 2)
    size = len(offsets)
    step = max(1, max_entries // (size * size))
    for begin in range(0, len(u), step):
        uu, vv = u[begin:begin + step, None, None], v[begin:begin + step, None, None]
        hh = hp[begin:begin + step, None, None]
        ix = np.floor(uu).astype(np.int64) + offsets[None, None, :]
        iy = np.floor(vv).astype(np.int64) + offsets[None, :, None]
        q = 1 - ((ix - uu) ** 2 + (iy - vv) ** 2) / (hh * hh)
        np.maximum(q, 0, out=q)
        w = q * np.sqrt(q)
        total = w.sum(axis=(1, 2))
        if np.any(total <= 0):
            raise ValueError("empty discrete kernel stencil")
        w *= (density[begin:begin + step] / total)[:, None, None]
        ix, iy = np.broadcast_arrays(ix, iy)
        ok = (w > 0) & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        np.add.at(flat, iy[ok] * n + ix[ok], w[ok])
