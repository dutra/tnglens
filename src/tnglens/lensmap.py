"""Read a lens file and evaluate its deflection, potential and their derivatives anywhere in the field.

The lens file (one per halo, snapshot and projection) holds maps indexed [y, x] on a uniform grid of pixel
centres ``x``, ``y`` [arcsec] about the halo centre, normalised to D_ds/D_s = 1.  Images obey the mapping of
the deflection map, beta = theta - s alpha(theta) with s = D_ds/D_s, so everything that describes an image
(position, magnification, parity, kappa, shear) comes from one interpolant of ``alpha_x`` and ``alpha_y``: a
cubic B-spline through the pixel values.  Its Jacobian is evaluated analytically from the same spline
coefficients, so the magnification is exactly the inverse determinant of the mapping the images solve.  The
``potential`` map gets its own cubic spline and is used only for time delays.

Only a square of half-width ``half + margin`` about the centre is splined, which keeps memory to a few
hundred MB for a 4096^2 map; the B-spline prefilter is local to ~20 pixels, so with the default margin of 64
pixels the cropped spline equals the full-map spline to round-off inside ``half``.

Boundary: the prefilter (``ndimage.spline_filter``) and the evaluation both use the ``mirror`` extension
(the samples reflected about the first and last pixel centres), so the spline passes through every pixel value,
the edge rows and columns included.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import ndimage

from . import cosmology

# Attributes of the lens-file contract and the types they are read as.
FLOAT_ATTRS = ("lens_redshift", "H0", "Om0", "kpc_per_arcsec", "sigma_critical_inf_Msun_pkpc2", "M200c_Msun",
               "R200c_pkpc", "box_ckpc_h", "h", "scale_factor", "trusted_half_width_arcsec")
INT_ATTRS = ("snapshot", "zoom_index", "halo_id")
# Older files used other names for some attributes; the first name is the contract's.
ALIASES = {"zoom_index": ("zoom_index", "zoom", "zoom_id"),
           "trusted_half_width_arcsec": ("trusted_half_width_arcsec", "trusted_half_arcsec", "trusted_half_width"),
           "scale_factor": ("scale_factor", "a")}
AXES = {"xy": (0, 1, 2), "yz": (1, 2, 0), "zx": (2, 0, 1)}   # (map x, map y, line of sight) simulation axes
SPLINE_FIELDS = ("alpha_x", "alpha_y", "potential")
SPLINE_MODE = "mirror"        # boundary extension of the prefilter; ``Spline`` evaluates with the same one
EDGE_PIXELS = 16              # the mirror boundary perturbs the coefficients by ~0.27**k at k pixels: 1e-9 at 16


def _plain(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def read_meta(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """The file attributes as plain Python values, with contract names, then ``overrides`` applied on top.

    Missing attributes stay missing; ``require`` checks for the ones a stage needs.
    """
    with h5py.File(path, "r") as f:
        raw = {k: _plain(v) for k, v in f.attrs.items()}
    meta = dict(raw)
    for name, alts in ALIASES.items():
        for a in alts:
            if a in raw:
                meta[name] = raw[a]
                break
    meta.update(overrides or {})
    for k in FLOAT_ATTRS:
        if k in meta and meta[k] is not None:
            meta[k] = float(meta[k])
    for k in INT_ATTRS:
        if k in meta and meta[k] is not None:
            meta[k] = int(meta[k])
    if "halo_position_ckpc_h" in meta and meta["halo_position_ckpc_h"] is not None:
        meta["halo_position_ckpc_h"] = [float(v) for v in meta["halo_position_ckpc_h"]]
    if "axis" in meta and meta["axis"] not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, not {meta['axis']!r}")
    return meta


def map_extent(path: str | Path):
    """(half-width of the largest centred square inside the map [arcsec], pixel size [arcsec]) of a lens file."""
    with h5py.File(path, "r") as f:
        x, y = f["x"][:].astype(np.float64), f["y"][:].astype(np.float64)
    return float(min(-x[0], x[-1], -y[0], y[-1])), float(x[1] - x[0])


def require(meta: dict, keys, what: str):
    missing = [k for k in keys if meta.get(k) is None]
    if missing:
        raise KeyError(f"{what} needs the lens-file attributes {missing}; they are missing from the file, "
                       "pass them as overrides")


# ----------------------------------------------------------------------------- cubic B-spline evaluation
def prefilter(a):
    """Cubic B-spline coefficients of the samples ``a`` with the ``SPLINE_MODE`` boundary (float64)."""
    return ndimage.spline_filter(np.asarray(a, dtype=np.float64), order=3, output=np.float64, mode=SPLINE_MODE)


def _mirror(i, n):
    """Indices reflected about 0 and n - 1 (the ``mirror`` extension), then clipped to the array."""
    if n > 1:
        i = np.abs(i)
        i = np.where(i > n - 1, 2 * (n - 1) - i, i)
    return np.clip(i, 0, n - 1)


def _bspline_weights(t):
    """Cubic B-spline weights of the four nodes floor(u)-1 .. floor(u)+2 and their derivatives in u."""
    t2, t3 = t * t, t * t * t
    w = np.stack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3], axis=-1) / 6.0
    d = np.stack([-(1 - t) ** 2, 3 * t2 - 4 * t, -3 * t2 + 2 * t + 1, t2], axis=-1) / 2.0
    return w, d


@dataclass
class Spline:
    """Cubic B-spline coefficients on a uniform grid; values and gradients at arbitrary points.

    ``fit`` prefilters with the ``mirror`` boundary and evaluation reflects the coefficient indices the same way,
    so values equal ``ndimage.map_coordinates(coef, ..., order=3, prefilter=False, mode='mirror')`` (and
    ``map_coordinates(data, ..., order=3, mode='mirror')``) everywhere on the grid, edges included, and the
    spline interpolates every sample.  The gradient is the analytic derivative of that spline.
    """
    x0: float
    y0: float
    dx: float
    dy: float
    coef: dict[str, np.ndarray]
    chunk: int = 200_000

    @classmethod
    def fit(cls, x, y, arrays: dict[str, np.ndarray]):
        coef = {k: prefilter(v) for k, v in arrays.items()}
        return cls(float(x[0]), float(y[0]), float(x[1] - x[0]), float(y[1] - y[0]), coef)

    def __call__(self, pts, names, deriv: bool = False):
        """Values (and d/dx, d/dy if ``deriv``) of each field in ``names`` at ``pts`` (N, 2) [x, y].

        Returns {name: values} or {name: (values, d/dx, d/dy)}.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out = {n: [] for n in names}
        for lo in range(0, len(pts), self.chunk):
            p = pts[lo:lo + self.chunk]
            u = (p[:, 0] - self.x0) / self.dx
            v = (p[:, 1] - self.y0) / self.dy
            iu, iv = np.floor(u), np.floor(v)
            wu, du = _bspline_weights(u - iu)
            wv, dv = _bspline_weights(v - iv)
            for n in names:
                c = self.coef[n]
                ix = _mirror(iu.astype(np.int64)[:, None] + np.arange(-1, 3), c.shape[1])
                iy = _mirror(iv.astype(np.int64)[:, None] + np.arange(-1, 3), c.shape[0])
                g = c[iy[:, :, None], ix[:, None, :]]                   # (N, 4 in y, 4 in x)
                gx = np.einsum("nij,nj->ni", g, wu)                      # summed over x
                val = np.einsum("ni,ni->n", gx, wv)
                if deriv:
                    ddx = np.einsum("nij,nj,ni->n", g, du, wv) / self.dx
                    ddy = np.einsum("ni,ni->n", gx, dv) / self.dy
                    out[n].append((val, ddx, ddy))
                else:
                    out[n].append(val)
        if deriv:
            return {n: tuple(np.concatenate([c[i] for c in out[n]]) for i in range(3)) for n in names}
        return {n: np.concatenate(out[n]) for n in names}


# ----------------------------------------------------------------------------- the lens map
@dataclass
class LensMap:
    """One lens file: metadata, cosmology and splines of the deflection and potential (D_ds/D_s = 1)."""
    path: str
    meta: dict
    cosmo: Any
    zl: float
    kpc_per_arcsec: float
    sigcr_inf: float
    half: float                       # half-width of the splined square [arcsec]
    x: np.ndarray                     # pixel centres of the splined square
    y: np.ndarray
    spline: Spline
    has_potential: bool
    notes: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path, overrides: dict | None = None, half: float | None = None, margin: int = 64,
                  potential: bool = True):
        """Read ``path`` and spline the square of half-width ``half`` (default: the whole map).

        ``overrides`` replaces or supplies file attributes (for files written before the contract had them).
        The cosmology is flat LCDM with the file's ``H0`` and ``Om0``.
        """
        meta = read_meta(path, overrides)
        require(meta, ("lens_redshift", "H0", "Om0"), "the lens map")
        cosmo = cosmology.make({"H0": meta["H0"], "Om0": meta["Om0"]})
        zl = meta["lens_redshift"]
        notes = []
        kpa = cosmology.kpc_per_arcsec(cosmo, zl)
        if meta.get("kpc_per_arcsec") is not None:
            if abs(meta["kpc_per_arcsec"] / kpa - 1) > 1e-4:
                notes.append(f"file kpc_per_arcsec {meta['kpc_per_arcsec']:.6g} differs from the cosmology's "
                             f"{kpa:.6g}; using the file's value for the map frame")
            kpa = meta["kpc_per_arcsec"]
        sig = cosmology.sigma_critical_inf(cosmo, zl)
        if meta.get("sigma_critical_inf_Msun_pkpc2") is not None:
            sig = meta["sigma_critical_inf_Msun_pkpc2"]
        with h5py.File(path, "r") as f:
            x, y = f["x"][:].astype(np.float64), f["y"][:].astype(np.float64)
            for a in (x, y):
                d = np.diff(a)
                if not (np.all(d > 0) and np.allclose(d, d[0], rtol=1e-6, atol=0)):
                    raise ValueError(f"{path}: x and y must be ascending and uniform")
            if half is None:
                ix, iy = slice(0, len(x)), slice(0, len(y))
            else:
                if half > min(-x[0], x[-1], -y[0], y[-1]):
                    raise ValueError(f"half-width {half} arcsec exceeds the map ({x[0]:.2f} .. {x[-1]:.2f})")
                jx = np.flatnonzero(np.abs(x) <= half)
                jy = np.flatnonzero(np.abs(y) <= half)
                ix = slice(max(jx[0] - margin, 0), min(jx[-1] + margin + 1, len(x)))
                iy = slice(max(jy[0] - margin, 0), min(jy[-1] + margin + 1, len(y)))
            names = ["alpha_x", "alpha_y"] + (["potential"] if potential and "potential" in f else [])
            coef = {}
            for n in names:      # one field in float64 at a time
                a = f[n][iy, ix]
                coef[n] = prefilter(a)
                del a
            xs, ys = x[ix], y[iy]
            spline = Spline(float(xs[0]), float(ys[0]), float(xs[1] - xs[0]), float(ys[1] - ys[0]), coef)
        half_eff = half if half is not None else float(min(-x[0], x[-1], -y[0], y[-1]))
        return cls(str(path), meta, cosmo, zl, float(kpa), float(sig), float(half_eff), xs, ys, spline,
                   "potential" in coef, notes)

    # --------------------------------------------------------------- geometry
    def efficiency(self, zs):
        return cosmology.efficiency(self.cosmo, self.zl, zs)

    def time_delay_days(self):
        return cosmology.time_delay_days(self.cosmo, self.zl)

    @property
    def trusted_half(self):
        return self.meta.get("trusted_half_width_arcsec")

    # --------------------------------------------------------------- fields
    def alpha(self, pts):
        """Deflection for D_ds/D_s = 1 [arcsec] at ``pts`` (N, 2); shape (N, 2)."""
        v = self.spline(pts, ("alpha_x", "alpha_y"))
        return np.column_stack([v["alpha_x"], v["alpha_y"]])

    def potential(self, pts):
        """Lensing potential for D_ds/D_s = 1 [arcsec^2] at ``pts``."""
        if not self.has_potential:
            raise KeyError(f"{self.path} has no potential map")
        return self.spline(pts, ("potential",))["potential"]

    def jacobian(self, pts):
        """Analytic derivatives of the deflection spline (D_ds/D_s = 1): a_xx, a_xy, a_yx, a_yy.

        a_xy = d alpha_x / d y.  Also returns the deflection itself as a fifth item, (N, 2).
        """
        v = self.spline(pts, ("alpha_x", "alpha_y"), deriv=True)
        ax, axx, axy = v["alpha_x"]
        ay, ayx, ayy = v["alpha_y"]
        return axx, axy, ayx, ayy, np.column_stack([ax, ay])

    def hessian(self, pts, s):
        """kappa, gamma1, gamma2 (scaled by ``s``) and det(A) from the deflection Jacobian.

        kappa = (a_xx + a_yy)/2, gamma1 = (a_xx - a_yy)/2, gamma2 = (a_xy + a_yx)/2; the antisymmetric part
        (a curl of the map, zero for a true gradient field) only enters det(A).
        """
        axx, axy, ayx, ayy, _ = self.jacobian(pts)
        axx, axy, ayx, ayy = s * axx, s * axy, s * ayx, s * ayy
        return 0.5 * (axx + ayy), 0.5 * (axx - ayy), 0.5 * (axy + ayx), (1 - axx) * (1 - ayy) - axy * ayx


# ----------------------------------------------------------------------------- synthetic lens files
def analytic_lens(b=15.0, rc=2.0, e=0.15, gamma_ext=(0.05, -0.03), phi=0.3):
    """A pseudo-elliptical cored isothermal potential plus external shear, normalised to D_ds/D_s = 1.

    psi = b sqrt(rc^2 + (1 - e) u^2 + (1 + e) v^2) + g1 (x^2 - y^2)/2 + g2 x y, with (u, v) the coordinates
    rotated by ``phi``.  Returns a function of (x, y) arrays giving dict(psi, ax, ay, pxx, pyy, pxy), all exact.
    """
    g1, g2 = gamma_ext
    c, sn = math.cos(phi), math.sin(phi)
    qa, qb = 1 - e, 1 + e

    def f(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        u, v = c * x + sn * y, -sn * x + c * y
        R = np.sqrt(rc ** 2 + qa * u * u + qb * v * v)
        pu, pv = b * qa * u / R, b * qb * v / R
        puu = b * qa / R - b * (qa * u) ** 2 / R ** 3
        pvv = b * qb / R - b * (qb * v) ** 2 / R ** 3
        puv = -b * qa * qb * u * v / R ** 3
        ax = c * pu - sn * pv + g1 * x + g2 * y
        ay = sn * pu + c * pv - g1 * y + g2 * x
        pxx = c * c * puu - 2 * c * sn * puv + sn * sn * pvv + g1
        pyy = sn * sn * puu + 2 * c * sn * puv + c * c * pvv - g1
        pxy = c * sn * (puu - pvv) + (c * c - sn * sn) * puv + g2
        psi = b * R + 0.5 * g1 * (x * x - y * y) + g2 * x * y
        return dict(psi=psi, ax=ax, ay=ay, pxx=pxx, pyy=pyy, pxy=pxy)

    return f


def write_synthetic_lens(path, lens=None, n=512, half=64.0, zl=0.3, H0=70.0, Om0=0.3, attrs=None):
    """Write a lens file in the contract format from an analytic lens (see ``analytic_lens``), for tests.

    Pixel centres are uniform in [-half, half).  Only the fields of the contract are written.
    """
    lens = lens or analytic_lens()
    dx = 2 * half / n
    x = -half + dx * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(x, x)
    v = lens(X, Y)
    cosmo = cosmology.make({"H0": H0, "Om0": Om0})
    kpa = cosmology.kpc_per_arcsec(cosmo, zl)
    sig = cosmology.sigma_critical_inf(cosmo, zl)
    kappa = 0.5 * (v["pxx"] + v["pyy"])
    with h5py.File(path, "w") as f:
        f["x"], f["y"] = x, x
        for k, a in (("kappa", kappa), ("alpha_x", v["ax"]), ("alpha_y", v["ay"]), ("potential", v["psi"]),
                     ("gamma1", 0.5 * (v["pxx"] - v["pyy"])), ("gamma2", v["pxy"]), ("Sigma", kappa * sig)):
            f.create_dataset(k, data=a.astype(np.float32))
        base = dict(lens_redshift=zl, snapshot=99, zoom_index=0, halo_id=0, axis="yz", H0=H0, Om0=Om0,
                    kpc_per_arcsec=kpa, sigma_critical_inf_Msun_pkpc2=sig, M200c_Msun=1e15, R200c_pkpc=2000.0,
                    halo_position_ckpc_h=[1000.0, 1000.0, 1000.0], box_ckpc_h=680000.0, h=0.6774,
                    scale_factor=1 / (1 + zl), group_catalog_base="", particle_types=[0, 1, 4, 5],
                    trusted_half_width_arcsec=0.9 * half, source_redshifts=[2.0])
        base.update(attrs or {})
        f.attrs.update(base)
    return path
