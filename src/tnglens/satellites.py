"""Single satellites of the lens: their own deflection, and the counterfactual that ties images to them.

The deflection of one satellite comes from its bound particles (the Subfind particles of the particle types of
the lens, read from the snapshot chunk): they are binned on a local grid about the satellite, smoothed with a
Gaussian, and convolved with the free-space deflection kernel (``Patch``).  Inside ``patch_keep`` of the satellite
the deflection is a cubic B-spline of that grid, with its analytic derivatives; outside it is a point mass.  The
smoothing (0.16" by default) is not the adaptive kernel of the lens map, so the result classifies images; it is not
a measurement of the satellite's field.

``Satellites.counterfactual`` asks, for each satellite near an observable image of a system: without this
satellite, would the image still be there?  The satellite's deflection is subtracted from the lens, the source
position is refitted to the observable images (weighted to image-plane offsets), and the images are solved again.
An image within the search radius of the satellite (``radius_einstein`` point-mass Einstein radii, clipped to
``radius_clip``) that vanishes or moves by more than ``shift_tol`` is tied to it.

``Satellites.seed_mask`` marks the image-plane mesh nodes near massive satellites, for the ``subhalo`` seeding kind
of ``tnglens.sources``.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from scipy import fft, ndimage
from scipy.optimize import linear_sum_assignment

from . import members as members_mod
from .images import ImageFinder
from .lensmap import AXES, Spline, require

DEFAULTS = dict(
    enabled="auto",            # auto: on when quotas.subhalo or the subhalo seeding weight is positive and
                               # the simulation files are found; true: an error if they are not; false
    min_mass=3e10,             # satellites tested: bound mass in the particle types of the lens [Msun]
    shift_tol=0.5,             # an image that moves by more than this without the satellite is tied to it [arcsec]
    radius_einstein=4.0,       # search radius about the satellite, in its point-mass Einstein radii ...
    radius_clip=[1.5, 6.0],    # ... clipped to this range [arcsec]
    match_radius=5.0,          # an image without the satellite is the same image if it is closer than this [arcsec]
    patch_half=40.0,           # half-width of the satellite's local grid [arcsec]
    patch_pixels=1000,         # pixels per side of that grid
    patch_keep=30.0,           # the grid is used inside this half-width; a point mass outside [arcsec]
    patch_sigma_pixels=2.0,    # Gaussian smoothing of the satellite's surface density [grid pixels]
    seed_min_mass=1e11,        # the subhalo seeding kind uses satellites more massive than this [Msun] ...
    seed_radius=4.0,           # ... and mesh nodes within this distance of one [arcsec] ...
    seed_min_centre=8.0,       # ... for satellites farther than this from the BCG [arcsec]
    max_cached=24,             # satellite grids kept in memory per process
)
EDGE = 24                      # grid pixels kept beyond patch_keep, so the cropped spline is unaffected inside it


def _solve(a, b):
    """Least-squares solution of a 2x2 system (a system with every image on a critical curve is singular)."""
    return np.linalg.lstsq(a, b, rcond=None)[0]


class Patch:
    """The isolated deflection of one satellite (D_ds/D_s = 1) about its centre ``(cx, cy)`` [arcsec]."""

    def __init__(self, centre, offsets, masses, sigcr_inf, kpc_per_arcsec, half=40.0, pixels=1000, keep=30.0,
                 sigma_pixels=2.0):
        self.cx, self.cy, self.keep = float(centre[0]), float(centre[1]), float(keep)
        n = int(pixels)
        edges = np.linspace(-half, half, n + 1)
        pix = float(edges[1] - edges[0])
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        mass, _, _ = np.histogram2d(offsets[:, 1], offsets[:, 0], bins=[edges, edges], weights=masses)
        kappa = mass / (pix * kpc_per_arcsec) ** 2 / sigcr_inf
        if sigma_pixels > 0:
            kappa = ndimage.gaussian_filter(kappa, sigma_pixels)
        size = fft.next_fast_len(2 * n, real=True)           # zero padding: no periodic images
        c = fft.fftfreq(size, 1.0 / size) * pix
        X, Y = np.meshgrid(c, c)
        R2 = X * X + Y * Y
        R2[0, 0] = np.inf
        fk = fft.rfft2(kappa, s=(size, size))
        centres = 0.5 * (edges[1:] + edges[:-1])
        sel = np.flatnonzero(np.abs(centres) <= keep + EDGE * pix)
        crop = np.ix_(sel, sel)
        fields = {}
        for name, kernel in (("alpha_x", X / R2), ("alpha_y", Y / R2)):
            fields[name] = (fft.irfft2(fk * fft.rfft2(kernel), s=(size, size))[:n, :n] * pix ** 2 / np.pi)[crop]
        self.theta_e2 = float(kappa.sum() * pix ** 2 / np.pi)      # point-mass Einstein radius squared [arcsec^2]
        self.spline = Spline.fit(centres[sel], centres[sel], fields)
        self.mass_fraction = float(mass.sum() / np.sum(masses)) if len(offsets) else 0.0

    def _point(self, dx, dy):
        r2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = self.theta_e2
        axx = t * (dy * dy - dx * dx) / r2 ** 2
        axy = -2.0 * t * dx * dy / r2 ** 2
        return t * dx / r2, t * dy / r2, axx, axy, axy.copy(), -axx

    def fields(self, pts):
        """Deflection (N, 2) and its derivatives a_xx, a_xy, a_yx, a_yy at ``pts`` (N, 2)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        dx, dy = pts[:, 0] - self.cx, pts[:, 1] - self.cy
        ax, ay, axx, axy, ayx, ayy = self._point(dx, dy)
        inside = (np.abs(dx) < self.keep) & (np.abs(dy) < self.keep)
        if inside.any():
            v = self.spline(np.column_stack([dx[inside], dy[inside]]), ("alpha_x", "alpha_y"), deriv=True)
            ax[inside], axx[inside], axy[inside] = v["alpha_x"]
            ay[inside], ayx[inside], ayy[inside] = v["alpha_y"]
        return np.column_stack([ax, ay]), axx, axy, ayx, ayy

    def alpha(self, pts):
        return self.fields(pts)[0]

    def on_mesh(self, xs, ys):
        """The same six fields on the mesh ``xs`` x ``ys``, each (len(ys), len(xs))."""
        dx, dy = np.asarray(xs) - self.cx, np.asarray(ys) - self.cy
        out = [np.array(a) for a in self._point(dx[None, :], dy[:, None])]
        jx, jy = np.flatnonzero(np.abs(dx) < self.keep), np.flatnonzero(np.abs(dy) < self.keep)
        if len(jx) and len(jy):
            X, Y = np.meshgrid(dx[jx], dy[jy])
            v = self.spline(np.column_stack([X.ravel(), Y.ravel()]), ("alpha_x", "alpha_y"), deriv=True)
            block = np.ix_(jy, jx)
            for target, values in zip(out, (v["alpha_x"][0], v["alpha_y"][0], v["alpha_x"][1], v["alpha_x"][2],
                                            v["alpha_y"][1], v["alpha_y"][2])):
                target[block] = values.reshape(X.shape)
        return out


class MinusLens:
    """A ``LensMap`` with one satellite's deflection removed (deflection and its derivatives only)."""
    has_potential = False

    def __init__(self, lens, patch):
        self.lens, self.patch = lens, patch
        self.zl, self.half, self.x, self.y = lens.zl, lens.half, lens.x, lens.y

    def alpha(self, pts):
        return self.lens.alpha(pts) - self.patch.alpha(pts)

    def jacobian(self, pts):
        axx, axy, ayx, ayy, a = self.lens.jacobian(pts)
        pa, pxx, pxy, pyx, pyy = self.patch.fields(pts)
        return axx - pxx, axy - pxy, ayx - pyx, ayy - pyy, a - pa

    def hessian(self, pts, s):
        axx, axy, ayx, ayy, _ = self.jacobian(pts)
        axx, axy, ayx, ayy = s * axx, s * axy, s * ayx, s * ayy
        return 0.5 * (axx + ayy), 0.5 * (axx - ayy), 0.5 * (axy + ayx), (1 - axx) * (1 - ayy) - axy * ayx


def finder_without(finder, patch):
    """An ``ImageFinder`` on the mesh of ``finder`` for the lens without the satellite of ``patch``."""
    ax, ay, axx, axy, ayx, ayy = patch.on_mesh(finder.xs, finder.ys)
    out = object.__new__(ImageFinder)
    out.lens, out.half, out.stride = MinusLens(finder.lens, patch), finder.half, finder.stride
    out.xs, out.ys, out.step = finder.xs, finder.ys, finder.step
    out.ax, out.ay = finder.ax - ax, finder.ay - ay
    out.jac = tuple(a - b for a, b in zip(finder.jac, (axx, axy, ayx, ayy)))
    return out


def particle_reader(meta, table, kpc_per_arcsec, snapshot_file):
    """k -> (offsets [arcsec] of satellite k's bound particles from its centre, masses [Msun]), all lens types."""
    require(meta, ("particle_types", "halo_position_ckpc_h", "box_ckpc_h", "h", "scale_factor", "axis"),
            "the satellite deflection")
    snapshot = Path(snapshot_file)
    if not snapshot.exists():
        raise FileNotFoundError(f"the satellite deflection needs the snapshot chunk {snapshot.absolute()}, which does "
                                "not exist (set catalog.members.snapshot_file, or catalog.subhalos.enabled: false)")
    h, a, box = meta["h"], meta["scale_factor"], meta["box_ckpc_h"]
    centre = np.asarray(meta["halo_position_ckpc_h"])
    ix, iy, _ = AXES[meta["axis"]]
    types = [int(t) for t in meta["particle_types"]]

    def read(k):
        offsets, masses = [], []
        with h5py.File(snapshot, "r") as f:
            table_mass = np.asarray(f["Header"].attrs["MassTable"], dtype=np.float64)
            for t in types:
                n, o = int(table["slen"][k, t]), int(table["offset"][k, t])
                if n == 0:
                    continue
                g = f[f"PartType{t}"]
                c = g["Coordinates"][o:o + n].astype(np.float64) - centre
                c = ((c + box / 2) % box - box / 2) * a / h
                m = g["Masses"][o:o + n].astype(np.float64) if "Masses" in g else np.full(n, table_mass[t])
                offsets.append(np.column_stack([c[:, ix], c[:, iy]]) / kpc_per_arcsec
                               - np.array([table["x"][k], table["y"][k]]))
                masses.append(m * 1e10 / h)
        if not offsets:
            return np.zeros((0, 2)), np.zeros(0)
        return np.concatenate(offsets), np.concatenate(masses)
    return read


def unavailable(meta, member_cfg=None):
    """Why the satellites of this lens file cannot be used (a short reason), or None when they can."""
    mcfg = {**members_mod.DEFAULTS, **(member_cfg or {})}
    if not members_mod.available(meta, mcfg):
        return "the Subfind group catalog was not found"
    lacking = [k for k in ("particle_types", "R200c_pkpc") if meta.get(k) is None]
    if lacking:
        return f"the lens file lacks the attributes {lacking}"
    snapshot = members_mod.catalog_paths(meta, mcfg)[2]
    if not snapshot.exists():
        return f"the snapshot chunk {snapshot} was not found"
    return None


class Satellites:
    """The satellites of one lens: who is tested, their deflection grids, the counterfactual and the seed mask.

    ``table`` is ``members.load_subhalos`` (arrays by local subhalo index: x, y [arcsec], los [pkpc], mlens
    [Msun], soff, central); ``reader(k)`` gives satellite k's particles (see ``particle_reader``).
    """

    def __init__(self, lens, finder, table, reader, cfg=None, radius=120.0, los_max=np.inf):
        self.lens, self.finder, self.table, self.reader = lens, finder, table, reader
        self.cfg = {**DEFAULTS, **(cfg or {})}
        mlens = np.asarray(table["mlens"], dtype=np.float64)
        if not np.isfinite(mlens).any():
            raise KeyError("the satellite counterfactual needs the bound mass in the particle types of the lens; the "
                           "lens file has no particle_types attribute (pass it as an override)")
        self.infield = (np.hypot(table["x"], table["y"]) < radius) & (np.abs(table["los"]) < los_max)
        self.infield[int(table["central"])] = False
        self.tested = np.flatnonzero(self.infield & (mlens >= self.cfg["min_mass"]))
        self._patches = OrderedDict()

    @classmethod
    def from_lens(cls, lens, finder, cfg, member_cfg):
        """From the lens file's metadata: the Subfind chunk and the snapshot chunk it points to."""
        meta = lens.meta
        mcfg = {**members_mod.DEFAULTS, **(member_cfg or {})}
        require(meta, ("R200c_pkpc", "particle_types"), "the satellite counterfactual")
        table = members_mod.load_subhalos(meta, lens.kpc_per_arcsec, mcfg)
        snapshot = members_mod.catalog_paths(meta, mcfg)[2]
        reader = particle_reader(meta, table, lens.kpc_per_arcsec, snapshot)
        return cls(lens, finder, table, reader, cfg, mcfg["radius"], mcfg["los_r200c"] * meta["R200c_pkpc"])

    def global_id(self, k):
        return int(k) + int(self.table["soff"])

    def patch(self, k):
        k = int(k)
        if k in self._patches:
            self._patches.move_to_end(k)
            return self._patches[k]
        c = self.cfg
        offsets, masses = self.reader(k)
        p = Patch((self.table["x"][k], self.table["y"][k]), offsets, masses, self.lens.sigcr_inf,
                  self.lens.kpc_per_arcsec, c["patch_half"], c["patch_pixels"], c["patch_keep"], c["patch_sigma_pixels"])
        self._patches[k] = p
        while len(self._patches) > int(c["max_cached"]):
            self._patches.popitem(last=False)
        return p

    def seed_mask(self, X, Y):
        """Mesh nodes within ``seed_radius`` of a satellite above ``seed_min_mass``, away from the BCG."""
        c, t = self.cfg, self.table
        central = int(t["central"])
        r_centre = np.hypot(t["x"] - t["x"][central], t["y"] - t["y"][central])
        ks = np.flatnonzero(self.infield & (np.asarray(t["mlens"]) > c["seed_min_mass"]) & (r_centre > c["seed_min_centre"]))
        mask = np.zeros(X.shape, bool)
        xs, ys = X[0], Y[:, 0]
        for k in ks:
            jx = np.flatnonzero(np.abs(xs - t["x"][k]) < c["seed_radius"])
            jy = np.flatnonzero(np.abs(ys - t["y"][k]) < c["seed_radius"])
            if len(jx) and len(jy):
                block = np.ix_(jy, jx)
                mask[block] |= np.hypot(X[block] - t["x"][k], Y[block] - t["y"][k]) < c["seed_radius"]
        return mask

    def counterfactual(self, s, tab, obs):
        """Satellites that hold observable images of a system in place.

        ``tab`` is ``images.image_table`` of the system, ``obs`` its observable mask.  Returns one dict per
        satellite with at least one tied image: sub (local index), id (global Subfind ID), theta_e and radius
        [arcsec], n_without (images of the refitted source without it), dist and shift per observable image
        (shift is inf for an image that vanishes), and tagged (indices of the tied images in ``tab``).
        """
        c, t = self.cfg, self.table
        obs = np.asarray(obs, bool)
        theta = tab["theta"][obs]
        if not len(theta) or not len(self.tested):
            return []
        k_, g1, g2 = tab["kappa"][obs], tab["gamma1"][obs], tab["gamma2"][obs]
        A = np.empty((len(theta), 2, 2))
        A[:, 0, 0], A[:, 1, 1] = 1 - k_ - g1, 1 - k_ + g1
        A[:, 0, 1] = A[:, 1, 0] = -g2
        M = np.linalg.pinv(A)
        W = np.einsum("nji,njk->nik", M, M)
        x, y = np.asarray(t["x"])[self.tested], np.asarray(t["y"])[self.tested]
        te = np.sqrt(s * np.asarray(t["mlens"])[self.tested] / (np.pi * self.lens.sigcr_inf * self.lens.kpc_per_arcsec ** 2))
        radius = np.clip(c["radius_einstein"] * te, *c["radius_clip"])
        dist = np.hypot(theta[:, None, 0] - x[None], theta[:, None, 1] - y[None])       # (images, satellites)
        results = []
        for j in np.flatnonzero(dist.min(axis=0) <= radius):
            k = int(self.tested[j])
            without = finder_without(self.finder, self.patch(k))
            b = theta - s * without.lens.alpha(theta)
            beta = _solve(W.sum(0), np.einsum("nij,nj->i", W, b))
            new, shift = np.zeros((0, 2)), np.full(len(theta), np.inf)
            for _ in range(2):
                new = without.find(beta, s)
                shift = np.full(len(theta), np.inf)
                match = np.full(len(theta), -1)
                if len(new):
                    d = np.linalg.norm(theta[:, None] - new[None], axis=2)
                    rows, cols = linear_sum_assignment(d)
                    good = d[rows, cols] < c["match_radius"]
                    shift[rows[good]], match[rows[good]] = d[rows, cols][good], cols[good]
                ok = match >= 0
                if ok.sum() < 2:
                    break
                # linearised update of the source that minimises the matched image-plane offsets
                resid = np.einsum("nij,nj->ni", A[ok], theta[ok] - new[match[ok]])
                beta = beta + _solve(W[ok].sum(0), np.einsum("nij,nj->i", W[ok], resid))
            tagged = (dist[:, j] < radius[j]) & (shift > c["shift_tol"])
            if tagged.any():
                results.append(dict(sub=k, id=self.global_id(k), theta_e=float(te[j]), radius=float(radius[j]),
                                    n_without=int(len(new)), dist=dist[:, j].tolist(), shift=shift.tolist(),
                                    tagged=np.flatnonzero(obs)[tagged].tolist()))
        return results
