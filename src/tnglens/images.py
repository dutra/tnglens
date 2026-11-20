"""Find every image of a point source and describe it.

``ImageFinder.find`` maps a regular mesh of the image plane to the source plane, splits each mesh cell into
two triangles, and takes every triangle whose source-plane image contains beta as a starting point (the
barycentric interpolation of its corners).  Each start is refined with Newton's method on the splined
deflection, all starts at once, and the few that do not converge are retried with ``scipy.optimize.root``.
Roots that solve the lens equation to ``ROOT_TOL`` arcsec inside the search square are kept, merging any
closer than ``MERGE_TOL``.  Two images closer than a mesh step (a fold pair) can share one triangle; when the
count comes out even (a smooth lens with every image in the square gives an odd number) the most magnified
images are searched again on a fine local mesh.  Images can still be missed where the mapping has structure
smaller than the mesh step (``stride`` map pixels) away from the brightest images.

``image_table`` gives magnification, parity, kappa and shear from the Jacobian of the same spline, and the
Fermat potential and time delay from the potential map.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

ROOT_TOL = 1e-7      # |theta - s alpha(theta) - beta| that counts as a root [arcsec]
MERGE_TOL = 2e-3     # roots closer than this are the same image [arcsec]
MAX_STEP = 0.5       # Newton step limit [arcsec]


class ImageFinder:
    """Triangle-mapping image finder on the square |x|, |y| <= ``half`` of a ``LensMap``."""

    def __init__(self, lens, half: float, stride: int = 2):
        if half > lens.half + 1e-9:
            raise ValueError(f"search half-width {half} exceeds the splined region {lens.half}")
        self.lens, self.half, self.stride = lens, float(half), int(stride)
        xs = lens.x[np.abs(lens.x) <= half][::stride]
        ys = lens.y[np.abs(lens.y) <= half][::stride]
        self.xs, self.ys = xs, ys
        X, Y = np.meshgrid(xs, ys)
        axx, axy, ayx, ayy, a = lens.jacobian(np.column_stack([X.ravel(), Y.ravel()]))
        shape = X.shape
        self.ax, self.ay = a[:, 0].reshape(shape), a[:, 1].reshape(shape)
        self.jac = tuple(v.reshape(shape) for v in (axx, axy, ayx, ayy))
        self.step = float(xs[1] - xs[0])

    @property
    def mesh(self):
        return np.meshgrid(self.xs, self.ys)

    def det(self, s):
        """det(A) on the mesh for efficiency ``s`` (the last one is cached; do not modify it)."""
        if getattr(self, "_det_s", None) != s:
            axx, axy, ayx, ayy = self.jac
            self._det = (1 - s * axx) * (1 - s * ayy) - s * s * axy * ayx
            self._det_s = s
        return self._det

    def source_plane(self, s):
        """beta of every mesh node, (ny, nx) each (the last one is cached; do not modify it)."""
        if getattr(self, "_bp_s", None) != s:
            self._bp = (self.xs[None, :] - s * self.ax, self.ys[:, None] - s * self.ay)
            self._bp_s = s
        return self._bp

    # --------------------------------------------------------------- triangles
    def guesses(self, beta, s):
        """Starting points: the barycentric interpolation of each triangle whose image contains ``beta``."""
        bx, by = self.source_plane(s)
        return triangle_guesses(self.xs, self.ys, bx - beta[0], by - beta[1])

    def critical_cells(self, beta, s, max_cells: int = 32):
        """Centres of mesh cells crossed by a critical curve whose source-plane bounding box contains ``beta``.

        In such a cell the mapped triangles fold over, and a fold pair inside it can be missed by the triangle
        test.  The box is padded by a quarter of its size; neighbouring hits are merged; at most ``max_cells``.
        """
        det = self.det(s)
        P = lambda a, i, j: a[i:a.shape[0] - 1 + i, j:a.shape[1] - 1 + j]
        corners = ((0, 0), (0, 1), (1, 0), (1, 1))
        pos = [P(det, i, j) > 0 for i, j in corners]
        mixed = ~((pos[0] == pos[1]) & (pos[0] == pos[2]) & (pos[0] == pos[3]))
        iy, ix = np.nonzero(mixed)
        if not len(iy):
            return np.zeros((0, 2))
        bx, by = self.source_plane(s)
        cx = np.stack([bx[iy + i, ix + j] for i, j in corners])
        cy = np.stack([by[iy + i, ix + j] for i, j in corners])
        lo_x, hi_x, lo_y, hi_y = cx.min(0), cx.max(0), cy.min(0), cy.max(0)
        pad = 0.25 * np.maximum(hi_x - lo_x, hi_y - lo_y)
        hit = (lo_x - pad <= beta[0]) & (beta[0] <= hi_x + pad) & (lo_y - pad <= beta[1]) & (beta[1] <= hi_y + pad)
        centres = np.column_stack([self.xs[ix[hit]], self.ys[iy[hit]]]) + 0.5 * self.step
        out = []
        for c in centres:
            if not any(np.max(np.abs(c - o)) < 1.5 * self.step for o in out):
                out.append(c)
            if len(out) >= max_cells:
                break
        return np.array(out).reshape(-1, 2)

    def local_guesses(self, centre, beta, s, half=None, n=41):
        """Triangle guesses on a fine mesh (n x n) of half-width ``half`` (default 3 mesh steps) about ``centre``."""
        half = 3 * self.step if half is None else half
        xs = centre[0] + np.linspace(-half, half, n)
        ys = centre[1] + np.linspace(-half, half, n)
        X, Y = np.meshgrid(xs, ys)
        a = self.lens.alpha(np.column_stack([X.ravel(), Y.ravel()]))
        bx = X - s * a[:, 0].reshape(X.shape) - beta[0]
        by = Y - s * a[:, 1].reshape(X.shape) - beta[1]
        return triangle_guesses(xs, ys, bx, by)

    # --------------------------------------------------------------- refinement
    def refine(self, guesses, beta, s, iters: int = 40):
        """Newton iteration on all starts; returns roots (N, 2) and residual norms."""
        return newton(self.lens, guesses, beta, s, iters)

    def _keep(self, th, res, roots):
        ok = (res < ROOT_TOL) & (np.max(np.abs(th), axis=1) <= self.half) if len(th) else np.zeros(0, bool)
        for p in th[ok][np.argsort(res[ok], kind="stable")]:
            if not any(np.hypot(*(p - r)) < MERGE_TOL for r in roots):
                roots.append(p)
        return roots

    def find(self, beta, s, n_local: int = 3):
        """All images of ``beta`` (arcsec, efficiency ``s``) in the search square, (N, 2).

        Two images closer than a mesh step (a fold pair) can share one cell and be missed by the triangle test,
        so every critical cell whose source-plane image may hold ``beta`` (``critical_cells``) is searched again
        on a 10x finer local mesh.  If the count is still even (a smooth lens with all images in the square has
        an odd number), the ``n_local`` most magnified images are searched on a ~7x finer mesh 6 steps wide.
        """
        beta = np.asarray(beta, dtype=np.float64)
        th, res = self.refine(self.guesses(beta, s), beta, s)
        roots = self._keep(th, res, [])
        for c in self.critical_cells(beta, s):
            th, res = self.refine(self.local_guesses(c, beta, s, half=1.5 * self.step, n=31), beta, s)
            roots = self._keep(th, res, roots)
        if n_local and len(roots) % 2 == 0 and len(roots):
            r = np.array(roots)
            _, _, _, det = self.lens.hessian(r, s)
            for i in np.argsort(np.abs(det), kind="stable")[:n_local]:
                g = self.local_guesses(r[i], beta, s)
                th, res = self.refine(g, beta, s)
                roots = self._keep(th, res, roots)
        roots = np.array(roots).reshape(-1, 2)
        # deterministic order: by position
        return roots[np.lexsort((roots[:, 1], roots[:, 0]))] if len(roots) else roots


def triangle_guesses(xs, ys, bx, by):
    """Barycentric start points of the mesh triangles whose source-plane image contains the origin.

    ``xs``, ``ys``: mesh coordinates; ``bx``, ``by`` (len(ys), len(xs)): beta(theta) - beta at the nodes.
    """
    def cross(ax_, ay_, bx_, by_):
        return ax_ * by_ - ay_ * bx_

    P = lambda a, i, j: a[i:a.shape[0] - 1 + i, j:a.shape[1] - 1 + j]
    x00, y00, x01, y01 = P(bx, 0, 0), P(by, 0, 0), P(bx, 0, 1), P(by, 0, 1)
    x10, y10, x11, y11 = P(bx, 1, 0), P(by, 1, 0), P(bx, 1, 1), P(by, 1, 1)
    h0 = cross(x00, y00, x01, y01)          # edge p00 -> p01
    h1 = cross(x10, y10, x11, y11)          # edge p10 -> p11
    v0 = cross(x00, y00, x10, y10)          # edge p00 -> p10
    v1 = cross(x01, y01, x11, y11)          # edge p01 -> p11
    dg = cross(x00, y00, x11, y11)          # diagonal p00 -> p11
    out = []
    # triangle (p00, p01, p11): cross(p00,p01)=h0, cross(p01,p11)=v1, cross(p11,p00)=-dg
    # triangle (p00, p11, p10): cross(p00,p11)=dg, cross(p11,p10)=-h1, cross(p10,p00)=-v0
    # the origin is inside when the three crosses share a sign; they are its barycentric weights (x2 area)
    for c_ab, c_bc, c_ca, corners in ((h0, v1, -dg, ((0, 0), (0, 1), (1, 1))),
                                      (dg, -h1, -v0, ((0, 0), (1, 1), (1, 0)))):
        inside = ((c_ab >= 0) & (c_bc >= 0) & (c_ca >= 0)) | ((c_ab <= 0) & (c_bc <= 0) & (c_ca <= 0))
        iy, ix = np.nonzero(inside)
        if not len(iy):
            continue
        ab, bc, ca = c_ab[iy, ix], c_bc[iy, ix], c_ca[iy, ix]
        tot = ab + bc + ca
        tot = np.where(tot == 0, 1.0, tot)
        w = (bc / tot, ca / tot, ab / tot)       # weights of a, b, c
        gx, gy = np.zeros(len(ix)), np.zeros(len(ix))
        for wk, (di, dj) in zip(w, corners):
            gx += wk * xs[ix + dj]
            gy += wk * ys[iy + di]
        out.append(np.column_stack([gx, gy]))
    return np.concatenate(out) if out else np.zeros((0, 2))


def newton(lens, guesses, beta, s, iters: int = 40):
    """Newton iteration of the lens equation from all ``guesses`` at once (step limited to ``MAX_STEP``),
    with ``scipy.optimize.root`` for the starts that do not converge.  Returns roots (N, 2) and residuals."""
    guesses = np.array(guesses, dtype=np.float64).reshape(-1, 2)
    th = guesses.copy()
    beta = np.asarray(beta, dtype=np.float64)
    if not len(th):
        return th, np.zeros(0)
    active = np.ones(len(th), bool)
    res = np.full(len(th), np.inf)
    for _ in range(iters):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        axx, axy, ayx, ayy, a = lens.jacobian(th[idx])
        f = th[idx] - s * a - beta
        res[idx] = np.hypot(f[:, 0], f[:, 1])
        done = res[idx] < 1e-11
        active[idx[done]] = False
        idx, f = idx[~done], f[~done]
        if not len(idx):
            break
        a11, a12, a21, a22 = 1 - s * axx[~done], -s * axy[~done], -s * ayx[~done], 1 - s * ayy[~done]
        det = a11 * a22 - a12 * a21
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        dx = (a22 * f[:, 0] - a12 * f[:, 1]) / det
        dy = (-a21 * f[:, 0] + a11 * f[:, 1]) / det
        n = np.hypot(dx, dy)
        scale = np.where(n > MAX_STEP, MAX_STEP / np.maximum(n, 1e-300), 1.0)
        th[idx, 0] -= scale * dx
        th[idx, 1] -= scale * dy
    a = lens.alpha(th)
    f = th - s * a - beta
    res = np.hypot(f[:, 0], f[:, 1])
    for i in np.flatnonzero(~(res < ROOT_TOL)):
        fun = lambda p: p - s * lens.alpha(p[None])[0] - beta
        sol = optimize.root(fun, guesses[i], method="hybr", options={"xtol": 1e-11})
        r = float(np.hypot(*fun(sol.x)))
        if r < res[i] or not np.isfinite(res[i]):
            th[i], res[i] = sol.x, r
    return th, res


def fermat(lens, theta, beta, s):
    """Fermat potential 0.5 |theta - beta|^2 / s - psi_inf(theta) [arcsec^2]; times ``lens.time_delay_days()``
    it is the arrival time in days up to a constant."""
    d = np.asarray(theta) - np.asarray(beta)
    return 0.5 * np.sum(d * d, axis=1) / s - lens.potential(theta)


def image_table(lens, theta, beta, s):
    """Per-image properties: kappa, gamma1, gamma2, det, mu, parity, residual, and if the lens has a potential,
    fermat [arcsec^2] and dt_days (relative to the first-arriving image)."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 2)
    k, g1, g2, det = lens.hessian(theta, s)
    resid = theta - s * lens.alpha(theta) - np.asarray(beta)
    out = dict(theta=theta, kappa=k, gamma1=g1, gamma2=g2, det=det, mu=1 / det,
               parity=np.sign(det).astype(int), resid=np.hypot(resid[:, 0], resid[:, 1]))
    if lens.has_potential and len(theta):
        tau = fermat(lens, theta, beta, s)
        out["fermat"] = tau
        out["dt_days"] = (tau - tau.min()) * lens.time_delay_days()
    else:
        out["fermat"] = np.full(len(theta), np.nan)
        out["dt_days"] = np.full(len(theta), np.nan)
    return out


def observable(tab, centre=(0.0, 0.0), r_exclude=3.0, mu_min=0.25):
    """Images farther than ``r_exclude`` from ``centre`` (the BCG) and with |mu| > ``mu_min``."""
    th = tab["theta"]
    r = np.hypot(th[:, 0] - centre[0], th[:, 1] - centre[1])
    return (r > r_exclude) & (np.abs(tab["mu"]) > mu_min)


def separations(theta):
    """(min, max) separation of a set of positions; (inf, 0) for fewer than two."""
    n = len(theta)
    if n < 2:
        return np.inf, 0.0
    d = np.linalg.norm(theta[:, None] - theta[None], axis=2)
    iu = np.triu_indices(n, 1)
    return float(d[iu].min()), float(d[iu].max())


def classify(theta, parity):
    """Image-family class of the observable images, as in the tngc014 mock (a label, not a selection).

    wide: closest pair >= 8"; intermediate: 3-8"; fold: an opposite-parity pair <= 3" with the next image
    more than max(2 sep, 2") away; cusp: three close images of alternating parity; other otherwise.
    """
    n = len(theta)
    if n < 2:
        return "single"
    d = np.linalg.norm(theta[:, None] - theta[None], axis=2) + np.eye(n) * 1e9
    i, j = np.unravel_index(np.argmin(d), d.shape)
    min_sep = float(d[i, j])
    cls = "wide" if min_sep >= 8 else "intermediate" if min_sep >= 3 else "other"
    if min_sep <= 3 and parity[i] != parity[j]:
        others = np.delete(np.arange(n), [i, j])
        third = float(np.min(np.minimum(d[i, others], d[j, others]))) if n > 2 else np.inf
        if third > max(2 * min_sep, 2.0):
            cls = "fold"
        elif third <= 8:
            cls = "cusp"
    if cls in ("intermediate", "wide") and n >= 3:
        for a in range(n):
            near = np.argsort(d[a])[:2]
            if d[a, near[1]] < 6 and parity[a] != parity[near[0]] and parity[a] != parity[near[1]]:
                cls = "cusp"
    return cls
