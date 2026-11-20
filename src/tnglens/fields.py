"""Lensing potential and deflection from a convergence map, without periodic images.

``isolated_fields`` convolves kappa with the analytic free-space Green kernels on a grid zero
padded to at least twice its size, so the mass on the grid has no periodic copies and no mean
is subtracted.  With pixel centres theta_ij and pixel area dA (arcsec^2):

    psi(theta)   = (1/pi) sum kappa_ij ln|theta - theta_ij| dA
    alpha(theta) = (1/pi) sum kappa_ij (theta - theta_ij)/|theta - theta_ij|^2 dA
    gamma1, gamma2: the matching second-derivative kernels (diagnostics only)

The self term of alpha and gamma vanishes by symmetry; the self term of psi is the exact
average of ln r over a square cell.  psi has an arbitrary additive constant.

``multiscale_fields`` adds the tidal field of the mass outside the core: the environment map
is multiplied by the exact area fraction of each coarse cell that lies outside the core
(``partition_environment``), its fields are solved on the coarse grid and interpolated
(linearly) onto the core grid, where they are added to the core's own fields.  kappa is exactly
the core density.  Near the core edge the coarse quadrature is not trustworthy, so a guard of two
coarse cells is excluded from the "trusted" region.

``deflection_jacobian`` differentiates a stored deflection map the way the image finder sees
it: the exact derivative of its cubic B-spline interpolant.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage
from scipy.interpolate import RegularGridInterpolator



def isolated_fields(kappa, dx: float, threads: int = 8, names=("potential", "alpha_x", "alpha_y")) -> dict:
    """Fields of a square convergence map with pixel size ``dx`` (output in units of ``dx``)."""
    kappa = np.asarray(kappa, dtype=np.float64)
    n = len(kappa)
    if kappa.shape != (n, n):
        raise ValueError("a square grid is required")
    m = fft.next_fast_len(2 * n, real=True)
    axis = np.arange(m, dtype=np.float64)
    axis[axis > m // 2] -= m
    axis *= dx
    x, y = axis[None, :], axis[:, None]
    r2 = x * x + y * y
    r2[0, 0] = 1.0
    fk = fft.rfft2(kappa, s=(m, m), workers=threads)
    result = {}
    for name in names:
        if name == "potential":
            kernel = 0.5 * np.log(r2) / np.pi
            # Average of ln r over a square cell of side dx centred on the origin.
            kernel[0, 0] = (np.log(dx) - 0.5 * np.log(2) - 1.5 + np.pi / 4) / np.pi
        elif name == "alpha_x":
            kernel = np.broadcast_to(x, r2.shape) / r2 / np.pi
        elif name == "alpha_y":
            kernel = np.broadcast_to(y, r2.shape) / r2 / np.pi
        elif name == "gamma1":
            kernel = (y * y - x * x) / (r2 * r2 * np.pi)
        elif name == "gamma2":
            kernel = -2 * x * y / (r2 * r2 * np.pi)
        else:
            raise ValueError(f"unknown field {name}")
        if name != "potential":
            kernel[0, 0] = 0.0
        transformed = fft.rfft2(kernel, workers=threads)
        del kernel
        transformed *= fk
        value = fft.irfft2(transformed, s=(m, m), workers=threads)[:n, :n].copy()
        del transformed
        value *= dx * dx
        result[name] = value
    return result


def _uniform_axis(axis, what):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all():
        raise ValueError(f"{what} must be a finite 1D axis with at least three centres")
    step = np.diff(axis)
    if np.any(step <= 0) or not np.allclose(step, step[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{what} must be uniform and ascending")
    return axis, float(step[0])


def partition_environment(sigma_environment, x_environment, x_fine):
    """Environment density with the core footprint removed (exact cell-area fractions).

    Returns ``(exterior, info)``.  ``info`` holds the core footprint, the trusted interior
    (footprint minus two coarse cells), and the mass bookkeeping of the split.
    """
    xe, de = _uniform_axis(x_environment, "environment axis")
    xf, df = _uniform_axis(x_fine, "core axis")
    sigma = np.asarray(sigma_environment, dtype=np.float64)
    if sigma.shape != (len(xe), len(xe)) or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("environment density must be finite, non-negative, square and match its axis")
    lo, hi = float(xf[0] - df / 2), float(xf[-1] + df / 2)
    if lo < xe[0] - de / 2 or hi > xe[-1] + de / 2 or xf[0] < xe[0] or xf[-1] > xe[-1]:
        raise ValueError("the core grid must lie inside the environment grid")
    guard = 2 * de
    if hi - lo <= 2 * guard:
        raise ValueError("the core must be wider than four environment cells")
    lower, upper = xe - de / 2, xe + de / 2
    overlap = np.clip((np.minimum(upper, hi) - np.maximum(lower, lo)) / de, 0.0, 1.0)
    overlap[(lower >= lo) & (upper <= hi)] = 1.0
    fraction = np.multiply.outer(overlap, overlap)
    exterior = sigma * (1 - fraction)
    full = float(np.sum(sigma, dtype=np.float64) * de * de)
    removed = float(np.sum(sigma * fraction, dtype=np.float64) * de * de)
    info = dict(core_footprint=[lo, hi], trusted_interior=[lo + guard, hi - guard], guard=guard,
                environment_cell=de, environment_mass=full, environment_mass_inside_core=removed,
                environment_mass_outside_core=float(np.sum(exterior, dtype=np.float64) * de * de))
    return exterior, info


def interpolate_grid(image, source_x, target_x, tile: int = 128) -> np.ndarray:
    """Bilinear interpolation of a square [y, x] map onto another square grid."""
    interp = RegularGridInterpolator((source_x, source_x), image, bounds_error=True)
    out = np.empty((len(target_x), len(target_x)))
    for j in range(0, len(target_x), tile):
        yy, xx = np.meshgrid(target_x[j:j + tile], target_x, indexing="ij")
        out[j:j + tile] = interp((yy, xx))
    return out


def multiscale_fields(kappa_fine, kappa_environment, x_fine, x_environment, threads: int = 8,
                      names=("potential", "alpha_x", "alpha_y")):
    """Fields on the core grid from the core map plus the exterior of the environment map.

    Both maps are convergence on square grids centred on 0; ``x_*`` are pixel centres in
    arcsec.  Returns ``(fields, info)`` with fields on the core grid (``kappa`` included).
    """
    x_fine, _ = _uniform_axis(x_fine, "core axis")
    x_environment, _ = _uniform_axis(x_environment, "environment axis")
    kappa_fine = np.asarray(kappa_fine, dtype=np.float64)
    if kappa_fine.shape != (len(x_fine),) * 2 or not np.isfinite(kappa_fine).all() or np.any(kappa_fine < 0):
        raise ValueError("core map must be finite, non-negative, square and match its axis")
    exterior, info = partition_environment(kappa_environment, x_environment, x_fine)
    coarse = isolated_fields(exterior, x_environment[1] - x_environment[0], threads, names)
    del exterior
    fields = isolated_fields(kappa_fine, x_fine[1] - x_fine[0], threads, names)
    for name in names:
        fields[name] += interpolate_grid(coarse.pop(name), x_environment, x_fine)
    fields["kappa"] = kappa_fine.copy()
    return fields, info


def deflection_jacobian(alpha_x, alpha_y, x, y, step: float | None = None, chunk: int = 2_000_000):
    """Derivatives of the cubic-spline interpolants of the deflection maps at every pixel centre.

    The interpolant is the ``scipy.ndimage`` cubic B-spline through the pixel values
    (``spline_filter``, order 3).  Returns ``(dax_dx, dax_dy, day_dx, day_dy)`` as [y, x] arrays
    (``dax_dy`` = d alpha_x / d y).

    With ``step=None`` (default) the derivative of the spline is evaluated exactly at the knots:
    along the differentiated axis (c[k+1] - c[k-1]) / 2, along the other (c[k-1] + 4 c[k] + c[k+1]) / 6,
    with coefficients extended by their edge value.  With a ``step`` in arcsec it instead takes central
    differences of ``map_coordinates(..., mode='nearest')`` at +-step, as the old
    ``build_tngc014_mock.jacobian`` did (step 0.02 arcsec); the two agree to O(step^2).
    """
    x, dx = _uniform_axis(x, "x axis")
    y, dy = _uniform_axis(y, "y axis")
    coef = [ndimage.spline_filter(np.asarray(a, dtype=np.float64), order=3, output=np.float64)
            for a in (alpha_x, alpha_y)]
    if step is None:
        out = []
        for c in coef:
            c = np.pad(c, 1, mode="edge")
            smooth_y = (c[:-2] + 4 * c[1:-1] + c[2:]) / 6          # rows -> n, columns n + 2
            out.append((smooth_y[:, 2:] - smooth_y[:, :-2]) / (2 * dx))
            del smooth_y
            smooth_x = (c[:, :-2] + 4 * c[:, 1:-1] + c[:, 2:]) / 6  # rows n + 2, columns -> n
            out.append((smooth_x[2:] - smooth_x[:-2]) / (2 * dy))
            del smooth_x, c
        return tuple(out)

    shape = (len(y), len(x))
    size = shape[0] * shape[1]
    sx, sy = step / dx, step / dy  # step in pixels
    out = [np.empty(size) for _ in range(4)]

    def value(c, px, py):
        return ndimage.map_coordinates(c, [py, px], order=3, prefilter=False, mode="nearest")

    for begin in range(0, size, chunk):
        py, px = np.divmod(np.arange(begin, min(begin + chunk, size)), shape[1])
        px, py = px.astype(np.float64), py.astype(np.float64)
        cases = [(coef[0], sx, 0.0), (coef[0], 0.0, sy), (coef[1], sx, 0.0), (coef[1], 0.0, sy)]
        for k, (c, ox, oy) in enumerate(cases):
            out[k][begin:begin + len(px)] = (value(c, px + ox, py + oy) - value(c, px - ox, py - oy)) / (2 * step)
    return tuple(a.reshape(shape) for a in out)
