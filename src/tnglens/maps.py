"""Turning surface-density grids into the lens file ``lens.h5``.

All lensing fields are normalised to D_ds/D_s = 1 (multiply kappa, alpha, psi, gamma by
``efficiency(z_s)`` for a source at z_s).  The file is a truth product, so it is
self-consistent with the deflection map that image positions obey:

* ``kappa`` = deposited Sigma / Sigma_crit,inf;
* ``alpha_x``, ``alpha_y``, ``potential`` from the isolated multiscale solver;
* ``gamma1``, ``gamma2`` and every ``mu_zs*`` from the Jacobian of the *stored* (float32)
  deflection map, the exact derivative of its cubic B-spline -- not from the FFT shear
  kernels, which differ from the derivatives of alpha (for zoom 14 at snapshot 73: median
  ~2.5e-5, 99th percentile ~4e-4 inside the trusted region); small, but enough to move
  magnifications near critical curves.

The residual between kappa and half the divergence of alpha (inside the trusted region) is
stored as attributes; it measures how well the solver's alpha reproduces the deposited mass
on the pixel scale.
"""
from __future__ import annotations

import h5py
import numpy as np

from . import __version__
from .cache import to_json
from .fields import deflection_jacobian, multiscale_fields

H5 = dict(compression="gzip", compression_opts=4, shuffle=True)

# The parts of the run configuration a lens file depends on.  They are part of the lens-file
# signature and are stored as ``config_json``, so a reused lens file never carries a stale config;
# runtime settings (workers, threads, paths, halos list, catalog) are not.
LENS_CONFIG_KEYS = ("simulation", "cosmology", "source_redshifts", "particle_types", "projection")


def lens_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in LENS_CONFIG_KEYS}

DESCRIPTION = (
    "TNG-Cluster lens plane: {what}, projected along simulation axis {los} (map x, y = simulation "
    "{mx}, {my}). Single lens plane at the snapshot redshift; all FoF particles of the original zoom "
    "(types {types}) within +-{depth:g} R200c of the halo along the line of sight; no outer (non-FoF) "
    "mass and no line-of-sight structure. kappa, gamma1, gamma2, alpha_x, alpha_y [arcsec] and "
    "potential [arcsec^2] are for D_ds/D_s = 1: multiply by efficiency(z_s) = D_ds/D_s. "
    "Sigma is in Msun/pkpc^2. mu_zs<z> is the signed magnification for a source at z. Arrays are [y, x]."
)
COORDINATES = ("x, y: pixel centres in arcsec relative to the halo centre (GroupPos), ascending; "
               "x = (ra0 - ra) cos(dec0) 3600 and y = (dec - dec0) 3600 for a lens at (ra0, dec0) seen "
               "with map x along the first simulation axis of the projection name")
GAMMA_CONVENTION = "gamma1 = (psi_xx - psi_yy)/2, gamma2 = psi_xy in map (x, y)"
CONSISTENCY = (
    "kappa is the deposited density divided by Sigma_crit,inf. alpha and potential come from the isolated "
    "Green-function solver. gamma1, gamma2 and mu_zs* are computed from the Jacobian of the stored float32 "
    "alpha maps: the exact derivative, at the pixel centres, of the cubic B-spline through the alpha pixel "
    "values (scipy.ndimage.spline_filter, order 3), so they are exactly the mapping image positions obey; they are not the FFT shear kernels, which differ slightly "
    "(fft_shear_vs_jacobian_* attributes). "
    "kappa_vs_div_alpha_* attributes give |kappa - div(alpha)/2| inside the trusted region. "
    "The potential has an arbitrary additive constant (zero at the map centre)."
)


def lens_fields(sigma_fine, sigma_environment, x_fine_pkpc, x_environment_pkpc, kpc_per_arcsec: float,
                sigma_critical_inf: float, threads: int = 8) -> tuple[np.ndarray, dict, dict]:
    """kappa, alpha, potential, Jacobian-based shear and diagnostics on the core grid (D_ds/D_s = 1).

    Returns ``(x_arcsec, maps, info)``; ``maps`` arrays are float64 except ``alpha_*`` which are
    already rounded to float32 (the stored values everything else is derived from).
    """
    x = np.asarray(x_fine_pkpc, dtype=np.float64) / kpc_per_arcsec
    x_env = np.asarray(x_environment_pkpc, dtype=np.float64) / kpc_per_arcsec
    fields, info = multiscale_fields(np.asarray(sigma_fine, dtype=np.float64) / sigma_critical_inf,
                                     np.asarray(sigma_environment, dtype=np.float64) / sigma_critical_inf,
                                     x, x_env, threads,
                                     names=("potential", "alpha_x", "alpha_y", "gamma1", "gamma2"))
    to_msun = sigma_critical_inf * kpc_per_arcsec ** 2  # kappa * arcsec^2 -> Msun
    info = {"core_footprint_arcsec": info["core_footprint"], "trusted_interior_arcsec": info["trusted_interior"],
            "guard_arcsec": info["guard"], "environment_cell_arcsec": info["environment_cell"],
            "environment_mass_Msun": info["environment_mass"] * to_msun,
            "environment_mass_inside_core_Msun": info["environment_mass_inside_core"] * to_msun,
            "environment_mass_outside_core_Msun": info["environment_mass_outside_core"] * to_msun}
    # The potential's zero is arbitrary; put it at the map centre so float32 keeps its precision
    # (the solver's raw values are ~1e5 arcsec^2 for a massive cluster, where one float32 ulp is ~1e-2).
    n = len(x)
    fields["potential"] -= fields["potential"][n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].mean()
    alpha_x = fields.pop("alpha_x").astype(np.float32)
    alpha_y = fields.pop("alpha_y").astype(np.float32)
    axx, axy, ayx, ayy = deflection_jacobian(alpha_x, alpha_y, x, x)
    maps = {"kappa": fields["kappa"], "alpha_x": alpha_x, "alpha_y": alpha_y, "potential": fields["potential"],
            "gamma1": 0.5 * (axx - ayy), "gamma2": 0.5 * (axy + ayx),
            "jacobian": (axx, axy, ayx, ayy)}

    lo, hi = info["trusted_interior_arcsec"]
    inner = (x >= lo) & (x <= hi)
    box = np.ix_(inner, inner)
    residual = np.abs(fields["kappa"] - 0.5 * (axx + ayy))[box]
    shear_fft = np.hypot(fields["gamma1"] - maps["gamma1"], fields["gamma2"] - maps["gamma2"])[box]
    info.update(
        trusted_half_width_arcsec=float(min(-lo, hi)),
        kappa_vs_div_alpha_median=float(np.median(residual)),
        kappa_vs_div_alpha_p99=float(np.percentile(residual, 99)),
        kappa_vs_div_alpha_max=float(residual.max()),
        fft_shear_vs_jacobian_median=float(np.median(shear_fft)),
        fft_shear_vs_jacobian_p99=float(np.percentile(shear_fft, 99)),
        mass_core_Msun=float(np.sum(sigma_fine, dtype=np.float64) * (x_fine_pkpc[1] - x_fine_pkpc[0]) ** 2),
    )
    return x, maps, info


def magnification(jacobian, s: float) -> np.ndarray:
    """Signed magnification 1/det(A) for D_ds/D_s = s from the D_ds/D_s = 1 deflection Jacobian."""
    axx, axy, ayx, ayy = jacobian
    det = (1 - s * axx) * (1 - s * ayy) - s * s * axy * ayx
    with np.errstate(divide="ignore"):
        return 1 / det


def write_lens_file(f: h5py.File, x, maps: dict, sigma_fine, efficiencies: dict, attrs: dict) -> None:
    """Write datasets and attributes into an open HDF5 file (see the README for the contract)."""
    f.create_dataset("x", data=np.asarray(x, dtype=np.float64))
    f.create_dataset("y", data=np.asarray(x, dtype=np.float64))
    for name in ("kappa", "alpha_x", "alpha_y", "potential", "gamma1", "gamma2"):
        f.create_dataset(name, data=np.asarray(maps[name], dtype=np.float32), **H5)
    f.create_dataset("Sigma", data=np.asarray(sigma_fine, dtype=np.float32), **H5)
    f["Sigma"].attrs["unit"] = "Msun / pkpc^2"
    f["alpha_x"].attrs["unit"] = f["alpha_y"].attrs["unit"] = "arcsec"
    f["potential"].attrs["unit"] = "arcsec^2"
    for zs, s in efficiencies.items():
        d = f.create_dataset(f"mu_zs{zs:g}", data=magnification(maps["jacobian"], s).astype(np.float32), **H5)
        d.attrs.update(source_redshift=float(zs), efficiency=float(s),
                       definition="signed magnification 1/det(A) from the Jacobian of the stored deflection map")
    f.attrs.update({k: (to_json(v) if isinstance(v, (dict, list)) and k.endswith("_json") else v)
                    for k, v in attrs.items()})


def lens_attributes(halo, axis: str, cosmology: dict, cfg: dict, info: dict, ledger: list) -> dict:
    """Attribute dictionary of the lens file."""
    from .projection import AXES
    names = "xyz"
    ix, iy, iz = AXES[axis]
    p = cfg["projection"]
    what = f"zoom {halo.zoom}, snapshot {halo.snapshot}, halo {halo.halo_id}" + ("" if halo.primary else " (not the zoom's primary)")
    return {
        "description": DESCRIPTION.format(what=what, los=names[iz], mx=names[ix], my=names[iy],
                                          types=", ".join(map(str, cfg["particle_types"])), depth=p["depth_half_r200"]),
        "coordinates": COORDINATES,
        "gamma_convention": GAMMA_CONVENTION,
        "consistency": CONSISTENCY,
        "lens_redshift": halo.redshift,
        "snapshot": halo.snapshot,
        "zoom_index": halo.zoom,
        "halo_id": halo.halo_id,
        "primary_target": halo.primary,
        "axis": axis,
        "line_of_sight_axis": names[iz],
        "H0": cosmology["H0"],
        "Om0": cosmology["Om0"],
        "cosmology_source": cosmology["source"],
        "kpc_per_arcsec": cosmology["kpc_per_arcsec"],
        "sigma_critical_inf_Msun_pkpc2": cosmology["sigma_critical_inf"],
        "M200c_Msun": halo.M200c_Msun,
        "R200c_pkpc": halo.R200c_pkpc,
        "halo_position_ckpc_h": np.asarray(halo.position_ckpc_h, dtype=np.float64),
        "box_ckpc_h": halo.box_ckpc_h,
        "h": halo.h,
        "scale_factor": halo.scale_factor,
        "group_catalog_base": halo.base,
        "group_file": halo.group_file,
        "snapshot_file": halo.snapshot_file,
        "particle_types": np.asarray(cfg["particle_types"], dtype=np.int64),
        "particle_scope": "all FoF particles of the original zoom (no outer fuzz), centres within "
                          f"+-{p['depth_half_r200']:g} R200c along the line of sight",
        "source_redshifts": np.asarray(cfg["source_redshifts"], dtype=np.float64),
        "trusted_half_width_arcsec": info["trusted_half_width_arcsec"],
        "core_half_width_arcsec": float(info["core_footprint_arcsec"][1]),
        "support_floor_pkpc": p["support_floor_pkpc"],
        "neighbors": p["neighbors"],
        "gas_support": p["gas_support"],
        "jacobian": "analytic derivative of the order-3 B-spline of alpha at the pixel centres",
        "kappa_vs_div_alpha_median": info["kappa_vs_div_alpha_median"],
        "kappa_vs_div_alpha_p99": info["kappa_vs_div_alpha_p99"],
        "kappa_vs_div_alpha_max": info["kappa_vs_div_alpha_max"],
        "fft_shear_vs_jacobian_median": info["fft_shear_vs_jacobian_median"],
        "fft_shear_vs_jacobian_p99": info["fft_shear_vs_jacobian_p99"],
        "potential_zero": "arbitrary additive constant, chosen so the mean of the four central pixels is 0",
        "mass_ledger_json": ledger,
        "solver_json": {k: v for k, v in info.items()},
        "config_json": lens_config(cfg),
        "tnglens_version": __version__,
    }
