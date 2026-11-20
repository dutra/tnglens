"""Distances for the lens geometry.

The default cosmology is the one the simulation was run in (read from the snapshot header).
A config may replace it with ``{H0: ..., Om0: ...}`` (flat LCDM); that changes angular scales
and critical densities only, never particle masses or physical lengths.
"""
from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

ARCSEC_PER_RADIAN = 206264.80624709636
TNG = {"H0": 67.74, "Om0": 0.3089}  # Planck 2015, as used by IllustrisTNG / TNG-Cluster


def make(spec="native", header=None):
    """Return a FlatLambdaCDM.

    ``spec`` is "native" (use ``header`` with HubbleParam and Omega0 if given, else TNG) or a
    mapping with H0 [km/s/Mpc] and Om0.
    """
    if spec is None or spec == "native":
        if header is not None:
            return FlatLambdaCDM(H0=100.0 * float(header["HubbleParam"]), Om0=float(header["Omega0"]))
        return FlatLambdaCDM(**TNG)
    return FlatLambdaCDM(H0=float(spec["H0"]), Om0=float(spec["Om0"]))


def angular_diameter_distance_between(cosmo, z1, z2):
    """Angular diameter distance from z1 to z2.

    Recent astropy (8.0 here) takes ``angular_diameter_distance(z1, z2)`` and deprecates
    ``angular_diameter_distance_z1z2``; older versions only have the latter.
    """
    try:
        return cosmo.angular_diameter_distance(z1, z2)
    except TypeError:  # older astropy: one-argument form only
        return cosmo.angular_diameter_distance_z1z2(z1, z2)


def kpc_per_arcsec(cosmo, zl):
    """Physical kpc per arcsec at the lens."""
    return cosmo.angular_diameter_distance(zl).to_value(u.kpc) / ARCSEC_PER_RADIAN


def efficiency(cosmo, zl, zs):
    """D_ds / D_s; zero for sources at or in front of the lens. Accepts arrays of zs."""
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    out = np.zeros_like(zs)
    behind = zs > zl
    if behind.any():
        dds = angular_diameter_distance_between(cosmo, zl, zs[behind]).value
        out[behind] = dds / cosmo.angular_diameter_distance(zs[behind]).value
    return out if out.size > 1 else float(out[0])


def sigma_critical_inf(cosmo, zl):
    """Critical surface density for D_ds/D_s = 1 [Msun / pkpc^2]; divide by efficiency for a given zs."""
    dl = cosmo.angular_diameter_distance(zl)
    return (const.c**2 / (4 * np.pi * const.G) / dl).to(u.Msun / u.kpc**2).value


def time_delay_days(cosmo, zl):
    """Factor T such that dt [days] = T * (fermat potential difference [arcsec^2]),

    where the Fermat potential is 0.5*|theta - beta|^2 / s - psi_inf(theta) with s = D_ds/D_s and
    psi_inf the potential for D_ds/D_s = 1.  T = (1 + zl) D_l / c in days per arcsec^2 (the D_s/D_ds
    factor is carried by the 1/s and by psi_inf being normalised to s = 1).
    """
    dl = cosmo.angular_diameter_distance(zl)
    return ((1 + zl) * dl / const.c).to(u.day).value / ARCSEC_PER_RADIAN**2
