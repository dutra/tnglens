"""Multiple-image and member catalogs from a lens file: truth tables and fit-ready (kappaccino) YAML.

``build_catalog(lens_path, cfg, outdir)`` draws candidate sources (``tnglens.sources``), finds all their images
(``tnglens.images``), keeps the first ``n_systems`` candidates with at least two observable images, builds the
member table (``tnglens.members``) and writes::

    outdir/truth/images_truth.yaml   every image in the search square, exact positions, true redshifts
    outdir/truth/images_truth.csv    one row per image
    outdir/truth/sources_truth.csv   one row per system
    outdir/truth/candidates.csv      every candidate drawn and why it was kept or rejected
    outdir/truth/members_truth.csv   Subfind properties of the members
    outdir/truth/selection.yaml      the resolved configuration and the selection counts
    outdir/observed/images.yaml      noisy positions of the observable images (lenscluster-images/1)
    outdir/observed/members.yaml     members (band, ra, dec, mag, shape, role)
    outdir/README.md

Every file is a function of the lens file, the configuration and the seed only (no dates, no timings, and the
lens file is recorded by its name and identifying attributes, not by the path it was given as), and the result
does not depend on ``workers``.  Run standalone with ``python -m tnglens.catalog LENS.h5 --config CFG.yaml
--out DIR``; the configuration is the ``catalog:`` section of the package YAML (see docs/catalog.md).

Three parts reproduce the tngc014 mock and are on by default (see docs/catalog.md):

``quotas``     keep a set number of systems per image-family class (wide, intermediate, cusp, fold by pair
               separation, subhalo) instead of the first ``n_systems`` candidates;
``subhalos``   the counterfactual of ``tnglens.satellites``: an image that vanishes or moves when one satellite is
               removed is tied to it, and a system with a tied image is of class ``subhalo`` and named S<n>;
``knots``      extra point sources 0.03-0.15 arcsec from the source of the most magnified systems, catalogued as
               knots A, B, C of the same images.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from . import __version__, cosmology, members as members_mod, satellites as satellites_mod, sources as sources_mod
from .images import ImageFinder, classify, image_table, observable, separations
from .lensmap import EDGE_PIXELS, LensMap, map_extent, read_meta, require

DEFAULTS = dict(
    seed=20261111,
    n_systems=None,                # systems to keep when quotas is null (the first that pass the cuts)
    max_candidates=20000,          # give up after drawing this many
    batch=64,                      # candidates evaluated per round (does not change the result)
    workers=1,                     # processes; >1 uses a spawn-context pool, each worker loads the lens file
    search_half="auto",            # image search square half-width [arcsec]; auto = trusted half-width, else 110
                                   # clipped to the map
    finder_stride=2,               # finder mesh step in map pixels
    overrides={},                  # lens-file attributes to supply or replace
    sources=dict(sources_mod.DEFAULTS, seed_radius=[5.0, "auto"],     # auto = 95/110 of the search half-width
                 # the pool of the tngc014 mock: 500 general, 600 near-critical and 600 near-satellite seeds
                 seed_kinds=dict(negative=0.30, critical=0.35, source=0.0, subhalo=0.35),
                 duplicate_dz=0.05, duplicate_dbeta=1.0),             # reject a source this close to a kept one
    observability=dict(mu_min=0.25, bcg_radius=3.0, min_separation=0.3, min_images=2),
    noise=dict(sigma_pos=0.03, photoz_fraction=0.12, photoz_sigma=0.06, photoz_zmin=0.8,
               source_mag=[26.0, 29.0]),
    members=dict(members_mod.DEFAULTS),
    # systems to keep per class (CLASSES), the numbers of the tngc014 mock; null keeps the first n_systems instead
    quotas=dict(wide=22, intermediate=14, cusp=8, fold=15, subhalo=10),
    selection=dict(fold_bins=[0.3, 0.7, 1.5, 3.0],    # fold quota split evenly over these pair separations [arcsec]
                   wide_pair_keep=0.3,                # fraction kept of wide systems with only two observable images
                   max_per_satellite=2),              # subhalo systems tied (first) to the same satellite
    subhalos=dict(satellites_mod.DEFAULTS),
    knots=dict(n_systems=12,                          # systems that get extra knots, most magnified first
               extra=[1, 2],                          # knots added per system: uniform integer in this range
               offset=[0.03, 0.15],                   # distance of a knot from the source centre [arcsec]
               tries=6,                               # positions tried per knot
               min_sep=0.08, max_sep=4.0,             # allowed image-plane separation from the same image of knot A
               mag_offset=1.0),                       # a knot is this much fainter than the source
)
CLASSES = ("wide", "intermediate", "cusp", "fold", "subhalo", "other")
KNOT_NAMES = "ABCDEFGH"

# Candidate outcomes, in the order they are tested.  close_pair: two observable images closer than
# min_separation at their true positions; close_pair_observed: the same after the position noise.
STATUS = ("accepted", "foreground", "no_seed", "single", "lt2_observable", "close_pair", "close_pair_observed",
          "duplicate", "no_quota", "quota_full", "thinned")
# With quotas: no_quota (its class has no quota, or a fold pair outside the bins), quota_full (the quota of its
# class, fold bin or satellite is already filled), thinned (a wide pair dropped at random, see wide_pair_keep).
PHOTOZ_LENS_GAP = 0.01            # a catalogued (photometric) redshift is at least z_lens + this
FALLBACK_SEARCH_HALF = 110.0      # search half-width without a trusted half-width [arcsec], clipped to the map


def resolve_config(cfg: dict | None) -> dict:
    """``DEFAULTS`` updated by ``cfg`` (nested dicts merged one level deep), validated.

    Raises ValueError naming the key on an unknown key or a bad value; needs no lens file (the checks that need
    the lens redshift or the map are made by ``build_catalog``).
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"the catalog section must be a mapping, not {type(cfg).__name__}")
    out = copy.deepcopy(DEFAULTS)
    if cfg.get("n_systems") is not None and "quotas" not in cfg:
        out["quotas"] = None                 # asking for a number of systems means: the first that pass the cuts
    for k, v in cfg.items():
        if k not in out:
            raise ValueError(f"unknown catalog key {k!r}; known {sorted(out)}")
        if isinstance(out[k], dict) and k not in ("overrides", "quotas"):
            if not isinstance(v, dict):
                raise ValueError(f"catalog.{k} must be a mapping, not {v!r}")
            unknown = set(v) - set(out[k])
            if unknown:
                raise ValueError(f"unknown keys {sorted(unknown)} in catalog.{k}; known {sorted(out[k])}")
            out[k].update(copy.deepcopy(v))
        else:
            out[k] = copy.deepcopy(v)
    _validate(out)
    return out


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and math.isfinite(float(v))


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _validate(c):
    """ValueError for the first bad value of a resolved configuration."""
    def bad(key, why):
        raise ValueError(f"catalog.{key}: {why}")

    def num(key, v, lo=None, strict=False):
        if not _is_num(v):
            bad(key, f"must be a number, not {v!r}")
        if lo is not None and (v <= lo if strict else v < lo):
            bad(key, f"must be {'>' if strict else '>='} {lo}, not {v!r}")

    def integer(key, v, lo):
        if not _is_int(v) or v < lo:
            bad(key, f"must be an integer >= {lo}, not {v!r}")

    def numlist(key, v, n=None, lo=None, strict=False):
        if not isinstance(v, (list, tuple)) or not v or (n is not None and len(v) != n):
            bad(key, f"must be a list of {n or 'one or more'} numbers, not {v!r}")
        for x in v:
            num(key, x, lo, strict)

    if not _is_int(c["seed"]):
        bad("seed", f"must be an integer, not {c['seed']!r}")
    for k in ("max_candidates", "batch", "workers", "finder_stride"):
        integer(k, c[k], 1)
    if c["n_systems"] is not None:
        integer("n_systems", c["n_systems"], 1)
        if c["quotas"] is not None:
            bad("n_systems", f"is {c['n_systems']!r}, but the systems are kept by class (quotas); set quotas: null to "
                             "keep the first n_systems candidates, or change the quotas")
    elif c["quotas"] is None:
        bad("n_systems", "is needed when quotas is null")
    if c["search_half"] not in (None, "auto"):
        num("search_half", c["search_half"], 0, strict=True)
    if not isinstance(c["overrides"], dict):
        bad("overrides", f"must be a mapping of lens-file attributes, not {c['overrides']!r}")
    sc = c["sources"]
    z = sc["redshift"]
    if not isinstance(z, dict) or set(z) - {"edges", "weights", "values"}:
        bad("sources.redshift", f"must be {{edges, weights}} or {{values[, weights]}}, not {z!r}")
    if z.get("values") is not None:
        numlist("sources.redshift.values", z["values"], lo=0, strict=True)
        if z.get("weights") is not None:
            numlist("sources.redshift.weights", z["weights"], n=len(z["values"]), lo=0)
            if sum(z["weights"]) <= 0:
                bad("sources.redshift.weights", "need a positive sum")
    else:
        e, w = z.get("edges"), z.get("weights")
        numlist("sources.redshift.edges", e, lo=0)
        if len(e) < 2 or any(b <= a for a, b in zip(e, e[1:])):
            bad("sources.redshift.edges", f"must be two or more increasing redshifts, not {e!r}")
        numlist("sources.redshift.weights", w, n=len(e) - 1, lo=0)
        if sum(w) <= 0:
            bad("sources.redshift.weights", "need a positive sum")
    k = sc["seed_kinds"]
    if not isinstance(k, dict) or set(k) - set(sources_mod.KINDS):
        bad("sources.seed_kinds", f"must map kinds among {sorted(sources_mod.KINDS)} to weights, not {k!r}")
    for name, w in k.items():
        num(f"sources.seed_kinds.{name}", w, 0)
    sub = c["subhalos"]
    if sub["enabled"] not in (True, False, "auto"):
        bad("subhalos.enabled", f"must be true, false or auto, not {sub['enabled']!r}")
    for key in ("min_mass", "shift_tol", "radius_einstein", "match_radius", "patch_half", "patch_keep", "seed_min_mass",
                "seed_radius"):
        num(f"subhalos.{key}", sub[key], 0, strict=True)
    for key in ("patch_sigma_pixels", "seed_min_centre"):
        num(f"subhalos.{key}", sub[key], 0)
    numlist("subhalos.radius_clip", sub["radius_clip"], n=2, lo=0)
    if sub["radius_clip"][1] < sub["radius_clip"][0]:
        bad("subhalos.radius_clip", f"must be [low, high], not {sub['radius_clip']!r}")
    integer("subhalos.patch_pixels", sub["patch_pixels"], 16)
    integer("subhalos.max_cached", sub["max_cached"], 1)
    if sub["patch_keep"] >= sub["patch_half"]:
        bad("subhalos.patch_keep", f"must be smaller than patch_half {sub['patch_half']!r}, not {sub['patch_keep']!r}")
    quotas = c["quotas"]
    if quotas is not None:
        if not isinstance(quotas, dict) or not quotas or set(quotas) - set(CLASSES):
            bad("quotas", f"must map classes among {list(CLASSES)} to numbers of systems, not {quotas!r}")
        for name, n_cls in quotas.items():
            integer(f"quotas.{name}", n_cls, 0)
        if sum(quotas.values()) < 1:
            bad("quotas", "needs at least one system")
    if sub["enabled"] is False and (float(k.get("subhalo", 0)) > 0 or (quotas or {}).get("subhalo", 0) > 0):
        bad("subhalos.enabled", "is false, but sources.seed_kinds.subhalo or quotas.subhalo asks for satellites")
    sel = c["selection"]
    numlist("selection.fold_bins", sel["fold_bins"], lo=0)
    if len(sel["fold_bins"]) < 2 or any(b <= a for a, b in zip(sel["fold_bins"], sel["fold_bins"][1:])):
        bad("selection.fold_bins", f"must be two or more increasing separations, not {sel['fold_bins']!r}")
    num("selection.wide_pair_keep", sel["wide_pair_keep"], 0)
    if sel["wide_pair_keep"] > 1:
        bad("selection.wide_pair_keep", f"must be within [0, 1], not {sel['wide_pair_keep']!r}")
    integer("selection.max_per_satellite", sel["max_per_satellite"], 1)
    kn = c["knots"]
    integer("knots.n_systems", kn["n_systems"], 0)
    integer("knots.tries", kn["tries"], 1)
    for key in ("extra", "offset"):
        numlist(f"knots.{key}", kn[key], n=2, lo=0)
        if kn[key][1] < kn[key][0]:
            bad(f"knots.{key}", f"must be [low, high], not {kn[key]!r}")
    if not all(_is_int(v) for v in kn["extra"]) or kn["extra"][0] < 1 or kn["extra"][1] > len(KNOT_NAMES) - 1:
        bad("knots.extra", f"must be two integers between 1 and {len(KNOT_NAMES) - 1}, not {kn['extra']!r}")
    for key in ("min_sep", "max_sep", "mag_offset"):
        num(f"knots.{key}", kn[key], 0)
    if not any(float(w) > 0 for w in k.values()):
        bad("sources.seed_kinds", "needs at least one positive weight")
    r = sc["seed_radius"]
    if not isinstance(r, (list, tuple)) or len(r) != 2:
        bad("sources.seed_radius", f"must be [inner, outer or auto], not {r!r}")
    num("sources.seed_radius", r[0], 0)
    if r[1] not in (None, "auto"):
        num("sources.seed_radius", r[1], r[0], strict=True)
    numlist("sources.critical_det", sc["critical_det"], lo=0, strict=True)
    num("sources.duplicate_dz", sc["duplicate_dz"], 0)
    num("sources.duplicate_dbeta", sc["duplicate_dbeta"], 0)
    o = c["observability"]
    for key in ("mu_min", "bcg_radius", "min_separation"):
        num(f"observability.{key}", o[key], 0)
    integer("observability.min_images", o["min_images"], 1)
    n = c["noise"]
    num("noise.sigma_pos", n["sigma_pos"], 0)
    num("noise.photoz_fraction", n["photoz_fraction"], 0)
    if n["photoz_fraction"] > 1:
        bad("noise.photoz_fraction", f"must be within [0, 1], not {n['photoz_fraction']!r}")
    num("noise.photoz_sigma", n["photoz_sigma"], 0)
    num("noise.photoz_zmin", n["photoz_zmin"], 0)
    numlist("noise.source_mag", n["source_mag"], n=2)
    if n["source_mag"][1] < n["source_mag"][0]:
        bad("noise.source_mag", f"must be [bright, faint], not {n['source_mag']!r}")
    m = c["members"]
    if m["enabled"] not in (True, False, "auto"):
        bad("members.enabled", f"must be true, false or auto, not {m['enabled']!r}")
    num("members.mag_limit", m["mag_limit"])
    num("members.radius", m["radius"], 0, strict=True)
    num("members.los_r200c", m["los_r200c"], 0, strict=True)
    if not _is_int(m["band_index"]) or not 0 <= m["band_index"] <= 7:
        bad("members.band_index", f"must be an integer 0..7 (U B V K g r i z), not {m['band_index']!r}")
    if not isinstance(m["band_name"], str) or not m["band_name"]:
        bad("members.band_name", f"must be a non-empty string, not {m['band_name']!r}")
    if m["shapes"] not in (None, False) and not isinstance(m["shapes"], str):
        bad("members.shapes", f"must be auto, snapshot, none or a directory, not {m['shapes']!r}")
    integer("members.shape_min_particles", m["shape_min_particles"], 0)
    for key in ("group_file", "offsets_file", "snapshot_file"):
        if m[key] is not None and not isinstance(m[key], str):
            bad(f"members.{key}", f"must be a path or null, not {m[key]!r}")


def _search_half(cfg, meta, extent, dx):
    """(search half-width, note or None); ``extent`` and ``dx`` from ``lensmap.map_extent``."""
    trusted = meta.get("trusted_half_width_arcsec")
    half = cfg["search_half"]
    if half in (None, "auto"):
        if trusted:
            return float(trusted), None
        # keep the finder's local searches (3 mesh steps) and the spline's boundary zone inside the map
        limit = extent - (EDGE_PIXELS + 3 * int(cfg["finder_stride"])) * dx
        if FALLBACK_SEARCH_HALF > limit:
            return round(limit, 3), (f"search_half auto: no trusted half-width in the lens file and the map is "
                                     f"smaller than {FALLBACK_SEARCH_HALF:g} arcsec; using {round(limit, 3):g} "
                                     f"(the map half-width {extent:g} minus {EDGE_PIXELS + 3 * int(cfg['finder_stride'])}"
                                     " pixels)")
        return FALLBACK_SEARCH_HALF, None
    half = float(half)
    if trusted and half > trusted:
        return float(trusted), f"search_half {half} reduced to the trusted half-width {trusted}"
    return half, None


# ----------------------------------------------------------------------------- candidate evaluation
class Evaluator:
    """Everything needed to evaluate candidate ``index``: lens, finder, sampler and cuts."""

    def __init__(self, lens_path, cfg, half, centre):
        self.cfg = cfg
        self.lens = LensMap.from_file(lens_path, cfg["overrides"], half=half)
        self.finder = ImageFinder(self.lens, half, cfg["finder_stride"])
        scfg = {k: cfg["sources"][k] for k in sources_mod.DEFAULTS}
        self.sampler = sources_mod.SourceSampler(self.finder, scfg, cfg["seed"])
        self.centre = tuple(centre)
        self.sats = None
        if subhalos_on(cfg):
            self.sats = satellites_mod.Satellites.from_lens(self.lens, self.finder, cfg["subhalos"], cfg["members"])
            self.sampler.subhalo_mask = self.sats.seed_mask(*self.finder.mesh)

    def __call__(self, index):
        o = self.cfg["observability"]
        c = self.sampler.draw(index)
        rec = dict(index=c.index, kind=c.kind, z=c.z, s=c.s, beta=c.beta, seed=c.seed, n_images=0, n_obs=0)
        if c.outcome != "ok":
            return dict(rec, status=c.outcome)
        theta = self.finder.find(c.beta, c.s)
        rec["n_images"] = len(theta)
        if len(theta) < 2:
            return dict(rec, status="single")
        tab = image_table(self.lens, theta, c.beta, c.s)
        obs = observable(tab, self.centre, o["bcg_radius"], o["mu_min"])
        in_bcg = np.hypot(theta[:, 0] - self.centre[0], theta[:, 1] - self.centre[1]) <= o["bcg_radius"]
        rec.update(tab=tab, obs=obs, in_bcg=in_bcg, n_obs=int(obs.sum()))
        if obs.sum() < max(2, int(o["min_images"])):
            return dict(rec, status="lt2_observable")
        min_sep, max_sep = separations(theta[obs])
        rec.update(min_sep=min_sep, max_sep=max_sep, cls=classify(theta[obs], tab["parity"][obs]))
        if min_sep < o["min_separation"]:
            return dict(rec, status="close_pair")
        # the noise is a function of (seed, candidate index) only, so the cut on the noisy positions does not
        # depend on the workers or on which other candidates were kept
        order = label_order(tab["mu"], obs)
        rec.update(order=order, **draw_noise(self.cfg, c.index, len(theta), c.z, self.lens.zl))
        noisy = theta.copy()
        noisy[order] += rec.pop("dpos")
        rec.update(noisy=noisy, min_sep_obs=separations(noisy[obs])[0])
        if rec["min_sep_obs"] < o["min_separation"]:
            return dict(rec, status="close_pair_observed")
        # which satellites hold its images in place; a system with a tied image is of class subhalo
        rec["cf"] = self.sats.counterfactual(c.s, tab, obs) if self.sats is not None else []
        rec["geometry"] = rec["cls"]
        if rec["cf"]:
            rec["cls"] = "subhalo"
        rec["wide_draw"] = float(np.random.default_rng([int(self.cfg["seed"]), 3, int(c.index)]).random())
        return dict(rec, status="accepted")


def subhalos_on(cfg) -> bool:
    """Whether the satellites are loaded: asked for, or needed by a subhalo quota or the subhalo seeding kind."""
    mode = cfg["subhalos"]["enabled"]
    if mode in (True, False):
        return mode
    return (cfg["quotas"] or {}).get("subhalo", 0) > 0 or float(cfg["sources"]["seed_kinds"].get("subhalo", 0)) > 0


def label_order(mu, obs):
    """Image indices in label order (a, b, ...): observable images first, then brightest first."""
    return np.lexsort((-np.abs(mu), ~np.asarray(obs, bool)))


def draw_noise(cfg, index, n_images, z, zl):
    """The observational noise of candidate ``index``: a function of the seed and the index only.

    Position offsets ``dpos`` (n_images, 2) in label order come from ``default_rng([seed, 1, index])``; the
    source magnitude and the photometric redshift from ``default_rng([seed, 2, index])``.  A photometric
    redshift scatters about the true one by photoz_sigma (1 + z_true), is at least max(photoz_zmin,
    z_lens + PHOTOZ_LENS_GAP), and is reported with z_err = photoz_sigma (1 + z_catalog).
    """
    nz, seed = cfg["noise"], int(cfg["seed"])
    dpos = np.random.default_rng([seed, 1, int(index)]).normal(0.0, nz["sigma_pos"], (int(n_images), 2))
    rng = np.random.default_rng([seed, 2, int(index)])
    m_src = float(rng.uniform(*nz["source_mag"]))
    photoz = bool(rng.random() < nz["photoz_fraction"])
    g = float(rng.normal())                       # drawn for every system, used only for a photometric z
    if photoz:
        zfloor = max(float(nz["photoz_zmin"]), zl + PHOTOZ_LENS_GAP)
        z_cat = round(max(zfloor, z + g * nz["photoz_sigma"] * (1 + z)), 3)
        z_cat = max(z_cat, math.ceil(zfloor * 1000) / 1000)       # rounding never takes it below the floor
        z_err = round(nz["photoz_sigma"] * (1 + z_cat), 3)
    else:
        z_cat, z_err = z, None
    return dict(dpos=dpos, m_src=m_src, photoz=photoz, z_cat=z_cat, z_err=z_err)


_WORKER = None


def _init_worker(lens_path, cfg, half, centre):
    global _WORKER
    _WORKER = Evaluator(lens_path, cfg, half, centre)


def _work(index):
    return _WORKER(index)


class Quotas:
    """Systems still wanted per class; ``offer`` says what happens to a candidate, ``take`` counts it.

    The fold quota is split evenly (rounded up) over the bins of ``selection.fold_bins``.  A candidate with a tied
    image is of class subhalo only: when that quota (or its satellite's share) is full it is not kept under its
    geometric class, as in the tngc014 mock.
    """

    def __init__(self, quotas, sel):
        self.sel = sel
        self.left = {k: int(quotas.get(k, 0)) for k in CLASSES if k != "fold"}
        self.edges = np.asarray(sel["fold_bins"], dtype=np.float64)
        nbins = len(self.edges) - 1
        self.fold_left = [math.ceil(int(quotas.get("fold", 0)) / nbins)] * nbins
        self.wanted = dict(self.left, fold=sum(self.fold_left))
        self.per_satellite = {}

    def full(self):
        return sum(self.left.values()) + sum(self.fold_left) <= 0

    def _fold_bin(self, r):
        b = int(np.searchsorted(self.edges, r["min_sep"], side="right")) - 1
        return b if 0 <= b < len(self.fold_left) else None

    def offer(self, r):
        cls = r["cls"]
        if cls == "subhalo":
            if not self.wanted["subhalo"]:
                return "no_quota"
            k = r["cf"][0]["sub"]
            if self.left["subhalo"] <= 0 or self.per_satellite.get(k, 0) >= self.sel["max_per_satellite"]:
                return "quota_full"
            return "accepted"
        if cls == "fold":
            b = self._fold_bin(r)
            if b is None or not self.wanted["fold"]:
                return "no_quota"
            return "accepted" if self.fold_left[b] > 0 else "quota_full"
        if not self.wanted.get(cls, 0):
            return "no_quota"
        if self.left[cls] <= 0:
            return "quota_full"
        if cls == "wide" and r["n_obs"] < 3 and r["wide_draw"] >= self.sel["wide_pair_keep"]:
            return "thinned"
        return "accepted"

    def take(self, r):
        cls = r["cls"]
        if cls == "fold":
            self.fold_left[self._fold_bin(r)] -= 1
            return
        self.left[cls] -= 1
        if cls == "subhalo":
            k = r["cf"][0]["sub"]
            self.per_satellite[k] = self.per_satellite.get(k, 0) + 1

    def missing(self):
        out = {k: v for k, v in self.left.items() if v > 0}
        if sum(self.fold_left) > 0:
            out["fold"] = sum(self.fold_left)
        return out


def _select(evaluate_batch, cfg):
    """Evaluate candidates in index order until the wanted systems are accepted.

    Without quotas the first ``n_systems`` accepted candidates are kept; with quotas each candidate is kept if
    its class still has room (``Quotas``).  Either way the outcome depends only on the candidates in index order,
    not on ``batch`` or ``workers``.  Returns (accepted, all records, the Quotas or None).
    """
    n_want, nmax, batch = int(cfg["n_systems"] or 0), int(cfg["max_candidates"]), int(cfg["batch"])
    dz, db = cfg["sources"]["duplicate_dz"], cfg["sources"]["duplicate_dbeta"]
    quotas = Quotas(cfg["quotas"], cfg["selection"]) if cfg["quotas"] is not None else None
    done = (lambda: quotas.full()) if quotas is not None else (lambda: len(accepted) >= n_want)
    accepted, records, start = [], [], 0
    while not done() and start < nmax:
        for r in evaluate_batch(range(start, min(start + batch, nmax))):
            if done():
                break
            if r["status"] == "accepted" and quotas is not None:
                r["status"] = quotas.offer(r)
            if r["status"] == "accepted" and any(abs(r["z"] - a["z"]) <= dz and np.hypot(*(r["beta"] - a["beta"])) <= db
                                                 for a in accepted):
                r["status"] = "duplicate"
            records.append(r)
            if r["status"] == "accepted":
                accepted.append(r)
                if quotas is not None:
                    quotas.take(r)
        start += batch
    return accepted, records, quotas


# ----------------------------------------------------------------------------- knots
def add_knots(accepted, lens, finder, cfg):
    """Extra point sources close to the source of the most magnified systems; returns how many systems got one.

    Systems are tried in order of their largest |mu|.  Each gets ``extra`` knots (B, C, ...): a source
    ``offset`` arcsec from the source centre in a random direction, kept when it has as many images as the
    source, each of the same parity and, for the observable ones, between ``min_sep`` and ``max_sep`` from the
    matching image of the source (knot A).  The draws of candidate i come from ``default_rng([seed, 4, i])`` and
    the position noise of its knot j from ``default_rng([seed, 5, i, j])``.
    """
    kc, seed = cfg["knots"], int(cfg["seed"])
    if kc["n_systems"] <= 0:
        return 0
    from scipy.optimize import linear_sum_assignment
    done = 0
    for c in sorted(accepted, key=lambda c: (-float(np.max(np.abs(c["tab"]["mu"]))), c["index"])):
        if done >= kc["n_systems"]:
            break
        rng = np.random.default_rng([seed, 4, int(c["index"])])
        theta, obs, parity = c["tab"]["theta"], c["obs"], c["tab"]["parity"]
        knots = {}
        for j, name in enumerate(KNOT_NAMES[1:1 + int(rng.integers(kc["extra"][0], kc["extra"][1] + 1))], start=1):
            for _ in range(int(kc["tries"])):
                off, ang = rng.uniform(*kc["offset"]), rng.uniform(0, 2 * np.pi)
                beta = np.asarray(c["beta"]) + off * np.array([np.cos(ang), np.sin(ang)])
                new = finder.find(beta, c["s"])
                if len(new) != len(theta):
                    continue
                d = np.linalg.norm(theta[:, None] - new[None], axis=2)
                rows, cols = linear_sum_assignment(d)
                sep = d[rows, cols]
                tab = image_table(lens, new[cols], beta, c["s"])
                if sep[obs].max() < kc["max_sep"] and sep[obs].min() > kc["min_sep"] and np.all(tab["parity"] == parity):
                    dpos = np.random.default_rng([seed, 5, int(c["index"]), j]).normal(0.0, cfg["noise"]["sigma_pos"],
                                                                                    (len(theta), 2))
                    knots[name] = dict(beta=beta, tab=tab, noisy=tab["theta"] + dpos)
                    break
        if knots:
            c["knots"] = knots
            done += 1
    return done


# ----------------------------------------------------------------------------- assembly
def _labels(n):
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for i in range(n):
        s, i1 = "", i
        while True:
            s = letters[i1 % 26] + s
            i1 = i1 // 26 - 1
            if i1 < 0:
                break
        out.append(s)
    return out


def _assemble(accepted, member_ids=(), knot_mag_offset=1.0):
    """Systems (truth and observed) and the per-image truth rows.

    Systems are named 1, 2, ... by true redshift; those of class subhalo are named S1, S2, ... instead.  A system
    with knots has one position per knot in every image (``pos`` is then a dict by knot name) and one truth row
    per image and knot.
    """
    member_ids = {str(m) for m in member_ids}
    systems, image_rows, source_rows = [], [], []
    n_regular = n_sub = 0
    for c in sorted(accepted, key=lambda c: (c["z"], c["index"])):
        if c["cls"] == "subhalo":
            n_sub += 1
            name = f"S{n_sub}"
        else:
            n_regular += 1
            name = str(n_regular)
        tab, obs, in_bcg, order = c["tab"], c["obs"], c["in_bcg"], c["order"]
        mu, par = tab["mu"], tab["parity"]
        labels = dict(zip(order.tolist(), _labels(len(order))))
        m_src, photoz, z_cat, z_err = c["m_src"], c["photoz"], c["z_cat"], c["z_err"]
        note = f"class={c['cls']}; n_images={c['n_obs']}; min_sep={c['min_sep']:.2f} arcsec"
        if photoz:
            note += "; photometric redshift"
        # images tied to satellites: {image index: [(global Subfind ID, shift without it)]}
        tied, obs_index = {}, np.flatnonzero(obs).tolist()
        for r in c.get("cf", []):
            for i in r["tagged"]:
                tied.setdefault(i, []).append((r["id"], r["shift"][obs_index.index(i)]))
        for sid in sorted({sid for v in tied.values() for sid, _ in v}):
            ims = ",".join(labels[i] for i in sorted(tied, key=lambda i: order.tolist().index(i))
                           if any(k == sid for k, _ in tied[i]))
            where = "" if not member_ids or str(sid) in member_ids else " (below the members.yaml magnitude limit)"
            note += f"; subhalo {sid}{where}: images {ims}"
        extra = c.get("knots", {})
        knot_names = ["A"] + list(extra) if extra else []
        knots = {"A": dict(beta=c["beta"], tab=tab, noisy=c["noisy"]), **extra}
        st = dict(name=name, z=c["z"], z_err=None, notes=note, knots=knot_names, images=[])
        so = dict(name=name, z=z_cat, z_err=z_err, notes=note, knots=knot_names, images=[])
        for i in order.tolist():
            dt = tab["dt_days"][i]
            tnote = None
            if i in tied:
                tnote = "; ".join(f"subhalo {sid} (shift without it: "
                                  f"{'vanishes' if not np.isfinite(sh) else f'{sh:.2f} arcsec'})" for sid, sh in tied[i])
            inote = (f"mu={mu[i]:.4g}; dt={dt:.2f} d" + ("" if obs[i] else "; not observable")
                     + ("; inside the BCG radius" if in_bcg[i] else "") + (f"; {tnote}" if tnote else ""))
            kmag = {kn: round(m_src + (0.0 if kn == "A" else knot_mag_offset)
                              - 2.5 * np.log10(abs(k["tab"]["mu"][i])), 2) for kn, k in knots.items()}
            common = dict(label=labels[i], parity=int(par[i]), mag=kmag["A"], kmag=kmag)
            st["images"].append(dict(common, pos={kn: k["tab"]["theta"][i] for kn, k in knots.items()}, notes=inote))
            if obs[i]:
                so["images"].append(dict(common, pos={kn: k["noisy"][i] for kn, k in knots.items()}, notes=tnote))
            for kn, k in knots.items():
                kt, true, noisy = k["tab"], k["tab"]["theta"][i], k["noisy"][i]
                image_rows.append(dict(
                    system=name, knot=kn if knot_names else "", image=labels[i], z_true=c["z"], z_catalog=z_cat,
                    efficiency=c["s"], beta_x=k["beta"][0], beta_y=k["beta"][1], x_true=true[0], y_true=true[1],
                    x_obs=noisy[0] if obs[i] else np.nan, y_obs=noisy[1] if obs[i] else np.nan,
                    observable=bool(obs[i]), inside_bcg_radius=bool(in_bcg[i]), mu=kt["mu"][i], parity=int(kt["parity"][i]),
                    kappa=kt["kappa"][i], gamma1=kt["gamma1"][i], gamma2=kt["gamma2"][i],
                    fermat_arcsec2=kt["fermat"][i], dt_days=kt["dt_days"][i], mag=kmag[kn],
                    source_mag=m_src + (0.0 if kn == "A" else knot_mag_offset),
                    residual_arcsec=kt["resid"][i], cls=c["cls"],
                    subhalo=";".join(str(sid) for sid, _ in tied.get(i, []))))
        systems.append((st, so))
        n_out = int(c["n_images"] - in_bcg.sum())
        source_rows.append(dict(
            system=name, z_true=c["z"], z_catalog=z_cat, z_err=z_err if z_err is not None else np.nan,
            efficiency=c["s"], beta_x=c["beta"][0], beta_y=c["beta"][1], n_images=c["n_images"],
            n_images_outside_bcg=n_out, n_observable=c["n_obs"], odd_number=bool(c["n_images"] % 2 == 1),
            cls=c["cls"], geometry=c.get("geometry", c["cls"]), min_sep=c["min_sep"],
            min_sep_observed=c["min_sep_obs"], max_sep=c["max_sep"], source_mag=m_src, n_knots=len(knot_names),
            subhalos=";".join(str(r["id"]) for r in c.get("cf", [])), seed_kind=c["kind"],
            candidate_index=c["index"],
            seed_x=c["seed"][0] if c["seed"] is not None else np.nan,
            seed_y=c["seed"][1] if c["seed"] is not None else np.nan))
    return systems, image_rows, source_rows


# ----------------------------------------------------------------------------- writers
def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def radec(x, y):
    """Map offsets [arcsec] to (ra, dec) [deg] about (0, 0): x = -ra 3600, y = dec 3600."""
    return -x / 3600.0, y / 3600.0


def write_images(path, systems, header, notes):
    lines = [f"# {h}" for h in header] + ["format: lenscluster-images/1", f"notes: {q(notes)}",
                                         "defaults: {shape: [0.3, 0.3, 0.0], mag_err: 0.05}",
                                         "systems:" if systems else "systems: {}"]
    for s in systems:
        lines += [f"  {q(s['name'])}:", f"    z: {s['z']}"]
        if s["z_err"]:
            lines.append(f"    z_err: {s['z_err']}")
        lines.append(f"    notes: {q(s['notes'])}")
        if s["knots"]:
            lines.append(f"    knots: [{', '.join(s['knots'])}]")
        lines.append("    images:")
        for im in s["images"]:
            extra = f", notes: {q(im['notes'])}" if im["notes"] else ""
            if s["knots"]:
                lines += [f"      {im['label']}:", f"        parity: {im['parity']}"]
                if im["notes"]:
                    lines.append(f"        notes: {q(im['notes'])}")
                lines.append("        knots:")
                for kn in s["knots"]:
                    ra, dec = radec(*im["pos"][kn])
                    lines.append(f"          {kn}: {{ra: {ra:.9f}, dec: {dec:.9f}, mag: {im['kmag'][kn]}}}")
                continue
            ra, dec = radec(*im["pos"]["A"])
            lines.append(f"      {im['label']}: {{ra: {ra:.9f}, dec: {dec:.9f}, parity: {im['parity']}, "
                         f"mag: {im['mag']}{extra}}}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_members(path, rows, header, band):
    lines = [f"# {h}" for h in header] + [f"band: {band}", "members:"]
    for m in rows:
        ra, dec = radec(m["x"], m["y"])
        parts = [f"ra: {ra:.9f}", f"dec: {dec:.9f}", f"mag: {{{band}: {m['mag']:.3f}}}"]
        if m["shape"]:
            parts.append("shape: [{:.3f}, {:.3f}, {:.1f}]".format(*m["shape"]))
        if m["role"] == "bcg":
            parts.append("role: bcg")
        lines.append(f"  {q(m['id'])}: {{{', '.join(parts)}}}")
    if not rows:
        lines[-1] = "members: {}"
    Path(path).write_text("\n".join(lines) + "\n")


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.9g}"         # 9 significant digits: residuals and tiny |mu| keep their precision
    if isinstance(v, np.integer):
        return str(int(v))
    return "" if v is None else v


def write_csv(path, rows, fieldnames=None):
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fieldnames})


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


# ----------------------------------------------------------------------------- driver
def build_catalog(lens_path, cfg, outdir) -> dict:
    """Build the catalog of ``lens_path`` with the ``catalog:`` configuration ``cfg`` into ``outdir``.

    Returns {name: path} of every file written, plus ``summary`` (counts and timings).
    """
    t0 = time.perf_counter()
    cfg = resolve_config(cfg)
    lens_path = str(lens_path)
    outdir = Path(outdir)
    meta = read_meta(lens_path, cfg["overrides"])
    require(meta, ("lens_redshift", "H0", "Om0"), "the catalog")
    cosmo = cosmology.make({"H0": meta["H0"], "Om0": meta["Om0"]})
    zl = meta["lens_redshift"]
    kpa = meta.get("kpc_per_arcsec") or cosmology.kpc_per_arcsec(cosmo, zl)
    notes = []
    fg = sources_mod.check_redshifts(cfg["sources"]["redshift"], zl)      # ValueError if all of it is foreground
    if fg > 0:
        msg = (f"{fg:.1%} of the sources.redshift distribution lies at or in front of the lens redshift "
               f"{zl:.4f}; those candidates are counted as 'foreground'")
        warnings.warn(msg, stacklevel=2)
        notes.append(msg)
    extent, dx = map_extent(lens_path)
    half, half_note = _search_half(cfg, meta, extent, dx)
    if half_note:
        notes.insert(0, half_note)
    if abs(kpa / cosmology.kpc_per_arcsec(cosmo, zl) - 1) > 1e-4:
        notes.append(f"file kpc_per_arcsec {kpa:.6g} differs from the cosmology's "
                     f"{cosmology.kpc_per_arcsec(cosmo, zl):.6g}; the file's value is used for the map frame")
    lo, hi = cfg["sources"]["seed_radius"]
    cfg["sources"]["seed_radius"] = [float(lo), round(95.0 / 110.0 * half, 3) if hi in (None, "auto") else float(hi)]

    # the satellites need the Subfind chunk and the snapshot chunk; without them (auto) that part is left out
    if cfg["subhalos"]["enabled"] == "auto" and subhalos_on(cfg):
        missing = satellites_mod.unavailable(meta, cfg["members"])
        if missing:
            cfg["subhalos"]["enabled"] = False
            cfg["sources"]["seed_kinds"] = dict(cfg["sources"]["seed_kinds"], subhalo=0.0)
            if cfg["quotas"] is not None:
                cfg["quotas"] = dict(cfg["quotas"], subhalo=0)
            notes.append(f"no systems tied to satellites, no subhalo quota and no subhalo seeds: {missing}")
            if not any(float(w) > 0 for w in cfg["sources"]["seed_kinds"].values()):
                raise ValueError(f"catalog.sources.seed_kinds has only the subhalo kind, which is not available: {missing}")
            if cfg["quotas"] is not None and sum(cfg["quotas"].values()) < 1:
                raise ValueError(f"catalog.quotas has only the subhalo class, which is not available: {missing}")

    # members first: the BCG position sets the exclusion region
    mcfg = cfg["members"]
    want = mcfg["enabled"]
    members, minfo = [], None
    if want is True:
        members_mod.group_file(meta, mcfg)          # a clear error when the group catalog cannot be found
    if want is True or (want == "auto" and members_mod.available(meta, mcfg)):
        members, minfo = members_mod.member_table(meta, cosmo, kpa, mcfg)
    elif want == "auto":
        notes.append("member table skipped: group catalog not found")
    centre = tuple(minfo["bcg_xy"]) if minfo else (0.0, 0.0)
    t_members = time.perf_counter() - t0

    # candidates
    workers = int(cfg["workers"])
    if workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=(lens_path, cfg, half, centre))
        evaluate_batch = lambda idx: pool.map(_work, list(idx), chunksize=1)
    else:
        ev = Evaluator(lens_path, cfg, half, centre)
        evaluate_batch = lambda idx: [ev(i) for i in idx]
        pool = None
    t1 = time.perf_counter()
    try:
        accepted, records, quotas = _select(evaluate_batch, cfg)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    t_select = time.perf_counter() - t1
    n_knotted = 0
    if cfg["knots"]["n_systems"] > 0 and accepted:
        if workers > 1:                              # the pool is gone; the knots need the lens in this process
            lens = LensMap.from_file(lens_path, cfg["overrides"], half=half)
            n_knotted = add_knots(accepted, lens, ImageFinder(lens, half, cfg["finder_stride"]), cfg)
        else:
            n_knotted = add_knots(accepted, ev.lens, ev.finder, cfg)
    systems, image_rows, source_rows = _assemble(accepted, [m["id"] for m in members], cfg["knots"]["mag_offset"])
    if not accepted:
        notes.append(f"no system accepted in {len(records)} candidates: the image catalogs have no systems")
    if quotas is not None and quotas.missing():
        notes.append(f"quotas not filled after {len(records)} candidates (max_candidates {cfg['max_candidates']}): "
                     + ", ".join(f"{k} short by {v}" for k, v in quotas.missing().items()))
    if cfg["knots"]["n_systems"] > n_knotted:
        notes.append(f"knots: {n_knotted} of the {cfg['knots']['n_systems']} systems asked for got one")
    by_class = {}
    for c in accepted:
        by_class[c["cls"]] = by_class.get(c["cls"], 0) + 1

    counts = {s: 0 for s in STATUS}
    by_kind = {}
    for r in records:
        counts[r["status"]] += 1
        k = r["kind"] or "none"
        by_kind.setdefault(k, {s: 0 for s in STATUS})[r["status"]] += 1
    mult = {}
    for r in source_rows:
        mult[str(r["n_observable"])] = mult.get(str(r["n_observable"]), 0) + 1
    selection = dict(
        tnglens_version=__version__, lens_file=Path(lens_path).name, lens_file_attributes=_lens_id(meta),
        seed=int(cfg["seed"]), search_half=half,
        centre_for_exclusion=list(centre),
        n_systems_requested=int(sum(quotas.wanted.values()) if quotas is not None else cfg["n_systems"]),
        n_systems=len(accepted), systems_by_class={k: by_class[k] for k in CLASSES if k in by_class},
        quotas=dict(quotas.wanted) if quotas is not None else None, n_systems_with_knots=n_knotted,
        n_systems_tied_to_satellites=sum(bool(c.get("cf")) for c in accepted),
        n_candidates=len(records), status_counts=counts, status_by_kind=by_kind,
        observable_multiplicity=dict(sorted(mult.items(), key=lambda kv: int(kv[0]))),
        cosmology=dict(H0=meta["H0"], Om0=meta["Om0"]), lens_redshift=zl, kpc_per_arcsec=kpa,
        time_delay_days_per_arcsec2=cosmology.time_delay_days(cosmo, zl), notes=notes,
        members=_plain(minfo) if minfo else None,
        config=_plain({k: v for k, v in cfg.items() if k not in ("workers", "batch")}))

    # write
    (outdir / "truth").mkdir(parents=True, exist_ok=True)
    (outdir / "observed").mkdir(parents=True, exist_ok=True)
    paths = {k: outdir / v for k, v in dict(
        images_truth_yaml="truth/images_truth.yaml", images_truth_csv="truth/images_truth.csv",
        sources_truth_csv="truth/sources_truth.csv", candidates_csv="truth/candidates.csv",
        members_truth_csv="truth/members_truth.csv", selection_yaml="truth/selection.yaml",
        images_yaml="observed/images.yaml", members_yaml="observed/members.yaml", readme="README.md").items()}
    o, nz = cfg["observability"], cfg["noise"]
    desc = _describe(meta, zl)
    # the catalog states its own cosmology: a fit must use the same distance ratios
    cosmo_note = (f"Cosmology: flat LCDM, H0 = {meta['H0']:g}, Om0 = {meta['Om0']:g}, Ode0 = {1 - meta['Om0']:g}; "
                  f"lens redshift {zl:.6f}.")
    head = [f"Mock multiple images by tnglens {__version__} (tnglens.catalog), seed {cfg['seed']}.", desc]
    if any(c.get("cf") for c in accepted):
        head[1] += " Systems S<n> have images tied to a satellite (see their notes)."
    write_images(paths["images_truth_yaml"], [s[0] for s in systems],
                 head + [f"Truth version: exact positions, every image inside the {2 * half:g} arcsec search square, "
                         "true redshifts; image notes give mu and the time delay after the first image."],
                 f"Truth mock catalog; see README.md. {cosmo_note}")
    write_images(paths["images_yaml"], [s[1] for s in systems],
                 head + [f"Observed version: Gaussian position noise of {nz['sigma_pos']} arcsec per axis; images within "
                         f"{o['bcg_radius']} arcsec of the BCG or with |mu| <= {o['mu_min']} removed; "
                         f"{nz['photoz_fraction']:.0%} of systems have a photometric z."],
                 f"Observed mock catalog; truth in ../truth/images_truth.yaml and ../truth/images_truth.csv. {cosmo_note}")
    write_csv(paths["images_truth_csv"], image_rows, IMAGE_COLUMNS)
    write_csv(paths["sources_truth_csv"], source_rows, SOURCE_COLUMNS)
    write_csv(paths["candidates_csv"], [_candidate_row(r) for r in records], CANDIDATE_COLUMNS)
    write_csv(paths["members_truth_csv"], members, list(members_mod.COLUMNS))
    write_members(paths["members_yaml"], members,
                  [f"Members by tnglens {__version__} (tnglens.members). {desc}",
                   f"Subfind galaxies within {mcfg['radius']:g} arcsec and {mcfg['los_r200c']:g} R200c along the line "
                   f"of sight, m < {mcfg['mag_limit']:g}.",
                   f"{mcfg['band_name']} = rest-frame SubhaloStellarPhotometrics[{mcfg['band_index']}] absolute magnitude "
                   "+ distance modulus - 2.5 log10(1 + z_lens) (AB, no dust).",
                   "Names are global Subfind IDs. Shapes are stellar second moments within twice the stellar half-mass "
                   "radius: [a, b, theta] in arcsec and degrees counter-clockwise from +x.",
                   "The BCG magnitude counts all stars bound to the central subhalo, so it includes the intracluster light."],
                  mcfg["band_name"])
    paths["selection_yaml"].write_text(yaml.safe_dump(_plain(selection), sort_keys=False, width=110))
    paths["readme"].write_text(_readme(selection, cfg, meta, half, centre, len(members), image_rows))
    summary = dict(n_systems=len(accepted), n_candidates=len(records), status_counts=counts,
                   systems_by_class=selection["systems_by_class"], n_systems_with_knots=n_knotted,
                   n_observed_images=sum(len(s[1]["images"]) for s in systems),
                   observable_multiplicity=selection["observable_multiplicity"],
                   n_members=len(members), seconds_members=t_members, seconds_select=t_select,
                   seconds_total=time.perf_counter() - t0)
    out = {k: str(v) for k, v in paths.items()}
    out["summary"] = summary
    return out


IMAGE_COLUMNS = ["system", "knot", "image", "z_true", "z_catalog", "efficiency", "beta_x", "beta_y", "x_true", "y_true",
                 "x_obs", "y_obs", "observable", "inside_bcg_radius", "mu", "parity", "kappa", "gamma1", "gamma2", "fermat_arcsec2",
                 "dt_days", "mag", "source_mag", "residual_arcsec", "cls", "subhalo"]
SOURCE_COLUMNS = ["system", "z_true", "z_catalog", "z_err", "efficiency", "beta_x", "beta_y", "n_images",
                  "n_images_outside_bcg", "n_observable", "odd_number", "cls", "geometry", "min_sep",
                  "min_sep_observed", "max_sep", "source_mag", "n_knots", "subhalos", "seed_kind", "candidate_index",
                  "seed_x", "seed_y"]
LENS_ID_ATTRS = ("signature", "tnglens_version", "zoom_index", "snapshot", "halo_id", "axis", "lens_redshift")


def _lens_id(meta):
    """The lens file's identifying attributes (those present), recorded instead of its path."""
    return {k: meta[k] for k in LENS_ID_ATTRS if meta.get(k) is not None}
CANDIDATE_COLUMNS = ["index", "kind", "z", "beta_x", "beta_y", "n_images", "n_observable", "status"]


def _candidate_row(r):
    b = r.get("beta")
    return dict(index=r["index"], kind=r["kind"] or "", z=r["z"] if r["z"] is not None else np.nan, beta_x=b[0] if b is not None else np.nan,
                beta_y=b[1] if b is not None else np.nan, n_images=r["n_images"], n_observable=r["n_obs"],
                status=r["status"])


def _describe(meta, zl):
    parts = ["TNG-Cluster"]
    if meta.get("zoom_index") is not None:
        parts.append(f"zoom {meta['zoom_index']:03d}")
    if meta.get("halo_id") is not None:
        parts.append(f"(halo {meta['halo_id']})")
    s = " ".join(parts)
    if meta.get("snapshot") is not None:
        s += f", snapshot {meta['snapshot']}"
    s += f" (z = {zl:.4f})"
    if meta.get("axis"):
        s += f", {meta['axis']} projection"
    return (s + ". Simulated cluster: positions are offsets about (0, 0) in degrees, x = -ra*3600, y = dec*3600 "
            f"arcsec in the lens-map frame. Flat LCDM H0={meta['H0']:g} Om0={meta['Om0']:g}.")


def _sig(sel):
    sig = sel["lens_file_attributes"].get("signature")
    return f" (signature `{sig}`)" if sig else ""


def _readme(sel, cfg, meta, half, centre, n_members, image_rows):
    o, nz, sc = cfg["observability"], cfg["noise"], cfg["sources"]
    zspec = sc["redshift"]
    if zspec.get("values") is not None:
        zline = f"drawn from the list {zspec['values']} with weights {zspec.get('weights') or 'equal'}"
    else:
        zline = (f"uniform within the bins {zspec['edges']} with weights {zspec['weights']}, rounded to 1e-4")
    kinds = ", ".join(f"{k} {v}" for k, v in sc["seed_kinds"].items() if float(v) > 0)
    c = sel["status_counts"]
    mult = ", ".join(f"{k} images: {v}" for k, v in sel["observable_multiplicity"].items())
    base_rows = [r for r in image_rows if r["knot"] in ("", "A")]
    n_obs = sum(r["observable"] for r in base_rows)
    n_core = sum(r["inside_bcg_radius"] for r in base_rows)
    kn, sub, sl = cfg["knots"], cfg["subhalos"], cfg["selection"]
    classes = ", ".join(f"{v} {k}" for k, v in sel["systems_by_class"].items())
    if sel["quotas"]:
        fold_bins = ", ".join(f"{a:g}-{b:g}" for a, b in zip(sl["fold_bins"], sl["fold_bins"][1:]))
        quota_text = f"""The systems are kept by class quota ({", ".join(f"{k} {v}" for k, v in sel["quotas"].items() if v)}): candidates
are taken in candidate order while their class has room. The fold quota is split evenly over pair separations of
{fold_bins} arcsec. A wide system with only two observable images is kept with probability {sl['wide_pair_keep']:g}.
At most {sl['max_per_satellite']} subhalo systems are tied (first) to the same satellite, and a system with a tied image
counts as subhalo only. Kept: {classes}. Not kept for the quotas: {c['no_quota']} without a quota, {c['quota_full']} with
the quota full, {c['thinned']} wide pairs thinned."""
    else:
        quota_text = f"No class quotas: the first candidates that pass the cuts are kept ({classes or 'none'})."
    if subhalos_on(cfg):
        sub_text = f"""Images tied to satellites: for each satellite with bound mass above {sub['min_mass']:.3g} Msun (in the particle
types of the lens) near an observable image, the satellite's bound particles are removed, the source position is
refitted and the images are solved again. An image within {sub['radius_einstein']:g} point-mass Einstein radii of the
satellite (clipped to {sub['radius_clip'][0]:g}-{sub['radius_clip'][1]:g} arcsec) that vanishes or moves by more than
{sub['shift_tol']:g} arcsec is tied to it. {sel['n_systems_tied_to_satellites']} systems have a tied image; they are of class
subhalo and named S1, S2, ... The system notes list the satellite (global Subfind ID) and its images, and each
tied image has a note with the shift. The satellite's deflection uses a Gaussian smoothing of
{sub['patch_sigma_pixels'] * 2 * sub['patch_half'] / sub['patch_pixels']:.2g} arcsec, not the adaptive kernel of the map, so the tie
is a classification and not a measurement."""
    else:
        sub_text = "No images are tied to satellites in this catalog (`subhalos.enabled` is off)."
    if sel["n_systems_with_knots"]:
        knot_text = f"""Knots: the {sel['n_systems_with_knots']} most magnified systems carry {kn['extra'][0]} to {kn['extra'][1]} extra knots (B, C),
point sources {kn['offset'][0]:g}-{kn['offset'][1]:g} arcsec from the source centre and {kn['mag_offset']:g} mag fainter. Every image of such
a system has one position per knot (knot A is the source centre), and `truth/images_truth.csv` has one row per
image and knot."""
    else:
        knot_text = "No system has knots in this catalog."
    return f"""# Mock lensing catalog

{_describe(meta, sel['lens_redshift'])}

Lens file: `{sel['lens_file']}`{_sig(sel)}. Built by tnglens {sel['tnglens_version']} (`tnglens.catalog`) with seed {sel['seed']};
the same lens file, configuration and seed give the same files. The resolved configuration and the selection
counts are in `truth/selection.yaml`.

{sel['n_systems']} systems ({sel['n_systems_requested']} requested), {len(base_rows)} images in the truth catalog, {n_obs}
observable, {n_core} inside the BCG radius; {n_members} members.

## Files

| file | content |
|---|---|
| `observed/images.yaml` | fit-ready catalog (lenscluster-images/1): noisy positions of the observable images |
| `observed/members.yaml` | members: ra, dec, {cfg['members']['band_name']} magnitude, stellar shape, BCG role |
| `truth/images_truth.yaml` | same systems: every image found in the search square, exact positions, true redshifts |
| `truth/images_truth.csv` | one row per image (columns below) |
| `truth/sources_truth.csv` | one row per system: source position and redshift, multiplicity, how it was seeded |
| `truth/candidates.csv` | every candidate source drawn, in order, and why it was kept or rejected |
| `truth/members_truth.csv` | Subfind properties of each member |
| `truth/selection.yaml` | configuration, counts and lens metadata |

## Frame and lens

Positions are offsets about (0, 0) in degrees: x = -ra 3600, y = dec 3600 arcsec in the frame of the lens map,
whose origin is the halo centre. Images solve beta = theta - s alpha(theta) with s = D_ds/D_s(z_s) and alpha
the deflection map for D_ds/D_s = 1, interpolated by a cubic B-spline. Magnification mu = 1/det(A), parity
sign(det A), kappa and shear come from the analytic Jacobian of the same spline (kappa = s (a_xx + a_yy)/2,
gamma1 = s (a_xx - a_yy)/2, gamma2 = s (a_xy + a_yx)/2), not from the kappa and gamma maps. Time delays use the
Fermat potential 0.5 |theta - beta|^2 / s - psi(theta) with psi the potential map (D_ds/D_s = 1), times
(1 + z_l) D_l / c = {sel['time_delay_days_per_arcsec2']:.6g} days per arcsec^2, relative to the first image of the
system. The search square is |x|, |y| <= {half:g} arcsec on a mesh of every {cfg['finder_stride']} map pixels;
images outside it are not in the catalog.

## Sources and selection

Redshifts are {zline}. Candidates are seeded with the kinds (weights) {kinds}: `negative` maps a random
image-plane mesh node with det A < 0 at radius {sc['seed_radius'][0]}-{sc['seed_radius'][1]} arcsec to the source plane, `critical`
a node with |det A| below one of {sc['critical_det']}, `source` is uniform in the source plane over the region
that holds every multiply imaged source, `subhalo` a node with det A < 0 within {sub['seed_radius']:g} arcsec of a satellite above
{sub['seed_min_mass']:.3g} Msun. Image-plane seeding favours sources with large negative-parity or
near-critical image area, so **the systems are not an unbiased source population**: they are candidates
taken in candidate order with at least {max(2, int(o['min_images']))} observable images, no pair of observable
images closer than {o['min_separation']} arcsec, at their true or at their noisy positions (such systems are
rejected whole), and not within dz <= {sc['duplicate_dz']} and {sc['duplicate_dbeta']} arcsec of an earlier kept source.
Rejecting whole systems with a close pair removes many of the most magnified fold pairs (see
`status_by_kind` in `truth/selection.yaml`): they are under-represented.

Candidates drawn: {sel['n_candidates']}. Accepted {c['accepted']}; source at or in front of the lens {c['foreground']};
no seed node {c['no_seed']}; one image or none in the square {c['single']}; fewer than two observable images
{c['lt2_observable']}; observable pair too close {c['close_pair']} (true positions) and {c['close_pair_observed']} (noisy
positions); duplicate {c['duplicate']}. Observable multiplicity of the kept systems: {mult}.

An image is observable when |mu| > {o['mu_min']} and it lies more than {o['bcg_radius']} arcsec from the BCG at
({centre[0]:.3f}, {centre[1]:.3f}) arcsec. The observed catalog adds Gaussian noise of {nz['sigma_pos']} arcsec per
axis to the observable images and drops the others; {nz['photoz_fraction']:.0%} of systems (drawn per system) get a
photometric redshift: a value scattered about the true z by {nz['photoz_sigma']} (1 + z_true), at least
max({nz['photoz_zmin']}, z_lens + {PHOTOZ_LENS_GAP}), reported with z_err = {nz['photoz_sigma']} (1 + z_catalog).
Magnitudes are a source magnitude uniform in {nz['source_mag']} minus 2.5 log10 |mu|; no magnitude limit.
Systems are named 1, 2, ... in order of true redshift. Image labels put observable images first, brightest first.
Systems of class subhalo are named S1, S2, ... The `notes` give the family class of the observable images (wide:
closest pair >= 8"; intermediate: 3-8"; fold: an opposite-parity pair <= 3" with the next image more than
max(2 sep, 2") away; cusp: three close images of alternating parity; other).

Images inside the BCG radius: the truth tables keep every image the finder solves, including central images
within {o['bcg_radius']} arcsec of the BCG. Those exist mostly because the pixelised deflection map is
interpolated through a core the simulation does not resolve (kappa of tens, |mu| ~ 1e-4), so they are not
reliable predictions; they are flagged `inside_bcg_radius` and are never observable. `n_images` counts them,
`n_images_outside_bcg` does not, and `odd_number` refers to `n_images`.

{quota_text}

{sub_text}

{knot_text}

## Truth columns (`truth/images_truth.csv`)

system, knot (empty unless the system has knots), image (label); z_true (source redshift), z_catalog (as in observed/images.yaml), efficiency (D_ds/D_s);
beta_x, beta_y (true source position, arcsec); x_true, y_true (exact image position solving the lens equation to
1e-7 arcsec, residual_arcsec); x_obs, y_obs (noisy, nan for unobservable images); observable;
inside_bcg_radius (within {o['bcg_radius']} arcsec of the BCG, see above); mu (signed);
parity; kappa, gamma1, gamma2 at the image for this source redshift; fermat_arcsec2; dt_days (arrival after the
first image); mag (observed-catalog magnitude), source_mag; cls (family class of the system); subhalo (global
Subfind IDs of the satellites the image is tied to).

`truth/sources_truth.csv`: n_images (all images in the square), n_images_outside_bcg (without those inside the
BCG radius), n_observable, odd_number (n_images odd, as a non-singular lens with every image inside the square
must give), min_sep/max_sep of the observable images (true positions), min_sep_observed (noisy positions),
geometry (the class from the image geometry alone; cls is subhalo when an image is tied to a satellite), n_knots,
subhalos (satellites with a tied image), seed_kind and candidate_index (the candidate that produced it),
seed_x/seed_y (the image-plane seed).

`truth/members_truth.csv`: id (global Subfind ID), index (local index in the chunk), x, y (arcsec), los_pkpc,
mag, role (bcg/member), fof_group, mtot, mstar, m_in_2rhalf (total mass within twice the stellar half-mass
radius), mstar_in_2rhalf, mlens (bound mass in the particle types of the lens) [Msun], veldisp, vmax [km/s],
rvmax_pkpc, rhalf_star_pkpc, rhalf_pkpc, shape_a, shape_b [arcsec], shape_theta [deg], n_star_shape.
"""


# ----------------------------------------------------------------------------- command line
def load_config(path):
    """The ``catalog:`` section of a YAML config (or the whole file if it has none)."""
    if path is None:
        return {}
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return doc.get("catalog", doc) if isinstance(doc, dict) else {}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tnglens.catalog", description=__doc__.split("\n")[0])
    ap.add_argument("lens", help="lens file (HDF5)")
    ap.add_argument("--config", help="YAML with a catalog: section")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="lens-file attribute override (value parsed as YAML); repeatable")
    ap.add_argument("--workers", type=int, help="override catalog.workers")
    ap.add_argument("--n-systems", type=int, help="keep the first N systems that pass the cuts (sets quotas to null)")
    ap.add_argument("--seed", type=int, help="override catalog.seed")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ov = dict(cfg.get("overrides") or {})
    for item in a.override:
        k, _, v = item.partition("=")
        ov[k] = yaml.safe_load(v)
    if ov:
        cfg["overrides"] = ov
    for k in ("workers", "n_systems", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.n_systems is not None:
        cfg["quotas"] = None
    out = build_catalog(a.lens, cfg, a.out)
    s = out.pop("summary")
    print(yaml.safe_dump(_plain(dict(summary=s, files=out)), sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
