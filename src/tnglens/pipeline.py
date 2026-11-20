"""Stages, caching and parallel execution.

For each halo (snapshot + zoom or halo id) the stages are

1. ``particles``  -- read every FoF particle of the zoom, per type   (cache_root/particles/<key>/)
2. ``smoothing``  -- k-th neighbour distances, per type              (cache_root/smoothing/<key>/n<k>/)
3. ``projection`` -- core and environment Sigma maps, per axis        (cache_root/projections/<name>/sigma-<sig>.h5)
4. ``maps``       -- the lens file                                    (output_root/<name>/lens.h5)
5. ``catalog``    -- tnglens.catalog.build_catalog, if the config has a ``catalog`` section

Each product carries a signature of its inputs and is reused when it matches; ``force``
names the first stage to recompute (later stages follow automatically, because their
signatures include the earlier ones).  Stages 1 and 2 run only when some projection of the
halo actually has to be computed.  Halos run in separate processes (``workers``), each using
``threads`` threads for the tree search, sphviewer2 and the FFTs.

Concurrency: the particle and smoothing caches of a halo are guarded by a lock file
(``cache_root/locks/<key>.lock``): building them takes it exclusively, using them takes it
shared, and ``keep_particles: false`` deletes only the caches this run built, and only if no
other run holds the lock.  The projection file name contains its signature, so runs with
different settings never share one.  Each product directory is written under its own lock
(``output_root/.locks/<name>.lock``).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import multiprocessing
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from . import __version__, cosmology
from .cache import (FileLock, atomic_h5, atomic_text, is_complete, remove_stale_partials, signature,
                    signature_of, stage, to_json)
from .config import ConfigError, requests
from .halos import Halo, find_halo, zoom_of_halo
from .maps import lens_attributes, lens_config, lens_fields, write_lens_file
from .particles import particle_signature, read_species, smooth_species, smoothing_signature
from .projection import make_grids, project

log = logging.getLogger("tnglens")

STAGES = ("particles", "smoothing", "projection", "maps", "catalog")
PROJECTION_VERSION = 2
MAPS_VERSION = 3
CATALOG_VERSION = 2
CATALOG_OUTPUTS = ("truth", "observed", "README.md")  # what build_catalog writes; cleared before a rebuild


# ----------------------------------------------------------------------------- planning
def _matches(names, only) -> bool:
    return any(fnmatch.fnmatchcase(n, pat) for n in names for pat in only)


def _request_label(req) -> str:
    what = f"zoom {req.zoom}" if req.zoom is not None else f"halo_id {req.halo_id}"
    return f"snapshot {req.snapshot} {what}"


def _candidate_names(cfg, req):
    """Product names (and halo keys) a request can produce, without opening its files; None if unknown."""
    if req.zoom is not None:
        stem = f"tngc_z{req.zoom:03d}_s{req.snapshot:03d}"
        return [stem] + [f"{stem}_{a}" for a in req.axes]
    try:
        zoom = zoom_of_halo(cfg["simulation"]["base"], req.snapshot, req.halo_id, cfg["simulation"]["offsets"])
    except Exception:
        return None
    # Whether the group is the zoom's primary (no _h suffix) is known only from its group file.
    stem, suffix = f"tngc_z{zoom:03d}_s{req.snapshot:03d}", f"_h{req.halo_id}"
    return ([stem, stem + suffix] + [f"{stem}_{a}" for a in req.axes]
            + [f"{stem}_{a}{suffix}" for a in req.axes])


def resolve(cfg: dict, only=None):
    """Look up the requested halos on disk.

    Returns ``(jobs, failures)``: ``jobs`` is a list of (halo, axes), one per halo; ``failures``
    holds one summary dict (with ``error`` and ``lookup: True``) per request whose lookup failed.
    With ``only``, requests whose product names cannot match are skipped without being looked up,
    so a missing halo does not affect unrelated products.
    """
    base, offsets = cfg["simulation"]["base"], cfg["simulation"]["offsets"]
    out: dict[str, tuple[Halo, tuple[str, ...]]] = {}
    failures = []
    for req in requests(cfg):
        if only:
            names = _candidate_names(cfg, req)
            if names is not None and not _matches(names, only):
                continue
        try:
            halo = find_halo(base, req.snapshot, zoom=req.zoom, halo_id=req.halo_id, offsets_dir=offsets)
        except (OSError, ValueError, KeyError) as exc:
            log.error("%s: lookup failed: %s", _request_label(req), exc)
            failures.append({"halo": _request_label(req), "error": f"lookup failed: {exc}", "lookup": True,
                             "stages": {}, "products": {}})
            continue
        if halo.key() in out:
            axes = tuple(dict.fromkeys(out[halo.key()][1] + req.axes))
        else:
            axes = req.axes
        out[halo.key()] = (halo, axes)
    return list(out.values()), failures


def select(jobs, only):
    """Keep the (halo, axes) whose product names match any of the glob patterns in ``only``."""
    if not only:
        return jobs
    kept = []
    for halo, axes in jobs:
        axes = tuple(a for a in axes if _matches([halo.name(a), halo.key()], only))
        if axes:
            kept.append((halo, axes))
    return kept


def lens_cosmology(cfg: dict, halo: Halo) -> dict:
    spec = cfg["cosmology"]
    cosmo = cosmology.make(spec, halo.header)
    zl = halo.redshift
    behind = [z for z in cfg["source_redshifts"] if z <= zl]
    if behind:
        raise ConfigError(f"source redshifts {behind} are not behind the lens {halo.key()} (z = {zl:.4f})")
    return {"H0": float(cosmo.H0.value), "Om0": float(cosmo.Om0),
            "source": "native" if spec == "native" else "config",
            "kpc_per_arcsec": float(cosmology.kpc_per_arcsec(cosmo, zl)),
            "sigma_critical_inf": float(cosmology.sigma_critical_inf(cosmo, zl)),
            "efficiencies": {z: float(cosmology.efficiency(cosmo, zl, z)) for z in cfg["source_redshifts"]}}


# ----------------------------------------------------------------------------- paths
def particle_path(cfg, halo, pt):
    return Path(cfg["cache_root"]) / "particles" / halo.key() / f"type{pt}.h5"


def smoothing_path(cfg, halo, pt):
    k = cfg["projection"]["neighbors"]
    return Path(cfg["cache_root"]) / "smoothing" / halo.key() / f"n{k:03d}" / f"type{pt}.h5"


def projection_path(cfg, halo, axis, sig: str):
    """One file per projection signature, so different settings or cosmologies never share a path."""
    return Path(cfg["cache_root"]) / "projections" / halo.name(axis) / f"sigma-{sig[:16]}.h5"


def halo_lock_path(cfg, halo):
    return Path(cfg["cache_root"]) / "locks" / f"{halo.key()}.lock"


def output_dir(cfg, halo, axis):
    return Path(cfg["output_root"]) / halo.name(axis)


def output_lock_path(cfg, halo, axis):
    return Path(cfg["output_root"]) / ".locks" / f"{halo.name(axis)}.lock"


# ----------------------------------------------------------------------------- one halo
def _plan_axis(cfg, halo, axis, cosmo, inputs) -> dict:
    """Signatures and paths of one axis's products (cheap: no particle data is read)."""
    p = cfg["projection"]
    fine, environment = make_grids(p, halo.R200c_pkpc, cosmo["kpc_per_arcsec"])
    proj_sig = signature({"version": PROJECTION_VERSION, "tnglens": __version__, "inputs": inputs, "axis": axis,
                          "settings": p, "R200c_pkpc": halo.R200c_pkpc,
                          "grids": [(g.width, g.n) for g in (fine, environment)]})
    lens_sig = signature({"version": MAPS_VERSION, "tnglens": __version__, "projection": proj_sig,
                          "cosmology": cosmo, "config": lens_config(cfg)})
    return {"axis": axis, "grids": (fine, environment), "proj_sig": proj_sig,
            "proj_file": projection_path(cfg, halo, axis, proj_sig), "lens_sig": lens_sig}


def _species_complete(cfg, halo, types) -> bool:
    k = cfg["projection"]["neighbors"]
    return all(is_complete(particle_path(cfg, halo, pt), particle_signature(halo, pt))
               and is_complete(smoothing_path(cfg, halo, pt), smoothing_signature(halo, pt, k)) for pt in types)


def _prepare_species(cfg, halo, types, forced, stages, label, lock) -> list[Path]:
    """Make sure the particle and smoothing caches exist; return the files this call built.

    Leaves ``lock`` held shared, so no other run deletes or rebuilds them while they are used.
    """
    k = cfg["projection"]["neighbors"]
    created = []
    lock.shared()
    rebuild = forced("particles") or forced("smoothing")
    if rebuild or not _species_complete(cfg, halo, types):
        lock.exclusive()  # waits for runs that are reading these caches
        with stage("particles", stages, label) as rec:
            remove_stale_partials(particle_path(cfg, halo, types[0]).parent)
            built = []
            for pt in types:
                if read_species(halo, pt, particle_path(cfg, halo, pt), cfg["read_chunk"], forced("particles")):
                    built.append(pt)
                    created.append(particle_path(cfg, halo, pt))
            rec.update(status="done" if built else "cached", built_types=built)
        with stage("smoothing", stages, label) as rec:
            remove_stale_partials(smoothing_path(cfg, halo, types[0]).parent)
            built = []
            for pt in types:
                if smooth_species(halo, pt, particle_path(cfg, halo, pt), smoothing_path(cfg, halo, pt), k,
                                  cfg["threads"], forced("smoothing")):
                    built.append(pt)
                    created.append(smoothing_path(cfg, halo, pt))
            rec.update(status="done" if built else "cached", built_types=built)
        lock.shared()
        if not _species_complete(cfg, halo, types):  # removed while the lock was being converted
            raise RuntimeError(f"{halo.key()}: particle or smoothing caches disappeared after being built")
    else:
        for name in ("particles", "smoothing"):
            with stage(name, stages, label) as rec:
                rec.update(status="cached", built_types=[])
    return created


def _remove_created(cfg, halo, created, lock, label):
    """keep_particles: false -- delete the caches this run built, unless another run is using them."""
    if not created:
        log.info("%skeep_particles: false, but the particle/smoothing caches predate this run; kept", label)
        return
    if not lock.try_exclusive():
        log.info("%skeep_particles: false, but another run is using the particle/smoothing caches; kept", label)
        return
    for path in created:
        path.unlink(missing_ok=True)
    for directory in {p.parent for p in created} | {p.parent.parent for p in created}:
        if directory.name in ("particles", "smoothing"):
            continue
        try:
            directory.rmdir()  # only if empty
        except OSError:
            pass
    log.info("%sremoved %d particle/smoothing cache file(s) built by this run (keep_particles: false)", label,
             len(created))


def run_halo(cfg: dict, halo: Halo, axes, force: str | None = None) -> dict:
    """Run all stages for one halo and the given axes; returns a summary dict."""
    forced = lambda name: force is not None and STAGES.index(name) >= STAGES.index(force)
    label = f"[{halo.key()}] "
    types = cfg["particle_types"]
    summary = {"halo": halo.key(), "stages": {}, "products": {}}
    cosmo = lens_cosmology(cfg, halo)

    inputs = {pt: (particle_signature(halo, pt), smoothing_signature(halo, pt, cfg["projection"]["neighbors"]))
              for pt in types}
    plans = {axis: _plan_axis(cfg, halo, axis, cosmo, inputs) for axis in axes}
    needed = [a for a, plan in plans.items()
              if forced("projection") or not is_complete(plan["proj_file"], plan["proj_sig"])]
    lock, created = None, []
    try:
        if needed:
            lock = FileLock(halo_lock_path(cfg, halo))
            created = _prepare_species(cfg, halo, types, forced, summary["stages"], label, lock)
        else:
            for name in ("particles", "smoothing"):
                summary["stages"][name] = {"status": "not needed", "seconds": 0.0, "peak_rss_gib": None}
            log.info("%severy projection is cached; particles and smoothing not needed", label)
        for axis in axes:
            record = summary["products"].setdefault(halo.name(axis), {})
            try:
                record.update(_run_axis(cfg, halo, plans[axis], cosmo, forced, label + f"{axis} ",
                                        summary["stages"]))
            except Exception as exc:  # one failing axis does not stop the others
                log.error("%s%s failed: %s", label, axis, exc)
                record["error"] = traceback.format_exc()
                record.setdefault("stages", {})
        if lock is not None and not cfg["keep_particles"] and not any("error" in r for r in summary["products"].values()):
            _remove_created(cfg, halo, created, lock, label)
    finally:
        if lock is not None:
            lock.release()
    return summary


def _read_projection(path, sig):
    """Open the projection file and check it is the one this run expects."""
    with h5py.File(path, "r") as f:
        found = f.attrs.get("signature")
        found = found.decode() if isinstance(found, bytes) else found
        if not bool(f.attrs.get("complete", False)) or found != sig:
            raise RuntimeError(f"projection file {path} has signature {found!r}, expected {sig!r} "
                               "(replaced or incomplete); rerun with --force projection")
        return (f["Sigma_fine"][:].astype(np.float64), f["Sigma_environment"][:].astype(np.float64),
                f["x_fine_pkpc"][:], f["x_environment_pkpc"][:], json.loads(f.attrs["mass_ledger_json"]))


def _clear_catalog(outdir: Path):
    for name in CATALOG_OUTPUTS:
        path = outdir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _run_axis(cfg, halo, plan, cosmo, forced, label, halo_stages) -> dict:
    stages: dict = {}
    record = {"stages": stages}
    axis = plan["axis"]
    p = cfg["projection"]
    fine, environment = plan["grids"]
    proj_file, proj_sig = plan["proj_file"], plan["proj_sig"]

    with stage("projection", stages, label) as rec:
        remove_stale_partials(proj_file.parent)
        if not forced("projection") and is_complete(proj_file, proj_sig):
            rec["status"] = "cached"
        else:
            species = [(pt, particle_path(cfg, halo, pt), smoothing_path(cfg, halo, pt)) for pt in cfg["particle_types"]]
            sigma, ledger = project(species, axis, (fine, environment), p, halo.R200c_pkpc,
                                    halo.physical_box_pkpc, cfg["threads"], cfg["read_chunk"])
            with atomic_h5(proj_file, proj_sig) as f:
                for g in (fine, environment):
                    f.create_dataset(f"Sigma_{g.name}", data=sigma[g.name].astype(np.float32),
                                     compression="gzip", compression_opts=4, shuffle=True)
                    f.create_dataset(f"x_{g.name}_pkpc", data=g.centers)
                f.attrs["mass_ledger_json"] = to_json(ledger)
                f.attrs.update(axis=axis, halo=halo.key(), tnglens_version=__version__)
            del sigma
        rec["file"] = str(proj_file)

    outdir = output_dir(cfg, halo, axis)
    lens_file = outdir / "lens.h5"
    lens_sig = plan["lens_sig"]
    with FileLock(output_lock_path(cfg, halo, axis)) as out_lock:
        out_lock.exclusive()  # one run at a time writes this product directory
        with stage("maps", stages, label) as rec:
            remove_stale_partials(outdir)
            if not forced("maps") and is_complete(lens_file, lens_sig):
                rec["status"] = "cached"
            else:
                sigma_fine, sigma_env, x_fine, x_env, ledger = _read_projection(proj_file, proj_sig)
                x, maps, info = lens_fields(sigma_fine, sigma_env, x_fine, x_env, cosmo["kpc_per_arcsec"],
                                            cosmo["sigma_critical_inf"], cfg["threads"])
                del sigma_env
                attrs = lens_attributes(halo, axis, cosmo, cfg, info, ledger)
                attrs["projection_signature"] = proj_sig
                with atomic_h5(lens_file, lens_sig) as f:
                    write_lens_file(f, x, maps, sigma_fine, cosmo["efficiencies"], attrs)
                del maps, sigma_fine
                rec.update({k: info[k] for k in ("kappa_vs_div_alpha_median", "kappa_vs_div_alpha_p99",
                                                  "fft_shear_vs_jacobian_median", "trusted_half_width_arcsec")})
                log.info("%skappa vs div(alpha)/2 in trusted region: median %.3g, p99 %.3g", label,
                         info["kappa_vs_div_alpha_median"], info["kappa_vs_div_alpha_p99"])
        record["lens_file"] = str(lens_file)

        if cfg.get("catalog") is not None:
            stamp = outdir / "catalog.done.json"
            with stage("catalog", stages, label) as rec:
                from .catalog import build_catalog, resolve_config  # lazy: the catalog half is optional
                cat_sig = signature({"version": CATALOG_VERSION, "tnglens": __version__, "lens": lens_sig,
                                     "catalog": resolve_config(cfg["catalog"])})
                try:
                    done = stamp.exists() and json.loads(stamp.read_text()).get("signature") == cat_sig
                except (OSError, ValueError):
                    done = False
                if not forced("catalog") and done:
                    rec["status"] = "cached"
                else:
                    stamp.unlink(missing_ok=True)
                    _clear_catalog(outdir)
                    if signature_of(lens_file) != lens_sig:
                        raise RuntimeError(f"{lens_file} does not have the expected signature")
                    result = build_catalog(str(lens_file), cfg["catalog"], str(outdir))
                    atomic_text(stamp, to_json({"signature": cat_sig, "tnglens_version": __version__,
                                                "result": result}, indent=2))
                    record["catalog"] = result
        atomic_text(outdir / "stages.json", to_json({**record, "stages": {**halo_stages, **stages}}, indent=2))
    return record


# ----------------------------------------------------------------------------- many halos
def _init_worker(level):
    logging.basicConfig(level=level, format="%(asctime)s %(processName)s %(message)s", datefmt="%H:%M:%S")


def _failure(key, error):
    return {"halo": key, "error": error, "stages": {}, "products": {}}


def _safe_run(cfg, halo, axes, force):
    try:
        return run_halo(cfg, halo, axes, force)
    except Exception:
        log.error("[%s] failed:\n%s", halo.key(), traceback.format_exc())
        return _failure(halo.key(), traceback.format_exc())


def failed_names(summaries) -> list[str]:
    """Halos (lookup or whole-halo failures) and products that failed."""
    out = []
    for s in summaries:
        if "error" in s:
            out.append(s["halo"])
        out += [name for name, product in s["products"].items() if "error" in product]
    return out


def run(cfg: dict, only=None, force: str | None = None) -> list[dict]:
    """Run every requested halo (optionally only those matching ``only``); returns their summaries.

    A halo that cannot be looked up, or fails, is reported in its summary (``error``) and the
    others still run.  Raises ConfigError for problems with the request itself (nothing runs).
    """
    if force is not None and force not in STAGES:
        raise ConfigError(f"force must be one of {STAGES}")
    if not cfg["halos"]:
        raise ConfigError("the config lists no halos")
    jobs, failures = resolve(cfg, only)
    jobs = select(jobs, only)
    if not jobs and not failures:
        raise ConfigError(f"--only {' '.join(only)} matches no product of this config")
    for halo, _ in jobs:
        lens_cosmology(cfg, halo)  # fail early on source redshifts in front of a lens
    results = list(failures)
    if jobs:
        Path(cfg["cache_root"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
        workers = min(cfg["workers"], len(jobs))
        log.info("%d halo(s), %d lens file(s), %d worker(s) x %d thread(s)", len(jobs),
                 sum(len(a) for _, a in jobs), workers, cfg["threads"])
        if workers == 1:
            results += [_safe_run(cfg, halo, axes, force) for halo, axes in jobs]
        else:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[var] = str(cfg["threads"])  # inherited by the spawned workers
            context = multiprocessing.get_context("spawn")
            done = {}
            with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker,
                                     initargs=(logging.getLogger().level or logging.INFO,)) as pool:
                futures = {pool.submit(_safe_run, cfg, halo, axes, force): halo.key() for halo, axes in jobs}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        done[key] = future.result()
                    except Exception as exc:  # the worker process died (killed, out of memory, ...)
                        log.error("[%s] worker failed: %r", key, exc)
                        done[key] = _failure(key, f"worker process failed: {exc!r}")
            results += [done[halo.key()] for halo, _ in jobs]
    failed = failed_names(results)
    n_products = sum(len(s["products"]) for s in results)
    if failed:
        log.error("finished: %d of %d lens product(s) ok; failed: %s",
                  n_products - sum(1 for s in results for p in s["products"].values() if "error" in p),
                  n_products, ", ".join(failed))
    else:
        log.info("finished: %d lens product(s) ok", n_products)
    return results
