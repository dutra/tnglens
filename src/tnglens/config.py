"""Reading and validating the YAML run configuration.

``load(path)`` returns a plain dict with every default filled in (see ``DEFAULTS`` and the
README for the meaning of each key).  ``requests(cfg)`` expands the ``halos`` list into one
entry per (snapshot, zoom or halo id) with its projection axes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

AXES = ("xy", "yz", "zx")
PARTICLE_TYPES = (0, 1, 2, 4, 5)  # gas, DM, low-res DM, stars+wind, black holes (3 = tracers, no coordinates)

DEFAULTS = {
    "simulation": {"name": "tngc", "base": "/ironwolf/TNGC/output", "offsets": None},
    "cache_root": "/data/dev/tnglens",
    "output_root": "./output",
    "cosmology": "native",
    "halos": [],
    "axes": ["xy", "yz", "zx"],
    "source_redshifts": [2.0, 4.0, 9.0],
    "particle_types": [0, 1, 2, 4, 5],
    "projection": {
        "neighbors": 64,
        "support_floor_pkpc": 1.48,
        "gas_support": "neighbors",
        "target_cells_per_h": 8,
        "core_width_arcsec": 400.0,
        "core_resolution": 4096,
        "environment_width_r200": 4.0,
        "environment_resolution": 2048,
        "depth_half_r200": 2.0,
    },
    "workers": 1,
    "threads": 8,
    "read_chunk": 2_000_000,
    "keep_particles": True,
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Request:
    """One halo to process: snapshot plus either a zoom index or a FoF group id."""

    snapshot: int
    zoom: int | None
    halo_id: int | None
    axes: tuple[str, ...]


def _merge(defaults: dict, given: dict, where: str) -> dict:
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        if key not in defaults:
            raise ConfigError(f"unknown key '{where}{key}'")
        if isinstance(defaults[key], dict) and key != "cosmology":
            if not isinstance(value, dict):
                raise ConfigError(f"'{where}{key}' must be a mapping")
            out[key] = _merge(defaults[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _power_of_two(n) -> bool:
    return _is_int(n) and n >= 16 and n & (n - 1) == 0


def _check_catalog(section):
    """Run the catalog half's own resolver so unknown keys fail before any heavy stage."""
    if not isinstance(section, dict):
        raise ConfigError("'catalog' must be a mapping (passed unchanged to tnglens.catalog.build_catalog)")
    from .catalog import resolve_config  # lazy: the catalog half is optional
    try:
        resolve_config(section)
    except (KeyError, TypeError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise ConfigError(f"catalog: {message}") from None


def validate(cfg: dict) -> dict:
    """Fill defaults, check types and ranges; return a new normalised dict."""
    extra = {k: v for k, v in cfg.items() if k == "catalog"}
    cfg = _merge(DEFAULTS, {k: v for k, v in cfg.items() if k != "catalog"}, "")
    if extra.get("catalog") is not None:
        _check_catalog(extra["catalog"])
        cfg["catalog"] = extra["catalog"]

    if cfg["simulation"]["name"] != "tngc":
        raise ConfigError("simulation.name: only 'tngc' (TNG-Cluster) is supported")
    for key in ("cache_root", "output_root"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    cfg["simulation"]["base"] = str(Path(cfg["simulation"]["base"]).expanduser())

    cosmo = cfg["cosmology"]
    if cosmo is None:
        cfg["cosmology"] = "native"
    elif isinstance(cosmo, dict):
        if set(cosmo) != {"H0", "Om0"}:
            raise ConfigError("cosmology must be 'native' or a mapping with exactly H0 and Om0")
        if not all(_is_number(v) for v in cosmo.values()):
            raise ConfigError("cosmology: H0 and Om0 must be numbers")
        cfg["cosmology"] = {"H0": float(cosmo["H0"]), "Om0": float(cosmo["Om0"])}
        if not (cfg["cosmology"]["H0"] > 0 and 0 < cfg["cosmology"]["Om0"] <= 1):
            raise ConfigError("cosmology: need H0 > 0 and 0 < Om0 <= 1")
    elif cosmo != "native":
        raise ConfigError("cosmology must be 'native' or {H0: ..., Om0: ...}")

    zs = cfg["source_redshifts"]
    if not isinstance(zs, list) or not zs or not all(_is_number(z) and z > 0 for z in zs):
        raise ConfigError("source_redshifts must be a non-empty list of positive numbers")
    cfg["source_redshifts"] = [float(z) for z in zs]
    if len({f"{z:g}" for z in cfg["source_redshifts"]}) != len(zs):
        raise ConfigError("source_redshifts must be distinct (dataset names use the %g format)")

    pts = cfg["particle_types"]
    if (not isinstance(pts, list) or not pts or not all(_is_int(p) and p in PARTICLE_TYPES for p in pts)
            or len(set(pts)) != len(pts)):
        raise ConfigError("particle_types must be a non-empty list of distinct integers from (0, 1, 2, 4, 5); type 3 is tracers")
    cfg["particle_types"] = sorted(int(p) for p in pts)

    axes = cfg["axes"]
    if not isinstance(axes, list) or not axes or any(a not in AXES for a in axes):
        raise ConfigError(f"axes must be a non-empty list drawn from {AXES}")

    p = cfg["projection"]
    if not (_is_int(p["neighbors"]) and p["neighbors"] >= 1):
        raise ConfigError("projection.neighbors must be a positive integer")
    for key in ("core_resolution", "environment_resolution"):
        if not _power_of_two(p[key]):
            raise ConfigError(f"projection.{key} must be a power of two >= 16 (required by sphviewer2)")
    for key in ("core_width_arcsec", "environment_width_r200", "depth_half_r200", "target_cells_per_h"):
        if not (_is_number(p[key]) and p[key] > 0):
            raise ConfigError(f"projection.{key} must be positive")
    if not (_is_number(p["support_floor_pkpc"]) and p["support_floor_pkpc"] >= 0):
        raise ConfigError("projection.support_floor_pkpc must be >= 0")
    if p["gas_support"] not in ("neighbors", "cells"):
        raise ConfigError("projection.gas_support must be 'neighbors' or 'cells'")
    for key in ("core_width_arcsec", "environment_width_r200", "depth_half_r200", "support_floor_pkpc"):
        p[key] = float(p[key])

    for key in ("workers", "threads", "read_chunk"):
        if not (_is_int(cfg[key]) and cfg[key] >= 1):
            raise ConfigError(f"{key} must be a positive integer")
    if not isinstance(cfg["keep_particles"], bool):
        raise ConfigError("keep_particles must be true or false")

    if not isinstance(cfg["halos"], list):
        raise ConfigError("halos must be a list")
    cfg["halos"] = [_halo_entry(h, i) for i, h in enumerate(cfg["halos"])]
    requests(cfg)  # raises on malformed entries
    return cfg


def _as_int_list(value, what):
    values = value if isinstance(value, list) else [value]
    if not values or not all(_is_int(v) and v >= 0 for v in values):
        raise ConfigError(f"{what} must be a non-negative integer or a list of them")
    return values


def _halo_entry(entry, index):
    where = f"halos[{index}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"{where} must be a mapping")
    unknown = set(entry) - {"zoom", "halo_id", "snapshot", "axes"}
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    if "snapshot" not in entry:
        raise ConfigError(f"{where}: 'snapshot' is required")
    if ("zoom" in entry) == ("halo_id" in entry):
        raise ConfigError(f"{where}: give exactly one of 'zoom' or 'halo_id'")
    out = {"snapshot": _as_int_list(entry["snapshot"], f"{where}.snapshot")}
    if "zoom" in entry:
        out["zoom"] = _as_int_list(entry["zoom"], f"{where}.zoom")
    else:
        out["halo_id"] = _as_int_list(entry["halo_id"], f"{where}.halo_id")
    if "axes" in entry:
        axes = entry["axes"] if isinstance(entry["axes"], list) else [entry["axes"]]
        if not axes or any(a not in AXES for a in axes):
            raise ConfigError(f"{where}.axes must be drawn from {AXES}")
        out["axes"] = list(dict.fromkeys(axes))
    return out


def requests(cfg: dict) -> list[Request]:
    """Expand ``cfg['halos']`` (lists of snapshots/zooms allowed) into individual requests."""
    out: dict[tuple, Request] = {}
    for entry in cfg["halos"]:
        axes = tuple(entry.get("axes", cfg["axes"]))
        for snap in entry["snapshot"]:
            for zoom in entry.get("zoom", [None]):
                for hid in entry.get("halo_id", [None]):
                    key = (snap, zoom, hid)
                    merged = tuple(dict.fromkeys(out[key].axes + axes)) if key in out else axes
                    out[key] = Request(snap, zoom, hid, merged)
    return list(out.values())


def load(path) -> dict:
    """Read and validate a YAML config file (every problem is raised as ConfigError)."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    cfg = validate(raw)
    cfg["config_path"] = str(path.resolve())
    return cfg
