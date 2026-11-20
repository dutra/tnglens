"""Command line: ``tnglens list`` and ``tnglens run``."""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import DEFAULTS


def _ranges(values):
    """[0, 1, 2, 5, 7, 8] -> '0-2,5,7-8'."""
    out, start, prev = [], None, None
    for v in values:
        if start is None:
            start = prev = v
        elif v == prev + 1:
            prev = v
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = v
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def cmd_list(args):
    from .halos import inventory
    base, offsets = args.base, None
    if args.config:
        from .config import load
        cfg = load(args.config)
        base, offsets = cfg["simulation"]["base"], cfg["simulation"]["offsets"]
    base = base or DEFAULTS["simulation"]["base"]
    inv = inventory(base, snapshots=set(args.snapshot) if args.snapshot else None,
                    offsets_dir=offsets, check=args.check)
    if not inv:
        print(f"no snapshot under {base} has both group catalogs and FoF particle files")
        return 1
    print(f"{base}: zoom/snapshot combinations with a group catalog and FoF particles"
          + (" (particle files checked)" if args.check else ""))
    print(f"{'snapshot':>8} {'zooms':>5}  zoom indices")
    for snap, zooms in inv.items():
        print(f"{snap:>8} {len(zooms):>5}  {_ranges(zooms)}")
    return 0


def cmd_run(args):
    """Exit status: 0 all products done; 1 some product or halo failed while running;
    2 a user error (bad config, --only matching nothing, a halo that cannot be looked up)."""
    from .config import load
    from .pipeline import run
    cfg = load(args.config)
    if args.workers is not None:
        cfg["workers"] = args.workers
    summaries = run(cfg, only=args.only, force=args.force)
    failed, lookup_failed, ok = [], [], 0
    for s in summaries:
        if s.get("lookup"):
            lookup_failed.append(s["halo"])
            print(f"FAILED {s['halo']}: {s['error']}")
            continue
        if "error" in s:
            failed.append(s["halo"])
            print(f"FAILED {s['halo']}:\n{s['error']}")
        for name, product in s["products"].items():
            if "error" in product:
                failed.append(name)
                print(f"FAILED {name}:\n{product['error']}")
                continue
            ok += 1
            stages = {**s["stages"], **product["stages"]}
            text = ", ".join(f"{k} {v['status']} {v['seconds']:.0f}s/{v['peak_rss_gib']}GiB" for k, v in stages.items())
            print(f"{name}: {product['lens_file']}  [{text}]")
    bad = lookup_failed + failed
    print(f"summary: {ok} product(s) done" + (f", {len(bad)} failed: {', '.join(bad)}" if bad else ""))
    return 2 if lookup_failed else 1 if failed else 0


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tnglens", description="Lens maps from TNG-Cluster halos.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="which zoom/snapshot combinations are usable on disk")
    p.add_argument("--base", help=f"simulation output directory (default {DEFAULTS['simulation']['base']})")
    p.add_argument("--config", help="take the simulation base from this config")
    p.add_argument("--snapshot", type=int, action="append", help="only this snapshot (repeatable)")
    p.add_argument("--check", action="store_true", help="open particle files and check their lengths (slow)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="build lens files (and catalogs) for the halos in a config")
    p.add_argument("config", help="YAML config file")
    p.add_argument("--only", action="append", metavar="NAME",
                   help="only products whose name matches this glob, e.g. tngc_z014_s073_yz (repeatable)")
    p.add_argument("--force", nargs="?", const="particles", default=None,
                   choices=("particles", "smoothing", "projection", "maps", "catalog"),
                   help="recompute from this stage on (bare --force: everything)")
    p.add_argument("--workers", type=_positive_int, help="override the config's workers (>= 1)")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from .config import ConfigError
    try:
        return args.func(args)
    except (ConfigError, OSError) as exc:  # user errors: one line, no traceback (-v shows it)
        if args.verbose:
            logging.getLogger("tnglens").exception("error")
        print(f"tnglens: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
