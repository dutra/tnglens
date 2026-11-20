"""Small helpers shared by the stages: signatures, atomic HDF5 writes, stage timing.

Every cached product is an HDF5 file whose attributes carry ``signature`` (a hash of
everything the product depends on) and ``complete``.  A product is reused only when both
match; it is written to a temporary name (``<name>.partial-<host>-<pid>``) and renamed into
place, so an interrupted run never leaves a file that looks complete.  Partial files left by
a killed process (same host, pid no longer alive) are removed the next time anything is
written into the same directory.

``FileLock`` serialises runs that build or delete the same halo's particle and smoothing
caches (``fcntl.flock`` on a file under ``<cache_root>/locks``).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import socket
import time
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger("tnglens")


def jsonable(obj):
    """Convert numpy scalars/arrays, paths and bytes into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def to_json(obj, **kwargs) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, **kwargs)


def signature(obj) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return hashlib.sha256(to_json(obj).encode()).hexdigest()


def is_complete(path, sig: str) -> bool:
    """True if ``path`` is a finished HDF5 product with this signature."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False)) and f.attrs.get("signature") == sig
    except OSError:
        return False


_HOST = socket.gethostname().replace("-", "_") or "localhost"


def _partial_name(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial-{_HOST}-{os.getpid()}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def remove_stale_partials(directory) -> list[Path]:
    """Delete ``*.partial-[<host>-]<pid>`` files in ``directory`` whose writer is dead.

    Only partials written on this host (or the old host-less form) are considered; a partial
    whose process is still alive is left alone.  Returns the removed paths.
    """
    directory = Path(directory)
    removed = []
    if not directory.is_dir():
        return removed
    for p in directory.glob("*.partial-*"):
        tag = p.name.rsplit(".partial-", 1)[1]
        host, _, pid = tag.rpartition("-")
        if not pid.isdigit() or (host and host != _HOST):
            continue
        if int(pid) == os.getpid() or _pid_alive(int(pid)):
            continue
        try:
            p.unlink()
            removed.append(p)
            log.info("removed stale partial file %s", p)
        except FileNotFoundError:
            pass
    return removed


@contextlib.contextmanager
def atomic_h5(path, sig: str):
    """Open ``path`` for writing under a temporary name; mark complete and rename on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_partials(path.parent)
    tmp = _partial_name(path)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["signature"] = sig
            f.attrs["complete"] = False
            yield f
            f.attrs["complete"] = True
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_text(path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_partials(path.parent)
    tmp = _partial_name(path)
    try:
        tmp.write_text(text)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def signature_of(path) -> str | None:
    """The ``signature`` attribute of a complete HDF5 product, or None."""
    try:
        with h5py.File(path, "r") as f:
            if not bool(f.attrs.get("complete", False)):
                return None
            sig = f.attrs.get("signature")
            return sig.decode() if isinstance(sig, bytes) else sig
    except OSError:
        return None


class FileLock:
    """``fcntl.flock`` on ``path`` (created if needed; never deleted, so every process locks the same inode).

    ``exclusive()`` blocks until no one else holds the lock; ``shared()`` converts to a shared lock
    (others may share it, nobody may take it exclusively); ``try_exclusive()`` returns False instead of
    waiting; ``release()`` drops it.  Use as a context manager to release on exit.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)

    def exclusive(self):
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def shared(self):
        fcntl.flock(self._fd, fcntl.LOCK_SH)

    def try_exclusive(self) -> bool:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def release(self):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def available_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def _reset_peak_rss() -> bool:
    """Reset this process's peak-RSS counter (Linux >= 4.0); False if unsupported."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def _peak_rss_gib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


@contextlib.contextmanager
def stage(name: str, record: dict, label: str = ""):
    """Time a stage and record its wall-clock seconds and peak resident memory.

    ``record[name]`` receives ``{"seconds", "peak_rss_gib", "status"}``.  The peak is the
    process high-water mark since the stage started (reset through /proc/self/clear_refs).
    """
    reset = _reset_peak_rss()
    free = available_memory_gib()
    log.info("%s%s: start (available memory %.1f GiB)", label, name, free if free is not None else float("nan"))
    entry = {"status": "running"}
    record[name] = entry
    start = time.monotonic()
    try:
        yield entry
    except BaseException:
        entry["status"] = "failed"
        raise
    finally:
        entry["seconds"] = round(time.monotonic() - start, 2)
        entry["peak_rss_gib"] = round(_peak_rss_gib(), 3) if reset and _peak_rss_gib() is not None else None
        if entry["status"] == "running":
            entry["status"] = "done"
        log.info("%s%s: %s in %.1f s, peak RSS %s GiB", label, name, entry["status"], entry["seconds"],
                 entry["peak_rss_gib"])
