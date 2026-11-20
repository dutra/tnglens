"""Finding halos and checking which TNG-Cluster files are on disk.

TNG-Cluster layout (``base`` is the ``output`` directory):

* ``groups_NNN/fof_subhalo_tab_NNN.i.hdf5`` -- group catalog of original zoom ``i``;
* ``snapdir_NNN/snapshot_NNN.i.hdf5`` -- for ``i < n_zooms`` all FoF particles of zoom ``i``;
  files ``i + n_zooms`` hold the outer (non-FoF) particles, which tnglens never reads;
* ``../postprocessing/offsets/offsets_NNN.hdf5`` -- global group offsets per file.

A halo is a FoF group.  Selecting by zoom index picks the zoom's primary target
(``GroupPrimaryZoomTarget == 1``); selecting by halo id picks that group, whose particles are
still all the FoF particles of the zoom it belongs to, centred on the selected group.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class Halo:
    snapshot: int
    zoom: int
    halo_id: int            # global FoF group index at this snapshot
    local_index: int        # row in the zoom's group catalog file
    primary: bool           # True if this is the zoom's primary target
    position_ckpc_h: list   # GroupPos [ckpc/h]
    M200c_Msun: float
    R200c_pkpc: float
    redshift: float
    scale_factor: float
    h: float
    omega_m: float
    omega_lambda: float
    box_ckpc_h: float
    mass_table: list        # header MassTable [1e10 Msun/h]
    num_part: list          # particles per type in the zoom's FoF file
    contamination_mass_fraction: float
    base: str
    group_file: str
    snapshot_file: str
    extra: dict = field(default_factory=dict)

    @property
    def physical_box_pkpc(self) -> float:
        return self.box_ckpc_h * self.scale_factor / self.h

    @property
    def header(self) -> dict:
        """The cosmology entries that tnglens.cosmology.make('native', header) needs."""
        return {"HubbleParam": self.h, "Omega0": self.omega_m}

    def name(self, axis: str) -> str:
        stem = f"tngc_z{self.zoom:03d}_s{self.snapshot:03d}_{axis}"
        return stem if self.primary else f"{stem}_h{self.halo_id}"

    def key(self) -> str:
        """Identifier of the particle cache (independent of the projection axis)."""
        stem = f"tngc_z{self.zoom:03d}_s{self.snapshot:03d}"
        return stem if self.primary else f"{stem}_h{self.halo_id}"

    def to_dict(self) -> dict:
        return asdict(self)


def _numbered_dirs(base: Path, prefix: str) -> dict[int, Path]:
    """Map snapshot number -> directory for ``groups_*`` / ``snapdir_*`` (suffix parsed as int)."""
    out = {}
    for d in base.glob(f"{prefix}_*"):
        suffix = d.name[len(prefix) + 1:]
        if d.is_dir() and suffix.isdigit():
            out[int(suffix)] = d
    return out


def _numbered_files(directory: Path, stem: str) -> set[int]:
    """Chunk indices i of files ``{stem}.i.hdf5`` in ``directory``."""
    out = set()
    for p in directory.glob(f"{stem}.*.hdf5"):
        middle = p.name[len(stem) + 1:-len(".hdf5")]
        if middle.isdigit():
            out.add(int(middle))
    return out


def group_dir(base, snap: int) -> Path:
    base = Path(base)
    dirs = _numbered_dirs(base, "groups")
    return dirs.get(snap, base / f"groups_{snap:03d}")


def snap_dir(base, snap: int) -> Path:
    base = Path(base)
    dirs = _numbered_dirs(base, "snapdir")
    return dirs.get(snap, base / f"snapdir_{snap:03d}")


def group_path(base, snap: int, zoom: int) -> Path:
    return group_dir(base, snap) / f"fof_subhalo_tab_{snap:03d}.{zoom}.hdf5"


def snapshot_path(base, snap: int, zoom: int) -> Path:
    return snap_dir(base, snap) / f"snapshot_{snap:03d}.{zoom}.hdf5"


def offsets_path(base, snap: int, offsets_dir=None) -> Path | None:
    directory = Path(offsets_dir) if offsets_dir else Path(base).parent / "postprocessing" / "offsets"
    for p in directory.glob("offsets_*.hdf5"):
        suffix = p.name[len("offsets_"):-len(".hdf5")]
        if suffix.isdigit() and int(suffix) == snap:
            return p
    return None


def _n_zooms(base, snap: int, offsets_dir=None) -> int:
    off = offsets_path(base, snap, offsets_dir)
    if off is not None:
        with h5py.File(off, "r") as f:
            return len(f["OriginalZooms/HaloIDs"])
    files = sorted(_numbered_files(group_dir(base, snap), f"fof_subhalo_tab_{snap:03d}"))
    if not files:
        raise FileNotFoundError(f"no group catalog for snapshot {snap} under {base}")
    with h5py.File(group_path(base, snap, files[0]), "r") as f:
        return int(f["Header"].attrs["NumFiles"])


def inventory(base, snapshots=None, offsets_dir=None, check=False) -> dict[int, list[int]]:
    """Zoom indices per snapshot that have both a group catalog and a FoF particle file.

    With ``check=True`` each candidate particle file is opened and its datasets are compared
    with ``NumPart_ThisFile`` (slow; catches truncated downloads).
    """
    base = Path(base)
    groups, snaps = _numbered_dirs(base, "groups"), _numbered_dirs(base, "snapdir")
    out = {}
    for snap in sorted(set(groups) & set(snaps)):
        if snapshots is not None and snap not in snapshots:
            continue
        have_groups = _numbered_files(groups[snap], f"fof_subhalo_tab_{snap:03d}")
        have_parts = _numbered_files(snaps[snap], f"snapshot_{snap:03d}")
        both = have_groups & have_parts
        if not both:
            continue
        n_zooms = _n_zooms(base, snap, offsets_dir)
        zooms = sorted(i for i in both if i < n_zooms)
        if check:
            zooms = [i for i in zooms if not particle_file_problems(snapshot_path(base, snap, i))]
        if zooms:
            out[snap] = zooms
    return out


GRAVITATING_TYPES = (0, 1, 2, 4, 5)  # PartType3 (tracers) has no Coordinates or Masses, only ParentID/TracerID


def particle_file_problems(path, types=GRAVITATING_TYPES) -> list[str]:
    """Problems found in a snapshot chunk (empty list if it looks complete).

    Checks that every type in ``types`` with particles in the header has a ``Coordinates``
    dataset of the advertised length (and ``Masses`` when the header mass table is zero).
    Tracers (type 3) carry no coordinates and are never checked.
    """
    problems = []
    try:
        with h5py.File(path, "r") as f:
            hdr = f["Header"].attrs
            counts = np.asarray(hdr["NumPart_ThisFile"], dtype=np.int64)
            table = np.asarray(hdr["MassTable"], dtype=np.float64)
            for pt in types:
                if counts[pt] == 0:
                    continue
                group = f.get(f"PartType{pt}")
                if group is None or "Coordinates" not in group:
                    problems.append(f"PartType{pt} missing")
                    continue
                if len(group["Coordinates"]) != counts[pt]:
                    problems.append(f"PartType{pt} has {len(group['Coordinates'])} of {counts[pt]} particles")
                if table[pt] == 0:
                    if "Masses" not in group:
                        problems.append(f"PartType{pt} has no Masses and MassTable[{pt}] is 0")
                    elif len(group["Masses"]) != counts[pt]:
                        problems.append(f"PartType{pt} Masses has {len(group['Masses'])} of {counts[pt]} entries")
    except (OSError, KeyError) as exc:
        problems.append(f"unreadable: {exc}")
    return problems


def _group_file_offsets(base, snap: int, offsets_dir=None) -> np.ndarray:
    """Global index of the first group in each zoom's catalog file."""
    off = offsets_path(base, snap, offsets_dir)
    if off is not None:
        with h5py.File(off, "r") as f:
            return f["FileOffsets/Group"][:]
    n = _n_zooms(base, snap, offsets_dir)
    counts = []
    for i in range(n):
        path = group_path(base, snap, i)
        if not path.exists():
            raise FileNotFoundError(f"no offsets file for snapshot {snap} and group file {path} missing; "
                                    "cannot convert halo ids to zooms")
        with h5py.File(path, "r") as f:
            counts.append(int(f["Header"].attrs["Ngroups_ThisFile"]))
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def zoom_of_halo(base, snapshot: int, halo_id: int, offsets_dir=None) -> int:
    """Original zoom whose group catalog holds global FoF group ``halo_id`` (no particle file needed)."""
    file_offsets = _group_file_offsets(base, snapshot, offsets_dir)
    zoom = int(np.searchsorted(file_offsets, halo_id, side="right") - 1)
    if zoom < 0:
        raise ValueError(f"halo id {halo_id} not found at snapshot {snapshot}")
    return zoom


def find_halo(base, snapshot: int, zoom: int | None = None, halo_id: int | None = None,
              offsets_dir=None) -> Halo:
    """Look up a halo by zoom index (its primary target) or by global FoF group id."""
    if (zoom is None) == (halo_id is None):
        raise ValueError("give exactly one of zoom or halo_id")
    base = Path(base)
    file_offsets = _group_file_offsets(base, snapshot, offsets_dir)
    n_zooms = len(file_offsets)
    if halo_id is not None:
        zoom = int(np.searchsorted(file_offsets, halo_id, side="right") - 1)
        if zoom < 0:
            raise ValueError(f"halo id {halo_id} not found at snapshot {snapshot}")
    if not 0 <= zoom < n_zooms:
        raise ValueError(f"zoom {zoom} out of range (snapshot {snapshot} has {n_zooms} zooms)")

    gpath, spath = group_path(base, snapshot, zoom), snapshot_path(base, snapshot, zoom)
    for p, what in ((gpath, "group catalog"), (spath, "particle file")):
        if not p.exists():
            raise FileNotFoundError(f"{what} for zoom {zoom} at snapshot {snapshot} not on disk: {p}")

    with h5py.File(gpath, "r") as f:
        g = f["Group"]
        n_groups = int(f["Header"].attrs["Ngroups_ThisFile"])
        primary_rows = np.flatnonzero(g["GroupPrimaryZoomTarget"][:] == 1)
        if halo_id is None:
            if len(primary_rows) != 1:
                raise ValueError(f"zoom {zoom}: expected one primary target, found {len(primary_rows)}")
            local = int(primary_rows[0])
        else:
            local = int(halo_id - file_offsets[zoom])
            if not 0 <= local < n_groups:
                raise ValueError(f"halo id {halo_id} not in zoom {zoom}")
        row = lambda name: g[name][local]
        position = np.asarray(row("GroupPos"), dtype=np.float64)
        m200, r200 = float(row("Group_M_Crit200")), float(row("Group_R_Crit200"))
        contamination = float(row("GroupContaminationFracByMass"))

    with h5py.File(spath, "r") as f:
        hdr = f["Header"].attrs
        h, a, z = float(hdr["HubbleParam"]), float(hdr["Time"]), float(hdr["Redshift"])
        if not np.isclose(z, 1 / a - 1, rtol=0, atol=1e-8):
            raise ValueError(f"{spath}: header Time and Redshift disagree")
        num_part = [int(n) for n in hdr["NumPart_ThisFile"]]
        info = dict(omega_m=float(hdr["Omega0"]), omega_lambda=float(hdr["OmegaLambda"]),
                    box=float(hdr["BoxSize"]), mass_table=[float(m) for m in hdr["MassTable"]])

    off = offsets_path(base, snapshot, offsets_dir)
    if off is not None:  # the FoF file must be exactly the zoom's FoF particles
        with h5py.File(off, "r") as f:
            expected = f["OriginalZooms/GroupsTotalLengthByType"][zoom]
        if list(map(int, expected)) != num_part:
            raise ValueError(f"{spath}: particle counts differ from OriginalZooms/GroupsTotalLengthByType")

    return Halo(snapshot=snapshot, zoom=zoom, halo_id=int(file_offsets[zoom] + local), local_index=local,
                primary=local in primary_rows, position_ckpc_h=position.tolist(),
                M200c_Msun=m200 * 1e10 / h, R200c_pkpc=r200 * a / h, redshift=z, scale_factor=a, h=h,
                omega_m=info["omega_m"], omega_lambda=info["omega_lambda"], box_ckpc_h=info["box"],
                mass_table=info["mass_table"], num_part=num_part, contamination_mass_fraction=contamination,
                base=str(base), group_file=str(gpath), snapshot_file=str(spath))
