# Running the pipeline

## Commands

```bash
tnglens list                          # which snapshot/zoom combinations are on disk
tnglens run my.yaml                   # all halos in the config
tnglens run my.yaml --only 'tngc_z014_s073_yz'      # one product (glob patterns allowed)
tnglens run my.yaml --force maps      # recompute lens files (and catalogs), reuse the rest
tnglens run my.yaml --force           # recompute everything
tnglens run my.yaml --workers 4       # override the config's workers
```

`--force STAGE` recomputes from that stage on (`particles`, `smoothing`, `projection`, `maps`,
`catalog`); a bare `--force` recomputes everything. `-v` turns on debug logging.

### Exit status of `tnglens run`

`tnglens run` exits with 0 when every product is done, 1 when some halo or product failed while
running (the others still finish; the last line, `summary: ...`, names the failures), and 2 for
user errors: an invalid config (including an unknown key in `catalog:`), an `--only` that matches
no product, `--workers` < 1, or a halo that cannot be looked up on disk. User errors print one
line (`tnglens: error: ...`; `-v` adds the traceback). A halo that cannot be looked up is
reported and skipped, and the other halos still run. With `--only`, halos whose product names
cannot match are not looked up at all.

### `tnglens list`

`tnglens list [--base DIR] [--config CFG] [--snapshot N] [--check]` prints, per snapshot, the
zoom indices that have both a group catalog (`groups_NNN/fof_subhalo_tab_NNN.<zoom>.hdf5`) and a
FoF particle file (`snapdir_NNN/snapshot_NNN.<zoom>.hdf5`). Only a few snapshots have particle
files for every zoom. `--check` also opens each particle file and checks, for the gravitating
types 0, 1, 2, 4, 5, the `Coordinates` (and, where the header mass table is zero, `Masses`)
lengths against its header. Tracers (type 3) have no coordinates and are not checked. It reads
headers only, which takes about 7 s for the 349 zooms of snapshot 73.

## Stages and caches

| stage | product | depends on |
|---|---|---|
| `particles` | `<cache_root>/particles/tngc_zZZZ_sSSS/type<T>.h5`: coordinates [pkpc, relative to the halo centre, float32], masses [Msun], gas cell size | snapshot file, halo centre |
| `smoothing` | `<cache_root>/smoothing/tngc_zZZZ_sSSS/n<k>/type<T>.h5`: distance to the k-th neighbour [pkpc] | particles, `neighbors` |
| `projection` | `<cache_root>/projections/<name>/sigma-<signature[:16]>.h5`: core and environment surface density, mass ledger | the above, `projection.*`, axis, cosmology (through the core width in kpc), tnglens version |
| `maps` | `<output_root>/<name>/lens.h5` | projection, cosmology, source redshifts, the lens-relevant config, tnglens version |
| `catalog` | `<output_root>/<name>/{truth/, observed/, README.md}` and the stamp `catalog.done.json` | lens file, resolved `catalog` section, tnglens version |

Projection caches store the total surface density only, not one map per particle type.

Every product stores a signature (a hash of its inputs) and is reused only when the signature
matches. Changing `support_floor_pkpc`, for example, reprojects without rerunning the tree
search, and changing only `source_redshifts` rewrites only the lens files. The particle and
smoothing stages run only when some projection of the halo has to be computed. A rerun whose
projections are cached never opens the particle caches (they are reported as `not needed`, and
may be absent). The projection file name contains its signature, so configs with different
cosmologies or settings keep separate projections, and the maps stage checks the signature of
the file it reads. Before a catalog is rebuilt, the previous `truth/`, `observed/` and
`README.md` of that product directory are deleted (nothing else is).
`<output_root>/<name>/stages.json` records the status (`done`, `cached`, `not needed`), wall
time and peak memory of every stage, particles and smoothing included.

## Atomic writes and locking

Files are written under a temporary name (`<file>.partial-<host>-<pid>`) and renamed when
complete. Partial files of processes that no longer exist on this host are deleted when a stage
next writes into that directory. Runs that share a `cache_root` can run at the same time. A lock
file `<cache_root>/locks/<key>.lock` is held exclusively while a halo's particle and smoothing
caches are built and shared while they are read, so a second run that needs them waits and then
reuses them. `<output_root>/.locks/<name>.lock` serialises writers of one product directory.

## Parallelism

Each halo (all of its axes) runs in one process, and `workers` processes run at once. Memory
scales with `workers`, so check what is free before raising it. Each worker uses `threads`
threads for the tree search, sphviewer2 and the FFTs.

## Cost

Measured for zoom 14 at snapshot 73 (1.0e8 particles of types 0, 1, 4, 5; 8 threads on a
busy 32-core machine):

| stage | wall time | peak memory | on disk |
|---|---|---|---|
| particles | 48 s | 0.4 GiB | 1.2 GB |
| smoothing | 14 min | 2.5 GiB | 0.33 GB |
| projection | 3.8 min | 2.2 GiB | 48 MB per axis |
| lens file | 21 s | 3.7 GiB | 335 MB |

Type 2 adds little: zoom 14 has 2.3e5 low-resolution particles (9.3e14 Msun, variable masses),
whose 64-neighbour search takes under a second.

## Package layout

```
src/tnglens/
  config.py      YAML loading, defaults, validation, halo request expansion
  halos.py       file layout, halo lookup, inventory of what is on disk
  particles.py   particle reader and neighbour distances (stages particles, smoothing)
  deposition.py  stencil/CIC deposit of small kernels
  projection.py  core and environment grids, sphviewer2 + small-kernel projection, mass ledger
  fields.py      isolated Green-function solver, environment partition, spline Jacobian
  maps.py        lens fields and the lens-file writer
  pipeline.py    stages, caching, parallel execution
  cli.py         `tnglens list`, `tnglens run`
  cache.py       signatures, atomic writes, stage timing
  cosmology.py   distances, critical density, time-delay factor
  lensmap.py, images.py, sources.py, members.py, satellites.py, catalog.py   the catalog half
```
