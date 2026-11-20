# tnglens

Gravitational-lensing maps of TNG-Cluster halos, and multiple-image catalogs simulated from them.

For each halo, snapshot and projection axis, tnglens projects all FoF particles of the halo's
original zoom, solves for the lensing fields, and writes one HDF5 *lens file* with the surface
density, convergence, deflection, potential, shear and magnification maps. If the config has a
`catalog:` section, the second half of the package (`tnglens.catalog`) then places point
sources behind the lens and writes their multiple images and truth tables. By default the catalog
follows the tngc014 mock: systems kept by class quota, images tied to single satellites, and knots
(`configs/tngc014_mock.yaml`, [docs/catalog.md](docs/catalog.md)).

It is a cleaned-up extraction of the map-making part of the `cosmic_telescope` research
pipeline. The physics and numerical method are unchanged; see [Differences from cosmic_telescope](#differences-from-cosmic_telescope).

## Installation

```bash
pip install -e .            # numpy, scipy, h5py, astropy, pyyaml, py-sphviewer2
pip install -e '.[test]'    # adds pytest
pytest                      # unit tests, no simulation files needed
```

Python >= 3.11. Only NumPy/SciPy, plus [py-sphviewer2](https://pypi.org/project/py-sphviewer2/)
for rendering resolved SPH kernels. TNG-Cluster files are read directly with h5py;
`illustris_python` is not needed.

## Quick start

```bash
tnglens list                          # which snapshot/zoom combinations are on disk
cp configs/example.yaml my.yaml       # edit the halos list
tnglens run my.yaml                   # all halos in the config
tnglens run my.yaml --only 'tngc_z014_s073_yz'      # one product (glob patterns allowed)
tnglens run my.yaml --force maps      # recompute lens files (and catalogs), reuse the rest
tnglens run my.yaml --force           # recompute everything
```

`tnglens run` exits with 0 when every product is done, 1 when some halo or product failed while
running (the others still finish; the last line, `summary: ...`, names the failures), and 2 for
user errors: an invalid config (including an unknown key in `catalog:`), an `--only` that matches
no product, `--workers` < 1, or a halo that cannot be looked up on disk. User errors print one
line (`tnglens: error: ...`; `-v` adds the traceback). A halo that cannot be looked up is
reported and skipped; the other halos still run. With `--only`, halos whose product names cannot
match are not looked up at all.

`tnglens list [--base DIR] [--snapshot N] [--check]` prints, per snapshot, the zoom indices
that have both a group catalog (`groups_NNN/fof_subhalo_tab_NNN.<zoom>.hdf5`) and a FoF
particle file (`snapdir_NNN/snapshot_NNN.<zoom>.hdf5`). Only a few snapshots have particle
files for every zoom. `--check` also opens each particle file and checks, for the gravitating
types 0, 1, 2, 4, 5, the `Coordinates` (and, where the header mass table is zero, `Masses`)
lengths against its header; tracers (type 3) have no coordinates and are not checked. It reads
headers only: about 7 s for the 349 zooms of snapshot 73.

## Configuration

A YAML file; every key except `halos` has a default. `configs/example.yaml` lists them all.

| key | default | meaning |
|---|---|---|
| `simulation.name` | `tngc` | only TNG-Cluster is supported |
| `simulation.base` | `/ironwolf/TNGC/output` | directory with `groups_NNN/`, `snapdir_NNN/` |
| `simulation.offsets` | `<base>/../postprocessing/offsets` | offsets directory (used to map halo ids to zooms and to check particle counts) |
| `cache_root` | `/data/dev/tnglens` | bulk caches: particles, smoothing lengths, projections |
| `output_root` | `./output` | lens files go to `<output_root>/<name>/lens.h5` |
| `cosmology` | `native` | `native` (from the snapshot header: Planck 2015, H0 = 67.74, Om0 = 0.3089) or `{H0, Om0}` (flat LCDM). Changes only angular scales and critical densities |
| `halos` | — | list of `{snapshot, zoom \| halo_id, axes}`; each of snapshot/zoom/halo_id may be a list (all combinations are run) |
| `axes` | `[xy, yz, zx]` | default projection axes |
| `source_redshifts` | `[2, 4, 9]` | one magnification map per redshift; each must be behind the lens |
| `particle_types` | `[0, 1, 2, 4, 5]` | gas, dark matter, low-resolution dark matter, stars and wind, black holes (3 = tracers is not allowed) |
| `projection.neighbors` | 64 | smoothing length = distance to the k-th nearest particle of the same type |
| `projection.support_floor_pkpc` | 1.48 | minimum smoothing length (the TNG-Cluster softening) |
| `projection.gas_support` | `neighbors` | `cells`: gas uses `2 (3 m / 4 pi rho)^(1/3)` instead (floor still applies) |
| `projection.target_cells_per_h` | 8 | kernels with h >= this many pixels are rendered by sphviewer2, smaller ones by the stencil/CIC deposit |
| `projection.core_width_arcsec` | 400 | side of the lens map |
| `projection.core_resolution` | 4096 | pixels per side (power of two) |
| `projection.environment_width_r200` | 4 | side of the coarse environment grid in R200c (at least 1.5 core widths) |
| `projection.environment_resolution` | 2048 | pixels per side (power of two) |
| `projection.depth_half_r200` | 2 | particles whose centres are within ±2 R200c along the line of sight |
| `workers` | 1 | halos processed at once, each in its own process |
| `threads` | 8 | threads per worker (tree search, sphviewer2, FFTs) |
| `read_chunk` | 2000000 | particles read at a time |
| `keep_particles` | `true` | `false` deletes the particle and smoothing cache files *this run built* once all of the halo's lens files are written (caches that existed before, or that another run is using, are kept) |
| `catalog` | absent | passed unchanged to `tnglens.catalog.build_catalog(lens_path, cfg, outdir)`; checked with `tnglens.catalog.resolve_config` when the config is loaded, so a typo fails before any stage runs |

Integer settings reject booleans (`neighbors: true` is an error, not 1).

A halo given by `zoom` is that zoom's primary target (`GroupPrimaryZoomTarget == 1`). A halo
given by `halo_id` (the global FoF group index at that snapshot) may be any group; its
particles are still all FoF particles of the zoom it belongs to, centred on that group, and
its product names get a `_h<halo_id>` suffix unless it is the primary.

## Stages and caches

| stage | product | depends on |
|---|---|---|
| `particles` | `<cache_root>/particles/tngc_zZZZ_sSSS/type<T>.h5`: coordinates [pkpc, relative to the halo centre, float32], masses [Msun], gas cell size | snapshot file, halo centre |
| `smoothing` | `<cache_root>/smoothing/tngc_zZZZ_sSSS/n<k>/type<T>.h5`: distance to the k-th neighbour [pkpc] | particles, `neighbors` |
| `projection` | `<cache_root>/projections/<name>/sigma-<signature[:16]>.h5`: core and environment surface density, mass ledger | the above, `projection.*`, axis, cosmology (through the core width in kpc), tnglens version |
| `maps` | `<output_root>/<name>/lens.h5` | projection, cosmology, source redshifts, the lens-relevant config, tnglens version |
| `catalog` | `<output_root>/<name>/{truth/, observed/, README.md}` and the stamp `catalog.done.json` | lens file, resolved `catalog` section, tnglens version |

Every product stores a signature (a hash of its inputs) and is reused only when the signature
matches, so changing e.g. `support_floor_pkpc` reprojects without rerunning the tree search, and
changing only `source_redshifts` rewrites only the lens files. The particle and smoothing stages
run only when some projection of the halo has to be computed: a rerun whose projections are
cached never opens the particle caches (they are reported as `not needed`, and may be absent).
The projection file name contains its signature, so configs with different cosmologies or
settings keep separate projections, and the maps stage checks the signature of the file it
reads. Before a catalog is rebuilt, the previous `truth/`, `observed/` and `README.md` of that
product directory are deleted (nothing else is). `<output_root>/<name>/stages.json` records the
status (`done`, `cached`, `not needed`), wall time and peak memory of every stage, particles and
smoothing included.

Files are written under a temporary name (`<file>.partial-<host>-<pid>`) and renamed when
complete; partial files of processes that no longer exist on this host are deleted when a stage
next writes into that directory. Runs that share a `cache_root` are safe to run at once: a lock
file `<cache_root>/locks/<key>.lock` is held exclusively while a halo's particle/smoothing caches
are built and shared while they are read (a second run needing them waits, then reuses them), and
`<output_root>/.locks/<name>.lock` serialises writers of one product directory.

Measured for zoom 14 at snapshot 73 (1.0e8 particles of types 0, 1, 4, 5; 8 threads on a
busy 32-core machine): particles 48 s / 0.4 GiB peak, 1.2 GB on disk; smoothing 14 min /
2.5 GiB, 0.33 GB; projection 3.8 min / 2.2 GiB, 48 MB per axis; lens file 21 s / 3.7 GiB,
335 MB. Type 2 adds little: zoom 14 has 2.3e5 low-resolution particles (9.3e14 Msun, variable
masses), whose 64-neighbour search takes under a second.

Parallelism: each halo (all of its axes) runs in one process; `workers` processes run at once.
Memory scales with `workers`, so check what is free before raising it.

## Method

1. **Particles.** All particles of the zoom's FoF file (`snapshot_NNN.<zoom>.hdf5`, i.e. the
   whole original zoom FoF, not just the target group, and no outer fuzz). Coordinates are
   converted to physical kpc relative to `GroupPos` with the periodic minimum image; masses to
   Msun (header `MassTable` where it is non-zero, i.e. high-resolution dark matter; the `Masses`
   dataset otherwise, e.g. for the variable-mass low-resolution dark matter of type 2).
2. **Smoothing lengths.** Exact distance to the 64th nearest neighbour of the same type in the
   full periodic box (`scipy.spatial.cKDTree`), over all FoF particles of the zoom. Black holes
   get zero. The support is `h = max(distance, 1.48 pkpc)`.
3. **Projection.** Two grids centred on the halo: the core (400" at 4096 px) and the
   environment (4 R200c at 2048 px). Particles with centres within ±2 R200c along the line of
   sight are deposited: by sphviewer2 when `h >= 8` pixels, otherwise by a stencil of the
   projected compact kernel `(1 - r²/h²)^{3/2}` normalised over all its cells before clipping
   (`1.5 <= h < 8` pixels) or cloud-in-cell (`h < 1.5` pixels). sphviewer2's kernel is similar
   but not identical: for one particle it differs from the stencil kernel by up to ~5% of the
   peak at h = 8 pixels (~12% for h >= 20 pixels) and conserves its mass to better than 0.4%. The mass
   ledger records, per type and grid, the selected mass, the mass of fully contained kernels, and
   the deposited mass; if the deposit is not between them (1% tolerance) the projection fails
   with an error and that lens file is not written.
4. **Fields.** kappa = Sigma / Sigma_crit,inf with Sigma_crit,inf = c² / (4 pi G D_l). The
   potential and deflection are free-space Green-function convolutions on zero-padded grids
   (no periodic images, no mean subtracted). The mass outside the core comes from the
   environment grid multiplied by the exact fraction of each coarse cell outside the core; its
   fields are solved on the coarse grid and interpolated onto the core. Two coarse cells inside
   the core edge are excluded from the *trusted* region.
5. **Derived maps.** The deflection is stored in float32; gamma1, gamma2 and the magnification
   maps are computed from the Jacobian of that stored map, as the exact derivative of its
   cubic B-spline (`scipy.ndimage.spline_filter`, order 3) at the pixel centres.

## The lens file

`<output_root>/<name>/lens.h5`, with `name = tngc_z{zoom:03d}_s{snap:03d}_{axis}` (plus
`_h<halo_id>` for a non-primary halo). `axis` is `xy`, `yz` or `zx`: map x and y are the
simulation axes named by the two letters, the line of sight is the remaining one.

Datasets (2D maps are float32, gzip-compressed, indexed `[y, x]`):

| dataset | unit | meaning |
|---|---|---|
| `x`, `y` | arcsec | pixel centres relative to the halo centre, ascending, uniform (float64, 1D) |
| `Sigma` | Msun/pkpc² | deposited surface density |
| `kappa` | — | Sigma / Sigma_crit,inf |
| `alpha_x`, `alpha_y` | arcsec | deflection |
| `potential` | arcsec² | lensing potential (arbitrary additive constant; zero at the map centre) |
| `gamma1`, `gamma2` | — | shear from the Jacobian of the stored deflection: gamma1 = (psi_xx − psi_yy)/2, gamma2 = psi_xy |
| `mu_zs<z>` | — | signed magnification 1/det(A) for a source at z (`%g` format, e.g. `mu_zs2`, `mu_zs2.5`) |

**Normalisation.** kappa, alpha, potential and gamma are for D_ds/D_s = 1: multiply by
`efficiency(z_s) = D_ds/D_s` (`tnglens.cosmology.efficiency`) for a source at z_s. The lens
equation is beta = theta − efficiency(z_s) alpha(theta). Each `mu_zs*` dataset has attributes
`source_redshift` and `efficiency`.

**Self-consistency.** Image positions obey the deflection map, so everything that describes an
image must come from it. The shear of the FFT kernels differs from the derivatives of alpha
(zoom 14, snapshot 73: median ~2.5e-5, 99th percentile ~4e-4 inside the trusted region), which
is small but moves magnifications near critical curves; the stored gamma and mu therefore come
from the Jacobian of the stored alpha. kappa is kept as the deposited density; its difference from half
the divergence of alpha is stored as `kappa_vs_div_alpha_{median,p99,max}` (absolute values
inside the trusted region), and the FFT-shear difference as `fft_shear_vs_jacobian_{median,p99}`.

Main attributes: `lens_redshift` (= snapshot redshift), `snapshot`, `zoom_index`, `halo_id`,
`primary_target`, `axis`, `line_of_sight_axis`, `H0`, `Om0`, `cosmology_source` (`native` or
`config`), `kpc_per_arcsec`, `sigma_critical_inf_Msun_pkpc2`, `M200c_Msun`, `R200c_pkpc`,
`halo_position_ckpc_h`, `box_ckpc_h`, `h`, `scale_factor`, `group_catalog_base`,
`group_file`, `snapshot_file`, `particle_types`, `particle_scope`, `source_redshifts`,
`trusted_half_width_arcsec`, `core_half_width_arcsec`, `support_floor_pkpc`, `neighbors`,
`gas_support`, `mass_ledger_json`, `solver_json`, `config_json` (the part of the config the
lens file depends on: `simulation`, `cosmology`, `source_redshifts`, `particle_types`,
`projection`; it is part of the signature, so a reused file never carries a stale one),
`projection_signature`, `tnglens_version`, and the plain-language `description`, `coordinates`, `gamma_convention`,
`consistency`.

Use the fields only inside `trusted_half_width_arcsec` (the core half-width minus two
environment cells, ~198" for the fiducial settings).

## What tnglens does not do

* **No line of sight.** One lens plane at the snapshot redshift; no structure in front of or
  behind the cluster, no multi-plane lensing.
* **No outer mass.** Only the FoF particles of the original zoom are used: the zoom's outer
  (non-FoF) particles, and anything beyond the zoom region, are left out, as is mass farther than
  `depth_half_r200` along the line of sight. The convergence therefore has no mass sheet from
  the surrounding large-scale structure.
* **No smoothing-sensitivity certification.** The 64-neighbour, 1.48 pkpc support is a choice;
  substructure on scales of a few kpc depends on it.
* Only TNG-Cluster, and only snapshots for which both group catalogs and particle files are on
  disk (`tnglens list`).

## Differences from cosmic_telescope

* The small-kernel deposit is a vectorised NumPy port of the numba `deposit_small` (identical
  to round-off, ~2.5x slower), so numba is not a dependency.
* gamma and mu use the exact derivative of the cubic B-spline of alpha at the pixel centres
  instead of central differences of ±0.02" of the same spline. That is the derivative the image
  finder in `tnglens.lensmap` evaluates (they agree to ~1e-13); the two forms differ by O(step²),
  up to ~1e-3 near sharp features.
* Particle types are configurable. The default, 0, 1, 2, 4, 5, is the old fiducial set. For
  zoom 14 at snapshot 73 no type-2 kernel reaches either grid (the nearest low-resolution
  particle is 3.95 R200c from the centre), so its maps equal those for types 0, 1, 4, 5; a larger
  environment or depth would otherwise drop real mass.
* Projection caches store total surface density only (not per type).
* Campaign, variant, legacy-cache and merger machinery is gone.

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
