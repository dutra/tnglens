# Configuration

A run is described by one YAML file. Every key except `halos` has a default, and
[`configs/example.yaml`](../configs/example.yaml) lists them all. Relative paths are resolved
against the current working directory.

| key | default | meaning |
|---|---|---|
| `simulation.name` | `tngc` | only TNG-Cluster is supported |
| `simulation.base` | `/ironwolf/TNGC/output` | directory with `groups_NNN/`, `snapdir_NNN/` |
| `simulation.offsets` | `<base>/../postprocessing/offsets` | offsets directory (used to map halo ids to zooms and to check particle counts) |
| `cache_root` | `/data/dev/tnglens` | bulk caches: particles, smoothing lengths, projections |
| `output_root` | `./output` | lens files go to `<output_root>/<name>/lens.h5` |
| `cosmology` | `native` | `native` (from the snapshot header: Planck 2015, H0 = 67.74, Om0 = 0.3089) or `{H0, Om0}` (flat LCDM). Changes only angular scales and critical densities |
| `halos` | required | list of `{snapshot, zoom \| halo_id, axes}`; each of snapshot/zoom/halo_id may be a list (all combinations are run) |
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
| `catalog` | absent | passed unchanged to `tnglens.catalog.build_catalog(lens_path, cfg, outdir)`; checked with `tnglens.catalog.resolve_config` when the config is loaded, so a typo fails before any stage runs. Its keys are listed in [catalog.md](catalog.md) |

Integer settings reject booleans (`neighbors: true` is an error, not 1).

## Halos

A halo given by `zoom` is that zoom's primary target (`GroupPrimaryZoomTarget == 1`). A halo
given by `halo_id` (the global FoF group index at that snapshot) may be any group. Its
particles are still all FoF particles of the zoom it belongs to, centred on that group, and
its product names get a `_h<halo_id>` suffix unless it is the primary.

The lens redshift is always the snapshot redshift. The cosmology changes only angular scales
and critical densities; particle masses and physical lengths never change.

Ready-made run files for four Frontier Fields analogs are in
[`configs/hff_analogs/`](../configs/hff_analogs/README.md), and
[`configs/tngc014_mock.yaml`](../configs/tngc014_mock.yaml) spells out the default catalog
recipe.
