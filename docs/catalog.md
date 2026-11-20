# Catalogs: multiple images and members

`tnglens.catalog` reads one lens file (`lens.h5`), draws point sources, finds all their images, builds the
table of cluster members from the Subfind group catalog, and writes truth tables plus fit-ready catalogs in
the YAML formats kappaccino reads (`lenscluster-images/1` and the member catalog).

```
python -m tnglens.catalog LENS.h5 --config CFG.yaml --out DIR [--override KEY=VALUE ...] [--workers N]
```

```python
from tnglens.catalog import build_catalog
paths = build_catalog("lens.h5", cfg["catalog"], "out/")   # {name: path, ..., "summary": {...}}
```

With `workers > 1` the candidates run in a spawn-context process pool and each worker loads the lens file
(~0.5 GB for a 4096^2 map). A script that calls `build_catalog` with workers needs the usual
`if __name__ == "__main__":` guard. The output does not depend on `workers` or `batch`.

## Method

- **Lens.** Images obey the mapping of the deflection map, beta = theta - s alpha(theta), with
  s = D_ds/D_s(z_s) from the file's cosmology (`H0`, `Om0`) and alpha normalised to s = 1. alpha_x and alpha_y
  are interpolated by a cubic B-spline through the pixel values, using only the search square plus a 64-pixel
  margin. The prefilter and the evaluation use the same (mirror) boundary, so the spline passes through
  every pixel value, the first and last rows and columns included. The Jacobian is the analytic derivative of the same spline, so mu = 1/det A is exactly the
  magnification of the mapping the images solve. kappa = s(a_xx + a_yy)/2, gamma1 = s(a_xx - a_yy)/2 and
  gamma2 = s(a_xy + a_yx)/2 at the images come from it too, not from the `kappa`/`gamma` maps.
- **Images.** A mesh (every `finder_stride` map pixels) is mapped to the source plane. Each mesh triangle
  whose image contains beta gives a starting point, and vectorised Newton refines it (with
  `scipy.optimize.root` as a fallback). Mesh cells crossed by a critical curve whose mapped bounding box
  holds beta are searched again on a fine local mesh, which recovers fold pairs closer than a mesh step. If
  the count is still even, the most magnified images get a further local search. Roots solve the lens
  equation to 1e-7 arcsec.
- **Time delays.** Fermat potential tau = 0.5 |theta - beta|^2 / s - psi(theta) with psi the `potential`
  map, times `cosmology.time_delay_days` = (1 + z_l) D_l / c, relative to the first image of each system.
- **Sources.** Candidate i uses its own generator `default_rng([seed, i])`. It draws a redshift and a seeding
  kind. `negative` takes a random image-plane mesh node with det A < 0. `critical` takes a node with
  |det A| < t, with t drawn from `critical_det`. Either node is jittered uniformly within half a mesh step
  in x and y, and the jittered point (the recorded seed, one of the images) is mapped back to the source
  plane. `source` is uniform in the source plane over the box that holds every multiply imaged source.
  Candidates are evaluated in index order, and the first `n_systems` with at least two observable images
  are kept. A candidate is rejected when two of its observable images are closer than `min_separation`,
  at their true positions or after the position noise, or when it falls within `duplicate_dz` and
  `duplicate_dbeta` of a kept source.
  **Image-plane seeding makes the kept systems a selected, not an unbiased, source population.** Each
  system records its seeding kind and candidate index. `truth/candidates.csv` lists every candidate with
  its outcome: `foreground` (redshift at or in front of the lens), `no_seed` (the kind found no seed node),
  `single` (fewer than two images in the square), `lt2_observable`, `close_pair` (true positions),
  `close_pair_observed` (noisy positions), `duplicate`, `accepted`. A redshift distribution entirely at or
  below the lens redshift is an error; one partly below it gives a warning and a note in `selection.yaml`.
- **Noise.** The observational noise of candidate i is a function of the seed and i only (position offsets
  from `default_rng([seed, 1, i])`, source magnitude and photometric redshift from
  `default_rng([seed, 2, i])`), so the noisy close-pair cut does not depend on `workers`, `batch` or on
  which other candidates were kept. A photometric redshift scatters about the true one by
  `photoz_sigma` (1 + z_true), is at least max(`photoz_zmin`, z_lens + 0.01), and is reported with
  z_err = `photoz_sigma` (1 + z_catalog).
- **Members.** Read from `groups_SSS/fof_subhalo_tab_SSS.Z.hdf5`, with Z = `zoom_index`, under
  `group_catalog_base`. Global IDs use `postprocessing/offsets/offsets_SSS.hdf5`, or the chunk headers when
  that file is absent. Positions are relative to `halo_position_ckpc_h` (periodic wrap, physical kpc), then
  map x and y are the simulation axes named by `axis`, divided by `kpc_per_arcsec`. The magnitude is the
  rest-frame `SubhaloStellarPhotometrics` band plus the distance modulus minus 2.5 log10(1 + z_l). The
  stellar shape is the mass-weighted second moment of the star particles within 2 r_half,*. It is read
  from `snapdir_SSS/snapshot_SSS.Z.hdf5` and left empty (no `shape` key) when the snapshot is not there.
  The BCG is the central subhalo of `halo_id`; the BCG-exclusion cut is centred on it.

### Known selection effects and caveats

- **Close pairs.** A system is rejected whole when any two of its observable images are closer than
  `min_separation`. On the tngc014 lens this removes about half of the `critical`-seeded candidates (9 of 17
  in a 20-system build; 1 of 13 `negative`-seeded), so the most magnified fold pairs are under-represented
  among the kept systems. `status_by_kind` in `truth/selection.yaml` gives the counts of each build.
- **Images inside the BCG.** The truth tables keep every image the finder solves, including central images
  within `bcg_radius` of the BCG. Most exist only because the pixelised deflection map is interpolated
  through a core the simulation does not resolve (kappa of tens, |mu| ~ 1e-4), so they are not reliable
  predictions. They are flagged `inside_bcg_radius` in `truth/images_truth.csv`, are never observable, are
  counted in `n_images` and `odd_number`, and are left out of `n_images_outside_bcg`.
- **No systems.** If no candidate is accepted, the image catalogs hold `systems: {}` and `selection.yaml`
  has a note; kappaccino's loader refuses a catalog without systems.

## The tngc014 mock: class quotas, satellite-tied systems and knots

Three parts reproduce the selection of the tngc014 mock. All are on by default with the numbers of that mock
(69 systems, about 300 images); `configs/tngc014_mock.yaml` spells them out. For a plain catalog, the first N
candidates that pass the cuts, set `n_systems: N` (which sets `quotas` to null) and `knots.n_systems: 0`.
Without the simulation files (the Subfind chunk and the snapshot chunk) the satellite part is left out with a note
in `truth/selection.yaml`: no systems are tied, the subhalo quota is dropped and the `subhalo` seeds are not drawn.

- **Class quotas** (`quotas`). Each candidate that passes the cuts has a class from the geometry of its
  observable images: `wide` (closest pair at least 8" apart), `intermediate` (3" to 8"), `fold` (an
  opposite-parity pair at most 3" apart with the next image more than max(2 sep, 2") away), `cusp` (three close
  images of alternating parity) or `other`. With `quotas`, candidates are taken in candidate order while their
  class has room; giving both `quotas` and `n_systems` is an error. The fold quota is split evenly, rounded up, over the pair
  separations of `selection.fold_bins`, so `fold: 15` with the default three bins keeps five per bin. A wide
  system with only two observable images is kept with probability `selection.wide_pair_keep`. Candidates end
  as `no_quota`, `quota_full` or `thinned` in `truth/candidates.csv` when they are not kept for these reasons.
  The run stops when every quota is full or at `max_candidates`; quotas left unfilled are listed in the notes
  of `truth/selection.yaml`.
- **Images tied to satellites** (`subhalos`, module `tnglens.satellites`). For each satellite with bound mass
  above `min_mass` (in the particle types of the lens) that lies near an observable image, the deflection of its
  bound particles is subtracted from the lens, the source position is refitted to the observable images, and
  the images are solved again. An image within `radius_einstein` point-mass Einstein radii of the satellite
  (clipped to `radius_clip`) that vanishes or moves by more than `shift_tol` is tied to it. A system with a tied
  image has class `subhalo` and is named S1, S2, ... Its notes list the satellite and its images, and each tied
  image has a note with the shift. At most `selection.max_per_satellite` subhalo systems have the same first
  satellite. The satellite's particles are read from the snapshot chunk (`members.snapshot_file`, or the
  default path under `group_catalog_base`), so this part needs the simulation files and the `particle_types`
  attribute of the lens file. The seeding kind `subhalo` (`sources.seed_kinds`) places seeds at det A < 0
  within `seed_radius` of satellites above `seed_min_mass` that are farther than `seed_min_centre` from the BCG.
  `enabled: auto` turns this part on when `quotas.subhalo` or the `subhalo` seeding weight is positive and the
  simulation files are found; `true` makes missing files an error.
- **Knots** (`knots`). The `n_systems` systems with the largest |mu| get `extra` more point sources (knots B,
  C), each `offset` arcsec from the source centre in a random direction and `mag_offset` fainter. A knot is kept
  when it has as many images as the source, with the same parities, and its observable images lie between
  `min_sep` and `max_sep` of the matching images of the source (knot A). The image catalogs then give one
  position per knot in every image of the system, in the knot format of kappaccino.

Caveats. The satellite deflection uses a fixed Gaussian smoothing (`patch_sigma_pixels` grid pixels, 0.16" by
default), not the adaptive kernel of the lens map, so a tie is a classification and not a measurement. A
candidate with a tied image is never kept under its geometric class, so the wide, cusp and fold systems exclude
those with a tied image. Unlike the original script, which drew a fixed pool and shuffled it, the selection here
takes candidates in index order, so it does not reproduce the systems of `catalogs/tngc014` one for one. Each
tested candidate costs about one image search over the whole mesh per nearby satellite.

## Configuration: the `catalog:` section

Every key is optional; the defaults are those of the tngc014 mock. `tnglens.catalog.resolve_config(section)`
merges and validates the section without the lens file and raises ValueError naming the key on an unknown key
or a bad value; the checks that need the lens (redshifts against z_lens, the map size) are made by
`build_catalog`.

| key | default | meaning |
|---|---|---|
| `seed` | 20261111 | master seed |
| `n_systems` | none | keep the first N candidates that pass the cuts; sets `quotas` to null |
| `max_candidates` | 20000 | stop after this many candidates |
| `batch` | 64 | candidates per round (no effect on the result) |
| `workers` | 1 | processes |
| `search_half` | auto | image search square half-width [arcsec]; auto = the file's `trusted_half_width_arcsec`, else 110 clipped to the map half-width minus 16 + 3 `finder_stride` pixels (with a note); values above the trusted width are reduced to it |
| `finder_stride` | 2 | mesh step in map pixels |
| `overrides` | {} | lens-file attributes to supply or replace (for files missing them) |
| `sources.redshift` | `{edges: [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 6, 8], weights: [.10, .17, .20, .17, .12, .12, .08, .04]}` | or `{values: [...], weights: [...]}` |
| `sources.seed_kinds` | `{negative: 0.30, critical: 0.35, source: 0, subhalo: 0.35}` | weights of the seeding kinds |
| `sources.seed_radius` | `[5, auto]` | image-plane seeds between these radii; auto = 95/110 of the search half-width |
| `sources.critical_det` | `[0.05, 0.02, 0.008, 0.003]` | thresholds of the `critical` kind |
| `sources.duplicate_dz`, `duplicate_dbeta` | 0.05, 1.0 | duplicate-source rejection |
| `observability.mu_min` | 0.25 | observable when \|mu\| > this |
| `observability.bcg_radius` | 3.0 | and farther than this from the BCG [arcsec] |
| `observability.min_separation` | 0.3 | reject a system with two observable images closer than this |
| `observability.min_images` | 2 | observable images a system needs |
| `noise.sigma_pos` | 0.03 | Gaussian position noise per axis [arcsec] |
| `noise.photoz_fraction`, `photoz_sigma`, `photoz_zmin` | 0.12, 0.06, 0.8 | systems with a photometric z: scatter sigma (1 + z_true), z_err = sigma (1 + z_catalog), z_catalog >= max(zmin, z_lens + 0.01) |
| `noise.source_mag` | [26, 29] | source magnitude range; image mag = m - 2.5 log10 \|mu\| |
| `quotas` | `{wide: 22, intermediate: 14, cusp: 8, fold: 15, subhalo: 10}` | systems to keep per class; null keeps the first `n_systems` instead |
| `selection.fold_bins` | [0.3, 0.7, 1.5, 3.0] | pair separations [arcsec] the fold quota is split over |
| `selection.wide_pair_keep` | 0.3 | fraction kept of wide systems with only two observable images |
| `selection.max_per_satellite` | 2 | subhalo systems with the same first satellite |
| `subhalos.enabled` | auto | tie images to satellites (auto: when `quotas.subhalo` or the `subhalo` seeding weight is positive and the simulation files are found) |
| `subhalos.min_mass` | 3e10 | satellites tested: bound mass in the particle types of the lens [Msun] |
| `subhalos.shift_tol` | 0.5 | an image that moves by more than this without the satellite is tied to it [arcsec] |
| `subhalos.radius_einstein`, `radius_clip` | 4, [1.5, 6] | search radius in point-mass Einstein radii of the satellite, clipped to this range [arcsec] |
| `subhalos.match_radius` | 5 | largest shift that still counts as the same image [arcsec] |
| `subhalos.patch_half`, `patch_pixels`, `patch_keep`, `patch_sigma_pixels` | 40, 1000, 30, 2 | the satellite's local grid: half-width [arcsec], pixels, half-width used (a point mass outside), Gaussian smoothing [pixels] |
| `subhalos.seed_min_mass`, `seed_radius`, `seed_min_centre` | 1e11, 4, 8 | the `subhalo` seeding kind: satellite mass [Msun], distance of the seed [arcsec], distance of the satellite from the BCG [arcsec] |
| `subhalos.max_cached` | 24 | satellite grids kept in memory per process |
| `knots.n_systems` | 12 | systems that get extra knots, most magnified first |
| `knots.extra`, `offset`, `tries` | [1, 2], [0.03, 0.15], 6 | knots added per system, their distance from the source centre [arcsec], positions tried per knot |
| `knots.min_sep`, `max_sep`, `mag_offset` | 0.08, 4, 1 | allowed separation from the same image of knot A [arcsec]; a knot is this much fainter |
| `members.enabled` | auto | build the member table (auto: if the group catalog is found; true: an error naming the missing attribute or path if it is not) |
| `members.mag_limit`, `radius`, `los_r200c` | 24, 120, 2 | members brighter than this, within this radius [arcsec] and \|LOS\| < this R200c |
| `members.band_index`, `band_name` | 7, F125W | photometry column (U B V K g r i z) and its name in members.yaml |
| `members.shapes` | auto | auto/snapshot, a directory holding a `type4.h5` cache, or none |
| `members.shape_min_particles` | 50 | minimum star particles for a shape |
| `members.group_file`, `offsets_file`, `snapshot_file` | derived | explicit paths |

Example for the pre-contract tngc014 file:

```yaml
catalog:
  n_systems: 50
  overrides: {H0: 70.0, Om0: 0.3, zoom_index: 14, snapshot: 73, axis: yz,
              group_catalog_base: /ironwolf/TNGC/output, halo_position_ckpc_h: [141836.671875, 294045.15625, 268843.0],
              h: 0.6774, scale_factor: 0.7245495448210172, box_ckpc_h: 680000.0, R200c_pkpc: 2192.744246298297,
              particle_types: [0, 1, 4, 5]}
```

## Outputs (in `--out`)

| file | content |
|---|---|
| `observed/images.yaml` | lenscluster-images/1: noisy positions of observable images (one per knot for a system with knots), catalog redshifts (some photometric), parity, mag, notes of tied images |
| `observed/members.yaml` | band, members keyed by global Subfind ID: ra, dec, mag, shape, `role: bcg` |
| `truth/images_truth.yaml` | same format: every image in the search square at its exact position, true z; image notes give mu and dt |
| `truth/images_truth.csv` | per image and knot: system, knot, image, z_true, z_catalog, efficiency, beta_x/y, x_true/y_true, x_obs/y_obs, observable, inside_bcg_radius, mu, parity, kappa, gamma1, gamma2, fermat_arcsec2, dt_days, mag, source_mag, residual_arcsec, cls, subhalo (satellites the image is tied to) |
| `truth/sources_truth.csv` | per system: z, source position, n_images, n_images_outside_bcg, n_observable, odd_number, class and geometry (the class without the satellite ties), min/max separation (true), min_sep_observed (noisy), n_knots, subhalos, seed kind, candidate index, seed position |
| `truth/candidates.csv` | every candidate drawn and its outcome |
| `truth/members_truth.csv` | id, index, x, y, los_pkpc, mag, role, fof_group, mtot, mstar, m_in_2rhalf, veldisp, vmax, rvmax_pkpc, rhalf_star_pkpc, rhalf_pkpc, mstar_in_2rhalf, mlens, shape_a/b/theta, n_star_shape |
| `truth/selection.yaml` | resolved configuration, counts per outcome and seeding kind, lens metadata; the lens file by name and identifying attributes (signature, version, zoom, snapshot, halo, axis, z) |
| `README.md` | conventions, cuts, seed and the meaning of each truth column |

Conventions: positions are offsets about (0, 0) in degrees with x = -ra 3600 and y = dec 3600 arcsec in the
lens-map frame. Systems are named 1, 2, ... by true redshift, and those of class subhalo S1, S2, ... Image labels a, b, ... put observable images
first, brightest first. Masses are in Msun, lengths in physical kpc, velocities in km/s, delays in days. CSV
floats have 9 significant digits. The files contain no dates, timings or the lens path as spelled (only the
file name), so the same lens file, configuration and seed give byte-identical outputs.
