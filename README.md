# tnglens

[![tests](https://github.com/dutra/tnglens/actions/workflows/tests.yml/badge.svg)](https://github.com/dutra/tnglens/actions/workflows/tests.yml)
[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](pyproject.toml)

tnglens makes gravitational-lensing maps of TNG-Cluster halos. For each halo, snapshot and
projection axis it projects the FoF particles of the halo's zoom, solves for the lensing fields
and writes them to one HDF5 lens file. It can also place point sources behind the lens and write
mock multiple-image catalogs with their truth tables.

## Installation

```bash
pip install -e .            # numpy, scipy, h5py, astropy, pyyaml, py-sphviewer2
pip install -e '.[test]'    # adds pytest
pytest                      # unit tests, no simulation files needed
```

tnglens needs Python >= 3.11. It reads the TNG-Cluster HDF5 files directly with h5py and uses
[py-sphviewer2](https://pypi.org/project/py-sphviewer2/) to render resolved SPH kernels.

## Quick start

```bash
tnglens list                          # snapshot/zoom combinations on disk
cp configs/example.yaml my.yaml       # edit the halos list
tnglens run my.yaml                   # all halos in the config
tnglens run my.yaml --only 'tngc_z014_s073_yz'   # one product
tnglens run my.yaml --force maps      # recompute lens files, reuse the cached projections
```

Every stage is cached and reused when its inputs have not changed. Run files for four Frontier
Fields analogs are in [`configs/hff_analogs/`](configs/hff_analogs/README.md).

## What it produces

The lens file `<output_root>/<name>/lens.h5`, with `name = tngc_z{zoom:03d}_s{snap:03d}_{axis}`,
holds these maps on a 400" grid of 4096² pixels by default:

| dataset | unit | quantity |
|---|---|---|
| `Sigma` | Msun/pkpc² | surface density |
| `kappa` | dimensionless | convergence |
| `alpha_x`, `alpha_y` | arcsec | deflection |
| `potential` | arcsec² | lensing potential |
| `gamma1`, `gamma2` | dimensionless | shear |
| `mu_zs<z>` | dimensionless | signed magnification for a source at z |

kappa, alpha, the potential and gamma are normalised to D_ds/D_s = 1. For a source at z_s the
lens equation is beta = theta - (D_ds/D_s) alpha(theta), with D_ds/D_s from
`tnglens.cosmology.efficiency`. Use the maps only inside `trusted_half_width_arcsec` (~198" by
default). The full list of datasets and attributes is in [docs/lens-file.md](docs/lens-file.md).

If the config has a `catalog:` section, `tnglens.catalog` draws sources, finds all their images
in the deflection map and builds the member table from the Subfind catalog. It writes truth
tables and observed catalogs, by default with the selection of the tngc014 mock
([docs/catalog.md](docs/catalog.md)).

## Method

1. **Particles.** All particles of the zoom's FoF file, in physical kpc about the halo centre.
   The default types are gas, dark matter, low-resolution dark matter, stars and black holes.
2. **Smoothing lengths.** The distance to the 64th nearest neighbour of the same type, with a
   floor of 1.48 pkpc (the TNG-Cluster softening).
3. **Projection.** Particles within ±2 R200c along the line of sight go onto a 400" core grid
   and a 4 R200c environment grid. sphviewer2 renders kernels of 8 pixels or more, and a
   mass-conserving stencil or cloud-in-cell deposit handles the smaller ones.
4. **Fields.** The deflection and potential are free-space Green-function convolutions on
   zero-padded grids. The mass outside the core comes from the coarse environment grid.
5. **Derived maps.** Shear and magnification come from the exact derivative of the cubic
   B-spline of the stored deflection, so they match the mapping the image finder solves.

Details, including the mass ledger and the comparison of the two kernels, are in [docs/method.md](docs/method.md).

## Limitations

* One lens plane at the snapshot redshift, with no line-of-sight structure.
* Only the FoF particles of the zoom within ±2 R200c (by default) along the line of sight are used, so the
  maps miss the zoom's outer particles and any mass sheet from large-scale structure.
* The 64-neighbour, 1.48 pkpc smoothing is a choice, and substructure on scales of a few kpc
  depends on it.
* Only TNG-Cluster, and only snapshots with both group catalogs and particle files on disk.

## Documentation

* [docs/configuration.md](docs/configuration.md): every configuration key and its default
* [docs/pipeline.md](docs/pipeline.md): commands, exit codes, stages, caches, locking and cost
* [docs/method.md](docs/method.md): particles, smoothing, projection and field solver
* [docs/lens-file.md](docs/lens-file.md): datasets, normalisation and attributes of the lens file
* [docs/catalog.md](docs/catalog.md): mock multiple-image and member catalogs

## License

BSD 3-Clause. See [LICENSE](LICENSE).
