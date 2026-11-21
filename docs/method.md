# Method

The settings quoted here are the defaults ([configuration.md](configuration.md)).

## 1. Particles

tnglens reads all particles of the zoom's FoF file (`snapshot_NNN.<zoom>.hdf5`), that is the
whole original zoom FoF, not just the target group, and no outer fuzz. Coordinates are converted
to physical kpc relative to `GroupPos` with the periodic minimum image. Masses are converted to
Msun, from the header `MassTable` where it is non-zero (high-resolution dark matter) and from the
`Masses` dataset otherwise (for example the variable-mass low-resolution dark matter of type 2).
TNG-Cluster files are read directly with h5py, so `illustris_python` is not needed.

The particle types are configurable. The default is 0, 1, 2, 4, 5 (gas, dark matter,
low-resolution dark matter, stars and wind, black holes). For zoom 14 at snapshot 73 no type-2
kernel reaches either grid, because the nearest low-resolution particle is 3.95 R200c from the
centre, so its maps equal those for types 0, 1, 4, 5. With a larger environment or depth,
leaving out type 2 would drop real mass.

## 2. Smoothing lengths

The smoothing length is the exact distance to the 64th nearest neighbour of the same type in the
full periodic box (`scipy.spatial.cKDTree`), over all FoF particles of the zoom. Black holes get
zero. The support is `h = max(distance, 1.48 pkpc)`. With `gas_support: cells`, gas uses
`2 (3 m / 4 pi rho)^(1/3)` instead, and the floor still applies.

## 3. Projection

Two grids are centred on the halo: the core (400" at 4096 px) and the environment (4 R200c at
2048 px). Particles with centres within ±2 R200c along the line of sight are deposited. Kernels
with `h >= 8` pixels are rendered by sphviewer2. Smaller ones go through a stencil of the
projected compact kernel `(1 - r²/h²)^{3/2}`, normalised over all its cells before clipping, for
`1.5 <= h < 8` pixels, and through cloud-in-cell for `h < 1.5` pixels. Cloud-in-cell conserves
mass and the first moments. The small-kernel deposit is vectorised NumPy, so numba is not a
dependency.

sphviewer2's kernel is similar to the stencil kernel but not identical. For one particle it
differs from the stencil kernel by up to ~5% of the peak at h = 8 pixels (~12% for h >= 20
pixels), and it conserves the particle's mass to better than 0.4%.

The mass ledger records, per type and grid, the selected mass, the mass of fully contained
kernels and the deposited mass. If the deposit is not between the first two (1% tolerance), the
projection fails with an error and that lens file is not written. The ledger is stored in the
lens file as `mass_ledger_json`.

## 4. Fields

kappa = Sigma / Sigma_crit,inf with Sigma_crit,inf = c² / (4 pi G D_l). The potential and
deflection are free-space Green-function convolutions on zero-padded grids, with no periodic
images and no mean subtracted. The mass outside the core comes from the environment grid,
multiplied by the exact fraction of each coarse cell outside the core. Its fields are solved on
the coarse grid and interpolated onto the core. Two coarse cells inside the core edge are
excluded from the *trusted* region (~198" half-width for the default settings).

## 5. Derived maps

The deflection is stored in float32. gamma1, gamma2 and the magnification maps are computed from
the Jacobian of that stored map, as the exact derivative of its cubic B-spline
(`scipy.ndimage.spline_filter`, order 3) at the pixel centres. This is the derivative the image
finder in `tnglens.lensmap` evaluates, and the two agree to ~1e-13. A central difference of
±0.02" of the same spline differs from it by O(step²), up to ~1e-3 near sharp features. Why the
shear comes from the deflection and not from the FFT kernels is explained in
[lens-file.md](lens-file.md#self-consistency).
