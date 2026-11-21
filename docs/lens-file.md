# The lens file

Each product is written to `<output_root>/<name>/lens.h5`, with
`name = tngc_z{zoom:03d}_s{snap:03d}_{axis}` (plus `_h<halo_id>` for a non-primary halo).
`axis` is `xy`, `yz` or `zx`. Map x and y are the simulation axes named by the two letters, and
the line of sight is the remaining one.

## Datasets

2D maps are float32, gzip-compressed and indexed `[y, x]`.

| dataset | unit | meaning |
|---|---|---|
| `x`, `y` | arcsec | pixel centres relative to the halo centre, ascending, uniform (float64, 1D) |
| `Sigma` | Msun/pkpc² | deposited surface density |
| `kappa` | dimensionless | Sigma / Sigma_crit,inf |
| `alpha_x`, `alpha_y` | arcsec | deflection |
| `potential` | arcsec² | lensing potential (arbitrary additive constant; zero at the map centre) |
| `gamma1`, `gamma2` | dimensionless | shear from the Jacobian of the stored deflection: gamma1 = (psi_xx - psi_yy)/2, gamma2 = psi_xy |
| `mu_zs<z>` | dimensionless | signed magnification 1/det(A) for a source at z (`%g` format, e.g. `mu_zs2`, `mu_zs2.5`) |

## Normalisation

kappa, alpha, potential and gamma are for D_ds/D_s = 1. Multiply them by
`efficiency(z_s) = D_ds/D_s` (`tnglens.cosmology.efficiency`) for a source at z_s. The lens
equation is beta = theta - efficiency(z_s) alpha(theta). Each `mu_zs*` dataset has the
attributes `source_redshift` and `efficiency`.

## Trusted region

Use the fields only inside `trusted_half_width_arcsec`, the core half-width minus two
environment cells (~198" for the default settings).

## Self-consistency

Image positions obey the deflection map, so everything that describes an image must come from
it. The shear of the FFT kernels differs from the derivatives of alpha (zoom 14, snapshot 73:
median ~2.5e-5, 99th percentile ~4e-4 inside the trusted region). The difference is small but
moves magnifications near critical curves, so the stored gamma and mu come from the Jacobian of
the stored alpha. kappa is kept as the deposited density. Its difference from half the divergence
of alpha is stored as `kappa_vs_div_alpha_{median,p99,max}` (absolute values inside the trusted
region), and the FFT-shear difference as `fft_shear_vs_jacobian_{median,p99}`.

## Attributes

The main attributes are `lens_redshift` (= snapshot redshift), `snapshot`, `zoom_index`,
`halo_id`, `primary_target`, `axis`, `line_of_sight_axis`, `H0`, `Om0`, `cosmology_source`
(`native` or `config`), `kpc_per_arcsec`, `sigma_critical_inf_Msun_pkpc2`, `M200c_Msun`,
`R200c_pkpc`, `halo_position_ckpc_h`, `box_ckpc_h`, `h`, `scale_factor`, `group_catalog_base`,
`group_file`, `snapshot_file`, `particle_types`, `particle_scope`, `source_redshifts`,
`trusted_half_width_arcsec`, `core_half_width_arcsec`, `support_floor_pkpc`, `neighbors`,
`gas_support`, `mass_ledger_json`, `solver_json`, `config_json`, `projection_signature`,
`tnglens_version`, and the plain-language `description`, `coordinates`, `gamma_convention` and
`consistency`.

`config_json` holds the part of the config the lens file depends on (`simulation`, `cosmology`,
`source_redshifts`, `particle_types`, `projection`). It is part of the signature, so a reused
file never carries a stale one.
