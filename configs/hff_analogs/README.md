# TNG-Cluster analogs of the Frontier Fields clusters

One run file per cluster. Each names the halo, the snapshot and the view, and writes out the cosmology, the map
settings and the catalog recipe, so `tnglens run configs/hff_analogs/<cluster>.yaml` produces the lens fields
(`lens.h5`) and the mock catalogs (`observed/`, `truth/`) of that analog.

| file | observed cluster | halo, snapshot, zoom | view | theta_E observed / analog (z_s = 6) | product |
|---|---|---|---|---|---|
| `abell2744.yaml` | Abell 2744, z = 0.3072 | 1805203, 78, 9 (z = 0.2977) | yz | 27.5" / 27.3" | `tngc_z009_s078_yz` |
| `macsj0416.yaml` | MACS J0416.1-2403, z = 0.3960 | 3535001, 72, 22 (z = 0.3999) | zx | 33.9" / 33.0" | `tngc_z022_s072_zx` |
| `abells1063.yaml` | Abell S1063, z = 0.3475 | 260530, 75, 1 (z = 0.3479) | zx | 41.5" / 41.9" | `tngc_z001_s075_zx` |
| `abell370.yaml` | Abell 370, z = 0.3750 | 2571291, 73, 14 (z = 0.3802) | yz | 45.4" / 39.6" | `tngc_z014_s073_yz` |

The analogs are the rank-one Einstein-radius matches of `ggsl_env/lenscluster/analog_matches.py`. MACS J0717 and
MACS J1149 have no match in that table, so they have no file here. The four files differ only in the header and in
the `halos` entry. The cosmology is the simulation's (Planck 2015: H0 = 67.74, Om0 = 0.3089, flat), and the map
settings, the catalog recipe and the seed are the same for all four.

Run all four, one after the other:

```bash
for f in configs/hff_analogs/*.yaml; do tnglens run "$f"; done
```

The products are generated, not tracked: `catalogs/` in this folder holds a local copy of `observed/` and `truth/`
of each product and is ignored by git.
