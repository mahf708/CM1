# Reference cases

Each subdirectory is one CM1 configuration used as a reference for the ports
(cm1k in C++/Kokkos, and the array-backend port). It holds a
`namelist.input`, optionally an `every` file (dump interval in steps,
default 1) and any extra input files the case needs (copied next to the
namelist when the case runs).

All cases are serial, double precision (`gfortran -fdefault-real-8
-ffp-contract=off`), 3D, no radiation (except `rrtmg_*`), no surface fluxes, no explicit
diffusion (`idiff = 0`; except the `*_cor_*`/`*diff*` cases below), periodic lateral boundaries (`wbc = ebc = sbc = nbc = 1`;
except the `*_openbc*` cases),
free-slip top and bottom, a warm bubble (`iinit = 1`) with no random
perturbations (`irandp = 0`), and the time-split compressible solver
(`psolver = 3`), `dtl = 6 s`, except the `cbl_sfc*`, `pbl2_*`, `ysu_*`, `mynn*`, `myj_*`, `gfs_*` and `nh_mountain_waves*` cases (see the table).

| Case | Grid (nx×ny×nz, dx, dz) | Length | Dumps | What it exercises |
|---|---|---|---|---|
| `dry_bubble` | 32×32×20, 2 km, 500 m | 20 steps (120 s) | every step | dry dynamics only: no moisture (`imoist = 0`), no subgrid turbulence (`sgsmodel = 0`), dry-adiabatic sounding (`isnd = 1`), zero winds |
| `dry_shear` | 32×32×20, 2 km, 500 m | 20 steps (120 s) | every step | as `dry_bubble` plus the Weisman-Klemp supercell wind profile (`iwnd = 2`): advection by a sheared base-state flow |
| `kessler_bubble` | 32×32×20, 2 km, 500 m | 150 steps (900 s) | every 10 steps | Kessler microphysics (`ptype = 1`; `qv qc qr`), Weisman-Klemp sounding and winds (`isnd = 5`, `iwnd = 2`), no subgrid turbulence |
| `kessler_tke` | 32×32×20, 2 km, 500 m | 150 steps (900 s) | every 10 steps | as `kessler_bubble` with TKE subgrid turbulence (`sgsmodel = 1`): `tkea`, `kmh/kmv/khh/khv` are live |
| `kessler_saturated` | 32×32×20, 2 km, 500 m | 30 steps (180 s) | every step | Kessler with a saturated, neutrally stable sounding (`isnd = 4`, Bryan-Fritsch 2002), zero winds: cloud water and saturation adjustment active from t = 0 |
| `kessler_periodic` | 40×40×40, 1 km, 500 m | 10 steps (60 s) | every step | Kessler, Weisman-Klemp sounding and winds, finer grid; shared with the array-backend port (writes netCDF output, so netCDF-Fortran is required) |
| `cbl_sfc` | 32×32×40, 100 m, 50 m | 150 steps (300 s, `dtl = 2 s`) | every 10 steps | dry convective boundary layer: surface layer (`sfcmodel = 1`, semi-slip `bbc = 3`, `isfcflx = 1`) with a constant heat flux (`set_flx = 1`) and roughness (`set_znt = 1`), `cecd = 3`, `pertflx = 1`; TKE turbulence (`sgsmodel = 1`, `idiss = 1`, implicit vertical diffusion); random θ perturbations (`irandp = 1`); also dumps the surface arrays (`tsk`, `znt`, `ust`, `thflux`, ...) |
| `cbl_sfc_moist` | 32×32×40, 100 m, 50 m | 150 steps (300 s, `dtl = 2 s`) | every 10 steps | as `cbl_sfc` with Kessler microphysics over water (`lu0 = 16`, `tsk0 = 305 K`): bulk heat and moisture fluxes (`cecd = 2`, z0 from the 10-m wind), Weisman-Klemp sounding and winds |
| `kessler_periodic_long` | 40×40×40, 1 km, 500 m | 300 steps (1800 s) | every 50 steps | `kessler_periodic` run for 30 min: a deep convective cloud with rain, for long-run drift checks |
| `pbl2_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | simple PBL scheme (`cm1setup = 2`, `ipbl = 2`: Bryan-Rotunno/Louis vertical eddy viscosity, bulk-Ri `hpbl`, implicit vertical diffusion) with the surface layer over land (`sfcmodel = 1`, `bbc = 3`, `isfcflx = 1`, `lu0 = 3`, `tsk0 = 305 K`, `cecd = 3`), no horizontal turbulence (`horizturb = 0`), dry, Weisman-Klemp sounding and winds; also dumps the surface arrays, `hpbl` and the PBL tendencies (`thpten`, `upten`, ...) |
| `pbl2_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_dry` with Kessler microphysics over water (`lu0 = 16`, `cecd = 2`), horizontal Smagorinsky turbulence (`horizturb = 1`) and dissipative heating (`idiss = 1`) |
| `pbl2_sfclay_land` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_kessler` but with the MM5 surface layer (`sfcmodel = 2`, `isfcflx = 1` forced by param.F) over land (`lu0 = 3`, `xland0 = 1`, `tmn0 = 297.28 K`) and no horizontal turbulence: the SLAB soil model (5 layers) updates `tsk`, `tslb`, `hfx`, `qfx` every step after the first; also dumps `mol` and `tslb` |
| `ysu_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | YSU PBL (`cm1setup = 2`, `ipbl = 1`, `l_inf = 0`) with the revised MM5 surface layer (`sfcmodel = 3`) and SLAB over land (`lu0 = 3`, `tsk0 = 305 K`), `horizturb = 0`, YSU dissipative heating (`idiss = 1`), dry, Weisman-Klemp sounding and winds; dumps the surface arrays, `mol`, `tslb`, `hpbl` and the YSU tendencies |
| `ysu_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | YSU with the MM5 surface layer (`sfcmodel = 2`) over water (`lu0 = 16`), horizontal Smagorinsky turbulence (`horizturb = 1`), `idiss = 1`, Kessler microphysics (YSU's qc tendency) |
| `rrtmg_ysu_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `ysu_kessler` over land (`lu0 = 3`, `xland0 = 1`: SLAB updates `tsk`, `tslb`) with RRTMG radiation (`radopt = 2`, LW and SW every `dtrad = 36 s` = 6 steps, so dumps fall between calls; 21:38 UTC on 15 May 2009 at 36.68 N, 98.35 W: daytime) on the Kessler clouds (`cal_cldfra1`); YSU gets `rthraten`, SLAB `gsw`/`glw`. Also dumps the radiation arrays (`swten`, `lwten`, `thraten`, `cldfra`, `o30`, the LW/SW fluxes, `coszen`, `gsw`, `glw`, `albd`, `emiss`, ...). `make_refs.sh` copies `run/RRTMG_LW_DATA` and `run/RRTMG_SW_DATA` into the run directory of a `radopt = 2` case |
| `mynn25_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_kessler` with the MYNN-EDMF level-2.5 PBL (`ipbl = 4`: prognostic `qke`, advected in each RK stage, mass flux, subgrid clouds, its own dissipative heating, so `param.F` resets `idiss` to 0) and the MYNN surface layer (`sfcmodel = 6`) over water (`lu0 = 16`, so SLAB only diagnoses `qsfc`); also dumps the MYNN carried arrays (`qke`, `qke3d`, `tsq`, `el_pbl`, `cldfra_bl`, ...), `mol`, `rmol`, `zol`, `ustm`, `tslb`, `qcpten`, `qipten`, `xkzh`, `xkzm` |
| `mynn3_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_dry` with the MYNN-EDMF level-3 PBL (`ipbl = 5`: `tsq`, `qsq`, `cov` prognostic) and the MYNN surface layer (`sfcmodel = 6`, `isftcflx = 1`) over water (`xland0 = 2`, `lu0 = 16`), dry, no horizontal turbulence; dumps as `mynn25_kessler` |
| `mynn25_land_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn3_dry` with the level-2.5 PBL (`ipbl = 4`) over land (`lu0 = 3`, `xland0 = 1`, `tsk0 = 305 K`, `tmn0 = 297.28 K`) with the Chen-Zhang thermal roughness (`iz0tlnd = 1`): SLAB (5 soil layers) runs after the MYNN surface layer every step after the first and updates `tsk`, `tslb`, `hfx`, `qfx`; dumps as `mynn25_kessler` (incl. `tslb`) |
| `mynn3_land_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn25_kessler` with the level-3 PBL (`ipbl = 5`) over land (`lu0 = 7`, `xland0 = 1`, `iz0tlnd = 0`) with SLAB: moist land fluxes (SLAB's `qfx`, `lh` from the saturation mixing ratio at `tsk`), Kessler |
| `myj_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn25_kessler` with the MYJ PBL (`ipbl = 6`: prognostic `tke_myj`, not advected, `l_inf = 0`; `idiss = 1` keeps `turbparam_horiz`'s dissipation and the surface term) and the MYJ surface layer (`sfcmodel = 7`, `isftcflx = 1`) over water (`lu0 = 16`, SLAB only diagnoses `qsfc`); also dumps `mol`, `rmol`, `tslb`, the MYJ surface state (`thz0`, `qz0`, `uz0`, `vz0`, `akhs`, `akms`, `zntmyj`, `z0base`, `th10`, `q10`, `u10e`, `v10e`), `hpbl`, the PBL tendencies incl. `qipten`, `xkzh`, `xkzm`, `tke_myj`, `el_myj`, `mixht` |
| `myj_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `ysu_dry` with the MYJ PBL (`ipbl = 6`) and surface layer (`sfcmodel = 7`, `isftcflx = 0`) over warm water (`xland0 = 2`, `lu0 = 16`, `tsk0 = 305 K`), `horizturb = 0`, `idiss = 1`, dry; dumps as `myj_kessler`. (Over land CM1's MYJ roughness `zntmyj` stays 0 and the Fortran run is NaN within 10 steps, so the case runs over water.) |
| `gfs_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | GFS-EDMF PBL (`cm1setup = 2`, `ipbl = 3`: K profile, mass flux, stratocumulus mixing, its own dissipative heating, so `param.F` resets `idiss` to 0) with the GFDL surface layer (`sfcmodel = 4`, HWRF settings: `ntsflg = 1`, `icoef_sf = 6`) over water (`lu0 = 16`, `tsk0 = 305 K`; z0 from the 10-m wind, carried in `mznt`), horizontal Smagorinsky turbulence (`horizturb = 1`), Kessler; dumps the surface arrays, `mznt`, `mol`, `rmol`, `zol`, `br`, `gz1oz0`, `psim`, `psih`, `hpbl`, the PBL tendencies, `xkzh`, `xkzm` |
| `gfs_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | GFS-EDMF (`ipbl = 3`) with the revised MM5 surface layer (`sfcmodel = 3`) and SLAB over land (`lu0 = 3`, `xland0 = 1`, `tsk0 = 305 K`), no horizontal turbulence, dry; dumps the surface arrays, `mol`, `tslb`, `hpbl`, the PBL tendencies, `xkzh`, `xkzm` |
| `gfs_sfclay_land` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `gfs_dry` with the MM5 surface layer (`sfcmodel = 2`) and Kessler microphysics |
| `rrtmg_gfs_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `gfs_sfclay_land` with RRTMG radiation as in `rrtmg_ysu_kessler` (`radopt = 2`, `dtrad = 36 s`, daytime): GFS-EDMF gets `rthraten`, SLAB `gsw`/`glw`; dumps the radiation arrays too |
| `nh_mountain_waves` | 100×1×100, 200 m, 200 m | 100 steps (200 s, `dtl = 2 s`) | every 10 steps | terrain-following coordinate (`terrain_flag = .true.`, `itern = 1`: 400 m bell, half width 1 km), nonhydrostatic mountain waves: 2D, open E/W boundaries (`irbc = 4`), `isnd = 9` (N = 0.01/s), `iwnd = 6` (U = 10 m/s), `cm1setup = 0`, WENO scalars (`advwenos = 2`), Coriolis with `lspgrad = 1`, Rayleigh damping (`irdamp = 1`, `zd = 14 km`); also dumps the terrain metrics (`zs`, `gz`, `rgzu`, `dzdx`, `gx`, ...) |
| `nh_mountain_waves_long` | 100×1×100, 200 m, 200 m | 900 steps (1800 s, `dtl = 2 s`) | every 150 steps | `nh_mountain_waves` run for 30 min, for long-run drift checks |
| `kessler_terrain_tke` | 32×32×20, 2 km, 500 m | 150 steps (900 s) | every 10 steps | `kessler_tke` over a 3D hill (`terrain_flag = .true.`, `itern = 3`: 500 m, half width 20 km): terrain branches of the TKE scheme (implicit vertical diffusion, `idiss = 1`), Kessler, advection, sound and the warm bubble on sigma levels |
| `kessler_terrain_tke_openbc` | 32×32×20, 2 km, 500 m | 50 steps (300 s) | every 10 steps | `kessler_terrain_tke` with open-radiative E/W boundaries and free-slip/no-slip walls S/N (`wbc = ebc = 2`, `sbc = 3`, `nbc = 4`, as `kessler_tke_openbc`): `ssopenbc*` and the open-boundary pieces of the terrain turbulence branches |
| `tc1_cbl` | 24×24×40, 100 m, 50 m | 60 steps (120 s, `dtl = 2 s`) | every 10 steps | `testcase = 1` (dry CBL) as `cbl_sfc` (`pertflx = 0`) with `isnd = 9`, `iwnd = 6` (1 m/s from the testcase), `icor = 1`, `lspgrad = 2`: `use_avg_sfc` (domain-mean `getcecd`/`sfcstress`), the max-gradient `hpbl`, the testcase random perturbations (`zh <= 100 m`) |
| `tc2_shear_long` | 24×24×20, 200 m, 100 m | 640 steps (3200 s, `dtl = 5 s`) | every 80 steps | `testcase = 2` (shear PBL): `iwnd = 6` (15 m/s), `use_avg_sfc`, the surface heat flux switched off after 3000 s (`sfcflux`) |
| `tc3_shcu` | 24×24×40, 100 m, 50 m | 60 steps (120 s) | every 10 steps | `testcase = 3` (shallow Cu) with Kessler (`imoist` forced to 1), `isnd = 5`, `iwnd = 2`, the surface layer over water (`lu0 = 16`, `set_flx = 1`), `lspgrad = 2`: large-scale subsidence (`dolsw`: `wsub` in advection), `thfrc`, `qvfrc`, `ug`, the initial TKE profile, θ/qv perturbations below 1600 m |
| `tc4_simplerad` | 24×24×40, 100 m, 100 m | 60 steps (120 s) | every 10 steps | `testcase = 4` (stratocumulus): the testcase constants (`cp = 1015`, ...), simple radiation (`frad`, `thraten`, `zir`) on Kessler clouds, `dolsw`, `iwnd = 6` (7, -5.5 m/s), TKE = 1 initially, `lspgrad = 1`, surface layer over water |
| `tc5_simplerad_nosfc` | 24×24×40, 100 m, 100 m | 60 steps (120 s) | every 10 steps | `testcase = 5` constants and simple radiation without a surface layer (`sfcmodel = 0`, `bbc = 1`) |
| `tc6_hurr_long` | 24×24×30, 300 m, 50 m | 770 steps (3080 s, `dtl = 4 s`) | every 70 steps | `testcase = 6` (hurricane BL): `isnd = 8`, `iwnd = 8`, `lspgrad = 3` (gradient-wind `ulspg`/`vlspg`, the mesoscale `ufrc`/`vfrc` from domain means) rotated by `hurr_rotate = 30`, heat flux off after 3000 s. cm1k starts it from the dump (isnd = 8 is the sounding worker's) |
| `tc7_dycoms` | 24×24×40, 100 m, 50 m | 60 steps (120 s) | every 10 steps | `testcase = 7` (DYCOMS): surface fluxes and stress from the lowest-level wind (`sfcflux`, `sfcstress`), `dolsw`, `thfrc`/`qvfrc`, `ug`/`vg`, `lspgrad = 2`, Kessler |
| `tc8_rrtmg` | 24×24×25, 2 km, 200 m | 60 steps (360 s) | every 10 steps | `rrtmg_ysu_kessler` with `testcase = 8`: RRTMG at the fixed `solcon = 650.83`, `coszen = cos(50.5°)`, testcase perturbations in k = 1..5 (`irandp = 1`, no bubble) |
| `tc9_sbl` | 24×24×30, 12.5 m, 10 m | 60 steps (30 s, `dtl = 0.5 s`) | every 10 steps | `testcase = 9` (stable BL): `use_avg_sfc`, initial TKE below 255 m, perturbations below 50 m, `iwnd = 6` (8 m/s) |
| `tc10_nudge` | 4×4×30, 200 m, 50 m | 100 steps (200 s) | every 10 steps | `testcase = 10`: the θ/qv nudging towards the base state (`tqnudge`), `lspgrad = 3`, the simple PBL (`cm1setup = 2`, `ipbl = 2`), Kessler, `isnd = 8`, `iwnd = 8` (from the dump in cm1k) |
| `tc11_sas` | 24×24×40, 100 m, 50 m | 60 steps (120 s) | every 10 steps | `testcase = 11` (SAS): the `sfcflux` branch (`thmag`), `use_avg_sfc`, initial TKE, `iwnd = 6` (2 m/s), Kessler |
| `tc12_windtunnel` | 24×16×20, 2 mm, 2 mm | 30 steps (1.2 ms, `dtl = 40 µs`) | every 10 steps | `testcase = 12` (wind tunnel): `lspgrad = 4` (`ulspg = 36`), `isnd = 9`, `iwnd = 6` (20 m/s), velocity perturbations (`ii = 5..nx-4`), no surface layer |
| `tc13_blwt` | 32×16×30, 50 m, 20 m | 60 steps (30 s, `dtl = 0.5 s`) | every 10 steps | `testcase = 13` (BL wind tunnel): `u0 = ug = 18 m/s`, `lspgrad = 4`, the upstream-third roughness (`nx > ny`), velocity perturbations, `isfcflx = 0` |
| `tc14_sgp` | 24×24×40, 100 m, 50 m | 60 steps (240 s, `dtl = 4 s`) | every 10 steps | `testcase = 14` (SGP shallow Cu): time-dependent `thfrc`/`qvfrc`, `use_avg_sfc`, `iwnd = 6`, Kessler |
| `tc15_hurr` | 24×24×30, 300 m, 50 m | 60 steps (240 s) | every 10 steps | `testcase = 15` (hurricane coast): `lspgrad = 3`, perturbations, `isfcflx = 0`, `isnd = 8`, `iwnd = 8` (from the dump in cm1k) |
| `tc3_nosfc`, `tc7_nosfc`, `tc9_nosfc`, `tc11_nosfc`, `tc13_nosfc`, `tc14_nosfc` | as `tc3_shcu`, `tc7_dycoms`, `tc9_sbl`, `tc11_sas`, `tc13_blwt`, `tc14_sgp` | 60 steps | every 10 steps | the same test cases without a surface layer (`sfcmodel = 0`, `isfcflx = 0`, `bbc = 1`): the configurations the array port (cm1x, isnd = 5, 9, no surface fluxes with TKE) runs too |
| `tc4_nosfc` | as `tc5_simplerad_nosfc` | 60 steps (120 s) | every 10 steps | `tc5_simplerad_nosfc` with `testcase = 4` (its constants) |
| `tc6_nosfc` | as `tc3_shcu` | 60 steps (120 s) | every 10 steps | `testcase = 6` on the `tc3_shcu` grid with `isnd = 5`, `iwnd = 2`, dry, no surface layer: `lspgrad = 3` mesoscale tendencies from the domain-mean (sheared) wind, rotated by `hurr_rotate = 30`, `dolsw`, perturbations below 100 m |
| `tc10_les` | as `tc3_shcu` | 60 steps (120 s) | every 10 steps | `testcase = 10` with LES TKE instead of the PBL scheme, no surface layer, Kessler: θ/qv nudging (`get_avg_uvtq`), `lspgrad = 3` |
diffusion (`idiff = 0`; except the `*_cor_*`/`*diff*` cases below), periodic lateral boundaries (`wbc = ebc = sbc = nbc = 1`;
except the `*_openbc*` cases),

The table is descriptive; the namelists are authoritative.

## Making the references

```bash
cxx/scripts/make_refs.sh OUT_DIR              # all cases
cxx/scripts/make_refs.sh OUT_DIR dry_bubble   # some cases
```

The script builds the Fortran out of tree (default
`build/cm1-fortran-<config>/` at the repo root, incremental), runs each case
with `CM1_DUMP_DIR` set, checks every dump against its `index.txt`, and fails
loudly (non-zero exit, log tail on stderr) on any build or run error.
Environment knobs: `CM1_USE_OPENMP`, `CM1_THREADS`, `CM1_DEBUG`,
`CM1_BUILD_DIR`, `JOBS`, `FC` (see the script header). The command-line
interface `make_refs.sh OUT_DIR [case ...]` and the layout below are stable;
other tools and CI jobs depend on them.

In CI the workflow `.github/workflows/cm1k.yml` (job `fortran-refs`) builds
all cases and uploads OUT_DIR as the artifact **`cm1-refs`**:

```bash
gh run download <run-id> -n cm1-refs -D refs     # same layout as OUT_DIR
```

## Layout (format_version 1)

```
OUT_DIR/
  MANIFEST.txt                 build info + one "case <name> every <n> steps <count> last <step_dir>" line per case
  <case>/
    MANIFEST.txt               build info (git commit, compiler, make options, OMP threads, date) + the case line
    namelist.input             the namelist that was run
    cm1.log                    the model's stdout/stderr
    step_NNNNNN/               NNNNNN = model step (zero padded to 6 digits); step_000000 is the initial state
      index.txt
      <name>.bin
```

`index.txt`: the first line is `#step_time <nstep> <model time in s>`, then
one line per array:

```
<name> <kind> <ndims> <lb1> <ub1> [<lb2> <ub2> ...]
```

`<name>.bin` is the array as raw native-endian (little-endian on x86/ARM)
reals of `<kind>` bytes (always 8: float64) in Fortran (column-major, i
fastest) order, halo points included, with Fortran index bounds `lb..ub` in
each dimension. No header, no record markers. In numpy:

```python
shape = [ub - lb + 1 for lb, ub in bounds]
a = np.fromfile(f"{step}/{name}.bin", "<f8").reshape(shape, order="F")
# a[i - lb1, j - lb2, k - lb3] is the Fortran a(i, j, k)
```

### Arrays

What is written is defined by `src/cm1dump_state.inc`; at format_version 1:

- 1D grid: `xh rxh uh ruh` (`ib:ie`), `xf rxf uf ruf` (`ib:ie+1`), `yh vh rvh`,
  `yf vf rvf`, `sigma rds` (`kb:ke`), `sigmaf rdsf` (`kb:ke+1`)
- 3D grid metrics: `zh mh rmh c1 c2`, `zf mf rmf` (`kb:ke+1`)
- base state: `pi0 th0 rho0 prs0 thv0 rth0 qv0 qc0 qi0 rr0 rf0 rrf0`,
  `u0` (`ib:ie+1`), `v0` (`jb:je+1`)
- prognostics/diagnostics: `ua` (`ie+1`), `va` (`je+1`), `wa` (`ke+1`),
  `ppi tha rho prs`
- moisture (only when `imoist = 1`): `qa_<qname>`, e.g. `qa_qv qa_qc qa_qr` for Kessler
- turbulence: `tkea`, `kmh kmv khh khv` (dimensioned `1:1` in every
  direction when the scheme that uses them is off; check the bounds)
- `rain` (`ib:ie, jb:je, 1:nrain`)
- test cases (`testcase >= 1` or `lspgrad >= 2`): the forcing profiles
  `wprof ufrc vfrc thfrc qvfrc ug vg dvdr ulspg vlspg` (`kb:ke`); with
  `testcase = 4, 5` also `frad` (`kb:ke+1`), `thraten` and `zir` (`1:1` in
  k); with `testcase = 1..7`, `bbc = 3` and no PBL scheme also `hpbl`

Typical bounds for the 32×32×20 cases: `ib = jb = -2`, `ie = je = 35`
(3 halo points), `kb = 0`, `ke = 21`.

Dumps are written at step 0 (after initialization) and after every
`every`-th step (after `mtime` is advanced), so `step_000010` is the state
after ten full time steps.

## Reproducibility

`cxx/scripts/check_determinism.sh WORK_DIR [case ...]` builds and runs the
Fortran twice (serial build), and with OpenMP at 1 and 2 threads, and
compares the dumps with `cxx/scripts/compare_dumps.py` (bitwise by default;
`--rtol/--atol` for tolerances). See the script header for which
comparisons are required to be bitwise.

Result (gfortran 13.3, x86_64, all seven cases, 2026-10): every comparison is
**bitwise identical**: run to run, serial build vs OpenMP build at 1 thread,
and OpenMP 1 vs 2 threads. The Fortran reference is deterministic and
thread-count independent for these configurations, so bitwise parity with
the Fortran is a realistic target for ports that keep its operation order
(reductions over the domain, if any are added later, would be the exception).
