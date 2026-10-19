# Reference cases

Each subdirectory is one CM1 configuration used as a reference for the ports
(cm1k in C++/Kokkos, and the array-backend port). It holds a
`namelist.input`, optionally an `every` file (dump interval in steps,
default 1) and any extra input files the case needs (copied next to the
namelist when the case runs).

All cases are serial, double precision (`gfortran -fdefault-real-8
-ffp-contract=off`), 3D, no radiation, no surface fluxes, no explicit
diffusion (`idiff = 0`; except the `*_cor_*`/`*diff*` cases below), periodic lateral boundaries (`wbc = ebc = sbc = nbc = 1`;
except the `*_openbc*` cases),
free-slip top and bottom, a warm bubble (`iinit = 1`) with no random
perturbations (`irandp = 0`), and the time-split compressible solver
(`psolver = 3`), `dtl = 6 s`, except the `cbl_sfc*`, `pbl2_*`, `ysu_*`, `mynn*`, `myj_*` and `gfs_*` cases (see the table).

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
| `mynn25_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_kessler` with the MYNN-EDMF level-2.5 PBL (`ipbl = 4`: prognostic `qke`, advected in each RK stage, mass flux, subgrid clouds, its own dissipative heating, so `param.F` resets `idiss` to 0) and the MYNN surface layer (`sfcmodel = 6`) over water (`lu0 = 16`, so SLAB only diagnoses `qsfc`); also dumps the MYNN carried arrays (`qke`, `qke3d`, `tsq`, `el_pbl`, `cldfra_bl`, ...), `mol`, `rmol`, `zol`, `ustm`, `tslb`, `qcpten`, `qipten`, `xkzh`, `xkzm` |
| `mynn3_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `pbl2_dry` with the MYNN-EDMF level-3 PBL (`ipbl = 5`: `tsq`, `qsq`, `cov` prognostic) and the MYNN surface layer (`sfcmodel = 6`, `isftcflx = 1`) over water (`xland0 = 2`, `lu0 = 16`), dry, no horizontal turbulence; dumps as `mynn25_kessler` |
| `mynn25_land_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn3_dry` with the level-2.5 PBL (`ipbl = 4`) over land (`lu0 = 3`, `xland0 = 1`, `tsk0 = 305 K`, `tmn0 = 297.28 K`) with the Chen-Zhang thermal roughness (`iz0tlnd = 1`): SLAB (5 soil layers) runs after the MYNN surface layer every step after the first and updates `tsk`, `tslb`, `hfx`, `qfx`; dumps as `mynn25_kessler` (incl. `tslb`) |
| `mynn3_land_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn25_kessler` with the level-3 PBL (`ipbl = 5`) over land (`lu0 = 7`, `xland0 = 1`, `iz0tlnd = 0`) with SLAB: moist land fluxes (SLAB's `qfx`, `lh` from the saturation mixing ratio at `tsk`), Kessler |
| `myj_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `mynn25_kessler` with the MYJ PBL (`ipbl = 6`: prognostic `tke_myj`, not advected, `l_inf = 0`; `idiss = 1` keeps `turbparam_horiz`'s dissipation and the surface term) and the MYJ surface layer (`sfcmodel = 7`, `isftcflx = 1`) over water (`lu0 = 16`, SLAB only diagnoses `qsfc`); also dumps `mol`, `rmol`, `tslb`, the MYJ surface state (`thz0`, `qz0`, `uz0`, `vz0`, `akhs`, `akms`, `zntmyj`, `z0base`, `th10`, `q10`, `u10e`, `v10e`), `hpbl`, the PBL tendencies incl. `qipten`, `xkzh`, `xkzm`, `tke_myj`, `el_myj`, `mixht` |
| `myj_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `ysu_dry` with the MYJ PBL (`ipbl = 6`) and surface layer (`sfcmodel = 7`, `isftcflx = 0`) over warm water (`xland0 = 2`, `lu0 = 16`, `tsk0 = 305 K`), `horizturb = 0`, `idiss = 1`, dry; dumps as `myj_kessler`. (Over land CM1's MYJ roughness `zntmyj` stays 0 and the Fortran run is NaN within 10 steps, so the case runs over water.) |
| `gfs_kessler` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | GFS-EDMF PBL (`cm1setup = 2`, `ipbl = 3`: K profile, mass flux, stratocumulus mixing, its own dissipative heating, so `param.F` resets `idiss` to 0) with the GFDL surface layer (`sfcmodel = 4`, HWRF settings: `ntsflg = 1`, `icoef_sf = 6`) over water (`lu0 = 16`, `tsk0 = 305 K`; z0 from the 10-m wind, carried in `mznt`), horizontal Smagorinsky turbulence (`horizturb = 1`), Kessler; dumps the surface arrays, `mznt`, `mol`, `rmol`, `zol`, `br`, `gz1oz0`, `psim`, `psih`, `hpbl`, the PBL tendencies, `xkzh`, `xkzm` |
| `gfs_dry` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | GFS-EDMF (`ipbl = 3`) with the revised MM5 surface layer (`sfcmodel = 3`) and SLAB over land (`lu0 = 3`, `xland0 = 1`, `tsk0 = 305 K`), no horizontal turbulence, dry; dumps the surface arrays, `mol`, `tslb`, `hpbl`, the PBL tendencies, `xkzh`, `xkzm` |
| `gfs_sfclay_land` | 24×24×25, 2 km, 200 m | 150 steps (900 s) | every 10 steps | as `gfs_dry` with the MM5 surface layer (`sfcmodel = 2`) and Kessler microphysics |
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

## MPI references

The `mpi_*` cases are copies of the serial cases above (same physics and
grid, except `mpi_kessler_2x1`: `kessler_bubble` on 48×24 for 180 s, so that
2 ranks split the domain in x) that also run with MPI. Each has a `ranks`
file listing the rank counts (`2 4`), writes netCDF output
(`output_format = 2`; CM1's GrADS output does not work with MPI in double
precision) and dumps every 5 to 30 steps (`every`).
`mpi_kessler_tke_nomc` is `mpi_kessler_tke` with `apmasscon = 0`.

```bash
cxx/scripts/make_refs.sh     OUT_DIR    # serial: OUT_DIR/<case>/step_*  (all cases, incl. mpi_*)
cxx/scripts/make_refs_mpi.sh OUT_DIR    # MPI:    OUT_DIR/<case>/np<N>/rank_RRRR/step_*
```

`make_refs_mpi.sh` builds the Fortran with `USE_MPI`, `USE_DOUBLE` and
`USE_NETCDF` (`src/Makefile` maps `MPI_REAL` to `MPI_DOUBLE_PRECISION`
in double precision) and runs every case with a `ranks` file (knobs:
`CM1_BUILD_DIR`, `MPIRUN`, `MPIRUN_ARGS`, `JOBS`, `FC`). Every rank dumps its
own subdomain, halos included, to `rank_RRRR/` in the layout above;
`index.txt` has one more line, `#decomp nodex nodey myi myj myi1 myi2 myj1 myj2`
(the process grid and the rank's global index range). Both scripts can
write to the same OUT_DIR. In CI, job `fortran-refs-mpi` uploads the
artifact **`cm1-refs-mpi`**, and job `cm1k-mpi` downloads it and `cm1-refs`
into one directory.

CM1 picks the process grid (`nodex × nodey`) from the rank count and the
grid: 32×32, 40×40 and 24×24 run as 1×2 on 2 ranks and 2×2 on 4 ranks;
48×24 as 2×1 and 2×2.

`tests/mpi` (`ctest -L mpi`: tests `mpi_tests_np2`, `mpi_tests_np4`) starts
each rank of cm1k from its own step-0 dump, steps, and compares every dumped
array of the rank, halos included, with the MPI Fortran at the same rank
count at tolerance 0. It also prints the largest interior difference from
the serial Fortran run of the case.

Rank-count dependence of the Fortran (gfortran 13.3, OpenMPI 4.1, 2026-10):
with `apmasscon = 1` (all cases except `mpi_kessler_tke_nomc`) the global
mass sum is an `MPI_ALLREDUCE` of per-rank partial sums, so the result
depends on the rank count at round-off level. Interior max |np2 or np4 −
serial| over 150 steps is about 1e-10 to 2e-9 (absolute, mostly on `prs`
and `psfc`, ~1e5 Pa, so about 1e-15 relative) for most cases;
`mpi_pbl2_sfclay_land` reaches 2e-7, and the moist turbulent cases
`mpi_kessler_cor_diff6` (5e-5) and `mpi_myj_kessler` (1e-4) amplify it
further. With `apmasscon = 0` (`mpi_kessler_tke_nomc`) the Fortran is
bitwise identical at 1, 2 and 4 ranks, and so is cm1k (the test checks it).
