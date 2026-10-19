# cm1k: CM1 in C++/Kokkos

`cm1k` is a port of CM1 (the Fortran model in `../src`) to C++ with
[Kokkos](https://github.com/kokkos/kokkos), built on E3SM's
[EKAT](https://github.com/E3SM-Project/EKAT) in the style of EAMxx. It runs on
CPUs (Serial/OpenMP) and GPUs (CUDA/HIP) from one source, and has Python
bindings (`python/`).

The Fortran model stays the reference: every ported piece is tested against
it, either routine by routine (through the Fortran bridge) or step by step
(through reference dumps).

## Build

```bash
git submodule update --init --recursive    # EKAT and its Kokkos
# EKAT's Kokkos submodule uses an ssh URL; over https:
#   git config --global url."https://github.com/".insteadOf git@github.com:

# Fortran references (gfortran, double precision)
cxx/scripts/make_refs.sh /path/to/refs            # all cases in cxx/cases

cmake -S cxx -B build -DKokkos_ENABLE_OPENMP=ON -DCM1K_REF_DIR=/path/to/refs
cmake --build build -j
cd build && ctest --output-on-failure
```

OpenMPI refuses to run as root unless `OMPI_ALLOW_RUN_AS_ROOT=1` and
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` are set (ctest launches tests via mpiexec).

Options: `CM1K_ENABLE_FORTRAN_BRIDGE` (default ON, needs gfortran),
`CM1K_ENABLE_PYTHON` (pybind11 module), `CM1K_REF_DIR`.

## Layout

| Path | What |
|---|---|
| `src/cm1k/share` | types, `FArray` (Fortran-indexed arrays), constants, namelist, grid, state, dump I/O |
| `src/cm1k/dynamics` | dynamical core: advection, acoustic solver, boundary conditions, solve1/2/3 |
| `src/cm1k/physics/<scheme>` | physics (e.g. `kessler`; `rrtmg`: RRTMG radiation, which reads `RRTMG_LW_DATA` / `RRTMG_SW_DATA` from `$CM1K_RRTMG_DATA_DIR`, the working directory or the source tree's `run/`) |
| `src/cm1k/init` | native initialization (grid, base state, initial perturbations) |
| `src/cm1k/driver` | `Model` (the public API) and the `cm1k` executable |
| `python/` | pybind11 module `cm1k` |
| `tests/<group>` | Catch2 tests; each subdirectory with a CMakeLists.txt is picked up |
| `tests/fortran_bridge` | bind(C) shims around Fortran routines + component tests |
| `cases/<name>` | namelists used for reference runs and step-level tests |
| `scripts/make_refs.sh` | builds the Fortran and writes reference dumps |

Sources are globbed, so adding a file never requires editing a shared
CMakeLists.txt.

## Conventions (read before porting)

- **Same names, same indices as the Fortran.** Arrays are `FArray<Rank>` with
  Fortran bounds and Fortran (LayoutLeft, i-fastest) memory order:
  `ua(i,j,k)` in C++ is `ua(i,j,k)` in Fortran. Keep variable and routine
  names (`advs`, `sound`, `tha`, `ppi`, `rds`, ...), so the two read side by side.
- **Loops:** `!$omp parallel do` nests become
  `Kokkos::parallel_for(box3(i1,i2,j1,j2,k1,k2), KOKKOS_LAMBDA(int i,int j,int k){...})`
  with inclusive Fortran bounds. Vertical recurrences (tridiagonal solves,
  sedimentation) are a `parallel_for` over columns with a serial `k` loop inside.
- **Round-off parity.** We match the double-precision Fortran to round-off,
  so: keep the Fortran's operation order and parenthesization; `x**2` is
  `x*x`, `x**0.5` is `std::pow(x,0.5)` unless the Fortran says `sqrt`; real
  literals are doubles; FMA contraction is off (`-ffp-contract=off`, also in
  the Fortran reference build). Do not "simplify" expressions.
- **Same math library paths.** glibc chooses libm/libmvec code paths (FMA,
  AVX2, ...) from the CPU at run time, and those can differ in the last bit.
  When references and tests run on different machines, run both with
  `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX512F,-AVX2,-FMA,-FMA4` (CI does).
- **Options:** read namelist options with `Namelist::geti/getr/getb`, using
  param.F's default. Unsupported option values must throw (`fail loudly`),
  never silently run something different.
- **Constants:** `Constants` (from `constants.F`, depends on `testcase`) is a
  small struct; capture it by value in kernels. Compile-time ones are in `phys::`.
- **Kernels** must be device-safe: no `std::` math except what Kokkos maps
  (use `Kokkos::pow`, `Kokkos::sqrt`, `Kokkos::exp`, `Kokkos::fabs`, ...), no
  virtuals, no host containers captured. Capture `FArray`s and structs by value.

## Testing

Two levels, both required for a ported component:

1. **Component tests** (`tests/fortran_bridge`): a `bind(C)` shim calls the
   original Fortran routine on synthetic or dumped inputs; the test runs the
   C++ port on the same inputs and compares (bit-for-bit where possible,
   otherwise a relative tolerance of ~1e-13 with a comment saying why).
   Shims set whatever `input`-module variables the routine reads (`ni`, `nk`,
   `ib`, ..., option flags). See `cm1f_bridge_constants.F90` for the pattern.
2. **Step tests** (`tests/<group>`): load `step_000000` of a reference case
   with `Model::init_from_dump`, advance with `Model::step`, compare every
   prognostic array against `step_NNNNNN` (`cm1k::diff`). Use
   `cm1k::testing::case_ref("<case>")`; tests skip with a warning when no
   references are available, and CI always provides them.

## Terrain-following coordinate

`terrain_flag = .true.` (Gal-Chen and Somerville sigma coordinate) is ported
on these paths, bit-for-bit with the double-precision gfortran build:
native initialization (`itern = 1, 2, 3`: init_terrain.F, zsgrad and the
metrics of param.F; the metrics are dumped and compared at step 0), scalar
and momentum advection (including WENO, `pdefx/y/z`, 6th-order diffusion),
the acoustic solver (`psolver = 3`, with `bcwsfc`), open-radiative
boundaries (`ssopenbc*`), Rayleigh damping (`irdamp`/`hrdamp = 1`),
Coriolis and `lspgrad = 1`, the cm1r19 `termod1` pressure-gradient term,
Kessler, the LES TKE scheme (`sgsmodel = 1`, `doimpl = 0/1`, bbc = 1 or
bbc = 3 with the surface layer `sfcmodel = 1`), the PBL schemes YSU
(`ipbl = 1`), the simple scheme (`ipbl = 2`) and GFS-EDMF (`ipbl = 3`) with
the surface layers `sfcmodel = 1, 2, 3` and SLAB, and RRTMG (`radopt = 2`).
turb.F's terrain lines of the surface block are the only terrain-specific
physics: `za = zh(k=1) - zs` for `getcecd`/`sfcdiags` and `hpbl = hpbl - zs`
before every surface layer call (also for the PBL schemes' own `hpbl`, and
for the LES, where `hpbl` keeps losing `zs` every step); `sfcstress`,
`radiation_driver.F` and the PBL drivers have no terrain branch and read
`zh`, `zf`, `dz*rmh` and the `cgs`/`cgt` pressures per column, as the ports
do (GFS-EDMF gets `z = zf`, heights above sea level).
Unported combinations throw: `difforder = 2` (diff2.F stops), `imove = 1`,
`itern = 0` or 4, the GFDL (`sfcmodel = 4`), MYNN (`sfcmodel = 6`,
`ipbl = 4, 5`) and MYJ (`sfcmodel = 7`, `ipbl = 6`) schemes, and everything
param.F already refuses with terrain.
Tests: cases `nh_mountain_waves`, `nh_mountain_waves_long`,
`kessler_terrain_tke`, `kessler_terrain_tke_openbc`,
`kessler_terrain_tke_sfc`, `pbl2_terrain_kessler`, `ysu_terrain_dry`,
`gfs_terrain_dry` and `rrtmg_ysu_terrain` (init, step from dump, step from
`Model::init`), and the `[terrain]` component tests
(adv.F, turb.F/turbtend.F with random metrics). Periodic halo copies wrap their source index (`pwrap` in
`dynamics/cm1k_bc.hpp`), which also makes them correct for `nj < ngxy`
(the 2D `ny = 1` case).

## Reference dumps

`src/cm1dump.F` (Fortran, enabled only when `CM1_DUMP_DIR` is set) writes
`$CM1_DUMP_DIR/step_NNNNNN/{index.txt,<name>.bin}` at step 0 and every
`CM1_DUMP_EVERY` steps: grid metrics, base state, `ua va wa ppi tha rho prs`,
`qa_<qname>`, `tkea kmh kmv khh khv rain` (plus the surface, PBL and, with
`radopt = 2`, radiation arrays: see `src/cm1dump_state.inc`). Raw float64 in Fortran order,
halos included; `index.txt` lists `name kind ndims lb ub ...`.
`Model::write_dump` writes the same format, so the Python tools and tests
can compare either side.
