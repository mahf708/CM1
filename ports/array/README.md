# cm1x — CM1 on any array backend

`cm1x` is a Python port of CM1 written once against the
[Python array API standard](https://data-apis.org/array-api/) (through
`array-api-compat`), so the same code runs on **numpy, jax, torch** (and cupy /
dask where available). It lives only on this fork; nothing here is upstreamed.

Initial scope: the dry/moist dynamical core used by `psolver=3` (RK3 + vertically
implicit acoustic steps, 5th-order flux-form advection) and Kessler warm-rain
microphysics, on a uniform periodic grid without terrain. Reference case:
`ports/reference/kessler_periodic/namelist.input`. Since added: open lateral
boundaries, WENO, the TKE LES closure (`supercell_kessler`) and RRTMG
radiation (`kessler_rrtmg`, below).

```bash
pip install -e ports/array[test]     # plus jax and/or torch if wanted
pytest ports/array                    # every test runs on each installed backend
```

## Porting conventions (binding for every module)

1. **One Python function per Fortran subroutine**, same name, in the module named
   after the Fortran file (`adv.F` → `cm1x/adv.py`, `kessler.F` → `cm1x/kessler.py`).
   Arguments keep the Fortran dummy-argument names (lower case) and are passed by
   keyword. Scalars that come from Fortran modules (`input`, `constants`) are read
   from `cm1x.constants` or from a `cfg` (namelist) object instead of arguments.
2. **Return, don't mutate.** Every `intent(out)` / `intent(inout)` array comes back
   in a dict keyed by its Fortran name. Inside, update arrays with
   `x = bk.set(x, idx, v)` / `bk.add(...)`, never `x[idx] = v` (jax arrays are
   immutable; numpy/torch updates happen in place).
3. **Backend only through `bk`.** `bk.xp` is the array-API namespace; `bk.dtype` the
   float type. No direct `import numpy` for computation (tests may use numpy).
   No Python `if`/`for` on array *values* (keeps `jax.jit` working): a loop whose
   trip count depends on the data goes through `bk.fori_loop(n, body, carry)`
   (`n` may be a 0-d array; `lax.while_loop` under jit), a repeated identical
   sub-step through `bk.fori_loop` with a static `n`, and a vertical recurrence
   (tridiagonal solves, sequential sums) through `bk.scan` on level-stacked
   coefficients computed vectorised over `[k, j, i]`. Eagerly these are plain
   Python loops doing the same operations; under `jax.jit` they stay loops in the
   compiled program instead of being unrolled.
4. **Layout.** Arrays are `a[k, j, i]` and keep CM1's halos (3 in x/y, 1 in z), so
   Fortran index `i` is Python index `i + 2` and Fortran `k` is Python `k`
   (`cm1x.grid.fi`, `fk`). Shapes: `Grid.s_shape`, `u_shape`, `v_shape`, `w_shape`.
   1-D vertical arrays (base state, `mh`, `rmh`, `mf`…) are carried as 3-D arrays
   exactly as CM1 stores them (e.g. `rho0(ib:ie,jb:je,kb:ke)`).
5. **Verification.** Each routine is checked against the original Fortran routine
   through the per-routine harness (`tests/fortran.py`, drivers in
   `fortran_harness/drv_<name>.F90`, linked against CM1's own object files built in
   double precision by `fortran_harness/build_objs.sh`). Target: agreement to
   ~1e-12 relative in float64, on random-but-physical inputs, on every backend.
   Whole-model runs are checked against CM1's netCDF output for the reference case.

## Radiation (radopt=2, RRTMG)

`module_ra_rrtmg_lw.py` / `module_ra_rrtmg_sw.py` port RRTMG as CM1 calls it
for warm-rain microphysics (ptype=1: qv, qc, qr; McICA maximum-random overlap
with the KISS generator; no aerosols; o3 from the built-in profile), every
column at once (`[layer, column, g-point]`), with the k-distribution tables
read from `run/RRTMG_LW_DATA` / `RRTMG_SW_DATA` and cached per backend.
`radiation_driver.py` is the radopt=2 branch of `radiation_driver.F`
(`radconst`, `calc_coszen`, `cal_cldfra1`, the p8w/t8w level interpolation,
the eqtset temperature-tendency conversion and `thraten`); `init_surface.py`
reads tsk0 and the LANDUSE.TBL albedo / emissivity / z0 / xland. `Model`
follows cm1.F's timing: radiation on the initial state, then at the end of
every step with `mtime + dt >= radtim - 0.1 dt` (dtrad = max(dtrad, dt)),
coszen at the middle of the radiation interval; `thraten` is added to the
theta tendency in solve1 until the next call. With `jit=True` the step is
compiled twice (with and without the radiation call).

CM1 requires a surface model with radiation, and sfcmodel>=1 forces the
semi-slip lower boundary (bbc=3), so the sfcmodel=1 surface layer without
fluxes is ported too (`sfcphys.getcecd`, `turb.sfcstress`, the turbu/turbv
surface drag at zero eddy viscosity, the bbc=3 surface-layer dissipative
heating) for sgsmodel=0. Not ported (raise `NotImplementedError`): radopt=1,
ice microphysics, the PBL subgrid-cloud coupling, testcase=8, surface fluxes
(isfcflx=1), sfcmodel>=1 with sgsmodel=1, use_avg_sfc. The McICA
generator needs 64-bit integers (jax: `jax_enable_x64`, else
`NotImplementedError`). Reference case: `ports/reference/kessler_rrtmg`
(daytime, dtrad=12 s). Partially cloudy layers are sensitive to the McICA
seeds (derived from the layer pressures: a 1e-13 relative change in
pressure changes their heating rates by up to ~10%), so once they appear
(after ~15 min with dtrad=60 s) the radiative fields agree with CM1 to
~2e-3 and, through them, the dynamics to ~1e-5 of each field's maximum
(30 min, measured); before that everything agrees to the float32 rounding
of the output.

The Fortran tests need one source change, shared with the cm1k fork: param.F
and radiation_driver.F accept radopt=2 with ptype=1 (WRF's mp_physics=1), and
the RRTMG table arrays are 4-byte reals (`kind_rf`), so a `-fdefault-real-8`
build reads the 4-byte data files unchanged.

## Performance

`Model(nml, backend="jax", jit=True)` compiles the whole time step into one
`jax.jit` function (state -> state); results equal the eager step to round-off
(`tests/test_jit.py`). On 4 CPUs, supercell_kessler at 40^3 runs 0.12 s/step
with jax.jit (~22 s compile) vs 0.58 s numpy and 0.36 s torch eager (4
threads); 6.6x numpy at 80^3. `Model(..., overrides={"nfall_max": 4})` gives
the sedimentation loop a static trip count so reverse-mode `jax.grad` works
through the jitted step. Sweep and notes (torch.compile, dask,
array-api-strict): `bench/sweep.py`, `bench/results_cpu.md`.
