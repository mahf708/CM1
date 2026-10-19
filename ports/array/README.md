# cm1x — CM1 on any array backend

`cm1x` is a Python port of CM1 written once against the
[Python array API standard](https://data-apis.org/array-api/) (through
`array-api-compat`), so the same code runs on **numpy, jax, torch** (and cupy /
dask where available). It lives only on this fork; nothing here is upstreamed.

Initial scope: the dry/moist dynamical core used by `psolver=3` (RK3 + vertically
implicit acoustic steps, 5th-order flux-form advection) and Kessler warm-rain
microphysics, on a uniform periodic grid without terrain. Reference case:
`ports/reference/kessler_periodic/namelist.input`.

Ice microphysics: Morrison two-moment (`ptype=5`, `cm1x/morrison.py`) runs
through the same `mp_driver`; the generic ice-scheme wiring (param.F moisture
tables for ptype=3/5, `iice=1` paths in solve2/turb/pdef/getcvm/init3d) is in
`cm1x/namelist.py` and the dynamics modules. Morrison notes:

* All columns and levels are computed at once; the Fortran's per-level GOTOs
  become masks, sedimentation sub-steps run as masked iterations
  (`bk.fori_loop`; `nfall_max` gives a static trip count for reverse-mode AD
  under jit), the fall-speed fill below the precipitation edge is a `bk.scan`.
* Switches as CM1 hard-wires them (IACT=2, IBASE=1, ISUB=1, ILIQ=0, INUC=0,
  IGRAUP=0, IRAIN=0); `ihail`, `ndcnst`, `eqtset`, `efall`, `testcase=5` from
  the namelist. `inum=0` (predicted droplet number, 11 moisture arrays) is a
  cm1x option that CM1 cannot select without editing GRAUPEL_INIT.
  Not ported: reflectivity (`getdbz`), effective radii (`dorad`), the
  dotbud/doqbud budget diagnostics.
* Harness: the scheme's routines are private to `module_mp_GRAUPEL`;
  `tests/fortran.py` can link a module's *public variant*
  (`build_driver(..., public=("morrison",))`, `public_module`) so
  `drv_morrison.F90` calls GRAUPEL_INIT, GAMMA, DERF1, satadj2 and MP_GRAUPEL
  directly; `drv_morrison_mp.F90` runs CM1's mp_driver. numpy reproduces them
  bit for bit (with numpy's AVX-512 libm disabled, `NPY_DISABLE_CPU_FEATURES`),
  jax to ~1e-14. Whole-model cases: `ports/reference/morrison_bubble`
  (300 steps, ihail=1) and `morrison_efall` (200 steps, efall=1;
  `tests/test_model_dumps_morrison.py`, `CM1X_DUMP_DIR_MORRISON`): numpy is
  bitwise against CM1's float64 dumps at every dumped step, jax ~1e-11.
* Real powers go through `_pow` (libm `pow`, as gfortran calls it even for
  `x**0.5`): `np.pow`/`**` with a scalar exponent of 0.5, 2 or -1 (and
  torch's `pow` with several more) take a sqrt/square/reciprocal fast path
  that can differ from `pow` in the last bit, so `_pow` passes those
  exponents as a full array. One such bit in the graupel fall speed
  (`DLAMG**BG`, BG=0.5) was enough to make morrison_bubble diverge after
  step 246.

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

## Performance

`Model(nml, backend="jax", jit=True)` compiles the whole time step into one
`jax.jit` function (state -> state); results equal the eager step to round-off
(`tests/test_jit.py`). On 4 CPUs, supercell_kessler at 40^3 runs 0.14 s/step
with jax.jit (~25 s compile) vs 0.57 s numpy and 0.36 s torch eager (4
threads); 5.5x numpy at 80^3. `Model(..., overrides={"nfall_max": 4})` gives
the sedimentation loop a static trip count so reverse-mode `jax.grad` works
through the jitted step.

`Model(nml, backend="torch", jit=True)` instead `torch.compile`s the hot
routines one by one (`Backend.kernel`: advection, the acoustic small step,
subgrid turbulence, Kessler; inductor with `dynamic=False`,
`fullgraph=True`) and runs the rest of the step eagerly: 0.20 s/step at 40^3
(1.8x torch eager) and 0.89 s at 80^3 (2.0x torch eager, 6.1x numpy, ahead
of jax.jit). Compiling takes ~2.5-3.5 min with a cold inductor cache, ~30 s
warm (inductor's on-disk cache); `jit_options` update the torch.compile
settings (`cm1x.backend.TORCH_KERNEL_DEFAULTS`), and
`jit_options={"whole_step": True}` compiles the whole step (impractically
slow to compile). Torch autograd works through the compiled routines (same
gradient as eager) but compiles a second set of graphs (~5 min cold).
Sweep and notes (dask, array-api-strict): `bench/sweep.py`,
`bench/results_cpu.md`.
