# cm1x — CM1 on any array backend

`cm1x` is a Python port of CM1 written once against the
[Python array API standard](https://data-apis.org/array-api/) (through
`array-api-compat`), so the same code runs on **numpy, jax, torch** (and cupy /
dask where available). It lives only on this fork; nothing here is upstreamed.

Initial scope: the dry/moist dynamical core used by `psolver=3` (RK3 + vertically
implicit acoustic steps, 5th-order flux-form advection) and Kessler warm-rain
microphysics, on a uniform periodic grid without terrain. Reference case:
`ports/reference/kessler_periodic/namelist.input`.

```bash
pip install -e ports/array[test]     # plus jax and/or torch if wanted
pytest ports/array                    # every test runs on each installed backend
```

### Terrain (`terrain_flag = .true.`)

The terrain-following (Gal-Chen) coordinate is ported on the psolver=3 path:
`init_terrain` (itern 1-3), the metrics of param.F (`init_terrain.terrain_metrics`),
the 3-D base state and its terrain surface branch (base.F), `bcwsfc`, and the
terrain branches of advection (flux form, WENO, pdef / pdefweno, `getdivx`,
`getdiv`), the acoustic solver (`sound`, `ssopenbc*` with open radiative BCs),
Rayleigh damping (irdamp=1) and the prognostic-TKE LES closure (sgsmodel=1:
`calcdef`, `turbs`, `turbt`, `turbu`, `turbv`, `turbw`), with Kessler. Per-routine
harnesses `drv_terrain`, `drv_adv_terrain`, `drv_sound_terrain`,
`drv_base_terrain`, `drv_turb_terrain`; whole-model cases
`ports/reference/nh_mountain_waves` (2-D x-z mountain waves, `CM1X_REF_DIR_NHMW`,
`CM1X_REF_DIR_NHMW_LONG`) and `ports/reference/kessler_hill` (moist 3-D hill
with TKE, `CM1X_REF_DIR_KH`, `CM1X_REF_DIR_KH_LONG`); see
`tests/test_model_terrain.py`. Raises `NotImplementedError`: itern=4 (terrain read
from `perts.dat`), imove=1 with terrain; everything not ported without terrain
(e.g. rigid-wall advection, surface fluxes, PBL schemes) stays unported with it.

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
(`tests/test_jit.py`). On 4 CPUs, supercell_kessler at 40^3 runs 0.12 s/step
with jax.jit (~22 s compile) vs 0.58 s numpy and 0.36 s torch eager (4
threads); 6.6x numpy at 80^3. `Model(..., overrides={"nfall_max": 4})` gives
the sedimentation loop a static trip count so reverse-mode `jax.grad` works
through the jitted step. Sweep and notes (torch.compile, dask,
array-api-strict): `bench/sweep.py`, `bench/results_cpu.md`.
