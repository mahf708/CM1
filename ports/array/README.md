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
   No Python `if`/`for` on array *values* (keeps `jax.jit` working); loops over the
   vertical are fine when the Fortran recurrence needs them (tridiagonal solves,
   sedimentation), everything else vectorised over `[k, j, i]`.
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
