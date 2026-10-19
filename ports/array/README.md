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

## Boundary layer (cm1setup=2) and surface layers

`pbl.py` holds the mesoscale path of `turb.F sfc_and_turb` (`sfc_and_turb_pbl`:
dissten=0, the surface layer, bc2d of the surface arrays, calcnm, calcdef,
`turbparam_horiz` (horizturb=1), `turbparam_vert` (ipbl=2), gettau, the PBL
scheme, bcs(upten/vpten)), the solve1 `use_pbl` blocks and the param.F option
adjustments (`pbl.apply_options`: bbc=3, isfcflx, tconfig=2, l_inf, dohturb/dovturb,
use_pbl, dosfcflx). PBL schemes and surface layers plug in through a registry
in `pbl.py`; adding one is a one-line entry:

```python
PBL_SCHEMES = {2: "cm1x.pbl", 1: "cm1x.pbl_ysu", ...}          # ipbl -> module
SURFACE_LAYERS = {1: "cm1x.sfcphys", 2: "cm1x.sfclay", ...}    # sfcmodel -> module
```

The module defines `pbl(ctx)` (or `surface_layer(ctx)`) returning a dict of
state updates, and optionally `options(cfg)` (param.F adjustments, run once),
`check(cfg)`, `init_state(ctx)` (arrays carried across steps), `WRF_INPUTS =
True` and `DUMP_KEYS`. `ctx` (`pbl.Context`) carries `bk, g, cfg, dt, nstep,
update_sfc, rtime`, the model state `s` (surface arrays 2-D `[j, i]` with
halos; K arrays, `dissten`, the PBL tendencies `thpten, qvpten, qcpten,
qipten, upten, vpten` 3-D), the base state `b`, the new time level `u3d, v3d,
w3d, pp3d, th3d, q3d` with `rho, rr, rf, prs`, and `ctx.w`, the WRF column
inputs of turb.F (`pbl.wrf_inputs`: u, v at scalar points, th, pi, t, dz8w,
p3di, qv, qc, qi, rthraten). PBL tendencies are returned exactly as the
Fortran scheme stores them (solve1 adds `thpten`, `q*pten` and the u/v-point
average of `upten`/`vpten` with no extra factor; any rho weighting is the
scheme's own, e.g. ipbl=2) and filled on the interior; the core applies bcs to
upten/vpten. WRF-type surface layers end with `pbl.wrf_sfc_epilogue` (SLAB
when `update_sfc`, else the qsfc diagnosis; ch, cq, s10; thflux, qvflux).

Supported here: cm1setup=2 with ipbl=2 (Bryan 2009 simple PBL) and
horizturb=1; LES (cm1setup=1, sgsmodel=1) with a surface layer; sfcmodel=1
(land and water, isfcflx, cecd 1-3, set_flx, set_znt, pertflx), sfcmodel=2
(`sfclay.py`, MM5 similarity) and 3 (revised MM5, isftcflx 0-2, iz0tlnd 0-2)
with the SLAB land model (`slab.py`, 5 soil layers); irandp=1 and isnd=1 in
the initial state. The sfclayinit tables call glibc's libmvec (as gfortran's
vectorised loop does) when it is available.

`tests/test_model_dumps_pbl.py` compares every array a CM1 dump holds with
the model state (`CM1X_DUMP_DIR_PBL`, layout `<case>/step_NNNNNN`; cases in
`CASES`, overridable with `CM1X_PBL_CASES`, steps capped by
`CM1X_PBL_MAX_STEP`): prognostics against step N, surface / K / PBL arrays
against step N-1 (CM1 dumps before sfc_and_turb). Tolerance 1e-10 relative to
each array's maximum; on jax and torch the PBL tendencies get 1e-9 (their
(new-old)/dt cancellation amplifies last-bit libm differences). The
surface-layer variant namelists without a C++ case live in
`ports/reference/pbl/`. On numpy the ipbl=2, cbl_sfc and sfcmodel=2/3 cases
match the float64 Fortran to 1e-12 or bitwise through 150 steps.

### MYJ (ipbl=6, sfcmodel=7)

`sfc_myj.py` ports module_sf_myjsfc.F (`myjsfcinit`'s psi tables, built
with libm on the host; `myjsfc`; `sfcdif`, water and land branches,
isftcflx 0/1), and `pbl_myj.py` ports module_bl_myjpbl.F (`myjpbl`, `mixlen`,
`prodq2`, `difcof`, `vdifq`, `vdifh`, `vdifv`; STEPBL=1, QCI mixed). They work
on every column at once, with the downward-counted column (K = KTE+1-k) kept
as Python lists of planes, and the level-wise parts stacked along K.
`surface_layer(ctx)` and `pbl(ctx)` hold the turb.F glue, with cm1k's
`cm1k_myj_coupling` as the reference. tke_myj is prognostic but not advected.
Harness: `drv_myjsfc`, `drv_myjpbl` (`tests/test_myj.py`): 1e-12 on numpy and
1e-11 on jax/torch, measured against each field's magnitude. The remaining
differences are the last bit of exp/log/pow. Over land, CM1 never sets zntmyj
(it stays 0), so the Fortran goes NaN; the land path is tested only per routine,
with positive roughness. Not ported: sfcmodel=7 without ipbl=6, cm1setup 4, terrain.

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
