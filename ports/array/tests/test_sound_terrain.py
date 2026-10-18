"""Terrain branch of cm1x.sound.sound vs. the Fortran sound (psolver=3,
drv_sound_terrain.F90), periodic and open lateral boundaries."""

import numpy as np
import pytest

from cm1x.sound import sound

from fortran import requires_gfortran, run_driver
from test_adv_terrain import make_metrics
from test_sound import BCN, CFG, DT, NI, NJ, NK, NSOUND, _compare, make_inputs
from test_sound import out_shapes

CG = dict(cgs1=1.8, cgs2=-1.0, cgs3=0.2, cgt1=1.7, cgt2=-0.9, cgt3=0.2)
TM = "rds sigma rdsf sigmaf gz rgz gzu rgzu gzv rgzv dzdx dzdy gx gxu gy gyv".split()


@pytest.fixture(scope="module")
def grid():
    from cm1x.grid import Grid

    return Grid(ni=NI, nj=NJ, nk=NK, dx=1000.0, dy=1000.0, dz=500.0)


def inputs(g, seed, bcs):
    d = make_inputs(g, seed=seed, bcs=bcs)
    cfg = {**dict(zip(BCN, bcs)), "terrain_flag": True}
    tm, zt = make_metrics(g, seed, cfg)
    assert set(TM) <= set(tm)
    d["rho0"] = 1.0 / d["rr0"]
    d["rrf0"] = 1.0 / d["rf0"]
    return d, tm, zt


def fortran(g, d, tm, zt, nrk, bcs):
    params = np.array(
        [g.ni, g.nj, g.nk, NSOUND, nrk, DT, CFG["alph"], CFG["kdiv"]]
        + [g.dx, g.dy, g.dz, CFG["smeps"], *bcs, zt, *CG.values()],
        dtype=float,
    )
    return run_driver("sound_terrain", {"params": params, **d, **tm}, out_shapes(g))


def python(bk, g, d, tm, zt, nrk, bcs):
    a = {k: bk.asarray(np.array(v)) for k, v in d.items()}
    m = {k: bk.asarray(np.array(v)) for k, v in tm.items()}
    cfg = {**CFG, **dict(zip(BCN, bcs)), "terrain_flag": True}
    return sound(bk, g, cfg, dt=DT, nrk=nrk, zt=zt, cg=CG, **a, **m)


@requires_gfortran
@pytest.mark.parametrize(
    "bcs",
    [(1, 1, 1, 1), (2, 2, 2, 2), (2, 2, 1, 1)],
    ids=lambda c: "bc" + "".join(map(str, c)),
)
@pytest.mark.parametrize("nrk", [1, 3])
def test_sound_terrain_vs_fortran(bk, grid, nrk, bcs):
    d, tm, zt = inputs(grid, 20 + nrk, bcs)
    ref = fortran(grid, d, tm, zt, nrk, bcs)
    res = python(bk, grid, d, tm, zt, nrk, bcs)
    _compare(bk, res, ref)


@requires_gfortran
def test_sound_terrain_jit(grid):
    pytest.importorskip("jax")
    import jax

    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    bcs = (2, 2, 2, 2)
    d, tm, zt = inputs(grid, 7, bcs)
    ref = fortran(grid, d, tm, zt, 3, bcs)
    cfg = {**CFG, **dict(zip(BCN, bcs)), "terrain_flag": True}
    f = jax.jit(lambda a, m: sound(bk, grid, cfg, dt=DT, nrk=3, zt=zt, cg=CG, **a, **m))
    res = f(
        {k: bk.asarray(v) for k, v in d.items()},
        {k: bk.asarray(v) for k, v in tm.items()},
    )
    _compare(bk, res, ref)
