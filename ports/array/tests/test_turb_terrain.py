"""Terrain branches (terrain_flag=.true.) of the LES subgrid turbulence:
cm1x.turb.calcdef and cm1x.turbtend turbs/turbt/turbu/turbv/turbw against
the double-precision Fortran (fortran_harness/drv_turb_terrain.F90).  Inputs
are test_turb's random fields plus test_adv_terrain's Gal-Chen metrics."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver
from test_adv_terrain import make_metrics
from test_turb import CASES, DT, GRID1, KS, SH, TAU, G, A, As, close, make_state
from test_turb import params as _params

from cm1x import turb as T
from cm1x import turbtend as TT

pytestmark = requires_gfortran

CG = dict(cgs1=1.8, cgs2=-1.0, cgs3=0.2, cgt1=1.7, cgt2=-0.9, cgt3=0.2)
TM_IN = "rds sigma rdsf sigmaf gz rgz gzu rgzu gzv rgzv gx gxu gy gyv".split()
TCASES = {k: dict(CASES[k], terrain_flag=True) for k in CASES}


def metrics(cfg, seed=3):
    tm, zt = make_metrics(G, seed, cfg)
    return tm, zt


def fortran(mode, cfg, st, tm, zt, ins, outs, renames=None, **kw):
    renames = renames or {}
    p = np.concatenate([_params(mode, cfg, **kw), [zt, *CG.values()]])
    data = {"params": p, **{k: tm[k] for k in TM_IN}}
    for n in ins:
        data[n] = st[renames.get(n, n)]
    return run_driver("turb_terrain", data, {n: SH[n] for n in outs})


def terr(bk, tm, zt):
    d = {k: A(bk, tm[k]) for k in TM_IN}
    d.update(zt=zt, **CG)
    return d


@pytest.mark.parametrize("case", list(TCASES))
def test_calcdef_terrain(bk, case):
    cfg = TCASES[case]
    st = make_state(9)
    tm, zt = metrics(cfg)
    ins = [*GRID1, "c1", "c2", "rho", "rr", "rf", "ua", "va", "wa", *TAU]
    ins += ["defv", "defh"]
    ref = fortran(2, cfg, st, tm, zt, ins, [*TAU, "defv", "defh"])
    out = T.calcdef(
        bk,
        G,
        cfg,
        terrain=terr(bk, tm, zt),
        **As(bk, st, *GRID1, "c1", "c2", "rho", "rf", "ua", "va", "wa"),
        **{f"s{n[1:]}": A(bk, st[n]) for n in TAU},
        defv=A(bk, st["defv"]),
        defh=A(bk, st["defh"]),
    )
    for n in TAU:
        close(bk, out[f"s{n[1:]}"], ref[n], name=n)
    close(bk, out["defv"], ref["defv"], name="defv")
    close(bk, out["defh"], ref["defh"], name="defh")
    # the terrain path really differs from the flat one
    flat = T.calcdef(
        bk,
        G,
        dict(cfg, terrain_flag=False),
        **As(bk, st, *GRID1, "c1", "c2", "rho", "rf", "ua", "va", "wa"),
    )
    d = np.abs(bk.to_numpy(flat["defh"]) - ref["defh"]).max()
    assert d > 1e-3 * np.abs(ref["defh"]).max()


@pytest.mark.parametrize("case", list(TCASES))
def test_turbt_terrain(bk, case):
    cfg = TCASES[case]
    st = make_state(10)
    tm, zt = metrics(cfg)
    ref = fortran(
        6,
        cfg,
        st,
        tm,
        zt,
        [*GRID1, "rho", "rr", "rf", "t", "tten", *KS],
        ["tten"],
        renames={"t": "tkea", "tten": "tketen"},
    )
    out = TT.turbt(
        bk,
        G,
        cfg,
        dt=DT,
        t=A(bk, st["tkea"]),
        tten=A(bk, st["tketen"]),
        terrain=terr(bk, tm, zt),
        **As(bk, st, *GRID1, "rho", "rr", "rf", "kmh", "kmv"),
    )
    close(bk, out["tten"], ref["tten"], name="tten")


@pytest.mark.parametrize("case", list(TCASES))
def test_turbu_terrain(bk, case):
    cfg = TCASES[case]
    st = make_state(11)
    tm, zt = metrics(cfg)
    ref = fortran(
        7,
        cfg,
        st,
        tm,
        zt,
        [*GRID1, "ruh", "rmf", "rho", "rf", "u", "uten", "w", *TAU, *KS],
        ["uten"],
        renames={"u": "ua", "w": "wa"},
    )
    out = TT.turbu(
        bk,
        G,
        cfg,
        dt=DT,
        u=A(bk, st["ua"]),
        w=A(bk, st["wa"]),
        terrain=terr(bk, tm, zt),
        **As(bk, st, "ruh", "uf", "vh", "mh", "mf", "rho", "rf", "uten"),
        **As(bk, st, "t11", "t12", "t13", "kmv"),
    )
    close(bk, out["uten"], ref["uten"], name="uten")


@pytest.mark.parametrize("case", list(TCASES))
def test_turbv_terrain(bk, case):
    cfg = TCASES[case]
    st = make_state(12)
    tm, zt = metrics(cfg)
    ref = fortran(
        8,
        cfg,
        st,
        tm,
        zt,
        [*GRID1, "rvh", "rho", "rr", "rf", "v", "vten", "w", *TAU, *KS],
        ["vten"],
        renames={"v": "va", "w": "wa"},
    )
    out = TT.turbv(
        bk,
        G,
        cfg,
        dt=DT,
        v=A(bk, st["va"]),
        w=A(bk, st["wa"]),
        terrain=terr(bk, tm, zt),
        **As(bk, st, "uh", "rvh", "vf", "mh", "mf", "rho", "rr", "rf", "vten"),
        **As(bk, st, "t12", "t22", "t23", "kmv"),
    )
    close(bk, out["vten"], ref["vten"], name="vten")


@pytest.mark.parametrize("case", list(TCASES))
def test_turbw_terrain(bk, case):
    cfg = TCASES[case]
    st = make_state(13)
    tm, zt = metrics(cfg)
    ref = fortran(
        9,
        cfg,
        st,
        tm,
        zt,
        [*GRID1, "rho", "rf", "w", "wten", *TAU, *KS],
        ["wten"],
        renames={"w": "wa"},
    )
    out = TT.turbw(
        bk,
        G,
        cfg,
        dt=DT,
        w=A(bk, st["wa"]),
        terrain=terr(bk, tm, zt),
        **As(bk, st, "uh", "vh", "mh", "mf", "rho", "rf", "wten"),
        **As(bk, st, "t13", "t23", "t33", "kmh"),
    )
    close(bk, out["wten"], ref["wten"], name="wten")


@pytest.mark.parametrize("case", list(TCASES))
@pytest.mark.parametrize("iflux", [1, 0])
def test_turbs_terrain(bk, case, iflux):
    cfg = TCASES[case]
    st = make_state(14)
    tm, zt = metrics(cfg)
    ref = fortran(
        10,
        cfg,
        st,
        tm,
        zt,
        [*GRID1, "rho", "rr", "rf", "s", "sten", *KS, "dum7", "dum8"],
        ["sten"],
        iflux=iflux,
    )
    out = TT.turbs(
        bk,
        G,
        cfg,
        iflux=iflux,
        dt=DT,
        terrain=terr(bk, tm, zt),
        **As(bk, st, *GRID1, "rho", "rr", "rf", "s", "sten", "khh", "khv"),
        **As(bk, st, "dum7", "dum8"),
    )
    close(bk, out["sten"], ref["sten"], name="sten")


def test_turb_terrain_needs_metrics(bk):
    cfg = TCASES["supercell"]
    st = make_state(9)
    with pytest.raises(ValueError, match="terrain"):
        T.calcdef(
            bk,
            G,
            cfg,
            **As(bk, st, *GRID1, "c1", "c2", "rho", "rf", "ua", "va", "wa"),
        )


def test_turb_terrain_jit():
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    cfg = TCASES["supercell"]
    st = make_state(9)
    tm, zt = metrics(cfg)
    names = (*GRID1, "c1", "c2", "rho", "rr", "rf", "ua", "va", "wa", "kmh", "kmv")
    a = As(bk, st, *names)
    m = {k: A(bk, tm[k]) for k in TM_IN}

    def f(a, m):
        tr = dict(m, zt=zt, **CG)
        d = T.calcdef(bk, G, cfg, terrain=tr, **a)
        u = TT.turbu(
            bk,
            G,
            cfg,
            dt=DT,
            u=a["ua"],
            w=a["wa"],
            uten=bk.zeros(G.u_shape),
            ruh=1.0 / a["uh"],
            t11=d["s11"],
            t12=d["s12"],
            t13=d["s13"],
            terrain=tr,
            **{k: a[k] for k in ("uf", "vh", "mh", "mf", "rho", "rf", "kmv")},
        )
        return d["defh"], u["uten"]

    e = f(a, m)
    j = jax.jit(f)(a, m)
    for x, y in zip(e, j):
        close(bk, y, bk.to_numpy(x), rtol=1e-13)
