"""Terrain branches of advs (with pdef and WENO), advu, advv, advw (adv.F) and
getdivx / getdiv (misclibs.F) vs CM1's Fortran (drv_adv_terrain.F90).  The
metrics come from terrain_metrics on a smooth random 2-D surface height."""

import numpy as np
import pytest

from cm1x import adv, misc
from cm1x.bc import bc2d
from cm1x.init_terrain import terrain_metrics

from fortran import requires_gfortran, run_driver
from test_adv import DT, G, _cmp, _regions, make_inputs

TM_KEYS = "gz rgz gzu rgzu gzv rgzv dzdx dzdy rds rdsf sigma sigmaf".split()


def make_metrics(g, seed, cfg):
    """Gal-Chen metrics for a smooth hill plus noise on stretched levels."""
    from cm1x import get_backend

    bk = get_backend("numpy")
    rng = np.random.default_rng(seed)
    nk = g.nk
    S, U, V = g.s_shape, g.u_shape, g.v_shape
    dzs = g.dz * (1.0 + 0.3 * rng.uniform(size=nk + 2))
    sigmaf = np.concatenate([[-dzs[0]], np.cumsum(dzs) - dzs[0]])
    sigma = 0.5 * (sigmaf[:-1] + sigmaf[1:])
    zt = float(sigmaf[nk + 1])
    x = (np.arange(S[2]) - g.ngxy + 0.5) * g.dx
    y = (np.arange(S[1]) - g.ngxy + 0.5) * g.dy
    xc, yc = 0.5 * g.ni * g.dx, 0.5 * g.nj * g.dy
    r2 = ((x[None] - xc) / 2500.0) ** 2 + ((y[:, None] - yc) / 2000.0) ** 2
    zs = 600.0 / (1.0 + r2) + 20.0 * rng.standard_normal(S[1:])
    zs = bc2d(bk, g, zs, cfg)
    t = terrain_metrics(
        bk,
        g,
        cfg,
        zs=zs,
        uh=1.0 + 0.02 * rng.standard_normal(S[2]),
        uf=1.0 + 0.02 * rng.standard_normal(U[2]),
        vh=1.0 + 0.02 * rng.standard_normal(S[1]),
        vf=1.0 + 0.02 * rng.standard_normal(V[1]),
        rds=np.ones(nk + 2),
        sigma=sigma,
        rdsf=np.ones(nk + 3),
        sigmaf=sigmaf,
        zt=zt,
    )
    m = {k: np.asarray(t[k]) for k in TM_KEYS if k in t}
    m.update(sigma=sigma, sigmaf=sigmaf)
    return m, zt


def _cfg(obc, advwenos):
    return dict(
        wbc=obc, ebc=obc, sbc=obc, nbc=obc, advwenos=advwenos, terrain_flag=True
    )


def fortran(inp, tm, zt, nrk, pdef, obc, advwenos, pdefweno, wepsf):
    g = G
    p = np.array(
        [g.ni, g.nj, g.nk, nrk, pdef, 1, DT, g.dx, g.dy, g.dz, obc, advwenos]
        + [pdefweno, wepsf, 0, 0, zt],
        float,
    )
    outs = dict(
        sten=g.s_shape,
        uten=g.u_shape,
        vten=g.v_shape,
        wten=g.w_shape,
        rru2=g.u_shape,
        rrv2=g.v_shape,
        rrw2=g.w_shape,
        divx2=g.s_shape,
        div=g.s_shape,
    )
    return run_driver("adv_terrain", {"params": p, **inp, **tm}, outs)


def python(bk, inp, tm, zt, nrk, pdef, obc, advwenos, pdefweno, wepsf):
    from cm1x import constants as c

    cfg = _cfg(obc, advwenos)
    a = {k: bk.asarray(np.array(v)) for k, v in inp.items()}
    m = {k: bk.asarray(np.array(v)) for k, v in tm.items()}
    com = dict(rru=a["rru"], rrv=a["rrv"], rrw=a["rrw"], divx=a["divx"], mh=a["mh"])
    com.update(c1=a["c1"], c2=a["c2"], cfg=cfg, nrk=nrk)
    out = {}
    out["sten"] = adv.advs(
        bk,
        G,
        wflag=1,
        rho0=a["rho0"],
        rr0=a["rr0"],
        uh=a["uh"],
        vh=a["vh"],
        s0=a["s0"],
        s=a["s"],
        sten=a["sten"],
        pdef=pdef,
        dt=DT,
        hadvorder=5,
        vadvorder=5,
        weps=wepsf * c.epsilon,
        pdefweno=pdefweno,
        ruh=a["ruh"],
        rvh=a["rvh"],
        rmh=a["rmh"],
        gz=m["gz"],
        rgz=m["rgz"],
        rdsf=m["rdsf"],
        **com,
    )["sten"]
    out["uten"] = adv.advu(
        bk,
        G,
        rr0=a["rr0"],
        rho0=a["rho0"],
        uf=a["uf"],
        vh=a["vh"],
        u3d=a["u3d"],
        uten=a["uten"],
        gzu=m["gzu"],
        rdsf=m["rdsf"],
        **com,
    )["uten"]
    out["vten"] = adv.advv(
        bk,
        G,
        rr0=a["rr0"],
        rho0=a["rho0"],
        uh=a["uh"],
        vf=a["vf"],
        v3d=a["v3d"],
        vten=a["vten"],
        gzv=m["gzv"],
        rdsf=m["rdsf"],
        **com,
    )["vten"]
    out["wten"] = adv.advw(
        bk,
        G,
        rrf0=a["rrf0"],
        mf=a["mf"],
        uh=a["uh"],
        vh=a["vh"],
        w3d=a["w3d"],
        wten=a["wten"],
        hadvorder=5,
        vadvorder=5,
        advweno=0,
        gz=m["gz"],
        rds=m["rds"],
        **com,
    )["wten"]
    mt = {k: m[k] for k in "rds rdsf sigma sigmaf gz rgzu rgzv dzdx dzdy".split()}
    gd = adv.getdivx(
        bk,
        G,
        uh=a["uh"],
        vh=a["vh"],
        mh=a["mh"],
        rho0=a["rho0"],
        rf0=a["rf0"],
        rru=bk.copy(a["u3d"]),
        rrv=bk.copy(a["v3d"]),
        rrw=bk.copy(a["w3d"]),
        divx=bk.copy(a["divx"]),
        cfg=cfg,
        zt=zt,
        **mt,
    )
    out.update(rru2=gd["rru"], rrv2=gd["rrv"], rrw2=gd["rrw"], divx2=gd["divx"])
    out["div"] = misc.getdiv(
        bk,
        G,
        uh=a["uh"][None, None, :],
        vh=a["vh"][None, :, None],
        mh=a["mh"],
        u=a["u3d"],
        v=a["v3d"],
        w=a["w3d"],
        terrain=dict(mt, zt=zt),
    )["div"]
    return {k: bk.to_numpy(v) for k, v in out.items()}


@requires_gfortran
@pytest.mark.parametrize(
    "nrk,pdef,obc,advwenos,pdefweno,wepsf",
    [
        (1, 0, 1, 0, 0, 0.01),  # flux advection, periodic
        (3, 1, 2, 0, 0, 0.01),  # open BCs + pdef
        (3, 0, 2, 2, 0, 10.0),  # WENO (nh_mountain_waves theta)
        (3, 1, 2, 2, 1, 0.01),  # WENO + pdef + pdefweno
        (3, 1, 1, 2, 0, 1.0),  # WENO + pdef, periodic
    ],
)
def test_adv_terrain_vs_fortran(bk, nrk, pdef, obc, advwenos, pdefweno, wepsf):
    seed = nrk + 10 * obc + 100 * pdefweno + 1000 * advwenos
    inp = make_inputs(seed=seed)
    tm, zt = make_metrics(G, seed, _cfg(obc, advwenos))
    assert np.max(np.abs(tm["dzdx"])) > 0.05
    args = (nrk, pdef, obc, advwenos, pdefweno, wepsf)
    ref = fortran(inp, tm, zt, *args)
    got = python(bk, inp, tm, zt, *args)
    base = dict(
        sten=inp["sten"],
        uten=inp["uten"],
        vten=inp["vten"],
        wten=inp["wten"],
        rru2=0 * inp["u3d"],
        rrv2=0 * inp["v3d"],
        rrw2=0 * inp["w3d"],
        divx2=0 * inp["divx"],
    )
    for name, reg in _regions().items():
        _cmp(name, got[name], ref[name], base[name], reg)
    _cmp("div", got["div"], ref["div"], 0 * inp["divx"], _regions()["divx2"])


@pytest.mark.parametrize("advwenos", [0, 2])
def test_adv_terrain_jit(advwenos):
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax")
    inp = make_inputs(seed=7)
    tm, zt = make_metrics(G, 7, _cfg(2, advwenos))
    args = (zt, 3, 1, 2, advwenos, 1, 0.01)

    def f(s):
        return python(bk, dict(inp, s=s), tm, *args)

    e = f(inp["s"])
    a = {k: bk.asarray(v) for k, v in inp.items()}
    m = {k: bk.asarray(v) for k, v in tm.items()}

    def g(s, gz, rgz, rdsf):
        return adv.advs(
            bk,
            G,
            nrk=3,
            wflag=1,
            rho0=a["rho0"],
            rr0=a["rr0"],
            mh=a["mh"],
            uh=a["uh"],
            vh=a["vh"],
            rru=a["rru"],
            rrv=a["rrv"],
            rrw=a["rrw"],
            divx=a["divx"],
            s0=a["s0"],
            s=s,
            sten=a["sten"],
            pdef=1,
            dt=DT,
            c1=a["c1"],
            c2=a["c2"],
            hadvorder=5,
            vadvorder=5,
            weps=1e-20,
            pdefweno=1,
            gz=gz,
            rgz=rgz,
            rdsf=rdsf,
            cfg=_cfg(2, advwenos),
        )["sten"]

    j = jax.jit(g)(a["s"], m["gz"], m["rgz"], m["rdsf"])
    ee = g(a["s"], m["gz"], m["rgz"], m["rdsf"])
    np.testing.assert_allclose(np.asarray(j), np.asarray(ee), rtol=1e-12, atol=1e-18)
    assert np.all(np.isfinite(e["sten"]))
