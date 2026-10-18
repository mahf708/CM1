"""Terrain-following grid vs CM1's Fortran: init_terrain (itern = 1, 2, 3; with
zsgrad and the Gal-Chen metrics), bcwsfc and the terrain branches of
ssopenbcw/e/s/n (drivers: fortran_harness/drv_terrain.F90)."""

import numpy as np
import pytest

from cm1x import bc
from cm1x.grid import Grid
from cm1x.init_terrain import init_terrain

from fortran import requires_gfortran, run_driver

G = Grid(ni=9, nj=8, nk=12, dx=400.0, dy=500.0, dz=300.0)
DTS = 0.5
CG = dict(cgs1=1.8, cgs2=-1.0, cgs3=0.2, cgt1=1.7, cgt2=-0.9, cgt3=0.2)
METRICS = "zs zh zf gz rgz gzu rgzu gzv rgzv dzdx dzdy gx gxu gy gyv rds rdsf".split()


def make_inputs(g, seed):
    rng = np.random.default_rng(seed)
    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    ng, nk = g.ngxy, g.nk
    # stretched nominal levels sigmaf(kb..ke+1), sigma at midpoints
    dzs = g.dz * (1.0 + 0.3 * rng.uniform(size=nk + 2))
    sigmaf = np.concatenate([[-dzs[0]], np.cumsum(dzs) - dzs[0]])
    sigma = 0.5 * (sigmaf[:-1] + sigmaf[1:])
    fi_x = np.arange(S[2]) - ng + 1  # Fortran i
    fi_y = np.arange(S[1]) - ng + 1
    xf = (np.arange(U[2]) - ng) * g.dx
    yf = (np.arange(V[1]) - ng) * g.dy
    d = {
        "xh": (fi_x - 0.5) * g.dx,
        "xf": xf,
        "yh": (fi_y - 0.5) * g.dy,
        "yf": yf,
        "uh": 1.0 + 0.05 * rng.standard_normal(S[2]),
        "uf": 1.0 + 0.05 * rng.standard_normal(U[2]),
        "vh": 1.0 + 0.05 * rng.standard_normal(S[1]),
        "vf": 1.0 + 0.05 * rng.standard_normal(V[1]),
        "sigma": sigma,
        "sigmaf": sigmaf,
        "rds": np.ones(nk + 2),
        "rdsf": np.ones(nk + 3),
        "u": 10 * rng.standard_normal(U),
        "v": 10 * rng.standard_normal(V),
        "w": rng.standard_normal(W),
        "u3d": 10 * rng.standard_normal(U),
        "v3d": 10 * rng.standard_normal(V),
        "uten": 1e-2 * rng.standard_normal(U),
        "vten": 1e-2 * rng.standard_normal(V),
        "radbcw": rng.uniform(0, 30, (S[0], S[1])),
        "radbce": rng.uniform(0, 30, (S[0], S[1])),
        "radbcs": rng.uniform(0, 30, (S[0], S[2])),
        "radbcn": rng.uniform(0, 30, (S[0], S[2])),
    }
    return d


def _scalars(g, d):
    # itern=1/2 hills are only ~400 m wide: put the model top low enough that
    # the metric terms are O(1)
    return dict(maxx=g.ni * g.dx, maxy=g.nj * g.dy, maxz=float(d["sigmaf"][g.nk + 1]))


def fortran(g, d, itern, obc, hord=(5, 5)):
    sc = _scalars(g, d)
    p = np.array(
        [
            g.ni,
            g.nj,
            g.nk,
            g.dx,
            g.dy,
            itern,
            obc,
            obc,
            sc["maxx"],
            sc["maxy"],
            sc["maxz"],
            DTS,
            *CG.values(),
            *hord,
        ],
        float,
    )
    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    s2 = S[1:]
    outs = dict(
        zs=s2,
        zh=S,
        zf=W,
        gz=s2,
        rgz=s2,
        gzu=s2,
        rgzu=s2,
        gzv=s2,
        rgzv=s2,
        dzdx=s2,
        dzdy=s2,
        gx=S,
        gxu=S,
        gy=S,
        gyv=S,
        rds=(g.nk + 2,),
        rdsf=(g.nk + 3,),
        w=W,
        uo=U,
        vo=V,
    )
    return run_driver("terrain", {"params": p, **d}, outs)


def python(bk, g, d, itern, obc, hord=(5, 5)):
    cfg = dict(wbc=obc, ebc=obc, sbc=obc, nbc=obc, itern=itern, terrain_flag=True)
    cfg.update(hadvordrs=hord[0], hadvordrv=hord[1])
    a = {k: bk.asarray(np.array(v)) for k, v in d.items()}
    t = init_terrain(
        bk,
        g,
        cfg,
        **{k: a[k] for k in "xh uh xf uf yh vh yf vf rds sigma rdsf sigmaf".split()},
        **_scalars(g, d),
    )
    out = {k: t[k] for k in METRICS}
    out["w"] = bc.bcwsfc(
        bk, g, gz=t["gz"], dzdx=t["dzdx"], dzdy=t["dzdy"], u=a["u"], v=a["v"], w=a["w"]
    )
    tm = dict(
        rds=t["rds"],
        sigma=a["sigma"],
        rdsf=t["rdsf"],
        sigmaf=a["sigmaf"],
        gz=t["gz"],
        rgzu=t["rgzu"],
        rgzv=t["rgzv"],
        gx=t["gx"],
        gy=t["gy"],
        **CG,
    )
    uo, vo = a["u3d"], a["v3d"]
    if obc == 2:
        kw = dict(uh=a["uh"], u3d=uo, uten=a["uten"], dts=DTS, terrain=tm)
        uo = bc.ssopenbcw(bk, g, cfg, radbcw=a["radbcw"], **kw)
        kw["u3d"] = uo
        uo = bc.ssopenbce(bk, g, cfg, radbce=a["radbce"], **kw)
        kw = dict(vh=a["vh"], v3d=vo, vten=a["vten"], dts=DTS, terrain=tm)
        vo = bc.ssopenbcs(bk, g, cfg, radbcs=a["radbcs"], **kw)
        kw["v3d"] = vo
        vo = bc.ssopenbcn(bk, g, cfg, radbcn=a["radbcn"], **kw)
    out["uo"], out["vo"] = uo, vo
    return {k: bk.to_numpy(v) for k, v in out.items()}


@requires_gfortran
@pytest.mark.parametrize(
    "itern,obc,hord",
    [(1, 1, (5, 5)), (2, 2, (5, 5)), (3, 2, (5, 6)), (3, 1, (3, 4)), (1, 2, (2, 2))],
)
def test_terrain_vs_fortran(bk, itern, obc, hord):
    d = make_inputs(G, seed=itern + 10 * obc)
    ref = fortran(G, d, itern, obc, hord)
    got = python(bk, G, d, itern, obc, hord)
    assert np.max(np.abs(ref["zs"])) > 10.0
    assert np.max(np.abs(ref["gx"])) > 1e-3
    for k in METRICS + ["w"]:
        sc = np.max(np.abs(ref[k])) + 1e-300
        err = np.max(np.abs(got[k] - ref[k])) / sc
        assert err < 1e-12, f"{k}: rel err {err:.3e}"
    for k, base in (("uo", d["u3d"]), ("vo", d["v3d"])):
        sc = np.max(np.abs(ref[k] - base)) + 1e-300
        if obc == 2:
            assert sc > 1e-3
        err = np.max(np.abs(got[k] - ref[k])) / sc
        assert err < 1e-12, f"{k}: rel err {err:.3e}"


def test_itern4_raises(bk):
    d = make_inputs(G, 0)
    a = {k: bk.asarray(v) for k, v in d.items()}
    with pytest.raises(NotImplementedError):
        init_terrain(
            bk,
            G,
            dict(itern=4, terrain_flag=True),
            **{
                k: a[k] for k in "xh uh xf uf yh vh yf vf rds sigma rdsf sigmaf".split()
            },
            **_scalars(G, d),
        )
