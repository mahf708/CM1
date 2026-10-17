"""cm1x.bc vs. the Fortran bc.F routines (non-MPI), every lateral BC type."""

import numpy as np
import pytest

from cm1x import bc
from cm1x.grid import Grid

from fortran import requires_gfortran, run_driver

NI, NJ, NK = 7, 6, 5
DT, DTS = 2.0, 0.5
CASES = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 4, 3, 4), (2, 1, 4, 2), (1, 3, 2, 1)]


@pytest.fixture(scope="module")
def grid():
    return Grid(ni=NI, nj=NJ, nk=NK, dx=1000.0, dy=1200.0, dz=500.0)


def make_inputs(g, seed):
    rng = np.random.default_rng(seed)
    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    d = {
        "s": rng.standard_normal(S),
        "p": rng.standard_normal(S),
        "u": rng.standard_normal(U),
        "v": rng.standard_normal(V),
        "w0": rng.standard_normal(W),
        "w1": rng.standard_normal(W),
        "s2d": rng.standard_normal(S[1:]),
        "ua": 10 * rng.standard_normal(U),
        "va": 10 * rng.standard_normal(V),
        "uten": 1e-2 * rng.standard_normal(U),
        "vten": 1e-2 * rng.standard_normal(V),
        "mh": 1.0 + 0.05 * rng.standard_normal(S),
        "ruf": 1.0 + 0.05 * rng.standard_normal(U[2]),
        "rvf": 1.0 + 0.05 * rng.standard_normal(V[1]),
        "uh": 1.0 + 0.05 * rng.standard_normal(S[2]),
        "vh": 1.0 + 0.05 * rng.standard_normal(S[1]),
        "radbcw": rng.standard_normal((S[0], S[1])),
        "radbce": rng.standard_normal((S[0], S[1])),
        "radbcs": rng.standard_normal((S[0], S[2])),
        "radbcn": rng.standard_normal((S[0], S[2])),
    }
    # new velocities: small change, so phase speeds span the clipping range
    d["u3d"] = d["ua"] + rng.standard_normal(U)
    d["v3d"] = d["va"] + rng.standard_normal(V)
    return d


def fortran_bc(g, d, case):
    params = np.array([g.ni, g.nj, g.nk, *case, g.dx, g.dy, g.dz, DT, DTS], dtype=float)
    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    outs = dict(
        s=S,
        p=S,
        u=U,
        v=V,
        w0=W,
        w1=W,
        s2d=S[1:],
        uo=U,
        vo=V,
        cfl=(1,),
        radbcw=(S[0], S[1]),
        radbce=(S[0], S[1]),
        radbcs=(S[0], S[2]),
        radbcn=(S[0], S[2]),
    )
    return run_driver("bc", {"params": params, **d}, outs)


def python_bc(bk, g, d, case):
    cfg = dict(zip(("wbc", "ebc", "sbc", "nbc"), case))
    a = {k: bk.asarray(v) for k, v in d.items()}
    r = {
        "s": bc.bcs(bk, g, bk.copy(a["s"]), cfg),
        "p": bc.bcp(bk, g, bk.copy(a["p"]), cfg),
        "u": bc.bcu(bk, g, bk.copy(a["u"]), cfg),
        "v": bc.bcv(bk, g, bk.copy(a["v"]), cfg),
        "w0": bc.bcw(bk, g, bk.copy(a["w0"]), 0, cfg),
        "w1": bc.bcw(bk, g, bk.copy(a["w1"]), 1, cfg),
        "s2d": bc.bc2d(bk, g, bk.copy(a["s2d"]), cfg),
    }
    r.update(
        bc.radbcew4(
            bk,
            g,
            cfg,
            ruf=a["ruf"],
            radbcw=a["radbcw"],
            radbce=a["radbce"],
            u1=a["ua"],
            u2=a["u3d"],
            dt=DT,
        )
    )
    r.update(
        bc.radbcns4(
            bk,
            g,
            cfg,
            rvf=a["rvf"],
            radbcs=a["radbcs"],
            radbcn=a["radbcn"],
            v1=a["va"],
            v2=a["v3d"],
            dt=DT,
        )
    )
    uo, vo = bk.copy(a["u3d"]), bk.copy(a["v3d"])
    kw = dict(dts=DTS)
    if case[0] == 2:
        uo = bc.ssopenbcw(
            bk, g, cfg, uh=a["uh"], radbcw=r["radbcw"], u3d=uo, uten=a["uten"], **kw
        )
    if case[1] == 2:
        uo = bc.ssopenbce(
            bk, g, cfg, uh=a["uh"], radbce=r["radbce"], u3d=uo, uten=a["uten"], **kw
        )
    if case[2] == 2:
        vo = bc.ssopenbcs(
            bk, g, cfg, vh=a["vh"], radbcs=r["radbcs"], v3d=vo, vten=a["vten"], **kw
        )
    if case[3] == 2:
        vo = bc.ssopenbcn(
            bk, g, cfg, vh=a["vh"], radbcn=r["radbcn"], v3d=vo, vten=a["vten"], **kw
        )
    r["uo"], r["vo"] = uo, vo
    # calccflquick on (ua, va, w1 after bcw)
    r["cfl"] = bk.xp.reshape(
        bc.calccflquick(
            bk,
            g,
            cfg,
            dt=DT,
            uh=a["uh"],
            vh=a["vh"],
            mh=a["mh"],
            ua=a["ua"],
            va=a["va"],
            wa=r["w1"],
        ),
        (1,),
    )
    return r


@requires_gfortran
@pytest.mark.parametrize("case", CASES, ids=lambda c: "".join(map(str, c)))
def test_bc_vs_fortran(bk, grid, case):
    d = make_inputs(grid, seed=sum(c * 10**i for i, c in enumerate(case)))
    ref = fortran_bc(grid, d, case)
    res = python_bc(bk, grid, d, case)
    assert set(res) == set(ref)
    for k, r in ref.items():
        a = bk.to_numpy(res[k])
        scale = max(np.max(np.abs(r)), 1e-300)
        np.testing.assert_allclose(a, r, rtol=1e-12, atol=1e-12 * scale, err_msg=k)


def test_dtu0_dtv0(bk, grid):
    dtu0, dtv0 = (
        bk.to_numpy(a) for a in bc.dtu0_dtv0(bk, grid, dict(wbc=2, ebc=2, sbc=1, nbc=2))
    )
    assert dtu0[3 : NJ + 3, 3].sum() == 0 and dtu0[3 : NJ + 3, NI + 3].sum() == 0
    assert dtu0[3 : NJ + 3, 4 : NI + 3].min() == 1 and dtu0.sum() == NJ * (NI - 1)
    assert dtv0[3, 3 : NI + 3].min() == 1 and dtv0[NJ + 3, 3 : NI + 3].sum() == 0
    assert dtv0.sum() == NJ * NI


def test_ib_flags():
    assert bc.ib_flags(None) == (0, 0, 0, 0)
    assert bc.ib_flags(dict(wbc=2, ebc=1, sbc=3, nbc=4)) == (1, 0, 1, 1)
