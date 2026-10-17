"""advs/advu/advv/advw (adv.F, with pdef.F) and getdivx (misclibs.F) vs CM1's Fortran."""

import numpy as np
import pytest

from cm1x import adv
from cm1x.bc import periodic_xy
from cm1x.grid import Grid, fi, fk

from fortran import requires_gfortran, run_driver

G = Grid(ni=9, nj=8, nk=12, dx=1000.0, dy=900.0, dz=500.0)
DT = 6.0


class _NP:  # numpy-only helper backend for periodic_xy in data generation
    name = "numpy"

    def __init__(self):
        import array_api_compat.numpy as xp

        self.xp = xp


def _per(a):
    return periodic_xy(_NP(), np.array(a), G)


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    g = G
    ks, ku, kv, kw = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    nk = g.nk
    zh = np.array(g.zh())
    zf = np.array(g.zf())
    prof = 1.15 * np.exp(-zh / 9000.0)
    proff = 1.15 * np.exp(-zf / 9000.0)

    def col3(p, shape):
        return np.broadcast_to(p[:, None, None], shape).copy()

    rho0 = col3(prof, ks)
    rr0 = 1.0 / rho0
    rf0 = col3(proff[: ks[0]], ks)
    rrf0 = 1.0 / rf0
    mh = col3(1.0 + 0.05 * rng.standard_normal(ks[0]), ks)
    mf = col3(1.0 + 0.05 * rng.standard_normal(kw[0]), kw)
    c2 = _per(0.5 + 0.05 * rng.standard_normal(ks))
    c1 = 1.0 - c2
    uh = 1.0 + 0.02 * rng.standard_normal(ks[2])
    uf = 1.0 + 0.02 * rng.standard_normal(ku[2])
    vh = 1.0 + 0.02 * rng.standard_normal(ks[1])
    vf = 1.0 + 0.02 * rng.standard_normal(kv[1])

    u3d = _per(10.0 * rng.standard_normal(ku))
    v3d = _per(10.0 * rng.standard_normal(kv))
    w3d = _per(3.0 * rng.standard_normal(kw))
    w3d[fk(1)] = 0.0
    w3d[fk(nk + 1)] = 0.0
    rru = rho0[:, :, :1] * u3d
    rrv = rho0[:, :1, :] * v3d
    rrw = np.concatenate([rf0, rf0[-1:]], axis=0)[:, :1, :1] * w3d
    divx = np.zeros(ks)
    divx[:, :, :] = (
        (rru[:, :, 1:] - rru[:, :, :-1]) * g.rdx * uh
        + (rrv[:, 1:, :] - rrv[:, :-1, :]) * g.rdy * vh[:, None]
    ) + (rrw[1:] - rrw[:-1]) * g.rdz * mh[:, :1, :1]

    # moisture-like scalar with zeros so that the pdef correction is exercised
    s = np.maximum(0.0, 1e-3 * rng.standard_normal(ks))
    s = _per(s)
    s0 = _per(np.maximum(0.0, s + 2e-4 * rng.standard_normal(ks)))
    sten = 1e-6 * rng.standard_normal(ks)
    uten = rng.standard_normal(ku)
    vten = rng.standard_normal(kv)
    wten = rng.standard_normal(kw)
    return dict(
        uh=uh,
        uf=uf,
        vh=vh,
        vf=vf,
        mh=mh,
        mf=mf,
        rho0=rho0,
        rr0=rr0,
        rf0=rf0,
        rrf0=rrf0,
        c1=c1,
        c2=c2,
        rru=rru,
        rrv=rrv,
        rrw=rrw,
        divx=divx,
        s0=s0,
        s=s,
        sten=sten,
        u3d=u3d,
        uten=uten,
        v3d=v3d,
        vten=vten,
        w3d=w3d,
        wten=wten,
    )


def fortran(inp, nrk, pdef, wflag=1):
    g = G
    p = np.array(
        [g.ni, g.nj, g.nk, nrk, pdef, wflag, DT, g.dx, g.dy, g.dz, 0, 0], float
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
    )
    return run_driver("adv", {"params": p, **inp}, outs)


def python(bk, inp, nrk, pdef, wflag=1):
    a = {k: bk.asarray(v) for k, v in inp.items()}
    out = {}
    out["sten"] = adv.advs(
        bk,
        G,
        nrk=nrk,
        wflag=wflag,
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
        s=a["s"],
        sten=a["sten"],
        pdef=pdef,
        dt=DT,
        c1=a["c1"],
        c2=a["c2"],
        hadvorder=5,
        vadvorder=5,
    )["sten"]
    out["uten"] = adv.advu(
        bk,
        G,
        nrk=nrk,
        rr0=a["rr0"],
        mh=a["mh"],
        uf=a["uf"],
        vh=a["vh"],
        rru=a["rru"],
        rrv=a["rrv"],
        rrw=a["rrw"],
        u3d=a["u3d"],
        uten=a["uten"],
        divx=a["divx"],
        c1=a["c1"],
        c2=a["c2"],
    )["uten"]
    out["vten"] = adv.advv(
        bk,
        G,
        nrk=nrk,
        rr0=a["rr0"],
        mh=a["mh"],
        uh=a["uh"],
        vf=a["vf"],
        rru=a["rru"],
        rrv=a["rrv"],
        rrw=a["rrw"],
        v3d=a["v3d"],
        vten=a["vten"],
        divx=a["divx"],
        c1=a["c1"],
        c2=a["c2"],
    )["vten"]
    out["wten"] = adv.advw(
        bk,
        G,
        nrk=nrk,
        rrf0=a["rrf0"],
        mf=a["mf"],
        uh=a["uh"],
        vh=a["vh"],
        rru=a["rru"],
        rrv=a["rrv"],
        rrw=a["rrw"],
        w3d=a["w3d"],
        wten=a["wten"],
        divx=a["divx"],
        c1=a["c1"],
        c2=a["c2"],
        hadvorder=5,
        vadvorder=5,
        advweno=0,
    )["wten"]
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
    )
    out.update(rru2=gd["rru"], rrv2=gd["rrv"], rrw2=gd["rrw"], divx2=gd["divx"])
    return {k: bk.to_numpy(v) for k, v in out.items()}


def _cmp(name, got, ref, base, region):
    """Max abs error over the whole array (halos included), relative to the
    magnitude of the change the routine makes in its update region."""
    scale = np.max(np.abs(ref[region] - base[region])) + 1e-300
    err = np.max(np.abs(got - ref)) / scale
    assert err < 1e-12, f"{name}: rel err {err:.3e}"


def _regions():
    g = G
    K, J, I = (
        slice(fk(1), fk(g.nk) + 1),
        slice(fi(1), fi(g.nj) + 1),
        slice(fi(1), fi(g.ni) + 1),
    )
    JF, IF = slice(fi(1), fi(g.nj + 1) + 1), slice(fi(1), fi(g.ni + 1) + 1)
    KW = slice(fk(1), fk(g.nk + 1) + 1)
    return dict(
        sten=(K, J, I),
        uten=(K, J, IF),
        vten=(K, JF, I),
        wten=(KW, J, I),
        rru2=(K, J, IF),
        rrv2=(K, JF, I),
        rrw2=(KW, J, I),
        divx2=(K, J, I),
    )


@requires_gfortran
@pytest.mark.parametrize("nrk,pdef", [(1, 0), (3, 1)])
def test_adv_vs_fortran(bk, nrk, pdef):
    inp = make_inputs(seed=nrk)
    ref = fortran(inp, nrk, pdef)
    got = python(bk, inp, nrk, pdef)
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


@requires_gfortran
def test_pdef_is_active():
    """The pdef=1 path must actually change the scalar tendency for these inputs."""
    from cm1x import get_backend

    bk = get_backend("numpy")
    inp = make_inputs(seed=3)
    a = python(bk, inp, 3, 1)["sten"]
    b = python(bk, inp, 3, 0)["sten"]
    assert np.max(np.abs(a - b)) > 0.0


def test_jit():
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax")
    inp = make_inputs(seed=5)
    a = {k: bk.asarray(v) for k, v in inp.items()}

    def f(s, s0, sten, u3d, uten):
        r1 = adv.advs(
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
            s0=s0,
            s=s,
            sten=sten,
            pdef=1,
            dt=DT,
            c1=a["c1"],
            c2=a["c2"],
            hadvorder=5,
            vadvorder=5,
        )["sten"]
        r2 = adv.advu(
            bk,
            G,
            nrk=3,
            rr0=a["rr0"],
            mh=a["mh"],
            uf=a["uf"],
            vh=a["vh"],
            rru=a["rru"],
            rrv=a["rrv"],
            rrw=a["rrw"],
            u3d=u3d,
            uten=uten,
            divx=a["divx"],
            c1=a["c1"],
            c2=a["c2"],
        )["uten"]
        return r1, r2

    args = (a["s"], a["s0"], a["sten"], a["u3d"], a["uten"])
    j1, j2 = jax.jit(f)(*args)
    e1, e2 = f(*args)
    np.testing.assert_allclose(np.asarray(j1), np.asarray(e1), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(j2), np.asarray(e2), rtol=1e-12, atol=1e-12)


def test_unsupported_options(bk):
    inp = make_inputs(seed=0)
    a = {k: bk.asarray(v) for k, v in inp.items()}
    with pytest.raises(NotImplementedError):
        adv.advu(
            bk,
            G,
            nrk=1,
            rr0=a["rr0"],
            mh=a["mh"],
            uf=a["uf"],
            vh=a["vh"],
            rru=a["rru"],
            rrv=a["rrv"],
            rrw=a["rrw"],
            u3d=a["u3d"],
            uten=a["uten"],
            divx=a["divx"],
            c1=a["c1"],
            c2=a["c2"],
            cfg={"advwenov": 1},
        )
    with pytest.raises(NotImplementedError):
        adv.advw(
            bk,
            G,
            nrk=1,
            rrf0=a["rrf0"],
            mf=a["mf"],
            uh=a["uh"],
            vh=a["vh"],
            rru=a["rru"],
            rrv=a["rrv"],
            rrw=a["rrw"],
            w3d=a["w3d"],
            wten=a["wten"],
            divx=a["divx"],
            c1=a["c1"],
            c2=a["c2"],
            hadvorder=3,
            vadvorder=5,
            advweno=0,
        )
