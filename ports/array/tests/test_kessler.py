"""Kessler microphysics (kessler.F), pdefq (misclibs.F) and the ptype=1 branch of
mp_driver.F against the original Fortran (double-precision CM1 objects)."""

import numpy as np
import pytest

from cm1x import constants as C
from cm1x.grid import Grid
from cm1x.kessler import fallout, geterain, getefall, k_fallout, kessler, satadj
from cm1x.mp_driver import mp_driver
from cm1x.pdef import pdefq

from fortran import requires_gfortran, run_driver

pytestmark = requires_gfortran

G = Grid(ni=7, nj=5, nk=14, dx=1000.0, dy=1000.0, dz=500.0)
DT = 6.0
TSMALL = 1.0e-4
QSMALL = 1.0e-14
RTOL = 1e-12


def _qvs(t, p):
    esl = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = np.minimum(esl, 0.5 * p)
    return C.eps * esl / (p - esl)


def make_state(seed=0, g=G):
    """Warm, moist, partly saturated columns with cloud and rain."""
    rng = np.random.default_rng(seed)
    sh = g.s_shape
    z = np.array(g.zh())[:, None, None] * np.ones(sh)
    th0 = 300.0 + 4.0e-3 * z
    pi0 = 1.0 - C.g * z / (C.cp * 302.0)
    th3d = rng.uniform(-1.0, 1.0, sh)
    pp3d = rng.uniform(-2e-4, 2e-4, sh)
    prs = C.p00 * (pi0 + pp3d) ** C.cpdrd
    t = (th0 + th3d) * (pi0 + pp3d)
    qvs = _qvs(t, prs)
    rh = rng.uniform(0.6, 1.08, sh)
    qv = rh * qvs
    qc = np.where(rng.random(sh) < 0.5, rng.uniform(0.0, 2.5e-3, sh), 0.0)
    qc = np.where(rng.random(sh) < 0.05, 1e-16, qc)
    qr = np.where(rng.random(sh) < 0.6, rng.uniform(0.0, 4e-3, sh), 0.0)
    qr = np.where(rng.random(sh) < 0.1, 1e-9, qr)  # below the 1e-8 fall-edge test
    rho = prs / (C.rd * t * (1.0 + qv * C.reps))
    q3d = np.stack([qv, qc, qr])
    ruh = 1.0 + 0.05 * rng.standard_normal(sh[2])
    rvh = 1.0 + 0.05 * rng.standard_normal(sh[1])
    rmh = 1.0 + 0.05 * rng.standard_normal(sh)
    mh = np.ones(sh)
    mf = np.ones(g.w_shape)
    return dict(
        th0=th0,
        pi0=pi0,
        th3d=th3d,
        pp3d=pp3d,
        prs=prs,
        rho=rho,
        q3d=q3d,
        ruh=ruh,
        rvh=rvh,
        rmh=rmh,
        mh=mh,
        mf=mf,
        zh=z,
        t=t,
        rr=rho * (1.0 + 0.01 * rng.standard_normal(sh)),
    )


def kparams(mode, eqtset=2, nrain=1, nrk=4, setzero=1, cpx=C.cpl, lx1=C.lv1, g=G):
    return np.array(
        [
            mode,
            g.ni,
            g.nj,
            g.nk,
            g.dx,
            g.dy,
            g.dz,
            DT,
            eqtset,
            TSMALL,
            nrain,
            nrk,
            setzero,
            cpx,
            lx1,
        ],
        dtype=float,
    )


def close(bk, got, ref, rtol=RTOL, scale=True, name=""):
    got = got if isinstance(got, np.ndarray) else bk.to_numpy(got)
    ref = np.asarray(ref)
    atol = rtol * np.max(np.abs(ref)) if scale else 0.0
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=name)


def A(bk, x):
    return bk.asarray(np.asarray(x))


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eqtset", [2, 1])
def test_kessler(bk, eqtset):
    s = make_state(1)
    tmp0 = np.zeros(G.s_shape)
    ins = dict(
        params=kparams(1, eqtset),
        ruh=s["ruh"],
        rvh=s["rvh"],
        rmh=s["rmh"],
        pi0=s["pi0"],
        th0=s["th0"],
        tmp=tmp0,
        rho=s["rho"],
        rr=s["rr"],
        pp3d=s["pp3d"],
        th3d=s["th3d"],
        prs=s["prs"],
        q3d=s["q3d"],
        bud=np.array([1.5]),
    )
    shp = G.s_shape
    ref = run_driver(
        "kessler",
        ins,
        dict(
            tmp=shp,
            rho=shp,
            pp3d=shp,
            th3d=shp,
            prs=shp,
            q3d=(3,) + shp,
            tauto=(1,),
            taccr=(1,),
            tevar=(1,),
        ),
    )
    q = A(bk, s["q3d"])
    out = kessler(
        bk,
        G,
        {"eqtset": eqtset},
        dt=DT,
        tauto=1.5,
        taccr=3.0,
        tevar=4.5,
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        rmh=A(bk, s["rmh"]),
        pi0=A(bk, s["pi0"]),
        th0=A(bk, s["th0"]),
        tmp=A(bk, tmp0),
        rho=A(bk, s["rho"]),
        rr=A(bk, s["rr"]),
        pp3d=A(bk, s["pp3d"]),
        th3d=A(bk, s["th3d"]),
        prs=A(bk, s["prs"]),
        qv3d=q[0],
        qc3d=q[1],
        qr3d=q[2],
    )
    for k in ("tmp", "rho", "pp3d", "th3d", "prs"):
        close(bk, out[k], ref[k], name=k)
    for n, k in enumerate(("qv3d", "qc3d", "qr3d")):
        close(bk, out[k], ref["q3d"][n], name=k)
    # budgets: double accumulations, only summation order differs
    for k in ("tauto", "taccr", "tevar"):
        close(bk, out[k], ref[k][0], rtol=1e-11, scale=False, name=k)
    # inputs were physical enough to exercise every process
    assert abs(ref["tevar"][0] - 4.5) > 0 and abs(ref["tauto"][0] - 1.5) > 0


@pytest.mark.parametrize("eqtset,nrk", [(2, 4), (2, 3), (1, 4)])
def test_satadj(bk, eqtset, nrk):
    s = make_state(2)
    ins = dict(
        params=kparams(2, eqtset, nrk=nrk),
        ruh=s["ruh"],
        rvh=s["rvh"],
        rmh=s["rmh"],
        pi0=s["pi0"],
        th0=s["th0"],
        rho=s["rho"],
        rr=s["rr"],
        pp3d=s["pp3d"],
        prs=s["prs"],
        th3d=s["th3d"],
        q3d=s["q3d"],
        bud=np.array([0.5]),
    )
    shp = G.s_shape
    ref = run_driver(
        "kessler",
        ins,
        dict(
            rho=shp, pp3d=shp, th3d=shp, prs=shp, q3d=(3,) + shp, tcond=(1,), tevac=(1,)
        ),
    )
    out = satadj(
        bk,
        G,
        {"eqtset": eqtset, "tsmall": TSMALL},
        nrk=nrk,
        dt=DT,
        tcond=0.5,
        tevac=1.0,
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        rmh=A(bk, s["rmh"]),
        pi0=A(bk, s["pi0"]),
        th0=A(bk, s["th0"]),
        rho=A(bk, s["rho"]),
        rr=A(bk, s["rr"]),
        pp3d=A(bk, s["pp3d"]),
        prs=A(bk, s["prs"]),
        th3d=A(bk, s["th3d"]),
        q3d=A(bk, s["q3d"]),
    )
    for k in ("rho", "pp3d", "th3d", "prs"):
        close(bk, out[k], ref[k], name=k)
    close(bk, out["q3d"][0], ref["q3d"][0], name="qv")
    close(bk, out["q3d"][1], ref["q3d"][1], name="qc")
    close(bk, out["q3d"][2], ref["q3d"][2], name="qr")
    for k in ("tcond", "tevac"):
        close(bk, out[k], ref[k][0], rtol=1e-11, scale=False, name=k)
    if eqtset == 2:
        nmax = int(bk.to_numpy(out["nmax"]))
        assert 1 < nmax < 20  # iterated, and converged well inside maxiter


def test_k_fallout(bk):
    s = make_state(3)
    vr0 = np.random.default_rng(5).random(G.s_shape)
    ref = run_driver(
        "kessler",
        dict(params=kparams(3), rho=s["rho"], qr3d=s["q3d"][2], vr=vr0),
        dict(vr=G.s_shape),
    )["vr"]
    out = k_fallout(bk, G, rho=A(bk, s["rho"]), qr3d=A(bk, s["q3d"][2]), vr=A(bk, vr0))[
        "vr"
    ]
    close(bk, out, ref)


@pytest.mark.parametrize("big", [False, True])
def test_fallout(bk, big):
    """big=True: a long dt-like CFL so columns sub-step (nfall up to ~4)."""
    s = make_state(4)
    rng = np.random.default_rng(6)
    qr = s["q3d"][2]
    vq = k_fallout(bk, G, rho=A(bk, s["rho"]), qr3d=A(bk, qr))["vr"]
    vq = bk.to_numpy(vq)
    if big:
        vq = vq * rng.uniform(1.0, 40.0, (1,) + G.s_shape[1:])
    nrain = 2
    rain = rng.random((nrain,) + G.s_shape[1:])
    prate = rng.random(G.s_shape[1:])
    mh = 1.0 + 0.1 * rng.random(G.s_shape)
    ins = dict(
        params=kparams(4, nrain=nrain),
        ruh=s["ruh"],
        rvh=s["rvh"],
        zh=s["zh"],
        mh=mh,
        mf=s["mf"],
        rain=rain,
        prate=prate,
        rr=s["rr"],
        rho=s["rho"],
        qr3d=qr,
        vr=vq,
        bud=np.array([2.0]),
    )
    ref = run_driver(
        "kessler",
        ins,
        dict(
            rain=rain.shape, prate=prate.shape, qr3d=G.s_shape, vr=G.s_shape, train=(1,)
        ),
    )
    out = fallout(
        bk,
        G,
        dt=DT,
        train=2.0,
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        zh=A(bk, s["zh"]),
        mh=A(bk, mh),
        mf=A(bk, s["mf"]),
        rain=A(bk, rain),
        prate=A(bk, prate),
        rr=A(bk, s["rr"]),
        rho=A(bk, s["rho"]),
        q3d=A(bk, qr),
        vq=A(bk, vq),
    )
    close(bk, out["rain"], ref["rain"], name="rain")
    close(bk, out["prate"], ref["prate"], name="prate")
    close(bk, out["q3d"], ref["qr3d"], name="qr")
    close(bk, out["vq"], ref["vr"], name="vq")
    close(bk, out["train"], ref["train"][0], rtol=1e-11, scale=False, name="train")
    if big:
        crmax = (vq[G.K, G.J, G.I] * DT / G.dz * mh[G.K, G.J, G.I]).max()
        assert crmax > 2.0


def test_geterain(bk):
    s = make_state(5)
    vr = bk.to_numpy(
        k_fallout(bk, G, rho=A(bk, s["rho"]), qr3d=A(bk, s["q3d"][2]))["vr"]
    )
    ins = dict(
        params=kparams(5),
        ruh=s["ruh"],
        rvh=s["rvh"],
        t=s["t"],
        rr=s["rr"],
        qr3d=s["q3d"][2],
        vr=vr,
        bud=np.array([10.0]),
    )
    ref = run_driver("kessler", ins, dict(erain=(1,)))["erain"][0]
    out = geterain(
        bk,
        G,
        dt=DT,
        cpx=C.cpl,
        lx1=C.lv1,
        erain=10.0,
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        t=A(bk, s["t"]),
        rr=A(bk, s["rr"]),
        q3d=A(bk, s["q3d"][2]),
        vr=A(bk, vr),
    )["erain"]
    close(bk, out, ref, rtol=1e-11, scale=False)


@pytest.mark.parametrize("setzero", [0, 1])
def test_getefall(bk, setzero):
    s = make_state(6)
    rng = np.random.default_rng(7)
    vr = bk.to_numpy(
        k_fallout(bk, G, rho=A(bk, s["rho"]), qr3d=A(bk, s["q3d"][2]))["vr"]
    )
    cvm = C.cv + rng.random(G.s_shape) * 10
    tten = rng.standard_normal(G.s_shape)
    mf = 1.0 + 0.1 * rng.random(G.w_shape)
    ins = dict(
        params=kparams(6, setzero=setzero),
        mf=mf,
        t=s["t"],
        cvm=cvm,
        tten=tten,
        qr3d=s["q3d"][2],
        vr=vr,
    )
    ref = run_driver("kessler", ins, dict(tten=G.s_shape))["tten"]
    out = getefall(
        bk,
        G,
        setzero=setzero,
        cpx=C.cpl,
        mf=A(bk, mf),
        t=A(bk, s["t"]),
        cvm=A(bk, cvm),
        tten=A(bk, tten),
        q3d=A(bk, s["q3d"][2]),
        vr=A(bk, vr),
    )
    close(bk, out["tten"], ref)


@pytest.mark.parametrize("pdscheme,rmax", [(1, 0.0), (1, QSMALL), (2, QSMALL)])
def test_pdefq(bk, pdscheme, rmax):
    s = make_state(8)
    rng = np.random.default_rng(9)
    q = s["q3d"][1] + np.where(
        rng.random(G.s_shape) < 0.3, -rng.uniform(0, 1e-4, G.s_shape), 0
    )
    q = np.where(rng.random(G.s_shape) < 0.1, 1e-15, q)
    ins = dict(
        params=np.array([G.ni, G.nj, G.nk, G.dx, G.dy, G.dz, pdscheme, rmax]),
        ruh=s["ruh"],
        rvh=s["rvh"],
        rmh=s["rmh"],
        rho=s["rho"],
        q3d=q,
        asq=np.array([3.0]),
    )
    ref = run_driver("pdefq", ins, dict(q3d=G.s_shape, asq=(1,)))
    out = pdefq(
        bk,
        G,
        {"pdscheme": pdscheme},
        rmax=rmax,
        asq=3.0,
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        rmh=A(bk, s["rmh"]),
        rho=A(bk, s["rho"]),
        q3d=A(bk, q),
    )
    close(bk, out["q3d"], ref["q3d"])
    # asq = 3 + sum(a2 - a1): a difference of column masses; summation order only
    w = s["ruh"][None, None, G.I] * s["rvh"][None, G.J, None] * s["rmh"][G.K, G.J, G.I]
    mass = (
        np.sum(np.abs(s["rho"][G.K, G.J, G.I] * q[G.K, G.J, G.I]) * w)
        * G.dx
        * G.dy
        * G.dz
    )
    np.testing.assert_allclose(
        bk.to_numpy(out["asq"]), ref["asq"][0], rtol=0, atol=1e-13 * mass
    )


# ---------------------------------------------------------------------------


def _mp_inputs(seed):
    s = make_state(seed)
    rng = np.random.default_rng(seed + 100)
    q3d = s["q3d"].copy()
    q3d[1] = np.where(
        rng.random(G.s_shape) < 0.1, -1e-6, q3d[1]
    )  # pdefq has work to do
    rain = rng.random((1,) + G.s_shape[1:])
    prate = rng.random(G.s_shape[1:])
    qb = np.arange(1.0, 11.0)
    asq = np.array([0.1, 0.2, 0.3])
    return s, q3d, rain, prate, qb, asq


def _mp_call(bk, s, q3d, rain, prate, qb, asq, **kw):
    return mp_driver(
        bk,
        G,
        {"eqtset": 2, "tsmall": TSMALL, "qsmall": QSMALL},
        dt=DT,
        qbudget=A(bk, qb),
        asq=A(bk, asq),
        ruh=A(bk, s["ruh"]),
        rvh=A(bk, s["rvh"]),
        mh=A(bk, s["mh"]),
        rmh=A(bk, s["rmh"]),
        zh=A(bk, s["zh"]),
        mf=A(bk, s["mf"]),
        rain=A(bk, rain),
        prate=A(bk, prate),
        pi0=A(bk, s["pi0"]),
        th0=A(bk, s["th0"]),
        rho=A(bk, s["rho"]),
        prs=A(bk, s["prs"]),
        pp3d=A(bk, s["pp3d"]),
        th3d=A(bk, s["th3d"]),
        q3d=A(bk, q3d),
        **kw,
    )


def test_mp_driver(bk):
    s, q3d, rain, prate, qb, asq = _mp_inputs(10)
    ins = dict(
        params=np.array([G.ni, G.nj, G.nk, G.dx, G.dy, G.dz, DT, 2, TSMALL, QSMALL, 1]),
        ruh=s["ruh"],
        rvh=s["rvh"],
        mh=s["mh"],
        rmh=s["rmh"],
        zh=s["zh"],
        mf=s["mf"],
        rain=rain,
        prate=prate,
        pi0=s["pi0"],
        th0=s["th0"],
        rho=s["rho"],
        prs=s["prs"],
        pp3d=s["pp3d"],
        th3d=s["th3d"],
        q3d=q3d,
        qbudget=qb,
        asq=asq,
    )
    shp = G.s_shape
    ref = run_driver(
        "kessler_mp",
        ins,
        dict(
            rho=shp,
            prs=shp,
            pp3d=shp,
            th3d=shp,
            q3d=(3,) + shp,
            qten=(3,) + shp,
            thten=shp,
            dum1=shp,
            dum3=shp,
            rain=rain.shape,
            prate=prate.shape,
            qbudget=(10,),
            asq=(3,),
        ),
    )
    out = _mp_call(bk, s, q3d, rain, prate, qb, asq)
    S = (G.K, G.J, G.I)
    for k in ("rho", "prs", "pp3d", "th3d", "rain", "prate"):
        close(bk, out[k], ref[k], name=k)
    for k in ("thten", "dum1", "dum3"):  # Fortran scratch: compare where it is defined
        close(bk, bk.to_numpy(out[k])[S], ref[k][S], name=k)
    for n in range(3):
        close(bk, out["q3d"][n], ref["q3d"][n], name=f"q3d[{n}]")
    for n in (0, 2):
        close(bk, bk.to_numpy(out["qten"][n])[S], ref["qten"][n][S], name=f"qten[{n}]")
    qb_out = bk.to_numpy(out["qbudget"])
    np.testing.assert_allclose(qb_out, ref["qbudget"], rtol=1e-11)
    # asq: differences of column masses (cancellation) -> tolerance relative to the mass
    w = s["ruh"][None, None, G.I] * s["rvh"][None, G.J, None] * s["rmh"][G.K, G.J, G.I]
    for n in range(3):
        mass = np.sum(np.abs(s["rho"][S] * q3d[n][S]) * w) * G.dx * G.dy * G.dz
        np.testing.assert_allclose(
            bk.to_numpy(out["asq"])[n],
            ref["asq"][n],
            rtol=0,
            atol=1e-13 * mass,
            err_msg=f"asq[{n}]",
        )
    assert np.all(qb_out[:7] != qb[:7])  # every Kessler budget term was exercised


def test_mp_driver_jit():
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    s, q3d, rain, prate, qb, asq = _mp_inputs(11)
    eager = _mp_call(bk, s, q3d, rain, prate, qb, asq)
    nf = 4  # static sub-step bound (>= max nfall in the domain, which is 1 here)

    @jax.jit
    def step(q3d_, th3d_):
        return mp_driver(
            bk,
            G,
            {"eqtset": 2, "tsmall": TSMALL, "qsmall": QSMALL},
            dt=DT,
            qbudget=A(bk, qb),
            asq=A(bk, asq),
            ruh=A(bk, s["ruh"]),
            rvh=A(bk, s["rvh"]),
            mh=A(bk, s["mh"]),
            rmh=A(bk, s["rmh"]),
            zh=A(bk, s["zh"]),
            mf=A(bk, s["mf"]),
            rain=A(bk, rain),
            prate=A(bk, prate),
            pi0=A(bk, s["pi0"]),
            th0=A(bk, s["th0"]),
            rho=A(bk, s["rho"]),
            prs=A(bk, s["prs"]),
            pp3d=A(bk, s["pp3d"]),
            th3d=th3d_,
            q3d=q3d_,
            nfall_max=nf,
        )

    jitted = step(A(bk, q3d), A(bk, s["th3d"]))
    for k in ("q3d", "th3d", "pp3d", "prs", "rain", "qbudget"):
        np.testing.assert_allclose(
            np.asarray(jitted[k]),
            np.asarray(eager[k]),
            rtol=1e-13,
            atol=1e-13 * float(np.max(np.abs(eager[k]))),
            err_msg=k,
        )
