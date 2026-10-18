"""RRTMG shortwave (module_ra_rrtmg_sw.F) against the original Fortran
(double-precision CM1 objects, tables read from run/RRTMG_SW_DATA): the
reduced k-distribution and cloud-optics tables, reftra_sw on every branch,
each routine of the rrtmg_sw chain (mcica_subcol_sw, inatm_sw, cldprmc_sw,
setcoef_sw, taumol_sw, spcvmc_sw), rrtmg_sw itself and RRTMG_SWRAD as
radiation_driver calls it (day and night columns)."""

import re

import numpy as np
import pytest
from fortran import HARNESS, requires_gfortran, run_driver

from cm1x import _ftables as ft
from cm1x import module_ra_rrtmg_sw as sw

pytestmark = requires_gfortran

RTOL = 1e-12
DATA = [ft.cm1_root() / "run" / "RRTMG_SW_DATA"]
NCOL, NL = 6, 31
SCON = 1361.3


def _params(mode, ncol=NCOL, nl=NL, scon=SCON, yr=2009, seed=1, mu=0.5):
    return np.array([mode, ncol, nl, scon, yr, seed, mu], dtype=float)


def _close(a, b, rtol=RTOL, name=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    scale = max(np.max(np.abs(b)), 1e-300) if b.size else 1.0
    err = np.max(np.abs(a - b)) / scale if a.size else 0.0
    assert err <= rtol, f"{name}: rel err {err:.3e}"


def np_(bk, x):
    return np.asarray(bk.to_numpy(x))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def test_tables():
    T = sw.rrtmg_swinit()
    names = re.findall(
        r"call wr\('(b\d\d_\w+)'", (HARNESS / "drv_rrtmg_sw.F90").read_text()
    )
    names += re.findall(
        r"b1\(1\) = \w+; call wr\('(b\d\d_\w+)'",
        (HARNESS / "drv_rrtmg_sw.F90").read_text(),
    )
    want = {n: np.asarray(T["bands"][int(n[1:3])][n[4:]], dtype=float) for n in names}
    for k in ("exp_tbl", "rwgt", "pref", "preflog", "tref"):
        want[k] = np.asarray(T[k])
    for k in (
        "extliq1",
        "ssaliq1",
        "asyliq1",
        "extice3",
        "ssaice3",
        "asyice3",
        "fdlice3",
    ):
        want[k] = T[k]
    # sfluxref(ng, n), rayla(ng, n) are stored g-last like every other table
    gfirst = {
        k for k, v in want.items() if k[4:] in ("sfluxref", "rayla") and v.ndim == 2
    }
    shapes = {
        k: (v.shape if k in gfirst else v.shape[::-1]) if v.ndim else (1,)
        for k, v in want.items()
    }
    out = run_driver("rrtmg_sw", {"params": _params(1)}, shapes, files=DATA)
    assert len(names) > 80
    for k, v in want.items():
        got = out[k] if (k in gfirst or v.ndim == 0) else out[k].T
        tol = {"exp_tbl": 1e-15, "rwgt": 1e-15}.get(k, 0.0)
        _close(v.reshape(got.shape), got, rtol=tol, name=k)


# ---------------------------------------------------------------------------
# reftra_sw
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu", [0.03, 0.4, 1.0])
def test_reftra_sw(bk, mu):
    rng = np.random.default_rng(3)
    n = 400
    chk = (rng.random(n) < 0.9).astype(float)
    pw = rng.uniform(0.0, 1.0, n)
    pw[:60] = 1.0 - rng.uniform(0.0, 1e-6, 60)  # conservative branch
    pw[60:70] = 1.0
    pw[70:75] = 0.0
    pgg = rng.uniform(0.0, 0.95, n)
    pgg[75:80] = 1.0
    ptau = 10.0 ** rng.uniform(-6.0, 3.5, n)
    ptau[80:90] = 0.0
    ins = {
        "params": _params(4, ncol=1, nl=n, mu=mu),
        "lrtchk": chk,
        "pgg": pgg,
        "ptau": ptau,
        "pw": pw,
    }
    ref = run_driver(
        "rrtmg_sw",
        ins,
        {k: (n,) for k in ("pref", "prefd", "ptra", "ptrad")},
        files=DATA,
    )
    a = {k: bk.asarray(v) for k, v in (("pgg", pgg), ("ptau", ptau), ("pw", pw))}
    r = sw.reftra_sw(bk, lrtchk=bk.asarray(chk) > 0.5, prmuz=mu, **a)
    for k in ("pref", "prefd", "ptra", "ptrad"):
        _close(np_(bk, r[k]), ref[k], name=k)


# ---------------------------------------------------------------------------
# column inputs
# ---------------------------------------------------------------------------
def _tprof(p):
    t = np.maximum(212.0, 301.0 * (p / 1000.0) ** 0.19)
    return t + np.where(p < 100.0, 40.0 * (1.0 - p / 100.0) ** 2, 0.0)


def make_columns(seed=0, ncol=NCOL, nl=NL):
    """Layer arrays [nl, ncol], level arrays [nl+1, ncol] (p in mb) shaped
    like RRTMG_SWRAD's: the top level at 1e-5 mb, the top layer at half the
    pressure of the level below.  Column 0 is clear, 1 overcast liquid, 2
    partly cloudy at all heights, 3 cold ice cloud, 4 mixed, 5 random."""
    rng = np.random.default_rng(seed)
    ps = 1005.0 + 10.0 * rng.random(ncol)
    eta = np.linspace(1.0, 0.0, nl)[:, None] ** 1.3
    plev = np.concatenate([20.0 + (ps - 20.0) * eta, np.full((1, ncol), 1.0e-5)])
    play = 0.5 * (plev[:-1] + plev[1:])
    play[-1] = 0.5 * plev[-2]
    tlev = _tprof(plev) + rng.uniform(-3.0, 3.0, plev.shape)
    tlay = _tprof(play) + rng.uniform(-3.0, 3.0, play.shape)
    tsfc = tlev[0] + rng.uniform(-2.0, 2.0, ncol)
    q = np.maximum(
        3e-6, 0.016 * (play / 1000.0) ** 3.5 * rng.uniform(0.5, 1.0, play.shape)
    )
    lp = np.log(np.maximum(play, 1e-3))
    o3 = 2e-8 + 7e-6 * np.exp(-(((lp - np.log(8.0)) / 1.2) ** 2))
    ones = np.ones_like(play)
    cld = np.zeros_like(play)
    cld[3:9, 1] = 1.0
    cld[:, 2] = np.where(rng.random(nl) < 0.6, rng.uniform(0.05, 0.95, nl), 0.0)
    cld[12:20, 3] = rng.uniform(0.3, 1.0, 8)
    cld[2:16, 4] = rng.choice([0.0, 0.4, 1.0], 14)
    cld[:, 5] = np.where(rng.random(nl) < 0.4, rng.random(nl), 0.0)
    cld[-3:] = 0.0
    warm = tlay > 263.0
    clwp = np.where((cld > 0) & warm, rng.uniform(1.0, 200.0, play.shape), 0.0)
    ciwp = np.where((cld > 0) & ~warm, rng.uniform(0.5, 60.0, play.shape), 0.0)
    ciwp[:, 4] = np.where(cld[:, 4] > 0, rng.uniform(0.0, 20.0, nl), 0.0)
    rel = rng.uniform(2.6, 59.0, play.shape)
    rel[5, 1] = 2.0  # liquid table index 0 -> 1 (fint < 0)
    return {
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev, "tsfc": tsfc,
        "h2ovmr": q * 1.607793, "o3vmr": o3, "co2vmr": 3.9e-4 * ones,
        "ch4vmr": 1774.0e-9 * ones, "n2ovmr": 319.0e-9 * ones, "o2vmr": 0.209488 * ones,
        "cldfrac": cld, "ciwp": ciwp, "clwp": clwp,
        "rei": rng.uniform(5.0, 130.0, play.shape), "rel": rel,
        "asdir": rng.uniform(0.05, 0.3, ncol), "asdif": rng.uniform(0.05, 0.3, ncol),
        "aldir": rng.uniform(0.05, 0.3, ncol), "aldif": rng.uniform(0.05, 0.3, ncol),
        "coszen": np.array([0.9, 0.5, 0.2, 0.05, 1.0, 0.7])[:ncol],
    }  # fmt: skip


@pytest.fixture(scope="module")
def chain_ref():
    c = make_columns()
    ins = dict(c, params=_params(2))
    nl, nc, ng = NL, NCOL, 112
    g3 = (nl, nc, ng)
    outs = {
        k: g3
        for k in (
            "cldfmcl",
            "ciwpmcl",
            "clwpmcl",
            "taucmcl",
            "ssacmcl",
            "asmcmcl",
            "fsfcmcl",
        )
    }
    for k in (
        "swuflx",
        "swdflx",
        "swuflxc",
        "swdflxc",
        "swdkdir",
        "swdkdif",
        "swdkdirc",
    ):
        outs[k] = (nl + 1, nc)
    outs.update(
        swhr=(nl, nc), swhrc=(nl, nc), coldry=(nl, nc), wkl=(nl, nc, 7), pdp=(nl, nc),
        adjflux=(nc,), cldprmc=(4, nl, nc, ng), lay=(nl, nc, 16), ilay=(nl, nc, 5),
        laytrop=(nc, 2), sfluxzen=(nc, ng), taug=g3, taur=g3, flx=(nl + 1, nc, 6),
    )  # fmt: skip
    return c, run_driver("rrtmg_sw", ins, outs, files=DATA)


def _mcica(bk, c):
    xp = bk.xp
    a = {k: bk.asarray(v) for k, v in c.items()}
    zb = xp.zeros(tuple(a["play"].shape) + (14,), dtype=a["play"].dtype)
    return sw.mcica_subcol_sw(
        bk, permuteseed=1, play=a["play"], cldfrac=a["cldfrac"], ciwp=a["ciwp"],
        clwp=a["clwp"], cswp=xp.zeros_like(a["play"]), rei=a["rei"], rel=a["rel"],
        res=10.0 * xp.ones_like(a["play"]), tauc=zb, ssac=zb + 1.0, asmc=zb, fsfc=zb,
    )  # fmt: skip


def test_mcica_subcol_sw(bk, chain_ref):
    c, ref = chain_ref
    mc = _mcica(bk, c)
    for k in (
        "cldfmcl",
        "ciwpmcl",
        "clwpmcl",
        "taucmcl",
        "ssacmcl",
        "asmcmcl",
        "fsfcmcl",
    ):
        _close(np_(bk, mc[k]), ref[k], rtol=0.0, name=k)


def _run_chain(bk, c):
    a = {k: bk.asarray(v) for k, v in c.items()}
    gas = {k: a[k] for k in c if k.endswith("vmr")}
    tr = {}
    r = sw.rrtmg_sw(
        bk, play=a["play"], plev=a["plev"], tlay=a["tlay"], tlev=a["tlev"],
        tsfc=a["tsfc"], **gas, asdir=a["asdir"], asdif=a["asdif"], aldir=a["aldir"],
        aldif=a["aldif"], coszen=a["coszen"], adjes=1.0, scon=SCON, **_mcica(bk, c),
        trace=tr,
    )  # fmt: skip
    return r, tr


def test_rrtmg_sw_chain(bk, chain_ref):
    c, ref = chain_ref
    r, tr = _run_chain(bk, c)

    def n(x):
        return np_(bk, x)

    ia = tr["inatm"]
    _close(n(ia["coldry"]), ref["coldry"], name="coldry")
    _close(n(ia["pdp"]), ref["pdp"], name="pdp")
    for m in (1, 2, 3, 4, 6, 7):
        _close(n(ia["wkl"][m]), ref["wkl"][..., m - 1], name=f"wkl{m}")
    assert abs(float(ia["adjflux"]) - ref["adjflux"][0]) <= 1e-15
    # cldprmc_sw
    for i, k in enumerate(("taormc", "taucmc", "ssacmc", "asmcmc")):
        _close(n(tr["cldprmc"][k]), ref["cldprmc"][i], name=k)
    # setcoef_sw
    co = tr["setcoef"]
    np.testing.assert_array_equal(n(co["laytrop"]), ref["laytrop"][:, 0].astype(int))
    np.testing.assert_array_equal(n(co["laylow"]), ref["laytrop"][:, 1].astype(int))
    names = (
        "colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2", "colmol", "co2mult",
        "fac00", "fac01", "fac10", "fac11", "selffac", "selffrac", "forfac", "forfrac",
    )  # fmt: skip
    for i, k in enumerate(names):
        _close(n(co[k]), ref["lay"][..., i], name=k)
    for i, k in enumerate(("jp", "jt", "jt1", "indself", "indfor")):
        np.testing.assert_array_equal(
            n(co[k]), ref["ilay"][..., i].astype(int), err_msg=k
        )
    # taumol_sw, per band
    ngs = np.concatenate([[0], np.cumsum(sw.rrtmg_swinit()["ngc"])])
    tau = tr["taumol"]
    for b in range(14):
        s = slice(ngs[b], ngs[b + 1])
        _close(n(tau["taug"])[..., s], ref["taug"][..., s], name=f"taug band {b + 16}")
        _close(n(tau["taur"])[..., s], ref["taur"][..., s], name=f"taur band {b + 16}")
        _close(
            n(tau["sfluxzen"])[..., s],
            ref["sfluxzen"][..., s],
            name=f"sfluxzen band {b + 16}",
        )
    # spcvmc_sw
    sp = tr["spcvmc"]
    for i, k in enumerate(("pbbfu", "pbbfd", "pbbcu", "pbbcd", "pbbfddir", "pbbcddir")):
        _close(n(sp[k]), ref["flx"][..., i], name=k)
    # rrtmg_sw
    for k in (
        "swuflx",
        "swdflx",
        "swuflxc",
        "swdflxc",
        "swdkdir",
        "swdkdif",
        "swdkdirc",
        "swhr",
        "swhrc",
    ):
        _close(n(r[k]), ref[k], name=k)
    assert np.all(ref["swdflx"][-1] > 50.0)


# ---------------------------------------------------------------------------
# RRTMG_SWRAD
# ---------------------------------------------------------------------------
NK = 30


def make_wrf_columns(seed=1, ncol=NCOL, nk=NK):
    """CM1-like model columns for RRTMG_SWRAD (top near 20 mb), Kessler-type
    water species, one night column and one with the sun just below the
    horizon."""
    rng = np.random.default_rng(seed)
    ps = 100500.0 + 1000.0 * rng.random(ncol)
    eta = np.linspace(0.0, 1.0, nk + 1)[:, None]
    p8w = 2000.0 + (ps - 2000.0) * (1.0 - eta) ** 1.2
    p3d = 0.5 * (p8w[:-1] + p8w[1:]) * (1.0 + 0.002 * rng.standard_normal((nk, ncol)))
    t8w = _tprof(p8w * 0.01) + rng.uniform(-2.0, 2.0, p8w.shape)
    t3d = _tprof(p3d * 0.01) + rng.uniform(-2.0, 2.0, p3d.shape)
    qv = np.maximum(
        1e-7, 0.017 * (p3d / 1.0e5) ** 3.5 * rng.uniform(0.4, 1.0, p3d.shape)
    )
    qv[0, 0] = -1e-9
    cf = np.where(
        rng.random(p3d.shape) < 0.35, rng.choice([0.3, 0.7, 1.0], p3d.shape), 0.0
    )
    cf[:, 0] = 0.0
    qc = np.where(cf > 0, rng.uniform(1e-5, 2e-3, p3d.shape), 0.0)
    qr = np.where(rng.random(p3d.shape) < 0.3, rng.uniform(0.0, 1e-3, p3d.shape), 0.0)
    return {
        "p8w": p8w, "p3d": p3d, "t3d": t3d, "t8w": t8w, "cldfra3d": cf, "qv3d": qv,
        "qc3d": qc, "qr3d": qr, "dz8w": rng.uniform(400.0, 600.0, (nk, ncol)),
        "tsk": t8w[0] + rng.uniform(-3.0, 3.0, ncol),
        "albedo": rng.uniform(0.05, 0.3, ncol),
        "xland": np.where(np.arange(ncol) % 2 == 0, 1.0, 2.0),
        "coszen": np.array([0.8, 0.45, -0.3, 0.12, 0.0, 0.95])[:ncol],
    }  # fmt: skip


@pytest.fixture(scope="module")
def swrad_ref():
    w = make_wrf_columns()
    nk, nc = NK, NCOL
    rng = np.random.default_rng(9)
    prev_tenc = rng.uniform(-1e-5, 1e-5, (nk, nc))
    prev_gsw = rng.uniform(0.0, 50.0, nc)

    def lev3(x):
        out = np.zeros((nk + 1, 1, nc))
        out[: x.shape[0], 0] = x
        return out

    ins = {
        k: (lev3(v) if v.ndim == 2 else v[None]) for k, v in w.items() if k != "coszen"
    }
    ins.update(
        xcoszen=w["coszen"][None], swtenc_in=lev3(prev_tenc), gsw_in=prev_gsw[None],
        params=_params(3, nl=nk),
    )  # fmt: skip
    outs = {"rthratensw": (nk + 1, 1, nc), "swtenc": (nk + 1, 1, nc), "co2out": (1,)}
    for k in (
        "gsw", "swcf", "coszr", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc",
        "swdnb", "swdnbc", "swddir", "swddni", "swddif", "swdownc", "swddnic", "swddirc",
    ):  # fmt: skip
        outs[k] = (1, nc)
    return w, prev_tenc, prev_gsw, run_driver("rrtmg_sw", ins, outs, files=DATA)


def test_rrtmg_swrad(bk, swrad_ref):
    w, prev_tenc, prev_gsw, ref = swrad_ref
    a = {k: bk.asarray(v) for k, v in w.items()}
    z = bk.xp.zeros_like(a["tsk"])
    r = sw.RRTMG_SWRAD(
        bk, solcon=SCON, xice=z, snow=z, yr=2009, swtenc=bk.asarray(prev_tenc),
        gsw=bk.asarray(prev_gsw), **a,
    )  # fmt: skip
    for k in ("rthratensw", "swtenc"):
        _close(np_(bk, r[k]), ref[k][:NK, 0], name=k)
    day = w["coszen"] > 0
    for k in (
        "gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc", "swdnb",
        "swdnbc", "swddir", "swddif", "swdownc", "swddirc", "coszr",
    ):  # fmt: skip
        _close(np_(bk, r[k]), ref[k][0], name=k)
    # swddni, swddnic divide by coszen (inf/nan at night in the Fortran)
    for k in ("swddni", "swddnic"):
        _close(np_(bk, r[k])[day], ref[k][0][day], name=k)
    assert abs(r["co2out"] - ref["co2out"][0]) <= 1e-15 * ref["co2out"][0]
    assert np.all(np.abs(ref["rthratensw"][:NK, 0][:, day]) > 0)
