"""RRTMG longwave (module_ra_rrtmg_lw.F) against the original Fortran
(double-precision CM1 objects, tables read from run/RRTMG_LW_DATA): the
reduced k-distribution tables, every routine of the rrtmg_lw chain
(mcica_subcol_lw, inatm, cldprmc, setcoef, taumol, rtrnmc), rrtmg_lw itself
and RRTMG_LWRAD as radiation_driver calls it."""

import re

import numpy as np
import pytest
from fortran import HARNESS, requires_gfortran, run_driver

from cm1x import _ftables as ft
from cm1x import module_ra_rrtmg_lw as lw

pytestmark = requires_gfortran

RTOL = 1e-12
DATA = [ft.cm1_root() / "run" / "RRTMG_LW_DATA"]
NCOL, NL = 6, 30
PTOP = 2000.0  # p_top for rrtmg_lwinit (mode 3: 5 buffer layers)


def _params(mode, ncol=NCOL, nl=NL, p_top=PTOP, yr=2009, seed=150):
    return np.array([mode, ncol, nl, p_top, yr, seed], dtype=float)


def _close(a, b, rtol=RTOL, atol=0.0, name=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    scale = max(np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale if a.size else 0.0
    assert err <= max(rtol, atol / scale), f"{name}: rel err {err:.3e}"


def np_(bk, x):
    return np.asarray(bk.to_numpy(x)) if hasattr(bk, "to_numpy") else np.asarray(x)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _dumped_tables():
    src = (HARNESS / "drv_rrtmg_lw.F90").read_text()
    return re.findall(r"call wr\('(b\d\d_\w+)'", src)


def test_tables():
    T = lw.rrtmg_lwinit()
    names = _dumped_tables()
    single = {
        "tau_tbl": T["tau_tbl"],
        "exp_tbl": T["exp_tbl"],
        "tfn_tbl": T["tfn_tbl"],
        "totplnk": T["totplnk"],
        "absliq1": T["absliq1"],
        "absice3": T["absice3"],
        "chi_mls": T["chi_mls"],
        "rwgt": T["rwgt"],
    }
    want = {}
    for n in names:
        b, key = int(n[1:3]), n[4:]
        want[n] = np.asarray(T["bands"][b][key])
    want.update({k: np.asarray(v) for k, v in single.items()})
    # fracref(ng, n) is stored g-last here, like every other table
    gfirst = {k for k, v in want.items() if "_fracref" in k and v.ndim == 2}
    shapes = {k: (v.shape if k in gfirst else v.shape[::-1]) for k, v in want.items()}
    out = run_driver("rrtmg_lw", {"params": _params(1)}, shapes, files=DATA)
    assert len(names) > 100
    for k, v in want.items():
        # data tables are bit-identical; tfn_tbl cancels near tau=0.06 (libm exp ulps)
        tol = {"tau_tbl": 1e-15, "exp_tbl": 1e-15, "tfn_tbl": 1e-13, "rwgt": 1e-15}.get(
            k, 0.0
        )
        _close(v, out[k] if k in gfirst else out[k].T, rtol=tol, name=k)


# ---------------------------------------------------------------------------
# column inputs
# ---------------------------------------------------------------------------
def _tprof(p):
    t = np.maximum(212.0, 301.0 * (p / 1000.0) ** 0.19)
    return t + np.where(p < 100.0, 40.0 * (1.0 - p / 100.0) ** 2, 0.0)


def make_columns(seed=0, ncol=NCOL, nl=NL):
    """Layer arrays [nl, ncol], level arrays [nl+1, ncol] (p in mb).  Column
    0 is clear, 1 overcast liquid, 2 partly cloudy at all heights, 3 cold
    ice cloud, 4 mixed, 5 random."""
    rng = np.random.default_rng(seed)
    ps = 1005.0 + 10.0 * rng.random(ncol)
    eta = np.linspace(1.0, 0.0, nl + 1)[:, None] ** 1.3
    plev = ps * eta
    plev[-1] = 0.0
    play = 0.5 * (plev[:-1] + plev[1:])
    tlev = _tprof(plev) + rng.uniform(-3.0, 3.0, plev.shape)
    tlay = _tprof(play) + rng.uniform(-3.0, 3.0, play.shape)
    tsfc = tlev[0] + rng.uniform(-2.0, 2.0, ncol)
    q = np.maximum(
        3e-6, 0.016 * (play / 1000.0) ** 3.5 * rng.uniform(0.5, 1.0, play.shape)
    )
    lp = np.log(np.maximum(play, 1e-3))
    o3 = 2e-8 + 7e-6 * np.exp(-(((lp - np.log(8.0)) / 1.2) ** 2))
    ones = np.ones_like(play)
    gas = {
        "h2ovmr": q * 1.607793,
        "o3vmr": o3,
        "co2vmr": 3.9e-4 * ones,
        "ch4vmr": 1774.0e-9 * ones,
        "n2ovmr": 319.0e-9 * ones,
        "o2vmr": 0.209488 * ones,
        "cfc11vmr": 0.251e-9 * ones,
        "cfc12vmr": 0.538e-9 * ones,
        "cfc22vmr": 0.169e-9 * ones,
        "ccl4vmr": 0.093e-9 * ones,
    }
    cld = np.zeros_like(play)
    cld[3:9, 1] = 1.0
    cld[:, 2] = np.where(rng.random(nl) < 0.6, rng.uniform(0.05, 0.95, nl), 0.0)
    cld[12:20, 3] = rng.uniform(0.3, 1.0, 8)
    cld[2:16, 4] = rng.choice([0.0, 0.4, 1.0], 14)
    cld[:, 5] = np.where(rng.random(nl) < 0.4, rng.random(nl), 0.0)
    cld[-3:] = 0.0
    warm = tlay > 263.0
    clwp = np.where(
        cld > 0, np.where(warm, rng.uniform(1.0, 200.0, play.shape), 0.0), 0.0
    )
    ciwp = np.where(
        cld > 0, np.where(warm, 0.0, rng.uniform(0.5, 60.0, play.shape)), 0.0
    )
    ciwp[:, 4] = np.where(cld[:, 4] > 0, rng.uniform(0.0, 20.0, nl), 0.0)
    rei = rng.uniform(5.0, 130.0, play.shape)
    rel = rng.uniform(2.6, 59.0, play.shape)
    emis = rng.uniform(0.9, 1.0, (ncol, 16))
    return dict(
        play=play, plev=plev, tlay=tlay, tlev=tlev, tsfc=tsfc, emis=emis,
        cldfrac=cld, ciwp=ciwp, clwp=clwp, rei=rei, rel=rel, **gas,
    )  # fmt: skip


@pytest.fixture(scope="module")
def chain_ref():
    c = make_columns()
    ins = {k: (v.T if k == "emis" else v) for k, v in c.items()}
    ins["params"] = _params(2)
    nl, nc, ng = NL, NCOL, 140
    outs = {
        "cldfmcl": (nl, nc, ng),
        "ciwpmcl": (nl, nc, ng),
        "clwpmcl": (nl, nc, ng),
        "taucmcl": (nl, nc, ng),
        "uflx": (nl + 1, nc),
        "dflx": (nl + 1, nc),
        "hr": (nl, nc),
        "uflxc": (nl + 1, nc),
        "dflxc": (nl + 1, nc),
        "hrc": (nl, nc),
        "coldry": (nl, nc),
        "wbrodl": (nl, nc),
        "wkl": (nl, nc, 7),
        "wx": (nl, nc, 4),
        "pwvcm": (nc,),
        "taucmc": (nl, nc, ng),
        "lay": (nl, nc, 37),
        "ilay": (nl, nc, 6),
        "laytrop": (nc,),
        "planklay": (nl, nc, 16),
        "planklev": (nl + 1, nc, 16),
        "plankbnd": (nc, 16),
        "fracs": (nl, nc, ng),
        "taug": (nl, nc, ng),
        "flx": (nl + 1, nc, 8),
    }
    return c, run_driver("rrtmg_lw", ins, outs, files=DATA)


def _mcica(bk, c):
    xp = bk.xp
    a = {k: bk.asarray(v) for k, v in c.items()}
    return lw.mcica_subcol_lw(
        bk,
        permuteseed=150,
        play=a["play"],
        cldfrac=a["cldfrac"],
        ciwp=a["ciwp"],
        clwp=a["clwp"],
        cswp=xp.zeros_like(a["play"]),
        rei=a["rei"],
        rel=a["rel"],
        res=10.0 * xp.ones_like(a["play"]),
        tauc=xp.zeros(tuple(a["play"].shape) + (16,), dtype=a["play"].dtype),
    )


def test_mcica_subcol_lw(bk, chain_ref):
    c, ref = chain_ref
    mc = _mcica(bk, c)
    for k in ("cldfmcl", "ciwpmcl", "clwpmcl", "taucmcl"):
        _close(np_(bk, mc[k]), ref[k], rtol=0.0, name=k)
    # every cloudy layer has some cloudy subcolumns, partial ones not all
    f = ref["cldfmcl"].mean(axis=-1)
    assert np.all((f > 0) == (c["cldfrac"] > 0))


def test_rrtmg_lw_chain(bk, chain_ref):
    c, ref = chain_ref
    mc = _mcica(bk, c)
    a = {k: bk.asarray(v) for k, v in c.items()}
    gas = {k: a[k] for k in c if k.endswith("vmr")}
    tr = {}
    r = lw.rrtmg_lw(
        bk,
        play=a["play"],
        plev=a["plev"],
        tlay=a["tlay"],
        tlev=a["tlev"],
        tsfc=a["tsfc"],
        emis=a["emis"],
        **gas,
        **mc,
        trace=tr,
    )

    def n(x):
        return np_(bk, x)

    # inatm
    ia = tr["inatm"]
    _close(n(ia["coldry"]), ref["coldry"], name="coldry")
    _close(n(ia["wbrodl"]), ref["wbrodl"], name="wbrodl")
    for m in range(1, 8):
        _close(n(ia["wkl"][m]), ref["wkl"][..., m - 1], name=f"wkl{m}")
    for m in range(1, 5):
        _close(n(ia["wx"][m]), ref["wx"][..., m - 1], name=f"wx{m}")
    _close(n(ia["pwvcm"]), ref["pwvcm"], name="pwvcm")
    # cldprmc
    _close(n(tr["taucmc"]), ref["taucmc"], name="taucmc")
    # setcoef
    co = tr["setcoef"]
    laytrop = ref["laytrop"].astype(int)
    np.testing.assert_array_equal(n(co["laytrop"]), laytrop)
    lower = np.arange(NL)[:, None] < laytrop[None, :]

    def every(x):
        return np.ones_like(x, dtype=bool)

    lay_names = [
        "colh2o", "colco2", "colo3", "coln2o", "colco", "colch4", "colo2", "colbrd",
        "fac00", "fac01", "fac10", "fac11",
        "rat_h2oco2", "rat_h2oco2_1", "rat_h2oo3", "rat_h2oo3_1", "rat_h2on2o",
        "rat_h2on2o_1", "rat_h2och4", "rat_h2och4_1", "rat_n2oco2", "rat_n2oco2_1",
        "rat_o3co2", "rat_o3co2_1", "selffac", "selffrac", "forfac", "forfrac",
        "minorfrac", "scaleminor", "scaleminorn2",
    ]  # fmt: skip
    lower_only = {"selffrac", "indself"} | {
        f"rat_{p}{s}"
        for p in ("h2oo3", "h2on2o", "h2och4", "n2oco2")
        for s in ("", "_1")
    }
    upper_only = {"rat_o3co2", "rat_o3co2_1"}

    def mask(name):
        if name in lower_only:
            return lower
        if name in upper_only:
            return ~lower
        return every(lower)

    for i, name in enumerate(lay_names):
        m = mask(name)
        _close(n(co[name])[m], ref["lay"][..., i][m], name=name)
    for i, name in enumerate(("jp", "jt", "jt1", "indself", "indfor", "indminor")):
        m = mask(name)
        np.testing.assert_array_equal(
            n(co[name])[m], ref["ilay"][..., i][m].astype(int)
        )
    for name in ("planklay", "planklev", "plankbnd"):
        _close(n(co[name]), ref[name], name=name)
    # taumol
    _close(n(tr["taumol"]["taug"]), ref["taug"], name="taug")
    _close(n(tr["taumol"]["fracs"]), ref["fracs"], name="fracs")
    # rtrnmc
    rt = tr["rtrnmc"]
    for i, name in enumerate(
        ("totuflux", "totdflux", "fnet", "htr", "totuclfl", "totdclfl", "fnetc", "htrc")
    ):
        _close(n(rt[name]), ref["flx"][..., i], name=name)
    # rrtmg_lw
    for k in ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc"):
        _close(n(r[k]), ref[k], name=k)


def test_taumol_bands(bk, chain_ref):
    """Per-band check of taumol: each band's g-points against the Fortran."""
    c, ref = chain_ref
    mc = _mcica(bk, c)
    a = {k: bk.asarray(v) for k, v in c.items()}
    gas = {k: a[k] for k in c if k.endswith("vmr")}
    tr = {}
    lw.rrtmg_lw(
        bk, play=a["play"], plev=a["plev"], tlay=a["tlay"], tlev=a["tlev"],
        tsfc=a["tsfc"], emis=a["emis"], **gas, **mc, trace=tr,
    )  # fmt: skip
    taug = np_(bk, tr["taumol"]["taug"])
    ngs = np.concatenate([[0], np.cumsum(lw.rrtmg_lwinit()["ngc"])])
    for b in range(16):
        s = slice(ngs[b], ngs[b + 1])
        _close(taug[..., s], ref["taug"][..., s], name=f"taug band {b + 1}")


# ---------------------------------------------------------------------------
# RRTMG_LWRAD
# ---------------------------------------------------------------------------
def make_wrf_columns(seed=1, ncol=NCOL, nk=NL):
    """CM1-like model columns for RRTMG_LWRAD: levels up to ~20 mb (p_top
    2000 Pa), Kessler-type water species (qc, qr; cold layers become ice)."""
    rng = np.random.default_rng(seed)
    ps = 100500.0 + 1000.0 * rng.random(ncol)
    eta = np.linspace(0.0, 1.0, nk + 1)[:, None]
    p8w = PTOP + (ps - PTOP) * (1.0 - eta) ** 1.2
    p3d = 0.5 * (p8w[:-1] + p8w[1:]) * (1.0 + 0.002 * rng.standard_normal((nk, ncol)))
    t8w = _tprof(p8w * 0.01) + rng.uniform(-2.0, 2.0, p8w.shape)
    t3d = _tprof(p3d * 0.01) + rng.uniform(-2.0, 2.0, p3d.shape)
    qv = np.maximum(
        1e-7, 0.017 * (p3d / 1.0e5) ** 3.5 * rng.uniform(0.4, 1.0, p3d.shape)
    )
    qv[0, 0] = -1e-9  # negative values are clipped
    cf = np.where(
        rng.random(p3d.shape) < 0.35, rng.choice([0.3, 0.7, 1.0], p3d.shape), 0.0
    )
    cf[:, 0] = 0.0
    qc = np.where(cf > 0, rng.uniform(1e-5, 2e-3, p3d.shape), 0.0)
    qr = np.where(rng.random(p3d.shape) < 0.3, rng.uniform(0.0, 1e-3, p3d.shape), 0.0)
    return {
        "p8w": p8w,
        "p3d": p3d,
        "t3d": t3d,
        "t8w": t8w,
        "cldfra3d": cf,
        "qv3d": qv,
        "qc3d": qc,
        "qr3d": qr,
        "tsk": t8w[0] + rng.uniform(-3.0, 3.0, ncol),
        "emiss": rng.uniform(0.9, 1.0, ncol),
        "xland": np.where(np.arange(ncol) % 2 == 0, 1.0, 2.0),
    }


@pytest.fixture(scope="module")
def lwrad_ref():
    w = make_wrf_columns()
    nk, nc = NL, NCOL

    def lev3(x):  # [nk(+1), C] -> Fortran (ncol, 1, nk+1)
        out = np.zeros((nk + 1, 1, nc))
        out[: x.shape[0], 0] = x
        return out

    ins = {k: (lev3(v) if v.ndim == 2 else v[None]) for k, v in w.items()}
    ins["params"] = _params(3)
    outs = {k: (nk + 1, 1, nc) for k in ("rthratenlw", "lwtenc", "o33d")}
    for k in ("glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc"):
        outs[k] = (1, nc)
    for k in ("lwupb", "lwupbc", "lwdnb", "lwdnbc"):
        outs[k] = (1, nc)
    outs["co2out"] = (1,)
    return w, run_driver("rrtmg_lw", ins, outs, files=DATA)


def test_rrtmg_lwrad(bk, lwrad_ref):
    w, ref = lwrad_ref
    a = {k: bk.asarray(v) for k, v in w.items()}
    z = bk.xp.zeros_like(a["tsk"])
    r = lw.RRTMG_LWRAD(
        bk, nlayers=lw.nlayers_lw(NL, PTOP), xice=z, snow=z, yr=2009, **a
    )
    for k in ("rthratenlw", "lwtenc", "o33d"):
        _close(np_(bk, r[k]), ref[k][:NL, 0], name=k)
    for k in ("glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc"):
        _close(np_(bk, r[k]), ref[k][0], name=k)
    for k in ("lwupb", "lwupbc", "lwdnb", "lwdnbc"):
        _close(np_(bk, r[k]), ref[k][0], name=k)
    assert abs(r["co2out"] - ref["co2out"][0]) <= 1e-15 * ref["co2out"][0]
    assert np.all(np.abs(ref["rthratenlw"][:NL]) > 0)
