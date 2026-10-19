"""LES subgrid turbulence (cm1setup=1, sgsmodel=1): cm1x.turb / cm1x.turbtend
against the double-precision Fortran (turb.F, turbtend.F, bc.F and the inline
solve1/solve2 code) via fortran_harness/drv_turb.F90."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x import constants as C
from cm1x import turb as T
from cm1x import turbtend as TT
from cm1x.grid import Grid

pytestmark = requires_gfortran

G = Grid(ni=7, nj=6, nk=12, dx=1000.0, dy=900.0, dz=500.0)
DT = 6.0
RTOL = 1e-12

MOIST = {
    "imoist": 1,
    "ptype": 1,
    "iice": 0,
    "numq": 3,
    "nqv": 1,
    "nqc": 2,
    "nqi": 0,
    "nql1": 2,
    "nql2": 3,
    "nqs1": 0,
    "nqs2": -1,
    "eqtset": 2,
}
DRY = {
    "imoist": 0,
    "ptype": 0,
    "iice": 0,
    "numq": 1,
    "nqv": 0,
    "nqc": 0,
    "nqi": 0,
    "nql1": 0,
    "nql2": -1,
    "nqs1": 0,
    "nqs2": -1,
    "eqtset": 2,
}

# supercell_kessler namelist settings (open lateral boundaries)
SUPER = dict(
    cm1setup=1,
    sgsmodel=1,
    tconfig=1,
    bcturbs=1,
    horizturb=0,
    doimpl=1,
    idiss=1,
    ipbl=0,
    isfcflx=0,
    bbc=1,
    tbc=1,
    wbc=2,
    ebc=2,
    sbc=2,
    nbc=2,
    **MOIST,
)

CASES = {
    "supercell": SUPER,
    "periodic": {**SUPER, "wbc": 1, "ebc": 1, "sbc": 1, "nbc": 1},
    "walls_explicit": {
        **SUPER,
        "wbc": 3,
        "ebc": 4,
        "sbc": 4,
        "nbc": 3,
        "doimpl": 0,
        "bcturbs": 2,
    },
    "mixed_tconfig2": {
        **SUPER,
        "wbc": 2,
        "ebc": 1,
        "sbc": 1,
        "nbc": 2,
        "tconfig": 2,
        "bcturbs": 2,
    },
    "dry": {**SUPER, **DRY, "idiss": 1},
}

PKEYS = [
    "mode",
    "ni",
    "nj",
    "nk",
    "dx",
    "dy",
    "dz",
    "dt",
    "tconfig",
    "bcturbs",
    "doimpl",
    "wbc",
    "ebc",
    "sbc",
    "nbc",
    "imoist",
    "ptype",
    "iice",
    "numq",
    "nqv",
    "nqc",
    "nqi",
    "nql1",
    "nql2",
    "nqs1",
    "nqs2",
    "eqtset",
    "idiss",
    "iflux",
    "iarg",
    "rtime",
    "ramp_time",
    "dttmp",
]


def params(mode, cfg, g=G, **kw):
    d = dict(
        cfg, mode=mode, ni=g.ni, nj=g.nj, nk=g.nk, dx=g.dx, dy=g.dy, dz=g.dz, dt=DT
    )
    d.update(kw)
    return np.array([float(d.get(k, 0)) for k in PKEYS])


def _qvs(t, p):
    esl = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = np.minimum(esl, 0.5 * p)
    return C.eps * esl / (p - esl)


def make_state(seed=0, g=G, moist=True):
    """Random but physical fields over the whole halo'd arrays."""
    rng = np.random.default_rng(seed)
    S, W, U, V = g.s_shape, g.w_shape, g.u_shape, g.v_shape

    def jit(shape, a=0.05):
        return 1.0 + a * rng.uniform(-1, 1, shape)

    z = np.array(g.zh())[:, None, None] * np.ones(S)
    th0 = 300.0 + 4.0e-3 * z
    pi0 = 1.0 - C.g * z / (C.cp * 302.0)
    th = rng.uniform(-1.0, 1.0, S)
    pp = rng.uniform(-2e-4, 2e-4, S)
    prs = C.p00 * (pi0 + pp) ** C.cpdrd
    t = (th0 + th) * (pi0 + pp)
    rho = prs / (C.rd * t)
    uh = jit(S[2])
    vh = jit(S[1])
    uf = jit(U[2])
    vf = jit(V[1])
    mh = jit(S)
    mf = jit(W)
    c2 = 0.5 + 0.05 * rng.uniform(-1, 1, S)
    st = {
        "th0": th0,
        "pi0": pi0,
        "th": th,
        "pp": pp,
        "prs": prs,
        "rho": rho,
        "rr": 1.0 / rho,
        "rf": rho * jit(S, 0.02),
        "uh": uh,
        "ruh": 1.0 / uh,
        "vh": vh,
        "rvh": 1.0 / vh,
        "uf": uf,
        "vf": vf,
        "mh": mh,
        "mf": mf,
        "rmf": 1.0 / mf,
        "c1": 1.0 - c2,
        "c2": c2,
        "ua": rng.uniform(-10, 10, U),
        "va": rng.uniform(-10, 10, V),
        "wa": rng.uniform(-5, 5, W),
        "uten": rng.uniform(-1e-2, 1e-2, U),
        "vten": rng.uniform(-1e-2, 1e-2, V),
        "wten": rng.uniform(-1e-2, 1e-2, W),
        "sten": rng.uniform(-1e-3, 1e-3, S),
        "s": rng.uniform(-2, 2, S),
    }
    tke = rng.uniform(0.0, 3.0, W)
    tke = np.where(rng.random(W) < 0.15, 0.0, tke)
    tke = np.where(rng.random(W) < 0.05, 1e-11, tke)
    st["tkea"] = tke
    st["tke3d"] = rng.uniform(0.0, 3.0, W)
    st["tketen"] = rng.uniform(-1e-3, 1e-3, W)
    nm = rng.uniform(-2e-4, 4e-4, W)
    st["nm"] = np.where(rng.random(W) < 0.1, 5e-7, nm)
    st["defv"] = rng.uniform(0.0, 1e-4, W)
    st["defh"] = rng.uniform(0.0, 1e-4, W)
    st["dissten"] = rng.uniform(0.0, 1e-2, W)
    st["lenscl"] = rng.uniform(0.0, 500.0, W)
    for n in ("kmh", "kmv", "khh", "khv"):
        st[n] = rng.uniform(0.0, 200.0, W)
    for n in ("t11", "t12", "t13", "t22", "t23", "t33"):
        st[n] = rng.uniform(-0.5, 0.5, S)
    st["dum7"] = rng.uniform(0.0, 1.0, S)
    st["dum8"] = rng.uniform(0.0, 1.0, S)
    st["thten1"] = rng.uniform(-1e-3, 1e-3, S)
    cm0 = np.full(S[1:], T.c_m)
    cm0[1, 2] = 5e-5  # below cmemin: left alone by the ramp
    st["cm0"] = cm0
    cc = T.turb_constants(_np_bk(), g, None, cm0=cm0)
    for n in ("cme", "csm", "ce1", "ce2"):
        st[n] = np.asarray(cc[n]) * jit(W, 0.01)
    if moist:
        qvs = _qvs(t, prs)
        qv = rng.uniform(0.85, 1.05, S) * qvs
        qc = np.where(rng.random(S) < 0.5, rng.uniform(0.0, 2e-3, S), 0.0)
        qr = np.where(rng.random(S) < 0.5, rng.uniform(0.0, 3e-3, S), 0.0)
        st["qa"] = np.stack([qv, qc, qr])
    else:
        st["qa"] = np.zeros((1,) + S)
    st["qten"] = rng.uniform(-1e-6, 1e-6, st["qa"].shape)
    return st


def _np_bk():
    from cm1x import get_backend

    return get_backend("numpy", "float64")


def close(bk, got, ref, rtol=RTOL, name=""):
    got = got if isinstance(got, np.ndarray) else bk.to_numpy(got)
    ref = np.asarray(ref)
    atol = rtol * max(np.max(np.abs(ref)), 1e-300)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=name)


def A(bk, x):
    return bk.asarray(np.asarray(x))


def As(bk, st, *names):
    return {n: A(bk, st[n]) for n in names}


def qlist(bk, qa):
    return [A(bk, q) for q in qa]


GRID1 = ("uh", "uf", "vh", "vf", "mh", "mf")
KS = ("kmh", "kmv", "khh", "khv")
TAU = ("t11", "t12", "t13", "t22", "t23", "t33")
CONST = ("cme", "csm", "ce1", "ce2")
SHAPES = {}


def shapes(g=G):
    S, W, U, V = g.s_shape, g.w_shape, g.u_shape, g.v_shape
    d = {
        n: W
        for n in (
            "nm",
            "defv",
            "defh",
            "lenscl",
            "dissten",
            "tkea",
            "tke3d",
            "tketen",
            "tten",
            "wten",
            "w",
            "t",
            *KS,
            *CONST,
        )
    }
    d.update({n: S for n in ("sten", "thten1", "dum7", "dum8", *TAU)})
    d.update(uten=U, vten=V, u=U, v=V)
    return d


SH = shapes()


def fortran(mode, cfg, st, ins, outs, renames=None, **kw):
    renames = renames or {}
    data = {"params": params(mode, cfg, **kw)}
    for n in ins:
        data[n] = st[renames.get(n, n)]
    return run_driver("turb", data, {n: SH[n] for n in outs})


# ---------------------------------------------------------------------------
# turb.F


@pytest.mark.parametrize("case", ["supercell", "dry"])
def test_calcnm(bk, case):
    cfg = CASES[case]
    st = make_state(1, moist=cfg["imoist"] == 1)
    ref = fortran(
        1,
        cfg,
        st,
        ["c1", "c2", "mf", "pi0", "th0", "prs", "pp", "th", "qa", "nm"],
        ["nm"],
    )
    out = T.calcnm(
        bk,
        G,
        cfg,
        qa=qlist(bk, st["qa"]),
        **As(bk, st, "c1", "c2", "mf", "pi0", "th0", "nm", "prs", "pp", "th"),
    )
    close(bk, out["nm"], ref["nm"], name="nm")
    if cfg["imoist"]:
        # both branches (saturated / unsaturated) were exercised
        assert np.ptp(ref["nm"][1:-2]) > 0


# Morrison-like moisture table (iice=1: qv qc qr qi qs qg + 4 numbers)
ICE = {
    **SUPER,
    "ptype": 5,
    "iice": 1,
    "numq": 10,
    "nqi": 4,
    "nqs1": 4,
    "nqs2": 6,
}


def make_cold_state(seed=3, g=G):
    """make_state with temperatures spanning the ice (< 233.15 K), mixed and
    liquid (> 273.15 K) branches of calcnm, and ice species."""
    st = make_state(seed, g)
    rng = np.random.default_rng(seed + 100)
    S = g.s_shape
    z = np.array(g.zh())[:, None, None] * np.ones(S)
    st["th0"] = 250.0 + 2.0e-3 * z
    st["th"] = rng.uniform(-10.0, 30.0, S)
    t = (st["th0"] + st["th"]) * (st["pi0"] + st["pp"])
    qv = rng.uniform(0.6, 1.05, S) * _qvs(t, st["prs"])
    q = [qv]
    for _ in range(5):
        q.append(np.where(rng.random(S) < 0.5, rng.uniform(0.0, 1e-3, S), 0.0))
    for _ in range(4):
        q.append(rng.uniform(0.0, 1e5, S))
    st["qa"] = np.stack(q)
    return st


def test_calcnm_ice(bk):
    """iice=1 (ice / mixed-phase / liquid saturated formulations)."""
    st = make_cold_state()
    ref = fortran(
        1,
        ICE,
        st,
        ["c1", "c2", "mf", "pi0", "th0", "prs", "pp", "th", "qa", "nm"],
        ["nm"],
    )
    out = T.calcnm(
        bk,
        G,
        ICE,
        qa=qlist(bk, st["qa"]),
        **As(bk, st, "c1", "c2", "mf", "pi0", "th0", "nm", "prs", "pp", "th"),
    )
    close(bk, out["nm"], ref["nm"], name="nm")
    t = (st["th0"] + st["th"]) * (st["pi0"] + st["pp"])
    ti = t[1:-1, 3:-3, 3:-3]
    assert ti.min() < 233.15 and ti.max() > 273.15


@pytest.mark.parametrize("case", list(CASES))
def test_calcdef(bk, case):
    cfg = CASES[case]
    st = make_state(2)
    ins = [
        *GRID1,
        "c1",
        "c2",
        "rho",
        "rr",
        "rf",
        "ua",
        "va",
        "wa",
        *TAU,
        "defv",
        "defh",
    ]
    ref = fortran(2, cfg, st, ins, [*TAU, "defv", "defh"])
    out = T.calcdef(
        bk,
        G,
        cfg,
        s11=A(bk, st["t11"]),
        s12=A(bk, st["t12"]),
        s13=A(bk, st["t13"]),
        s22=A(bk, st["t22"]),
        s23=A(bk, st["t23"]),
        s33=A(bk, st["t33"]),
        **As(bk, st, *GRID1, "c1", "c2", "rho", "rf", "ua", "va", "wa", "defv", "defh"),
    )
    for n in TAU:
        close(bk, out["s" + n[1:]], ref[n], name=n)
    close(bk, out["defv"], ref["defv"], name="defv")
    close(bk, out["defh"], ref["defh"], name="defh")


@pytest.mark.parametrize("case", list(CASES))
def test_tkekm(bk, case):
    cfg = CASES[case]
    st = make_state(3)
    ins = [
        "ruh",
        "rvh",
        "mf",
        "rmf",
        "rf",
        "nm",
        "defv",
        "defh",
        *KS,
        *CONST,
        "tkea",
        "lenscl",
        "dissten",
    ]
    ref = fortran(3, cfg, st, ins, [*KS, "tkea", "lenscl", "dissten"])
    out = T.tkekm(bk, G, cfg, **As(bk, st, *ins))
    for n in (*KS, "lenscl", "dissten"):
        close(bk, out[n], ref[n], name=n)
    np.testing.assert_array_equal(ref["tkea"], st["tkea"])


@pytest.mark.parametrize("case", list(CASES))
def test_tkebc(bk, case):
    cfg = CASES[case]
    st = make_state(4)
    ref = fortran(4, cfg, st, ["tkea", *KS], ["tkea", *KS])
    out = T.tkebc(bk, G, cfg, **As(bk, st, "tkea", *KS))
    for n in ("tkea", *KS):
        close(bk, out[n], ref[n], name=n)


@pytest.mark.parametrize("case", list(CASES))
def test_gettau(bk, case):
    cfg = CASES[case]
    st = make_state(5)
    ref = fortran(5, cfg, st, ["rf", "mf", *KS, "ua", "va", "wa", *TAU], TAU)
    out = T.gettau(bk, G, cfg, **As(bk, st, "kmh", "kmv", *TAU))
    for n in TAU:
        close(bk, out[n], ref[n], name=n)


@pytest.mark.parametrize("case", ["supercell", "periodic", "walls_explicit"])
@pytest.mark.parametrize("flag", [0, 1])
def test_bcw(bk, case, flag):
    cfg = CASES[case]
    st = make_state(6)
    ref = fortran(15, cfg, st, ["w"], ["w"], renames={"w": "wa"}, iarg=flag)
    out = T.bcw(bk, G, cfg, A(bk, st["wa"]), flag)
    close(bk, out, ref["w"], name="w")


@pytest.mark.parametrize("case", ["supercell", "periodic", "walls_explicit"])
def test_bc2d_level(bk, case):
    cfg = CASES[case]
    st = make_state(7)
    ref = fortran(16, cfg, st, ["w"], ["w"], renames={"w": "wa"}, iarg=1)
    out = T.bc2d_level(bk, G, cfg, A(bk, st["wa"]), 1)
    close(bk, out, ref["w"], name="w")


@pytest.mark.parametrize("rtime", [0.0, 600.0, 2400.0])
def test_ramp(bk, rtime):
    cfg = dict(SUPER, ramp_sgs=1, ramp_time=1800.0)
    st = make_state(8)
    ref = fortran(14, cfg, st, ["cm0", *CONST], CONST, rtime=rtime, ramp_time=1800.0)
    tf = T.turbfac_of(cfg, rtime)
    out = T.turb_constants(
        bk, G, cfg, cm0=A(bk, st["cm0"]), turbfac=tf, **As(bk, st, *CONST)
    )
    for n in CONST:
        close(bk, out[n], ref[n], name=n)


@pytest.mark.parametrize("case", list(CASES))
def test_sfc_and_turb(bk, case):
    cfg = dict(CASES[case], ramp_sgs=1, ramp_time=1800.0)
    st = make_state(9, moist=cfg["imoist"] == 1)
    rtime = 900.0
    ins = [
        *GRID1,
        "ruh",
        "rvh",
        "c1",
        "c2",
        "rmf",
        "pi0",
        "th0",
        "rho",
        "rr",
        "rf",
        "prs",
        "ua",
        "va",
        "wa",
        "ppi",
        "tha",
        "qa",
        "tkea",
        "tke3d",
        "cm0",
        *CONST,
        *KS,
        *TAU,
        "nm",
        "defv",
        "defh",
        "lenscl",
        "dissten",
    ]
    outs = [
        "tkea",
        "tke3d",
        *KS,
        *CONST,
        "nm",
        "defv",
        "defh",
        "lenscl",
        "dissten",
        *TAU,
    ]
    ref = fortran(
        18,
        cfg,
        st,
        ins,
        outs,
        renames={"ppi": "pp", "tha": "th"},
        rtime=rtime,
        ramp_time=1800.0,
    )
    names = [n for n in ins if n not in ("rr", "ppi", "tha", "qa")]
    out = T.sfc_and_turb(
        bk,
        G,
        cfg,
        rtime=rtime,
        ppi=A(bk, st["pp"]),
        tha=A(bk, st["th"]),
        qa=qlist(bk, st["qa"]),
        **As(bk, st, *names),
    )
    for n in outs:
        close(bk, out[n], ref[n], name=n)


# ---------------------------------------------------------------------------
# turbtend.F


@pytest.mark.parametrize("case", list(CASES))
def test_turbt(bk, case):
    cfg = CASES[case]
    st = make_state(10)
    ref = fortran(
        6,
        cfg,
        st,
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
        **As(bk, st, *GRID1, "rho", "rr", "rf", "kmh", "kmv"),
    )
    close(bk, out["tten"], ref["tten"], name="tten")


@pytest.mark.parametrize("case", list(CASES))
def test_turbu(bk, case):
    cfg = CASES[case]
    st = make_state(11)
    ref = fortran(
        7,
        cfg,
        st,
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
        **As(
            bk,
            st,
            "ruh",
            "uf",
            "vh",
            "mh",
            "mf",
            "rho",
            "rf",
            "uten",
            "t11",
            "t12",
            "t13",
            "kmv",
        ),
    )
    close(bk, out["uten"], ref["uten"], name="uten")


@pytest.mark.parametrize("case", list(CASES))
def test_turbv(bk, case):
    cfg = CASES[case]
    st = make_state(12)
    ref = fortran(
        8,
        cfg,
        st,
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
        **As(
            bk,
            st,
            "uh",
            "rvh",
            "vf",
            "mh",
            "mf",
            "rho",
            "rr",
            "rf",
            "vten",
            "t12",
            "t22",
            "t23",
            "kmv",
        ),
    )
    close(bk, out["vten"], ref["vten"], name="vten")


@pytest.mark.parametrize("case", list(CASES))
def test_turbw(bk, case):
    cfg = CASES[case]
    st = make_state(13)
    ref = fortran(
        9,
        cfg,
        st,
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
        **As(
            bk,
            st,
            "uh",
            "vh",
            "mh",
            "mf",
            "rho",
            "rf",
            "wten",
            "t13",
            "t23",
            "t33",
            "kmh",
        ),
    )
    close(bk, out["wten"], ref["wten"], name="wten")


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("iflux", [1, 0])
def test_turbs(bk, case, iflux):
    cfg = CASES[case]
    st = make_state(14)
    ref = fortran(
        10,
        cfg,
        st,
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
        **As(
            bk, st, *GRID1, "rho", "rr", "rf", "s", "sten", "khh", "khv", "dum7", "dum8"
        ),
    )
    close(bk, out["sten"], ref["sten"], name="sten")


# ---------------------------------------------------------------------------
# solve1 / solve2 inline code


@pytest.mark.parametrize("case", ["supercell", "mixed_tconfig2"])
def test_tke_source(bk, case):
    cfg = CASES[case]
    st = make_state(15)
    ins = ["nm", "defv", "defh", "dissten", *KS, "tketen"]
    ref = fortran(11, cfg, st, ins, ["tketen"])
    out = T.tke_source(
        bk,
        G,
        cfg,
        **As(bk, st, "nm", "defv", "defh", "dissten", "kmh", "kmv", "khv", "tketen"),
    )
    close(bk, out["tketen"], ref["tketen"], name="tketen")


@pytest.mark.parametrize("case,idiss", [("supercell", 1), ("supercell", 0), ("dry", 1)])
def test_dissheat(bk, case, idiss):
    cfg = dict(CASES[case], idiss=idiss)
    st = make_state(16, moist=cfg["imoist"] == 1)
    ref = fortran(
        12,
        cfg,
        st,
        ["pi0", "ppi", "qa", "dissten", "thten1"],
        ["thten1"],
        renames={"ppi": "pp"},
    )
    out = T.dissheat(
        bk,
        G,
        cfg,
        ppi=A(bk, st["pp"]),
        qa=qlist(bk, st["qa"]),
        **As(bk, st, "pi0", "dissten", "thten1"),
    )
    close(bk, out["thten1"], ref["thten1"], name="thten1")


def test_vimpl(bk):
    st = make_state(17)
    ref = fortran(
        13, SUPER, st, [*KS, "mf", "rf", "mh", "rr", "dum7", "dum8"], ["dum7", "dum8"]
    )
    out = T.vimpl_coefs(
        bk, G, SUPER, **As(bk, st, "khv", "mf", "rf", "mh", "rr", "dum7", "dum8")
    )
    close(bk, out["dum7"], ref["dum7"], name="dum7")
    close(bk, out["dum8"], ref["dum8"], name="dum8")


@pytest.mark.parametrize("case", ["supercell", "periodic"])
def test_tke_rk_update(bk, case):
    cfg = CASES[case]
    st = make_state(18)
    st["tten"] = np.random.default_rng(5).uniform(-0.6, 0.2, G.w_shape)
    ref = fortran(17, cfg, st, ["tkea", "tten", "tke3d"], ["tke3d"], dttmp=2.0)
    out = T.tke_rk_update(bk, G, cfg, dttmp=2.0, **As(bk, st, "tkea", "tten", "tke3d"))
    close(bk, out["tke3d"], ref["tke3d"], name="tke3d")
    assert np.any(ref["tke3d"][1:-2] == 0.0)


@pytest.mark.parametrize("case", ["supercell", "dry", "walls_explicit"])
def test_turb_tendencies(bk, case):
    """solve1's doturb block = turbu, turbv, turbw, vimpl, turbs(theta - th0r),
    turbs(q_n), each checked against the Fortran in sequence."""
    cfg = CASES[case]
    st = make_state(19, moist=cfg["imoist"] == 1)
    r = {}
    r["uten"] = fortran(
        7,
        cfg,
        st,
        [*GRID1, "ruh", "rmf", "rho", "rf", "u", "uten", "w", *TAU, *KS],
        ["uten"],
        renames={"u": "ua", "w": "wa"},
    )["uten"]
    r["vten"] = fortran(
        8,
        cfg,
        st,
        [*GRID1, "rvh", "rho", "rr", "rf", "v", "vten", "w", *TAU, *KS],
        ["vten"],
        renames={"v": "va", "w": "wa"},
    )["vten"]
    r["wten"] = fortran(
        9,
        cfg,
        st,
        [*GRID1, "rho", "rf", "w", "wten", *TAU, *KS],
        ["wten"],
        renames={"w": "wa"},
    )["wten"]
    vi = fortran(
        13, cfg, st, [*KS, "mf", "rf", "mh", "rr", "dum7", "dum8"], ["dum7", "dum8"]
    )
    st2 = dict(st, dum7=vi["dum7"], dum8=vi["dum8"])
    nk, z, n = G.nk, G.ngz, G.ngxy
    sadv = np.zeros(G.s_shape)
    sadv[z : z + nk] = (st["th0"][z : z + nk, n : n + 1, n : n + 1] - T.th0r) + st[
        "th"
    ][z : z + nk]
    ins = [*GRID1, "rho", "rr", "rf", "s", "sten", *KS, "dum7", "dum8"]
    r["thten1"] = fortran(
        10, cfg, dict(st2, s=sadv, sten=st["thten1"]), ins, ["sten"], iflux=1
    )["sten"]
    qref = []
    if cfg["imoist"]:
        for q in range(cfg["numq"]):
            qref.append(
                fortran(
                    10,
                    cfg,
                    dict(st2, s=st["qa"][q], sten=st["qten"][q]),
                    ins,
                    ["sten"],
                    iflux=int(q + 1 == cfg["nqv"]),
                )["sten"]
            )
    out = T.turb_tendencies(
        bk,
        G,
        cfg,
        dt=DT,
        tha=A(bk, st["th"]),
        qa=qlist(bk, st["qa"]),
        qten=qlist(bk, st["qten"]),
        uten1=A(bk, st["uten"]),
        vten1=A(bk, st["vten"]),
        wten1=A(bk, st["wten"]),
        thten1=A(bk, st["thten1"]),
        **As(bk, st, *GRID1, "rho", "rr", "rf", "th0", "ua", "va", "wa", *KS, *TAU),
    )
    for a, b in (
        ("uten1", "uten"),
        ("vten1", "vten"),
        ("wten1", "wten"),
        ("thten1", "thten1"),
    ):
        close(bk, out[a], r[b], name=a)
    for q, ref in enumerate(qref):
        close(bk, out["qten"][q], ref, name=f"qten{q}")


def test_tke_copy(bk):
    st = make_state(20)
    out = T.tke_copy(bk, G, SUPER, **As(bk, st, "tke3d", "tkea"))
    ref = st["tkea"].copy()
    ref[1 : G.nk + 1, 2 : G.nj + 4, 2 : G.ni + 4] = st["tke3d"][
        1 : G.nk + 1, 2 : G.nj + 4, 2 : G.ni + 4
    ]
    close(bk, out["tkea"], ref, rtol=0, name="tkea")


# ---------------------------------------------------------------------------
# jax.jit


def test_jax_jit():
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    cfg = dict(SUPER, ramp_sgs=1, ramp_time=1800.0)
    st = make_state(9)
    names = [
        *GRID1,
        "ruh",
        "rvh",
        "c1",
        "c2",
        "rmf",
        "pi0",
        "th0",
        "rho",
        "rf",
        "prs",
        "ua",
        "va",
        "wa",
        "tkea",
        "tke3d",
        "cm0",
        *CONST,
    ]

    @jax.jit
    def step(d, qa, pp, th):
        o = T.sfc_and_turb(bk, G, cfg, rtime=900.0, ppi=pp, tha=th, qa=qa, **d)
        src = T.tke_source(bk, G, cfg, **o)["tketen"]
        return o["kmh"], o["t13"], src

    d = As(bk, st, *names)
    kmh, t13, src = step(d, qlist(bk, st["qa"]), A(bk, st["pp"]), A(bk, st["th"]))
    o = T.sfc_and_turb(
        _np_bk(),
        G,
        cfg,
        rtime=900.0,
        ppi=st["pp"],
        tha=st["th"],
        qa=list(st["qa"]),
        **{n: st[n] for n in names},
    )
    s2 = T.tke_source(_np_bk(), G, cfg, **o)["tketen"]
    close(bk, kmh, o["kmh"], name="kmh")
    close(bk, t13, o["t13"], name="t13")
    close(bk, src, s2, name="tketen")
