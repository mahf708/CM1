"""Morrison two-moment microphysics (src/morrison.F, ptype=5) and the ptype=5
branch of mp_driver.F against the original Fortran (double-precision CM1
objects).

GRAUPEL_INIT, GAMMA, DERF1, satadj2 and MP_GRAUPEL (with M2005MICRO_GRAUPEL)
are private to module_mp_GRAUPEL; drv_morrison links the module's public
variant (tests/fortran.py ``public_module``), so the scheme's own routines
are called directly, including the INUM=0 (predicted droplet number) path
that CM1 hard-wires off.  drv_morrison_mp runs CM1's real mp_driver."""

import numpy as np
import pytest

from cm1x import morrison as mo
from cm1x.grid import Grid

from fortran import requires_gfortran, run_driver

pytestmark = requires_gfortran

PUB = ("morrison",)
G = Grid(ni=6, nj=5, nk=26, dx=1000.0, dy=1000.0, dz=150.0)
DT = 20.0
RTOL = 1e-12

CONST_NAMES = (
    "ai ac as ar bi bc bs br ag bg r rv cp ep_2 cv cvv cpv cpl cpi lv1 lv2 "
    "ls1 ls2 tmelt rhosu rhow rhoi rhosn rhog aimm bimm ecr dcs mi0 mg0 "
    "f1s f2s f1r f2r g qsmall qthr eii eci cpw ci di cs ds cg dg rin mmult "
    "lammaxi lammini lammaxr lamminr d_b lammaxs lammins lammaxg lamming "
    "k1 c1 mw osm vi epsm rhoa map ma rr bact rm1 sig1 nanew1 f11 f21 rm2 "
    "sig2 nanew2 f12 f22"
).split() + [f"cons{n}" for n in range(1, 42)]


def close(bk, got, ref, rtol=RTOL, name=""):
    got = got if isinstance(got, np.ndarray) else bk.to_numpy(got)
    ref = np.asarray(ref)
    atol = rtol * np.max(np.abs(ref))
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=name)


def A(bk, x):
    return bk.asarray(np.asarray(x))


def _qvs(t, p):
    e = np.minimum(0.5 * p, 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65)))
    return 287.04 / 461.5 * e / (p - e)


def make_state(seed=0, g=G, inum=1):
    """Columns from a warm moist boundary layer to below 220 K, with every
    hydrometeor present at random (also tiny amounts, zero numbers, small
    snow/graupel above freezing), one dry column, and updrafts."""
    rng = np.random.default_rng(seed)
    sh = g.s_shape
    z = np.array(g.zh())[:, None, None] * np.ones(sh)
    t = 298.0 - 6.5e-3 * z * 4.0 + rng.uniform(-1.5, 1.5, sh)
    p = 1.0e5 * np.exp(-z * 4.0 / 8000.0)
    rho = p / (287.04 * t)
    rh = rng.uniform(0.5, 1.12, sh)
    qv = rh * _qvs(t, p)
    cold = t < 273.15

    def amt(frac, hi, tiny=0.0):
        a = np.where(rng.random(sh) < frac, rng.uniform(0.0, hi, sh), 0.0)
        if tiny:
            a = np.where(rng.random(sh) < 0.08, tiny, a)
        return a

    qc = amt(0.5, 2.0e-3, 1.0e-12)
    qr = amt(0.5, 3.0e-3, 5.0e-9)
    qi = np.where(cold, amt(0.5, 5.0e-4, 1.0e-12), amt(0.1, 1.0e-6))
    qs = np.where(cold, amt(0.5, 2.0e-3, 2.0e-9), amt(0.3, 2.0e-6))
    qg = np.where(cold, amt(0.4, 4.0e-3, 2.0e-9), amt(0.3, 3.0e-6))

    def num(q, m):
        n = q / m * np.exp(rng.uniform(-3.0, 3.0, sh))
        return np.where(rng.random(sh) < 0.05, 0.0, n)

    ni = num(qi, 1.0e-11)
    ns = num(qs, 1.0e-9)
    nr = num(qr, 1.0e-7)
    ng = num(qg, 1.0e-7)
    nc = num(qc, 1.0e-11) if inum == 0 else np.zeros(sh)
    # one dry column (no hydrometeors, subsaturated) and one ice-free column
    for a in (qc, qr, qi, qs, qg, ni, ns, nr, ng, nc):
        a[:, 4, 4] = 0.0
    qv[:, 4, 4] = 0.5 * _qvs(t[:, 4, 4], p[:, 4, 4])
    w = rng.uniform(-2.0, 6.0, g.w_shape)
    rmh = 1.0 + 0.05 * rng.standard_normal(sh)
    return dict(
        t3d=t,
        qv=qv,
        qc=qc,
        qr=qr,
        qi=qi,
        qs=qs,
        qg=qg,
        nni=ni,
        ns=ns,
        nr=nr,
        ng=ng,
        nc=nc,
        p=p,
        rho=rho,
        ddz=g.dz * rmh,
        w=w,
        rmh=rmh,
        rr=rho * (1.0 + 0.01 * rng.standard_normal(sh)),
        rainnc=rng.uniform(0.0, 1.0, (2, sh[1], sh[2])),
        prate=np.zeros(sh[1:]),
        ruh=1.0 + 0.05 * rng.standard_normal(sh[2]),
        rvh=1.0 + 0.05 * rng.standard_normal(sh[1]),
    )


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ihail,ndcnst", [(0, 250.0), (1, 100.0)])
def test_graupel_init(ihail, ndcnst):
    ref = run_driver(
        "morrison",
        dict(params=np.array([1, ihail, ndcnst, 1.0])),
        dict(consts=(len(CONST_NAMES),)),
        public=PUB,
    )["consts"]
    M = mo.graupel_init(ihail, ndcnst)
    got = np.array([M[n] for n in CONST_NAMES])
    bad = [n for n, a, b in zip(CONST_NAMES, got, ref) if a != b]
    assert not bad, f"constants differ: {bad}"


def test_gamma_derf1(bk):
    rng = np.random.default_rng(3)
    x = np.concatenate([np.linspace(1.0, 35.0, 3401), rng.uniform(3.0, 16.0, 600)])
    ref = run_driver(
        "morrison",
        dict(params=np.array([2, x.size]), x=x),
        dict(gam=(x.size,), erf=(x.size,)),
        public=PUB,
    )
    close(bk, mo.gamma(bk, A(bk, x)), ref["gam"], name="gamma")
    assert all(mo.gamma_scalar(v) == r for v, r in zip(x[::7], ref["gam"][::7]))
    xe = np.concatenate([np.linspace(-7.0, 7.0, 3401), rng.uniform(-3.0, 3.0, 600)])
    ref = run_driver(
        "morrison",
        dict(params=np.array([2, xe.size]), x=xe),
        dict(gam=(xe.size,), erf=(xe.size,)),
        public=PUB,
    )
    np.testing.assert_allclose(
        bk.to_numpy(mo.derf1(bk, A(bk, xe))), ref["erf"], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("eqtset", [2, 1])
def test_satadj2(bk, eqtset):
    rng = np.random.default_rng(5)
    n = 4000
    t = rng.uniform(230.0, 305.0, n)
    p = rng.uniform(2.0e4, 1.0e5, n)
    rho = p / (287.04 * t)
    qv = rng.uniform(0.8, 1.15, n) * _qvs(t, p)
    qc = np.where(rng.random(n) < 0.5, rng.uniform(0.0, 2e-3, n), 0.0)
    qr, qi, qs, qg = (rng.uniform(0.0, 1e-3, n) for _ in range(4))
    ins = dict(t=t, p=p, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, rho=rho)
    ref = run_driver(
        "morrison",
        dict(params=np.array([3, n, eqtset, 1.0e-4]), **ins),
        dict(t=(n,), p=(n,), qv=(n,), qc=(n,)),
        public=PUB,
    )
    M = mo.graupel_init()
    o = mo.satadj2(
        bk,
        M,
        tnew=A(bk, t),
        prs=A(bk, p),
        qvnew=A(bk, qv),
        qcnew=A(bk, qc),
        qrnew=A(bk, qr),
        qinew=A(bk, qi),
        qsnew=A(bk, qs),
        qgnew=A(bk, qg),
        rho=A(bk, rho),
        eqtset=eqtset,
        tsmall=1.0e-4,
        maxiter=100,
    )
    for k, r in (("tnew", "t"), ("prs", "p"), ("qvnew", "qv"), ("qcnew", "qc")):
        close(bk, o[k], ref[r], name=k)


CASES = {
    "default": dict(ihail=0, ndcnst=250.0, inum=1, eqtset=2, testcase=0),
    "hail": dict(ihail=1, ndcnst=100.0, inum=1, eqtset=2, testcase=0),
    "inum0": dict(ihail=0, ndcnst=250.0, inum=0, eqtset=2, testcase=0),
    "eqtset1": dict(ihail=0, ndcnst=250.0, inum=1, eqtset=1, testcase=0),
    "testcase5": dict(ihail=0, ndcnst=250.0, inum=1, eqtset=2, testcase=5),
}
OUT3 = ("t3d qv qc qr qi qs qg nni ns nr ng nc").split()


def run_mp_graupel_ref(case, s, g=G, dt=DT):
    c = CASES[case]
    params = np.array(
        [
            4,
            g.ni,
            g.nj,
            g.nk,
            g.dx,
            g.dy,
            g.dz,
            dt,
            c["ihail"],
            c["ndcnst"],
            c["inum"],
            c["eqtset"],
            1.0e-4,
            c["testcase"],
            s["rainnc"].shape[0],
        ]
    )
    bud = np.array([1.0, 2.0, 3.0, 4.0])
    ins = dict(params=params, bud=bud, **s)
    outs = {k: g.s_shape for k in OUT3}
    outs.update({k: g.s_shape for k in ("vtc", "vtr", "vti", "vts", "vtg")})
    outs.update(rainnc=s["rainnc"].shape, prate=s["prate"].shape, bud=(4,))
    return run_driver("morrison", ins, outs, public=PUB)


def run_mp_graupel(bk, case, s, g=G, dt=DT, jit=False):
    c = CASES[case]
    M = mo.graupel_init(c["ihail"], c["ndcnst"], c["inum"])
    cfg = dict(eqtset=c["eqtset"], tsmall=1.0e-4, testcase=c["testcase"])
    kw = {k: A(bk, v) for k, v in s.items() if k not in ("rmh",)}
    kw["rmh"] = A(bk, s["rmh"])
    bud = dict(tcond=1.0, tevac=2.0, tevar=3.0, train=4.0)

    def f(kw):
        return mo.mp_graupel(bk, g, cfg, M, dt=dt, **kw, **bud, nfall_max=None)

    if jit:
        import jax

        return jax.jit(f)(kw)
    return f(kw)


@pytest.mark.parametrize("case", list(CASES))
def test_mp_graupel(bk, case):
    s = make_state(7, inum=CASES[case]["inum"])
    ref = run_mp_graupel_ref(case, s)
    o = run_mp_graupel(bk, case, s)
    for k in OUT3 + ["rainnc", "prate"]:
        close(bk, o[k], ref[k], name=k)
    S = (G.K, G.J, G.I)
    for k in ("vtc", "vtr", "vti", "vts", "vtg"):
        close(bk, o[k], ref[k][S], name=k)
    got = [float(bk.to_numpy(o[k])) for k in ("tcond", "tevac", "tevar", "train")]
    np.testing.assert_allclose(got, ref["bud"], rtol=RTOL)


# ---------------------------------------------------------------------------
# mp_driver, ptype=5 (CM1's own mp_driver: pdefqtest, MP_GRAUPEL, efall, the
# theta/pressure update)

MP_CASES = {
    "default": dict(eqtset=2, efall=0, numq=10, ihail=0, ndcnst=250.0, testcase=0),
    "efall": dict(eqtset=2, efall=1, numq=10, ihail=0, ndcnst=250.0, testcase=0),
    "eqtset1": dict(eqtset=1, efall=0, numq=10, ihail=1, ndcnst=100.0, testcase=0),
    "inum0": dict(eqtset=2, efall=1, numq=11, ihail=0, ndcnst=250.0, testcase=0),
}


def make_mp_state(seed, numq, g=G):
    from cm1x import constants as C

    s = make_state(seed, g, inum=0 if numq == 11 else 1)
    rng = np.random.default_rng(seed + 100)
    sh = g.s_shape
    t, prs, qv = s["t3d"], s["p"], s["qv"]
    pi0 = (prs / C.p00) ** C.rovcp * (1.0 + 1e-4 * rng.standard_normal(sh))
    pp3d = (prs / C.p00) ** C.rovcp - pi0
    th = t / (pi0 + pp3d)
    th0 = th - rng.uniform(-1.0, 1.0, sh)
    th3d = th - th0
    rho = prs / (C.rd * t * (1.0 + qv * C.reps))
    names = ["qv", "qc", "qr", "qi", "qs", "qg", "nni", "ns", "nr", "ng"]
    if numq == 11:
        names.append("nc")
    q3d = np.stack([s[n] for n in names])
    # some negative values for pdefqtest
    q3d[1:6] = np.where(rng.random(q3d[1:6].shape) < 0.03, -1e-7, q3d[1:6])
    return dict(
        th0=th0,
        pi0=pi0,
        th3d=th3d,
        pp3d=pp3d,
        prs=prs,
        rho=rho,
        q3d=q3d,
        w3d=s["w"],
        ruh=s["ruh"],
        rvh=s["rvh"],
        rmh=s["rmh"],
        mh=np.ones(sh),
        zh=np.zeros(sh),
        mf=1.0 + 0.05 * rng.standard_normal(g.w_shape),
        rain=s["rainnc"],
        prate=s["prate"],
        qbudget=rng.uniform(0.0, 1.0, 10),
        asq=rng.uniform(0.0, 1.0, numq),
    )


@pytest.mark.parametrize("case", list(MP_CASES))
def test_mp_driver_ptype5(bk, case):
    from cm1x.mp_driver import mp_driver
    from cm1x.namelist import Config

    c = MP_CASES[case]
    numq = c["numq"]
    s = make_mp_state(11, numq)
    params = np.array(
        [
            G.ni,
            G.nj,
            G.nk,
            G.dx,
            G.dy,
            G.dz,
            DT,
            c["eqtset"],
            1.0e-4,
            1.0e-14,
            s["rain"].shape[0],
            c["ihail"],
            c["ndcnst"],
            c["efall"],
            numq,
            c["testcase"],
        ]
    )
    ins = dict(params=params, **s)
    sh = G.s_shape
    outs = dict(
        rho=sh,
        prs=sh,
        pp3d=sh,
        th3d=sh,
        q3d=(numq,) + sh,
        qten=(numq,) + sh,
        thten=sh,
        dum1=sh,
        dum3=sh,
        rain=s["rain"].shape,
        prate=s["prate"].shape,
        qbudget=(10,),
        asq=(numq,),
    )
    ref = run_driver("morrison_mp", ins, outs, public=PUB)
    cfg = Config(
        dict(
            ptype=5,
            imoist=1,
            inum=0 if numq == 11 else 1,
            eqtset=c["eqtset"],
            efall=c["efall"],
            ihail=c["ihail"],
            ndcnst=c["ndcnst"],
            testcase=c["testcase"],
            pdscheme=1,
            tsmall=1.0e-4,
            qsmall=1.0e-14,
            dx=G.dx,
            dy=G.dy,
            dz=G.dz,
        )
    )
    assert cfg.numq == numq
    kw = {k: A(bk, v) for k, v in s.items()}
    o = mp_driver(bk, G, cfg, dt=DT, satadj_maxiter=100, **kw)
    S = (G.K, G.J, G.I)
    for k in ("rho", "prs", "pp3d", "th3d", "thten", "dum1", "dum3", "rain", "prate"):
        close(bk, o[k], ref[k], name=k)
    got_q, ref_q = bk.to_numpy(o["q3d"]), ref["q3d"]
    for n in range(numq):
        close(bk, got_q[n], ref_q[n], name=f"q3d[{n}]")
    got_t = bk.to_numpy(o["qten"])
    for n in (0, 1, 2, 3, 4, 5) if c["efall"] else (0,):  # qv, fall speeds
        close(bk, got_t[n][S], ref["qten"][n][S], name=f"qten[{n}]")
    np.testing.assert_allclose(bk.to_numpy(o["qbudget"]), ref["qbudget"], rtol=RTOL)
    np.testing.assert_allclose(bk.to_numpy(o["asq"]), ref["asq"], rtol=RTOL)
