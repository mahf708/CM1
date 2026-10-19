"""pbl_gfs.bl_gfsedmf (the GFS-EDMF PBL, ipbl=3) against BL_GFSEDMF of
src/module_bl_gfsedmf.F on columns that cover its branches: stable,
weakly unstable (counter-gradient) and convective (mass-flux) boundary
layers, stratocumulus with cloud-top cooling, land and water (the
alpha < 0 wind cap), with and without cloud ice, and the alpha = 1 /
constant-Ric variants."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver
from libm_numpy import libm_backend

from cm1x import constants as C
from cm1x.pbl_gfs import UndefinedWspm, bl_gfsedmf

pytestmark = requires_gfortran

NI, NJ, NK = 9, 7, 24
DT = 6.0
# name: (p_qi, alpha, var_ric, disheat)
CASES = {
    "hwrf": (0, -1.0, 1.0, 1),
    "qi": (1, -1.0, 1.0, 1),
    "alpha1": (0, 1.0, 1.0, 1),
    "const_ric": (1, 0.5, 0.0, 1),
    "nodisheat": (0, -1.0, 1.0, 0),
}
OUT3 = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten")
OUT3 += ("dku3d", "dkt3d", "dissten", "dheat")
OUT2 = ("ust", "wspd", "hpbl", "kpbl2d")


def make_inputs(seed=7):
    rng = np.random.default_rng(seed)
    s2, s3 = (NJ, NI), (NK, NJ, NI)
    # stretched w levels (no terrain: zf(1) = 0)
    dzs = (
        60.0 * 1.12 ** np.arange(NK)[:, None, None] * rng.uniform(0.9, 1.1, (NK, 1, 1))
    )
    zf = np.concatenate([np.zeros((1, NJ, NI)), np.cumsum(dzs + 0 * np.ones(s3), 0)])
    zh = 0.5 * (zf[1:] + zf[:-1])
    dz8w = zf[1:] - zf[:-1]
    # theta: a mixed layer below h, stable above, a superadiabatic layer
    h = rng.uniform(300.0, 1500.0, s2)
    th_s = rng.uniform(295.0, 302.0, s2)
    lapse = rng.uniform(3e-3, 8e-3, s2)
    th = th_s + np.where(zh > h, lapse * (zh - h), 0.0)
    th = th - rng.uniform(-1.5, 2.0, s2) * np.exp(-zh / 50.0)
    th = th + rng.normal(0.0, 0.05, s3)
    qv = rng.uniform(0.006, 0.016, s2) * np.exp(-zh / 2500.0)
    # hydrostatic pressure on the scalar levels
    psfc = rng.uniform(98000.0, 101500.0, s2)
    pi_s = (psfc / C.p00) ** C.rovcp
    dzz = np.diff(np.concatenate([np.zeros((1, NJ, NI)), zh]), axis=0)
    pi = pi_s - np.cumsum(C.g / (C.cp * th) * dzz, 0)
    p = C.p00 * pi ** (C.cp / C.rd)
    t = th * pi
    # stratocumulus: cloud below the inversion of half the columns
    cloudy = rng.random(s2) < 0.5
    qc = np.where(cloudy & (zh < h) & (zh > 0.6 * h), rng.uniform(1e-4, 6e-4, s3), 0.0)
    qc = qc + np.where(rng.random(s3) < 0.1, 1e-6, 0.0)
    qi = np.where(rng.random(s3) < 0.3, rng.uniform(0.0, 5e-5, s3), 0.0)
    rth = np.where(
        cloudy & (zh < h) & (zh > 0.8 * h), -rng.uniform(1e-4, 8e-4, s3), 0.0
    )
    rth = rth + rng.normal(0.0, 1e-6, s3)
    u = rng.uniform(-3.0, 12.0, s2) + 4e-3 * zh * rng.uniform(-1.0, 1.0, s2)
    v = rng.uniform(-6.0, 6.0, s2) + 2e-3 * zh * rng.uniform(-1.0, 1.0, s2)
    u = u + rng.normal(0.0, 0.3, s3)
    v = v + rng.normal(0.0, 0.3, s3)
    xland = np.where(rng.random(s2) < 0.5, 2.0, 1.0)
    znt = np.where(
        xland == 2,
        10.0 ** rng.uniform(-4.5, -3.0, s2),
        10.0 ** rng.uniform(-2.5, -0.7, s2),
    )
    za = zh[0]
    gz1oz0 = np.log(za / znt)
    # surface layer: stable / weakly unstable / convective
    regime = rng.integers(0, 3, s2)
    br = np.choose(
        regime,
        [
            rng.uniform(0.01, 0.4, s2),
            rng.uniform(-0.06, -0.005, s2),
            rng.uniform(-3.0, -0.12, s2),
        ],
    )
    hfx = np.choose(
        regime,
        [
            rng.uniform(-60.0, -2.0, s2),
            rng.uniform(5.0, 80.0, s2),
            rng.uniform(80.0, 450.0, s2),
        ],
    )
    qfx = rng.uniform(0.0, 2.5e-4, s2)
    psim = np.where(br > 0, -rng.uniform(0.1, 3.0, s2), rng.uniform(0.1, 2.0, s2))
    psih = np.where(br > 0, -rng.uniform(0.1, 3.0, s2), rng.uniform(0.1, 2.5, s2))
    tsk = t[0] + np.where(br > 0, -rng.uniform(0.2, 3.0, s2), rng.uniform(0.2, 4.0, s2))
    wspd = np.sqrt(u[0] ** 2 + v[0] ** 2) + 1e-9
    fac = np.log(10.0 / znt) / gz1oz0
    u10, v10 = u[0] * fac, v[0] * fac
    return {
        "u3d": u,
        "v3d": v,
        "t3d": t,
        "qv3d": qv,
        "qc3d": qc,
        "qi3d": qi,
        "p3d": p,
        "pi3d": pi,
        "dz8w": dz8w,
        "z": zf[:NK],
        "rthraten": rth,
        "psfc": psfc,
        "psim": psim,
        "psih": psih,
        "hfx": hfx,
        "qfx": qfx,
        "tsk": tsk,
        "gz1oz0": gz1oz0,
        "br": br,
        "u10": u10,
        "v10": v10,
        "znt": znt,
        "xland": xland,
        "wspd": wspd,
    }


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    p_qi, alpha, var_ric, disheat = CASES[request.param]
    ins = make_inputs()
    params = [NI, NJ, NK, DT, p_qi, alpha, var_ric, 0.16, 0.25, disheat]
    outs = {k: (NK, NJ, NI) for k in OUT3}
    outs.update({k: (NJ, NI) for k in OUT2})
    ref = run_driver("pbl_gfs", {"params": np.array(params, float), **ins}, outs)
    opts = {
        "qi": p_qi >= 1,
        "alpha": alpha,
        "var_ric": var_ric,
        "disheat": bool(disheat),
    }
    return request.param, ins, opts, ref


def _cmp(name, got, want, tol):
    scale = max(np.max(np.abs(want)), 1e-300)
    err = np.max(np.abs(got - want)) / scale
    assert err <= tol, f"{name}: err {err:.2e} (max |ref| {scale:.3e})"


def test_bl_gfsedmf(bk, case):
    _, ins, opts, ref = case
    r = bl_gfsedmf(
        bk, dt=DT, ust=None, **{k: bk.asarray(v) for k, v in ins.items()}, **opts
    )
    tol = 1e-12
    for k in OUT3:
        want = ref[k]
        if k == "rqiblten" and not opts["qi"]:
            continue
        if k in ("dku3d", "dkt3d"):
            got = bk.to_numpy(r[k])
            _cmp(k, got, want[1:], tol)
            continue
        if k == "dissten" and not opts["disheat"]:
            continue  # undefined in the Fortran
        _cmp(k, bk.to_numpy(r[k]), want, tol)
    for k in OUT2:
        _cmp(k, bk.to_numpy(r[k]).astype(float), ref[k], tol)


def test_bl_gfsedmf_bitwise(case):
    """With glibc's exp/log/pow the port is bit-for-bit."""
    _, ins, opts, ref = case
    bk = libm_backend()
    r = bl_gfsedmf(bk, dt=DT, ust=None, **ins, **opts)
    for k in OUT3 + OUT2:
        if r[k] is None or (k == "dissten" and not opts["disheat"]):
            continue
        want = ref[k][1:] if k in ("dku3d", "dkt3d") else ref[k]
        nd = int(np.sum(np.asarray(r[k], float) != want))
        assert nd == 0, f"{k}: {nd} values differ"


def test_regimes_covered():
    """The inputs reach the convective, counter-gradient and
    stratocumulus branches."""
    from cm1x import get_backend

    bk = get_backend("numpy")
    ins = make_inputs()
    r = bl_gfsedmf(bk, dt=DT, ust=None, **ins)
    for k, f in r["flags"].items():
        n = int(np.sum(f))
        assert 0 < n < f.size, f"{k}: {n} of {f.size} columns"


def test_undefined_wspm():
    from cm1x import get_backend

    bk = get_backend("numpy")
    ins = make_inputs()
    ins["z"] = ins["z"] * 0.05  # no level brackets 500 m below nk/2
    ins["dz8w"] = ins["dz8w"] * 0.05
    ins["xland"] = ins["xland"] * 0 + 2.0
    with pytest.raises(UndefinedWspm):
        bl_gfsedmf(bk, dt=DT, ust=None, **ins)


def test_bl_gfsedmf_grad():
    """Reverse-mode gradients of the tendencies are finite."""
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax")
    ins = {k: bk.asarray(v) for k, v in make_inputs().items()}

    def f(t, u):
        r = bl_gfsedmf(bk, dt=DT, ust=None, **{**ins, "t3d": t, "u3d": u})
        return bk.xp.sum(r["rthblten"] ** 2) + bk.xp.sum(r["rublten"] ** 2)

    g = jax.grad(f, argnums=(0, 1))(ins["t3d"], ins["u3d"])
    assert all(bool(bk.xp.all(bk.xp.isfinite(a))) for a in g)
