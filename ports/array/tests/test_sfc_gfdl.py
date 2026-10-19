"""sfc_gfdl.sf_gfdl (the GFDL surface layer, sfcmodel=4) against SF_GFDL of
src/module_sf_gfdl.F: stable and unstable points over water (every branch
of the icoef_sf=6 roughness fits, calm and first-step 10-m winds) and over
land (the slab T* prediction, the soil-moisture availability, the flux
caps), plus the ntsflg=0 / isfflx=0 / sea-spray variants."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver
from libm_numpy import libm_backend

from cm1x import constants as C
from cm1x.sfc_gfdl import fpvs, gpvs, sf_gfdl

pytestmark = requires_gfortran

NI, NJ = 11, 9
DT = 6.0
# name: (ntsflg, sfenth, isfflx, isltyp)
CASES = {
    "hwrf": (1, 0.0, 1, 14),
    "loam": (1, 0.0, 1, 6),
    "nots": (0, 0.5, 1, 14),
    "noflx": (1, 0.0, 0, 3),
}
OUTS = [
    "tsk",
    "znt",
    "mznt",
    "ust",
    "u10",
    "v10",
    "chs",
    "chs2",
    "cqs2",
    "cpm",
    "psim",
    "psih",
    "hfx",
    "qfx",
    "lh",
    "taux",
    "tauy",
    "flhc",
    "flqc",
    "qgh",
    "qsfc",
    "gz1oz0",
    "wspd",
    "br",
    "cd_out",
    "ch_out",
    "mol",
    "zol",
    "z0t",
    "fm",
    "fh",
]


def make_inputs(seed=11):
    rng = np.random.default_rng(seed)
    s = (NJ, NI)
    xland = np.where(rng.random(s) < 0.55, 2.0, 1.0)
    spd = np.concatenate(
        [
            rng.uniform(0.0, 3.0, 20),
            rng.uniform(3.0, 20.0, 40),
            rng.uniform(20.0, 75.0, 39),
        ]
    )
    rng.shuffle(spd)
    spd = spd.reshape(s)
    spd[0, 3] = 0.0  # calm point
    ang = rng.uniform(0.0, 2 * np.pi, s)
    u, v = spd * np.cos(ang), spd * np.sin(ang)
    za = rng.uniform(8.0, 60.0, s)
    psfc = rng.uniform(95000.0, 102000.0, s)
    t = rng.uniform(270.0, 305.0, s)
    p = psfc * np.exp(-C.g * za / (C.rd * t))
    qv = rng.uniform(0.002, 0.02, s)
    tsk = t + rng.uniform(-6.0, 6.0, s)
    znt = np.where(
        xland == 2, 10.0 ** rng.uniform(-5.0, -3.0, s), 10.0 ** rng.uniform(-3, -0.5, s)
    )
    w10 = spd * rng.uniform(0.6, 0.95, s)
    w10 = np.where(rng.random(s) < 0.25, 0.0, w10)  # first step: log profile
    return {
        "u3d": u,
        "v3d": v,
        "t3d": t,
        "qv3d": qv,
        "p3d": p,
        "psfc": psfc,
        "xland": xland,
        "gsw": rng.uniform(0.0, 800.0, s),
        "glw": rng.uniform(250.0, 420.0, s),
        "tsk": tsk,
        "znt": znt,
        "u10": w10 * np.cos(ang),
        "v10": w10 * np.sin(ang),
        "zkmax": za,
        "smois": rng.uniform(0.05, 0.45, s),
    }


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    ntsflg, sfenth, isfflx, isltyp = CASES[request.param]
    ins = make_inputs()
    params = [NI, NJ, DT, ntsflg, sfenth, 6, isfflx, isltyp]
    ref = run_driver(
        "sfc_gfdl",
        {"params": np.array(params, float), **ins},
        {k: (NJ, NI) for k in OUTS},
    )
    opts = {"ntsflg": ntsflg, "sfenth": sfenth, "isfflx": isfflx, "isltyp": isltyp}
    return ins, opts, ref


def _run(bk, ins, opts):
    return sf_gfdl(bk, dt=DT, **{k: bk.asarray(v) for k, v in ins.items()}, **opts)


def test_sf_gfdl(bk, case):
    ins, opts, ref = case
    r = _run(bk, ins, opts)
    for k in OUTS:
        got, want = bk.to_numpy(r[k]), ref[k]
        # relative to the field's largest value (psim, psih cancel near 0)
        err = np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300)
        assert err <= 1e-12, f"{k}: err {err:.2e}"


def test_sf_gfdl_bitwise(case):
    """With glibc's exp/log/pow/atan the port is bit-for-bit."""
    ins, opts, ref = case
    r = _run(libm_backend(), ins, opts)
    for k in OUTS:
        nd = int(np.sum(np.asarray(r[k]) != ref[k]))
        assert nd == 0, f"{k}: {nd} values differ"


def test_fpvs_table(bk):
    c1, c2, _ = gpvs()
    t = bk.asarray(np.linspace(175.0, 335.0, 1001))
    e = bk.to_numpy(fpvs(bk, t))
    assert np.all(np.diff(e) >= 0.0) and abs(c1 + c2 * 180.0 - 1.0) < 1e-12
    assert 600.0 < np.interp(273.16, np.linspace(175.0, 335.0, 1001), e) < 620.0


def test_sf_gfdl_grad():
    """Reverse-mode gradients are finite away from calm points."""
    jax = pytest.importorskip("jax")
    from cm1x import get_backend

    bk = get_backend("jax")
    raw = make_inputs()
    raw["u3d"][0, 3] = 0.3
    ins = {k: bk.asarray(v) for k, v in raw.items()}

    def f(t, q):
        r = sf_gfdl(bk, dt=DT, **{**ins, "t3d": t, "qv3d": q})
        return bk.xp.sum(r["hfx"] + r["qfx"] * 2.5e6 + r["ust"] + r["tsk"])

    g = jax.grad(f, argnums=(0, 1))(ins["t3d"], ins["qv3d"])
    assert all(bool(bk.xp.all(bk.xp.isfinite(a))) for a in g)
