"""pbl_ysu.bl_ysu_run (the YSU PBL, ipbl=1) against src/bl_ysu.F called as
turb.F calls it (ysu_wrapper, one row of columns at a time), on columns
covering the scheme's branches: convective boundary layers over land and
water (counter-gradient and entrainment fluxes, with and without the
thermal enhancing the PBL height), stable layers (land: brcr_sb, water:
the Rossby-number brcr), near-neutral and windless columns, and cloudy
layers (the moist Richardson number)."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x import constants as C
from cm1x.pbl_ysu import bl_ysu_run

pytestmark = requires_gfortran

NI, NJ, NK = 9, 6, 32
DT = 6.0
F3 = ("ux", "vx", "tx", "qvx", "qcx", "qix", "p2d", "pi2d", "dz8w", "rthraten")
F2 = ("psfc", "znt", "ust", "psim", "psih", "xland", "hfx", "qfx", "wspd", "br")
OUT3 = ("utnp", "vtnp", "ttnp", "qvtnp", "qctnp", "qitnp", "exch_hx", "exch_mx")
OUT2 = ("hpbl", "wstar", "delta", "u10", "v10")


def make_columns(seed):
    """[k, j, i] profiles on a stretched grid (dz 40 m -> ~600 m)."""
    rng = np.random.default_rng(seed)
    shp2 = (NJ, NI)
    s = np.linspace(0.0, 1.0, NK)
    dz1 = 40.0 + 560.0 * s**2  # layer depths, same for every column
    dz = np.broadcast_to(dz1[:, None, None], (NK, NJ, NI)).copy()
    dz *= rng.uniform(0.98, 1.02, shp2)  # mild variation between columns
    zw = np.concatenate([np.zeros((1,) + shp2), np.cumsum(dz, axis=0)])
    zs = 0.5 * (zw[:-1] + zw[1:])

    # regimes: unstable (hfx > 0, br < 0), stable (hfx < 0, br > 0), neutral
    regime = rng.integers(0, 3, shp2)
    regime[0, 0], regime[0, 1], regime[0, 2] = 0, 1, 2
    hml = rng.uniform(150.0, 1800.0, shp2)  # mixed-layer depth
    th_s = rng.uniform(290.0, 305.0, shp2)
    gam = rng.uniform(3.0e-3, 7.0e-3, shp2)
    th = th_s + gam * np.maximum(zs - hml, 0.0)
    th += np.where(regime == 0, -rng.uniform(0.2, 2.0, shp2) * np.exp(-zs / 60.0), 0.0)
    th += np.where(
        regime == 1, rng.uniform(1.0, 6.0, shp2) * (1 - np.exp(-zs / 200.0)), 0.0
    )
    th += rng.normal(0.0, 0.05, th.shape)
    qv = rng.uniform(0.004, 0.016, shp2) * np.exp(-zs / 2500.0)
    qv *= 1 + rng.normal(0.0, 0.02, qv.shape)
    # cloud layers near the top of the mixed layer in some columns
    cloudy = rng.random(shp2) < 0.4
    band = (zs > hml - 150) & (zs < hml + 400)
    qc = np.where(cloudy & band, rng.uniform(2e-5, 8e-4, qv.shape), 0.0)
    qi = np.where(cloudy & band & (rng.random(qv.shape) < 0.3), 2e-5, 0.0)

    # hydrostatic Exner function and pressure
    psfc = rng.uniform(95000.0, 102000.0, shp2)
    thv = th * (1 + 0.61 * qv)
    piw = np.empty((NK + 1,) + shp2)
    piw[0] = (psfc / C.p00) ** C.rovcp
    for k in range(NK):
        piw[k + 1] = piw[k] - C.g * dz[k] / (C.cp * thv[k])
    pis = 0.5 * (piw[:-1] + piw[1:])
    prs = C.p00 * pis ** (C.cp / C.rd)
    p8w = C.p00 * piw ** (C.cp / C.rd)
    tk = th * pis

    # winds: log layer + shear aloft + noise
    ust = rng.uniform(0.05, 0.7, shp2)
    znt = 10.0 ** rng.uniform(-4.5, -0.7, shp2)
    ang = rng.uniform(0, 2 * np.pi, shp2)
    spd = ust / C.karman * np.log((zs + znt) / znt) + 2e-3 * zs * rng.uniform(
        0, 1, shp2
    )
    u = spd * np.cos(ang) + rng.normal(0, 0.3, spd.shape)
    v = spd * np.sin(ang) + rng.normal(0, 0.3, spd.shape)
    calm = (2, 3)
    u[:, calm[0], calm[1]] *= 0.02
    v[:, calm[0], calm[1]] *= 0.02

    za = zs[0]
    hfx = np.select(
        [regime == 0, regime == 1],
        [rng.uniform(10, 400, shp2), rng.uniform(-60, -1, shp2)],
        0.0,
    )
    qfx = np.select(
        [regime == 0, regime == 1],
        [rng.uniform(1e-5, 2e-4, shp2), rng.uniform(-1e-5, 1e-5, shp2)],
        0.0,
    )
    br = np.select(
        [regime == 0, regime == 1],
        [rng.uniform(-3.0, -0.01, shp2), rng.uniform(0.005, 0.25, shp2)],
        0.0,
    )
    fm = np.log((za + znt) / znt) + np.where(
        br < 0, -rng.uniform(0, 2, shp2), rng.uniform(0, 4, shp2)
    )
    fh = np.log((za + znt) / (0.1 * znt)) + np.where(
        br < 0, -rng.uniform(0, 2, shp2), rng.uniform(0, 4, shp2)
    )
    wspd = np.maximum(np.sqrt(u[0] ** 2 + v[0] ** 2), 0.1)
    s10 = wspd * np.log(10.0 / znt) / np.log(za / znt)
    u10 = s10 * u[0] / wspd
    v10 = s10 * v[0] / wspd
    xland = np.where(rng.random(shp2) < 0.5, 2.0, 1.0)

    f3 = {
        "ux": u,
        "vx": v,
        "tx": tk,
        "qvx": qv,
        "qcx": qc,
        "qix": qi,
        "p2d": prs,
        "pi2d": pis,
        "dz8w": dz,
    }
    f3["rthraten"] = rng.normal(0.0, 1e-5, qv.shape)
    f2 = {"psfc": psfc, "znt": znt, "ust": ust, "psim": fm, "psih": fh, "xland": xland}
    f2.update(hfx=hfx, qfx=qfx, wspd=wspd, br=br)
    return f3, p8w, f2, u10, v10


@pytest.fixture(scope="module", params=[11, 12])
def case(request):
    f3, p8w, f2, u10, v10 = make_columns(request.param)
    ins = {"params": np.array([NI, NJ, NK, DT], float), **f3, "p2di": p8w, **f2}
    ins["u10"], ins["v10"] = u10, v10
    outs = {k: (NK, NJ, NI) for k in OUT3}
    outs.update({k: (NJ, NI) for k in OUT2 + ("kpbl",)})
    ref = run_driver("ysu", ins, outs)
    return ins, ref


def test_bl_ysu_run(bk, case):
    ins, ref = case
    a = {k: bk.asarray(ins[k]) for k in F3 + F2 + ("p2di", "u10", "v10")}
    a["psfcpa"] = a.pop("psfc")
    a["dz8w2d"] = a.pop("dz8w")
    r = bl_ysu_run(bk, dt=DT, **a)
    kp = bk.to_numpy(r["kpbl1d"])
    np.testing.assert_array_equal(kp, ref["kpbl"])
    # every branch is exercised
    assert (kp > 1).any() and (kp == 1).any()
    for k in OUT3 + OUT2:
        got = bk.to_numpy(r[k])
        want = ref[k]
        scale = max(np.max(np.abs(want)), 1e-300)
        err = np.max(np.abs(got - want)) / scale
        assert err <= 1e-12, f"{k}: err {err:.2e} (of max {scale:.2e})"


def test_bl_ysu_run_grad():
    """Reverse-mode gradients through the scheme are finite (jax)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    f3, p8w, f2, u10, v10 = make_columns(11)
    a = {k: bk.asarray(v) for k, v in {**f3, **f2}.items()}
    a.update(p2di=bk.asarray(p8w), u10=bk.asarray(u10), v10=bk.asarray(v10))
    a["psfcpa"] = a.pop("psfc")
    a["dz8w2d"] = a.pop("dz8w")

    def loss(tx, hfx, ux):
        r = bl_ysu_run(bk, dt=DT, **{**a, "tx": tx, "hfx": hfx, "ux": ux})
        return (r["ttnp"] ** 2).sum() + (r["utnp"] ** 2).sum() + (r["qvtnp"] ** 2).sum()

    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(a["tx"], a["hfx"], a["ux"])
    for x in g:
        assert np.isfinite(np.asarray(x)).all()
        assert np.abs(np.asarray(x)).max() > 0


def test_ysudiss(bk):
    """ysudiss against a direct transcription of the turb.F loop."""
    from cm1x.pbl_ysu import ysudiss

    rng = np.random.default_rng(5)
    nk, shp = 12, (4, 5)
    f = {
        k: rng.normal(size=(nk + 2,) + shp) for k in ("upten", "vpten", "dum1", "dum2")
    }
    f["rmh"] = rng.uniform(0.5, 2.0, (nk + 2,) + shp)
    f["mf"] = rng.uniform(0.5, 2.0, (nk + 2,) + shp)
    f["dissten"] = rng.uniform(0, 1, (nk + 2,) + shp)
    dz = 200.0
    want = f["dissten"].copy()
    t13 = np.zeros((nk + 2,) + shp)
    t23 = np.zeros((nk + 2,) + shp)
    for k in range(nk, 1, -1):
        tem1 = dz * f["rmh"][k]
        tem2 = (1.0 / dz) * f["mf"][k]
        t13[k] = t13[k + 1] - f["upten"][k] * tem1
        t23[k] = t23[k + 1] - f["vpten"][k] * tem1
        want[k] = want[k] + (
            t13[k] * (f["dum1"][k] - f["dum1"][k - 1]) * tem2
            + t23[k] * (f["dum2"][k] - f["dum2"][k - 1]) * tem2
        )
    got = ysudiss(bk, dz=dz, **{k: bk.asarray(v) for k, v in f.items()})
    np.testing.assert_allclose(bk.to_numpy(got), want, rtol=1e-14, atol=1e-14)
