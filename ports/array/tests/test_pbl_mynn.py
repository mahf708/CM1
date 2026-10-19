"""pbl_mynn.mynnedmf_wrapper_run (the MYNN-EDMF PBL, ipbl=4/5) against
src/module_bl_mynn*.F, called as turb.F's domynn block does
(fortran_harness/drv_mynn_pbl.F90).

Synthetic but physical columns on a stretched grid up to ~12 km (as the
cm1k bridge test): water and land; convective (superadiabatic surface
layer, strong upward fluxes: the mass-flux plumes fire), near-neutral and
stable; moist (near saturation at the boundary-layer top: subgrid clouds and
plume condensation) and dry; calm to strong winds.  A run is the
initialisation call (nstep=0) and four more steps with the state carried
and th/qv/qc/qi/u/v advanced with the Fortran tendencies; every step both
sides start from the Fortran state and every output is compared."""

import os

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x import constants as C
from cm1x.pbl_mynn import INPUT_NAMES, STATE_NAMES, TEND_NAMES, mynnedmf_wrapper_run

pytestmark = requires_gfortran


def _exact(bk):
    """numpy with its AVX-512 paths off calls glibc's exp/log/pow like the
    Fortran: the port is then bitwise identical."""
    return bk.name == "numpy" and "AVX512F" in os.environ.get(
        "NPY_DISABLE_CPU_FEATURES", ""
    )

NI, NJ, NK, NSTEP = 6, 4, 40, 4
SFC_NAMES = (
    "znt", "xland", "tsk", "qsfc", "psfc", "ust", "ch", "hfx", "qfx", "wspd", "rmol", "hpbl",
    "kpbl", "nupdraft", "maxmf", "ktop_plume",
)  # fmt: skip
# name: (ipbl, edmf, edmf_mom, moist, dx, dt, ptype, seed)
CASES = {
    "l25_mf_mom_moist": (4, 1, 1, True, 1000.0, 10.0, 5, 11),
    "l3_mf_mom_moist": (5, 1, 1, True, 1000.0, 10.0, 5, 12),
    "l25_mf_nomom_dry": (4, 1, 0, False, 1000.0, 10.0, 5, 13),
    "l3_nomf_moist": (5, 0, 0, True, 1000.0, 10.0, 5, 14),
    "l3_mf_dry": (5, 1, 1, False, 1000.0, 10.0, 5, 15),
    "l25_dx3000_dt30": (4, 1, 1, True, 3000.0, 30.0, 1, 101),
    "l3_dx250_dt2": (5, 1, 1, True, 250.0, 2.0, 5, 102),
}


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _columns(moist, seed):
    rng = np.random.default_rng(seed)
    U = rng.uniform
    inp = np.zeros((12, NK, NJ, NI))
    s2 = np.zeros((16, NJ, NI))
    dzv = 20.0 * 1.11 ** np.arange(NK)
    zh = np.cumsum(dzv) - 0.5 * dzv
    for j in range(1, NJ + 1):
        for i in range(1, NI + 1):
            reg = (i + j) % 3
            water = i % 2 == 0
            hbl = U(800, 1800) if reg == 0 else U(300, 800) if reg == 1 else U(80, 250)
            ths = 290.0 + U(-5, 8) + (-20.0 if j == NJ else 0.0)
            psfc = 1.0e5 + U(-1500, 1500)
            wind = U(0.2, 1.0) if j == 1 else (U(3, 7) if j == 2 else U(10, 20))
            d = U(0, 2 * np.pi)
            rhs = (U(0.75, 0.9) if reg == 0 else U(0.5, 0.85)) if moist else 0.0
            pif = (psfc / C.p00) ** C.rovcp
            for k in range(NK):
                z = zh[k]
                if reg == 0:
                    th = (
                        ths
                        + (1.5 * (1 - z / 100) if z < 100 else 0)
                        + (3 + 0.004 * (z - hbl) if z > hbl else 0)
                    )
                elif reg == 1:
                    th = ths + (1.0 + 0.004 * (z - hbl) if z > hbl else 0.0005 * z)
                else:
                    th = ths + 6.0 * (1 - np.exp(-z / hbl)) + 0.004 * z
                th += U(-0.05, 0.05)
                rh = 0.0
                if moist:
                    rh = (
                        rhs + (0.995 - rhs) * (z / hbl)
                        if z < hbl
                        else max(0.05, 0.6 * np.exp(-(z - hbl) / 2500))
                    )
                    if z > 9000:
                        rh = 0.02
                pih = pif - 0.5 * C.g * dzv[k] / (C.cp * th)
                t = th * pih
                p = C.p00 * pih**C.cpdrd
                qv = rh * _qsat(t, p)
                pif = pih - 0.5 * C.g * dzv[k] / (C.cp * th)
                prof = min(1.2, np.log(1 + z / 0.1) / np.log(1 + 1000 / 0.1))
                c = inp[:, k, j - 1, i - 1]
                c[0] = dzv[k]
                c[1] = wind * np.cos(d) * prof + U(-0.2, 0.2)
                c[2] = wind * np.sin(d) * prof + U(-0.2, 0.2)
                c[3] = U(-0.05, 0.05)
                c[4] = th
                c[5] = qv
                c[6] = (
                    U(0, 2e-4) if (moist and abs(z - hbl) < 150 and reg == 0) else 0.0
                )
                c[7] = U(0, 2e-5) if (moist and 7000 < z < 9000) else 0.0
                c[8] = p
                c[9] = pih
                c[10] = p / (C.rd * t * (1 + 0.61 * qv))
                c[11] = t
            t1 = inp[11, 0, j - 1, i - 1]
            dts = U(2, 8) if reg == 0 else U(-0.3, 0.3) if reg == 1 else U(-6, -1)
            tsk = t1 + dts + 0.3
            ust = U(0.05, 0.25) if reg == 2 else max(0.1, 0.05 * wind + U(0, 0.2))
            hfx = U(150, 450) if reg == 0 else U(-5, 20) if reg == 1 else U(-60, -5)
            qfx = (U(5e-5, 2e-4) if water else U(0, 8e-5)) if moist else 0.0
            rho1 = inp[10, 0, j - 1, i - 1]
            s = s2[:, j - 1, i - 1]
            s[0] = 1.0e-4 if water else U(0.01, 0.5)
            s[1] = 2.0 if water else 1.0
            s[2] = tsk
            s[3] = _qsat(tsk, psfc) if moist else 0.0
            s[4] = psfc
            s[5] = ust
            s[6] = U(0.002, 0.02)
            s[7] = hfx
            s[8] = qfx
            s[9] = max(1.0, np.hypot(inp[1, 0, j - 1, i - 1], inp[2, 0, j - 1, i - 1]))
            s[10] = -0.4 * C.g * hfx / (rho1 * C.cp * inp[4, 0, j - 1, i - 1] * ust**3)
            s[11] = hbl
    return inp, s2, rng


def _steps(ipbl, edmf, mom, moist, dx, dt, ptype, seed):
    inp, s2, rng = _columns(moist, seed)
    st = np.zeros((26, NK, NJ, NI))
    rthr = -2.0e-5
    for nstep in range(NSTEP + 1):
        params = [NI, NJ, NK, nstep, ipbl, ptype, edmf, mom, dt, dx, rthr]
        ref = run_driver(
            "mynn_pbl",
            {"params": np.array(params, float), "in3": inp, "st3": st, "s2": s2},
            {"st3": (26, NK, NJ, NI), "te3": (9, NK, NJ, NI), "s2": (16, NJ, NI)},
        )
        yield nstep, inp.copy(), s2.copy(), st.copy(), ref
        te, st = ref["te3"], ref["st3"].copy()
        s2 = ref["s2"].copy()
        inp = inp.copy()
        inp[1] += dt * te[0]
        inp[2] += dt * te[1]
        inp[4] += dt * te[2]
        for n, m in ((5, 3), (6, 4), (7, 5)):
            inp[n] = np.maximum(0.0, inp[n] + dt * te[m])
        inp[11] = inp[4] * inp[9]
        kk, jj, ii = np.meshgrid(
            np.arange(1, NK + 1),
            np.arange(1, NJ + 1),
            np.arange(1, NI + 1),
            indexing="ij",
        )
        st[1] = np.maximum(
            0.0, st[1] * (1.0 + 0.01 * np.sin(0.7 * ii + 0.3 * jj + 0.1 * kk + nstep))
        )
        s2[7] *= rng.uniform(0.9, 1.1, (NJ, NI))
        s2[8] *= rng.uniform(0.9, 1.1, (NJ, NI))
        s2[5] *= rng.uniform(0.95, 1.05, (NJ, NI))


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    ipbl, edmf, mom, moist, dx, dt, ptype, seed = CASES[request.param]
    cfg = _Cfg(ipbl=ipbl, bl_mynn_edmf=edmf, bl_mynn_edmf_mom=mom, cm1setup=2)
    return cfg, dx, dt, list(_steps(ipbl, edmf, mom, moist, dx, dt, ptype, seed))


def _err(got, want):
    """Relative error against max(|want|, 1% of the field's largest value)
    (last-bit differences of exp/log/pow between numpy/jax/torch and glibc,
    amplified where a field is a near-cancelling difference)."""
    both_nan = np.isnan(got) & np.isnan(want)
    scale = 1e-2 * np.nanmax(np.abs(want)) + 1e-300
    err = np.abs(got - want) / np.maximum(np.abs(want), scale)
    return float(np.max(np.where(both_nan | (got == want), 0.0, err)))


TOL = 1e-10
# tendency -> the field it updates (dt * tendency is compared with that field:
# the tendencies are (x_new - x)/dt, so their own relative error is the
# round-off of x divided by the small change)
TEND_FIELD = {
    "upten": 1,
    "vpten": 2,
    "thpten": 4,
    "qvpten": 5,
    "qcpten": 6,
    "qipten": 7,
}
COVER = {"edmf_qc": 0.0, "neg_maxmf": 0}


def test_mynn_pbl(bk, case):
    cfg, dx, dt, steps = case
    A = bk.asarray
    worst = {}
    nup = 0
    for nstep, inp, s2, st, ref in steps:
        r_sfc, r_st, r_te = mynnedmf_wrapper_run(
            bk,
            cfg,
            initflag=(nstep == 0),
            delt=dt,
            dxc=dx,
            inp={k: A(inp[n]) for n, k in enumerate(INPUT_NAMES)},
            sfc={k: A(s2[n]) for n, k in enumerate(SFC_NAMES[:12])},
            st={k: A(st[n]) for n, k in enumerate(STATE_NAMES)},
        )
        for n, k in enumerate(STATE_NAMES):
            worst[k] = max(worst.get(k, 0.0), _err(bk.to_numpy(r_st[k]), ref["st3"][n]))
        for n, k in enumerate(TEND_NAMES):
            got, want = bk.to_numpy(r_te[k]), ref["te3"][n]
            if k in TEND_FIELD:
                x = inp[TEND_FIELD[k]]
                # dt * error against the field it updates (1% of the field's
                # largest value as the floor, 1e-12 for q), x100; or, where
                # the field is itself ~dt*tendency (the dry cases' qv is only
                # what qfx put in), the error relative to the tendency
                floor = 1e-2 * np.max(np.abs(x)) + (1e-12 if k[0] == "q" else 1e-300)
                e = np.max(dt * np.abs(got - want) / np.maximum(np.abs(x), floor))
                e = min(100.0 * float(e), _err(got, want))
                worst[k] = max(worst.get(k, 0.0), e)
            else:
                worst[k] = max(worst.get(k, 0.0), _err(got, want))
        for n, k in enumerate(SFC_NAMES):
            if k in ("rmol", "hpbl", "kpbl", "nupdraft", "maxmf", "ktop_plume"):
                worst[k] = max(
                    worst.get(k, 0.0), _err(bk.to_numpy(r_sfc[k]), ref["s2"][n])
                )
        nup = max(nup, ref["s2"][13].max())
        COVER["edmf_qc"] = max(COVER["edmf_qc"], ref["st3"][16].max())
        COVER["neg_maxmf"] += int((ref["s2"][14] < 0).sum())
    tol = 0.0 if _exact(bk) else TOL
    bad = {k: v for k, v in worst.items() if v > tol}
    assert not bad, f"rel err > {tol}: {bad}"
    if cfg["bl_mynn_edmf"] == 1:
        assert nup > 0  # the plumes fire


def test_coverage():
    """(after the cases) condensation in the plumes and dry plumes (maxmf < 0)"""
    if COVER["edmf_qc"] == 0.0 and COVER["neg_maxmf"] == 0:
        pytest.skip("run with test_mynn_pbl")
    assert COVER["edmf_qc"] > 0.0
    assert COVER["neg_maxmf"] > 0
