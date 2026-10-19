"""sfc_mynn.sfclay_mynn (the MYNN surface layer, sfcmodel=6) against
src/module_sf_mynn.F's SFCLAY_mynn, called as turb.F does
(fortran_harness/drv_mynn_sfc.F90).

Synthetic lowest-level columns: water and land (some snow covered),
stable / neutral / convective, moist and dry, calm to strong winds, layer
depths that reach every u10 branch.  Five steps per case with the
persistent arrays (ust, mol, qsfc, znt, ustm, hfx, qfx, zol, rmol, wspd)
carried over (each step both sides start from the Fortran state) and the
inputs drifting; every isftcflx (0..3) over water and iz0tlnd (0, 1, 3) over
land, isfflx = 0 once."""

import os

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x import constants as C
from cm1x.sfc_mynn import psi_init, sfclay_mynn

pytestmark = requires_gfortran


def _exact(bk):
    """numpy with its AVX-512 paths off calls glibc's exp/log/pow like the
    Fortran: the port is then bitwise identical."""
    return bk.name == "numpy" and "AVX512F" in os.environ.get(
        "NPY_DISABLE_CPU_FEATURES", ""
    )

NI, NJ, NSTEP = 10, 6, 5
STATE = [
    "ust",
    "mol",
    "qsfc",
    "znt",
    "ustm",
    "hfx",
    "qfx",
    "zol",
    "rmol",
    "wspd",
    "chs",
    "chs2",
    "cqs2",
    "cpm",
    "regime",
    "psim",
    "psih",
    "lh",
    "flhc",
    "flqc",
    "qgh",
    "u10",
    "v10",
    "th2",
    "t2",
    "q2",
    "gz1oz0",
    "br",
    "ch",
    "ck",
    "cka",
    "cd",
    "cda",
]
CASES = {
    "isftcflx0_iz0tlnd0": (0, 0, 1, True),
    "isftcflx1_iz0tlnd1": (1, 1, 1, True),
    "isftcflx2_iz0tlnd3": (2, 3, 1, True),
    "isftcflx3_iz0tlnd0_dry": (3, 0, 1, False),
    "isfflx0": (0, 1, 0, True),
}


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _steps(isftcflx, iz0tlnd, isfflx, moist, dx=1000.0, seed=11):
    """Yield (step, inputs, s2in, state_in, ref_state_out) for NSTEP steps."""
    rng = np.random.default_rng(seed + 7 * isftcflx + 3 * iz0tlnd)
    U = rng.uniform
    jj, ii = np.meshgrid(np.arange(1, NJ + 1), np.arange(1, NI + 1), indexing="ij")
    water = (ii + jj) % 2 == 0
    s2 = np.zeros((6, NJ, NI))
    psfc = 1.0e5 + U(-2000.0, 2000.0, (NJ, NI))
    s2[0] = psfc
    s2[1] = U(100.0, 1500.0, (NJ, NI))
    s2[2] = np.where(water, 1.0, U(0.05, 0.6, (NJ, NI)))
    s2[3] = np.where(water, 2.0, 1.0)
    s2[5] = np.where(~water & (jj == NJ), 0.2, 0.0)
    st = np.zeros((33, NJ, NI))
    st[STATE.index("znt")] = np.where(water, 1.0e-4, U(0.01, 0.5, (NJ, NI)))
    st[STATE.index("ust")] = 1.0e-6
    st[STATE.index("qsfc")] = 1.0e-5
    dzs = np.array([8.0, 12.0, 20.0, 40.0, 100.0])
    dz = np.zeros((2, NJ, NI))
    dz[0] = dzs[(ii - 1) % 5]
    dz[1] = dzs[(ii + jj) % 5]
    for step in range(1, NSTEP + 1):
        reg = (ii + 2 * jj + step) % 4
        t1 = (
            285.0
            + 8.0 * np.sin(0.3 * ii + 0.7 * jj)
            + U(-1, 1, (NJ, NI))
            - np.where(jj == 1, 25.0, 0.0)
        )
        dts = np.where(
            reg == 0,
            U(-6.0, -0.5, (NJ, NI)),
            np.where(reg == 1, 0.0, U(0.5, 8.0, (NJ, NI))),
        )
        s2[4] = t1 + dts + 0.5
        wind = np.where(
            ii == 1,
            U(0, 0.05, (NJ, NI)),
            np.where(ii == 2, U(0.5, 2, (NJ, NI)), U(2, 25, (NJ, NI))),
        )
        d = U(0, 2 * np.pi, (NJ, NI))
        u = np.stack([wind * np.cos(d), 1.15 * wind * np.cos(d)])
        v = np.stack([wind * np.sin(d), 1.15 * wind * np.sin(d)])
        p = psfc - 1.16 * 9.81 * 0.5 * dz[0]
        qv = U(0.3, 0.95, (NJ, NI)) * _qsat(t1, p) if moist else np.zeros((NJ, NI))
        rho = p / (C.rd * t1 * (1.0 + 0.61 * qv))
        s2[1] *= U(0.9, 1.1, (NJ, NI))
        ins = {
            "u3d": u,
            "v3d": v,
            "dz8w": dz.copy(),
            "t3d": t1,
            "qv3d": qv,
            "p3d": p,
            "rho3d": rho,
        }
        params = [NI, NJ, step, isfflx, isftcflx, iz0tlnd, dx]
        ref = run_driver(
            "mynn_sfc",
            {"params": np.array(params, float), "s2in": s2, "st": st, **ins},
            {"st": (33, NJ, NI), "psi": (4, 1001)},
        )
        yield step, ins, s2.copy(), st.copy(), ref
        st = ref["st"]


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    isftcflx, iz0tlnd, isfflx, moist = CASES[request.param]
    cfg = _Cfg(sfcmodel=6, isftcflx=isftcflx, iz0tlnd=iz0tlnd)
    return cfg, isfflx, list(_steps(isftcflx, iz0tlnd, isfflx, moist))


def _cmp(name, got, want, tol=1e-12):
    """Relative error, measured against max(|want|, 1% of the field's
    largest value): numpy/jax/torch's exp, log, pow, atan differ from glibc
    in the last bit, which near-cancelling differences (thv1 - thvg in br,
    flhc*(thg - th1) + ... in hfx) amplify where the field is near 0."""
    both_nan = np.isnan(got) & np.isnan(want)
    scale = 1e-2 * np.nanmax(np.abs(want)) + 1e-300
    err = np.abs(got - want) / np.maximum(np.abs(want), scale)
    err = np.where(both_nan | (got == want), 0.0, err)
    assert np.max(err) <= tol, f"{name}: rel err {np.max(err):.2e}"


def test_psi_tables(bk):
    _, _, _, _, ref = next(_steps(0, 0, 1, True))
    T = psi_init(bk)
    for n, key in enumerate(("psim_stab", "psim_unstab", "psih_stab", "psih_unstab")):
        _cmp(key, bk.to_numpy(T[key]), ref["psi"][n], 0.0 if _exact(bk) else 1e-14)


def test_sfclay_mynn(bk, case):
    cfg, isfflx, steps = case
    T = psi_init(bk)
    A = bk.asarray
    for step, ins, s2, st, ref in steps:
        stin = {k: A(st[n]) for n, k in enumerate(STATE[:10])}
        r = sfclay_mynn(
            bk,
            cfg,
            T,
            **{k: A(v) for k, v in ins.items()},
            psfcpa=A(s2[0]),
            pblh=A(s2[1]),
            mavail=A(s2[2]),
            xland=A(s2[3]),
            tsk=A(s2[4]),
            snowh=A(s2[5]),
            **stin,
            itimestep=step,
            dx=1000.0,
            isfflx=isfflx,
        )
        for n, k in enumerate(STATE):
            _cmp(
                f"step {step} {k}",
                bk.to_numpy(r[k]),
                ref["st"][n],
                0.0 if _exact(bk) else 1e-12,
            )
