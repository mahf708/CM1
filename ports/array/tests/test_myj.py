"""The MYJ surface layer (sfc_myj.myjsfc: MYJSFCINIT + MYJSFC + SFCDIF,
src/module_sf_myjsfc.F) and PBL (pbl_myj.myjpbl: MYJPBL + MIXLEN, PRODQ2,
DIFCOF, VDIFQ, VDIFH, VDIFV, src/module_bl_myjpbl.F) against the Fortran on
realistic random columns: water points over every viscous-sublayer regime
(u* below 0.225, between 0.225 and 0.7, above 0.7), first and later steps,
isftcflx 0 and 1, and land points with positive roughness."""

import numpy as np
import pytest
from fortran import requires_gfortran, run_driver

from cm1x.pbl_myj import myjpbl
from cm1x.sfc_myj import myjsfc

pytestmark = requires_gfortran

NI, NJ, NK = 9, 4, 20


def _column_state(rng, land_frac):
    shp = (NJ, NI)
    dz = np.linspace(60.0, 400.0, NK)[:, None, None] * rng.uniform(0.9, 1.1, (NK, *shp))
    z = np.cumsum(dz, axis=0) - 0.5 * dz
    zi = np.concatenate([np.zeros((1, *shp)), np.cumsum(dz, axis=0)])
    # mixed layer up to ~800-1500 m, stable above; a few cloudy levels
    zml = rng.uniform(600.0, 1500.0, shp)
    th = 300.0 + rng.uniform(-1, 1, shp) + np.where(z > zml, 0.004 * (z - zml), 0.0)
    th += rng.normal(0.0, 0.05, th.shape)
    qv = 0.014 * np.exp(-z / 2500.0) * rng.uniform(0.9, 1.0, th.shape)
    qc = np.where(
        (z > zml - 200) & (z < zml + 200), rng.uniform(0, 5e-4, th.shape), 0.0
    )
    qi = np.where((z > 3000.0), rng.uniform(0, 1e-5, th.shape), 0.0)
    psfc = rng.uniform(99500.0, 101500.0, shp)
    pint = psfc * np.exp(-zi / 8000.0)
    pmid = psfc * np.exp(-z / 8000.0)
    exner = (pmid / 1.0e5) ** (287.04 / 1005.7)
    t = th * exner
    u = 5.0 + 0.004 * z + rng.normal(0, 1.5, th.shape)
    v = -2.0 + 0.002 * z + rng.normal(0, 1.5, th.shape)
    tke = np.where(z < zml, rng.uniform(0.15, 1.5, th.shape), 0.1)
    tke[0] = rng.uniform(0.2, 1.0, shp)
    xland = np.where(rng.random(shp) < land_frac, 1.0, 2.0)
    return {
        "dz": dz,
        "pmid": pmid,
        "pint": pint,
        "th": th,
        "t": t,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "qi": qi,
        "u": u,
        "v": v,
        "tke": tke,
        "xland": xland,
        "psfc": psfc,
    }


SFC_OUT = [
    "qsfc",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "ustar",
    "znt",
    "pblh",
    "akhs",
    "akms",
    "rmol",
    "rib",
    "chs",
    "chs2",
    "cqs2",
    "hfx",
    "qfx",
    "flx_lh",
    "flhc",
    "flqc",
    "qgh",
    "cpm",
    "ct",
    "u10",
    "v10",
    "t02",
    "th02",
    "tshltr",
    "th10",
    "q02",
    "qshltr",
    "q10",
    "pshltr",
    "u10e",
    "v10e",
    "psim",
    "psih",
    "fm",
    "fhh",
    "z0out",
]

# (itimestep, isftcflx, land fraction)
SFC_CASES = {
    "first_water": (1, 0, 0.0),
    "later_water": (7, 0, 0.0),
    "later_water_isftcflx": (7, 1, 0.0),
    "first_mixed": (1, 1, 0.4),
    "later_mixed": (3, 0, 0.4),
}


@pytest.fixture(scope="module", params=list(SFC_CASES))
def sfc_case(request):
    return sfc_inputs(request.param)


def sfc_inputs(name):
    itimestep, isftcflx, land = SFC_CASES[name]
    rng = np.random.default_rng(11 + itimestep + 3 * isftcflx)
    c = _column_state(rng, land)
    shp = (NJ, NI)
    tlow = c["t"][0]
    tsk = tlow + rng.uniform(-3.0, 4.0, shp)
    thsk = tsk / (c["psfc"] / 1.0e5) ** (287.04 / 1005.7)
    # u* across the three regimes over water (u* < 0.225, < 0.7, >= 0.7)
    ust = rng.choice([0.05, 0.15, 0.3, 0.5, 0.8, 1.0], shp) * rng.uniform(0.9, 1.1, shp)
    sea = c["xland"] > 1.5
    ins = {
        "ht": np.zeros(shp),
        "dz": c["dz"],
        "pmid": c["pmid"],
        "pint": c["pint"],
        "th": c["th"],
        "t": c["t"],
        "qv": c["qv"],
        "qc": c["qc"],
        "u": c["u"],
        "v": c["v"],
        "q2": c["tke"],
        "tsk": tsk,
        "qsfc": c["qv"][0] * rng.uniform(1.0, 1.3, shp),
        "thz0": thsk + rng.uniform(-0.5, 0.5, shp),
        "qz0": c["qv"][0] * rng.uniform(1.0, 1.2, shp),
        "uz0": rng.uniform(-0.3, 0.3, shp),
        "vz0": rng.uniform(-0.3, 0.3, shp),
        "xland": c["xland"],
        "ust": ust,
        "znt": np.where(sea, 2e-4, rng.uniform(0.02, 0.5, shp)),
        "z0base": np.where(sea, 2e-4, rng.uniform(0.02, 0.5, shp)),
        "pblh": rng.uniform(300.0, 1800.0, shp),
        "mavail": np.where(sea, 1.0, rng.uniform(0.2, 0.9, shp)),
        "akhs": rng.uniform(0.005, 0.05, shp),
        "akms": rng.uniform(0.005, 0.05, shp),
        "rhosfc": rng.uniform(1.1, 1.2, shp),
    }
    params = np.array([NI, NJ, NK, itimestep, 1, isftcflx], float)
    ref = run_driver("myjsfc", {"params": params, **ins}, {k: shp for k in SFC_OUT})
    return itimestep, isftcflx, ins, ref


# absolute floors where the reference can be pure round-off (at the first
# step over water qz0 = qlow, so the moisture flux is a vanishing difference)
FLOOR = {"qfx": 1e-7, "flx_lh": 1e-7 * 2.501e6}


def _tol(bk):
    # numpy's exp/log/pow differ from glibc's in the last bit for a few
    # percent of arguments; jax's and torch's somewhat more often
    return 1e-12 if bk.name == "numpy" else 1e-11


def _rel(got, want, floor=0.0):
    """Max error relative to max(|want|, 1e-3 max|field|): values near zero
    (fluxes of a vanishing difference at the first step, psi near zeta = 0)
    are measured against the field's magnitude."""
    scale = np.maximum(np.abs(want), max(1e-3 * np.max(np.abs(want)), floor))
    return float(np.max(np.abs(got - want) / np.maximum(scale, 1e-300)))


def test_myjsfc(bk, sfc_case):
    itimestep, isftcflx, ins, ref = sfc_case
    a = {k: bk.asarray(v) for k, v in ins.items()}
    a["ustar"] = a.pop("ust")
    r = myjsfc(bk, itimestep=itimestep, isftcflx=isftcflx, usemyjpbl=True, **a)
    errs = {k: _rel(bk.to_numpy(r[k]), ref[k], FLOOR.get(k, 0.0)) for k in SFC_OUT}
    bad = {k: e for k, e in errs.items() if not e <= _tol(bk)}
    assert not bad, bad


PBL_OUT3 = [
    "tke_myj",
    "el_myj",
    "exch_h",
    "exch_m",
    "rublten",
    "rvblten",
    "rthblten",
    "rqvblten",
    "rqcblten",
    "rqiblten",
]
PBL_OUT2 = ["pblh", "kpbl", "mixht", "ct", "qsfc", "thz0", "qz0"]


@pytest.fixture(scope="module", params=[0.0, 0.4])
def pbl_case(request):
    return pbl_inputs(request.param)


def pbl_inputs(land):
    rng = np.random.default_rng(5 + int(10 * land))
    c = _column_state(rng, land)
    shp = (NJ, NI)
    tsk = c["t"][0] + rng.uniform(-2.0, 3.0, shp)
    thsk = tsk / (c["psfc"] / 1.0e5) ** (287.04 / 1005.7)
    sea = c["xland"] > 1.5
    ins = {
        "dz": c["dz"],
        "pmid": c["pmid"],
        "pint": c["pint"],
        "th": c["th"],
        "t": c["t"],
        "exner": c["exner"],
        "qv": c["qv"],
        "qcw": c["qc"],
        "qci": c["qi"],
        "u": c["u"],
        "v": c["v"],
        "tke_myj": c["tke"],
        "ht": np.zeros(shp),
        "tsk": tsk,
        "qsfc": c["qv"][0] * rng.uniform(1.0, 1.3, shp),
        "chklowq": np.where(sea, 1.0, rng.uniform(0.0, 0.9, shp)),
        "thz0": thsk + rng.uniform(-0.5, 0.5, shp),
        "qz0": c["qv"][0] * rng.uniform(1.0, 1.2, shp),
        "uz0": rng.uniform(-0.3, 0.3, shp),
        "vz0": rng.uniform(-0.3, 0.3, shp),
        "xland": c["xland"],
        "ust": rng.uniform(0.05, 0.8, shp),
        "ct": rng.uniform(-0.01, 0.01, shp),
        "akhs": rng.uniform(0.005, 0.05, shp),
        "akms": rng.uniform(0.005, 0.05, shp),
        "elflx": rng.uniform(-20.0, 300.0, shp),
    }
    params = np.array([NI, NJ, NK, 6.0], float)
    outs = {k: (NK, *shp) for k in PBL_OUT3}
    outs.update({k: shp for k in PBL_OUT2})
    ref = run_driver("myjpbl", {"params": params, **ins}, outs)
    return ins, ref


def test_myjpbl(bk, pbl_case):
    ins, ref = pbl_case
    a = {k: bk.asarray(v) for k, v in ins.items()}
    a["ustar"] = a.pop("ust")
    r = myjpbl(bk, dt=6.0, **a)
    errs = {}
    for k in PBL_OUT3 + PBL_OUT2:
        got = bk.to_numpy(r[k])
        want = ref[k]
        if k == "rthblten":
            # (new theta - old theta)/dt: compare the new theta (the
            # difference of two ~300 K values loses ~13 digits)
            got = ins["th"] + 6.0 * got
            want = ins["th"] + 6.0 * want
        scale = np.maximum(np.abs(want), 1e-30)
        if k.endswith("blten") or k in ("exch_h", "exch_m", "el_myj"):
            # relative to the field's largest magnitude
            scale = np.maximum(scale, np.max(np.abs(want)))
        errs[k] = float(np.max(np.abs(got - want) / scale))
    bad = {k: e for k, e in errs.items() if not e <= _tol(bk)}
    assert not bad, bad
