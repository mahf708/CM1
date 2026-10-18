"""Whole-model check of cm1x.model.Model with RRTMG radiation (radopt=2)
against CM1 for the reference case ports/reference/kessler_rrtmg: the
kessler_periodic warm bubble on a 20x20x40 grid, daytime (15 May 2009,
21:38 UTC, 36.68 N 98.35 W), sfcmodel=1 over water (LANDUSE.TBL category
16) with isfcflx=0, and dtrad=12 s, so radiation runs at t=0 and after every
second step and the steps in between reuse its tendency.

Needs a double-precision (-fdefault-real-8) Fortran run of exactly that
namelist with tapfrq=6 (t = 0, 6, ..., 60 s); fields are stored as float32,
so only that rounding is tolerated:

    ports/reference/run_reference.sh ports/reference/kessler_rrtmg /tmp/kr r8 \\
        timax=60.0 tapfrq=6.0
    CM1X_REF_DIR_RAD=/tmp/kr pytest ports/array/tests/test_model_radiation.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.model import FIELDS, RAD2D, RAD3D, Model

NAMELIST = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "kessler_rrtmg"
    / "namelist.input"
)
REF_DIR = os.environ.get("CM1X_REF_DIR_RAD")
NSTEPS = 10
RTOL = 2.0e-6
ULPS = 2.0
SCALE = {"th": "thpert", "prs": "prspert", "pi": "pipert"}
FLOOR = {"w": 1.0e-3, "qc": 1.0e-6, "qr": 1.0e-6, "cldfra": 1.0}


def _compare(fm, ds, t):
    bad = []
    for k in FIELDS + RAD3D + RAD2D:
        if k not in ds.variables:
            continue
        r = np.asarray(ds[k][t], dtype=np.float64)
        rs = np.asarray(ds[SCALE.get(k, k)][t], dtype=np.float64)
        err = np.abs(fm[k] - r).max()
        tol = RTOL * max(np.abs(rs).max(), FLOOR.get(k, 1.0e-30))
        if k in SCALE:
            tol = max(tol, ULPS * float(np.spacing(np.float32(np.abs(r).max()))))
        if not err <= tol:
            bad.append(f"{k}: max|diff|={err:.3e} > tol={tol:.3e}")
    return bad


def test_radiation_model_setup():
    m = Model(NAMELIST, overrides={"nx": 4, "ny": 4})
    assert m.rad and m.dtrad == 12.0 and m.radtim == 12.0
    # water surface from LANDUSE.TBL (category 16, summer)
    assert float(m.base["albd"].max()) == 0.08
    assert float(m.base["emiss"].min()) == 0.98
    f = m.fields()
    for k in RAD3D:
        assert f[k].shape == (m.grid.nk, m.grid.nj, m.grid.ni)
    assert f["swdnb"].min() > 500.0  # daytime
    m.run(2)
    assert m.radtim == 24.0


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR, reason="set CM1X_REF_DIR_RAD to the kessler_rrtmg reference"
)
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_radiation_10steps_vs_cm1(backend):
    if backend != "numpy":
        pytest.importorskip(backend)
        if os.environ.get("CM1X_TEST_ALL_BACKENDS") != "1":
            pytest.skip(
                "set CM1X_TEST_ALL_BACKENDS=1 to run the (slow) non-numpy backends"
            )
    elif "numpy" not in os.environ.get("CM1X_BACKENDS", "numpy"):
        pytest.skip("numpy not in CM1X_BACKENDS")
    netCDF4 = pytest.importorskip("netCDF4")
    m = Model(NAMELIST, backend=backend, dtype="float64")
    failures = []
    with netCDF4.Dataset(Path(REF_DIR) / "cm1out.nc") as ds:
        assert ds["time"].shape[0] >= NSTEPS + 1
        for k in RAD3D + RAD2D:
            assert k in ds.variables, k
        for t in range(NSTEPS + 1):
            if t:
                m.step()
            assert abs(m.mtime - float(ds["time"][t])) < 1e-6
            failures += [f"t={t}: {b}" for b in _compare(m.fields(), ds, t)]
    assert not failures, "\n".join(failures)
