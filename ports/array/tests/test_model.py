"""Whole-model check of cm1x.model.Model against CM1's own netCDF output for
the reference case ports/reference/kessler_periodic.

Needs a Fortran reference run of exactly that namelist: set CM1X_REF_DIR to a
directory containing its cm1out.nc (11 output times, t = 0, 6, ..., 60 s).
Accepted: a double-precision run (-fdefault-real-8; fields still stored as
float32 in the file) or CM1's default single-precision run.  CM1X_REF_PREC
(r8 / r4) selects the tolerance; if unset it is r8 when the directory name
contains "r8", else the looser r4.

    CM1X_REF_DIR=/tmp/ref_r8 pytest ports/array/tests/test_model.py
    CM1X_REF_DIR=/tmp/r1     pytest ports/array/tests/test_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.model import FIELDS, Model

HERE = Path(__file__).resolve().parent
NAMELIST = HERE.parent.parent / "reference" / "kessler_periodic" / "namelist.input"
REF_DIR = os.environ.get("CM1X_REF_DIR")
REF = Path(REF_DIR) / "cm1out.nc" if REF_DIR else None
PREC = os.environ.get("CM1X_REF_PREC") or (
    "r8" if REF_DIR and "r8" in Path(REF_DIR).name else "r4"
)
NSTEPS = 10

requires_ref = pytest.mark.skipif(
    REF is None or not REF.exists(),
    reason="set CM1X_REF_DIR to a directory with the reference cm1out.nc",
)

# Absolute tolerance per field, as a fraction of the field's max |value| at
# that time (with a floor relative to the field's typical magnitude).
# r8 reference: only the float32 rounding of the netCDF file remains
#   (observed max rel. errors ~1e-7 .. 5e-8).
# r4 reference: CM1's single-precision drift (observed after 10 steps:
#   w 2.7e-5 of 0.2, thpert 1.8e-5,
#   prspert 0.02 Pa of 30 Pa, pi 1.8e-7 of pipert 8.9e-5).
RTOL = {"r8": 2.0e-6, "r4": 5.0e-3}
# floor for full fields (th, prs, pi), in float32 ulps of the field maximum
# (the r4 base state itself differs by a few ulps)
ULPS = {"r8": 2.0, "r4": 8.0}
# fields whose perturbation part is what matters: compare against the
# perturbation scale, not the full value
SCALE = {"th": "thpert", "prs": "prspert", "pi": "pipert"}


def _ref_fields(ds, t):
    return {k: np.asarray(ds[k][t], dtype=np.float64) for k in FIELDS}


def _compare(fm, fr, rtol, ulps):
    bad = []
    for k in FIELDS:
        scale = np.abs(fr[SCALE.get(k, k)]).max()
        if k in ("qc", "qr"):
            scale = max(scale, 1.0e-6)
        if k == "w":
            scale = max(scale, 1.0e-3)
        err = np.abs(fm[k] - fr[k]).max()
        tol = rtol * max(scale, 1e-30)
        if k in SCALE:  # absolute rounding of the stored full field
            tol = max(tol, ulps * np.spacing(np.float32(np.abs(fr[k]).max())))
        if err > tol:
            bad.append(f"{k}: max|diff|={err:.3e} > tol={tol:.3e} (scale {scale:.3e})")
    return bad


@requires_ref
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_model_vs_cm1(backend):
    if backend != "numpy":
        pytest.importorskip(backend)
        if os.environ.get("CM1X_TEST_ALL_BACKENDS") != "1":
            pytest.skip(
                "set CM1X_TEST_ALL_BACKENDS=1 to run the (slow) non-numpy backends"
            )
    netCDF4 = pytest.importorskip("netCDF4")
    m = Model(NAMELIST, backend=backend, dtype="float64")
    rtol = RTOL[PREC]
    with netCDF4.Dataset(REF) as ds:
        assert ds["time"].shape[0] >= NSTEPS + 1
        failures = []
        for t in range(NSTEPS + 1):
            if t:
                m.step()
            assert abs(m.mtime - float(ds["time"][t])) < 1e-6
            bad = _compare(m.fields(), _ref_fields(ds, t), rtol, ULPS[PREC])
            failures += [f"t={t}: {b}" for b in bad]
    assert not failures, "\n".join(failures)


def test_model_api():
    m = Model(NAMELIST)
    assert m.cfg.nsound == 4
    f = m.fields()
    g = m.grid
    assert f["u"].shape == (g.nk, g.nj, g.ni + 1)
    assert f["v"].shape == (g.nk, g.nj + 1, g.ni)
    assert f["w"].shape == (g.nk + 1, g.nj, g.ni)
    for k in (
        "th",
        "thpert",
        "prs",
        "prspert",
        "pi",
        "pipert",
        "qv",
        "qc",
        "qr",
        "rho",
    ):
        assert f[k].shape == (g.nk, g.nj, g.ni)
