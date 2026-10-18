"""Whole-model check of cm1x.model.Model against CM1 for the reference case
ports/reference/supercell_kessler (40x40x40, open radiative lateral
boundaries irbc=4, WENO scalar advection advwenos=2, prognostic-TKE LES
closure sgsmodel=1 with ramp_sgs, idiss=1, imove=1, Kessler).

Needs double-precision (-fdefault-real-8) Fortran runs of exactly that
namelist; their fields are stored as float32 in cm1out.nc, so only that
rounding is tolerated:

  CM1X_REF_DIR_SC       run with tapfrq=6 (t = 0, 6, ..., 60 s; 10 steps)
  CM1X_REF_DIR_SC_LONG  run with the namelist's tapfrq=300 (30 min; numpy only)

    CM1X_REF_DIR_SC=/tmp/sc_r8_short CM1X_REF_DIR_SC_LONG=/tmp/sc_r8_long \\
        pytest ports/array/tests/test_model_supercell.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.model import Model

NAMELIST = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "supercell_kessler"
    / "namelist.input"
)
REF_DIR = os.environ.get("CM1X_REF_DIR_SC")
REF_DIR_LONG = os.environ.get("CM1X_REF_DIR_SC_LONG")
NSTEPS = 10

# fields in the supercell cm1out.nc (no thpert/prspert/rho in this output)
FIELDS = ("u", "v", "w", "th", "prs", "qv", "qc", "qr")
TURB = ("tke", "kmh", "kmv", "khh", "khv")
# full fields: tolerance in float32 ulps of the field maximum
FULL = ("th", "prs")
ULPS = 2.0
# everything else: fraction of max|ref| (observed <= 6e-8 over 30 minutes)
RTOL = 1.0e-6
# floors for fields that start at (or near) zero
FLOOR = {"w": 1.0e-3, "qc": 1.0e-6, "qr": 1.0e-6, "tke": 1.0e-6}


def _compare(fm, ds, t, names):
    bad = []
    for k in names:
        r = np.asarray(ds[k][t], dtype=np.float64)
        err = np.abs(fm[k] - r).max()
        if k in FULL:
            tol = ULPS * float(np.spacing(np.float32(np.abs(r).max())))
        else:
            tol = RTOL * max(np.abs(r).max(), FLOOR.get(k, 1.0e-12))
        if not err <= tol:
            bad.append(f"{k}: max|diff|={err:.3e} > tol={tol:.3e}")
    return bad


def test_supercell_model_setup():
    m = Model(NAMELIST)
    g = m.grid
    assert (g.ni, g.nj, g.nk) == (40, 40, 40)
    assert m.les and m.cfg.nsound == 4
    # open edges: the boundary-normal velocity is not stepped by the
    # generic small-step update (ssopenbc* does it)
    assert float(m.dtu0[g.J, 3].max()) == 0.0
    assert float(m.dtv0[3, g.I].max()) == 0.0
    f = m.fields()
    for k in TURB:
        assert f[k].shape == (g.nk + 1, g.nj, g.ni)
    assert float(np.abs(f["tke"]).max()) == 0.0


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR, reason="set CM1X_REF_DIR_SC to the 10-step supercell reference"
)
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_supercell_10steps_vs_cm1(backend):
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
        for t in range(NSTEPS + 1):
            if t:
                m.step()
            assert abs(m.mtime - float(ds["time"][t])) < 1e-6
            bad = _compare(m.fields(), ds, t, FIELDS + TURB)
            failures += [f"t={t}: {b}" for b in bad]
    assert not failures, "\n".join(failures)


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR_LONG,
    reason="set CM1X_REF_DIR_SC_LONG to the 30-minute supercell reference",
)
@pytest.mark.skipif(
    "numpy" not in os.environ.get("CM1X_BACKENDS", "numpy"), reason="numpy-only test"
)
def test_supercell_30min_vs_cm1():
    netCDF4 = pytest.importorskip("netCDF4")
    m = Model(NAMELIST, backend="numpy")
    failures = []
    with netCDF4.Dataset(Path(REF_DIR_LONG) / "cm1out.nc") as ds:
        times = np.asarray(ds["time"][:])
        for it in range(1, len(times)):
            m.run(int(round((times[it] - times[it - 1]) / m.dt)))
            assert abs(m.mtime - float(times[it])) < 1e-6
            f = m.fields()
            bad = _compare(f, ds, it, FIELDS + TURB)
            failures += [f"t={times[it]:.0f}s: {b}" for b in bad]
    assert not failures, "\n".join(failures)
    # a storm with rain and resolved turbulence must have developed, or the
    # open boundaries / WENO / TKE code went untested
    assert f["w"].max() > 10.0 and f["qr"].max() > 1e-3 and f["tke"].max() > 1.0
