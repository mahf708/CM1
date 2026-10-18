"""Whole-model check of the terrain-following coordinate against CM1 for the
reference case ports/reference/nh_mountain_waves (run/config_files copy with
netCDF output: 2-D x-z, 200 m grid, bell hill itern=1, isnd=9 / iwnd=6,
icor=1 with lspgrad=1, open radiative x boundaries, WENO scalars, Rayleigh
damping, dry, no turbulence).

Needs double-precision (-fdefault-real-8) Fortran runs of that namelist; the
fields are stored as float32, so only that rounding is tolerated:

  CM1X_REF_DIR_NHMW       timax=20, tapfrq=2 (an output every step; 10 steps)
  CM1X_REF_DIR_NHMW_LONG  the namelist as is (36 min, an output every 108
                          steps; numpy only)

    CM1X_REF_DIR_NHMW=/tmp/nhmw_r8 pytest ports/array/tests/test_model_terrain.py
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
    / "nh_mountain_waves"
    / "namelist.input"
)
REF_DIR = os.environ.get("CM1X_REF_DIR_NHMW")
REF_DIR_LONG = os.environ.get("CM1X_REF_DIR_NHMW_LONG")
NSTEPS = 10
# fields written by this namelist (output_pipert, output_rho; no thpert / q's)
FIELDS = ("u", "v", "w", "th", "prs", "pipert", "rho")
# full fields: tolerance in float32 ulps of the field maximum
FULL = ("th", "prs", "rho")
ULPS = 2.0
# everything else: fraction of max|ref| (observed <= 4e-8 after 10 steps)
RTOL = 1.0e-6
FLOOR = {"v": 1.0e-6, "pipert": 1.0e-9}


def _compare(fm, ds, t, rtol=RTOL):
    bad = []
    for k in FIELDS:
        r = np.asarray(ds[k][t], dtype=np.float64)
        err = np.abs(fm[k] - r).max()
        if k in FULL:
            tol = ULPS * float(np.spacing(np.float32(np.abs(r).max())))
        else:
            tol = rtol * max(np.abs(r).max(), FLOOR.get(k, 1.0e-12))
        if not err <= tol:
            bad.append(f"{k}: max|diff|={err:.3e} > tol={tol:.3e}")
    return bad


def test_nh_mountain_waves_setup():
    m = Model(NAMELIST)
    g = m.grid
    assert (g.ni, g.nj, g.nk) == (100, 1, 100)
    assert m.terrain and not m.moist and not m.les
    zs = np.asarray(m.bk.to_numpy(m.base["zs"]))
    assert abs(zs.max() - 400.0) < 1.0  # bell-shaped hill, h = 400 m
    # terrain-following levels: the lowest level follows the surface
    zh = np.asarray(m.bk.to_numpy(m.base["zh"]))
    np.testing.assert_allclose(zh[1] - zs, 100.0 * (1 - zs / 20000.0), rtol=1e-12)
    # the kinematic lower boundary condition holds at t = 0
    f = m.fields()
    assert np.abs(f["w"][0]).max() > 0.1


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR, reason="set CM1X_REF_DIR_NHMW to the 10-step nh_mountain_waves run"
)
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_nh_mountain_waves_vs_cm1(backend):
    if backend != "numpy":
        pytest.importorskip(backend)
        if os.environ.get("CM1X_TEST_ALL_BACKENDS") != "1":
            pytest.skip("set CM1X_TEST_ALL_BACKENDS=1 for the non-numpy backends")
    netCDF4 = pytest.importorskip("netCDF4")
    m = Model(NAMELIST, backend=backend, dtype="float64")
    with netCDF4.Dataset(Path(REF_DIR) / "cm1out.nc") as ds:
        assert ds["time"].shape[0] >= NSTEPS + 1
        failures = []
        for t in range(NSTEPS + 1):
            if t:
                m.step()
            assert abs(m.mtime - float(ds["time"][t])) < 1e-6
            failures += [f"t={t}: {b}" for b in _compare(m.fields(), ds, t)]
    assert not failures, "\n".join(failures)


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR_LONG, reason="set CM1X_REF_DIR_NHMW_LONG to the 36-min run"
)
@pytest.mark.skipif(
    "numpy" not in os.environ.get("CM1X_BACKENDS", "numpy"), reason="numpy-only test"
)
def test_nh_mountain_waves_long_vs_cm1():
    netCDF4 = pytest.importorskip("netCDF4")
    m = Model(NAMELIST, backend="numpy")
    with netCDF4.Dataset(Path(REF_DIR_LONG) / "cm1out.nc") as ds:
        times = np.asarray(ds["time"][:])
        failures = []
        for it in range(1, len(times)):
            m.run(int(round((times[it] - times[it - 1]) / m.dt)))
            rtol = LONG_RTOL if times[it] <= T_SENSITIVE else LATE_RTOL
            bad = _compare(m.fields(), ds, it, rtol=rtol)
            failures += [f"t={times[it]:.0f}s: {b}" for b in bad]
        # mountain waves must actually develop aloft
        assert np.abs(np.asarray(ds["w"][-1])[60:]).max() > 0.1
    assert not failures, "\n".join(failures)


# Up to t = 1728 s only the float32 rounding of the file remains (<= 5e-8 of
# max|ref|).  Later, once the waves reach the east (open, irbc=4) boundary,
# CM1's radiative phase speed radbcew4 -- a ratio of two differences that
# are both near round-off there -- amplifies round-off: perturbing the
# velocities by one ulp at t = 1080 s changes u(ni+1) and w(ni) by 3e-8
# m/s at t = 1836 s and 1.6e-5 m/s at t = 2160 s (~10x per 108 steps), and
# cm1x vs CM1 differ by 1.4e-4 m/s (1e-5 of max|u|) at that same point at
# t = 2160 s.
T_SENSITIVE = 1728.0
LONG_RTOL = 1.0e-6
LATE_RTOL = 1.0e-3


@pytest.mark.parametrize("backend", ["jax"])
def test_nh_mountain_waves_jit_matches_eager(backend):
    pytest.importorskip(backend)
    ov = {"nx": 24}
    me = Model(NAMELIST, backend=backend, overrides=ov)
    mj = Model(NAMELIST, backend=backend, overrides=ov, jit=True)
    me.run(3)
    mj.run(3)
    fe, fj = me.fields(), mj.fields()
    for k in fe:
        scale = max(float(np.abs(fe[k]).max()), 1.0e-30)
        err = float(np.abs(fe[k] - fj[k]).max())
        assert err <= 1.0e-10 * scale, f"{k}: {err:.3e} (max {scale:.3e})"
