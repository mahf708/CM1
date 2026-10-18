"""30-minute kessler_periodic run (300 steps; a storm with cloud and rain
develops) against the double-precision CM1 run, every 5 minutes.

Needs CM1X_REF_DIR_LONG pointing at that run's cm1out.nc (tapfrq=300). Runs on
numpy only: the other backends give identical results in the 10-step test and
are much slower eagerly.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.model import Model

REF_DIR = os.environ.get("CM1X_REF_DIR_LONG")
NAMELIST = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "kessler_periodic"
    / "namelist.input"
)
FIELDS = ("u", "v", "w", "thpert", "prspert", "qv", "qc", "qr", "rho")


@pytest.mark.reference
@pytest.mark.skipif(
    not REF_DIR, reason="set CM1X_REF_DIR_LONG to the 30-minute reference run"
)
@pytest.mark.skipif(
    "numpy" not in os.environ.get("CM1X_BACKENDS", "numpy"), reason="numpy-only test"
)
def test_storm_30min_vs_cm1():
    netCDF4 = pytest.importorskip("netCDF4")
    ref = netCDF4.Dataset(Path(REF_DIR) / "cm1out.nc")
    times = np.asarray(ref["time"][:])
    m = Model(NAMELIST, backend="numpy")
    for it in range(1, len(times)):
        m.run(int(round((times[it] - times[it - 1]) / m.dt)))
        f = m.fields()
        for v in FIELDS:
            b = np.asarray(ref[v][it], dtype=np.float64)
            scale = max(np.abs(b).max(), 1e-30)
            err = np.abs(f[v] - b).max() / scale
            # the reference is stored as float32: allow a few float32 ulps
            assert err < 1e-6, (
                f"{v} at t={times[it]:.0f}s: max|diff|/max|ref| = {err:.2e}"
            )
    # the run must actually produce cloud and rain, or Kessler went untested
    assert np.asarray(ref["qr"][-1]).max() > 1e-3 and f["qr"].max() > 1e-3
