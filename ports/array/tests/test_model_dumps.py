"""Step-by-step comparison with CM1's float64 state dumps (cm1dump module on
the Kokkos branch, -DDP build), shared with the C++/Kokkos port.

Needs CM1X_DUMP_DIR pointing at the kessler_periodic dump directory (with
step_000000 ... step_000010).  Interior points only: CM1 leaves the outermost
halo of the a-arrays stale, which the port fills.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.dump import read_dump
from cm1x.model import Model

DUMP_DIR = os.environ.get("CM1X_DUMP_DIR")
NAMELIST = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "kessler_periodic"
    / "namelist.input"
)
FIELDS = ("ua", "va", "wa", "tha", "ppi", "prs", "rho")
QNAMES = ("qv", "qc", "qr")
INTERIOR = (slice(1, -1), slice(3, -3), slice(3, -3))


def _tol(step):
    # Round-off for the first steps; from step 5 differences grow to ~1e-8
    # (two Fortran builds with different flags differ by the same amount).
    return 1e-10 if step <= 4 else 1e-6


@pytest.mark.reference
@pytest.mark.skipif(
    not DUMP_DIR, reason="set CM1X_DUMP_DIR to the kessler_periodic dumps"
)
def test_steps_vs_cm1_dumps(bk):
    if bk.name != "numpy" and os.environ.get("CM1X_TEST_ALL_BACKENDS") != "1":
        pytest.skip("set CM1X_TEST_ALL_BACKENDS=1 to run the (slow) non-numpy backends")
    m = Model(NAMELIST, backend=bk.name, fortran_dp=True)
    steps = sorted(p for p in Path(DUMP_DIR).glob("step_*") if p.name != "step_000000")
    assert steps, f"no dumps in {DUMP_DIR}"
    for p in steps[:10]:
        n = int(p.name.split("_")[1])
        while m.nstep < n:
            m.step()
        d, _ = read_dump(p)
        s = m.state
        qa = s["qa"] if isinstance(s["qa"], list) else list(s["qa"])
        pairs = [(k, s[k], d[k]) for k in FIELDS]
        pairs += [(q, qa[i], d["qa_" + q]) for i, q in enumerate(QNAMES)]
        for name, a, b in pairs:
            a = bk.to_numpy(a)[INTERIOR]
            b = b[INTERIOR]
            scale = np.abs(b).max()
            if scale == 0.0:
                assert np.abs(a).max() == 0.0, (name, n)
                continue
            err = np.abs(a - b).max() / scale
            assert err < _tol(n), f"{name} at step {n}: {err:.2e}"
