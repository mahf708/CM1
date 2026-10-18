"""supercell_kessler (open radiative lateral boundaries, irbc=4) vs CM1's
float64 state dumps (cm1dump, -DDP build; see test_model_dumps.py).

Needs CM1X_DUMP_DIR_SC pointing at a supercell_kessler dump directory
(step_000000 plus any later steps, e.g. CM1_DUMP_EVERY=5 over 10 steps).

The radiative phase speeds (bc.F radbcew4/radbcns4: a clipped ratio of two
small velocity differences) amplify any last-bit difference by orders of
magnitude within a few steps: a one-ulp perturbation of ua/va grows to
~1e-9 relative by step 5 and ~1e-7 by step 10. So this case only stays at
round-off if cm1x reproduces CM1 bit for bit, including

* the summation order of the apmasscon dry-mass/pressure sums
  (misc.level_sums / misc.qp_sum; numpy's pairwise sum drifted to 4e-8);
* libm: numpy's AVX-512 SIMD pow/exp/log differ from glibc's in the last
  bit, so the model runs in a subprocess with those dispatch targets
  disabled (NPY_DISABLE_CPU_FEATURES) and calls glibc like the Fortran.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DUMP_DIR = os.environ.get("CM1X_DUMP_DIR_SC")
NAMELIST = (
    Path(__file__).resolve().parents[2]
    / "reference"
    / "supercell_kessler"
    / "namelist.input"
)
TOL = 1e-10
NPY_NO_AVX512 = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)

_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from cm1x.dump import read_dump
from cm1x.model import Model

nml, ddir = sys.argv[1], Path(sys.argv[2])
F = ("ua", "va", "wa", "tha", "ppi", "prs", "rho")
Q = ("qv", "qc", "qr")
I = (slice(1, -1), slice(3, -3), slice(3, -3))
steps = sorted(int(p.name.split("_")[1]) for p in ddir.glob("step_*"))
steps = [n for n in steps if n > 0][:4]
m = Model(nml, backend="numpy", fortran_dp=True)
out = {}
for n in steps:
    while m.nstep < n:
        m.step()
    d, _ = read_dump(ddir / f"step_{n:06d}")
    s = m.state
    pairs = [(k, s[k], d[k]) for k in F]
    pairs += [(q, s["qa"][i], d["qa_" + q]) for i, q in enumerate(Q)]
    errs = {}
    for name, a, b in pairs:
        a, b = np.asarray(a)[I], b[I]
        sc = np.abs(b).max()
        errs[name] = float(np.abs(a - b).max() / sc) if sc > 0 else float(np.abs(a).max())
    out[n] = errs
print(json.dumps(out))
"""


@pytest.mark.reference
@pytest.mark.skipif(
    not DUMP_DIR, reason="set CM1X_DUMP_DIR_SC to the supercell_kessler dumps"
)
def test_supercell_openbc_vs_cm1_dumps(bk):
    if bk.name != "numpy":
        # jax/torch use their own exp/pow/log (not glibc's): the last-bit
        # differences are then amplified by the radiative boundaries
        pytest.skip("bitwise libm parity is only available with numpy")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_NO_AVX512)
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(NAMELIST), DUMP_DIR],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res, f"no dumps after step 0 in {DUMP_DIR}"
    bad = [
        f"{name} at step {n}: {e:.2e}"
        for n, errs in res.items()
        for name, e in errs.items()
        if not e < TOL
    ]
    assert not bad, "; ".join(bad)
