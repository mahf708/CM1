"""Whole runs of CM1's idealized test cases (testcase 3-14 forcing, lspgrad
3/4, dolsw subsidence, simple radiation, nudging) vs CM1's float64 state dumps.

Needs CM1X_TC_REFS pointing at a directory of reference cases made by the
cm1k scripts (cxx/scripts/make_refs.sh <dir> tc3_nosfc tc4_nosfc ...), each
case holding its namelist.input and step_NNNNNN dumps. The model runs in a
subprocess with numpy's AVX-512 libm dispatch disabled (glibc parity, as in
test_model_dumps_supercell.py).

Step 0 is skipped: cm1x runs the startup turbulence inside init, so the
dumped tkea at step 0 is not comparable (later steps are).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REFS = os.environ.get("CM1X_TC_REFS")
NPY_NO_AVX512 = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)
# cases that cm1x can run (no surface layer); all are bitwise on numpy except
# the mm-scale wind tunnel (tc12): last-few-ulp differences from step 10 on
# (prs 1e-10 Pa, tha 1e-14 K), which is 4e-8 relative to its ~1e-7 K
# theta perturbation
CASES = {
    "tc3_nosfc": 0.0,
    "tc4_nosfc": 0.0,
    "tc5_simplerad_nosfc": 0.0,
    "tc6_nosfc": 0.0,
    "tc7_nosfc": 0.0,
    "tc9_nosfc": 0.0,
    "tc10_les": 0.0,
    "tc11_nosfc": 0.0,
    "tc12_windtunnel": 1e-7,
    "tc13_nosfc": 0.0,
    "tc14_nosfc": 0.0,
}

_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from cm1x.dump import read_dump
from cm1x.model import Model

ddir = Path(sys.argv[1])
F = ("ua", "va", "wa", "tha", "ppi", "prs", "rho", "tkea")
Q = ("qv", "qc", "qr")
I = (slice(1, -1), slice(3, -3), slice(3, -3))
steps = sorted(int(p.name.split("_")[1]) for p in ddir.glob("step_*"))
steps = [n for n in steps if n > 0][:3]
m = Model(ddir / "namelist.input", backend="numpy", fortran_dp=True)
out = {}
for n in steps:
    while m.nstep < n:
        m.step()
    d, _ = read_dump(ddir / f"step_{n:06d}")
    s = m.state
    pairs = [(k, s[k], d[k]) for k in F if k in s and k in d]
    pairs += [(q, s["qa"][i], d["qa_" + q]) for i, q in enumerate(Q) if "qa_" + q in d]
    errs = {}
    for name, a, b in pairs:
        a, b = np.asarray(a)[I], b[I]
        sc = np.abs(b).max()
        e = np.abs(a - b).max()
        errs[name] = float(e / sc) if sc > 0 else float(e)
    out[n] = errs
print(json.dumps(out))
"""


@pytest.mark.reference
@pytest.mark.skipif(not REFS, reason="set CM1X_TC_REFS to the testcase reference cases")
@pytest.mark.parametrize("case", list(CASES))
def test_testcase_vs_cm1_dumps(bk, case):
    if bk.name != "numpy":
        pytest.skip("bitwise libm parity is only available with numpy")
    ddir = Path(REFS) / case
    if not (ddir / "step_000000").exists():
        pytest.skip(f"no dumps in {ddir}")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_NO_AVX512)
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(ddir)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res, f"no dumps after step 0 in {ddir}"
    tol = CASES[case]
    bad = [
        f"{name} at step {n}: {e:.2e}"
        for n, errs in res.items()
        for name, e in errs.items()
        if not e <= tol
    ]
    assert not bad, "; ".join(bad)
