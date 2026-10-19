"""Whole-model runs with Morrison microphysics (ptype=5) vs CM1's float64
state dumps (cm1dump, -DDP build; see test_model_dumps.py).

Needs CM1X_DUMP_DIR_MORRISON pointing at a directory with one dump
directory per case (make_refs.sh output): morrison_bubble (32x32x20,
periodic, eqtset=2, ihail=1, 300 steps dumped every 30) and morrison_efall
(eqtset=2, efall=1, ihail=0, 200 steps dumped every 50); namelists in
ports/reference/<case>.

numpy runs in a subprocess with numpy's AVX-512 libm disabled (as
test_model_dumps_supercell.py) and reproduces CM1 to round-off; jax and
torch use their own exp/pow/log, whose last-bit differences grow through
the microphysics thresholds to ~1e-11 (relative to each field's maximum)
over 300 steps.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DUMP_DIR = os.environ.get("CM1X_DUMP_DIR_MORRISON")
REF = Path(__file__).resolve().parents[2] / "reference"
TOL = {"numpy": 1e-10, "jax": 1e-9, "torch": 1e-9}
NPY_NO_AVX512 = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)
CASES = ("morrison_bubble", "morrison_efall")

_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from cm1x.dump import read_dump
from cm1x.model import Model

nml, ddir, backend = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
F = ("ua", "va", "wa", "tha", "ppi", "prs", "rho")
I = (slice(1, -1), slice(3, -3), slice(3, -3))
steps = sorted(int(p.name.split("_")[1]) for p in ddir.glob("step_*"))
steps = [n for n in steps if n > 0]
m = Model(nml, backend=backend, fortran_dp=True, jit=backend == "jax")
Q = m.cfg.qname
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
        a = m.bk.to_numpy(a) if not isinstance(a, np.ndarray) else a
        a, b = a[I], b[I]
        sc = np.abs(b).max()
        errs[name] = float(np.abs(a - b).max() / sc) if sc > 0 else float(np.abs(a).max())
    out[n] = errs
print(json.dumps(out))
"""


@pytest.mark.reference
@pytest.mark.skipif(
    not DUMP_DIR, reason="set CM1X_DUMP_DIR_MORRISON to the Morrison dump directories"
)
@pytest.mark.parametrize("case", CASES)
def test_morrison_vs_cm1_dumps(bk, case):
    ddir = Path(DUMP_DIR) / case
    if not ddir.is_dir():
        pytest.skip(f"no dumps for {case} in {DUMP_DIR}")
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_NO_AVX512)
    if bk.name == "jax":
        env["JAX_ENABLE_X64"] = "1"
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _SCRIPT,
            str(REF / case / "namelist.input"),
            str(ddir),
            bk.name,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res, f"no dumps after step 0 in {ddir}"
    tol = TOL[bk.name]
    bad = [
        f"{name} at step {n}: {e:.2e}"
        for n, errs in res.items()
        for name, e in errs.items()
        if not e < tol
    ]
    assert not bad, "; ".join(bad)
