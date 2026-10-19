"""Whole-model comparison with CM1's float64 state dumps for the boundary
layer and surface-layer cases (cm1setup=2 PBL schemes, LES with a surface
layer, SLAB), built by the Kokkos branch's cxx/scripts/make_refs.sh.

Needs CM1X_DUMP_DIR_PBL pointing at a directory with one subdirectory per
case (``<case>/step_NNNNNN``, every 10 steps), each with its namelist.input
(and LANDUSE.TBL); a case's namelist and LANDUSE.TBL may also come from
ports/reference/<case>/.  Every array present in both the dump and the
model state is compared on the interior:

* prognostics (PROGNOSTIC below, and qa_*) against the state after step N;
* everything else (surface arrays, K, PBL tendencies, ...) against the
  state after step N-1, because CM1 dumps before ``sfc_and_turb`` and cm1x
  runs it at the end of a step.  At step 0 only the prognostics are
  compared (CM1's surface arrays are still the allocation defaults).

Each case runs in a subprocess with numpy's AVX-512 paths off (glibc libm
for exp/log/pow, as the Fortran uses).  CM1X_PBL_MAX_STEP caps the steps.
Add a case: append its name to CASES (and any extra prognostic to
PROGNOSTIC).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DUMP_DIR = os.environ.get("CM1X_DUMP_DIR_PBL")
REF_DIR = Path(__file__).resolve().parents[2] / "reference"

CASES = [
    "pbl2_dry",
    "pbl2_kessler",
    "pbl2_sfclay_land",
    "cbl_sfc",
    "cbl_sfc_moist",
]
PROGNOSTIC = ("ua", "va", "wa", "tha", "ppi", "prs", "rho", "tkea", "qke")
TOL = 1e-10  # max |cm1x - cm1| / max |cm1| per array and step
NPY_DISABLE = (
    "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR X86_V4"
)

_DRIVER = r"""
import json, sys
from pathlib import Path
import numpy as np
from cm1x.dump import read_dump
from cm1x.model import Model

case_dir, namelist, backend, max_step = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
prognostic = set(json.loads(sys.argv[5]))
m = Model(namelist, backend=backend, fortran_dp=True)
bk, g = m.bk, m.grid
n = g.ngxy

def host(v):
    if isinstance(v, (list, tuple)):
        return [np.array(bk.to_numpy(x)) for x in v]
    return np.array(bk.to_numpy(v)) if hasattr(v, "shape") else None

def flat(s):
    out = {}
    for k, v in s.items():
        if k == "qa":
            for i, a in enumerate(host(v)):
                out[i] = a
            continue
        a = host(v)
        if isinstance(a, np.ndarray) and a.ndim in (2, 3):
            out[k] = a
    return out

qnames = {}
for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
    i = int(m.cfg.get("nq" + name[1], 0) or 0)
    if i >= 1:
        qnames["qa_" + name] = i - 1

def inner(a):
    if a.ndim == 2:
        a = a[None]
    return a[1:-1, n:-n, n:-n] if a.shape[0] > 1 and a.shape[0] in (g.nk + 2, g.nk + 3) else a[:, n:-n, n:-n]

steps = sorted(int(p.name.split("_")[1]) for p in case_dir.glob("step_*"))
steps = [s for s in steps if s <= max_step]
res = {"steps": steps, "errors": [], "compared": {}, "missing": []}
prev = None
for step in steps:
    while m.nstep < step:
        if m.nstep == step - 1:
            prev = flat(m.state)
        m.step()
    cur = flat(m.state)
    d, _ = read_dump(case_dir / f"step_{step:06d}")
    for k, b in d.items():
        key = qnames.get(k, k)
        isprog = k in prognostic or k.startswith("qa_")
        if not isprog and step == 0:
            continue
        src = cur if isprog else prev
        if key not in src:
            if k not in res["missing"]:
                res["missing"].append(k)
            continue
        a = src[key]
        if a.ndim == 2:
            a = a[None]
        if a.shape != b.shape:
            res["errors"].append([k, step, f"shape {a.shape} != {b.shape}"])
            continue
        ai, bi = inner(a), inner(b)
        sc = float(np.abs(bi).max())
        e = float(np.abs(ai - bi).max()) / (sc if sc > 0 else 1.0)
        res["compared"][k] = max(res["compared"].get(k, 0.0), e)
print("RESULT" + json.dumps(res))
"""


def _case_files(case):
    d = Path(DUMP_DIR) / case
    nml = d / "namelist.input"
    if not nml.exists():
        nml = REF_DIR / case / "namelist.input"
    lu = d / "LANDUSE.TBL"
    if not lu.exists():
        lu = REF_DIR / case / "LANDUSE.TBL"
    return d, nml, lu


@pytest.mark.reference
@pytest.mark.skipif(
    not DUMP_DIR, reason="set CM1X_DUMP_DIR_PBL to the PBL dump directory"
)
@pytest.mark.parametrize("case", CASES)
def test_pbl_case_vs_cm1_dumps(case, bk, tmp_path):
    if bk.name != "numpy" and os.environ.get("CM1X_TEST_ALL_BACKENDS") != "1":
        pytest.skip("set CM1X_TEST_ALL_BACKENDS=1 to run the (slow) non-numpy backends")
    d, nml, lu = _case_files(case)
    if not d.is_dir() or not any(d.glob("step_*")):
        pytest.skip(f"no dumps for {case} in {DUMP_DIR}")
    assert nml.exists(), f"no namelist.input for {case}"
    env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NPY_DISABLE)
    if lu.exists():
        env["CM1X_LANDUSE_TBL"] = str(lu)
    max_step = os.environ.get("CM1X_PBL_MAX_STEP", "1000000000")
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            _DRIVER,
            str(d),
            str(nml),
            bk.name,
            max_step,
            json.dumps(PROGNOSTIC),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert p.returncode == 0 and lines, p.stderr[-4000:]
    res = json.loads(lines[-1][len("RESULT") :])
    assert res["steps"], f"no steps compared for {case}"
    bad = {k: e for k, e in res["compared"].items() if not e <= TOL}
    assert not res["errors"], res["errors"]
    assert not bad, f"{case}: max relative error over steps {res['steps'][-1]}: {bad}"
    assert any(k in res["compared"] for k in ("ua", "tha")), res
