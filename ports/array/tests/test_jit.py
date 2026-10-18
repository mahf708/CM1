"""The compiled time step (``Model(..., jit=True)``: jax.jit of the whole
step; torch.compile of the hot routines, Backend.kernel) reproduces the
eager step to round-off.

Runs a few steps of both reference cases on a small grid (nx=ny=12, nz=16)
eagerly and compiled on the same backend and compares every output field.
The compiled program evaluates the same operations (fixed-trip / masked
loops, ``Backend.fori_loop`` / ``Backend.scan`` for the data-dependent and
vertical loops), so only floating-point re-association by the compiler is
allowed for: max |diff| <= RTOL * max |field|.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.model import Model

REF = Path(__file__).resolve().parents[2] / "reference"
CASES = ("kessler_periodic", "supercell_kessler")
SMALL = {"nx": 12, "ny": 12, "nz": 16}
NSTEPS = 3
RTOL = 1.0e-10
# torch.compile of the hot routines takes ~3 min per case with a cold inductor
# cache (~30 s warm); opt in with CM1X_TEST_TORCH_COMPILE=1
TORCH_COMPILE = os.environ.get("CM1X_TEST_TORCH_COMPILE", "0") == "1"


def _run(case, backend, jit):
    m = Model(REF / case / "namelist.input", backend=backend, jit=jit, overrides=SMALL)
    m.run(NSTEPS)
    return m


@pytest.mark.parametrize("case", CASES)
def test_jit_matches_eager(bk, case):
    if bk.name == "numpy":
        pytest.skip("numpy has no compiler (jit=True runs the same function)")
    if bk.name == "torch" and not TORCH_COMPILE:
        pytest.skip("set CM1X_TEST_TORCH_COMPILE=1 to test torch.compile")
    me = _run(case, bk.name, jit=False)
    mj = _run(case, bk.name, jit=True)
    assert mj.nstep == me.nstep == NSTEPS
    assert abs(mj.mtime - me.mtime) == 0.0
    fe, fj = me.fields(), mj.fields()
    assert fe.keys() == fj.keys()
    bad = []
    for k in fe:
        scale = max(float(np.abs(fe[k]).max()), 1.0e-30)
        err = float(np.abs(fe[k] - fj[k]).max())
        if not err <= RTOL * scale:
            bad.append(f"{k}: max|diff|={err:.3e} (max|f|={scale:.3e})")
    assert not bad, f"{case} {bk.name} jit vs eager after {NSTEPS} steps: " + "; ".join(
        bad
    )


def test_numpy_jit_flag_is_eager_path():
    """jit=True on numpy runs the same pure step function: bitwise equal."""
    case = "supercell_kessler"
    fe = _run(case, "numpy", jit=False).fields()
    fj = _run(case, "numpy", jit=True).fields()
    for k in fe:
        np.testing.assert_array_equal(fe[k], fj[k], err_msg=k)


def test_jax_jit_reverse_grad_with_static_nfall_max():
    """Reverse-mode AD through Model(jit=True): with a static sedimentation
    bound (cfg nfall_max) every loop of the compiled step is fixed-trip, so
    jax.grad works and matches the eager gradient to round-off."""
    if "jax" not in os.environ.get("CM1X_BACKENDS", "numpy,jax,torch").split(","):
        pytest.skip("jax not in CM1X_BACKENDS")
    if os.environ.get("CM1X_GRAD_SLOW", "0") != "1":
        pytest.skip("set CM1X_GRAD_SLOW=1 (compiles the step under jax.grad, ~3 min)")
    jax = pytest.importorskip("jax")
    from cm1x._grad_utils import initial_state, rollout

    nml = REF / "kessler_periodic" / "namelist.input"
    out = {}
    for jit in (False, True):
        m = Model(nml, backend="jax", jit=jit, overrides=dict(SMALL, nfall_max=4))
        g = m.grid
        tha0 = m.state["tha"][g.K, g.J, g.I]
        qv = m.state["qa"][0][g.K, g.J, g.I]

        def loss(a, m=m, tha0=tha0, qv=qv):
            s = rollout(m, initial_state(m, a * tha0, dqv=0.5 * qv * tha0), 2)
            return (s["wa"] ** 2).sum() + 1.0e3 * s["qa"][2].mean()

        out[jit] = float(jax.grad(loss)(1.0))
    assert abs(out[True] - out[False]) <= 1.0e-10 * abs(out[False]), out
