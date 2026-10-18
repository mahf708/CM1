"""Gradients of cm1x model rollouts (jax / torch autodiff) vs finite differences.

A rollout is differentiated w.r.t. two scalar parameters of the initial state
(cm1x._grad_utils.initial_state):

  amp    amplitude of the warm bubble (tha = amp * tha0, tha0 = namelist
         bubble), and
  qvamp  amplitude of a water-vapour excess in the bubble (dqv = qvamp * qv0 *
         tha0; 0.5 makes the bubble supersaturated so that saturation
         adjustment, autoconversion, accretion, rain evaporation, fallout and
         pdefq are all active within 3 steps),

and the AD derivative of a few end-of-run scalars (sum w^2, domain-mean qc,
qr, sum th'^2) is compared with central finite differences of the float64
numpy model (same forward results as jax/torch to round-off).  A dot-product
test (<u, J v> == <J^T u, v>) checks forward vs reverse mode on a field input.

Small grid (12 x 12 x 16) of the reference namelists:
  kessler_periodic   periodic, 5th-order advection, Kessler
  supercell_kessler  open radiative BCs (irbc=4), WENO5 scalars, prognostic
                     TKE (sgsmodel=1), imove=1, Kessler
  kessler_hill       terrain-following coordinate (itern=3 hill), open x
                     BCs, prognostic TKE, Kessler

"dry" below means the kessler_periodic case without the vapour excess: no
condensation happens, so only the dynamics (and the inactive microphysics
branches) carry derivatives.

The model is piecewise differentiable (clipping in pdefq / positive-definite
advection, saturation-adjustment and autoconversion switches, upwind and
limiter selections, TKE floor ...).  AD gives the derivative of the local
smooth piece; FD agrees with it once the step is below the distance to the
nearest kink, hence the small FD steps and few time steps used here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x._grad_utils import initial_state, rollout, small_namelist
from cm1x.model import Model

REF = Path(__file__).resolve().parents[2] / "reference"
CASES = {
    "kessler_periodic": REF / "kessler_periodic" / "namelist.input",
    "supercell_kessler": REF / "supercell_kessler" / "namelist.input",
    "kessler_hill": REF / "kessler_hill" / "namelist.input",
}
GRID = dict(nx=12, ny=12, nz=16)
QVAMP = 0.5
NAMES = ("w2", "qc", "qr", "th2")
_requested = os.environ.get("CM1X_BACKENDS", "numpy,jax,torch").split(",")


def _backend(name):
    if name not in _requested:
        pytest.skip(f"{name} not in CM1X_BACKENDS")
    pytest.importorskip(name)
    return name


@pytest.fixture(scope="module")
def nml(tmp_path_factory):
    d = tmp_path_factory.mktemp("grad_nml")
    return {k: small_namelist(p, d / f"{k}.input", **GRID) for k, p in CASES.items()}


_MODELS: dict = {}


def _model(nml, case, backend, **kw):
    key = (case, backend, tuple(sorted(kw.items())))
    if key not in _MODELS:
        _MODELS[key] = Model(nml[case], backend=backend, **kw)
    return _MODELS[key]


def _objective(m, nsteps, moist):
    """f(p) -> stacked end-of-run scalars, p = [amp, qvamp] (arrays of m.bk)."""
    g = m.grid
    K, J, I = g.K, g.J, g.I
    xp = m.bk.xp
    tha0 = m.state["tha"][K, J, I]
    qv0 = m.state["qa"][int(m.cfg.nqv) - 1][K, J, I]

    def f(p):
        dqv = p[1] * qv0 * tha0 if moist else None
        s = rollout(m, initial_state(m, p[0] * tha0, dqv=dqv), nsteps)
        return xp.stack(
            [
                xp.sum(s["wa"] ** 2),
                1.0e3 * xp.mean(s["qa"][1][K, J, I]),
                1.0e3 * xp.mean(s["qa"][2][K, J, I]),
                xp.sum(s["tha"][K, J, I] ** 2),
            ]
        )

    return f


def _fd_jacobian(nml, case, nsteps, moist, p0, rel_h):
    """Central-difference Jacobian of the numpy model, [len(NAMES), 2]."""
    f = _objective(_model(nml, case, "numpy"), nsteps, moist)
    cols = []
    for k in range(2 if moist else 1):
        h = rel_h * max(abs(p0[k]), 1.0e-3)
        e = np.zeros(2)
        e[k] = h
        cols.append((f(p0 + e) - f(p0 - e)) / (2 * h))
    return np.stack(cols, axis=1)


def _ad_jacobian(backend, m, nsteps, moist, p0, mode):
    f = _objective(m, nsteps, moist)
    n = 2 if moist else 1
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        p = jnp.asarray(p0)
        if mode == "rev":
            return np.asarray(jax.jacrev(f)(p))[:, :n]
        # one jax.jvp per column (cheaper to compile than jacfwd's vmap)
        cols = [jax.jvp(f, (p,), (jnp.eye(2)[k],))[1] for k in range(n)]
        return np.stack([np.asarray(c) for c in cols], axis=1)
    import torch

    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    if mode == "rev":
        y = f(p)
        rows = [
            torch.autograd.grad(y[i], p, retain_graph=i < len(NAMES) - 1)[0]
            for i in range(len(NAMES))
        ]
        return torch.stack(rows).numpy()[:, :n]
    import torch.autograd.forward_ad as fwAD

    cols = []
    with torch.no_grad(), fwAD.dual_level():
        for k in range(n):
            t = torch.zeros(2, dtype=torch.float64)
            t[k] = 1.0
            y = f(fwAD.make_dual(p.detach(), t))
            cols.append(fwAD.unpack_dual(y).tangent)
    return torch.stack(cols, axis=1).numpy()


def _check(jac_ad, jac_fd, rtol, what):
    bad = []
    for i, name in enumerate(NAMES):
        for k in range(jac_fd.shape[1]):
            ad, fd = jac_ad[i, k], jac_fd[i, k]
            scale = max(abs(fd), 1e-12 * np.abs(jac_fd[i]).max(), 1e-300)
            err = abs(ad - fd) / scale
            if not err <= rtol:  # also catches NaN
                bad.append(f"d{name}/d{('amp', 'qvamp')[k]}: ad={ad:.9e} fd={fd:.9e}")
    assert not bad, f"{what}: AD vs FD (rtol {rtol:g}):\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# torch: eager autograd (reverse) and forward-mode AD

# (case, moist, nsteps, FD step relative to the parameter, rtol)
TORCH_CASES = [
    pytest.param("kessler_periodic", False, 2, 1e-4, 1e-8, id="dry-2steps"),
    pytest.param("kessler_periodic", True, 3, 1e-5, 1e-6, id="kessler-3steps"),
    pytest.param("supercell_kessler", True, 2, 1e-5, 1e-6, id="supercell-2steps"),
    pytest.param("kessler_hill", True, 2, 1e-5, 1e-6, id="terrain-2steps"),
]


@pytest.mark.parametrize("case,moist,nsteps,rel_h,rtol", TORCH_CASES)
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_torch_grad_vs_fd(nml, case, moist, nsteps, rel_h, rtol, mode):
    _backend("torch")
    m = _model(nml, case, "torch")
    p0 = np.array([1.0, QVAMP])
    jac = _ad_jacobian("torch", m, nsteps, moist, p0, mode)
    _check(jac, _fd_jacobian(nml, case, nsteps, moist, p0, rel_h), rtol, case)


def test_torch_forward_unchanged_under_autograd(nml):
    """Backend.set/add go out of place while autograd records: same values."""
    torch = pytest.importorskip("torch")
    _backend("torch")
    m = _model(nml, "kessler_periodic", "torch")
    f = _objective(m, 2, True)
    p = torch.tensor([1.0, QVAMP], dtype=torch.float64)
    with torch.no_grad():
        y0 = f(p).numpy()
    y1 = f(p.clone().requires_grad_(True)).detach().numpy()
    np.testing.assert_array_equal(y1, y0)


def _dot_test_fn(m, nsteps):
    g = m.grid
    K, J, I = g.K, g.J, g.I
    xp = m.bk.xp
    tha0 = m.state["tha"][K, J, I]
    qv0 = m.state["qa"][int(m.cfg.nqv) - 1][K, J, I]

    def f(x):  # th' perturbation field -> (w, 1e3 qr) fields
        s = initial_state(m, tha0 + x, dqv=QVAMP * qv0 * tha0)
        s = rollout(m, s, nsteps)
        return xp.concat(
            [
                xp.reshape(s["wa"][g.KF, J, I], (-1,)),
                1e3 * xp.reshape(s["qa"][2][K, J, I], (-1,)),
            ]
        )

    return f, tuple(tha0.shape)


def test_torch_dot_product(nml):
    """<u, J v> (forward mode) == <J^T u, v> (reverse mode), Kessler case."""
    torch = pytest.importorskip("torch")
    _backend("torch")
    import torch.autograd.forward_ad as fwAD

    m = _model(nml, "kessler_periodic", "torch")
    f, shape = _dot_test_fn(m, 3)
    gen = torch.Generator().manual_seed(0)
    v = 0.1 * torch.randn(shape, generator=gen, dtype=torch.float64)
    x = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
    y = f(x)
    u = torch.randn(y.shape, generator=gen, dtype=torch.float64)
    (jtu,) = torch.autograd.grad(y, x, u)
    with torch.no_grad(), fwAD.dual_level():
        jv = fwAD.unpack_dual(
            f(fwAD.make_dual(torch.zeros(shape, dtype=torch.float64), v))
        ).tangent
    lhs, rhs = float((jtu * v).sum()), float((u * jv).sum())
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (lhs, rhs)


# ---------------------------------------------------------------------------
# jax.  Compile (jit) and eager-dispatch costs dominate on a small grid
# (~20 s to compile one step, 10-50 s for an eager reverse pass per step), so
# the default run checks forward mode through the compiled step for one step;
# CM1X_GRAD_SLOW=1 adds reverse mode, more steps, the supercell case and the
# dot-product test.

slow = pytest.mark.skipif(
    os.environ.get("CM1X_GRAD_SLOW") != "1", reason="set CM1X_GRAD_SLOW=1"
)


def _jax_jit_kw():
    import inspect

    return {"jit": True} if "jit" in inspect.signature(Model).parameters else {}


def test_jax_jvp_jit_vs_fd(nml):
    """Forward mode (jax.jacfwd) through the compiled step, Kessler case."""
    _backend("jax")
    m = _model(nml, "kessler_periodic", "jax", **_jax_jit_kw())
    p0 = np.array([1.0, QVAMP])
    jac = _ad_jacobian("jax", m, 1, True, p0, "fwd")
    fd = _fd_jacobian(nml, "kessler_periodic", 1, True, p0, 1e-5)
    _check(jac, fd, 1e-6, "kessler")


JAX_SLOW = [
    pytest.param("kessler_periodic", False, 2, 1e-4, 1e-8, id="dry-2steps"),
    pytest.param("kessler_periodic", True, 3, 1e-5, 1e-6, id="kessler-3steps"),
    pytest.param("supercell_kessler", True, 2, 1e-5, 1e-6, id="supercell-2steps"),
    pytest.param("kessler_hill", True, 2, 1e-5, 1e-6, id="terrain-2steps"),
]


@slow
@pytest.mark.parametrize("case,moist,nsteps,rel_h,rtol", JAX_SLOW)
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_jax_grad_eager_vs_fd(nml, case, moist, nsteps, rel_h, rtol, mode):
    """jax.jacrev / jax.jacfwd through the eager (un-jitted) step."""
    _backend("jax")
    m = _model(nml, case, "jax")
    p0 = np.array([1.0, QVAMP])
    jac = _ad_jacobian("jax", m, nsteps, moist, p0, mode)
    _check(jac, _fd_jacobian(nml, case, nsteps, moist, p0, rel_h), rtol, case)


@slow
def test_jax_dot_product(nml):
    """<u, J v> (jax.jvp) == <J^T u, v> (jax.vjp), Kessler case, eager."""
    _backend("jax")
    import jax
    import jax.numpy as jnp

    m = _model(nml, "kessler_periodic", "jax")
    f, shape = _dot_test_fn(m, 2)
    rng = np.random.default_rng(0)
    v = jnp.asarray(0.1 * rng.standard_normal(shape))
    x0 = jnp.zeros(shape)
    y, jv = jax.jvp(f, (x0,), (v,))
    u = jnp.asarray(rng.standard_normal(y.shape))
    _, pullback = jax.vjp(f, x0)
    (jtu,) = pullback(u)
    lhs, rhs = float(jnp.sum(jtu * v)), float(jnp.sum(u * jv))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (lhs, rhs)


def test_jax_fori_loop_eager_grad():
    """A data-dependent trip count (kessler.fallout's nfall) stays
    reverse-differentiable under eager jax.grad: only the carry is traced
    there, so Backend.fori_loop runs a static loop."""
    _backend("jax")
    import jax

    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    xp = bk.xp

    def f(x):
        n = xp.max(xp.floor(xp.abs(x) + 2.0))  # concrete (zero derivative)
        return bk.fori_loop(n, lambda i, c: c * x, xp.asarray(1.0))

    assert abs(float(jax.grad(f)(1.5)) - 3 * 1.5**2) < 1e-12


def test_jax_fori_loop_jit_grad():
    """Under jax.jit a data-dependent trip count is a lax.while_loop (not
    reverse-differentiable); with a static bound ``max_n`` Backend.fori_loop
    runs that many masked iterations instead, which jax.grad handles (this is
    what kessler.fallout does with cfg ``nfall_max``)."""
    _backend("jax")
    import jax

    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    xp = bk.xp

    @jax.jit
    def f(x):
        n = xp.max(xp.floor(xp.abs(x) + 2.0))  # traced under jit
        return bk.fori_loop(n, lambda i, c: c * x, xp.asarray(1.0), max_n=8)

    assert abs(float(f(1.5)) - 1.5**3) < 1e-12
    assert abs(float(jax.grad(f)(1.5)) - 3 * 1.5**2) < 1e-12
