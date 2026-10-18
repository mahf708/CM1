import numpy as np

from cm1x.bc import periodic_xy
from cm1x.grid import Grid, fi, fk


def test_index_map():
    assert fi(1) == 3 and fi(-2) == 0 and fk(1) == 1 and fk(0) == 0


def test_set_add(bk):
    a = bk.zeros((3, 4))
    a = bk.set(a, (1, slice(None)), 2.0)
    a = bk.add(a, (1, 0), 1.0)
    out = bk.to_numpy(a)
    assert out[1, 0] == 3.0 and out[1, 1] == 2.0 and out[0].sum() == 0


def test_periodic(bk):
    g = Grid(ni=5, nj=4, nk=2, dx=1.0, dy=1.0, dz=1.0)
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(g.s_shape)
    a = periodic_xy(bk, bk.asarray(raw), g)
    out = bk.to_numpy(a)
    core = raw[:, g.J, g.I]
    ref = np.pad(core, ((0, 0), (3, 3), (3, 3)), mode="wrap")
    np.testing.assert_array_equal(out, ref)
    # u-staggered: u(ni+1) == u(1)
    rawu = rng.standard_normal(g.u_shape)
    u = bk.to_numpy(periodic_xy(bk, bk.asarray(rawu), g, stag_x=True))
    np.testing.assert_array_equal(u[:, :, 3 + g.ni], u[:, :, 3])


def test_fori_loop_until(bk):
    """``until`` stops an eager loop early; same result as the full loop when
    the body is a no-op once ``until`` holds."""
    xp = bk.xp

    def body(i, c):
        x, active = c
        x = xp.where(active, x - 1.0, x)
        return x, active & (x > 0.0)

    x0 = bk.asarray(np.array([3.0, 1.0, 0.5]))
    c0 = (x0, x0 > 0.0)
    calls = []

    def until(c):
        calls.append(1)
        return ~xp.any(c[1])

    full = bk.fori_loop(20, body, c0)
    early = bk.fori_loop(20, body, c0, until=until)
    np.testing.assert_array_equal(bk.to_numpy(full[0]), bk.to_numpy(early[0]))
    assert len(calls) == 4  # stopped after 3 iterations


def test_kernel_is_identity_without_kernels(bk):
    def f(x, *, dt):
        return x * dt

    assert bk.kernel(f, traced=("dt",)) is f


def test_torch_kernel_compiles_and_matches():
    """Backend.kernel on torch with kernels set: torch.compile of the routine
    (``traced`` floats become inputs: no recompile for a new value)."""
    import dataclasses
    import os

    import pytest

    if "torch" not in os.environ.get("CM1X_BACKENDS", "numpy,jax,torch").split(","):
        pytest.skip("torch not in CM1X_BACKENDS")
    torch = pytest.importorskip("torch")
    from cm1x.backend import get_backend, torch_kernel_options

    bk = get_backend("torch")
    bkc = dataclasses.replace(bk, kernels=torch_kernel_options())

    def f(bk, x, *, dt, flag):
        y = bk.set(bk.copy(x), (slice(1, 3),), x[1:3] * dt + 1.0)
        return y if flag else -y

    x = torch.arange(5.0, dtype=torch.float64)
    k = bkc.kernel(f, traced=("dt",))
    assert k is not f and bkc.kernel(f, traced=("dt",)) is k
    for dt in (0.5, 0.25):
        for flag in (True, False):
            ref = f(bk, x, dt=dt, flag=flag)
            assert torch.equal(k(bkc, x, dt=dt, flag=flag), ref)
