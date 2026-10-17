"""cm1x.sound.sound vs. the Fortran sound (src/sound.F), psolver=3, periodic."""

import numpy as np
import pytest

from cm1x import constants as C
from cm1x.grid import Grid
from cm1x.sound import bcp, sound

from fortran import requires_gfortran, run_driver

NI, NJ, NK = 7, 6, 12
DT, NSOUND = 6.0, 6
CFG = dict(nsound=NSOUND, alph=0.6, kdiv=0.1, smeps=1.0e-30)

OUT_S = ["pp3d", "ppd", "ppx", "fpk", "qk", "pk1", "pk2", "mm", "dum1", "dum2"]


def make_inputs(g, seed=0):
    """Stable hydrostatic base state plus random perturbations."""
    rng = np.random.default_rng(seed)
    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    zh = np.array(g.zh())
    nn = 1.0e-2
    th = lambda z: 300.0 * np.exp(nn * nn * z / C.g)  # noqa: E731
    # hydrostatic Exner function: d(pi)/dz = -g/(cp*theta)
    pis = 1.0 - C.g * C.g / (C.cp * 300.0 * nn * nn) * (
        1.0 - np.exp(-nn * nn * zh / C.g)
    )
    rhos = C.p00 * pis ** (C.cv / C.rd) / (C.rd * th(zh))
    rfs = np.empty(W[0])
    rfs[1:-1] = 0.5 * (rhos[1:] + rhos[:-1])
    rfs[0], rfs[-1] = rfs[1], rfs[-2]

    def s3(prof, shape=S):
        return np.broadcast_to(np.asarray(prof)[:, None, None], shape).copy()

    c1v = 0.5 + 0.05 * rng.standard_normal(S[0])
    d = dict(
        uh=1.0 + 0.01 * rng.standard_normal(S[2]),
        uf=1.0 + 0.01 * rng.standard_normal(U[2]),
        vh=1.0 + 0.01 * rng.standard_normal(S[1]),
        vf=1.0 + 0.01 * rng.standard_normal(V[1]),
        mh=s3(1.0 + 0.02 * rng.standard_normal(S[0])),
        mf=s3(1.0 + 0.02 * rng.standard_normal(W[0]), W),
        c1=s3(c1v),
        c2=s3(1.0 - c1v),
        rr0=s3(1.0 / rhos),
        rf0=s3(rfs[: S[0]]),
        dtu0=1.0 + 0.01 * rng.random(S[1:]),
        dtv0=1.0 + 0.01 * rng.random(S[1:]),
    )
    thv = s3(th(zh)) + 0.5 * rng.standard_normal(S)
    d["thv"] = thv
    d["ppterm"] = C.rddcv * s3(pis) * (1.0 + 1e-3 * rng.standard_normal(S))
    d["piadv"] = s3(pis) + 1e-4 * rng.standard_normal(S)
    for nm, sh, amp in [
        ("ua", U, 10.0),
        ("u3d", U, 10.0),
        ("uten", U, 1e-2),
        ("va", V, 10.0),
        ("v3d", V, 10.0),
        ("vten", V, 1e-2),
        ("wa", W, 1.0),
        ("w3d", W, 1.0),
        ("wten", W, 1e-2),
        ("ppi", S, 1e-4),
        ("pp3d", S, 1e-4),
        ("ppten", S, 1e-6),
        ("ppx", S, 1e-6),
        ("rru", U, 1.0),
        ("rrv", V, 1.0),
        ("rrw", W, 1.0),
    ]:
        d[nm] = amp * rng.standard_normal(sh)
    for nm in ("wa", "w3d"):  # rigid lids: w = 0 at k=1 and nk+1
        d[nm][1] = 0.0
        d[nm][g.nk + 1] = 0.0
    for nm in ("dtu", "dtv"):
        d[nm] = np.zeros(S[1:])
    for nm in ("ppd", "fpk", "qk", "pk1", "pk2", "mm", "dum1", "dum2"):
        d[nm] = np.zeros(S)
    return d


def out_shapes(g):
    S = g.s_shape
    o = {nm: S for nm in OUT_S}
    o.update(
        u3d=g.u_shape,
        rru=g.u_shape,
        v3d=g.v_shape,
        rrv=g.v_shape,
        w3d=g.w_shape,
        rrw=g.w_shape,
        dtu=S[1:],
        dtv=S[1:],
    )
    return o


def fortran_sound(g, d, nrk):
    params = np.array(
        [
            g.ni,
            g.nj,
            g.nk,
            NSOUND,
            nrk,
            DT,
            CFG["alph"],
            CFG["kdiv"],
            g.dx,
            g.dy,
            g.dz,
            CFG["smeps"],
        ],
        dtype=float,
    )
    return run_driver("sound", {"params": params, **d}, out_shapes(g))


def py_sound(bk, g, d, nrk):
    a = {k: bk.asarray(v) for k, v in d.items()}
    return sound(bk, g, CFG, dt=DT, nrk=nrk, **a)


@pytest.fixture(scope="module")
def grid():
    return Grid(ni=NI, nj=NJ, nk=NK, dx=1000.0, dy=1000.0, dz=500.0)


def _compare(bk, res, ref):
    assert set(res) == set(ref)
    for k, r in ref.items():
        a = bk.to_numpy(res[k])
        scale = max(np.max(np.abs(r)), 1e-300)
        np.testing.assert_allclose(a, r, rtol=1e-12, atol=1e-12 * scale, err_msg=k)


@requires_gfortran
@pytest.mark.parametrize("nrk", [1, 2, 3])
def test_sound_vs_fortran(bk, grid, nrk):
    d = make_inputs(grid, seed=nrk)
    ref = fortran_sound(grid, d, nrk)
    res = py_sound(bk, grid, d, nrk)
    _compare(bk, res, ref)


@requires_gfortran
def test_sound_jit(grid):
    pytest.importorskip("jax")
    import jax

    from cm1x import get_backend

    bk = get_backend("jax", "float64")
    d = make_inputs(grid, seed=7)
    ref = fortran_sound(grid, d, 3)
    f = jax.jit(lambda a: sound(bk, grid, CFG, dt=DT, nrk=3, **a))
    res = f({k: bk.asarray(v) for k, v in d.items()})
    _compare(bk, res, ref)


def test_bcp_periodic(bk, grid):
    rng = np.random.default_rng(3)
    p = rng.standard_normal(grid.s_shape)
    out = bk.to_numpy(bcp(bk, grid, None, bk.asarray(p)))
    K, n = slice(1, NK + 1), 3
    core = p[K, n : n + NJ, n : n + NI]
    full = np.pad(core, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    np.testing.assert_array_equal(out[K, 2 : NJ + 4, 2 : NI + 4], full)


def test_unsupported(bk, grid):
    d = make_inputs(grid)
    a = {k: bk.asarray(v) for k, v in d.items()}
    for opt in (
        dict(terrain_flag=True),
        dict(axisymm=1),
        dict(wbc=2),
        dict(convinit=1),
    ):
        with pytest.raises(NotImplementedError):
            sound(bk, grid, {**CFG, **opt}, dt=DT, nrk=1, **a)
