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
