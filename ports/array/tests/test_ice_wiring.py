"""Generic ice-scheme wiring (param.F moisture tables, getcvm, pdefqtest)."""

import numpy as np

from cm1x import constants as C
from cm1x.grid import Grid
from cm1x.misc import getcvm
from cm1x.namelist import Config, moisture_table
from cm1x.pdef import pdefq, pdefqtest


def test_moisture_tables():
    k = moisture_table(1)
    assert (k["numq"], k["nqv"], k["nqc"], k["nqr"], k["iice"]) == (3, 1, 2, 3, 0)
    t = moisture_table(3)
    assert t["numq"] == 8 and t["iice"] == 1 and t["idm"] == 1
    assert (t["nqi"], t["nqs"], t["nqg"], t["nnci"], t["nncr"]) == (4, 5, 6, 7, 8)
    assert (t["nqs1"], t["nqs2"], t["nnc1"], t["nnc2"]) == (4, 6, 7, 8)
    m = moisture_table(5)
    assert m["numq"] == 10 and m["nncc"] == 0 and m["inum"] == 1
    assert (m["nnci"], m["nncs"], m["nncr"], m["nncg"]) == (7, 8, 9, 10)
    assert m["qmag"][6:] == (1.0e6,) * 4
    m0 = moisture_table(5, inum=0)
    assert m0["numq"] == 11 and m0["nncc"] == 11 and m0["qmag"][6:] == (1.0e5,) * 5
    cfg = Config({"imoist": 1, "ptype": 5})
    assert cfg.numq == 10 and cfg.qname[3] == "qi" and cfg.iice == 1


def test_getcvm(bk):
    g = Grid(4, 3, 5, 100.0, 100.0, 100.0)
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1e-2, (10,) + g.s_shape)
    cfg = Config({"imoist": 1, "ptype": 5, "eqtset": 2})
    got = bk.to_numpy(getcvm(bk, g, cfg, q=[bk.asarray(a) for a in q])["cvm"])
    S = (g.K, g.J, g.I)
    want = C.cv + C.cvv * q[0][S]
    for n in (1, 2):
        want = want + C.cpl * q[n][S]
    for n in (3, 4, 5):
        want = want + C.cpi * q[n][S]
    np.testing.assert_array_equal(got[S], want)


def test_pdefqtest_matches_pdefq(bk):
    """Same math as pdefq (sequential sums): equal to round-off."""
    g = Grid(5, 4, 6, 100.0, 100.0, 100.0)
    rng = np.random.default_rng(1)
    S = g.s_shape
    q = rng.uniform(-1e-5, 1e-3, S)
    rho = rng.uniform(0.8, 1.2, S)
    rmh = rng.uniform(0.9, 1.1, S)
    ruh = rng.uniform(0.9, 1.1, S[2])
    rvh = rng.uniform(0.9, 1.1, S[1])
    kw = dict(
        rmax=1e-14,
        asq=bk.asarray(0.0),
        ruh=bk.asarray(ruh),
        rvh=bk.asarray(rvh),
        rmh=bk.asarray(rmh),
        rho=bk.asarray(rho),
        q3d=bk.asarray(q),
    )
    a = pdefq(bk, g, None, **kw)
    b = pdefqtest(bk, g, None, **kw)
    np.testing.assert_allclose(bk.to_numpy(b["q3d"]), bk.to_numpy(a["q3d"]), 1e-14)
    np.testing.assert_allclose(float(b["asq"]), float(a["asq"]), 1e-10)
