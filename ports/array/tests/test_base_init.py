"""Namelist, base state (base.F) and initial conditions (init3d.F) for the
reference case ports/reference/kessler_periodic.

* base / init3d vs the Fortran routines in double precision (harness drivers
  drv_base.F90, drv_init3d.F90), ~1e-12 relative.
* full ``initialize`` in float32 vs CM1's own single-precision output at t=0
  (tests/data/kessler_periodic_t0.nc, a column/compressed subset of cm1out.nc),
  ~1e-6 relative.  If env CM1X_REF_DIR points at a directory holding a full
  cm1out.nc of the same case, that file is checked too.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cm1x.base import base, extrap_coeffs, param_grid
from cm1x.init3d import init3d
from cm1x.namelist import parse_namelist, read_namelist
from cm1x.state import initialize

from fortran import requires_gfortran, run_driver

HERE = Path(__file__).resolve().parent
NAMELIST = HERE.parent.parent / "reference" / "kessler_periodic" / "namelist.input"
REF_T0 = HERE / "data" / "kessler_periodic_t0.nc"


@pytest.fixture(scope="module")
def cfg():
    return read_namelist(NAMELIST)


def np_(bk, a):
    return np.asarray(bk.to_numpy(a), dtype=np.float64)


def relerr(a, r):
    a, r = np.asarray(a, np.float64), np.asarray(r, np.float64)
    return np.abs(a - r).max() / max(np.abs(r).max(), 1e-300)


# ---------------------------------------------------------------------------
# namelist


def test_namelist_reference(cfg):
    assert (cfg.nx, cfg.ny, cfg.nz) == (40, 40, 40)
    assert (cfg.ni, cfg.nj, cfg.nk) == (40, 40, 40)
    assert (cfg.dx, cfg.dy, cfg.dz) == (1000.0, 1000.0, 500.0)
    assert (cfg.isnd, cfg.iwnd, cfg.iinit, cfg.imoist, cfg.ptype) == (5, 2, 1, 1, 1)
    assert (
        cfg.irandp == 0 and cfg.ibalance == 0 and cfg.iorigin == 2 and cfg.psolver == 3
    )
    assert cfg.terrain_flag is False and cfg.restart_reset_frqtim is True
    assert cfg.zd == 15000.0 and cfg.rdalpha == pytest.approx(3.3333333333e-3)
    assert cfg.imove == 0 and cfg.umove == 0.0  # param.F: imove=0 -> umove=0
    assert cfg.ccn == 0.6e9 and cfg.alphahl == 0.5  # group with inline comments
    assert cfg.tsmall == 0.0001 and cfg.outunits == 1  # cm1.F defaults
    assert cfg.not_a_variable == 0  # unset -> zero
    assert (cfg.numq, cfg.nqv, cfg.nqc, cfg.nqr) == (3, 1, 2, 3)
    assert "param2" in cfg.groups and cfg.groups["param2"]["isnd"] == 5


def test_namelist_parser_misc():
    g = parse_namelist("""
 &a
  x = 1, y = 2.5d0 ! comment, z = 3
  s = 'it''s', l = .false.,
  arr = 1, 2, 3*4,
 /
 &b n=7 /
""")
    assert g["a"] == {"x": 1, "y": 2.5, "s": "it's", "l": False, "arr": [1, 2, 4, 4, 4]}
    assert g["b"] == {"n": 7}


# ---------------------------------------------------------------------------
# param.F pieces


def test_extrap_coeffs_uniform():
    nk = 40
    sigmaf = [500.0 * (k - 1) for k in range(nk + 3)]
    sigma = [0.5 * (sigmaf[k] + sigmaf[k + 1]) for k in range(nk + 2)]
    c = extrap_coeffs(sigma, sigmaf, nk)
    for p in ("cgs", "cgt"):
        assert [c[f"{p}{q}"] for q in (1, 2, 3)] == pytest.approx(
            [1.875, -1.25, 0.375], abs=1e-14
        )
    assert sum(c[f"wbe{q}"] for q in (1, 2, 3)) == pytest.approx(1.0)


def test_param_grid(bk, cfg):
    pg = param_grid(cfg, bk)
    zh, zf = np_(bk, pg["zh"]), np_(bk, pg["zf"])
    assert zh.shape == (42, 46, 46) and zf.shape == (43, 46, 46)
    np.testing.assert_allclose(zh[:, 5, 7], 500.0 * (np.arange(42) - 0.5))
    np.testing.assert_allclose(zf[:, 0, 0], 500.0 * (np.arange(43) - 1.0))
    c1, c2 = np_(bk, pg["c1"]), np_(bk, pg["c2"])
    assert np.all(c1[1:] == 0.5) and np.all(c2[1:] == 0.5) and np.all(c1[0] == 0)
    xh = np_(bk, pg["xh"])
    np.testing.assert_allclose(xh[3:43], np.arange(40) * 1000.0 - 19500.0)  # iorigin=2
    assert pg["centerx"] == 0.0 and pg["centery"] == 0.0 and pg["maxz"] == 20000.0
    tauh = np_(bk, pg["tauh"])[:, 0, 0]
    assert np.all(tauh[1:31] == 0) and 0 < tauh[31] < tauh[40] < 1 and tauh[41] == 0


# ---------------------------------------------------------------------------
# Fortran harness, double precision

BASE_OUT = [
    "pi0",
    "prs0",
    "rho0",
    "thv0",
    "th0",
    "rth0",
    "qv0",
    "qc0",
    "qi0",
    "rr0",
    "rf0",
    "rrf0",
    "t0",
    "rh0",
    "u0",
    "v0",
    "rho0s",
    "pi0s",
    "prs0s",
    "rth0s",
]


def _base_py(bk, cfg):
    pg = param_grid(cfg, bk)
    b = base(
        bk,
        cfg,
        zh=pg["zh"],
        mh=pg["mh"],
        rmh=pg["rmh"],
        c1=pg["c1"],
        c2=pg["c2"],
        zf=pg["zf"],
        mf=pg["mf"],
        cg=pg,
    )
    return pg, b


@requires_gfortran
@pytest.mark.parametrize("variant", ["reference", "dry_iwnd1", "imove"])
def test_base_vs_fortran(bk, cfg, variant):
    if variant == "dry_iwnd1":
        cfg = cfg.replace(imoist=0, iwnd=1)
    elif variant == "imove":
        cfg = cfg.replace(imove=1, umove=12.5, vmove=3.0)
    pg, b = _base_py(bk, cfg)
    params = [
        cfg.ni,
        cfg.nj,
        cfg.nk,
        cfg.dx,
        cfg.dy,
        cfg.dz,
        cfg.isnd,
        cfg.iwnd,
        cfg.imoist,
        cfg.ptype,
        cfg.psolver,
        cfg.imove,
        cfg.umove,
        cfg.vmove,
        *(pg[k] for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")),
        cfg.tsmall,
        # testcase, eqtset, iice, tsk0, hurr_vg, hurr_rotate, lspgrad, wbc..nbc
        0,
        2,
        0,
        299.28,
        40.0,
        0.0,
        0,
        1,
        1,
        1,
        1,
    ]
    ins = {k: np_(bk, pg[k]) for k in ("zh", "mh", "rmh", "c1", "c2", "zf", "mf")}
    ins["params"] = np.array(params, dtype=np.float64)
    ref = run_driver("base", ins, {k: tuple(b[k].shape) for k in BASE_OUT})
    for k in BASE_OUT:
        assert relerr(np_(bk, b[k]), ref[k]) < 1e-12, k


@requires_gfortran
@pytest.mark.parametrize("ibalance", [0, 1])
def test_init3d_vs_fortran(bk, cfg, ibalance):
    cfg = cfg.replace(ibalance=ibalance)
    pg, b = _base_py(bk, cfg)
    st = init3d(
        bk,
        cfg,
        xh=pg["xh"],
        yh=pg["yh"],
        zh=pg["zh"],
        pi0=b["pi0"],
        prs0=b["prs0"],
        th0=b["th0"],
        thv0=b["thv0"],
        qv0=b["qv0"],
        u0=b["u0"],
        v0=b["v0"],
        centerx=pg["centerx"],
        centery=pg["centery"],
    )
    params = [
        cfg.ni,
        cfg.nj,
        cfg.nk,
        cfg.dx,
        cfg.dy,
        cfg.dz,
        cfg.iinit,
        cfg.ibalance,
        cfg.imoist,
        cfg.ptype,
        cfg.isnd,
        cfg.psolver,
        cfg.irandp,
        pg["centerx"],
        pg["centery"],
    ]
    ins = {k: np_(bk, pg[k]) for k in ("xh", "yh", "zh", "mh", "mf", "zf")}
    ins.update(
        {
            k: np_(bk, b[k])
            for k in ("pi0", "prs0", "th0", "thv0", "qv0", "rho0", "rf0", "u0", "v0")
        }
    )
    ins["params"] = np.array(params, dtype=np.float64)
    outs = {k: tuple(st[k].shape) for k in ("ua", "u3d", "va", "v3d", "tha", "ppi")}
    outs["qa"] = tuple(st["qa"].shape)  # Fortran qa(i,j,k,n) <-> python [n,k,j,i]
    ref = run_driver("init3d", ins, outs)
    for k in outs:
        a = np_(bk, st[k])
        if k == "ppi":
            scale = max(np.abs(ref[k]).max(), 1e-3)
            assert np.abs(a - ref[k]).max() / scale < 1e-12, k
        else:
            assert relerr(a, ref[k]) < 1e-12, k
    if ibalance == 1:
        assert np.abs(ref["ppi"]).max() > 1e-5  # bubble is hydrostatically balanced
    tha = np_(bk, st["tha"])
    assert 0.9 < tha.max() <= 1.0 and tha.min() == 0.0


# ---------------------------------------------------------------------------
# whole initialisation vs CM1 single-precision output


def _check_against_ref(bk, g, b, s, ref, column):
    K, J, I, IF, JF = g.K, g.J, g.I, g.IF, g.JF

    def ck(name, arr, r, tol=1e-6):
        a = np_(bk, arr)
        if column:  # reference stored as (i,j)=(1,1) column of a homogeneous field
            assert np.ptp(a, axis=(1, 2)).max() <= 1e-6 * np.abs(a).max(), name
            a = a[:, 0, 0]
        err = relerr(a, np.asarray(r))
        assert err < tol, (name, err)

    def get(v):
        return ref[v]

    for v in ("th0", "pi0", "prs0", "qv0"):
        ck(v, b[v][K, J, I], get(v))
    ck("u0", b["u0"][K, J, IF], get("u0"))
    ck("v0", b["v0"][K, JF, I], get("v0"))
    ck("u", s["u3d"][K, J, IF], get("u"))
    ck("v", s["v3d"][K, JF, I], get("v"))
    ck("w", s["w3d"][g.KF, J, I], get("w"))
    ck("qv", s["qv3d"][K, J, I], get("qv"))
    ck("qc", s["qc3d"][K, J, I], get("qc"))
    ck("qr", s["qr3d"][K, J, I], get("qr"))
    ck("prs", s["prs"][K, J, I], get("prs"))
    ck("pipert", s["pp3d"][K, J, I], get("pipert"))
    # 3-D fields
    assert relerr(np_(bk, s["th3d"][K, J, I]), ref["thpert"]) < 1e-6
    assert relerr(np_(bk, s["rho"][K, J, I]), ref["rho"]) < 1e-6


def test_initialize_vs_cm1_output(bk32, cfg):
    nc = pytest.importorskip("netCDF4")
    g, b, s = initialize(cfg, bk32)
    with nc.Dataset(REF_T0) as d:
        d.set_auto_mask(False)
        ref = {k: np.asarray(d[k][...]) for k in d.variables}
    _check_against_ref(bk32, g, b, s, ref, column=True)


@pytest.mark.skipif(not os.environ.get("CM1X_REF_DIR"), reason="CM1X_REF_DIR not set")
def test_initialize_vs_full_cm1out(bk32, cfg):
    nc = pytest.importorskip("netCDF4")
    path = Path(os.environ["CM1X_REF_DIR"]) / "cm1out.nc"
    if not path.exists():
        pytest.skip(f"{path} missing")
    g, b, s = initialize(cfg, bk32)
    with nc.Dataset(path) as d:
        d.set_auto_mask(False)
        names = [
            "th0",
            "pi0",
            "prs0",
            "qv0",
            "u0",
            "v0",
            "u",
            "v",
            "w",
            "qv",
            "qc",
            "qr",
            "prs",
            "pipert",
            "thpert",
            "rho",
        ]
        ref = {k: np.asarray(d[k][0]) for k in names}
    _check_against_ref(bk32, g, b, s, ref, column=False)


def test_initialize_float64_structure(bk, cfg):
    g, b, s = initialize(cfg, bk)
    assert tuple(s["u3d"].shape) == g.u_shape and tuple(s["v3d"].shape) == g.v_shape
    assert (
        tuple(s["w3d"].shape) == g.w_shape and tuple(s["q3d"].shape) == (3,) + g.s_shape
    )
    for k in (
        "th0",
        "pi0",
        "prs0",
        "rho0",
        "rr0",
        "rf0",
        "rrf0",
        "thv0",
        "qv0",
        "rth0",
    ):
        assert tuple(b[k].shape) == g.s_shape, k
    # periodic halos
    th3d = np_(bk, s["th3d"])
    np.testing.assert_array_equal(th3d[:, :, 0:3], th3d[:, :, 40:43])
    np.testing.assert_array_equal(th3d[:, 0:3, :], th3d[:, 40:43, :])
    rr, rho = np_(bk, s["rr"]), np_(bk, s["rho"])
    np.testing.assert_allclose(rr[1:41] * rho[1:41], 1.0, rtol=1e-14)
    assert float(np_(bk, s["mass1"])) == pytest.approx(
        rho[1:41, 3:43, 3:43].sum(), rel=1e-12
    )
    # rf: interior average, 2nd-order extrapolation at k=1, nk+1
    rf = np_(bk, s["rf"])
    np.testing.assert_allclose(
        rf[2:41, 3:43, 3:43], 0.5 * (rho[1:40] + rho[2:41])[:, 3:43, 3:43], rtol=1e-14
    )
    np.testing.assert_allclose(
        rf[1, 2:44, 2:44],
        (1.875 * rho[1] - 1.25 * rho[2] + 0.375 * rho[3])[2:44, 2:44],
        rtol=1e-13,
    )
