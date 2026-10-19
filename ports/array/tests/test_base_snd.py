"""Every base-state sounding (isnd) and wind profile (iwnd) of src/base.F vs
the Fortran routine (harness driver drv_base.F90), double precision.

The runs mirror the cm1k Fortran-bridge test (cxx/tests/fortran_bridge/
base_tests.cpp): each isnd with the namelist settings of the bundled
run/config_files cases that use it (testcase, ptype, eqtset, psolver, iwnd),
isnd = 7 / 17 with the shipped and a synthetic input_sounding, and every
iwnd with imove and open lateral boundaries. Grids: "deep" is a stretched
(synthetic) 40-level grid to ~19 km, "shallow" and "pbl" are uniform.
The testcase-dependent profiles are compared with ``profiles_only`` (the
testcase forcing profiles of base.F are not ported).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from cm1x import base_snd
from cm1x.base import base, extrap_coeffs, param_grid
from cm1x.namelist import read_namelist

from fortran import requires_gfortran, run_driver

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMELIST = HERE.parent.parent / "reference" / "kessler_periodic" / "namelist.input"
CONFIGS = ROOT / "run" / "config_files"

OUT = [
    "pi0", "prs0", "rho0", "thv0", "th0", "rth0", "qv0", "qc0", "qi0", "rr0",
    "rf0", "rrf0", "t0", "rh0", "u0", "v0", "rho0s", "pi0s", "prs0s", "rth0s",
]  # fmt: skip


def np_(bk, a):
    return np.asarray(bk.to_numpy(a), dtype=np.float64)


@pytest.fixture(scope="module")
def soundings(tmp_path_factory):
    d = tmp_path_factory.mktemp("snd")
    out = {}
    for cs in ("hurricane_axisymmetric", "sea_breeze"):
        out[cs] = d / f"input_sounding_{cs}"
        shutil.copy(CONFIGS / cs / "input_sounding", out[cs])
    # winds varying with height, uneven levels, comma separators, d exponents
    lines = ["  1000.0 , 300.0, 14.0"]
    for k in range(1, 61):
        z = 37.0 * k + 3.1 * k * k + 0.4 * k * k * k
        th = 300.0 + 0.004 * z + 1.0e-7 * z * z
        qv = 14.0 * math.exp(-z / 2500.0)
        u = 3.0 + 12.0 * math.tanh(z / 3000.0)
        v = -2.0 + 1.0e-3 * z - 4.0e-8 * z * z
        lines.append(f"{z:.4f}, {th:.5f} {qv:.6f}d0 {u:.3E}   {v:.4f}")
    out["wind"] = d / "input_sounding_wind"
    out["wind"].write_text("\n".join(lines) + "\n")
    return out


def _grid(bk, cfg, grid):
    """param_grid on a uniform grid; "deep": zh, zf, c1, c2 and the
    extrapolation coefficients of a stretched column instead."""
    pg = param_grid(cfg, bk)
    if grid != "deep":
        return pg
    nk = cfg.nk
    dzs = [100.0 + 600.0 * (0.5 - 0.5 * math.cos(math.pi * min(k, 30) / 30)) for k in range(nk + 2)]
    zf1 = [-dzs[0], 0.0]
    for k in range(1, nk + 2):
        zf1.append(zf1[-1] + dzs[k])
    zh1 = [0.5 * (zf1[k] + zf1[k + 1]) for k in range(nk + 2)]
    c2c = [0.0] + [(zf1[k] - zh1[k - 1]) / (zh1[k] - zh1[k - 1]) for k in range(1, nk + 2)]
    c1c = [0.0] + [1.0 - c for c in c2c[1:]]
    shp_s, shp_w = pg["zh"].shape, pg["zf"].shape

    def col(v, shp):
        return bk.asarray(np.broadcast_to(np.asarray(v)[:, None, None], shp).copy())

    pg = dict(pg)
    pg.update(
        zh=col(zh1, shp_s), zf=col(zf1, shp_w), c1=col(c1c, shp_s), c2=col(c2c, shp_s)
    )
    pg.update(extrap_coeffs(zh1, zf1, nk))
    return pg


GRIDS = {
    "deep": dict(nz=40, dz=500.0),
    "shallow": dict(nz=40, dz=60.0),
    "pbl": dict(nz=60, dz=25.0),
}

M1 = dict(imoist=1, ptype=1)
RUNS = [
    # ---- soundings (zero wind unless noted)
    ("isnd1", "deep", dict(isnd=1), None),
    ("isnd1 les_ib_windtunnel", "pbl", dict(isnd=1, iwnd=6, testcase=12, psolver=6), None),
    ("isnd2", "deep", dict(isnd=2), None),
    ("isnd3", "deep", dict(isnd=3), None),
    ("isnd4 kessler", "deep", dict(M1, isnd=4), None),
    ("isnd5 moist", "deep", dict(M1, isnd=5, iwnd=2), None),
    ("isnd5 dry", "deep", dict(isnd=5, iwnd=1), None),
    ("isnd5 ptype4", "deep", dict(isnd=5, imoist=1, ptype=4), None),
    ("isnd7 hurricane", "deep", dict(M1, isnd=7), "hurricane_axisymmetric"),
    ("isnd7 sea_breeze", "shallow", dict(M1, isnd=7, wbc=2, ebc=2), "sea_breeze"),
    ("isnd7 wind moist", "deep", dict(M1, isnd=7), "wind"),
    ("isnd7 wind dry imove", "deep", dict(isnd=7, imove=1, umove=3.5, vmove=-1.25), "wind"),
    ("isnd7 wind open bc", "deep", dict(M1, isnd=7, wbc=2, ebc=2, sbc=3, nbc=4), "wind"),
    ("isnd17 + iwnd2", "deep", dict(M1, isnd=17, iwnd=2), "wind"),
    ("isnd17 dry + iwnd5", "deep", dict(isnd=17, iwnd=5), "hurricane_axisymmetric"),
    ("isnd8 dns_RayleighBenard", "pbl", dict(isnd=8), None),
    ("isnd8 les_HurrBoundLayer", "pbl", dict(isnd=8, iwnd=8, testcase=6, lspgrad=3), None),
    (
        "isnd8 scm_HurrBoundLayer_tqnudge",
        "pbl",
        dict(isnd=8, iwnd=8, testcase=10, imoist=1, ptype=6, lspgrad=3),
        None,
    ),
    (
        "isnd8 les_HurrCoast",
        "pbl",
        dict(isnd=8, iwnd=8, testcase=15, imoist=1, ptype=0, eqtset=1, psolver=2,
             hurr_rotate=-90.0, hurr_vg=35.0, lspgrad=3),
        None,
    ),  # fmt: skip
    (
        "iwnd8 rotate lspgrad3 imove open",
        "pbl",
        dict(isnd=8, iwnd=8, hurr_rotate=30.0, lspgrad=3, imove=1, umove=3.0, vmove=-1.5,
             wbc=2, ebc=2, sbc=3, nbc=3),
        None,
    ),  # fmt: skip
    ("isnd9", "deep", dict(isnd=9, iwnd=6), None),
    ("isnd9 moist", "deep", dict(M1, isnd=9), None),
    ("isnd10 kessler eqtset2", "deep", dict(M1, isnd=10), None),
    ("isnd10 kessler eqtset1", "deep", dict(M1, isnd=10, eqtset=1), None),
    ("isnd10 ice ptype5", "deep", dict(isnd=10, imoist=1, ptype=5), None),
    ("isnd10 ice ptype2", "deep", dict(isnd=10, imoist=1, ptype=2), None),
    ("isnd11", "deep", dict(M1, isnd=11, tsk0=301.15), None),
    ("isnd12", "pbl", dict(isnd=12), None),
    ("isnd13", "deep", dict(isnd=13, iwnd=3), None),
    ("isnd14 les_ConvBoundLayer", "pbl", dict(isnd=14, iwnd=6, testcase=1), None),
    (
        "isnd15 les_StratoCuNoPrecip",
        "pbl",
        dict(isnd=15, iwnd=6, testcase=4, imoist=1, ptype=6),
        None,
    ),
    (
        "isnd15 les_StratoCuDrizzle",
        "pbl",
        dict(isnd=15, iwnd=10, testcase=5, imoist=1, ptype=5),
        None,
    ),
    ("isnd18 les_ShearBoundLayer", "pbl", dict(isnd=18, iwnd=6, testcase=2), None),
    (
        "isnd19 les_ShallowCu",
        "shallow",
        dict(isnd=19, iwnd=9, testcase=3, imoist=1, ptype=5),
        None,
    ),
    (
        "isnd19 les_ConvPBL_moisture",
        "deep",
        dict(isnd=19, iwnd=6, testcase=11, imoist=1, ptype=0, eqtset=1, psolver=7),
        None,
    ),
    ("isnd19 dry", "shallow", dict(isnd=19, iwnd=9), None),
    (
        "isnd20 les_ShallowCuPrecip",
        "shallow",
        dict(isnd=20, iwnd=11, testcase=7, imoist=1, ptype=5),
        None,
    ),
    ("isnd21", "deep", dict(M1, isnd=21, tsk0=300.0), None),
    (
        "isnd22 les_StableBoundLayer",
        "pbl",
        dict(isnd=22, iwnd=6, testcase=9, eqtset=1, psolver=2),
        None,
    ),
    (
        "isnd23 les_ShallowCuLand",
        "shallow",
        dict(isnd=23, iwnd=6, testcase=14, imoist=1, ptype=6, eqtset=1),
        None,
    ),
]
for _w in range(12):
    RUNS.append((f"iwnd{_w}", "deep", dict(isnd=5, iwnd=_w), None))
    RUNS.append(
        (
            f"iwnd{_w} imove open",
            "deep",
            dict(isnd=1, iwnd=_w, imove=1, umove=7.5, vmove=2.25, wbc=2, ebc=3, sbc=2,
                 nbc=4, hurr_rotate=30.0),
            None,
        )
    )  # fmt: skip
for _tc in (0, 1, 2, 4, 9, 11, 12, 14, 8):
    RUNS.append((f"iwnd6 testcase {_tc}", "pbl", dict(isnd=1, iwnd=6, testcase=_tc), None))


def _cfg(grid, over):
    cfg = read_namelist(NAMELIST)
    kw = dict(nx=5, ny=4, iwnd=0, imoist=0, ptype=0, testcase=0, eqtset=2, psolver=3,
              imove=0, umove=0.0, vmove=0.0, tsk0=299.28, hurr_vg=40.0, hurr_rotate=0.0,
              lspgrad=0, wbc=1, ebc=1, sbc=1, nbc=1, irdamp=0)  # fmt: skip
    kw.update(GRIDS[grid])
    kw.update(over)
    return cfg.replace(**kw)


@requires_gfortran
@pytest.mark.parametrize("label,grid,over,snd", RUNS, ids=[r[0] for r in RUNS])
def test_base_snd_vs_fortran(bk, soundings, tmp_path, label, grid, over, snd):
    cfg = _cfg(grid, over)
    files = []
    if snd:
        p = tmp_path / "input_sounding"
        shutil.copy(soundings[snd], p)
        cfg = cfg.replace(input_sounding=str(p))
        files = [p]
    pg = _grid(bk, cfg, grid)
    kw = {k: pg[k] for k in ("zh", "mh", "rmh", "c1", "c2", "zf", "mf")}
    b = base(bk, cfg, **kw, cg=pg, profiles_only=True)
    params = [
        cfg.ni, cfg.nj, cfg.nk, cfg.dx, cfg.dy, cfg.dz, cfg.isnd, cfg.iwnd,
        cfg.imoist, cfg.ptype, cfg.psolver, cfg.imove, cfg.umove, cfg.vmove,
        *(pg[k] for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")),
        cfg.tsmall, cfg.testcase, base_snd.param_eqtset(cfg), base_snd.param_iice(cfg),
        cfg.tsk0, cfg.hurr_vg, cfg.hurr_rotate, cfg.lspgrad,
        cfg.wbc, cfg.ebc, cfg.sbc, cfg.nbc,
    ]  # fmt: skip
    ins = {k: np_(bk, v) for k, v in kw.items()}
    ins["params"] = np.array(params, dtype=np.float64)
    ref = run_driver("base", ins, {k: tuple(b[k].shape) for k in OUT}, files=files)
    worst = {}
    for k in OUT:
        a, r = np_(bk, b[k]), ref[k]
        d = np.abs(a - r).max() / max(np.abs(r).max(), 1e-300)
        worst[k] = d
    bad = {k: v for k, v in worst.items() if not v <= 1e-12}
    assert not bad, f"{label}: {bad}"


@pytest.mark.parametrize(
    "over",
    [
        dict(isnd=6),
        dict(isnd=16),
        dict(isnd=24),
        dict(isnd=2, imoist=1, ptype=1),
        dict(isnd=11),
        dict(isnd=15),
        dict(isnd=1, iwnd=12),
        dict(isnd=7, input_sounding="/no/such/input_sounding"),
    ],
)
def test_base_snd_stops(bk, over):
    """Options CM1 stops on (or that are not CM1 options) raise."""
    cfg = _cfg("pbl", over)
    pg = param_grid(cfg, bk)
    kw = {k: pg[k] for k in ("zh", "mh", "rmh", "c1", "c2", "zf", "mf")}
    with pytest.raises((RuntimeError, NotImplementedError, OSError)):
        base(bk, cfg, **kw, cg=pg, profiles_only=True)


def test_base_testcase_forcing_raises(bk):
    cfg = _cfg("pbl", dict(isnd=19, iwnd=9, testcase=3, imoist=1, ptype=1))
    pg = param_grid(cfg, bk)
    kw = {k: pg[k] for k in ("zh", "mh", "rmh", "c1", "c2", "zf", "mf")}
    with pytest.raises(NotImplementedError, match="forcing"):
        base(bk, cfg, **kw, cg=pg)
