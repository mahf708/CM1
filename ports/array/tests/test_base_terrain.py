"""Terrain grid set-up (param.F with init_terrain) and the 3-D base state
(base.F, terrain surface branch) vs the Fortran base (drv_base_terrain.F90),
for nh_mountain_waves (isnd=9, iwnd=6, open x boundaries) and a moist
Weisman-Klemp sounding (isnd=5, iwnd=2) over a 3-D hill (itern=3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cm1x.base import base, param_grid
from cm1x.namelist import read_namelist

from fortran import requires_gfortran, run_driver
from test_base_init import BASE_OUT, np_, relerr

REF = Path(__file__).resolve().parent.parent.parent / "reference"


def _cfg(variant):
    if variant == "nh_mountain_waves":
        return read_namelist(REF / "nh_mountain_waves" / "namelist.input")
    cfg = read_namelist(REF / "kessler_periodic" / "namelist.input")
    return cfg.replace(
        terrain_flag=True, itern=3, nx=24, ny=20, ni=24, nj=20, wbc=2, ebc=2
    )


@requires_gfortran
@pytest.mark.parametrize("variant", ["nh_mountain_waves", "kessler_hill"])
def test_base_terrain_vs_fortran(bk, variant):
    cfg = _cfg(variant)
    pg = param_grid(cfg, bk)
    assert np.ptp(np_(bk, pg["zh"])[1]) > 100.0  # terrain is there
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
        cfg.wbc,
        cfg.sbc,
        cfg.get("testcase", 0),
    ]
    ins = {k: np_(bk, pg[k]) for k in ("zh", "mh", "rmh", "c1", "c2", "zf", "mf")}
    ins["params"] = np.array(params, dtype=np.float64)
    ref = run_driver("base_terrain", ins, {k: tuple(b[k].shape) for k in BASE_OUT})
    for k in BASE_OUT:
        assert relerr(np_(bk, b[k]), ref[k]) < 1e-12, k
