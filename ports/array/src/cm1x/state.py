"""Model initialisation: namelist -> grid, base state, prognostic state.

    from cm1x import get_backend
    from cm1x.namelist import read_namelist
    from cm1x.state import initialize

    cfg = read_namelist("ports/reference/kessler_periodic/namelist.input")
    grid, base, state = initialize(cfg, get_backend("numpy", "float64"))

``base`` holds everything that is fixed in time (grid metrics from param.F and
the base state from base.F); ``state`` holds the prognostic/diagnostic arrays
after init3d.F's ``init3d`` + ``getset``. All arrays are keyed by their Fortran
names and laid out ``[k, j, i]`` (``[n, k, j, i]`` for qa/q3d) with CM1's halos.
"""

from __future__ import annotations

from .base import base as _base
from .base import param_grid
from .grid import Grid
from .init3d import getset, init3d

__all__ = ["initialize"]


def initialize(cfg, bk):
    """Return ``(grid, base, state)`` for the namelist ``cfg`` on backend ``bk``.

    base keys: grid metrics (xh, xf, yh, yf, uh, ruh, uf, ruf, vh, rvh, vf, rvf,
      rxh, rxf, arh1, arh2, arf1, arf2, sigma, sigmaf, rds, rdsf, zh, mh, rmh,
      zf, mf, rmf, c1, c2, tauh, taus, tauf, zs), base state (pi0, prs0, rho0,
      thv0, th0, rth0, qv0, qc0, qi0, rr0, rf0, rrf0, t0, rh0, u0, v0, rho0s,
      pi0s, prs0s, rth0s, pmin) and Python-float scalars (cgs1..3, cgt1..3,
      dgs*, dgt*, wbe*, wte*, minx, maxx, miny, maxy, centerx, centery, maxz,
      psurf, tsurf, qsurf, thsurf).
    state keys: ua, u3d, va, v3d, wa, w3d, ppi, pp3d, ppx, tha, th3d, qa, q3d,
      rho, prs, rr, rf, mass1, and convenience views qv3d/qc3d/qr3d (= q3d[n]).
    """
    g = Grid(cfg.ni, cfg.nj, cfg.nk, float(cfg.dx), float(cfg.dy), float(cfg.dz))
    pg = param_grid(cfg, bk, g)
    bs = _base(
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
    b = dict(pg)
    b.update(bs)
    st = init3d(
        bk,
        cfg,
        xh=b["xh"],
        yh=b["yh"],
        zh=b["zh"],
        pi0=b["pi0"],
        prs0=b["prs0"],
        th0=b["th0"],
        thv0=b["thv0"],
        qv0=b["qv0"],
        u0=b["u0"],
        v0=b["v0"],
        centerx=b["centerx"],
        centery=b["centery"],
    )
    gs = getset(
        bk,
        cfg,
        ua=st["ua"],
        va=st["va"],
        wa=st["wa"],
        ppi=st["ppi"],
        tha=st["tha"],
        qa=st["qa"],
        pi0=b["pi0"],
        th0=b["th0"],
        rho0=b["rho0"],
        rf0=b["rf0"],
        prs0=b["prs0"],
        c1=b["c1"],
        c2=b["c2"],
        rmh=b["rmh"],
        ruh=b["ruh"],
        rvh=b["rvh"],
        cg=pg,
    )
    b["rth0"] = gs.pop("rth0")
    state = dict(st)
    state.update(gs)
    if cfg.imoist == 1:
        for name, n in (("qv3d", cfg.nqv), ("qc3d", cfg.nqc), ("qr3d", cfg.nqr)):
            if n:
                state[name] = state["q3d"][n - 1]
    return g, b, state
