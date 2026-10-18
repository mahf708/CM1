"""Surface-layer physics, ported from src/sfcphys.F (``getcecd``: the
neutral surface layer of sfcmodel=1).

Only what the semi-slip lower boundary (bbc=3) needs without surface fluxes
(isfcflx=0) is computed: drag coefficient, roughness length and friction
velocity (plus the 10-m winds).  ``use_avg_sfc`` (testcase 1, 2, 9, 11, 14)
is not ported.
"""

from __future__ import annotations

from .constants import karman

DCD1 = 1.0e-3
DCD2 = 2.4e-3
DWSP1 = 5.0
DWSP2 = 25.0
DFAC = (DCD2 - DCD1) / (DWSP2 - DWSP1)

# getcecd stops CM1 when the water iteration needs more than 20 passes
NITER = 20


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def getcecd(bk, cfg, *, u1, v1, s1, za, xland, znt):
    """``getcecd`` on 2-D arrays (any shape; CM1 uses i=1..ni, j=1..nj):
    lowest-level winds u1, v1, speed s1 at height za, land/water flag xland
    and the previous roughness length znt.  Returns cd, znt, ust, u10, v10,
    s10.

    Options: set_znt=1 (given z0), set_ust=1 (given u*), cecd=1 (given Cd),
    else Cd(s10) over water (cecd=2 Deacon, cecd=3 Fairall/Donelan), iterated
    to |ds10| <= 0.001 m/s as a fixed number of masked passes (CM1 stops if
    more than 20 are needed; here the 20th pass is kept), and z0 given over
    land."""
    if int(_opt(cfg, "testcase", 0)) in (1, 2, 9, 11, 14):
        raise NotImplementedError("getcecd: use_avg_sfc is not ported")
    xp = bk.xp
    log, exp, sqrt = xp.log, xp.exp, xp.sqrt
    if int(_opt(cfg, "set_znt", 0)) == 1:
        rznt = 1.0 / znt
        cd = (karman / log(10.0 * rznt)) ** 2
        var = log(10.0 * rznt) / log(za * rznt)
        s10 = s1 * var
        ust = xp.clip(s1 * karman / log(za * rznt), 1.0e-6, None)
    elif int(_opt(cfg, "set_ust", 0)) == 1:
        ust = xp.zeros_like(s1) + float(cfg.cnst_ust)
        znt = za / (
            exp(xp.clip(s1, 1.0e-10, None) * karman / xp.clip(ust, 1.0e-10, None)) - 1.0
        )
        rznt = 1.0 / znt
        cd = (karman / log((10.0 + znt) * rznt)) ** 2
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
    elif int(_opt(cfg, "cecd", 3)) == 1:
        cd = xp.zeros_like(s1) + max(1.0e-10, float(cfg.cnstcd))
        znt = 10.0 / (exp(karman / sqrt(cd)) - 1.0)
        rznt = 1.0 / znt
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
        ust = xp.clip(s1 * karman / log((za + znt) * rznt), 1.0e-6, None)
    else:
        cecd = int(_opt(cfg, "cecd", 3))
        water = xland > 1.5
        # land (and the first guess over water): z0 as given
        rznt = 1.0 / znt
        cd_land = (karman / log((10.0 + znt) * rznt)) ** 2
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
        cd = cd_land
        wlast = xp.zeros_like(s1) - 1000.0
        for _ in range(NITER):
            active = water & (xp.abs(s10 - wlast) > 0.001)
            if cecd == 2:
                cdn = 1.1e-3 + (4.0e-5 * s10)
            else:
                cdn = DCD1 + (s10 - DWSP1) * DFAC
                cdn = xp.clip(xp.clip(cdn, None, DCD2), DCD1, None)
            zn = xp.clip(10.0 / (exp(karman / sqrt(cdn)) - 1.0), 1.0e-20, None)
            rz = 1.0 / zn
            vn = log((10.0 + zn) * rz) / log((za + zn) * rz)
            wlast = xp.where(active, s10, wlast)
            cd = xp.where(active, cdn, cd)
            znt = xp.where(active, zn, znt)
            rznt = xp.where(active, rz, rznt)
            var = xp.where(active, vn, var)
            s10 = xp.where(active, s1 * vn, s10)
        ust = xp.clip(s1 * karman / log((za + znt) * rznt), 1.0e-6, None)
    return {
        "cd": cd,
        "znt": znt,
        "ust": ust,
        "u10": u1 * var,
        "v10": v1 * var,
        "s10": s10,
    }
