"""The MM5/WRF SLAB land-surface model (src/slab.F: SLAB, SLAB1D), as CM1
calls it from turb.F sfc_and_turb after a WRF surface layer (sfcmodel 2, 3,
4, 6, 7) when update_sfc is true, and the soil-layer setup of
init_surface.F (slab_zs, slab_dzs, the initial tslb).

SLAB predicts the land skin temperature from the residual of the surface
energy budget, with heat diffusion through ``num_soil_layers`` soil layers
(CM1: 5; the deepest is held fixed), and recomputes the land hfx, qfx, lh
from the surface layer's exchange coefficients flhc, flqc and the old tsk.
Over water (xland > 1.5) only capg and the qsfc / chklowq diagnosis apply.

CM1's call fixes ifsnow = 0, radiation = .true. (the upward long-wave flux
stbolt*tg0**4 is always on) and gsw = glw = 0 without radiation.  Every
column is independent, so the port is vectorised over [j, i]; the soil
sub-steps and layers are Python loops in the Fortran's order.
"""

from __future__ import annotations

from . import constants as C

__all__ = ["NUM_SOIL_LAYERS", "soil_layers", "init_state", "slab", "slab_update"]

NUM_SOIL_LAYERS = 5  # cm1.F
DIFSL = 5.0e-7  # soil diffusion constant (m^2/s)
SOILFAC = 1.25  # makes the soil step more conservative
EOMEG = 7.2921e-5
STBOLT = 5.67051e-8
P1000MB = 100000.0

SLAB_SFCMODELS = (2, 3, 4, 6, 7)


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def soil_layers(n=NUM_SOIL_LAYERS):
    """init_surface.F: slab_dzs(1) = 0.01 m doubling with depth, slab_zs =
    layer centres (zero for a single layer).  Returns (zs, dzs) lists."""
    zs, dzs = [0.0] * n, [0.0] * n
    if n != 1:
        dzs[0] = 0.01
        zs[0] = 0.5 * dzs[0]
        for m in range(1, n):
            dzs[m] = 2 * dzs[m - 1]
            zs[m] = zs[m - 1] + 0.5 * dzs[m - 1] + 0.5 * dzs[m]
    return zs, dzs


def init_state(ctx):
    """SLAB's state (cm1.F allocations, init_surface.F initsfc=1): tmn =
    tmn0, emiss and thc from LANDUSE.TBL (getlanduse(season, lu0)), capg =
    chklowq = gsw = glw = lh = flhc = flqc = 0 (kept where already in the
    state), and tslb [nsoil, j, i] between tsk and tmn.  Returns the state
    additions (2-D [j, i] arrays with halos)."""
    from .init_surface import getlanduse

    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    z = bk.zeros(g.s_shape[1:])
    lu = getlanduse(int(cfg.season), int(cfg.lu0))
    out = {
        "tmn": z + float(_opt(cfg, "tmn0", 297.28)),
        "emiss": z + lu["emiss"],
        "thc": z + lu["thc"],
    }
    for k in ("capg", "chklowq", "gsw", "glw", "lh", "flhc", "flqc"):
        if k not in s:
            out[k] = z
    zs, _ = soil_layers()
    n = len(zs)
    tsk, tmn = s["tsk"], out["tmn"]
    if n == 1:
        out["tslb"] = bk.zeros((1,) + g.s_shape[1:])
    else:
        out["tslb"] = bk.xp.stack(
            [
                (tsk * (zs[n - 1] - zs[m]) + tmn * (zs[m] - zs[0]))
                / (zs[n - 1] - zs[0])
                for m in range(n)
            ]
        )
    return out


def slab(bk, *, dt, t1, qv1, p1, flhc, flqc, psfc, xland, tmn, hfx, qfx, lh, tsk, gsw, glw, thc,
         emiss, mavail, tslb, zs, dzs):  # fmt: skip
    """slab.F SLAB / SLAB1D (ifsnow = 0, radiation on) on [j, i] interior
    arrays; ``t1``, ``qv1``, ``p1`` are temperature, qv and pressure at
    k = 1, ``tslb`` is [nsoil, j, i].  Returns tsk, qsfc, hfx, qfx, lh,
    chklowq, capg, tslb."""
    xp = bk.xp
    nsl = len(zs)
    rovcp, lv1, lv2 = C.rovcp, C.lv1, C.lv2
    svp1, svp2, svp3, svpt0, ep2 = C.svp1, C.svp2, C.svp3, C.svpt0, C.rd / C.rv
    nsoil, rnsoil = 1, 1.0
    if nsl > 1:
        nsoil = 1 + int(SOILFAC * 4 * DIFSL / dzs[0] * dt / dzs[0])
        rnsoil = 1.0 / float(nsoil)

    psfcc = psfc / 1000.0  # cmb
    pl = p1 / 1000.0
    thcon = (P1000MB * 0.001 / pl) ** rovcp
    thx = t1 * thcon
    qx = qv1
    capg = 3.298e6 * thc if nsl == 1 else 5.9114e7 * thc
    land = ~((xland - 1.5) >= 0)

    tg0 = tsk
    upflux = STBOLT * ((tg0 * tg0) * (tg0 * tg0))
    xinet = emiss * (glw - upflux)
    rnet = gsw + xinet
    hm = 1.18 * EOMEG * (tg0 - tmn)
    esg = svp1 * xp.exp(svp2 * (tg0 - svpt0) / (tg0 - svp3))
    qsg = ep2 * esg / (psfcc - esg)
    thg = tsk * (100.0 / psfcc) ** rovcp
    hfx_l = flhc * (thg - thx)
    qfx_l = flqc * (qsg - qx)
    lh_l = qfx_l * (lv1 - lv2 * tsk)
    qs = hfx_l + qfx_l * (lv1 - lv2 * tsk)
    tl = [tslb[m] for m in range(nsl)]
    if nsl == 1:
        dthgdt = (rnet - qs) / capg - hm
    else:
        dthgdt = xp.zeros_like(tsk)
        flux = [None] * (nsl + 1)
        ps1 = (psfc / P1000MB) ** rovcp
        for itsoil in range(1, nsoil + 1):
            for L in range(1, nsl):
                if L == 1 and itsoil > 1:
                    T1 = tl[0]
                    thg = T1 / ps1
                    esg = svp1 * xp.exp(svp2 * (T1 - svpt0) / (T1 - svp3))
                    qsg = ep2 * esg / (psfcc - esg)
                    hfxt = flhc * (thg - thx)
                    qfxt = flqc * (qsg - qx)
                    qs = hfxt + qfxt * (lv1 - lv2 * tsk)
                    hfx_l = hfx_l + hfxt
                    qfx_l = qfx_l + qfxt
                flux[1] = rnet - qs
                flux[L + 1] = -DIFSL * capg * (tl[L] - tl[L - 1]) / (zs[L] - zs[L - 1])
                dtsdt = -(flux[L + 1] - flux[L]) / (dzs[L - 1] * capg)
                tl[L - 1] = tl[L - 1] + dtsdt * dt * rnsoil
                if L == 1:
                    dthgdt = dthgdt + rnsoil * dtsdt
                if itsoil == nsoil and L == 1:
                    hfx_l = hfx_l * rnsoil
                    qfx_l = qfx_l * rnsoil
                    lh_l = qfx_l * (lv1 - lv2 * tsk)
    tskx = tg0 + dt * dthgdt

    tsk_n = xp.where(land, tskx, tsk)
    hfx_n = xp.where(land, hfx_l, hfx)
    qfx_n = xp.where(land, qfx_l, qfx)
    lh_n = xp.where(land, lh_l, lh)
    tslb_n = xp.stack([xp.where(land, tl[m], tslb[m]) for m in range(nsl)])
    safe = xp.where(flqc != 0.0, flqc, xp.ones_like(flqc))
    qsfc = xp.where(flqc != 0.0, qx + qfx_n / safe, qx)
    return {
        "tsk": tsk_n,
        "qsfc": qsfc,
        "hfx": hfx_n,
        "qfx": qfx_n,
        "lh": lh_n,
        "chklowq": mavail,
        "capg": capg,
        "tslb": tslb_n,
    }


def slab_update(ctx, *, flhc, flqc, hfx, qfx, lh, tsk, qsfc=None):
    """turb.F 1019-1031 (update_sfc and testcase /= 11): ``call SLAB(dum4,
    divx, prs, ...)`` with this step's surface-layer results (interior
    [nj, ni] arrays).  Reads T and qv at k=1 from ``ctx.w``, prs from the
    context and psfc, xland, mavail, tmn, thc, emiss, gsw, glw, tslb from
    ``ctx.s``.  ``qsfc`` is ignored (SLAB rediagnoses it).  Returns interior
    tsk, qsfc, hfx, qfx, lh, chklowq, capg and tslb [nsoil, nj, ni]."""
    from .pbl import interior2

    g, s = ctx.g, ctx.s
    k1, n = g.ngz, g.ngxy
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    zs, dzs = soil_layers(s["tslb"].shape[0])
    i2 = {
        k: interior2(g, s[k])
        for k in ("psfc", "xland", "mavail", "tmn", "thc", "emiss", "gsw", "glw")
    }
    return slab(
        ctx.bk, dt=float(ctx.dt), t1=ctx.w["t"][k1, J, I], qv1=ctx.w["qv"][k1, J, I],
        p1=ctx.prs[k1, J, I], flhc=flhc, flqc=flqc, hfx=hfx, qfx=qfx, lh=lh, tsk=tsk,
        tslb=s["tslb"][:, J, I], zs=zs, dzs=dzs, **i2,
    )  # fmt: skip


def slab_step(ctx, out):
    """``wrf_sfc_epilogue`` hook: SLAB on a surface layer's ``out`` dict
    (interior arrays; missing lh / tsk come from the state).  Returns the
    updated entries, tslb with halos (the state's halos kept)."""
    from .pbl import interior2

    g, s = ctx.g, ctx.s

    def get(k):
        return out[k] if k in out else interior2(g, s[k])

    r = slab_update(
        ctx, flhc=out["flhc"], flqc=out["flqc"], hfx=get("hfx"), qfx=get("qfx"), lh=get("lh"),
        tsk=get("tsk"),
    )  # fmt: skip
    n = g.ngxy
    r["tslb"] = ctx.bk.set(
        s["tslb"], (slice(None), slice(n, n + g.nj), slice(n, n + g.ni)), r["tslb"]
    )
    return r
