"""Terrain-following grid (port of src/init_terrain.F).

``init_terrain`` specifies the surface height ``zs`` (itern = 1, 2, 3: the
analytic bell / Schaer / Doernbrack shapes; itern = 4 reads ``perts.dat`` and
is not ported) and derives the Gal-Chen coordinate metrics that every
terrain branch of CM1 uses:

    zh = zs + sigma*(zt-zs)/zt,  zf = zs + sigmaf*(zt-zs)/zt
    gz = zt/(zt-zs), rgz = 1/gz, gzu/rgzu, gzv/rgzv (u, v points),
    dzdx, dzdy (zsgrad), gx, gxu, gy, gyv (d sigma / dx terms on w levels),
    rds = 1/(sigma(k)-sigma(k-1)), rdsf = 1/(sigmaf(k+1)-sigmaf(k)).

Layout (README): 2-D arrays ``a[j, i]`` over CM1's full halo'd range
(``(ib:ie, jb:je)`` -> ``Grid.s_shape[1:]``); gx, gxu, gy, gyv are
``(ib:ie, jb:je, kb:ke)`` -> ``Grid.s_shape``; zh ``s_shape``, zf ``w_shape``;
rds/sigma length nk+2 and rdsf/sigmaf length nk+3 (Python index == Fortran k).
Points CM1 does not set keep param.F's initial values (gz.. = 1, dz/dx = 0,
gx.. = 0, rds = rdsf = 1).
"""

from __future__ import annotations

from . import bc as _bc
from . import constants as C
from .adv_routines import zsgrad
from .grid import Grid

__all__ = ["init_terrain", "terrain_metrics"]


def init_terrain(
    bk, g: Grid, cfg, *, xh, uh, xf, uf, yh, vh, yf, vf, rds, sigma, rdsf, sigmaf, **kw
):
    """Port of ``init_terrain``.  ``xh .. vf`` are the 1-D grid arrays of
    param.F (Python index = Fortran index + ngxy - 1); ``sigma``/``sigmaf``
    the nominal heights.  Module scalars: ``maxx``, ``maxy`` (itern = 3) and
    ``maxz`` (= sigmaf(nk+1), CM1's zt) come from ``kw`` or are computed.

    Returns zs, zh, zf, gz, rgz, gzu, rgzu, gzv, rgzv, dzdx, dzdy, gx, gxu,
    gy, gyv, rds, rdsf and the Python float zt (= rzt**-1).
    """
    xp = bk.xp
    S = bk.asarray
    ni, nk, ng = g.ni, g.nk, g.ngxy
    dx = float(g.dx)
    itern = int(cfg.get("itern", 0))
    shp2 = g.s_shape[1:]

    # ---------------- surface height (SPECIFY TERRAIN HERE)
    if itern == 1:
        # bell-shaped (Durran and Klemp 1983)
        hh, aa, xc = S(400.0), S(1000.0), S(0.0 + 0.5 * dx)
        x = (xh - xc) / aa
        zs = hh / (S(1.0) + x**2)
        zs = zs[None, :] + bk.zeros(shp2)
    elif itern == 2:
        # Schaer case (Klemp et al. 2003): xval = dx*(i-ni/2), Fortran i
        ii = bk.asarray([float(i - ng + 1 - ni // 2) for i in range(shp2[1])])
        xval = S(dx) * ii
        zs = (
            S(250.0)
            * xp.exp(-((xval / S(5000.0)) ** 2))
            * (xp.cos(S(C.pi) * xval / S(4000.0)) ** 2)
        )
        zs = zs[None, :] + bk.zeros(shp2)
    elif itern == 3:
        # Doernbrack et al. (2005)
        hh, aa = S(500.0), S(20000.0)
        maxx, maxy = float(kw["maxx"]), float(kw["maxy"])
        xval = xp.sqrt(
            (xh[None, :] - S(0.25 * maxx)) ** 2 + (yh[:, None] - S(0.25 * maxy)) ** 2
        )
        zs = hh * ((S(1.0) + (xval / aa) ** 2) ** S(-1.5))
    elif itern == 4:
        raise NotImplementedError("itern=4 (terrain read from perts.dat) not ported")
    else:
        raise NotImplementedError(f"itern={itern} not ported")

    zt = float(kw.get("maxz", float(bk.to_numpy(sigmaf[nk + 1]))))
    return terrain_metrics(
        bk,
        g,
        cfg,
        zs=zs,
        uh=uh,
        uf=uf,
        vh=vh,
        vf=vf,
        rds=rds,
        sigma=sigma,
        rdsf=rdsf,
        sigmaf=sigmaf,
        zt=zt,
    )


def terrain_metrics(
    bk, g: Grid, cfg, *, zs, uh, uf, vh, vf, rds, sigma, rdsf, sigmaf, zt
):
    """The part of ``init_terrain`` below "DO NOT CHANGE ANYTHING BELOW HERE":
    from a surface height ``zs`` (2-D ``[j, i]``, halos filled by bc2d here)
    and the model top ``zt`` (Python float) all Gal-Chen metrics.  Returns
    the same dict as init_terrain."""
    xp = bk.xp
    S = bk.asarray
    nk = g.nk
    dx, dy = float(g.dx), float(g.dy)
    shp2 = g.s_shape[1:]
    zs = _bc.bc2d(bk, g, zs, cfg)

    zts = S(zt)
    rzt = 1.0 / zt

    # zh(k=1..nk), zf(kb..ke+1) on all columns
    sig = sigma[1 : nk + 1][:, None, None]
    zh_in = zs[None] + sig * (zts - zs[None]) / zts
    zf = zs[None] + sigmaf[:, None, None] * (zts - zs[None]) / zts

    gz0 = bk.full(shp2, 1.0)  # param.F: gz = 1 (zsgrad uses it as scratch)
    zg = zsgrad(
        bk,
        g,
        cfg,
        zs=zs,
        gz=gz0,
        dzdx=bk.zeros(shp2),
        dzdy=bk.zeros(shp2),
        uh=uh,
        vh=vh,
    )
    dzdx = _bc.bc2d(bk, g, zg["dzdx"], cfg)
    dzdy = _bc.bc2d(bk, g, zg["dzdy"], cfg)

    # metric terms
    gz = zts / (zts - zs)
    rgz = (zts - zs) / zts
    rgz = _bc.bc2d(bk, g, rgz, cfg)

    one2 = bk.full(shp2, 1.0)
    half = S(0.5)
    zsu = half * (zs[1:, :-1] + zs[1:, 1:])  # i = ib+1..ie, j = jb+1..je
    zsv = half * (zs[:-1, 1:] + zs[1:, 1:])
    gzu = bk.set(one2, (slice(1, None), slice(1, None)), zts / (zts - zsu))
    rgzu = bk.set(bk.copy(one2), (slice(1, None), slice(1, None)), (zts - zsu) / zts)
    gzv = bk.set(bk.copy(one2), (slice(1, None), slice(1, None)), zts / (zts - zsv))
    rgzv = bk.set(bk.copy(one2), (slice(1, None), slice(1, None)), (zts - zsv) / zts)

    # gx, gxu, gy, gyv: k = 1..nk+1, j = jb+1..je-1, i = ib+1..ie-1
    rdx, rdy = S(1.0 / dx), S(1.0 / dy)
    Ji, Ii = slice(1, shp2[0] - 1), slice(1, shp2[1] - 1)
    Kf = slice(1, nk + 2)
    zmsf = (zts - sigmaf[Kf])[:, None, None]
    uhi = uh[Ii][None, None, :]
    ufi = uf[Ii][None, None, :]
    vhj = vh[Ji][None, :, None]
    vfj = vf[Ji][None, :, None]
    gx_ = zmsf * gz[Ji, Ii] * (rgzu[Ji, 2 : shp2[1]] - rgzu[Ji, Ii]) * rdx * uhi
    gxu_ = zmsf * gzu[Ji, Ii] * (rgz[Ji, Ii] - rgz[Ji, 0 : shp2[1] - 2]) * rdx * ufi
    gy_ = zmsf * gz[Ji, Ii] * (rgzv[2 : shp2[0], Ii] - rgzv[Ji, Ii]) * rdy * vhj
    gyv_ = zmsf * gzv[Ji, Ii] * (rgz[Ji, Ii] - rgz[0 : shp2[0] - 2, Ii]) * rdy * vfj
    z3 = bk.zeros(g.s_shape)
    gx = bk.set(z3, (Kf, Ji, Ii), gx_)
    gxu = bk.set(bk.zeros(g.s_shape), (Kf, Ji, Ii), gxu_)
    gy = bk.set(bk.zeros(g.s_shape), (Kf, Ji, Ii), gy_)
    gyv = bk.set(bk.zeros(g.s_shape), (Kf, Ji, Ii), gyv_)

    # zf(0), zf(nk+2), zh(0), zh(nk+1)
    zf0 = zf[1] - (zf[2] - zf[1])
    zftop = zf[nk + 1] + (zf[nk + 1] - zf[nk])
    zf = xp.concat([zf0[None], zf[1 : nk + 2], zftop[None]], axis=0)
    zh = xp.concat(
        [
            (half * (zf[0] + zf[1]))[None],
            zh_in,
            (half * (zf[nk + 1] + zf[nk + 2]))[None],
        ],
        axis=0,
    )

    one = S(1.0)
    rds = bk.set(
        bk.copy(rds), slice(2, nk + 1), one / (sigma[2 : nk + 1] - sigma[1:nk])
    )
    rdsf = bk.set(
        bk.copy(rdsf), slice(1, nk + 1), one / (sigmaf[2 : nk + 2] - sigmaf[1 : nk + 1])
    )
    return dict(
        zs=zs,
        zh=zh,
        zf=zf,
        gz=gz,
        rgz=rgz,
        gzu=gzu,
        rgzu=rgzu,
        gzv=gzv,
        rgzv=rgzv,
        dzdx=dzdx,
        dzdy=dzdy,
        gx=gx,
        gxu=gxu,
        gy=gy,
        gyv=gyv,
        rds=rds,
        rdsf=rdsf,
        zt=zt,
        rzt=rzt,
    )
