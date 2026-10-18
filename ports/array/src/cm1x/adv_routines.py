"""Flux-form advection kernels, ported from src/adv_routines.F.

Only the routines selected by the reference cases (hadvordr*=vadvordr*=5,
WENO5 for scalars, periodic or open lateral boundaries) are ported:

  * ``flx3``, ``flx5``                 (statement-like flux functions)
  * ``hadv_flx5``                      (horizontal 5th-order upwind fluxes)
  * ``vadv_flx5``                      (vertical 5th-order upwind fluxes)
  * ``vadv_lwr_flx_23``, ``vadv_lwr_flx_35`` (lower-order fluxes near bottom/top)
  * ``weno5``, ``upstrpd``, ``hadv_weno5``, ``vadv_weno5``,
    ``vadv_lwr_weno_23``, ``vadv_lwr_weno_35``, ``vadv_lwr_weno_35_b``
    (scalars, stag=1, only)

``stag`` follows CM1: 1 = scalar, 2 = u, 3 = v, 4 = w.

The flux arrays ``dumx``, ``dumy``, ``dumz`` are scalar-shaped
(``ib:ie, jb:je, kb:ke``) exactly as in CM1. Points that CM1 does not compute
are returned as zero (in Fortran they hold stale scratch values).

Index helpers: Fortran ``i`` -> Python ``fi(i)``, Fortran ``k`` -> ``fk(k)``.
"""

from __future__ import annotations

from .grid import fi, fk


def _opt(cfg, name, default):
    if cfg is None:
        return default
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def ibflags(cfg):
    """(ibw, ibe, ibs, ibn) for a single-process run: 1 where the lateral
    boundary is open (bc=2), as set in param.F; 0 for periodic (bc=1)."""
    out = []
    for b in ("wbc", "ebc", "sbc", "nbc"):
        v = _opt(cfg, b, 1)
        if v not in (1, 2):
            raise NotImplementedError(
                f"{b}={v}: only periodic (1) or open (2) BCs are ported"
            )
        out.append(1 if v == 2 else 0)
    return tuple(out)


def _irange(g, cfg):
    """(i1, i2) of CM1's u-staggered loops."""
    ibw, ibe, _, _ = ibflags(cfg)
    return (2 if ibw else 1), (g.ni if ibe else g.ni + 1)


def _jrange(g, cfg):
    """(j1, j2) of CM1's v-staggered loops."""
    _, _, ibs, ibn = ibflags(cfg)
    return (2 if ibs else 1), (g.nj if ibn else g.nj + 1)


# ---------------------------------------------------------------------------
# small indexing helpers (Fortran inclusive ranges + offsets)
# ---------------------------------------------------------------------------


def _r(lo, hi, off=0):
    return slice(lo + off, hi + off + 1)


def sub3(a, k, j, i, dk=0, dj=0, di=0):
    """``a(i1+di:i2+di, j1+dj:j2+dj, k1+dk:k2+dk)`` for Fortran inclusive ranges."""
    return a[
        fk(k[0]) + dk : fk(k[1]) + dk + 1,
        fi(j[0]) + dj : fi(j[1]) + dj + 1,
        fi(i[0]) + di : fi(i[1]) + di + 1,
    ]


def put3(bk, a, k, j, i, v):
    """Return ``a`` with Fortran region (i, j, k) set to ``v``."""
    idx = (
        slice(fk(k[0]), fk(k[1]) + 1),
        slice(fi(j[0]), fi(j[1]) + 1),
        slice(fi(i[0]), fi(i[1]) + 1),
    )
    return bk.set(a, idx, v)


def xrow(a1, i, di=0):
    """1-D x-array (``ib:ie``) restricted to Fortran range i, broadcastable on [k,j,i]."""
    return a1[fi(i[0]) + di : fi(i[1]) + di + 1]


def yrow(a1, j, dj=0):
    """1-D y-array (``jb:je``) restricted to range j, shaped (nj', 1)."""
    return a1[fi(j[0]) + dj : fi(j[1]) + dj + 1][:, None]


def col(a3, k, dk=0):
    """``a3(1,1,k)`` for a 3-D array: shape (nk', 1, 1)."""
    return a3[fk(k[0]) + dk : fk(k[1]) + dk + 1, fi(1) : fi(1) + 1, fi(1) : fi(1) + 1]


# ---------------------------------------------------------------------------
# flux functions
# ---------------------------------------------------------------------------


def flx3(s1, s2, s3):
    # 3rd-order flux (Wicker and Skamarock 2002)
    return (-1.0 / 6.0) * s1 + (5.0 / 6.0) * s2 + (2.0 / 6.0) * s3


def flx5(s1, s2, s3, s4, s5):
    # 5th-order flux (Wicker and Skamarock 2002)
    return (
        (((2.0 / 60.0) * s1 + (-13.0 / 60.0) * s2) + (47.0 / 60.0) * s3)
        + (27.0 / 60.0) * s4
    ) + (-3.0 / 60.0) * s5


def _upwind5(xp, vel, a_up, a_dn):
    """vel*flx5(a_up...) if vel>=0 else vel*flx5(a_dn...); a_up/a_dn are 5-tuples."""
    return xp.where(vel >= 0.0, vel * flx5(*a_up), vel * flx5(*a_dn))


# ---------------------------------------------------------------------------
# horizontal 5th-order fluxes
# ---------------------------------------------------------------------------


def hadv_flx5(bk, g, *, stag, c1, c2, rru, rrv, a, cfg=None):
    """Port of ``hadv_flx5`` (periodic lateral boundaries: ibw=ibe=ibs=ibn=0).

    Returns {'dumx', 'dumy'} (scalar-shaped arrays). Only the points that the
    advs/advu/advv/advw tendency loops use are computed (identical values to CM1).
    """
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    dumx = bk.zeros(g.s_shape)
    dumy = bk.zeros(g.s_shape)

    def A(k, j, i, dk=0, dj=0, di=0):
        return sub3(a, k, j, i, dk, dj, di)

    if stag == 1:
        K = (1, nk)
        # dumx: i=1..ni+1, j=1..nj
        J, I = (1, nj), (1, ni + 1)
        u = sub3(rru, K, J, I)
        fx = _upwind5(
            xp,
            u,
            [A(K, J, I, di=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, di=d) for d in (2, 1, 0, -1, -2)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        # dumy: i=1..ni, j=1..nj+1
        J, I = (1, nj + 1), (1, ni)
        v = sub3(rrv, K, J, I)
        fy = _upwind5(
            xp,
            v,
            [A(K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, dj=d) for d in (2, 1, 0, -1, -2)],
        )
        dumy = put3(bk, dumy, K, J, I, fy)

    elif stag == 2:
        # u-staggered (i1=1, i2=ni+1 periodic; i1=2 / i2=ni at open boundaries)
        i1, i2 = _irange(g, cfg)
        K = (1, nk)
        J, I = (1, nj), (i1 - 1, i2)
        ubar = 0.5 * (sub3(rru, K, J, I) + sub3(rru, K, J, I, di=1))
        fx = _upwind5(
            xp,
            ubar,
            [A(K, J, I, di=d) for d in (-2, -1, 0, 1, 2)],
            [A(K, J, I, di=d) for d in (3, 2, 1, 0, -1)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        J, I = (1, nj + 1), (i1, i2)
        vbar = 0.5 * (sub3(rrv, K, J, I) + sub3(rrv, K, J, I, di=-1))
        fy = _upwind5(
            xp,
            vbar,
            [A(K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, dj=d) for d in (2, 1, 0, -1, -2)],
        )
        dumy = put3(bk, dumy, K, J, I, fy)

    elif stag == 3:
        # v-staggered (j1=1, j2=nj+1 periodic; j1=2 / j2=nj at open boundaries)
        j1, j2 = _jrange(g, cfg)
        K = (1, nk)
        J, I = (j1, j2), (1, ni + 1)
        ubar = 0.5 * (sub3(rru, K, J, I) + sub3(rru, K, J, I, dj=-1))
        fx = _upwind5(
            xp,
            ubar,
            [A(K, J, I, di=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, di=d) for d in (2, 1, 0, -1, -2)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        J, I = (j1 - 1, j2), (1, ni)
        vbar = 0.5 * (sub3(rrv, K, J, I) + sub3(rrv, K, J, I, dj=1))
        fy = _upwind5(
            xp,
            vbar,
            [A(K, J, I, dj=d) for d in (-2, -1, 0, 1, 2)],
            [A(K, J, I, dj=d) for d in (3, 2, 1, 0, -1)],
        )
        dumy = put3(bk, dumy, K, J, I, fy)

    elif stag == 4:
        # w-staggered: k=2..nk
        K = (2, nk)
        J, I = (1, nj), (1, ni + 1)
        cc2 = 0.5 * (sub3(c2, K, J, I, di=-1) + sub3(c2, K, J, I))
        cc1 = 1.0 - cc2
        ubar = cc2 * sub3(rru, K, J, I) + cc1 * sub3(rru, K, J, I, dk=-1)
        fx = _upwind5(
            xp,
            ubar,
            [A(K, J, I, di=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, di=d) for d in (2, 1, 0, -1, -2)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        J, I = (1, nj + 1), (1, ni)
        cc2 = 0.5 * (sub3(c2, K, J, I, dj=-1) + sub3(c2, K, J, I))
        cc1 = 1.0 - cc2
        vbar = cc2 * sub3(rrv, K, J, I) + cc1 * sub3(rrv, K, J, I, dk=-1)
        fy = _upwind5(
            xp,
            vbar,
            [A(K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, dj=d) for d in (2, 1, 0, -1, -2)],
        )
        dumy = put3(bk, dumy, K, J, I, fy)
    else:
        raise ValueError(f"bad stag {stag}")

    return {"dumx": dumx, "dumy": dumy}


# ---------------------------------------------------------------------------
# vertical 5th-order fluxes
# ---------------------------------------------------------------------------


def _vgrid(g, stag, cfg=None):
    """(J, I) horizontal ranges for the vertical-flux loops."""
    if stag == 1 or stag == 4:
        return (1, g.nj), (1, g.ni)
    if stag == 2:
        return (1, g.nj), _irange(g, cfg)
    if stag == 3:
        return _jrange(g, cfg), (1, g.ni)
    raise ValueError(f"bad stag {stag}")


def _wvel(g, stag, rrw, K, J, I):
    """Vertical mass flux at the flux point for each staggering."""
    if stag == 1:
        return sub3(rrw, K, J, I)
    if stag == 2:
        return 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, di=-1))
    if stag == 3:
        return 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dj=-1))
    # stag == 4
    return 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dk=1))


def _cc2(g, stag, c2, K, J, I):
    if stag == 2:
        return 0.5 * (sub3(c2, K, J, I, di=-1) + sub3(c2, K, J, I))
    return 0.5 * (sub3(c2, K, J, I, dj=-1) + sub3(c2, K, J, I))


def vadv_lwr_flx_23(bk, g, *, stag, c1, c2, rrw, dumz, a, cfg=None):
    """Port of ``vadv_lwr_flx_23`` (periodic). Returns {'dumz'}."""
    xp = bk.xp
    nk = g.nk
    J, I = _vgrid(g, stag, cfg)

    def A(k, dk):
        return sub3(a, (k, k), J, I, dk=dk)

    if stag in (1, 2, 3):
        for k, low in ((2, True), (nk, False)):
            K = (k, k)
            w = _wvel(g, stag, rrw, K, J, I)
            if stag == 1:
                cen = sub3(c1, K, J, I) * A(k, -1) + sub3(c2, K, J, I) * A(k, 0)
            else:
                cc2 = _cc2(g, stag, c2, K, J, I)
                cc1 = 1.0 - cc2
                cen = cc1 * A(k, -1) + cc2 * A(k, 0)
            if low:
                v = xp.where(w >= 0.0, w * cen, w * flx3(A(k, 1), A(k, 0), A(k, -1)))
            else:
                v = xp.where(w >= 0.0, w * flx3(A(k, -2), A(k, -1), A(k, 0)), w * cen)
            dumz = put3(bk, dumz, K, J, I, v)
    else:
        K = (1, 1)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            (w * 0.5) * (A(1, 0) + A(1, 1)),
            w * flx3(A(1, 2), A(1, 1), A(1, 0)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
        K = (nk, nk)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            w * flx3(A(nk, -1), A(nk, 0), A(nk, 1)),
            (w * 0.5) * (A(nk, 0) + A(nk, 1)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
    return {"dumz": dumz}


def vadv_lwr_flx_35(bk, g, *, stag, c1, c2, rrw, dumz, a, cfg=None):
    """Port of ``vadv_lwr_flx_35`` (periodic). Returns {'dumz'}."""
    xp = bk.xp
    nk = g.nk
    J, I = _vgrid(g, stag, cfg)

    def A(k, dk):
        return sub3(a, (k, k), J, I, dk=dk)

    if stag in (1, 2, 3):
        k = 3
        K = (k, k)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            w * flx3(A(k, -2), A(k, -1), A(k, 0)),
            w * flx5(A(k, 2), A(k, 1), A(k, 0), A(k, -1), A(k, -2)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
        k = nk - 1
        K = (k, k)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            w * flx5(A(k, -3), A(k, -2), A(k, -1), A(k, 0), A(k, 1)),
            w * flx3(A(k, 1), A(k, 0), A(k, -1)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
    else:
        k = 2
        K = (k, k)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            w * flx3(A(k, -1), A(k, 0), A(k, 1)),
            w * flx5(A(k, 3), A(k, 2), A(k, 1), A(k, 0), A(k, -1)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
        k = nk - 1
        K = (k, k)
        w = _wvel(g, stag, rrw, K, J, I)
        v = xp.where(
            w >= 0.0,
            w * flx5(A(k, -2), A(k, -1), A(k, 0), A(k, 1), A(k, 2)),
            w * flx3(A(k, 2), A(k, 1), A(k, 0)),
        )
        dumz = put3(bk, dumz, K, J, I, v)
    return {"dumz": dumz}


def vadv_flx5(bk, g, *, stag, c1, c2, rrw, a, cfg=None):
    """Port of ``vadv_flx5`` (periodic), including the lower-order fluxes near
    the bottom and top. Returns {'dumz'} (scalar-shaped).

    stag 1/2/3: fluxes at w levels k=2..nk; stag 4: at scalar levels k=1..nk.
    """
    xp = bk.xp
    nk = g.nk
    if nk < 6:
        raise NotImplementedError("vadv_flx5 needs nk >= 6")
    dumz = bk.zeros(g.s_shape)
    J, I = _vgrid(g, stag, cfg)
    if stag in (1, 2, 3):
        K = (4, nk - 2)
        w = _wvel(g, stag, rrw, K, J, I)
        v = _upwind5(
            xp,
            w,
            [sub3(a, K, J, I, dk=d) for d in (-3, -2, -1, 0, 1)],
            [sub3(a, K, J, I, dk=d) for d in (2, 1, 0, -1, -2)],
        )
    elif stag == 4:
        K = (3, nk - 2)
        w = _wvel(g, stag, rrw, K, J, I)
        v = _upwind5(
            xp,
            w,
            [sub3(a, K, J, I, dk=d) for d in (-2, -1, 0, 1, 2)],
            [sub3(a, K, J, I, dk=d) for d in (3, 2, 1, 0, -1)],
        )
    else:
        raise ValueError(f"bad stag {stag}")
    if K[1] >= K[0]:
        dumz = put3(bk, dumz, K, J, I, v)
    dumz = vadv_lwr_flx_23(
        bk, g, stag=stag, c1=c1, c2=c2, rrw=rrw, dumz=dumz, a=a, cfg=cfg
    )["dumz"]
    dumz = vadv_lwr_flx_35(
        bk, g, stag=stag, c1=c1, c2=c2, rrw=rrw, dumz=dumz, a=a, cfg=cfg
    )["dumz"]
    return {"dumz": dumz}


# ---------------------------------------------------------------------------
# WENO (weno_order=5, siform=2: Borges et al. 2008 smoothness indicators)
# ---------------------------------------------------------------------------


def weno5(xp, s1, s2, s3, s4, s5, weps):
    """Port of the ``weno5`` function (siform=2), same operation order."""
    d1 = (s1 - 2.0 * s2) + s3
    e1 = (s1 - 4.0 * s2) + 3.0 * s3
    d2 = (s2 - 2.0 * s3) + s4
    e2 = s2 - s4
    d3 = (s3 - 2.0 * s4) + s5
    e3 = (3.0 * s3 - 4.0 * s4) + s5
    b1 = (13.0 / 12.0) * (d1 * d1) + 0.25 * (e1 * e1)
    b2 = (13.0 / 12.0) * (d2 * d2) + 0.25 * (e2 * e2)
    b3 = (13.0 / 12.0) * (d3 * d3) + 0.25 * (e3 * e3)
    t = xp.abs(b1 - b3)
    r1 = xp.minimum(1.0e30 + 0.0 * t, t / (b1 + weps))
    r2 = xp.minimum(1.0e30 + 0.0 * t, t / (b2 + weps))
    r3 = xp.minimum(1.0e30 + 0.0 * t, t / (b3 + weps))
    w1 = 0.1 * (1.0 + r1 * r1)
    w2 = 0.6 * (1.0 + r2 * r2)
    w3 = 0.3 * (1.0 + r3 * r3)
    p1 = ((2.0 / 6.0) * s1 + (-7.0 / 6.0) * s2) + (11.0 / 6.0) * s3
    p2 = ((-1.0 / 6.0) * s2 + (5.0 / 6.0) * s3) + (2.0 / 6.0) * s4
    p3 = ((2.0 / 6.0) * s3 + (5.0 / 6.0) * s4) + (-1.0 / 6.0) * s5
    return ((w1 * p1 + w2 * p2) + w3 * p3) / ((w1 + w2) + w3)


def upstrpd(xp, s1, s2, s3, weps):
    """Port of ``upstrpd`` (positive-definite upstream scheme, Beets & Koren)."""
    dd = s2 - s1
    rr = (s3 - s2) / (xp.copysign(weps + 0.0 * dd, dd) + dd)
    phi = xp.minimum(rr, xp.minimum((1.0 / 6.0) + (2.0 / 6.0) * rr, 1.0 + 0.0 * rr))
    phi = xp.maximum(0.0 * rr, phi)
    return s2 + phi * (s2 - s1)


def _sel5(xp, vel, up, dn):
    """s1..s5 of the upwind stencil: up if vel>=0 else dn."""
    return [xp.where(vel >= 0.0, u, d) for u, d in zip(up, dn)]


def _bmax_ok(xp, s, bmin):
    m = xp.abs(s[0])
    for x in s[1:]:
        m = xp.maximum(m, xp.abs(x))
    return m > bmin


def hadv_weno5(bk, g, *, stag, c1, c2, rru, rrv, a, pdef, weps, cfg=None):
    """Port of ``hadv_weno5`` for scalars (stag=1) and w-staggered fields
    (stag=4, k=2..nk; CM1 uses it for TKE with advwenos). With pdef=1 the
    flux is zero where max|s1..s5| <= weps (stag=1 only, as in CM1).
    Returns {'dumx', 'dumy'}."""
    if stag not in (1, 4):
        raise NotImplementedError("hadv_weno5: only stag=1 and stag=4 ported")
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    dumx = bk.zeros(g.s_shape)
    dumy = bk.zeros(g.s_shape)
    if stag == 4:
        K = (2, nk)
        # x: i=1..ni+1, j=1..nj ; y: i=1..ni, j=1..nj+1 (points advw uses)
        J, I = (1, nj), (1, ni + 1)
        cc2 = 0.5 * (sub3(c2, K, J, I, di=-1) + sub3(c2, K, J, I))
        cc1 = 1.0 - cc2
        vel = cc2 * sub3(rru, K, J, I) + cc1 * sub3(rru, K, J, I, dk=-1)
        up = [sub3(a, K, J, I, di=d) for d in (-3, -2, -1, 0, 1)]
        dn = [sub3(a, K, J, I, di=d) for d in (2, 1, 0, -1, -2)]
        dumx = put3(bk, dumx, K, J, I, vel * weno5(xp, *_sel5(xp, vel, up, dn), weps))
        J, I = (1, nj + 1), (1, ni)
        cc2 = 0.5 * (sub3(c2, K, J, I, dj=-1) + sub3(c2, K, J, I))
        cc1 = 1.0 - cc2
        vel = cc2 * sub3(rrv, K, J, I) + cc1 * sub3(rrv, K, J, I, dk=-1)
        up = [sub3(a, K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)]
        dn = [sub3(a, K, J, I, dj=d) for d in (2, 1, 0, -1, -2)]
        dumy = put3(bk, dumy, K, J, I, vel * weno5(xp, *_sel5(xp, vel, up, dn), weps))
        return {"dumx": dumx, "dumy": dumy}
    K = (1, nk)
    for which in ("x", "y"):
        if which == "x":
            J, I = (1, nj), (1, ni + 1)
            vel = sub3(rru, K, J, I)
            up = [sub3(a, K, J, I, di=d) for d in (-3, -2, -1, 0, 1)]
            dn = [sub3(a, K, J, I, di=d) for d in (2, 1, 0, -1, -2)]
        else:
            J, I = (1, nj + 1), (1, ni)
            vel = sub3(rrv, K, J, I)
            up = [sub3(a, K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)]
            dn = [sub3(a, K, J, I, dj=d) for d in (2, 1, 0, -1, -2)]
        s = _sel5(xp, vel, up, dn)
        f = vel * weno5(xp, *s, weps)
        if pdef == 1:
            f = xp.where(_bmax_ok(xp, s, weps), f, 0.0 * f)
        if which == "x":
            dumx = put3(bk, dumx, K, J, I, f)
        else:
            dumy = put3(bk, dumy, K, J, I, f)
    return {"dumx": dumx, "dumy": dumy}


def vadv_lwr_weno_23(bk, g, *, stag, c1, c2, rrw, dumz, a, pdef, weps, cfg=None):
    """Port of ``vadv_lwr_weno_23`` (stag=1 and stag=4). Returns {'dumz'}."""
    if stag not in (1, 4):
        raise NotImplementedError("vadv_lwr_weno_23: only stag=1, 4 ported")
    xp = bk.xp
    nk = g.nk
    J, I = (1, g.nj), (1, g.ni)

    def A(k, dk):
        return sub3(a, (k, k), J, I, dk=dk)

    if stag == 4:
        for k in (1, nk):
            K = (k, k)
            w = 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dk=1))
            cen = (w * 0.5) * (A(k, 0) + A(k, 1))
            if k == 1:
                up = w * upstrpd(xp, A(k, 2), A(k, 1), A(k, 0), weps)
                v = xp.where(w >= 0.0, cen, up)
            else:
                up = w * upstrpd(xp, A(k, -1), A(k, 0), A(k, 1), weps)
                v = xp.where(w > 0.0, up, cen)
            dumz = put3(bk, dumz, K, J, I, v)
        return {"dumz": dumz}

    k = 2
    K = (k, k)
    w = sub3(rrw, K, J, I)
    cen = sub3(c1, K, J, I) * A(k, -1) + sub3(c2, K, J, I) * A(k, 0)
    v = xp.where(w >= 0.0, w * cen, w * upstrpd(xp, A(k, 1), A(k, 0), A(k, -1), weps))
    dumz = put3(bk, dumz, K, J, I, v)
    k = nk
    K = (k, k)
    w = sub3(rrw, K, J, I)
    cen = sub3(c1, K, J, I) * A(k, -1) + sub3(c2, K, J, I) * A(k, 0)
    v = xp.where(w > 0.0, w * upstrpd(xp, A(k, -2), A(k, -1), A(k, 0), weps), w * cen)
    dumz = put3(bk, dumz, K, J, I, v)
    return {"dumz": dumz}


def vadv_lwr_weno_35(bk, g, *, stag, c1, c2, rrw, dumz, a, pdef, weps, cfg=None):
    """Port of ``vadv_lwr_weno_35`` (stag=1 and stag=4). Returns {'dumz'}."""
    if stag not in (1, 4):
        raise NotImplementedError("vadv_lwr_weno_35: only stag=1, 4 ported")
    xp = bk.xp
    nk = g.nk
    J, I = (1, g.nj), (1, g.ni)

    def A(k, dk):
        return sub3(a, (k, k), J, I, dk=dk)

    if stag == 4:
        k = 2
        K = (k, k)
        w = 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dk=1))
        v = xp.where(
            w >= 0.0,
            w * upstrpd(xp, A(k, -1), A(k, 0), A(k, 1), weps),
            w * weno5(xp, A(k, 3), A(k, 2), A(k, 1), A(k, 0), A(k, -1), weps),
        )
        dumz = put3(bk, dumz, K, J, I, v)
        k = nk - 1
        K = (k, k)
        w = 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dk=1))
        v = xp.where(
            w > 0.0,
            w * weno5(xp, A(k, -2), A(k, -1), A(k, 0), A(k, 1), A(k, 2), weps),
            w * upstrpd(xp, A(k, 2), A(k, 1), A(k, 0), weps),
        )
        dumz = put3(bk, dumz, K, J, I, v)
        return {"dumz": dumz}

    k = 3
    K = (k, k)
    w = sub3(rrw, K, J, I)
    v = xp.where(
        w >= 0.0,
        w * upstrpd(xp, A(k, -2), A(k, -1), A(k, 0), weps),
        w * weno5(xp, A(k, 2), A(k, 1), A(k, 0), A(k, -1), A(k, -2), weps),
    )
    dumz = put3(bk, dumz, K, J, I, v)
    k = nk - 1
    K = (k, k)
    w = sub3(rrw, K, J, I)
    v = xp.where(
        w >= 0.0,
        w * weno5(xp, A(k, -3), A(k, -2), A(k, -1), A(k, 0), A(k, 1), weps),
        w * upstrpd(xp, A(k, 1), A(k, 0), A(k, -1), weps),
    )
    dumz = put3(bk, dumz, K, J, I, v)
    return {"dumz": dumz}


def vadv_lwr_weno_35_b(bk, g, *, stag, c1, c2, rrw, dumz, a, pdef, weps, cfg=None):
    """Port of ``vadv_lwr_weno_35_b`` (only overwrites the upward flux at k=3
    and the downward flux at k=nk-1). Returns {'dumz'}."""
    xp = bk.xp
    nk = g.nk
    J, I = (1, g.nj), (1, g.ni)

    def A(k, dk):
        return sub3(a, (k, k), J, I, dk=dk)

    k = 3
    K = (k, k)
    w = sub3(rrw, K, J, I)
    v = xp.where(
        w > 0.0, w * upstrpd(xp, A(k, -2), A(k, -1), A(k, 0), weps), sub3(dumz, K, J, I)
    )
    dumz = put3(bk, dumz, K, J, I, v)
    k = nk - 1
    K = (k, k)
    w = sub3(rrw, K, J, I)
    v = xp.where(
        w < 0.0, w * upstrpd(xp, A(k, 1), A(k, 0), A(k, -1), weps), sub3(dumz, K, J, I)
    )
    dumz = put3(bk, dumz, K, J, I, v)
    return {"dumz": dumz}


def vadv_weno5(bk, g, *, stag, c1, c2, rrw, a, pdef, pdefweno, weps, cfg=None):
    """Port of ``vadv_weno5`` for scalars (stag=1), both the pdefweno=1 branch
    (k=3..nk-1 with max|s|>weps check, then lwr_weno_23 + lwr_weno_35_b) and
    the pdefweno=0 branch (k=4..nk-2, then lwr_weno_23 + lwr_weno_35).
    For stag=4 (w-staggered, e.g. TKE): k=3..nk-2 with wbar, then the stag=4
    lwr_weno_23/35 (pdef/pdefweno unused, as in CM1).
    Returns {'dumz'}."""
    if stag not in (1, 4):
        raise NotImplementedError("vadv_weno5: only stag=1 and stag=4 ported")
    xp = bk.xp
    nk = g.nk
    if nk < 6:
        raise NotImplementedError("vadv_weno5 needs nk >= 6")
    dumz = bk.zeros(g.s_shape)
    J, I = (1, g.nj), (1, g.ni)
    kw = dict(stag=stag, c1=c1, c2=c2, rrw=rrw, a=a, pdef=pdef, weps=weps, cfg=cfg)
    if stag == 4:
        K = (3, nk - 2)
        w = 0.5 * (sub3(rrw, K, J, I) + sub3(rrw, K, J, I, dk=1))
        up = [sub3(a, K, J, I, dk=d) for d in (-2, -1, 0, 1, 2)]
        dn = [sub3(a, K, J, I, dk=d) for d in (3, 2, 1, 0, -1)]
        dumz = put3(bk, dumz, K, J, I, w * weno5(xp, *_sel5(xp, w, up, dn), weps))
        dumz = vadv_lwr_weno_23(bk, g, dumz=dumz, **kw)["dumz"]
        dumz = vadv_lwr_weno_35(bk, g, dumz=dumz, **kw)["dumz"]
        return {"dumz": dumz}
    if pdefweno == 1:
        K = (3, nk - 1)
        w = sub3(rrw, K, J, I)
        up = [sub3(a, K, J, I, dk=d) for d in (-3, -2, -1, 0, 1)]
        dn = [sub3(a, K, J, I, dk=d) for d in (2, 1, 0, -1, -2)]
        s = _sel5(xp, w, up, dn)
        doit = _bmax_ok(xp, s, weps)
        kk = xp.reshape(xp.arange(K[0], K[1] + 1), (-1, 1, 1))
        doit = doit & ~((kk == 3) & (w > 0.0)) & ~((kk == nk - 1) & (w < 0.0))
        f = w * weno5(xp, *s, weps)
        dumz = put3(bk, dumz, K, J, I, xp.where(doit, f, 0.0 * f))
        dumz = vadv_lwr_weno_23(bk, g, dumz=dumz, **kw)["dumz"]
        dumz = vadv_lwr_weno_35_b(bk, g, dumz=dumz, **kw)["dumz"]
    else:
        K = (4, nk - 2)
        w = sub3(rrw, K, J, I)
        up = [sub3(a, K, J, I, dk=d) for d in (-3, -2, -1, 0, 1)]
        dn = [sub3(a, K, J, I, dk=d) for d in (2, 1, 0, -1, -2)]
        s = _sel5(xp, w, up, dn)
        dumz = put3(bk, dumz, K, J, I, w * weno5(xp, *s, weps))
        dumz = vadv_lwr_weno_23(bk, g, dumz=dumz, **kw)["dumz"]
        dumz = vadv_lwr_weno_35(bk, g, dumz=dumz, **kw)["dumz"]
    return {"dumz": dumz}


# ---------------------------------------------------------------------------
# terrain slope
# ---------------------------------------------------------------------------


def _flx4(s1, s2, s3, s4):
    return (7.0 / 12.0) * (s3 + s2) - (1.0 / 12.0) * (s4 + s1)


def _flx6(s1, s2, s3, s4, s5, s6):
    return ((37.0 / 60.0) * (s4 + s3) + (-8.0 / 60.0) * (s5 + s2)) + (1.0 / 60.0) * (
        s6 + s1
    )


def zsgrad(bk, g, cfg=None, *, zs, gz, dzdx, dzdy, uh, vh):
    """Port of ``zsgrad``: terrain slopes dzdx (i, j = 1..ni, 1..nj) and dzdy
    from zs interpolated to u / v points with the order of the horizontal
    advection scheme, n = max(hadvordrs, hadvordrv).  ``zs``, ``gz``,
    ``dzdx``, ``dzdy`` are 2-D ``[j, i]`` (``Grid.s_shape[1:]``); gz is
    CM1's scratch array (left holding the v-point zs on j=1..nj+1, i=1..ni,
    and the u-point zs at j=1..nj, i=ni+1 / unchanged elsewhere).
    Returns {'gz', 'dzdx', 'dzdy'}."""
    n = max(int(_opt(cfg, "hadvordrs", 5)), int(_opt(cfg, "hadvordrv", 5)))
    ni, nj = g.ni, g.nj
    J, I = slice(fi(1), fi(nj) + 1), slice(fi(1), fi(ni) + 1)
    Ip, Jp = slice(fi(1), fi(ni + 1) + 1), slice(fi(1), fi(nj + 1) + 1)

    def sx(d):  # zs(i+d, j) for i = 1..ni+1, j = 1..nj
        return zs[J, fi(1) + d : fi(ni + 1) + d + 1]

    def sy(d):  # zs(i, j+d) for j = 1..nj+1, i = 1..ni
        return zs[fi(1) + d : fi(nj + 1) + d + 1, I]

    if n <= 2:
        fx, fy = 0.5 * (sx(-1) + sx(0)), 0.5 * (sy(-1) + sy(0))
    elif n in (3, 4):
        fx = _flx4(sx(-2), sx(-1), sx(0), sx(1))
        fy = _flx4(sy(-2), sy(-1), sy(0), sy(1))
    elif n in (5, 6):
        fx = _flx6(sx(-3), sx(-2), sx(-1), sx(0), sx(1), sx(2))
        fy = _flx6(sy(-3), sy(-2), sy(-1), sy(0), sy(1), sy(2))
    else:
        raise NotImplementedError(f"zsgrad: advection order {n} > 6 not ported")
    rdx, rdy = 1.0 / g.dx, 1.0 / g.dy
    gz = bk.set(bk.copy(gz), (J, Ip), fx)
    dzdx = bk.set(bk.copy(dzdx), (J, I), (fx[:, 1:] - fx[:, :-1]) * rdx * uh[I])
    gz = bk.set(gz, (Jp, I), fy)
    dzdy = bk.set(bk.copy(dzdy), (J, I), (fy[1:] - fy[:-1]) * rdy * vh[J][:, None])
    return {"gz": gz, "dzdx": dzdx, "dzdy": dzdy}
