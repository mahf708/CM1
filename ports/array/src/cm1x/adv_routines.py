"""Flux-form advection kernels, ported from src/adv_routines.F.

Only the routines selected by the reference case (hadvordr*=vadvordr*=5, no WENO,
periodic lateral boundaries, i.e. ibw=ibe=ibs=ibn=0) are ported:

  * ``flx3``, ``flx5``                 (statement-like flux functions)
  * ``hadv_flx5``                      (horizontal 5th-order upwind fluxes)
  * ``vadv_flx5``                      (vertical 5th-order upwind fluxes)
  * ``vadv_lwr_flx_23``, ``vadv_lwr_flx_35`` (lower-order fluxes near bottom/top)

``stag`` follows CM1: 1 = scalar, 2 = u, 3 = v, 4 = w.

The flux arrays ``dumx``, ``dumy``, ``dumz`` are scalar-shaped
(``ib:ie, jb:je, kb:ke``) exactly as in CM1. Points that CM1 does not compute
are returned as zero (in Fortran they hold stale scratch values).

Index helpers: Fortran ``i`` -> Python ``fi(i)``, Fortran ``k`` -> ``fk(k)``.
"""

from __future__ import annotations

from .grid import fi, fk

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
        # u-staggered; periodic: i1=1, i2=ni+1
        K = (1, nk)
        J, I = (1, nj), (0, ni + 1)
        ubar = 0.5 * (sub3(rru, K, J, I) + sub3(rru, K, J, I, di=1))
        fx = _upwind5(
            xp,
            ubar,
            [A(K, J, I, di=d) for d in (-2, -1, 0, 1, 2)],
            [A(K, J, I, di=d) for d in (3, 2, 1, 0, -1)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        J, I = (1, nj + 1), (1, ni + 1)
        vbar = 0.5 * (sub3(rrv, K, J, I) + sub3(rrv, K, J, I, di=-1))
        fy = _upwind5(
            xp,
            vbar,
            [A(K, J, I, dj=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, dj=d) for d in (2, 1, 0, -1, -2)],
        )
        dumy = put3(bk, dumy, K, J, I, fy)

    elif stag == 3:
        # v-staggered; periodic: j1=1, j2=nj+1
        K = (1, nk)
        J, I = (1, nj + 1), (1, ni + 1)
        ubar = 0.5 * (sub3(rru, K, J, I) + sub3(rru, K, J, I, dj=-1))
        fx = _upwind5(
            xp,
            ubar,
            [A(K, J, I, di=d) for d in (-3, -2, -1, 0, 1)],
            [A(K, J, I, di=d) for d in (2, 1, 0, -1, -2)],
        )
        dumx = put3(bk, dumx, K, J, I, fx)
        J, I = (0, nj + 1), (1, ni)
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


def _vgrid(g, stag):
    """(J, I) horizontal ranges for the vertical-flux loops (periodic)."""
    if stag == 1 or stag == 4:
        return (1, g.nj), (1, g.ni)
    if stag == 2:
        return (1, g.nj), (1, g.ni + 1)
    if stag == 3:
        return (1, g.nj + 1), (1, g.ni)
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


def vadv_lwr_flx_23(bk, g, *, stag, c1, c2, rrw, dumz, a):
    """Port of ``vadv_lwr_flx_23`` (periodic). Returns {'dumz'}."""
    xp = bk.xp
    nk = g.nk
    J, I = _vgrid(g, stag)

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


def vadv_lwr_flx_35(bk, g, *, stag, c1, c2, rrw, dumz, a):
    """Port of ``vadv_lwr_flx_35`` (periodic). Returns {'dumz'}."""
    xp = bk.xp
    nk = g.nk
    J, I = _vgrid(g, stag)

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
    J, I = _vgrid(g, stag)
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
    dumz = vadv_lwr_flx_23(bk, g, stag=stag, c1=c1, c2=c2, rrw=rrw, dumz=dumz, a=a)[
        "dumz"
    ]
    dumz = vadv_lwr_flx_35(bk, g, stag=stag, c1=c1, c2=c2, rrw=rrw, dumz=dumz, a=a)[
        "dumz"
    ]
    return {"dumz": dumz}
