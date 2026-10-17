"""Acoustic (small-time-step) solver for psolver=3, ported from src/sound.F.

Split-explicit, vertically implicit sound-wave integration (Klemp-Wilhelmson
style) as used by CM1's RK3 large step. Only the code path of the reference
case is supported: no terrain, Cartesian (axisymm=0), periodic lateral
boundaries (wbc=ebc=sbc=nbc=1), no convergence forcing / updraft nudging /
immersed boundaries. Rayleigh damping of w is *not* applied inside ``sound``
(CM1 adds it to ``wten`` before the call), so ``irdamp`` is irrelevant here.

Index conventions follow cm1x.grid: arrays are ``a[k, j, i]`` with halos, and a
Fortran index ``i`` is Python ``i + 2`` (``k`` unchanged).
"""

from __future__ import annotations

import math

from . import constants as C

nrkmax = 3  # constants.F: integer, parameter :: nrkmax = 3

_DEFAULTS = dict(
    nsound=6,
    alph=0.6,
    kdiv=0.1,
    smeps=1.0e-30,  # cm1.F default (1e-60 only with -DDP)
    terrain_flag=False,
    axisymm=0,
    wbc=1,
    ebc=1,
    sbc=1,
    nbc=1,
    irbc=4,
    roflux=0,
    convinit=0,
    wnudge=0,
    do_ib=False,
)


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    # dict.get / Config.get: an unset option falls back to this module's
    # default rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])


def _check_options(cfg):
    if _opt(cfg, "terrain_flag"):
        raise NotImplementedError("sound: terrain_flag=.true. not ported")
    if _opt(cfg, "axisymm") != 0:
        raise NotImplementedError("sound: axisymm != 0 not ported")
    for b in ("wbc", "ebc", "sbc", "nbc"):
        if _opt(cfg, b) != 1:
            raise NotImplementedError(
                f"sound: {b}={_opt(cfg, b)} (only periodic, 1, ported)"
            )
    if _opt(cfg, "roflux") != 0:
        raise NotImplementedError("sound: roflux=1 not ported")
    if _opt(cfg, "convinit") != 0:
        raise NotImplementedError("sound: convinit=1 not ported")
    if _opt(cfg, "wnudge") != 0:
        raise NotImplementedError("sound: wnudge=1 not ported")
    if _opt(cfg, "do_ib"):
        raise NotImplementedError("sound: immersed boundaries (do_ib) not ported")


def small_steps(dt, nrk, nsound):
    """Number of acoustic steps ``nloop`` and their length ``dts`` (sound.F)."""
    if nrk == 1:
        nloop = int(math.floor(float(nsound) / 3.0 + 0.5))  # nint, positive arg
        dts = dt / (nloop * 3.0)
        if dts > (dt / nsound):
            nloop = nloop + 1
            dts = dt / (nloop * 3.0)
    elif nrk == 2:
        nloop = int(0.5 * nsound)
        dts = dt / nsound
    elif nrk == 3:
        nloop = nsound
        dts = dt / nsound
    else:
        raise ValueError(f"nrk={nrk} (nrkmax=3)")
    return nloop, dts


def bcp(bk, g, cfg, p):
    """bc.F ``bcp`` (periodic branch, non-MPI): 1-point halo of a pressure
    array, k=1..nk.  West/east over j=0..nj+1, then south/north over
    i=0..ni+1 (so corners come out doubly periodic)."""
    for b in ("wbc", "ebc", "sbc", "nbc"):
        if _opt(cfg, b) != 1:
            raise NotImplementedError(f"bcp: {b}={_opt(cfg, b)} not ported")
    ni, nj, nk = g.ni, g.nj, g.nk
    K = slice(1, nk + 1)
    J0 = slice(2, nj + 4)  # Fortran j = 0..nj+1
    I0 = slice(2, ni + 4)
    # Fortran 0 -> py 2, 1 -> 3, ni -> ni+2, ni+1 -> ni+3
    p = bk.set(p, (K, J0, 2), p[K, J0, ni + 2])
    p = bk.set(p, (K, J0, ni + 3), p[K, J0, 3])
    p = bk.set(p, (K, 2, I0), p[K, nj + 2, I0])
    p = bk.set(p, (K, nj + 3, I0), p[K, 3, I0])
    return p


def sound(
    bk,
    g,
    cfg=None,
    *,
    dt,
    nrk,
    uh,
    uf,
    vh,
    vf,
    mh,
    mf,
    c1,
    c2,
    rr0,
    rf0,
    dtu0,
    dtv0,
    ua,
    u3d,
    uten,
    va,
    v3d,
    vten,
    wa,
    w3d,
    wten,
    ppi,
    pp3d,
    piadv,
    ppten,
    ppx,
    thv,
    ppterm,
    rru=None,
    rrv=None,
    rrw=None,
    dtu=None,
    dtv=None,
    ppd=None,
    fpk=None,
    qk=None,
    pk1=None,
    pk2=None,
    mm=None,
    dum1=None,
    dum2=None,
    **unused,
):
    """Port of sound.F ``sound`` (psolver=3), no terrain, periodic.

    Shapes: scalars ``g.s_shape`` (mh, c1, c2, rr0, rf0, ppi, pp3d, piadv,
    ppten, ppx, thv, ppterm and the scratch arrays), ``g.u_shape`` (ua, u3d,
    uten, rru), ``g.v_shape`` (va, v3d, vten, rrv), ``g.w_shape`` (mf, wa,
    w3d, wten, rrw).  ``uh``/``vh`` are 1-D over the i/j scalar points
    (length s_shape[2]/s_shape[1]); ``uf``/``vf`` over the staggered points.
    ``dtu0``/``dtv0`` are 2-D ``[j, i]`` (s_shape[1:]).  ``dt`` and ``nrk`` are
    Python numbers (they set the number of small steps).

    Optional intent(inout) work arrays default to zeros; only their ranges
    written by the Fortran are meaningful. Unused Fortran dummy arguments
    (xh, arh1, gz, radbcw, ...) may be passed and are ignored.

    Returns a dict with u3d, v3d, w3d, pp3d, ppd, ppx, rru, rrv, rrw, dtu, dtv
    and the work arrays fpk, qk, pk1, pk2, mm, dum1, dum2.
    """
    _check_options(cfg)
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    if nk < 3:
        raise NotImplementedError("sound: nk >= 3 required")
    alph = float(_opt(cfg, "alph"))
    kdiv = float(_opt(cfg, "kdiv"))
    smeps = float(_opt(cfg, "smeps"))
    nsound = int(_opt(cfg, "nsound"))
    rdx, rdy, rdz = 1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz

    def z(shape, a):
        return bk.zeros(shape) if a is None else bk.copy(a)

    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    rru, rrv, rrw = z(U, rru), z(V, rrv), z(W, rrw)
    dtu, dtv = z(S[1:], dtu), z(S[1:], dtv)
    ppd, fpk, qk, pk1, pk2, mm = (z(S, a) for a in (ppd, fpk, qk, pk1, pk2, mm))
    dum1, dum2 = z(S, dum1), z(S, dum2)
    u3d, v3d, w3d, pp3d, ppx = (bk.copy(a) for a in (u3d, v3d, w3d, pp3d, ppx))

    beta = 1.0 - alph
    asq = alph * alph
    nloop, dts = small_steps(float(dt), int(nrk), nsound)

    # python slices of Fortran ranges
    Ii = slice(3, ni + 3)  # i = 1..ni
    Ip = slice(3, ni + 4)  # i = 1..ni+1
    I0 = slice(2, ni + 4)  # i = 0..ni+1
    Ji = slice(3, nj + 3)
    Jp = slice(3, nj + 4)
    J0 = slice(2, nj + 4)
    K = slice(1, nk + 1)  # k = 1..nk
    Km, Kp = slice(0, nk), slice(2, nk + 2)  # k-1, k+1

    # dtu, dtv  (j,i = 1..nj+1, 1..ni+1)
    dtu = bk.set(dtu, (Jp, Ip), dts * dtu0[Jp, Ip])
    dtv = bk.set(dtv, (Jp, Ip), dts * dtv0[Jp, Ip])

    # --- arrays for the vertically implicit solver (no terrain) -----------
    def col(a, ks):
        return xp.reshape(a[ks, 3, 3], (-1, 1, 1))  # a(1,1,k)

    mhk = col(mh, K)
    tem = dts * rdz * mhk
    r2 = dts * rdz * mhk * col(rr0, K) * col(rf0, Kp)
    r1 = dts * rdz * mhk * col(rr0, K) * col(rf0, K)
    tem1 = dts * rdz * C.cp * col(mf, K)
    c1a, c1b = col(c1, K), col(c1, Kp)
    c2a, c2b = col(c2, K), col(c2, Kp)
    pa, pam, pap = piadv[K, Ji, Ii], piadv[Km, Ji, Ii], piadv[Kp, Ji, Ii]
    qk = bk.set(qk, (K, Ji, Ii), tem * ppterm[K, Ji, Ii])
    pk2 = bk.set(pk2, (K, Ji, Ii), r2 * (-c2b * pap + (1.0 - c1b) * pa))
    pk1 = bk.set(pk1, (K, Ji, Ii), r1 * (+c1a * pam + (c2a - 1.0) * pa))
    mm = bk.set(mm, (K, Ji, Ii), tem1 * (c2a * thv[K, Ji, Ii] + c1a * thv[Km, Ji, Ii]))
    pk1 = bk.set(pk1, (1, Ji, Ii), 0.0)
    pk2 = bk.set(pk2, (nk, Ji, Ii), 0.0)

    # --- prepare for acoustic steps --------------------------------------
    R0 = (K, J0, I0)
    if nrk != 1:
        u3d = bk.set(u3d, R0, ua[R0])
        v3d = bk.set(v3d, R0, va[R0])
        w3d = bk.set(w3d, R0, wa[R0])
        pp3d = bk.set(pp3d, R0, ppi[R0])
    ppd = bk.set(ppd, R0, ppx[R0])
    rru = bk.set(rru, R0, 0.0)
    rrv = bk.set(rrv, R0, 0.0)
    rrw = bk.set(rrw, R0, wa[R0] * beta)

    # loop-invariant pieces
    tu1 = rdx * C.cp * 0.5
    tv2 = rdy * C.cp * 0.5
    ufp = xp.reshape(uf[Ip], (1, 1, -1))
    vfp = xp.reshape(vf[Jp], (1, -1, 1))
    uhi = xp.reshape(uh[Ii], (1, 1, -1))
    vhj = xp.reshape(vh[Ji], (1, -1, 1))
    temx = dts * 0.5 * rdx
    temy = dts * 0.5 * rdy
    thv_u = thv[K, Jp, Ip] + thv[K, Jp, 2 : ni + 3]
    thv_v = thv[K, Jp, Ip] + thv[K, 2 : nj + 3, Ip]
    dtu_k = dtu[Jp, Ip]
    dtv_k = dtv[Jp, Ip]
    uten_r = uten[K, Jp, Ip]
    vten_r = vten[K, Jp, Ip]
    ppten_i = ppten[K, Ji, Ii]
    ppterm_i = ppterm[K, Ji, Ii]
    pa_i = piadv[K, Ji, Ii]
    dpx_m = pa_i - piadv[K, Ji, 2 : ni + 2]  # piadv(i)-piadv(i-1)
    dpx_p = piadv[K, Ji, 4 : ni + 4] - pa_i  # piadv(i+1)-piadv(i)
    dpy_m = pa_i - piadv[K, 2 : nj + 2, Ii]
    dpy_p = piadv[K, 4 : nj + 4, Ii] - pa_i
    qk_i, pk1_i, pk2_i = qk[K, Ji, Ii], pk1[K, Ji, Ii], pk2[K, Ji, Ii]
    wten_i = wten[:, Ji, Ii]
    zero = bk.asarray(0.0)

    for n in range(1, nloop + 1):
        # --- integrate u, v forward (k=1..nk, j=1..nj+1, i=1..ni+1)
        pc = ppd[K, Jp, Ip]
        u3d = bk.set(
            u3d,
            (K, Jp, Ip),
            u3d[K, Jp, Ip]
            + dtu_k * (uten_r - tu1 * (pc - ppd[K, Jp, 2 : ni + 3]) * ufp * thv_u),
        )
        v3d = bk.set(
            v3d,
            (K, Jp, Ip),
            v3d[K, Jp, Ip]
            + dtv_k * (vten_r - tv2 * (pc - ppd[K, 2 : nj + 3, Ip]) * vfp * thv_v),
        )

        # --- forcing for the pressure equation
        ul, ur = u3d[K, Ji, Ii], u3d[K, Ji, 4 : ni + 4]
        vl, vr = v3d[K, Ji, Ii], v3d[K, 4 : nj + 4, Ii]
        div = (ur - ul) * rdx * uhi + (vr - vl) * rdy * vhj
        div = xp.where(xp.abs(div) < smeps, zero, div)
        w_lo, w_hi = w3d[K, Ji, Ii], w3d[2 : nk + 2, Ji, Ii]
        fpk_i = (
            dts * (ppten_i - ppterm_i * div)
            + (
                -(ul * dpx_m + ur * dpx_p) * temx * uhi
                - (vl * dpy_m + vr * dpy_p) * temy * vhj
            )
            + beta * (w_lo * (pk1_i + qk_i) + w_hi * (pk2_i - qk_i))
        )
        fpk = bk.set(fpk, (K, Ji, Ii), fpk_i)

        # --- vertically implicit solve for w (k = 2..nk), Thomas algorithm
        pp = pp3d[:, Ji, Ii]
        fp = fpk[:, Ji, Ii]
        cof1, cof2 = {}, {}
        wnk = None
        for k in range(2, nk + 1):
            m = mm[k, Ji, Ii]
            bb = 1.0 - asq * (
                +m
                * (
                    -(pk1[k, Ji, Ii] + qk[k, Ji, Ii])
                    + (pk2[k - 1, Ji, Ii] - qk[k - 1, Ji, Ii])
                )
            )
            fwk = dts * wten_i[k] + beta * (-m * (pp[k] - pp[k - 1]))
            dd = (
                w3d[k, Ji, Ii]
                + fwk
                + alph * (-m * ((pp[k] + fp[k]) - (pp[k - 1] + fp[k - 1])))
            )
            if k == 2:
                cc = -asq * (-m * (pk2[k, Ji, Ii] - qk[k, Ji, Ii]))
                r1 = 1.0 / bb
                cof1[k] = -cc * r1
                cof2[k] = dd * r1
            else:
                aa = -asq * (+m * (pk1[k - 1, Ji, Ii] + qk[k - 1, Ji, Ii]))
                r1 = 1.0 / (aa * cof1[k - 1] + bb)
                if k < nk:
                    cc = -asq * (-m * (pk2[k, Ji, Ii] - qk[k, Ji, Ii]))
                    cof1[k] = -cc * r1
                    cof2[k] = (dd - aa * cof2[k - 1]) * r1
                else:
                    wnk = (dd - aa * cof2[k - 1]) * r1
        wcol = [None] * (nk + 1)
        wcol[nk] = wnk
        for k in range(nk - 1, 1, -1):
            wcol[k] = cof1[k] * wcol[k + 1] + cof2[k]
        w3d = bk.set(
            w3d, (slice(2, nk + 1), Ji, Ii), xp.stack(wcol[2 : nk + 1], axis=0)
        )
        dum1 = bk.set(
            dum1,
            (slice(2, nk), Ji, Ii),
            xp.stack([cof1[k] for k in range(2, nk)], axis=0),
        )
        dum2 = bk.set(
            dum2,
            (slice(2, nk), Ji, Ii),
            xp.stack([cof2[k] for k in range(2, nk)], axis=0),
        )

        # --- new pp, divergence damping
        w_lo, w_hi = w3d[K, Ji, Ii], w3d[2 : nk + 2, Ji, Ii]
        ppold = pp3d[K, Ji, Ii]
        ppnew = ppold + fpk_i + alph * (w_lo * (pk1_i + qk_i) + w_hi * (pk2_i - qk_i))
        ppnew = xp.where(xp.abs(ppnew) < smeps, zero, ppnew)
        d1 = kdiv * (ppnew - ppold)
        pp3d = bk.set(pp3d, (K, Ji, Ii), ppnew)
        dum1 = bk.set(dum1, (K, Ji, Ii), d1)
        ppd = bk.set(ppd, (K, Ji, Ii), ppnew + d1)
        if n < nloop:
            ppd = bcp(bk, g, cfg, ppd)

        # --- time-averaged velocities
        R = (K, Jp, Ip)
        if n < nloop:
            rru = bk.set(rru, R, rru[R] + u3d[R])
            rrv = bk.set(rrv, R, rrv[R] + v3d[R])
            rrw = bk.set(rrw, R, rrw[R] + w3d[R])
        else:
            tavg = 1.0 / float(nloop)
            rru = bk.set(rru, R, (rru[R] + u3d[R]) * tavg)
            rrv = bk.set(rrv, R, (rrv[R] + v3d[R]) * tavg)
            rrw = bk.set(rrw, R, (rrw[R] + w3d[R] * alph) * tavg)

    if nrk == nrkmax:
        ppx = bk.set(ppx, (K, Ji, Ii), dum1[K, Ji, Ii])

    return dict(
        u3d=u3d,
        v3d=v3d,
        w3d=w3d,
        pp3d=pp3d,
        ppd=ppd,
        ppx=ppx,
        rru=rru,
        rrv=rrv,
        rrw=rrw,
        dtu=dtu,
        dtv=dtv,
        fpk=fpk,
        qk=qk,
        pk1=pk1,
        pk2=pk2,
        mm=mm,
        dum1=dum1,
        dum2=dum2,
    )
