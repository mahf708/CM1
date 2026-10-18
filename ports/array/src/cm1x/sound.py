"""Acoustic (small-time-step) solver for psolver=3, ported from src/sound.F.

Split-explicit, vertically implicit sound-wave integration (Klemp-Wilhelmson
style) as used by CM1's RK3 large step. Only the code path of the reference
cases are supported: no terrain, Cartesian (axisymm=0), any non-MPI lateral
boundary type (wbc/ebc/sbc/nbc = 1 periodic, 2 open -- with irbc=4 the
radiative phase speeds radbc* come in from solve2's radbcew4/radbcns4 -- or
3/4 walls), no convergence forcing / updraft nudging / immersed boundaries. Rayleigh damping of w is *not* applied inside ``sound``
(CM1 adds it to ``wten`` before the call), so ``irdamp`` is irrelevant here.

Index conventions follow cm1x.grid: arrays are ``a[k, j, i]`` with halos, and a
Fortran index ``i`` is Python ``i + 2`` (``k`` unchanged).
"""

from __future__ import annotations

import math

from . import bc as _bc
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
    if _opt(cfg, "axisymm") != 0:
        raise NotImplementedError("sound: axisymm != 0 not ported")
    bcs = [int(_opt(cfg, b)) for b in ("wbc", "ebc", "sbc", "nbc")]
    for b, v in zip(("wbc", "ebc", "sbc", "nbc"), bcs):
        if v not in (1, 2, 3, 4):
            raise NotImplementedError(f"sound: {b}={v} not supported")
    if int(_opt(cfg, "irbc")) == 2 and any(v != 1 for v in bcs):
        raise NotImplementedError("sound: irbc=2 (radbcew/radbcns) not ported")
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
    """bc.F ``bcp``; kept here for backward compatibility, see cm1x.bc.bcp."""
    return _bc.bcp(bk, g, p, _BcCfg(cfg))


class _BcCfg:
    """Read lateral-BC options through sound's defaults (Config returns 0 for
    unset names)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def get(self, name, default=None):
        if name in _DEFAULTS:
            return _opt(self.cfg, name)
        return default


def _make_terrain_uv_fpk(
    bk,
    g,
    cfg,
    tm,
    *,
    dts,
    beta,
    smeps,
    uh,
    uf,
    vh,
    vf,
    rr0,
    uten,
    vten,
    ppten,
    ppterm,
    thv,
    piadv,
    qk,
    pk1,
    pk2,
    dum1,
    dum2,
):
    """Terrain branch of one acoustic step of sound.F, from the u/v update to
    the pressure forcing fpk (incl. bcwsfc and the advection of pi).  Returns
    ``f(u3d, v3d, w3d, ppd, pp3d, fpk) -> (u3d, v3d, w3d, dum1, dum2, fpk_i,
    fpk)`` (loop invariants precomputed)."""
    xp = bk.xp
    S = bk.asarray
    ni, nj, nk = g.ni, g.nj, g.nk
    rdx, rdy = 1.0 / g.dx, 1.0 / g.dy
    ibw, ibe, ibs, ibn = _bc.ib_flags(_BcCfg(cfg))
    rds, sigma, rdsf, sigmaf = tm["rds"], tm["sigma"], tm["rdsf"], tm["sigmaf"]
    gz, rgzu, rgzv = tm["gz"], tm["rgzu"], tm["rgzv"]
    dzdx, dzdy, gxu, gyv = tm["dzdx"], tm["dzdy"], tm["gxu"], tm["gyv"]
    rho0, rrf0 = tm["rho0"], tm["rrf0"]
    zts, rzt = S(float(tm["zt"])), S(1.0 / float(tm["zt"]))

    def fs(a, b):  # Fortran inclusive range -> python slice (i, j)
        return slice(a + 2, b + 3)

    K = slice(1, nk + 1)
    Ii, Ji = fs(1, ni), fs(1, nj)
    I0, J0 = fs(0, ni + 1), fs(0, nj + 1)
    Iu = fs(1 + ibw, ni + 1 - ibe)  # u points updated
    Jv = fs(1 + ibs, nj + 1 - ibn)  # v points updated
    Iu1, Jv1 = fs(1, ni + 1), fs(1, nj + 1)

    def m1(sl):  # shift a python slice by -1
        return slice(sl.start - 1, sl.stop - 1)

    def p1(sl):
        return slice(sl.start + 1, sl.stop + 1)

    def r3(a1):
        return xp.reshape(a1, (-1, 1, 1))

    tem = C.cp * 0.5
    Kd = slice(2, nk + 1)  # k = 2..nk
    rdsk = r3(rds[Kd])
    rdsfk = r3(rdsf[K])
    # r2, r1 of the sigma-dot terms (k = 2..nk), not halved here
    r2 = r3((sigmaf[Kd] - sigma[1:nk]) * rds[Kd])
    r1 = S(1.0) - r2
    sfz = r3(sigmaf[Kd]) - zts
    # u update (j = 1..nj, i = Iu), v update (j = Jv, i = 1..ni)
    thv_u = thv[K, Ji, Iu] + thv[K, Ji, m1(Iu)]
    thv_v = thv[K, Jv, Ii] + thv[K, m1(Jv), Ii]
    gxu_s = gxu[K, Ji, Iu] + gxu[2 : nk + 2, Ji, Iu]
    gyv_s = gyv[K, Jv, Ii] + gyv[2 : nk + 2, Jv, Ii]
    ufu = xp.reshape(uf[Iu], (1, 1, -1))
    vfv = xp.reshape(vf[Jv], (1, -1, 1))
    uhi = xp.reshape(uh[Ii], (1, 1, -1))
    vhj = xp.reshape(vh[Ji], (1, -1, 1))
    uten_u, vten_v = uten[K, Ji, Iu], vten[K, Jv, Ii]
    rgzu_u = rgzu[None, Ji, Iu1]
    rgzv_v = rgzv[None, Jv1, Ii]
    gzi = gz[None, Ji, Ii]
    dzdxi, dzdyi = dzdx[None, Ji, Ii], dzdy[None, Ji, Ii]
    rho_u = 0.5 * (rho0[K, Ji, m1(Iu1)] + rho0[K, Ji, Iu1])
    rho_v = 0.5 * (rho0[K, m1(Jv1), Ii] + rho0[K, Jv1, Ii])
    rrf0w = rrf0[Kd, Ji, Ii]
    ppten_i, ppterm_i, qk_i = ppten[K, Ji, Ii], ppterm[K, Ji, Ii], qk[K, Ji, Ii]
    pk1_i, pk2_i, rr0_i = pk1[K, Ji, Ii], pk2[K, Ji, Ii], rr0[K, Ji, Ii]
    pa = piadv[K, Ji, Ii]
    dpx_p = piadv[K, Ji, p1(Ii)] - pa  # piadv(i+1)-piadv(i)
    dpx_m = pa - piadv[K, Ji, m1(Ii)]
    dpy_p = piadv[K, p1(Ji), Ii] - pa
    dpy_m = pa - piadv[K, m1(Ji), Ii]
    temx = dts * 0.5 * rdx
    temy = dts * 0.5 * rdy
    zero = S(0.0)
    zk = bk.zeros((1, nj, ni))

    def sigma_dot(d1, d2):
        # 0.5*((r2*(d1(i)+d1(i+1))(k)+r1*(..)(k-1))*dzdx + (..)*dzdy)*(sigmaf-zt)*gz*rzt
        ax = d1[:, Ji, Ii] + d1[:, Ji, p1(Ii)]
        ay = d2[:, Ji, Ii] + d2[:, p1(Ji), Ii]
        sx = r2 * ax[Kd] + r1 * ax[1:nk]
        sy = r2 * ay[Kd] + r1 * ay[1:nk]
        return 0.5 * (sx * dzdxi + sy * dzdyi) * sfz * gzi * rzt

    def f(u3d, v3d, w3d, ppd, pp3d, fpk):
        # dum1 = d ppd / d sigma on w levels (j, i = 0..nj+1, 0..ni+1)
        dz = (ppd[Kd, J0, I0] - ppd[1:nk, J0, I0]) * rdsk
        zr = xp.zeros_like(dz[:1])
        dum1 = bk.copy(dum1_0)
        dum1 = bk.set(dum1, (slice(1, nk + 2), J0, I0), xp.concat([zr, dz, zr], axis=0))
        # u, v forward (uses dts, not dtu)
        dsu = (dum1[2 : nk + 2, Ji, Iu] + dum1[2 : nk + 2, Ji, m1(Iu)]) + (
            dum1[K, Ji, Iu] + dum1[K, Ji, m1(Iu)]
        )
        u3d = bk.set(
            u3d,
            (K, Ji, Iu),
            u3d[K, Ji, Iu]
            + dts
            * (
                uten_u
                - tem
                * thv_u
                * (
                    (ppd[K, Ji, Iu] - ppd[K, Ji, m1(Iu)]) * rdx * ufu
                    + 0.125 * dsu * gxu_s
                )
            ),
        )
        dsv = (dum1[2 : nk + 2, Jv, Ii] + dum1[2 : nk + 2, m1(Jv), Ii]) + (
            dum1[K, Jv, Ii] + dum1[K, m1(Jv), Ii]
        )
        v3d = bk.set(
            v3d,
            (K, Jv, Ii),
            v3d[K, Jv, Ii]
            + dts
            * (
                vten_v
                - tem
                * thv_v
                * (
                    (ppd[K, Jv, Ii] - ppd[K, m1(Jv), Ii]) * rdy * vfv
                    + 0.125 * dsv * gyv_s
                )
            ),
        )
        # terms for the horizontal divergence
        dum1 = bk.set(dum1, (K, Ji, Iu1), u3d[K, Ji, Iu1] * rgzu_u)
        dum2 = bk.copy(dum2_0)
        dum2 = bk.set(dum2, (K, Jv1, Ii), v3d[K, Jv1, Ii] * rgzv_v)
        d3 = xp.concat([zk, sigma_dot(dum1, dum2), zk], axis=0)  # k = 1..nk+1
        div = gzi * (
            (
                (dum1[K, Ji, p1(Ii)] - dum1[K, Ji, Ii]) * rdx * uhi
                + (dum2[K, p1(Ji), Ii] - dum2[K, Ji, Ii]) * rdy * vhj
            )
            + (d3[1:] - d3[:-1]) * rdsfk
        )
        div = xp.where(xp.abs(div) < smeps, zero, div)
        w1 = w3d[K, Ji, Ii]
        w2 = w3d[2 : nk + 2, Ji, Ii]
        w1 = xp.concat([xp.zeros_like(w1[:1]), w1[1:]], axis=0)
        w2 = xp.concat([w2[:-1], xp.zeros_like(w2[:1])], axis=0)
        fpk_i = dts * (ppten_i - ppterm_i * div) + beta * (-qk_i * (w2 - w1))
        # advection of pi (terrain): lower bc on w first
        w3d = _bc.bcwsfc(bk, g, gz=gz, dzdx=dzdx, dzdy=dzdy, u=u3d, v=v3d, w=w3d)
        dum1 = bk.set(dum1, (K, Ji, Iu1), rho_u * dum1[K, Ji, Iu1])
        dum2 = bk.set(dum2, (K, Jv1, Ii), rho_v * dum2[K, Jv1, Ii])
        d3 = xp.concat([zk, sigma_dot(dum1, dum2) * rrf0w, zk], axis=0)
        u2 = temx * dum1[K, Ji, p1(Ii)] * uhi
        u1 = temx * dum1[K, Ji, Ii] * uhi
        v2 = temy * dum2[K, p1(Ji), Ii] * vhj
        v1 = temy * dum2[K, Ji, Ii] * vhj
        ww2 = d3[1:] + beta * w3d[2 : nk + 2, Ji, Ii]
        ww1 = d3[:-1] + beta * w3d[K, Ji, Ii]
        fpk_i = (
            fpk_i
            + ((-(u2 * dpx_p + u1 * dpx_m)) - (v2 * dpy_p + v1 * dpy_m)) * rr0_i * gzi
        ) + (ww1 * pk1_i + ww2 * pk2_i)
        fpk = bk.set(fpk, (K, Ji, Ii), fpk_i)
        return u3d, v3d, w3d, dum1, dum2, fpk_i, fpk

    dum1_0 = bk.copy(dum1)
    dum2_0 = bk.copy(dum2)
    return f


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
    radbcw=None,
    radbce=None,
    radbcs=None,
    radbcn=None,
    rho0=None,
    rrf0=None,
    rds=None,
    sigma=None,
    rdsf=None,
    sigmaf=None,
    gz=None,
    rgzu=None,
    rgzv=None,
    dzdx=None,
    dzdy=None,
    gx=None,
    gxu=None,
    gy=None,
    gyv=None,
    zt=None,
    cg=None,
    **unused,
):
    """Port of sound.F ``sound`` (psolver=3), any non-MPI lateral BCs, with
    or without terrain.

    With ``terrain_flag`` the terrain-following branch is used; it needs the
    metric arrays rho0, rrf0 (3-D), rds, sigma (nk+2), rdsf, sigmaf (nk+3),
    gz, rgzu, rgzv, dzdx, dzdy (2-D), gx, gxu, gy, gyv (3-D), the Python float
    ``zt`` and, with open boundaries, ``cg`` (cgs1..3, cgt1..3).

    Shapes: scalars ``g.s_shape`` (mh, c1, c2, rr0, rf0, ppi, pp3d, piadv,
    ppten, ppx, thv, ppterm and the scratch arrays), ``g.u_shape`` (ua, u3d,
    uten, rru), ``g.v_shape`` (va, v3d, vten, rrv), ``g.w_shape`` (mf, wa,
    w3d, wten, rrw).  ``uh``/``vh`` are 1-D over the i/j scalar points
    (length s_shape[2]/s_shape[1]); ``uf``/``vf`` over the staggered points.
    ``dtu0``/``dtv0`` are 2-D ``[j, i]`` (s_shape[1:]).  ``dt`` and ``nrk`` are
    Python numbers (they set the number of small steps).

    Optional intent(inout) work arrays default to zeros; only their ranges
    written by the Fortran are meaningful. ``radbcw``/``radbce`` (``[k, j]``)
    and ``radbcs``/``radbcn`` (``[k, i]``) are the open-boundary phase speeds
    (from cm1x.bc.radbcew4/radbcns4; zeros if omitted); with wbc=2 etc. the
    boundary-normal velocity is advanced by cm1x.bc.ssopenbc* each small step
    (dtu0/dtv0 must then be zero there, see cm1x.bc.dtu0_dtv0). Unused Fortran
    dummy arguments (xh, arh1, gz, ...) may be passed and are ignored.

    Returns a dict with u3d, v3d, w3d, pp3d, ppd, ppx, rru, rrv, rrw, dtu, dtv
    and the work arrays fpk, qk, pk1, pk2, mm, dum1, dum2.
    """
    _check_options(cfg)
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    terr = bool(_opt(cfg, "terrain_flag"))
    if terr:
        tmk = dict(
            rho0=rho0,
            rrf0=rrf0,
            rds=rds,
            sigma=sigma,
            rdsf=rdsf,
            sigmaf=sigmaf,
            gz=gz,
            rgzu=rgzu,
            rgzv=rgzv,
            dzdx=dzdx,
            dzdy=dzdy,
            gx=gx,
            gxu=gxu,
            gy=gy,
            gyv=gyv,
            zt=zt,
        )
        miss = [k for k, v in tmk.items() if v is None]
        if miss:
            raise ValueError("sound: terrain_flag=.true. needs " + ", ".join(miss))
        tm = dict(tmk)
        tm.update(cg or {})
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
    bcfg = _BcCfg(cfg)
    tmo = tm if terr else None
    wbc, ebc, sbc, nbc = (int(_opt(cfg, b)) for b in ("wbc", "ebc", "sbc", "nbc"))
    radbcw = bk.zeros((S[0], S[1])) if radbcw is None else radbcw
    radbce = bk.zeros((S[0], S[1])) if radbce is None else radbce
    radbcs = bk.zeros((S[0], S[2])) if radbcs is None else radbcs
    radbcn = bk.zeros((S[0], S[2])) if radbcn is None else radbcn
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

    pa, pam, pap = piadv[K, Ji, Ii], piadv[Km, Ji, Ii], piadv[Kp, Ji, Ii]
    if terr:
        gzi = gz[None, Ji, Ii]
        rdsfk = xp.reshape(rdsf[K], (-1, 1, 1))
        c1a, c1b = c1[K, Ji, Ii], c1[Kp, Ji, Ii]
        c2a, c2b = c2[K, Ji, Ii], c2[Kp, Ji, Ii]
        qk = bk.set(qk, (K, Ji, Ii), dts * rdsfk * ppterm[K, Ji, Ii] * gzi)
        tem2 = dts * gzi * rdsfk * rr0[K, Ji, Ii]
        pk2 = bk.set(
            pk2, (K, Ji, Ii), tem2 * rf0[Kp, Ji, Ii] * (-c2b * pap + (1.0 - c1b) * pa)
        )
        pk1 = bk.set(
            pk1, (K, Ji, Ii), tem2 * rf0[K, Ji, Ii] * (+c1a * pam + (c2a - 1.0) * pa)
        )
        Kq = slice(2, nk + 1)
        tem1 = dts * xp.reshape(rds[Kq], (-1, 1, 1)) * C.cp
        mm = bk.set(
            mm,
            (Kq, Ji, Ii),
            tem1
            * (c2[Kq, Ji, Ii] * thv[Kq, Ji, Ii] + c1[Kq, Ji, Ii] * thv[1:nk, Ji, Ii])
            * gzi,
        )
    else:
        mhk = col(mh, K)
        tem = dts * rdz * mhk
        r2 = dts * rdz * mhk * col(rr0, K) * col(rf0, Kp)
        r1 = dts * rdz * mhk * col(rr0, K) * col(rf0, K)
        tem1 = dts * rdz * C.cp * col(mf, K)
        c1a, c1b = col(c1, K), col(c1, Kp)
        c2a, c2b = col(c2, K), col(c2, Kp)
        qk = bk.set(qk, (K, Ji, Ii), tem * ppterm[K, Ji, Ii])
        pk2 = bk.set(pk2, (K, Ji, Ii), r2 * (-c2b * pap + (1.0 - c1b) * pa))
        pk1 = bk.set(pk1, (K, Ji, Ii), r1 * (+c1a * pam + (c2a - 1.0) * pa))
        mm = bk.set(
            mm, (K, Ji, Ii), tem1 * (c2a * thv[K, Ji, Ii] + c1a * thv[Km, Ji, Ii])
        )
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

    # --- loop-invariant coefficients of the vertically implicit w solve
    # (Thomas algorithm over k = 2..nk; only the right-hand side dd changes
    # between acoustic steps).  Python index k == Fortran k (ngz = 1).
    Kw = slice(2, nk + 1)  # k = 2..nk
    Kwm = slice(1, nk)  # k-1
    m_w = mm[Kw, Ji, Ii]
    bb_w = 1.0 - asq * (
        +m_w
        * (-(pk1[Kw, Ji, Ii] + qk[Kw, Ji, Ii]) + (pk2[Kwm, Ji, Ii] - qk[Kwm, Ji, Ii]))
    )
    cc_w = -asq * (-m_w * (pk2[Kw, Ji, Ii] - qk[Kw, Ji, Ii]))  # used for k < nk
    aa_w = -asq * (+m_w * (pk1[Kwm, Ji, Ii] + qk[Kwm, Ji, Ii]))  # used for k > 2
    r1_2 = 1.0 / bb_w[0]
    cof1_2 = -cc_w[0] * r1_2

    def _fwd1(c1prev, x):  # k = 3..nk: r1(k), cof1(k)
        aa, bb, cc = x
        r1 = 1.0 / (aa * c1prev + bb)
        c1 = -cc * r1
        return c1, (r1, c1)

    _, (r1_w, cof1_w) = bk.scan(_fwd1, cof1_2, (aa_w[1:], bb_w[1:], cc_w[1:]))
    r1_w = xp.concat([r1_2[None], r1_w], axis=0)  # k = 2..nk
    cof1_w = xp.concat([cof1_2[None], cof1_w[:-1]], axis=0)  # k = 2..nk-1
    wten_w = wten_i[Kw]
    w3d_rows = (slice(2, nk + 1), Ji, Ii)

    def _fwd2(c2prev, x):  # k = 3..nk: cof2(k) (k = nk: w(nk))
        dd, aa, r1 = x
        c2 = (dd - aa * c2prev) * r1
        return c2, c2

    def _bwd(wnext, x):  # k = nk-1..2: w(k) = cof1(k) w(k+1) + cof2(k)
        c1, c2 = x
        w = c1 * wnext + c2
        return w, w

    def small_step(last, c):
        u3d, v3d, w3d, pp3d, ppd, rru, rrv, rrw, dum1, dum2, fpk = c
        # --- open boundaries: boundary-normal velocities (bc.F ssopenbc*)
        if wbc == 2:
            u3d = _bc.ssopenbcw(
                bk,
                g,
                bcfg,
                uh=uh,
                radbcw=radbcw,
                u3d=u3d,
                uten=uten,
                dts=dts,
                terrain=tmo,
            )
        if ebc == 2:
            u3d = _bc.ssopenbce(
                bk,
                g,
                bcfg,
                uh=uh,
                radbce=radbce,
                u3d=u3d,
                uten=uten,
                dts=dts,
                terrain=tmo,
            )
        if sbc == 2:
            v3d = _bc.ssopenbcs(
                bk,
                g,
                bcfg,
                vh=vh,
                radbcs=radbcs,
                v3d=v3d,
                vten=vten,
                dts=dts,
                terrain=tmo,
            )
        if nbc == 2:
            v3d = _bc.ssopenbcn(
                bk,
                g,
                bcfg,
                vh=vh,
                radbcn=radbcn,
                v3d=v3d,
                vten=vten,
                dts=dts,
                terrain=tmo,
            )

        if terr:
            u3d, v3d, w3d, dum1, dum2, fpk_i, fpk = _terrain_uv_fpk(
                u3d, v3d, w3d, ppd, pp3d, fpk
            )
        else:
            u3d, v3d, fpk_i, fpk = _flat_uv_fpk(u3d, v3d, w3d, ppd, fpk)

        # --- vertically implicit solve for w (k = 2..nk), Thomas algorithm
        pp = pp3d[:, Ji, Ii]
        fp = fpk[:, Ji, Ii]
        fwk = dts * wten_w + beta * (-m_w * (pp[Kw] - pp[Kwm]))
        dd = (
            w3d[w3d_rows]
            + fwk
            + alph * (-m_w * ((pp[Kw] + fp[Kw]) - (pp[Kwm] + fp[Kwm])))
        )
        cof2_2 = dd[0] * r1_w[0]
        wnk, cof2_w = bk.scan(_fwd2, cof2_2, (dd[1:], aa_w[1:], r1_w[1:]))
        cof2_w = xp.concat([cof2_2[None], cof2_w[:-1]], axis=0)  # k=2..nk-1
        _, wcol = bk.scan(_bwd, wnk, (cof1_w, cof2_w), reverse=True)
        w3d = bk.set(w3d, w3d_rows, xp.concat([wcol, wnk[None]], axis=0))
        dum1 = bk.set(dum1, (slice(2, nk), Ji, Ii), cof1_w)
        dum2 = bk.set(dum2, (slice(2, nk), Ji, Ii), cof2_w)

        # --- new pp, divergence damping
        w_lo, w_hi = w3d[K, Ji, Ii], w3d[2 : nk + 2, Ji, Ii]
        if terr:  # w1 = 0 at k = 1, w2 = 0 at k = nk
            w_lo = xp.concat([xp.zeros_like(w_lo[:1]), w_lo[1:]], axis=0)
            w_hi = xp.concat([w_hi[:-1], xp.zeros_like(w_hi[:1])], axis=0)
        ppold = pp3d[K, Ji, Ii]
        ppnew = ppold + fpk_i + alph * (w_lo * (pk1_i + qk_i) + w_hi * (pk2_i - qk_i))
        ppnew = xp.where(xp.abs(ppnew) < smeps, zero, ppnew)
        d1 = kdiv * (ppnew - ppold)
        pp3d = bk.set(pp3d, (K, Ji, Ii), ppnew)
        dum1 = bk.set(dum1, (K, Ji, Ii), d1)
        ppd = bk.set(ppd, (K, Ji, Ii), ppnew + d1)
        if not last:
            ppd = _bc.bcp(bk, g, ppd, bcfg)

        # --- time-averaged velocities
        R = (K, Jp, Ip)
        if not last:
            rru = bk.set(rru, R, rru[R] + u3d[R])
            rrv = bk.set(rrv, R, rrv[R] + v3d[R])
            rrw = bk.set(rrw, R, rrw[R] + w3d[R])
        else:
            tavg = 1.0 / float(nloop)
            rru = bk.set(rru, R, (rru[R] + u3d[R]) * tavg)
            rrv = bk.set(rrv, R, (rrv[R] + v3d[R]) * tavg)
            rrw = bk.set(rrw, R, (rrw[R] + w3d[R] * alph) * tavg)
        return u3d, v3d, w3d, pp3d, ppd, rru, rrv, rrw, dum1, dum2, fpk

    def _flat_uv_fpk(u3d, v3d, w3d, ppd, fpk):
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
        return u3d, v3d, fpk_i, fpk

    if terr:
        _terrain_uv_fpk = _make_terrain_uv_fpk(
            bk,
            g,
            cfg,
            tm,
            dts=dts,
            beta=beta,
            smeps=smeps,
            uh=uh,
            uf=uf,
            vh=vh,
            vf=vf,
            rr0=rr0,
            uten=uten,
            vten=vten,
            ppten=ppten,
            ppterm=ppterm,
            thv=thv,
            piadv=piadv,
            qk=qk,
            pk1=pk1,
            pk2=pk2,
            dum1=dum1,
            dum2=dum2,
        )

    # acoustic steps 1..nloop-1 are identical; the last one differs (no
    # pressure halo update, time averages divided by nloop)
    carry = (u3d, v3d, w3d, pp3d, ppd, rru, rrv, rrw, dum1, dum2, fpk)
    carry = bk.fori_loop(nloop - 1, lambda n, c: small_step(False, c), carry)
    carry = small_step(True, carry)
    u3d, v3d, w3d, pp3d, ppd, rru, rrv, rrw, dum1, dum2, fpk = carry

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
