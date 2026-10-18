"""GFS-EDMF PBL scheme (ipbl=3, "GFS-EDMF as configured in HWRF"), ported
from src/module_bl_gfsedmf.F (with the constants of module_gfs_physcons):

  ``bl_gfsedmf``  the WRF-style driver (column packing, unit conversions,
                  tendencies, u*, wspd, PBL height, diffusivities)
  ``moninedmf``   the hybrid EDMF scheme: local K above the PBL, the K profile
                  with Weiguo Wang's height-dependent shape (useshape=2) and
                  RGF's wind-dependent cap over water (alpha < 0),
                  stratocumulus cloud-top mixing, implicit diffusion and the
                  TKE dissipative heating (dspheat)
  ``mfpbl``       the mass flux of the strongly unstable (convective) PBL
  ``tridi2``, ``tridin``  the tridiagonal solvers

as src/turb.F (sfc_and_turb, dogfsedmf) calls it; ``gfsedmf_cm1`` is that
glue (turb.F inputs -> BL_GFSEDMF -> the PBL tendency arrays).  The order of
operations follows the Fortran (and the bitwise C++ translation in
cm1k/physics/pbl_gfs): ``x**2`` is ``x*x``, ``real**real`` is ``pow``.

Every column is computed at once: column arrays carry the vertical index first
(``a[k-1, ...]`` holds Fortran level k) and any trailing column shape; the
Fortran's per-column ``if``s become ``xp.where`` and its per-column level
indices (kpbl, krad, ...) integer arrays read with ``take_along_axis``.

What CM1's call fixes (turb.F), and so what is ported: disheat=.true.,
gfs_alpha=-1, var_ric=1, coef_ric_l=0.16, coef_ric_s=0.25, pert_pbl=.false.
(the ran1 perturbation is commented out), ntrac=3 with cloud ice (nqi>=1) else
2, ntcw=2, xkzm_m=xkzm_h=xkzm_s=1, kinver=nk.  The other alpha / var_ric /
coef_ric_* / disheat values are accepted as keyword options.  The diagnostics
that never feed back (stau, hpbl2d, evap2d, heat2d, brcr, wscale, wscaleu,
prkpp, dusfc, dvsfc, dtsfc, dqsfc) are not computed.

alpha < 0 over water (xland=2) reads WSPM(i,2), the level that brackets 500 m
among k=1..nk/2; the Fortran leaves it undefined when no level does
(zi(nk/2+1) <= 500 m): ``bl_gfsedmf`` raises then (eagerly; under a jax trace
such columns keep the first K profile).
"""

from __future__ import annotations

from . import constants as C

# module_gfs_physcons
con_g = 9.80665e0
con_rd = 2.8705e2
con_rv = 4.6150e2
con_cp = 1.0046e3
con_hvap = 2.5000e6
con_fvirt = con_rv / con_rd - 1.0

# moninedmf parameters
grav = con_g
cp = con_cp
hvap = con_hvap
fv = con_fvirt
gravi = 1.0 / grav
g = grav
gocp = g / cp
rlam = 30.0
vk = 0.4
prmin = 0.25
prmax = 4.0
zolcr = 0.2
zolcru = -0.5
dw2min = 0.0001
dkmin = 0.0
dkmax = 1000.0
rimin = -100.0
crbcon = 0.25
crbmin = 0.15
crbmax = 0.35
wfac = 7.0
cfac = 6.5
sfcfrac = 0.1
qmin = 1.0e-8
zfmin = 1.0e-8
aphi5 = 5.0
aphi16 = 16.0
tdzmin = 1.0e-3
qlmin = 1.0e-12
f0 = 1.0e-4
h1 = 0.33333333
cldtime = 500.0
xkzminv = 0.3
gamcrt = 3.0
gamcrq = 0.0
rlamun = 150.0
rentf1 = 0.2
rentf2 = 1.0
radfac = 0.85
zstblmax = 2500.0
qlcr = 3.5e-5
actei = 0.7
# BL_GFSEDMF: background diffusivities
xkzm_m = 1.0
xkzm_h = 1.0
xkzm_s = 1.0
# mfpbl parameters
ce0 = 0.38
alp = 1.0
pgcon = 0.55
a1c = 0.08
bb1 = 1.8
bb2 = 3.5


class UndefinedWspm(RuntimeError):
    """alpha < 0 over water with no level bracketing 500 m among k=1..nk/2."""


# ---------------------------------------------------------------------------
# helpers
def _mx(xp, a, b):
    """Fortran max of two numbers (either may be a Python scalar)."""
    if isinstance(a, (int, float)):
        a, b = b, a
    if isinstance(b, (int, float)):
        return xp.clip(a, min=b)
    return xp.maximum(a, b)


def _mn(xp, a, b):
    """Fortran min of two numbers (either may be a Python scalar)."""
    if isinstance(a, (int, float)):
        a, b = b, a
    if isinstance(b, (int, float)):
        return xp.clip(a, max=b)
    return xp.minimum(a, b)


def _rdiv(xp, s, a):
    """s / a for a Python scalar s (torch computes scalar/tensor as
    s * (1/a), which is not correctly rounded)."""
    return xp.full_like(a, s) / a


def _levels(xp, n, like, start=1):
    """Fortran level numbers start..start+n-1 shaped (n, 1, ..., 1)."""
    shp = (n,) + (1,) * (like.ndim)
    return xp.reshape(xp.arange(start, start + n, dtype=xp.int64), shp)


def _at(xp, a, k):
    """a(k) per column: a is (n, *C) with a[0] Fortran level 1; k an integer
    array of shape C (clipped into 1..n)."""
    idx = xp.clip(k - 1, 0, a.shape[0] - 1)
    return xp.take_along_axis(a, idx[None, ...], axis=0)[0]


def _tracing(bk, *arrays):
    if bk.name != "jax":
        return False
    from .backend import _jax_tracing

    return _jax_tracing(arrays)


def _trisolve(bk, cl, cm, cu, rs):
    """The forward/backward sweeps of tridi2/tridin: cl(2..n) lower (n-1
    rows), cm(1..n), cu(1..n-1); rs a tuple of right-hand sides (n rows).
    Returns the solutions (same operations as the Fortran)."""
    xp = bk.xp
    n = cm.shape[0]
    fk = 1.0 / cm[0]
    au1 = fk * cu[0]
    r1 = tuple(fk * r[0] for r in rs)
    cu_pad = xp.concat([cu[1:], xp.zeros_like(cu[:1])], axis=0)

    def fwd(carry, x):
        au_p, a_p = carry
        cl_k, cm_k, cu_k = x[0], x[1], x[2]
        r_k = x[3:]
        f = 1.0 / (cm_k - cl_k * au_p)
        au_k = f * cu_k
        a_k = tuple(f * (r - cl_k * a) for r, a in zip(r_k, a_p))
        return (au_k, a_k), (au_k,) + a_k

    _, ys = bk.scan(fwd, (au1, r1), (cl, cm[1:], cu_pad) + tuple(r[1:] for r in rs))
    # au(1..n-1); a(1..n) after the forward sweep
    au = xp.concat([au1[None], ys[0][:-1]], axis=0)
    a = [xp.concat([r1[m][None], ys[1 + m]], axis=0) for m in range(len(rs))]

    def bwd(carry, x):
        au_k = x[0]
        a_k = x[1:]
        new = tuple(ak - au_k * c for ak, c in zip(a_k, carry))
        return new, new

    last = tuple(am[n - 1] for am in a)
    _, ys = bk.scan(bwd, last, (au,) + tuple(am[: n - 1] for am in a), reverse=True)
    return tuple(xp.concat([ys[m], last[m][None]], axis=0) for m in range(len(rs)))


def tridi2(bk, *, cl, cm, cu, r1, r2):
    """tridi2: two right-hand sides r1, r2 (cl: rows 2..n)."""
    a1, a2 = _trisolve(bk, cl, cm, cu, (r1, r2))
    return {"a1": a1, "a2": a2}


def tridin(bk, *, cl, cm, cu, r1, r2):
    """tridin: r1 and the ntrac right-hand sides of the list r2."""
    out = _trisolve(bk, cl, cm, cu, (r1, *r2))
    return {"a1": out[0], "a2": list(out[1:])}


# ---------------------------------------------------------------------------
def mfpbl(
    bk,
    *,
    delt,
    cnvflg,
    zl,
    zm,
    thvx,
    q1,
    t1,
    u1,
    v1,
    hpbl,
    kpbl,
    sflx,
    ustar,
    wstar,
    xmf,
    tcko,
    qcko,
    ucko,
    vcko,
):
    """mfpbl on the columns where ``cnvflg`` (others keep their inputs).
    zl (km), zm (km+1) in m; q1 a list of the ntrac tracers.  Returns hpbl,
    kpbl, xmf, tcko, qcko, ucko, vcko."""
    xp = bk.xp
    W = xp.where
    km = zl.shape[0]
    kmpbl = km // 2 + 1
    dt2 = delt
    zi = zm[1:]  # zi(k) = zm(k+1)
    one = xp.ones_like(hpbl)
    wst = W(cnvflg, wstar, one)
    hp0 = W(cnvflg, hpbl, one)

    def entrain(kp, hp):
        k = _mx(xp, kp // 2, 1)
        delz = _at(xp, zl, k + 1) - _at(xp, zl, k)
        xlamax = _rdiv(xp, ce0, delz)
        kk = _levels(xp, kmpbl, hp)
        ptem = 1.0 / (zi[:kmpbl] + delz)
        tem = _mx(xp, (hp - zi[:kmpbl] + delz), delz)
        ptem1 = 1.0 / tem
        return W(kk < kp, ce0 * (ptem + ptem1), xlamax)

    xlamue = entrain(kpbl, hp0)
    # thermal excess
    tem = zl[0] / hp0
    usws3 = xp.pow(ustar / wst, 3.0)
    tem1 = usws3 + 0.6 * tem
    tem2 = _mx(xp, (1.0 - tem), zfmin)
    ptem = xp.pow(tem1, h1) * xp.sqrt(tem2)
    sigw1 = 1.3 * ptem * wst
    ptem1 = alp * sflx / sigw1
    thvu1 = thvx[0] + ptem1
    buo1 = g * (thvu1 / thvx[0] - 1.0)

    def thv_step(thvu_p, x):
        xl, zl_k, zl_p, thvx_k, thvx_p = x
        dz = zl_k - zl_p
        tem = xl * dz
        ptem = 2.0 + tem
        ptem1 = (2.0 - tem) / ptem
        tem1 = tem * (thvx_k + thvx_p) / ptem
        thvu = ptem1 * thvu_p + tem1
        return thvu, g * (thvu / thvx_k - 1.0)

    _, buo_r = bk.scan(
        thv_step,
        thvu1,
        (
            xlamue[: kmpbl - 1],
            zl[1:kmpbl],
            zl[: kmpbl - 1],
            thvx[1:kmpbl],
            thvx[: kmpbl - 1],
        ),
    )
    buo = xp.concat([buo1[None], buo_r], axis=0)
    # updraft velocity squared
    dz = zi[0]
    tem = 0.5 * bb1 * xlamue[0] * dz
    tem1 = bb2 * buo[0] * dz
    wu21 = tem1 / (1.0 + tem)

    def wu2_step(wp, x):
        xl_k, xl_p, zi_k, zi_p, buo_k = x
        dz = zi_k - zi_p
        tem = 0.25 * bb1 * (xl_k + xl_p) * dz
        tem1 = bb2 * buo_k * dz
        ptem = (1.0 - tem) * wp
        w = (ptem + tem1) / (1.0 + tem)
        return w, w

    _, wu2_r = bk.scan(
        wu2_step,
        wu21,
        (xlamue[1:], xlamue[:-1], zi[1:kmpbl], zi[: kmpbl - 1], buo[1:]),
    )
    wu2 = xp.concat([wu21[None], wu2_r], axis=0)
    # PBL height where the updraft velocity vanishes
    flg = ~cnvflg
    rbup = wu2[0]
    rbdn = xp.zeros_like(rbup)
    kp = kpbl
    for k in range(2, kmpbl + 1):
        nf = ~flg
        rbdn = W(nf, rbup, rbdn)
        rbup = W(nf, wu2[k - 1], rbup)
        kp = W(nf, k, kp)
        flg = W(nf, rbup <= 0.0, flg)
    den = W(rbdn - rbup == 0.0, one, rbdn - rbup)
    rbint = W(rbdn <= 0.0, 0.0, W(rbup >= 0.0, 1.0, rbdn / den))
    zik = _at(xp, zm, kp)
    hp = zik + rbint * (_at(xp, zm, kp + 1) - zik)
    hp = W(cnvflg, hp, hp0)
    xlamue = entrain(kp, hp)
    # updraft mass flux
    kk = _levels(xp, kmpbl, hp)
    msk = cnvflg & (kk < kp)
    w2 = W(msk, wu2, 1.0)
    xm = a1c * xp.sqrt(w2)
    xmmx = (zl[1 : kmpbl + 1] - zl[:kmpbl]) / dt2
    xm = _mn(xp, xm, xmmx)
    xmf = xp.concat([W(msk, xm, xmf[:kmpbl]), xmf[kmpbl:]], axis=0)

    # updraft properties
    def props(ck, x):
        tc_p, uc_p, vc_p, qc_p = ck
        on, xl, zl_k, zl_p, t_k, t_p, u_k, u_p, v_k, v_p, tc0, uc0, vc0 = x[:13]
        qk = x[13:]
        nq = len(qk) // 3
        q_k, q_p, qc0 = qk[:nq], qk[nq : 2 * nq], qk[2 * nq :]
        dz = zl_k - zl_p
        tem = 0.5 * xl * dz
        factor = 1.0 + tem
        ptem = tem + pgcon
        ptem1 = tem - pgcon
        tc = ((1.0 - tem) * tc_p + tem * (t_k + t_p) - gocp * dz) / factor
        uc = ((1.0 - tem) * uc_p + ptem * u_k + ptem1 * u_p) / factor
        vc = ((1.0 - tem) * vc_p + ptem * v_k + ptem1 * v_p) / factor
        qc = tuple(
            ((1.0 - tem) * a + tem * (b + c)) / factor
            for a, b, c in zip(qc_p, q_k, q_p)
        )
        tc = W(on, tc, tc0)
        uc = W(on, uc, uc0)
        vc = W(on, vc, vc0)
        qc = tuple(W(on, a, b) for a, b in zip(qc, qc0))
        return (tc, uc, vc, qc), (tc, uc, vc) + qc

    kk2 = _levels(xp, kmpbl - 1, hp, start=2)
    on = cnvflg & (kk2 <= kp)
    s = slice(1, kmpbl)
    sp = slice(0, kmpbl - 1)
    xs = (
        on,
        xlamue[: kmpbl - 1],
        zl[s],
        zl[sp],
        t1[s],
        t1[sp],
        u1[s],
        u1[sp],
        v1[s],
        v1[sp],
        tcko[s],
        ucko[s],
        vcko[s],
        *(q[s] for q in q1),
        *(q[sp] for q in q1),
        *(q[s] for q in qcko),
    )
    _, ys = bk.scan(props, (tcko[0], ucko[0], vcko[0], tuple(q[0] for q in qcko)), xs)

    def merge(a, new):
        return xp.concat([a[:1], new, a[kmpbl:]], axis=0)

    nq = len(q1)
    return {
        "hpbl": W(cnvflg, hp, hpbl),
        "kpbl": W(cnvflg, kp, kpbl),
        "xmf": xmf,
        "tcko": merge(tcko, ys[0]),
        "ucko": merge(ucko, ys[1]),
        "vcko": merge(vcko, ys[2]),
        "qcko": [merge(qcko[n], ys[3 + n]) for n in range(nq)],
    }


# ---------------------------------------------------------------------------
def moninedmf(
    bk,
    *,
    u1,
    v1,
    t1,
    q1,
    thraten,
    psk,
    rbsoil,
    zorl,
    u10m,
    v10m,
    fm,
    fh,
    tsea,
    heat,
    evap,
    stress,
    spd1,
    prsi,
    dl,
    prsl,
    prslk,
    phii,
    phil,
    delt,
    xland1,
    dspheat=True,
    var_ric=1.0,
    coef_ric_l=0.16,
    coef_ric_s=0.25,
    alpha=-1.0,
):
    """moninedmf (kinver = km, xkzm_* = 1, ntcw = 2).  Column arrays are
    (km, *C) (prsi, phii: km+1); q1 is the list of the ntrac tracers (qv, qc
    [, qi]); dl is the Fortran's del.  Returns dv, du, tau, dishx, diss
    (levels 0..km-1 of diss(0:km)), rtg (list), kpbl, hpbl, dkt, dku
    (km-1 levels), the flag ``bad`` (alpha < 0 over water without the
    500-m level) and, as diagnostics, the regime flags pcnvflg, ublflg,
    scuflg and wspm_adj (the 500-m wind cap applied)."""
    xp = bk.xp
    W = xp.where
    km = t1.shape[0]
    km1 = km - 1
    kmpbl = km // 2
    ntrac = len(q1)
    dt2 = delt
    rdt = 1.0 / dt2
    like = psk
    zero = xp.zeros_like(like)
    one = xp.ones_like(like)
    ione = xp.ones(like.shape, dtype=xp.int64)

    zi = phii * gravi
    zl = phil * gravi
    rdzt = 1.0 / (zl[1:] - zl[:-1])

    # background diffusivities (k < kinver = km for every k <= km-1)
    kx1 = ione
    tx1 = 1.0 / prsi[0]
    tx2 = tx1
    xkzo_l, xkzmo_l = [], []
    for k in range(1, km1 + 1):
        ptem = prsi[k] * tx1
        tem1 = 1.0 - ptem
        tem1 = tem1 * tem1 * 10.0
        xkzo_l.append(xkzm_h * _mn(xp, 1.0, xp.exp(-tem1)))
        above = ptem >= xkzm_s
        if k > 1:
            tx2 = W((~above) & (kx1 == k), 1.0 / prsi[k - 1], tx2)
        tem1 = 1.0 - prsi[k] * tx2
        tem1 = tem1 * tem1 * 5.0
        xkzmo_l.append(W(above, xkzm_m, xkzm_m * _mn(xp, 1.0, xp.exp(-tem1))))
        kx1 = W(above, k + 1, kx1)
    xkzo = xp.stack(xkzo_l, axis=0)
    xkzmo = xp.stack(xkzmo_l, axis=0)
    # diffusivity in the inversion layer
    tem1 = (t1[1 : kmpbl + 1] - t1[:kmpbl]) * rdzt[:kmpbl]
    inv = (zi[1 : kmpbl + 1] > 250.0) & (tem1 > 1.0e-5)
    xkzo = xp.concat(
        [W(inv, _mn(xp, xkzo[:kmpbl], xkzminv), xkzo[:kmpbl]), xkzo[kmpbl:]], axis=0
    )

    z0 = 0.01 * zorl
    sfcflg = ~(rbsoil > 0.0)
    radmin = zero
    hrad = zi[0]
    zd = zero
    lcld = ione * km1
    kcld = ione * km1
    krad = ione
    icld = 0 * ione

    theta = t1 * psk / prslk
    qv = q1[0]
    qlx = _mx(xp, q1[1], qlmin)
    qtx = _mx(xp, qv, qmin) + qlx
    ptem1 = hvap * _mx(xp, qv, qmin) / (cp * t1)
    thetae = theta * (1.0 + ptem1)
    thvx = theta * (1.0 + fv * _mx(xp, qv, qmin) - qlx)
    ptem2 = theta - (hvap / cp) * qlx
    thlvx = ptem2 * (1.0 + fv * qtx)
    radx = (zi[1:km] - zi[:km1]) * thraten[:km1]
    scuflg = xp.ones(like.shape, dtype=xp.bool)
    flg = scuflg
    for k in range(1, km1 + 1):
        hit = flg & (zl[k - 1] >= zstblmax)
        lcld = W(hit, k, lcld)
        flg = flg & ~hit
    # virtual potential temperature gradient and wind shear squared
    rdz = rdzt
    bf = (thvx[1:] - thvx[:-1]) * rdz
    ti = 2.0 / (t1[:-1] + t1[1:])
    dud = u1[:-1] - u1[1:]
    dvd = v1[:-1] - v1[1:]
    dw2 = dud * dud + dvd * dvd
    shr2 = _mx(xp, dw2, dw2min) * rdz * rdz

    govrth = _rdiv(xp, g, theta[0])
    beta = _rdiv(xp, dt2, zi[1] - zi[0])
    ustar = xp.sqrt(stress)
    sflux = heat + evap * fv * theta[0]
    pblflg = sfcflg & (sflux > 0.0)

    # PBL height (variable critical Ri)
    thermal = W(pblflg, thvx[0], tsea * (1.0 + fv * _mx(xp, qv[0], qmin)))
    tem = xp.sqrt(u10m * u10m + v10m * v10m)
    tem = _mx(xp, tem, 1.0)
    robn = tem / (f0 * z0)
    tem1 = 1.0e-7 * robn
    crb = crbcon * one
    if var_ric == 1.0:
        crb = W(xland1 == 1, coef_ric_l * xp.pow(tem1, -0.18), crb)
        crb = W(xland1 == 2, coef_ric_s * xp.pow(tem1, -0.18), crb)
    crb = _mx(xp, _mn(xp, crb, crbmax), crbmin)

    spdk2 = _mx(xp, (u1 * u1 + v1 * v1), 1.0)
    rbk = (thvx[:kmpbl] - thermal) * (g * zl[:kmpbl] / thvx[0]) / spdk2[:kmpbl]
    flg = xp.zeros(like.shape, dtype=xp.bool)
    rbup = rbsoil
    rbdn = zero
    kpbl = ione
    for k in range(1, kmpbl + 1):
        nf = ~flg
        rbdn = W(nf, rbup, rbdn)
        rbup = W(nf, rbk[k - 1], rbup)
        kpbl = W(nf, k, kpbl)
        flg = W(nf, rbup > crb, flg)

    def interp(kp, rbdn, rbup):
        den = W(rbup - rbdn == 0.0, one, rbup - rbdn)
        rbint = W(rbdn >= crb, 0.0, W(rbup <= crb, 1.0, (crb - rbdn) / den))
        zlm = _at(xp, zl, kp - 1)
        return zlm + rbint * (_at(xp, zl, kp) - zlm)

    big = kpbl > 1
    hpbl = interp(kpbl, rbdn, rbup)
    kpbl = W(big & (hpbl < _at(xp, zi, kpbl)), kpbl - 1, kpbl)
    hpbl = W(big, hpbl, zl[0])
    kpbl = W(big, kpbl, 1)
    kpblx = kpbl
    hpblx = hpbl

    # similarity parameters
    zol = _mx(xp, rbsoil * fm * fm / fh, rimin)
    zol = W(sfcflg, _mn(xp, zol, -zfmin), _mx(xp, zol, zfmin))
    zol1 = zol * sfcfrac * hpbl / zl[0]
    tem = 1.0 / (1.0 - aphi16 * W(sfcflg, zol1, zero))
    phih_u = xp.sqrt(tem)
    phim_u = xp.sqrt(phih_u)
    phim_s = 1.0 + aphi5 * zol1
    phim = W(sfcflg, phim_u, phim_s)
    phih = W(sfcflg, phih_u, phim_s)
    wscale = ustar / phim
    ustmin = ustar / aphi5
    wscale = _mx(xp, wscale, ustmin)
    pcnvflg = pblflg & (zol < zolcru) & (kpbl > 1)
    ublflg = pblflg & ~pcnvflg
    wst3 = govrth * sflux * hpbl
    wst3s = W(pblflg, wst3, one)
    wstar = W(pblflg, xp.pow(wst3s, h1), zero)
    ust3 = xp.pow(ustar, 3.0)
    wsu = ust3 + wfac * vk * W(pblflg, wst3, zero) * sfcfrac
    wscaleu = W(pblflg, _mx(xp, xp.pow(W(pblflg, wsu, one), h1), ustmin), zero)
    # counter-gradient terms
    wsu_s = W(ublflg, wscaleu, one)
    hgamt = _mn(xp, cfac * heat / wsu_s, gamcrt)
    hgamq = _mn(xp, cfac * evap / wsu_s, gamcrq)
    vpert = hgamt + hgamq * fv * theta[0]
    vpert = _mn(xp, vpert, gamcrt)
    thermal = W(ublflg, thermal + _mx(xp, vpert, 0.0), thermal)
    hgamt = W(ublflg, _mx(xp, hgamt, 0.0), zero)
    hgamq = W(ublflg, _mx(xp, hgamq, 0.0), zero)

    # PBL height with the thermal excess
    rbk = (thvx[:kmpbl] - thermal) * (g * zl[:kmpbl] / thvx[0]) / spdk2[:kmpbl]
    flg = ~ublflg
    rbup = W(ublflg, rbsoil, rbup)
    for k in range(2, kmpbl + 1):
        nf = ~flg
        rbdn = W(nf, rbup, rbdn)
        rbup = W(nf, rbk[k - 1], rbup)
        kpbl = W(nf, k, kpbl)
        flg = W(nf, rbup > crb, flg)
    hp_u = interp(kpbl, rbdn, rbup)
    kp_u = W(hp_u < _at(xp, zi, kpbl), kpbl - 1, kpbl)
    hpbl = W(ublflg, hp_u, hpbl)
    kpbl = W(ublflg, kp_u, kpbl)
    off = ublflg & (kpbl <= 1)
    ublflg = ublflg & ~off
    pblflg = pblflg & ~off

    # stratocumulus
    flg = scuflg
    for k in range(kmpbl, 0, -1):
        hit = flg & (k <= lcld) & (qlx[k - 1] >= qlcr)
        kcld = W(hit, k, kcld)
        flg = flg & ~hit
    scuflg = scuflg & ~(kcld == km1)
    flg = scuflg
    for k in range(kmpbl, 0, -1):
        on = flg & (k <= kcld)
        cl = qlx[k - 1] >= qlcr
        lower = on & cl & (radx[k - 1] < radmin)
        radmin = W(lower, radx[k - 1], radmin)
        krad = W(lower, k, krad)
        flg = flg & ~(on & ~cl)
    scuflg = scuflg & ~(krad <= 1)
    scuflg = scuflg & ~(radmin >= 0.0)
    flg = scuflg
    for k in range(kmpbl, 1, -1):
        on = flg & (k <= krad)
        cl = qlx[k - 1] >= qlcr
        icld = W(on & cl, icld + 1, icld)
        flg = flg & ~(on & ~cl)
    scuflg = scuflg & ~(icld < 1)
    hrad = W(scuflg, _at(xp, zi, krad + 1), hrad)
    scuflg = scuflg & ~(hrad < zi[1])
    dzk = zi[1:] - zi[:-1]  # zi(k+1) - zi(k), k = 1..km
    tem = _at(xp, dzk, krad)
    thlvx1 = W(scuflg, _at(xp, thlvx, krad) + cldtime * radmin / tem, zero)
    flg = scuflg
    for k in range(kmpbl, 0, -1):
        on = flg & (k <= krad)
        ok = thlvx1 <= thlvx[k - 1]
        zd = W(on & ok, zd + dzk[k - 1], zd)
        flg = flg & ~(on & ~ok)
    kk = _mx(xp, krad + 1 - icld, 1)
    zdd = W(scuflg, hrad - _at(xp, zi, kk), zero)
    zd = W(scuflg, _mn(xp, _mx(xp, zd, zdd), hrad), zd)
    tem = govrth * zd * (-radmin)
    vrad = W(scuflg, xp.pow(W(scuflg, tem, one), h1), zero)

    # inverse Prandtl number
    tem = W(ublflg, phih / phim + cfac * vk * sfcfrac, phih / phim)
    prinv = 1.0 / tem
    prinv = _mn(xp, prinv, prmax)
    prinv = _mx(xp, prinv, prmin)
    kpbl = W(zol > zolcr, 1, kpbl)

    # RGF: wind speed near 500 m (alpha < 0)
    wspm1 = zero
    wspm2 = 0 * ione
    wset = xp.zeros(like.shape, dtype=xp.bool)
    if alpha < 0:
        for k in range(1, kmpbl + 1):
            hit = (zi[k - 1] <= 500) & (zi[k] > 500.0)
            spd = xp.sqrt(u1[k - 1] * u1[k - 1] + v1[k - 1] * v1[k - 1])
            wspm1 = W(hit, spd / 0.6, wspm1)
            wspm2 = W(hit, k, wspm2)
            wset = wset | hit

    # diffusion coefficients below the PBL (useshape = 2)
    kl = _levels(xp, kmpbl, like)
    zik1 = zi[1 : kmpbl + 1]
    hmax = 0.333
    omh = 1.0 - hmax
    skmax = hmax * (omh * omh)
    sksfc = _mn(xp, zi[1] / hpbl, 0.05)
    omk = 1 - sksfc
    sksfc = sksfc * (omk * omk)
    dhp = hpbl - zl[0]
    sz2h = (zik1 - zl[0]) / W(dhp == 0.0, one, dhp)
    sz2h = _mx(xp, sz2h, zfmin)
    sz2h = _mn(xp, sz2h, 1.0)
    oms = 1.0 - sz2h
    zfac = oms * oms
    zfac = _mx(xp, zfac, zfmin)
    skminusk0 = zik1 * zfac - hpbl * sksfc
    wsc = W(pblflg, wscaleu, wscale)

    def below_pbl(afac, dku, dkt):
        ashape = _mx(xp, afac, 0.2)
        ashape1 = W(skmax > sksfc, (skmax * ashape - sksfc) / (skmax - sksfc), 1.0)
        tem = W(skminusk0 > 0, skminusk0 * ashape1 + hpbl * sksfc, zik1 * (zfac))
        tem1 = vk * wsc * tem
        du_ = _mx(xp, _mn(xp, tem1, dkmax), xkzmo[:kmpbl])
        dt_ = _mx(xp, _mn(xp, tem1 * prinv, dkmax), xkzo[:kmpbl])
        on = kl < kpbl
        return W(on, du_, dku), W(on, dt_, dkt)

    zk3 = xp.zeros_like(rdzt[:kmpbl])
    dku_b, dkt_b = below_pbl(abs(alpha) * one, zk3, zk3)
    bad = xp.zeros(like.shape, dtype=xp.bool)
    if alpha < 0:
        water = xland1 == 2
        bad = water & ~wset
        kloc = _mx(xp, wspm2, 1)
        xdku = _at(xp, dku_b, kloc)
        adj = water & wset & (kpbl > kloc) & (xdku >= wspm1)
        wspm3 = wspm1 / W(adj, xdku, one)
        wspm4 = _mn(xp, wspm3, 1.0)
        du2, dt2_ = below_pbl(wspm4, dku_b, dkt_b)
        dku_b = W(adj, du2, dku_b)
        dkt_b = W(adj, dt2_, dkt_b)
    dktx = xp.concat([dkt_b, xp.zeros_like(rdzt[kmpbl:])], axis=0)
    dktx = W(_levels(xp, km1, like) < kpbl, dktx, 0.0)

    # local diffusion above the PBL
    k1 = _levels(xp, km1, like)
    bvf2 = g * bf * ti
    ri = _mx(xp, bvf2 / shr2, rimin)
    zk = vk * zi[1:km]
    neg = ri < 0.0
    rl2 = W(neg, zk * rlamun / (rlamun + zk), zk * rlam / (rlam + zk))
    dk = rl2 * rl2 * xp.sqrt(shr2)
    sri = xp.sqrt(W(neg, -ri, 0.0))
    du_n = dk * (1 + 8.0 * (-ri) / (1 + 1.746 * sri))
    dt_n = dk * (1 + 8.0 * (-ri) / (1 + 1.286 * sri))
    t5 = 1 + 5.0 * ri
    tem1 = dk / (t5 * t5)
    prnum = W(k1 >= kpblx, _mn(xp, 1.0 + 2.1 * ri, prmax), 1.0)
    du_s = tem1 * prnum
    du_l = _mx(xp, _mn(xp, W(neg, du_n, du_s), dkmax), xkzmo)
    dt_l = _mx(xp, _mn(xp, W(neg, dt_n, tem1), dkmax), xkzo)
    dku = xp.concat([dku_b, xp.zeros_like(rdzt[kmpbl:])], axis=0)
    dkt = xp.concat([dkt_b, xp.zeros_like(rdzt[kmpbl:])], axis=0)
    loc = k1 >= kpbl
    dku = W(loc, du_l, dku)
    dkt = W(loc, dt_l, dkt)

    # mass flux of the large thermals
    zkm = xp.zeros_like(t1)
    mf = mfpbl(
        bk,
        delt=dt2,
        cnvflg=pcnvflg,
        zl=zl,
        zm=zi,
        thvx=thvx,
        q1=q1,
        t1=t1,
        u1=u1,
        v1=v1,
        hpbl=hpbl,
        kpbl=kpbl,
        sflx=sflux,
        ustar=ustar,
        wstar=wstar,
        xmf=zkm,
        tcko=t1,
        qcko=list(q1),
        ucko=u1,
        vcko=v1,
    )
    hpbl, kpbl, xmf = mf["hpbl"], mf["kpbl"], mf["xmf"]
    tcko, ucko, vcko, qcko = mf["tcko"], mf["ucko"], mf["vcko"], mf["qcko"]

    # cloud-top driven diffusion
    tem = _at(xp, thetae[:km1], krad) - _at(xp, thetae[1:], krad)
    tem1 = _at(xp, qtx[:km1], krad) - _at(xp, qtx[1:], krad)
    pos = scuflg & (tem > 0.0) & (tem1 > 0.0)
    cteit = cp * tem / (hvap * W(pos, tem1, one))
    rent = W(pos & (cteit > actei), rentf2 * one, rentf1 * one)
    tem1 = _mx(xp, _at(xp, bf, krad), tdzmin)
    ckt_r = -rent * radmin / tem1
    kp = _levels(xp, kmpbl, like)
    tem1 = hrad - zd
    tem2 = zik1 - tem1
    zds = W(zd == 0.0, one, zd)
    ptem = tem2 / zds
    ptem = W(ptem >= 1.0, 1.0, ptem)
    ptem = tem2 * ptem * xp.sqrt(W(tem2 > 0.0, 1.0 - ptem, 0.0))
    ckt = radfac * vk * vrad * ptem
    cku = 0.75 * ckt
    ckt = _mn(xp, _mx(xp, ckt, dkmin), dkmax)
    cku = _mn(xp, _mx(xp, cku, dkmin), dkmax)
    below = scuflg & (kp < krad) & (tem2 > 0.0)
    ckt = W(below, ckt, 0.0)
    cku = W(below, cku, 0.0)
    atr = scuflg & (kp == krad)
    ckt = W(atr, ckt_r, ckt)
    cku = W(atr, ckt_r, cku)
    dkt_p, dku_p = dkt[:kmpbl], dku[:kmpbl]
    if alpha == 1.0:
        dkt_p = W(scuflg, dkt_p + ckt, dkt_p)
        dku_p = W(scuflg, dku_p + cku, dku_p)
    dkt_p = W(scuflg, _mn(xp, dkt_p, dkmax), dkt_p)
    dku_p = W(scuflg, _mn(xp, dku_p, dkmax), dku_p)
    dkt = xp.concat([dkt_p, dkt[kmpbl:]], axis=0)
    dku = xp.concat([dku_p, dku[kmpbl:]], axis=0)

    # tridiagonal matrix for heat and moisture
    kk = _levels(xp, km1, like)
    dtodsd = _rdiv(xp, dt2, dl[:km1])
    dtodsu = _rdiv(xp, dt2, dl[1:])
    dsig = prsl[:km1] - prsl[1:]
    tem1 = dsig * dkt * rdz
    dsdz2 = tem1 * rdz
    au0 = -dtodsd * dsdz2
    al0 = -dtodsu * dsdz2
    pc = pcnvflg & (kk < kpbl)
    ub = (~pc) & ublflg & (kk < kpbl)
    tem2 = dsig * rdz
    ptemx = 0.5 * tem2 * xmf[:km1]
    ptem1x = dtodsd * ptemx
    ptem2x = dtodsu * ptemx
    au = W(pc, au0 - ptem1x, au0)
    al = W(pc, al0 + ptem2x, al0)
    # ad(k) = ad(k) - au(k) [- ptem1], with ad(k) from row k-1 (or 1.)
    adn = W(pc, 1.0 - al0 + ptem2x, 1.0 - al0)  # ad(k+1) before row k+1
    adp = xp.concat([one[None], adn[:-1]], axis=0)
    adk = W(pc, adp - au0 - ptem1x, adp - au0)
    ad = xp.concat([adk, adn[-1:]], axis=0)
    # a1: temperature
    tcs = tcko[:km1] + tcko[1:]
    dsdzt0 = tem1 * gocp
    ptem1u = dsig * dktx * rdz
    temh = 1.0 / W(ublflg, hpbl, one)
    dsdzt_u = tem1 * gocp - ptem1u * hgamt * temh
    dsdzq_u = -ptem1u * hgamq * temh
    dsdzt = W(ub, dsdzt_u, dsdzt0)
    a1n = W(pc, t1[1:] - dtodsu * dsdzt + ptem2x * tcs, t1[1:] - dtodsu * dsdzt)
    a1p = xp.concat([(t1[0] + beta * heat)[None], a1n[:-1]], axis=0)
    a1k = W(pc, a1p + dtodsd * dsdzt - ptem1x * tcs, a1p + dtodsd * dsdzt)
    a1 = xp.concat([a1k, a1n[-1:]], axis=0)
    # a2(1): qv
    qcs = qcko[0][:km1] + qcko[0][1:]
    a2n = W(
        pc,
        q1[0][1:] + ptem2x * qcs,
        W(ub, q1[0][1:] - dtodsu * dsdzq_u, q1[0][1:]),
    )
    a2p = xp.concat([(q1[0][0] + beta * evap)[None], a2n[:-1]], axis=0)
    a2k = W(pc, a2p - ptem1x * qcs, W(ub, a2p + dtodsd * dsdzq_u, a2p))
    a2 = [xp.concat([a2k, a2n[-1:]], axis=0)]
    for n in range(1, ntrac):
        qcs = qcko[n][:km1] + qcko[n][1:]
        qn = W(pc, q1[n][1:] + ptem2x * qcs, q1[n][1:])
        qp = xp.concat([q1[n][:1], qn[:-1]], axis=0)
        qk = W(pc, qp - ptem1x * qcs, qp)
        a2.append(xp.concat([qk, qn[-1:]], axis=0))
    sol = tridin(bk, cl=al, cm=ad, cu=au, r1=a1, r2=a2)
    a1, a2 = sol["a1"], sol["a2"]
    tau = 0.0 + (a1 - t1) * rdt
    rtg = [0.0 + (a2[n] - q1[n]) * rdt for n in range(ntrac)]

    # TKE dissipation rate and dissipative heating
    if dspheat:
        dss = dku * shr2 - g * ti * dkt * bf  # diss(1..km-1)
        tem = govrth * sflux
        tem1 = tem + stress * spd1 / zl[0]
        tem2 = 0.5 * (tem1 + dss[0])
        tem2 = _mx(xp, tem2, 0.0)
        tt1 = tem2 / cp
        temk = 0.5 * (dss[:-1] + dss[1:])  # k = 2..km-1
        temk = _mx(xp, temk, 0.0)
        ttk = temk / cp
        dh = xp.concat([(0.7 * tt1)[None], 0.7 * ttk], axis=0)
        dish = xp.concat([dh, zero[None]], axis=0)
        tau = xp.concat([tau[:km1] + dh, tau[km1:]], axis=0)
        diss = xp.concat([tem1[None], dss], axis=0)  # diss(0..km-1)
    else:
        dish = xp.zeros_like(t1)
        diss = xp.zeros_like(t1)
    # tridiagonal matrix for momentum
    tem1 = dsig * dku * rdz
    dsdz2 = tem1 * rdz
    au0 = -dtodsd * dsdz2
    al0 = -dtodsu * dsdz2
    au = W(pc, au0 - ptem1x, au0)
    al = W(pc, al0 + ptem2x, al0)
    adn = W(pc, 1.0 - al0 + ptem2x, 1.0 - al0)
    adp = xp.concat([(1.0 + beta * stress / spd1)[None], adn[:-1]], axis=0)
    adk = W(pc, adp - au0 - ptem1x, adp - au0)
    ad = xp.concat([adk, adn[-1:]], axis=0)
    ucs = ucko[:km1] + ucko[1:]
    vcs = vcko[:km1] + vcko[1:]
    un = W(pc, u1[1:] + ptem2x * ucs, u1[1:])
    up = xp.concat([u1[:1], un[:-1]], axis=0)
    uk = W(pc, up - ptem1x * ucs, up)
    vn = W(pc, v1[1:] + ptem2x * vcs, v1[1:])
    vp = xp.concat([v1[:1], vn[:-1]], axis=0)
    vk_ = W(pc, vp - ptem1x * vcs, vp)
    sol = tridi2(
        bk,
        cl=al,
        cm=ad,
        cu=au,
        r1=xp.concat([uk, un[-1:]], axis=0),
        r2=xp.concat([vk_, vn[-1:]], axis=0),
    )
    du = 0.0 + (sol["a1"] - u1) * rdt
    dv = 0.0 + (sol["a2"] - v1) * rdt
    return {
        "dv": dv,
        "du": du,
        "tau": tau,
        "dishx": dish,
        "diss": diss,
        "rtg": rtg,
        "kpbl": kpblx,
        "hpbl": hpblx,
        "dkt": dkt,
        "dku": dku,
        "bad": bad,
        "pcnvflg": pcnvflg,
        "ublflg": ublflg,
        "scuflg": scuflg,
        "wspm_adj": adj if alpha < 0 else bad,
    }


# ---------------------------------------------------------------------------
def bl_gfsedmf(
    bk,
    *,
    u3d,
    v3d,
    t3d,
    qv3d,
    qc3d,
    qi3d,
    p3d,
    pi3d,
    dz8w,
    z,
    rthraten,
    psfc,
    psim,
    psih,
    hfx,
    qfx,
    tsk,
    gz1oz0,
    br,
    u10,
    v10,
    znt,
    xland,
    ust,
    wspd,
    dt,
    cp_=C.cp,
    g_=C.g,
    rovcp=C.rovcp,
    r=C.rd,
    rovg=C.rd / C.g,
    ep1=C.rv / C.rd - 1.0,
    karman=C.karman,
    qi=False,
    disheat=True,
    var_ric=1.0,
    coef_ric_l=0.16,
    coef_ric_s=0.25,
    alpha=-1.0,
):
    """BL_GFSEDMF on the interior columns: 3-D arrays (nk, *C) with [k-1]
    Fortran level k (z: the w levels zf, nk levels read), 2-D arrays C.
    ``qi``: P_QI >= P_FIRST_SCALAR (ntrac = 3).  Returns rublten, rvblten,
    rthblten, rqvblten, rqcblten, rqiblten (None without qi), dissten
    (diss(k-1)), dheat, dku3d/dkt3d (levels 2..nk: nk-1 arrays), ust, wspd,
    hpbl, kpbl2d (integer)."""
    xp = bk.xp
    km = t3d.shape[0]
    if km < 4:
        raise ValueError("bl_gfsedmf: needs nk >= 4")
    qv1 = qv3d[0]
    rrhox = (r * t3d[0] * (1.0 + ep1 * qv1)) / psfc
    cpm = cp_ * (1.0 + 0.8 * qv1)
    fmtmp = gz1oz0 - psim
    psk = xp.pow(psfc * 0.00001, rovcp)
    fh = gz1oz0 - psih
    heat = hfx / cpm * rrhox
    evap = qfx * rrhox
    stress = karman * karman * wspd * wspd / (fmtmp * fmtmp)
    q1 = [qv3d / (1.0 + qv3d), qc3d / (1.0 + qc3d)]
    if qi:
        q1.append(qi3d / (1.0 + qi3d))
    prsl = p3d * 0.001
    prslk = xp.pow(prsl * 0.01, rovcp)
    dl = prsl[: km - 1] / rovg * dz8w[: km - 1] / t3d[: km - 1]
    dl = xp.concat([dl, dl[-1:]], axis=0)

    def pstep(p, d):
        pn = p - d
        return pn, pn

    p1 = psfc * 0.001
    _, pr = bk.scan(pstep, p1, dl[: km - 1])
    prsi = xp.concat([p1[None], pr, (pr[-1] - dl[km - 2])[None]], axis=0)
    phii = xp.concat(
        [
            xp.zeros_like(p1)[None],
            (z[1:km] - z[0]) * g_,
            ((z[km - 1] - z[0]) * g_ + dz8w[km - 1] * g_)[None],
        ],
        axis=0,
    )
    phil = 0.5 * (z[1:km] + z[: km - 1] - 2.0 * z[0]) * g_
    phil = xp.concat([phil, (phii[km - 1] - phil[km - 2] + phii[km - 1])[None]], axis=0)
    m = moninedmf(
        bk,
        u1=u3d,
        v1=v3d,
        t1=t3d,
        q1=q1,
        thraten=rthraten,
        psk=psk,
        rbsoil=br,
        zorl=znt * 100.0,
        u10m=u10,
        v10m=v10,
        fm=fmtmp,
        fh=fh,
        tsea=tsk,
        heat=heat,
        evap=evap,
        stress=stress,
        spd1=wspd,
        prsi=prsi,
        dl=dl,
        prsl=prsl,
        prslk=prslk,
        phii=phii,
        phil=phil,
        delt=dt,
        xland1=xland,
        dspheat=disheat,
        var_ric=var_ric,
        coef_ric_l=coef_ric_l,
        coef_ric_s=coef_ric_s,
        alpha=alpha,
    )
    if alpha < 0 and not _tracing(bk, m["bad"]) and bool(bk.to_numpy(xp.any(m["bad"]))):
        raise UndefinedWspm(
            "bl_gfsedmf: alpha < 0 over water with no level bracketing 500 m "
            "among k = 1..nk/2 (WSPM(i,2) is undefined in module_bl_gfsedmf.F)"
        )
    rtg = m["rtg"]
    om = [1.0 - q for q in q1]
    out = {
        "rvblten": m["dv"],
        "rublten": m["du"],
        "rthblten": m["tau"] / pi3d,
        "rqvblten": rtg[0] / (om[0] * om[0]),
        "rqcblten": rtg[1] / (om[1] * om[1]),
        "rqiblten": rtg[2] / (om[2] * om[2]) if qi else None,
        "dheat": m["dishx"],
        "dissten": m["diss"],
        "ust": xp.sqrt(stress),
        "wspd": xp.sqrt(u3d[0] * u3d[0] + v3d[0] * v3d[0]) + 1.0e-9,
        "hpbl": m["hpbl"],
        "kpbl2d": m["kpbl"],
        "dku3d": m["dku"],
        "dkt3d": m["dkt"],
    }
    out["flags"] = {k: m[k] for k in ("pcnvflg", "ublflg", "scuflg", "wspm_adj")}
    return out


# ---------------------------------------------------------------------------
# CM1 glue (registered as ipbl=3 in cm1x.pbl)
WRF_INPUTS = True
DUMP_KEYS = ("xkzh", "xkzm")


def _cfg(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def options(cfg):
    """param.F: the dissipative heating is part of the GFS-EDMF theta
    tendency, so idiss = 0 (cm1setup /= 4)."""
    if int(_cfg(cfg, "cm1setup", 1)) != 4:
        cfg.idiss = 0


def check(cfg):
    sfc = int(_cfg(cfg, "sfcmodel", 0))
    if sfc not in (2, 3, 4):
        raise NotImplementedError(
            f"ipbl=3 (GFS-EDMF) with sfcmodel={sfc} is not ported"
        )
    if int(_cfg(cfg, "nk", 10)) < 4:
        raise NotImplementedError("ipbl=3 needs nk >= 4")


def init_state(ctx):
    """cm1.F: xkzh, xkzm (DKT3D, DKU3D) allocated with zeros."""
    z = ctx.bk.zeros(ctx.g.s_shape)
    return {"xkzh": z, "xkzm": ctx.bk.copy(z)}


def pbl(ctx):
    """turb.F dogfsedmf: BL_GFSEDMF on the interior columns with the turb.F
    inputs (``ctx.w``), the surface layer's 2-D fields and, for sfcmodel=4,
    the GFDL momentum roughness mznt (else mznt = znt).  Returns the PBL
    tendencies (as BL_GFSEDMF gives them), xkzh/xkzm (levels 2..nk),
    dissten (1..nk), ust, wspd, hpbl."""
    bk, g, cfg, s, w = ctx.bk, ctx.g, ctx.cfg, ctx.s, ctx.w
    ni, nj, nk, n, z = g.ni, g.nj, g.nk, g.ngxy, g.ngz
    K, J, I = slice(z, z + nk), slice(n, n + nj), slice(n, n + ni)

    def c3(a):
        return a[K, J, I]

    def c2(a):
        return a[J, I]

    imoist = int(_cfg(cfg, "imoist", 0)) == 1
    qi = imoist and int(_cfg(cfg, "nqi", 0)) >= 1
    znt = s["mznt"] if int(_cfg(cfg, "sfcmodel", 0)) == 4 else s["znt"]
    r = bl_gfsedmf(
        bk,
        u3d=c3(w["u"]),
        v3d=c3(w["v"]),
        t3d=c3(w["t"]),
        qv3d=c3(w["qv"]),
        qc3d=c3(w["qc"]),
        qi3d=c3(w["qi"]),
        p3d=c3(ctx.prs),
        pi3d=c3(w["pi"]),
        dz8w=c3(w["dz8w"]),
        z=ctx.b["zf"][K, J, I],
        rthraten=c3(w["rthraten"]),
        psfc=c2(s["psfc"]),
        psim=c2(s["psim"]),
        psih=c2(s["psih"]),
        hfx=c2(s["hfx"]),
        qfx=c2(s["qfx"]),
        tsk=c2(s["tsk"]),
        gz1oz0=c2(s["gz1oz0"]),
        br=c2(s["br"]),
        u10=c2(s["u10"]),
        v10=c2(s["v10"]),
        znt=c2(znt),
        xland=c2(s["xland"]),
        ust=c2(s["ust"]),
        wspd=c2(s["wspd"]),
        dt=ctx.dt,
        qi=qi,
    )
    out = {}
    names = [("upten", "rublten"), ("vpten", "rvblten"), ("thpten", "rthblten")]
    if imoist:
        names += [("qvpten", "rqvblten"), ("qcpten", "rqcblten")]
        if qi:
            names.append(("qipten", "rqiblten"))
    for name, key in names:
        out[name] = bk.set(bk.copy(s[name]), (K, J, I), r[key])
    K2 = slice(z + 1, z + nk)  # k = 2..nk
    out["xkzm"] = bk.set(bk.copy(s["xkzm"]), (K2, J, I), r["dku3d"])
    out["xkzh"] = bk.set(bk.copy(s["xkzh"]), (K2, J, I), r["dkt3d"])
    out["dissten"] = bk.set(bk.copy(s["dissten"]), (K, J, I), r["dissten"])
    for name in ("ust", "wspd", "hpbl"):
        out[name] = bk.set(bk.copy(s[name]), (J, I), r[name])
    return out
