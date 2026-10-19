"""MYNN surface layer (sfcmodel=6), ported from src/module_sf_mynn.F.

``psi_init`` (psi_opt=0: the stability-function lookup tables),
``sfclay_mynn`` (SFCLAY_mynn + SFCLAY1D_mynn, every column at once) with
``charnock_1955``, ``davis_etal_2008``, ``taylor_yelland_2001``,
``zilitinkevich_1995``, ``garratt_1992``, ``fairall_etal_2003``,
``andreas_2002``, ``li_etal_2010``, ``zolrib`` and the psi functions, plus the
CM1 glue around the call in src/turb.F (``prepare_inputs``: the k2=2
"variables for wrf physics" block; ``finish``: the sfcmodel=6
post-processing).

Reachable in CM1 and ported: COARE_OPT=3.0, psi_opt=0, spp_pbl=0 (fixed by
the module / CM1), isftcflx = 0..3 over water, iz0tlnd = 0, 1, 3 over land,
snowh >= 0.1 (Andreas 2002), isfflx = 0/1, itimestep == 1 and later.  Not
ported (``NotImplementedError``): isftcflx=4 (z_t/z_q left undefined by the
Fortran), iz0tlnd=2 (Yang 2008 reads an uninitialised ``qstar``).

Branches on column values are ``xp.where`` selections; the iterative z/L
solution (``zolrib``, at most 19 passes) is a fixed number of masked passes
with the Fortran's convergence test.  The order of operations follows the
Fortran (and the bit-for-bit C++ port in cm1k/physics/sfc_mynn).
"""

from __future__ import annotations

import math

from . import constants as C

WMIN = 0.1
VCONVC = 1.25
KARMAN = 0.4
EP_3 = 1.0 - 287.04 / 461.5  # module_sf_mynn: 1 - ep_2 (module_model_constants)


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def _fmax(xp, a, b):
    return xp.where(a > b, a, b)


def _fmin(xp, a, b):
    return xp.where(a < b, a, b)


def check(cfg):
    if int(_opt(cfg, "sfcmodel", 1)) != 6:
        return
    isftcflx = int(_opt(cfg, "isftcflx", 0))
    iz0tlnd = int(_opt(cfg, "iz0tlnd", 0))
    if isftcflx not in (0, 1, 2, 3):
        raise NotImplementedError(
            f"sfc_mynn: isftcflx={isftcflx} not ported (0..3; 4 leaves z_t undefined)"
        )
    if iz0tlnd not in (0, 1, 3):
        raise NotImplementedError(
            f"sfc_mynn: iz0tlnd={iz0tlnd} not ported (0, 1, 3; 2 reads an "
            "uninitialised qstar in module_sf_mynn)"
        )


# ---------------------------------------------------------------------------
# stability functions (psi_opt = 0)


def psim_stable_full(xp, zolf):
    return -6.1 * xp.log(zolf + (1 + zolf**2.5) ** (1.0 / 2.5))


def psih_stable_full(xp, zolf):
    return -5.3 * xp.log(zolf + (1 + zolf**1.1) ** (1.0 / 1.1))


def psim_unstable_full(xp, zolf):
    x = (1.0 - 16.0 * zolf) ** 0.25
    psimk = (
        2 * xp.log(0.5 * (1 + x))
        + xp.log(0.5 * (1 + x * x))
        - 2.0 * xp.atan(x)
        + 2.0 * math.atan(1.0)
    )
    ym = (1.0 - 10.0 * zolf) ** 0.33
    s3 = math.sqrt(3.0)
    psimc = (
        (3.0 / 2.0) * xp.log((ym * ym + ym + 1.0) / 3.0)
        - s3 * xp.atan((2.0 * ym + 1) / s3)
        + 4.0 * math.atan(1.0) / s3
    )
    return (psimk + zolf * zolf * (psimc)) / (1 + zolf * zolf)


def psih_unstable_full(xp, zolf):
    y = (1.0 - 16.0 * zolf) ** 0.5
    psihk = 2.0 * xp.log((1 + y) / 2.0)
    yh = (1.0 - 34.0 * zolf) ** 0.33
    s3 = math.sqrt(3.0)
    psihc = (
        (3.0 / 2.0) * xp.log((yh * yh + yh + 1.0) / 3.0)
        - s3 * xp.atan((2.0 * yh + 1) / s3)
        + 4.0 * math.atan(1.0) / s3
    )
    return (psihk + zolf * zolf * (psihc)) / (1 + zolf * zolf)


def psi_init(bk):
    """psi_init(psi_opt=0): psim_stab, psim_unstab, psih_stab, psih_unstab
    on zolf = +-0.01*N, N = 0..1000."""
    xp = bk.xp
    n = xp.astype(xp.arange(1001), bk.dtype)
    zs = n * 0.01
    zu = -n * 0.01
    return {
        "psim_stab": psim_stable_full(xp, zs),
        "psih_stab": psih_stable_full(xp, zs),
        "psim_unstab": psim_unstable_full(xp, zu),
        "psih_unstab": psih_unstable_full(xp, zu),
    }


def _lookup(bk, tab, x, full):
    """``nzol = int(x*100); rzol = x*100 - nzol``; table interpolation when
    nzol+1 <= 1000, else the full function (x = +-zolf)."""
    xp = bk.xp
    y = x * 100.0
    nz = xp.trunc(y)
    rz = y - nz
    ok = (nz + 1) <= 1000
    idx = xp.astype(xp.clip(nz, 0, 999), xp.int64)
    shp = idx.shape
    fi = xp.reshape(idx, (-1,))
    t0 = xp.reshape(xp.take(tab, fi), shp)
    t1 = xp.reshape(xp.take(tab, fi + 1), shp)
    return xp.where(ok, t0 + rz * (t1 - t0), full)


def _safe(xp, cond, x, v):
    """x where cond, else the harmless value v (keeps the unused branch of a
    where finite)."""
    return xp.where(cond, x, v)


def psim_stable(bk, T, zolf):
    xp = bk.xp
    big = zolf * 100.0 >= 1000.0
    full = psim_stable_full(xp, _safe(xp, big, zolf, 10.0))
    return _lookup(bk, T["psim_stab"], zolf, full)


def psih_stable(bk, T, zolf):
    xp = bk.xp
    big = zolf * 100.0 >= 1000.0
    full = psih_stable_full(xp, _safe(xp, big, zolf, 10.0))
    return _lookup(bk, T["psih_stab"], zolf, full)


def psim_unstable(bk, T, zolf):
    xp = bk.xp
    big = -zolf * 100.0 >= 1000.0
    full = psim_unstable_full(xp, _safe(xp, big, zolf, -10.0))
    return _lookup(bk, T["psim_unstab"], -zolf, full)


def psih_unstable(bk, T, zolf):
    xp = bk.xp
    big = -zolf * 100.0 >= 1000.0
    full = psih_unstable_full(xp, _safe(xp, big, zolf, -10.0))
    return _lookup(bk, T["psih_unstab"], -zolf, full)


# ---------------------------------------------------------------------------
# roughness lengths


def charnock_1955(xp, ustar, wsp10, visc, zu):
    g, czo2 = 9.81, 0.011
    wsp10m = wsp10 * math.log(10.0 / 1e-4) / xp.log(zu / 1e-4)
    czc = czo2 + 0.007 * _fmin(xp, _fmax(xp, (wsp10m - 10.0) / 8.0, 0.0), 1.0)
    z0 = czc * ustar * ustar / g + (0.11 * visc / _fmax(xp, ustar, 0.05))
    z0 = _fmax(xp, z0, 1.27e-7)
    return _fmin(xp, z0, 2.85e-3)


def davis_etal_2008(xp, ustar):
    g, ozo = 9.81, 1.59e-5
    zw = _fmin(xp, (ustar / 1.06) ** 0.3, 1.0)
    zn1 = 0.011 * ustar * ustar / g + ozo
    zn2 = 10.0 * xp.exp(-9.5 * ustar**-0.3333) + 0.11 * 1.5e-5 / _fmax(xp, ustar, 0.01)
    z0 = (1.0 - zw) * zn1 + zw * zn2
    z0 = _fmax(xp, z0, 1.27e-7)
    return _fmin(xp, z0, 2.85e-3)


def taylor_yelland_2001(xp, ustar, wsp10):
    g, pi = 9.81, 3.14159265
    hs = 0.0248 * (wsp10 * wsp10)
    tp = 0.729 * _fmax(xp, wsp10, 0.1)
    lp = g * (tp * tp) / (2 * pi)
    z0 = 1200.0 * hs * (hs / lp) ** 4.5
    z0 = _fmax(xp, z0, 1.27e-7)
    return _fmin(xp, z0, 2.85e-3)


def zilitinkevich_1995(xp, z_0, restar, landsea, iz0tlnd):
    """Returns (zt, zq)."""
    sr = xp.sqrt(restar)
    # water (landsea > 1.5)
    zt_s = z_0 * math.exp(KARMAN * 2.0)
    zt_s = _fmax(xp, _fmin(xp, zt_s, 6.0e-5), 2.0e-9)
    zq_s = z_0 * math.exp(KARMAN * 3.0)
    zq_s = _fmax(xp, _fmin(xp, zq_s, 6.0e-5), 2.0e-9)
    zt_r = z_0 * xp.exp(-KARMAN * (4.0 * sr - 3.2))
    zt_r = _fmax(xp, _fmin(xp, zt_r, 6.0e-5), 2.0e-9)
    zq_r = _fmax(xp, zt_r, 2.0e-9)  # sic: zq = max(zt, 2e-9)
    small = restar < 0.1
    zt_w = xp.where(small, zt_s, zt_r)
    zq_w = xp.where(small, zq_s, zq_r)
    # land
    if iz0tlnd == 1:
        czil = 10.0 ** (-0.40 * (z_0 / 0.07))
    else:
        czil = 0.085
    zt_l = z_0 * xp.exp(-KARMAN * czil * sr)
    zt_l = _fmin(xp, zt_l, 0.75 * z_0)
    zq_l = z_0 * xp.exp(-KARMAN * czil * sr)
    zq_l = _fmin(xp, zq_l, 0.75 * z_0)
    water = landsea - 1.5 > 0
    return xp.where(water, zt_w, zt_l), xp.where(water, zq_w, zq_l)


def garratt_1992(xp, z_0, ren, landsea):
    """Returns (zt, zq)."""
    e = 2.71828183
    r4 = ren**0.25
    zt = z_0 * xp.exp(2.0 - (2.48 * r4))
    zq = z_0 * xp.exp(2.0 - (2.28 * r4))
    zq = _fmax(xp, _fmin(xp, zq, 5.5e-5), 2.0e-9)
    zt = _fmax(xp, _fmin(xp, zt, 5.5e-5), 2.0e-9)
    zl = z_0 / (e * e)
    water = landsea - 1.5 > 0
    return xp.where(water, zt, zl), xp.where(water, zq, zl)


def fairall_etal_2003(xp, ren):
    """Returns (zt, zq) (spp_pbl = 0; both Ren branches are the same)."""
    zt = (5.5e-5) * ren**-0.60
    zt = _fmax(xp, _fmin(xp, zt, 1.0e-4), 2.0e-9)
    zq = _fmax(xp, zt, 2.0e-9)  # sic: zq = max(min(zt,..)) on zt
    return zt, zq


def andreas_2002(xp, z_0, bvisc, ustar):
    """Returns (zt, zq)."""
    bt = ((1.25, 0.0, 0.0), (0.149, -0.55, 0.0), (0.317, -0.565, -0.183))
    bq = ((1.61, 0.0, 0.0), (0.351, -0.628, 0.0), (0.396, -0.512, -0.180))
    a = (ustar - 0.18) / 0.1
    zntsno = 0.135 * bvisc / ustar + (0.035 * (ustar * ustar) / 9.8) * (
        5.0 * xp.exp(-1.0 * (a * a)) + 1.0
    )
    ren2 = ustar * zntsno / bvisc
    ren2 = xp.where(ren2 > 1000.0, 1000.0, ren2)
    lg = xp.log(ren2)
    s = ren2 <= 0.135
    t = (ren2 > 0.135) & (ren2 < 2.5)

    def ev(c):
        return zntsno * xp.exp(c[0] + c[1] * lg + c[2] * (lg * lg))

    zt = xp.where(s, ev(bt[0]), xp.where(t, ev(bt[1]), ev(bt[2])))
    zq = xp.where(s, ev(bq[0]), xp.where(t, ev(bq[1]), ev(bq[2])))
    return zt, zq


# ---------------------------------------------------------------------------
# z/L from the bulk Richardson number


def li_etal_2010(xp, rib, zaz0, z0zt):
    au11, bu11, bu12, bu21, bu22 = 0.045, 0.003, 0.0059, -0.0828, 0.8845
    bu31, bu32, bu33 = 0.1739, -0.9213, -0.1057
    aw11, aw12, aw21, aw22 = 0.5738, -0.4399, -4.901, 52.50
    bw11, bw12, bw21, bw22 = -0.0539, 1.540, -0.669, -3.282
    as11, as21, bs11, bs21, bs22 = 0.7529, 14.94, 0.1569, -0.3091, -1.303
    zaz02 = xp.where(zaz0 < 100.0, 100.0, zaz0)
    zaz02 = xp.where(zaz0 > 100000.0, 100000.0, zaz02)
    z0zt2 = xp.where(z0zt < 0.5, 0.5, z0zt)
    z0zt2 = xp.where(z0zt > 100.0, 100.0, z0zt2)
    alfa = xp.log(zaz02)
    beta = xp.log(z0zt2)
    zl1 = (
        au11 * alfa * (rib * rib)
        + (
            (bu11 * beta + bu12) * (alfa * alfa)
            + (bu21 * beta + bu22) * alfa
            + (bu31 * (beta * beta) + bu32 * beta + bu33)
        )
        * rib
    )
    zl1 = _fmin(xp, _fmax(xp, zl1, -15.0), 0.0)
    zl2 = ((aw11 * beta + aw12) * alfa + (aw21 * beta + aw22)) * (rib * rib) + (
        (bw11 * beta + bw12) * alfa + (bw21 * beta + bw22)
    ) * rib
    zl2 = _fmax(xp, _fmin(xp, zl2, 4.0), 0.0)
    zl3 = (as11 * alfa + as21) * rib + bs11 * alfa + bs21 * beta + bs22
    zl3 = _fmax(xp, _fmin(xp, zl3, 20.0), 1.0)
    return xp.where(rib <= 0.0, zl1, xp.where(rib <= 0.2, zl2, zl3))


def zolrib(bk, T, ri, za, z0, zt, logz0, logzt, zol1):
    """Iterative z/L (at most nmax-1 = 19 passes, masked), Li et al. (2010)
    if it does not converge."""
    xp = bk.xp
    nmax = 20
    zol1 = xp.where(zol1 * ri < 0.0, 0.0, zol1)
    neg = ri < 0.0
    zolold = xp.where(neg, xp.zeros_like(ri) - 99999.0, 99999.0)
    res = xp.where(neg, xp.zeros_like(ri) - 66666.0, 66666.0)
    n = xp.ones_like(ri)
    for it in range(nmax - 1):
        act = (xp.abs(zolold - res) > 0.01) & (n < nmax)
        zo = zol1 if it == 0 else res
        zolold = xp.where(act, zo, zolold)
        zol20 = zolold * z0 / za
        zol3 = zolold + zol20
        zolt = zolold * zt / za
        psit_u = _fmax(
            xp,
            logzt
            - (
                psih_unstable(bk, T, _fmin(xp, zol3, 0.0))
                - psih_unstable(bk, T, _fmin(xp, zolt, 0.0))
            ),
            1.0,
        )
        psix_u = _fmax(
            xp,
            logz0
            - (
                psim_unstable(bk, T, _fmin(xp, zol3, 0.0))
                - psim_unstable(bk, T, _fmin(xp, zol20, 0.0))
            ),
            1.0,
        )
        psit_s = _fmax(
            xp,
            logzt
            - (
                psih_stable(bk, T, _fmax(xp, zol3, 0.0))
                - psih_stable(bk, T, _fmax(xp, zolt, 0.0))
            ),
            1.0,
        )
        psix_s = _fmax(
            xp,
            logz0
            - (
                psim_stable(bk, T, _fmax(xp, zol3, 0.0))
                - psim_stable(bk, T, _fmax(xp, zol20, 0.0))
            ),
            1.0,
        )
        psit2 = xp.where(neg, psit_u, psit_s)
        psix2 = xp.where(neg, psix_u, psix_s)
        res = xp.where(act, ri * (psix2 * psix2) / psit2, res)
        n = xp.where(act, n + 1, n)
    fail = (n == nmax) & (xp.abs(zolold - res) > 0.01)
    return xp.where(fail, li_etal_2010(xp, ri, za / z0, z0 / zt), res)


# ---------------------------------------------------------------------------


def sfclay_mynn(
    bk,
    cfg,
    tables,
    *,
    u3d,
    v3d,
    t3d,
    qv3d,
    p3d,
    dz8w,
    rho3d,
    psfcpa,
    pblh,
    mavail,
    xland,
    tsk,
    snowh,
    ust,
    mol,
    qsfc,
    znt,
    ustm,
    hfx,
    qfx,
    zol,
    rmol,
    wspd,
    itimestep,
    dx,
    isfflx=1,
):
    """SFCLAY_mynn + SFCLAY1D_mynn on every column at once.

    Column inputs: u3d, v3d, dz8w at the two lowest levels (``u3d[0]`` is
    k=1, ``u3d[1]`` k=2), t3d, qv3d (mixing ratio), p3d, rho3d at k=1 (2-D).
    2-D inputs psfcpa, pblh, mavail, xland, tsk, snowh; persistent inout ust,
    mol, qsfc, znt, ustm, hfx, qfx, zol, rmol, wspd.  ``itimestep`` is a
    Python int (CM1's nstep), ``dx`` CM1's dx.  Returns the inout arrays and
    chs, chs2, cqs2, cpm, regime, psim, psih, lh, flhc, flqc, qgh, u10, v10,
    th2, t2, q2, gz1oz0, br, ch, ck, cka, cd, cda.  Constants: CM1's (cp, g,
    rovcp, xlv, svp1..3, svpt0), ep1 = rv/rd-1, ep2 = rd/rv as turb.F."""
    check(cfg)
    xp = bk.xp
    T = tables
    isftcflx = int(_opt(cfg, "isftcflx", 0))
    iz0tlnd = int(_opt(cfg, "iz0tlnd", 0))
    cp, g, rovcp, xlv = C.cp, C.g, C.rovcp, C.xlv
    svp1, svp2, svp3, svpt0 = C.svp1, C.svp2, C.svp3, C.svpt0
    ep1 = C.rv / C.rd - 1.0
    ep2 = C.rd / C.rv
    prt = 1.0
    log, exp, sqrt = xp.log, xp.exp, xp.sqrt

    dz8w1d, dz2w1d = dz8w[0], dz8w[1]
    u1d, v1d, u1d2, v1d2 = u3d[0], v3d[0], u3d[1], v3d[1]
    qv1d, p1d, t1d, rho1d = qv3d, p3d, t3d, rho3d
    # itimestep: a Python int, or a 0-d array (a step counter carried in the
    # model state, so that a compiled step sees the current value)
    static = isinstance(itimestep, (int, float))
    if static:
        if itimestep == 1:
            ust = _fmax(xp, 0.04 * sqrt(u1d * u1d + v1d * v1d), 0.001)
            mol = xp.zeros_like(u1d)
            qsfc = qv3d / (1.0 + qv3d)
    else:
        first = itimestep == 1
        ust = xp.where(first, _fmax(xp, 0.04 * sqrt(u1d * u1d + v1d * v1d), 0.001), ust)
        mol = xp.where(first, xp.zeros_like(u1d), mol)
        qsfc = xp.where(first, qv3d / (1.0 + qv3d), qsfc)

    psfc = psfcpa / 1000.0
    thgb = tsk * (100.0 / psfc) ** rovcp
    pl = p1d / 1000.0
    thcon = (100.0 / pl) ** rovcp
    th1d = t1d * thcon
    tc1d = t1d - 273.15
    qvsh = qv1d / (1.0 + qv1d)
    tvcon = 1.0 + ep1 * qvsh
    thv1d = th1d * tvcon
    za = 0.5 * dz8w1d
    za2 = dz8w1d + 0.5 * dz2w1d
    govrth = g / th1d

    cold = tsk < 273.15
    tsk_c = xp.where(cold, tsk, 250.0)
    e1c = svp1 * exp(
        4648 * (1.0 / 273.15 - 1.0 / tsk_c)
        - 11.64 * log(273.15 / tsk_c)
        + 0.02265 * (273.15 - tsk_c)
    )
    e1w = svp1 * exp(svp2 * (tsk - svpt0) / (tsk - svp3))
    e1 = xp.where(cold, e1c, e1w)
    wq = (xland > 1.5) | (qsfc <= 0.0)
    qsfc = xp.where(wq, ep2 * e1 / (psfc - EP_3 * e1), qsfc)
    qsfcmr = xp.where(wq, ep2 * e1 / (psfc - e1), qsfc / (1.0 - qsfc))
    t1_c = xp.where(cold, t1d, 250.0)
    e1c = svp1 * exp(
        4648 * (1.0 / 273.15 - 1.0 / t1_c)
        - 11.64 * log(273.15 / t1_c)
        + 0.02265 * (273.15 - t1_c)
    )
    e1w = svp1 * exp(svp2 * (t1d - svpt0) / (t1d - svp3))
    e1 = xp.where(cold, e1c, e1w)
    pl = p1d / 1000.0
    qgh = ep2 * e1 / (pl - e1)
    cpm = cp * (1.0 + 0.84 * qv1d)

    wspd = sqrt(u1d * u1d + v1d * v1d)
    thvgb = thgb * (1.0 + ep1 * qsfc)
    dthvdz = thv1d - thvgb
    fluxc = _fmax(xp, hfx / rho1d / cp + ep1 * thvgb * qfx / rho1d, 0.0)
    wst_w = VCONVC * (g / tsk * pblh * fluxc) ** 0.33
    wst_l = VCONVC * (g / tsk * _fmin(xp, 1.5 * pblh, 4000.0) * fluxc) ** 0.33
    wstar = xp.where((xland > 1.5) | (qsfc <= 0.0), wst_w, wst_l)  # updated qsfc
    vsgd = 0.32 * max(dx / 5000.0 - 1.0, 0.0) ** 0.33
    wspd = sqrt(wspd * wspd + wstar * wstar + vsgd * vsgd)
    wspd = _fmax(xp, wspd, WMIN)
    br = govrth * za * dthvdz / (wspd * wspd)
    if static:
        lim = 2.0 if itimestep == 1 else 4.0
    else:
        lim = xp.where(itimestep == 1, 2.0 + 0.0 * br, 4.0 + 0.0 * br)
    br = _fmin(xp, _fmax(xp, br, -lim), lim)

    visc = 1.326e-5 * (
        1.0 + 6.542e-3 * tc1d + 8.301e-6 * tc1d * tc1d - 4.84e-9 * tc1d * tc1d * tc1d
    )
    water = (xland - 1.5) >= 0
    # water roughness
    if isftcflx == 0:
        znt_w = charnock_1955(xp, ust, wspd, visc, za)
    elif isftcflx in (1, 2):
        znt_w = davis_etal_2008(xp, ust)
    else:
        znt_w = taylor_yelland_2001(xp, ust, wspd)
    znt = xp.where(water, znt_w, znt)
    zntstoch = znt
    restar = _fmax(xp, ust * zntstoch / visc, 0.1)
    if isftcflx == 2:
        zt_w, zq_w = garratt_1992(xp, zntstoch, restar, xland)
    else:
        zt_w, zq_w = fairall_etal_2003(xp, restar)
    # land
    sn = snowh >= 0.1
    ust_a = xp.where(sn & ~water, ust, 1.0)
    zt_a, zq_a = andreas_2002(xp, zntstoch, visc, ust_a)
    if iz0tlnd <= 1:
        zt_n, zq_n = zilitinkevich_1995(xp, zntstoch, restar, xland, iz0tlnd)
    else:
        zt_n, zq_n = garratt_1992(xp, zntstoch, restar, xland)
    z_t = xp.where(water, zt_w, xp.where(sn, zt_a, zt_n))
    z_q = xp.where(water, zq_w, xp.where(sn, zq_a, zq_n))

    zratio = zntstoch / z_t
    gz1oz0 = log((za + zntstoch) / zntstoch)
    gz1ozt = log((za + zntstoch) / z_t)
    gz2ozt = log((2.0 + zntstoch) / z_t)
    gz10oz0 = log((10.0 + zntstoch) / zntstoch)
    gz10ozt = log((10.0 + zntstoch) / z_t)

    stable = br > 0.0
    unstable = br < 0.0
    neutral = br == 0.0
    # first guess of z/L
    if static and itimestep <= 1:
        zg = li_etal_2010(xp, br, za / zntstoch, zratio)
        zg_s = zg
        zg_u = zg
    else:
        zg_s = za * KARMAN * g * mol / (th1d * _fmax(xp, ust * ust, 0.0001))
        zg_s = _fmin(xp, _fmax(xp, zg_s, 0.0), 20.0)
        zg_u = za * KARMAN * g * mol / (th1d * _fmax(xp, ust * ust, 0.001))
        zg_u = _fmin(xp, _fmax(xp, zg_u, -20.0), 0.0)
        if not static:
            zg = li_etal_2010(xp, br, za / zntstoch, zratio)
            zg_s = xp.where(itimestep <= 1, zg, zg_s)
            zg_u = xp.where(itimestep <= 1, zg, zg_u)
    zg = xp.where(stable, zg_s, zg_u)
    zr = zolrib(bk, T, br, za, zntstoch, z_t, gz1oz0, gz1ozt, zg)
    zr_s = _fmin(xp, _fmax(xp, zr, 0.0), 20.0)
    zr_u = _fmin(xp, _fmax(xp, zr, -20.0), 0.0)
    zol = xp.where(stable, zr_s, xp.where(unstable, zr_u, xp.where(neutral, 0.0, zol)))

    zolzt = zol * z_t / za
    zolz0 = zol * zntstoch / za
    zolza = zol * (za + zntstoch) / za
    zol10 = zol * (10.0 + zntstoch) / za
    zol2 = zol * (2.0 + zntstoch) / za

    def p(f, x, s):  # psi on the clamped side (s: +1 stable, -1 unstable)
        return f(bk, T, _fmax(xp, x, 0.0) if s > 0 else _fmin(xp, x, 0.0))

    psim_s = p(psim_stable, zolza, 1) - p(psim_stable, zolz0, 1)
    psih_s = p(psih_stable, zolza, 1) - p(psih_stable, zolzt, 1)
    psim10_s = p(psim_stable, zol10, 1) - p(psim_stable, zolz0, 1)
    psih10_s = p(psih_stable, zol10, 1) - p(psih_stable, zolz0, 1)
    psih2_s = p(psih_stable, zol2, 1) - p(psih_stable, zolz0, 1)
    psim_u = p(psim_unstable, zolza, -1) - p(psim_unstable, zolz0, -1)
    psih_u = p(psih_unstable, zolza, -1) - p(psih_unstable, zolzt, -1)
    psim10_u = p(psim_unstable, zol10, -1) - p(psim_unstable, zolz0, -1)
    psih10_u = p(psih_unstable, zol10, -1) - p(psih_unstable, zolz0, -1)
    psih2_u = p(psih_unstable, zol2, -1) - p(psih_unstable, zolz0, -1)
    psih_u = _fmin(xp, psih_u, 0.9 * gz1ozt)
    psim_u = _fmin(xp, psim_u, 0.9 * gz1oz0)
    psih2_u = _fmin(xp, psih2_u, 0.9 * gz2ozt)
    psim10_u = _fmin(xp, psim10_u, 0.9 * gz10oz0)
    psih10_u = _fmin(xp, psih10_u, 0.9 * gz10ozt)

    def sel(s, u, n):
        return xp.where(stable, s, xp.where(unstable, u, n))

    nan = float("nan")
    regime = xp.where(
        stable,
        xp.where(br > 0.2, xp.ones_like(br), 2.0),
        xp.where(neutral, xp.ones_like(br) * 3.0, 4.0),
    )
    psim = sel(psim_s, psim_u, 0.0)
    psih = sel(psih_s, psih_u, 0.0)
    nn = xp.where(neutral, 0.0, nan)
    psim10 = sel(psim10_s, psim10_u, nn)
    psih10 = sel(psih10_s, psih10_u, nn)
    psih2 = sel(psih2_s, psih2_u, nn)
    rmol = xp.where(stable | unstable | neutral, zol / za, rmol)

    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    ust = 0.5 * ust + 0.5 * KARMAN * wspd / psix
    wspdi = _fmax(xp, sqrt(u1d * u1d + v1d * v1d), WMIN)
    ustm = 0.5 * ustm + 0.5 * KARMAN * wspdi / psix
    land = (xland - 1.5) < 0.0
    ust = xp.where(land, _fmax(xp, ust, 0.005), ust)
    ustm = xp.where(land, ust, ustm)
    gz1ozt = log((za + zntstoch) / z_t)
    gz2ozt = log((2.0 + zntstoch) / z_t)
    dtg1 = thv1d - thvgb
    mol = KARMAN * dtg1 / _fmax(xp, gz1ozt - psih, 1.0) / prt

    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    psit = _fmax(xp, gz1ozt - psih, 1.0)
    psit2 = _fmax(xp, gz2ozt - psih2, 1.0)
    psiq = _fmax(xp, log((za + z_q) / z_q) - psih, 1.0)
    psiq2 = _fmax(xp, log((2.0 + z_q) / z_q) - psih2, 1.0)
    psiq10 = _fmax(xp, log((10.0 + z_q) / z_q) - psih10, 1.0)
    z = xp.zeros_like(ust)
    if isfflx < 1:
        qfx = hfx = flhc = flqc = lh = chs = ch = chs2 = cqs2 = ck = cd = cka = cda = z
    else:
        flqc = rho1d * mavail * ust * KARMAN / psiq
        flhc = rho1d * cpm * ust * KARMAN / psit
        qfx = flqc * (qsfcmr - qv1d)
        qfx = _fmax(xp, qfx, -0.02)
        lh = xlv * qfx
        hw = flhc * (thgb - th1d)
        if isftcflx != 0:
            hw = hw + rho1d * ustm * ustm * wspdi
        hl = _fmax(xp, flhc * (thgb - th1d), -250.0)
        hfx = xp.where(xland - 1.5 > 0.0, hw, xp.where(xland - 1.5 < 0.0, hl, hfx))
        chs = ust * KARMAN / psit
        ch = flhc / (cpm * rho1d)
        cqs2 = ust * KARMAN / psiq2
        chs2 = ust * KARMAN / psit2
        ck = (KARMAN / psix10) * (KARMAN / psiq10)
        cd = (KARMAN / psix10) * (KARMAN / psix10)
        cka = (KARMAN / psix) * (KARMAN / psiq)
        cda = (KARMAN / psix) * (KARMAN / psix)

    rlog = log(10.0 / zntstoch) / log(za / zntstoch)
    ul, vl = u1d * rlog, v1d * rlog
    up, vp = u1d * psix10 / psix, v1d * psix10 / psix
    low = za <= 7.0
    mid = (za > 7.0) & (za < 13.0)
    lv2 = (za2 > 7.0) & (za2 < 13.0)
    u10 = xp.where(low, xp.where(lv2, u1d2, ul), xp.where(mid, ul, up))
    v10 = xp.where(low, xp.where(lv2, v1d2, vl), xp.where(mid, vl, vp))
    dtg = th1d - thgb
    th2 = thgb + dtg * psit2 / psit
    bad = ((th1d > thgb) & ((th2 < thgb) | (th2 > th1d))) | (
        (th1d < thgb) & ((th2 > thgb) | (th2 < th1d))
    )
    th2 = xp.where(bad, thgb + 2.0 * (th1d - thgb) / za, th2)
    t2 = th2 * (psfc / 100.0) ** rovcp
    q2 = qsfcmr + (qv1d - qsfcmr) * psiq2 / psiq
    q2 = _fmax(xp, q2, _fmin(xp, qsfcmr, qv1d))
    q2 = _fmin(xp, q2, 1.05 * qv1d)
    return {
        "ust": ust,
        "mol": mol,
        "qsfc": qsfc,
        "znt": znt,
        "ustm": ustm,
        "hfx": hfx,
        "qfx": qfx,
        "zol": zol,
        "rmol": rmol,
        "wspd": wspd,
        "chs": chs,
        "chs2": chs2,
        "cqs2": cqs2,
        "cpm": cpm,
        "regime": regime,
        "psim": psim,
        "psih": psih,
        "lh": lh,
        "flhc": flhc,
        "flqc": flqc,
        "qgh": qgh,
        "u10": u10,
        "v10": v10,
        "th2": th2,
        "t2": t2,
        "q2": q2,
        "gz1oz0": gz1oz0,
        "br": br,
        "ch": ch,
        "ck": ck,
        "cka": cka,
        "cd": cd,
        "cda": cda,
    }


# ---------------------------------------------------------------------------
# the model glue (cm1x.pbl registry: SURFACE_LAYERS[6])

# the surface arrays SFCLAY_mynn writes that cm1x.pbl.SFC_KEYS and SLAB's
# init_state do not create (cm1.F allocates them as 0)
_OWN_2D = ("ustm", "chs", "chs2", "cqs2", "regime", "qgh", "ck", "cka", "snowh")
# carried step counter (itimestep = nstep; see surface_layer)
NSTEP_KEY = "sfc_mynn_nstep"
DUMP_KEYS = ("mol", "rmol", "zol", "ustm", "tslb")


def init_state(ctx):
    """The MYNN surface layer's own 2-D arrays (0, as cm1.F allocates them)
    and the step counter (-1: the startup pass makes it 0)."""
    bk, g, s = ctx.bk, ctx.g, ctx.s
    # distinct arrays (bk.set works in place on numpy/torch)
    out = {k: bk.zeros(g.s_shape[1:]) for k in _OWN_2D if k not in s}
    out[NSTEP_KEY] = bk.zeros(()) - 1.0
    return out


def surface_layer(ctx):
    """turb.F, sfcmodel=6: the k2=2 "variables for wrf physics", the
    mynn_sf_j_loop (SFCLAY_mynn with isfflx=1, itimestep=nstep, DX=dx), then
    SLAB (update_sfc) or the qsfc diagnosis, ch = chs2 / cq = cqs2 and s10,
    ch = cq = ck (sfcmodel=6), thflux / qvflux with dosfcflx or
    output_sfcflx (``cm1x.pbl.wrf_sfc_epilogue``).

    itimestep comes from a counter carried in the state (startup pass 0,
    then 1, 2, ...), equal to CM1's nstep, so that a compiled step uses the
    current value."""
    from .pbl import interior2, wrf_sfc_epilogue

    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    n, z = g.ngxy, g.ngz
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    K2 = slice(z, z + 2)
    w = ctx.w

    def i2(k):
        return interior2(g, s[k])

    nstep = s[NSTEP_KEY] + 1.0
    r = sfclay_mynn(
        bk,
        cfg,
        _tables(bk),
        u3d=w["u"][K2, J, I],
        v3d=w["v"][K2, J, I],
        dz8w=w["dz8w"][K2, J, I],
        t3d=w["t"][z, J, I],
        qv3d=w["qv"][z, J, I],
        p3d=ctx.prs[z, J, I],
        rho3d=ctx.rho[z, J, I],
        psfcpa=i2("psfc"),
        pblh=i2("hpbl"),
        mavail=i2("mavail"),
        xland=i2("xland"),
        tsk=i2("tsk"),
        snowh=i2("snowh"),
        **{k: i2(k) for k in _STATE_IN},
        itimestep=nstep,
        dx=float(cfg.dx),
        isfflx=1,
    )
    r["cpmm"] = r.pop("cpm")
    r["tsk"] = i2("tsk")
    out = wrf_sfc_epilogue(ctx, r)
    # turb.F (sfcmodel=6): ch = cq = ck, after ch = chs2, cq = cqs2
    # (copies: wrf_sfc_epilogue writes the state arrays in place, so they
    # must not alias)
    out["ch"] = bk.copy(out["ck"])
    out["cq"] = bk.copy(out["ck"])
    out[NSTEP_KEY] = nstep
    return out


_STATE_IN = ("ust", "mol", "qsfc", "znt", "ustm", "hfx", "qfx", "zol", "rmol", "wspd")


def _tables(bk):
    # rebuilt per call (4 x 1001 values): nothing cached across jit traces
    return psi_init(bk)
