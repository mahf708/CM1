"""The YSU PBL scheme (ipbl=1), ported from src/bl_ysu.F (``bl_ysu_run``,
``tridin_ysu``, ``tridi2n``) as CM1 calls it through ``ysu_wrapper``
(src/mmm_physics_wrapper.F) from the ``ysu_j_loop`` of src/turb.F
(sfc_and_turb), plus the YSU dissipative heating of turb.F (``ysudiss``,
idiss=1).

Every column is computed at once: level-stacked arrays ``[k, column...]``
with Fortran level k at index k-1; the per-column ``if`` blocks are
``xp.where`` selections and the values at the PBL-top level ``kpbl`` are
gathers (``take_along_axis``).  The three bulk-Richardson searches of the
PBL height are evaluated in closed form (the Fortran's ``stable`` loop stops
at the first level whose Richardson number exceeds the critical one) and
the tridiagonal solves run through ``bk.scan``.  Expressions keep the
Fortran's operation order (``x**2`` is ``x*x``; ``x**3.``, ``x**4.`` and
the ``**h1`` / ``**h2`` powers are ``pow``).

What CM1's call fixes, and so what is ported:
  * ctopo, ctopo2, uox, vox, the BEP urban arrays and dusfc..dqsfc are not
    passed (absent optionals): bepswitch = 0, uoxl = voxl = 0 (the
    always-zero BEP and ocean-current terms are dropped), ad(1) = 1 +
    fric(1), u10/v10 are not modified.
  * f_qc = f_qi = .true.; nmix = 1 with qmix = 0 and qmixtnp discarded:
    not computed.
  * ysu_topdown_pblmix = .false. (turb.F); the .true. branch (cloud-top
    entrainment) raises ``NotImplementedError``.  Without it wstar3_2 = 0,
    and the terms it multiplies vanish exactly (they are left out).
  * the "paj tke" block and get_pblh only feed vconv, which only matters
    with ctopo present: not ported.
"""

from __future__ import annotations

from . import constants as C

# bl_ysu_run parameters
XKZMINM, XKZMINH = 0.1, 0.01
XKZMAX, RIMIN = 1000.0, -100.0
RLAM, PRMIN, PRMAX = 30.0, 0.25, 4.0
BRCR_UB, BRCR_SB, CORI = 0.0, 0.25, 1.0e-4
AFAC, BFAC = 6.8, 6.8
PHIFAC, SFCFRAC = 8.0, 0.1
D1, D2, D3 = 0.02, 0.05, 0.001
H1, H2 = 0.33333335, 0.6666667
ZFMIN, APHI5, APHI16 = 1.0e-8, 5.0, 16.0
TMIN = 1.0e-2
GAMCRT, GAMCRQ = 3.0, 2.0e-3

# constants as turb.F passes them to the WRF physics
EP1 = C.rv / C.rd - 1.0
EP2 = C.rd / C.rv


def _cat(xp, *a):
    return xp.concat(a, axis=0)


def _take(xp, a, idx):
    """``a(kidx)`` per column: ``a`` level-stacked (Fortran k at index k-1),
    ``idx`` the 0-based level index per column."""
    return xp.take_along_axis(a, idx[None, ...], axis=0)[0]


def _mx(xp, a, b):
    """Fortran ``max(a, b)`` (either may be a Python scalar)."""
    if isinstance(b, (int, float)):
        return xp.clip(a, b, None)
    if isinstance(a, (int, float)):
        return xp.clip(b, a, None)
    return xp.maximum(a, b)


def _mn(xp, a, b):
    """Fortran ``min(a, b)`` (either may be a Python scalar)."""
    if isinstance(b, (int, float)):
        return xp.clip(a, None, b)
    if isinstance(a, (int, float)):
        return xp.clip(b, None, a)
    return xp.minimum(a, b)


def _pow0(xp, x, p):
    """``x**p`` for ``x >= 0`` with a finite gradient at ``x = 0``
    (``pow(0, p) = 0`` exactly, so the value is unchanged)."""
    pos = x > 0.0
    return xp.where(pos, xp.pow(xp.where(pos, x, 1.0), p), 0.0)


def tridin_ysu(bk, al, cm, cu, fs):
    """``tridin_ysu`` (nt = 1) for several right-hand sides ``fs`` sharing
    the matrix: sub-diagonal ``cl(k) = al(k-1)`` (k = 2..n), diagonal ``cm``,
    super-diagonal ``cu`` (all level-stacked, n levels).  ``tridi2n`` is the
    same elimination for two right-hand sides with diagonals ``cm`` and
    ``cm1``, which CM1's call makes identical, so it is this function too.
    Returns the solutions (tuple, like ``fs``)."""
    xp = bk.xp
    fk = 1.0 / cm[0]
    c0 = (fk * cu[0],) + tuple(fk * f[0] for f in fs)

    def fwd(c, x):
        a, b, u = x[0], x[1], x[2]
        fk = 1.0 / (b - a * c[0])
        n = (fk * u,) + tuple(fk * (f - a * fp) for f, fp in zip(x[3:], c[1:]))
        return n, n

    (_, *last), ys = bk.scan(
        fwd, c0, (al[:-1], cm[1:], cu[1:]) + tuple(f[1:] for f in fs)
    )
    aul = _cat(xp, c0[0][None], ys[0][:-1])
    out = []
    for m in range(len(fs)):
        fwdv = _cat(xp, c0[1 + m][None], ys[1 + m][:-1])

        def bwd(fnext, x):
            v = x[0] - x[1] * fnext
            return v, v

        _, xs = bk.scan(bwd, last[m], (fwdv, aul), reverse=True)
        out.append(_cat(xp, xs, last[m][None]))
    return tuple(out)


def _search(bk, thvx, ux, vx, za, thermal, brcr, brup0, kmax):
    """The bulk-Richardson loop over k = 2..kmax (from ``stable = .false.``):
    returns (kpbl, brup, brdn) as the Fortran leaves them."""
    xp = bk.xp
    spdk2 = _mx(xp, ux[1:kmax] * ux[1:kmax] + vx[1:kmax] * vx[1:kmax], 1.0)
    brupk = (thvx[1:kmax] - thermal) * (C.g * za[1:kmax] / thvx[0]) / spdk2
    ext = _cat(xp, brup0[None], brupk)  # Fortran k = 1..kmax (k = 1: brup0)
    over = brupk > brcr
    kpbl = xp.full(brup0.shape, kmax, dtype=xp.int64, device=_dev(brup0))
    for k in range(kmax, 1, -1):
        kpbl = xp.where(over[k - 2], k, kpbl)
    return kpbl, _take(xp, ext, kpbl - 1), _take(xp, ext, kpbl - 2)


def _dev(x):
    from array_api_compat import device

    return device(x)


def _interp(bk, kpbl, brup, brdn, brcr, za, zq2):
    """The ``brint`` interpolation of the PBL height at level kpbl, then
    kpbl = 1 if hpbl < zq(2).  Returns (hpbl, kpbl)."""
    xp = bk.xp
    hi = brdn >= brcr
    lo = brup <= brcr
    den = xp.where(hi | lo, 1.0, brup - brdn)
    brint = xp.where(hi, 0.0, xp.where(lo, 1.0, (brcr - brdn) / den))
    km = xp.clip(kpbl - 2, 0, None)
    zlo = _take(xp, za, km)
    hpbl = zlo + brint * (_take(xp, za, km + 1) - zlo)
    kpbl = xp.where(hpbl < zq2, 1, kpbl)
    return hpbl, kpbl


def bl_ysu_run(
    bk,
    *,
    ux,
    vx,
    tx,
    qvx,
    qcx,
    qix,
    p2d,
    p2di,
    pi2d,
    dz8w2d,
    rthraten,
    psfcpa,
    znt,
    ust,
    psim,
    psih,
    xland,
    hfx,
    qfx,
    wspd,
    br,
    u10,
    v10,
    dt,
    ysu_topdown_pblmix=False,
):
    """``bl_ysu_run`` as ``ysu_wrapper`` calls it, on every column at once.

    3-D inputs are level-stacked ``[nk, ...]`` (Fortran k = 1..nk at index
    0..nk-1; ``p2di`` has nk+1 levels), 2-D inputs ``[...]``.  ``rthraten``
    only feeds the topdown branch.  Returns utnp, vtnp, ttnp, qvtnp, qctnp,
    qitnp, exch_hx, exch_mx (``[nk, ...]``; exch(1) = 0, exch(k+1) = K at the
    top of layer k), hpbl, kpbl1d (int), wstar, delta, u10, v10."""
    if ysu_topdown_pblmix:
        raise NotImplementedError("bl_ysu_run: ysu_topdown_pblmix is not ported")
    del rthraten
    xp = bk.xp
    g, cp, rd, xlv, rv, karman = C.g, C.cp, C.rd, C.xlv, C.rv, C.karman
    ep1 = EP1
    nk = ux.shape[0]
    if nk < 3:
        raise ValueError("bl_ysu_run: needs nk >= 3")
    klpbl = nk
    exp, sqrt, pow, where = xp.exp, xp.sqrt, xp.pow, xp.where

    def mx(a, b):
        return _mx(xp, a, b)

    def mn(a, b):
        return _mn(xp, a, b)

    cont = cp / g
    conpr = BFAC * karman * SFCFRAC

    thx = tx / pi2d
    tvcon = 1.0 + ep1 * qvx
    thvx = thx * tvcon
    rhox = psfcpa / (rd * tx[0] * tvcon[0])
    govrth = g / thx[0]

    # height of full- and half-sigma levels above ground, layer thicknesses
    zl = [xp.zeros_like(psfcpa)]
    for k in range(nk):
        zl.append(dz8w2d[k] + zl[k])
    zq = xp.stack(zl, axis=0)  # nk+1 levels
    za = 0.5 * (zq[:-1] + zq[1:])
    dl = p2di[:-1] - p2di[1:]
    dza = _cat(xp, za[:1], za[1:] - za[:-1])

    wspd1 = sqrt(ux[0] * ux[0] + vx[0] * vx[0]) + 1.0e-9
    dt2 = 2.0 * dt
    rdt = 1.0 / dt2

    zl1 = za[0]
    thermal = thvx[0]
    sflux = hfx / rhox / cp + qfx / rhox * ep1 * thx[0]
    sfcflg = ~(br > 0.0)
    zq2 = zq[1]

    # first guess of pbl height
    zero = xp.zeros_like(br)
    kpbl, brup, brdn = _search(bk, thvx, ux, vx, za, thermal, BRCR_UB, br, klpbl)
    hpbl, kpbl = _interp(bk, kpbl, brup, brdn, BRCR_UB, za, zq2)
    pblflg = kpbl > 1

    zol1 = mx(br * psim * psim / psih, RIMIN)
    zol1 = where(sfcflg, mn(zol1, -ZFMIN), mx(zol1, ZFMIN))
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    a16 = where(sfcflg, 1.0 - APHI16 * hol1, 1.0)
    phim = where(sfcflg, pow(a16, -1.0 / 4.0), 1.0 + APHI5 * hol1)
    phih = where(sfcflg, pow(a16, -1.0 / 2.0), phim)
    bfx0 = mx(sflux, 0.0)
    wstar3 = where(sfcflg, govrth * bfx0 * hpbl, 0.0)
    wstar = _pow0(xp, wstar3, H1)
    ust3 = pow(ust, 3.0)
    wscale = pow(ust3 + PHIFAC * karman * wstar3 * 0.5, H1)
    wscale = mn(wscale, ust * APHI16)
    wscale = mx(wscale, ust / APHI5)

    # surface variables for pbl height estimation under unstable conditions
    unst = sfcflg & (sflux > 0.0)
    gamfac = BFAC / rhox / wscale
    hgamt = mn(gamfac * hfx / cp, GAMCRT)
    hgamq = mn(gamfac * qfx, GAMCRQ)
    vpert = (hgamt + ep1 * thx[0] * hgamq) / BFAC * AFAC
    hsafe = where(hpbl > 0.0, hpbl, 1.0)
    thermal = where(
        unst, thermal + mx(vpert, 0.0) * mn(za[0] / (SFCFRAC * hsafe), 1.0), thermal
    )
    hgamt = where(unst, mx(hgamt, 0.0), 0.0)
    hgamq = where(unst, mx(hgamq, 0.0), 0.0)
    brint = -15.9 * ust * ust / wspd * wstar3 / pow(wscale, 4.0)
    hgamu = where(unst, brint * ux[0], 0.0)
    hgamv = where(unst, brint * vx[0], 0.0)
    pblflg = pblflg & unst

    # enhance the pbl height by considering the thermal
    k2, bu2, bd2 = _search(bk, thvx, ux, vx, za, thermal, BRCR_UB, br, klpbl)
    h2, k2 = _interp(bk, k2, bu2, bd2, BRCR_UB, za, zq2)
    hpbl = where(pblflg, h2, hpbl)
    kpbl = where(pblflg, k2, kpbl)
    pblflg = pblflg & (k2 > 1)

    # stable boundary layer
    sbl = (~sfcflg) & (hpbl < zq2)
    water = (xland - 1.5) >= 0.0
    wspd10 = sqrt(u10 * u10 + v10 * v10)
    ross = wspd10 / (CORI * znt)
    rpos = ross > 0.0
    brcr_sbro = where(
        rpos, mn(0.16 * pow(where(rpos, 1.0e-7 * ross, 1.0), -0.18), 0.3), 0.3
    )
    brcr = where(sbl, where(water, brcr_sbro, BRCR_SB), BRCR_UB)
    k3, bu3, bd3 = _search(bk, thvx, ux, vx, za, thermal, brcr, br, klpbl)
    h3, k3 = _interp(bk, k3, bu3, bd3, brcr, za, zq2)
    hpbl = where(sbl, h3, hpbl)
    kpbl = where(sbl, k3, kpbl)
    pblflg = pblflg & ~(sbl & (k3 <= 1))

    # estimate the entrainment parameters (at k = kpbl-1)
    hp = where(pblflg, hpbl, 1.0)
    kb = xp.clip(kpbl - 2, 0, None)  # 0-based index of kpbl-1
    kt = kb + 1  # of kpbl

    def jump(a):
        return _take(xp, a, kt) - _take(xp, a, kb)

    wm3 = wstar3 + 5.0 * ust3
    wm2 = pow(wm3, H2)
    bfxpbl = -0.15 * thvx[0] / g * wm3 / hp
    dthvx = mx(jump(thvx), TMIN)
    we = mx(bfxpbl / dthvx, -sqrt(wm2))
    we = where(pblflg, we, 0.0)
    wm2 = where(pblflg, wm2, 0.0)
    prpbl = where(pblflg, zero + 1.0, zero)
    dthx = mx(jump(thx), TMIN)
    dqx = mn(jump(qvx), 0.0)
    hfxpbl = we * dthx
    qfxpbl = we * dqx

    def momflx(du):
        return where(
            du > TMIN,
            mx(prpbl * we * du, -ust * ust),
            where(du < -TMIN, mn(prpbl * we * du, ust * ust), 0.0),
        )

    ufxpbl = where(pblflg, momflx(jump(ux)), 0.0)
    vfxpbl = where(pblflg, momflx(jump(vx)), 0.0)
    delb = govrth * D3 * hp
    delta = where(pblflg, mn(D1 * hpbl + D2 * wm2 / delb, 100.0), zero)

    kk = xp.reshape(
        xp.arange(1, nk + 1, dtype=xp.int64, device=_dev(br)),
        (nk,) + (1,) * br.ndim,
    )
    inpbl = kk >= kpbl  # k >= kpbl (Fortran k)
    below = kk < kpbl
    dsafe = where(pblflg, delta, 1.0)
    x = (zq[1:] - hpbl) / dsafe
    entfac = where(pblflg & inpbl, x * x, 1.0e30)

    # diffusion coefficients below pbl
    hl = where(below, hpbl - zl1, 1.0)
    zfac = mn(mx(1.0 - (zq[1:] - zl1) / hl, ZFMIN), 1.0)
    zfacent = pow(1.0 - zfac, 3.0)
    wscalek_u = pow(ust3 + PHIFAC * karman * wstar3 * (1.0 - zfac), H1)
    prfac2_u = 15.9 * wstar3 / ust3 / (1.0 + 4.0 * karman * wstar3 / ust3)
    m = mx(zq[1:] - SFCFRAC * hpbl, 0.0)
    hb = where(kpbl > 1, hpbl, 1.0)
    prnumfac_u = -3.0 * (m * m) / (hb * hb)
    phim8z = 1.0 + APHI5 * zol1 * zq[1:] / zl1
    wscalek_s = mx(ust / phim8z, 0.001)
    wscalek = where(sfcflg, wscalek_u, wscalek_s)
    prfac = where(sfcflg, zero + conpr, zero)
    prfac2 = where(sfcflg, prfac2_u, 0.0)
    prnumfac = where(sfcflg, prnumfac_u, 0.0)
    prnum0 = phih / phim + prfac
    prnum0 = mx(mn(prnum0, PRMAX), PRMIN)
    xkzm_b = wscalek * karman * zq[1:] * (zfac * zfac)
    prnum = 1.0 + (prnum0 - 1.0) * exp(prnumfac)
    xkzq_b = xkzm_b / prnum
    prnum0 = prnum0 / (1.0 + prfac2 * karman * SFCFRAC)
    prnum = 1.0 + (prnum0 - 1.0) * exp(prnumfac)
    xkzh_b = xkzm_b / prnum
    xkzm_b = mn(xkzm_b + XKZMINM, XKZMAX)
    xkzh_b = mn(xkzh_b + XKZMINH, XKZMAX)
    xkzq_b = mn(xkzq_b + XKZMINH, XKZMAX)

    # diffusion coefficients over pbl (free atmosphere), k = 1..nk-1
    du = ux[1:] - ux[:-1]
    dv = vx[1:] - vx[:-1]
    dzu = dza[1:]
    ss = (du * du + dv * dv) / (dzu * dzu) + 1.0e-9
    govrthv = g / (0.5 * (thvx[1:] + thvx[:-1]))
    ri = govrthv * (thvx[1:] - thvx[:-1]) / (ss * dzu)
    qci = qcx + qix
    cld = (qci[:-1] > 0.01e-3) & (qci[1:] > 0.01e-3)
    qmean = 0.5 * (qvx[:-1] + qvx[1:])
    tmean = 0.5 * (tx[:-1] + tx[1:])
    alph = xlv * qmean / rd / tmean
    chi = xlv * xlv * qmean / cp / rv / tmean / tmean
    ri = where(
        cld,
        (1.0 + alph) * (ri - g * g / ss / tmean / cp * ((chi - alph) / (1.0 + chi))),
        ri,
    )
    zk = karman * zq[1:-1]
    rlamdz = mn(mx(0.1 * dzu, RLAM), 300.0)
    rlamdz = mn(dzu, rlamdz)
    rl = zk * rlamdz / (rlamdz + zk)
    rl2 = rl * rl
    dk = rl2 * sqrt(ss)
    neg = ri < 0.0
    rin = mx(ri, RIMIN)
    sri = sqrt(where(neg, -rin, 1.0))
    xkzm_u = dk * (1 + 8.0 * (-rin) / (1 + 1.746 * sri))
    xkzh_u = dk * (1 + 8.0 * (-rin) / (1 + 1.286 * sri))
    r = 1 + 5.0 * ri
    xkzh_s = dk / (r * r)
    xkzm_s = xkzh_s * mn(1.0 + 2.1 * ri, PRMAX)
    xkzml = mn(where(neg, xkzm_u, xkzm_s) + XKZMINM, XKZMAX)
    xkzhl = mn(where(neg, xkzh_u, xkzh_s) + XKZMINH, XKZMAX)

    # the coefficients at k = 1..nk-1 (the k = nk ones are never used)
    b1, i1 = below[:-1], inpbl[:-1]
    zero1 = xp.zeros_like(xkzml)
    xkzm = where(i1, xkzml, where(b1, xkzm_b[:-1], zero1))
    xkzh = where(i1, xkzhl, where(b1, xkzh_b[:-1], zero1))
    xkzq0 = where(b1, xkzq_b[:-1], zero1)
    xkzhl = where(i1, xkzhl, zero1)
    xkzml = where(i1, xkzml, zero1)
    zfe = zfacent[:-1]

    dtodsd = 1.0 * dt2 / dl[:-1]
    dtodsu = 1.0 * dt2 / dl[1:]
    dsig = p2d[:-1] - p2d[1:]
    rdz = 1.0 / dzu
    brA = pblflg & b1  # pblflg .and. k < kpbl
    brB = pblflg & i1 & (entfac[:-1] < 4.6)
    dzak = _take(xp, dza, xp.clip(kpbl - 1, 0, None))

    def entk(xl):
        v = -we * dzak * exp(-entfac[:-1])
        v = sqrt(where(brB, v * xl, 1.0))
        return mn(mx(v, XKZMINH), XKZMAX)

    lastk = xp.zeros_like(brA[:1])  # level nk: no k-th iteration
    brAn = _cat(xp, brA, lastk)

    def assemble(kz, fsrc, f1b, dsdz, ad1=1.0):
        """f1, au, al, ad of the matrix loop k = 1..nk-1: f1(1) = f1b and,
        where brA (pblflg, k < kpbl), f1(k) += dtodsd dsdz(k) and f1(k+1) =
        fsrc(k+1) - dtodsu dsdz(k), else f1(k+1) = fsrc(k+1); ad(1) = ad1."""
        tem1 = dsig * kz * rdz
        dsdz2 = tem1 * rdz
        au = -dtodsd * dsdz2 / 1.0
        al = -dtodsu * dsdz2 / 1.0
        z1 = xp.zeros_like(f1b)[None]
        base = _cat(xp, f1b[None], where(brA, fsrc[1:] - dtodsu * dsdz, fsrc[1:]))
        f1 = where(brAn, base + _cat(xp, dtodsd * dsdz, z1), base)
        ad = _cat(xp, (ad1 + z1), 1.0 - al) - _cat(xp, au, z1)
        return f1, _cat(xp, au, z1), _cat(xp, al, z1), ad

    # ---- heat
    xkzhA = where(brA, xkzh, 1.0)
    hpA = where(brA, hpbl, 1.0)
    dsdzt = dsig * xkzh * rdz * (-hgamt / hpA - hfxpbl * zfe / xkzhA)
    xkzh = where(brB, entk(xkzhl), xkzh)
    f1b = thx[0] - 300.0 + (1.0 - 0.0) * hfx / cont / dl[0] * dt2
    f1, au, al, ad = assemble(xkzh, thx - 300.0, f1b, dsdzt)
    (f1,) = tridin_ysu(bk, al, ad, au, (f1,))
    ttnp = (f1 - thx + 300.0) * rdt
    z1 = xp.zeros_like(br)[None]
    exch_hx = _cat(xp, z1, xkzh)

    # ---- water vapor, cloud water, cloud ice
    xkzq = where(i1, xkzh, xkzq0)
    xkzqA = where(brA, xkzq, 1.0)
    dsdzq = dsig * xkzq * rdz * (-qfxpbl * zfe / xkzqA)
    xkzq = where(brB, entk(xkzhl), xkzq)
    f1b = qvx[0] + (1.0 - 0.0) * qfx * g / dl[0] * dt2
    f1, au, al, ad = assemble(xkzq, qvx, f1b, dsdzq)
    fq, fc, fi = tridin_ysu(bk, al, ad, au, (f1, qcx, qix))
    qvtnp = (fq - qvx) * rdt
    qctnp = (fc - qcx) * rdt
    qitnp = (fi - qix) * rdt

    # ---- momentum
    rr = wspd1 / wspd
    fric1 = ust * ust / wspd1 * rhox * g / dl[0] * dt2 * (rr * rr)
    xkzmA = where(brA, xkzm, 1.0)
    tm = dsig * xkzm * rdz
    dsdzu = tm * (-hgamu / hpA - ufxpbl * zfe / xkzmA)
    dsdzv = tm * (-hgamv / hpA - vfxpbl * zfe / xkzmA)
    xm = sqrt(where(brB, prpbl * xkzh * xkzml, 1.0))
    xkzm = where(brB, mn(mx(xm, XKZMINM), XKZMAX), xkzm)
    f1, au, al, ad = assemble(xkzm, ux, ux[0], dsdzu, 1.0 + fric1)
    f2, _, _, _ = assemble(xkzm, vx, vx[0], dsdzv)
    f1, f2 = tridin_ysu(bk, al, ad, au, (f1, f2))
    utnp = (f1 - ux) * rdt
    vtnp = (f2 - vx) * rdt
    exch_mx = _cat(xp, z1, xkzm)

    return {
        "utnp": utnp,
        "vtnp": vtnp,
        "ttnp": ttnp,
        "qvtnp": qvtnp,
        "qctnp": qctnp,
        "qitnp": qitnp,
        "exch_hx": exch_hx,
        "exch_mx": exch_mx,
        "hpbl": hpbl,
        "kpbl1d": kpbl,
        "wstar": wstar,
        "delta": delta,
        "u10": u10,
        "v10": v10,
    }


# ---------------------------------------------------------------------------
# CM1 glue (src/turb.F sfc_and_turb)

WRF_INPUTS = True  # the scheme reads ctx.w (turb.F get_vars)
DUMP_KEYS = ("hpbl",)


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def check(cfg):
    """What CM1's YSU call needs that is not ported."""
    if int(_opt(cfg, "sfcmodel", 0)) not in (2, 3):
        raise NotImplementedError("ipbl=1: only sfcmodel=2, 3 are ported")
    if int(_opt(cfg, "cm1setup", 1)) != 2:
        raise NotImplementedError("ipbl=1: only cm1setup=2 is ported")


def init_state(ctx):
    """The YSU diagnostics carried in the state (zero until the first call):
    kpbl2d, wstar, delta (2-D) and xkzh, xkzm (3-D)."""
    bk, g = ctx.bk, ctx.g
    out = {k: bk.zeros(g.s_shape[1:]) for k in ("kpbl2d", "wstar", "delta")}
    out.update({k: bk.zeros(g.s_shape) for k in ("xkzh", "xkzm")})
    return out


def ysudiss(bk, *, dz, rmh, mf, upten, vpten, dum1, dum2, dissten):
    """turb.F ``ysudiss`` (idiss=1): the column stresses t13 = -sum(upten dz
    rmh), t23 (from the top down, zero at nk+1) times the shear of the
    scalar-point winds dum1, dum2, added to dissten on w levels k = 2..nk.
    Arrays are level-stacked ``[k, ...]`` with Fortran k at index k (CM1's
    kb = 0), interior columns only; returns dissten."""
    xp = bk.xp
    nk = upten.shape[0] - 2
    rdz = 1.0 / dz
    ks = slice(2, nk + 1)
    tem1 = dz * rmh[ks]
    tem2 = rdz * mf[ks]
    z0 = xp.zeros_like(upten[0])

    def f(c, x):
        u, v, t = x
        n = (c[0] - u * t, c[1] - v * t)
        return n, n

    _, (t13, t23) = bk.scan(f, (z0, z0), (upten[ks], vpten[ks], tem1), reverse=True)
    du = dum1[ks] - dum1[1:nk]
    dv = dum2[ks] - dum2[1:nk]
    add = t13 * du * tem2 + t23 * dv * tem2
    return bk.set(bk.copy(dissten), ks, dissten[ks] + add)


def pbl(ctx):
    """turb.F ``doysu`` (the ysu_j_loop calling ysu_wrapper on every column)
    and ``ysudiss`` (idiss=1).  Returns the PBL tendencies thpten, qvpten,
    qcpten, qipten, upten, vpten (interior filled; plain tendencies, which
    solve1 adds as they are), hpbl, kpbl2d, wstar, delta, xkzh/xkzm (k=1..nk,
    exch_hx/exch_mx) and dissten (idiss=1)."""
    bk, g, cfg, s, w = ctx.bk, ctx.g, ctx.cfg, ctx.s, ctx.w
    xp = bk.xp
    ni, nj, nk, n, z = g.ni, g.nj, g.nk, g.ngxy, g.ngz
    K, J, I = slice(z, z + nk), slice(n, n + nj), slice(n, n + ni)
    Kw = slice(z, z + nk + 1)

    def c3(a):
        return a[K, J, I]

    def c2(a):
        return a[J, I]

    r = bl_ysu_run(
        bk,
        ux=c3(w["u"]),
        vx=c3(w["v"]),
        tx=c3(w["t"]),
        qvx=c3(w["qv"]),
        qcx=c3(w["qc"]),
        qix=c3(w["qi"]),
        p2d=c3(ctx.prs),
        p2di=w["p3di"][Kw, J, I],
        pi2d=c3(w["pi"]),
        dz8w2d=c3(w["dz8w"]),
        rthraten=c3(w["rthraten"]),
        psfcpa=c2(s["psfc"]),
        znt=c2(s["znt"]),  # zntmp = znt for sfcmodel 2, 3
        ust=c2(s["ust"]),
        psim=c2(s["fm"]),
        psih=c2(s["fh"]),
        xland=c2(s["xland"]),
        hfx=c2(s["hfx"]),
        qfx=c2(s["qfx"]),
        wspd=c2(s["wspd"]),
        br=c2(s["br"]),
        u10=c2(s["u10"]),
        v10=c2(s["v10"]),
        dt=ctx.dt,
        ysu_topdown_pblmix=False,
    )

    def put3(v):
        # a fresh array each time: bk.set updates numpy/torch arrays in place
        return bk.set(bk.zeros(g.s_shape), (K, J, I), v)

    def put2(name, v):
        old = s.get(name)
        base = bk.copy(old) if old is not None else bk.zeros(g.s_shape[1:])
        return bk.set(base, (J, I), xp.astype(v, base.dtype))

    out = {
        "upten": put3(r["utnp"]),
        "vpten": put3(r["vtnp"]),
        "thpten": put3(r["ttnp"]),
        "qvpten": put3(r["qvtnp"]),
        "qcpten": put3(r["qctnp"]),
        "qipten": put3(r["qitnp"]),
        "xkzh": put3(r["exch_hx"]),
        "xkzm": put3(r["exch_mx"]),
        "hpbl": put2("hpbl", r["hpbl"]),
        "kpbl2d": put2("kpbl2d", r["kpbl1d"]),
        "wstar": put2("wstar", r["wstar"]),
        "delta": put2("delta", r["delta"]),
    }
    if int(_opt(cfg, "idiss", 0)) == 1:
        dt3 = s["dissten"]
        KK = slice(z - 1, z + nk + 1)  # Fortran k = 0..nk+1
        col = ysudiss(
            bk,
            dz=float(cfg.dz),
            rmh=ctx.b["rmh"][KK, J, I],
            mf=ctx.b["mf"][KK, J, I],
            upten=out["upten"][KK, J, I],
            vpten=out["vpten"][KK, J, I],
            dum1=w["u"][KK, J, I],
            dum2=w["v"][KK, J, I],
            dissten=dt3[KK, J, I],
        )
        out["dissten"] = bk.set(bk.copy(dt3), (KK, J, I), col)
    return out
