"""Base state (port of src/base.F) and the grid-metric parts of src/param.F.

Scope: no terrain, no grid stretching, uniform dx/dy/dz;
``isnd = 5`` (Weisman-Klemp analytic sounding) and ``iwnd = 0/1/2``; dry
(``imoist = 0``) or moist (``imoist = 1``); ``imove = 0/1``.

The base state is horizontally homogeneous, so each routine computes one
vertical column (Python index == Fortran ``k`` because ``ngz = 1``) with the
same operation order as CM1, then broadcasts it to the 3-D halo'd arrays CM1
uses, ``a[k, j, i]``. Lateral ``bcs``/``bcu``/``bcv`` calls are therefore exact
no-ops; ``extrapbcs`` (k = 0 and nk+1) is applied to the column.

All literal constants are turned into ``bk.dtype`` scalars before arithmetic so
that a float32 backend reproduces CM1's single-precision build closely.
"""

from __future__ import annotations

from . import constants as C
from .grid import Grid

__all__ = ["extrap_coeffs", "param_grid", "rslf", "extrapbcs", "base"]


# ---------------------------------------------------------------------------
# param.F pieces


def extrap_coeffs(sigma, sigmaf, nk):
    """2nd-order extrapolation coefficients (Fornberg 1988), as in param.F.

    ``sigma``/``sigmaf`` are Python sequences indexed by Fortran ``k`` (index 0 ->
    k=0). Computed in double precision like CM1. Returns a dict with
    cgs1..3, cgt1..3, dgs1..3, dgt1..3, wbe1..3, wte1..3.
    """
    bign, bigm = 3, 1
    out = {}
    for nn in (1, 2, 3, 4):
        if nn == 1:
            d = sigmaf[2] - sigmaf[1]
            x0 = sigmaf[1] / d
            alpha = [sigma[1] / d, sigma[2] / d, sigma[3] / d, sigma[4] / d]
        elif nn == 2:
            d = sigmaf[nk + 1] - sigmaf[nk]
            x0 = sigmaf[nk + 1] / d
            alpha = [
                sigma[nk] / d,
                sigma[nk - 1] / d,
                sigma[nk - 2] / d,
                sigma[nk - 3] / d,
            ]
        elif nn == 3:
            d = sigmaf[2] - sigmaf[1]
            x0 = sigmaf[1] / d
            alpha = [sigmaf[2] / d, sigmaf[3] / d, sigmaf[4] / d, sigmaf[5] / d]
        else:
            d = sigmaf[nk + 1] - sigmaf[nk]
            x0 = sigmaf[nk + 1] / d
            alpha = [
                sigmaf[nk] / d,
                sigmaf[nk - 1] / d,
                sigmaf[nk - 2] / d,
                sigmaf[nk - 3] / d,
            ]
        # delta(n, m, nu), m from -1
        delta = {
            (n, m, nu): 0.0
            for n in range(bign + 1)
            for m in range(-1, bigm + 1)
            for nu in range(bign + 1)
        }
        delta[(0, 0, 0)] = 1.0
        b1 = 1.0
        for n in range(1, bign + 1):
            b2 = 1.0
            for nu in range(n):
                b3 = alpha[n] - alpha[nu]
                b2 = b2 * b3
                if n <= bigm:
                    delta[(n - 1, n, nu)] = 0.0
                for m in range(min(n, bigm) + 1):
                    delta[(n, m, nu)] = (
                        (alpha[n] - x0) * delta[(n - 1, m, nu)]
                        - m * delta[(n - 1, m - 1, nu)]
                    ) / b3
            for m in range(min(n, bigm) + 1):
                delta[(n, m, n)] = (b1 / b2) * (
                    m * delta[(n - 1, m - 1, n - 1)]
                    - (alpha[n - 1] - x0) * delta[(n - 1, m, n - 1)]
                )
            b1 = b2
        c = [delta[(2, 0, q)] for q in range(3)]
        dd = [delta[(2, 1, q)] for q in range(3)]
        name = {
            1: ("cgs", "dgs"),
            2: ("cgt", "dgt"),
            3: ("wbe", None),
            4: ("wte", None),
        }[nn]
        for q in range(3):
            out[f"{name[0]}{q + 1}"] = c[q]
            if name[1]:
                out[f"{name[1]}{q + 1}"] = dd[q]
    return out


def param_grid(cfg, bk, g: Grid | None = None) -> dict:
    """Grid/metric arrays of param.F for a uniform grid without terrain.

    Returns a dict keyed by Fortran names:
      1-D horizontal (Python index = Fortran index + ngxy - 1):
        xh, rxh, uh, ruh (len ni+2*ngxy), xf, rxf, uf, ruf (len ni+2*ngxy+1),
        arh1, arh2, arf1, arf2, yh, vh, rvh (len nj+2*ngxy), yf, vf, rvf
      1-D vertical (Python index = Fortran k): sigma, rds (nk+2), sigmaf, rdsf (nk+3)
      3-D (like CM1, ``[k, j, i]``): zh, mh, rmh, c1, c2, tauh, taus (scalar
        points), zf, mf, rmf, tauf (w points); 2-D: zs
      Python-float scalars (module ``input`` in CM1): cgs1..3, cgt1..3, dgs*, dgt*,
        wbe*, wte*, minx, maxx, miny, maxy, centerx, centery, maxz
    """
    if cfg.terrain_flag or cfg.stretch_x or cfg.stretch_y or cfg.stretch_z:
        raise NotImplementedError("only uniform grids without terrain are supported")
    xp = bk.xp
    if g is None:
        g = Grid(cfg.ni, cfg.nj, cfg.nk, float(cfg.dx), float(cfg.dy), float(cfg.dz))
    ni, nj, nk, ng = g.ni, g.nj, g.nk, g.ngxy
    dx, dy, dz = float(cfg.dx), float(cfg.dy), float(cfg.dz)
    smeps = float(cfg.smeps)

    # xfdp/yfdp are double precision in CM1
    def fdp(n, d):
        if cfg.iorigin == 2:
            return [d * (i - 1) - 0.5 * d * n for i in range(1 - ng, n + ng + 2)]
        return [d * (i - 1) for i in range(1 - ng, n + ng + 2)]

    xfdp, yfdp = fdp(ni, dx), fdp(nj, dy)
    xf = bk.asarray(xfdp)
    xh = bk.asarray([0.5 * (xfdp[i] + xfdp[i + 1]) for i in range(ni + 2 * ng)])
    yf = bk.asarray(yfdp)
    yh = bk.asarray([0.5 * (yfdp[j] + yfdp[j + 1]) for j in range(nj + 2 * ng)])
    one_h = bk.full((ni + 2 * ng,), 1.0)
    one_f = bk.full((ni + 2 * ng + 1,), 1.0)
    out = dict(
        xh=xh,
        xf=xf,
        yh=yh,
        yf=yf,
        uh=one_h,
        ruh=one_h,
        uf=one_f,
        ruf=one_f,
        arh1=one_h,
        arh2=one_h,
        arf1=one_f,
        arf2=one_f,
        vh=bk.full((nj + 2 * ng,), 1.0),
        rvh=bk.full((nj + 2 * ng,), 1.0),
        vf=bk.full((nj + 2 * ng + 1,), 1.0),
        rvf=bk.full((nj + 2 * ng + 1,), 1.0),
        rxh=1.0 / (smeps + xh),
        rxf=1.0 / (smeps + xf),
    )
    minx, maxx = xfdp[ng], xfdp[ng + ni]
    miny, maxy = yfdp[ng], yfdp[ng + nj]

    # vertical (Python index == Fortran k)
    zf1 = [dz * (k - 1) for k in range(0, nk + 3)]
    zh1 = [0.5 * (zf1[k] + zf1[k + 1]) for k in range(0, nk + 2)]
    shp_s, shp_w = g.s_shape, g.w_shape

    def col3(vals, shp):
        return bk.asarray(vals)[:, None, None] + bk.zeros(shp)

    zf = col3(zf1, shp_w)
    zh = col3(zh1, shp_s)
    mh = bk.full(shp_s, 1.0)
    mf = bk.full(shp_w, 1.0)
    # cc2(k) = (zf(k)-zh(k-1))/(zh(k)-zh(k-1)) for k=1..nk+1, cc1 = 1-cc2 ; 0 at k=0
    zfa, zha = bk.asarray(zf1), bk.asarray(zh1)
    c2col = (zfa[1 : nk + 2] - zha[0 : nk + 1]) / (zha[1 : nk + 2] - zha[0 : nk + 1])
    c1col = 1.0 - c2col
    z0 = bk.zeros((1,))
    c1 = xp.concat([z0, c1col])[:, None, None] + bk.zeros(shp_s)
    c2 = xp.concat([z0, c2col])[:, None, None] + bk.zeros(shp_s)

    sigmaf = [float(z) for z in zf1]
    sigma = [0.5 * (sigmaf[k] + sigmaf[k + 1]) for k in range(nk + 2)]
    maxz = sigmaf[nk + 1]

    # Rayleigh damping coefficients (irdamp >= 1)
    zd = float(cfg.zd)
    pi = bk.asarray(C.pi)
    if cfg.irdamp >= 1 and zd < maxz:
        ztop = zfa[nk + 1]
        th_ = 0.5 * (1.0 - xp.cos(pi * (zha - zd) / (ztop - zd)))
        th_ = xp.where(zha > zd, th_, bk.asarray(0.0))
        th_ = xp.concat([bk.zeros((1,)), th_[1 : nk + 1], bk.zeros((1,))])
        tf_ = 0.5 * (1.0 - xp.cos(pi * (zfa - zd) / (ztop - zd)))
        tf_ = xp.where(zfa > zd, tf_, bk.asarray(0.0))
        tf_ = xp.concat([bk.zeros((1,)), tf_[1 : nk + 2], bk.zeros((1,))])
        tauh = th_[:, None, None] + bk.zeros(shp_s)
        tauf = tf_[:, None, None] + bk.zeros(shp_w)
    else:
        tauh, tauf = bk.zeros(shp_s), bk.zeros(shp_w)

    out.update(
        sigma=bk.asarray(sigma),
        sigmaf=bk.asarray(sigmaf),
        rds=bk.full((nk + 2,), 1.0),
        rdsf=bk.full((nk + 3,), 1.0),
        zh=zh,
        mh=mh,
        rmh=1.0 / mh,
        zf=zf,
        mf=mf,
        rmf=1.0 / mf,
        c1=c1,
        c2=c2,
        tauh=tauh,
        taus=bk.copy(tauh),
        tauf=tauf,
        zs=bk.zeros(shp_s[1:]),
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        centerx=minx + 0.5 * (maxx - minx),
        centery=miny + 0.5 * (maxy - miny),
        maxz=maxz,
    )
    out.update(extrap_coeffs(sigma, sigmaf, nk))
    return out


# ---------------------------------------------------------------------------
# helpers


def rslf(bk, p, t, ptype: int = 1):
    """Liquid saturation mixing ratio (cm1libs.F RSLF)."""
    xp = bk.xp
    S = bk.asarray
    if ptype == 4:
        return S(380.00) * xp.exp(S(17.2693882) - S(4097.8531) / (t - S(35.86))) / p
    esl = S(611.2) * xp.exp(S(17.67) * (t - S(273.15)) / (t - S(29.65)))
    esl = xp.minimum(esl, p * S(0.5))
    eps = S(C.rd) / S(C.rv)
    return eps * esl / (p - esl)


def extrapbcs(bk, s, cg, nk):
    """bc.F extrapbcs on a column (k axis first): fill k=0 and k=nk+1."""
    s = bk.set(s, 0, cg["cgs1"] * s[1] + cg["cgs2"] * s[2] + cg["cgs3"] * s[3])
    s = bk.set(
        s, nk + 1, cg["cgt1"] * s[nk] + cg["cgt2"] * s[nk - 1] + cg["cgt3"] * s[nk - 2]
    )
    return s


# ---------------------------------------------------------------------------
# base.F


def base(bk, cfg, *, zh, mh, rmh, c1, c2, zf, mf, cg: dict) -> dict:
    """Port of ``base`` (src/base.F): sounding, hydrostatic base state, winds.

    ``cg`` holds the extrapolation coefficients cgs1..3 / cgt1..3 (module
    ``input`` scalars in CM1; e.g. the dict returned by ``param_grid``).

    Returns arrays keyed by Fortran name: pi0, prs0, rho0, thv0, th0, rth0, qv0,
    qc0, qi0, rr0, rf0, rrf0, t0, rh0 (scalar points), u0, v0 (staggered),
    rho0s, pi0s, prs0s, rth0s (2-D), plus Python-float scalars psurf, tsurf,
    qsurf, thsurf and the 0-d array pmin.
    """
    xp = bk.xp
    S = bk.asarray
    nk = zh.shape[0] - 2
    shp_s = zh.shape
    shp_u = (shp_s[0], shp_s[1], shp_s[2] + 1)
    shp_v = (shp_s[0], shp_s[1] + 1, shp_s[2])
    imoist = cfg.imoist
    ptype = cfg.ptype
    cg = {k: S(cg[k]) for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}

    g, cp, rd, rv, p00 = S(C.g), S(C.cp), S(C.rd), S(C.rv), S(C.p00)
    reps = rv / rd
    cpdrd = cp / rd
    rovcp = rd / cp
    rp00 = S(1.0) / p00
    one, zero = S(1.0), S(0.0)

    zc = zh[:, 0, 0]  # column, Python index == Fortran k

    if cfg.isnd != 5:
        raise NotImplementedError(f"isnd={cfg.isnd} not ported (only isnd=5)")

    # ---------------- isnd = 5: Weisman-Klemp analytic sounding
    z_trop, th_trop, t_trop = S(12000.0), S(343.0), S(213.0)
    th_sfc, prs_sfc, qv_pbl = S(300.0), S(100000.0), S(0.014)

    pi_sfc = (prs_sfc / p00) ** (rd / cp)
    qv_sfc = rslf(bk, prs_sfc, th_sfc * pi_sfc, ptype)
    thv_sfc = th_sfc * (one + qv_sfc * reps) / (one + qv_sfc)

    zi = zc[1 : nk + 1]
    ratio = (zi / z_trop) ** S(1.25)
    below = zi < z_trop
    th_in = xp.where(
        below,
        th_sfc + (th_trop - th_sfc) * ratio,
        th_trop * xp.exp((g / (t_trop * cp)) * (zi - z_trop)),
    )
    if imoist == 1:
        rh_in = xp.where(below, one - S(0.75) * ratio, S(0.25))
    else:
        rh_in = bk.zeros((nk,))
    th0 = xp.concat([th_in[:1], th_in, th_in[-1:]])
    z1 = bk.zeros((1,))
    rh0 = xp.concat([z1, rh_in, z1])
    qv0 = bk.zeros((nk + 2,))
    pi0 = bk.zeros((nk + 2,))
    prs0 = bk.zeros((nk + 2,))
    thv0 = None

    half = S(0.5)
    for _n in range(20):
        thv0 = th0 * (one + reps * qv0) / (one + qv0)
        pis = [pi_sfc - g * zc[1] / (cp * half * (thv_sfc + thv0[1]))]
        for k in range(2, nk + 1):
            pis.append(
                pis[-1]
                - g * (zc[k] - zc[k - 1]) / (cp * half * (thv0[k] + thv0[k - 1]))
            )
        pi_in = xp.stack(pis)
        pi0 = xp.concat([pi0[:1], pi_in, pi0[-1:]])
        prs_in = p00 * (pi_in ** (cp / rd))
        prs0 = xp.concat([prs0[:1], prs_in, prs0[-1:]])
        qv_in = rh_in * rslf(bk, prs_in, th_in * pi_in, ptype)
        qv_in = xp.where(qv_in > qv_pbl, qv_pbl, qv_in)
        qv0 = xp.concat([qv0[:1], qv_in, qv0[-1:]])

    rh_in = qv0[1 : nk + 1] / rslf(
        bk, prs0[1 : nk + 1], th0[1 : nk + 1] * pi0[1 : nk + 1], ptype
    )
    rh0 = xp.concat([rh0[:1], rh_in, rh0[-1:]])

    psurf = prs_sfc
    tsurf = th_sfc * ((psurf * rp00) ** rovcp)
    if imoist == 1:
        q1 = cg["cgs1"] * qv0[1] + cg["cgs2"] * qv0[2] + cg["cgs3"] * qv0[3]
        qsurf = xp.where(qv0[1] < qv_pbl, q1, qv_pbl)
    else:
        qsurf = zero

    # ---------------- ghost cells (bcs is a no-op on homogeneous fields)
    qc0 = bk.zeros((nk + 2,))
    qi0 = bk.zeros((nk + 2,))
    pi0 = extrapbcs(bk, pi0, cg, nk)
    prs0 = extrapbcs(bk, prs0, cg, nk)
    th0 = extrapbcs(bk, th0, cg, nk)
    qv0 = extrapbcs(bk, qv0, cg, nk)
    rh0 = extrapbcs(bk, rh0, cg, nk)

    if imoist == 1:
        thv0 = th0 * (one + reps * qv0) / (one + qv0 + qc0 + qi0)
    else:
        qv0 = bk.zeros((nk + 2,))
        rh0 = bk.zeros((nk + 2,))
        thv0 = th0

    prs0 = p00 * (pi0**cpdrd)
    rho0 = prs0 / (rd * th0 * pi0 * (one + qv0 * reps))
    t0 = th0 * pi0

    if imoist == 1:
        qvp = xp.maximum(zero, qv0)
        thv0 = (
            (th0 + zero)
            * (one + reps * qvp)
            / (one + qvp + xp.maximum(zero, qc0 + qi0))
        )
    else:
        thv0 = th0

    rho0 = extrapbcs(bk, rho0, cg, nk)
    if cfg.psolver in (5, 6):
        rho0 = bk.full((nk + 2,), 1.0)

    rr0 = one / rho0
    c1c, c2c = c1[:, 0, 0], c2[:, 0, 0]
    rf_mid = c1c[2 : nk + 1] * rho0[1:nk] + c2c[2 : nk + 1] * rho0[2 : nk + 1]
    rf1 = cg["cgs1"] * rho0[1] + cg["cgs2"] * rho0[2] + cg["cgs3"] * rho0[3]
    rftop = (
        cg["cgt1"] * rho0[nk] + cg["cgt2"] * rho0[nk - 1] + cg["cgt3"] * rho0[nk - 2]
    )
    rf0 = xp.concat([xp.stack([rf1, rf1]), rf_mid, xp.stack([rftop])])
    rrf0 = one / rf0
    rho0s = rf1

    # ---------------- surface values (no terrain, psurf/tsurf set by isnd=5)
    tsm = float(cfg.tsmall)
    if not (float(psurf) >= tsm and float(tsurf) >= tsm and not cfg.terrain_flag):
        raise NotImplementedError(
            "surface-value branch for terrain / unset psurf not ported"
        )
    thsurf = tsurf / ((psurf * rp00) ** rovcp)
    prs0s = psurf
    pi0s = (psurf * rp00) ** rovcp
    rth0s = thsurf ** (-one)

    # ---------------- base-state wind (iwnd)
    zu = half * (zc + zc)  # 0.5*(zh(i-1)+zh(i)), homogeneous
    if cfg.iwnd in (0, None):
        u_in = bk.zeros((nk,))
        v_in = bk.zeros((nk,))
    elif cfg.iwnd == 1:
        udep1, udep2, uc1, uc2 = S(0.0), S(2500.0), S(0.0), S(10.0)
        z = zu[1 : nk + 1]
        u_in = xp.where(
            z < udep1,
            uc1,
            xp.where(
                (z > udep1) & (z < udep2),
                (uc2 - uc1) * (z - udep1) / (udep2 - udep1) + uc1,
                uc2,
            ),
        )
        v_in = bk.zeros((nk,))
    elif cfg.iwnd == 2:
        udep1, udep2, umax1, umax2 = S(2000.0), S(6000.0), S(7.0), S(31.0)
        pi180 = S(C.pi) / S(180.0)
        z = zu[1 : nk + 1]
        angle = S(90.0) * (z / udep1) * pi180
        u_in = xp.where(
            z <= udep1,
            umax1 - umax1 * xp.cos(angle),
            xp.where(
                z <= udep2,
                umax1 + (z - udep1) * (umax2 - umax1) / (udep2 - udep1),
                umax2,
            ),
        )
        vmax1 = umax1
        v_in = xp.where(z <= udep1, vmax1 * xp.sin(angle), vmax1)
    else:
        raise NotImplementedError(f"iwnd={cfg.iwnd} not ported")

    if cfg.imove == 1:
        u_in = u_in - S(float(cfg.umove))
        v_in = v_in - S(float(cfg.vmove))

    def ext_full(col_in):
        col = xp.concat([z1, col_in, z1])
        return extrapbcs(bk, col, cg, nk)

    u0c = ext_full(u_in)
    v0c = ext_full(v_in)

    rth0 = one / th0

    def to3(col, shp):
        return col[:, None, None] + bk.zeros(shp)

    def to2(x):
        return x + bk.zeros(shp_s[1:])

    out = dict(
        pi0=to3(pi0, shp_s),
        prs0=to3(prs0, shp_s),
        rho0=to3(rho0, shp_s),
        thv0=to3(thv0, shp_s),
        th0=to3(th0, shp_s),
        rth0=to3(rth0, shp_s),
        qv0=to3(qv0, shp_s),
        qc0=to3(qc0, shp_s),
        qi0=to3(qi0, shp_s),
        rr0=to3(rr0, shp_s),
        rf0=to3(rf0, shp_s),
        rrf0=to3(rrf0, shp_s),
        t0=to3(t0, shp_s),
        rh0=to3(rh0, shp_s),
        u0=to3(u0c, shp_u),
        v0=to3(v0c, shp_v),
        rho0s=to2(rho0s),
        pi0s=to2(pi0s),
        prs0s=to2(prs0s),
        rth0s=to2(rth0s),
        pmin=xp.min(prs0[1 : nk + 1]),
        psurf=float(psurf),
        tsurf=float(tsurf),
        qsurf=float(qsurf),
        thsurf=float(thsurf),
    )
    return out
