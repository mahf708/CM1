"""Base state (port of src/base.F) and the grid-metric parts of src/param.F.

Scope: uniform dx/dy/dz (terrain via init_terrain); every sounding
(``isnd`` 1-5, 7-15, 17-23) and wind profile (``iwnd`` 0-11) of base.F (see
:mod:`cm1x.base_snd`); dry (``imoist = 0``) or moist (``imoist = 1``);
``imove = 0/1``. The testcase large-scale forcing profiles raise.

Every column is computed with the same operation order as CM1 on the 3-D
halo'd arrays CM1 uses, ``a[k, j, i]`` (Python index == Fortran ``k`` because
``ngz = 1``); then the lateral ``bcs``/``bcu``/``bcv`` and ``extrapbcs``
(k = 0 and nk+1) fill the ghost cells as in CM1.

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
    if cfg.stretch_x or cfg.stretch_y or cfg.stretch_z:
        raise NotImplementedError("only uniform grids (no stretching) are supported")
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
    if cfg.get("terrain_flag", False):
        out.update(_terrain_metrics(cfg, bk, g, out))
    return out


def _terrain_metrics(cfg, bk, g: Grid, pg: dict) -> dict:
    """param.F with ``terrain_flag``: ``init_terrain`` (zs, zh, zf, metric
    terms) then the column-dependent mh, rmh, mf, rmf, c1, c2 (cc1/cc2) and
    Rayleigh-damping coefficients tauh, taus, tauf."""
    from . import bc as _bc
    from .init_terrain import init_terrain

    xp = bk.xp
    S = bk.asarray
    nk = g.nk
    dz = S(float(cfg.dz))
    t = init_terrain(
        bk,
        g,
        cfg,
        **{k: pg[k] for k in ("xh", "uh", "xf", "uf", "yh", "vh", "yf", "vf")},
        **{k: pg[k] for k in ("rds", "sigma", "rdsf", "sigmaf")},
        maxx=pg["maxx"],
        maxy=pg["maxy"],
        maxz=pg["maxz"],
    )
    zh, zf = t["zh"], t["zf"]
    one = S(1.0)
    mh = dz / (zf[1:] - zf[:-1])  # k = kb..ke
    mf_in = dz / (zh[1:] - zh[:-1])  # k = 1..nk+1
    mf = xp.concat([mf_in[:1], mf_in, mf_in[-1:]])
    cc2 = (zf[1 : nk + 2] - zh[0 : nk + 1]) / (zh[1 : nk + 2] - zh[0 : nk + 1])
    z0 = bk.zeros((1,) + g.s_shape[1:])
    cc2 = xp.concat([z0, cc2])
    cc2 = _bc.bcs(bk, g, cc2, cfg)
    cc1 = xp.concat([z0, one - cc2[1:]])

    zd = float(cfg.zd)
    if cfg.irdamp >= 1 and zd < pg["maxz"]:
        pi, zds, half = S(C.pi), S(zd), S(0.5)
        ztop = zf[nk + 1]
        th_ = half * (one - xp.cos(pi * (zh - zds) / (ztop - zds)))
        th_ = xp.where(zh > zds, th_, S(0.0))
        kk = xp.arange(nk + 2)[:, None, None]
        th_ = xp.where((kk >= 1) & (kk <= nk), th_, S(0.0))
        tf_ = half * (one - xp.cos(pi * (zf - zds) / (ztop - zds)))
        tf_ = xp.where(zf > zds, tf_, S(0.0))
        kf = xp.arange(nk + 3)[:, None, None]
        tf_ = xp.where((kf >= 1) & (kf <= nk + 1), tf_, S(0.0))
        tauh, tauf = th_, tf_
    else:
        tauh, tauf = bk.zeros(g.s_shape), bk.zeros(g.w_shape)
    out = dict(t)
    out.update(
        mh=mh,
        rmh=one / mh,
        mf=mf,
        rmf=one / mf,
        c1=cc1,
        c2=cc2,
        tauh=tauh,
        taus=bk.copy(tauh),
        tauf=tauf,
    )
    return out


# ---------------------------------------------------------------------------
# helpers


def rslf(bk, p, t, ptype: int = 1, eps=None):
    """Liquid saturation mixing ratio (cm1libs.F RSLF); ``eps`` = rd/rv
    (default: the testcase = 0 constants)."""
    xp = bk.xp
    S = bk.asarray
    if ptype == 4:
        return S(380.00) * xp.exp(S(17.2693882) - S(4097.8531) / (t - S(35.86))) / p
    esl = S(611.2) * xp.exp(S(17.67) * (t - S(273.15)) / (t - S(29.65)))
    esl = xp.minimum(esl, p * S(0.5))
    if eps is None:
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


# testcases for which base.F also sets large-scale forcing profiles (wprof,
# ufrc/vfrc, ug/vg, thfrc/qvfrc, ulspg/vlspg, dvdr, nudging): not ported
FORCING_TESTCASES = (3, 4, 5, 6, 7, 10, 11, 12, 13, 15)


def base(
    bk,
    cfg,
    *,
    zh,
    mh,
    rmh,
    c1,
    c2,
    zf,
    mf,
    cg: dict,
    g: Grid | None = None,
    profiles_only: bool = False,
) -> dict:
    """Port of ``base`` (src/base.F): sounding, hydrostatic base state, winds.

    ``cg`` holds the extrapolation coefficients cgs1..3 / cgt1..3 (module
    ``input`` scalars in CM1; e.g. the dict returned by ``param_grid``).
    Every column is computed (vectorised over ``[j, i]``, same per-column
    operation order as CM1), so with terrain (``zh`` varying in x/y) the base
    state is 3-D exactly like CM1's. The soundings (isnd) and wind profiles
    (iwnd) are in :mod:`cm1x.base_snd`.

    base.F also sets large-scale forcing profiles for some testcases
    (``FORCING_TESTCASES``); those are not ported and raise
    NotImplementedError, unless ``profiles_only`` (the harness tests of the
    testcase-dependent soundings/winds), which returns the base state alone.

    Returns arrays keyed by Fortran name: pi0, prs0, rho0, thv0, th0, rth0, qv0,
    qc0, qi0, rr0, rf0, rrf0, t0, rh0 (scalar points), u0, v0 (staggered),
    rho0s, pi0s, prs0s, rth0s (2-D), plus Python-float scalars psurf, tsurf,
    qsurf, thsurf and the 0-d array pmin.
    """
    from . import bc as _bc
    from . import base_snd

    xp = bk.xp
    S = bk.asarray
    nk = zh.shape[0] - 2
    shp_s = zh.shape
    sh2 = shp_s[1:]
    if g is None:
        g = Grid(cfg.ni, cfg.nj, cfg.nk, float(cfg.dx), float(cfg.dy), float(cfg.dz))
    imoist = cfg.imoist
    terrain = bool(cfg.get("terrain_flag", False))
    testcase = int(cfg.get("testcase", 0) or 0)
    if testcase in FORCING_TESTCASES and not profiles_only:
        raise NotImplementedError(
            f"testcase={testcase}: base.F large-scale forcing profiles are not ported"
        )
    cg = {k: S(cg[k]) for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}

    K = base_snd.consts(testcase)
    grav, cp, rd, rv, p00 = S(K["g"]), S(K["cp"]), S(K["rd"]), S(K["rv"]), S(C.p00)
    reps = rv / rd
    cpdrd = cp / rd
    rovcp = rd / cp
    rp00 = S(1.0) / p00
    one, zero = S(1.0), S(0.0)
    half = S(0.5)

    def Z(n):
        return bk.zeros((n,) + sh2)

    zc = zh  # [k, j, i], Python index == Fortran k

    snd = base_snd.sounding(bk, cfg, zh=zh, cg=cg, g=g)
    pi0, prs0, th0, qv0 = snd["pi0"], snd["prs0"], snd["th0"], snd["qv0"]
    qc0, qi0, rh0 = snd["qc0"], snd["qi0"], snd["rh0"]
    psurf, tsurf, qsurf = snd["psurf"], snd["tsurf"], snd["qsurf"]

    # ---------------- ghost cells
    def bcx(a):
        return extrapbcs(bk, _bc.bcs(bk, g, a, cfg), cg, nk)

    pi0 = bcx(pi0)
    prs0 = bcx(prs0)
    th0 = bcx(th0)
    qv0 = bcx(qv0)
    qc0 = bcx(qc0)
    qi0 = bcx(qi0)
    rh0 = bcx(rh0)

    if imoist == 1:
        thv0 = th0 * (one + reps * qv0) / (one + qv0 + qc0 + qi0)
    else:
        qv0 = Z(nk + 2)
        rh0 = Z(nk + 2)
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

    rho0 = bcx(rho0)
    if cfg.psolver in (5, 6):
        rho0 = bk.full(shp_s, 1.0)

    rr0 = one / rho0
    rf_mid = c1[2 : nk + 1] * rho0[1:nk] + c2[2 : nk + 1] * rho0[2 : nk + 1]
    rf1 = cg["cgs1"] * rho0[1] + cg["cgs2"] * rho0[2] + cg["cgs3"] * rho0[3]
    rftop = (
        cg["cgt1"] * rho0[nk] + cg["cgt2"] * rho0[nk - 1] + cg["cgt3"] * rho0[nk - 2]
    )
    rf0 = xp.concat([xp.stack([rf1, rf1]), rf_mid, xp.stack([rftop])])
    rrf0 = one / rf0
    rho0s = rf1

    # ---------------- surface values
    tsm = float(cfg.tsmall)
    if float(psurf) >= tsm and float(tsurf) >= tsm and not terrain:
        thsurf = tsurf / ((psurf * rp00) ** rovcp)
        prs0s = psurf + bk.zeros(sh2)
        pi0s = (psurf * rp00) ** rovcp + bk.zeros(sh2)
        rth0s = thsurf ** (-one) + bk.zeros(sh2)
    else:
        ths = cg["cgs1"] * th0[1] + cg["cgs2"] * th0[2] + cg["cgs3"] * th0[3]
        qs = cg["cgs1"] * qv0[1] + cg["cgs2"] * qv0[2] + cg["cgs3"] * qv0[3]
        thvsurf = ths * (one + qs * reps) / (one + qs)
        pi0s = pi0[1] + zc[1] * grav / (cp * half * (thvsurf + thv0[1]))
        prs0s = p00 * (pi0s ** (cp / rd))
        rth0s = ths ** (-one)
        # module scalars keep the last column's values (i = ie, j = je)
        thsurf = ths[-1, -1]
        qsurf = qs[-1, -1]
    # (Fortran i = j = 1 is Python index 3)
    if float(psurf) < tsm:
        psurf = prs0s[3, 3]
    if float(tsurf) <= tsm:
        tsurf = cg["cgs1"] * t0[1, 3, 3] + cg["cgs2"] * t0[2, 3, 3] + cg["cgs3"] * t0[3, 3, 3]
    if float(qsurf) <= tsm:
        qsurf = (
            cg["cgs1"] * qv0[1, 3, 3] + cg["cgs2"] * qv0[2, 3, 3] + cg["cgs3"] * qv0[3, 3, 3]
        )
    thsurf = tsurf / ((psurf * rp00) ** rovcp)

    # ---------------- base-state wind (iwnd; isnd = 7: from input_sounding)
    if int(cfg.isnd) == 7:
        u0, v0 = snd["u0"], snd["v0"]
    else:
        u0, v0 = base_snd.wind(bk, cfg, zh=zh, g=g)
    if cfg.imove == 1:
        # k = 1..nk, j = 1..nj+1, i = 1..ni+1 (Python index = Fortran + 2)
        if terrain:
            raise NotImplementedError("imove=1 with terrain")
        kk = xp.arange(nk + 2)[:, None, None]
        mk = (kk >= 1) & (kk <= nk)
        mu = mk & base_snd._ji_mask(bk, u0.shape[1:], 1, g.nj + 1, 1, g.ni + 1)[None]
        mv = mk & base_snd._ji_mask(bk, v0.shape[1:], 1, g.nj + 1, 1, g.ni + 1)[None]
        u0 = xp.where(mu, u0 - S(float(cfg.umove)), u0)
        v0 = xp.where(mv, v0 - S(float(cfg.vmove)), v0)
    u0 = extrapbcs(bk, _bc.bcu(bk, g, u0, cfg), cg, nk)
    v0 = extrapbcs(bk, _bc.bcv(bk, g, v0, cfg), cg, nk)

    rth0 = one / th0

    out = dict(
        pi0=pi0,
        prs0=prs0,
        rho0=rho0,
        thv0=thv0,
        th0=th0,
        rth0=rth0,
        qv0=qv0,
        qc0=qc0,
        qi0=qi0,
        rr0=rr0,
        rf0=rf0,
        rrf0=rrf0,
        t0=t0,
        rh0=rh0,
        u0=u0,
        v0=v0,
        rho0s=rho0s,
        pi0s=pi0s,
        prs0s=prs0s,
        rth0s=rth0s,
        pmin=xp.min(prs0[1 : nk + 1, 3 : 3 + g.nj, 3 : 3 + g.ni]),
        psurf=float(psurf),
        tsurf=float(tsurf),
        qsurf=float(qsurf),
        thsurf=float(thsurf),
    )
    return out
