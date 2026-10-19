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

__all__ = ["extrap_coeffs", "param_grid", "vertical_grid", "zgrid", "rslf", "extrapbcs", "base"]


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


def _powi(x: float, m: int) -> float:
    """x**m (real x, integer m) as gfortran evaluates it (libgcc __powidf2:
    binary exponentiation), not C pow()."""
    n = -m if m < 0 else m
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def _zheight(dzbot, r, nz, dzmax, n1, zctop, ztopr, dzmax2):
    """param.F FUNCTION ZHEIGHT (geometrically stretched grid height)."""
    total = 0.0
    n2 = 0
    dzm = 0.0
    for k in range(1, nz + 1):
        if k <= n1:
            dznew = dzbot
        else:
            k2 = k - n1
            dznew = min(dzbot * _powi(r, k2 - 1), dzmax)
            if total >= zctop:
                if n2 == 0:
                    dzm = min(dznew, dzmax)
                n2 = n2 + 1
                dznew = min(dzm * _powi(ztopr, n2), dzmax2)
        total = total + dznew
    return total


def zgrid(dz, dz_stretch, nz, nbndlyr, dzmax, ztopstr=100000.0, rtop=1.0, dzmaxtop=700.0):
    """param.F SUBROUTINE ZGRID (stretch_z = 2): w levels ``gze[1..nz]`` of
    a geometric stretch (bisection for the factor); returns a list indexed
    0..nz+1 with gze[0] and gze[nz+1] left at 0."""
    ztop = dz * (nz - 1)
    zx, stretch = 1.0, 1.0
    if dz > dz_stretch:
        for _ in range(50):
            if abs(zx) > 1.0e-12:
                zx = zx * 0.5
                xmid = stretch + zx
                fmid = (
                    _zheight(dz_stretch, xmid, nz - 1, dzmax, nbndlyr, ztopstr, rtop, dzmaxtop)
                    - ztop
                )
                if fmid <= 0.0:
                    stretch = xmid
                if fmid == 0.0:
                    break
    if stretch > 1.1:
        raise ValueError(
            f"stretch_z=2: stretch factor too big ({stretch}); increase nz or dz_bot"
        )
    gze = [0.0] * (nz + 2)
    for k in range(1, nz):
        gze[k + 1] = _zheight(dz_stretch, stretch, k, dzmax, nbndlyr, ztopstr, rtop, dzmaxtop)
    return gze


def _read_grid_z(cfg, n):
    """The first value of each of the first ``n`` lines of ``input_grid_z``
    (list-directed ``read(40,*)``), next to the namelist file."""
    import os

    path = os.path.join(str(cfg.get("namelist_dir", "") or "."), "input_grid_z")
    vals = []
    with open(path) as f:
        for line in f:
            tok = line.replace(",", " ").split()
            if not tok:
                continue
            vals.append(float(tok[0].lower().replace("d", "e")))
            if len(vals) == n:
                break
    if len(vals) < n:
        raise ValueError(f"{path}: expected {n} values, found {len(vals)}")
    return vals


def vertical_grid(cfg, nk):
    """param.F's vertical grid columns (Python floats, Python index ==
    Fortran k): zf (k = 0..nk+2), zh, mh (0..nk+1), mf (0..nk+2) and the
    nominal heights sigmaf (0..nk+2), for stretch_z = 0 (uniform dz), 1
    (analytic), 2 (ZGRID, dz = ztop/nk), 3 (scalar levels) and 4 (w levels)
    from ``input_grid_z``.  Every column is the same (no terrain here)."""
    stretch_z = int(cfg.get("stretch_z", 0) or 0)
    if stretch_z not in (0, 1, 2, 3, 4):
        raise NotImplementedError(f"stretch_z={stretch_z} is not a CM1 option")
    dz = float(cfg.dz)
    ztop = float(cfg.get("ztop", 0.0) or 0.0)
    zf = [dz * (k - 1) for k in range(0, nk + 3)]
    zh = [0.5 * (zf[k] + zf[k + 1]) for k in range(0, nk + 2)]
    mh = [1.0] * (nk + 2)
    mf = [1.0] * (nk + 3)
    sigmaf = list(zf)
    if stretch_z == 0:
        return zf, zh, mh, mf, sigmaf
    if stretch_z == 1:
        str_bot, str_top = float(cfg.str_bot), float(cfg.str_top)
        dz_bot, dz_top = float(cfg.dz_bot), float(cfg.dz_top)
        nominal_dz = 0.5 * (dz_bot + dz_top)

        def nint(x):
            return int(x + 0.5) if x >= 0 else -int(-x + 0.5)

        nk1 = nint(str_bot / dz_bot)
        nk3 = nint((ztop - str_top) / dz_top)
        nk2 = nk - (nk1 + nk3)
        if abs(nk1 * dz_bot - str_bot) > 0.01:
            raise ValueError("stretch_z=1: depth of bottom layer does not divide by dz_bot")
        if abs(nk2 * nominal_dz - (str_top - str_bot)) >= 0.01:
            raise ValueError("stretch_z=1: depth of middle layer does not divide by nominal_dz")
        if abs(nk3 * dz_top - (ztop - str_top)) > 0.01:
            raise ValueError("stretch_z=1: depth of top layer does not divide by dz_top")
        nominal_dz = (str_top - str_bot) / nk2
        c2 = (nominal_dz - dz_bot) / (nominal_dz * nominal_dz * float(nk2 - 1))
        c1 = (dz_bot / nominal_dz) - c2 * nominal_dz
        test_len = dz_bot * nk1 + 0.5 * (dz_bot + dz_top) * nk2 + dz_top * nk3
        if (
            int(cfg.nz) != nk1 + nk2 + nk3
            or min(nk1, nk2, nk3) < 0
            or abs(ztop - test_len) > 1.0e-6 * ztop
        ):
            raise ValueError("stretch_z=1: inconsistent nz/ztop/str_bot/str_top/dz_bot/dz_top")
        col = list(zf)
        for k in range(1, nk1 + 2):
            col[k] = float(k - 1) * dz_bot
        for k in range(nk1 + 1, nk1 + nk2 + 2):
            col[k] = col[nk1 + 1] + (c1 + c2 * float(k - 1 - nk1) * nominal_dz) * float(
                k - 1 - nk1
            ) * nominal_dz
        for k in range(nk1 + nk2 + 2, nk1 + nk2 + nk3 + 2):
            col[k] = col[k - 1] + dz_top
        zf = col
        sigmaf = list(col)  # same recurrences in param.F
        sigmaf[0] = -sigmaf[2]
        sigmaf[nk + 2] = sigmaf[nk + 1] + (sigmaf[nk + 1] - sigmaf[nk])
    elif stretch_z == 2:
        if cfg.get("terrain_flag", False):
            raise NotImplementedError(
                "stretch_z=2 with terrain (param.F: not yet compatible)"
            )
        dz_bot, dz_top = float(cfg.dz_bot), float(cfg.dz_top)
        nbndlyr = max(0, int(float(cfg.str_bot) / dz_bot + 0.01) - 1)
        ke = nk + 1
        gze = zgrid(dz, dz_bot, ke, nbndlyr, dz_top)
        gze[0] = -gze[2]
        gze[ke + 1] = gze[ke] + (gze[ke] - gze[ke - 1])
        zf = list(gze)
        sigmaf = list(gze)
    else:
        # cm1r19: levels in input_grid_z; sigmaf(0), sigmaf(nk+2) keep the
        # uniform-grid values (param.F)
        if stretch_z == 3:
            sig = _read_grid_z(cfg, nk)
            sigmaf[1] = 0.0
            for k in range(1, nk + 1):
                sigmaf[k + 1] = 2.0 * sig[k - 1] - sigmaf[k]
        else:
            sf = _read_grid_z(cfg, nk + 1)
            for k in range(1, nk + 2):
                sigmaf[k] = sf[k - 1]
        zf = list(zf)
        for k in range(1, nk + 2):
            zf[k] = sigmaf[k]
    # param.F, common to every stretch_z
    zf[0] = -zf[2]
    zf[nk + 2] = zf[nk + 1] + (zf[nk + 1] - zf[nk])
    zh = [0.5 * (zf[k + 1] + zf[k]) for k in range(0, nk + 2)]
    mh = [dz / (zf[k + 1] - zf[k]) for k in range(0, nk + 2)]
    zh[0] = -zh[1]
    zh[nk + 1] = zh[nk] + 2.0 * (zf[nk + 1] - zh[nk])
    mf = [0.0] * (nk + 3)
    for k in range(1, nk + 2):
        mf[k] = dz / (zh[k] - zh[k - 1])
    mf[0] = mf[1]
    mf[nk + 2] = mf[nk + 1]
    for k in range(1, nk + 1):
        if zf[k + 1] - zf[k] < 1.0e-60:
            raise ValueError(f"vertical grid: dz <= 0 at k = {k}")
    return zf, zh, mh, mf, sigmaf



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
    if cfg.stretch_x or cfg.stretch_y:
        raise NotImplementedError("horizontal grid stretching (stretch_x/y) is not ported")
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

    # vertical (Python index == Fortran k), param.F incl. stretch_z = 1..4
    zf1, zh1, mh1, mf1, sigmaf = vertical_grid(cfg, nk)
    shp_s, shp_w = g.s_shape, g.w_shape

    def col3(vals, shp):
        return bk.asarray(vals)[:, None, None] + bk.zeros(shp)

    zf = col3(zf1, shp_w)
    zh = col3(zh1, shp_s)
    mh = col3(mh1, shp_s)
    mf = col3(mf1, shp_w)
    # cc2(k) = (zf(k)-zh(k-1))/(zh(k)-zh(k-1)) for k=1..nk+1, cc1 = 1-cc2 ; 0 at k=0
    zfa, zha = bk.asarray(zf1), bk.asarray(zh1)
    c2col = (zfa[1 : nk + 2] - zha[0 : nk + 1]) / (zha[1 : nk + 2] - zha[0 : nk + 1])
    c1col = 1.0 - c2col
    z0 = bk.zeros((1,))
    c1 = xp.concat([z0, c1col])[:, None, None] + bk.zeros(shp_s)
    c2 = xp.concat([z0, c2col])[:, None, None] + bk.zeros(shp_s)

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
        rmh=col3([1.0 / m for m in mh1], shp_s),
        zf=zf,
        mf=mf,
        rmf=col3([1.0 / m for m in mf1], shp_w),
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
    # cm1r18 ghost points for zh (param.F, after the terrain section): index
    # 0 is the surface, nk+1 the model top
    zh, zf = out["zh"], out["zf"]
    zh = bk.set(zh, 0, zf[1])
    out["zh"] = bk.set(zh, nk + 1, zf[nk + 1])
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


def base(
    bk, cfg, *, zh, mh, rmh, c1, c2, zf, mf, cg: dict, g: Grid | None = None
) -> dict:
    """Port of ``base`` (src/base.F): sounding, hydrostatic base state, winds.

    ``cg`` holds the extrapolation coefficients cgs1..3 / cgt1..3 (module
    ``input`` scalars in CM1; e.g. the dict returned by ``param_grid``).
    Every column is computed (vectorised over ``[j, i]``, same per-column
    operation order as CM1), so with terrain (``zh`` varying in x/y) the base
    state is 3-D exactly like CM1's.

    Returns arrays keyed by Fortran name: pi0, prs0, rho0, thv0, th0, rth0, qv0,
    qc0, qi0, rr0, rf0, rrf0, t0, rh0 (scalar points), u0, v0 (staggered),
    rho0s, pi0s, prs0s, rth0s (2-D), plus Python-float scalars psurf, tsurf,
    qsurf, thsurf and the 0-d array pmin.
    """
    from . import bc as _bc

    xp = bk.xp
    S = bk.asarray
    nk = zh.shape[0] - 2
    shp_s = zh.shape
    sh2 = shp_s[1:]
    shp_u = (shp_s[0], shp_s[1], shp_s[2] + 1)
    shp_v = (shp_s[0], shp_s[1] + 1, shp_s[2])
    if g is None:
        g = Grid(cfg.ni, cfg.nj, cfg.nk, float(cfg.dx), float(cfg.dy), float(cfg.dz))
    imoist = cfg.imoist
    ptype = cfg.ptype
    terrain = bool(cfg.get("terrain_flag", False))
    cg = {k: S(cg[k]) for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}

    grav, cp, rd, rv, p00 = S(C.g), S(C.cp), S(C.rd), S(C.rv), S(C.p00)
    reps = rv / rd
    cpdrd = cp / rd
    rovcp = rd / cp
    rp00 = S(1.0) / p00
    one, zero = S(1.0), S(0.0)
    half = S(0.5)

    def Z(n):
        return bk.zeros((n,) + sh2)

    zc = zh  # [k, j, i], Python index == Fortran k
    z1 = Z(1)
    qc0 = Z(nk + 2)
    qi0 = Z(nk + 2)

    if cfg.isnd == 5:
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
            th_trop * xp.exp((grav / (t_trop * cp)) * (zi - z_trop)),
        )
        if imoist == 1:
            rh_in = xp.where(below, one - S(0.75) * ratio, S(0.25))
        else:
            rh_in = Z(nk)
        th0 = xp.concat([th_in[:1], th_in, th_in[-1:]])
        rh0 = xp.concat([z1, rh_in, z1])
        qv0 = Z(nk + 2)
        pi0 = Z(nk + 2)
        prs0 = Z(nk + 2)

        for _n in range(20):
            thv0 = th0 * (one + reps * qv0) / (one + qv0)
            pis = [pi_sfc - grav * zc[1] / (cp * half * (thv_sfc + thv0[1]))]
            for k in range(2, nk + 1):
                pis.append(
                    pis[-1]
                    - grav * (zc[k] - zc[k - 1]) / (cp * half * (thv0[k] + thv0[k - 1]))
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
            q1 = cg["cgs1"] * qv0[1, 0, 0] + cg["cgs2"] * qv0[2, 0, 0]
            q1 = q1 + cg["cgs3"] * qv0[3, 0, 0]
            qsurf = xp.where(qv0[1, 0, 0] < qv_pbl, q1, qv_pbl)
        else:
            qsurf = zero
    elif cfg.isnd == 9:
        # ---------------- isnd = 9: dry, constant Brunt-Vaisala frequency
        ns1 = S(0.0001)
        zl1 = 40000.0
        if float(xp.max(zc[1 : nk + 2])) > zl1 or float(xp.min(zc[1])) >= zl1:
            raise NotImplementedError("isnd=9 above zl1 = 40 km not ported")
        th_sfc, pi_sfc = S(288.0), S(1.0)
        psurf = p00 * (pi_sfc ** (cp / rd))
        tsurf = th_sfc * pi_sfc
        zi = zc[1 : nk + 1]
        thv_in = th_sfc * xp.exp(ns1 * zi / grav)
        pi_in = pi_sfc + grav * grav / (cp * ns1 * th_sfc) * (
            xp.exp(-ns1 * zi / grav) - one
        )
        prs_in = p00 * (pi_in**cpdrd)
        # rh0 = 0 => qv0 = 0 and th0 = thv0*(1+0+0)/(1+0) = thv0 exactly
        th0 = xp.concat([z1, thv_in, z1])
        pi0 = xp.concat([z1, pi_in, z1])
        prs0 = xp.concat([z1, prs_in, z1])
        qv0 = Z(nk + 2)
        rh0 = Z(nk + 2)
        qsurf = zero
    else:
        raise NotImplementedError(f"isnd={cfg.isnd} not ported (only isnd=5, 9)")

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
        if float(psurf) < tsm:
            psurf = prs0s[3, 3]
        if float(tsurf) <= tsm:
            tsurf = (
                cg["cgs1"] * t0[1, 3, 3]
                + cg["cgs2"] * t0[2, 3, 3]
                + cg["cgs3"] * t0[3, 3, 3]
            )
        if float(qsurf) <= tsm:
            qsurf = (
                cg["cgs1"] * qv0[1, 3, 3]
                + cg["cgs2"] * qv0[2, 3, 3]
                + cg["cgs3"] * qv0[3, 3, 3]
            )
        thsurf = tsurf / ((psurf * rp00) ** rovcp)

    # ---------------- base-state wind (iwnd)
    # computed on every column, then cut to the (i, j) range CM1 sets for
    # this iwnd (zero elsewhere, as CM1's initial u0/v0), then bcu/bcv and
    # the k = 0, nk+1 extrapolation (with mixed open/periodic boundaries
    # the halo corners keep those zeros, as in CM1).
    zfull = xp.concat([zc[:, :, :1], zc, zc[:, :, -1:]], axis=2)
    zu = half * (zfull[:, :, :-1] + zfull[:, :, 1:])  # u points
    zfull = xp.concat([zc[:, :1, :], zc, zc[:, -1:, :]], axis=1)
    zv = half * (zfull[:, :-1, :] + zfull[:, 1:, :])  # v points
    zu, zv = zu[1 : nk + 1], zv[1 : nk + 1]
    if cfg.iwnd in (0, None):
        u_in = bk.zeros(zu.shape)
        v_in = bk.zeros(zv.shape)
    elif cfg.iwnd == 1:
        udep1, udep2, uc1, uc2 = S(0.0), S(2500.0), S(0.0), S(10.0)
        z = zu
        u_in = xp.where(
            z < udep1,
            uc1,
            xp.where(
                (z > udep1) & (z < udep2),
                (uc2 - uc1) * (z - udep1) / (udep2 - udep1) + uc1,
                uc2,
            ),
        )
        v_in = bk.zeros(zv.shape)
    elif cfg.iwnd == 2:
        udep1, udep2, umax1, umax2 = S(2000.0), S(6000.0), S(7.0), S(31.0)
        pi180 = S(C.pi) / S(180.0)
        z = zu
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
        anglev = S(90.0) * (zv / udep1) * pi180
        v_in = xp.where(zv <= udep1, vmax1 * xp.sin(anglev), vmax1)
    elif cfg.iwnd == 6:
        tc = int(cfg.get("testcase", 0))
        uval, vval = {1: (1.0, 0.0), 2: (15.0, 0.0), 4: (7.0, -5.5), 9: (8.0, 0.0)}.get(
            tc, {11: (2.0, 0.0), 12: (20.0, 0.0)}.get(tc, (10.0, 0.0))
        )
        u_in = S(uval) + bk.zeros(zu.shape)
        v_in = S(vval) + bk.zeros(zv.shape)
    else:
        raise NotImplementedError(f"iwnd={cfg.iwnd} not ported")

    if cfg.imove == 1:
        if terrain:
            raise NotImplementedError("imove=1 with terrain")
        u_in = u_in - S(float(cfg.umove))
        v_in = v_in - S(float(cfg.vmove))

    # Fortran (j, i) ranges set by each iwnd option; Python index = Fortran + 2
    rng = {
        1: (((0, g.nj + 1), (0, g.ni + 2)), ((0, g.nj + 2), (0, g.ni + 1))),
        2: (((1, g.nj), (1, g.ni + 1)), ((1, g.nj + 1), (1, g.ni))),
    }.get(int(cfg.iwnd or 0))
    if rng is not None:

        def _cut(a, jr, ir):
            m = bk.zeros(a.shape[1:])
            m = bk.set(
                m, (slice(jr[0] + 2, jr[1] + 3), slice(ir[0] + 2, ir[1] + 3)), 1.0
            )
            return a * m[None]

        u_in, v_in = _cut(u_in, *rng[0]), _cut(v_in, *rng[1])

    u0 = xp.concat([bk.zeros((1,) + shp_u[1:]), u_in, bk.zeros((1,) + shp_u[1:])])
    v0 = xp.concat([bk.zeros((1,) + shp_v[1:]), v_in, bk.zeros((1,) + shp_v[1:])])
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
