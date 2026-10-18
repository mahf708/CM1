"""Lateral boundary conditions, ported from src/bc.F (single domain, non-MPI).

Every bc.F routine used by the dynamical core is here with all of its non-MPI
branches: ``wbc/ebc/sbc/nbc`` = 1 (periodic), 2 (open: zero-gradient halos),
3 or 4 (rigid free-slip / no-slip walls, mirrored halos). The MPI-only
corner routines (``bcs2``, ``bcu2``, ``bcv2``, ``bcw2``, ...) are not
compiled without MPI and are not needed.

Open (radiative) boundaries with ``irbc=4`` (Durran & Klemp 1983 style) use
``radbcew4``/``radbcns4`` (phase speeds, called once per RK step in solve2) and
``ssopenbcw/e/s/n`` (boundary-normal velocity update on every acoustic step,
called from sound).

Calling convention: ``bcs(bk, g, s, cfg=None)`` etc. return the updated array.
``cfg`` is anything with ``.get(name, default)`` (dict, cm1x Config) or
attributes; ``cfg=None`` means periodic everywhere. Index mapping as in
cm1x.grid: Fortran i/j -> Python i+2, k unchanged.
"""

from __future__ import annotations

csmax = 350.0  # constants.F: real, parameter :: csmax = 350.0

_DEFAULTS = dict(wbc=1, ebc=1, sbc=1, nbc=1, axisymm=0, irbc=4, terrain_flag=False)


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    get = getattr(cfg, "get", None)
    if get is not None:
        return get(name, _DEFAULTS[name])
    return getattr(cfg, name, _DEFAULTS[name])


def _bcs(cfg):
    out = []
    for b in ("wbc", "ebc", "sbc", "nbc"):
        v = int(_opt(cfg, b))
        if v not in (1, 2, 3, 4):
            raise NotImplementedError(f"{b}={v} not supported")
        out.append(v)
    return out


def ib_flags(cfg):
    """param.F (single domain): ibw=1 when the west edge is not periodic, etc.
    Returns (ibw, ibe, ibs, ibn)."""
    wbc, ebc, sbc, nbc = _bcs(cfg)
    return tuple(int(b != 1) for b in (wbc, ebc, sbc, nbc))


def periodic_xy(bk, a, g, stag_x: bool = False, stag_y: bool = False):
    """Fill the ngxy-wide horizontal halos of ``a`` periodically.

    For a u-staggered array (stag_x) the interior is i = 1..ni+1 in Fortran, and
    CM1 sets u(ni+1) = u(1) periodically; same for v in y.
    """
    xp = bk.xp
    n = g.ngxy
    # x direction (last axis)
    ni = g.ni
    core = a[..., n : n + ni]
    left = core[..., ni - n :]
    right_len = a.shape[-1] - (n + ni)
    right = core[..., :right_len]
    a = xp.concat([left, core, right], axis=-1)
    # y direction
    nj = g.nj
    core = a[..., n : n + nj, :]
    lo = core[..., nj - n :, :]
    hi_len = a.shape[-2] - (n + nj)
    hi = core[..., :hi_len, :]
    a = xp.concat([lo, core, hi], axis=-2)
    return a


# ---------------------------------------------------------------------------
# generic engine


def _fs(a, b):
    """Fortran inclusive i/j range a..b -> python slice."""
    return slice(a + 2, b + 3)


_ALL = slice(None)


def _side(bk, a, *, axis, mode, side, n, stag, ng, lead, t):
    """Apply one edge of a bc.F routine.

    axis 'x' (last dim) or 'y' (second to last); mode = wbc-style value;
    side 'lo' (west/south) or 'hi' (east/north); n = ni or nj; stag: the field
    is staggered along ``axis`` (u in x, v in y); ng: halo width touched;
    lead: tuple of leading-dim slices (k); t: slice of the transverse dim.
    """

    def ix(p):  # p is a Fortran index along axis
        p = p + 2
        return (*lead, t, p) if axis == "x" else (*lead, p, t)

    def cp(a, dst, src, sign=1.0):
        v = a[ix(src)]
        return bk.set(a, ix(dst), -v if sign < 0 else v)

    lo = side == "lo"
    if mode == 1:
        for i in range(1, ng + 1):
            if lo:
                a = cp(a, 1 - i, n + 1 - i)
            else:
                a = cp(a, n + 1 + i, 1 + i) if stag else cp(a, n + i, i)
    elif mode == 2:
        for i in range(1, ng + 1):
            if lo:
                a = cp(a, 1 - i, 1)
            else:
                a = cp(a, n + 1 + i, n + 1) if stag else cp(a, n + i, n)
    else:  # 3, 4: walls
        if stag:
            b = 1 if lo else n + 1
            a = bk.set(a, ix(b), 0.0)
            for i in range(1, ng + 1):
                a = cp(a, b - i, b + i, -1.0) if lo else cp(a, b + i, b - i, -1.0)
        else:
            for i in range(1, ng + 1):
                a = cp(a, 1 - i, i) if lo else cp(a, n + i, n + 1 - i)
    return a


def _x_then_y(
    bk, g, a, cfg, *, lead, ng, tx, ty, stag_x=False, stag_y=False, y_first=False
):
    """tx(mode) / ty(mode): transverse slice for the x / y passes."""
    wbc, ebc, sbc, nbc = _bcs(cfg)

    def xpass(a):
        a = _side(
            bk,
            a,
            axis="x",
            mode=wbc,
            side="lo",
            n=g.ni,
            stag=stag_x,
            ng=ng,
            lead=lead,
            t=tx(wbc),
        )
        a = _side(
            bk,
            a,
            axis="x",
            mode=ebc,
            side="hi",
            n=g.ni,
            stag=stag_x,
            ng=ng,
            lead=lead,
            t=tx(ebc),
        )
        return a

    def ypass(a):
        a = _side(
            bk,
            a,
            axis="y",
            mode=sbc,
            side="lo",
            n=g.nj,
            stag=stag_y,
            ng=ng,
            lead=lead,
            t=ty(sbc),
        )
        a = _side(
            bk,
            a,
            axis="y",
            mode=nbc,
            side="hi",
            n=g.nj,
            stag=stag_y,
            ng=ng,
            lead=lead,
            t=ty(nbc),
        )
        return a

    if y_first:
        return xpass(ypass(a))
    return ypass(xpass(a))


def _k(a, b):
    return slice(a, b + 1)


# ---------------------------------------------------------------------------
# bc.F routines


def bc2d(bk, g, s, cfg=None):
    """bc.F ``bc2d``: 2-D ``s[j, i]``, all halo points, full transverse range."""
    return _x_then_y(
        bk, g, s, cfg, lead=(), ng=g.ngxy, tx=lambda m: _ALL, ty=lambda m: _ALL
    )


def bcs(bk, g, s, cfg=None):
    """bc.F ``bcs``: scalar ``s[k, j, i]``, ngxy halos, k=1..nk. Periodic edges
    fill the full transverse range; other edges j (i) = 0..nj+1 (0..ni+1)."""
    return _x_then_y(
        bk,
        g,
        s,
        cfg,
        lead=(_k(1, g.nk),),
        ng=g.ngxy,
        tx=lambda m: _ALL if m == 1 else _fs(0, g.nj + 1),
        ty=lambda m: _ALL if m == 1 else _fs(0, g.ni + 1),
    )


def bcp(bk, g, p, cfg=None):
    """bc.F ``bcp``: one halo row/column of a pressure-like array, k=1..nk,
    j (i) = 0..nj+1 (0..ni+1); west/east first, then south/north."""
    return _x_then_y(
        bk,
        g,
        p,
        cfg,
        lead=(_k(1, g.nk),),
        ng=1,
        tx=lambda m: _fs(0, g.nj + 1),
        ty=lambda m: _fs(0, g.ni + 1),
    )


def bcu(bk, g, u, cfg=None):
    """bc.F ``bcu``: u (staggered in x), k=1..nk. Walls set u(1)=u(ni+1)=0
    and antisymmetric halos."""
    return _x_then_y(
        bk,
        g,
        u,
        cfg,
        lead=(_k(1, g.nk),),
        ng=g.ngxy,
        stag_x=True,
        tx=lambda m: _ALL if m == 1 else _fs(0, g.nj + 1),
        ty=lambda m: _ALL if m == 1 else _fs(0, g.ni + 2),
    )


def bcv(bk, g, v, cfg=None):
    """bc.F ``bcv``: v (staggered in y), k=1..nk; south/north first, then
    west/east (axisymm=1 not ported)."""
    if int(_opt(cfg, "axisymm")) != 0:
        raise NotImplementedError("bcv: axisymm=1 not ported")
    return _x_then_y(
        bk,
        g,
        v,
        cfg,
        lead=(_k(1, g.nk),),
        ng=g.ngxy,
        stag_y=True,
        y_first=True,
        tx=lambda m: _ALL if m == 1 else _fs(0, g.nj + 2),
        ty=lambda m: _ALL if m == 1 else _fs(0, g.ni + 1),
    )


def bcw(bk, g, w, flag, cfg=None):
    """bc.F ``bcw``: w halos for k=2..nk; ``flag=1`` also sets
    w(0:ni+1, 0:nj+1, 1) = w(..., nk+1) = 0."""
    w = _x_then_y(
        bk,
        g,
        w,
        cfg,
        lead=(_k(2, g.nk),),
        ng=g.ngxy,
        tx=lambda m: _ALL if m == 1 else _fs(0, g.nj + 1),
        ty=lambda m: _ALL if m == 1 else _fs(0, g.ni + 1),
    )
    if flag == 1:
        J0, I0 = _fs(0, g.nj + 1), _fs(0, g.ni + 1)
        w = bk.set(w, (1, J0, I0), 0.0)
        w = bk.set(w, (g.nk + 1, J0, I0), 0.0)
    return w


def bcwsfc(bk, g, *, gz, dzdx, dzdy, u, v, w):
    """bc.F ``bcwsfc`` (terrain): kinematic lower boundary condition,
    w(i,j,1) = 0.5*((u(i)+u(i+1))*dzdx + (v(j)+v(j+1))*dzdy)*gz for
    i, j = 0..ni+1, 0..nj+1.  ``gz``, ``dzdx``, ``dzdy`` are 2-D ``[j, i]``."""
    J0, I0 = _fs(0, g.nj + 1), _fs(0, g.ni + 1)
    J1, I1 = _fs(1, g.nj + 2), _fs(1, g.ni + 2)
    ws = (
        0.5
        * (
            (u[1, J0, I0] + u[1, J0, I1]) * dzdx[J0, I0]
            + (v[1, J0, I0] + v[1, J1, I0]) * dzdy[J0, I0]
        )
        * gz[J0, I0]
    )
    return bk.set(w, (1, J0, I0), ws)


# ---------------------------------------------------------------------------
# radiative open boundaries (irbc=4)


def _rad_avg(bk, g, cfg, num, foo, dt, lo):
    """Per-column vertical average of the clipped phase speed (k loop kept
    sequential to match the Fortran sum exactly). num/foo: [k, n] for k=1..nk."""
    xp = bk.xp
    tiny = bk.asarray(1.0e-10)
    zero, cmax = bk.asarray(0.0), bk.asarray(csmax)
    f = foo[: g.nk]
    c = num[: g.nk] / (dt * (xp.copysign(tiny, f) + f))
    if lo:  # max(min(c, 0), -umax)
        c = xp.maximum(xp.minimum(c, zero), -cmax)
    else:  # min(max(c, 0), umax)
        c = xp.minimum(xp.maximum(c, zero), cmax)
    # sequential sum over k, as the Fortran
    acc, _ = bk.scan(lambda a, ck: (a + ck, None), c[0], c[1:])
    return acc / float(g.nk)


def radbcew4(bk, g, cfg=None, *, ruf, radbcw, radbce, u1, u2, dt):
    """bc.F ``radbcew4``: column-mean outward phase speed at the west/east
    open boundaries. u1 = ua (past), u2 = u3d (new), dt = dttmp.
    radbcw/radbce are ``[k, j]`` (Fortran ``(jb:je,kb:ke)``); rows j=1..nj,
    k=1..nk are set. Returns {'radbcw', 'radbce'}."""
    wbc, ebc, _, _ = _bcs(cfg)
    ni, nj, nk = g.ni, g.nj, g.nk
    K, J = _k(1, nk), _fs(1, nj)
    radbcw, radbce = bk.copy(radbcw), bk.copy(radbce)
    xp = bk.xp
    if wbc == 2:
        foo = u1[K, J, 3 + 2] - u1[K, J, 2 + 2]
        num = g.dx * ruf[2 + 2] * (u1[K, J, 2 + 2] - u2[K, J, 2 + 2])
        avg = _rad_avg(bk, g, cfg, num, foo, dt, lo=True)
        radbcw = bk.set(radbcw, (K, J), xp.broadcast_to(avg[None, :], (nk, nj)))
    if ebc == 2:
        foo = u1[K, J, ni + 2] - u1[K, J, ni - 1 + 2]
        num = g.dx * ruf[ni + 2] * (u1[K, J, ni + 2] - u2[K, J, ni + 2])
        avg = _rad_avg(bk, g, cfg, num, foo, dt, lo=False)
        radbce = bk.set(radbce, (K, J), xp.broadcast_to(avg[None, :], (nk, nj)))
    return {"radbcw": radbcw, "radbce": radbce}


def radbcns4(bk, g, cfg=None, *, rvf, radbcs, radbcn, v1, v2, dt):
    """bc.F ``radbcns4``: as radbcew4 for the south/north boundaries.
    radbcs/radbcn are ``[k, i]``. Returns {'radbcs', 'radbcn'}."""
    _, _, sbc, nbc = _bcs(cfg)
    ni, nj, nk = g.ni, g.nj, g.nk
    K, I = _k(1, nk), _fs(1, ni)
    radbcs, radbcn = bk.copy(radbcs), bk.copy(radbcn)
    xp = bk.xp
    if sbc == 2:
        foo = v1[K, 3 + 2, I] - v1[K, 2 + 2, I]
        num = g.dy * rvf[2 + 2] * (v1[K, 2 + 2, I] - v2[K, 2 + 2, I])
        avg = _rad_avg(bk, g, cfg, num, foo, dt, lo=True)
        radbcs = bk.set(radbcs, (K, I), xp.broadcast_to(avg[None, :], (nk, ni)))
    if nbc == 2:
        foo = v1[K, nj + 2, I] - v1[K, nj - 1 + 2, I]
        num = g.dy * rvf[nj + 2] * (v1[K, nj + 2, I] - v2[K, nj + 2, I])
        avg = _rad_avg(bk, g, cfg, num, foo, dt, lo=False)
        radbcn = bk.set(radbcn, (K, I), xp.broadcast_to(avg[None, :], (nk, ni)))
    return {"radbcs": radbcs, "radbcn": radbcn}


def _terrain_on(cfg):
    return bool(_opt(cfg, "terrain_flag"))


def _at_w(bk, a, nk, tm):
    """bc.F ssopenbc* (terrain): ``a`` (u or v, k axis first, Python index ==
    Fortran k) at w levels k = 1..nk+1 (returned index 0 == k=1): cgs/cgt
    extrapolation at k=1 and nk+1, r1*a(k-1)+r2*a(k) inside with
    r2 = (sigmaf(k)-sigma(k-1))*rds(k)."""
    xp = bk.xp
    S = bk.asarray
    cg = {k: S(float(tm[k])) for k in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}
    lo = cg["cgs1"] * a[1] + cg["cgs2"] * a[2] + cg["cgs3"] * a[3]
    hi = cg["cgt1"] * a[nk] + cg["cgt2"] * a[nk - 1] + cg["cgt3"] * a[nk - 2]
    sh = (nk - 1,) + (1,) * (a.ndim - 1)
    r2 = xp.reshape(
        (tm["sigmaf"][2 : nk + 1] - tm["sigma"][1:nk]) * tm["rds"][2 : nk + 1], sh
    )
    r1 = S(1.0) - r2
    mid = r1 * a[1:nk] + r2 * a[2 : nk + 1]
    return xp.concat([lo[None], mid, hi[None]], axis=0)


def _need_tm(who, tm, keys):
    if tm is None or any(tm.get(k) is None for k in keys):
        raise ValueError(
            f"{who}: terrain_flag=.true. needs terrain={{{', '.join(keys)}}}"
        )


_TM_U = ("rds", "sigma", "rdsf", "sigmaf", "gz", "rgzu", "gx", "cgs1", "cgt1")
_TM_V = ("rds", "sigma", "rdsf", "sigmaf", "gz", "rgzv", "gy", "cgs1", "cgt1")


def _ssopen_x(bk, g, tm, u3d, uten, rad, dts, uh, east):
    """Terrain branch of ssopenbcw (east=False) / ssopenbce (east=True)."""
    nk = g.nk
    K, J = _k(1, nk), _fs(1, g.nj)
    b = (g.ni + 1 + 2) if east else (1 + 2)
    i0, i1 = (b - 1, b) if east else (b, b + 1)  # the two u columns used
    ic = b - 1 if east else b  # scalar column (gz, gx, uh): ni or 1
    d = _at_w(bk, u3d[:, J, i0 : i1 + 1], nk, tm)  # [k=1..nk+1, j, 2]
    dsum = d[..., 1] + d[..., 0]  # dum1(i0+1) + dum1(i0)
    gxc = tm["gx"][1 : nk + 2, J, ic]
    gz, rgzu = tm["gz"], tm["rgzu"]
    hx = (
        gz[J, ic][None]
        * (u3d[K, J, i1] * rgzu[J, i1][None] - u3d[K, J, i0] * rgzu[J, i0][None])
        * g.rdx
        * uh[ic]
    )
    vz = (
        0.5
        * (gxc[1:] * dsum[1:] - gxc[:-1] * dsum[:-1])
        * tm["rdsf"][1 : nk + 1][:, None]
    )
    new = u3d[K, J, b] + dts * (-rad[K, J] * (hx + vz) + uten[K, J, b])
    return bk.set(u3d, (K, J, b), new)


def _ssopen_y(bk, g, tm, v3d, vten, rad, dts, vh, north):
    """Terrain branch of ssopenbcs (north=False) / ssopenbcn (north=True)."""
    nk = g.nk
    K, I = _k(1, nk), _fs(1, g.ni)
    b = (g.nj + 1 + 2) if north else (1 + 2)
    j0, j1 = (b - 1, b) if north else (b, b + 1)
    jc = b - 1 if north else b
    d = _at_w(bk, v3d[:, j0 : j1 + 1, I], nk, tm)  # [k=1..nk+1, 2, i]
    dsum = d[:, 1] + d[:, 0]
    gyc = tm["gy"][1 : nk + 2, jc, I]
    gz, rgzv = tm["gz"], tm["rgzv"]
    hy = (
        gz[jc, I][None]
        * (v3d[K, j1, I] * rgzv[j1, I][None] - v3d[K, j0, I] * rgzv[j0, I][None])
        * g.rdy
        * vh[jc]
    )
    vz = (
        0.5
        * (gyc[1:] * dsum[1:] - gyc[:-1] * dsum[:-1])
        * tm["rdsf"][1 : nk + 1][:, None]
    )
    new = v3d[K, b, I] + dts * (-rad[K, I] * (hy + vz) + vten[K, b, I])
    return bk.set(v3d, (K, b, I), new)


def ssopenbcw(bk, g, cfg=None, *, uh, radbcw, u3d, uten, dts, terrain=None):
    """bc.F ``ssopenbcw``: small-step update of u(1, 1:nj, 1:nk).  With
    terrain_flag, ``terrain`` is a dict with rds, sigma, rdsf, sigmaf, gz,
    rgzu, gx and the extrapolation coefficients cgs1..3, cgt1..3."""
    if _terrain_on(cfg):
        _need_tm("ssopenbcw", terrain, _TM_U)
        return _ssopen_x(bk, g, terrain, u3d, uten, radbcw, dts, uh, east=False)
    K, J, b = _k(1, g.nk), _fs(1, g.nj), 1 + 2
    new = u3d[K, J, b] + dts * (
        -radbcw[K, J] * (u3d[K, J, b + 1] - u3d[K, J, b]) * g.rdx * uh[b]
        + uten[K, J, b]
    )
    return bk.set(u3d, (K, J, b), new)


def ssopenbce(bk, g, cfg=None, *, uh, radbce, u3d, uten, dts, terrain=None):
    """bc.F ``ssopenbce``: small-step update of u(ni+1, :, :) (``terrain`` as
    in ssopenbcw)."""
    if _terrain_on(cfg):
        _need_tm("ssopenbce", terrain, _TM_U)
        return _ssopen_x(bk, g, terrain, u3d, uten, radbce, dts, uh, east=True)
    K, J, b = _k(1, g.nk), _fs(1, g.nj), g.ni + 1 + 2
    new = u3d[K, J, b] + dts * (
        -radbce[K, J] * (u3d[K, J, b] - u3d[K, J, b - 1]) * g.rdx * uh[b - 1]
        + uten[K, J, b]
    )
    return bk.set(u3d, (K, J, b), new)


def ssopenbcs(bk, g, cfg=None, *, vh, radbcs, v3d, vten, dts, terrain=None):
    """bc.F ``ssopenbcs``: small-step update of v(1:ni, 1, 1:nk).  With
    terrain_flag, ``terrain`` holds rds, sigma, rdsf, sigmaf, gz, rgzv, gy and
    cgs1..3, cgt1..3."""
    if _terrain_on(cfg):
        _need_tm("ssopenbcs", terrain, _TM_V)
        return _ssopen_y(bk, g, terrain, v3d, vten, radbcs, dts, vh, north=False)
    K, I, b = _k(1, g.nk), _fs(1, g.ni), 1 + 2
    new = v3d[K, b, I] + dts * (
        -radbcs[K, I] * (v3d[K, b + 1, I] - v3d[K, b, I]) * g.rdy * vh[b]
        + vten[K, b, I]
    )
    return bk.set(v3d, (K, b, I), new)


def ssopenbcn(bk, g, cfg=None, *, vh, radbcn, v3d, vten, dts, terrain=None):
    """bc.F ``ssopenbcn``: small-step update of v(:, nj+1, :) (``terrain`` as
    in ssopenbcs)."""
    if _terrain_on(cfg):
        _need_tm("ssopenbcn", terrain, _TM_V)
        return _ssopen_y(bk, g, terrain, v3d, vten, radbcn, dts, vh, north=True)
    K, I, b = _k(1, g.nk), _fs(1, g.ni), g.nj + 1 + 2
    new = v3d[K, b, I] + dts * (
        -radbcn[K, I] * (v3d[K, b, I] - v3d[K, b - 1, I]) * g.rdy * vh[b - 1]
        + vten[K, b, I]
    )
    return bk.set(v3d, (K, b, I), new)


# ---------------------------------------------------------------------------
# helpers from param.F / misclibs.F that go with the open boundaries


def dtu0_dtv0(bk, g, cfg=None):
    """param.F: the small-step masks. dtu0 = 1 on i=1..ni+1, j=1..nj and dtv0
    = 1 on i=1..ni, j=1..nj+1, zero elsewhere; additionally zero on
    non-periodic edges (u(1), u(ni+1), v(1), v(nj+1) are then advanced by
    ssopenbc* or held fixed). Both are ``[j, i]`` (s_shape[1:])."""
    ibw, ibe, ibs, ibn = ib_flags(cfg)
    ni, nj = g.ni, g.nj
    dtu0 = bk.set(bk.zeros(g.s_shape[1:]), (_fs(1, nj), _fs(1, ni + 1)), 1.0)
    if ibw:
        dtu0 = bk.set(dtu0, (_fs(1, nj), 1 + 2), 0.0)
    if ibe:
        dtu0 = bk.set(dtu0, (_fs(1, nj), ni + 1 + 2), 0.0)
    dtv0 = bk.set(bk.zeros(g.s_shape[1:]), (_fs(1, nj + 1), _fs(1, ni)), 1.0)
    if ibs:
        dtv0 = bk.set(dtv0, (1 + 2, _fs(1, ni)), 0.0)
    if ibn:
        dtv0 = bk.set(dtv0, (nj + 1 + 2, _fs(1, ni)), 0.0)
    return dtu0, dtv0


def calccflquick(bk, g, cfg=None, *, dt, uh, vh, mh, ua, va, wa, nx=None, ny=None):
    """misclibs.F ``calccflquick``: max advective CFL number (diagnostic; CM1
    sets stopit when it reaches 1.5). Returns a 0-d array (``cflmax``)."""
    xp = bk.xp
    nx = g.ni if nx is None else nx
    ny = g.nj if ny is None else ny
    dtdx, dtdy, dtdz = 0.5 * dt * g.rdx, 0.5 * dt * g.rdy, 0.5 * dt * g.rdz
    K, J, I = _k(1, g.nk), _fs(1, g.nj), _fs(1, g.ni)
    cu = (ua[K, J, I] + ua[K, J, 4 : g.ni + 4]) * dtdx * xp.reshape(uh[I], (1, 1, -1))
    cv = (va[K, J, I] + va[K, 4 : g.nj + 4, I]) * dtdy * xp.reshape(vh[J], (1, -1, 1))
    cw = (wa[K, J, I] + wa[2 : g.nk + 2, J, I]) * dtdz * mh[K, J, I]
    if nx > 1 and ny > 1:
        wsp = xp.sqrt((cu**2 + cv**2) + cw**2)
    elif nx > 1:
        wsp = xp.sqrt(cu**2 + cw**2)
    elif int(_opt(cfg, "axisymm")) == 0 and ny > 1:
        wsp = xp.sqrt(cv**2 + cw**2)
    else:
        return bk.asarray(0.0)
    return xp.maximum(xp.max(wsp), bk.asarray(0.0))
