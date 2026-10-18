"""Subgrid-turbulence tendencies, ported from src/turbtend.F.

Routines: ``turbs`` (+ ``turbsz``), ``turbt``, ``turbu`` (+ ``turbuz``),
``turbv`` (+ ``turbvz``), ``turbw``.  Path ported: Cartesian (axisymm=0), no
terrain, cm1setup=1 (LES boundary conditions in turbsz), dohturb=dovturb=.true.,
no surface fluxes (dosfcflx=.false.), no budget diagnostics, doimpl=0 or 1,
any of the lateral boundary conditions wbc..nbc = 1..4 (only the open-boundary
branch, 2, changes anything in these routines).

Layout (cm1x.grid): ``a[k, j, i]`` with halos; a Fortran index ``i`` is Python
``i + ngxy - 1``, ``k`` is ``k + ngz - 1``.  Shapes follow CM1's declarations:
scalars and ``rho, rr, rf, t11..t33, dum7, dum8`` are ``g.s_shape``; ``mf``,
``kmh, kmv, khh, khv`` (``ibc:iec,..,kbc:kec`` with kec = ke+1) and the TKE
arrays ``t, tten`` (``ibt:..:ket``) are ``g.w_shape``.  ``uh, ruh`` are 1-D over
the i scalar points (``s_shape[2]``), ``uf`` over u points (``u_shape[2]``);
likewise ``vh, rvh`` / ``vf`` in j.

Every expression keeps the Fortran operation order, so numpy agrees with the
double-precision Fortran to round-off (in practice bitwise).  Vertical
tridiagonal solves are Python loops over k on ``[j, i]`` slabs.
"""

from __future__ import annotations

vialpha = 1.0  # constants.F (implicit Euler for the implicit vertical diffusion)
vibeta = 1.0 - vialpha


def _opt(cfg, name, default):
    if cfg is None:
        return default
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def _bcflags(cfg):
    """Lateral BC options and the serial-run ibw.. flags (1 iff not periodic)."""
    b = {n: int(_opt(cfg, n, 1)) for n in ("wbc", "ebc", "sbc", "nbc")}
    b["ibw"] = int(b["wbc"] != 1)
    b["ibe"] = int(b["ebc"] != 1)
    b["ibs"] = int(b["sbc"] != 1)
    b["ibn"] = int(b["nbc"] != 1)
    return b


def _check(cfg, who):
    bad = []
    for name, ok in (("axisymm", (0,)), ("cm1setup", (1,))):
        v = int(_opt(cfg, name, ok[0]))
        if v not in ok:
            bad.append(f"{name}={v}")
    if _opt(cfg, "dosfcflx", False):
        bad.append("dosfcflx=T")
    if bad:
        raise NotImplementedError(f"{who}: options not ported: " + ", ".join(bad))


class Reg:
    """A Fortran index box ``k=k0..k1, j=j0..j1, i=i0..i1`` (inclusive).

    ``r(a, dk, dj, di)`` is the slab of ``a`` at ``(k+dk, j+dj, i+di)``;
    ``r.x(a1, di)`` / ``r.y(a1, dj)`` broadcast a 1-D i / j array; ``r.idx``
    is the index tuple for ``bk.set``.
    """

    def __init__(self, g, k, j, i):
        self.g = g
        self.k, self.j, self.i = k, j, i

    def _h(self, lo, hi, d):
        n = self.g.ngxy
        return slice(lo + d + n - 1, hi + d + n)

    def _v(self, lo, hi, d):
        z = self.g.ngz
        return slice(lo + d + z - 1, hi + d + z)

    def __call__(self, a, dk=0, dj=0, di=0):
        return a[self._v(*self.k, dk), self._h(*self.j, dj), self._h(*self.i, di)]

    def x(self, a1, di=0):
        return a1[self._h(*self.i, di)][None, None, :]

    def y(self, a1, dj=0):
        return a1[self._h(*self.j, dj)][None, :, None]

    @property
    def idx(self):
        return (self._v(*self.k, 0), self._h(*self.j, 0), self._h(*self.i, 0))

    def kidx(self, k):
        """Index of one level k (Fortran) over this box's j, i."""
        return (k + self.g.ngz - 1, self._h(*self.j, 0), self._h(*self.i, 0))

    def m2(self, a2, dj=0, di=0):
        """A 2-D ``[j, i]`` terrain metric at (j+dj, i+di), broadcast over k."""
        return a2[self._h(*self.j, dj), self._h(*self.i, di)][None]

    def z(self, a1, dk=0):
        """A 1-D vertical array (rds, sigma, rdsf, sigmaf; Python index ==
        Fortran k) at k+dk, broadcast over j, i."""
        return a1[self._v(*self.k, dk)][:, None, None]


# ---------------------------------------------------------------------------
# terrain helpers

_TM_KEYS = ("rds", "sigma", "rdsf", "sigmaf", "gz", "rgz", "zt", "cgs1", "cgt1")


def _terrain(cfg, terrain, who):
    """The terrain metrics dict when terrain_flag (else None).  ``terrain``
    holds rds, sigma (nk+2), rdsf, sigmaf (nk+3), gz, rgz, gzu, rgzu, gzv,
    rgzv (2-D ``[j, i]``), gx, gxu, gy, gyv (``g.s_shape``), the Python
    float zt and the extrapolation coefficients cgs1..3, cgt1..3."""
    if not _opt(cfg, "terrain_flag", False):
        return None
    if terrain is None or any(terrain.get(k) is None for k in _TM_KEYS):
        raise ValueError(f"{who}: terrain_flag=.true. needs terrain={{...}}")
    return terrain


def _atw(bk, g, tm, a, j, i, shape=None):
    """``a`` (scalar levels) interpolated to w levels k=1..nk+1 on the box
    j=j0..j1, i=i0..i1 as CM1's terrain code does (cgs/cgt extrapolation at
    k=1 and nk+1, r1*a(k-1)+r2*a(k) with r2=(sigmaf(k)-sigma(k-1))*rds(k)
    inside), stored into a zero array of ``shape`` (default ``g.w_shape``)."""
    xp = bk.xp
    nk = g.nk
    S = bk.asarray
    cg = {n: S(float(tm[n])) for n in ("cgs1", "cgs2", "cgs3", "cgt1", "cgt2", "cgt3")}
    L = lambda k0, k1: Reg(g, (k0, k1), j, i)(a)  # noqa: E731
    lo = cg["cgs1"] * L(1, 1) + cg["cgs2"] * L(2, 2) + cg["cgs3"] * L(3, 3)
    hi = (
        cg["cgt1"] * L(nk, nk)
        + cg["cgt2"] * L(nk - 1, nk - 1)
        + cg["cgt3"] * L(nk - 2, nk - 2)
    )
    R = Reg(g, (2, nk), j, i)
    r2 = (R.z(tm["sigmaf"]) - R.z(tm["sigma"], dk=-1)) * R.z(tm["rds"])
    r1 = 1.0 - r2
    mid = r1 * L(1, nk - 1) + r2 * L(2, nk)
    out = bk.zeros(shape or g.w_shape)
    return bk.set(out, Reg(g, (1, nk + 1), j, i).idx, _cat(xp, lo, mid, hi))


def _setcol(bk, g, a, i_dst, i_src, k, j):
    """a(i_dst, j0..j1, k0..k1) = a(i_src, ...) (Fortran indices)."""
    r = Reg(g, k, j, (i_dst, i_dst))
    s = Reg(g, k, j, (i_src, i_src))
    return bk.set(a, r.idx, s(a))


def _setrow(bk, g, a, j_dst, j_src, k, i):
    r = Reg(g, k, (j_dst, j_dst), i)
    s = Reg(g, k, (j_src, j_src), i)
    return bk.set(a, r.idx, s(a))


def _zcol(bk, g, a, i_dst, k, j):
    r = Reg(g, k, j, (i_dst, i_dst))
    return bk.set(a, r.idx, 0.0)


def _zrow(bk, g, a, j_dst, k, i):
    r = Reg(g, k, (j_dst, j_dst), i)
    return bk.set(a, r.idx, 0.0)


def _open_zero(bk, g, b, turbx, turby, k, ix_e, jy_n, jx=None, iy=None):
    """Open-boundary zeroing of turbx at i=1 / i=ix_e and turby at j=1 / j=jy_n."""
    nj, ni = g.nj, g.ni
    jx = jx or (1, nj)
    iy = iy or (1, ni)
    if b["wbc"] == 2 and b["ibw"] == 1:
        turbx = _zcol(bk, g, turbx, 1, k, jx)
    if b["ebc"] == 2 and b["ibe"] == 1:
        turbx = _zcol(bk, g, turbx, ix_e, k, jx)
    if b["sbc"] == 2 and b["ibs"] == 1:
        turby = _zrow(bk, g, turby, 1, k, iy)
    if b["nbc"] == 2 and b["ibn"] == 1:
        turby = _zrow(bk, g, turby, jy_n, k, iy)
    return turbx, turby


def _stack(xp, levels):
    return xp.stack(levels, axis=0)


def _cat(xp, *parts):
    return xp.concat(parts, axis=0)


def _tridiag(bk, cfa, cfb, cfc, cfd):
    """Thomas algorithm on level-stacked coefficients (axis 0 = the n
    unknowns, bottom to top), exactly CM1's recurrences::

        tem = 1/cfb(1);  d1(1) = -cfc(1) tem;  d2(1) = cfd(1) tem
        tem = 1/(cfa(k) d1(k-1) + cfb(k));  d1(k) = -cfc(k) tem
        d2(k) = (cfd(k) - cfa(k) d2(k-1)) tem          (k = 2..n)
        x(n) = d2(n);  x(k) = d1(k) x(k+1) + d2(k)     (k = n-1..1)

    (``cfa[0]`` and ``cfc[-1]`` are not used.)  Both sweeps run through
    ``Backend.scan``.  Returns ``x`` stacked like the inputs."""
    xp = bk.xp
    tem = 1.0 / cfb[0]
    c0 = (-cfc[0] * tem, cfd[0] * tem)

    def fwd(c, x):
        a, b, cc, d = x
        tem = 1.0 / (a * c[0] + b)
        n = (-cc * tem, (d - a * c[1]) * tem)
        return n, n

    (_, xn), (d1, d2) = bk.scan(fwd, c0, (cfa[1:], cfb[1:], cfc[1:], cfd[1:]))
    d1 = _cat(xp, c0[0][None], d1[:-1])
    d2 = _cat(xp, c0[1][None], d2[:-1])

    def bwd(xnext, x):
        v = x[0] * xnext + x[1]
        return v, v

    _, xs = bk.scan(bwd, xn, (d1, d2), reverse=True)
    return _cat(xp, xs, xn[None])


# ---------------------------------------------------------------------------
# turbs / turbsz


def turbsz(
    bk, g, cfg=None, *, iflux, dt, mh, mf, rho, rr, rf, s, khv, dum7, dum8, sflux=None
):
    """turbtend.F ``turbsz`` (cm1setup=1, no surface flux): vertical subgrid
    tendency ``turbz`` (k=1..nk, i=1..ni, j=1..nj), returned as the interior
    ``[nk, nj, ni]`` array."""
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    doimpl = int(_opt(cfg, "doimpl", 1))
    bcturbs = int(_opt(cfg, "bcturbs", 1))
    rdz = g.rdz
    J, I = (1, nj), (1, ni)

    def lev(a, k):
        return Reg(g, (k, k), J, I)(a)[0]

    if doimpl == 0:
        dz_ = [None] * (nk + 2)
        for k in range(2, nk + 1):
            dz_[k] = (
                -lev(khv, k)
                * (lev(s, k) - lev(s, k - 1))
                * rdz
                * lev(mf, k)
                * lev(rf, k)
            )
        if bcturbs == 1:
            z = xp.zeros_like(lev(s, 1))
            dz_[1] = z
            dz_[nk + 1] = z
        else:
            dz_[1] = dz_[2]
            dz_[nk + 1] = dz_[nk]
        tz = [-(dz_[k + 1] - dz_[k]) * rdz * lev(mh, k) for k in range(1, nk + 1)]
        return _stack(xp, tz)

    rdt = 1.0 / dt
    tema = -1.0 * dt * vialpha * rdz * rdz
    temb = dt * vibeta * rdz * rdz
    temc = dt * rdz
    if bcturbs == 1:
        dzb = xp.zeros_like(lev(s, 1))
        dzt = dzb
    else:
        dzb = -lev(khv, 2) * (lev(s, 2) - lev(s, 1)) * rdz * lev(mf, 2) * lev(rf, 2)
        dzt = (
            -lev(khv, nk)
            * (lev(s, nk) - lev(s, nk - 1))
            * rdz
            * lev(mf, nk)
            * lev(rf, nk)
        )

    def L(a, k0, k1):  # levels k0..k1 (Fortran) over j, i
        return Reg(g, (k0, k1), J, I)(a)

    zz = xp.zeros_like(L(s, 1, 1))
    # k = 1
    r2 = L(dum8, 1, 1)
    cfc1 = tema * r2
    cfb1 = 1.0 - cfc1
    cfd1 = (
        L(s, 1, 1)
        + temb * (-r2 * L(s, 1, 1) + r2 * L(s, 2, 2))
        + temc * dzb * L(mh, 1, 1) * L(rr, 1, 1)
    )
    # k = 2..nk-1
    r1 = L(dum7, 2, nk - 1)
    r2 = L(dum8, 2, nk - 1)
    cfa = tema * r1
    cfc = tema * r2
    cfb = 1.0 - cfa - cfc
    cfd = L(s, 2, nk - 1) + temb * (
        r1 * L(s, 1, nk - 2) - (r1 + r2) * L(s, 2, nk - 1) + r2 * L(s, 3, nk)
    )
    # k = nk
    r1 = L(dum7, nk, nk)
    cfan = tema * r1
    cfbn = 1.0 - cfan
    cfdn = (
        L(s, nk, nk)
        + temb * (r1 * L(s, nk - 1, nk - 1) - r1 * L(s, nk, nk))
        - temc * dzt * L(mh, nk, nk) * L(rr, nk, nk)
    )
    dz_ = _tridiag(
        bk,
        _cat(xp, zz, cfa, cfan),
        _cat(xp, cfb1, cfb, cfbn),
        _cat(xp, cfc1, cfc, zz),
        _cat(xp, cfd1, cfd, cfdn),
    )
    return L(rho, 1, nk) * (dz_ - L(s, 1, nk)) * rdt


def turbs(
    bk,
    g,
    cfg=None,
    *,
    iflux,
    dt,
    uh,
    uf,
    vh,
    vf,
    mh,
    mf,
    rho,
    rr,
    rf,
    s,
    sten,
    khh,
    khv,
    dum7=None,
    dum8=None,
    sflux=None,
    terrain=None,
    **unused,
):
    """turbtend.F ``turbs``: ``sten(1:ni,1:nj,1:nk) +=`` subgrid diffusion of
    ``s``.  ``dum7/dum8`` are solve1's vimpl coefficients (see
    ``turb.vimpl_coefs``; required when doimpl=1).  With terrain_flag,
    ``terrain`` holds the metrics (see ``_terrain``).  Returns {"sten"}."""
    _check(cfg, "turbs")
    tm = _terrain(cfg, terrain, "turbs")
    nk, nj, ni = g.nk, g.nj, g.ni
    b = _bcflags(cfg)
    K = (1, nk)
    R = Reg(g, K, (1, nj), (1, ni))
    if tm is not None:
        tx, ty = _turbs_terrain_h(bk, g, tm, s, rho, khh, uh, uf, vh, vf)
    else:
        tx, ty = _turbs_flat_h(bk, g, b, s, rho, khh, uh, uf, vh, vf)
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.s_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.s_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni, nj)
        tx, ty = R(tx), R(ty)

    tz = turbsz(
        bk,
        g,
        cfg,
        iflux=iflux,
        dt=dt,
        mh=mh,
        mf=mf,
        rho=rho,
        rr=rr,
        rf=rf,
        s=s,
        khv=khv,
        dum7=dum7,
        dum8=dum8,
        sflux=sflux,
    )
    out = R(sten) + ((tx + ty) + tz) * R(rr)
    return {"sten": bk.set(bk.copy(sten), R.idx, out)}


def _turbs_terrain_h(bk, g, tm, s, rho, khh, uh, uf, vh, vf):
    """Horizontal part of turbs, Cartesian with terrain: turbx, turby on
    k=1..nk, j=1..nj, i=1..ni."""
    nk, nj, ni = g.nk, g.nj, g.ni
    rdx, rdy = g.rdx, g.rdy
    K = (1, nk)
    gz, rgz, rgzu, rgzv = tm["gz"], tm["rgz"], tm["rgzu"], tm["rgzv"]
    rdsf = tm["rdsf"]
    # s at w points (j, i = 0..n+1)
    sw = _atw(bk, g, tm, s, (0, nj + 1), (0, ni + 1))
    X = Reg(g, K, (1, nj), (1, ni + 1))
    dumx = (
        -0.125
        * (X.m2(gz) * X(rho) + X.m2(gz, di=-1) * X(rho, di=-1))
        * ((X(khh) + X(khh, di=-1)) + (X(khh, dk=1) + X(khh, dk=1, di=-1)))
        * (
            (X(s) * X.m2(rgz) - X(s, di=-1) * X.m2(rgz, di=-1)) * rdx * X.x(uf)
            + 0.5
            * (
                X(tm["gxu"], dk=1) * (X(sw, dk=1) + X(sw, dk=1, di=-1))
                - X(tm["gxu"]) * (X(sw) + X(sw, di=-1))
            )
            * X.z(rdsf)
            * X.m2(rgzu)
        )
    )
    Y = Reg(g, K, (1, nj + 1), (1, ni))
    dumy = (
        -0.125
        * (Y.m2(gz) * Y(rho) + Y.m2(gz, dj=-1) * Y(rho, dj=-1))
        * ((Y(khh) + Y(khh, dj=-1)) + (Y(khh, dk=1) + Y(khh, dk=1, dj=-1)))
        * (
            (Y(s) * Y.m2(rgz) - Y(s, dj=-1) * Y.m2(rgz, dj=-1)) * rdy * Y.y(vf)
            + 0.5
            * (
                Y(tm["gyv"], dk=1) * (Y(sw, dk=1) + Y(sw, dk=1, dj=-1))
                - Y(tm["gyv"]) * (Y(sw) + Y(sw, dj=-1))
            )
            * Y.z(rdsf)
            * Y.m2(rgzv)
        )
    )
    wx = bk.set(bk.zeros(g.s_shape), X.idx, dumx)
    wy = bk.set(bk.zeros(g.s_shape), Y.idx, dumy)
    # the fluxes at w points (the points turbx / turby use)
    fxw = _atw(bk, g, tm, wx, (1, nj), (1, ni + 1))
    fyw = _atw(bk, g, tm, wy, (1, nj + 1), (1, ni))
    R = Reg(g, K, (1, nj), (1, ni))
    tx = -R.m2(gz) * (R(wx, di=1) * R.m2(rgzu, di=1) - R(wx) * R.m2(rgzu)) * rdx * R.x(
        uh
    ) - (
        R(tm["gx"], dk=1) * (R(fxw, dk=1) + R(fxw, dk=1, di=1))
        - R(tm["gx"]) * (R(fxw) + R(fxw, di=1))
    ) * 0.5 * R.z(rdsf)
    ty = -R.m2(gz) * (R(wy, dj=1) * R.m2(rgzv, dj=1) - R(wy) * R.m2(rgzv)) * rdy * R.y(
        vh
    ) - (
        R(tm["gy"], dk=1) * (R(fyw, dk=1) + R(fyw, dk=1, dj=1))
        - R(tm["gy"]) * (R(fyw) + R(fyw, dj=1))
    ) * 0.5 * R.z(rdsf)
    return tx, ty


def _turbs_flat_h(bk, g, b, s, rho, khh, uh, uf, vh, vf):
    """Horizontal part of turbs, Cartesian without terrain."""
    nk, nj, ni = g.nk, g.nj, g.ni
    rdx, rdy = g.rdx, g.rdy
    K = (1, nk)

    F = Reg(g, K, (1, nj + 1), (1, ni + 1))
    dumx = (
        -0.125
        * (F(rho) + F(rho, di=-1))
        * ((F(khh) + F(khh, di=-1)) + (F(khh, dk=1) + F(khh, dk=1, di=-1)))
        * (F(s) - F(s, di=-1))
        * rdx
        * F.x(uf)
    )
    dumy = (
        -0.125
        * (F(rho) + F(rho, dj=-1))
        * ((F(khh) + F(khh, dj=-1)) + (F(khh, dk=1) + F(khh, dk=1, dj=-1)))
        * (F(s) - F(s, dj=-1))
        * rdy
        * F.y(vf)
    )
    # work on full-size copies so the boundary copies index like the Fortran
    wx = bk.set(bk.zeros(g.s_shape), F.idx, dumx)
    wy = bk.set(bk.zeros(g.s_shape), F.idx, dumy)
    if b["wbc"] == 2 and b["ibw"] == 1:
        wx = _setcol(bk, g, wx, 1, 2, K, (1, nj))
    if b["ebc"] == 2 and b["ibe"] == 1:
        wx = _setcol(bk, g, wx, ni + 1, ni, K, (1, nj))
    if b["sbc"] == 2 and b["ibs"] == 1:
        wy = _setrow(bk, g, wy, 1, 2, K, (1, ni))
    if b["nbc"] == 2 and b["ibn"] == 1:
        wy = _setrow(bk, g, wy, nj + 1, nj, K, (1, ni))
    R = Reg(g, K, (1, nj), (1, ni))
    tx = -(R(wx, di=1) - R(wx)) * rdx * R.x(uh)
    ty = -(R(wy, dj=1) - R(wy)) * rdy * R.y(vh)
    return tx, ty


# ---------------------------------------------------------------------------
# turbt (TKE, w levels)


def turbt(
    bk,
    g,
    cfg=None,
    *,
    dt,
    uh,
    uf,
    vh,
    vf,
    mh,
    mf,
    rho,
    rr,
    rf,
    t,
    tten,
    kmh,
    kmv,
    terrain=None,
    **unused,
):
    """turbtend.F ``turbt``: ``tten(1:ni,1:nj,2:nk) +=`` subgrid diffusion of
    the w-level scalar ``t`` (TKE).  ``terrain``: metrics with terrain_flag.
    Returns {"tten"}."""
    _check(cfg, "turbt")
    tm = _terrain(cfg, terrain, "turbt")
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbt: nk >= 3 required")
    b = _bcflags(cfg)
    doimpl = int(_opt(cfg, "doimpl", 1))
    bbc = int(_opt(cfg, "bbc", 1))
    K = (2, nk)
    R = Reg(g, K, (1, nj), (1, ni))
    if tm is not None:
        tx, ty = _turbt_terrain_h(bk, g, tm, t, rf, kmh, uh, uf, vh, vf)
    else:
        tx, ty = _turbt_flat_h(bk, g, b, t, rf, kmh, uh, uf, vh, vf)
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.s_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.s_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni, nj)
        tx, ty = R(tx), R(ty)
    return _turbt_z(bk, g, R, doimpl, bbc, dt, t, tten, tx, ty, mh, mf, rho, rf, kmv)


def _turbt_terrain_h(bk, g, tm, t, rf, kmh, uh, uf, vh, vf):
    """Horizontal part of turbt with terrain (k=2..nk, j=1..nj, i=1..ni)."""
    nk, nj, ni = g.nk, g.nj, g.ni
    rdx, rdy = g.rdx, g.rdy
    zt = float(tm["zt"])
    gz, rgz, rgzu, rgzv = tm["gz"], tm["rgz"], tm["rgzu"], tm["rgzv"]
    sig, rds = tm["sigma"], tm["rds"]
    K = (2, nk)
    # t at scalar points (k = 1..nk, j, i = 0..n+1)
    A = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
    ts = bk.set(bk.zeros(g.s_shape), A.idx, 0.5 * (A(t) + A(t, dk=1)))
    X = Reg(g, K, (1, nj), (1, ni + 1))
    dumx = (
        -0.25
        * (X.m2(gz) * X(rf) + X.m2(gz, di=-1) * X(rf, di=-1))
        * 2.0
        * (X(kmh) + X(kmh, di=-1))
        * (
            (X(t) * X.m2(rgz) - X(t, di=-1) * X.m2(rgz, di=-1)) * rdx * X.x(uf)
            + 0.5
            * (
                (zt - X.z(sig)) * (X(ts, di=-1) + X(ts))
                - (zt - X.z(sig, dk=-1)) * (X(ts, dk=-1, di=-1) + X(ts, dk=-1))
            )
            * X.z(rds)
            * (X.m2(rgz) - X.m2(rgz, di=-1))
            * rdx
            * X.x(uf)
        )
    )
    Y = Reg(g, K, (1, nj + 1), (1, ni))
    dumy = (
        -0.25
        * (Y.m2(gz) * Y(rf) + Y.m2(gz, dj=-1) * Y(rf, dj=-1))
        * 2.0
        * (Y(kmh) + Y(kmh, dj=-1))
        * (
            (Y(t) * Y.m2(rgz) - Y(t, dj=-1) * Y.m2(rgz, dj=-1)) * rdy * Y.y(vf)
            + 0.5
            * (
                (zt - Y.z(sig)) * (Y(ts, dj=-1) + Y(ts))
                - (zt - Y.z(sig, dk=-1)) * (Y(ts, dk=-1, dj=-1) + Y(ts, dk=-1))
            )
            * Y.z(rds)
            * (Y.m2(rgz) - Y.m2(rgz, dj=-1))
            * rdy
            * Y.y(vf)
        )
    )
    # dumx, dumy on k = 1..nk+1 (0 at k = 1 and nk+1)
    wx = bk.set(bk.zeros(g.s_shape), X.idx, dumx)
    wy = bk.set(bk.zeros(g.s_shape), Y.idx, dumy)
    # at scalar points (k = 1..nk)
    S1 = Reg(g, (1, nk), (1, nj), (1, ni + 1))
    S2 = Reg(g, (1, nk), (1, nj + 1), (1, ni))
    fx = bk.set(bk.zeros(g.s_shape), S1.idx, 0.5 * (S1(wx) + S1(wx, dk=1)))
    fy = bk.set(bk.zeros(g.s_shape), S2.idx, 0.5 * (S2(wy) + S2(wy, dk=1)))
    R = Reg(g, K, (1, nj), (1, ni))
    tx = -(R(wx, di=1) * R.m2(rgzu, di=1) - R(wx) * R.m2(rgzu)) * R.m2(gz) * rdx * R.x(
        uh
    ) - 0.5 * (
        (zt - R.z(sig)) * (R(fx) + R(fx, di=1))
        - (zt - R.z(sig, dk=-1)) * (R(fx, dk=-1) + R(fx, dk=-1, di=1))
    ) * R.z(rds) * (R.m2(rgzu, di=1) - R.m2(rgzu)) * R.m2(gz) * rdx * R.x(uh)
    ty = -(R(wy, dj=1) * R.m2(rgzv, dj=1) - R(wy) * R.m2(rgzv)) * R.m2(gz) * rdy * R.y(
        vh
    ) - 0.5 * (
        (zt - R.z(sig)) * (R(fy) + R(fy, dj=1))
        - (zt - R.z(sig, dk=-1)) * (R(fy, dk=-1) + R(fy, dk=-1, dj=1))
    ) * R.z(rds) * (R.m2(rgzv, dj=1) - R.m2(rgzv)) * R.m2(gz) * rdy * R.y(vh)
    return tx, ty


def _turbt_flat_h(bk, g, b, t, rf, kmh, uh, uf, vh, vf):
    """Horizontal part of turbt without terrain."""
    nk, nj, ni = g.nk, g.nj, g.ni
    rdx, rdy = g.rdx, g.rdy
    K = (2, nk)

    F = Reg(g, K, (1, nj + 1), (1, ni + 1))
    dumx = (
        -0.25
        * (F(rf) + F(rf, di=-1))
        * 2.0
        * (F(kmh) + F(kmh, di=-1))
        * (F(t) - F(t, di=-1))
        * rdx
        * F.x(uf)
    )
    dumy = (
        -0.25
        * (F(rf) + F(rf, dj=-1))
        * 2.0
        * (F(kmh) + F(kmh, dj=-1))
        * (F(t) - F(t, dj=-1))
        * rdy
        * F.y(vf)
    )
    wx = bk.set(bk.zeros(g.s_shape), F.idx, dumx)
    wy = bk.set(bk.zeros(g.s_shape), F.idx, dumy)
    if b["wbc"] == 2 and b["ibw"] == 1:
        wx = _setcol(bk, g, wx, 1, 2, K, (1, nj))
    if b["ebc"] == 2 and b["ibe"] == 1:
        wx = _setcol(bk, g, wx, ni + 1, ni, K, (1, nj))
    if b["sbc"] == 2 and b["ibs"] == 1:
        wy = _setrow(bk, g, wy, 1, 2, K, (1, ni))
    if b["nbc"] == 2 and b["ibn"] == 1:
        wy = _setrow(bk, g, wy, nj + 1, nj, K, (1, ni))
    R = Reg(g, K, (1, nj), (1, ni))
    tx = -(R(wx, di=1) - R(wx)) * rdx * R.x(uh)
    ty = -(R(wy, dj=1) - R(wy)) * rdy * R.y(vh)
    return tx, ty


def _turbt_z(bk, g, R, doimpl, bbc, dt, t, tten, tx, ty, mh, mf, rho, rf, kmv):
    """Vertical part of turbt and the tten update."""
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    rdz = g.rdz
    J, I = (1, nj), (1, ni)

    def lev(a, k):
        return Reg(g, (k, k), J, I)(a)[0]

    if doimpl == 0:
        dz_ = {}
        for k in range(1, nk + 1):
            dz_[k] = (
                -(lev(kmv, k) + lev(kmv, k + 1))
                * (lev(t, k + 1) - lev(t, k))
                * rdz
                * lev(mh, k)
                * lev(rho, k)
            )
        if bbc == 3:
            dz_[1] = xp.zeros_like(dz_[1])
        tz = [-(dz_[k] - dz_[k - 1]) * rdz * lev(mf, k) for k in range(2, nk + 1)]
    else:
        rdt = 1.0 / dt
        tema = -1.0 * dt * vialpha * rdz * rdz
        temb = dt * vibeta * rdz * rdz

        def L(a, k0, k1):
            return Reg(g, (k0, k1), J, I)(a)

        zz = xp.zeros_like(L(t, 2, 2))
        # k = 2
        rrf = L(mf, 2, 2) / L(rf, 2, 2)
        r2 = (L(kmv, 2, 2) + L(kmv, 3, 3)) * L(mh, 2, 2) * L(rho, 2, 2) * rrf
        cfc2 = tema * r2
        cfb2 = 1.0 - cfc2
        cfd2 = L(t, 2, 2) + temb * (r2 * L(t, 3, 3) - r2 * L(t, 2, 2))
        # k = 3..nk-1
        rrf = L(mf, 3, nk - 1) / L(rf, 3, nk - 1)
        r1 = (
            (L(kmv, 2, nk - 2) + L(kmv, 3, nk - 1))
            * L(mh, 2, nk - 2)
            * L(rho, 2, nk - 2)
            * rrf
        )
        r2 = (
            (L(kmv, 3, nk - 1) + L(kmv, 4, nk))
            * L(mh, 3, nk - 1)
            * L(rho, 3, nk - 1)
            * rrf
        )
        cfa = tema * r1
        cfc = tema * r2
        cfb = 1.0 - cfa - cfc
        cfd = L(t, 3, nk - 1) + temb * (
            r2 * L(t, 4, nk) - (r1 + r2) * L(t, 3, nk - 1) + r1 * L(t, 2, nk - 2)
        )
        # k = nk
        rrf = L(mf, nk, nk) / L(rf, nk, nk)
        r1 = (
            (L(kmv, nk - 1, nk - 1) + L(kmv, nk, nk))
            * L(mh, nk - 1, nk - 1)
            * L(rho, nk - 1, nk - 1)
            * rrf
        )
        cfan = tema * r1
        cfbn = 1.0 - cfan
        cfdn = L(t, nk, nk) + temb * (-r1 * L(t, nk, nk) + r1 * L(t, nk - 1, nk - 1))
        dz_ = _tridiag(
            bk,
            _cat(xp, zz, cfa, cfan),
            _cat(xp, cfb2, cfb, cfbn),
            _cat(xp, cfc2, cfc, zz),
            _cat(xp, cfd2, cfd, cfdn),
        )
        tz = L(rf, 2, nk) * (dz_ - L(t, 2, nk)) * rdt
    if isinstance(tz, list):
        tz = _stack(xp, tz)
    out = R(tten) + ((tx + ty) + tz) / R(rf)
    return {"tten": bk.set(bk.copy(tten), R.idx, out)}


# ---------------------------------------------------------------------------
# turbu / turbuz, turbv / turbvz


def _uvz(bk, g, cfg, *, dt, ax, u, rho, rf, mh, mf, kmv, t13, w, uf_or_vf, tm=None):
    """turbuz (ax='i') / turbvz (ax='j'): vertical part ``turbz`` on the
    u (v) points i=1..ni+1, j=1..nj (i=1..ni, j=1..nj+1), k=1..nk."""
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    doimpl = int(_opt(cfg, "doimpl", 1))
    rdz = g.rdz
    if ax == "i":
        J, I = (1, nj), (1, ni + 1)
        sh = {"di": -1}
        rdh = g.rdx
    else:
        J, I = (1, nj + 1), (1, ni)
        sh = {"dj": -1}
        rdh = g.rdy

    def lev(a, k, m=False):
        r = Reg(g, (k, k), J, I)
        return (r(a, **sh) if m else r(a))[0]

    if doimpl == 0:
        tem = rdz * 0.5
        tz = [
            (lev(t13, k + 1) - lev(t13, k)) * tem * (lev(mh, k, True) + lev(mh, k))
            for k in range(1, nk + 1)
        ]
        return _stack(xp, tz)

    rdt = 0.5 / dt
    tema = -0.0625 * dt * vialpha * rdz * rdz
    temb = 0.0625 * dt * vibeta * rdz * rdz
    temc = 0.5 * dt * rdz

    def L(a, k0, k1, m=False):
        r = Reg(g, (k0, k1), J, I)
        return r(a, **sh) if m else r(a)

    def mm(a, k0, k1):  # a(i-1) + a(i)  (or j-1, j)
        return L(a, k0, k1, True) + L(a, k0, k1)

    def rcoef(k0, k1, tem):  # levels kk = k0..k1
        return mm(kmv, k0, k1) * mm(mf, k0, k1) * mm(rf, k0, k1) * tem

    r1d = Reg(g, (1, 1), J, I)
    h1 = r1d.x(uf_or_vf) if ax == "i" else r1d.y(uf_or_vf)

    def rr0(k0, k1):
        return 1.0 / (0.5 * mm(rho, k0, k1))

    zz = xp.zeros_like(L(u, 1, 1))
    # k = 1
    rru0 = rr0(1, 1)
    tem = mm(mh, 1, 1) * rru0
    r2 = rcoef(2, 2, tem)
    cfc1 = tema * r2
    cfb1 = 1.0 - cfc1
    cfd1 = (
        L(u, 1, 1)
        + temb * (r2 * L(u, 2, 2) - r2 * L(u, 1, 1))
        - temc * L(t13, 1, 1) * mm(mh, 1, 1) * rru0
    )
    # k = 2..nk-1
    rru0 = rr0(2, nk - 1)
    tem = mm(mh, 2, nk - 1) * rru0
    r1 = rcoef(2, nk - 1, tem)
    r2 = rcoef(3, nk, tem)
    cfa = tema * r1
    cfc = tema * r2
    cfb = 1.0 - cfa - cfc
    cfd = L(u, 2, nk - 1) + temb * (
        r2 * L(u, 3, nk) - (r1 + r2) * L(u, 2, nk - 1) + r1 * L(u, 1, nk - 2)
    )
    # k = nk
    rru0 = rr0(nk, nk)
    tem = mm(mh, nk, nk) * rru0
    r1 = rcoef(nk, nk, tem)
    cfan = tema * r1
    cfbn = 1.0 - cfan
    cfdn = (
        L(u, nk, nk)
        + temb * (-r1 * L(u, nk, nk) + r1 * L(u, nk - 1, nk - 1))
        + temc * L(t13, nk + 1, nk + 1) * mm(mh, nk, nk) * rru0
    )
    d3 = _tridiag(
        bk,
        _cat(xp, zz, cfa, cfan),
        _cat(xp, cfb1, cfb, cfbn),
        _cat(xp, cfc1, cfc, zz),
        _cat(xp, cfd1, cfd, cfdn),
    )
    # explicit part d7(k), k = 2..nk (d7(1) = 0; 0 above k = nk)
    if tm is None:
        d7 = (
            (L(w, 2, nk) - L(w, 2, nk, True))
            * rdh
            * h1
            * 0.25
            * mm(kmv, 2, nk)
            * mm(rf, 2, nk)
        )
    else:
        # dum9: w at scalar points (k = 1..nk, j, i = 0..n+1)
        zt = float(tm["zt"])
        rgz, gz = tm["rgz"], tm["gz"]
        A = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
        d9 = bk.set(bk.zeros(g.s_shape), A.idx, 0.5 * (A(w) + A(w, dk=1)))
        R = Reg(g, (2, nk), J, I)
        if ax == "i":
            hh = R.x(uf_or_vf)
            dd = {"di": -1}
        else:
            hh = R.y(uf_or_vf)
            dd = {"dj": -1}
        d7 = (R(w) * R.m2(rgz) - R(w, **dd) * R.m2(rgz, **dd)) * rdh * hh + 0.5 * R.z(
            tm["rds"]
        ) * (
            (zt - R.z(tm["sigma"])) * (R(d9) + R(d9, **dd))
            - (zt - R.z(tm["sigma"], dk=-1)) * (R(d9, dk=-1) + R(d9, dk=-1, **dd))
        ) * (R.m2(rgz) - R.m2(rgz, **dd)) * rdh * hh
        d7 = (
            d7
            * 0.25
            * (R(kmv, **dd) + R(kmv))
            * (R.m2(gz, **dd) * R(rf, **dd) + R.m2(gz) * R(rf))
        )
    d7lo = _cat(xp, zz, d7)  # k = 1..nk
    d7hi = _cat(xp, d7, zz)  # k+1
    tz = mm(rho, 1, nk) * (d3 - L(u, 1, nk)) * rdt
    return tz + (d7hi - d7lo) * rdz * 0.5 * mm(mh, 1, nk)


def turbu(
    bk,
    g,
    cfg=None,
    *,
    dt,
    ruh=None,
    uf,
    vh,
    mh,
    mf,
    rho,
    rf,
    u,
    uten,
    w,
    t11,
    t12,
    t13,
    kmv,
    terrain=None,
    **unused,
):
    """turbtend.F ``turbu`` (+ turbuz): ``uten(1:ni+1,1:nj,1:nk) +=`` subgrid
    stress divergence.  ``u``/``uten`` are ``g.u_shape``, ``w`` ``g.w_shape``.
    ``terrain``: metrics with terrain_flag.  Returns {"uten"}."""
    _check(cfg, "turbu")
    tm = _terrain(cfg, terrain, "turbu")
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbu: nk >= 3 required")
    b = _bcflags(cfg)
    K = (1, nk)
    R = Reg(g, K, (1, nj), (1, ni + 1))
    if tm is None:
        tx = (R(t11) - R(t11, di=-1)) * g.rdx * R.x(uf)
        ty = (R(t12, dj=1) - R(t12)) * g.rdy * R.y(vh)
    else:
        zt = float(tm["zt"])
        gzu, rgz, rdsf, sf = tm["gzu"], tm["rgz"], tm["rdsf"], tm["sigmaf"]
        d1 = _atw(bk, g, tm, t11, (1, nj + 1), (0, ni + 1))
        d2 = _atw(bk, g, tm, t12, (1, nj + 1), (0, ni + 1))
        tx = R.m2(gzu) * (
            R(t11) * R.m2(rgz) - R(t11, di=-1) * R.m2(rgz, di=-1)
        ) * g.rdx * R.x(uf) + 0.5 * (
            R(tm["gxu"], dk=1) * (R(d1, dk=1, di=-1) + R(d1, dk=1))
            - R(tm["gxu"]) * (R(d1, di=-1) + R(d1))
        ) * R.z(rdsf)
        r1 = 0.25 * (
            (R.m2(rgz, dj=-1, di=-1) + R.m2(rgz))
            + (R.m2(rgz, di=-1) + R.m2(rgz, dj=-1))
        )
        r2 = 0.25 * (
            (R.m2(rgz, dj=1, di=-1) + R.m2(rgz)) + (R.m2(rgz, di=-1) + R.m2(rgz, dj=1))
        )
        ty = R.m2(gzu) * (R(t12, dj=1) * r2 - R(t12) * r1) * g.rdy * R.y(vh) + 0.5 * (
            (zt - R.z(sf, dk=1)) * (R(d2, dk=1) + R(d2, dk=1, dj=1))
            - (zt - R.z(sf)) * (R(d2) + R(d2, dj=1))
        ) * R.m2(gzu) * (r2 - r1) * g.rdy * R.y(vh) * R.z(rdsf)
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.u_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.u_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni + 1, nj, iy=(1, ni + 1))
        tx, ty = R(tx), R(ty)
    tz = _uvz(
        bk,
        g,
        cfg,
        dt=dt,
        ax="i",
        u=u,
        rho=rho,
        rf=rf,
        mh=mh,
        mf=mf,
        kmv=kmv,
        t13=t13,
        w=w,
        uf_or_vf=uf,
        tm=tm,
    )
    rru0 = 1.0 / (0.5 * (R(rho, di=-1) + R(rho)))
    out = R(uten) + ((tx + ty) + tz) * rru0
    return {"uten": bk.set(bk.copy(uten), R.idx, out)}


def turbv(
    bk,
    g,
    cfg=None,
    *,
    dt,
    uh,
    rvh=None,
    vf,
    mh,
    mf,
    rho,
    rr=None,
    rf,
    v,
    vten,
    w,
    t12,
    t22,
    t23,
    kmv,
    terrain=None,
    **unused,
):
    """turbtend.F ``turbv`` (+ turbvz): ``vten(1:ni,1:nj+1,1:nk) +=`` subgrid
    stress divergence.  ``v``/``vten`` are ``g.v_shape``.  ``terrain``:
    metrics with terrain_flag.  Returns {"vten"}."""
    _check(cfg, "turbv")
    tm = _terrain(cfg, terrain, "turbv")
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbv: nk >= 3 required")
    b = _bcflags(cfg)
    K = (1, nk)
    R = Reg(g, K, (1, nj + 1), (1, ni))
    if tm is None:
        tx = (R(t12, di=1) - R(t12)) * g.rdx * R.x(uh)
        ty = (R(t22) - R(t22, dj=-1)) * g.rdy * R.y(vf)
    else:
        zt = float(tm["zt"])
        gzv, rgz, rdsf, sf = tm["gzv"], tm["rgz"], tm["rdsf"], tm["sigmaf"]
        d1 = _atw(bk, g, tm, t12, (0, nj + 1), (1, ni + 1))
        d2 = _atw(bk, g, tm, t22, (0, nj + 1), (1, ni + 1))
        r1 = 0.25 * (
            (R.m2(rgz, dj=-1, di=-1) + R.m2(rgz))
            + (R.m2(rgz, di=-1) + R.m2(rgz, dj=-1))
        )
        r2 = 0.25 * (
            (R.m2(rgz, dj=-1, di=1) + R.m2(rgz)) + (R.m2(rgz, di=1) + R.m2(rgz, dj=-1))
        )
        tx = R.m2(gzv) * (R(t12, di=1) * r2 - R(t12) * r1) * g.rdx * R.x(uh) + 0.5 * (
            (zt - R.z(sf, dk=1)) * (R(d1, dk=1) + R(d1, dk=1, di=1))
            - (zt - R.z(sf)) * (R(d1) + R(d1, di=1))
        ) * R.m2(gzv) * (r2 - r1) * g.rdx * R.x(uh) * R.z(rdsf)
        ty = R.m2(gzv) * (
            R(t22) * R.m2(rgz) - R(t22, dj=-1) * R.m2(rgz, dj=-1)
        ) * g.rdy * R.y(vf) + 0.5 * (
            R(tm["gyv"], dk=1) * (R(d2, dk=1, dj=-1) + R(d2, dk=1))
            - R(tm["gyv"]) * (R(d2, dj=-1) + R(d2))
        ) * R.z(rdsf)
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.v_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.v_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni, nj + 1, jx=(1, nj + 1))
        tx, ty = R(tx), R(ty)
    tz = _uvz(
        bk,
        g,
        cfg,
        dt=dt,
        ax="j",
        u=v,
        rho=rho,
        rf=rf,
        mh=mh,
        mf=mf,
        kmv=kmv,
        t13=t23,
        w=w,
        uf_or_vf=vf,
        tm=tm,
    )
    rrv0 = 1.0 / (0.5 * (R(rho, dj=-1) + R(rho)))
    out = R(vten) + ((tx + ty) + tz) * rrv0
    return {"vten": bk.set(bk.copy(vten), R.idx, out)}


# ---------------------------------------------------------------------------
# turbw


def turbw(
    bk,
    g,
    cfg=None,
    *,
    dt,
    uh,
    vh,
    mh,
    mf,
    rho,
    rf,
    w,
    wten,
    t13,
    t23,
    t33,
    kmh,
    terrain=None,
    **unused,
):
    """turbtend.F ``turbw``: ``wten(1:ni,1:nj,2:nk) +=`` subgrid stress
    divergence (implicit vertical part uses ``kmh`` as CM1 does).
    ``w``/``wten`` are ``g.w_shape``.  ``terrain``: metrics with
    terrain_flag.  Returns {"wten"}."""
    _check(cfg, "turbw")
    tm = _terrain(cfg, terrain, "turbw")
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbw: nk >= 3 required")
    b = _bcflags(cfg)
    doimpl = int(_opt(cfg, "doimpl", 1))
    rdz = g.rdz
    K = (2, nk)
    R = Reg(g, K, (1, nj), (1, ni))
    if tm is None:
        tx = (R(t13, di=1) - R(t13)) * g.rdx * R.x(uh)
        ty = (R(t23, dj=1) - R(t23)) * g.rdy * R.y(vh)
    else:
        zt = float(tm["zt"])
        gz, rgzu, rgzv = tm["gz"], tm["rgzu"], tm["rgzv"]
        sig, rds = tm["sigma"], tm["rds"]
        A = Reg(g, (1, nk), (1, nj), (1, ni))
        d1 = bk.set(
            bk.zeros(g.s_shape),
            A.idx,
            0.25 * ((A(t13, dk=1) + A(t13, dk=1, di=1)) + (A(t13) + A(t13, di=1))),
        )
        d2 = bk.set(
            bk.zeros(g.s_shape),
            A.idx,
            0.25 * ((A(t23, dk=1) + A(t23, dk=1, dj=1)) + (A(t23) + A(t23, dj=1))),
        )
        tx = R.m2(gz) * (
            R(t13, di=1) * R.m2(rgzu, di=1) - R(t13) * R.m2(rgzu)
        ) * g.rdx * R.x(uh) + (
            (zt - R.z(sig)) * R(d1) - (zt - R.z(sig, dk=-1)) * R(d1, dk=-1)
        ) * R.m2(gz) * (R.m2(rgzu, di=1) - R.m2(rgzu)) * g.rdx * R.x(uh) * R.z(rds)
        ty = R.m2(gz) * (
            R(t23, dj=1) * R.m2(rgzv, dj=1) - R(t23) * R.m2(rgzv)
        ) * g.rdy * R.y(vh) + (
            (zt - R.z(sig)) * R(d2) - (zt - R.z(sig, dk=-1)) * R(d2, dk=-1)
        ) * R.m2(gz) * (R.m2(rgzv, dj=1) - R.m2(rgzv)) * g.rdy * R.y(vh) * R.z(rds)
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.w_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.w_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni, nj)
        tx, ty = R(tx), R(ty)

    J, I = (1, nj), (1, ni)

    def lev(a, k):
        return Reg(g, (k, k), J, I)(a)[0]

    if doimpl == 0:
        tz = _stack(
            xp,
            [
                (lev(t33, k) - lev(t33, k - 1)) * rdz * lev(mf, k)
                for k in range(2, nk + 1)
            ],
        )
    else:
        rdt = 1.0 / dt
        tema = -1.0 * dt * vialpha * rdz * rdz
        temb = dt * vibeta * rdz * rdz
        temc = dt * rdz

        def L(a, k0, k1):
            return Reg(g, (k0, k1), J, I)(a)

        zz = xp.zeros_like(L(w, 2, 2))
        # k = 2
        rrf = L(mf, 2, 2) / L(rf, 2, 2)
        r2 = (L(kmh, 2, 2) + L(kmh, 3, 3)) * L(mh, 2, 2) * L(rho, 2, 2) * rrf
        cfc2 = tema * r2
        cfb2 = 1.0 - cfc2
        cfd2 = (
            L(w, 2, 2)
            + temb * (r2 * L(w, 3, 3) - r2 * L(w, 2, 2))
            - temc * L(t33, 1, 1) * rrf
        )
        # k = 3..nk-1
        rrf = L(mf, 3, nk - 1) / L(rf, 3, nk - 1)
        r1 = (
            (L(kmh, 2, nk - 2) + L(kmh, 3, nk - 1))
            * L(mh, 2, nk - 2)
            * L(rho, 2, nk - 2)
            * rrf
        )
        r2 = (
            (L(kmh, 3, nk - 1) + L(kmh, 4, nk))
            * L(mh, 3, nk - 1)
            * L(rho, 3, nk - 1)
            * rrf
        )
        cfa = tema * r1
        cfc = tema * r2
        cfb = 1.0 - cfa - cfc
        cfd = L(w, 3, nk - 1) + temb * (
            r2 * L(w, 4, nk) - (r1 + r2) * L(w, 3, nk - 1) + r1 * L(w, 2, nk - 2)
        )
        # k = nk
        rrf = L(mf, nk, nk) / L(rf, nk, nk)
        r1 = (
            (L(kmh, nk - 1, nk - 1) + L(kmh, nk, nk))
            * L(mh, nk - 1, nk - 1)
            * L(rho, nk - 1, nk - 1)
            * rrf
        )
        cfan = tema * r1
        cfbn = 1.0 - cfan
        cfdn = (
            L(w, nk, nk)
            + temb * (-r1 * L(w, nk, nk) + r1 * L(w, nk - 1, nk - 1))
            + temc * L(t33, nk, nk) * rrf
        )
        d3 = _tridiag(
            bk,
            _cat(xp, zz, cfa, cfan),
            _cat(xp, cfb2, cfb, cfbn),
            _cat(xp, cfc2, cfc, zz),
            _cat(xp, cfd2, cfd, cfdn),
        )
        tz = L(rf, 2, nk) * (d3 - L(w, 2, nk)) * rdt
    out = R(wten) + ((tx + ty) + tz) / R(rf)
    return {"wten": bk.set(bk.copy(wten), R.idx, out)}
