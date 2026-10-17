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
    if _opt(cfg, "terrain_flag", False):
        bad.append("terrain_flag=T")
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
    d1 = [None] * (nk + 1)
    d2 = [None] * (nk + 1)
    k = 1
    r2 = lev(dum8, k)
    cfc = tema * r2
    cfb = 1.0 - cfc
    cfd = (
        lev(s, k)
        + temb * (-r2 * lev(s, k) + r2 * lev(s, k + 1))
        + temc * dzb * lev(mh, 1) * lev(rr, 1)
    )
    tem = 1.0 / cfb
    d1[1] = -cfc * tem
    d2[1] = cfd * tem
    for k in range(2, nk):
        r1 = lev(dum7, k)
        r2 = lev(dum8, k)
        cfa = tema * r1
        cfc = tema * r2
        cfb = 1.0 - cfa - cfc
        cfd = lev(s, k) + temb * (
            r1 * lev(s, k - 1) - (r1 + r2) * lev(s, k) + r2 * lev(s, k + 1)
        )
        tem = 1.0 / (cfa * d1[k - 1] + cfb)
        d1[k] = -cfc * tem
        d2[k] = (cfd - cfa * d2[k - 1]) * tem
    k = nk
    r1 = lev(dum7, k)
    cfa = tema * r1
    cfb = 1.0 - cfa
    cfd = (
        lev(s, k)
        + temb * (r1 * lev(s, k - 1) - r1 * lev(s, k))
        - temc * dzt * lev(mh, nk) * lev(rr, nk)
    )
    tem = 1.0 / (cfa * d1[k - 1] + cfb)
    dz_ = [None] * (nk + 2)
    tz = [None] * (nk + 1)
    dz_[k] = (cfd - cfa * d2[k - 1]) * tem
    tz[k] = lev(rho, k) * (dz_[k] - lev(s, k)) * rdt
    for k in range(nk - 1, 0, -1):
        dz_[k] = d1[k] * dz_[k + 1] + d2[k]
        tz[k] = lev(rho, k) * (dz_[k] - lev(s, k)) * rdt
    return _stack(xp, tz[1:])


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
    **unused,
):
    """turbtend.F ``turbs``: ``sten(1:ni,1:nj,1:nk) +=`` subgrid diffusion of
    ``s``.  ``dum7/dum8`` are solve1's vimpl coefficients (see
    ``turb.vimpl_coefs``; required when doimpl=1).  Returns {"sten"}."""
    _check(cfg, "turbs")
    nk, nj, ni = g.nk, g.nj, g.ni
    b = _bcflags(cfg)
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
    **unused,
):
    """turbtend.F ``turbt``: ``tten(1:ni,1:nj,2:nk) +=`` subgrid diffusion of
    the w-level scalar ``t`` (TKE).  Returns {"tten"}."""
    _check(cfg, "turbt")
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbt: nk >= 3 required")
    b = _bcflags(cfg)
    doimpl = int(_opt(cfg, "doimpl", 1))
    bbc = int(_opt(cfg, "bbc", 1))
    rdx, rdy, rdz = g.rdx, g.rdy, g.rdz
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
    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        tx = bk.set(bk.zeros(g.s_shape), R.idx, tx)
        ty = bk.set(bk.zeros(g.s_shape), R.idx, ty)
        tx, ty = _open_zero(bk, g, b, tx, ty, K, ni, nj)
        tx, ty = R(tx), R(ty)

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
        d1, d2 = {}, {}
        k = 2
        rrf = lev(mf, k) / lev(rf, k)
        r2 = (lev(kmv, k) + lev(kmv, k + 1)) * lev(mh, k) * lev(rho, k) * rrf
        cfc = tema * r2
        cfb = 1.0 - cfc
        cfd = lev(t, k) + temb * (r2 * lev(t, k + 1) - r2 * lev(t, k))
        tem = 1.0 / cfb
        d1[k] = -cfc * tem
        d2[k] = cfd * tem
        for k in range(3, nk):
            rrf = lev(mf, k) / lev(rf, k)
            r1 = (
                (lev(kmv, k - 1) + lev(kmv, k)) * lev(mh, k - 1) * lev(rho, k - 1) * rrf
            )
            r2 = (lev(kmv, k) + lev(kmv, k + 1)) * lev(mh, k) * lev(rho, k) * rrf
            cfa = tema * r1
            cfc = tema * r2
            cfb = 1.0 - cfa - cfc
            cfd = lev(t, k) + temb * (
                r2 * lev(t, k + 1) - (r1 + r2) * lev(t, k) + r1 * lev(t, k - 1)
            )
            tem = 1.0 / (cfa * d1[k - 1] + cfb)
            d1[k] = -cfc * tem
            d2[k] = (cfd - cfa * d2[k - 1]) * tem
        k = nk
        rrf = lev(mf, k) / lev(rf, k)
        r1 = (lev(kmv, k - 1) + lev(kmv, k)) * lev(mh, k - 1) * lev(rho, k - 1) * rrf
        cfa = tema * r1
        cfb = 1.0 - cfa
        cfd = lev(t, k) + temb * (-r1 * lev(t, k) + r1 * lev(t, k - 1))
        tem = 1.0 / (cfa * d1[k - 1] + cfb)
        dz_ = {k: (cfd - cfa * d2[k - 1]) * tem}
        tzd = {k: lev(rf, k) * (dz_[k] - lev(t, k)) * rdt}
        for k in range(nk - 1, 1, -1):
            dz_[k] = d1[k] * dz_[k + 1] + d2[k]
            tzd[k] = lev(rf, k) * (dz_[k] - lev(t, k)) * rdt
        tz = [tzd[k] for k in range(2, nk + 1)]
    tz = _stack(xp, tz)
    out = R(tten) + ((tx + ty) + tz) / R(rf)
    return {"tten": bk.set(bk.copy(tten), R.idx, out)}


# ---------------------------------------------------------------------------
# turbu / turbuz, turbv / turbvz


def _uvz(bk, g, cfg, *, dt, ax, u, rho, rf, mh, mf, kmv, t13, w, uf_or_vf):
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

    def h1(a1):
        r = Reg(g, (1, 1), J, I)
        return (r.x(a1) if ax == "i" else r.y(a1))[0]

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
    d1, d2, d7 = {}, {}, {}

    def rr0(k):
        return 1.0 / (0.5 * (lev(rho, k, True) + lev(rho, k)))

    def rcoef(k, kk, tem):
        return (
            (lev(kmv, kk, True) + lev(kmv, kk))
            * (lev(mf, kk, True) + lev(mf, kk))
            * (lev(rf, kk, True) + lev(rf, kk))
            * tem
        )

    def expl(k):
        return (
            (lev(w, k) - lev(w, k, True))
            * rdh
            * h1(uf_or_vf)
            * 0.25
            * (lev(kmv, k, True) + lev(kmv, k))
            * (lev(rf, k, True) + lev(rf, k))
        )

    k = 1
    rru0 = rr0(k)
    tem = (lev(mh, k, True) + lev(mh, k)) * rru0
    r2 = rcoef(k, k + 1, tem)
    cfc = tema * r2
    cfb = 1.0 - cfc
    cfd = (
        lev(u, k)
        + temb * (r2 * lev(u, k + 1) - r2 * lev(u, k))
        - temc * lev(t13, 1) * (lev(mh, 1, True) + lev(mh, 1)) * rru0
    )
    tem = 1.0 / cfb
    d1[1] = -cfc * tem
    d2[1] = cfd * tem
    d7[1] = xp.zeros_like(d2[1])
    for k in range(2, nk):
        rru0 = rr0(k)
        tem = (lev(mh, k, True) + lev(mh, k)) * rru0
        r1 = rcoef(k, k, tem)
        r2 = rcoef(k, k + 1, tem)
        cfa = tema * r1
        cfc = tema * r2
        cfb = 1.0 - cfa - cfc
        cfd = lev(u, k) + temb * (
            r2 * lev(u, k + 1) - (r1 + r2) * lev(u, k) + r1 * lev(u, k - 1)
        )
        tem = 1.0 / (cfa * d1[k - 1] + cfb)
        d1[k] = -cfc * tem
        d2[k] = (cfd - cfa * d2[k - 1]) * tem
        d7[k] = expl(k)
    k = nk
    rru0 = rr0(k)
    tem = (lev(mh, k, True) + lev(mh, k)) * rru0
    r1 = rcoef(k, k, tem)
    cfa = tema * r1
    cfb = 1.0 - cfa
    cfd = (
        lev(u, k)
        + temb * (-r1 * lev(u, k) + r1 * lev(u, k - 1))
        + temc * lev(t13, nk + 1) * (lev(mh, nk, True) + lev(mh, nk)) * rru0
    )
    tem = 1.0 / (cfa * d1[k - 1] + cfb)
    d3 = {k: (cfd - cfa * d2[k - 1]) * tem}
    tz = {}
    tz[k] = (lev(rho, k, True) + lev(rho, k)) * (d3[k] - lev(u, k)) * rdt
    d7[k] = expl(k)
    tz[k] = tz[k] + (0.0 - d7[k]) * rdz * 0.5 * (lev(mh, k, True) + lev(mh, k))
    for k in range(nk - 1, 0, -1):
        d3[k] = d1[k] * d3[k + 1] + d2[k]
        tz[k] = (lev(rho, k, True) + lev(rho, k)) * (d3[k] - lev(u, k)) * rdt
        tz[k] = tz[k] + (d7[k + 1] - d7[k]) * rdz * 0.5 * (
            lev(mh, k, True) + lev(mh, k)
        )
    return _stack(xp, [tz[k] for k in range(1, nk + 1)])


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
    **unused,
):
    """turbtend.F ``turbu`` (+ turbuz): ``uten(1:ni+1,1:nj,1:nk) +=`` subgrid
    stress divergence.  ``u``/``uten`` are ``g.u_shape``, ``w`` ``g.w_shape``.
    Returns {"uten"}."""
    _check(cfg, "turbu")
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbu: nk >= 3 required")
    b = _bcflags(cfg)
    K = (1, nk)
    R = Reg(g, K, (1, nj), (1, ni + 1))
    tx = (R(t11) - R(t11, di=-1)) * g.rdx * R.x(uf)
    ty = (R(t12, dj=1) - R(t12)) * g.rdy * R.y(vh)
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
    **unused,
):
    """turbtend.F ``turbv`` (+ turbvz): ``vten(1:ni,1:nj+1,1:nk) +=`` subgrid
    stress divergence.  ``v``/``vten`` are ``g.v_shape``.  Returns {"vten"}."""
    _check(cfg, "turbv")
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbv: nk >= 3 required")
    b = _bcflags(cfg)
    K = (1, nk)
    R = Reg(g, K, (1, nj + 1), (1, ni))
    tx = (R(t12, di=1) - R(t12)) * g.rdx * R.x(uh)
    ty = (R(t22) - R(t22, dj=-1)) * g.rdy * R.y(vf)
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
    **unused,
):
    """turbtend.F ``turbw``: ``wten(1:ni,1:nj,2:nk) +=`` subgrid stress
    divergence (implicit vertical part uses ``kmh`` as CM1 does).
    ``w``/``wten`` are ``g.w_shape``.  Returns {"wten"}."""
    _check(cfg, "turbw")
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    if nk < 3:
        raise NotImplementedError("turbw: nk >= 3 required")
    b = _bcflags(cfg)
    doimpl = int(_opt(cfg, "doimpl", 1))
    rdz = g.rdz
    K = (2, nk)
    R = Reg(g, K, (1, nj), (1, ni))
    tx = (R(t13, di=1) - R(t13)) * g.rdx * R.x(uh)
    ty = (R(t23, dj=1) - R(t23)) * g.rdy * R.y(vh)
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
        d1, d2 = {}, {}
        k = 2
        rrf = lev(mf, k) / lev(rf, k)
        r2 = (lev(kmh, k) + lev(kmh, k + 1)) * lev(mh, k) * lev(rho, k) * rrf
        cfc = tema * r2
        cfb = 1.0 - cfc
        cfd = (
            lev(w, k)
            + temb * (r2 * lev(w, k + 1) - r2 * lev(w, k))
            - temc * lev(t33, k - 1) * rrf
        )
        tem = 1.0 / cfb
        d1[k] = -cfc * tem
        d2[k] = cfd * tem
        for k in range(3, nk):
            rrf = lev(mf, k) / lev(rf, k)
            r1 = (
                (lev(kmh, k - 1) + lev(kmh, k)) * lev(mh, k - 1) * lev(rho, k - 1) * rrf
            )
            r2 = (lev(kmh, k) + lev(kmh, k + 1)) * lev(mh, k) * lev(rho, k) * rrf
            cfa = tema * r1
            cfc = tema * r2
            cfb = 1.0 - cfa - cfc
            cfd = lev(w, k) + temb * (
                r2 * lev(w, k + 1) - (r1 + r2) * lev(w, k) + r1 * lev(w, k - 1)
            )
            tem = 1.0 / (cfa * d1[k - 1] + cfb)
            d1[k] = -cfc * tem
            d2[k] = (cfd - cfa * d2[k - 1]) * tem
        k = nk
        rrf = lev(mf, k) / lev(rf, k)
        r1 = (lev(kmh, k - 1) + lev(kmh, k)) * lev(mh, k - 1) * lev(rho, k - 1) * rrf
        cfa = tema * r1
        cfb = 1.0 - cfa
        cfd = (
            lev(w, k)
            + temb * (-r1 * lev(w, k) + r1 * lev(w, k - 1))
            + temc * lev(t33, k) * rrf
        )
        tem = 1.0 / (cfa * d1[k - 1] + cfb)
        d3 = {k: (cfd - cfa * d2[k - 1]) * tem}
        tzd = {k: lev(rf, k) * (d3[k] - lev(w, k)) * rdt}
        for k in range(nk - 1, 1, -1):
            d3[k] = d1[k] * d3[k + 1] + d2[k]
            tzd[k] = lev(rf, k) * (d3[k] - lev(w, k)) * rdt
        tz = _stack(xp, [tzd[k] for k in range(2, nk + 1)])
    out = R(wten) + ((tx + ty) + tz) / R(rf)
    return {"wten": bk.set(bk.copy(wten), R.idx, out)}
