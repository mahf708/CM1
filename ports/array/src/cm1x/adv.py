"""Advection tendencies, ported from src/adv.F (advs, advu, advv, advw).

Also contains, because they are only used by these routines:
  * the positive-definite correction of src/pdef.F (pdefx1, pdefx2, pdefy1,
    pdefy2, pdefz), active in ``advs`` when ``pdef=1`` (CM1 hard-wires
    pdscheme=1; solve2 passes pdef=1 for moisture on the last RK stage);
  * ``getdivx`` from src/misclibs.F (rescales the small-step-averaged
    velocities to mass fluxes and recomputes divx).

Supported configuration (reference cases kessler_periodic and
supercell_kessler): flux-form advection of order 5; WENO5 (weno_order=5,
siform=2) for scalars when advwenos=1, or advwenos=2 on the last RK stage
(advwenov must be 0); periodic (bc=1) or open (bc=2, single process:
ibw=ibe=ibs=ibn=1) lateral boundaries, including the stat_qsrc boundary-flux
statistic ``bsq``; no terrain, Cartesian grid, no immersed boundaries, no
large-scale subsidence, no 6th-order artificial diffusion, no budgets. Any
other option raises NotImplementedError. (imove/umove/vmove do not enter
these routines: CM1 only uses them in movesfc, for sfcmodel=2..4.)

Options come from ``cfg`` (a namelist-like object or dict; missing entries fall
back to ``DEFAULTS``, which mirror ports/reference/kessler_periodic).

Array conventions (README): ``a[k, j, i]`` with CM1 halos. 1-D x arrays
(``uh``, ``uf``) are length ``ni+2*ngxy`` (``+1`` for ``uf``), 1-D y arrays
(``vh``, ``vf``) length ``nj+2*ngxy`` (``+1`` for ``vf``). Base-state arrays
(``rho0``, ``rr0``, ``rrf0``, ``mh``, ``mf``) are 3-D as stored by CM1; only
their ``(1,1,k)`` column is used here (no terrain).
"""

from __future__ import annotations

from .adv_routines import (
    col,
    hadv_flx5,
    hadv_weno5,
    ibflags,
    put3,
    sub3,
    vadv_flx5,
    vadv_weno5,
    xrow,
    yrow,
)
from .grid import fi, fk

DEFAULTS = dict(
    hadvordrs=5,
    vadvordrs=5,
    hadvordrv=5,
    vadvordrv=5,
    advwenos=0,
    advwenov=0,
    weno_order=5,
    idiff=0,
    difforder=6,
    terrain_flag=False,
    axisymm=0,
    wbc=1,
    ebc=1,
    sbc=1,
    nbc=1,
    do_ib=False,
    dolsw=False,
    pdscheme=1,
    stat_qsrc=0,
    smeps=1.0e-30,
    nrkmax=3,
)


def _opt(cfg, name):
    if cfg is None:
        return DEFAULTS[name]
    # dict.get / Config.get: an unset option falls back to this module's
    # default rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, DEFAULTS[name]) if get else getattr(cfg, name, DEFAULTS[name])


def _terrain(cfg):
    return bool(_opt(cfg, "terrain_flag"))


def _need(who, **kw):
    miss = [k for k, v in kw.items() if v is None]
    if miss:
        raise ValueError(f"{who}: terrain_flag=.true. needs {', '.join(miss)}")


def g2(a2, j, i, dj=0, di=0):
    """2-D ``a2(i1+di:i2+di, j1+dj:j2+dj)`` (terrain metrics ``[j, i]``),
    broadcastable on ``[k, j, i]``."""
    return a2[
        None, fi(j[0]) + dj : fi(j[1]) + dj + 1, fi(i[0]) + di : fi(i[1]) + di + 1
    ]


def kv(a1, k, dk=0):
    """1-D vertical array (rds, rdsf; Python index == Fortran k) on range k,
    shaped (nk', 1, 1)."""
    return a1[fk(k[0]) + dk : fk(k[1]) + dk + 1][:, None, None]


def _check_common(cfg, who):
    if _opt(cfg, "axisymm") != 0:
        raise NotImplementedError(f"{who}: axisymm=1 not ported")
    for a, b in (("wbc", "ebc"), ("sbc", "nbc")):
        va, vb = _opt(cfg, a), _opt(cfg, b)
        if va not in (1, 2) or vb not in (1, 2) or ((va == 1) != (vb == 1)):
            raise NotImplementedError(
                f"{who}: only periodic (1) or open (2) lateral BCs are ported "
                f"({a}={va}, {b}={vb})"
            )
    if _opt(cfg, "do_ib"):
        raise NotImplementedError(f"{who}: immersed boundaries not ported")
    if _opt(cfg, "dolsw"):
        raise NotImplementedError(f"{who}: large-scale subsidence (dolsw) not ported")


def _check_order(who, hadvorder, vadvorder, doweno, cfg=None, weno_ok=False):
    if doweno:
        if not weno_ok:
            raise NotImplementedError(f"{who}: WENO advection not ported")
        if _opt(cfg, "weno_order") != 5:
            raise NotImplementedError(f"{who}: only weno_order=5 ported")
        return
    if hadvorder != 5 or vadvorder != 5:
        raise NotImplementedError(
            f"{who}: only 5th-order advection ported (got {hadvorder},{vadvorder})"
        )


def _check_diff(cfg, who):
    if _opt(cfg, "idiff") >= 1 and _opt(cfg, "difforder") == 6:
        raise NotImplementedError(f"{who}: 6th-order artificial diffusion not ported")


def _vflux_to_tend(bk, g, dumz, K, J, I, mh, rdsf=None):
    """advz(k) = -(dumz(k+1)-dumz(k))*rdz*mh(1,1,k), with the CM1 end cases
    advz(1) = -dumz(2)*.., advz(nk) = +dumz(nk)*..; with terrain (``rdsf``
    given) the factor is rdsf(k) instead of rdz*mh(1,1,k)."""
    xp = bk.xp
    nk = g.nk
    rdz = g.rdz

    def mul(x, k):
        if rdsf is not None:
            return x * kv(rdsf, k)
        return x * rdz * col(mh, k)

    top = sub3(dumz, (2, nk), J, I)
    zero = bk.zeros(top[:1].shape)
    up = xp.concat([top, zero], axis=0)  # dumz(k+1), k=1..nk; 0 at k=nk
    dn = xp.concat([zero, top], axis=0)  # dumz(k), k=1..nk; 0 at k=1
    # k=1: -(dumz(2)-0) == -dumz(2); k=nk: -(0-dumz(nk)) == +dumz(nk)  (exact)
    first = mul(-up[:1], (1, 1))
    last = mul(dn[-1:], (nk, nk))
    mid = mul(-(up[1:-1] - dn[1:-1]), (2, nk - 1))
    return xp.concat([first, mid, last], axis=0)


def _setwhere(bk, a, K, J, I, cond, new):
    """Return ``a`` with region (K, J, I) set to ``new`` where ``cond``."""
    return put3(bk, a, K, J, I, bk.xp.where(cond, new, sub3(a, K, J, I)))


def _bcs(bk, g, s, cfg=None):
    """bc.F ``bcs`` for k=1..nk: periodic (bc=1) or open/zero-gradient
    (bc=2) halos; west, east, then south, north (as in CM1)."""
    xp = bk.xp
    n, ni, nj = g.ngxy, g.ni, g.nj
    K = slice(fk(1), fk(g.nk) + 1)
    ibw, ibe, ibs, ibn = ibflags(cfg)
    if ibw:
        J0 = slice(fi(0), fi(nj + 1) + 1)
        edge = s[K, J0, fi(1) : fi(1) + 1]
        s = bk.set(s, (K, J0, slice(0, fi(1))), xp.concat([edge] * n, axis=-1))
    else:
        s = bk.set(s, (K, slice(None), slice(0, n)), s[K, :, ni : ni + n])
    if ibe:
        J0 = slice(fi(0), fi(nj + 1) + 1)
        edge = s[K, J0, fi(ni) : fi(ni) + 1]
        s = bk.set(
            s,
            (K, J0, slice(fi(ni + 1), fi(ni + n) + 1)),
            xp.concat([edge] * n, axis=-1),
        )
    else:
        s = bk.set(s, (K, slice(None), slice(n + ni, 2 * n + ni)), s[K, :, n : 2 * n])
    if ibs:
        I0 = slice(fi(0), fi(ni + 1) + 1)
        edge = s[K, fi(1) : fi(1) + 1, I0]
        s = bk.set(s, (K, slice(0, fi(1)), I0), xp.concat([edge] * n, axis=-2))
    else:
        s = bk.set(s, (K, slice(0, n), slice(None)), s[K, nj : nj + n, :])
    if ibn:
        I0 = slice(fi(0), fi(ni + 1) + 1)
        edge = s[K, fi(nj) : fi(nj) + 1, I0]
        s = bk.set(
            s,
            (K, slice(fi(nj + 1), fi(nj + n) + 1), I0),
            xp.concat([edge] * n, axis=-2),
        )
    else:
        s = bk.set(s, (K, slice(n + nj, 2 * n + nj), slice(None)), s[K, n : 2 * n, :])
    return s


# ===========================================================================
# pdef.F
# ===========================================================================


def _pd_redistribute(bk, dm, dc, dp, smeps):
    """Shared core of pdefx2/pdefy2/pdefz: for each *source* point with value
    dc and neighbours dm (i-1), dp (i+1), whether it borrows mass (active) and
    foo1, foo2, foo1+foo2, foo3 as in CM1."""
    xp = bk.xp
    foo1 = xp.where(dm > 0.0, dm, 0.0 * dm)  # max(0, dum(i-1))
    foo2 = xp.where(dp > 0.0, dp, 0.0 * dp)  # max(0, dum(i+1))
    tot = foo1 + foo2
    active = (dc < 0.0) & (tot > smeps)
    den = xp.where(active, tot, 1.0 + 0.0 * tot)
    num = xp.where(dc > -tot, dc, -tot)  # max(dum(i), -(foo1+foo2))
    foo3 = num / den
    return active, foo1, foo2, tot, foo3


def _pd_mass_flag(bk, d, axis, smeps):
    """Given ``d`` = dum over the source range [t0-1 .. t1+1] along ``axis``
    (with one extra point on each side for neighbour access, i.e. length n+4
    for n target points), return (mass, flag) at the n target points.

    Contributions to mass(t) are accumulated in CM1's sequential order:
    from source t-1 (foo2*foo3), then t (-(foo1+foo2)*foo3), then t+1 (foo1*foo3).
    """
    xp = bk.xp
    n_all = d.shape[axis]

    def sl(a, lo, hi):  # python slice along axis, hi exclusive
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(lo, hi)
        return a[tuple(idx)]

    # sources s = 1 .. n_all-2 (have both neighbours)
    dm = sl(d, 0, n_all - 2)
    dc = sl(d, 1, n_all - 1)
    dp = sl(d, 2, n_all)
    active, foo1, foo2, tot, foo3 = _pd_redistribute(bk, dm, dc, dp, smeps)
    z = 0.0 * foo3
    to_right = xp.where(active, foo2 * foo3, z)  # contributes to s+1
    to_self = xp.where(active, tot * foo3, z)  # subtracted at s
    to_left = xp.where(active, foo1 * foo3, z)  # contributes to s-1
    pos_c = sl(d, 1, n_all - 1) > smeps  # dum at source-indexed point
    # targets t = sources 1 .. ns-2 (in source indexing), ns = n_all-2
    ns = n_all - 2
    L = sl(to_right, 0, ns - 2)
    M = sl(to_self, 1, ns - 1)
    R = sl(to_left, 2, ns)
    mass = (L - M) + R
    act_l = sl(active, 0, ns - 2)
    act_c = sl(active, 1, ns - 1)
    act_r = sl(active, 2, ns)
    pos = sl(pos_c, 1, ns - 1)
    flag = act_c | ((act_l | act_r) & pos)
    return mass, flag


def pdefx1(bk, g, *, uh, rho0, rru, advx, s0, s, dt, cfg=None, gz=None, **unused):
    """Port of ``pdefx1`` (Cartesian; periodic or open; with terrain the
    3-D rho0 and ``gz`` enter). Returns {'dum'} with the halo filled
    (``call bcs(dum)``)."""
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    rdx = g.rdx
    a = sub3(advx, K, J, I)
    inner = a + sub3(s, K, J, I) * (
        sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I)
    ) * rdx * xrow(uh, I)
    if _terrain(cfg):
        _need("pdefx1", gz=gz)
        v = sub3(rho0, K, J, I) * sub3(s0, K, J, I) + dt * inner * g2(gz, J, I)
    else:
        v = col(rho0, K) * sub3(s0, K, J, I) + dt * inner
    dum = put3(bk, bk.zeros(g.s_shape), K, J, I, v)
    ibw, ibe, _, _ = ibflags(cfg)
    if ibw:
        Ib = (1, 1)
        dum = _setwhere(
            bk,
            dum,
            K,
            J,
            Ib,
            sub3(rru, K, J, Ib) >= 0.0,
            sub3(rho0, K, J, Ib) * sub3(s0, K, J, Ib),
        )
    if ibe:
        Ib = (g.ni, g.ni)
        dum = _setwhere(
            bk,
            dum,
            K,
            J,
            Ib,
            sub3(rru, K, J, Ib, di=1) <= 0.0,
            sub3(rho0, K, J, Ib) * sub3(s0, K, J, Ib),
        )
    return {"dum": _bcs(bk, g, dum, cfg)}


def pdefx2(bk, g, *, uh, rho0, rru, advx, dum, s0, s, dt, cfg=None, rgz=None, **unused):
    """Port of ``pdefx2`` (Cartesian; terrain: 3-D rho0 and ``rgz``). ``dum``
    must have its halo filled. Returns {'advx'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    d = sub3(dum, K, J, (-1, g.ni + 2))
    mass, flag = _pd_mass_flag(bk, d, 2, smeps)
    rdt = 1.0 / dt
    rdx = g.rdx
    div = sub3(s, K, J, I) * (sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I))
    div = div * rdx * xrow(uh, I)
    if _terrain(cfg):
        _need("pdefx2", rgz=rgz)
        r0 = sub3(rho0, K, J, I)
        new = ((sub3(dum, K, J, I) + mass) - r0 * sub3(s0, K, J, I)) * rdt
        new = new * g2(rgz, J, I) - div
    else:
        r0 = col(rho0, K)
        new = ((sub3(dum, K, J, I) + mass) - r0 * sub3(s0, K, J, I)) * rdt - div
    v = xp.where(flag, new, sub3(advx, K, J, I))
    return {"advx": put3(bk, bk.copy(advx), K, J, I, v)}


def pdefy1(bk, g, *, vh, rho0, rrv, advy, s0, s, dt, cfg=None, gz=None, **unused):
    """Port of ``pdefy1`` (periodic or open; terrain: 3-D rho0 and ``gz``).
    Returns {'dum'} (halo filled)."""
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    rdy = g.rdy
    inner = sub3(advy, K, J, I) + sub3(s, K, J, I) * (
        sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I)
    ) * rdy * yrow(vh, J)
    if _terrain(cfg):
        _need("pdefy1", gz=gz)
        v = sub3(rho0, K, J, I) * sub3(s0, K, J, I) + dt * inner * g2(gz, J, I)
    else:
        v = col(rho0, K) * sub3(s0, K, J, I) + dt * inner
    dum = put3(bk, bk.zeros(g.s_shape), K, J, I, v)
    _, _, ibs, ibn = ibflags(cfg)
    if ibs:
        Jb = (1, 1)
        dum = _setwhere(
            bk,
            dum,
            K,
            Jb,
            I,
            sub3(rrv, K, Jb, I) >= 0.0,
            sub3(rho0, K, Jb, I) * sub3(s0, K, Jb, I),
        )
    if ibn:
        Jb = (g.nj, g.nj)
        dum = _setwhere(
            bk,
            dum,
            K,
            Jb,
            I,
            sub3(rrv, K, Jb, I, dj=1) <= 0.0,
            sub3(rho0, K, Jb, I) * sub3(s0, K, Jb, I),
        )
    return {"dum": _bcs(bk, g, dum, cfg)}


def pdefy2(bk, g, *, vh, rho0, rrv, advy, dum, s0, s, dt, cfg=None, rgz=None, **unused):
    """Port of ``pdefy2`` (terrain: 3-D rho0 and ``rgz``). Returns {'advy'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    d = sub3(dum, K, (-1, g.nj + 2), I)
    mass, flag = _pd_mass_flag(bk, d, 1, smeps)
    rdt = 1.0 / dt
    rdy = g.rdy
    div = sub3(s, K, J, I) * (sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I))
    div = div * rdy * yrow(vh, J)
    if _terrain(cfg):
        _need("pdefy2", rgz=rgz)
        r0 = sub3(rho0, K, J, I)
        new = ((sub3(dum, K, J, I) + mass) - r0 * sub3(s0, K, J, I)) * rdt
        new = new * g2(rgz, J, I) - div
    else:
        r0 = col(rho0, K)
        new = ((sub3(dum, K, J, I) + mass) - r0 * sub3(s0, K, J, I)) * rdt - div
    v = xp.where(flag, new, sub3(advy, K, J, I))
    return {"advy": put3(bk, bk.copy(advy), K, J, I, v)}


def pdefz(
    bk,
    g,
    *,
    mh,
    rho0,
    rrw,
    advz,
    s0,
    s,
    dt,
    cfg=None,
    gz=None,
    rgz=None,
    rdsf=None,
    **unused,
):
    """Port of ``pdefz`` (terrain: 3-D rho0, ``gz``, ``rgz``, ``rdsf``).
    Returns {'advz'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    nk = g.nk
    K, J, I = (1, nk), (1, g.nj), (1, g.ni)
    rdz = g.rdz
    terr = _terrain(cfg)
    if terr:
        _need("pdefz", gz=gz, rgz=rgz, rdsf=rdsf)
    a = sub3(advz, K, J, I)
    dw = sub3(s, K, J, I) * (sub3(rrw, K, J, I, dk=1) - sub3(rrw, K, J, I))
    if terr:
        r0 = sub3(rho0, K, J, I)
        dw = dw * kv(rdsf, K)
        dk = r0 * sub3(s0, K, J, I) + dt * (a + dw) * g2(gz, J, I)
    else:
        r0 = col(rho0, K)
        dw = dw * rdz * col(mh, K)
        dk = r0 * sub3(s0, K, J, I) + dt * (a + dw)
    zero = 0.0 * dk[:1]
    # dum(0)=dum(nk+1)=0; pad one more zero on each side for neighbour access
    d = xp.concat([zero, zero, dk, zero, zero], axis=0)
    mass, flag = _pd_mass_flag(bk, d, 0, smeps)
    rdt = 1.0 / dt
    new = ((dk + mass) - r0 * sub3(s0, K, J, I)) * rdt
    if terr:
        new = new * g2(rgz, J, I)
    new = new - dw
    v = xp.where(flag, new, a)
    return {"advz": put3(bk, bk.copy(advz), K, J, I, v)}


# ===========================================================================
# adv.F
# ===========================================================================


def advs(
    bk,
    g,
    *,
    nrk,
    wflag,
    rho0,
    rr0,
    mh,
    uh,
    vh,
    rru,
    rrv,
    rrw,
    divx,
    s0,
    s,
    sten,
    pdef,
    dt,
    c1,
    c2,
    hadvorder,
    vadvorder,
    diffit=0,
    dobud=False,
    bflag=0,
    weps=None,
    pdefweno=0,
    bsq=0.0,
    ruh=None,
    rvh=None,
    rmh=None,
    cfg=None,
    gz=None,
    rgz=None,
    rdsf=None,
):
    """Port of ``advs``: adds the advective tendency of scalar ``s`` to ``sten``.
    With ``terrain_flag`` the metrics ``gz``, ``rgz`` (2-D) and ``rdsf`` are
    required (vertical flux divergence with rdsf, tendency times rr0*gz).

    ``s0`` is the scalar at the start of the RK step (used by the pdef
    correction, active when ``pdef=1``). WENO (weno_order=5) is used when
    ``wflag=1`` and advwenos=1, or advwenos=2 on the last RK stage; ``weps``
    is then required. Open lateral boundaries (wbc/ebc/sbc/nbc=2) follow
    CM1's inflow treatment.

    Returns {'sten'}; with ``bflag=1``, stat_qsrc=1 and an open boundary it
    also returns 'bsq' (``bsq`` plus the lateral boundary flux of this
    stage; needs ``ruh``, ``rvh``, ``rmh``).
    """
    _check_common(cfg, "advs")
    nrkmax = _opt(cfg, "nrkmax")
    advwenos = _opt(cfg, "advwenos")
    doweno = wflag == 1 and (advwenos == 1 or (advwenos == 2 and nrk == nrkmax))
    _check_order("advs", hadvorder, vadvorder, doweno, cfg, weno_ok=True)
    if doweno and weps is None:
        raise ValueError("advs: weps is required when WENO is active")
    if diffit == 1:
        raise NotImplementedError("advs: diffit=1 (6th-order diffusion) not ported")
    if dobud:
        raise NotImplementedError("advs: budget diagnostics not ported")
    ni, nj = g.ni, g.nj
    K, J, I = (1, g.nk), (1, nj), (1, ni)
    rdx, rdy = g.rdx, g.rdy
    ibw, ibe, ibs, ibn = ibflags(cfg)
    terr = _terrain(cfg)
    if terr:
        _need("advs", gz=gz, rgz=rgz, rdsf=rdsf)
    tk = dict(gz=gz, rgz=rgz, rdsf=rdsf)

    if doweno:
        fl = hadv_weno5(
            bk,
            g,
            stag=1,
            c1=c1,
            c2=c2,
            rru=rru,
            rrv=rrv,
            a=s,
            pdef=pdef,
            weps=weps,
            cfg=cfg,
        )
    else:
        fl = hadv_flx5(bk, g, stag=1, c1=c1, c2=c2, rru=rru, rrv=rrv, a=s, cfg=cfg)
    dumx, dumy = fl["dumx"], fl["dumy"]

    # open bc: set hadv to zero at inflow
    if ibw:
        Ib = (1, 1)
        dumx = _setwhere(
            bk, dumx, K, J, Ib, sub3(rru, K, J, Ib) >= 0.0, sub3(dumx, K, J, Ib, di=1)
        )
    if ibe:
        Ib = (ni + 1, ni + 1)
        dumx = _setwhere(
            bk, dumx, K, J, Ib, sub3(rru, K, J, Ib) <= 0.0, sub3(dumx, K, J, Ib, di=-1)
        )
    if ibs:
        Jb = (1, 1)
        dumy = _setwhere(
            bk, dumy, K, Jb, I, sub3(rrv, K, Jb, I) >= 0.0, sub3(dumy, K, Jb, I, dj=1)
        )
    if ibn:
        Jb = (nj + 1, nj + 1)
        dumy = _setwhere(
            bk, dumy, K, Jb, I, sub3(rrv, K, Jb, I) <= 0.0, sub3(dumy, K, Jb, I, dj=-1)
        )

    # boundary-flux statistics (stat_qsrc): dumk1 (x), dumk2 (y)
    dobsq = bflag == 1 and _opt(cfg, "stat_qsrc") == 1
    out_bsq = None
    if dobsq and (ibw or ibe or ibs or ibn):
        xp = bk.xp
        out_bsq = bsq
        if ibw or ibe:
            dk1 = 0.0
            if ibw:
                Ib = (1, 1)
                dk1 = dk1 + xp.sum(
                    sub3(dumx, K, J, Ib) * yrow(rvh, J) * sub3(rmh, K, J, Ib),
                    axis=(1, 2),
                )
            if ibe:
                Ib = (ni + 1, ni + 1)
                dk1 = dk1 - xp.sum(
                    sub3(dumx, K, J, Ib) * yrow(rvh, J) * sub3(rmh, K, J, Ib),
                    axis=(1, 2),
                )
            out_bsq = out_bsq + xp.sum(dk1 * (dt * g.dy * g.dz))
        if ibs or ibn:
            dk2 = 0.0
            if ibs:
                Jb = (1, 1)
                dk2 = dk2 + xp.sum(
                    sub3(dumy, K, Jb, I) * xrow(ruh, I) * sub3(rmh, K, Jb, I),
                    axis=(1, 2),
                )
            if ibn:
                Jb = (nj + 1, nj + 1)
                dk2 = dk2 - xp.sum(
                    sub3(dumy, K, Jb, I) * xrow(ruh, I) * sub3(rmh, K, Jb, I),
                    axis=(1, 2),
                )
            out_bsq = out_bsq + xp.sum(dk2 * (dt * g.dx * g.dz))

    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)
    advx = put3(bk, bk.zeros(g.s_shape), K, J, I, advx)
    advy = put3(bk, bk.zeros(g.s_shape), K, J, I, advy)

    # open bc: subtract-off the divx piece at inflow points
    def _cx(advx, Ib, cond):
        new = sub3(advx, K, J, Ib) - sub3(s, K, J, Ib) * (
            sub3(rru, K, J, Ib, di=1) - sub3(rru, K, J, Ib)
        ) * rdx * xrow(uh, Ib)
        return _setwhere(bk, advx, K, J, Ib, cond, new)

    def _cy(advy, Jb, cond):
        new = sub3(advy, K, Jb, I) - sub3(s, K, Jb, I) * (
            sub3(rrv, K, Jb, I, dj=1) - sub3(rrv, K, Jb, I)
        ) * rdy * yrow(vh, Jb)
        return _setwhere(bk, advy, K, Jb, I, cond, new)

    if ibw:
        advx = _cx(advx, (1, 1), sub3(rru, K, J, (1, 1)) >= 0.0)
    if ibe:
        advx = _cx(advx, (ni, ni), sub3(rru, K, J, (ni + 1, ni + 1)) <= 0.0)
    if ibs:
        advy = _cy(advy, (1, 1), sub3(rrv, K, (1, 1), I) >= 0.0)
    if ibn:
        advy = _cy(advy, (nj, nj), sub3(rrv, K, (nj + 1, nj + 1), I) <= 0.0)

    dopd = _opt(cfg, "pdscheme") == 1 and pdef == 1
    if dopd:
        dumx = pdefx1(
            bk,
            g,
            uh=uh,
            rho0=rho0,
            rru=rru,
            advx=advx,
            s0=s0,
            s=s,
            dt=dt,
            cfg=cfg,
            **tk,
        )["dum"]
        dumy = pdefy1(
            bk,
            g,
            vh=vh,
            rho0=rho0,
            rrv=rrv,
            advy=advy,
            s0=s0,
            s=s,
            dt=dt,
            cfg=cfg,
            **tk,
        )["dum"]

    if doweno:
        dumz = vadv_weno5(
            bk,
            g,
            stag=1,
            c1=c1,
            c2=c2,
            rrw=rrw,
            a=s,
            pdef=pdef,
            pdefweno=pdefweno,
            weps=weps,
            cfg=cfg,
        )["dumz"]
    else:
        dumz = vadv_flx5(bk, g, stag=1, c1=c1, c2=c2, rrw=rrw, a=s, cfg=cfg)["dumz"]
    advz = put3(
        bk,
        bk.zeros(g.s_shape),
        K,
        J,
        I,
        _vflux_to_tend(bk, g, dumz, K, J, I, mh, rdsf if terr else None),
    )

    if dopd:
        advz = pdefz(
            bk,
            g,
            mh=mh,
            rho0=rho0,
            rrw=rrw,
            advz=advz,
            s0=s0,
            s=s,
            dt=dt,
            cfg=cfg,
            **tk,
        )["advz"]
        advx = pdefx2(
            bk,
            g,
            uh=uh,
            rho0=rho0,
            rru=rru,
            advx=advx,
            dum=dumx,
            s0=s0,
            s=s,
            dt=dt,
            cfg=cfg,
            **tk,
        )["advx"]
        advy = pdefy2(
            bk,
            g,
            vh=vh,
            rho0=rho0,
            rrv=rrv,
            advy=advy,
            dum=dumy,
            s0=s0,
            s=s,
            dt=dt,
            cfg=cfg,
            **tk,
        )["advy"]

    tend = (
        (sub3(advx, K, J, I) + sub3(advy, K, J, I))
        + sub3(advz, K, J, I)
        + sub3(s, K, J, I) * sub3(divx, K, J, I)
    )
    if terr:
        tend = tend * sub3(rr0, K, J, I) * g2(gz, J, I)
    else:
        tend = tend * col(rr0, K)
    sten = put3(bk, bk.copy(sten), K, J, I, sub3(sten, K, J, I) + tend)
    if out_bsq is not None:
        return {"sten": sten, "bsq": out_bsq}
    return {"sten": sten}


def advu(
    bk,
    g,
    *,
    nrk,
    rr0,
    mh,
    uf,
    vh,
    rru,
    rrv,
    rrw,
    u3d,
    uten,
    divx,
    c1,
    c2,
    dt=None,
    doubud=False,
    cfg=None,
    rho0=None,
    gzu=None,
    rdsf=None,
    **unused,
):
    """Port of ``advu`` (Cartesian; periodic or open). Returns {'uten'}.
    With ``terrain_flag`` needs ``rho0``, ``gzu`` and ``rdsf``.

    Order options come from cfg (hadvordrv, vadvordrv, advwenov, weno_order).
    """
    _check_common(cfg, "advu")
    _check_diff(cfg, "advu")
    nrkmax = _opt(cfg, "nrkmax")
    advwenov = _opt(cfg, "advwenov")
    doweno = advwenov == 1 or (advwenov == 2 and nrk == nrkmax)
    _check_order("advu", _opt(cfg, "hadvordrv"), _opt(cfg, "vadvordrv"), doweno)
    if doubud:
        raise NotImplementedError("advu: budget diagnostics not ported")
    nj = g.nj
    _, _, ibs, ibn = ibflags(cfg)
    ibw, ibe = ibflags(cfg)[:2]
    K, J = (1, g.nk), (1, nj)
    I = (2 if ibw else 1, g.ni if ibe else g.ni + 1)  # i1..i2
    rdx, rdy = g.rdx, g.rdy

    fl = hadv_flx5(bk, g, stag=2, c1=c1, c2=c2, rru=rru, rrv=rrv, a=u3d, cfg=cfg)
    dumx, dumy = fl["dumx"], fl["dumy"]
    if ibs:
        Jb = (1, 1)
        cond = (sub3(rrv, K, Jb, I) + sub3(rrv, K, Jb, I, di=-1)) >= 0.0
        dumy = _setwhere(bk, dumy, K, Jb, I, cond, sub3(dumy, K, Jb, I, dj=1))
    if ibn:
        Jb = (nj + 1, nj + 1)
        cond = (sub3(rrv, K, Jb, I) + sub3(rrv, K, Jb, I, di=-1)) <= 0.0
        dumy = _setwhere(bk, dumy, K, Jb, I, cond, sub3(dumy, K, Jb, I, dj=-1))
    advx = -(sub3(dumx, K, J, I) - sub3(dumx, K, J, I, di=-1)) * rdx * xrow(uf, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)

    def _cy(advy, j, jc, ge):
        # advy(:,j) -= u*0.5*((rrv(i-1,j+1)-rrv(i-1,j))+(rrv(i,j+1)-rrv(i,j)))*rdy*vh(j)
        Jb = (j, j)
        v = sub3(rrv, K, (jc, jc), I) + sub3(rrv, K, (jc, jc), I, di=-1)
        cond = v >= 0.0 if ge else v <= 0.0
        old = advy[:, j - 1 : j, :]
        new = old - sub3(u3d, K, Jb, I) * 0.5 * (
            (sub3(rrv, K, Jb, I, dj=1, di=-1) - sub3(rrv, K, Jb, I, di=-1))
            + (sub3(rrv, K, Jb, I, dj=1) - sub3(rrv, K, Jb, I))
        ) * rdy * yrow(vh, Jb)
        xp = bk.xp
        return xp.concat(
            [advy[:, : j - 1, :], xp.where(cond, new, old), advy[:, j:, :]], axis=1
        )

    if ibs:
        advy = _cy(advy, 1, 1, True)
    if ibn:
        advy = _cy(advy, nj, nj + 1, False)

    terr = _terrain(cfg)
    if terr:
        _need("advu", rho0=rho0, gzu=gzu, rdsf=rdsf)
    dumz = vadv_flx5(bk, g, stag=2, c1=c1, c2=c2, rrw=rrw, a=u3d, cfg=cfg)["dumz"]
    advz = _vflux_to_tend(bk, g, dumz, K, J, I, mh, rdsf if terr else None)

    tend = (
        (advx + advy)
        + advz
        + sub3(u3d, K, J, I) * 0.5 * (sub3(divx, K, J, I) + sub3(divx, K, J, I, di=-1))
    )
    if terr:
        rb = 0.5 * (sub3(rho0, K, J, I, di=-1) + sub3(rho0, K, J, I))
        tend = tend * g2(gzu, J, I) / rb
    else:
        tend = tend * col(rr0, K)
    uten = put3(bk, bk.copy(uten), K, J, I, sub3(uten, K, J, I) + tend)
    return {"uten": uten}


def advv(
    bk,
    g,
    *,
    nrk,
    rr0,
    mh,
    uh,
    vf,
    rru,
    rrv,
    rrw,
    v3d,
    vten,
    divx,
    c1,
    c2,
    dt=None,
    dovbud=False,
    cfg=None,
    rho0=None,
    gzv=None,
    rdsf=None,
    **unused,
):
    """Port of ``advv`` (Cartesian; periodic or open). Returns {'vten'}.
    With ``terrain_flag`` needs ``rho0``, ``gzv`` and ``rdsf``."""
    _check_common(cfg, "advv")
    _check_diff(cfg, "advv")
    nrkmax = _opt(cfg, "nrkmax")
    advwenov = _opt(cfg, "advwenov")
    doweno = advwenov == 1 or (advwenov == 2 and nrk == nrkmax)
    _check_order("advv", _opt(cfg, "hadvordrv"), _opt(cfg, "vadvordrv"), doweno)
    if dovbud:
        raise NotImplementedError("advv: budget diagnostics not ported")
    ni = g.ni
    ibw, ibe, ibs, ibn = ibflags(cfg)
    K, I = (1, g.nk), (1, ni)
    J = (2 if ibs else 1, g.nj if ibn else g.nj + 1)  # j1..j2
    rdx, rdy = g.rdx, g.rdy

    fl = hadv_flx5(bk, g, stag=3, c1=c1, c2=c2, rru=rru, rrv=rrv, a=v3d, cfg=cfg)
    dumx, dumy = fl["dumx"], fl["dumy"]
    if ibw:
        Ib = (1, 1)
        cond = (sub3(rru, K, J, Ib) + sub3(rru, K, J, Ib, dj=-1)) >= 0.0
        dumx = _setwhere(bk, dumx, K, J, Ib, cond, sub3(dumx, K, J, Ib, di=1))
    if ibe:
        Ib = (ni + 1, ni + 1)
        cond = (sub3(rru, K, J, Ib) + sub3(rru, K, J, Ib, dj=-1)) <= 0.0
        dumx = _setwhere(bk, dumx, K, J, Ib, cond, sub3(dumx, K, J, Ib, di=-1))
    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I) - sub3(dumy, K, J, I, dj=-1)) * rdy * yrow(vf, J)

    def _cx(advx, i, ic, ge):
        # advx(i,:) -= v*0.5*((rru(i+1,j-1)-rru(i,j-1))+(rru(i+1,j)-rru(i,j)))*rdx*uh(i)
        Ib = (i, i)
        u = sub3(rru, K, J, (ic, ic)) + sub3(rru, K, J, (ic, ic), dj=-1)
        cond = u >= 0.0 if ge else u <= 0.0
        old = advx[:, :, i - 1 : i]
        new = old - sub3(v3d, K, J, Ib) * 0.5 * (
            (sub3(rru, K, J, Ib, di=1, dj=-1) - sub3(rru, K, J, Ib, dj=-1))
            + (sub3(rru, K, J, Ib, di=1) - sub3(rru, K, J, Ib))
        ) * rdx * xrow(uh, Ib)
        xp = bk.xp
        return xp.concat(
            [advx[:, :, : i - 1], xp.where(cond, new, old), advx[:, :, i:]], axis=2
        )

    if ibw:
        advx = _cx(advx, 1, 1, True)
    if ibe:
        advx = _cx(advx, ni, ni + 1, False)

    terr = _terrain(cfg)
    if terr:
        _need("advv", rho0=rho0, gzv=gzv, rdsf=rdsf)
    dumz = vadv_flx5(bk, g, stag=3, c1=c1, c2=c2, rrw=rrw, a=v3d, cfg=cfg)["dumz"]
    advz = _vflux_to_tend(bk, g, dumz, K, J, I, mh, rdsf if terr else None)

    tend = (
        (advx + advy)
        + advz
        + sub3(v3d, K, J, I) * 0.5 * (sub3(divx, K, J, I) + sub3(divx, K, J, I, dj=-1))
    )
    if terr:
        rb = 0.5 * (sub3(rho0, K, J, I, dj=-1) + sub3(rho0, K, J, I))
        tend = tend * g2(gzv, J, I) / rb
    else:
        tend = tend * col(rr0, K)
    vten = put3(bk, bk.copy(vten), K, J, I, sub3(vten, K, J, I) + tend)
    return {"vten": vten}


def advw(
    bk,
    g,
    *,
    nrk,
    rrf0,
    mf,
    uh,
    vh,
    rru,
    rrv,
    rrw,
    w3d,
    wten,
    divx,
    c1,
    c2,
    hadvorder,
    vadvorder,
    advweno,
    dt=None,
    dowbud=False,
    iamw=True,
    cfg=None,
    gz=None,
    rds=None,
    **unused,
):
    """Port of ``advw`` (Cartesian; periodic or open). Updates wten at
    k=2..nk. Returns {'wten'}.  With ``terrain_flag`` needs ``gz`` and
    ``rds``."""
    _check_common(cfg, "advw")
    _check_diff(cfg, "advw")
    nrkmax = _opt(cfg, "nrkmax")
    doweno = advweno == 1 or (advweno == 2 and nrk == nrkmax)
    _check_order("advw", hadvorder, vadvorder, doweno, cfg, weno_ok=True)
    weps = 100.0 * 1.0e-18  # advw.F: weps = 100.0*epsilon
    if dowbud:
        raise NotImplementedError("advw: budget diagnostics not ported")
    nk = g.nk
    K, J, I = (2, nk), (1, g.nj), (1, g.ni)
    rdx, rdy, rdz = g.rdx, g.rdy, g.rdz

    ni, nj = g.ni, g.nj
    ibw, ibe, ibs, ibn = ibflags(cfg)

    def ubar(Ib):
        cc2 = 0.5 * (sub3(c2, K, J, Ib, di=-1) + sub3(c2, K, J, Ib))
        cc1 = 1.0 - cc2
        return cc2 * sub3(rru, K, J, Ib) + cc1 * sub3(rru, K, J, Ib, dk=-1)

    def vbar(Jb):
        cc2 = 0.5 * (sub3(c2, K, Jb, I, dj=-1) + sub3(c2, K, Jb, I))
        cc1 = 1.0 - cc2
        return cc2 * sub3(rrv, K, Jb, I) + cc1 * sub3(rrv, K, Jb, I, dk=-1)

    if doweno:
        fl = hadv_weno5(
            bk,
            g,
            stag=4,
            c1=c1,
            c2=c2,
            rru=rru,
            rrv=rrv,
            a=w3d,
            pdef=0,
            weps=weps,
            cfg=cfg,
        )
    else:
        fl = hadv_flx5(bk, g, stag=4, c1=c1, c2=c2, rru=rru, rrv=rrv, a=w3d, cfg=cfg)
    dumx, dumy = fl["dumx"], fl["dumy"]
    if ibw:
        Ib = (1, 1)
        dumx = _setwhere(
            bk, dumx, K, J, Ib, ubar(Ib) >= 0.0, sub3(dumx, K, J, Ib, di=1)
        )
    if ibe:
        Ib = (ni + 1, ni + 1)
        dumx = _setwhere(
            bk, dumx, K, J, Ib, ubar(Ib) <= 0.0, sub3(dumx, K, J, Ib, di=-1)
        )
    if ibs:
        Jb = (1, 1)
        dumy = _setwhere(
            bk, dumy, K, Jb, I, vbar(Jb) >= 0.0, sub3(dumy, K, Jb, I, dj=1)
        )
    if ibn:
        Jb = (nj + 1, nj + 1)
        dumy = _setwhere(
            bk, dumy, K, Jb, I, vbar(Jb) <= 0.0, sub3(dumy, K, Jb, I, dj=-1)
        )
    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)
    xp = bk.xp

    def _cx(advx, i, ic, ge):
        Ib = (i, i)
        ub = ubar((ic, ic))
        cond = ub >= 0.0 if ge else ub <= 0.0
        old = advx[:, :, i - 1 : i]
        new = old - sub3(w3d, K, J, Ib) * (
            sub3(c1, K, J, Ib)
            * (sub3(rru, K, J, Ib, di=1, dk=-1) - sub3(rru, K, J, Ib, dk=-1))
            + sub3(c2, K, J, Ib) * (sub3(rru, K, J, Ib, di=1) - sub3(rru, K, J, Ib))
        ) * rdx * xrow(uh, Ib)
        return xp.concat(
            [advx[:, :, : i - 1], xp.where(cond, new, old), advx[:, :, i:]], axis=2
        )

    def _cy(advy, j, jc, ge):
        Jb = (j, j)
        vb = vbar((jc, jc))
        cond = vb >= 0.0 if ge else vb <= 0.0
        old = advy[:, j - 1 : j, :]
        new = old - sub3(w3d, K, Jb, I) * (
            sub3(c1, K, Jb, I)
            * (sub3(rrv, K, Jb, I, dj=1, dk=-1) - sub3(rrv, K, Jb, I, dk=-1))
            + sub3(c2, K, Jb, I) * (sub3(rrv, K, Jb, I, dj=1) - sub3(rrv, K, Jb, I))
        ) * rdy * yrow(vh, Jb)
        return xp.concat(
            [advy[:, : j - 1, :], xp.where(cond, new, old), advy[:, j:, :]], axis=1
        )

    if ibw:
        advx = _cx(advx, 1, 1, True)
    if ibe:
        advx = _cx(advx, ni, ni + 1, False)
    if ibs:
        advy = _cy(advy, 1, 1, True)
    if ibn:
        advy = _cy(advy, nj, nj + 1, False)

    if doweno:
        dumz = vadv_weno5(
            bk,
            g,
            stag=4,
            c1=c1,
            c2=c2,
            rrw=rrw,
            a=w3d,
            pdef=0,
            pdefweno=0,
            weps=weps,
            cfg=cfg,
        )["dumz"]
    else:
        dumz = vadv_flx5(bk, g, stag=4, c1=c1, c2=c2, rrw=rrw, a=w3d, cfg=cfg)["dumz"]
    if _terrain(cfg):
        _need("advw", gz=gz, rds=rds)
        advz = -(sub3(dumz, K, J, I) - sub3(dumz, K, J, I, dk=-1)) * kv(rds, K)
        tend = (
            (advx + advy)
            + advz
            + sub3(w3d, K, J, I)
            * (
                sub3(c2, K, J, I) * sub3(divx, K, J, I)
                + sub3(c1, K, J, I) * sub3(divx, K, J, I, dk=-1)
            )
        )
        tend = tend * sub3(rrf0, K, J, I) * g2(gz, J, I)
    else:
        advz = -(sub3(dumz, K, J, I) - sub3(dumz, K, J, I, dk=-1)) * rdz * col(mf, K)
        tend = (
            (advx + advy)
            + advz
            + sub3(w3d, K, J, I)
            * (
                col(c2, K) * sub3(divx, K, J, I)
                + col(c1, K) * sub3(divx, K, J, I, dk=-1)
            )
        ) * col(rrf0, K)
    wten = put3(bk, bk.copy(wten), K, J, I, sub3(wten, K, J, I) + tend)
    return {"wten": wten}


# ===========================================================================
# misclibs.F: getdivx
# ===========================================================================


def sigdot(bk, w0, u, v, K, J, I, *, rds, sigma, sigmaf, gz, dzdx, dzdy, zt):
    """Terrain-following vertical mass flux on w levels K (k >= 2), the
    expression shared by solve2 (rrw), getdivx and getdiv:

        w0 + ( (r2*(u(i)+u(i+1))(k) + r1*(u(i)+u(i+1))(k-1))*dzdx
             + (r2*(v(j)+v(j+1))(k) + r1*(v(j)+v(j+1))(k-1))*dzdy
             )*(sigmaf(k)-zt)*gz*rzt

    with r2 = 0.5*(sigmaf(k)-sigma(k-1))*rds(k), r1 = 0.5*(1-...).  ``w0``
    is the (already weighted) w term on (K, J, I)."""
    S = bk.asarray
    r2 = (kv(sigmaf, K) - kv(sigma, K, dk=-1)) * kv(rds, K)
    r1 = S(1.0) - r2
    r1 = 0.5 * r1
    r2 = 0.5 * r2
    zts = S(float(zt))
    rzt = S(1.0 / float(zt))
    ux = r2 * (sub3(u, K, J, I) + sub3(u, K, J, I, di=1)) + r1 * (
        sub3(u, K, J, I, dk=-1) + sub3(u, K, J, I, dk=-1, di=1)
    )
    vy = r2 * (sub3(v, K, J, I) + sub3(v, K, J, I, dj=1)) + r1 * (
        sub3(v, K, J, I, dk=-1) + sub3(v, K, J, I, dk=-1, dj=1)
    )
    return (
        w0
        + (ux * g2(dzdx, J, I) + vy * g2(dzdy, J, I))
        * (kv(sigmaf, K) - zts)
        * g2(gz, J, I)
        * rzt
    )


def getdivx(
    bk,
    g,
    *,
    uh,
    vh,
    mh,
    rho0,
    rf0,
    rru,
    rrv,
    rrw,
    divx,
    cfg=None,
    rds=None,
    rdsf=None,
    sigma=None,
    sigmaf=None,
    gz=None,
    rgzu=None,
    rgzv=None,
    dzdx=None,
    dzdy=None,
    zt=None,
    **unused,
):
    """Port of ``getdivx`` (Cartesian): converts the (averaged) velocities in
    rru/rrv/rrw to mass fluxes in the interior and recomputes divx in the
    interior. With ``terrain_flag`` the metric arguments (rds, rdsf, sigma,
    sigmaf, gz, rgzu, rgzv, dzdx, dzdy and the Python float zt) are needed.
    Returns {'rru', 'rrv', 'rrw', 'divx'}."""
    _check_common(cfg, "getdivx")
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    nk = g.nk
    K, J, I = (1, nk), (1, g.nj), (1, g.ni)
    if _terrain(cfg):
        _need(
            "getdivx",
            rds=rds,
            rdsf=rdsf,
            sigma=sigma,
            sigmaf=sigmaf,
            gz=gz,
            rgzu=rgzu,
            rgzv=rgzv,
            dzdx=dzdx,
            dzdy=dzdy,
            zt=zt,
        )
        Iu, Jv = (1, g.ni + 1), (1, g.nj + 1)
        ru = 0.5 * (sub3(rho0, K, J, Iu, di=-1) + sub3(rho0, K, J, Iu))
        ru = ru * sub3(rru, K, J, Iu) * g2(rgzu, J, Iu)
        rru = put3(bk, bk.copy(rru), K, J, Iu, ru)
        rv = 0.5 * (sub3(rho0, K, Jv, I, dj=-1) + sub3(rho0, K, Jv, I))
        rv = rv * sub3(rrv, K, Jv, I) * g2(rgzv, Jv, I)
        rrv = put3(bk, bk.copy(rrv), K, Jv, I, rv)
        rrw = bk.copy(rrw)
        z = bk.zeros(sub3(rrw, (1, 1), J, I).shape)
        rrw = put3(bk, rrw, (1, 1), J, I, z)
        rrw = put3(bk, rrw, (nk + 1, nk + 1), J, I, z)
        if nk >= 2:
            Kw = (2, nk)
            rrw = put3(
                bk,
                rrw,
                Kw,
                J,
                I,
                sigdot(
                    bk,
                    sub3(rf0, Kw, J, I) * sub3(rrw, Kw, J, I),
                    rru,
                    rrv,
                    Kw,
                    J,
                    I,
                    rds=rds,
                    sigma=sigma,
                    sigmaf=sigmaf,
                    gz=gz,
                    dzdx=dzdx,
                    dzdy=dzdy,
                    zt=zt,
                ),
            )
        d = (
            (sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I)) * g.rdx * xrow(uh, I)
            + (sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I)) * g.rdy * yrow(vh, J)
        ) + (sub3(rrw, K, J, I, dk=1) - sub3(rrw, K, J, I)) * kv(rdsf, K)
        d = xp.where(xp.abs(d) < smeps, 0.0 * d, d)
        divx = put3(bk, bk.copy(divx), K, J, I, d)
        return {"rru": rru, "rrv": rrv, "rrw": rrw, "divx": divx}
    r0 = col(rho0, K)
    rru = put3(
        bk, bk.copy(rru), K, J, (1, g.ni + 1), sub3(rru, K, J, (1, g.ni + 1)) * r0
    )
    rrv = put3(
        bk, bk.copy(rrv), K, (1, g.nj + 1), I, sub3(rrv, K, (1, g.nj + 1), I) * r0
    )
    rrw = bk.copy(rrw)
    if nk >= 2:
        rrw = put3(bk, rrw, (2, nk), J, I, sub3(rrw, (2, nk), J, I) * col(rf0, (2, nk)))
    z = bk.zeros(sub3(rrw, (1, 1), J, I).shape)
    rrw = put3(bk, rrw, (1, 1), J, I, z)
    rrw = put3(bk, rrw, (nk + 1, nk + 1), J, I, z)
    d = (
        (sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I)) * g.rdx * xrow(uh, I)
        + (sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I)) * g.rdy * yrow(vh, J)
    ) + (sub3(rrw, K, J, I, dk=1) - sub3(rrw, K, J, I)) * g.rdz * col(mh, K)
    d = xp.where(xp.abs(d) < smeps, 0.0 * d, d)
    divx = put3(bk, bk.copy(divx), K, J, I, d)
    return {"rru": rru, "rrv": rrv, "rrw": rrw, "divx": divx}


__all__ = [
    "advs",
    "advu",
    "advv",
    "advw",
    "pdefx1",
    "pdefx2",
    "pdefy1",
    "pdefy2",
    "pdefz",
    "getdivx",
    "sigdot",
    "DEFAULTS",
]
