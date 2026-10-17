"""Advection tendencies, ported from src/adv.F (advs, advu, advv, advw).

Also contains, because they are only used by these routines:
  * the positive-definite correction of src/pdef.F (pdefx1, pdefx2, pdefy1,
    pdefy2, pdefz), active in ``advs`` when ``pdef=1`` (CM1 hard-wires
    pdscheme=1; solve2 passes pdef=1 for moisture on the last RK stage);
  * ``getdivx`` from src/misclibs.F (rescales the small-step-averaged
    velocities to mass fluxes and recomputes divx).

Supported configuration (the reference case kessler_periodic): flux-form
advection of order 5 in both directions, no WENO, no terrain, Cartesian grid,
periodic lateral boundaries, no immersed boundaries, no large-scale subsidence,
no 6th-order artificial diffusion, no budgets. Any other option raises
NotImplementedError.

Options come from ``cfg`` (a namelist-like object or dict; missing entries fall
back to ``DEFAULTS``, which mirror ports/reference/kessler_periodic).

Array conventions (README): ``a[k, j, i]`` with CM1 halos. 1-D x arrays
(``uh``, ``uf``) are length ``ni+2*ngxy`` (``+1`` for ``uf``), 1-D y arrays
(``vh``, ``vf``) length ``nj+2*ngxy`` (``+1`` for ``vf``). Base-state arrays
(``rho0``, ``rr0``, ``rrf0``, ``mh``, ``mf``) are 3-D as stored by CM1; only
their ``(1,1,k)`` column is used here (no terrain).
"""

from __future__ import annotations

from .adv_routines import col, hadv_flx5, put3, sub3, vadv_flx5, xrow, yrow
from .bc import periodic_xy

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
    smeps=1.0e-30,
    nrkmax=3,
)


def _opt(cfg, name):
    if cfg is None:
        return DEFAULTS[name]
    if isinstance(cfg, dict):
        return cfg.get(name, DEFAULTS[name])
    return getattr(cfg, name, DEFAULTS[name])


def _check_common(cfg, who):
    if _opt(cfg, "terrain_flag"):
        raise NotImplementedError(f"{who}: terrain_flag=.true. not ported")
    if _opt(cfg, "axisymm") != 0:
        raise NotImplementedError(f"{who}: axisymm=1 not ported")
    for b in ("wbc", "ebc", "sbc", "nbc"):
        if _opt(cfg, b) != 1:
            raise NotImplementedError(
                f"{who}: only periodic lateral BCs (={b}=1) are ported"
            )
    if _opt(cfg, "do_ib"):
        raise NotImplementedError(f"{who}: immersed boundaries not ported")
    if _opt(cfg, "dolsw"):
        raise NotImplementedError(f"{who}: large-scale subsidence (dolsw) not ported")


def _check_order(who, hadvorder, vadvorder, doweno):
    if doweno:
        raise NotImplementedError(f"{who}: WENO advection not ported")
    if hadvorder != 5 or vadvorder != 5:
        raise NotImplementedError(
            f"{who}: only 5th-order advection ported (got {hadvorder},{vadvorder})"
        )


def _check_diff(cfg, who):
    if _opt(cfg, "idiff") >= 1 and _opt(cfg, "difforder") == 6:
        raise NotImplementedError(f"{who}: 6th-order artificial diffusion not ported")


def _vflux_to_tend(bk, g, dumz, K, J, I, mh):
    """advz(k) = -(dumz(k+1)-dumz(k))*rdz*mh(1,1,k), with the CM1 end cases
    advz(1) = -dumz(2)*.., advz(nk) = +dumz(nk)*.. (no terrain)."""
    xp = bk.xp
    nk = g.nk
    rdz = g.rdz
    top = sub3(dumz, (2, nk), J, I)
    zero = bk.zeros(top[:1].shape)
    up = xp.concat([top, zero], axis=0)  # dumz(k+1), k=1..nk; 0 at k=nk
    dn = xp.concat([zero, top], axis=0)  # dumz(k), k=1..nk; 0 at k=1
    # k=1: -(dumz(2)-0) == -dumz(2); k=nk: -(0-dumz(nk)) == +dumz(nk)  (exact)
    first = -up[:1] * rdz * col(mh, (1, 1))
    last = (dn[-1:]) * rdz * col(mh, (nk, nk))
    mid = -(up[1:-1] - dn[1:-1]) * rdz * col(mh, (2, nk - 1))
    return xp.concat([first, mid, last], axis=0)


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


def pdefx1(bk, g, *, uh, rho0, rru, advx, s0, s, dt, cfg=None):
    """Port of ``pdefx1`` (no terrain, Cartesian, periodic). Returns {'dum'}
    with the periodic halo filled (``call bcs(dum)``)."""
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    rdx = g.rdx
    a = sub3(advx, K, J, I)
    v = col(rho0, K) * sub3(s0, K, J, I) + dt * (
        a
        + sub3(s, K, J, I)
        * (sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I))
        * rdx
        * xrow(uh, I)
    )
    dum = put3(bk, bk.zeros(g.s_shape), K, J, I, v)
    return {"dum": periodic_xy(bk, dum, g)}


def pdefx2(bk, g, *, uh, rho0, rru, advx, dum, s0, s, dt, cfg=None):
    """Port of ``pdefx2`` (no terrain, Cartesian). ``dum`` must have its
    periodic halo filled. Returns {'advx'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    d = sub3(dum, K, J, (-1, g.ni + 2))
    mass, flag = _pd_mass_flag(bk, d, 2, smeps)
    rdt = 1.0 / dt
    rdx = g.rdx
    new = ((sub3(dum, K, J, I) + mass) - col(rho0, K) * sub3(s0, K, J, I)) * rdt - sub3(
        s, K, J, I
    ) * (sub3(rru, K, J, I, di=1) - sub3(rru, K, J, I)) * rdx * xrow(uh, I)
    v = xp.where(flag, new, sub3(advx, K, J, I))
    return {"advx": put3(bk, bk.copy(advx), K, J, I, v)}


def pdefy1(bk, g, *, vh, rho0, rrv, advy, s0, s, dt, cfg=None):
    """Port of ``pdefy1`` (no terrain, periodic). Returns {'dum'} (halo filled)."""
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    rdy = g.rdy
    v = col(rho0, K) * sub3(s0, K, J, I) + dt * (
        sub3(advy, K, J, I)
        + sub3(s, K, J, I)
        * (sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I))
        * rdy
        * yrow(vh, J)
    )
    dum = put3(bk, bk.zeros(g.s_shape), K, J, I, v)
    return {"dum": periodic_xy(bk, dum, g)}


def pdefy2(bk, g, *, vh, rho0, rrv, advy, dum, s0, s, dt, cfg=None):
    """Port of ``pdefy2`` (no terrain). Returns {'advy'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    d = sub3(dum, K, (-1, g.nj + 2), I)
    mass, flag = _pd_mass_flag(bk, d, 1, smeps)
    rdt = 1.0 / dt
    rdy = g.rdy
    new = ((sub3(dum, K, J, I) + mass) - col(rho0, K) * sub3(s0, K, J, I)) * rdt - sub3(
        s, K, J, I
    ) * (sub3(rrv, K, J, I, dj=1) - sub3(rrv, K, J, I)) * rdy * yrow(vh, J)
    v = xp.where(flag, new, sub3(advy, K, J, I))
    return {"advy": put3(bk, bk.copy(advy), K, J, I, v)}


def pdefz(bk, g, *, mh, rho0, rrw, advz, s0, s, dt, cfg=None):
    """Port of ``pdefz`` (no terrain). Returns {'advz'}."""
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    nk = g.nk
    K, J, I = (1, nk), (1, g.nj), (1, g.ni)
    rdz = g.rdz
    a = sub3(advz, K, J, I)
    dk = col(rho0, K) * sub3(s0, K, J, I) + dt * (
        a
        + sub3(s, K, J, I)
        * (sub3(rrw, K, J, I, dk=1) - sub3(rrw, K, J, I))
        * rdz
        * col(mh, K)
    )
    zero = 0.0 * dk[:1]
    # dum(0)=dum(nk+1)=0; pad one more zero on each side for neighbour access
    d = xp.concat([zero, zero, dk, zero, zero], axis=0)
    mass, flag = _pd_mass_flag(bk, d, 0, smeps)
    rdt = 1.0 / dt
    new = ((dk + mass) - col(rho0, K) * sub3(s0, K, J, I)) * rdt - sub3(s, K, J, I) * (
        sub3(rrw, K, J, I, dk=1) - sub3(rrw, K, J, I)
    ) * rdz * col(mh, K)
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
    cfg=None,
):
    """Port of ``advs``: adds the advective tendency of scalar ``s`` to ``sten``.

    ``s0`` is the scalar at the start of the RK step (used by the pdef
    correction, active when ``pdef=1``). Returns {'sten'}.
    """
    _check_common(cfg, "advs")
    nrkmax = _opt(cfg, "nrkmax")
    advwenos = _opt(cfg, "advwenos")
    doweno = wflag == 1 and (advwenos == 1 or (advwenos == 2 and nrk == nrkmax))
    _check_order("advs", hadvorder, vadvorder, doweno)
    if diffit == 1:
        raise NotImplementedError("advs: diffit=1 (6th-order diffusion) not ported")
    if dobud:
        raise NotImplementedError("advs: budget diagnostics not ported")
    K, J, I = (1, g.nk), (1, g.nj), (1, g.ni)
    rdx, rdy = g.rdx, g.rdy

    fl = hadv_flx5(bk, g, stag=1, c1=c1, c2=c2, rru=rru, rrv=rrv, a=s)
    dumx, dumy = fl["dumx"], fl["dumy"]
    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)
    advx = put3(bk, bk.zeros(g.s_shape), K, J, I, advx)
    advy = put3(bk, bk.zeros(g.s_shape), K, J, I, advy)

    dopd = _opt(cfg, "pdscheme") == 1 and pdef == 1
    if dopd:
        dumx = pdefx1(
            bk, g, uh=uh, rho0=rho0, rru=rru, advx=advx, s0=s0, s=s, dt=dt, cfg=cfg
        )["dum"]
        dumy = pdefy1(
            bk, g, vh=vh, rho0=rho0, rrv=rrv, advy=advy, s0=s0, s=s, dt=dt, cfg=cfg
        )["dum"]

    dumz = vadv_flx5(bk, g, stag=1, c1=c1, c2=c2, rrw=rrw, a=s)["dumz"]
    advz = put3(
        bk, bk.zeros(g.s_shape), K, J, I, _vflux_to_tend(bk, g, dumz, K, J, I, mh)
    )

    if dopd:
        advz = pdefz(
            bk, g, mh=mh, rho0=rho0, rrw=rrw, advz=advz, s0=s0, s=s, dt=dt, cfg=cfg
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
        )["advy"]

    tend = (
        (sub3(advx, K, J, I) + sub3(advy, K, J, I))
        + sub3(advz, K, J, I)
        + sub3(s, K, J, I) * sub3(divx, K, J, I)
    ) * col(rr0, K)
    sten = put3(bk, bk.copy(sten), K, J, I, sub3(sten, K, J, I) + tend)
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
    **unused,
):
    """Port of ``advu`` (Cartesian, no terrain, periodic). Returns {'uten'}.

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
    K, J = (1, g.nk), (1, g.nj)
    I = (1, g.ni + 1)  # periodic: i1=1, i2=ni+1
    rdx, rdy = g.rdx, g.rdy

    fl = hadv_flx5(bk, g, stag=2, c1=c1, c2=c2, rru=rru, rrv=rrv, a=u3d)
    dumx, dumy = fl["dumx"], fl["dumy"]
    advx = -(sub3(dumx, K, J, I) - sub3(dumx, K, J, I, di=-1)) * rdx * xrow(uf, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)

    dumz = vadv_flx5(bk, g, stag=2, c1=c1, c2=c2, rrw=rrw, a=u3d)["dumz"]
    advz = _vflux_to_tend(bk, g, dumz, K, J, I, mh)

    tend = (
        (advx + advy)
        + advz
        + sub3(u3d, K, J, I) * 0.5 * (sub3(divx, K, J, I) + sub3(divx, K, J, I, di=-1))
    ) * col(rr0, K)
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
    **unused,
):
    """Port of ``advv`` (Cartesian, no terrain, periodic). Returns {'vten'}."""
    _check_common(cfg, "advv")
    _check_diff(cfg, "advv")
    nrkmax = _opt(cfg, "nrkmax")
    advwenov = _opt(cfg, "advwenov")
    doweno = advwenov == 1 or (advwenov == 2 and nrk == nrkmax)
    _check_order("advv", _opt(cfg, "hadvordrv"), _opt(cfg, "vadvordrv"), doweno)
    if dovbud:
        raise NotImplementedError("advv: budget diagnostics not ported")
    K, I = (1, g.nk), (1, g.ni)
    J = (1, g.nj + 1)  # periodic: j1=1, j2=nj+1
    rdx, rdy = g.rdx, g.rdy

    fl = hadv_flx5(bk, g, stag=3, c1=c1, c2=c2, rru=rru, rrv=rrv, a=v3d)
    dumx, dumy = fl["dumx"], fl["dumy"]
    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I) - sub3(dumy, K, J, I, dj=-1)) * rdy * yrow(vf, J)

    dumz = vadv_flx5(bk, g, stag=3, c1=c1, c2=c2, rrw=rrw, a=v3d)["dumz"]
    advz = _vflux_to_tend(bk, g, dumz, K, J, I, mh)

    tend = (
        (advx + advy)
        + advz
        + sub3(v3d, K, J, I) * 0.5 * (sub3(divx, K, J, I) + sub3(divx, K, J, I, dj=-1))
    ) * col(rr0, K)
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
    **unused,
):
    """Port of ``advw`` (Cartesian, no terrain, periodic). Updates wten at
    k=2..nk. Returns {'wten'}."""
    _check_common(cfg, "advw")
    _check_diff(cfg, "advw")
    nrkmax = _opt(cfg, "nrkmax")
    doweno = advweno == 1 or (advweno == 2 and nrk == nrkmax)
    _check_order("advw", hadvorder, vadvorder, doweno)
    if dowbud:
        raise NotImplementedError("advw: budget diagnostics not ported")
    nk = g.nk
    K, J, I = (2, nk), (1, g.nj), (1, g.ni)
    rdx, rdy, rdz = g.rdx, g.rdy, g.rdz

    fl = hadv_flx5(bk, g, stag=4, c1=c1, c2=c2, rru=rru, rrv=rrv, a=w3d)
    dumx, dumy = fl["dumx"], fl["dumy"]
    advx = -(sub3(dumx, K, J, I, di=1) - sub3(dumx, K, J, I)) * rdx * xrow(uh, I)
    advy = -(sub3(dumy, K, J, I, dj=1) - sub3(dumy, K, J, I)) * rdy * yrow(vh, J)

    dumz = vadv_flx5(bk, g, stag=4, c1=c1, c2=c2, rrw=rrw, a=w3d)["dumz"]
    advz = -(sub3(dumz, K, J, I) - sub3(dumz, K, J, I, dk=-1)) * rdz * col(mf, K)

    tend = (
        (advx + advy)
        + advz
        + sub3(w3d, K, J, I)
        * (col(c2, K) * sub3(divx, K, J, I) + col(c1, K) * sub3(divx, K, J, I, dk=-1))
    ) * col(rrf0, K)
    wten = put3(bk, bk.copy(wten), K, J, I, sub3(wten, K, J, I) + tend)
    return {"wten": wten}


# ===========================================================================
# misclibs.F: getdivx
# ===========================================================================


def getdivx(bk, g, *, uh, vh, mh, rho0, rf0, rru, rrv, rrw, divx, cfg=None, **unused):
    """Port of ``getdivx`` (no terrain, Cartesian): converts the (averaged)
    velocities in rru/rrv/rrw to mass fluxes in the interior and recomputes
    divx in the interior. Returns {'rru', 'rrv', 'rrw', 'divx'}."""
    _check_common(cfg, "getdivx")
    xp = bk.xp
    smeps = _opt(cfg, "smeps")
    nk = g.nk
    K, J, I = (1, nk), (1, g.nj), (1, g.ni)
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
    "DEFAULTS",
]
