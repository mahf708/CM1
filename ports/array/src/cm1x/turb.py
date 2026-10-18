"""LES subgrid turbulence (cm1setup=1, sgsmodel=1: prognostic-TKE Deardorff
scheme), ported from src/turb.F, plus the TKE pieces of solve1/2/3.F.

turb.F
  ``sfc_and_turb`` (the LES path only: ramp of the "constants" for
  ramp_sgs>=1, dissten=0, calcnm, calcdef, tkekm, tkebc, the tke3d boundary
  copy, gettau), ``calcnm``, ``calcdef``, ``tkekm``, ``tkebc``, ``gettau``
  (with ``sfcstress`` for bbc=1);
init3d.F
  ``turb_constants`` (cm0 = c_m and cme/ce1/ce2/csm, cm1setup=1);
solve1.F (inline code, no subroutine of its own in CM1)
  ``tke_source`` (shear/buoyancy production - dissipation), ``dissheat``
  (dissipative heating, idiss=1, with cvm), ``vimpl_coefs`` (dum7/dum8 for
  implicit turbs), ``turb_tendencies`` (the ``doturb`` block: turbu, turbv,
  turbw, vimpl arrays, turbs on theta-th0r and on every moisture species);
solve2.F / solve3.F
  ``tke_rk_update`` (tke3d = tkea + dt*tten, floor at 1e-10, bcw) and
  ``tke_copy`` (tkea = tke3d after the RK loop);
bc.F (non-MPI branches, as the turbulence code calls them)
  ``bcw`` and ``bc2d_level``.

Supported: cm1setup=1, sgsmodel=1, ipbl=0, horizturb=0, tconfig=1/2,
bcturbs=1/2, doimpl=0/1, idiss=0/1, ramp_sgs>=0, bbc=1, tbc=1, lateral BCs
wbc..nbc=1..4, imoist=0 or 1 with liquid-only moisture (iice=0, e.g. Kessler),
no surface fluxes (isfcflx=0, sfcmodel=0 or 1), no terrain, axisymm=0,
do_ib=F.  Diagnostics that never feed back (budgets, out3d, iamsat, stau,
psfc, ugr/vgr, the nstep=0 print) are not ported.

Shapes (cm1x.grid, ``a[k, j, i]`` with halos): w-level arrays ``mf, rmf, nm,
defv, defh, lenscl, dissten, kmh, kmv, khh, khv, cme, csm, ce1, ce2, tkea,
tke3d, tketen`` are ``g.w_shape``; ``rho, rr, rf, prs, pi0, th0, mh, c1, c2,
t11..t33`` are ``g.s_shape``; ``cm0`` is ``s_shape[1:]``; ``uh, ruh`` /
``uf`` are 1-D over i scalar / u points, ``vh, rvh`` / ``vf`` over j.
``qa`` is a sequence of numq scalar arrays (or a ``[numq, k, j, i]`` array).
"""

from __future__ import annotations

from . import constants as C
from .turbtend import (
    Reg,
    _atw,
    _bcflags,
    _opt,
    _terrain,
    turbs,
    turbu,
    turbv,
    turbw,
)

c_m = 0.10
c_l = 0.8165
ri_c = 0.25
cmemin = 1.0e-4
th0r = 300.0
_pi = 3.1415926535897932384626433  # constants.F (real, parameter)


def _check(cfg, who):
    bad = []
    for name, ok in (
        ("cm1setup", (1,)),
        ("sgsmodel", (1,)),
        ("ipbl", (0,)),
        ("horizturb", (0,)),
        ("cuparam", (0,)),
        ("tconfig", (1, 2)),
        ("bcturbs", (1, 2)),
        ("doimpl", (0, 1)),
        ("idiss", (0, 1)),
        ("bbc", (1,)),
        ("tbc", (1,)),
        ("axisymm", (0,)),
        ("isfcflx", (0,)),
        ("iice", (0,)),
    ):
        v = int(_opt(cfg, name, ok[0]))
        if v not in ok:
            bad.append(f"{name}={v}")
    sfcmodel = int(_opt(cfg, "sfcmodel", 0))
    if sfcmodel in (2, 3, 4, 6, 7):
        bad.append(f"sfcmodel={sfcmodel} (surface fluxes)")
    for n in ("wbc", "ebc", "sbc", "nbc"):
        v = int(_opt(cfg, n, 1))
        if v not in (1, 2, 3, 4):
            bad.append(f"{n}={v}")
    for n in ("do_ib", "dosfcflx"):
        if _opt(cfg, n, False):
            bad.append(f"{n}=T")
    if bad:
        raise NotImplementedError(f"{who}: options not ported: " + ", ".join(bad))


def _q(qa, n):
    """Fortran species n (1-based) of qa."""
    return qa[n - 1]


def _moist(cfg):
    imoist = int(_opt(cfg, "imoist", 0))
    ptype = int(_opt(cfg, "ptype", 1 if imoist else 0))
    d = {
        "imoist": imoist,
        "ptype": ptype,
        "iice": int(_opt(cfg, "iice", 0)),
        "nqv": int(_opt(cfg, "nqv", 1)),
        "nqc": int(_opt(cfg, "nqc", 2)),
        "nql1": int(_opt(cfg, "nql1", 2 if imoist else 0)),
        "nql2": int(_opt(cfg, "nql2", 3 if imoist else 0)),
        "nqs1": int(_opt(cfg, "nqs1", -2)),
        "nqs2": int(_opt(cfg, "nqs2", -2)),
        "numq": int(_opt(cfg, "numq", 3 if imoist else 1)),
    }
    return d


def _copy_or_zeros(bk, a, shape):
    return bk.zeros(shape) if a is None else bk.copy(a)


# ---------------------------------------------------------------------------
# bc.F (serial)


def bcw(bk, g, cfg, w, flag):
    """bc.F ``bcw`` (non-MPI) on a w-level array; flag=1 also zeroes k=1 and
    k=nk+1 on i,j=0..n+1.  Returns the new array."""
    ni, nj, nk, n = g.ni, g.nj, g.nk, g.ngxy
    b = _bcflags(cfg)
    K = (2, nk)
    jall, iall = (1 - n, nj + n), (1 - n, ni + n)

    def cp(a, dst, src):
        return bk.set(a, Reg(g, *dst).idx, Reg(g, *src)(a))

    if nk >= 2:
        if b["wbc"] == 1:
            w = cp(w, (K, jall, (1 - n, 0)), (K, jall, (ni + 1 - n, ni)))
        elif b["wbc"] == 2:
            for i in range(1 - n, 1):
                w = cp(w, (K, (0, nj + 1), (i, i)), (K, (0, nj + 1), (1, 1)))
        else:
            for i in range(1, n + 1):
                w = cp(w, (K, (0, nj + 1), (1 - i, 1 - i)), (K, (0, nj + 1), (i, i)))
        if b["ebc"] == 1:
            w = cp(w, (K, jall, (ni + 1, ni + n)), (K, jall, (1, n)))
        elif b["ebc"] == 2:
            for i in range(ni + 1, ni + n + 1):
                w = cp(w, (K, (0, nj + 1), (i, i)), (K, (0, nj + 1), (ni, ni)))
        else:
            for i in range(1, n + 1):
                w = cp(
                    w,
                    (K, (0, nj + 1), (ni + i, ni + i)),
                    (K, (0, nj + 1), (ni + 1 - i,) * 2),
                )
        if b["sbc"] == 1:
            w = cp(w, (K, (1 - n, 0), iall), (K, (nj + 1 - n, nj), iall))
        elif b["sbc"] == 2:
            for j in range(1 - n, 1):
                w = cp(w, (K, (j, j), (0, ni + 1)), (K, (1, 1), (0, ni + 1)))
        else:
            for j in range(1, n + 1):
                w = cp(w, (K, (1 - j, 1 - j), (0, ni + 1)), (K, (j, j), (0, ni + 1)))
        if b["nbc"] == 1:
            w = cp(w, (K, (nj + 1, nj + n), iall), (K, (1, n), iall))
        elif b["nbc"] == 2:
            for j in range(nj + 1, nj + n + 1):
                w = cp(w, (K, (j, j), (0, ni + 1)), (K, (nj, nj), (0, ni + 1)))
        else:
            for j in range(1, n + 1):
                w = cp(
                    w,
                    (K, (nj + j, nj + j), (0, ni + 1)),
                    (K, (nj + 1 - j,) * 2, (0, ni + 1)),
                )
    if flag == 1:
        for k in (1, nk + 1):
            w = bk.set(w, Reg(g, (k, k), (0, nj + 1), (0, ni + 1)).idx, 0.0)
    return w


def bc2d_level(bk, g, cfg, a, k):
    """bc.F ``bc2d`` applied to level k (Fortran) of a 3-D array."""
    ni, nj, n = g.ni, g.nj, g.ngxy
    b = _bcflags(cfg)
    K = (k, k)
    jall, iall = (1 - n, nj + n), (1 - n, ni + n)

    def cp(x, dst, src):
        return bk.set(x, Reg(g, *dst).idx, Reg(g, *src)(x))

    if b["wbc"] == 1:
        a = cp(a, (K, jall, (1 - n, 0)), (K, jall, (ni + 1 - n, ni)))
    elif b["wbc"] == 2:
        for i in range(1 - n, 1):
            a = cp(a, (K, jall, (i, i)), (K, jall, (1, 1)))
    else:
        for i in range(1, n + 1):
            a = cp(a, (K, jall, (1 - i,) * 2), (K, jall, (i, i)))
    if b["ebc"] == 1:
        a = cp(a, (K, jall, (ni + 1, ni + n)), (K, jall, (1, n)))
    elif b["ebc"] == 2:
        for i in range(ni + 1, ni + n + 1):
            a = cp(a, (K, jall, (i, i)), (K, jall, (ni, ni)))
    else:
        for i in range(1, n + 1):
            a = cp(a, (K, jall, (ni + i,) * 2), (K, jall, (ni + 1 - i,) * 2))
    if b["sbc"] == 1:
        a = cp(a, (K, (1 - n, 0), iall), (K, (nj + 1 - n, nj), iall))
    elif b["sbc"] == 2:
        for j in range(1 - n, 1):
            a = cp(a, (K, (j, j), iall), (K, (1, 1), iall))
    else:
        for j in range(1, n + 1):
            a = cp(a, (K, (1 - j,) * 2, iall), (K, (j, j), iall))
    if b["nbc"] == 1:
        a = cp(a, (K, (nj + 1, nj + n), iall), (K, (1, n), iall))
    elif b["nbc"] == 2:
        for j in range(nj + 1, nj + n + 1):
            a = cp(a, (K, (j, j), iall), (K, (nj, nj), iall))
    else:
        for j in range(1, n + 1):
            a = cp(a, (K, (nj + j,) * 2, iall), (K, (nj + 1 - j,) * 2, iall))
    return a


# ---------------------------------------------------------------------------
# init3d / ramp


def turb_constants(
    bk, g, cfg=None, *, cm0=None, turbfac=None, cme=None, csm=None, ce1=None, ce2=None
):
    """Turbulence-model "constants".

    turbfac=None: init3d.F (cm1setup=1): cm0 = c_m everywhere (unless given)
    and cme, ce1, ce2, csm on k=kb..ke at all i, j.
    turbfac given: the ramp_sgs block of sfc_and_turb: cme = max(0.001,
    turbfac*cm0), ... on k=1..nk+1, i=1..ni, j=1..nj (where cm0 > cmemin).
    Returns {"cm0", "cme", "csm", "ce1", "ce2"} (cme.. ``g.w_shape``).
    """
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    n = g.ngxy
    if cm0 is None:
        cm0 = bk.full(g.s_shape[1:], c_m)
    cme = _copy_or_zeros(bk, cme, g.w_shape)
    csm = _copy_or_zeros(bk, csm, g.w_shape)
    ce1 = _copy_or_zeros(bk, ce1, g.w_shape)
    ce2 = _copy_or_zeros(bk, ce2, g.w_shape)
    if turbfac is None:
        r = Reg(g, (1 - g.ngz, nk + g.ngz), (1 - n, nj + n), (1 - n, ni + n))
        c0 = cm0[None, :, :]
        cmev = c0 * xp.ones_like(r(cme))
    else:
        r = Reg(g, (1, nk + 1), (1, nj), (1, ni))
        c0 = cm0[r.idx[1], r.idx[2]][None]
        cmev = xp.maximum(xp.full_like(r(cme), 0.001), turbfac * c0)
    ce1v = cmev * c_l * c_l * (1.0 / ri_c - 1.0)
    ce2v = xp.maximum(xp.zeros_like(cmev), cmev * _pi * _pi - ce1v)
    csmv = (cmev * cmev * cmev / (ce1v + ce2v)) ** 0.25
    ok = c0 > cmemin
    out = {"cm0": cm0}
    for name, arr, val in (
        ("cme", cme, cmev),
        ("ce1", ce1, ce1v),
        ("ce2", ce2, ce2v),
        ("csm", csm, csmv),
    ):
        out[name] = bk.set(arr, r.idx, xp.where(ok, val, r(arr)))
    return out


def turbfac_of(cfg, rtime, bk=None):
    """sfc_and_turb: turbfac = clip((rtime-0)/ramp_time, 0, 1).

    A Python float for a Python-number ``rtime``; for a 0-d array ``rtime``
    (a compiled step) the same arithmetic on arrays, cast to ``bk.dtype``."""
    if isinstance(rtime, (int, float)):
        tf = (float(rtime) - 0.0) / float(_opt(cfg, "ramp_time", 0.0))
        return min(1.0, max(0.0, tf))
    xp = bk.xp
    tf = (rtime - 0.0) / float(_opt(cfg, "ramp_time", 0.0))
    tf = xp.minimum(xp.maximum(tf, xp.zeros_like(tf)), xp.ones_like(tf))
    return xp.astype(tf, bk.dtype)


# ---------------------------------------------------------------------------
# calcnm


def _rslf(xp, p, t, ptype):
    if ptype == 4:
        return 380.00 * xp.exp(17.2693882 - 4097.8531 / (t - 35.86)) / p
    esl = 611.2 * xp.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = xp.minimum(esl, p * 0.5)
    return C.eps * esl / (p - esl)


def calcnm(
    bk, g, cfg=None, *, c1, c2, mf, pi0, th0, nm=None, prs, pp, th, qa=None, **unused
):
    """turb.F ``calcnm``: squared Brunt-Vaisala frequency on w levels
    (k=2..nk; 0 at k=1 and nk+1; i=1..ni, j=1..nj).  Liquid-only moist
    formulation (iice=0).  Returns {"nm"}."""
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    m = _moist(cfg)
    if m["imoist"] == 1 and m["iice"] == 1:
        raise NotImplementedError("calcnm: iice=1 not ported")
    rdz = g.rdz
    gg = C.g
    nm = _copy_or_zeros(bk, nm, g.w_shape)
    R = Reg(g, (2, nk), (1, nj), (1, ni))
    if m["imoist"] == 0:
        val = (
            xp.log((R(th0) + R(th)) / (R(th0, dk=-1) + R(th, dk=-1))) * gg * rdz * R(mf)
        )
    else:
        A = Reg(g, (1, nk), (1, nj), (1, ni))
        nqv, nqc = m["nqv"], m["nqc"]
        qv = A(_q(qa, nqv))
        if m["ptype"] == 0:
            qvci = qv
        else:
            qvci = qv + A(_q(qa, nqc))
        qt = xp.zeros_like(qv)
        for n in range(1, m["numq"] + 1):
            if (
                n == nqv
                or (m["nql1"] <= n <= m["nql2"])
                or (m["nqs1"] <= n <= m["nqs2"] and m["iice"] == 1)
            ):
                qt = qt + A(_q(qa, n))
        t = (A(th0) + A(th)) * (A(pi0) + A(pp))
        thv = (A(th0) + A(th)) * (1.0 + C.reps * qv) / (1.0 + qt)
        p = A(prs)
        cc1, cc2, mff = R(c1), R(c2), R(mf)
        # k = 2..nk vs k-1
        hi, lo = slice(1, nk), slice(0, nk - 1)
        sub = gg * xp.log(thv[hi] / thv[lo]) * rdz * mff
        pavg = cc1 * p[lo] + cc2 * p[hi]
        tavg = cc1 * t[lo] + cc2 * t[hi]
        qtavg = cc1 * qt[lo] + cc2 * qt[hi]
        qvciavg = cc1 * qvci[lo] + cc2 * qvci[hi]
        qvs = _rslf(xp, pavg, tavg, m["ptype"])
        ql = xp.maximum(qvciavg - qvs, xp.zeros_like(qvs))
        qvv = xp.maximum(qvciavg - ql, xp.zeros_like(qvs))
        cpml = C.cp + C.cpv * qvv + C.cpl * ql
        lh = C.lv1 - C.lv2 * tavg
        dtd = tavg - 29.65
        drdt = 17.67 * (273.15 - 29.65) * qvs / (dtd * dtd)
        gamma = (
            gg * (1.0 + qtavg) * (1.0 + lh * qvs / (C.rd * tavg)) / (cpml + lh * drdt)
        )
        sat = gg * (
            (xp.log(t[hi] / t[lo]) * rdz * mff + gamma / tavg)
            * (1.0 + tavg * drdt / (C.eps + qvs))
            - xp.log((1.0 + qt[hi]) / (1.0 + qt[lo])) * rdz * mff
        )
        val = xp.where(ql > 1.0e-10, sat, sub)
    nm = bk.set(nm, R.idx, val)
    for k in (1, nk + 1):
        nm = bk.set(nm, Reg(g, (k, k), (1, nj), (1, ni)).idx, 0.0)
    return {"nm": nm}


# ---------------------------------------------------------------------------
# calcdef


def calcdef(
    bk,
    g,
    cfg=None,
    *,
    uh,
    uf,
    vh,
    vf,
    mh,
    c1,
    c2,
    mf,
    rho,
    rf,
    ua,
    va,
    wa,
    s11=None,
    s12=None,
    s13=None,
    s22=None,
    s23=None,
    s33=None,
    defv=None,
    defh=None,
    terrain=None,
    **unused,
):
    """turb.F ``calcdef`` (Cartesian): rho*S_ij into s11..s33
    (``g.s_shape``) and the deformation terms defv, defh (w levels,
    k=2..nk).  With terrain_flag, ``terrain`` holds the metrics (see
    ``turbtend._terrain``).  Returns dict s11, s12, s13, s22, s23, s33,
    defv, defh."""
    ni, nj, nk = g.ni, g.nj, g.nk
    b = _bcflags(cfg)
    tm = _terrain(cfg, terrain, "calcdef")
    s11 = _copy_or_zeros(bk, s11, g.s_shape)
    s12 = _copy_or_zeros(bk, s12, g.s_shape)
    s13 = _copy_or_zeros(bk, s13, g.s_shape)
    s22 = _copy_or_zeros(bk, s22, g.s_shape)
    s23 = _copy_or_zeros(bk, s23, g.s_shape)
    s33 = _copy_or_zeros(bk, s33, g.s_shape)
    defv = _copy_or_zeros(bk, defv, g.w_shape)
    defh = _copy_or_zeros(bk, defh, g.w_shape)

    def cp(a, dst, src):
        return bk.set(a, Reg(g, *dst).idx, Reg(g, *src)(a))

    if tm is not None:
        s11, s12, s13, s22, s23, s33 = _calcdef_terrain(
            bk, g, tm, uh, uf, vh, vf, rho, rf, ua, va, wa, s11, s12, s13, s22, s23, s33
        )
    else:
        s11, s12, s13, s22, s23, s33 = _calcdef_flat(
            bk,
            g,
            b,
            uh,
            uf,
            vh,
            vf,
            mh,
            mf,
            rho,
            rf,
            ua,
            va,
            wa,
            s11,
            s12,
            s13,
            s22,
            s23,
            s33,
        )
    K = (1, nk)
    Kw = (2, nk)

    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        w2 = b["wbc"] == 2 and b["ibw"] == 1
        e2 = b["ebc"] == 2 and b["ibe"] == 1
        s2 = b["sbc"] == 2 and b["ibs"] == 1
        n2 = b["nbc"] == 2 and b["ibn"] == 1
        if w2:
            s11 = cp(s11, (K, (0, nj + 1), (0, 0)), (K, (0, nj + 1), (1, 1)))
        if e2:
            s11 = cp(
                s11, (K, (0, nj + 1), (ni + 1, ni + 1)), (K, (0, nj + 1), (ni, ni))
            )
        if s2:
            s22 = cp(s22, (K, (0, 0), (0, ni + 1)), (K, (1, 1), (0, ni + 1)))
        if n2:
            s22 = cp(
                s22, (K, (nj + 1, nj + 1), (0, ni + 1)), (K, (nj, nj), (0, ni + 1))
            )
        if w2:
            s12 = cp(s12, (K, (1, nj + 1), (1, 1)), (K, (1, nj + 1), (2, 2)))
        if e2:
            s12 = cp(
                s12, (K, (1, nj + 1), (ni + 1, ni + 1)), (K, (1, nj + 1), (ni, ni))
            )
        if s2:
            s12 = cp(s12, (K, (1, 1), (1, ni + 1)), (K, (2, 2), (1, ni + 1)))
        if n2:
            s12 = cp(
                s12, (K, (nj + 1, nj + 1), (1, ni + 1)), (K, (nj, nj), (1, ni + 1))
            )
        if s2 and w2:
            s12 = cp(s12, (K, (1, 1), (1, 1)), (K, (2, 2), (2, 2)))
        if s2 and e2:
            s12 = cp(s12, (K, (1, 1), (ni + 1, ni + 1)), (K, (2, 2), (ni, ni)))
        if n2 and w2:
            s12 = cp(s12, (K, (nj + 1, nj + 1), (1, 1)), (K, (nj, nj), (2, 2)))
        if n2 and e2:
            s12 = cp(
                s12, (K, (nj + 1, nj + 1), (ni + 1, ni + 1)), (K, (nj, nj), (ni, ni))
            )

    D = Reg(g, Kw, (1, nj), (1, ni))
    cc1, cc2 = D(c1), D(c2)

    def sq(x):
        return x * x

    tmp11 = cc1 * sq(D(s11, dk=-1)) + cc2 * sq(D(s11))
    tmp22 = cc1 * sq(D(s22, dk=-1)) + cc2 * sq(D(s22))
    tmp33 = cc1 * sq(D(s33, dk=-1)) + cc2 * sq(D(s33))
    tmp12 = 0.25 * (
        cc1
        * (
            (sq(D(s12, dk=-1)) + sq(D(s12, dk=-1, dj=1, di=1)))
            + (sq(D(s12, dk=-1, dj=1)) + sq(D(s12, dk=-1, di=1)))
        )
        + cc2
        * (
            (sq(D(s12)) + sq(D(s12, dj=1, di=1)))
            + (sq(D(s12, dj=1)) + sq(D(s12, di=1)))
        )
    )
    tmp13 = 0.5 * (sq(D(s13)) + sq(D(s13, di=1)))
    tmp23 = 0.5 * (sq(D(s23)) + sq(D(s23, dj=1)))
    rrf = 1.0 / sq(D(rf))
    defv = bk.set(defv, D.idx, 4.0 * (tmp13 + tmp23) * rrf)
    defh = bk.set(defh, D.idx, (2.0 * ((tmp11 + tmp22) + tmp33) + 4.0 * tmp12) * rrf)
    return {
        "s11": s11,
        "s12": s12,
        "s13": s13,
        "s22": s22,
        "s23": s23,
        "s33": s33,
        "defv": defv,
        "defh": defh,
    }


def _calcdef_flat(
    bk, g, b, uh, uf, vh, vf, mh, mf, rho, rf, ua, va, wa, s11, s12, s13, s22, s23, s33
):
    """calcdef's Cartesian no-terrain rho*S_ij (with the wall BCs)."""
    ni, nj, nk = g.ni, g.nj, g.nk
    rdx, rdy, rdz = g.rdx, g.rdy, g.rdz

    def cp(a, dst, src):
        return bk.set(a, Reg(g, *dst).idx, Reg(g, *src)(a))

    R = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
    mhk = mh[R.idx[0], g.ngxy : g.ngxy + 1, g.ngxy : g.ngxy + 1]  # mh(1,1,k)
    temz = rdz * mhk
    s11 = bk.set(s11, R.idx, R(rho) * (R(ua, di=1) - R(ua)) * rdx * R.x(uh))
    s22 = bk.set(s22, R.idx, R(rho) * (R(va, dj=1) - R(va)) * rdy * R.y(vh))
    s33 = bk.set(s33, R.idx, R(rho) * (R(wa, dk=1) - R(wa)) * temz)

    def rho4(r):
        return 0.25 * (
            (r(rho, dj=-1, di=-1) + r(rho)) + (r(rho, di=-1) + r(rho, dj=-1))
        )

    s12 = bk.set(
        s12,
        R.idx,
        0.5
        * (
            (R(ua) - R(ua, dj=-1)) * rdy * R.y(vf)
            + (R(va) - R(va, di=-1)) * rdx * R.x(uf)
        )
        * rho4(R),
    )
    K = (1, nk)
    if b["wbc"] == 3 and b["ibw"] == 1:
        s12 = cp(s12, (K, (1, nj + 1), (1, 1)), (K, (1, nj + 1), (2, 2)))
    elif b["wbc"] == 4 and b["ibw"] == 1:
        r = Reg(g, K, (1, nj + 1), (1, 1))
        s12 = bk.set(s12, r.idx, 2.0 * r(va) * rdx * r.x(uf) * rho4(r))
    if b["ebc"] == 3 and b["ibe"] == 1:
        s12 = cp(s12, (K, (1, nj + 1), (ni + 1, ni + 1)), (K, (1, nj + 1), (ni, ni)))
    elif b["ebc"] == 4 and b["ibe"] == 1:
        r = Reg(g, K, (1, nj + 1), (ni + 1, ni + 1))
        s12 = bk.set(s12, r.idx, -2.0 * r(va, di=-1) * rdx * r.x(uf) * rho4(r))
    if b["sbc"] == 3 and b["ibs"] == 1:
        s12 = cp(s12, (K, (1, 1), (1, ni + 1)), (K, (2, 2), (1, ni + 1)))
    elif b["sbc"] == 4 and b["ibs"] == 1:
        r = Reg(g, K, (1, 1), (1, ni + 1))
        s12 = bk.set(s12, r.idx, 2.0 * r(ua) * rdy * r.y(vf) * rho4(r))
    if b["nbc"] == 3 and b["ibn"] == 1:
        s12 = cp(s12, (K, (nj + 1, nj + 1), (1, ni + 1)), (K, (nj, nj), (1, ni + 1)))
    elif b["nbc"] == 4 and b["ibn"] == 1:
        r = Reg(g, K, (nj + 1, nj + 1), (1, ni + 1))
        s12 = bk.set(s12, r.idx, -2.0 * r(ua, dj=-1) * rdy * r.y(vf) * rho4(r))

    Kw = (2, nk)
    V = Reg(g, Kw, (1, nj + 1), (1, ni + 1))
    s13 = bk.set(
        s13,
        V.idx,
        0.5
        * (
            (V(wa) - V(wa, di=-1)) * rdx * V.x(uf)
            + (V(ua) - V(ua, dk=-1)) * rdz * 0.5 * (V(mf, di=-1) + V(mf))
        )
        * 0.5
        * (V(rf, di=-1) + V(rf)),
    )
    s23 = bk.set(
        s23,
        V.idx,
        0.5
        * (
            (V(wa) - V(wa, dj=-1)) * rdy * V.y(vf)
            + (V(va) - V(va, dk=-1)) * rdz * 0.5 * (V(mf, dj=-1) + V(mf))
        )
        * 0.5
        * (V(rf, dj=-1) + V(rf)),
    )
    if b["wbc"] == 3 and b["ibw"] == 1:
        s13 = cp(s13, (Kw, (1, nj), (1, 1)), (Kw, (1, nj), (2, 2)))
    elif b["wbc"] == 4 and b["ibw"] == 1:
        r = Reg(g, Kw, (1, nj), (1, 1))
        s13 = bk.set(s13, r.idx, 2.0 * r(wa) * rdx * r.x(uf))
    if b["ebc"] == 3 and b["ibe"] == 1:
        s13 = cp(s13, (Kw, (1, nj), (ni + 1, ni + 1)), (Kw, (1, nj), (ni, ni)))
    elif b["ebc"] == 4 and b["ibe"] == 1:
        r = Reg(g, Kw, (1, nj), (ni + 1, ni + 1))
        s13 = bk.set(s13, r.idx, -2.0 * r(wa, di=-1) * rdx * r.x(uf))
    if b["sbc"] == 3 and b["ibs"] == 1:
        s23 = cp(s23, (Kw, (1, 1), (1, ni)), (Kw, (2, 2), (1, ni)))
    elif b["sbc"] == 4 and b["ibs"] == 1:
        r = Reg(g, Kw, (1, 1), (1, ni))
        s23 = bk.set(s23, r.idx, 2.0 * r(wa) * rdy * r.y(vf))
    if b["nbc"] == 3 and b["ibn"] == 1:
        s23 = cp(s23, (Kw, (nj + 1, nj + 1), (1, ni)), (Kw, (nj, nj), (1, ni)))
    elif b["nbc"] == 4 and b["ibn"] == 1:
        r = Reg(g, Kw, (nj + 1, nj + 1), (1, ni))
        s23 = bk.set(s23, r.idx, -2.0 * r(wa, dj=-1) * rdy * r.y(vf))

    return s11, s12, s13, s22, s23, s33


def _calcdef_terrain(
    bk, g, tm, uh, uf, vh, vf, rho, rf, ua, va, wa, s11, s12, s13, s22, s23, s33
):
    """calcdef's "Cartesian with terrain" rho*S_ij (no wall BCs there)."""
    ni, nj, nk = g.ni, g.nj, g.nk
    rdx, rdy = g.rdx, g.rdy
    zt = float(tm["zt"])
    gz, rgz, rgzu, rgzv = tm["gz"], tm["rgz"], tm["rgzu"], tm["rgzv"]
    gx, gy = tm["gx"], tm["gy"]
    rds, sig, rdsf, sigf = tm["rds"], tm["sigma"], tm["rdsf"], tm["sigmaf"]
    # dum1 / dum2: u / v at w points
    d1 = _atw(bk, g, tm, ua, (0, nj + 2), (0, ni + 2))
    d2 = _atw(bk, g, tm, va, (0, nj + 2), (0, ni + 2))
    R = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
    rz = R.z(rdsf)
    s11 = bk.set(
        s11,
        R.idx,
        (
            R.m2(gz)
            * (R(ua, di=1) * R.m2(rgzu, di=1) - R(ua) * R.m2(rgzu))
            * rdx
            * R.x(uh)
            + (
                R(gx, dk=1) * (R(d1, dk=1) + R(d1, dk=1, di=1))
                - R(gx) * (R(d1) + R(d1, di=1))
            )
            * 0.5
            * rz
        )
        * R(rho),
    )
    s22 = bk.set(
        s22,
        R.idx,
        (
            R.m2(gz)
            * (R(va, dj=1) * R.m2(rgzv, dj=1) - R(va) * R.m2(rgzv))
            * rdy
            * R.y(vh)
            + (
                R(gy, dk=1) * (R(d2, dk=1) + R(d2, dk=1, dj=1))
                - R(gy) * (R(d2) + R(d2, dj=1))
            )
            * 0.5
            * rz
        )
        * R(rho),
    )
    s33 = bk.set(s33, R.idx, (R(wa, dk=1) - R(wa)) * rz * R.m2(gz) * R(rho))
    P = Reg(g, (1, nk), (1, nj + 1), (1, ni + 1))

    def rg(dj, di):
        return P(rho, dj=dj, di=di) * P.m2(gz, dj=dj, di=di)

    r1 = 0.25 * ((rg(-1, -1) + rg(0, 0)) + (rg(0, -1) + rg(-1, 0)))
    pz = P.z(rdsf)
    ztf1, ztf0 = zt - P.z(sigf, dk=1), zt - P.z(sigf)
    s12 = bk.set(
        s12,
        P.idx,
        0.5
        * (
            (
                r1
                * (P(ua) * P.m2(rgzu) - P(ua, dj=-1) * P.m2(rgzu, dj=-1))
                * rdy
                * P.y(vf)
                + 0.5
                * (
                    ztf1 * (P(d1, dj=-1, dk=1) + P(d1, dk=1))
                    - ztf0 * (P(d1, dj=-1) + P(d1))
                )
                * pz
                * r1
                * (P.m2(rgzu) - P.m2(rgzu, dj=-1))
                * rdy
                * P.y(vf)
            )
            + (
                r1
                * (P(va) * P.m2(rgzv) - P(va, di=-1) * P.m2(rgzv, di=-1))
                * rdx
                * P.x(uf)
                + 0.5
                * (
                    ztf1 * (P(d2, di=-1, dk=1) + P(d2, dk=1))
                    - ztf0 * (P(d2, di=-1) + P(d2))
                )
                * pz
                * r1
                * (P.m2(rgzv) - P.m2(rgzv, di=-1))
                * rdx
                * P.x(uf)
            )
        ),
    )
    # dum1: w at scalar points
    W = Reg(g, (1, nk), (0, nj + 1), (0, ni + 1))
    ws = bk.set(bk.zeros(g.s_shape), W.idx, 0.5 * (W(wa) + W(wa, dk=1)))
    X = Reg(g, (2, nk), (1, nj), (1, ni + 1))
    s13 = bk.set(
        s13,
        X.idx,
        0.5
        * (
            (X(ua) - X(ua, dk=-1)) * X.z(rds)
            + (X(wa) * X.m2(rgz) - X(wa, di=-1) * X.m2(rgz, di=-1)) * rdx * X.x(uf)
            + 0.5
            * X.z(rds)
            * (
                (zt - X.z(sig)) * (X(ws) + X(ws, di=-1))
                - (zt - X.z(sig, dk=-1)) * (X(ws, dk=-1) + X(ws, dk=-1, di=-1))
            )
            * (X.m2(rgz) - X.m2(rgz, di=-1))
            * rdx
            * X.x(uf)
        )
        * 0.5
        * (X.m2(gz, di=-1) * X(rf, di=-1) + X.m2(gz) * X(rf)),
    )
    Y = Reg(g, (2, nk), (1, nj + 1), (1, ni))
    s23 = bk.set(
        s23,
        Y.idx,
        0.5
        * (
            (Y(va) - Y(va, dk=-1)) * Y.z(rds)
            + (Y(wa) * Y.m2(rgz) - Y(wa, dj=-1) * Y.m2(rgz, dj=-1)) * rdy * Y.y(vf)
            + 0.5
            * Y.z(rds)
            * (
                (zt - Y.z(sig)) * (Y(ws) + Y(ws, dj=-1))
                - (zt - Y.z(sig, dk=-1)) * (Y(ws, dk=-1) + Y(ws, dk=-1, dj=-1))
            )
            * (Y.m2(rgz) - Y.m2(rgz, dj=-1))
            * rdy
            * Y.y(vf)
        )
        * 0.5
        * (Y.m2(gz, dj=-1) * Y(rf, dj=-1) + Y.m2(gz) * Y(rf)),
    )
    return s11, s12, s13, s22, s23, s33


# ---------------------------------------------------------------------------
# tkekm, tkebc


def tkekm(
    bk,
    g,
    cfg=None,
    *,
    ruh,
    rvh,
    rmf,
    nm,
    kmh=None,
    kmv=None,
    khh=None,
    khv=None,
    cme,
    ce1,
    ce2,
    tkea,
    lenscl=None,
    dissten=None,
    **unused,
):
    """turb.F ``tkekm``: length scale, dissipation (``dissten +=``) and eddy
    viscosities/diffusivities on k=2..nk, then ``bcw(K, 1)`` on all four.
    Returns dict kmh, kmv, khh, khv, lenscl, dissten."""
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    tconfig = int(_opt(cfg, "tconfig", 1))
    tke_min, nm_min, small = 1.0e-10, 1.0e-6, 0.001
    kmh = _copy_or_zeros(bk, kmh, g.w_shape)
    kmv = _copy_or_zeros(bk, kmv, g.w_shape)
    khh = _copy_or_zeros(bk, khh, g.w_shape)
    khv = _copy_or_zeros(bk, khv, g.w_shape)
    lenscl = _copy_or_zeros(bk, lenscl, g.w_shape)
    dissten = _copy_or_zeros(bk, dissten, g.w_shape)
    R = Reg(g, (2, nk), (1, nj), (1, ni))
    dxr = g.dx * R.x(ruh)
    dyr = g.dy * R.y(rvh)
    if tconfig == 1:
        grdscl = ((dxr * dyr) * (g.dz * R(rmf))) ** 0.33333333
    else:
        lenh = xp.sqrt(dxr * dyr) * xp.ones_like(R(rmf))
        grdscl = g.dz * R(rmf)
    rgrdscl = 1.0 / grdscl
    tke = R(tkea)
    n_ = R(nm)
    low = tke <= tke_min
    tk = xp.where(low, xp.full_like(tke, tke_min), tke)
    pos = n_ > nm_min
    nsafe = xp.where(pos, n_, xp.ones_like(n_))
    ls = c_l * xp.sqrt(tk / nsafe)
    ls = xp.minimum(ls, grdscl)
    ls = xp.maximum(ls, small * grdscl)
    ls = xp.where(pos, ls, grdscl)
    ls = xp.where(low, small * grdscl, ls)
    cc1, cc2, cmev = R(ce1), R(ce2), R(cme)
    dis = R(dissten) + (cc1 + cc2 * ls * rgrdscl) * tk * xp.sqrt(tk) / ls
    prinv = xp.where(
        pos,
        xp.minimum(1.0 + 2.00 * ls * rgrdscl, xp.full_like(ls, 3.00)),
        xp.full_like(ls, 3.00),
    )
    if tconfig == 1:
        km_h = cmev * xp.sqrt(tk) * ls
        km_v = km_h
        kh_h = km_h * prinv
        kh_v = kh_h
    else:
        km_h = cmev * xp.sqrt(tk) * lenh
        km_v = cmev * xp.sqrt(tk) * ls
        kh_h = km_h * prinv
        kh_v = km_v * prinv
    lenscl = bk.set(lenscl, R.idx, ls)
    dissten = bk.set(dissten, R.idx, dis)
    kmh = bcw(bk, g, cfg, bk.set(kmh, R.idx, km_h), 1)
    kmv = bcw(bk, g, cfg, bk.set(kmv, R.idx, km_v), 1)
    khh = bcw(bk, g, cfg, bk.set(khh, R.idx, kh_h), 1)
    khv = bcw(bk, g, cfg, bk.set(khv, R.idx, kh_v), 1)
    return {
        "kmh": kmh,
        "kmv": kmv,
        "khh": khh,
        "khv": khv,
        "lenscl": lenscl,
        "dissten": dissten,
    }


def tkebc(bk, g, cfg=None, *, tkea, kmh, kmv, khh, khv, **unused):
    """turb.F ``tkebc`` (bbc=1 free slip, tbc=1): TKE and K at k=1.
    Returns dict tkea, kmh, kmv, khh, khv."""
    if int(_opt(cfg, "bbc", 1)) != 1 or int(_opt(cfg, "tbc", 1)) != 1:
        raise NotImplementedError("tkebc: only bbc=1, tbc=1 ported")
    ni, nj = g.ni, g.nj
    r = Reg(g, (1, 1), (1, nj), (1, ni))
    tkea = bc2d_level(bk, g, cfg, bk.set(bk.copy(tkea), r.idx, 0.0), 1)
    kmh = bc2d_level(bk, g, cfg, bk.set(bk.copy(kmh), r.idx, 0.0), 1)
    r = Reg(g, (1, 1), (0, nj + 1), (0, ni + 1))
    kmv = bk.set(bk.copy(kmv), r.idx, r(kmh))
    khv = bk.set(
        bk.copy(khv), r.idx, r(kmv) * (r(khv, dk=1) / (1.0e-10 + r(kmv, dk=1)))
    )
    khh = bk.set(
        bk.copy(khh), r.idx, r(kmh) * (r(khh, dk=1) / (1.0e-10 + r(kmh, dk=1)))
    )
    return {"tkea": tkea, "kmh": kmh, "kmv": kmv, "khh": khh, "khv": khv}


# ---------------------------------------------------------------------------
# gettau (+ sfcstress, bbc=1)


def gettau(bk, g, cfg=None, *, kmh, kmv, t11, t12, t13, t22, t23, t33, **unused):
    """turb.F ``gettau`` (Cartesian; bbc=1 via ``sfcstress``; tbc=1):
    t_ij = (rho S_ij) * 2K plus boundary values.  On entry t_ij hold calcdef's
    rho*S_ij.  Returns dict t11, t12, t13, t22, t23, t33."""
    if int(_opt(cfg, "bbc", 1)) != 1 or int(_opt(cfg, "tbc", 1)) != 1:
        raise NotImplementedError("gettau: only bbc=1, tbc=1 ported")
    ni, nj, nk = g.ni, g.nj, g.nk
    b = _bcflags(cfg)
    t11, t12, t13 = bk.copy(t11), bk.copy(t12), bk.copy(t13)
    t22, t23, t33 = bk.copy(t22), bk.copy(t23), bk.copy(t33)

    def cp(a, dst, src):
        return bk.set(a, Reg(g, *dst).idx, Reg(g, *src)(a))

    K = (1, nk)
    R = Reg(g, K, (0, nj + 1), (0, ni + 1))
    tem = R(kmh) + R(kmh, dk=1)
    t11 = bk.set(t11, R.idx, R(t11) * tem)
    t22 = bk.set(t22, R.idx, R(t22) * tem)
    t33 = bk.set(t33, R.idx, R(t33) * tem)
    t12 = bk.set(
        t12,
        R.idx,
        R(t12)
        * 0.25
        * (
            ((R(kmh, dj=-1, di=-1) + R(kmh)) + (R(kmh, di=-1) + R(kmh, dj=-1)))
            + (
                (R(kmh, dk=1, dj=-1, di=-1) + R(kmh, dk=1))
                + (R(kmh, dk=1, di=-1) + R(kmh, dk=1, dj=-1))
            )
        ),
    )
    if b["wbc"] == 3 and b["ibw"] == 1:
        t12 = cp(t12, (K, (1, nj + 1), (1, 1)), (K, (1, nj + 1), (2, 2)))
    if b["ebc"] == 3 and b["ibe"] == 1:
        t12 = cp(t12, (K, (1, nj + 1), (ni + 1, ni + 1)), (K, (1, nj + 1), (ni, ni)))
    if b["sbc"] == 3 and b["ibs"] == 1:
        t12 = cp(t12, (K, (1, 1), (1, ni + 1)), (K, (2, 2), (1, ni + 1)))
    if b["nbc"] == 3 and b["ibn"] == 1:
        t12 = cp(t12, (K, (nj + 1, nj + 1), (1, ni + 1)), (K, (nj, nj), (1, ni + 1)))

    Kw = (2, nk)
    V = Reg(g, Kw, (1, nj + 1), (1, ni + 1))
    t13 = bk.set(t13, V.idx, V(t13) * (V(kmv, di=-1) + V(kmv)))
    t23 = bk.set(t23, V.idx, V(t23) * (V(kmv, dj=-1) + V(kmv)))
    if b["wbc"] == 3 and b["ibw"] == 1:
        t13 = cp(t13, (Kw, (1, nj), (1, 1)), (Kw, (1, nj), (2, 2)))
    if b["ebc"] == 3 and b["ibe"] == 1:
        t13 = cp(t13, (Kw, (1, nj), (ni + 1, ni + 1)), (Kw, (1, nj), (ni, ni)))
    if b["sbc"] == 3 and b["ibs"] == 1:
        t23 = cp(t23, (Kw, (1, 1), (1, ni)), (Kw, (2, 2), (1, ni)))
    if b["nbc"] == 3 and b["ibn"] == 1:
        t23 = cp(t23, (Kw, (nj + 1, nj + 1), (1, ni)), (Kw, (nj, nj), (1, ni)))

    if any(b[n] == 2 for n in ("wbc", "ebc", "sbc", "nbc")):
        w2 = b["wbc"] == 2 and b["ibw"] == 1
        e2 = b["ebc"] == 2 and b["ibe"] == 1
        s2 = b["sbc"] == 2 and b["ibs"] == 1
        n2 = b["nbc"] == 2 and b["ibn"] == 1
        if w2:
            t11 = cp(t11, (K, (0, nj + 1), (0, 0)), (K, (0, nj + 1), (1, 1)))
        if e2:
            t11 = cp(
                t11, (K, (0, nj + 1), (ni + 1, ni + 1)), (K, (0, nj + 1), (ni, ni))
            )
        if s2:
            t22 = cp(t22, (K, (0, 0), (0, ni + 1)), (K, (1, 1), (0, ni + 1)))
        if n2:
            t22 = cp(
                t22, (K, (nj + 1, nj + 1), (0, ni + 1)), (K, (nj, nj), (0, ni + 1))
            )
        if w2:
            t12 = cp(t12, (K, (1, nj + 1), (1, 1)), (K, (1, nj + 1), (2, 2)))
        if e2:
            t12 = cp(
                t12, (K, (1, nj + 1), (ni + 1, ni + 1)), (K, (1, nj + 1), (ni, ni))
            )
        if s2:
            t12 = cp(t12, (K, (1, 1), (1, ni + 1)), (K, (2, 2), (1, ni + 1)))
        if n2:
            t12 = cp(
                t12, (K, (nj + 1, nj + 1), (1, ni + 1)), (K, (nj, nj), (1, ni + 1))
            )
        if s2 and w2:
            t12 = cp(t12, (K, (1, 1), (1, 1)), (K, (2, 2), (2, 2)))
        if s2 and e2:
            t12 = cp(t12, (K, (1, 1), (ni + 1, ni + 1)), (K, (2, 2), (ni, ni)))
        if n2 and w2:
            t12 = cp(t12, (K, (nj + 1, nj + 1), (1, 1)), (K, (nj, nj), (2, 2)))
        if n2 and e2:
            t12 = cp(
                t12, (K, (nj + 1, nj + 1), (ni + 1, ni + 1)), (K, (nj, nj), (ni, ni))
            )

    H = ((1, nj + 1), (1, ni + 1))
    # sfcstress, bbc=1 (free slip); then tbc=1 (free slip)
    t13 = cp(t13, ((1, 1),) + H, ((2, 2),) + H)
    t23 = cp(t23, ((1, 1),) + H, ((2, 2),) + H)
    t13 = cp(t13, ((nk + 1, nk + 1),) + H, ((nk, nk),) + H)
    t23 = cp(t23, ((nk + 1, nk + 1),) + H, ((nk, nk),) + H)
    return {"t11": t11, "t12": t12, "t13": t13, "t22": t22, "t23": t23, "t33": t33}


# ---------------------------------------------------------------------------
# sfc_and_turb (LES path)


def sfc_and_turb(
    bk,
    g,
    cfg=None,
    *,
    rtime,
    ruh,
    rvh,
    uh,
    uf,
    vh,
    vf,
    mh,
    c1,
    c2,
    mf,
    rmf,
    pi0,
    th0,
    rho,
    rf,
    prs,
    ua,
    va,
    wa,
    ppi,
    tha,
    qa=None,
    tkea,
    tke3d,
    cm0,
    cme,
    csm,
    ce1,
    ce2,
    kmh=None,
    kmv=None,
    khh=None,
    khv=None,
    nm=None,
    defv=None,
    defh=None,
    lenscl=None,
    dissten=None,
    t11=None,
    t12=None,
    t13=None,
    t22=None,
    t23=None,
    t33=None,
    terrain=None,
    **unused,
):
    """turb.F ``sfc_and_turb`` for cm1setup=1, sgsmodel=1 (no surface layer,
    no PBL).  Inputs are the state of the *new* time level (CM1 calls it at
    the end of each step: u3d, v3d, w3d, pp3d, th3d, q3d as ua..qa, with prs,
    rho, rf) and ``rtime`` = model time at the end of the step (Python float,
    used by ramp_sgs).

    Returns dict: tkea (k=1 zeroed), tke3d (k=1 and nk+1 copied from tkea),
    kmh, kmv, khh, khv, cme, csm, ce1, ce2, nm, defv, defh, lenscl, dissten,
    t11..t33 (the stresses rho*2K*S_ij used by the next solve1).
    """
    _check(cfg, "sfc_and_turb")
    ni, nj, nk = g.ni, g.nj, g.nk
    out = {}
    if int(_opt(cfg, "ramp_sgs", 0)) >= 1:
        cc = turb_constants(
            bk,
            g,
            cfg,
            cm0=cm0,
            turbfac=turbfac_of(cfg, rtime, bk),
            cme=cme,
            csm=csm,
            ce1=ce1,
            ce2=ce2,
        )
        cme, csm, ce1, ce2 = cc["cme"], cc["csm"], cc["ce1"], cc["ce2"]
    out.update(cme=cme, csm=csm, ce1=ce1, ce2=ce2)
    dissten = _copy_or_zeros(bk, dissten, g.w_shape)
    dissten = bk.set(dissten, Reg(g, (1, nk + 1), (1, nj), (1, ni)).idx, 0.0)
    nm = calcnm(
        bk,
        g,
        cfg,
        c1=c1,
        c2=c2,
        mf=mf,
        pi0=pi0,
        th0=th0,
        nm=nm,
        prs=prs,
        pp=ppi,
        th=tha,
        qa=qa,
    )["nm"]
    d = calcdef(
        bk,
        g,
        cfg,
        uh=uh,
        uf=uf,
        vh=vh,
        vf=vf,
        mh=mh,
        c1=c1,
        c2=c2,
        mf=mf,
        rho=rho,
        rf=rf,
        ua=ua,
        va=va,
        wa=wa,
        s11=t11,
        s12=t12,
        s13=t13,
        s22=t22,
        s23=t23,
        s33=t33,
        defv=defv,
        defh=defh,
        terrain=terrain,
    )
    k = tkekm(
        bk,
        g,
        cfg,
        ruh=ruh,
        rvh=rvh,
        rmf=rmf,
        nm=nm,
        kmh=kmh,
        kmv=kmv,
        khh=khh,
        khv=khv,
        cme=cme,
        ce1=ce1,
        ce2=ce2,
        tkea=tkea,
        lenscl=lenscl,
        dissten=dissten,
    )
    bc = tkebc(
        bk, g, cfg, tkea=tkea, kmh=k["kmh"], kmv=k["kmv"], khh=k["khh"], khv=k["khv"]
    )
    tkea = bc["tkea"]
    tke3d = bk.copy(tke3d)
    for kk in (1, nk + 1):
        r = Reg(g, (kk, kk), (0, nj + 1), (0, ni + 1))
        tke3d = bk.set(tke3d, r.idx, r(tkea))
    tau = gettau(
        bk,
        g,
        cfg,
        kmh=bc["kmh"],
        kmv=bc["kmv"],
        t11=d["s11"],
        t12=d["s12"],
        t13=d["s13"],
        t22=d["s22"],
        t23=d["s23"],
        t33=d["s33"],
    )
    out.update(
        tkea=tkea,
        tke3d=tke3d,
        kmh=bc["kmh"],
        kmv=bc["kmv"],
        khh=bc["khh"],
        khv=bc["khv"],
        nm=nm,
        defv=d["defv"],
        defh=d["defh"],
        lenscl=k["lenscl"],
        dissten=k["dissten"],
        **tau,
    )
    return out


# ---------------------------------------------------------------------------
# solve1.F pieces


def tke_source(
    bk, g, cfg=None, *, nm, defv, defh, dissten, kmh, kmv, khv, tketen=None, **unused
):
    """solve1.F: ``tketen(1:ni,1:nj,2:nk) =`` buoyancy + shear production -
    dissipation (tconfig 1 or 2).  Returns {"tketen"}."""
    xp = bk.xp
    tconfig = int(_opt(cfg, "tconfig", 1))
    tketen = _copy_or_zeros(bk, tketen, g.w_shape)
    R = Reg(g, (2, g.nk), (1, g.nj), (1, g.ni))
    z = xp.zeros_like(R(nm))
    if tconfig == 1:
        v = -R(khv) * R(nm) - R(dissten) + R(kmv) * xp.maximum(z, (R(defv) + R(defh)))
    else:
        v = (
            -R(khv) * R(nm)
            - R(dissten)
            + R(kmv) * xp.maximum(z, R(defv))
            + R(kmh) * xp.maximum(z, R(defh))
        )
    return {"tketen": bk.set(tketen, R.idx, v)}


def dissheat(bk, g, cfg=None, *, pi0, ppi, qa=None, dissten, thten1, **unused):
    """solve1.F: dissipative heating (idiss=1), ``thten1(1:ni,1:nj,1:nk) +=
    epsilon/(cp_v pi)`` with cvm from qa for eqtset=2, imoist=1 (bbc=1, so
    k1=1).  Returns {"thten1", "cvm"} (cvm = solve1's dum1 on the interior)."""
    xp = bk.xp
    idiss = int(_opt(cfg, "idiss", 1))
    eqtset = int(_opt(cfg, "eqtset", 2))
    rterm = int(_opt(cfg, "rterm", 0))
    m = _moist(cfg)
    R = Reg(g, (1, g.nk), (1, g.nj), (1, g.ni))
    if eqtset == 2 and m["imoist"] == 1 and (idiss == 1 or rterm == 1):
        if m["nql1"] >= 1:
            ql = R(_q(qa, m["nql1"]))
            for n in range(m["nql1"] + 1, m["nql2"] + 1):
                ql = ql + R(_q(qa, n))
        else:
            ql = xp.zeros_like(R(pi0))
        if m["iice"] == 1:
            qi = R(_q(qa, m["nqs1"]))
            for n in range(m["nqs1"] + 1, m["nqs2"] + 1):
                qi = qi + R(_q(qa, n))
        else:
            qi = xp.zeros_like(R(pi0))
        cvm = C.cv + C.cvv * R(_q(qa, m["nqv"])) + C.cpl * ql + C.cpi * qi
    else:
        cvm = xp.full_like(R(pi0), C.cv)
    out = {"cvm": cvm, "thten1": thten1}
    if idiss != 1:
        return out
    if int(_opt(cfg, "bbc", 1)) == 3:
        raise NotImplementedError("dissheat: bbc=3 not ported")
    epsd = 0.5 * (R(dissten) + R(dissten, dk=1))
    if m["imoist"] == 1 and eqtset == 2:
        dheat = epsd / (C.cpdcv * cvm * (R(pi0) + R(ppi)))
    else:
        dheat = epsd / (C.cp * (R(pi0) + R(ppi)))
    out["thten1"] = bk.set(bk.copy(thten1), R.idx, R(thten1) + dheat)
    return out


def vimpl_coefs(
    bk, g, cfg=None, *, khv, mf, rf, mh, rr, dum7=None, dum8=None, **unused
):
    """solve1.F: arrays for implicit turbs, on k=1..nk:
    dum7 = khv(k)mf(k)rf(k)mh rr, dum8 = khv(k+1)mf(k+1)rf(k+1)mh rr."""
    R = Reg(g, (1, g.nk), (1, g.nj), (1, g.ni))
    dum7 = _copy_or_zeros(bk, dum7, g.s_shape)
    dum8 = _copy_or_zeros(bk, dum8, g.s_shape)
    dum7 = bk.set(dum7, R.idx, R(khv) * R(mf) * R(rf) * R(mh) * R(rr))
    dum8 = bk.set(dum8, R.idx, R(khv, dk=1) * R(mf, dk=1) * R(rf, dk=1) * R(mh) * R(rr))
    return {"dum7": dum7, "dum8": dum8}


def turb_tendencies(
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
    th0,
    ua,
    va,
    wa,
    tha,
    qa=None,
    uten1,
    vten1,
    wten1,
    thten1,
    qten=None,
    kmh,
    kmv,
    khh,
    khv,
    t11,
    t12,
    t13,
    t22,
    t23,
    t33,
    terrain=None,
    **unused,
):
    """solve1.F ``doturb`` block (sgsmodel=1): turbu(ua)->uten1,
    turbv(va)->vten1, turbw(wa)->wten1, vimpl arrays (doimpl=1),
    turbs(tha + th0(1,1,k) - th0r)->thten1, turbs(qa(n))->qten(n) for
    n=1..numq (imoist=1).  Returns dict uten1, vten1, wten1, thten1, qten
    (qten a list)."""
    _check(cfg, "turb_tendencies")
    xp = bk.xp
    nk = g.nk
    m = _moist(cfg)
    uten1 = turbu(
        bk,
        g,
        cfg,
        dt=dt,
        uf=uf,
        vh=vh,
        mh=mh,
        mf=mf,
        rho=rho,
        rf=rf,
        u=ua,
        uten=uten1,
        w=wa,
        t11=t11,
        t12=t12,
        t13=t13,
        kmv=kmv,
        terrain=terrain,
    )["uten"]
    vten1 = turbv(
        bk,
        g,
        cfg,
        dt=dt,
        uh=uh,
        vf=vf,
        mh=mh,
        mf=mf,
        rho=rho,
        rf=rf,
        v=va,
        vten=vten1,
        w=wa,
        t12=t12,
        t22=t22,
        t23=t23,
        kmv=kmv,
        terrain=terrain,
    )["vten"]
    wten1 = turbw(
        bk,
        g,
        cfg,
        dt=dt,
        uh=uh,
        vh=vh,
        mh=mh,
        mf=mf,
        rho=rho,
        rf=rf,
        w=wa,
        wten=wten1,
        t13=t13,
        t23=t23,
        t33=t33,
        kmh=kmh,
        terrain=terrain,
    )["wten"]
    dum7 = dum8 = None
    if int(_opt(cfg, "doimpl", 1)) == 1:
        v = vimpl_coefs(bk, g, khv=khv, mf=mf, rf=rf, mh=mh, rr=rr)
        dum7, dum8 = v["dum7"], v["dum8"]
    zs = g.ngz
    n = g.ngxy
    if _opt(cfg, "terrain_flag", False):
        R = Reg(g, (1, nk), (0, g.nj + 1), (0, g.ni + 1))
        sadv = bk.set(bk.zeros(g.s_shape), R.idx, (R(th0) - th0r) + R(tha))
    else:
        tem = th0[zs : zs + nk, n : n + 1, n : n + 1] - th0r  # th0(1,1,k) - th0r
        sadv = bk.set(bk.zeros(g.s_shape), slice(zs, zs + nk), tem + tha[zs : zs + nk])
    common = {
        "dt": dt,
        "uh": uh,
        "uf": uf,
        "vh": vh,
        "vf": vf,
        "mh": mh,
        "mf": mf,
        "rho": rho,
        "rr": rr,
        "rf": rf,
        "khh": khh,
        "khv": khv,
        "dum7": dum7,
        "dum8": dum8,
        "terrain": terrain,
    }
    thten1 = turbs(bk, g, cfg, iflux=1, s=sadv, sten=thten1, **common)["sten"]
    qout = None
    if m["imoist"] == 1:
        qout = []
        for nn in range(1, m["numq"] + 1):
            qout.append(
                turbs(
                    bk,
                    g,
                    cfg,
                    iflux=1 if nn == m["nqv"] else 0,
                    s=_q(qa, nn),
                    sten=_q(qten, nn),
                    **common,
                )["sten"]
            )
    del xp
    return {
        "uten1": uten1,
        "vten1": vten1,
        "wten1": wten1,
        "thten1": thten1,
        "qten": qout,
    }


# ---------------------------------------------------------------------------
# solve2.F / solve3.F pieces


def tke_rk_update(bk, g, cfg=None, *, dttmp, tkea, tten, tke3d, **unused):
    """solve2.F (cm1setup/=4): tke3d(1:ni,1:nj,2:nk) = tkea + dttmp*tten,
    zeroed below 1e-10, then bcw(tke3d, 1).  ``tten`` is solve2's wten after
    wten(2:nk) = tketen and advw.  Returns {"tke3d"}."""
    xp = bk.xp
    R = Reg(g, (2, g.nk), (1, g.nj), (1, g.ni))
    v = R(tkea) + dttmp * R(tten)
    v = xp.where(v < 1.0e-10, xp.zeros_like(v), v)
    tke3d = bk.set(bk.copy(tke3d), R.idx, v)
    return {"tke3d": bcw(bk, g, cfg, tke3d, 1)}


def tke_copy(bk, g, cfg=None, *, tke3d, tkea, **unused):
    """solve3.F: tkea(0:ni+1,0:nj+1,1:nk) = tke3d.  Returns {"tkea"}."""
    R = Reg(g, (1, g.nk), (0, g.nj + 1), (0, g.ni + 1))
    return {"tkea": bk.set(bk.copy(tkea), R.idx, R(tke3d))}
