"""The isnd (base-state sounding) and iwnd (base-state wind) blocks of
src/base.F, used by :func:`cm1x.base.base`.

Every column ``[j, i]`` is computed at once with CM1's per-column operation
order; vertical recurrences are Python loops over ``k`` (static trip count).
Loops that iterate to convergence per column (``GO TO`` / ``DO WHILE`` in the
Fortran: isnd 4, 10, 11, 15 and ``revthe``) run as masked iterations through
``bk.fori_loop``: a column stops changing once its own test fails, so each
column takes exactly the iterations CM1 takes. Each sounding then keeps the
values only on the ``(i, j, k)`` points the Fortran sets (zero elsewhere, as
CM1's freshly allocated arrays), before the ghost-cell fill in ``base``.

Supported: isnd 1-5, 7-15, 17-23 (isnd = 6 stops in CM1 and raises here;
isnd = 7 and 17 read ``input_sounding``: ``cfg.input_sounding`` if set, else
``./input_sounding``), iwnd 0-11 (base.F has no iwnd = 7 branch: zero wind).
``testcase`` only selects the analytic profiles (isnd 8, 15, 19; iwnd 6, 8);
the testcase forcing profiles of base.F (wprof, ug/vg, ...) are not here.
"""

from __future__ import annotations

import math
import re

from . import constants as C

__all__ = [
    "consts",
    "param_eqtset",
    "param_iice",
    "read_input_sounding",
    "sounding",
    "wind",
]


# ---------------------------------------------------------------------------
# constants.F and param.F settings base.F reads


def consts(testcase: int) -> dict:
    """The constants of constants.F ``set_constants(testcase)`` base.F uses
    (Python floats; derived ones are formed in the backend dtype by the
    callers, as cm1x.constants does for testcase /= 4, 5)."""
    if testcase == 4:
        g, rd, cp, xlv = 9.81, 287.0, 1015.0, 2.47e6
    elif testcase == 5:
        g, rd, cp, xlv = 9.81, 287.0, 1004.0, 2.50e6
    else:
        g, rd, cp, xlv = C.g, C.rd, C.cp, C.xlv
    cpv, cpl, cpi, xls, to = C.cpv, C.cpl, C.cpi, C.xls, C.to
    lv1, lv2 = xlv + (cpl - cpv) * to, cpl - cpv
    if testcase in (4, 5):
        lv1, lv2 = xlv, 0.0
    return dict(
        g=g, rd=rd, cp=cp, rv=C.rv, xlv=xlv, cpv=cpv, cpl=cpl, cpi=cpi,
        lv1=lv1, lv2=lv2, ls1=xls + (cpi - cpv) * to, ls2=cpi - cpv,
    )  # fmt: skip



def param_eqtset(cfg) -> int:
    """eqtset after param.F: clamped to 1..2; 1 for psolver 4-7 and for
    ptype 4, 7, 50-53, 55."""
    e = min(max(int(cfg.get("eqtset", 2) or 2), 1), 2)
    if int(cfg.get("psolver", 3)) in (4, 5, 6, 7) and e == 2:
        e = 1
    if int(cfg.get("ptype", 0)) in (4, 7, 50, 51, 52, 53, 55) and e == 2:
        e = 1
    return e


def param_iice(cfg) -> int:
    """iice as param.F sets it (an ice scheme with imoist = 1)."""
    if int(cfg.get("imoist", 0)) != 1:
        return 0
    return 0 if int(cfg.get("ptype", 0)) in (0, 1, 6) else 1


def _goddard_t(ptype: int):
    """t0k, t00k, rt0 of goddard.F: set by consat, which param.F calls for
    ptype = 2 only (0 otherwise)."""
    if ptype == 2:
        t0k, t00k = 273.15, 233.15
        return t0k, t00k, 1.0 / (t0k - t00k)
    return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# input_sounding


def read_input_sounding(path, with_wind: bool) -> dict:
    """input_sounding as base.F reads it (list-directed): a header record
    p_sfc (mb), th_sfc (K), qv_sfc (g/kg), then records z, th, qv (g/kg)
    [, u, v]. Returns Python floats/lists; qv of the levels in g/g."""
    with open(path) as f:
        lines = f.read().splitlines()
    pos = [0]

    def read(n):
        vals = []
        while len(vals) < n:
            if pos[0] >= len(lines):
                if not vals:
                    return None
                raise RuntimeError("input_sounding: end of file inside a record")
            toks = re.split(r"[\s,]+", lines[pos[0]].strip())
            pos[0] += 1
            for t in toks:
                if not t:
                    continue
                if t == "/":
                    return vals + [0.0] * (n - len(vals))
                if len(vals) < n:
                    vals.append(float(t.replace("d", "e").replace("D", "e")))
        return vals

    hdr = read(3)
    if hdr is None:
        raise RuntimeError("input_sounding: empty file")
    nv = 5 if with_wind else 3
    z, th, qv, u, v = [], [], [], [], []
    while (rec := read(nv)) is not None:
        z.append(rec[0])
        th.append(rec[1])
        qv.append(rec[2] / 1000.0)
        u.append(rec[3] if with_wind else 0.0)
        v.append(rec[4] if with_wind else 0.0)
    return dict(p_sfc=hdr[0], th_sfc=hdr[1], qv_sfc=hdr[2], z=z, th=th, qv=qv, u=u, v=v)


# ---------------------------------------------------------------------------
# helpers


def _rslf(bk, p, t, ptype, eps):
    from .base import rslf

    return rslf(bk, p, t, ptype, eps=eps)


def _rsif(bk, p, t, ptype, eps):
    """cm1libs.F RSIF (``eps`` = rd/rv of set_constants)."""
    xp = bk.xp
    S = bk.asarray
    if ptype == 4:
        return S(380.00) * xp.exp(S(21.87455) - S(5807.4743) / (t - S(7.66))) / p
    esi = S(611.2) * xp.exp(S(21.8745584) * (t - S(273.15)) / (t - S(7.66)))
    esi = xp.minimum(esi, p * S(0.5))
    return eps * esi / (p - esi)


def _loop(bk, body, carry, max_n, act="act"):
    """Masked per-column iteration: ``body(carry) -> new carry`` is applied
    to the columns whose ``carry[act]`` is true, up to ``max_n`` times;
    ``body`` sets the new ``act``."""
    xp = bk.xp

    def step(_i, c):
        new = body(c)
        a = c[act]
        return {k: xp.where(a, new[k], c[k]) for k in c}

    return bk.fori_loop(max_n, step, carry, until=lambda c: ~xp.any(c[act]))


def _any(bk, m) -> bool:
    return bool(bk.to_numpy(bk.xp.any(m)))


def _ji_mask(bk, sh2, jlo, jhi, ilo, ihi, ng=3):
    """[j, i] mask of Fortran j = jlo..jhi, i = ilo..ihi."""
    xp = bk.xp
    J = xp.arange(sh2[0])[:, None] - (ng - 1)
    I = xp.arange(sh2[1])[None, :] - (ng - 1)  # noqa: E741
    return (J >= jlo) & (J <= jhi) & (I >= ilo) & (I <= ihi)


def _keep(bk, a, kr, m2=None):
    """Zero ``a`` [k, j, i] outside Fortran k in kr (inclusive) and the
    [j, i] mask m2."""
    xp = bk.xp
    kk = xp.arange(a.shape[0])[:, None, None]
    m = (kk >= kr[0]) & (kk <= kr[1])
    if m2 is not None:
        m = m & m2[None]
    return xp.where(m, a, bk.asarray(0.0))


# ---------------------------------------------------------------------------
# base.F: isnd


def sounding(bk, cfg, *, zh, cg: dict, g) -> dict:
    """The isnd block of base.F. ``zh`` is ``[k, j, i]`` (k = 0..nk+1), ``cg``
    the extrapolation coefficients. Returns pi0, prs0, th0, thv0, qv0, qc0,
    qi0, rh0, t0 ([k, j, i]); u0, v0 (staggered; isnd = 7, else None); and the
    module scalars psurf, tsurf, qsurf (0-d arrays or Python floats)."""
    xp = bk.xp
    S = bk.asarray
    nk = zh.shape[0] - 2
    sh2 = zh.shape[1:]
    ni, nj = g.ni, g.nj
    isnd = int(cfg.isnd)
    imoist = int(cfg.imoist)
    ptype = int(cfg.get("ptype", 0) or 0)
    testcase = int(cfg.get("testcase", 0) or 0)

    K = consts(testcase)
    grav, cp, rd, rv, p00 = S(K["g"]), S(K["cp"]), S(K["rd"]), S(K["rv"]), S(C.p00)
    xlv, lv1, lv2 = S(K["xlv"]), S(K["lv1"]), S(K["lv2"])
    reps = rv / rd
    eps = rd / rv
    cpdrd = cp / rd
    rovcp = rd / cp
    rp00 = S(1.0) / p00
    cpl, cpv, cpi = S(K["cpl"]), S(K["cpv"]), S(K["cpi"])
    ls1, ls2 = S(K["ls1"]), S(K["ls2"])
    one, zero, half = S(1.0), S(0.0), S(0.5)
    cgs1, cgs2, cgs3 = (S(cg[k]) for k in ("cgs1", "cgs2", "cgs3"))

    def rslf(p, t):
        return _rslf(bk, p, t, ptype, eps)

    def rsif(p, t):
        return _rsif(bk, p, t, ptype, eps)

    def Z(n=None):
        return bk.zeros(((nk + 2) if n is None else n,) + sh2)

    def col(levels):
        """stack a list of [j, i] arrays for k = 1..nk into [k, j, i] with
        zero k = 0, nk+1"""
        z1 = bk.zeros((1,) + sh2)
        return xp.concat([z1, xp.stack(levels), z1])

    def surf_q(qv0):
        return cgs1 * qv0[1, 3, 3] + cgs2 * qv0[2, 3, 3] + cgs3 * qv0[3, 3, 3]

    zc = zh
    interior = _ji_mask(bk, sh2, 1, nj, 1, ni)
    KB = (0, nk + 1)
    K1 = (1, nk)
    out = dict(u0=None, v0=None)
    psurf, tsurf, qsurf = zero, zero, zero
    qc0, qi0 = Z(), Z()
    t0, thv0, rh0 = Z(), Z(), Z()

    if imoist == 1 and isnd in (2, 3):
        raise RuntimeError(f"isnd={isnd} requires imoist=0 (param.F stops)")

    if isnd == 1:
        th_sfc, pi_sfc = S(300.0), S(1.0)
        th0 = th_sfc + bk.zeros(zc.shape)
        pi0 = pi_sfc - grav * zc / (cp * th_sfc)
        prs0 = p00 * (pi0**cpdrd)
        qv0, rh0 = Z(), Z()
        psurf = p00 * (pi_sfc**cpdrd)
        tsurf = th_sfc * ((psurf * rp00) ** rovcp)

    elif isnd == 2:
        t_sfc, prs_sfc = S(250.0), p00
        hs = rd * t_sfc / grav
        prs0 = prs_sfc * xp.exp(-zc / hs)
        pi0 = (prs0 / p00) ** (rd / cp)
        th0 = t_sfc / pi0
        qv0, rh0 = Z(), Z()
        psurf, tsurf = prs_sfc, t_sfc

    elif isnd == 3:
        th_sfc, prs_sfc, lapse = S(300.0), p00, S(0.0065)
        t0 = th_sfc - lapse * zc
        prs0 = p00 * (t0 / th_sfc) ** (grav / (lapse * rd))
        pi0 = (prs0 / p00) ** (rd / cp)
        th0 = th_sfc * (t0 / th_sfc) ** (one - grav / (lapse * cp))
        qv0, rh0 = Z(), Z()
        psurf = prs_sfc
        tsurf = th_sfc * ((psurf * rp00) ** rovcp)

    elif isnd == 4:
        thec_mb, qt_mb, prs_sfc = S(320.0), S(0.020), S(100000.0)
        cqt = cp + cpl * qt_mb
        p1 = prs_sfc + bk.zeros(sh2)
        qv1 = qt_mb + bk.zeros(sh2)
        tmp = thec_mb + bk.zeros(sh2)
        tlast = tmp
        for _n in range(20):
            thbar = (tmp * (p00 / p1) ** (rd / cp)) * (one + qv1 * reps) / (one + qt_mb)
            pi1 = one - grav * zc[1] / (cp * thbar)
            p1 = p00 * (pi1**cpdrd)
            qv1 = rslf(p1, tmp)
            tmp = thec_mb * ((p1 / (one + qv1 / eps) / p00) ** (rd / cqt)) / xp.exp(
                (lv1 - lv2 * tmp) * qv1 / (tmp * cqt)
            )
            tmp = tlast + S(0.3) * (tmp - tlast)
            tlast = tmp
        pis, prss, ts, ths, qvs = [pi1], [p1], [tmp], [tmp / pi1], [rslf(p1, tmp)]
        t1, th1, qv1 = tmp, ths[0], qvs[0]
        stuck = bk.zeros(sh2) > one
        for k in range(2, nk + 1):
            dzk = zc[k] - zc[k - 1]
            pik = pis[-1] - grav * dzk / (cp * th1)

            def body(c, k=k, dzk=dzk, th1=th1, qv1=qv1):
                n = c["n"] + one
                th2 = c["tlast"]
                t2 = th2 * c["pi"]
                thbar = half * (
                    th1 * (one + qv1 * reps) / (one + qt_mb)
                    + th2 * (one + c["qv2"] * reps) / (one + qt_mb)
                )
                pi = pis[-1] - grav * dzk / (cp * thbar)
                prs = p00 * (pi**cpdrd)
                qv2 = rslf(prs, t2)
                t2 = thec_mb * ((prs / (one + qv2 / eps) / p00) ** (rd / cqt)) / xp.exp(
                    (lv1 - lv2 * t2) * qv2 / (t2 * cqt)
                )
                th2 = t2 / pi
                cont = (xp.abs(th2 - c["tlast"]) > S(0.0001)) & (n < S(100.0))
                return dict(
                    act=cont,
                    stuck=(~cont) & (n >= S(100.0)),
                    n=n,
                    tlast=xp.where(cont, c["tlast"] + S(0.3) * (th2 - c["tlast"]), c["tlast"]),
                    pi=pi,
                    prs=prs,
                    qv2=qv2,
                    t2=t2,
                    th2=th2,
                )

            c0 = dict(
                act=bk.zeros(sh2) < one,
                stuck=stuck,
                n=bk.zeros(sh2),
                tlast=th1,
                pi=pik,
                prs=bk.zeros(sh2),
                qv2=qv1,
                t2=t1,
                th2=th1,
            )
            c = _loop(bk, body, c0, 100)
            stuck = c["stuck"]
            pis.append(c["pi"])
            prss.append(c["prs"])
            ts.append(c["t2"])
            ths.append(c["th2"])
            qvs.append(rslf(c["prs"], c["th2"] * c["pi"]))
            th1, qv1, t1 = c["th2"], c["qv2"], c["t2"]
        if _any(bk, stuck):
            raise RuntimeError("isnd=4: stuck in loop!")
        pi0, prs0, t0, th0, qv0 = col(pis), col(prss), col(ts), col(ths), col(qvs)
        rh0 = _keep(bk, one + Z(), K1)
        qc0 = _keep(bk, qt_mb - qv0, K1)
        psurf = prs_sfc
        tsurf = cgs1 * t0[1, 3, 3] + cgs2 * t0[2, 3, 3] + cgs3 * t0[3, 3, 3]
        qsurf = surf_q(qv0)

    elif isnd == 5:
        z_trop, th_trop, t_trop = S(12000.0), S(343.0), S(213.0)
        th_sfc, prs_sfc, qv_pbl = S(300.0), S(100000.0), S(0.014)
        pi_sfc = (prs_sfc / p00) ** (rd / cp)
        qv_sfc = rslf(prs_sfc, th_sfc * pi_sfc)
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
        rh0 = col(list(rh_in))
        qv0 = Z()
        pi0 = Z()
        prs0 = Z()
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
            qv_in = rh_in * rslf(prs_in, th_in * pi_in)
            qv_in = xp.where(qv_in > qv_pbl, qv_pbl, qv_in)
            qv0 = xp.concat([qv0[:1], qv_in, qv0[-1:]])
        rh_in = qv0[1 : nk + 1] / rslf(
            prs0[1 : nk + 1], th0[1 : nk + 1] * pi0[1 : nk + 1]
        )
        rh0 = xp.concat([rh0[:1], rh_in, rh0[-1:]])
        psurf = prs_sfc
        tsurf = th_sfc * ((psurf * rp00) ** rovcp)
        if imoist == 1:
            qsurf = xp.where(qv0[1, 3, 3] < qv_pbl, surf_q(qv0), qv_pbl)

    elif isnd == 6:
        raise RuntimeError("isnd=6 code has been removed (CM1 stops)")

    elif isnd in (7, 17):
        path = cfg.get("input_sounding", None) or "input_sounding"
        snd = read_input_sounding(path, isnd == 7)
        nsnd = len(snd["z"]) + 1
        if nsnd <= 2:
            raise RuntimeError(
                "isnd=7/17: only one u,v level; at least two are needed (CM1 stops)"
            )
        # sounding columns: python index m = Fortran k - 1 (python floats are
        # float64, as is CM1's double-precision build; kept as 0-d arrays)
        zs = [0.0] + snd["z"]
        ths = [0.0] + snd["th"]
        qvs = [0.0] + snd["qv"]
        us = [0.0] + snd["u"]
        vs = [0.0] + snd["v"]
        zs, ths, qvs, us, vs = ([S(x) for x in a] for a in (zs, ths, qvs, us, vs))
        p_sfc, th_sfc, qv_sfc = S(snd["p_sfc"]), S(snd["th_sfc"]), S(snd["qv_sfc"])
        if imoist == 0:
            qv_sfc = zero
        p_sfc = p_sfc * S(100.0)
        qv_sfc = qv_sfc / S(1000.0)
        pi_sfc = (p_sfc / p00) ** (rd / cp)
        thv_sfc = th_sfc * (one + qv_sfc * reps) / (one + qv_sfc)
        psurf = p_sfc
        tsurf = th_sfc * pi_sfc
        qsurf = qv_sfc
        zs[0], ths[0], qvs[0] = zero, th_sfc, qv_sfc
        if imoist == 0:
            qvs = [zero] * nsnd
        if isnd == 7:
            us[0] = us[1] - (zs[1] - zs[0]) * (us[2] - us[1]) / (zs[2] - zs[1])
            vs[0] = vs[1] - (zs[1] - zs[0]) * (vs[2] - vs[1]) / (zs[2] - zs[1])
        thvs = [ths[m] * (one + reps * qvs[m]) / (one + qvs[m]) for m in range(nsnd)]
        pisn = [pi_sfc]
        for m in range(1, nsnd):
            pisn.append(
                pisn[-1] - grav * (zs[m] - zs[m - 1]) / (cp * half * (thvs[m] + thvs[m - 1]))
            )
        psn = [p00 * (pisn[m] ** (cp / rd)) for m in range(nsnd)]
        tsn = [ths[m] * pisn[m] for m in range(nsnd)]
        rhs = [zero] * nsnd
        if imoist == 1:
            for m in range(nsnd):
                if float(qvs[m]) < 1.0e-12:
                    qvs[m] = S(0.0001) * rslf(psn[m], ths[m] * pisn[m])
                rhs[m] = qvs[m] / rslf(psn[m], tsn[m])
        Zs = xp.stack(zs)
        if float(bk.to_numpy(xp.max(zc[nk]))) > float(zs[-1]):
            raise RuntimeError("isnd=7/17: zmax of sounding < zmax of grid (CM1 stops)")

        # level index per grid point: kup = first m with zs[m] >= zh
        def kup_of(z):
            return xp.sum(
                xp.astype(Zs[:, None, None, None] < z[None], xp.int64), axis=0
            )

        zin = zc[1 : nk + 1]
        kup = kup_of(zin)
        if _any(bk, kup < 1):
            raise RuntimeError("isnd=7/17: grid level at or below the sounding's surface")
        kdn = kup - 1

        def take(lst, idx):
            return xp.take(xp.stack(lst), xp.reshape(idx, (-1,)), axis=0).reshape(idx.shape)

        zdn, zup = take(zs, kdn), take(zs, kup)

        def lin(lst, frac):
            a, b = take(lst, kdn), take(lst, kup)
            return a + (b - a) * frac

        frac = (zin - zdn) / (zup - zdn)
        th_i = lin(ths, frac)
        qv_i = lin(qvs, frac)
        p_i = lin(psn, frac)
        t_i = lin(tsn, frac)
        rh_i = lin(rhs, frac)
        # u at i (zu from i-1, clamped at ib), v at j
        zim = xp.concat([zin[:, :, :1], zin[:, :, :-1]], axis=2)
        zu = half * (zim + zin)
        u_i = lin(us, (zu - zdn) / (zup - zdn))
        zjm = xp.concat([zin[:, :1, :], zin[:, :-1, :]], axis=1)
        zv = half * (zjm + zin)
        v_i = lin(vs, (zv - zdn) / (zup - zdn))
        qv_i = rh_i * rslf(p_i, t_i)
        thv_i = th_i * (one + reps * qv_i) / (one + qv_i)
        pis = [pi_sfc - grav * zin[0] / (cp * half * (thv_sfc + thv_i[0]))]
        for k in range(1, nk):
            pis.append(
                pis[-1] - grav * (zin[k] - zin[k - 1]) / (cp * half * (thv_i[k] + thv_i[k - 1]))
            )
        pi_i = xp.stack(pis)
        p_i = p00 * (pi_i ** (cp / rd))
        if imoist == 1:
            rh_i = qv_i / rslf(p_i, th_i * pi_i)
        pi0, prs0, th0, qv0 = col(list(pi_i)), col(list(p_i)), col(list(th_i)), col(list(qv_i))
        t0, rh0, thv0 = col(list(t_i)), col(list(rh_i)), col(list(thv_i))
        z1u = bk.zeros((1, sh2[0], sh2[1] + 1))
        u0 = xp.concat([u_i, bk.zeros(u_i.shape[:2] + (1,))], axis=2)  # u(ie+1) = 0
        out["u0"] = xp.concat([z1u, u0, z1u])
        z1v = bk.zeros((1, sh2[0] + 1, sh2[1]))
        v0 = xp.concat([v_i, bk.zeros((v_i.shape[0], 1, v_i.shape[2]))], axis=1)
        out["v0"] = xp.concat([z1v, v0, z1v])

    elif isnd == 8:
        th_sfc, pi_sfc = S(300.0), S(1.0)
        lapse, rhval = -0.0030581, 0.0
        if testcase in (6, 10):
            lapse = 0.005
        if testcase == 10:
            rhval = 0.90
        if testcase == 15:
            lapse, rhval = 0.005, 0.80
        lapse, rhval = S(lapse), S(rhval)
        th0 = th_sfc + lapse * zc
        pi0 = pi_sfc - (grav / (cp * lapse)) * xp.log(th0 / th_sfc)
        prs0 = p00 * (pi0 ** (cp / rd))
        t0 = th0 * pi0
        rh0 = rhval + bk.zeros(zc.shape)
        qv0 = rhval * rslf(prs0, t0)
        thsurf = th_sfc  # noqa: F841 (CM1 sets it; recomputed in base)
        psurf = p00 * (pi_sfc ** (cp / rd))
        tsurf = th_sfc * pi_sfc
        qsurf = surf_q(qv0)

    elif isnd in (9, 13):
        klemp = isnd == 13
        ns1 = S(0.0001)
        ns2 = S(0.0004) if klemp else S(0.0000)
        ns3 = S(0.0001)
        zl1 = S(2000.0) if klemp else S(40000.0)
        zl2 = S(3000.0)
        th_s = S(288.0) + bk.zeros(sh2)
        pi_s = S(1.0) + bk.zeros(sh2)
        psurf = p00 * (S(1.0) ** (cp / rd))
        tsurf = S(288.0) * S(1.0)
        above = zc[1] >= zl1

        def jump(pi_s, th_s, ns_a, dz_a):
            # (CM1 tests abs(ns1) < 1e-6 in every jump; ns1 /= 0 here)
            if abs(float(ns1)) < 1.0e-6:
                return pi_s - grav * dz_a / (cp * th_s), th_s
            return (
                pi_s + grav * grav / (cp * ns_a * th_s) * (xp.exp(-ns_a * dz_a / grav) - one),
                th_s * xp.exp(ns_a * dz_a / grav),
            )

        pj, tj = jump(pi_s, th_s, ns1, zl1)
        pi_s = xp.where(above, pj, pi_s)
        th_s = xp.where(above, tj, th_s)
        zsfc = xp.where(above, zl1, S(0.0))
        ns = xp.where(above, ns2, ns1)
        pis, ths, thvs, prss, ts = [], [], [], [], []
        for k in range(1, nk + 1):
            small = xp.abs(ns) < S(1.0e-6)
            dzk = zc[k] - zsfc
            thv = xp.where(small, th_s, th_s * xp.exp(ns * dzk / grav))
            pik = xp.where(
                small,
                pi_s - grav * dzk / (cp * th_s),
                pi_s + grav * grav / (cp * ns * th_s) * (xp.exp(-ns * dzk / grav) - one),
            )
            prs = p00 * (pik**cpdrd)
            if not klemp:
                # rh0 = 0: qv0 = 0 and th0 = thv0*(1+0+0)/(1+0) every iteration
                th = thv
                t = th * pik
                for _n in range(20):
                    t = th * pik
                    qv = zero * rslf(prs, t)
                    th = thv * (one + qv + zero) / (one + reps * qv)
            else:
                th = thv
                t = th * pik
            pis.append(pik)
            ths.append(th)
            thvs.append(thv)
            prss.append(prs)
            ts.append(t)
            c1 = (zc[k + 1] > zl1) & (zsfc < S(1.0))
            pj, tj = jump(pi_s, th_s, ns1, zl1)
            if klemp:
                c2 = (~c1) & (zc[k + 1] > zl2) & (zsfc < zl2)
                pj2, tj2 = jump(pi_s, th_s, ns2, zl2 - zl1)
                pi_s = xp.where(c1, pj, xp.where(c2, pj2, pi_s))
                th_s = xp.where(c1, tj, xp.where(c2, tj2, th_s))
                zsfc = xp.where(c1, zl1, xp.where(c2, zl2, zsfc))
                ns = xp.where(c1, ns2, xp.where(c2, ns3, ns))
            else:
                pi_s = xp.where(c1, pj, pi_s)
                th_s = xp.where(c1, tj, th_s)
                zsfc = xp.where(c1, zl1, zsfc)
                ns = xp.where(c1, ns2, ns)
        pi0, th0, thv0, prs0, t0 = col(pis), col(ths), col(thvs), col(prss), col(ts)
        qv0, rh0 = Z(), Z()
        if not klemp:
            qsurf = surf_q(qv0)

    elif isnd == 10:
        th0, pi0, prs0, qv0, t0, thv0, qc0, qi0, rh0 = _isnd10(
            bk, cfg, zc, sh2, nk, ptype, rslf, rsif,
            dict(grav=grav, cp=cp, rd=rd, rv=rv, p00=p00, reps=reps, eps=eps,
                 lv1=lv1, lv2=lv2, ls1=ls1, ls2=ls2, cpl=cpl, cpv=cpv, cpi=cpi),
        )  # fmt: skip
        psurf = p00 * (S(1.0) ** (cp / rd))
        tsurf = S(288.0) * S(1.0)
        qsurf = surf_q(qv0)

    elif isnd == 11:
        if imoist == 0:
            raise RuntimeError("isnd=11 requires imoist=1 (CM1 stops)")
        th0, pi0, prs0, qv0, t0, thv0, qc0, rh0, psurf, tsurf = _isnd11(
            bk, cfg, zc, sh2, nk, rslf,
            dict(grav=grav, cp=cp, rd=rd, p00=p00, reps=reps, rovcp=rovcp, rp00=rp00,
                 cpdrd=cpdrd, lv1=lv1, lv2=lv2, cpl=cpl),
        )  # fmt: skip
        qsurf = surf_q(qv0)

    elif isnd == 12:
        pi_sfc, th_sfc, zl1, lapse = S(1.0), S(300.0), S(960.0), S(0.010)
        pisfc = pi_sfc - grav * zl1 / (cp * th_sfc)
        zi = zc[1 : nk + 1]
        th_i = xp.where(zi <= zl1, th_sfc + zero * zi, th_sfc + lapse * (zi - zl1))
        pi_i = xp.where(
            zi <= zl1,
            pi_sfc - grav * zi / (cp * th_sfc),
            pisfc - (grav / (cp * lapse)) * xp.log(th_i / th_sfc),
        )
        th0, pi0 = col(list(th_i)), col(list(pi_i))
        prs0 = _keep(bk, p00 * (pi0 ** (cp / rd)), K1)
        t0 = th0 * pi0
        qv0, rh0 = Z(), Z()

    elif isnd in (14, 18, 22):
        pi_sfc, th_sfc = S(1.0), S(300.0)
        lr2, zl2 = S(0.0), S(0.0)
        if isnd == 14:
            lr1, lr2, zl1, zl2 = S(0.08), S(0.003), S(974.0), S(1074.0)
        elif isnd == 18:
            delz = S(60.0)
            lr1, lr2 = S(8.0) / delz, S(0.003)
            zl1 = S(465.0) - delz / S(2.0)
            zl2 = S(465.0) + delz / S(2.0)
        else:
            th_sfc, zl1, lr1 = S(265.0), S(100.0), S(0.01)
        pis, ths = [Z(1)[0]], [Z(1)[0]]  # k = 0 placeholders (pi0(k-1) at k=1 is 0)
        for k in range(1, nk + 1):
            z = zc[k]
            low = z <= zl1
            if isnd == 22:
                th = xp.where(low, th_sfc + zero * z, th_sfc + (z - zl1) * lr1)
            else:
                th = xp.where(
                    low,
                    th_sfc + zero * z,
                    xp.where(
                        z <= zl2,
                        th_sfc + (z - zl1) * lr1,
                        th_sfc + (zl2 - zl1) * lr1 + (z - zl2) * lr2,
                    ),
                )
            pi = xp.where(
                low,
                pi_sfc - grav * z / (cp * th_sfc),
                pis[-1] - grav * (z - zc[k - 1]) / (cp * half * (th + ths[-1])),
            )
            pis.append(pi)
            ths.append(th)
        pi0, th0 = col(pis[1:]), col(ths[1:])
        prs0 = _keep(bk, p00 * (pi0 ** (cp / rd)), K1)
        t0 = th0 * pi0
        thv0 = th0
        rh0 = Z()
        qv0 = _keep(bk, rh0 * rslf(prs0, t0), K1) if isnd == 22 else Z()

    elif isnd == 15:
        if testcase not in (4, 5):
            raise RuntimeError("isnd=15: testcase must be 4 or 5 (CM1 stops)")
        if testcase == 4:
            th_sfc, qv_sfc, prs_sfc, z_inv = S(289.0), S(0.009), S(101780.0), S(840.0)
        else:
            th_sfc, qv_sfc, prs_sfc, z_inv = S(288.3), S(0.00945), S(101780.0), S(795.0)
        pi_sfc = (prs_sfc / p00) ** (rd / cp)
        thv_sfc = th_sfc * (one + qv_sfc * reps) / (one + qv_sfc)
        thvlast = thv_sfc + bk.zeros(sh2)
        pilast = pi_sfc + bk.zeros(sh2)
        zlast = bk.zeros(sh2)
        lv = dict(th=[], thv=[], pi=[], prs=[], t=[], qv=[], qc=[])
        bad = bk.zeros(sh2) > one
        for k in range(1, nk + 1):
            z = zc[k]
            if testcase == 4:
                qt_hi = S(0.0015)
                qt_hi = qt_hi / (one - qt_hi)
                thl_hi = S(297.5) + xp.abs(z - z_inv) ** (one / S(3.0))
            else:
                qt_hi = S(0.001) * (
                    S(5.0) - S(3.0) * (one - xp.exp((z_inv - z) / S(500.0)))
                )
                thl_hi = S(295.0) + xp.abs(z - z_inv) ** (one / S(3.0))
            low = z <= z_inv
            thl = xp.where(low, th_sfc, thl_hi)
            qt = xp.where(low, qv_sfc, qt_hi)

            def body(c, thl=thl, qt=qt, z=z, thvlast=thvlast, pilast=pilast, zlast=zlast):
                thlast = c["th"]
                thv = c["th"] * (one + reps * c["qv"]) / (one + c["qv"] + c["qc"])
                pi = pilast - grav * (z - zlast) / (cp * half * (thvlast + thv))
                prs = p00 * (pi ** (cp / rd))
                t = c["th"] * pi
                qvs = rslf(prs, t)
                sub = qt < qvs
                qv = xp.where(sub, qt, qvs)
                qc = xp.where(sub, zero, qt - qvs)
                th = xp.where(sub, thl, thl / (one - xlv * (qt - qvs) / (cp * t)))
                th = thlast + S(0.3) * (th - thlast)
                n = c["n"] + one
                return dict(
                    act=xp.abs(th - thlast) >= S(0.0001),
                    bad=c["bad"] | (n > S(100.0)),
                    n=n, th=th, thv=thv, pi=pi, prs=prs, t=t, qv=qv, qc=qc,
                )  # fmt: skip

            c0 = dict(
                act=xp.abs(thl - zero) >= S(0.0001),
                bad=bad,
                n=bk.zeros(sh2),
                th=thl,
                thv=bk.zeros(sh2),
                pi=bk.zeros(sh2),
                prs=bk.zeros(sh2),
                t=bk.zeros(sh2),
                qv=qt,
                qc=bk.zeros(sh2),
            )
            c = _loop(bk, body, c0, 102)
            bad = c["bad"]
            c["t"] = c["th"] * c["pi"]
            for key in lv:
                lv[key].append(c[key])
            thvlast, pilast, zlast = c["thv"], c["pi"], z
        if _any(bk, bad):
            raise RuntimeError("isnd=15: no convergence (CM1: stop 1111)")
        th0, thv0, pi0, prs0 = (
            _keep(bk, col(lv[x]), K1, interior) for x in ("th", "thv", "pi", "prs")
        )
        t0, qv0, qc0 = (_keep(bk, col(lv[x]), K1, interior) for x in ("t", "qv", "qc"))
        rh0 = Z()
        psurf, qsurf = prs_sfc, qv_sfc
        tsurf = th_sfc * pi_sfc

    elif isnd in (19, 20, 23):
        clip = isnd == 19
        if isnd == 19:
            z = [0.0, 520.0, 1480.0, 2000.0, 3000.0]
            th = [298.7, 298.7, 302.4, 308.2, 311.85]
            qv = [0.0170, 0.0163, 0.0107, 0.0042, 0.0030]
            prs_sfc = 101500.0
            if testcase == 11:
                z = [0.0, 352.5, 442.5, 3000.0, 20000.0]
                th = [296.6, 296.6, 298.1]
                th += [th[2] + 0.003 * (z[3] - z[2]), th[2] + 0.003 * (z[4] - z[2])]
                qv = [11.8e-3, 11.8e-3, 7.8e-3]
                qv += [qv[2] - 0.004e-3 * (z[3] - z[2]), qv[2] - 0.004e-3 * (z[4] - z[2])]
                prs_sfc = 100000.0
        elif isnd == 20:
            z = [0.0, 740.0, 3260.0, 4000.0]
            th = [297.9, 297.9, 312.6644, 317.0]
            qv = [0.0160, 0.0138, 0.0024, 0.0018]
            prs_sfc = 101540.0
        else:
            z = [0.0, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0, 10000.0]
            th = [299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2]
            qv = [0.0152, 0.01517, 0.01498, 0.0148, 0.0147, 0.0135, 0.003, 0.003]
            th.append(th[7] + (z[8] - z[7]) * (th[7] - th[6]) / (z[7] - z[6]))
            qv.append(qv[7] + (z[8] - z[7]) * (qv[7] - qv[6]) / (z[7] - z[6]))
        # (python-float arithmetic above is IEEE double, as CM1's DP build)
        z, th, qv = ([S(x) for x in a] for a in (z, th, qv))
        prs_sfc = S(prs_sfc)
        pi_sfc = (prs_sfc * rp00) ** rovcp
        thv_sfc = th[0] * (one + reps * qv[0]) / (one + qv[0])
        th_a, qv_a = Z(), Z()  # values where no interval applies stay 0
        done = bk.zeros(zc.shape) > one
        for m in range(1, len(z)):
            sel = (~done) & (zc <= z[m])
            th_a = xp.where(sel, th[m - 1] + (th[m] - th[m - 1]) * (zc - z[m - 1]) / (z[m] - z[m - 1]), th_a)
            qv_a = xp.where(sel, qv[m - 1] + (qv[m] - qv[m - 1]) * (zc - z[m - 1]) / (z[m] - z[m - 1]), qv_a)
            done = done | sel
        if clip:
            qv_a = xp.maximum(zero, qv_a / (one - qv_a))
        else:
            qv_a = qv_a / (one - qv_a)
        thv_a = th_a * (one + reps * qv_a) / (one + qv_a)
        pis = [pi_sfc - grav * zc[1] / (cp * half * (thv_sfc + thv_a[1]))]
        for k in range(2, nk + 1):
            pis.append(
                pis[-1] - grav * (zc[k] - zc[k - 1]) / (cp * half * (thv_a[k] + thv_a[k - 1]))
            )
        pi0 = _keep(bk, col(pis), K1, interior)
        prs0 = _keep(bk, p00 * (pi0 ** (cp / rd)), K1, interior)
        rh0 = _keep(bk, qv_a / rslf(prs0, th_a * pi0), K1, interior)
        th0 = _keep(bk, th_a, KB, interior)
        qv0 = _keep(bk, qv_a, KB, interior)
        thv0 = _keep(bk, thv_a, KB, interior)
        psurf = prs_sfc
        tsurf = th[0] * ((psurf * rp00) ** rovcp)
        if imoist == 1:
            qsurf = xp.where(qv0[1, 3, 3] < qv[0], surf_q(qv0), qv[0])

    elif isnd == 21:
        ztrop, qsfc, qt = S(15000.0), S(0.01865), S(1.0e-11)  # 10.0**(-11)
        zq1, zq2 = S(4000.0), S(7500.0)
        tsfc = S(float(cfg.get("tsk0", 299.28)))
        tvsfc = tsfc * (one + S(0.608) * qsfc)
        psfc = S(101480.0)
        lr = S(0.0067)
        tvtrop = tvsfc - lr * ztrop
        ptrop = psfc * (tvtrop / tvsfc) ** (grav / (rd * lr))
        zi = zc[1 : nk + 1]
        low = zi < ztrop
        r = zi / zq2
        qv_i = xp.where(low, qsfc * xp.exp(-zi / zq1) * xp.exp(-(r * r)), qt)
        tv = xp.where(low, tvsfc - lr * zi, tvtrop)
        p_i = xp.where(
            low,
            psfc * (((tvsfc - lr * zi) / tvsfc) ** (grav / (rd * lr))),
            ptrop * xp.exp(-(grav * (zi - ztrop) / (rd * tvtrop))),
        )
        t_i = tv / (one + S(0.608) * qv_i)
        pi_i = (p_i * rp00) ** rovcp
        th_i = t_i / pi_i
        qv_i = qv_i / (one - qv_i)
        thv_i = th_i * (one + reps * qv_i) / (one + qv_i)
        qv0, t0, prs0 = col(list(qv_i)), col(list(t_i)), col(list(p_i))
        pi0, th0, thv0 = col(list(pi_i)), col(list(th_i)), col(list(thv_i))
        rh0 = Z()

    else:
        raise NotImplementedError(f"isnd={isnd} is not a CM1 option")

    out.update(
        pi0=pi0, prs0=prs0, th0=th0, thv0=thv0, qv0=qv0, qc0=qc0, qi0=qi0,
        rh0=rh0, t0=t0, psurf=psurf, tsurf=tsurf, qsurf=qsurf,
    )  # fmt: skip
    return out


def _isnd10(bk, cfg, zc, sh2, nk, ptype, rslf, rsif, k_):
    """isnd = 10: moist, constant Brunt-Vaisala frequency."""
    xp = bk.xp
    S = bk.asarray
    grav, cp, rd, rv, p00 = k_["grav"], k_["cp"], k_["rd"], k_["rv"], k_["p00"]
    reps, eps, lv1, lv2 = k_["reps"], k_["eps"], k_["lv1"], k_["lv2"]
    ls1, ls2, cpl, cpv, cpi = k_["ls1"], k_["ls2"], k_["cpl"], k_["cpv"], k_["cpi"]
    one, zero, half = S(1.0), S(0.0), S(0.5)
    eqtset = param_eqtset(cfg)
    iice = param_iice(cfg)
    t0k, t00k, rt0 = (S(x) for x in _goddard_t(ptype))
    qcval = S(0.005)
    ns1, ns2, zl1 = S(0.0), S(4.0e-4), S(111700.0)
    c17 = S(17.67) * (S(273.15) - S(29.65))
    c21 = S(21.8745584) * (S(273.15) - S(7.66))
    z0 = bk.zeros(sh2)
    rh = one  # rh0 = 1 everywhere

    def sq(x):
        return x * x

    th_sfc, pi_sfc = S(288.0), S(1.0)
    p_sfc = p00 * (pi_sfc ** (cp / rd))
    t_sfc = th_sfc * pi_sfc
    qv_sfc = rh * rslf(p_sfc, t_sfc)
    ql_sfc = qcval
    thv_sfc = th_sfc * (one + reps * qv_sfc) / (one + qv_sfc + ql_sfc)
    z1 = zc[1]

    # ---- k = 1
    def body1(c):
        th = c["thn"]
        thv = th * (one + reps * c["qv"]) / (one + c["qv"] + c["qc"])
        pi = pi_sfc - grav * z1 / (cp * half * (thv + thv_sfc))
        prs = p00 * (pi ** (cp / rd))
        t = th * pi
        qv = rh * rslf(prs, t)
        qc = qcval + z0
        qtm = qv_sfc + ql_sfc
        qtp = qv + qc
        tavg = half * (t + t_sfc)
        lhv = lv1 - lv2 * tavg
        qvavg = half * (qv_sfc + qv)
        qlavg = half * (ql_sfc + qc)
        qtavg = qvavg + qlavg
        desdt = c17 / sq(tavg - S(29.65))
        drdt = c17 * qvavg / sq(tavg - S(29.65))
        if eqtset == 2:
            gamma = grav * (one + qtavg) * (one + lhv * qvavg / rd / tavg) / (
                cp + cpv * qvavg + cpl * qlavg + lhv * drdt
            )
        else:
            gamma = grav * (one + qtavg) * (rd / (rd + rv * qvavg) + lhv * qvavg / rd / tavg) / (
                cp + lhv * (one + qvavg * reps) * qvavg * desdt
            )
        tp = t_sfc * xp.exp(
            z1 * (((S(0.0) / grav) + xp.log((one + qtp) / (one + qtm)) / z1)
                  / (one + tavg * drdt / (eps + qvavg)) - gamma / tavg)
        )  # fmt: skip
        thn = tp / pi
        n = c["n"] + one
        cont = (xp.abs(thn - c["thlast"]) > S(0.0001)) & (n < S(200.0))
        thn2 = xp.where(cont, c["thlast"] + S(0.3) * (thn - c["thlast"]), thn)
        return dict(
            act=cont, stuck=c["stuck"] | ((~cont) & (n >= S(200.0))), n=n,
            thn=thn2, thlast=xp.where(cont, thn2, c["thlast"]),
            th=th, thv=thv, pi=pi, prs=prs, t=t, qv=qv, qc=qc, desdt=desdt,
        )  # fmt: skip

    c = dict(
        act=z0 < one, stuck=z0 > one, n=z0 + one, thn=th_sfc + z0, thlast=th_sfc + z0,
        th=z0, thv=z0, pi=z0, prs=z0, t=z0, qv=qv_sfc + z0, qc=ql_sfc + z0, desdt=z0,
    )  # fmt: skip
    c = _loop(bk, body1, c, 200)
    if _any(bk, c["stuck"]):
        raise RuntimeError("isnd=10: stuck in loop (111)!")
    desdt = c["desdt"]
    th = c["thn"]
    pi = c["pi"]
    prs = p00 * (pi ** (cp / rd))
    t = th * pi
    lev = dict(
        th=[th], thv=[c["thv"]], pi=[pi], prs=[prs], t=[t],
        qv=[rh * rslf(prs, t)], qc=[qcval + z0], qi=[z0],
    )  # fmt: skip
    ns = ns1 + z0
    stuck = z0 > one
    for k in range(2, nk + 1):
        zk, zkm = zc[k], zc[k - 1]

        def state_at(src):
            return dict(
                pim=src["pi"], thm=src["th"], thvm=src["thv"],
                qtm=src["qv"] + src["qc"] + src["qi"], qvm=src["qv"],
                qlm=src["qc"], qim=src["qi"], tm=src["t"],
            )  # fmt: skip

        prev = {x: lev[x][-1] for x in lev}
        prs_km1 = prev["prs"]

        def solve(m, delz, ns, act0, stuck):
            """the 572 loop from state m, over delz, for columns act0"""

            def body(c):
                th = c["thn"]
                thv = th * (one + reps * c["qv"]) / (one + c["qv"] + c["qc"] + c["qi"])
                pi = m["pim"] - grav * delz / (cp * half * (thv + m["thvm"]))
                prs = p00 * (pi ** (cp / rd))
                t = th * pi
                prsavg = half * (prs_km1 + prs)
                tavg = half * (t + m["tm"])
                if iice == 0:
                    qv = rh * rslf(prs, t)
                    qc = qcval + z0
                    qi = c["qi"]
                    qtp = qv + qc
                    qvavg = rh * rslf(prsavg, tavg)
                    qlavg = half * (m["qlm"] + qc)
                    qtavg = half * (m["qtm"] + qtp)
                    drdt = c17 * qvavg / sq(tavg - S(29.65))
                    lhv = lv1 - lv2 * tavg
                    cpml = cp + cpv * qvavg + cpl * qlavg
                else:
                    qvl = rh * rslf(prs, t)
                    qvi = rh * rsif(prs, t)
                    fliq = xp.maximum(xp.minimum((t - t00k) * rt0, one), zero)
                    fice = one - fliq
                    qv = fliq * qvl + fice * qvi
                    qc = fliq * qcval
                    qi = fice * qcval
                    qtp = qv + qc + qi
                    qvl = rh * rslf(prsavg, tavg)
                    qvi = rh * rsif(prsavg, tavg)
                    fliq = xp.maximum(xp.minimum((tavg - t00k) * rt0, one), zero)
                    fice = one - fliq
                    qvavg = fliq * qvl + fice * qvi
                    qlavg = half * (m["qlm"] + qc)
                    qtavg = half * (m["qtm"] + qtp)
                    drdt = fliq * c17 * qvl / sq(tavg - S(29.65)) + fice * c21 * qvi / sq(
                        tavg - S(7.66)
                    )
                    drdt = xp.where((tavg > t00k) & (tavg < t0k), drdt + (qvl - qvi) * rt0, drdt)
                    lhv = fliq * (lv1 - lv2 * tavg) + fice * (ls1 - ls2 * tavg)
                    cpml = cp + cpv * qvavg + cpl * qlavg + cpi * (qtavg - qlavg - qvavg)
                if eqtset == 2:
                    gamma = grav * (one + qtavg) * (one + lhv * qvavg / rd / tavg) / (cpml + lhv * drdt)
                else:
                    gamma = grav * (one + qtavg) * (
                        rd / (rd + rv * qvavg) + lhv * qvavg / rd / tavg
                    ) / (cp + lhv * (one + qvavg * reps) * qvavg * desdt)
                tp = m["tm"] * xp.exp(
                    delz * (((ns / grav) + xp.log((one + qtp) / (one + m["qtm"])) / delz)
                            / (one + tavg * drdt / (eps + qvavg)) - gamma / tavg)
                )  # fmt: skip
                thn = tp / pi
                n = c["n"] + one
                cont = (xp.abs(thn - c["thlast"]) > S(0.0001)) & (n < S(200.0))
                thn2 = xp.where(cont, c["thlast"] + S(0.3) * (thn - c["thlast"]), thn)
                return dict(
                    act=cont, stuck=c["stuck"] | ((~cont) & (n >= S(200.0))), n=n,
                    thn=thn2, thlast=xp.where(cont, thn2, c["thlast"]),
                    thv=thv, pi=pi, prs=prs, qv=qv, qc=qc, qi=qi,
                )  # fmt: skip

            c = dict(
                act=act0, stuck=stuck, n=z0, thn=m["thm"], thlast=m["thm"],
                thv=m["thvm"], pi=z0, prs=z0, qv=m["qvm"], qc=m["qlm"], qi=m["qim"],
            )  # fmt: skip
            c = _loop(bk, body, c, 200)
            th = c["thn"]
            pi = c["pi"]
            prs = p00 * (pi ** (cp / rd))
            t = th * pi
            qc, qi = c["qc"], c["qi"]
            if iice == 0:
                qv = rh * rslf(prs, t)
            else:
                qvl = rh * rslf(prs, t)
                qvi = rh * rsif(prs, t)
                fliq = xp.maximum(xp.minimum((t - t00k) * rt0, one), zero)
                fice = one - fliq
                qv = fliq * qvl + fice * qvi
                qc = fliq * qcval
                qi = fice * qcval
            res = dict(th=th, thv=c["thv"], pi=pi, prs=prs, t=t, qv=qv, qc=qc, qi=qi)
            return res, c["stuck"]

        cross = (zk > zl1) & (ns == ns1) & (zkm < zl1)
        delz1 = xp.where(cross, zl1 - zkm, zk - zkm)
        r1, stuck = solve(state_at(prev), delz1, ns, z0 < one, stuck)
        if _any(bk, cross):
            ns = xp.where(cross, ns2, ns)
            r2, stuck = solve(state_at(r1), zk - zl1, ns, cross, stuck)
            r1 = {x: xp.where(cross, r2[x], r1[x]) for x in r1}
        for x in lev:
            lev[x].append(r1[x])
    if _any(bk, stuck):
        raise RuntimeError("isnd=10: stuck in loop (333)!")

    def col(levels, edge=z0):
        e = edge[None]
        return xp.concat([e, xp.stack(levels), e])

    q = S(0.005) + z0
    th0, pi0, prs0 = col(lev["th"]), col(lev["pi"]), col(lev["prs"])
    qv0, t0, thv0 = col(lev["qv"]), col(lev["t"]), col(lev["thv"])
    qc0 = col(lev["qc"], q)  # k = 0, nk+1 keep the initial qcval
    qi0 = col(lev["qi"])
    rh0 = one + bk.zeros(th0.shape)
    return th0, pi0, prs0, qv0, t0, thv0, qc0, qi0, rh0


def _isnd11(bk, cfg, zc, sh2, nk, rslf, k_):
    """isnd = 11: constant theta-e, saturated (pseudoadiabatic, ttype = 2)."""
    xp = bk.xp
    S = bk.asarray
    grav, cp, rd, p00, reps = k_["grav"], k_["cp"], k_["rd"], k_["p00"], k_["reps"]
    rovcp, rp00, cpdrd = k_["rovcp"], k_["rp00"], k_["cpdrd"]
    one, zero, half = S(1.0), S(0.0), S(0.5)
    z0 = bk.zeros(sh2)
    zl1 = S(15000.0)
    psurf = S(101510.0)
    tsk0 = S(float(cfg.get("tsk0", 299.28)))

    def thx_of(p, t, qv):  # ttype = 2
        return t * ((p00 / p) ** (S(0.2854) * (one - S(0.28) * qv))) * xp.exp(
            ((S(3376.0) / t) - S(2.54)) * qv * (one + S(0.81) * qv)
        )

    def revthe(the, p, t, act0):
        """returns (t, qv, stuck); t is the first guess"""

        def body(c):
            t = c["tlast"] + c["tinc"]
            qv = rslf(p, t)
            diff = the - thx_of(p, t, qv)
            c1 = (xp.abs(diff) > S(0.0001)) & (c["n"] < S(50.0))
            tinc = xp.where(c1, S(0.30) * diff, c["tinc"])
            return dict(
                act=c1 & (xp.abs(tinc) >= S(0.0001)),
                stuck=c["stuck"] | ((~c1) & (c["n"] >= S(50.0))),
                n=xp.where(c1, c["n"] + one, c["n"]),
                tinc=tinc, tlast=xp.where(c1, t, c["tlast"]), t=t, qv=qv,
            )  # fmt: skip

        c = dict(act=act0, stuck=z0 > one, n=z0 + one, tinc=z0, tlast=t, t=t, qv=z0)
        c = _loop(bk, body, c, 60)
        return c["t"], c["qv"], c["stuck"] & act0

    pi_sfc = (psurf * rp00) ** rovcp
    th_sfc = (tsk0 - S(3.0)) / pi_sfc
    tsurf = th_sfc * pi_sfc
    t_sfc = th_sfc * pi_sfc
    p_sfc = p00 * (pi_sfc ** (cp / rd))
    qv_sfc = rslf(p_sfc, t_sfc)
    thv_sfc = th_sfc * (one + reps * qv_sfc) / (one + qv_sfc)
    the_sfc = thx_of(p_sfc, t_sfc, qv_sfc)  # getthe
    t_sfc, _qv, st = revthe(the_sfc + z0, p_sfc + z0, t_sfc + z0, z0 < one)
    if _any(bk, st):
        raise RuntimeError("isnd=11: revthe: n exceeded 50 (CM1 stops)")

    def tropo(z2, s, act0):
        """DO WHILE (abs(t2-tlast) > 0.0001) from state s (t2 = t1, tlast = 0)"""

        def body(c):
            tlast = c["t2"]
            pi2 = s["pi1"] - grav * (z2 - s["z1"]) / (cp * half * (s["thv1"] + c["thv2"]))
            p2 = p00 * (pi2 ** (cp / rd))
            t2, qv2, st = revthe(the_sfc + z0, p2, c["t2"], c["act"])
            th2 = t2 / pi2
            ql2 = z0
            thv2 = th2 * (one + reps * qv2) / (one + qv2 + ql2)
            return dict(
                act=xp.abs(t2 - tlast) > S(0.0001), stuck=c["stuck"] | st,
                t2=t2, tlast=tlast, pi2=pi2, p2=p2, qv2=qv2, th2=th2, thv2=thv2,
            )  # fmt: skip

        c = dict(
            act=act0 & (xp.abs(s["t1"] - zero) > S(0.0001)), stuck=z0 > one,
            t2=s["t1"], tlast=z0, pi2=s["pi2"], p2=s["p2"], qv2=s["qv2"],
            th2=s["th2"], thv2=s["thv2"],
        )  # fmt: skip
        c = _loop(bk, body, c, 1000)
        return c

    s = dict(
        pi1=pi_sfc + z0, z1=z0, t1=t_sfc, thv1=thv_sfc + z0, thv2=thv_sfc + z0,
        pi2=z0, p2=z0, qv2=z0, th2=z0,
    )  # fmt: skip
    flag = z0 > one
    th_s, pi_s, zsfc = z0, z0, z0
    ns = S(4.0e-4)
    lev = dict(pi=[], prs=[], t=[], th=[], thv=[], qv=[])
    stuck = z0 > one
    for k in range(1, nk + 1):
        zk = zc[k]
        strat = zk > zl1
        # troposphere (and, the first time above zl1, the tropopause values)
        first = strat & (~flag)
        ztr = xp.where(strat, zl1, zk)
        c = tropo(ztr, s, (~strat) | first)
        stuck = stuck | c["stuck"]
        th_s = xp.where(first, c["thv2"], th_s)
        pi_s = xp.where(first, c["pi2"], pi_s)
        zsfc = xp.where(first, zl1, zsfc)
        flag = flag | first
        # stratosphere (ns /= 0)
        thv2s = th_s * xp.exp(ns * (zk - zsfc) / grav)
        pi2s = pi_s + grav * grav / (cp * ns * th_s) * (xp.exp(-ns * (zk - zsfc) / grav) - one)
        p2s = p00 * (pi2s**cpdrd)
        th2s = thv2s
        for _n in range(20):
            t2s = th2s * pi2s
            qv2s = rslf(p2s, t2s)
            th2s = thv2s * (one + qv2s + zero) / (one + reps * qv2s)
        pi2 = xp.where(strat, pi2s, c["pi2"])
        p2 = xp.where(strat, p2s, c["p2"])
        t2 = xp.where(strat, t2s, c["t2"])
        th2 = xp.where(strat, th2s, c["th2"])
        thv2 = xp.where(strat, thv2s, c["thv2"])
        qv2 = xp.where(strat, qv2s, c["qv2"])
        s = dict(pi1=pi2, z1=zk, t1=t2, thv1=thv2, thv2=thv2, pi2=pi2, p2=p2, qv2=qv2, th2=th2)
        for x, v in (("pi", pi2), ("prs", p2), ("t", t2), ("th", th2), ("thv", thv2), ("qv", qv2)):
            lev[x].append(v)
    if _any(bk, stuck):
        raise RuntimeError("isnd=11: revthe: n exceeded 50 (CM1 stops)")

    def col(levels):
        e = z0[None]
        return xp.concat([e, xp.stack(levels), e])

    th0, pi0, prs0 = col(lev["th"]), col(lev["pi"]), col(lev["prs"])
    qv0, t0, thv0 = col(lev["qv"]), col(lev["t"]), col(lev["thv"])
    qc0 = bk.zeros(th0.shape)
    kk = xp.arange(nk + 2)[:, None, None]
    rh0 = xp.where((kk >= 1) & (kk <= nk), one, zero) + bk.zeros(th0.shape)
    return th0, pi0, prs0, qv0, t0, thv0, qc0, rh0, psurf, tsurf


# ---------------------------------------------------------------------------
# base.F: iwnd


def wind(bk, cfg, *, zh, g):
    """The iwnd block of base.F (before umove/vmove): u0, v0 (staggered,
    [k, j, i]) with values only on the points CM1 sets (zero elsewhere).
    Not called for isnd = 7 (winds from input_sounding)."""
    xp = bk.xp
    S = bk.asarray
    nk = zh.shape[0] - 2
    ni, nj = g.ni, g.nj
    iwnd = int(cfg.get("iwnd", 0) or 0)
    testcase = int(cfg.get("testcase", 0) or 0)
    if iwnd < 0 or iwnd > 11:
        raise NotImplementedError(f"iwnd={iwnd} is not a CM1 option")
    pi = S(C.pi)
    half = S(0.5)
    zc = zh
    # zu at u point i: 0.5*(zh(i-1)+zh(i)) (i = ib+1..ie; ib, ie+1 padded)
    zfull = xp.concat([zc[:, :, :1], zc, zc[:, :, -1:]], axis=2)
    zu = half * (zfull[:, :, :-1] + zfull[:, :, 1:])
    zfull = xp.concat([zc[:, :1, :], zc, zc[:, -1:, :]], axis=1)
    zv = half * (zfull[:, :-1, :] + zfull[:, 1:, :])
    shu, shv = zu.shape, zv.shape
    zero = S(0.0)

    def mask(shape2, jlo, jhi, ilo, ihi, klo=1, khi=None):
        khi = nk if khi is None else khi
        m2 = _ji_mask(bk, shape2, jlo, jhi, ilo, ihi)
        kk = xp.arange(nk + 2)[:, None, None]
        return ((kk >= klo) & (kk <= khi)) & m2[None]

    def setu(val, m):
        return xp.where(m, val, zero)

    # default ranges (iwnd = 2, 3, 4, 5): u i=1..ni+1 j=1..nj; v i=1..ni j=1..nj+1
    mu = mask(shu[1:], 1, nj, 1, ni + 1)
    mv = mask(shv[1:], 1, nj + 1, 1, ni)
    # wide ranges (iwnd = 1, 9, 10, 11): u i=0..ni+2 j=0..nj+1; v i=0..ni+1 j=0..nj+2
    mu_w = mask(shu[1:], 0, nj + 1, 0, ni + 2)
    mv_w = mask(shv[1:], 0, nj + 2, 0, ni + 1)

    def shear(z, d1, d2, c1, c2, le):
        d1, d2, c1, c2 = S(d1), S(d2), S(c1), S(c2)
        lo = (z <= d1) if le else (z < d1)
        return xp.where(
            lo, c1, xp.where((z > d1) & (z < d2), (c2 - c1) * (z - d1) / (d2 - d1) + c1, c2)
        )

    u0, v0 = bk.zeros(shu), bk.zeros(shv)
    if iwnd in (0, 7):
        pass
    elif iwnd == 1:
        u0 = setu(shear(zu, 0.0, 2500.0, 0.0, 10.0, False), mu_w)
    elif iwnd == 2:
        udep1, udep2, umax1, umax2 = S(2000.0), S(6000.0), S(7.0), S(31.0)
        pi180 = pi / S(180.0)
        angle = S(90.0) * (zu / udep1) * pi180
        u = xp.where(
            zu <= udep1,
            umax1 - umax1 * xp.cos(angle),
            xp.where(zu <= udep2, umax1 + (zu - udep1) * (umax2 - umax1) / (udep2 - udep1), umax2),
        )
        anglev = S(90.0) * (zv / udep1) * pi180
        v = xp.where(zv <= udep1, umax1 * xp.sin(anglev), umax1)
        u0, v0 = setu(u, mu), setu(v, mv)
    elif iwnd == 3:
        udep1, udep2 = S(0.0), S(7500.0)
        umax1 = S(-40.0) / pi
        umax2 = S(40.0) / pi + S(40.0)
        u = xp.where(zu <= udep2, umax1 + (zu - udep1) * (umax2 - umax1) / (udep2 - udep1), umax2)
        u0, v0 = setu(u, mu), setu(S(40.0) / pi + bk.zeros(shv), mv)
    elif iwnd == 4:
        u0 = setu(S(35.0) * xp.tanh(zu / S(3000.0)), mu)
    elif iwnd == 5:
        umax, udep1, udep2 = S(15.0), S(4000.0), S(6000.0)
        q = S(0.25) * pi

        def alpha(z):
            return q * (S(1.0) + xp.cos(pi * (z - udep1) / (udep2 - udep1)))

        u = xp.where(zu < udep1, umax, xp.where(zu < udep2, umax * xp.sin(alpha(zu)), zero))
        v = xp.where(zv < udep1, zero, xp.where(zv < udep2, umax * xp.cos(alpha(zv)), umax))
        u0, v0 = setu(u, mu), setu(v, mv)
    elif iwnd == 6:
        uval, vval = {
            1: (1.0, 0.0), 2: (15.0, 0.0), 4: (7.0, -5.5), 9: (8.0, 0.0),
            11: (2.0, 0.0), 12: (20.0, 0.0), 14: (10.0, 0.0),
        }.get(testcase, (10.0, 0.0))  # fmt: skip
        u0 = S(uval) + bk.zeros(shu)  # every point (k = kb..ke)
        v0 = S(vval) + bk.zeros(shv)
    elif iwnd == 8:
        hurr_vg = S(float(cfg.get("hurr_vg", 40.0)))
        # param.F sets hurr_angle from hurr_rotate for lspgrad = 3 only
        ang = 0.0
        if int(cfg.get("lspgrad", 0) or 0) == 3:
            ang = float(cfg.get("hurr_rotate", 0.0)) * (4.0 * math.atan(1.0) / 180.0)
        hurr_angle = S(ang)
        u = -hurr_vg * xp.sin(-hurr_angle) + bk.zeros(shu)
        v = hurr_vg * xp.cos(-hurr_angle) + bk.zeros(shv)
        u0 = setu(u, mask(shu[1:], 0, nj + 2, 0, ni + 2))
        v0 = setu(v, mask(shv[1:], 0, nj + 2, 0, ni + 2))
    elif iwnd == 9:
        u0 = setu(shear(zu, 700.0, 3000.0, -8.75, -4.61, True), mu_w)
    elif iwnd == 10:
        u = S(3.0) + S(4.3) * zu / S(1000.0)
        v = S(-9.0) + S(5.6) * zv / S(1000.0)
        u0, v0 = setu(u, mu_w), setu(v, mv_w)
    elif iwnd == 11:
        u0 = setu(shear(zu, 0.0, 4000.0, -9.9, -1.9, True), mu_w)
        v0 = setu(S(-3.8) + bk.zeros(shv), mv_w)
    return u0, v0
