"""Kessler warm-rain microphysics (ptype=1), ported from src/kessler.F.

Routines: ``kessler`` (autoconversion, accretion, rain evaporation),
``satadj`` (iterative saturation adjustment, eqtset=2, plus the eqtset=1
one-step branch), ``k_fallout`` (rain fall speed), ``fallout``
(sedimentation), ``getefall`` and ``geterain`` (fall-energy terms/budget).

Arrays are ``a[k, j, i]`` with CM1's halos (cm1x.grid).  Only interior points
(Fortran i=1..ni, j=1..nj, k=1..nk) are computed; halos pass through
unchanged.  ``q3d`` is ``[numq, k, j, i]`` (Fortran ``q3d(i,j,k,n)``), with
qv, qc, qr at Python indices 0, 1, 2.  ``ruh``/``rvh`` are 1-D over the i/j
scalar points (with halos).  Budget scalars (tauto, tcond, train, ...) are
CM1 ``double precision`` accumulators: pass the running value, get back the
updated one as a 0-d array.
"""

from __future__ import annotations

from . import constants as C

nrkmax = 3  # constants.F

_DEFAULTS = dict(
    eqtset=2,
    tsmall=1.0e-4,  # cm1.F default (1e-12 only with -DDP)
    qsmall=1.0e-14,
    nqv=1,
    nqc=2,
    nql1=2,
    nql2=3,
    iice=0,
    nqs1=-2,
    nqs2=-2,
    satadj_maxiter=20,  # see satadj docstring
)


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    # dict.get / Config.get: an unset option falls back to this module's
    # default rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])


def _hw(g, ruh, rvh):
    """ruh(i)*rvh(j) on the interior, broadcastable to [k, j, i]."""
    ru = ruh[g.I] if ruh.ndim == 1 else ruh[..., g.I]
    rv = rvh[g.J] if rvh.ndim == 1 else rvh[..., g.J, :]
    if ruh.ndim == 1:
        ru = ru[None, None, :]
    if rvh.ndim == 1:
        rv = rv[None, :, None]
    return ru, rv


def _esat_qvs(xp, t, prs, pden):
    """esl = 611.2 exp(17.67 (T-273.15)/(T-29.65)), capped at prs/2; qvs."""
    esl = 611.2 * xp.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = xp.minimum(esl, prs * 0.5)
    return C.eps * esl / (pden - esl)


# ---------------------------------------------------------------------------


def kessler(
    bk,
    g,
    cfg=None,
    *,
    dt,
    tauto,
    taccr,
    tevar,
    ruh,
    rvh,
    rmh,
    pi0,
    th0,
    tmp,
    rho,
    rr,
    pp3d,
    th3d,
    prs,
    qv3d,
    qc3d,
    qr3d,
):
    """kessler.F ``kessler``.  Returns dict with tmp, rho, pp3d, th3d, prs,
    qv3d, qc3d, qr3d, tauto, taccr, tevar."""
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    eqtset = int(_opt(cfg, "eqtset"))
    S = (K, J, I)

    th0i, pi0i = th0[S], pi0[S]
    thi, ppi_, prsi, rhoi = th3d[S], pp3d[S], prs[S], rho[S]
    qv, qc, qr = qv3d[S], qc3d[S], qr3d[S]

    t = (th0i + thi) * (pi0i + ppi_)

    # autoconversion
    ar = xp.where(
        qc > 0.0,
        xp.maximum(0.001 * (qc - 0.001), xp.zeros_like(qc)) * dt,
        xp.zeros_like(qc),
    )
    # accretion
    qrp = xp.where(qr > 0.0, qr, xp.ones_like(qr))  # safe base for powers
    cr = xp.where(
        (qc > 0.0) & (qr > 0.0), 2.2 * qc * (qrp**0.875) * dt, xp.zeros_like(qc)
    )
    # rain evaporation
    qvs = _esat_qvs(xp, t, prsi, prsi)
    rq = rhoi * qrp
    er = (
        (1.6 + 30.3922 * (rq**0.2046))
        * (1.0 - (qv / qvs))
        * (rq**0.525)
        / ((2.03e4 + 9.584e6 / (qvs * prsi)) * rhoi)
    )
    er = xp.minimum(er * dt, qr)
    er = xp.where((qv + er) > qvs, qvs - qv, er)
    er = xp.where((qr > 0.0) & (qv < qvs), er, xp.zeros_like(er))

    term1 = ar + cr
    resc = term1 > qc
    den = xp.where(resc, term1, xp.ones_like(term1))
    ar = xp.where(resc, qc * ar / den, ar)
    cr = xp.where(resc, qc * cr / den, cr)

    qvn = qv + er
    qcn = qc - (ar + cr)
    qrn = qr + (ar + cr - er)

    doit = er > 1.0e-7
    if eqtset == 2:
        cpml = C.cp + C.cpv * qvn + C.cpl * (qcn + qrn)
        cvml = C.cv + C.cvv * qvn + C.cpl * (qcn + qrn)
        rm = C.rd + C.rv * qvn
        thn = thi - er * (
            (C.lv1 - C.lv2 * t) / (C.cpdcv * cvml * (pi0i + ppi_))
            - (th0i + thi) * (C.rv / cvml) * (1.0 - C.rovcp * cpml / rm)
        )
        ppn = ((rhoi * rm * (th0i + thn) * C.rp00) ** C.rddcv) - pi0i
        prn = C.p00 * ((pi0i + ppn) ** C.cpdrd)
        thi = xp.where(doit, thn, thi)
        ppi_ = xp.where(doit, ppn, ppi_)
        prsi = xp.where(doit, prn, prsi)
    else:
        thn = thi - er * ((C.lv1 - C.lv2 * t) / (C.cp * (pi0i + ppi_)))
        rhn = prsi / (C.rd * (th0i + thn) * (pi0i + ppi_) * (1.0 + qvn * C.reps))
        thi = xp.where(doit, thn, thi)
        rhoi = xp.where(doit, rhn, rhoi)

    ru, rv = _hw(g, ruh, rvh)
    w = rr[S] * (ru * rv * rmh[S])
    dum = g.dx * g.dy * g.dz
    return {
        "tmp": bk.set(bk.copy(tmp), S, t),
        "rho": bk.set(bk.copy(rho), S, rhoi),
        "pp3d": bk.set(bk.copy(pp3d), S, ppi_),
        "th3d": bk.set(bk.copy(th3d), S, thi),
        "prs": bk.set(bk.copy(prs), S, prsi),
        "qv3d": bk.set(bk.copy(qv3d), S, qvn),
        "qc3d": bk.set(bk.copy(qc3d), S, qcn),
        "qr3d": bk.set(bk.copy(qr3d), S, qrn),
        "tauto": tauto + xp.sum(w * ar) * dum,
        "taccr": taccr + xp.sum(w * cr) * dum,
        "tevar": tevar + xp.sum(w * er) * dum,
    }


# ---------------------------------------------------------------------------


def satadj(
    bk,
    g,
    cfg=None,
    *,
    nrk,
    dt,
    tcond,
    tevac,
    ruh,
    rvh,
    rmh,
    pi0,
    th0,
    rho,
    rr,
    pp3d,
    prs,
    th3d,
    q3d,
    maxiter=None,
):
    """kessler.F ``satadj`` (warm saturation adjustment).

    eqtset=2: CM1's per-point iteration (``do while`` until |dT| <= converge,
    stopping at n=100) is run as ``maxiter`` masked sweeps over the whole
    field; each point freezes once converged, so results equal the Fortran
    whenever every point converges within ``maxiter`` iterations (typically
    2-4).  ``maxiter`` defaults to cfg ``satadj_maxiter`` (20).  The returned
    ``nmax`` (0-d int array) is the largest per-point iteration count, as in
    the Fortran; ``nmax == maxiter`` means some point may not have converged
    (Fortran stops the model when n reaches 100).

    ``nrk`` is a Python int (4 from mp_driver).  Returns dict with rho, pp3d,
    prs, th3d, q3d, tcond, tevac, nmax.
    """
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    S = (K, J, I)
    eqtset = int(_opt(cfg, "eqtset"))
    tsmall = float(_opt(cfg, "tsmall"))
    nqv = int(_opt(cfg, "nqv")) - 1
    nqc = int(_opt(cfg, "nqc")) - 1
    nql1, nql2 = int(_opt(cfg, "nql1")), int(_opt(cfg, "nql2"))
    iice = int(_opt(cfg, "iice"))
    nqs1, nqs2 = int(_opt(cfg, "nqs1")), int(_opt(cfg, "nqs2"))
    if maxiter is None:
        maxiter = int(_opt(cfg, "satadj_maxiter"))
    maxiter = min(int(maxiter), 100)

    th0i, pi0i = th0[S], pi0[S]
    thi, ppi_, prsi, rhoi = th3d[S], pp3d[S], prs[S], rho[S]
    qv0 = q3d[(nqv,) + S]
    qc0 = q3d[(nqc,) + S]

    if eqtset == 2:
        converge = 2.0 * tsmall if nrk == 4 else 20.0 * tsmall
        tnew = (th0i + thi) * (pi0i + ppi_)
        qvs = _esat_qvs(xp, tnew, prsi, prsi)
        act0 = (qc0 > 1.0e-12) | (qv0 > qvs)

        ql = xp.zeros_like(qv0)
        for n in range(nql1, nql2 + 1):
            ql = ql + q3d[(n - 1,) + S]
        qi = xp.zeros_like(qv0)
        if iice == 1:
            for n in range(nqs1, nqs2 + 1):
                qi = qi + q3d[(n - 1,) + S]
        ql = xp.maximum(xp.zeros_like(ql), ql)
        qi = xp.maximum(xp.zeros_like(qi), qi)
        cvml = C.cv + C.cvv * qv0 + C.cpl * ql + C.cpi * qi
        lhv = C.lv1 - C.lv2 * tnew
        t1 = (lhv - C.rv * tnew) / cvml
        d1 = t1 * 17.67 * 243.5

        nit = xp.zeros(qv0.shape, dtype=xp.int32, device=bk.device)

        def sweep(last, c):
            qvnew, qcnew, tnew, pnew, nit, tlast, qvs, active = c
            dqv = (qvs - qvnew) / (1.0 + d1 * qvs / ((tnew - 29.65) ** 2))
            dqv = xp.minimum(dqv, qcnew)
            dqv = xp.where((qvnew + dqv) < 1.0e-20, 1.0e-20 - qvnew, dqv)
            qv_n = qvnew + dqv
            qc_n = qcnew - dqv
            t_n = tnew - dqv * t1
            p_n = rhoi * (C.rd + C.rv * qv_n) * t_n
            qvnew = xp.where(active, qv_n, qvnew)
            qcnew = xp.where(active, qc_n, qcnew)
            tnew = xp.where(active, t_n, tnew)
            pnew = xp.where(active, p_n, pnew)
            nit = xp.where(active, nit + 1, nit)
            again = active & (xp.abs(tnew - tlast) > converge)
            if last:  # Fortran gives up at n = 100
                again = xp.zeros_like(again)
            tlast = xp.where(again, tnew, tlast)
            qvs_n = _esat_qvs(xp, tnew, prsi, pnew)
            qvs = xp.where(again, qvs_n, qvs)
            return qvnew, qcnew, tnew, pnew, nit, tlast, qvs, again

        c = (qv0, qc0, tnew, prsi, nit, tnew, qvs, act0)
        nfree = maxiter - 1 if maxiter == 100 else maxiter
        c = bk.fori_loop(nfree, lambda it, c: sweep(False, c), c)
        if maxiter == 100:
            c = sweep(True, c)
        qvnew, qcnew, tnew, pnew, nit, tlast, qvs, active = c

        pp_n = (pnew * C.rp00) ** C.rovcp - pi0i
        th_n = tnew / (pi0i + pp_n) - th0i
        prsi = xp.where(act0, pnew, prsi)
        ppi_ = xp.where(act0, pp_n, ppi_)
        thi = xp.where(act0, th_n, thi)
        nmax = xp.max(nit)
    else:
        qvnew, qcnew = qv0, qc0
        tnew = (th0i + thi) * (pi0i + ppi_)
        qvs = _esat_qvs(xp, tnew, prsi, prsi)
        lhv = C.lv1 - C.lv2 * tnew
        dqv = (qvs - qvnew) / (
            1.0 + lhv * qvs * 17.67 * 243.5 / (C.cp * ((tnew - 29.65) ** 2))
        )
        dqv = xp.where((qvnew + dqv) < 1.0e-20, 1.0e-20 - qvnew, dqv)
        dqv = xp.minimum(dqv, xp.maximum(xp.zeros_like(qcnew), qcnew))
        qvnew = qvnew + dqv
        qcnew = qcnew - dqv
        thi = thi - dqv * (lhv / (C.cp * (pi0i + ppi_)))
        rhoi = prsi / (C.rd * (th0i + thi) * (pi0i + ppi_) * (1.0 + qvnew * C.reps))
        nmax = xp.asarray(1)

    q = bk.copy(q3d)
    q = bk.set(q, (nqc,) + S, qcnew)
    q = bk.set(q, (nqv,) + S, qvnew)

    out = {
        "rho": bk.set(bk.copy(rho), S, rhoi),
        "pp3d": bk.set(bk.copy(pp3d), S, ppi_),
        "prs": bk.set(bk.copy(prs), S, prsi),
        "th3d": bk.set(bk.copy(th3d), S, thi),
        "q3d": q,
        "tcond": tcond,
        "tevac": tevac,
        "nmax": nmax,
    }
    if nrk >= nrkmax:
        ru, rv = _hw(g, ruh, rvh)
        w = rr[S] * (ru * rv * rmh[S])
        z = xp.zeros_like(qv0)
        dum = g.dx * g.dy * g.dz
        out["tcond"] = tcond + xp.sum(w * xp.maximum(qcnew - qc0, z)) * dum
        out["tevac"] = tevac + xp.sum(w * xp.maximum(qvnew - qv0, z)) * dum
    return out


# ---------------------------------------------------------------------------


def _traced(bk, x):
    """True inside a jax trace (jit/grad), where values are not concrete."""
    if bk.name != "jax":
        return False
    import jax

    return isinstance(x, jax.core.Tracer)


def k_fallout(bk, g, cfg=None, *, rho, qr3d, vr=None):
    """kessler.F ``k_fallout``: rain terminal velocity on the interior.
    ``vr`` (optional) supplies the halo values; returns {"vr": ...}."""
    xp = bk.xp
    S = (g.K, g.J, g.I)
    r, q = rho[S], qr3d[S]
    qs = xp.where(q > 1.0e-12, q, xp.ones_like(q))
    v = xp.where(
        q > 1.0e-12, 14.34 * ((r * qs) ** 0.1346) * xp.sqrt(1.15 / r), xp.zeros_like(q)
    )
    base = bk.zeros(g.s_shape) if vr is None else bk.copy(vr)
    return {"vr": bk.set(base, S, v)}


def fallout(
    bk,
    g,
    cfg=None,
    *,
    dt,
    train,
    ruh,
    rvh,
    zh,
    mh,
    mf,
    rain,
    prate,
    rr,
    rho,
    q3d,
    vq,
    nfall_max=None,
):
    """kessler.F ``fallout``: flux-form sedimentation of ``q3d`` with fall
    speed ``vq`` (both ``g.s_shape``), sub-stepped per column with
    ``nfall = max(1, int(CFL+1))`` steps.

    ``rain`` is ``[nrain, j, i]`` and ``prate`` ``[j, i]`` (with halos).
    Columns are vectorised: a Python loop runs ``nfall_max`` sub-steps and a
    column is updated only for its first ``nfall`` of them.  ``nfall_max``
    defaults to the domain maximum of ``nfall`` (a data-dependent trip count:
    ``Backend.fori_loop`` runs it as a Python loop eagerly and as
    ``lax.while_loop`` under ``jax.jit``); a static int >= that maximum gives
    the same result with a fixed-trip loop (needed for reverse-mode AD under
    jit).  A static bound is checked eagerly (ValueError if exceeded); under
    a jax trace it cannot be (the extra sub-steps are masked no-ops, but a
    too-small bound silently truncates the sedimentation).  The fill of
    ``vq`` below the precipitation edge is a downward recurrence (loop over
    k).

    Returns dict with rain, prate, q3d, vq, train.  (zh and mf are unused,
    as in the Fortran.)
    """
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    S = (K, J, I)
    nk = g.nk
    rdz = 1.0 / g.dz

    q = q3d[S]
    v = vq[S]
    r = rho[S]
    m = mh[S]

    crmax = xp.max(v * dt * rdz * m, axis=0)
    crmax = xp.maximum(crmax, xp.zeros_like(crmax))
    nfall = xp.maximum(xp.floor(crmax + 1.0), xp.ones_like(crmax))  # [j, i], float

    # below the edge of falling precip, set fall speed to value above
    lev = [v[k] for k in range(nk)]
    for k in range(nk - 2, -1, -1):
        lev[k] = xp.where(q[k] < 1.0e-8, lev[k + 1], lev[k])
    v = xp.stack(lev, axis=0)

    if nfall_max is None:
        nfall_max = xp.max(nfall)  # 0-d array: Backend.fori_loop handles it
    elif not _traced(bk, nfall):  # eager: the static bound must hold
        nmax = int(bk.to_numpy(xp.max(nfall)))
        if nmax > int(nfall_max):
            raise ValueError(
                f"fallout: max nfall = {nmax} exceeds the static nfall_max = "
                f"{int(nfall_max)}"
            )
    dtfall = dt / nfall

    ru, rv = _hw(g, ruh, rvh)
    ru, rv = ru[0], rv[0]  # [1, ni], [nj, 1]
    rr1 = rr[K.start, J, I]
    v1 = v[0]
    zk = xp.zeros_like(q[:1])

    def substep(n, c):
        q, rn, pr, bud = c
        active = nfall > n
        rq = r * v * xp.maximum(xp.zeros_like(q), q)
        rqp = xp.concat([rq[1:], zk], axis=0)
        qn = q + dtfall[None] * (rqp - rq) * rdz * m / r
        q = xp.where(active[None], qn, q)
        rq1 = rq[0]
        rn = xp.where(active[None], rn + 0.1 * dtfall[None] * rq1[None], rn)
        pr = xp.where(active, rq1, pr)
        b = dtfall * rr1 * v1 * xp.maximum(xp.zeros_like(pr), q[0]) * ru * rv
        bud = xp.where(active, bud + b, bud)
        return q, rn, pr, bud

    pr = prate[J, I]
    carry = (q, rain[:, J, I], pr, xp.zeros_like(pr))
    if isinstance(nfall_max, (int, float)):
        nfall_max = int(nfall_max)
    q, rn, pr, bud = bk.fori_loop(nfall_max, substep, carry)

    return {
        "rain": bk.set(bk.copy(rain), (slice(None), J, I), rn),
        "prate": bk.set(bk.copy(prate), (J, I), pr),
        "q3d": bk.set(bk.copy(q3d), S, q),
        "vq": bk.set(bk.copy(vq), S, v),
        "train": train + xp.sum(bud) * (g.dx * g.dy),
    }


def getefall(bk, g, cfg=None, *, setzero, cpx, mf, t, cvm, tten, q3d, vr):
    """kessler.F ``getefall``: heating from falling hydrometeors, k=1..nk-1.
    ``mf`` is ``g.w_shape``.  Returns {"tten": ...}."""
    xp = bk.xp
    J, I = g.J, g.I
    nk = g.nk
    rdz = 1.0 / g.dz
    Kl = slice(1, nk)  # Fortran k = 1..nk-1
    Ku = slice(2, nk + 1)  # k+1
    S = (Kl, J, I)
    base = tten[S]
    if int(setzero) == 1:
        base = xp.zeros_like(base)
    val = (
        base
        + q3d[S]
        * vr[S]
        * (cpx * (t[Ku, J, I] - t[S]) * rdz * mf[Ku, J, I] + C.g)
        / cvm[S]
    )
    return {"tten": bk.set(bk.copy(tten), S, val)}


def geterain(bk, g, cfg=None, *, dt, cpx, lx1, erain, ruh, rvh, t, rr, q3d, vr):
    """kessler.F ``geterain``: energy budget of rain reaching the surface.
    Returns {"erain": erain + ...}."""
    xp = bk.xp
    J, I = g.J, g.I
    k1 = g.K.start
    ru, rv = _hw(g, ruh, rvh)
    b = (
        dt
        * vr[k1, J, I]
        * rr[k1, J, I]
        * q3d[k1, J, I]
        * (cpx * t[k1, J, I] - lx1)
        * ru[0]
        * rv[0]
    )
    return {"erain": erain + xp.sum(b) * g.dx * g.dy}
