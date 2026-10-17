"""RK3 loop and acoustic solver call, ported from src/solve2.F.

Active path for the reference cases only: psolver=3, imoist=1, eqtset=2,
ptype>=1 (Kessler: qv, qc, qr), icor=0, no terrain, Cartesian, apmasscon=1,
no PBL/passive tracers.  Lateral boundaries: any type of cm1x.bc (periodic,
open, walls), with the irbc=4 radiative phase speeds (radbcew4/radbcns4
after each acoustic-step sequence).  Prognostic TKE (sgsmodel=1): advected
with advw (advwenos) and stepped on each RK stage.
"""

from __future__ import annotations

from . import constants as c
from .adv import advs, advu, advv, advw, getdivx
from . import bc as _bc
from .misc import NRKMAX, SMEPS, getdiv


def _opt(cfg, name, default):
    if cfg is None:
        return default
    # dict.get / Config.get: an unset option falls back to ``default``
    # rather than Config's zero for unknown names
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def _check(cfg):
    bad = []
    for name, ok in (
        ("psolver", (3,)),
        ("imoist", (1,)),
        ("eqtset", (2,)),
        ("icor", (0,)),
        ("axisymm", (0,)),
        ("iptra", (0,)),
        ("ipbl", (0,)),
        ("cuparam", (0,)),
        ("iprcl", (0,)),
        ("nudgeobc", (0,)),
        ("iice", (0,)),
    ):
        v = _opt(cfg, name, ok[0])
        if v not in ok:
            bad.append(f"{name}={v}")
    if _opt(cfg, "idiff", 0) >= 1 and _opt(cfg, "difforder", 6) == 6:
        bad.append("idiff>=1 with difforder=6")
    if _opt(cfg, "terrain_flag", False):
        bad.append("terrain_flag=T")
    _bc.ib_flags(cfg)  # validates wbc..nbc
    if int(_opt(cfg, "irbc", 4)) not in (4,) and any(_bc.ib_flags(cfg)):
        bad.append(f"irbc={_opt(cfg, 'irbc', 4)}")
    if bad:
        raise NotImplementedError("solve2: options not ported: " + ", ".join(bad))


def solve2(
    bk,
    g,
    cfg=None,
    *,
    dt,
    mass1,
    uh,
    vh,
    ruh,
    rvh,
    uf,
    vf,
    mh,
    rmh,
    mf,
    c1,
    c2,
    pi0,
    rho0,
    th0,
    rth0,
    qv0,
    qc0,
    qi0,
    rr0,
    rf0,
    rrf0,
    dtu0,
    dtv0,
    divx,
    rho,
    prs,
    u0,
    ua,
    u3d,
    uten1,
    v0,
    va,
    v3d,
    vten1,
    wa,
    w3d,
    wten1,
    ppi,
    pp3d,
    ppten,
    ppx,
    tha,
    th3d,
    thten1,
    qpten,
    qtten,
    qvten,
    qcten,
    qa,
    q3d,
    qten,
    ruf=None,
    rvf=None,
    radbcw=None,
    radbce=None,
    radbcs=None,
    radbcn=None,
    tkea=None,
    tke3d=None,
    tketen=None,
    hooks=None,
    **unused,
):
    """solve2.F ``solve2`` (active path).  ``qa``, ``q3d``, ``qten`` are
    sequences of numq moisture arrays.  ``mass1`` is the initial dry-air
    mass (init3d.F getset).  Returns the updated prognostic arrays
    (u3d, v3d, w3d, pp3d, th3d, q3d, prs, rho, ppx, divx, rru, rrv, rrw,
    qpten, qtten, qvten, qcten, thterm, mass2, radbcw, radbce, radbcs,
    radbcn and, with prognostic TKE, tke3d).

    ``radbc*`` are the persistent open-boundary phase speeds (``[k, j]`` for
    w/e, ``[k, i]`` for s/n; zeros at t=0), ``ruf``/``rvf`` = 1/uf, 1/vf.
    ``tkea``, ``tke3d``, ``tketen`` (solve1) are needed for sgsmodel=1.

    ``hooks`` (optional callable ``hooks(stage_name, nrk, locals_dict)``)
    lets tests inspect intermediate results.
    """
    _check(cfg)
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    n, nz = g.ngxy, g.ngz
    numq = len(qa)
    nqv = int(_opt(cfg, "nqv", 1)) - 1
    nqc = int(_opt(cfg, "nqc", 2)) - 1
    nql1 = int(_opt(cfg, "nql1", 2)) - 1
    nql2 = int(_opt(cfg, "nql2", 3)) - 1
    ptype = int(_opt(cfg, "ptype", 1))
    apmasscon = int(_opt(cfg, "apmasscon", 1))
    hs, vs = int(_opt(cfg, "hadvordrs", 5)), int(_opt(cfg, "vadvordrs", 5))
    hv, vv = int(_opt(cfg, "hadvordrv", 5)), int(_opt(cfg, "vadvordrv", 5))
    advwenov = int(_opt(cfg, "advwenov", 0))
    nx, ny, nz_ = (
        int(_opt(cfg, "nx", ni)),
        int(_opt(cfg, "ny", nj)),
        int(_opt(cfg, "nz", nk)),
    )
    qmag = [1.0e-2] * numq

    from .sound import sound  # local import: keeps module import order free

    ibw, ibe, ibs, ibn = _bc.ib_flags(cfg)
    irbc = int(_opt(cfg, "irbc", 4))
    S0 = g.s_shape
    radbcw = bk.zeros((S0[0], S0[1])) if radbcw is None else radbcw
    radbce = bk.zeros((S0[0], S0[1])) if radbce is None else radbce
    radbcs = bk.zeros((S0[0], S0[2])) if radbcs is None else radbcs
    radbcn = bk.zeros((S0[0], S0[2])) if radbcn is None else radbcn
    les = int(_opt(cfg, "sgsmodel", 0)) == 1 and int(_opt(cfg, "cm1setup", 1)) in (
        1,
        4,
    )
    if les:
        if tkea is None or tke3d is None or tketen is None:
            raise ValueError("solve2: sgsmodel=1 needs tkea, tke3d, tketen")
        tke3d = bk.copy(tke3d)
    advwenos = int(_opt(cfg, "advwenos", 0))

    u3d, v3d, w3d = bk.copy(u3d), bk.copy(v3d), bk.copy(w3d)
    pp3d, th3d = bk.copy(pp3d), bk.copy(th3d)
    q3d = [bk.copy(q) for q in q3d]
    rho, prs, divx, ppx = bk.copy(rho), bk.copy(prs), bk.copy(divx), bk.copy(ppx)
    qpten, qtten, qvten, qcten = (bk.copy(a) for a in (qpten, qtten, qvten, qcten))

    S, U, V, W = g.s_shape, g.u_shape, g.v_shape, g.w_shape
    K = slice(nz, nz + nk)
    Ji, Ii = slice(n, n + nj), slice(n, n + ni)
    Jp, Ip = slice(n, n + nj + 1), slice(n, n + ni + 1)
    J0, I0 = slice(n - 1, n + nj + 1), slice(n - 1, n + ni + 1)  # 0..n+1
    J02, I02 = slice(n - 1, n + nj + 2), slice(n - 1, n + ni + 2)  # 0..n+2
    SI = (K, Ji, Ii)

    def col(a, ks):
        return a[ks, n : n + 1, n : n + 1]

    thterm = bk.zeros(S)
    t11 = bk.zeros(S)
    t22 = bk.zeros(S)
    dum8 = bk.zeros(S)
    sadv = bk.zeros(S)
    rru, rrv, rrw = bk.zeros(U), bk.zeros(V), bk.zeros(W)
    uten, vten, wten = bk.zeros(U), bk.zeros(V), bk.zeros(W)
    sret = {}
    mass2 = None

    for nrk in range(1, NRKMAX + 1):
        dttmp = dt / float(NRKMAX + 1 - nrk)

        # ---- misc variables (moist, eqtset=2) -----------------------------
        ql = q3d[nql1][SI]
        for m in range(nql1 + 1, nql2 + 1):
            ql = ql + q3d[m][SI]
        zero = xp.zeros_like(ql)
        qv = xp.maximum(q3d[nqv][SI], zero)
        qli = xp.maximum(zero, ql)  # + dum3 (=0, no ice)
        cpli = c.cpl * xp.maximum(zero, ql)
        dum8 = bk.set(
            dum8,
            SI,
            c.g
            * (
                th3d[SI] * rth0[SI]
                + c.repsm1 * (qv - qv0[SI])
                - (qli - qc0[SI] - qi0[SI])
            ),
        )
        t11 = bk.set(
            t11, SI, (th0[SI] + th3d[SI]) * (1.0 + c.reps * qv) / (1.0 + qv + qli)
        )
        cpm = c.cp + c.cpv * qv + cpli
        cvm = 1.0 / (c.cv + c.cvv * qv + cpli)
        thterm = bk.set(
            thterm, SI, (th0[SI] + th3d[SI]) * (c.rd + c.rv * qv - c.rovcp * cpm) * cvm
        )
        t22 = bk.set(t22, SI, (pi0[SI] + pp3d[SI]) * c.rovcp * cpm * cvm)
        t11 = _bc.bcp(bk, g, t11, cfg)

        # ---- RK tendency arrays --------------------------------------------
        R = (K, Jp, Ip)
        uten = bk.set(uten, R, uten1[R])
        vten = bk.set(vten, R, vten1[R])
        wten = bk.set(wten, R, wten1[R])

        # ---- rru, rrv, rrw, divx (no terrain) ------------------------------
        rru = bk.set(rru, (K, J0, I02), col(rho0, K) * u3d[K, J0, I02])
        rrv = bk.set(rrv, (K, J02, I0), col(rho0, K) * v3d[K, J02, I0])
        rrw = bk.set(rrw, (nz, J0, I0), bk.zeros(rrw[nz, J0, I0].shape))
        rrw = bk.set(rrw, (nz + nk, J0, I0), bk.zeros(rrw[nz, J0, I0].shape))
        Kw = slice(nz + 1, nz + nk)
        rrw = bk.set(rrw, (Kw, J0, I0), col(rf0, Kw) * w3d[Kw, J0, I0])

        uh0 = uh[I0][None, None, :]
        vh0 = vh[J0][None, :, None]
        d = (
            (rru[K, J0, n : n + ni + 2] - rru[K, J0, n - 1 : n + ni + 1]) * g.rdx * uh0
            + (rrv[K, n : n + nj + 2, I0] - rrv[K, n - 1 : n + nj + 1, I0])
            * g.rdy
            * vh0
        ) + (rrw[nz + 1 : nz + nk + 1, J0, I0] - rrw[K, J0, I0]) * g.rdz * col(mh, K)
        d = xp.where(xp.abs(d) < SMEPS, xp.zeros_like(d), d)
        divx = bk.set(divx, (K, J0, I0), d)

        # ---- advection of momentum -----------------------------------------
        uten = advu(
            bk,
            g,
            nrk=nrk,
            rr0=rr0,
            mh=mh,
            uf=uf,
            vh=vh,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            u3d=u3d,
            uten=uten,
            divx=divx,
            c1=c1,
            c2=c2,
            dt=dttmp,
            cfg=cfg,
        )["uten"]
        vten = advv(
            bk,
            g,
            nrk=nrk,
            rr0=rr0,
            mh=mh,
            uh=uh,
            vf=vf,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            v3d=v3d,
            vten=vten,
            divx=divx,
            c1=c1,
            c2=c2,
            dt=dttmp,
            cfg=cfg,
        )["vten"]
        wten = advw(
            bk,
            g,
            nrk=nrk,
            rrf0=rrf0,
            mf=mf,
            uh=uh,
            vh=vh,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            w3d=w3d,
            wten=wten,
            divx=divx,
            c1=c1,
            c2=c2,
            hadvorder=hv,
            vadvorder=vv,
            advweno=advwenov,
            dt=dttmp,
            iamw=True,
            cfg=cfg,
        )["wten"]

        # ---- buoyancy --------------------------------------------------------
        Kb = slice(nz + 1, nz + nk)  # k = 2..nk
        wten = bk.set(
            wten,
            (Kb, Ji, Ii),
            wten[Kb, Ji, Ii]
            + (
                c1[Kb, Ji, Ii] * dum8[nz : nz + nk - 1, Ji, Ii]
                + c2[Kb, Ji, Ii] * dum8[Kb, Ji, Ii]
            ),
        )

        if hooks is not None:
            hooks(
                "pre_sound",
                nrk,
                dict(
                    uten=uten,
                    vten=vten,
                    wten=wten,
                    dum8=dum8,
                    t11=t11,
                    t22=t22,
                    divx=divx,
                    rru=rru,
                    rrv=rrv,
                    rrw=rrw,
                ),
            )

        # ---- pressure equation: sadv = pi0 + pp3d ---------------------------
        sadv = bk.set(sadv, (K, J0, I0), col(pi0, K) + pp3d[K, J0, I0])

        # ---- acoustic steps ---------------------------------------------------
        so = sound(
            bk,
            g,
            cfg,
            dt=dt,
            nrk=nrk,
            uh=uh,
            uf=uf,
            vh=vh,
            vf=vf,
            mh=mh,
            mf=mf,
            c1=c1,
            c2=c2,
            rr0=rr0,
            rf0=rf0,
            dtu0=dtu0,
            dtv0=dtv0,
            ua=ua,
            u3d=u3d,
            uten=uten,
            va=va,
            v3d=v3d,
            vten=vten,
            wa=wa,
            w3d=w3d,
            wten=wten,
            ppi=ppi,
            pp3d=pp3d,
            piadv=sadv,
            ppten=ppten,
            ppx=ppx,
            thv=t11,
            ppterm=t22,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            radbcw=radbcw,
            radbce=radbce,
            radbcs=radbcs,
            radbcn=radbcn,
        )
        u3d, v3d, w3d, pp3d, ppx = (
            so["u3d"],
            so["v3d"],
            so["w3d"],
            so["pp3d"],
            so["ppx"],
        )
        rru, rrv, rrw = so["rru"], so["rrv"], so["rrw"]
        sret = so

        # ---- radiative open boundaries: phase speeds for the next stage -------
        if irbc == 4:
            if ibw or ibe:
                rb = _bc.radbcew4(
                    bk,
                    g,
                    cfg,
                    ruf=ruf,
                    radbcw=radbcw,
                    radbce=radbce,
                    u1=ua,
                    u2=u3d,
                    dt=dttmp,
                )
                radbcw, radbce = rb["radbcw"], rb["radbce"]
            if ibs or ibn:
                rb = _bc.radbcns4(
                    bk,
                    g,
                    cfg,
                    rvf=rvf,
                    radbcs=radbcs,
                    radbcn=radbcn,
                    v1=va,
                    v2=v3d,
                    dt=dttmp,
                )
                radbcs, radbcn = rb["radbcs"], rb["radbcn"]

        if hooks is not None:
            hooks(
                "post_sound",
                nrk,
                dict(
                    u3d=u3d,
                    v3d=v3d,
                    w3d=w3d,
                    pp3d=pp3d,
                    ppx=ppx,
                    rru=rru,
                    rrv=rrv,
                    rrw=rrw,
                ),
            )

        # ---- 3-D divergence (time-averaged velocities) into t11 ------------
        t11 = getdiv(
            bk,
            g,
            uh=uh[None, None, :],
            vh=vh[None, :, None],
            mh=mh,
            u=rru,
            v=rrv,
            w=rrw,
            div=t11,
        )["div"]
        gd = getdivx(
            bk,
            g,
            uh=uh,
            vh=vh,
            mh=mh,
            rho0=rho0,
            rf0=rf0,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            divx=divx,
            cfg=cfg,
        )
        rru, rrv, rrw, divx = gd["rru"], gd["rrv"], gd["rrw"], gd["divx"]

        # ---- theta --------------------------------------------------------------
        thten = bk.zeros(S)
        thten = bk.set(thten, SI, thten1[SI] - t11[SI] * thterm[SI])
        sadv = bk.set(sadv, (K,), col(th0, K) - c.th0r + th3d[K])
        thten = advs(
            bk,
            g,
            nrk=nrk,
            wflag=1,
            rho0=rho0,
            rr0=rr0,
            mh=mh,
            uh=uh,
            vh=vh,
            rru=rru,
            rrv=rrv,
            rrw=rrw,
            divx=divx,
            s0=tha,
            s=sadv,
            sten=thten,
            pdef=0,
            dt=dttmp,
            c1=c1,
            c2=c2,
            hadvorder=hs,
            vadvorder=vs,
            weps=10.0 * c.epsilon,
            pdefweno=0,
            cfg=cfg,
        )["sten"]
        tn = tha[SI] + dttmp * thten[SI]
        tn = xp.where(xp.abs(tn) < SMEPS, xp.zeros_like(tn), tn)
        th3d = bk.set(th3d, SI, tn)

        # ---- moisture --------------------------------------------------------
        pdef = 1 if nrk == NRKMAX else 0
        for m in range(numq):
            sten = bk.zeros(S)
            sten = bk.set(sten, SI, qten[m][SI])
            sten = advs(
                bk,
                g,
                nrk=nrk,
                wflag=1,
                rho0=rho0,
                rr0=rr0,
                mh=mh,
                uh=uh,
                vh=vh,
                rru=rru,
                rrv=rrv,
                rrw=rrw,
                divx=divx,
                s0=qa[m],
                s=q3d[m],
                sten=sten,
                pdef=pdef,
                dt=dttmp,
                c1=c1,
                c2=c2,
                hadvorder=hs,
                vadvorder=vs,
                weps=qmag[m] * c.epsilon,
                pdefweno=(0 if m == nqv else 1),
                cfg=cfg,
            )["sten"]
            qn = qa[m][SI] + dttmp * sten[SI]
            qn = xp.where(xp.abs(qn) < SMEPS, xp.zeros_like(qn), qn)
            q3d[m] = bk.set(q3d[m], SI, qn)

        if hooks is not None:
            hooks("post_scalars", nrk, dict(th3d=th3d, q3d=q3d, thten=thten, t11=t11))

        # ---- pressure, density ------------------------------------------------
        dopf = False
        if nrk == NRKMAX and ptype >= 1:
            # subtract-off estimated diabatic terms used during RK steps,
            # save values before microphysics:
            pp3d = bk.set(pp3d, SI, pp3d[SI] - dt * qpten[SI])
            qpten = bk.set(qpten, SI, pp3d[SI])
            th3d = bk.set(th3d, SI, th3d[SI] - dt * qtten[SI])
            qtten = bk.set(qtten, SI, th3d[SI])
            q3d[nqv] = bk.set(q3d[nqv], SI, q3d[nqv][SI] - dt * qvten[SI])
            qvten = bk.set(qvten, SI, q3d[nqv][SI])
            q3d[nqc] = bk.set(q3d[nqc], SI, q3d[nqc][SI] - dt * qcten[SI])
            qcten = bk.set(qcten, SI, q3d[nqc][SI])

        def _prsrho(pp3d, prs, rho):
            pit = pi0[SI] + pp3d[SI]
            p = c.p00 * (pit**c.cpdrd)
            r = p / (
                (th0[SI] + th3d[SI])
                * pit
                * (c.rd + xp.maximum(q3d[nqv][SI], xp.zeros_like(p)) * c.rv)
            )
            return bk.set(prs, SI, p), bk.set(rho, SI, r)

        if nrk == NRKMAX:
            prs, rho = _prsrho(pp3d, prs, rho)
            if apmasscon == 1:
                dumk3 = xp.sum(
                    rho[SI] * ruh[Ii][None, None, :] * rvh[Ji][None, :, None] * rmh[SI],
                    axis=(1, 2),
                )
                dumk4 = xp.sum(pi0[SI] + pp3d[SI], axis=(1, 2))
                mass2 = xp.sum(dumk3)
                p2 = xp.sum(dumk4)
                dopf = True

        # ---- bcs ------------------------------------------------------------------
        u3d = _bc.bcu(bk, g, u3d, cfg)
        v3d = _bc.bcv(bk, g, v3d, cfg)
        w3d = _bc.bcw(bk, g, w3d, 1, cfg)
        if nrk < NRKMAX:
            rho = _bc.bcp(bk, g, rho, cfg)
            pp3d = _bc.bcp(bk, g, pp3d, cfg)
            th3d = _bc.bcs(bk, g, th3d, cfg)
            q3d = [_bc.bcs(bk, g, q, cfg) for q in q3d]

        # ---- TKE (sgsmodel=1) ------------------------------------------------
        if les:
            from .turb import tke_rk_update

            Kt = slice(nz + 1, nz + nk)  # k = 2..nk
            wten = bk.set(wten, (Kt, Ji, Ii), tketen[Kt, Ji, Ii])
            wten = advw(
                bk,
                g,
                nrk=nrk,
                rrf0=rrf0,
                mf=mf,
                uh=uh,
                vh=vh,
                rru=rru,
                rrv=rrv,
                rrw=rrw,
                w3d=tke3d,
                wten=wten,
                divx=divx,
                c1=c1,
                c2=c2,
                hadvorder=hs,
                vadvorder=vs,
                advweno=advwenos,
                dt=dttmp,
                iamw=False,
                cfg=cfg,
            )["wten"]
            tke3d = tke_rk_update(
                bk, g, cfg, dttmp=dttmp, tkea=tkea, tten=wten, tke3d=tke3d
            )["tke3d"]

        # ---- finish pressure / density (apmasscon) -----------------------------
        if dopf:
            tem = ((mass1 / mass2) ** c.rddcv - 1.0) * p2 / float(nx * ny * nz_)
            pp3d = bk.set(pp3d, SI, pp3d[SI] + tem)
            prs, rho = _prsrho(pp3d, prs, rho)

    return {
        "u3d": u3d,
        "v3d": v3d,
        "w3d": w3d,
        "pp3d": pp3d,
        "th3d": th3d,
        "q3d": q3d,
        "prs": prs,
        "rho": rho,
        "ppx": ppx,
        "divx": divx,
        "rru": rru,
        "rrv": rrv,
        "rrw": rrw,
        "qpten": qpten,
        "qtten": qtten,
        "qvten": qvten,
        "qcten": qcten,
        "thterm": thterm,
        "mass2": mass2,
        "uten": uten,
        "vten": vten,
        "wten": wten,
        "dtu": sret.get("dtu"),
        "dtv": sret.get("dtv"),
        "radbcw": radbcw,
        "radbce": radbce,
        "radbcs": radbcs,
        "radbcn": radbcn,
        "tke3d": tke3d,
    }
