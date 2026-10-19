"""Pre-RK tendencies, ported from src/solve1.F.

Only the code paths active for the reference cases (irdamp=0/1, hrdamp=0,
sgsmodel=0 or 1 (cm1setup=1), idiff=0, lspgrad=0/1, testcase=0, no
radiation/PBL/cumulus/nudging/recycling, psolver=3, eqtset=2, imoist=0/1,
ptype>=1 when moist; terrain_flag with or without TKE as far as cm1x.turb
supports it) are ported, in solve1.F's order:

  * sgsmodel=1: the TKE tendency tketen (shear/buoyancy production -
    dissipation, cm1x.turb.tke_source) plus its subgrid diffusion (turbt);
  * Rayleigh damping (irdamp=1) into uten1, vten1, wten1, thten1;
  * qten = 0;
  * lspgrad=1: large-scale pressure gradient in geostrophic balance with
    the base-state wind (icor=1, fcor);
  * sgsmodel=1: dissipative heating (idiss=1, cm1x.turb.dissheat) and the
    ``doturb`` block (turbu/turbv/turbw/turbs, cm1x.turb.turb_tendencies);
  * the mass-conserving pressure tendency ppten (eqtset=2, moist or dry)
    plus, when moist, the diabatic tendencies of the previous step (qpten,
    qtten, qvten, qcten).

With sgsmodel=0 the turbulence terms are identically zero (zero eddy
viscosities, stresses and dissipation) and are skipped, except with the
semi-slip lower boundary (bbc=3, which sfcmodel>=1 forces): then the surface
stress acts on u, v at k=1 and idiss=1 adds the surface-layer dissipation.
Radiation (radopt=2) adds thraten to thten1 after the dissipative heating.
"""

from __future__ import annotations

from . import constants as c
from .misc import QSMALL, TSMALL


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
        ("irdamp", (0, 1)),
        ("hrdamp", (0,)),
        ("lspgrad", (0, 1, 2, 3, 4)),
        ("idiff", (0,)),
        ("sgsmodel", (0, 1)),
        ("ipbl", (0,)),
        ("radopt", (0, 2)),
        ("cuparam", (0,)),
        ("testcase", tuple(range(16))),
        ("rterm", (0,)),
        ("eqtset", (2,)),
        ("psolver", (3,)),
        ("imoist", (0, 1)),
        ("isfcflx", (0,)),
        ("axisymm", (0,)),
    ):
        v = _opt(cfg, name, ok[0])
        if v not in ok:
            bad.append(f"{name}={v}")
    for name in (
        "do_lsnudge",
        "do_recycle_w",
        "do_recycle_e",
        "do_recycle_s",
        "do_recycle_n",
    ):
        if _opt(cfg, name, False):
            bad.append(f"{name}=T")
    if int(_opt(cfg, "sgsmodel", 0)) == 1 and int(_opt(cfg, "cm1setup", 1)) not in (
        0,
        1,
    ):
        bad.append(f"sgsmodel=1 with cm1setup={_opt(cfg, 'cm1setup', 1)}")
    if int(_opt(cfg, "lspgrad", 0)) >= 1 and int(_opt(cfg, "imove", 0)) != 0:
        bad.append("lspgrad=1 with imove=1")
    if bad:
        raise NotImplementedError("solve1: options not ported: " + ", ".join(bad))


def solve1(
    bk,
    g,
    cfg=None,
    *,
    dt,
    tauh,
    taus,
    tauf,
    rho,
    pi0,
    th0,
    u0,
    ua,
    v0,
    va,
    wa,
    ppi,
    tha,
    qa,
    qpten,
    qtten,
    qvten,
    qcten,
    uten1=None,
    vten1=None,
    wten1=None,
    thten1=None,
    ppten=None,
    qten=None,
    turb=None,
    thraten=None,
    tc=None,
    **unused,
):
    """solve1.F ``solve1`` (active path only).

    ``qa`` is a sequence of the numq moisture arrays (Fortran qa(:,:,:,n),
    n = nqv, nqc, nqr for Kessler).  Returns uten1, vten1, wten1, thten1,
    ppten, qten (list of numq arrays) and, with sgsmodel=1, tketen.

    ``turb`` (required for sgsmodel=1) is a dict with the grid metrics uh, uf,
    vh, vf, mh, mf, the density arrays rr, rf, and the output of the previous
    cm1x.turb.sfc_and_turb call: tkea, kmh, kmv, khh, khv, nm, defv, defh,
    dissten, t11, t12, t13, t22, t23, t33.

    With sgsmodel=0 and bbc=3 (sfcmodel=1) ``turb`` holds uh, uf, vh, vf,
    mh, mf, zf, rf and the surface layer of cm1x.turb.sfc_and_turb_nosgs:
    ust, znt, t13, t23 (surface drag on uten1/vten1, surface-layer
    dissipative heating for idiss=1).

    ``thraten`` (radopt=2): the potential-temperature tendency of the last
    radiation call (cm1x.radiation_driver), added to thten1.

    ``tc`` (testcase >= 1 or lspgrad >= 2, cm1x.testcase): a dict with the
    forcing profiles of base.F (wprof .. vlspg), ``mtime`` (model time at the
    start of the step) and the base/grid arrays rho0, qv0, zh, zf, mh, ruh,
    ruf, rvh, rvf, xf, yf.  Adds lspgrad = 2, 3, 4, the testcase 10 nudging
    and testcase_simple_phys; the updated profiles (and frad, thraten, zir
    of testcase 4, 5) are returned under ``"tc"``.
    """
    _check(cfg)
    xp = bk.xp
    ni, nj, nk = g.ni, g.nj, g.nk
    n, nz = g.ngxy, g.ngz
    numq = len(qa)
    nqv = int(_opt(cfg, "nqv", 1)) - 1
    nqc = int(_opt(cfg, "nqc", 2)) - 1
    irdamp = int(_opt(cfg, "irdamp", 1))
    rdalpha = float(_opt(cfg, "rdalpha", 3.3333333333e-3))
    ptype = int(_opt(cfg, "ptype", 1))
    # cm1.F: tsmall/qsmall (1e-12/1e-24 with -DDP, see namelist.DP_DEFAULTS)
    tsmall = float(_opt(cfg, "tsmall", TSMALL))
    qsmall = float(_opt(cfg, "qsmall", QSMALL))

    uten1 = bk.zeros(g.u_shape) if uten1 is None else bk.copy(uten1)
    vten1 = bk.zeros(g.v_shape) if vten1 is None else bk.copy(vten1)
    wten1 = bk.zeros(g.w_shape) if wten1 is None else bk.copy(wten1)
    thten1 = bk.zeros(g.s_shape) if thten1 is None else bk.copy(thten1)
    ppten = bk.zeros(g.s_shape) if ppten is None else bk.copy(ppten)
    if qten is None:
        qten = [bk.zeros(g.s_shape) for _ in range(numq)]
    else:
        qten = [bk.copy(q) for q in qten]

    # cm1setup=0: no subgrid turbulence (idoles = .false.) whatever sgsmodel
    les = int(_opt(cfg, "sgsmodel", 0)) == 1 and int(_opt(cfg, "cm1setup", 1)) == 1
    # sgsmodel=0 with the semi-slip lower boundary (bbc=3, forced by
    # sfcmodel>=1): only the surface stress and surface-layer dissipation
    sfc = (not les) and int(_opt(cfg, "bbc", 1)) == 3
    if sfc:
        from .turb import dissheat, sfc_drag_tendencies

        if turb is None:
            raise ValueError("solve1: bbc=3 needs the turb dict (surface layer)")
        T = turb
    tketen = None
    if les:
        from .turb import dissheat, tke_source, turb_tendencies
        from .turbtend import turbt

        if turb is None:
            raise ValueError("solve1: sgsmodel=1 needs the turb dict")
        T = turb
        tketen = tke_source(
            bk,
            g,
            cfg,
            nm=T["nm"],
            defv=T["defv"],
            defh=T["defh"],
            dissten=T["dissten"],
            kmh=T["kmh"],
            kmv=T["kmv"],
            khv=T["khv"],
        )["tketen"]
        tketen = turbt(
            bk,
            g,
            cfg,
            dt=dt,
            uh=T["uh"],
            uf=T["uf"],
            vh=T["vh"],
            vf=T["vf"],
            mh=T["mh"],
            mf=T["mf"],
            rho=rho,
            rr=T["rr"],
            rf=T["rf"],
            t=T["tkea"],
            tten=tketen,
            kmh=T["kmh"],
            kmv=T["kmv"],
            terrain=T.get("terrain"),
        )["tten"]

    # Fortran k=1..nk, j=1..nj+1, i=1..ni+1
    K = slice(nz, nz + nk)
    Jp = slice(n, n + nj + 1)
    Ip = slice(n, n + ni + 1)
    Jm = slice(n - 1, n + nj)  # j-1
    Im = slice(n - 1, n + ni)  # i-1
    R = (K, Jp, Ip)

    # ---- Rayleigh damping -------------------------------------------------
    if irdamp == 1:
        uten1 = bk.set(
            uten1, R, -rdalpha * 0.5 * (tauh[K, Jp, Im] + tauh[R]) * (ua[R] - u0[R])
        )
        vten1 = bk.set(
            vten1, R, -rdalpha * 0.5 * (tauh[K, Jm, Ip] + tauh[R]) * (va[R] - v0[R])
        )
        wten1 = bk.set(wten1, R, -rdalpha * tauf[R] * wa[R])
        thten1 = bk.set(thten1, R, -rdalpha * taus[R] * tha[R])
    else:
        z = bk.zeros(ua[R].shape)
        uten1 = bk.set(uten1, R, z)
        vten1 = bk.set(vten1, R, z)
        wten1 = bk.set(wten1, R, z)
        thten1 = bk.set(thten1, R, z)

    # ---- moisture tendencies ----------------------------------------------
    S = (K, slice(n, n + nj), slice(n, n + ni))
    for m in range(numq):
        qten[m] = bk.set(qten[m], S, bk.zeros(qa[m][S].shape))

    # ---- large-scale pressure gradient (lspgrad=1) ---------------------------
    if int(_opt(cfg, "lspgrad", 0)) == 1:
        fcor = float(_opt(cfg, "fcor", 0.0))
        vmove = umove = 0.0  # imove=0
        Jq = slice(n + 1, n + nj + 2)  # j+1
        Iq = slice(n + 1, n + ni + 2)  # i+1
        vs = 0.25 * ((v0[K, Jp, Ip] + v0[K, Jq, Ip]) + (v0[K, Jp, Im] + v0[K, Jq, Im]))
        us = 0.25 * ((u0[K, Jp, Ip] + u0[K, Jp, Iq]) + (u0[K, Jm, Ip] + u0[K, Jm, Iq]))
        uten1 = bk.set(uten1, R, uten1[R] - fcor * (vs + vmove))
        vten1 = bk.set(vten1, R, vten1[R] + fcor * (us + umove))

    # ---- large-scale pressure gradient (lspgrad=2, 3, 4) ----------------------
    tcout = {}
    if int(_opt(cfg, "lspgrad", 0)) >= 2:
        from .testcase import lspgrad_tend

        uten1, vten1, ul, vl = lspgrad_tend(bk, g, cfg, tc, uten1, vten1)
        tcout.update(ulspg=ul, vlspg=vl)

    # ---- dissipative heating (sgsmodel=1, idiss=1) ---------------------------
    if (les or sfc) and int(_opt(cfg, "idiss", 1)) == 1:
        thten1 = dissheat(
            bk,
            g,
            cfg,
            pi0=pi0,
            ppi=ppi,
            qa=qa,
            dissten=T["dissten"] if les else None,
            thten1=thten1,
            ust=T.get("ust"),
            znt=T.get("znt"),
            zf=T.get("zf"),
        )["thten1"]

    # ---- radiation tendency (radopt=2) ----------------------------------------
    if thraten is not None:
        thten1 = bk.set(thten1, S, thten1[S] + thraten[S])

    # ---- subgrid turbulence (sgsmodel=1) ------------------------------------
    if les:
        tt = turb_tendencies(
            bk,
            g,
            cfg,
            dt=dt,
            uh=T["uh"],
            uf=T["uf"],
            vh=T["vh"],
            vf=T["vf"],
            mh=T["mh"],
            mf=T["mf"],
            rho=rho,
            rr=T["rr"],
            rf=T["rf"],
            th0=th0,
            ua=ua,
            va=va,
            wa=wa,
            tha=tha,
            qa=qa,
            uten1=uten1,
            vten1=vten1,
            wten1=wten1,
            thten1=thten1,
            qten=qten,
            kmh=T["kmh"],
            kmv=T["kmv"],
            khh=T["khh"],
            khv=T["khv"],
            t11=T["t11"],
            t12=T["t12"],
            t13=T["t13"],
            t22=T["t22"],
            t23=T["t23"],
            t33=T["t33"],
            terrain=T.get("terrain"),
        )
        uten1, vten1, wten1, thten1 = (
            tt["uten1"],
            tt["vten1"],
            tt["wten1"],
            tt["thten1"],
        )
        if tt["qten"] is not None:  # dry: no moisture tendencies
            qten = list(tt["qten"])
    elif sfc:
        sd = sfc_drag_tendencies(
            bk,
            g,
            cfg,
            dt=dt,
            uf=T["uf"],
            vh=T["vh"],
            uh=T["uh"],
            vf=T["vf"],
            mh=T["mh"],
            mf=T["mf"],
            rho=rho,
            rf=T["rf"],
            ua=ua,
            va=va,
            wa=wa,
            uten1=uten1,
            vten1=vten1,
            t13=T["t13"],
            t23=T["t23"],
        )
        uten1, vten1 = sd["uten1"], sd["vten1"]

    # ---- test cases: theta/qv nudging (testcase 10), testcase_simple_phys ----
    tcase = int(_opt(cfg, "testcase", 0))
    if tcase == 10:
        from .testcase import get_avg, tqnudge

        avg = get_avg(
            bk,
            g,
            cfg,
            th0=th0,
            ua=ua,
            va=va,
            tha=tha,
            qa=qa if int(_opt(cfg, "imoist", 1)) == 1 else None,
            **{k: tc[k] for k in ("ruh", "ruf", "rvh", "rvf", "xf", "yf")},
        )
        thten1, qten = tqnudge(
            bk,
            g,
            cfg,
            mtime=tc["mtime"],
            avg=avg,
            th0=th0,
            qv0=tc["qv0"],
            thten1=thten1,
            qten=qten,
        )
    if tcase >= 1:
        from .testcase import simple_phys

        sp = simple_phys(
            bk,
            g,
            cfg,
            prof=tc,
            th0=th0,
            ua=ua,
            va=va,
            tha=tha,
            qa=qa,
            uten1=uten1,
            vten1=vten1,
            thten1=thten1,
            qten=qten,
            **{
                k: tc[k]
                for k in (
                    "mtime",
                    "rho0",
                    "zh",
                    "zf",
                    "mh",
                    "ruh",
                    "ruf",
                    "rvh",
                    "rvf",
                    "xf",
                    "yf",
                )
            },
        )
        uten1, vten1, thten1, qten = (
            sp.pop("uten1"),
            sp.pop("vten1"),
            sp.pop("thten1"),
            sp.pop("qten"),
        )
        tcout.update(sp)

    # ---- pressure tendency for mass conservation (eqtset=2) ---------------
    rdt = 1.0 / dt
    tem = 0.0001 * tsmall
    th1 = thten1[S]
    thnew = tha[S] + dt * th1
    moist = int(_opt(cfg, "imoist", 1)) == 1
    if moist:
        qv1 = qten[nqv][S]
        qvnew = qa[nqv][S] + dt * qv1
        pinew = (
            rho[S] * (th0[S] + thnew) * (c.rd + c.rv * qvnew) * c.rp00
        ) ** c.rddcv - pi0[S]
        cond = (xp.abs(dt * th1) > tem) | (xp.abs(dt * qv1) > qsmall)
    else:
        pinew = (rho[S] * (th0[S] + thnew) * c.rd * c.rp00) ** c.rddcv - pi0[S]
        cond = xp.abs(dt * th1) > tem
    pt = xp.where(cond, (pinew - ppi[S]) * rdt, xp.zeros_like(pinew))
    ppten = bk.set(ppten, S, pt)

    if moist and ptype >= 1:
        # diabatic tendencies from last timestep as a good estimate
        ppten = bk.set(ppten, S, ppten[S] + qpten[S])
        thten1 = bk.set(thten1, S, thten1[S] + qtten[S])
        qten[nqv] = bk.set(qten[nqv], S, qten[nqv][S] + qvten[S])
        qten[nqc] = bk.set(qten[nqc], S, qten[nqc][S] + qcten[S])

    out = {
        "uten1": uten1,
        "vten1": vten1,
        "wten1": wten1,
        "thten1": thten1,
        "ppten": ppten,
        "qten": qten,
    }
    if les:
        out["tketen"] = tketen
    if tcout:
        out["tc"] = tcout
    return out
