"""Surface-layer physics, ported from src/sfcphys.F: the neutral surface
layer of sfcmodel=1 (``getcecd`` with ch/cq, ``sfcflux``, ``sfcdiags``),
``gethpbl2`` (the bulk-Richardson boundary-layer depth of ipbl=2), and the
surface block of turb.F ``sfc_and_turb`` shared by every surface layer
(``sfc_prelude``: psfc, u1, v1, s1, t1, gethpbl2), plus ``surface_layer``,
the sfcmodel=1 hook of the cm1x.pbl registry.

``use_avg_sfc`` (testcase 1, 2, 9, 11, 14), set_ust with tbc=3, the qv
budget and the testcase flux switches are not ported.  2-D arrays are the
interior [nj, ni] unless noted.
"""

from __future__ import annotations

from . import constants as C
from .constants import karman

DCD1 = 1.0e-3
DCD2 = 2.4e-3
DWSP1 = 5.0
DWSP2 = 25.0
DFAC = (DCD2 - DCD1) / (DWSP2 - DWSP1)

# getcecd stops CM1 when the water iteration needs more than 20 passes
NITER = 20


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def getcecd(bk, cfg, *, u1, v1, s1, za, xland, znt):  # noqa: C901
    """``getcecd`` on 2-D arrays (any shape; CM1 uses i=1..ni, j=1..nj):
    lowest-level winds u1, v1, speed s1 at height za, land/water flag xland
    and the previous roughness length znt.  Returns cd, znt, ust, u10, v10,
    s10.

    Options: set_znt=1 (given z0), set_ust=1 (given u*), cecd=1 (given Cd),
    else Cd(s10) over water (cecd=2 Deacon, cecd=3 Fairall/Donelan), iterated
    to |ds10| <= 0.001 m/s as a fixed number of masked passes (CM1 stops if
    more than 20 are needed; here the 20th pass is kept), and z0 given over
    land."""
    if int(_opt(cfg, "testcase", 0)) in (1, 2, 9, 11, 14):
        raise NotImplementedError("getcecd: use_avg_sfc is not ported")
    xp = bk.xp
    log, exp, sqrt = xp.log, xp.exp, xp.sqrt
    if int(_opt(cfg, "set_znt", 0)) == 1:
        rznt = 1.0 / znt
        cd = (karman / log(10.0 * rznt)) ** 2
        var = log(10.0 * rznt) / log(za * rznt)
        s10 = s1 * var
        ust = xp.clip(s1 * karman / log(za * rznt), 1.0e-6, None)
    elif int(_opt(cfg, "set_ust", 0)) == 1:
        ust = xp.zeros_like(s1) + float(cfg.cnst_ust)
        znt = za / (
            exp(xp.clip(s1, 1.0e-10, None) * karman / xp.clip(ust, 1.0e-10, None)) - 1.0
        )
        rznt = 1.0 / znt
        cd = (karman / log((10.0 + znt) * rznt)) ** 2
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
    elif int(_opt(cfg, "cecd", 3)) == 1:
        cd = xp.zeros_like(s1) + max(1.0e-10, float(cfg.cnstcd))
        znt = 10.0 / (exp(karman / sqrt(cd)) - 1.0)
        rznt = 1.0 / znt
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
        ust = xp.clip(s1 * karman / log((za + znt) * rznt), 1.0e-6, None)
    else:
        cecd = int(_opt(cfg, "cecd", 3))
        water = xland > 1.5
        # land (and the first guess over water): z0 as given
        rznt = 1.0 / znt
        cd_land = (karman / log((10.0 + znt) * rznt)) ** 2
        var = log((10.0 + znt) * rznt) / log((za + znt) * rznt)
        s10 = s1 * var
        cd = cd_land
        wlast = xp.zeros_like(s1) - 1000.0
        for _ in range(NITER):
            active = water & (xp.abs(s10 - wlast) > 0.001)
            if cecd == 2:
                cdn = 1.1e-3 + (4.0e-5 * s10)
            else:
                cdn = DCD1 + (s10 - DWSP1) * DFAC
                cdn = xp.clip(xp.clip(cdn, None, DCD2), DCD1, None)
            zn = xp.clip(10.0 / (exp(karman / sqrt(cdn)) - 1.0), 1.0e-20, None)
            rz = 1.0 / zn
            vn = log((10.0 + zn) * rz) / log((za + zn) * rz)
            wlast = xp.where(active, s10, wlast)
            cd = xp.where(active, cdn, cd)
            znt = xp.where(active, zn, znt)
            rznt = xp.where(active, rz, rznt)
            var = xp.where(active, vn, var)
            s10 = xp.where(active, s1 * vn, s10)
        ust = xp.clip(s1 * karman / log((za + znt) * rznt), 1.0e-6, None)
    out = {
        "cd": cd,
        "znt": znt,
        "ust": ust,
        "u10": u1 * var,
        "v10": v1 * var,
        "s10": s10,
    }
    if int(_opt(cfg, "isfcflx", 0)) == 1:
        # ch, cq: over water from cecd, over land = cd
        cecd = int(_opt(cfg, "cecd", 3))
        if cecd == 1:
            cw = xp.zeros_like(s1) + float(_opt(cfg, "cnstce", 0.0011))
        elif cecd == 2:
            cw = 1.1e-3 + (4.0e-5 * s10)
        else:
            cw = xp.zeros_like(s1) + 1.20e-3
        ch = xp.where(xland > 1.5, cw, cd)
        out["ch"] = ch
        out["cq"] = ch
    return out


def rslf(xp, p, t, ptype):
    """cm1libs ``rslf``: saturation mixing ratio over liquid."""
    if ptype == 4:
        return 380.00 * xp.exp(17.2693882 - 4097.8531 / (t - 35.86)) / p
    esl = 611.2 * xp.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = xp.minimum(esl, p * 0.5)
    return C.eps * esl / (p - esl)


def _ptype(cfg):
    imoist = int(_opt(cfg, "imoist", 0))
    return int(_opt(cfg, "ptype", 1 if imoist else 0))


def sfcflux(
    bk, cfg, *, psfc, ch, cq, s10, tsk, th0, tha, qv1, mavail, thflux, qvflux, qsfc
):
    """sfcphys.F ``sfcflux`` (testcase=0): surface fluxes of theta and qv
    (``th0``, ``tha``, ``qv1`` = qv at k=1; thflux/qvflux/qsfc are the
    previous values, kept where CM1 does not set them).  set_flx=1 gives
    cnst_shflx / cnst_lhflx.  Returns thflux, qvflux, qsfc."""
    xp = bk.xp
    moist = int(_opt(cfg, "imoist", 0)) == 1
    if int(_opt(cfg, "set_flx", 0)) == 1:
        thflux = xp.zeros_like(psfc) + float(_opt(cfg, "cnst_shflx", 0.24))
        if moist:
            qvflux = xp.zeros_like(psfc) + float(_opt(cfg, "cnst_lhflx", 5.2e-5))
        return {"thflux": thflux, "qvflux": qvflux, "qsfc": qsfc}
    pisfc = (psfc * C.rp00) ** C.rovcp
    thflux = ch * s10 * ((tsk / pisfc - th0) - tha)
    if moist:
        qvsat = rslf(xp, psfc, tsk, _ptype(cfg))
        qsfc = qvsat
        qvflux = cq * s10 * (qvsat - qv1) * mavail
    return {"thflux": thflux, "qvflux": qvflux, "qsfc": qsfc}


def sfcdiags(
    bk, cfg, *, tsk, thflux, qvflux, cd, s1, psfc, znt, hpbl, dsxy, za, th1, qv1,
    rf1,
):  # fmt: skip
    """sfcphys.F ``sfcdiags`` (sfcmodel=1: neutral surface layer): qsfc,
    gz1oz0, wspd (with the convective velocity and the Mahrt-Sun
    correction), br, hfx, qfx, cda, psim = psih = zol = mol = 0, fm = fh =
    gz1oz0, and the 2-m th2, q2, t2.  ``qv1`` is solve1's divx (qv or 0),
    ``rf1`` = rf at k=1, ``za`` the height of the lowest level."""
    xp = bk.xp
    log, sqrt = xp.log, xp.sqrt
    ep1 = C.rv / C.rd - 1.0
    vconvc = 1.0
    pisfc = (psfc * C.rp00) ** C.rovcp
    thgb = tsk / pisfc
    thvx = th1 * (1.0 + ep1 * qv1)
    qsfc = rslf(xp, psfc, tsk, _ptype(cfg))
    tskv = thgb * (1.0 + ep1 * qsfc)
    govrth = C.g / th1
    rznt = 1.0 / znt
    gz1oz0 = log((za + znt) * rznt)
    dthvdz = thvx - tskv
    z = xp.zeros_like(psfc)
    fluxc = xp.maximum(thflux + ep1 * tskv * qvflux, z)
    vconv = vconvc * (C.g / tsk * hpbl * fluxc) ** 0.33
    vsgd = 0.32 * (xp.maximum(dsxy / 5000.0 - 1.0, z)) ** 0.33
    wspd = sqrt(s1 * s1 + vconv * vconv + vsgd * vsgd)
    wspd = xp.maximum(z + 0.1, wspd)
    br = govrth * za * dthvdz / (wspd * wspd)
    val = log((2.0 + znt) * rznt) / log((za + znt) * rznt)
    th2 = thgb + (th1 - thgb) * val
    return {
        "qsfc": qsfc,
        "gz1oz0": gz1oz0,
        "wspd": wspd,
        "br": br,
        "hfx": thflux * C.cp * rf1,
        "qfx": qvflux * rf1,
        "cda": cd,
        "psim": z,
        "psih": z,
        "zol": z,
        "mol": z,
        "fm": gz1oz0,
        "fh": gz1oz0,
        "th2": th2,
        "q2": qsfc + (qv1 - qsfc) * val,
        "t2": th2 * pisfc,
    }


BRIC = 0.125  # gethpbl2: critical bulk Richardson number


def gethpbl2(bk, cfg, *, psfc, tsk, thflux, qvflux, ust, hpbl, zh, th, qv, ugr, vgr):
    """sfcphys.F ``gethpbl2`` (ipbl=2): qsfc (rslf, or 0 when imoist=0) and
    the boundary-layer depth hpbl from the bulk Richardson number (two
    passes; the second adds the YSU-style thermal excess from the previous
    ust, thflux, qvflux and the first pass's hpbl).  Columns: 2-D [nj, ni];
    ``zh``, ``th`` (th0 + tha), ``qv`` (divx) are interior [nk, nj, ni];
    ``ugr``/``vgr`` are interior [nk, nj, ni+1] / [nk, nj+1, ni] (u, v
    points).  Returns {"qsfc", "hpbl"} (hpbl = -1 where no crossing).

    CM1 stops the level search once ``ntot`` (the number of crossings found
    so far in this pass, counted over all columns) equals ni*nj; that is a
    global count, so a running count and a "done" flag are carried through
    the levels."""
    xp = bk.xp
    nk, nj, ni = th.shape
    moist = int(_opt(cfg, "imoist", 0)) == 1
    if moist:
        qsfc = rslf(xp, psfc, tsk, _ptype(cfg))
    else:
        qsfc = xp.zeros_like(psfc)
    rep = C.repsm1
    thv0 = th[0] * (1.0 + rep * qv[0])
    govthv = C.g / thv0
    one = xp.zeros_like(psfc) + 1.0
    zero = xp.zeros_like(psfc)

    def uv2(k):
        uavg = 0.5 * (ugr[k, :, :ni] + ugr[k, :, 1:])
        vavg = 0.5 * (vgr[k, :nj, :] + vgr[k, 1:, :])
        return xp.maximum(one, uavg * uavg + vavg * vavg)

    thvsfc = tsk * ((C.p00 / psfc) ** C.rovcp) * (1.0 + rep * qsfc)
    total = ni * nj
    for nloop in (1, 2):
        thv = th[0] * (1.0 + rep * qv[0])
        if nloop == 1:
            thv1 = thv
        else:
            thf1 = 1.0 + rep * qv[0]
            thf2 = rep * th[0]
            thvflux = xp.maximum(zero, thf1 * thflux + thf2 * qvflux)
            ws = ((ust * ust) * ust + 8.0 * karman * 0.5 * govthv * thvflux * hpbl) ** (
                0.3333333
            )
            ws = xp.minimum(ws, ust * 16.0)
            ws = xp.maximum(ws, ust / 5.0)
            ws = xp.maximum(ws, zero + 0.0001)
            gamfac = 6.8 / ws
            tmpprt = xp.maximum(
                zero,
                xp.minimum(gamfac * thflux, zero + 3.0) * thf1
                + xp.minimum(gamfac * qvflux, zero + 2.0e-3) * thf2,
            )
            thv1 = thv + tmpprt
        brilast = govthv * zh[0] * (thv - thvsfc) / uv2(0)
        hpbl = zero - 1.0
        ntot = xp.sum(zero)
        done = ntot > 1.0  # .false.
        for k in range(1, nk):
            thvk = th[k] * (1.0 + rep * qv[k])
            bri = govthv * zh[k] * (thvk - thv1) / uv2(k)
            act = (hpbl < 0.0) & ~done
            hit = act & (bri > BRIC)
            hnew = xp.where(
                xp.abs(bri - brilast) < 0.000001,
                zh[k],
                zh[k - 1] + (zh[k] - zh[k - 1]) * (BRIC - brilast) / (bri - brilast),
            )
            hpbl = xp.where(hit, hnew, hpbl)
            brilast = xp.where(act, bri, brilast)
            ntot = ntot + xp.sum(xp.where(hit, one, zero))
            done = done | (ntot == total)
    return {"qsfc": qsfc, "hpbl": hpbl}


def surface_layer(ctx):
    """The sfcmodel=1 surface layer (cm1x.pbl registry hook; turb.F
    sfc_and_turb, after ``pbl.sfc_prelude``): getcecd, sfcflux (isfcflx=1),
    sfcdiags; with a PBL scheme other than ipbl=2, hfx/qfx from the moist
    cp.  Returns the state updates (2-D arrays with halos)."""
    from .pbl import interior2, put2

    bk, g, cfg, b, s = ctx.bk, ctx.g, ctx.cfg, ctx.b, ctx.s
    xp = bk.xp
    if int(_opt(cfg, "tbc", 1)) == 3:
        raise NotImplementedError("sfcmodel=1: tbc=3 is not ported")
    if _opt(cfg, "terrain_flag", False):
        raise NotImplementedError("sfcmodel=1 surface layer: terrain is not ported")
    I2 = lambda k: interior2(g, s[k])  # noqa: E731
    n, z = g.ngxy, g.ngz
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    za = b["zh"][z, J, I]
    th01, tha1 = b["th0"][z, J, I], ctx.th3d[z, J, I]
    th1 = th01 + tha1
    nqv = int(_opt(cfg, "nqv", 0))
    moist = int(_opt(cfg, "imoist", 0)) == 1 and nqv >= 1
    qv1 = ctx.q3d[nqv - 1][z, J, I] if moist else xp.zeros_like(za)
    rf1 = ctx.rf[z, J, I]
    r = getcecd(
        bk, cfg, u1=I2("u1"), v1=I2("v1"), s1=I2("s1"), za=za, xland=I2("xland"),
        znt=I2("znt"),
    )  # fmt: skip
    out = dict(r)
    for k in ("ch", "cq", "thflux", "qvflux", "qsfc"):
        out.setdefault(k, I2(k))
    if int(_opt(cfg, "isfcflx", 0)) == 1:
        out.update(
            sfcflux(
                bk,
                cfg,
                psfc=I2("psfc"),
                ch=out["ch"],
                cq=out["cq"],
                s10=out["s10"],
                tsk=I2("tsk"),
                th0=th01,
                tha=tha1,
                qv1=qv1,
                mavail=I2("mavail"),
                thflux=out["thflux"],
                qvflux=out["qvflux"],
                qsfc=out["qsfc"],
            )  # fmt: skip
        )
    out.update(
        sfcdiags(
            bk,
            cfg,
            tsk=I2("tsk"),
            thflux=out["thflux"],
            qvflux=out["qvflux"],
            cd=out["cd"],
            s1=I2("s1"),
            psfc=I2("psfc"),
            znt=out["znt"],
            hpbl=I2("hpbl"),
            dsxy=I2("dsxy"),
            za=za,
            th1=th1,
            qv1=qv1,
            rf1=rf1,
        )  # fmt: skip
    )
    if bool(_opt(cfg, "use_pbl", False)) and int(_opt(cfg, "ipbl", 0)) != 2:
        cpmm = C.cp * (1.0 + 0.8 * qv1)
        out["cpmm"] = cpmm
        out["hfx"] = out["thflux"] * cpmm * rf1
        out["qfx"] = out["qvflux"] * rf1
    return {k: put2(bk, g, s[k], v) for k, v in out.items()}
