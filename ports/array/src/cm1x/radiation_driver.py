"""Radiation driver, ported from src/radiation_driver.F (radopt=2, RRTMG).
The surface parameters it reads (tsk, albd, emiss, xland) come from
cm1x.init_surface.

Only the path CM1 takes for warm-rain (ptype=1, Kessler) microphysics is
ported: qv, qc, qr (no ice species), cal_cldfra1 cloud fraction, no PBL
subgrid clouds (ipbl=0), testcase /= 8, uniform latitude/longitude
(xlat = ctrlat, xlong = ctrlon everywhere, as radiation_driver sets them).

The time-dependent scalars (julian day, declination, solar constant,
cos(zenith)) depend only on the model time, so ``radconst`` and
``calc_coszen`` work on Python floats on the host; the compiled time step
receives coszen and solcon as 0-d arrays.
"""

from __future__ import annotations

import math

from . import constants as c
from . import module_ra_rrtmg_lw as lw
from . import module_ra_rrtmg_sw as sw

DPD = 360.0 / 365.0
DEGRAD = 3.1415926535897932384626433 / 180.0


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def _check(cfg):
    bad = []
    for name, ok in (
        ("radopt", (2,)),
        ("ptype", (1,)),
        ("testcase", (0,)),
        ("ipbl", (0,)),
        ("eqtset", (1, 2)),
    ):
        v = int(_opt(cfg, name, ok[0]))
        if v not in ok:
            bad.append(f"{name}={v}")
    if bad:
        raise NotImplementedError(
            "radiation_driver: options not ported: " + ", ".join(bad)
        )


# ---------------------------------------------------------------------------
def julday(year, month, day):
    """radtrns3d.F ``julday``: day of the year."""
    mndys = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    leap = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    return mndys[month - 1] + day + (1 if (month > 2 and leap) else 0)


def julian_day(cfg, mtime):
    """radiation_driver: julian = jday - 1 + hour/24 + minute/1440 +
    (second+mtime)/86400 (fractional day of the year at model time mtime)."""
    jday = julday(int(cfg.year), int(cfg.month), int(cfg.day))
    hour, minute, second = int(cfg.hour), int(cfg.minute), int(cfg.second)
    return (
        jday
        - 1.0
        + hour / 24.0
        + minute / (60.0 * 24.0)
        + (second + mtime) / (60.0 * 60.0 * 24.0)
    )


def radconst(*, xtime, julian, degrad=DEGRAD, dpd=DPD):
    """``radconst``: solar declination (rad) and solar constant (W m-2)."""
    del xtime  # unused by the Fortran too
    obecl = 23.5 * degrad
    sinob = math.sin(obecl)
    if julian >= 80.0:
        sxlong = dpd * (julian - 80.0)
    else:
        sxlong = dpd * (julian + 285.0)
    sxlong = sxlong * degrad
    declin = math.asin(sinob * math.sin(sxlong))
    djul = julian * 360.0 / 365.0
    rjul = djul * degrad
    eccfac = (
        1.000110
        + 0.034221 * math.cos(rjul)
        + 0.001280 * math.sin(rjul)
        + 0.000719 * math.cos(2 * rjul)
        + 0.000077 * math.sin(2 * rjul)
    )
    return {"declin": declin, "solcon": 1370.0 * eccfac}


def calc_coszen(*, julian, xtime, gmt, declin, xlon, xlat, degrad=DEGRAD):
    """``calc_coszen`` for one (xlat, xlon) point: cos(solar zenith angle)
    and hour angle (rad).  ``xtime`` in minutes since the start."""
    da = 6.2831853071795862 * (julian - 1) / 365.0
    eot = (
        0.000075
        + 0.001868 * math.cos(da)
        - 0.032077 * math.sin(da)
        - 0.014615 * math.cos(2 * da)
        - 0.04089 * math.sin(2 * da)
    ) * (229.18)
    xt24 = math.fmod(xtime, 1440.0) + eot
    tloctm = gmt + xt24 / 60.0 + xlon / 15.0
    hrang = 15.0 * (tloctm - 12.0) * degrad
    xxlat = xlat * degrad
    coszen = math.sin(xxlat) * math.sin(declin) + math.cos(xxlat) * math.cos(
        declin
    ) * math.cos(hrang)
    coszen = min(max(coszen, -1.0), 1.0)
    return {"coszen": coszen, "hrang": hrang}


def solar(cfg, mtime, dtrad):
    """The time-dependent part of radiation_driver: julian, xtime, declin,
    solcon, coszen and hrang at model time ``mtime`` (s), with coszen taken
    at the middle of the radiation interval (tfoo = mtime + dtrad/2)."""
    julian = julian_day(cfg, mtime)
    xtime = mtime / 60.0
    rc = radconst(xtime=xtime, julian=julian)
    tfoo = (mtime + 0.5 * dtrad) / 60.0
    cz = calc_coszen(
        julian=julian,
        xtime=tfoo,
        gmt=float(int(cfg.hour)),
        declin=rc["declin"],
        xlon=float(cfg.ctrlon),
        xlat=float(cfg.ctrlat),
    )
    return {"julian": julian, "xtime": xtime, **rc, **cz}


# ---------------------------------------------------------------------------
def cal_cldfra1(bk, *, qv, qc, t, p):
    """``cal_cldfra1`` with f_qc only (f_qi = f_qs = .false.): the cloud
    fraction of Xu and Randall (1996) from cloud water, with the saturation
    mixing ratio over ice where t <= 273.15 K.  Returns cldfra (the
    diagnostic cldfra1_flag is not computed)."""
    xp = bk.xp
    svpt0, r_d, r_v = 273.15, 287.04, 461.5
    ep_2 = r_d / r_v
    tc = t - svpt0
    esw = 1000.0 * 0.6112 * xp.exp(17.67 * tc / (t - 29.65))
    esi = 1000.0 * 0.6112 * xp.exp(21.8745584 * tc / (t - 7.66))
    qvsw = ep_2 * esw / (p - esw)
    qvsi = ep_2 * esi / (p - esi)
    qcld = qc
    noq = qcld < 1.0e-12
    weight = xp.where(noq | (t > 273.15), xp.zeros_like(t), xp.ones_like(t))
    qvs_weight = (1 - weight) * qvsw + weight * qvsi
    rhum = qv / qvs_weight
    sat = rhum >= 1.0
    subsat = xp.clip(1.0 * qvs_weight - qv, 1.0e-10, None)
    denom = subsat**0.49
    arg = xp.clip(-100.0 * qcld / denom, -6.9, None)
    rh = xp.clip(rhum, 1.0e-10, None)
    part = (rh / 1.0) ** 0.25 * (1.0 - xp.exp(arg))
    part = xp.where(part < 0.01, 0.0, part)
    cldfra = xp.where(noq, 0.0, xp.where(sat, 1.0, part))
    return {"cldfra": cldfra}


def p_top(base, nk):
    """init_physics.F: model-top pressure for rrtmg_lwinit (column (1,1))."""
    prs0 = base["prs0"]
    j = i = 3  # fortran (1, 1) with ngxy = 3
    return float(
        base["cgt1"] * prs0[nk, j, i]
        + base["cgt2"] * prs0[nk - 1, j, i]
        + base["cgt3"] * prs0[nk - 2, j, i]
    )


# ---------------------------------------------------------------------------
def radiation_driver(
    bk,
    g,
    cfg,
    *,
    coszen,
    solcon,
    nlayers,
    c1,
    c2,
    rmh,
    pi0,
    th0,
    prs,
    ppi,
    tha,
    qa,
    tsk,
    albd,
    emiss,
    xland,
    xice,
    xsnow,
    swtenc=None,
    cg,
    Tlw=None,
    Tsw=None,
):
    """``radiation_driver``, radopt=2 branch, for one radiation call.

    ``coszen`` and ``solcon`` come from ``solar`` (Python floats or 0-d
    arrays), ``nlayers`` from ``module_ra_rrtmg_lw.nlayers_lw``; ``cg`` holds
    cgs1..3, cgt1..3.  3-D arrays are [k, j, i] with halos (``qa``: list of
    numq arrays), surface arrays [j, i] with halos.  ``swtenc`` is the
    previous clear-sky shortwave tendency, kept where the sun is down.

    Returns swten, lwten, swtenc, lwtenc, cldfra, thraten (3-D, zero in the
    halos) and the 2-D diagnostics (glw, gsw, olr, lwcf, swcf, lwupt ...,
    swddir, swddni, swddif, coszr) on the interior [nj, ni]."""
    _check(cfg)
    xp = bk.xp
    nk, nj, ni = g.nk, g.nj, g.ni
    C = nj * ni
    K, J, I = g.K, g.J, g.I
    nqv, nqc, nqr = int(cfg.nqv), int(cfg.nqc), int(cfg.nqr)

    def col3(a, ks):
        return xp.reshape(a[ks, J, I], (-1, C))

    def col2(a):
        return xp.reshape(a[J, I], (C,))

    km = slice(0, nk)  # k-1 for k = 1..nk
    pi3d = col3(pi0, K) + col3(ppi, K)  # dum4
    t3d = (col3(th0, K) + col3(tha, K)) * pi3d  # dum3
    p3d = col3(prs, K)
    c1k, c2k = col3(c1, K), col3(c2, K)
    p8w = c1k * col3(prs, km) + c2k * p3d  # dum1
    # dum2 at k=2..nk (k=1 is replaced by tsk below)
    t8w_in = c1k[1:] * t3d[:-1] + c2k[1:] * t3d[1:]
    dz8w = float(cfg.dz) * col3(rmh, K)  # dum5
    tskc = col2(tsk)
    p8w_s = cg["cgs1"] * p3d[0] + cg["cgs2"] * p3d[1] + cg["cgs3"] * p3d[2]
    p8w_t = (
        cg["cgt1"] * p3d[nk - 1] + cg["cgt2"] * p3d[nk - 2] + cg["cgt3"] * p3d[nk - 3]
    )
    t8w_t = (
        cg["cgt1"] * t3d[nk - 1] + cg["cgt2"] * t3d[nk - 2] + cg["cgt3"] * t3d[nk - 3]
    )
    p8w = xp.concat([p8w_s[None], p8w[1:], p8w_t[None]], axis=0)
    t8w = xp.concat([tskc[None], t8w_in, t8w_t[None]], axis=0)

    qv = col3(qa[nqv - 1], K)
    qc = col3(qa[nqc - 1], K) if nqc else xp.zeros_like(qv)
    qr = col3(qa[nqr - 1], K) if nqr else xp.zeros_like(qv)

    cf = cal_cldfra1(bk, qv=qv, qc=qc, t=t3d, p=p3d)
    cldfra = cf["cldfra"]
    yr = int(cfg.year)
    surf = {
        "xland": col2(xland),
        "xice": col2(xice),
        "snow": col2(xsnow),
        "qv3d": qv,
        "qc3d": qc,
    }

    L = lw.RRTMG_LWRAD(
        bk,
        nlayers=nlayers,
        emiss=col2(emiss),
        p8w=p8w,
        p3d=p3d,
        tsk=tskc,
        t3d=t3d,
        t8w=t8w,
        cldfra3d=cldfra,
        qr3d=qr,
        yr=yr,
        g=c.g,
        T=Tlw,
        **surf,
    )
    S = sw.RRTMG_SWRAD(
        bk,
        coszen=xp.zeros_like(tskc) + coszen,
        solcon=solcon,
        albedo=col2(albd),
        t3d=t3d,
        t8w=t8w,
        tsk=tskc,
        p3d=p3d,
        p8w=p8w,
        dz8w=dz8w,
        cldfra3d=cldfra,
        qr3d=qr,
        yr=yr,
        g=c.g,
        swtenc=None if swtenc is None else col3(swtenc, K),
        T=Tsw,
        **surf,
    )

    lwten, lwtenc = L["rthratenlw"], L["lwtenc"]
    swten, swtenc_n = S["rthratensw"], S["swtenc"]
    eqtset = int(_opt(cfg, "eqtset", 2))
    if eqtset == 2:
        rcvm = c.cp / (
            c.cv
            + c.cvv * xp.clip(qv, 0.0, None)
            + c.cpl * xp.clip(qc + qr, 0.0, None)
            + c.cpi * 0.0
        )
        lwten, swten = lwten * rcvm, swten * rcvm
        lwtenc, swtenc_n = lwtenc * rcvm, swtenc_n * rcvm
        thraten = (swten + lwten) * c.cvdcp / pi3d
    else:
        thraten = (swten + lwten) / pi3d

    def full(a):
        z = bk.zeros(g.s_shape)
        return bk.set(z, (K, J, I), xp.reshape(a, (nk, nj, ni)))

    out = {
        "swten": full(swten),
        "lwten": full(lwten),
        "swtenc": full(swtenc_n),
        "lwtenc": full(lwtenc),
        "cldfra": full(cldfra),
        "thraten": full(thraten),
    }
    for k in (
        "glw",
        "olr",
        "lwcf",
        "lwupt",
        "lwuptc",
        "lwdnt",
        "lwdntc",
        "lwupb",
        "lwupbc",
        "lwdnb",
        "lwdnbc",
    ):
        out[k] = xp.reshape(L[k], (nj, ni))
    for k in (
        "gsw",
        "swcf",
        "swupt",
        "swuptc",
        "swdnt",
        "swdntc",
        "swupb",
        "swupbc",
        "swdnb",
        "swdnbc",
        "swddir",
        "swddni",
        "swddif",
        "coszr",
    ):
        out[k] = xp.reshape(S[k], (nj, ni))
    return out
