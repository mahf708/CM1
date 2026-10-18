"""The Mellor-Yamada-Janjic surface layer (sfcmodel=7), ported from
src/module_sf_myjsfc.F (``myjsfcinit``, ``myjsfc``, ``sfcdif``) and its CM1
glue in src/turb.F (``sfc_myj``: sfc_and_turb's sfcmodel=7 block).

As CM1 runs it: LOWLYR = 1 (so PSFC = PINT(1) and LMH = KTE), ISURBAN = 0,
IVGTYP / IZ0TLND unused, the NMM_CORE branches compiled out (NTSD == 1 is
the first step), ``isftcflx`` present (/= 0: the high-wind roughness over
water).  Every column at once: 3-D inputs are ``[k, ...]`` arrays whose
leading index 0 is Fortran k = 1 (``pint`` has nk+1 levels), 2-D inputs
have the trailing (column) shape.  The scheme counts levels downward from
the model top (K = KTE+1-k); the column work arrays here are Python lists
indexed by that K (entry 0 unused) so the code reads side by side with the
Fortran (and with cxx/src/cm1k/physics/myj/cm1k_myj.cpp).

Over land CM1 never sets ``zntmyj`` (it stays 0), so the land roughness is
zero and the Fortran run goes NaN; the land code path is ported as written
(and tested with positive roughness) but whole-model runs are over water.
"""

from __future__ import annotations

import contextlib
import math

# module_model_constants (src/module_ra_etc.F)
r_d = 287.04
cp = 1005.7
g = 9.81
r_v = 461.5
xls = 2.834e6
xlv = 2.501e6
rvovrd = r_v / r_d
pq0 = 379.90516
epsq2 = 0.2
a2 = 17.2693882
a3 = 273.16
a4 = 35.86
p608 = rvovrd - 1.0
ep_1 = r_v / r_d - 1.0
p1000mb = 100000.0

# module_sf_myjsfc parameters
ITRMX = 5
vkarman = 0.4
capa = r_d / cp
elocp = 2.72e6 / cp
rcap = 1.0 / capa
gocp02 = g / cp * 2.0
gocp10 = g / cp * 10.0
epsu2 = 1.0e-6
epsust = 1.0e-9
epszt = 1.0e-28
a2s = 17.2693882
a3s = 273.16
a4s = 35.86
seafc = 0.98
pq0sea = pq0 * seafc
beta = 1.0 / 273.0
excml = 0.0001
excms = 0.0001
glkbr = 10.0
glkbs = 30.0
qvisc = 2.1e-5
ric = 0.505
small = 0.35
sqpr = 0.84
sqsc = 0.84
sqvisc = 258.2
tvisc = 2.1e-5
ustc = 0.7
ustr = 0.225
visc = 1.5e-5
fh = 1.01
wwst = 1.2
ztfc = 1.0
btg = beta * g
cziv = small * glkbs
grrs = glkbr / glkbs
rtvisc = 1.0 / tvisc
rvisc = 1.0 / visc
zqrzt = sqsc / sqpr
fzq1 = rtvisc * qvisc * zqrzt
fzq2 = rtvisc * qvisc * zqrzt
fzt1 = rvisc * tvisc * sqpr
fzt2 = cziv * grrs * tvisc * sqpr
fzu1 = cziv * visc
rqvisc = 1.0 / qvisc
ustfc = 0.018 / g
wwst2 = wwst * wwst
ozo = 1.59e-5
KZTM = 10001
KZTM2 = KZTM - 2


def _max(xp, a, b):
    """Fortran MAX(a, b) as gfortran evaluates it (a > b ? a : b)."""
    return xp.where(a > b, a, b)


def _min(xp, a, b):
    return xp.where(a < b, a, b)


def _sel(xp, c, a, b):
    """``a if c else b`` for a Python bool or an array condition."""
    if isinstance(c, bool):
        return a if c else b
    return xp.where(c, a, b)


_TABLES: dict = {}


def _host_tables():
    """MYJSFCINIT's psi tables and scalars, computed with libm (Python's
    ``math``) in the Fortran's order: the zeta recurrences are serial."""
    if "host" in _TABLES:
        return _TABLES["host"]
    pihf = 3.1415926 / 2.0
    eps = 1.0e-6
    ztmin1 = -5.0
    ztmax1 = 1.0
    ztmin2 = -5.0
    ztmax2 = 1.0
    dzeta1 = (ztmax1 - ztmin1) / (KZTM - 1)
    dzeta2 = (ztmax2 - ztmin2) / (KZTM - 1)
    m1, h1, m2, h2 = [], [], [], []
    zeta1, zeta2 = ztmin1, ztmin2

    def psi(zeta):
        if zeta < 0.0:
            # Paulson 1970
            x = math.sqrt(math.sqrt(1.0 - 16.0 * zeta))
            pm = (
                -2.0 * math.log((x + 1.0) / 2.0)
                - math.log((x * x + 1.0) / 2.0)
                + 2.0 * math.atan(x)
                - pihf
            )
            ph = -2.0 * math.log((x * x + 1.0) / 2.0)
        else:
            # Holtslag and de Bruin 1988
            pm = 0.7 * zeta + 0.75 * zeta * (6.0 - 0.35 * zeta) * math.exp(-0.35 * zeta)
            ph = pm
        return pm, ph

    for k in range(1, KZTM + 1):
        pm, ph = psi(zeta1)
        m1.append(pm)
        h1.append(ph)
        pm, ph = psi(zeta2)
        m2.append(pm)
        h2.append(ph)
        if k == KZTM:
            ztmax1 = zeta1
            ztmax2 = zeta2
        zeta1 = zeta1 + dzeta1
        zeta2 = zeta2 + dzeta2
    t = {
        "psim1": m1,
        "psih1": h1,
        "psim2": m2,
        "psih2": h2,
        "fh01": 1.0,
        "fh02": 1.0,
        "ztmin1": ztmin1,
        "ztmin2": ztmin2,
        "ztmax1": ztmax1 - eps,
        "ztmax2": ztmax2 - eps,
        "dzeta1": dzeta1,
        "dzeta2": dzeta2,
    }
    _TABLES["host"] = t
    return t


def myjsfcinit(bk):
    """MYJSFCINIT's integral functions: the psi tables PSIM1/PSIH1 (water),
    PSIM2/PSIH2 (land), as backend arrays (index K-1 holds PSI(K)), and
    DZETA*, FH0*, ZTMIN*, ZTMAX*.  (USTAR = 0.1 and LOWLYR = 1 are set by
    the caller: ``init_myj``.)"""
    key = (bk.name, str(bk.dtype), str(bk.device))
    if key not in _TABLES:
        h = _host_tables()
        t = dict(h)
        if bk.name == "jax":
            # concrete arrays even when first called inside a jit trace
            import jax

            ctx = jax.ensure_compile_time_eval()
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            for n in ("psim1", "psih1", "psim2", "psih2"):
                t[n] = bk.asarray(h[n])
        _TABLES[key] = t
    return _TABLES[key]


def _lookup(bk, tab, zeta, ztmin, dzeta):
    """RZ=(ZETA-ZTMIN)/DZETA; K=INT(RZ); RDZT=RZ-REAL(K); K=MIN(K,KZTM2);
    K=MAX(K,0); PSI=(TAB(K+2)-TAB(K+1))*RDZT+TAB(K+1)."""
    xp = bk.xp
    rz = (zeta - ztmin) / dzeta
    kf = xp.trunc(rz)
    rdzt = rz - kf
    k = xp.astype(xp.clip(kf, 0.0, float(KZTM2)), xp.int64)
    kk = xp.reshape(k, (-1,))
    t0 = xp.reshape(xp.take(tab, kk, axis=0), k.shape)
    t1 = xp.reshape(xp.take(tab, kk + 1, axis=0), k.shape)
    return (t1 - t0) * rdzt + t0


def sfcdif(
    bk,
    tab,
    *,
    ntsd,
    seamask,
    ths,
    qs,
    psfc,
    uz0,
    vz0,
    tz0,
    tsk,
    thz0,
    qz0,
    ustar,
    z0,
    z0base,
    akms,
    akhs,
    pblh,
    wetm,
    ulow,
    vlow,
    tlow,
    thlow,
    thelow,
    qlow,
    cwmlow,
    zsl,
    plow,
    rhosfc,
    isftcflx=0,
):
    """SFCDIF on 2-D (column) arrays: the surface exchange coefficients
    (water: viscous sublayer of Janjic 1994, optionally the isftcflx /= 0
    roughness; land: Zilitinkevich zt) iterated ITRMX times, then the 2-m
    and 10-m diagnostics.  ``ntsd`` is a Python int or a 0-d array.
    Returns the INOUT and OUT arguments by their Fortran names (lower
    case; ``rlmo`` is RMOL)."""
    xp = bk.xp
    sqrt, log, exp = xp.sqrt, xp.log, xp.exp
    first = ntsd == 1
    later = ntsd > 1
    psm1, psh1, psm2, psh2 = tab["psim1"], tab["psih1"], tab["psim2"], tab["psih2"]
    ztmin1, ztmax1, dzeta1, fh01 = (
        tab["ztmin1"],
        tab["ztmax1"],
        tab["dzeta1"],
        tab["fh01"],
    )
    ztmin2, ztmax2, dzeta2, fh02 = (
        tab["ztmin2"],
        tab["ztmax2"],
        tab["dzeta2"],
        tab["fh02"],
    )

    rdz = 1.0 / zsl
    cxchl = excml * rdz
    cxchs = excms * rdz
    btgx = g / thlow
    elfc = vkarman * btgx
    btgh = xp.where(pblh > 1000.0, btgx * pblh, btgx * 1000.0)
    sea = seamask > 0.5

    def clamp(z, lo, hi):
        return _min(xp, _max(xp, z, lo), hi)

    def dthv_of(thz0_, qz0_):
        tem = (tlow + tz0) * 0.5
        thm = (thelow + thz0_) * 0.5
        a = thm * p608
        b = (elocp / tem - 1.0 - p608) * thm
        return (
            (thelow - thz0_) * ((qlow + qz0_ + cwmlow) * (0.5 * p608) + 1.0)
            + (qlow - qz0_ + cwmlow) * a
            + cwmlow * b
        )

    # ------------------------------------------------------------ sea points
    s_qs, s_uz0, s_vz0, s_thz0, s_qz0 = qs, uz0, vz0, thz0, qz0
    s_ust, s_z0, s_akms, s_akhs = ustar, z0, akms, akhs
    # (positive placeholders where the branch is not taken keep the
    # unselected arithmetic finite)
    s_ust = xp.where(sea, s_ust, 0.1)
    for _ in range(ITRMX):
        z0s = _max(xp, ustfc * s_ust * s_ust, 1.59e-5)
        if int(isftcflx) != 0:
            # GHB, 200621: level-off roughness length at high wind speeds
            zw = _min(xp, (s_ust / 1.06) ** 0.3, 1.0)
            zn1 = 0.011 * s_ust * s_ust / g + ozo
            zn2 = 10.0 * exp(-9.5 * s_ust ** (-0.3333)) + 0.11 * 1.5e-5 / _max(
                xp, s_ust, 0.01
            )
            z0s = (1.0 - zw) * zn1 + zw * zn2
            z0s = _min(xp, z0s, 2.85e-3)
            z0s = _max(xp, z0s, 1.59e-5)
        s_z0 = z0s
        lo = s_ust < ustr  # viscous sublayer
        hi = s_ust >= ustc
        # ustar < ustr
        lo1 = lo & first
        s_akms = xp.where(lo1, cxchs, s_akms)
        s_akhs = xp.where(lo1, cxchs, s_akhs)
        s_qs = xp.where(lo1, qlow, s_qs)
        sq4 = sqrt(sqrt(z0s * s_ust * rvisc))
        zu_lo = fzu1 * sq4 / s_ust
        wght = s_akms * zu_lo * rvisc
        rwgh = wght / (wght + 1.0)
        uz0_lo = (ulow * rwgh + s_uz0) * 0.5
        vz0_lo = (vlow * rwgh + s_vz0) * 0.5
        zt_lo = fzt1 * zu_lo
        zq_lo = fzq1 * zt_lo
        zt_mid = fzt2 * sq4 / s_ust
        zq_mid = fzq2 * zt_mid
        zt_lm = xp.where(lo, zt_lo, zt_mid)
        zq_lm = xp.where(lo, zq_lo, zq_mid)
        wghtt = s_akhs * zt_lm * rtvisc
        wghtq = s_akhs * zq_lm * rqvisc
        thz0_lm = _sel(
            xp,
            later,
            ((wghtt * thlow + ths) / (wghtt + 1.0) + s_thz0) * 0.5,
            (wghtt * thlow + ths) / (wghtt + 1.0),
        )
        qz0_lm = _sel(
            xp,
            later,
            ((wghtq * qlow + s_qs) / (wghtq + 1.0) + s_qz0) * 0.5,
            (wghtq * qlow + s_qs) / (wghtq + 1.0),
        )
        # ustar >= ustc
        qs_hi = pq0sea / psfc * exp(a2s * (tsk - a3s) / (tsk - a4s))
        zu = xp.where(lo, zu_lo, z0s)
        zt = xp.where(hi, z0s, zt_lm)
        s_uz0 = xp.where(lo, uz0_lo, 0.0)
        s_vz0 = xp.where(lo, vz0_lo, 0.0)
        s_thz0 = xp.where(hi, ths, thz0_lm)
        s_qs = xp.where(hi, qs_hi, s_qs)
        s_qz0 = xp.where(hi, qs_hi, qz0_lm)
        dthv_s = dthv_of(s_thz0, s_qz0)
        du2_s = _max(
            xp, (ulow - s_uz0) * (ulow - s_uz0) + (vlow - s_vz0) * (vlow - s_vz0), epsu2
        )
        rib_s = btgx * dthv_s * zsl / du2_s
        # turbulent branch
        zslu = zsl + zu
        zslt = zsl + zt
        rzsu = zslu / zu
        rzst = zslt / zt
        rlogu = log(rzsu)
        rlogt = log(rzst)
        rlmo_s = elfc * s_akhs * dthv_s / ((s_ust * s_ust) * s_ust)
        zetalu = clamp(zslu * rlmo_s, ztmin1, ztmax1)
        zetalt = clamp(zslt * rlmo_s, ztmin1, ztmax1)
        zetau = clamp(zu * rlmo_s, ztmin1 / rzsu, ztmax1 / rzsu)
        zetat = clamp(zt * rlmo_s, ztmin1 / rzst, ztmax1 / rzst)
        psmz_s = _lookup(bk, psm1, zetau, ztmin1, dzeta1)
        psmzl = _lookup(bk, psm1, zetalu, ztmin1, dzeta1)
        psim_s = -psmzl
        simm = psmzl - psmz_s + rlogu
        fm_s = psmzl + rlogu
        pshz_s = _lookup(bk, psh1, zetat, ztmin1, dzeta1)
        pshzl = _lookup(bk, psh1, zetalt, ztmin1, dzeta1)
        psih_s = -pshzl
        simh = (pshzl - pshz_s + rlogt) * fh01
        fhh_s = (pshzl + rlogt) * fh01
        ustark_s = s_ust * vkarman
        s_akms = _max(xp, ustark_s / simm, cxchs)
        s_akhs = _max(xp, ustark_s / simh, cxchs)
        # Beljaars correction for ustar
        wstar2 = xp.where(
            dthv_s <= 0.0,
            wwst2 * xp.abs(btgh * s_akhs * dthv_s) ** (2.0 / 3.0),
            0.0,
        )
        s_ust = _max(xp, sqrt(s_akms * sqrt(du2_s + wstar2)), epsust)
    s_zu, s_zt = zu, zt

    # ----------------------------------------------------------- land points
    land = ~sea
    l_z0 = xp.where(land, z0, 1.0)
    l_z0base = xp.where(land, z0base, 1.0)
    l_ust = xp.where(land, ustar, 0.1)
    l_akhs = akhs
    l_akms = akms
    l_qs = _sel(xp, first, qlow, qs)
    l_zu = l_z0
    l_thz0 = ths
    l_qz0 = l_qs
    dthv_l = dthv_of(l_thz0, l_qz0)
    du2_l = _max(xp, ulow * ulow + vlow * vlow, epsu2)
    rib_l = btgx * dthv_l * zsl / du2_l
    zslu_l = zsl + l_zu
    rzsu_l = zslu_l / l_zu
    rlogu_l = log(rzsu_l)
    zslt_l = zsl + l_zu  # u, v and t are at the same level
    czil = 0.1
    zilfc = -czil * vkarman * sqvisc
    czetmax = 10.0
    zzil = xp.where(
        dthv_l > 0.0,
        xp.where(
            rib_l < ric,
            zilfc * (1.0 + (rib_l / ric) * (rib_l / ric) * czetmax),
            zilfc * (1.0 + czetmax),
        ),
        zilfc,
    )
    for _ in range(ITRMX):
        # Zilitinkevitch fix for zt
        l_zt = _max(xp, exp(zzil * sqrt(l_ust * l_z0base)) * l_z0base, epszt)
        rzst_l = zslt_l / l_zt
        rlogt_l = log(rzst_l)
        rlmo_l = elfc * l_akhs * dthv_l / ((l_ust * l_ust) * l_ust)
        zetalu = clamp(zslu_l * rlmo_l, ztmin2, ztmax2)
        zetalt = clamp(zslt_l * rlmo_l, ztmin2, ztmax2)
        zetau = clamp(l_zu * rlmo_l, ztmin2 / rzsu_l, ztmax2 / rzsu_l)
        zetat = clamp(l_zt * rlmo_l, ztmin2 / rzst_l, ztmax2 / rzst_l)
        psmz_l = _lookup(bk, psm2, zetau, ztmin2, dzeta2)
        psmzl = _lookup(bk, psm2, zetalu, ztmin2, dzeta2)
        psim_l = -psmzl
        simm = psmzl - psmz_l + rlogu_l
        fm_l = psmzl + rlogu_l
        pshz_l = _lookup(bk, psh2, zetat, ztmin2, dzeta2)
        pshzl = _lookup(bk, psh2, zetalt, ztmin2, dzeta2)
        psih_l = -pshzl
        simh = (pshzl - pshz_l + rlogt_l) * fh02
        fhh_l = (pshzl + rlogt_l) * fh02
        ustark_l = l_ust * vkarman
        l_akms = _max(xp, ustark_l / simm, cxchl)
        l_akhs = _max(xp, ustark_l / simh, cxchl)
        wstar2 = xp.where(
            dthv_l <= 0.0,
            wwst2 * xp.abs(btgh * l_akhs * dthv_l) ** (2.0 / 3.0),
            0.0,
        )
        l_ust = _max(xp, sqrt(l_akms * sqrt(du2_l + wstar2)), epsust)

    def pick(a, b):
        return xp.where(sea, a, b)

    qs = pick(s_qs, l_qs)
    uz0 = pick(s_uz0, 0.0)
    vz0 = pick(s_vz0, 0.0)
    thz0 = pick(s_thz0, l_thz0)
    qz0 = pick(s_qz0, l_qz0)
    ustar_o = pick(s_ust, l_ust)
    z0_o = pick(s_z0, z0)
    akms = pick(s_akms, l_akms)
    akhs = pick(s_akhs, l_akhs)
    zu = pick(s_zu, l_zu)
    zt = pick(s_zt, l_zt)
    psmz = pick(psmz_s, psmz_l)
    pshz = pick(pshz_s, pshz_l)
    ustark = pick(ustark_s, ustark_l)
    rib = pick(rib_s, rib_l)
    rlmo = pick(rlmo_s, rlmo_l)
    out = {
        "psim": pick(psim_s, psim_l),
        "psih": pick(psih_s, psih_l),
        "fm": pick(fm_s, fm_l),
        "fhh": pick(fhh_s, fhh_l),
    }
    # countergradient fix (commented out in the Fortran)
    ct = xp.zeros_like(rib)

    # 2-m and 10-m diagnostics
    umflx = akms * (ulow - uz0)
    vmflx = akms * (vlow - vz0)
    hsflx = akhs * (thlow - thz0)
    hlflx = akhs * (qlow - qz0)
    zu10 = zu + 10.0
    zt02 = zt + 2.0
    zt10 = zt + 10.0
    rlnu10 = log(zu10 / zu)
    rlnt02 = log(zt02 / zt)
    rlnt10 = log(zt10 / zt)
    ztau10 = zu10 * rlmo
    ztat02 = zt02 * rlmo
    ztat10 = zt10 * rlmo

    def diag(psm, psh, ztmin, ztmax, dzeta, fh0, cxch):
        psm10 = _lookup(bk, psm, clamp(ztau10, ztmin, ztmax), ztmin, dzeta)
        simm10 = psm10 - psmz + rlnu10
        psh02 = _lookup(bk, psh, clamp(ztat02, ztmin, ztmax), ztmin, dzeta)
        simh02 = (psh02 - pshz + rlnt02) * fh0
        psh10 = _lookup(bk, psh, clamp(ztat10, ztmin, ztmax), ztmin, dzeta)
        simh10 = (psh10 - pshz + rlnt10) * fh0
        return (
            _max(xp, ustark / simm10, cxch),
            _max(xp, ustark / simh02, cxch),
            _max(xp, ustark / simh10, cxch),
        )

    d1 = diag(psm1, psh1, ztmin1, ztmax1, dzeta1, fh01, cxchs)
    d2 = diag(psm2, psh2, ztmin2, ztmax2, dzeta2, fh02, cxchl)
    akms10, akhs02, akhs10 = (pick(a, b) for a, b in zip(d1, d2))
    u10 = umflx / akms10 + uz0
    v10 = vmflx / akms10 + vz0

    def bracket(th, h):
        bad = ((thlow > thz0) & ((th < thz0) | (th > thlow))) | (
            (thlow < thz0) & ((th > thz0) | (th < thlow))
        )
        return xp.where(bad, thz0 + h * rdz * (thlow - thz0), th)

    # be certain that the 2-m and 10-m theta are bracketed by thz0 and thlow
    th02 = bracket(hsflx / akhs02 + thz0, 2.0)
    th10 = bracket(hsflx / akhs10 + thz0, 10.0)
    q02 = hlflx / akhs02 + qz0
    q10 = hlflx / akhs10 + qz0
    term1 = -0.068283 / tlow
    pshltr = psfc * exp(term1)
    z0out = zu
    # "equivalent" z0 to approximate local shelter readings (land)
    zuuz = _min(xp, zu * 0.50, 0.18)
    zue = _max(xp, zu * 0.35, zuuz)
    zu10e = zue + 10.0
    rlnu10e = log(zu10e / zue)
    ztau10e = clamp(zu10e * rlmo, ztmin2, ztmax2)
    psm10 = _lookup(bk, psm2, ztau10e, ztmin2, dzeta2)
    simm10 = psm10 - psmz + rlnu10e
    ekms10 = _max(xp, ustark / simm10, cxchl)
    landd = seamask < 0.5
    u10e = xp.where(landd, umflx / ekms10 + uz0, u10)
    v10e = xp.where(landd, vmflx / ekms10 + vz0, v10)
    # other WRF driver arrays
    rlow = rhosfc
    hfx = -rlow * cp * hsflx
    qfx = -rlow * hlflx * wetm
    qgh = (
        ((1.0 - seamask) * pq0 + seamask * pq0sea)
        / plow
        * exp(a2s * (tlow - a3s) / (tlow - a4s))
    )
    qgh = qgh / (1.0 - qgh)  # convert to mixing ratio
    out.update(
        qs=qs,
        uz0=uz0,
        vz0=vz0,
        thz0=thz0,
        qz0=qz0,
        ustar=ustar_o,
        z0=z0_o,
        akms=akms,
        akhs=akhs,
        rib=rib,
        rlmo=rlmo,
        ct=ct,
        u10=u10,
        v10=v10,
        th02=th02,
        th10=th10,
        q02=q02,
        q10=q10,
        pshltr=pshltr,
        z0out=z0out,
        u10e=u10e,
        v10e=v10e,
        chs=akhs,
        chs2=akhs02,
        cqs2=akhs02,
        hfx=hfx,
        qfx=qfx,
        flx_lh=xlv * qfx,
        flhc=rlow * cp * akhs,
        flqc=rlow * akhs * wetm,
        qgh=qgh,
        cpm=cp * (1.0 + 0.8 * qlow),
    )
    return out


def myjsfc(
    bk,
    *,
    itimestep,
    ht,
    dz,
    pmid,
    pint,
    th,
    t,
    qv,
    qc,
    u,
    v,
    q2,
    tsk,
    qsfc,
    thz0,
    qz0,
    uz0,
    vz0,
    xland,
    ustar,
    znt,
    z0base,
    pblh,
    mavail,
    akhs,
    akms,
    rhosfc,
    usemyjpbl=True,
    isftcflx=0,
):
    """MYJSFC (P1000mb = 1e5).  3-D inputs ``[k, ...]`` with index 0 =
    Fortran k = 1 (k = 1..nk; ``pint`` 1..nk+1, only k = 1 is read); 2-D
    inputs on the columns.  Returns the INOUT/OUT arguments keyed by their
    Fortran names (lower case): qsfc, thz0, qz0, uz0, vz0, ustar, znt,
    pblh, akhs, akms, rmol, rib, chs, chs2, cqs2, hfx, qfx, flx_lh, flhc,
    flqc, qgh, cpm, ct, u10, v10, t02, th02, tshltr, th10, q02, qshltr,
    q10, pshltr, u10e, v10e, psim, psih, fm, fhh, z0out."""
    xp = bk.xp
    tab = myjsfcinit(bk)
    kte = dz.shape[0]
    ntsd = itimestep
    # Z at the interfaces, counted downward (K = 1 the model top)
    zint = [None] * (kte + 2)
    zint[kte + 1] = ht
    for k in range(kte, 0, -1):
        zint[k] = zint[k + 1] + dz[kte - k, ...]
    if isinstance(ntsd, int):
        if ntsd == 1:
            ustar = xp.zeros_like(ustar) + 0.1
    else:
        ustar = xp.where(ntsd == 1, 0.1, ustar)
    lmh = kte
    psfc = pint[0, ...]
    thsk = tsk / (psfc / p1000mb) ** capa
    seamask = xland - 1.0
    # lowest level (K = LMH is Fortran k = 1)
    thlow = th[0, ...]
    tlow = t[0, ...]
    qlow = qv[0, ...] / (1.0 + qv[0, ...])
    plow = pmid[0, ...]
    cwmlow = qc[0, ...]
    thelow = (cwmlow * (-elocp / tlow) + 1.0) * thlow
    ulow = u[0, ...]
    vlow = v[0, ...]
    # the height of the PBL
    if usemyjpbl:
        lpbl = xp.zeros_like(ht) + 1.0
        for k in range(1, lmh):
            q2k = 2.0 * q2[kte - k, ...]
            lpbl = xp.where(q2k <= epsq2 * fh, float(k), lpbl)
        zl = xp.zeros_like(ht)
        for k in range(1, lmh):
            zl = xp.where(lpbl == float(k), zint[k], zl)
        pblh_new = zl - zint[lmh + 1]
    else:
        pblh_new = pblh
    zsl = (zint[lmh] - zint[lmh + 1]) * 0.5
    apesfc = (psfc / p1000mb) ** capa
    tz0 = _sel(xp, ntsd == 1, tsk, thz0 * apesfc)
    o = sfcdif(
        bk,
        tab,
        ntsd=ntsd,
        seamask=seamask,
        ths=thsk,
        qs=qsfc,
        psfc=psfc,
        uz0=uz0,
        vz0=vz0,
        tz0=tz0,
        tsk=tsk,
        thz0=thz0,
        qz0=qz0,
        ustar=ustar,
        z0=znt,
        z0base=z0base,
        akms=akms,
        akhs=akhs,
        pblh=pblh_new,
        wetm=mavail,
        ulow=ulow,
        vlow=vlow,
        tlow=tlow,
        thlow=thlow,
        thelow=thelow,
        qlow=qlow,
        cwmlow=cwmlow,
        zsl=zsl,
        plow=plow,
        rhosfc=rhosfc,
        isftcflx=isftcflx,
    )
    # remove supersaturation at 2 m and 10 m
    rapa = apesfc
    th02p = o["th02"]
    th10p = o["th10"]
    rapa02 = rapa - gocp02 / th02p
    rapa10 = rapa - gocp10 / th10p
    t02p = th02p * rapa02
    t10p = th10p * rapa10
    p02p = rapa02**rcap * p1000mb
    p10p = rapa10**rcap * p1000mb
    qs02 = pq0 / p02p * xp.exp(a2 * (t02p - a3) / (t02p - a4))
    qs10 = pq0 / p10p * xp.exp(a2 * (t10p - a3) / (t10p - a4))
    qshltr = xp.where(o["q02"] > qs02, qs02, o["q02"])
    q10 = xp.where(o["q10"] > qs10, qs10, o["q10"])
    return {
        "qsfc": o["qs"],
        "thz0": o["thz0"],
        "qz0": o["qz0"],
        "uz0": o["uz0"],
        "vz0": o["vz0"],
        "ustar": o["ustar"],
        "znt": o["z0"],
        "pblh": pblh_new,
        "akhs": o["akhs"],
        "akms": o["akms"],
        "rmol": o["rlmo"],
        "rib": o["rib"],
        "chs": o["chs"],
        "chs2": o["chs2"],
        "cqs2": o["cqs2"],
        "hfx": o["hfx"],
        "qfx": o["qfx"],
        "flx_lh": o["flx_lh"],
        "flhc": o["flhc"],
        "flqc": o["flqc"],
        "qgh": o["qgh"],
        "cpm": o["cpm"],
        "ct": o["ct"],
        "u10": o["u10"],
        "v10": o["v10"],
        "t02": th02p * apesfc,
        "th02": th02p,
        "tshltr": th02p,
        "th10": th10p,
        "q02": qshltr / (1.0 - qshltr),
        "qshltr": qshltr,
        "q10": q10,
        "pshltr": o["pshltr"],
        "u10e": o["u10e"],
        "v10e": o["v10e"],
        "psim": o["psim"],
        "psih": o["psih"],
        "fm": o["fm"],
        "fhh": o["fhh"],
        "z0out": o["z0out"],
    }


# ---------------------------------------------------------------------------
# CM1 glue (turb.F sfc_and_turb, sfcmodel=7; init_physics.F, init_surface.F)

# cm1.F (double precision)
SMEPS = 1.0e-60
RKARMAN = 1.0 / 0.4

# MYJSFC's carried / output 2-D arrays (cm1.F names) the model state needs
# besides the shared surface arrays (tsk, znt, ust, hpbl, u10, ...)
CARRIED_2D = (
    "zntmyj",
    "z0base",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "akhs",
    "akms",
    "th10",
    "q10",
    "u10e",
    "v10e",
    "tshltr",
    "qshltr",
    "pshltr",
    "chs",
    "chs2",
    "cqs2",
    "cpmm",
    "flhc",
    "flqc",
    "qgh",
    "lh",
    "ct",
    "ustm",
    "mol",
    "rmol",
    "psim",
    "psih",
    "fm",
    "fh",
    "t2",
    "th2",
    "q2",
    "br",
    "gz1oz0",
    "chklowq",
)

DUMP_KEYS = (
    "mol",
    "rmol",
    "tslb",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "akhs",
    "akms",
    "zntmyj",
    "z0base",
    "th10",
    "q10",
    "u10e",
    "v10e",
)


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def check(cfg):
    """What is not ported: sfcmodel=7 without the MYJ PBL (usemyjpbl =
    .false.), cm1setup other than 2, terrain."""
    if int(_opt(cfg, "ipbl", 0)) != 6:
        raise NotImplementedError(
            "sfcmodel=7 is only ported with the MYJ PBL (ipbl=6); "
            "usemyjpbl=.false. is not ported"
        )
    if int(_opt(cfg, "cm1setup", 1)) != 2:
        raise NotImplementedError("sfcmodel=7: only cm1setup=2 is ported")
    if bool(_opt(cfg, "terrain_flag", False)):
        raise NotImplementedError("sfcmodel=7 with terrain is not ported")


def init_state(ctx):
    """cm1.F allocations (0) of the MYJ surface arrays the state does not
    already hold, MYJSFCINIT's ust = 0.1 on 1..ni x 1..nj and
    init_surface.F's z0base = znt on the whole plane."""
    bk, g, s = ctx.bk, ctx.g, ctx.s
    n = g.ngxy
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    shp = g.s_shape[1:]
    out = {k: bk.zeros(shp) for k in CARRIED_2D if k not in s}
    out["zntmyj"] = bk.zeros(shp)
    out["z0base"] = s["znt"] + 0.0
    ust = s["ust"] if "ust" in s else bk.zeros(shp)
    out["ust"] = bk.set(ust + 0.0, (J, I), 0.1)
    return out


def surface_layer(ctx):
    """sfc_and_turb's sfcmodel=7 block: MYJSFC (ITIMESTEP = nstep, ZNT =
    zntmyj, z0out = znt, rhosfc = rf(k=1)) on the WRF column inputs, br
    clipped to [-10, 10], gz1oz0 = log(zh(k=1)/znt); SLAB (update_sfc and
    testcase /= 11; the startup pass does nothing for sfcmodel=7); then ch
    = chs2, cq = cqs2, s10, the sfcmodel=7 block (ustm, wspd, mol, cd) and
    thflux / qvflux (dosfcflx or output_sfcflx)."""
    bk, g, cfg, s, b = ctx.bk, ctx.g, ctx.cfg, ctx.s, ctx.b
    xp = bk.xp
    n, z, nk = g.ngxy, g.ngz, g.nk
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    K = slice(z, z + nk)
    w = ctx.w

    def i3(a):
        return a[K, J, I]

    def i2(name):
        return s[name][J, I]

    r = myjsfc(
        bk,
        itimestep=int(ctx.nstep) if isinstance(ctx.nstep, int) else ctx.nstep,
        ht=bk.zeros(i2("tsk").shape),
        dz=i3(w["dz8w"]),
        pmid=i3(ctx.prs),
        pint=w["p3di"][z : z + nk + 1, J, I],
        th=i3(w["th"]),
        t=i3(w["t"]),
        qv=i3(w["qv"]),
        qc=i3(w["qc"]),
        u=i3(w["u"]),
        v=i3(w["v"]),
        q2=i3(s["tke_myj"]),
        tsk=i2("tsk"),
        qsfc=i2("qsfc"),
        thz0=i2("thz0"),
        qz0=i2("qz0"),
        uz0=i2("uz0"),
        vz0=i2("vz0"),
        xland=i2("xland"),
        ustar=i2("ust"),
        znt=i2("zntmyj"),
        z0base=i2("z0base"),
        pblh=i2("hpbl"),
        mavail=i2("mavail"),
        akhs=i2("akhs"),
        akms=i2("akms"),
        rhosfc=ctx.rf[z, J, I],
        usemyjpbl=int(_opt(cfg, "ipbl", 0)) == 6,
        isftcflx=int(_opt(cfg, "isftcflx", 0)),
    )
    names = {
        "qsfc": "qsfc",
        "thz0": "thz0",
        "qz0": "qz0",
        "uz0": "uz0",
        "vz0": "vz0",
        "ust": "ustar",
        "zntmyj": "znt",
        "hpbl": "pblh",
        "akhs": "akhs",
        "akms": "akms",
        "rmol": "rmol",
        "chs": "chs",
        "chs2": "chs2",
        "cqs2": "cqs2",
        "hfx": "hfx",
        "qfx": "qfx",
        "lh": "flx_lh",
        "flhc": "flhc",
        "flqc": "flqc",
        "qgh": "qgh",
        "cpmm": "cpm",
        "ct": "ct",
        "u10": "u10",
        "v10": "v10",
        "t2": "t02",
        "th2": "th02",
        "tshltr": "tshltr",
        "th10": "th10",
        "q2": "q02",
        "qshltr": "qshltr",
        "q10": "q10",
        "pshltr": "pshltr",
        "u10e": "u10e",
        "v10e": "v10e",
        "psim": "psim",
        "psih": "psih",
        "fm": "fm",
        "fh": "fhh",
        "znt": "z0out",
    }
    v = {k: r[m] for k, m in names.items()}
    br = _max(xp, -10.0, r["rib"])
    v["br"] = xp.where(10.0 < br, 10.0, br)
    v["gz1oz0"] = xp.log(b["zh"][z, J, I] / v["znt"])

    # SLAB (land and water; ifsnow = 0, radiation on)
    if ctx.update_sfc and int(_opt(cfg, "testcase", 0)) != 11:
        from . import slab

        upd = slab.slab_update(
            ctx,
            flhc=v["flhc"],
            flqc=v["flqc"],
            hfx=v["hfx"],
            qfx=v["qfx"],
            lh=v["lh"],
            tsk=i2("tsk"),
            qsfc=v["qsfc"],
        )
        v.update(upd)

    # WRF parameters into CM1 arrays, the sfcmodel=7 block
    u10, v10 = v["u10"], v["v10"]
    s10 = xp.sqrt(u10 * u10 + v10 * v10)
    v["ch"] = v["chs2"]
    v["cq"] = v["cqs2"]
    v["s10"] = s10
    ust = v["ust"]
    v["ustm"] = ust
    u1, v1 = w["u"][z, J, I], w["v"][z, J, I]
    wspd = _max(xp, xp.sqrt(u1 * u1 + v1 * v1), 0.001)
    rmol = v["rmol"]
    small = xp.abs(rmol) <= SMEPS
    v["mol"] = xp.where(
        small,
        xp.copysign(xp.zeros_like(rmol) + 1.0e20, rmol),
        1.0 / xp.where(small, 1.0, rmol),
    )
    tem = _max(xp, 0.1, ust * RKARMAN * v["fm"])
    v["wspd"] = _max(xp, wspd, tem)
    v["cd"] = ust * ust / _max(xp, 1.0e-20, s10 * s10)
    if bool(_opt(cfg, "dosfcflx", False)) or int(_opt(cfg, "output_sfcflx", 0)) == 1:
        rf1 = ctx.rf[z, J, I]
        v["thflux"] = v["hfx"] / (v["cpmm"] * rf1)
        v["qvflux"] = v["qfx"] / rf1
    out = {}
    for k, a in v.items():
        if a.ndim == 3:  # tslb [nsoil, j, i]
            out[k] = bk.set(s[k] + 0.0, (slice(None), J, I), a)
        else:
            base = s[k] + 0.0 if k in s else bk.zeros(g.s_shape[1:])
            out[k] = bk.set(base, (J, I), a)
    return out
