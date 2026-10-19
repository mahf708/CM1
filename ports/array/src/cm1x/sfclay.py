"""The MM5-type surface layers CM1 takes from WRF/MMM:

* sfcmodel = 2: src/sfclay.F (module_sf_sfclay: SFCLAY / SFCLAY1D and the
  sfclayinit stability tables, "MM5 similarity theory");
* sfcmodel = 3: src/sf_sfclayrev.F (sf_sfclayrev_run through the
  mmm_physics wrapper, sf_sfclayrev_init, zolri and the psi* table
  functions, "revised MM5");

and the turb.F glue after them (SLAB or the qsfc diagnosis, ch = chs2,
cq = cqs2, s10, thflux / qvflux: ``pbl.wrf_sfc_epilogue``).

Columns are independent, so both are vectorised over [j, i] with the
Fortran's per-point operation order (x**2 as x*x, Fortran max/min of two
non-NaN numbers as maximum/minimum).  The stability classes become
``where`` selections; zolri's secant iteration runs its at most 10 steps
with a per-point "still iterating" mask.

The sfclayinit tables: gfortran -O2 vectorises that loop through glibc's
libmvec (pow, log, atan two lanes at a time), whose results differ from
the scalar libm in the last bit for some arguments.  The tables are built
on the host with the same libmvec entry points when libmvec is available
(x86-64 glibc), else with Python's math (scalar libm).

Not ported: shalwater_z0, scm_force_flux (CM1 passes .false.), isfflx = 0
(turb.F passes 1).  Quirk kept: SFCLAY1D sets z0t / z0q only over water
(z0q also with isftcflx = 1); elsewhere they keep their previous values.
"""

from __future__ import annotations

import functools
import math

from . import constants as C

WRF_INPUTS = True
DUMP_KEYS = ("mol",)

# module parameters
VCONVC = 1.0
CZO = 0.0185
OZO = 1.59e-5
XKA = 2.4e-5
PRT = 1.0
P1000MB = 100000.0
EP1 = C.rv / C.rd - 1.0
EP2 = C.rd / C.rv

# arrays the WRF layers carry besides pbl.SFC_KEYS (cm1.F allocates them 0)
EXTRA_KEYS = (
    "qgh", "flhc", "flqc", "regime", "lh", "ck", "cka", "z0t", "z0q", "ustm",
    "chs", "chs2", "cqs2",
)  # fmt: skip


def _opt(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def check(cfg):
    sfcmodel = int(_opt(cfg, "sfcmodel", 0))
    if sfcmodel not in (2, 3):
        raise NotImplementedError(f"cm1x.sfclay: sfcmodel={sfcmodel}")
    for name in ("isftcflx", "iz0tlnd"):
        v = int(_opt(cfg, name, 0))
        if v not in (0, 1, 2):
            raise NotImplementedError(f"cm1x.sfclay: {name}={v} not ported")
    if int(_opt(cfg, "oceanmodel", 1)) == 2:
        raise NotImplementedError("cm1x.sfclay: oceanmodel=2 not ported")


def init_state(ctx):
    """cm1.F allocations of the WRF surface-layer arrays (0)."""
    z = ctx.bk.zeros(ctx.g.s_shape[1:])
    return {k: z + 0.0 for k in EXTRA_KEYS if k not in ctx.s}


# ---------------------------------------------------------------------------
# tables


def _libmvec():
    try:
        import ctypes
        import platform

        if platform.machine() not in ("x86_64", "AMD64"):
            return None
        lib = ctypes.CDLL("libmvec.so.1")
        d = ctypes.c_double
        # SSE2 two-lane variants: on x86-64 an __m128d argument / result is
        # passed in xmm registers, so lane 0 is read and written as a double.
        pw, lg, at = lib._ZGVbN2vv_pow, lib._ZGVbN2v_log, lib._ZGVbN2v_atan
        pw.argtypes, pw.restype = [d, d], d
        lg.argtypes, lg.restype = [d], d
        at.argtypes, at.restype = [d], d
        return pw, lg, at
    except (OSError, AttributeError):
        return None


@functools.cache
def sfclayinit():
    """sfclay.F sfclayinit: psimtb, psihtb (0..1000) as Python lists."""
    mv = _libmvec()
    if mv is None:
        pw, lg, at = math.pow, math.log, math.atan
    else:
        pw, lg, at = mv
    two_atan1 = 2.0 * math.atan(1.0)  # constant-folded by gfortran
    m, h = [], []
    for n in range(1001):
        zoln = -float(n) * 0.01
        x = pw(1 - 16.0 * zoln, 0.25)
        m.append(
            2 * lg(0.5 * (1 + x)) + lg(0.5 * (1 + x * x)) - 2.0 * at(x) + two_atan1
        )
        y = math.sqrt(1 - 16 * zoln)
        h.append(2 * lg(0.5 * (1 + y)))
    return m, h


def _psim_stable_full(xp, z):
    return -6.1 * xp.log(z + (1 + z**2.5) ** (1.0 / 2.5))


def _psih_stable_full(xp, z):
    return -5.3 * xp.log(z + (1 + z**1.1) ** (1.0 / 1.1))


_S3 = math.sqrt(3.0)
_A1 = math.atan(1.0)


def _psim_unstable_full(xp, z):
    x = (1.0 - 16.0 * z) ** 0.25
    psimk = (
        2 * xp.log(0.5 * (1 + x))
        + xp.log(0.5 * (1 + x * x))
        - 2.0 * xp.atan(x)
        + 2.0 * _A1
    )
    ym = (1.0 - 10.0 * z) ** 0.33
    psimc = (
        (3.0 / 2.0) * xp.log((ym * ym + ym + 1.0) / 3.0)
        - _S3 * xp.atan((2.0 * ym + 1) / _S3)
        + 4.0 * _A1 / _S3
    )
    return (psimk + z * z * (psimc)) / (1 + z * z)


def _psih_unstable_full(xp, z):
    y = (1.0 - 16.0 * z) ** 0.5
    psihk = 2.0 * xp.log((1 + y) / 2.0)
    yh = (1.0 - 34.0 * z) ** 0.33
    psihc = (
        (3.0 / 2.0) * xp.log((yh * yh + yh + 1.0) / 3.0)
        - _S3 * xp.atan((2.0 * yh + 1) / _S3)
        + 4.0 * _A1 / _S3
    )
    return (psihk + z * z * (psihc)) / (1 + z * z)


class _MathNS:
    log = staticmethod(math.log)
    atan = staticmethod(math.atan)


math_ns = _MathNS()


@functools.cache
def sf_sfclayrev_init():
    """sf_sfclayrev.F sf_sfclayrev_init: psim_stab, psim_unstab, psih_stab,
    psih_unstab (0..1000), scalar libm (Python math)."""
    zs = [float(n) * 0.01 for n in range(1001)]
    zu = [-float(n) * 0.01 for n in range(1001)]
    out = {
        "psim_stab": [_psim_stable_full(math_ns, z) for z in zs],
        "psih_stab": [_psih_stable_full(math_ns, z) for z in zs],
        "psim_unstab": [_psim_unstable_full(math_ns, z) for z in zu],
        "psih_unstab": [_psih_unstable_full(math_ns, z) for z in zu],
    }
    return out


def _ity(bk):
    return bk.xp.int64


def _lookup(bk, tab, zz):
    """tab(n) + r*(tab(n+1) - tab(n)) with n = int(zz), r = zz - n, for
    0 <= zz (indices clamped to 0..999; callers select valid points)."""
    xp = bk.xp
    t = bk.asarray(tab)
    n = xp.astype(xp.trunc(zz), _ity(bk))
    n = xp.minimum(xp.maximum(n, 0), 999)
    r = zz - xp.astype(n, zz.dtype)
    sh = zz.shape
    flat = xp.reshape(n, (-1,))
    t0 = xp.reshape(xp.take(t, flat), sh)
    t1 = xp.reshape(xp.take(t, flat + 1), sh)
    return t0 + r * (t1 - t0)


# ---------------------------------------------------------------------------
# sfcmodel = 2


def sfclay(bk, cfg, *, ux, vx, t1d, qv1d, p1d, dz8w1d, rhox, psfcpa, tsk, xland, mavail, pblh,
           dx, qsfc, znt, ust, mol, zol, hfx, qfx, ustm, z0t, z0q):  # fmt: skip
    """sfclay.F SFCLAY / SFCLAY1D (isfflx = 1) on [j, i] interior arrays
    (k = 1 inputs; rhox = rf(1)).  Returns the updated surface arrays."""
    xp = bk.xp
    isftcflx = int(_opt(cfg, "isftcflx", 0))
    iz0tlnd = int(_opt(cfg, "iz0tlnd", 0))
    CP, G, ROVCP, K = C.cp, C.g, C.rovcp, C.karman
    lv1, lv2 = C.lv1, C.lv2
    psimtb, psihtb = sfclayinit()

    def mx(a, b):
        return xp.maximum(a, b)

    def mn(a, b):
        return xp.minimum(a, b)

    PSFC = psfcpa / 1000.0
    TGDSA = tsk
    THGB = tsk * (P1000MB / psfcpa) ** ROVCP
    PL = p1d / 1000.0
    SCR3 = t1d
    THCON = (P1000MB * 0.001 / PL) ** ROVCP
    THX = SCR3 * THCON
    QX = qv1d
    TVCON = 1.0 + EP1 * QX
    THVX = THX * TVCON
    E1 = C.svp1 * xp.exp(C.svp2 * (TGDSA - C.svpt0) / (TGDSA - C.svp3))
    QSFC = xp.where((xland > 1.5) | (qsfc <= 2.0e-10), EP2 * E1 / (PSFC - E1), qsfc)
    E1 = C.svp1 * xp.exp(C.svp2 * (t1d - C.svpt0) / (t1d - C.svp3))
    QGH = EP2 * E1 / (PL - E1)
    CPM = CP * (1.0 + 0.8 * QX)
    ZQKL = dz8w1d + 0.0
    ZA = 0.5 * (ZQKL + 0.0)
    GOVRTH = G / THX

    GZ1OZ0 = xp.log(ZA / znt)
    GZ2OZ0 = xp.log(2.0 / znt)
    GZ10OZ0 = xp.log(10.0 / znt)
    WSPD = xp.sqrt(ux * ux + vx * vx)
    TSKV = THGB * (1.0 + EP1 * QSFC)
    DTHVDZ = THVX - TSKV
    land = xland < 1.5
    fluxc = mx(hfx / rhox / CP + EP1 * TSKV * qfx / rhox, 0.0)
    arg = G / TGDSA * pblh * fluxc
    vc_land = VCONVC * xp.where(land & (arg >= 0), arg, 0.0) ** 0.33
    DTHVM = xp.where(-DTHVDZ >= 0, -DTHVDZ, 0.0)
    VCONV = xp.where(land, vc_land, xp.sqrt(DTHVM))
    VSGD = 0.32 * mx(dx / 5000.0 - 1.0, 0.0) ** 0.33
    WSPD = xp.sqrt(WSPD * WSPD + VCONV * VCONV + VSGD * VSGD)
    WSPD = mx(WSPD, 0.1)
    BR = GOVRTH * ZA * DTHVDZ / (WSPD * WSPD)
    BR = xp.where(mol < 0.0, mn(BR, 0.0), BR)
    RMOL = -GOVRTH * DTHVDZ * ZA * K

    ust_s = xp.where(ust < 0.01, 1.0, ust)
    zol_m = xp.where(ust < 0.01, BR * GZ1OZ0, K * GOVRTH * ZA * mol / (ust_s * ust_s))
    c4 = BR < 0.0
    c1 = ~c4 & ~(BR < 0.2)
    c2 = ~c4 & ~c1 & ~(BR == 0.0)
    # class 4: free convection
    Z4 = zol_m
    Z10 = 10.0 / ZA * Z4
    Z2 = 2.0 / ZA * Z4
    Z4 = mx(mn(Z4, 0.0), -9.9999)
    Z10 = mx(mn(Z10, 0.0), -9.9999)
    Z2 = mx(mn(Z2, 0.0), -9.9999)
    P4 = {}
    for nm, zz in (("", Z4), ("10", Z10), ("2", Z2)):
        a = -zz * 100.0
        P4["m" + nm] = _lookup(bk, psimtb, a)
        P4["h" + nm] = _lookup(bk, psihtb, a)
    psim4 = mn(P4["m"], 0.9 * GZ1OZ0)
    psih4 = mn(P4["h"], 0.9 * GZ1OZ0)
    psih2_4 = mn(P4["h2"], 0.9 * GZ2OZ0)
    psim10_4 = mn(P4["m10"], 0.9 * GZ10OZ0)
    psih10_4 = mn(P4["h10"], 0.9 * GZ10OZ0)
    # class 1: stable
    psim1 = mx(-10.0 * GZ1OZ0, -10.0)
    psim10_1 = mx(10.0 / ZA * psim1, -10.0)
    psim2_1 = mx(2.0 / ZA * psim1, -10.0)
    rmol1 = mn(zol_m, 9.999) / ZA
    # class 2: damped mechanical turbulence
    psim2c = mx(-5.0 * BR * GZ1OZ0 / (1.1 - 5.0 * BR), -10.0)
    psim10_2 = mx(10.0 / ZA * psim2c, -10.0)
    psim2_2 = mx(2.0 / ZA * psim2c, -10.0)
    zol2c = BR * GZ1OZ0 / (1.00001 - 5.0 * BR)
    zol2b = mn((1.89 * GZ1OZ0 + 44.2) * BR * BR + (1.18 * GZ1OZ0 - 1.37) * BR, 9.999)
    zol2c = xp.where(zol2c > 0.5, zol2b, zol2c)

    def sel(v4, v1, v2, v3):
        return xp.where(c4, v4, xp.where(c1, v1, xp.where(c2, v2, v3)))

    zero = xp.zeros_like(BR)
    REGIME = sel(zero + 4.0, zero + 1.0, zero + 2.0, zero + 3.0)
    PSIM = sel(psim4, psim1, psim2c, zero)
    PSIH = sel(psih4, psim1, psim2c, zero)
    PSIM10 = sel(psim10_4, psim10_1, psim10_2, zero)
    PSIH10 = sel(psih10_4, psim10_1, psim10_2, zero)
    PSIH2 = sel(psih2_4, psim2_1, psim2_2, zero)
    ZOL = sel(Z4, zol, zol2c, zol_m)
    RMOL = sel(Z4 / ZA, rmol1, zol2c / ZA, zol_m / ZA)

    # loop 330: frictional velocity
    DTG = THX - THGB
    PSIX = GZ1OZ0 - PSIM
    PSIX10 = GZ10OZ0 - PSIM10
    PSIT = mx(GZ1OZ0 - PSIH, 2.0)
    water = (xland - 1.5) >= 0
    ZL = xp.where(water, znt, 0.01)
    PSIQ = xp.log(K * ust * ZA / XKA + ZA / ZL) - PSIH
    PSIT2 = GZ2OZ0 - PSIH2
    PSIQ2 = xp.log(K * ust * 2.0 / XKA + 2.0 / ZL) - PSIH2
    PSIQ10 = xp.log(K * ust * 10.0 / XKA + 10.0 / ZL) - PSIH10
    VISC = (1.32 + 0.009 * (SCR3 - 273.15)) * 1.0e-5
    RESTAR = ust * znt / VISC
    Z0T = (5.5e-5) * (xp.where(water, RESTAR, 1.0) ** -0.60)
    Z0T = mx(mn(Z0T, 1.0e-4), 2.0e-9)
    Z0Q = Z0T
    PSIQ = xp.where(water, mx(xp.log((ZA + Z0Q) / Z0Q) - PSIH, 2.0), PSIQ)
    PSIT = xp.where(water, mx(xp.log((ZA + Z0T) / Z0T) - PSIH, 2.0), PSIT)
    PSIQ2 = xp.where(water, mx(xp.log((2.0 + Z0Q) / Z0Q) - PSIH2, 2.0), PSIQ2)
    PSIT2 = xp.where(water, mx(xp.log((2.0 + Z0T) / Z0T) - PSIH2, 2.0), PSIT2)
    PSIQ10 = xp.where(water, mx(xp.log((10.0 + Z0Q) / Z0Q) - PSIH10, 2.0), PSIQ10)
    if isftcflx == 1:
        Z0Q = xp.where(water, 1.0e-4 + zero, Z0Q)
        pq = xp.log(ZA / Z0Q) - PSIH
        pq2 = xp.log(2.0 / Z0Q) - PSIH2
        PSIQ = xp.where(water, pq, PSIQ)
        PSIT = xp.where(water, pq, PSIT)
        PSIQ2 = xp.where(water, pq2, PSIQ2)
        PSIQ10 = xp.where(water, xp.log(10.0 / Z0Q) - PSIH10, PSIQ10)
        PSIT2 = xp.where(water, pq2, PSIT2)
    if isftcflx == 2:
        rs = xp.sqrt(xp.sqrt(xp.where(water, RESTAR, 1.0)))
        GZ0OZT = 0.40 * (7.3 * rs * math.sqrt(0.71) - 5.0)
        GZ0OZQ = 0.40 * (7.3 * rs * math.sqrt(0.60) - 5.0)
        PSIT = xp.where(water, GZ1OZ0 - PSIH + GZ0OZT, PSIT)
        PSIQ = xp.where(water, GZ1OZ0 - PSIH + GZ0OZQ, PSIQ)
        PSIT2 = xp.where(water, GZ2OZ0 - PSIH2 + GZ0OZT, PSIT2)
        PSIQ2 = xp.where(water, GZ2OZ0 - PSIH2 + GZ0OZQ, PSIQ2)
        PSIQ10 = xp.where(water, GZ10OZ0 - PSIH + GZ0OZQ, PSIQ10)
    out = {
        "ck": (K / PSIX10) * (K / PSIQ10),
        "cd": (K / PSIX10) * (K / PSIX10),
        "cka": (K / PSIX) * (K / PSIQ),
        "cda": (K / PSIX) * (K / PSIX),
    }
    if iz0tlnd >= 1:
        lnd = (xland - 1.5) <= 0.0
        rs = xp.sqrt(ust * znt / VISC)
        czil = 10.0 ** (-0.40 * (znt / 0.07)) if iz0tlnd == 1 else 0.1
        PSIT = xp.where(lnd, GZ1OZ0 - PSIH + czil * K * rs, PSIT)
        PSIQ = xp.where(lnd, GZ1OZ0 - PSIH + czil * K * rs, PSIQ)
        PSIT2 = xp.where(lnd, GZ2OZ0 - PSIH2 + czil * K * rs, PSIT2)
        PSIQ2 = xp.where(lnd, GZ2OZ0 - PSIH2 + czil * K * rs, PSIQ2)
    out["z0t"] = xp.where(water, Z0T, z0t)
    out["z0q"] = xp.where(water, Z0Q, z0q)
    UST = 0.5 * ust + 0.5 * K * WSPD / PSIX
    WSPDI = xp.sqrt(ux * ux + vx * vx)
    out["ustm"] = 0.5 * ustm + 0.5 * K * WSPDI / PSIX
    out["u10"] = ux * PSIX10 / PSIX
    out["v10"] = vx * PSIX10 / PSIX
    TH2 = THGB + DTG * PSIT2 / PSIT
    out["th2"] = TH2
    out["q2"] = QSFC + (QX - QSFC) * PSIQ2 / PSIQ
    out["t2"] = TH2 * (psfcpa / P1000MB) ** ROVCP
    UST = xp.where((xland - 1.5) < 0.0, mx(UST, 0.1), UST)
    MOL = K * DTG / PSIT / PRT
    out["fm"] = PSIX
    out["fh"] = PSIT

    # isfflx = 1: fluxes; over water the roughness length follows ust
    ZNT = mn(CZO * UST * UST / G + 0.11 * 1.5e-5 / UST, 2.85e-3)
    if isftcflx != 0:
        ZW = mn((UST / 1.06) ** 0.3, 1.0)
        ZN1 = 0.011 * UST * UST / G + OZO
        ZN2 = 10.0 * xp.exp(-9.5 * UST ** (-0.3333)) + 0.11 * 1.5e-5 / mx(UST, 0.01)
        ZNT = mx(mn((1.0 - ZW) * ZN1 + ZW * ZN2, 2.85e-3), 1.27e-7)
    ZNT = xp.where(water, ZNT, znt)
    FLQC = rhox * mavail * UST * K / PSIQ
    DTTHX = xp.abs(THX - THGB)
    dth = xp.where(DTTHX > 1.0e-5, THX - THGB, 1.0)
    FLHC = xp.where(DTTHX > 1.0e-5, CPM * rhox * UST * MOL / dth, 0.0)
    QFX = mx(FLQC * (QSFC - QX), 0.0)
    h = FLHC * (THGB - THX)
    HFX = xp.where(
        xland - 1.5 > 0.0, h, xp.where(xland - 1.5 < 0.0, mx(h, -250.0), 0.0)
    )
    out.update(
        qgh=QGH, flhc=FLHC, flqc=FLQC, cpmm=CPM, qsfc=QSFC, znt=ZNT, ust=UST,
        gz1oz0=GZ1OZ0, wspd=WSPD, br=BR, mol=MOL, rmol=RMOL, regime=REGIME,
        psim=PSIM, psih=PSIH, zol=ZOL, hfx=HFX, qfx=QFX,
        lh=(lv1 - lv2 * tsk) * QFX,
        chs=UST * K / PSIQ, cqs2=UST * K / PSIQ2, chs2=UST * K / PSIT2,
    )  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# sfcmodel = 3


class _RevTab:
    def __init__(self, bk):
        self.bk = bk
        self.t = sf_sfclayrev_init()

    def _f(self, name, full, z, sign):
        xp = self.bk.xp
        zz = sign * z * 100.0
        n = xp.trunc(zz)
        use_tab = (n + 1) < 1000
        zt = xp.where(use_tab, zz, 0.0)
        tab = _lookup(self.bk, self.t[name], zt)
        zf = xp.where(use_tab, 0.0, z)
        return xp.where(use_tab, tab, full(xp, zf))

    def psim_stable(self, z):
        return self._f("psim_stab", _psim_stable_full, z, 1.0)

    def psih_stable(self, z):
        return self._f("psih_stab", _psih_stable_full, z, 1.0)

    def psim_unstable(self, z):
        return self._f("psim_unstab", _psim_unstable_full, z, -1.0)

    def psih_unstable(self, z):
        return self._f("psih_unstab", _psih_unstable_full, z, -1.0)

    def zolri2(self, zol2, ri2, z, z0):
        """Returns (f, zol2) (zol2 is intent(inout): 0 when its sign differs
        from ri2)."""
        xp = self.bk.xp
        zol2 = xp.where(zol2 * ri2 < 0.0, 0.0, zol2)
        zol20 = zol2 * z0 / z
        zol3 = zol2 + zol20
        un = ri2 < 0
        lg = xp.log((z + z0) / z0)
        u3, u20 = xp.where(un, zol3, 0.0), xp.where(un, zol20, 0.0)
        s3, s20 = xp.where(un, 0.0, zol3), xp.where(un, 0.0, zol20)
        psix2 = xp.where(
            un,
            lg - (self.psim_unstable(u3) - self.psim_unstable(u20)),
            lg - (self.psim_stable(s3) - self.psim_stable(s20)),
        )
        psih2 = xp.where(
            un,
            lg - (self.psih_unstable(u3) - self.psih_unstable(u20)),
            lg - (self.psih_stable(s3) - self.psih_stable(s20)),
        )
        return zol2 * psih2 / (psix2 * psix2) - ri2, zol2

    def zolri(self, ri, z, z0):
        xp = self.bk.xp
        neg = ri < 0.0
        x1 = xp.where(neg, -5.0, 0.0)
        x2 = xp.where(neg, 0.0, 5.0)
        result = xp.zeros_like(ri)
        fx1, x1 = self.zolri2(x1, ri, z, z0)
        fx2, x2 = self.zolri2(x2, ri, z, z0)
        active = xp.abs(x1 - x2) > -1.0  # all True
        for _ in range(10):
            go = active & (xp.abs(x1 - x2) > 0.01) & (fx1 != fx2)
            den = xp.where(go, fx2 - fx1, 1.0)
            left = go & (xp.abs(fx2) < xp.abs(fx1))
            right = go & ~(xp.abs(fx2) < xp.abs(fx1))
            x1n = x1 - fx1 / den * (x2 - x1)
            x2n = x2 - fx2 / den * (x2 - x1)
            f1n, x1n = self.zolri2(xp.where(left, x1n, x1), ri, z, z0)
            f2n, x2n = self.zolri2(xp.where(right, x2n, x2), ri, z, z0)
            x1 = xp.where(left, x1n, x1)
            fx1 = xp.where(left, f1n, fx1)
            x2 = xp.where(right, x2n, x2)
            fx2 = xp.where(right, f2n, fx2)
            result = xp.where(left, x1, xp.where(right, x2, result))
            active = go
        return result

    def psih_for(self, zol, za, z0x):
        xp = self.bk.xp
        zolzz = zol * (za + z0x) / za
        zol10 = zol * (10.0 + z0x) / za
        zol2 = zol * (2.0 + z0x) / za
        zol0 = zol * z0x / za
        st = zol > 0.0
        un = ~st & ~(zol == 0)

        def pair(a):
            return xp.where(st, a, 0.0), xp.where(un, a, 0.0)

        out = []
        for a in (zolzz, zol10, zol2):
            sa, ua = pair(a)
            s0, u0 = pair(zol0)
            out.append(
                xp.where(
                    st,
                    self.psih_stable(sa) - self.psih_stable(s0),
                    xp.where(un, self.psih_unstable(ua) - self.psih_unstable(u0), 0.0),
                )
            )
        return out  # psih, psih10, psih2


def sfclayrev(bk, cfg, *, ux, vx, t1d, qv1d, p1d, dz8w1d, psfcpa, tsk, xland, mavail, pblh, dx,
              qsfc, znt, ust, mol, zol, hfx, qfx, ustm, z0t, z0q, **_):  # fmt: skip
    """sf_sfclayrev.F sf_sfclayrev_run (isfflx, no shalwater_z0, no
    scm_force_flux) on [j, i] interior arrays.  Returns the updated surface
    arrays."""
    xp = bk.xp
    T = _RevTab(bk)
    isftcflx = int(_opt(cfg, "isftcflx", 0))
    iz0tlnd = int(_opt(cfg, "iz0tlnd", 0))
    cp, g, rovcp, r, K = C.cp, C.g, C.rovcp, C.rd, C.karman

    def mx(a, b):
        return xp.maximum(a, b)

    def mn(a, b):
        return xp.minimum(a, b)

    psfc = psfcpa / 1000.0
    tgdsa = tsk
    thgb = tsk * (P1000MB / psfcpa) ** rovcp
    pl = p1d / 1000.0
    scr3 = t1d
    thcon = (P1000MB * 0.001 / pl) ** rovcp
    thx = scr3 * thcon
    qx = qv1d
    tvcon = 1.0 + EP1 * qx
    thvx = thx * tvcon
    scr4 = scr3 * tvcon
    e1 = C.svp1 * xp.exp(C.svp2 * (tgdsa - C.svpt0) / (tgdsa - C.svp3))
    qsfc = xp.where((xland > 1.5) | (qsfc <= 0.0), EP2 * e1 / (psfc - e1), qsfc)
    e1 = C.svp1 * xp.exp(C.svp2 * (t1d - C.svpt0) / (t1d - C.svp3))
    qgh = EP2 * e1 / (pl - e1)
    cpm = cp * (1.0 + 0.8 * qx)
    rhox = psfc * 1000.0 / (r * scr4)
    zqkl = dz8w1d + 0.0
    za = 0.5 * (zqkl + 0.0)
    govrth = g / thx

    gz1oz0 = xp.log((za + znt) / znt)
    gz2oz0 = xp.log((2.0 + znt) / znt)
    gz10oz0 = xp.log((10.0 + znt) / znt)
    wspd = xp.sqrt(ux * ux + vx * vx)
    tskv = thgb * (1.0 + EP1 * qsfc)
    dthvdz = thvx - tskv
    land = xland < 1.5
    fluxc = mx(hfx / rhox / cp + EP1 * tskv * qfx / rhox, 0.0)
    arg = g / tgdsa * pblh * fluxc
    vc_land = VCONVC * xp.where(land & (arg >= 0), arg, 0.0) ** 0.33
    dthvm = xp.where(-dthvdz >= 0, -dthvdz, 0.0)
    vconv = xp.where(land, vc_land, xp.sqrt(dthvm))
    vsgd = 0.32 * mx(dx / 5000.0 - 1.0, 0.0) ** 0.33
    wspd = xp.sqrt(wspd * wspd + vconv * vconv + vsgd * vsgd)
    wspd = mx(wspd, 0.1)
    br = govrth * za * dthvdz / (wspd * wspd)
    br = xp.where(mol < 0.0, mn(br, 0.0), br)

    # loop 320
    brc = xp.where(br > 250.0, 250.0, xp.where(br < -250.0, -250.0, br))
    zr = T.zolri(brc, za, znt)
    zol = xp.where(br > 0, zr, zol)
    zol = xp.where(br < 0, xp.where(ust < 0.001, br * gz1oz0, zr), zol)
    zolzz = zol * (za + znt) / za
    zol10 = zol * (10.0 + znt) / za
    zol2 = zol * (2.0 + znt) / za
    zol0 = zol * znt / za
    zl2 = (2.0) / za * zol
    zl10 = (10.0) / za * zol
    zl = xp.where((xland - 1.5) < 0.0, (0.01) / za * zol, zol0)

    c4 = br < 0.0
    c3 = br == 0.0
    c1 = ~c4 & ~c3

    def stab(fs, fu, a, b):
        sa, sb = xp.where(c1, a, 0.0), xp.where(c1, b, 0.0)
        ua, ub = xp.where(c4, a, 0.0), xp.where(c4, b, 0.0)
        return xp.where(c4, fu(ua) - fu(ub), xp.where(c1, fs(sa) - fs(sb), 0.0))

    PM, PH = (T.psim_stable, T.psim_unstable), (T.psih_stable, T.psih_unstable)
    psim = stab(*PM, zolzz, zol0)
    psih = stab(*PH, zolzz, zol0)
    psim10 = stab(*PM, zol10, zol0)
    psih10 = stab(*PH, zol10, zol0)
    psih2 = stab(*PH, zol2, zol0)
    pq = stab(*PH, zol, zl)
    pq2 = stab(*PH, zl2, zl)
    pq10 = stab(*PH, zl10, zl)
    psih = xp.where(c4, mn(psih, 0.9 * gz1oz0), psih)
    psim = xp.where(c4, mn(psim, 0.9 * gz1oz0), psim)
    psih2 = xp.where(c4, mn(psih2, 0.9 * gz2oz0), psih2)
    psim10 = xp.where(c4, mn(psim10, 0.9 * gz10oz0), psim10)
    psih10 = xp.where(c4, mn(psih10, 0.9 * gz10oz0), psih10)
    zero = xp.zeros_like(br)
    regime = xp.where(c4, 4.0 + zero, xp.where(c3, 3.0 + zero, 1.0 + zero))
    zol = xp.where(c3, 0.0, zol)
    rmol = zol / za

    # loop 330
    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    psit = gz1oz0 - psih
    psit2 = gz2oz0 - psih2
    water = (xland - 1.5) >= 0
    zl = xp.where(water, znt, 0.01)
    psiq = xp.log(K * ust * za / XKA + za / zl) - pq
    psiq2 = xp.log(K * ust * 2.0 / XKA + 2.0 / zl) - pq2
    psiq10 = xp.log(K * ust * 10.0 / XKA + 10.0 / zl) - pq10
    visc = (1.32 + 0.009 * (scr3 - 273.15)) * 1.0e-5
    restar = ust * znt / visc
    z0t_n = (5.5e-5) * (xp.where(water, restar, 1.0) ** -0.60)
    z0t_n = mx(mn(z0t_n, 1.0e-4), 2.0e-9)
    z0q_n = z0t_n
    ph, ph10, ph2 = T.psih_for(zol, za, z0t_n)
    psit = xp.where(water, xp.log((za + z0t_n) / z0t_n) - ph, psit)
    psit2 = xp.where(water, xp.log((2.0 + z0t_n) / z0t_n) - ph2, psit2)
    ph, ph10, ph2 = T.psih_for(zol, za, z0q_n)
    psih = xp.where(water, ph, psih)
    psih10 = xp.where(water, ph10, psih10)
    psih2 = xp.where(water, ph2, psih2)
    psiq = xp.where(water, xp.log((za + z0q_n) / z0q_n) - psih, psiq)
    psiq2 = xp.where(water, xp.log((2.0 + z0q_n) / z0q_n) - psih2, psiq2)
    psiq10 = xp.where(water, xp.log((10.0 + z0q_n) / z0q_n) - psih10, psiq10)
    if isftcflx == 1:
        z0q_n = xp.where(water, 1.0e-4 + zero, z0q_n)
        ph, ph10, ph2 = T.psih_for(zol, za, z0q_n)
        psih = xp.where(water, ph, psih)
        psih10 = xp.where(water, ph10, psih10)
        psih2 = xp.where(water, ph2, psih2)
        psiq = xp.where(water, xp.log((za + z0q_n) / z0q_n) - psih, psiq)
        psit = xp.where(water, psiq, psit)
        psiq2 = xp.where(water, xp.log((2.0 + z0q_n) / z0q_n) - psih2, psiq2)
        psiq10 = xp.where(water, xp.log((10.0 + z0q_n) / z0q_n) - psih10, psiq10)
        psit2 = xp.where(water, psiq2, psit2)
    if isftcflx == 2:
        rs = xp.sqrt(xp.sqrt(xp.where(water, restar, 1.0)))
        gz0ozt = 0.40 * (7.3 * rs * math.sqrt(0.71) - 5.0)
        z0t_n = xp.where(water, znt / xp.exp(gz0ozt), z0t_n)
        ph, ph10, ph2 = T.psih_for(zol, za, z0t_n)
        psih = xp.where(water, ph, psih)
        psih10 = xp.where(water, ph10, psih10)
        psih2 = xp.where(water, ph2, psih2)
        psit = xp.where(water, xp.log((za + z0t_n) / z0t_n) - psih, psit)
        psit2 = xp.where(water, xp.log((2.0 + z0t_n) / z0t_n) - psih2, psit2)
        gz0ozq = 0.40 * (7.3 * rs * math.sqrt(0.60) - 5.0)
        z0q_n = xp.where(water, znt / xp.exp(gz0ozq), z0q_n)
        ph, ph10, ph2 = T.psih_for(zol, za, z0q_n)
        psih = xp.where(water, ph, psih)
        psih10 = xp.where(water, ph10, psih10)
        psih2 = xp.where(water, ph2, psih2)
        psiq = xp.where(water, xp.log((za + z0q_n) / z0q_n) - psih, psiq)
        psiq2 = xp.where(water, xp.log((2.0 + z0q_n) / z0q_n) - psih2, psiq2)
        psiq10 = xp.where(water, xp.log((10.0 + z0q_n) / z0q_n) - psih10, psiq10)
    out = {
        "ck": (K / psix10) * (K / psiq10),
        "cd": (K / psix10) * (K / psix10),
        "cka": (K / psix) * (K / psiq),
        "cda": (K / psix) * (K / psix),
    }
    z0t_o, z0q_o = xp.where(water, z0t_n, z0t), xp.where(water, z0q_n, z0q)
    if iz0tlnd >= 1:
        lnd = (xland - 1.5) <= 0.0
        rs = xp.sqrt(ust * znt / visc)
        czil = 10.0 ** (-0.40 * (znt / 0.07)) if iz0tlnd == 1 else 0.1
        z0tl = znt / xp.exp(czil * K * rs)
        ph, ph10, ph2 = T.psih_for(zol, za, z0tl)
        psih = xp.where(lnd, ph, psih)
        psih2 = xp.where(lnd, ph2, psih2)
        pql = xp.log((za + z0tl) / z0tl) - psih
        pql2 = xp.log((2.0 + z0tl) / z0tl) - psih2
        psiq = xp.where(lnd, pql, psiq)
        psiq2 = xp.where(lnd, pql2, psiq2)
        psit = xp.where(lnd, pql, psit)
        psit2 = xp.where(lnd, pql2, psit2)
    out["z0t"], out["z0q"] = z0t_o, z0q_o
    ust_n = 0.5 * ust + 0.5 * K * wspd / psix
    wspdi = xp.sqrt(ux * ux + vx * vx)
    out["ustm"] = 0.5 * ustm + 0.5 * K * wspdi / psix
    out["u10"] = ux * psix10 / psix
    out["v10"] = vx * psix10 / psix
    th2 = thgb + dtg * psit2 / psit
    out["th2"] = th2
    out["q2"] = qsfc + (qx - qsfc) * psiq2 / psiq
    out["t2"] = th2 * (psfcpa / P1000MB) ** rovcp
    ust_n = xp.where((xland - 1.5) < 0.0, mx(ust_n, 0.001), ust_n)
    mol_n = K * dtg / psit / PRT
    out["fm"] = psix
    out["fh"] = psit

    znt_w = mn(CZO * ust_n * ust_n / g + 0.11 * 1.5e-5 / ust_n, 2.85e-3)
    if isftcflx != 0:
        zw = mn((ust_n / 1.06) ** (0.3), 1.0)
        zn1 = 0.011 * ust_n * ust_n / g + OZO
        zn2 = 10.0 * xp.exp(-9.5 * ust_n ** (-0.3333)) + 0.11 * 1.5e-5 / mx(ust_n, 0.01)
        znt_w = mx(mn((1.0 - zw) * zn1 + zw * zn2, 2.85e-3), 1.27e-7)
    znt_n = xp.where(water, znt_w, znt)
    flqc = rhox * mavail * ust_n * K / psiq
    dtthx = xp.abs(thx - thgb)
    dth = xp.where(dtthx > 1.0e-5, thx - thgb, 1.0)
    flhc = xp.where(dtthx > 1.0e-5, cpm * rhox * ust_n * mol_n / dth, 0.0)
    qfx_n = mx(flqc * (qsfc - qx), 0.0)
    h = flhc * (thgb - thx)
    hfx_n = xp.where(
        xland - 1.5 > 0.0, h, xp.where(xland - 1.5 < 0.0, mx(h, -250.0), 0.0)
    )
    out.update(
        qgh=qgh, flhc=flhc, flqc=flqc, cpmm=cpm, qsfc=qsfc, znt=znt_n, ust=ust_n,
        gz1oz0=gz1oz0, wspd=wspd, br=br, mol=mol_n, rmol=rmol, regime=regime,
        psim=psim, psih=psih, zol=zol, hfx=hfx_n, qfx=qfx_n, lh=C.xlv * qfx_n,
        chs=ust_n * K / psiq, cqs2=ust_n * K / psiq2, chs2=ust_n * K / psit2,
    )  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# registry hook


def surface_layer(ctx):
    """turb.F sfc_and_turb, sfcmodel = 2 / 3: SFCLAY or sf_sfclayrev on the
    k = 1 WRF inputs, then ``pbl.wrf_sfc_epilogue`` (SLAB or the qsfc
    diagnosis, ch / cq / s10, thflux / qvflux)."""
    from .pbl import interior2, wrf_sfc_epilogue

    bk, g, cfg, s = ctx.bk, ctx.g, ctx.cfg, ctx.s
    k1, n = g.ngz, g.ngxy
    J, I = slice(n, n + g.nj), slice(n, n + g.ni)
    w = ctx.w
    i2 = {k: interior2(g, s[k]) for k in ("psfc", "tsk", "xland", "mavail", "qsfc", "znt", "ust", "mol",
                                          "zol", "hfx", "qfx", "ustm", "z0t", "z0q")}  # fmt: skip
    args = dict(
        ux=w["u"][k1, J, I], vx=w["v"][k1, J, I], t1d=w["t"][k1, J, I], qv1d=w["qv"][k1, J, I],
        p1d=ctx.prs[k1, J, I], dz8w1d=w["dz8w"][k1, J, I], psfcpa=i2.pop("psfc"),
        pblh=interior2(g, s["hpbl"]), dx=interior2(g, s["dsxy"]), **i2,
    )  # fmt: skip
    if int(cfg.sfcmodel) == 2:
        out = sfclay(bk, cfg, rhox=ctx.rf[k1, J, I], **args)
    else:
        out = sfclayrev(bk, cfg, **args)
    return wrf_sfc_epilogue(ctx, out)
