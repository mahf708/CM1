"""Thompson microphysics constants and lookup tables (src/thompson.F init).

Ported from ``thompson_init`` (constants, size bins) and the table
routines ``table_Efrw``, ``table_Efsw``, ``table_dropEvap``, ``qr_acr_qg``,
``qr_acr_qs``, ``freezeH2O``, ``qi_aut_qs`` plus their helpers ``GAMMLN``,
``GSER``, ``GCF``, ``GAMMP``, ``WGAMMA``.  ``table_ccnAct`` (and the
aerosol-aware path) is not used by CM1: ``thompson_init`` is called without
nwfa/nifa, so ``is_aerosol_aware`` is false.

Everything here is numpy float64, computed once at init and then handed to
the array backend by ``cm1x.thompson.thompson_init``.  The goal is bitwise
equality with CM1's double-precision tables:

* transcendental functions of scalars go through Python's ``math`` module,
  i.e. the same glibc libm entry points (exp, log, pow, atan, ...) the
  Fortran calls (src/thompson.F carries ``!GCC$ NOVECTOR`` on the table
  loops so that gfortran does not switch them to libmvec);
* every expression keeps Fortran's left-to-right evaluation order, and the
  long collection sums keep the Fortran loop order (vectorised over the
  table indices only, which never mixes summation order);
* ``real ** integer`` is libgcc's ``__powidf2`` (square-and-multiply),
  ``x ** 2.0`` with a constant exponent is ``x*x`` (as GCC folds it).

The 4-D collection tables cost ~1-2 min of numpy time; ``tables()`` caches
the result in ``$CM1X_THOMPSON_CACHE`` (default ``~/.cache/cm1x``) keyed by
``TABLE_VERSION``.

Arrays keep the Fortran index order with 0-based indices, e.g.
``tcg_racg[i, j, k, m]`` is Fortran ``tcg_racg(i+1, j+1, k+1, m+1)``.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

TABLE_VERSION = 1

# ---------------------------------------------------------------------------
# module parameters (thompson.F declarations; double precision with -DDP)

T_0 = 273.15
PI = 3.1415926536
rho_w = 1000.0
rho_s = 100.0
rho_g = 500.0
rho_i = 890.0
Nt_c_max = 1999.0e6
naIN0 = 1.5e6
naIN1 = 0.5e6
naCCN0 = 300.0e6
naCCN1 = 50.0e6
mu_r = 0.0
mu_g = 0.0
mu_i = 0.0
mu_s = 0.6357
Kap0 = 490.6
Kap1 = 17.46
Lam0 = 20.78
Lam1 = 3.29
gonv_min = 1.0e2
gonv_max = 1.0e6
am_r = PI * rho_w / 6.0
bm_r = 3.0
am_s = 0.069
bm_s = 2.0
am_g = PI * rho_g / 6.0
bm_g = 3.0
am_i = PI * rho_i / 6.0
bm_i = 3.0
av_r = 4854.0
bv_r = 1.0
fv_r = 195.0
av_s = 40.0
bv_s = 0.55
fv_s = 100.0
av_g = 442.0
bv_g = 0.89
av_i = 1847.5
bv_i = 1.0
av_c = 0.316946e8
bv_c = 2.0
C_cube = 0.5
C_sqrd = 0.15
Ef_si = 0.05
Ef_rs = 0.95
Ef_rg = 0.75
Ef_ri = 0.95
R1 = 1.0e-12
R2 = 1.0e-6
eps = 1.0e-15
TNO = 5.0
ATO = 0.304
rho_not = 101325.0 / (287.05 * 298.0)
Sc = 0.632
HGFR = 235.16
Rv = 461.5
oRv = 1.0 / Rv
R = 287.04
Cp = 1004.0
cv = Cp - R
cvv = 1408.5
cpl = 4190.0
cpi = 2106.0
lsub = 2.834e6
lvap0 = 2.5e6
lfus = lsub - lvap0
olfus = 1.0 / lfus
xm0i = 1.0e-12
D0c = 1.0e-6
D0r = 50.0e-6
D0s = 200.0e-6
D0g = 250.0e-6
IFDRY = 0
RE_QC_BG = 2.49e-6
RE_QI_BG = 4.99e-6
RE_QS_BG = 9.99e-6

nbins = 100
nbc = nbi = nbr = nbs = nbg = nbins
ntb_c = 37
ntb_i = 64
ntb_r = 37
ntb_s = 37
ntb_g = 37
ntb_g1 = 37
ntb_r1 = 37
ntb_i1 = 55
ntb_t = 9
ntb_IN = 55


def _decades(first_exp, ndec, last):
    """1,2,..9 x 10**e for ndec decades, then ``last`` (the r_c/N0r_exp/...
    parameter arrays, exactly as the Fortran literals ``k.e<e>``)."""
    out = []
    for e in range(first_exp, first_exp + ndec):
        out += [float(f"{d}.e{e}") for d in range(1, 10)]
    return np.array(out + [last])


r_c = _decades(-6, 4, 1.0e-2)
r_i = _decades(-10, 7, 1.0e-3)
r_r = _decades(-6, 4, 1.0e-2)
r_g = _decades(-6, 4, 1.0e-2)
r_s = _decades(-6, 4, 1.0e-2)
N0r_exp = _decades(6, 4, 1.0e10)
N0g_exp = _decades(2, 4, 1.0e6)
Nt_i = _decades(0, 6, 1.0e6)
Nt_IN = _decades(0, 6, 1.0e6)
assert len(r_i) == ntb_i and len(Nt_i) == ntb_i1 and len(r_c) == ntb_c

sa = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204,
      0.003199, 0.0, -0.015952)  # fmt: skip
sb = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079,
      0.000594, 0.0, -0.003577)  # fmt: skip
Tc = (-0.01, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0)

# ---------------------------------------------------------------------------
# scalar helpers (libm through ``math``)


def nint(x):
    """Fortran NINT (round half away from zero) of a float."""
    r = math.floor(x)
    if x - r >= 0.5:
        r += 1
    return int(r)


def powi(x, m):
    """libgcc ``__powidf2``: Fortran ``real ** integer``."""
    n = -m if m < 0 else m
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def _v(f, *args):
    """Elementwise scalar libm function on numpy arrays (float64)."""
    b = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in args])
    out = np.empty(b[0].shape)
    flat = [a.ravel() for a in b]
    o = out.ravel()
    for n in range(o.size):
        o[n] = f(*(a[n] for a in flat))
    return out


def gammln(xx):
    cof = (76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)  # fmt: skip
    stp = 2.5066282746310005
    x = xx
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.000000000190015
    for c in cof:
        y = y + 1.0
        ser = ser + c / y
    return tmp + math.log(stp * ser / x)


def gser(a, x):
    gln = gammln(a)
    if x <= 0.0:
        return 0.0, gln
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(100):
        ap = ap + 1.0
        d = d * x / ap
        s = s + d
        if abs(d) < abs(s) * 3.0e-7:
            break
    return s * math.exp(-x + a * math.log(x) - gln), gln


def gcf(a, x):
    fpmin = 1.0e-30
    gln = gammln(a)
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 101):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h = h * de
        if abs(de - 1.0) < 3.0e-7:
            break
    return math.exp(-x + a * math.log(x) - gln) * h, gln


def gammp(a, x):
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x < a + 1.0:
        return gser(a, x)[0]
    return 1.0 - gcf(a, x)[0]


def wgamma(y):
    return math.exp(gammln(y))


# ---------------------------------------------------------------------------


def init_constants(nt_c_in=100.0):
    """The scalar/vector constants set by ``thompson_init`` (micro_init
    block, before the tables).  ``nt_c_in`` is the namelist ``nt_c`` (per
    cc).  Returns a dict keyed by the Fortran variable names (arrays
    0-based, ``ccg[c, n]`` is Fortran ``ccg(c+1, n+1)``)."""
    c = {}
    Nt_c = 1.0e6 * nt_c_in
    c["Nt_c"] = Nt_c
    c["mu_c"] = min(15.0, (1000.0e6 / Nt_c + 2.0))
    c["Sc3"] = Sc ** (1.0 / 3.0)
    c["D0i"] = (xm0i / am_i) ** (1.0 / bm_i)
    c["xm0s"] = am_s * (D0s * D0s)
    # D0g**bm_g is a compile-time constant (MPFR, correctly rounded)
    c["xm0g"] = am_g * _cr_pow_int(D0g, 3)

    cce = np.zeros((5, 15))
    ccg = np.zeros((5, 15))
    ocg1 = np.zeros(15)
    ocg2 = np.zeros(15)
    for i in range(15):
        n = i + 1
        cce[0, i] = n + 1.0
        cce[1, i] = bm_r + n + 1.0
        cce[2, i] = bm_r + n + 4.0
        cce[3, i] = n + bv_c + 1.0
        cce[4, i] = bm_r + n + bv_c + 1.0
        for r in range(5):
            ccg[r, i] = wgamma(cce[r, i])
        ocg1[i] = 1.0 / ccg[0, i]
        ocg2[i] = 1.0 / ccg[1, i]
    c.update(cce=cce, ccg=ccg, ocg1=ocg1, ocg2=ocg2)

    cie = [
        mu_i + 1.0,
        bm_i + mu_i + 1.0,
        bm_i + mu_i + bv_i + 1.0,
        mu_i + bv_i + 1.0,
        mu_i + 2.0,
        bm_i * 0.5 + mu_i + bv_i + 1.0,
        bm_i * 0.5 + mu_i + 1.0,
    ]
    cig = [wgamma(x) for x in cie]
    c.update(cie=np.array(cie), cig=np.array(cig))
    c["oig1"] = 1.0 / cig[0]
    c["oig2"] = 1.0 / cig[1]
    c["obmi"] = 1.0 / bm_i

    cre = [
        bm_r + 1.0,
        mu_r + 1.0,
        bm_r + mu_r + 1.0,
        bm_r * 2.0 + mu_r + 1.0,
        mu_r + bv_r + 1.0,
        bm_r + mu_r + bv_r + 1.0,
        bm_r * 0.5 + mu_r + bv_r + 1.0,
        bm_r + mu_r + bv_r + 3.0,
        mu_r + bv_r + 3.0,
        mu_r + 2.0,
        0.5 * (bv_r + 5.0 + 2.0 * mu_r),
        bm_r * 0.5 + mu_r + 1.0,
        bm_r * 2.0 + mu_r + bv_r + 1.0,
    ]
    crg = [wgamma(x) for x in cre]
    c.update(cre=np.array(cre), crg=np.array(crg))
    c["obmr"] = 1.0 / bm_r
    c["ore1"] = 1.0 / cre[0]
    c["org1"] = 1.0 / crg[0]
    c["org2"] = 1.0 / crg[1]
    c["org3"] = 1.0 / crg[2]

    cse = [0.0] * 18
    cse[0] = bm_s + 1.0
    cse[1] = bm_s + 2.0
    cse[2] = bm_s * 2.0
    cse[3] = bm_s + bv_s + 1.0
    cse[4] = bm_s * 2.0 + bv_s + 1.0
    cse[5] = bm_s * 2.0 + 1.0
    cse[6] = bm_s + mu_s + 1.0
    cse[7] = bm_s + mu_s + 2.0
    cse[8] = bm_s + mu_s + 3.0
    cse[9] = bm_s + mu_s + bv_s + 1.0
    cse[10] = bm_s * 2.0 + mu_s + bv_s + 1.0
    cse[11] = bm_s * 2.0 + mu_s + 1.0
    cse[12] = bv_s + 2.0
    cse[13] = bm_s + bv_s
    cse[14] = mu_s + 1.0
    cse[15] = 1.0 + (1.0 + bv_s) / 2.0
    cse[16] = cse[15] + mu_s + 1.0
    cse[17] = bv_s + mu_s + 3.0
    csg = [wgamma(x) for x in cse]
    c.update(cse=np.array(cse), csg=np.array(csg))
    c["oams"] = 1.0 / am_s
    c["obms"] = 1.0 / bm_s
    c["ocms"] = c["oams"] ** c["obms"]

    cge = [
        bm_g + 1.0,
        mu_g + 1.0,
        bm_g + mu_g + 1.0,
        bm_g * 2.0 + mu_g + 1.0,
        bm_g * 2.0 + mu_g + bv_g + 1.0,
        bm_g + mu_g + bv_g + 1.0,
        bm_g + mu_g + bv_g + 2.0,
        bm_g + mu_g + bv_g + 3.0,
        mu_g + bv_g + 3.0,
        mu_g + 2.0,
        0.5 * (bv_g + 5.0 + 2.0 * mu_g),
        0.5 * (bv_g + 5.0) + mu_g,
    ]
    cgg = [wgamma(x) for x in cge]
    c.update(cge=np.array(cge), cgg=np.array(cgg))
    c["oamg"] = 1.0 / am_g
    c["obmg"] = 1.0 / bm_g
    c["ocmg"] = c["oamg"] ** c["obmg"]
    c["oge1"] = 1.0 / cge[0]
    c["ogg1"] = 1.0 / cgg[0]
    c["ogg2"] = 1.0 / cgg[1]
    c["ogg3"] = 1.0 / cgg[2]

    Sc3 = c["Sc3"]
    # SQRT(av_x) of a parameter: compile-time constant, correctly rounded
    c["t1_qr_qc"] = PI * 0.25 * av_r * crg[8]
    c["t1_qr_qi"] = PI * 0.25 * av_r * crg[8]
    c["t2_qr_qi"] = PI * 0.25 * am_r * av_r * crg[7]
    c["t1_qg_qc"] = PI * 0.25 * av_g * cgg[8]
    c["t1_qs_qc"] = PI * 0.25 * av_s
    c["t1_qs_qi"] = PI * 0.25 * av_s
    c["t1_qr_ev"] = 0.78 * crg[9]
    c["t2_qr_ev"] = 0.308 * Sc3 * math.sqrt(av_r) * crg[10]
    c["t1_qs_sd"] = 0.86
    c["t2_qs_sd"] = 0.28 * Sc3 * math.sqrt(av_s)
    c["t1_qs_me"] = PI * 4.0 * C_sqrd * olfus * 0.86
    c["t2_qs_me"] = PI * 4.0 * C_sqrd * olfus * 0.28 * Sc3 * math.sqrt(av_s)
    c["t1_qg_sd"] = 0.86 * cgg[9]
    c["t2_qg_sd"] = 0.28 * Sc3 * math.sqrt(av_g) * cgg[10]
    c["t1_qg_me"] = PI * 4.0 * C_cube * olfus * 0.86 * cgg[9]
    c["t2_qg_me"] = PI * 4.0 * C_cube * olfus * 0.28 * Sc3 * math.sqrt(av_g) * cgg[10]

    c["nic2"] = nint(math.log10(r_c[0]))
    c["nii2"] = nint(math.log10(r_i[0]))
    c["nii3"] = nint(math.log10(Nt_i[0]))
    c["nir2"] = nint(math.log10(r_r[0]))
    c["nir3"] = nint(math.log10(N0r_exp[0]))
    c["nis2"] = nint(math.log10(r_s[0]))
    c["nig2"] = nint(math.log10(r_g[0]))
    c["nig3"] = nint(math.log10(N0g_exp[0]))
    c["niIN2"] = nint(math.log10(Nt_IN[0]))

    # size bins
    Dc = np.zeros(nbc)
    dtc = np.zeros(nbc)
    Dc[0] = D0c
    dtc[0] = D0c
    for n in range(1, nbc):
        Dc[n] = Dc[n - 1] + 1.0e-6
        dtc[n] = Dc[n] - Dc[n - 1]
    c["Dc"], c["dtc"] = Dc, dtc

    def bins(x1, xn, nb):
        x = np.zeros(nb + 1)
        x[0] = x1
        x[nb] = xn
        lr = math.log(x[nb] / x[0])
        l1 = math.log(x[0])
        for n in range(1, nb):
            x[n] = math.exp(float(n) / float(nb) * lr + l1)
        D = np.sqrt(x[:-1] * x[1:])
        return D, x[1:] - x[:-1]

    c["Di"], c["dti"] = bins(c["D0i"], 5.0 * D0s, nbi)
    c["Dr"], c["dtr"] = bins(D0r, 0.005, nbr)
    c["Ds"], c["dts"] = bins(D0s, 0.02, nbs)
    c["Dg"], c["dtg"] = bins(D0g, 0.05, nbg)
    tn, _ = bins(1.0, 3000.0, nbc)
    c["t_Nc"] = tn * 1.0e6
    c["nic1"] = int(math.log(c["t_Nc"][nbc - 1] / c["t_Nc"][0]))
    return c


def _cr_pow_int(x, n):
    """Correctly rounded x**n (GCC folds a constant ``x**3.0`` with MPFR)."""
    from fractions import Fraction

    return float(Fraction(x) ** n)


def _pow3(x):
    """``x**bm`` with the PARAMETER bm=3.0 on a variable: gfortran calls
    ``pow(x, 3.0)``."""
    return _v(lambda a: math.pow(a, 3.0), x)


# ---------------------------------------------------------------------------


def table_Efrw(c):
    Dc, Dr = c["Dc"], c["Dr"]
    t = np.zeros((nbr, nbc))
    for j in range(nbc):
        for i in range(nbr):
            ef = 0.0
            p = Dc[j] / Dr[i]
            if Dr[i] < 50.0e-6 or Dc[j] < 3.0e-6:
                ef = 0.0
            elif p > 0.25:
                X = Dc[j] * 1.0e6
                d = Dr[i]
                if d < 75.0e-6:
                    ef = 0.026794 * X - 0.20604
                elif d < 125.0e-6:
                    ef = -0.00066842 * X * X + 0.061542 * X - 0.37089
                elif d < 175.0e-6:
                    ef = (4.091e-06 * X * X * X * X - 0.00030908 * X * X * X
                          + 0.0066237 * X * X - 0.0013687 * X - 0.073022)  # fmt: skip
                elif d < 250.0e-6:
                    ef = (
                        9.6719e-5 * X * X * X
                        - 0.0068901 * X * X
                        + 0.17305 * X
                        - 0.65988
                    )
                elif d < 350.0e-6:
                    ef = (
                        9.0488e-5 * X * X * X - 0.006585 * X * X + 0.16606 * X - 0.56125
                    )
                else:
                    ef = (
                        0.00010721 * X * X * X
                        - 0.0072962 * X * X
                        + 0.1704 * X
                        - 0.46929
                    )
            else:
                d = Dr[i]
                vtr = (-0.1021 + 4.932e3 * d - 0.9551e6 * d * d
                       + 0.07934e9 * d * d * d - 0.002362e12 * d * d * d * d)  # fmt: skip
                stokes = Dc[j] * Dc[j] * vtr * rho_w / (9.0 * 1.718e-5 * d)
                reynolds = 9.0 * stokes / (p * p * rho_w)
                ef = _bg_eff(stokes, reynolds, p)
            t[i, j] = max(0.0, min(ef, 0.95))
    return t


def _bg_eff(stokes, reynolds, p):
    F = math.log(reynolds)
    G = -0.1007 - 0.358 * F + 0.0261 * F * F
    K0 = math.exp(G)
    z = math.log(stokes / (K0 + 1.0e-15))
    H = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z
    yc0 = 2.0 / PI * math.atan(H)
    return (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p))


def table_Efsw(c):
    Dc, Ds, obmr = c["Dc"], c["Ds"], c["obmr"]
    t = np.zeros((nbs, nbc))
    for j in range(nbc):
        vtc = 1.19e4 * (1.0e4 * Dc[j] * Dc[j] * 0.25)
        for i in range(nbs):
            d = Ds[i]
            vts = av_s * math.pow(d, bv_s) * math.exp(-fv_s * d) - vtc
            Ds_m = math.pow(am_s * (d * d) / am_r, obmr)
            p = Dc[j] / Ds_m
            if p > 0.25 or d < D0s or Dc[j] < 6.0e-6 or vts < 1.0e-3:
                t[i, j] = 0.0
            else:
                stokes = Dc[j] * Dc[j] * vts * rho_w / (9.0 * 1.718e-5 * Ds_m)
                reynolds = 9.0 * stokes / (p * p * rho_w)
                t[i, j] = max(0.0, min(_bg_eff(stokes, reynolds, p), 0.95))
    return t


def table_dropEvap(c):
    """tpc_wev, tnc_wev [i, j, k] (only used by the aerosol-aware path)."""
    Dc, dtc, t_Nc = c["Dc"], c["dtc"], c["t_Nc"]
    massc = am_r * _pow3(Dc)
    tpc = np.zeros((nbc, ntb_c, nbc))
    tnc = np.zeros((nbc, ntb_c, nbc))
    for k in range(nbc):
        nu_c = min(15, nint(1000.0e6 / t_Nc[k]) + 2)
        for j in range(ntb_c):
            lamc = math.pow(
                t_Nc[k] * am_r * c["ccg"][1, nu_c - 1] * c["ocg1"][nu_c - 1] / r_c[j],
                c["obmr"],
            )
            N0_c = t_Nc[k] * c["ocg1"][nu_c - 1] * math.pow(lamc, c["cce"][0, nu_c - 1])
            Nc = np.array(
                [
                    N0_c * powi(Dc[i], nu_c) * math.exp(-lamc * Dc[i]) * dtc[i]
                    for i in range(nbc)
                ]
            )
            tpc[:, j, k] = np.cumsum(massc * Nc)
            tnc[:, j, k] = np.cumsum(Nc)
    return tpc, tnc


def _rain_dists(c):
    """N_r[k, m, n2] of qr_acr_qg/qr_acr_qs/freezeH2O (k: N0r_exp, m: r_r)."""
    Dr, dtr = c["Dr"], c["dtr"]
    out = np.zeros((ntb_r1, ntb_r, nbr))
    pw = math.pow(c["crg"][2] * c["org2"] * c["org1"], c["obmr"])
    for m in range(ntb_r):
        for k in range(ntb_r1):
            lam_exp = math.pow(N0r_exp[k] * am_r * c["crg"][0] / r_r[m], c["ore1"])
            lamr = lam_exp * pw
            N0_r = N0r_exp[k] / (c["crg"][1] * lam_exp) * math.pow(lamr, c["cre"][1])
            for n2 in range(nbr):
                out[k, m, n2] = (
                    N0_r * math.pow(Dr[n2], mu_r) * math.exp(-lamr * Dr[n2]) * dtr[n2]
                )
    return out


def _vr_poly(Dr):
    d = Dr
    return (-0.1021 + 4.932e3 * d - 0.9551e6 * d * d
            + 0.07934e9 * d * d * d - 0.002362e12 * d * d * d * d)  # fmt: skip


def _accumulate(coefs, A, B, shape):
    """Sequential collection sums: ``acc[q] = sum_{n2} sum_{n} coefs[q][n2, n]
    * A[:, n] * B[:, n2]`` (outer product over the A and B index sets), in
    Fortran loop order (n2 outer, n inner), terms evaluated as
    ``(coef * A) * B``.  Zero coefficients are skipped (adding +0 changes
    nothing)."""
    na, nb = A.shape[0], B.shape[0]
    acc = [np.zeros((na, nb)) for _ in coefs]
    buf = np.empty((na, nb))
    n2max, nmax = coefs[0].shape
    for n2 in range(n2max):
        b = B[:, n2][None, :]
        for n in range(nmax):
            a = A[:, n]
            for q, cf in enumerate(coefs):
                cq = cf[n2, n]
                if cq == 0.0:
                    continue
                np.multiply((cq * a)[:, None], b, out=buf)
                np.add(acc[q], buf, out=acc[q])
    return [x.reshape(shape) for x in acc]


def qr_acr_qg(c, Nr=None):
    """tcg_racg, tmr_racg, tcr_gacr, tmg_gacr, tnr_racg, tnr_gacr, each
    [i, j, k, m] = (N0g_exp, r_g, N0r_exp, r_r)."""
    Dr, Dg, dtg = c["Dr"], c["Dg"], c["dtg"]
    vr = _vr_poly(Dr)
    vg = _v(lambda d: av_g * math.pow(d, bv_g), Dg)
    if Nr is None:
        Nr = _rain_dists(c)
    # graupel distributions N_g[i, j, n]
    Ng = np.zeros((ntb_g1, ntb_g, nbg))
    pw = math.pow(c["cgg"][2] * c["ogg2"] * c["ogg1"], c["obmg"])
    for j in range(ntb_g):
        for i in range(ntb_g1):
            lam_exp = math.pow(N0g_exp[i] * am_g * c["cgg"][0] / r_g[j], c["oge1"])
            lamg = lam_exp * pw
            N0_g = N0g_exp[i] / (c["cgg"][1] * lam_exp) * math.pow(lamg, c["cge"][1])
            for n in range(nbg):
                Ng[i, j, n] = (
                    N0_g * math.pow(Dg[n], mu_g) * math.exp(-lamg * Dg[n]) * dtg[n]
                )
    massr = am_r * _pow3(Dr)
    massg = am_g * _pow3(Dg)
    s = Dg[None, :] + Dr[:, None]  # [n2, n]
    dvg = 0.5 * ((vr[:, None] - vg[None, :]) + np.abs(vr[:, None] - vg[None, :]))
    dvr = 0.5 * ((vg[None, :] - vr[:, None]) + np.abs(vg[None, :] - vr[:, None]))
    c0 = PI * 0.25 * Ef_rg
    bg = c0 * s * s * dvg
    br = c0 * s * s * dvr
    coefs = [
        bg * massg[None, :],  # t1
        bg * massr[:, None],  # z1
        bg,  # y1
        br * massr[:, None],  # t2
        br,  # y2
        br * massg[None, :],  # z2
    ]
    t1, z1, y1, t2, y2, z2 = _accumulate(
        coefs,
        Ng.reshape(ntb_g1 * ntb_g, nbg),
        Nr.reshape(ntb_r1 * ntb_r, nbr),
        (ntb_g1, ntb_g, ntb_r1, ntb_r),
    )
    return dict(
        tcg_racg=t1,
        tmr_racg=np.minimum(z1, r_r[None, None, None, :]),
        tcr_gacr=t2,
        tmg_gacr=np.minimum(z2, r_g[None, :, None, None]),
        tnr_racg=y1,
        tnr_gacr=y2,
    )


def _snow_moment(tc, cexp):
    """a_ and b_ of the Field et al. (2005) moment relation for moment
    ``cexp`` (the general sa/sb polynomials)."""
    loga_ = (sa[0] + sa[1] * tc + sa[2] * cexp + sa[3] * tc * cexp + sa[4] * tc * tc
             + sa[5] * cexp * cexp + sa[6] * tc * tc * cexp + sa[7] * tc * cexp * cexp
             + sa[8] * tc * tc * tc + sa[9] * cexp * cexp * cexp)  # fmt: skip
    b_ = (sb[0] + sb[1] * tc + sb[2] * cexp + sb[3] * tc * cexp + sb[4] * tc * tc
          + sb[5] * cexp * cexp + sb[6] * tc * tc * cexp + sb[7] * tc * cexp * cexp
          + sb[8] * tc * tc * tc + sb[9] * cexp * cexp * cexp)  # fmt: skip
    return math.pow(10.0, loga_), b_


def qr_acr_qs(c, Nr=None):
    """The 12 rain/snow collection tables, each [i, j, k, m] =
    (r_s, Tc, N0r_exp, r_r)."""
    Dr, Ds, dts = c["Dr"], c["Ds"], c["dts"]
    vr = _vr_poly(Dr)
    vs = _v(lambda d: 1.5 * av_s * math.pow(d, bv_s) * math.exp(-fv_s * d), Ds)
    if Nr is None:
        Nr = _rain_dists(c)
    Ns = np.zeros((ntb_s, ntb_t, nbs))
    cse1 = c["cse"][0]
    for j in range(ntb_t):
        for i in range(ntb_s):
            M2 = r_s[i] * c["oams"] * 1.0
            if 2.0 - 1.0e-3 < bm_s < 2.0 + 1.0e-3:
                a_, b_ = _snow_moment(Tc[j], bm_s)
                second = math.pow(M2 / a_, 1.0 / b_)
            else:
                second = M2
            a_, b_ = _snow_moment(Tc[j], cse1)
            M3 = a_ * math.pow(second, b_)
            oM3 = 1.0 / M3
            Mrat = M2 * (M2 * oM3) * (M2 * oM3) * (M2 * oM3)
            M0 = math.pow(M2 * oM3, mu_s)
            slam1 = M2 * oM3 * Lam0
            slam2 = M2 * oM3 * Lam1
            for n in range(nbs):
                d = Ds[n]
                Ns[i, j, n] = (
                    Mrat
                    * (
                        Kap0 * math.exp(-slam1 * d)
                        + Kap1 * M0 * math.pow(d, mu_s) * math.exp(-slam2 * d)
                    )
                    * dts[n]
                )
    massr = am_r * _pow3(Dr)
    masss = am_s * (Ds * Ds)
    s = Ds[None, :] + Dr[:, None]
    dvs = 0.5 * ((vr[:, None] - vs[None, :]) + np.abs(vr[:, None] - vs[None, :]))
    dvr = 0.5 * ((vs[None, :] - vr[:, None]) + np.abs(vs[None, :] - vr[:, None]))
    big = massr[:, None] > 1.5 * masss[None, :]
    c0 = PI * 0.25 * Ef_rs
    bs = c0 * s * s * dvs
    br = c0 * s * s * dvr
    z = np.zeros_like(bs)

    def sel(mask, a):
        return np.where(mask, a, z)

    ms = np.broadcast_to(masss[None, :], bs.shape)
    mr = np.broadcast_to(massr[:, None], bs.shape)
    coefs = [
        sel(big, bs * ms),  # t1
        sel(big, bs * mr),  # z1
        sel(big, bs),  # y1
        sel(~big, bs * ms),  # t3
        sel(~big, bs * mr),  # z3
        sel(~big, bs),  # y3
        sel(big, br * mr),  # t2
        sel(big, br),  # y2
        sel(big, br * ms),  # z2
        sel(~big, br * mr),  # t4
        sel(~big, br),  # y4
        sel(~big, br * ms),  # z4
    ]
    t1, z1, y1, t3, z3, y3, t2, y2, z2, t4, y4, z4 = _accumulate(
        coefs,
        Ns.reshape(ntb_s * ntb_t, nbs),
        Nr.reshape(ntb_r1 * ntb_r, nbr),
        (ntb_s, ntb_t, ntb_r1, ntb_r),
    )
    return dict(
        tcs_racs1=t1,
        tmr_racs1=np.minimum(z1, r_r[None, None, None, :]),
        tcs_racs2=t3,
        tmr_racs2=z3,
        tcr_sacr1=t2,
        tms_sacr1=z2,
        tcr_sacr2=t4,
        tms_sacr2=z4,
        tnr_racs1=y1,
        tnr_racs2=y3,
        tnr_sacr1=y2,
        tnr_sacr2=y4,
    )


def freezeH2O(c, Nr=None):
    """tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz [i, j, k, m] = (r_r, N0r_exp,
    -T, Nt_IN) and tpi_qcfz, tni_qcfz [i, j, k, m] = (r_c, t_Nc, -T, Nt_IN).
    The per-entry early EXIT of the bin sums is a mask."""
    Dr, Dc, dtc, t_Nc = c["Dr"], c["Dc"], c["dtc"], c["t_Nc"]
    if Nr is None:
        Nr = _rain_dists(c)
    orho_w = 1.0 / rho_w
    massr = am_r * _pow3(Dr)
    massc = am_r * _pow3(Dc)
    xm0g = c["xm0g"]
    Texp = np.zeros((45, ntb_IN))
    for m in range(ntb_IN):
        T_adjust = max(-3.0, min(3.0 - math.log10(Nt_IN[m]), 3.0))
        for k in range(45):
            Texp[k, m] = math.exp(float(k + 1) - T_adjust * 1.0) - 1.0

    def prob_of(vol):
        return np.maximum(
            0.0, 1.0 - _v(lambda t: math.exp(-120.0 * vol * 5.2e-4 * t), Texp)
        )

    # rain: entries [i (r_r), j (N0r_exp), k, m]
    NrT = np.transpose(Nr, (1, 0, 2))  # [m(r_r)->i, k(N0r)->j, n2]
    shp = (ntb_r, ntb_r1, 45, ntb_IN)
    s1, s2, sn1, sn2 = (np.zeros(shp) for _ in range(4))
    act = np.ones(shp, dtype=bool)
    rr = r_r[:, None, None, None]
    for n2 in range(nbr - 1, -1, -1):
        Nn = NrT[:, :, n2][:, :, None, None]
        prob = prob_of(massr[n2] * orho_w)[None, None, :, :]
        pn = prob * Nn
        pnm = pn * massr[n2]
        if massr[n2] < xm0g:
            sn1 = np.where(act, sn1 + pn, sn1)
            s1 = np.where(act, s1 + pnm, s1)
        else:
            sn2 = np.where(act, sn2 + pn, sn2)
            s2 = np.where(act, s2 + pnm, s2)
        act = act & ~((s1 + s2) >= rr)
    out = dict(tpi_qrfz=s1, tni_qrfz=sn1, tpg_qrfz=s2, tnr_qrfz=sn2)

    # cloud: entries [i (r_c), j (t_Nc), k, m]
    shp = (ntb_c, nbc, 45, ntb_IN)
    Nc = np.zeros((ntb_c, nbc, nbc))
    for j in range(nbc):
        nu_c = min(15, nint(1000.0e6 / t_Nc[j]) + 2)
        for i in range(ntb_c):
            lamc = math.pow(
                t_Nc[j] * am_r * c["ccg"][1, nu_c - 1] * c["ocg1"][nu_c - 1] / r_c[i],
                c["obmr"],
            )
            N0_c = t_Nc[j] * c["ocg1"][nu_c - 1] * math.pow(lamc, c["cce"][0, nu_c - 1])
            for n in range(nbc):
                Nc[i, j, n] = (
                    N0_c * powi(Dc[n], nu_c) * math.exp(-lamc * Dc[n]) * dtc[n]
                )
    s1 = np.zeros(shp)
    sn2 = np.zeros(shp)
    act = np.ones(shp, dtype=bool)
    tn = t_Nc[None, :, None, None]
    rc = r_c[:, None, None, None]
    for n in range(nbc - 1, -1, -1):
        prob = prob_of(massc[n] * orho_w)[None, None, :, :]
        pn = prob * Nc[:, :, n][:, :, None, None]
        sn2 = np.where(act, np.minimum(tn, sn2 + pn), sn2)
        s1 = np.where(act, s1 + pn * massc[n], s1)
        act = act & ~(s1 >= rc)
    out.update(tpi_qcfz=s1, tni_qcfz=sn2)
    return out


def qi_aut_qs(c):
    """tps_iaus, tni_iaus, tpi_ide [i, j] = (r_i, Nt_i)."""
    Di, dti = c["Di"], c["dti"]
    tps = np.zeros((ntb_i, ntb_i1))
    tni = np.zeros((ntb_i, ntb_i1))
    tpi = np.zeros((ntb_i, ntb_i1))
    for j in range(ntb_i1):
        for i in range(ntb_i):
            lami = math.pow(
                am_i * c["cig"][1] * c["oig1"] * Nt_i[j] / r_i[i], c["obmi"]
            )
            Di_mean = (bm_i + mu_i + 1.0) / lami
            N0_i = Nt_i[j] * c["oig1"] * math.pow(lami, c["cie"][0])
            t1 = t2 = 0.0
            if Di_mean > 5.0 * D0s:
                t1 = r_i[i]
                t2 = Nt_i[j]
                tpi[i, j] = 0.0
            elif Di_mean < c["D0i"]:
                tpi[i, j] = 1.0
            else:
                xlimit_intg = lami * D0s
                tpi[i, j] = gammp(mu_i + 2.0, xlimit_intg) * 1.0
                for n2 in range(nbi):
                    Ni = (
                        N0_i
                        * math.pow(Di[n2], mu_i)
                        * math.exp(-lami * Di[n2])
                        * dti[n2]
                    )
                    if Di[n2] >= D0s:
                        t1 = t1 + Ni * am_i * math.pow(Di[n2], bm_i)
                        t2 = t2 + Ni
            tps[i, j] = t1
            tni[i, j] = t2
    return dict(tps_iaus=tps, tni_iaus=tni, tpi_ide=tpi)


# ---------------------------------------------------------------------------


def compute_tables(nt_c_in=100.0, which=None):
    """All constants and tables as one dict of numpy arrays / scalars."""
    c = init_constants(nt_c_in)
    out = dict(c)
    out["t_Efrw"] = table_Efrw(c)
    out["t_Efsw"] = table_Efsw(c)
    out["tpc_wev"], out["tnc_wev"] = table_dropEvap(c)
    Nr = _rain_dists(c)
    out.update(qr_acr_qg(c, Nr))
    out.update(qr_acr_qs(c, Nr))
    out.update(freezeH2O(c, Nr))
    out.update(qi_aut_qs(c))
    return out


def _cache_dir():
    d = os.environ.get("CM1X_THOMPSON_CACHE")
    return Path(d) if d else Path.home() / ".cache" / "cm1x"


def tables(nt_c_in=100.0, cache=True):
    """``compute_tables`` with an on-disk cache (npz)."""
    key = hashlib.sha1(
        f"v{TABLE_VERSION}:{float(nt_c_in)!r}:{np.__version__}".encode()
    ).hexdigest()[:12]
    path = _cache_dir() / f"thompson_tables_{key}.npz"
    if cache and path.exists():
        with np.load(path) as z:
            return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    out = compute_tables(nt_c_in)
    if cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{k: np.asarray(v) for k, v in out.items()})
            os.replace(tmp, path)
        except OSError:
            pass
    return out
