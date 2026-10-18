"""Morrison two-moment microphysics (ptype=5), ported from src/morrison.F
(module_mp_graupel, V3.6 as modified for CM1) plus the ptype=5 branch of
src/mp_driver.F.

Routines (one per Fortran subroutine/function, same names):

* ``graupel_init``      GRAUPEL_INIT: the scheme's constants (a dict of
  Python floats, computed with the Fortran's operation order; the radar
  tables are not needed on the step path and are not built).
* ``gamma``, ``derf1``  the scheme's GAMMA (Cody & Stoltz) and DERF1
  (table-based erf), vectorised.
* ``satadj2``           the CM1 iterative saturation adjustment.
* ``m2005micro_graupel`` the column microphysics, vectorised over every
  column and level at once (``[k, j, i]`` arrays without halos).
* ``mp_graupel``        MP_GRAUPEL: 3-D interface, precipitation and budgets.
* ``mp_driver_ptype5``  the ptype=5 branch of mp_driver.F (pdefqtest, the
  call, efall, and the final theta/pressure update).

Switches.  GRAUPEL_INIT hard-wires IACT=2, IBASE=1, ISUB=1, ILIQ=0, INUC=0,
IGRAUP=0, IRAIN=0 and INUM=1; only IHAIL (namelist ``ihail``) and NDCNST
(``ndcnst``) come from CM1.  This port supports exactly those switches plus
INUM=0 (predicted droplet number; cm1x option ``inum=0``, which CM1 itself
cannot select without editing GRAUPEL_INIT).  The reflectivity diagnostic
(calc_refl10cm, getdbz) and the effective radii for radiation (dorad) are
not on the step path and are not ported; thermodynamic/moisture budget
diagnostics (dotbud/doqbud) are not ported either.

Vectorisation.  The Fortran loops over columns and, within a column, over
levels with GOTOs (skip a level, skip the warm-rain rates, skip a column).
Here every level of every column is computed at once and each Fortran branch
becomes a mask; values that the Fortran leaves uninitialised in a branch are
computed from guarded ("safe") inputs so that every lane stays finite (no
NaN/inf leaks into gradients through ``where``).  Vertical recurrences
(fall-speed fill below the precipitation edge) use ``Backend.scan``; the
per-species sedimentation sub-steps run as masked iterations
(``Backend.fori_loop``, data-dependent trip count, or a static bound
``nfall_max`` for reverse-mode AD under jit).  Sums keep the Fortran's
order where it matters for the state (sedimentation, column sums); the
precipitation and budget accumulators are summed level by level / point by
point as CM1 does.

Integer powers are written as products the way gfortran expands them
(x**2 = x*x, x**3 = (x*x)*x, x**4 = (x*x)*(x*x)); real exponents use
``xp.pow`` (numpy's ndarray ``**`` would turn x**0.5 into sqrt).
"""

from __future__ import annotations

import math

from . import constants as C

# module parameters
PI = 3.1415926535897932384626434
XXX = 0.9189385332046727417803297

_DEFAULTS = dict(
    ihail=0,
    ndcnst=0.0,
    inum=1,
    eqtset=2,
    tsmall=1.0e-4,
    qsmall=1.0e-14,
    testcase=0,
    efall=0,
    axisymm=0,
    satadj_maxiter=20,
    nfall_max=0,
)


def _opt(cfg, name):
    if cfg is None:
        return _DEFAULTS[name]
    get = getattr(cfg, "get", None)
    v = get(name, _DEFAULTS[name]) if get else getattr(cfg, name, _DEFAULTS[name])
    return _DEFAULTS[name] if v is None else v


# ---------------------------------------------------------------------------
# GAMMA / DERF1


_GP = (
    -1.71618513886549492533811e0,
    2.47656508055759199108314e1,
    -3.79804256470945635097577e2,
    6.29331155312818442661052e2,
    8.66966202790413211295064e2,
    -3.14512729688483675254357e4,
    -3.61444134186911729807069e4,
    6.64561438202405440627855e4,
)
_GQ = (
    -3.08402300119738975254353e1,
    3.15350626979604161529144e2,
    -1.01515636749021914166146e3,
    -3.10777167157231109440444e3,
    2.25381184209801510330112e4,
    4.75584627752788110767815e3,
    -1.34659959864969306392456e5,
    -1.15132259675553483497211e5,
)
_GC = (
    -1.910444077728e-03,
    8.4171387781295e-04,
    -5.952379913043012e-04,
    7.93650793500350248e-04,
    -2.777777777777681622553e-03,
    8.333333333333333331554247e-02,
    5.7083835261e-03,
)
_XBIG = 35.040
_XMININ = 1.18e-38
_EPS = 1.19e-7
_XINF = 3.4e38


def gamma_scalar(x: float) -> float:
    """GAMMA(X) for a Python float, statement by statement (glibc exp/log/sin
    via ``math``, as the Fortran)."""
    parity = False
    fact = 1.0
    n = 0
    y = x
    if y <= 0.0:
        y = -x
        y1 = float(math.trunc(y))
        res = y - y1
        if res != 0.0:
            if y1 != float(math.trunc(y1 * 0.5)) * 2.0:
                parity = True
            fact = -PI / math.sin(PI * res)
            y = y + 1.0
        else:
            return _XINF
    if y < _EPS:
        if y >= _XMININ:
            res = 1.0 / y
        else:
            return _XINF
    elif y < 12.0:
        y1 = y
        if y < 1.0:
            z = y
            y = y + 1.0
        else:
            n = int(y) - 1
            y = y - float(n)
            z = y - 1.0
        xnum, xden = 0.0, 1.0
        for i in range(8):
            xnum = (xnum + _GP[i]) * z
            xden = xden * z + _GQ[i]
        res = xnum / xden + 1.0
        if y1 < y:
            res = res / y1
        elif y1 > y:
            for _ in range(n):
                res = res * y
                y = y + 1.0
    else:
        if y <= _XBIG:
            ysq = y * y
            s = _GC[6]
            for i in range(6):
                s = s / ysq + _GC[i]
            s = s / y - y + XXX
            s = s + (y - 0.5) * math.log(y)
            res = math.exp(s)
        else:
            return _XINF
    if parity:
        res = -res
    if fact != 1.0:
        res = fact / res
    return res


def gamma(bk, x):
    """GAMMA(X) element-wise for 1 <= x <= 35.04 (the scheme only evaluates
    it at PGAM + 1..7 with 2 <= PGAM <= 10); same operations as the Fortran
    branch by branch."""
    xp = bk.xp
    one = xp.ones_like(x)
    # 1 <= y < 12: reduce to (1, 2), rational approximation, multiply up
    small = x < 12.0
    ys = xp.where(small, x, one)
    nf = xp.floor(ys) - 1.0  # INT(Y)-1 (y > 0)
    y = ys - nf
    z = y - 1.0
    xnum = xp.zeros_like(x)
    xden = one
    for i in range(8):
        xnum = (xnum + _GP[i]) * z
        xden = xden * z + _GQ[i]
    res = xnum / xden + 1.0
    for i in range(10):  # N <= 10 for y < 12
        m = nf > float(i)
        res = xp.where(m, res * y, res)
        y = xp.where(m, y + 1.0, y)
    # y >= 12: Stirling series
    yb = xp.where(small, 12.0 * one, x)
    ysq = yb * yb
    s = _GC[6] * one
    for i in range(6):
        s = s / ysq + _GC[i]
    s = s / yb - yb + XXX
    s = s + (yb - 0.5) * xp.log(yb)
    big = xp.exp(s)
    return xp.where(small, res, big)


_ERF_A = (
    0.00000000005958930743e0,
    -0.00000000113739022964e0,
    0.00000001466005199839e0,
    -0.00000016350354461960e0,
    0.00000164610044809620e0,
    -0.00001492559551950604e0,
    0.00012055331122299265e0,
    -0.00085483269811296660e0,
    0.00522397762482322257e0,
    -0.02686617064507733420e0,
    0.11283791670954881569e0,
    -0.37612638903183748117e0,
    1.12837916709551257377e0,
    0.00000000002372510631e0,
    -0.00000000045493253732e0,
    0.00000000590362766598e0,
    -0.00000006642090827576e0,
    0.00000067595634268133e0,
    -0.00000621188515924000e0,
    0.00005103883009709690e0,
    -0.00037015410692956173e0,
    0.00233307631218880978e0,
    -0.01254988477182192210e0,
    0.05657061146827041994e0,
    -0.21379664776456006580e0,
    0.84270079294971486929e0,
    0.00000000000949905026e0,
    -0.00000000018310229805e0,
    0.00000000239463074000e0,
    -0.00000002721444369609e0,
    0.00000028045522331686e0,
    -0.00000261830022482897e0,
    0.00002195455056768781e0,
    -0.00016358986921372656e0,
    0.00107052153564110318e0,
    -0.00608284718113590151e0,
    0.02986978465246258244e0,
    -0.13055593046562267625e0,
    0.67493323603965504676e0,
    0.00000000000382722073e0,
    -0.00000000007421598602e0,
    0.00000000097930574080e0,
    -0.00000001126008898854e0,
    0.00000011775134830784e0,
    -0.00000111992758382650e0,
    0.00000962023443095201e0,
    -0.00007404402135070773e0,
    0.00050689993654144881e0,
    -0.00307553051439272889e0,
    0.01668977892553165586e0,
    -0.08548534594781312114e0,
    0.56909076642393639985e0,
    0.00000000000155296588e0,
    -0.00000000003032205868e0,
    0.00000000040424830707e0,
    -0.00000000471135111493e0,
    0.00000005011915876293e0,
    -0.00000048722516178974e0,
    0.00000430683284629395e0,
    -0.00003445026145385764e0,
    0.00024879276133931664e0,
    -0.00162940941748079288e0,
    0.00988786373932350462e0,
    -0.05962426839442303805e0,
    0.49766113250947636708e0,
)
_ERF_B = (
    -0.00000000029734388465e0,
    0.00000000269776334046e0,
    -0.00000000640788827665e0,
    -0.00000001667820132100e0,
    -0.00000021854388148686e0,
    0.00000266246030457984e0,
    0.00001612722157047886e0,
    -0.00025616361025506629e0,
    0.00015380842432375365e0,
    0.00815533022524927908e0,
    -0.01402283663896319337e0,
    -0.19746892495383021487e0,
    0.71511720328842845913e0,
    -0.00000000001951073787e0,
    -0.00000000032302692214e0,
    0.00000000522461866919e0,
    0.00000000342940918551e0,
    -0.00000035772874310272e0,
    0.00000019999935792654e0,
    0.00002687044575042908e0,
    -0.00011843240273775776e0,
    -0.00080991728956032271e0,
    0.00661062970502241174e0,
    0.00909530922354827295e0,
    -0.20160072778491013140e0,
    0.51169696718727644908e0,
    0.00000000003147682272e0,
    -0.00000000048465972408e0,
    0.00000000063675740242e0,
    0.00000003377623323271e0,
    -0.00000015451139637086e0,
    -0.00000203340624738438e0,
    0.00001947204525295057e0,
    0.00002854147231653228e0,
    -0.00101565063152200272e0,
    0.00271187003520095655e0,
    0.02328095035422810727e0,
    -0.16725021123116877197e0,
    0.32490054966649436974e0,
    0.00000000002319363370e0,
    -0.00000000006303206648e0,
    -0.00000000264888267434e0,
    0.00000002050708040581e0,
    0.00000011371857327578e0,
    -0.00000211211337219663e0,
    0.00000368797328322935e0,
    0.00009823686253424796e0,
    -0.00065860243990455368e0,
    -0.00075285814895230877e0,
    0.02585434424202960464e0,
    -0.11637092784486193258e0,
    0.18267336775296612024e0,
    -0.00000000000367789363e0,
    0.00000000020876046746e0,
    -0.00000000193319027226e0,
    -0.00000000435953392472e0,
    0.00000018006992266137e0,
    -0.00000078441223763969e0,
    -0.00000675407647949153e0,
    0.00008428418334440096e0,
    -0.00017604388937031815e0,
    -0.00239729611435071610e0,
    0.02064129023876022970e0,
    -0.06905562880005864105e0,
    0.09084526782065478489e0,
)


def _seg_poly(bk, coef, seg, t, nseg):
    """Horner polynomial of 13 coefficients ``coef[13*seg + 0..12]`` in t,
    with ``seg`` (float array of segment numbers 0..nseg-1) selecting the
    coefficient set: ((c0*t + c1)*t + ... )*t + c12."""
    xp = bk.xp
    cs = []
    for m in range(13):
        c = bk.full(t.shape, coef[m])
        for s in range(1, nseg):
            c = xp.where(seg == float(s), coef[13 * s + m], c)
        cs.append(c)
    y = cs[0] * t + cs[1]
    for m in range(2, 13):
        y = y * t + cs[m] if m < 12 else y * t + cs[m]
    return y


def derf1(bk, x):
    """DERF1(X): the scheme's table-driven error function, element-wise."""
    xp = bk.xp
    w = xp.abs(x)
    # |x| < 2.2: K = INT(W*W), 5 segments in t = W*W - K
    lo = w < 2.2
    wl = xp.where(lo, w, xp.zeros_like(w))
    tt = wl * wl
    ka = xp.floor(tt)
    ya = _seg_poly(bk, _ERF_A, ka, tt - ka, 5) * wl
    # 2.2 <= |x| < 6.9: K = INT(W), segments K-2 = 0..4
    mid = (~lo) & (w < 6.9)
    wm = xp.where(mid, w, 2.5 * xp.ones_like(w))
    kb = xp.floor(wm)
    yb = _seg_poly(bk, _ERF_B, kb - 2.0, wm - kb, 5)
    yb = yb * yb
    yb = yb * yb
    yb = yb * yb
    yb = 1.0 - yb * yb
    y = xp.where(lo, ya, xp.where(mid, yb, xp.ones_like(w)))
    return xp.where(x < 0.0, -y, y)


# ---------------------------------------------------------------------------
# GRAUPEL_INIT


def graupel_init(ihail=0, ndcnst=0.0, inum=1):
    """GRAUPEL_INIT(cm1hail, cm1inum, cm1ndcnst): the module constants as a
    dict of Python floats (keys lower-case Fortran names).  ``inum`` is the
    cm1x extension described in the module docstring (CM1: always 1)."""
    M = {}
    M["inum"] = int(inum)
    M["ndcnst"] = float(ndcnst)
    M["ihail"] = int(ihail)
    # fall speed parameters
    ai, ac, as_, ar = 700.0, 3.0e7, 11.72, 841.99667
    bi, bc, bs, br = 1.0, 2.0, 0.41, 0.8
    if ihail == 0:
        ag, bg = 19.3, 0.37
    else:
        ag, bg = 114.5, 0.5
    r, rv, cp = 287.04, 461.5, 1005.7
    ep_2 = r / rv
    cv = cp - r
    cvv, cpv, cpl, cpi = 1408.5, 1870.0, 4190.0, 2106.0
    xlv0, xls0 = 2501000.0, 2834000.0
    lv1 = xlv0 + (cpl - cpv) * 273.15
    lv2 = cpl - cpv
    ls1 = xls0 + (cpi - cpv) * 273.15
    ls2 = cpi - cpv
    tmelt = 273.15
    rhosu = 85000.0 / (r * tmelt)
    rhow, rhoi, rhosn = 997.0, 500.0, 100.0
    rhog = 400.0 if ihail == 0 else 900.0
    aimm, bimm, ecr, dcs = 0.66, 100.0, 1.0, 125.0e-6
    pi43 = 4.0 / 3.0 * PI  # folded by the compiler
    mi0 = pi43 * rhoi * _cube_s(10.0e-6)
    mg0 = 1.6e-10
    f1s, f2s, f1r, f2r = 0.86, 0.28, 0.78, 0.308
    g = 9.81
    qsmall, qthr = 1.0e-14, 1.0e-10
    eii, eci = 0.1, 0.7
    cpw = cpl
    ci = rhoi * PI / 6.0
    di = 3.0
    cs = rhosn * PI / 6.0
    ds = 3.0
    cg = rhog * PI / 6.0
    dg = 3.0
    rin = 0.1e-6
    mmult = pi43 * rhoi * _cube_s(5.0e-6)
    lammaxi = 1.0 / 1.0e-6
    lammini = 1.0 / (2.0 * dcs + 100.0e-6)
    lammaxr = 1.0 / 20.0e-6
    lamminr = 1.0 / 2800.0e-6
    d_b = 300.0e-6
    lammaxs = 1.0 / 10.0e-6
    lammins = 1.0 / 2000.0e-6
    lammaxg = 1.0 / 20.0e-6
    lamming = 1.0 / 2000.0e-6
    k1, c1 = 0.4, 120.0
    mw, osm, vi, epsm, rhoa, map_, ma = 0.018, 1.0, 3.0, 0.7, 1777.0, 0.132, 0.0284
    rr = 8.3145
    bact = vi * osm * epsm * mw * rhoa / (map_ * rhow)
    rm1, sig1, nanew1 = 0.052e-6, 2.04, 72.2e6
    f11 = 0.5 * math.exp(2.5 * _sq(math.log(sig1)))
    f21 = 1.0 + 0.25 * math.log(sig1)
    rm2, sig2, nanew2 = 1.3e-6, 2.5, 1.8e6
    f12 = 0.5 * math.exp(2.5 * _sq(math.log(sig2)))
    f22 = 1.0 + 0.25 * math.log(sig2)
    G = gamma_scalar
    cons = {}
    cons[1] = G(1.0 + ds) * cs
    cons[2] = G(1.0 + dg) * cg
    cons[3] = G(4.0 + bs) / 6.0
    cons[4] = G(4.0 + br) / 6.0
    cons[5] = G(1.0 + bs)
    cons[6] = G(1.0 + br)
    cons[7] = G(4.0 + bg) / 6.0
    cons[8] = G(1.0 + bg)
    cons[9] = G(5.0 / 2.0 + br / 2.0)
    cons[10] = G(5.0 / 2.0 + bs / 2.0)
    cons[11] = G(5.0 / 2.0 + bg / 2.0)
    cons[12] = G(1.0 + di) * ci
    cons[13] = G(bs + 3.0) * PI / 4.0 * eci
    cons[14] = G(bg + 3.0) * PI / 4.0 * eci
    cons[15] = (
        -1108.0
        * eii
        * math.pow(PI, (1.0 - bs) / 3.0)
        * math.pow(rhosn, (-2.0 - bs) / 3.0)
        / (4.0 * 720.0)
    )
    cons[16] = G(bi + 3.0) * PI / 4.0 * eci
    cons[17] = (4.0 * 2.0 * 3.0 * rhosu * PI * eci * eci * G(2.0 * bs + 2.0)) / (
        8.0 * (rhog - rhosn)
    )
    cons[18] = rhosn * rhosn
    cons[19] = rhow * rhow
    cons[20] = 20.0 * PI * PI * rhow * bimm
    cons[21] = 4.0 / (dcs * rhoi)
    cons[22] = PI * rhoi * ((dcs * dcs) * dcs) / 6.0  # DCS**3 at run time
    cons[23] = PI / 4.0 * eii * G(bs + 3.0)
    cons[24] = PI / 4.0 * ecr * G(br + 3.0)
    cons[25] = PI * PI / 24.0 * rhow * ecr * G(br + 6.0)
    cons[26] = PI / 6.0 * rhow
    cons[27] = G(1.0 + bi)
    cons[28] = G(4.0 + bi) / 6.0
    cons[29] = pi43 * rhow * _cube_s(25.0e-6)
    cons[30] = pi43 * rhow
    cons[31] = PI * PI * ecr * rhosn
    cons[32] = PI / 2.0 * ecr
    cons[33] = PI * PI * ecr * rhog
    cons[34] = 5.0 / 2.0 + br / 2.0
    cons[35] = 5.0 / 2.0 + bs / 2.0
    cons[36] = 5.0 / 2.0 + bg / 2.0
    cons[37] = 4.0 * PI * 1.38e-23 / (6.0 * PI * rin)
    cons[38] = PI * PI / 3.0 * rhow
    cons[39] = PI * PI / 36.0 * rhow * bimm
    cons[40] = PI / 6.0 * bimm
    cons[41] = PI * PI * ecr * rhow
    loc = dict(locals())
    for k in (
        "ai ac ar bi bc bs br ag bg r rv cp ep_2 cv cvv cpv cpl cpi lv1 lv2 "
        "ls1 ls2 tmelt rhosu rhow rhoi rhosn rhog aimm bimm ecr dcs mi0 mg0 "
        "f1s f2s f1r f2r g qsmall qthr eii eci cpw ci di cs ds cg dg rin mmult "
        "lammaxi lammini lammaxr lamminr d_b lammaxs lammins lammaxg lamming "
        "k1 c1 mw osm vi epsm rhoa ma rr bact rm1 sig1 nanew1 f11 f21 rm2 "
        "sig2 nanew2 f12 f22"
    ).split():
        M[k] = loc[k]
    M["as"] = as_
    M["map"] = map_
    for n, v in cons.items():
        M[f"cons{n}"] = v
    return M


def _sq(x):
    return x * x


def _cube_s(x):
    """A literal real**3: gfortran folds it at compile time, correctly
    rounded (MPFR), not as (x*x)*x."""
    from fractions import Fraction

    return float(Fraction(x) ** 3)


# ---------------------------------------------------------------------------
# satadj2


def satadj2(
    bk,
    M,
    *,
    tnew,
    prs,
    qvnew,
    qcnew,
    qrnew,
    qinew,
    qsnew,
    qgnew,
    rho,
    eqtset,
    tsmall,
    mask=None,
    maxiter=20,
):
    """satadj2 element-wise on arrays.  ``mask`` (bool array) selects the
    points the Fortran calls it for (others are returned unchanged).

    eqtset=2: the per-point ``do while`` (until |dT| <= 2*tsmall, giving up
    at n=100) runs as up to ``maxiter`` masked sweeps (default 20; a point
    freezes once converged, so results equal the Fortran whenever every
    point converges within ``maxiter``; the returned ``nmax`` is the
    largest iteration count).  Returns dict tnew, prs, qvnew, qcnew, nmax."""
    xp = bk.xp
    if mask is None:
        mask = xp.ones_like(tnew) > 0.0
    if eqtset == 2:
        converge = 2.0 * tsmall
        esl = xp.minimum(
            prs * 0.5,
            611.2 * xp.exp(17.67 * (tnew - 273.15) / (tnew - 29.65)),
        )
        qvs = M["ep_2"] * esl / (prs - esl)
        act0 = mask & ((qcnew > M["qsmall"]) | (qvnew > qvs))
        zero = xp.zeros_like(tnew)
        ql = xp.maximum(zero, qcnew + qrnew)
        qi = xp.maximum(zero, qinew + qsnew + qgnew)
        cvml = M["cv"] + M["cvv"] * qvnew + M["cpl"] * ql + M["cpi"] * qi
        lhv = M["lv1"] - M["lv2"] * tnew
        t1 = (lhv - M["rv"] * tnew) / cvml
        d1 = t1 * 17.67 * 243.5
        nit = xp.zeros(tnew.shape, dtype=xp.int32, device=bk.device)

        def sweep(_last, c):
            qv, qc, t, p, nit, tlast, qvs, active = c
            dtd = t - 29.65
            dqv = (qvs - qv) / (1.0 + d1 * qvs / (dtd * dtd))
            dqv = xp.minimum(dqv, qc)
            dqv = xp.where((qv + dqv) < 1.0e-20, 1.0e-20 - qv, dqv)
            qv_n = qv + dqv
            qc_n = qc - dqv
            t_n = t - dqv * t1
            p_n = rho * (M["r"] + M["rv"] * qv_n) * t_n
            qv = xp.where(active, qv_n, qv)
            qc = xp.where(active, qc_n, qc)
            t = xp.where(active, t_n, t)
            p = xp.where(active, p_n, p)
            nit = xp.where(active, nit + 1, nit)
            again = active & (xp.abs(t - tlast) > converge)
            tlast = xp.where(again, t, tlast)
            e = xp.minimum(p * 0.5, 611.2 * xp.exp(17.67 * (t - 273.15) / (t - 29.65)))
            qvs = xp.where(again, M["ep_2"] * e / (p - e), qvs)
            return qv, qc, t, p, nit, tlast, qvs, again

        c = (qvnew, qcnew, tnew, prs, nit, tnew, qvs, act0)
        # CM1 gives up (stopcm1) at n=100; eagerly the sweeps stop as soon as
        # every point has converged
        c = bk.fori_loop(
            min(int(maxiter), 100),
            lambda it, c: sweep(False, c),
            c,
            until=lambda c: ~xp.any(c[7]),
        )
        qv, qc, t, p, nit, _, _, _ = c
        return {"tnew": t, "prs": p, "qvnew": qv, "qcnew": qc, "nmax": xp.max(nit)}
    esl = xp.minimum(
        prs * 0.5, 611.2 * xp.exp(17.67 * (tnew - 273.15) / (tnew - 29.65))
    )
    qvs = M["ep_2"] * esl / (prs - esl)
    lhv = M["lv1"] - M["lv2"] * tnew
    dtd = tnew - 29.65
    dqv = (qvs - qvnew) / (1.0 + lhv * qvs * 17.67 * 243.5 / (M["cp"] * (dtd * dtd)))
    dqv = xp.where((qvnew + dqv) < 1.0e-20, 1.0e-20 - qvnew, dqv)
    dqv = xp.minimum(dqv, xp.maximum(xp.zeros_like(qcnew), qcnew))
    qv = qvnew + dqv
    qc = qcnew - dqv
    t = tnew - dqv * lhv / (M["cp"] * (1.0 + 0.887 * qv))
    return {
        "tnew": xp.where(mask, t, tnew),
        "prs": prs,
        "qvnew": xp.where(mask, qv, qvnew),
        "qcnew": xp.where(mask, qc, qcnew),
        "nmax": xp.asarray(1),
    }


# ---------------------------------------------------------------------------
# M2005MICRO_GRAUPEL


def _isarr(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _traced(bk, x):
    if bk.name != "jax":
        return False
    import jax

    return isinstance(x, jax.core.Tracer)


def m2005micro_graupel(
    bk,
    M,
    *,
    t,
    qv,
    qc,
    qi,
    qs,
    qr,
    nc,
    ni,
    ns,
    nr,
    qg,
    ng,
    pres,
    rho,
    dzq,
    w,
    dt,
    eqtset=2,
    tsmall=1.0e-4,
    testcase=0,
    nfall_max=None,
    satadj_maxiter=20,
):
    """M2005MICRO_GRAUPEL on every column at once.

    All arrays are ``[k, j, i]`` without halos (``k`` = KTS..KTE along axis
    0): temperature ``t`` (K), mixing ratios qv, qc, qi, qs (snow, QNI3D),
    qr, qg, numbers nc, ni, ns, nr, ng (1/kg), pressure ``pres``, density
    ``rho``, layer depth ``dzq`` and vertical velocity ``w`` (cell centre).
    ``M`` is ``graupel_init(...)``.  For inum=1 pass ``nc`` = 0 (as
    MP_GRAUPEL does).

    Returns dict with the updated t, qv, qc, qi, qs, qr, nc, ni, ns, nr, qg,
    ng; ``precrt``, ``snowrt`` ``[j, i]``; the process rates pre, evpms,
    evpmg, pcc (for MP_GRAUPEL's budgets) and fall speeds fc, fr, fi, fs, fg
    (``[k, j, i]``, zero in columns without hydrometeors)."""
    xp = bk.xp
    W = xp.where
    inum = int(M["inum"])
    eqtset = int(eqtset)
    shape = t.shape
    nk = shape[0]

    def full(v):
        return bk.full(shape, float(v))

    zero = full(0.0)
    one = full(1.0)

    def wh(m, a, b):
        if not _isarr(a):
            a = bk.full(m.shape, float(a))
        if not _isarr(b):
            b = bk.full(m.shape, float(b))
        return W(m, a, b)

    def mx(a, v):
        return xp.maximum(a, v if _isarr(v) else xp.full_like(a, float(v)))

    def mn(a, v):
        return xp.minimum(a, v if _isarr(v) else xp.full_like(a, float(v)))

    def pw(x, e):
        return xp.pow(x, e)

    def safe(m, x, v=1.0):
        return wh(m, x, v)

    # module constants
    g_ = M["g"]
    qsmall, qthr = M["qsmall"], M["qthr"]
    tmelt = M["tmelt"]
    rhow, rhosu = M["rhow"], M["rhosu"]
    cons = {n: M[f"cons{n}"] for n in range(1, 42)}
    rdt = 1.0 / dt
    ndcnst = M["ndcnst"]

    # ---------------- kloop1 ---------------------------------------------
    xxlv = M["lv1"] - M["lv2"] * t
    xxls = M["ls1"] - M["ls2"] * t
    cpm = M["cp"] * (1.0 + 0.887 * qv)
    if eqtset == 2:
        cvm = 1.0 / (
            M["cv"] + M["cvv"] * qv + M["cpl"] * (qc + qr) + M["cpi"] * (qi + qs + qg)
        )
        ecnd = (xxlv - M["rv"] * t) * cvm
        edep = (xxls - M["rv"] * t) * cvm
        efrz = (xxls - xxlv) * cvm
    else:
        ecnd = xxlv / cpm
        edep = xxls / cpm
        efrz = (xxls - xxlv) / cpm

    def satvap(t, pres):
        evs = xp.minimum(pres * 0.5, 611.2 * xp.exp(17.67 * (t - 273.15) / (t - 29.65)))
        eis = xp.minimum(
            pres * 0.5, 611.2 * xp.exp(21.8745584 * (t - 273.15) / (t - 7.66))
        )
        eis = W(eis > evs, evs, eis)
        qvs = M["ep_2"] * evs / (pres - evs)
        qvi = M["ep_2"] * eis / (pres - eis)
        return evs, qvs, qvi

    def subsat(m, qv, t, qc, qr, qi, qs, qg, qvqvs, qvqvsi):
        """remove small amounts of condensate at subsaturation"""
        ml = m & (qvqvs < 0.9)
        out = []
        for q in (qr, qc):
            c = ml & (q < qthr)
            qv = W(c, qv + q, qv)
            t = W(c, t - q * ecnd, t)
            out.append(W(c, zero, q))
        qr, qc = out
        mi = m & (qvqvsi < 0.9)
        out = []
        for q in (qi, qs, qg):
            c = mi & (q < qthr)
            qv = W(c, qv + q, qv)
            t = W(c, t - q * edep, t)
            out.append(W(c, zero, q))
        qi, qs, qg = out
        return qv, t, qc, qr, qi, qs, qg

    def zsmall(m, q, n):
        c = m & (q < qsmall)
        return W(c, zero, q), W(c, zero, n)

    evs, qvs, qvi = satvap(t, pres)
    qvqvs = qv / qvs
    qvqvsi = qv / qvi
    alltrue = t == t
    qv, t, qc, qr, qi, qs, qg = subsat(
        alltrue, qv, t, qc, qr, qi, qs, qg, qvqvs, qvqvsi
    )
    xlf = xxls - xxlv
    qc, nc = zsmall(alltrue, qc, nc)
    qr, nr = zsmall(alltrue, qr, nr)
    qi, ni = zsmall(alltrue, qi, ni)
    qs, ns = zsmall(alltrue, qs, ns)
    qg, ng = zsmall(alltrue, qg, ng)

    mu = 1.496e-6 * pw(t, 1.5) / (t + 120.0)
    dum54 = pw(rhosu / rho, 0.54)
    dum35 = pw(rhosu / rho, 0.35)
    ain = dum35 * M["ai"]
    arn = dum54 * M["ar"]
    asn = dum54 * M["as"]
    acn = g_ * rhow / (18.0 * mu)
    agn = dum54 * M["ag"]

    allsmall = (
        (qc < qsmall) & (qi < qsmall) & (qs < qsmall) & (qr < qsmall) & (qg < qsmall)
    )
    skip = allsmall & (
        ((t < tmelt) & (qvqvsi < 0.999)) | ((t >= tmelt) & (qvqvs < 0.999))
    )
    act = ~skip

    kap = 1.414e3 * mu
    dv = 8.794e-5 * pw(t, 1.81) / pres
    sc = mu / (rho * dv)
    dumt2 = M["rv"] * (t * t)
    dqsdt = xxlv * qvs / dumt2
    dqsidt = xxls * qvi / dumt2
    abi = 1.0 + dqsidt * xxls / cpm
    ab = 1.0 + dqsdt * xxlv / cpm

    warm = act & (t >= tmelt)
    cold = act & ~(t >= tmelt)
    if inum == 1:
        nc = W(act, ndcnst * 1.0e6 / rho, nc)

    # warm: melt small snow/graupel instantly
    for which in ("s", "g"):
        q, n = (qs, ns) if which == "s" else (qg, ng)
        c = warm & (q < 1.0e-6)
        qr = W(c, qr + q, qr)
        nr = W(c, nr + n, nr)
        t = W(c, t - q * efrz, t)
        q = W(c, zero, q)
        n = W(c, zero, n)
        if which == "s":
            qs, ns = q, n
        else:
            qg, ng = q, n
    skip300 = warm & (qc < qsmall) & (qs < 1.0e-8) & (qr < qsmall) & (qg < 1.0e-8)
    wr = warm & ~skip300
    dist = wr | cold

    ni = W(cold, mx(ni, 0.0), ni)
    ns = W(dist, mx(ns, 0.0), ns)
    if inum == 0:
        nc = W(dist, mx(nc, 0.0), nc)
    nr = W(dist, mx(nr, 0.0), nr)
    ng = W(dist, mx(ng, 0.0), ng)

    def lamlim(m, q, n, cfac, rexp, lmin, lmax, n0fn):
        """slope/intercept with the lambda limits (n changed when limited)"""
        mm = m & (q >= qsmall)
        qq = safe(mm, q)
        nn = safe(mm, n)
        lam = pw(cfac * nn / qq, rexp)
        n0 = nn * lam
        lo = lam < lmin
        hi = ~lo & (lam > lmax)
        lam = W(lo, full(lmin), W(hi, full(lmax), lam))
        n0c = n0fn(lam, qq)
        lim = lo | hi
        n0 = W(lim, n0c, n0)
        n = W(mm & lim, n0c / lam, n)
        return lam, n0, n

    di, ds, dg = M["di"], M["ds"], M["dg"]
    pirw = PI * rhow
    lami_c, n0i, ni = lamlim(
        cold,
        qi,
        ni,
        cons[12],
        1.0 / di,
        M["lammini"],
        M["lammaxi"],
        lambda lam, q: pw(lam, di + 1.0) * q / cons[12],
    )
    lamr, n0rr, nr = lamlim(
        dist,
        qr,
        nr,
        pirw,
        1.0 / 3.0,
        M["lamminr"],
        M["lammaxr"],
        lambda lam, q: ((lam * lam) * (lam * lam)) * q / pirw,
    )
    lams, n0s, ns = lamlim(
        dist,
        qs,
        ns,
        cons[1],
        1.0 / ds,
        M["lammins"],
        M["lammaxs"],
        lambda lam, q: pw(lam, ds + 1.0) * q / cons[1],
    )
    lamg, n0g, ng = lamlim(
        dist,
        qg,
        ng,
        cons[2],
        1.0 / dg,
        M["lamming"],
        M["lammaxg"],
        lambda lam, q: pw(lam, dg + 1.0) * q / cons[2],
    )
    # LAMI is zeroed at every level before the skips, set in the cold branch
    lami = W(cold & (qi >= qsmall), lami_c, zero)

    def pgam_of(n):
        p = 0.0005714 * (n / 1.0e6 * rho) + 0.2714
        p = 1.0 / (p * p) - 1.0
        return mn(mx(p, 2.0), 10.0)

    def lamc_lim(m, q, n):
        mm = m & (q >= qsmall)
        qq = safe(mm, q)
        nn = safe(mm, n)
        pgam = pgam_of(nn)
        g1 = gamma(bk, pgam + 1.0)
        g4 = gamma(bk, pgam + 4.0)
        lamc = pw(cons[26] * nn * g4 / (qq * g1), 1.0 / 3.0)
        lammin = (pgam + 1.0) / 60.0e-6
        lammax = (pgam + 1.0) / 1.0e-6
        lo = lamc < lammin
        hi = ~lo & (lamc > lammax)
        lamc = W(lo, lammin, W(hi, lammax, lamc))
        ncn = (
            xp.exp(3.0 * xp.log(lamc) + xp.log(qq) + xp.log(g1) - xp.log(g4)) / cons[26]
        )
        n = W(mm & (lo | hi), ncn, n)
        return pgam, lamc, n, g1, g4, mm

    pgam, lamc, nc, g1c, g4c, mc = lamc_lim(dist, qc, nc)
    cdist1 = safe(mc, nc) / g1c

    # ---- process rates shared by the warm and cold branches ---------------
    lr2 = lamr * lamr
    lr3 = lr2 * lamr
    ls2 = lams * lams
    ls3 = ls2 * lams
    lg2 = lamg * lamg
    lg3 = lg2 * lamg
    bs, br, bg, bi = M["bs"], M["br"], M["bg"], M["bi"]
    pls = pw(lams, bs)
    plr = pw(lamr, br)
    plg = pw(lamg, bg)
    ums = mn(asn * cons[3] / pls, 1.2 * dum54)
    uns = mn(asn * cons[5] / pls, 1.2 * dum54)
    umr = mn(arn * cons[4] / plr, 9.1 * dum54)
    unr = mn(arn * cons[6] / plr, 9.1 * dum54)
    umg = mn(agn * cons[7] / plg, 20.0 * dum54)
    ung = mn(agn * cons[8] / plg, 20.0 * dum54)

    def sq(x):
        return x * x

    c_rs = dist & (qr >= 1.0e-8) & (qs >= 1.0e-8)
    c_rg = dist & (qr >= 1.0e-8) & (qg >= 1.0e-8)
    prs_arg = pw(sq(1.2 * umr - 0.95 * ums) + 0.08 * ums * umr, 0.5)
    pracs_raw = cons[41] * (
        prs_arg
        * rho
        * n0rr
        * n0s
        / lr3
        * (5.0 / (lr3 * lams) + 2.0 / (lr2 * ls2) + 0.5 / (lamr * ls3))
    )
    pracs_raw = W(c_rs, pracs_raw, zero)
    npracs_raw = (
        cons[32]
        * rho
        * pw(1.7 * sq(unr - uns) + 0.3 * unr * uns, 0.5)
        * n0rr
        * n0s
        * (1.0 / (lr3 * lams) + 1.0 / (lr2 * ls2) + 1.0 / (lamr * ls3))
    )
    psacr_raw = cons[31] * (
        prs_arg
        * rho
        * n0rr
        * n0s
        / ls3
        * (5.0 / (ls3 * lamr) + 2.0 / (ls2 * lr2) + 0.5 / (lams * lr3))
    )
    pracg_raw = cons[41] * (
        pw(sq(1.2 * umr - 0.95 * umg) + 0.08 * umg * umr, 0.5)
        * rho
        * n0rr
        * n0g
        / lr3
        * (5.0 / (lr3 * lamg) + 2.0 / (lr2 * lg2) + 0.5 / (lamr * lg3))
    )
    pracg_raw = W(c_rg, pracg_raw, zero)
    npracg_raw = (
        cons[32]
        * rho
        * pw(1.7 * sq(unr - ung) + 0.3 * unr * ung, 0.5)
        * n0rr
        * n0g
        * (1.0 / (lr3 * lamg) + 1.0 / (lr2 * lg2) + 1.0 / (lamr * lg3))
    )

    # autoconversion (IRAIN=0)
    c = dist & (qc >= 1.0e-6)
    qcp = safe(c, qc)
    ncp = safe(c, nc)
    prc = 1350.0 * pw(qcp, 2.47) * pw(ncp / 1.0e6 * rho, -1.79)
    nprc1 = prc / cons[29]
    nprc = prc / (qcp / ncp)
    nprc = mn(nprc, ncp / dt)
    nprc1 = mn(nprc1, nprc)
    prc = W(c, prc, zero)
    nprc = W(c, nprc, zero)
    nprc1 = W(c, nprc1, zero)

    # accretion of droplets by rain (IRAIN=0)
    c = dist & (qr >= 1.0e-8) & (qc >= 1.0e-8)
    qcp = safe(c, qc)
    pra = 67.0 * pw(qcp * safe(c, qr), 1.15)
    npra = pra / (qcp / safe(c, nc))
    pra = W(c, pra, zero)
    npra = W(c, npra, zero)

    # self-collection/breakup of rain
    c = dist & (qr >= 1.0e-8)
    rl = 1.0 / lamr
    db = M["d_b"]
    dumb = W(rl < db, one, 2.0 - xp.exp(2300.0 * (rl - db)))
    nragg = W(c, -5.78 * dumb * qr * nr * rho, zero)

    def bracket(lam, f1, f2, an, cn, ce):
        return f1 / (lam * lam) + f2 * pw(an * rho / mu, 0.5) * pw(
            sc, 1.0 / 3.0
        ) * cn / pw(lam, ce)

    br_r = bracket(lamr, M["f1r"], M["f2r"], arn, cons[9], cons[34])
    br_s = bracket(lams, M["f1s"], M["f2s"], asn, cons[10], cons[35])
    br_g = bracket(lamg, M["f1s"], M["f2s"], agn, cons[11], cons[36])
    twopi = 2.0 * PI
    epsr = W(dist & (qr >= qsmall), twopi * n0rr * rho * dv * br_r, zero)
    epss_f = twopi * n0s * rho * dv * br_s
    epsg_f = twopi * n0g * rho * dv * br_g

    def rain_evap():
        pre = epsr * (qv - qvs) / ab
        pre = mn(pre, 0.0)
        pre = W((qv - pre * dt) > qvs, (qv - qvs) / dt, pre)
        return W(qv < qvs, pre, zero)

    pre_all = rain_evap()

    def nloss(c, rate, q, n):
        """number loss for a (negative) mass sink: max(-1, rate*dt/q)*n/dt"""
        d = rate * dt / safe(c, q)
        d = mx(d, -1.0)
        return W(c, d * n / dt, zero)

    # ---------------- warm branch (T >= TMELT) -----------------------------
    pre_w = W(wr, pre_all, zero)
    npracg_w = W(c_rg, npracg_raw - pracg_raw / 5.2e-7, zero)
    cpw = M["cpw"]
    c = wr & (qs >= 1.0e-8)
    dum = -cpw / xlf * (t - 273.15) * pracs_raw
    psmlt = twopi * n0s * kap * (tmelt - t) / xlf * br_s + dum
    cv_ = c & (qvqvs < 1.0)
    evpms = (qv - qvs) * epss_f / ab
    evpms = mx(evpms, psmlt)
    psmlt = W(cv_, psmlt - evpms, psmlt)
    evpms = W(cv_, evpms, zero)
    psmlt = W(c, psmlt, zero)
    c = wr & (qg >= 1.0e-8)
    dum = -cpw / xlf * (t - 273.15) * pracg_raw
    pgmlt = twopi * n0g * kap * (tmelt - t) / xlf * br_g + dum
    cv_ = c & (qvqvs < 1.0)
    evpmg = (qv - qvs) * epsg_f / ab
    evpmg = mx(evpmg, pgmlt)
    pgmlt = W(cv_, pgmlt - evpmg, pgmlt)
    evpmg = W(cv_, evpmg, zero)
    pgmlt = W(c, pgmlt, zero)
    prc_w = W(wr, prc, zero)
    pra_w = W(wr, pra, zero)
    # PRACG = PRACS = 0 from here on in the warm branch
    dum = (prc_w + pra_w) * dt
    c = wr & (dum > qc) & (qc >= qsmall)
    ratio = qc / safe(c, dum)
    prc_w = W(c, prc_w * ratio, prc_w)
    pra_w = W(c, pra_w * ratio, pra_w)
    dum = (-psmlt - evpms + zero) * dt
    c = wr & (dum > qs) & (qs >= qsmall)
    ratio = qs / safe(c, dum)
    psmlt = W(c, psmlt * ratio, psmlt)
    evpms = W(c, evpms * ratio, evpms)
    dum = (-pgmlt - evpmg + zero) * dt
    c = wr & (dum > qg) & (qg >= qsmall)
    ratio = qg / safe(c, dum)
    pgmlt = W(c, pgmlt * ratio, pgmlt)
    evpmg = W(c, evpmg * ratio, evpmg)
    dum = (-zero - zero - pre_w - pra_w - prc_w + psmlt + pgmlt) * dt
    c = wr & (dum > qr) & (qr >= qsmall)
    ratio = (qr / dt + zero + zero + pra_w + prc_w - psmlt - pgmlt) / safe(c, -pre_w)
    pre_w = W(c, pre_w * ratio, pre_w)

    qvt_w = zero + (-pre_w - evpms - evpmg)
    tt_w = zero + (
        pre_w * ecnd + (evpms + evpmg) * edep + (psmlt + pgmlt - zero - zero) * efrz
    )
    qct_w = zero + (-pra_w - prc_w)
    qrt_w = zero + (pre_w + pra_w + prc_w - psmlt - pgmlt + zero + zero)
    qst_w = zero + (psmlt + evpms - zero)
    qgt_w = zero + (pgmlt + evpmg - zero)
    nct_w = zero + (-W(wr, npra, zero) - W(wr, nprc, zero))
    nrt_w = zero + (W(wr, nprc1, zero) + W(wr, nragg, zero) - W(wr, npracg_w, zero))
    nsubr = nloss(wr & (pre_w < 0.0), pre_w, qr, nr)
    nsmlts = nloss(wr & (evpms + psmlt < 0.0), evpms + psmlt, qs, ns)
    nsmltr = nloss(wr & (psmlt < 0.0), psmlt, qs, ns)
    ngmltg = nloss(wr & (evpmg + pgmlt < 0.0), evpmg + pgmlt, qg, ng)
    ngmltr = nloss(wr & (pgmlt < 0.0), pgmlt, qg, ng)
    nst_w = zero + nsmlts
    ngt_w = zero + ngmltg
    nrt_w = nrt_w + (nsubr - nsmltr - ngmltr)

    # ---------------- cold branch (T < TMELT) ------------------------------
    cc = cold
    # contact and immersion freezing of droplets
    c = cc & (qc >= qsmall) & (t < 269.15)
    nacnt = xp.exp(-2.80 + 0.262 * (tmelt - t)) * 1000.0
    dum = 7.37 * t / (288.0 * 10.0 * pres) / 100.0
    dap = cons[37] * t * (1.0 + dum / M["rin"]) / mu
    lcd = xp.log(cdist1)
    llc = xp.log(lamc)
    eimm = xp.exp(M["aimm"] * (273.15 - t)) - 1.0
    mnuccc = (
        cons[38] * dap * nacnt * xp.exp(lcd + xp.log(gamma(bk, pgam + 5.0)) - 4.0 * llc)
    )
    nnuccc = twopi * dap * nacnt * cdist1 * gamma(bk, pgam + 2.0) / lamc
    mnuccc = (
        mnuccc
        + cons[39] * xp.exp(lcd + xp.log(gamma(bk, 7.0 + pgam)) - 6.0 * llc) * eimm
    )
    nnuccc = nnuccc + cons[40] * xp.exp(lcd + xp.log(g4c) - 3.0 * llc) * eimm
    nnuccc = mn(nnuccc, nc / dt)
    mnuccc = W(c, mnuccc, zero)
    nnuccc = W(c, nnuccc, zero)

    prc_c = W(cc, prc, zero)
    nprc_c = W(cc, nprc, zero)
    nprc1_c = W(cc, nprc1, zero)

    c = cc & (qs >= 1.0e-8)
    e1 = (2.0 + bs) / 3.0
    nsagg = (
        cons[15]
        * asn
        * pw(rho, e1)
        * pw(safe(c, qs), e1)
        * pw(safe(c, ns) * rho, (4.0 - bs) / 3.0)
        / rho
    )
    nsagg = W(c, nsagg, zero)

    def collect(c, cn, an, q, n0, lam, b):
        p = pw(lam, b + 3.0)
        return W(c, cn * an * q * rho * n0 / p, zero)

    c = cc & (qs >= 1.0e-8) & (qc >= qsmall)
    psacws = collect(c, cons[13], asn, qc, n0s, lams, bs)
    npsacws = collect(c, cons[13], asn, nc, n0s, lams, bs)
    c = cc & (qg >= 1.0e-8) & (qc >= qsmall)
    psacwg = collect(c, cons[14], agn, qc, n0g, lamg, bg)
    npsacwg = collect(c, cons[14], agn, nc, n0g, lamg, bg)
    lami_s = safe(lami > 0.0, lami)
    c = cc & (qi >= 1.0e-8) & (qc >= qsmall) & (1.0 / lami_s >= 100.0e-6)
    psacwi = collect(c, cons[16], ain, qc, n0i, lami_s, bi)
    npsacwi = collect(c, cons[16], ain, nc, n0i, lami_s, bi)

    c = cc & c_rs
    qr_dt = qr / dt
    pracs = W(c, mn(pracs_raw, qr_dt), zero)
    npracs = W(c, npracs_raw, zero)
    psacr = W(c & (qs >= 0.1e-3) & (qr >= 0.1e-3), psacr_raw, zero)
    c = cc & c_rg
    pracg = W(c, mn(pracg_raw, qr_dt), zero)
    npracg = W(c, npracg_raw, zero)

    # rime splintering
    tm = (t < 270.16) & (t > 265.16)
    fmult = W(
        t > 270.16,
        zero,
        W(
            (t <= 270.16) & (t > 268.16),
            (270.16 - t) / 2.0,
            W((t >= 265.16) & (t <= 268.16), (t - 265.16) / 3.0, zero),
        ),
    )
    liq = (qc >= 0.5e-3) | (qr >= 0.1e-3)
    mmult = M["mmult"]

    def splinter(c, rate):
        nm = 35.0e4 * rate * fmult * 1000.0
        qm = mn(nm * mmult, rate)
        return W(c, nm, zero), W(c, qm, zero), W(c, rate - qm, rate)

    cs_ = cc & (qs >= 0.1e-3) & liq & ((psacws > 0.0) | (pracs > 0.0)) & tm
    nmults, qmults, psacws = splinter(cs_ & (psacws > 0.0), psacws)
    nmultr, qmultr, pracs = splinter(cs_ & (pracs > 0.0), pracs)
    cg_ = cc & (qg >= 0.1e-3) & liq & ((psacwg > 0.0) | (pracg > 0.0)) & tm
    nmultg, qmultg, psacwg = splinter(cg_ & (psacwg > 0.0), psacwg)
    nmultrg, qmultrg, pracg = splinter(cg_ & (pracg > 0.0), pracg)

    # conversion of rimed snow to graupel
    c = cc & (psacws > 0.0) & (qs >= 0.1e-3) & (qc >= 0.5e-3)
    pgsacw = mn(
        psacws,
        cons[17] * dt * n0s * qc * qc * asn * asn / (rho * pw(lams, 2.0 * bs + 2.0)),
    )
    rhosn, rhog = M["rhosn"], M["rhog"]
    dum = mx(rhosn / (rhog - rhosn) * pgsacw, 0.0)
    nscng = mn(dum / M["mg0"] * rho, ns / dt)
    pgsacw = W(c, pgsacw, zero)
    nscng = W(c, nscng, zero)
    psacws = W(c, psacws - pgsacw, psacws)
    c = cc & (pracs > 0.0) & (qs >= 0.1e-3) & (qr >= 0.1e-3)
    x = 4.0 / lams
    x3 = (x * x) * x
    y = 4.0 / lamr
    y3 = (y * y) * y
    dum = cons[18] * x3 * x3 / (cons[18] * x3 * x3 + cons[19] * y3 * y3)
    dum = mx(mn(dum, 1.0), 0.0)
    pgracs = W(c, (1.0 - dum) * pracs, zero)
    ngracs = mn(mn((1.0 - dum) * npracs, nr / dt), ns / dt)
    ngracs = W(c, ngracs, zero)
    pracs = W(c, pracs - pgracs, pracs)
    npracs = W(c, npracs - ngracs, npracs)
    psacr = W(c, psacr * (1.0 - dum), psacr)

    # immersion freezing of rain
    c = cc & (t < 269.15) & (qr >= qsmall)
    mnuccr = W(c, cons[20] * nr * (eimm / lr3 / lr3), zero)
    nnuccr = W(c, mn(PI * nr * M["bimm"] * eimm / lr3, nr / dt), zero)

    pra_c = W(cc, pra, zero)
    npra_c = W(cc, npra, zero)
    nragg_c = W(cc, nragg, zero)

    # ice autoconversion to snow, accretion of ice by snow
    dcs = M["dcs"]
    c = cc & (qi >= 1.0e-8) & (qvqvsi >= 1.0)
    nprci = cons[21] * (qv - qvi) * rho * n0i * xp.exp(-lami * dcs) * dv / abi
    prci = W(c, cons[22] * nprci, zero)
    nprci = W(c, mn(nprci, ni / dt), zero)
    c = cc & (qs >= 1.0e-8) & (qi >= qsmall)
    prai = collect(c, cons[23], asn, qi, n0s, lams, bs)
    nprai = W(c, mn(collect(c, cons[23], asn, ni, n0s, lams, bs), ni / dt), zero)

    # rain-ice collection
    c = cc & (qr >= 1.0e-8) & (qi >= 1.0e-8) & (t <= tmelt)
    pb = pw(lamr, br + 3.0)
    a_ = mn(mn(cons[24] * ni * n0rr * arn / pb * rho, nr / dt), ni / dt)
    b_ = cons[25] * ni * n0rr * arn / pb / lr3 * rho
    c_ = cons[24] * qi * n0rr * arn / pb * rho
    big = c & (qr >= 0.1e-3)
    sml = c & ~(qr >= 0.1e-3)
    niacr, piacr, praci = W(big, a_, zero), W(big, b_, zero), W(big, c_, zero)
    niacrs, piacrs, pracis = W(sml, a_, zero), W(sml, b_, zero), W(sml, c_, zero)

    # ice nucleation (INUC=0)
    c = cc & (((qvqvs >= 0.999) & (t <= 265.15)) | (qvqvsi >= 1.25))
    kc2 = 0.005 * xp.exp(0.304 * (tmelt - t)) * 1000.0
    kc2 = mn(kc2, 500.0e3)
    kc2 = mx(kc2 / rho, 0.0)
    c = c & (kc2 > ni + ns + ng)
    nnuccd = W(c, (kc2 - ni - ns - ng) / dt, zero)
    mnuccd = W(c, nnuccd * M["mi0"], zero)

    # deposition/sublimation
    mi = qi >= qsmall
    epsi = W(cc & mi, twopi * n0i * rho * dv / (lami_s * lami_s), zero)
    epss = W(cc & (qs >= qsmall), epss_f, zero)
    epsg = W(cc & (qg >= qsmall), epsg_f, zero)
    dqvi = qv - qvi
    dumd = W(mi, 1.0 - xp.exp(-lami * dcs) * (1.0 + lami * dcs), zero)
    prd = W(mi, epsi * dqvi / abi * dumd, zero)
    msn = qs >= qsmall
    rest = epsi * dqvi / abi * (1.0 - dumd)
    prds = W(msn, epss * dqvi / abi + rest, zero)
    prd = W(msn, prd, prd + rest)
    prdg = epsg * dqvi / abi
    prd, prds, prdg = (W(cc, a, zero) for a in (prd, prds, prdg))
    pre_c = W(cc, pre_all, zero)
    dum = dqvi / dt
    fudgef = 0.9999
    sum_dep = prd + prds + mnuccd + prdg
    c = cc & (
        ((dum > 0.0) & (sum_dep > dum * fudgef))
        | ((dum < 0.0) & (sum_dep < dum * fudgef))
    )
    sd = safe(c, sum_dep)
    mnuccd = W(c, fudgef * mnuccd * dum / sd, mnuccd)
    prd = W(c, fudgef * prd * dum / sd, prd)
    prds = W(c, fudgef * prds * dum / sd, prds)
    prdg = W(c, fudgef * prdg * dum / sd, prdg)
    eprd = W(prd < 0.0, prd, zero)
    prd = W(prd < 0.0, zero, prd)
    eprds = W(prds < 0.0, prds, zero)
    prds = W(prds < 0.0, zero, prds)
    eprdg = W(prdg < 0.0, prdg, zero)
    prdg = W(prdg < 0.0, zero, prdg)

    # conservation of qc
    dum = (
        prc_c + pra_c + mnuccc + psacws + psacwi + qmults + psacwg + pgsacw + qmultg
    ) * dt
    c = cc & (dum > qc) & (qc >= qsmall)
    ratio = qc / safe(c, dum)

    def sc_(a):
        return W(c, a * ratio, a)

    prc_c, pra_c, mnuccc, psacws, psacwi = (
        sc_(a) for a in (prc_c, pra_c, mnuccc, psacws, psacwi)
    )
    qmults, qmultg, psacwg, pgsacw = (sc_(a) for a in (qmults, qmultg, psacwg, pgsacw))
    # conservation of qi
    dum = (
        -prd
        - mnuccc
        + prci
        + prai
        - qmults
        - qmultg
        - qmultr
        - qmultrg
        - mnuccd
        + praci
        + pracis
        - eprd
        - psacwi
    ) * dt
    c = cc & (dum > qi) & (qi >= qsmall)
    ratio = (
        qi / dt + prd + mnuccc + qmults + qmultg + qmultr + qmultrg + mnuccd + psacwi
    ) / safe(c, prci + prai + praci + pracis - eprd)
    prci, prai, praci, pracis, eprd = (
        sc_(a) for a in (prci, prai, praci, pracis, eprd)
    )
    # conservation of qr
    dum = (
        (pracs - pre_c)
        + (qmultr + qmultrg - prc_c)
        + (mnuccr - pra_c)
        + piacr
        + piacrs
        + pgracs
        + pracg
    ) * dt
    c = cc & (dum > qr) & (qr >= qsmall)
    ratio = (qr / dt + prc_c + pra_c) / safe(
        c,
        -pre_c + qmultr + qmultrg + pracs + mnuccr + piacr + piacrs + pgracs + pracg,
    )
    pre_c, pracs, qmultr, qmultrg, mnuccr = (
        sc_(a) for a in (pre_c, pracs, qmultr, qmultrg, mnuccr)
    )
    piacr, piacrs, pgracs, pracg = (sc_(a) for a in (piacr, piacrs, pgracs, pracg))
    # conservation of snow (IGRAUP=0)
    dum = (-prds - psacws - prai - prci - pracs - eprds + psacr - piacrs - pracis) * dt
    c = cc & (dum > qs) & (qs >= qsmall)
    ratio = (qs / dt + prds + psacws + prai + prci + pracs + piacrs + pracis) / safe(
        c, -eprds + psacr
    )
    eprds, psacr = sc_(eprds), sc_(psacr)
    # conservation of graupel
    dum = (
        -psacwg
        - pracg
        - pgsacw
        - pgracs
        - prdg
        - mnuccr
        - eprdg
        - piacr
        - praci
        - psacr
    ) * dt
    c = cc & (dum > qg) & (qg >= qsmall)
    ratio = (
        qg / dt
        + psacwg
        + pracg
        + pgsacw
        + pgracs
        + prdg
        + mnuccr
        + psacr
        + piacr
        + praci
    ) / safe(c, -eprdg)
    eprdg = sc_(eprdg)

    qvt_c = zero + (-pre_c - prd - prds - mnuccd - eprd - eprds - prdg - eprdg)
    tt_c = zero + (
        pre_c * ecnd
        + (prd + prds + mnuccd + eprd + eprds + prdg + eprdg) * edep
        + (
            psacws
            + psacwi
            + mnuccc
            + mnuccr
            + qmults
            + qmultg
            + qmultr
            + qmultrg
            + pracs
            + psacwg
            + pracg
            + pgsacw
            + pgracs
            + piacr
            + piacrs
        )
        * efrz
    )
    qct_c = zero + (
        -pra_c
        - prc_c
        - mnuccc
        + zero
        - psacws
        - psacwi
        - qmults
        - qmultg
        - psacwg
        - pgsacw
    )
    qit_c = zero + (
        prd
        + eprd
        + psacwi
        + mnuccc
        - prci
        - prai
        + qmults
        + qmultg
        + qmultr
        + qmultrg
        + mnuccd
        - praci
        - pracis
    )
    qrt_c = zero + (
        pre_c
        + pra_c
        + prc_c
        - pracs
        - mnuccr
        - qmultr
        - qmultrg
        - piacr
        - piacrs
        - pracg
        - pgracs
    )
    qst_c = zero + (
        prai + psacws + prds + pracs + prci + eprds - psacr + piacrs + pracis
    )
    nst_c = zero + (nsagg + nprci - nscng - ngracs + niacrs)
    qgt_c = zero + (
        pracg + psacwg + pgsacw + pgracs + prdg + eprdg + mnuccr + piacr + praci + psacr
    )
    ngt_c = zero + (nscng + ngracs + nnuccr + niacr)
    nct_c = zero + (-nnuccc - npsacws - npra_c - nprc_c - npsacwi - npsacwg)
    nit_c = zero + (
        nnuccc
        - nprci
        - nprai
        + nmults
        + nmultg
        + nmultr
        + nmultrg
        + nnuccd
        - niacr
        - niacrs
    )
    nrt_c = zero + (
        nprc1_c
        - npracs
        - nnuccr
        + nragg_c
        - niacr
        - niacrs
        - W(cc, npracg, zero)
        - ngracs
    )
    nsubi = nloss(cc & (eprd < 0.0), eprd, qi, ni)
    nsubs = nloss(cc & (eprds < 0.0), eprds, qs, ns)
    nsubr_c = nloss(cc & (pre_c < 0.0), pre_c, qr, nr)
    nsubg = nloss(cc & (eprdg < 0.0), eprdg, qg, ng)
    nit_c = nit_c + nsubi
    nst_c = nst_c + nsubs
    ngt_c = ngt_c + nsubg
    nrt_c = nrt_c + nsubr_c

    def pick(a_w, a_c):
        return W(wr, a_w, W(cc, a_c, zero))

    qv3dten = pick(qvt_w, qvt_c)
    t3dten = pick(tt_w, tt_c)
    qc3dten = pick(qct_w, qct_c)
    qi3dten = W(cc, qit_c, zero)
    qr3dten = pick(qrt_w, qrt_c)
    qni3dten = pick(qst_w, qst_c)
    qg3dten = pick(qgt_w, qgt_c)
    nc3dten = pick(nct_w, nct_c)
    ni3dten = W(cc, nit_c, zero)
    ns3dten = pick(nst_w, nst_c)
    nr3dten = pick(nrt_w, nrt_c)
    ng3dten = pick(ngt_w, ngt_c)
    PRE = pick(pre_w, pre_c)
    EVPMS = W(wr, evpms, zero)
    EVPMG = W(wr, evpmg, zero)

    # ---- saturation adjustment (every processed level) --------------------
    dumt = t + dt * t3dten
    dumqv = qv + dt * qv3dten
    dumqc = qc + dt * qc3dten
    dumqr = qr + dt * qr3dten
    dummqs = qs + dt * qni3dten
    dumqg = qg + dt * qg3dten
    dumqi = W(cc, qi + dt * qi3dten, zero)
    if eqtset == 2:
        pres = W(act, rho * dumt * (M["r"] + M["rv"] * dumqv), pres)
    sa = satadj2(
        bk,
        M,
        tnew=dumt,
        prs=pres,
        qvnew=dumqv,
        qcnew=dumqc,
        qrnew=dumqr,
        qinew=dumqi,
        qsnew=dummqs,
        qgnew=dumqg,
        rho=rho,
        eqtset=eqtset,
        tsmall=tsmall,
        mask=act,
        maxiter=satadj_maxiter,
    )
    pres = sa["prs"]
    c = act & (xp.abs(sa["tnew"] - dumt) >= 0.0001)
    PCC = W(c, (sa["qcnew"] - dumqc) * rdt, zero)
    qv3dten = W(c, qv3dten - PCC, qv3dten)
    qc3dten = W(c, qc3dten + PCC, qc3dten)
    t3dten = W(c, t3dten + (sa["tnew"] - dumt) * rdt, t3dten)

    # ---- droplet activation (INUM=0; IACT=2, IBASE=1, ISUB=1) -------------
    if inum == 0:
        nc3dten = nc3dten + _activation(
            bk,
            M,
            wh,
            mn,
            mx,
            safe,
            act=act,
            cold=cc,
            t=t,
            qc=qc,
            qc3dten=qc3dten,
            nc=nc,
            w=w,
            rho=rho,
            pres=pres,
            evs=evs,
            qvs=qvs,
            qvi=qvi,
            xxlv=xxlv,
            cpm=cpm,
            dv=dv,
            kap=kap,
            dqsdt=dqsdt,
            pgam=pgam,
            lamc=lamc,
            epsr=epsr,
            epsi=epsi,
            epss=epss,
            epsg=epsg,
            dt=dt,
            zero=zero,
        )

    ltrue = xp.any(act, axis=0)  # [j, i]
    lt = ltrue[None]

    # ---------------- kloop2: fall speeds -----------------------------------
    dumi = qi + qi3dten * dt
    dumqs = qs + qni3dten * dt
    dumr = qr + qr3dten * dt
    dumfni = ni + ni3dten * dt
    dumfns = ns + ns3dten * dt
    dumfnr = nr + nr3dten * dt
    dumc = qc + qc3dten * dt
    dumfnc = nc + nc3dten * dt
    dumg = qg + qg3dten * dt
    dumfng = ng + ng3dten * dt
    if inum == 1:
        dumfnc = nc
    dumfni, dumfns, dumfnc, dumfnr, dumfng = (
        mx(a, 0.0) for a in (dumfni, dumfns, dumfnc, dumfnr, dumfng)
    )

    def dlam(q, n, cfac, rexp, lmin, lmax):
        m = q >= qsmall
        lam = pw(cfac * safe(m, n) / safe(m, q), rexp)
        return mn(mx(lam, lmin), lmax), m

    dlami, mI = dlam(dumi, dumfni, cons[12], 1.0 / di, M["lammini"], M["lammaxi"])
    dlamr, mR = dlam(dumr, dumfnr, pirw, 1.0 / 3.0, M["lamminr"], M["lammaxr"])
    dlams, mS = dlam(dumqs, dumfns, cons[1], 1.0 / ds, M["lammins"], M["lammaxs"])
    dlamg, mG = dlam(dumg, dumfng, cons[2], 1.0 / dg, M["lamming"], M["lammaxg"])
    mC = dumc >= qsmall
    pg2 = pgam_of(safe(mC, nc))
    g1 = gamma(bk, pg2 + 1.0)
    g4 = gamma(bk, pg2 + 4.0)
    dlamc = pw(cons[26] * safe(mC, dumfnc) * g4 / (safe(mC, dumc) * g1), 1.0 / 3.0)
    dlamc = mn(mx(dlamc, (pg2 + 1.0) / 60.0e-6), (pg2 + 1.0) / 1.0e-6)
    bc = M["bc"]
    pbc = pw(dlamc, bc)
    unc = W(mC, acn * gamma(bk, 1.0 + bc + pg2) / (pbc * g1), zero)
    umc = W(mC, acn * gamma(bk, 4.0 + bc + pg2) / (pbc * g4), zero)
    if int(testcase) == 5:
        umc5 = (
            1.19e8
            * math.pow(3.0 / (4.0 * PI * rhow * ndcnst * 1.0e6), 2.0 / 3.0)
            * pw(rho * safe(mC, dumc), 2.0 / 3.0)
            * math.exp(5.0 * math.log(1.5) ** 2)
        )
        umc = W(mC, umc5, zero)
    pbi = pw(dlami, bi)
    uni = W(mI, ain * cons[27] / pbi, zero)
    umi = W(mI, ain * cons[28] / pbi, zero)
    pbr = pw(dlamr, br)
    unr2 = W(mR, arn * cons[6] / pbr, zero)
    umr2 = W(mR, arn * cons[4] / pbr, zero)
    pbs = pw(dlams, bs)
    ums2 = W(mS, asn * cons[3] / pbs, zero)
    uns2 = W(mS, asn * cons[5] / pbs, zero)
    pbg = pw(dlamg, bg)
    umg2 = W(mG, agn * cons[7] / pbg, zero)
    ung2 = W(mG, agn * cons[8] / pbg, zero)
    ums2 = mn(ums2, 1.2 * dum54)
    uns2 = mn(uns2, 1.2 * dum54)
    umi = mn(umi, 1.2 * dum35)
    uni = mn(uni, 1.2 * dum35)
    umr2 = mn(umr2, 9.1 * dum54)
    unr2 = mn(unr2, 9.1 * dum54)
    umg2 = mn(umg2, 20.0 * dum54)
    ung2 = mn(ung2, 20.0 * dum54)

    # below the precipitation edge, take the fall speed from the level above
    def fill(carry, x):
        new = tuple(W(v < 1.0e-10, cv, v) for v, cv in zip(x, carry))
        return new, new

    fs_in = (umr2, umi, uni, ums2, uns2, unr2, umc, unc, umg2, ung2)
    _, filled = bk.scan(fill, tuple(a[nk - 1] for a in fs_in), fs_in, reverse=True)
    fr, fi, fni, fs, fns, fnr, fc, fnc, fg, fng = filled

    temz = 1.0 / dzq

    def nsteps(fq, fn):
        rgvm = xp.maximum(fq, fn)
        n = xp.max(xp.floor(rgvm * dt * temz + 1.0), axis=0)
        return W(ltrue, mx(n, 1.0), xp.zeros_like(n))

    dumr, dumi, dumfni = dumr * rho, dumi * rho, dumfni * rho
    dumqs, dumfns, dumfnr = dumqs * rho, dumfns * rho, dumfnr * rho
    dumc, dumfnc, dumg, dumfng = dumc * rho, dumfnc * rho, dumg * rho, dumfng * rho

    # ---------------- sedimentation -----------------------------------------
    zk = xp.zeros_like(rho[:1])
    precrt = xp.zeros_like(rho[0])
    snowrt = xp.zeros_like(rho[0])
    sten = {}
    nten = {"r": nr3dten, "i": ni3dten, "c": nc3dten, "s": ns3dten, "g": ng3dten}
    spec = (
        ("r", fr, fnr, dumr, dumfnr, False),
        ("i", fi, fni, dumi, dumfni, True),
        ("c", fc, fnc, dumc, dumfnc, False),
        ("s", fs, fns, dumqs, dumfns, True),
        ("g", fg, fng, dumg, dumfng, True),
    )
    for name, fq, fn, dq, dn, snow in spec:
        nstep = nsteps(fq, fn)
        nsafe = mx(nstep, 1.0)
        dttmp = dt / nsafe
        temr = 1.0 / (nsafe[None] * rho)

        def body(n, c, fq=fq, fn=fn, nstep=nstep, dttmp=dttmp, temr=temr, snow=snow):
            dq, dn, qst, nt, prc_, snr = c
            on = nstep > n
            fo = fq * dq
            fon = fn * dn
            ft = (xp.concat([fo[1:], zk], axis=0) - fo) * temz
            ftn = (xp.concat([fon[1:], zk], axis=0) - fon) * temz
            o3 = on[None]
            qst = W(o3, qst + ft * temr, qst)
            nt = W(o3, nt + ftn * temr, nt)
            dq = W(o3, dq + ft * dttmp[None], dq)
            dn = W(o3, dn + ftn * dttmp[None], dn)
            prc_ = W(on, prc_ + fo[0] * dttmp, prc_)
            if snow:
                snr = W(on, snr + fo[0] * dttmp, snr)
            return dq, dn, qst, nt, prc_, snr

        if nfall_max is None:
            nmax = xp.max(nstep)
        else:
            nmax = int(nfall_max)
            if not _traced(bk, nstep):
                nn = int(bk.to_numpy(xp.max(nstep)))
                if nn > nmax:
                    raise ValueError(
                        f"m2005micro_graupel: sedimentation needs {nn} sub-steps "
                        f"({name}), more than nfall_max = {nmax}"
                    )
        c = (dq, dn, zero, nten[name], precrt, snowrt)
        _, _, sten[name], nten[name], precrt, snowrt = bk.fori_loop(nmax, body, c)
    nr3dten, ni3dten, nc3dten = nten["r"], nten["i"], nten["c"]
    ns3dten, ng3dten = nten["s"], nten["g"]

    # ---------------- final kloop (columns with hydrometeors) -------------
    qr3dten = qr3dten + sten["r"]
    qi3dten = qi3dten + sten["i"]
    qc3dten = qc3dten + sten["c"]
    qg3dten = qg3dten + sten["g"]
    qni3dten = qni3dten + sten["s"]
    c = (qi >= qsmall) & (t < tmelt) & (lami >= 1.0e-10)
    c = c & (1.0 / safe(c, lami) >= 2.0 * dcs)
    qni3dten = W(c, qni3dten + qi / dt + qi3dten, qni3dten)
    ns3dten = W(c, ns3dten + ni / dt + ni3dten, ns3dten)
    qi3dten = W(c, -qi / dt, qi3dten)
    ni3dten = W(c, -ni / dt, ni3dten)

    f_qc = qc + qc3dten * dt
    f_qi = qi + qi3dten * dt
    f_qs = qs + qni3dten * dt
    f_qr = qr + qr3dten * dt
    f_nc = nc + nc3dten * dt
    f_ni = ni + ni3dten * dt
    f_ns = ns + ns3dten * dt
    f_nr = nr + nr3dten * dt
    f_qg = qg + qg3dten * dt
    f_ng = ng + ng3dten * dt
    f_t = t + t3dten * dt
    f_qv = qv + qv3dten * dt
    f_p = pres
    if eqtset == 2:
        f_p = rho * f_t * (M["r"] + M["rv"] * f_qv)
    _, f_qvs, f_qvi = satvap(f_t, f_p)
    f_qv, f_t, f_qc, f_qr, f_qi, f_qs, f_qg = subsat(
        alltrue, f_qv, f_t, f_qc, f_qr, f_qi, f_qs, f_qg, f_qv / f_qvs, f_qv / f_qvi
    )
    f_qc, f_nc = zsmall(alltrue, f_qc, f_nc)
    f_qr, f_nr = zsmall(alltrue, f_qr, f_nr)
    f_qi, f_ni = zsmall(alltrue, f_qi, f_ni)
    f_qs, f_ns = zsmall(alltrue, f_qs, f_ns)
    f_qg, f_ng = zsmall(alltrue, f_qg, f_ng)
    go = ~(
        (f_qc < qsmall)
        & (f_qi < qsmall)
        & (f_qs < qsmall)
        & (f_qr < qsmall)
        & (f_qg < qsmall)
    )
    # instantaneous melting of cloud ice
    c = go & (f_qi >= qsmall) & (f_t >= tmelt)
    f_qr = W(c, f_qr + f_qi, f_qr)
    f_t = W(c, f_t - f_qi * efrz, f_t)
    f_qi = W(c, zero, f_qi)
    f_nr = W(c, f_nr + f_ni, f_nr)
    f_ni = W(c, zero, f_ni)
    # homogeneous freezing of cloud water and rain (ILIQ=0, IGRAUP=0)
    c = go & (f_t <= 233.15) & (f_qc >= qsmall)
    f_qi = W(c, f_qi + f_qc, f_qi)
    f_t = W(c, f_t + f_qc * efrz, f_t)
    f_qc = W(c, zero, f_qc)
    f_ni = W(c, f_ni + f_nc, f_ni)
    f_nc = W(c, zero, f_nc)
    c = go & (f_t <= 233.15) & (f_qr >= qsmall)
    f_qg = W(c, f_qg + f_qr, f_qg)
    f_t = W(c, f_t + f_qr * efrz, f_t)
    f_qr = W(c, zero, f_qr)
    f_ng = W(c, f_ng + f_nr, f_ng)
    f_nr = W(c, zero, f_nr)
    f_ni, f_ns, f_nc, f_nr, f_ng = (
        W(go, mx(a, 0.0), a) for a in (f_ni, f_ns, f_nc, f_nr, f_ng)
    )
    _, _, f_ni = lamlim(
        go,
        f_qi,
        f_ni,
        cons[12],
        1.0 / di,
        M["lammini"],
        M["lammaxi"],
        lambda lam, q: pw(lam, di + 1.0) * q / cons[12],
    )
    _, _, f_nr = lamlim(
        go,
        f_qr,
        f_nr,
        pirw,
        1.0 / 3.0,
        M["lamminr"],
        M["lammaxr"],
        lambda lam, q: ((lam * lam) * (lam * lam)) * q / pirw,
    )
    _, _, f_nc, _, _, _ = lamc_lim(go, f_qc, f_nc)
    _, _, f_ns = lamlim(
        go,
        f_qs,
        f_ns,
        cons[1],
        1.0 / ds,
        M["lammins"],
        M["lammaxs"],
        lambda lam, q: pw(lam, ds + 1.0) * q / cons[1],
    )
    _, _, f_ng = lamlim(
        go,
        f_qg,
        f_ng,
        cons[2],
        1.0 / dg,
        M["lamming"],
        M["lammaxg"],
        lambda lam, q: pw(lam, dg + 1.0) * q / cons[2],
    )
    f_ni = mn(f_ni, 0.3e6 / rho)
    if inum == 0:
        f_nc = mn(f_nc, (M["nanew1"] + M["nanew2"]) / rho)
    else:
        f_nc = ndcnst * 1.0e6 / rho

    def fin(a, b):
        return W(lt, a, b)

    zero2 = xp.zeros_like(precrt)
    return {
        "t": fin(f_t, t),
        "qv": fin(f_qv, qv),
        "qc": fin(f_qc, qc),
        "qi": fin(f_qi, qi),
        "qs": fin(f_qs, qs),
        "qr": fin(f_qr, qr),
        "nc": fin(f_nc, nc),
        "ni": fin(f_ni, ni),
        "ns": fin(f_ns, ns),
        "nr": fin(f_nr, nr),
        "qg": fin(f_qg, qg),
        "ng": fin(f_ng, ng),
        "precrt": W(ltrue, precrt, zero2),
        "snowrt": W(ltrue, snowrt, zero2),
        "pre": PRE,
        "evpms": EVPMS,
        "evpmg": EVPMG,
        "pcc": PCC,
        "fc": fin(fc, zero),
        "fr": fin(fr, zero),
        "fi": fin(fi, zero),
        "fs": fin(fs, zero),
        "fg": fin(fg, zero),
    }


def _activation(
    bk,
    M,
    wh,
    mn,
    mx,
    safe,
    *,
    act,
    cold,
    t,
    qc,
    qc3dten,
    nc,
    w,
    rho,
    pres,
    evs,
    qvs,
    qvi,
    xxlv,
    cpm,
    dv,
    kap,
    dqsdt,
    pgam,
    lamc,
    epsr,
    epsi,
    epss,
    epsg,
    dt,
    zero,
):
    """Droplet activation for INUM=0 (IACT=2, IBASE=1, ISUB=1): the NC3DTEN
    increment of M2005MICRO_GRAUPEL's warm and cold branches."""
    xp = bk.xp
    W = xp.where

    def pw(x, e):
        return xp.pow(x, e)

    g_, mw, rhow, rr, ma = M["g"], M["mw"], M["rhow"], M["rr"], M["ma"]
    nanew1, nanew2 = M["nanew1"], M["nanew2"]
    tmelt = M["tmelt"]
    rsb = math.pow(M["bact"], 0.5)
    lsig1 = math.log(M["sig1"])
    lsig2 = math.log(M["sig2"])

    wv = w  # ISUB=1
    ca = act & (qc + qc3dten * dt >= M["qsmall"]) & (wv >= 0.001)
    thr = 0.05e-3 / rho
    qcp = xp.concat([qc[:1], qc[:-1]], axis=0)
    thrp = xp.concat([thr[:1], thr[:-1]], axis=0)
    first = bk.set(bk.full(qc.shape, 0.0), 0, 1.0) > 0.5  # K = KTS
    idrop = (qc <= thr) | first | ((qc > thr) & (qcp <= thrp))

    # safe inputs on lanes that do not activate
    tt = safe(ca, t, 280.0)
    ev = safe(ca, evs, 1000.0)
    pr = safe(ca, pres, 1.0e5)
    xl = safe(ca, xxlv, 2.5e6)
    cp_ = safe(ca, cpm, 1005.0)
    dv_ = safe(ca, dv, 2.0e-5)
    ka = safe(ca, kap, 2.0e-2)
    rh = safe(ca, rho, 1.0)
    ww = safe(ca, wv, 1.0)

    sigvl = 0.0761 - 1.55e-4 * (tt - tmelt)
    aact = 2.0 * mw / (rhow * rr) * sigvl / tt
    sm1 = 2.0 / rsb * pw(aact / (3.0 * M["rm1"]), 1.5)
    sm2 = 2.0 / rsb * pw(aact / (3.0 * M["rm2"]), 1.5)

    def nact(smax):
        uu1 = 2.0 * xp.log(sm1 / smax) / (4.242 * lsig1)
        uu2 = 2.0 * xp.log(sm2 / smax) / (4.242 * lsig2)
        d1 = nanew1 / 2.0 * (1.0 - derf1(bk, uu1))
        d2 = nanew2 / 2.0 * (1.0 - derf1(bk, uu2))
        d2 = (d1 + d2) / rh
        return mn(d2, (nanew1 + nanew2) / rh)

    alpha = g_ * mw * xl / (cp_ * rr * (tt * tt)) - g_ * ma / (rr * tt)
    gamm = rr * tt / (ev * mw) + mw * (xl * xl) / (cp_ * pr * ma * tt)
    gg = 1.0 / (
        rhow * rr * tt / (ev * dv_ * mw)
        + xl * rhow / (ka * tt) * (xl * mw / (tt * rr) - 1.0)
    )
    awg = alpha * ww / gg
    psi = 2.0 / 3.0 * pw(awg, 0.5) * aact
    eta1 = pw(awg, 1.5) / (2.0 * PI * rhow * gamm * nanew1)
    eta2 = pw(awg, 1.5) / (2.0 * PI * rhow * gamm * nanew2)
    dum1 = (
        1.0
        / (sm1 * sm1)
        * (
            M["f11"] * pw(psi / eta1, 1.5)
            + M["f21"] * pw((sm1 * sm1) / (eta1 + 3.0 * psi), 0.75)
        )
    )
    dum2 = (
        1.0
        / (sm2 * sm2)
        * (
            M["f12"] * pw(psi / eta2, 1.5)
            + M["f22"] * pw((sm2 * sm2) / (eta2 + 3.0 * psi), 0.75)
        )
    )
    smax = 1.0 / pw(dum1 + dum2, 0.5)
    dumact = nact(smax)  # N activated from the vertical velocity

    # IDROP=0: in-cloud activation from the quasi-steady supersaturation
    c0 = ca & ~idrop
    rho0 = safe(c0, rho)
    ncs = safe(c0, nc)
    tauc = 1.0 / (2.0 * PI * rho0 * safe(c0, dv) * ncs * (pgam + 1.0) / safe(c0, lamc))

    def tau(eps):
        m = eps > 1.0e-8
        return wh(m, 1.0 / safe(m, eps), 1.0e8)

    taur = tau(epsr)
    taui = tau(epsi)
    taus = tau(epss)
    taug = tau(epsg)
    qvs0 = safe(c0, qvs)
    dum3 = (
        (
            -qvs0 * rho0 / (safe(c0, pres, 1.0e5) - safe(c0, evs, 1.0e3))
            + dqsdt / M["cp"]
        )
        * g_
        * ww
    )
    d3w = dum3 * tauc * taur / (tauc + taur)
    a1 = tauc * taur * taui * taug
    a2 = tauc * taur * taus * taug
    a3 = tauc * taur * taui * taus
    a4 = taur * taui * taus * taug
    a5 = tauc * taui * taus * taug
    d3c = (
        dum3 * tauc * taur * taui * taus * taug
        - (qvs0 - safe(c0, qvi)) * (a1 + a2 + a3)
    ) / (a1 + a2 + a3 + a4 + a5)
    dum3 = W(cold, d3c, d3w)
    sm = dum3 / qvs0
    c1 = c0 & (sm >= 1.0e-6)
    n0 = mn(nact(safe(c1, sm)), dumact)

    nnew = W(idrop, dumact, n0)
    inc = mx((nnew - nc) / dt, 0.0)
    return W(ca & (idrop | c1), inc, zero)


# ---------------------------------------------------------------------------
# MP_GRAUPEL


def _seq_rows(bk, x):
    """Row sums of a 2-D ``x[r, n]`` accumulated sequentially along n."""
    from .misc import level_sums

    return level_sums(bk, x)


def mp_graupel(
    bk,
    g,
    cfg,
    M,
    *,
    dt,
    t3d,
    qv,
    qc,
    qr,
    qi,
    qs,
    qg,
    nni,
    ns,
    nr,
    ng,
    nc,
    p,
    rho,
    ddz,
    w,
    rainnc,
    prate,
    tcond,
    tevac,
    tevar,
    train,
    ruh,
    rvh,
    rmh,
    rr,
    nfall_max=None,
):
    """MP_GRAUPEL: run M2005MICRO_GRAUPEL on the interior columns, then the
    precipitation accumulation, the budgets (CM1 order: per j, i then k;
    then over j) and the fall speeds (vtc, vtr, vti, vts, vtg).

    3-D arrays are ``g.s_shape`` (``w`` is ``g.w_shape``), ``rainnc`` is
    ``[nrain, j, i]``, ``prate`` ``[j, i]`` (with halos).  ``nc`` is read and
    written only for inum=0.  Returns dict with t3d, qv, qc, qr, qi, qs, qg,
    nni, ns, nr, ng, nc, rainnc, prate, tcond, tevac, tevar, train, vtc, vtr,
    vti, vts, vtg (the vt arrays are interior-only ``[k, j, i]``)."""
    xp = bk.xp
    K, J, I = g.K, g.J, g.I
    S = (K, J, I)
    nk = g.nk
    inum = int(M["inum"])
    eqtset = int(_opt(cfg, "eqtset"))
    tsmall = float(_opt(cfg, "tsmall"))
    testcase = int(_opt(cfg, "testcase"))
    if int(_opt(cfg, "axisymm")) != 0:
        raise NotImplementedError("mp_graupel: axisymm not ported")
    rdt = 1.0 / dt
    k0 = K.start
    wc = 0.5 * (w[k0 : k0 + nk, J, I] + w[k0 + 1 : k0 + nk + 1, J, I])
    ncin = nc[S] if inum == 0 else xp.zeros_like(t3d[S])
    o = m2005micro_graupel(
        bk,
        M,
        t=t3d[S],
        qv=qv[S],
        qc=qc[S],
        qi=qi[S],
        qs=qs[S],
        qr=qr[S],
        nc=ncin,
        ni=nni[S],
        ns=ns[S],
        nr=nr[S],
        qg=qg[S],
        ng=ng[S],
        pres=p[S],
        rho=rho[S],
        dzq=ddz[S],
        w=wc,
        dt=dt,
        eqtset=eqtset,
        tsmall=tsmall,
        testcase=testcase,
        nfall_max=nfall_max,
        satadj_maxiter=int(_opt(cfg, "satadj_maxiter")),
    )

    def put(a, v):
        return bk.set(bk.copy(a), S, v)

    out = {
        "t3d": put(t3d, o["t"]),
        "qv": put(qv, o["qv"]),
        "qc": put(qc, o["qc"]),
        "qr": put(qr, o["qr"]),
        "qi": put(qi, o["qi"]),
        "qs": put(qs, o["qs"]),
        "qg": put(qg, o["qg"]),
        "nni": put(nni, o["ni"]),
        "ns": put(ns, o["ns"]),
        "nr": put(nr, o["nr"]),
        "ng": put(ng, o["ng"]),
        "nc": put(nc, o["nc"]) if inum == 0 else nc,
    }
    pr = o["precrt"]
    out["rainnc"] = bk.set(
        bk.copy(rainnc), (slice(None), J, I), rainnc[:, J, I] + 0.1 * pr[None]
    )
    out["prate"] = bk.set(bk.copy(prate), (J, I), pr * rdt)

    # budgets
    ru = ruh[I][None, :] if ruh.ndim == 1 else ruh[..., I]
    rv = rvh[J][:, None] if rvh.ndim == 1 else rvh[..., J, :]
    dx, dy, dz = g.dx, g.dy, g.dz
    tem1 = dx * dy * dz
    b1 = _seq_rows(bk, pr * ru * rv * dx * dy)  # [j]
    tem2 = dt * rr[S] * ru[None] * rv[None] * rmh[S] * tem1  # [k, j, i]
    pcc = o["pcc"]
    zero = xp.zeros_like(pcc)

    def jik(a):  # [k, j, i] -> [j, i*k] with k fastest
        a = xp.permute_dims(a, (1, 2, 0))
        return xp.reshape(a, (a.shape[0], -1))

    b2 = -_seq_rows(bk, jik(tem2 * (o["pre"] + o["evpms"] + o["evpmg"])))
    b3 = _seq_rows(bk, jik(tem2 * xp.maximum(pcc, zero)))
    b4 = -_seq_rows(bk, jik(tem2 * xp.minimum(pcc, zero)))

    def acc(tot, b):
        tot = xp.asarray(tot, dtype=b.dtype)
        return _seq_rows(bk, xp.concat([xp.reshape(tot, (1,)), b])[None])[0]

    out["train"] = acc(train, b1)
    out["tevar"] = acc(tevar, b2)
    out["tcond"] = acc(tcond, b3)
    out["tevac"] = acc(tevac, b4)
    for n in ("c", "r", "i", "s", "g"):
        out["vt" + n] = o["f" + n]
    return out


# ---------------------------------------------------------------------------
# mp_driver.F, ptype=5


def mp_driver_ptype5(
    bk,
    g,
    cfg=None,
    *,
    dt,
    qbudget,
    asq,
    ruh,
    rvh,
    rmh,
    mf,
    rain,
    prate,
    pi0,
    th0,
    rho,
    prs,
    pp3d,
    th3d,
    q3d,
    w3d,
    qten=None,
    thten=None,
    dum1=None,
    dum3=None,
    nfall_max=None,
    satadj_maxiter=None,
    getvt=False,
    **unused,
):
    """mp_driver.F ``mp_driver`` for ptype=5 (Morrison).

    ``q3d``/``qten`` are ``[numq, k, j, i]`` in param.F's order (qv, qc, qr,
    qi, qs, qg, nci, ncs, ncr, ncg[, ncc for inum=0]); other shapes as
    ``mp_driver_kessler``.  Sequence (as CM1): T -> dum1 and thten, qv ->
    qten(nqv), rho -> dum3; pdefqtest on qv..qg; MP_GRAUPEL (fall speeds
    into qten(nqc,nqr,nqi,nqs,nqg) when ``getvt`` or efall=1); efall=1: getcvm and getefall, T +=
    dt*heating; then th3d/pp3d/prs (eqtset=2) or th3d/rho (eqtset=1) where
    T or qv changed.

    Returns dict with q3d, th3d, pp3d, prs, rho, qbudget, asq, rain, prate,
    qten, thten, dum1, dum3."""
    from .kessler import getefall
    from .misc import getcvm
    from .pdef import pdefqtest

    xp = bk.xp
    S = (g.K, g.J, g.I)
    numq = int(cfg.numq)
    inum = 0 if numq == 11 else 1
    M = graupel_init(
        ihail=int(_opt(cfg, "ihail")), ndcnst=float(_opt(cfg, "ndcnst")), inum=inum
    )
    qsmall = float(_opt(cfg, "qsmall"))
    tsmall = float(_opt(cfg, "tsmall"))
    eqtset = int(_opt(cfg, "eqtset"))
    efall = int(_opt(cfg, "efall"))
    if nfall_max is None and int(_opt(cfg, "nfall_max") or 0) > 0:
        nfall_max = int(_opt(cfg, "nfall_max"))
    if satadj_maxiter is not None:
        cfg = _Over(cfg, satadj_maxiter=int(satadj_maxiter))

    def z(a, shape):
        return bk.zeros(shape) if a is None else bk.copy(a)

    qten = z(qten, (numq,) + tuple(g.s_shape))
    thten = z(thten, g.s_shape)
    dum1 = z(dum1, g.s_shape)
    dum3 = z(dum3, g.s_shape)

    t = (th0[S] + th3d[S]) * (pi0[S] + pp3d[S])
    dum1 = bk.set(dum1, S, t)
    thten = bk.set(thten, S, t)
    qten = bk.set(qten, (0,) + S, q3d[(0,) + S])
    dum3 = bk.set(dum3, S, rho[S])

    q = [q3d[n] for n in range(numq)]
    asq_l = [asq[n] for n in range(asq.shape[0])]
    for n in range(6):
        o = pdefqtest(
            bk,
            g,
            cfg,
            rmax=0.0 if n == 0 else qsmall,
            asq=asq_l[n],
            ruh=ruh,
            rvh=rvh,
            rmh=rmh,
            rho=rho,
            q3d=q[n],
        )
        q[n], asq_l[n] = o["q3d"], o["asq"]
    dum4 = bk.set(bk.zeros(g.s_shape), S, g.dz * rmh[S])
    nc = q[10] if numq == 11 else bk.zeros(g.s_shape)

    qb = [qbudget[n] for n in range(qbudget.shape[0])]
    o = bk.kernel(mp_graupel, traced=("dt",))(
        bk,
        g,
        cfg,
        M,
        dt=dt,
        t3d=dum1,
        qv=q[0],
        qc=q[1],
        qr=q[2],
        qi=q[3],
        qs=q[4],
        qg=q[5],
        nni=q[6],
        ns=q[7],
        nr=q[8],
        ng=q[9],
        nc=nc,
        p=prs,
        rho=rho,
        ddz=dum4,
        w=w3d,
        rainnc=rain,
        prate=prate,
        tcond=qb[0],
        tevac=qb[1],
        tevar=qb[4],
        train=qb[5],
        ruh=ruh,
        rvh=rvh,
        rmh=rmh,
        rr=dum3,
        nfall_max=nfall_max,
    )
    dum1 = o["t3d"]
    for n, k in enumerate(
        ("qv", "qc", "qr", "qi", "qs", "qg", "nni", "ns", "nr", "ng")
    ):
        q[n] = o[k]
    if numq == 11:
        q[10] = o["nc"]
    rain, prate = o["rainnc"], o["prate"]
    qb[0], qb[1], qb[4], qb[5] = o["tcond"], o["tevac"], o["tevar"], o["train"]
    getvt = bool(getvt) or efall == 1  # mp_driver.F: efall=1 sets getvt
    vt = {}
    for name, key in (
        ("nqc", "vtc"),
        ("nqr", "vtr"),
        ("nqi", "vti"),
        ("nqs", "vts"),
        ("nqg", "vtg"),
    ):
        idx = int(getattr(cfg, name)) - 1
        if getvt:
            qten = bk.set(qten, (idx,) + S, o[key])
        vt[name] = idx

    if efall == 1:
        dum2 = getcvm(bk, g, cfg, q=q)["cvm"]
        dum4 = bk.zeros(g.s_shape)
        for setzero, cpx, name in (
            (1, C.cpl, "nqc"),
            (0, C.cpl, "nqr"),
            (0, C.cpi, "nqi"),
            (0, C.cpi, "nqs"),
            (0, C.cpi, "nqg"),
        ):
            n = vt[name]
            dum4 = getefall(
                bk,
                g,
                cfg,
                setzero=setzero,
                cpx=cpx,
                mf=mf,
                t=dum1,
                cvm=dum2,
                tten=dum4,
                q3d=q[n],
                vr=qten[n],
            )["tten"]
        Kl = (slice(g.K.start, g.K.start + g.nk - 1), g.J, g.I)
        dum1 = bk.set(dum1, Kl, dum1[Kl] + dt * dum4[Kl])
    elif efall != 0:
        raise NotImplementedError(f"mp_driver ptype=5: efall={efall} not ported")

    qv = q[0]
    T = dum1[S]
    qvi = qv[S]
    ch = (xp.abs(T - thten[S]) >= tsmall) | (xp.abs(qvi - qten[(0,) + S]) >= qsmall)
    if eqtset == 2:
        pn = rho[S] * (C.rd + C.rv * qvi) * T
        ppn = xp.pow(pn * C.rp00, C.rovcp) - pi0[S]
        thn = T / (pi0[S] + ppn) - th0[S]
        prs = bk.set(bk.copy(prs), S, xp.where(ch, pn, prs[S]))
        pp3d = bk.set(bk.copy(pp3d), S, xp.where(ch, ppn, pp3d[S]))
        th3d = bk.set(bk.copy(th3d), S, xp.where(ch, thn, th3d[S]))
    else:
        thn = T / (pi0[S] + pp3d[S]) - th0[S]
        rhn = prs[S] / (C.rd * T * (1.0 + qvi * C.reps))
        th3d = bk.set(bk.copy(th3d), S, xp.where(ch, thn, th3d[S]))
        rho = bk.set(bk.copy(rho), S, xp.where(ch, rhn, rho[S]))

    dt_ = qbudget.dtype
    return {
        "q3d": xp.stack(q, axis=0),
        "th3d": th3d,
        "pp3d": pp3d,
        "prs": prs,
        "rho": rho,
        "qbudget": xp.stack([xp.asarray(b, dtype=dt_) for b in qb]),
        "asq": xp.stack([xp.asarray(a, dtype=dt_) for a in asq_l]),
        "rain": rain,
        "prate": prate,
        "qten": qten,
        "thten": thten,
        "dum1": dum1,
        "dum3": dum3,
    }


class _Over:
    """``cfg`` with a few options overridden (attribute and ``get`` access)."""

    def __init__(self, cfg, **over):
        self._cfg, self._over = cfg, over

    def get(self, name, default=None):
        if name in self._over:
            return self._over[name]
        get = getattr(self._cfg, "get", None)
        return get(name, default) if get else getattr(self._cfg, name, default)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._over:
            return self._over[name]
        return getattr(self._cfg, name)
