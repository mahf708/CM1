"""GFDL surface layer (sfcmodel=4, the HWRF "GFDL surface layer"), ported
from

  src/module_sf_gfdl.F      ``sf_gfdl`` (SF_GFDL) and ``mflux2`` (MFLUX2:
                            Monin-Obukhov fluxes with the Hicks universal
                            functions and Wegstein iterations, the GFDL slab
                            T* prediction over land)
  src/module_sf_exchcoef.F  ``znot_wind10m`` with icoef_sf=6 (``znot_m_v6``,
                            ``znot_t_v6``: roughness lengths over water from
                            the 10-m wind)
  src/module_gfs_funcphys.F ``fpvs`` (the saturation vapor pressure table) and
                            the table it reads (``gpvs`` / ``fpvsx``)

as src/turb.F (sfc_and_turb) calls it; ``surface_layer(ctx)`` is that glue
(registered as sfcmodel=4 in cm1x.pbl).  The order of operations follows the
Fortran (and the bitwise C++ translation in cm1k/physics/sfc_gfdl):
``x**2`` is ``x*x``, integer powers follow GCC's powi chains.

Every point is computed at once (arrays of any shape); MFLUX2's packing of
the stable and unstable points becomes both iterations on every point,
selected with ``xp.where``, and its 30-pass Wegstein iterations a fixed
number of masked passes (a point stops updating once converged, as in the
Fortran).

What CM1's call fixes (turb.F), and so what is ported: ntsflg=1 (GFDL slab
T* prediction over land, and the tsk patch at the i=1 and j=1 rows),
sfenth=0, pert_cd=.false., icoef_sf=6, iwavecpl=0, lcurr_sf=.false.,
isltyp=14, isfflx=1, num_soil_layers=1; ntsflg=0, isfflx=0, other sfenth and
isltyp are keyword options.  Over land SF_GFDL reads smois(i,1,j) of CM1's
2-D smois(ib:ie,jb:je) passed as (ims:ime,jms:jme,1:1): out of bounds for
j>1 (undefined), so the model runs sfcmodel=4 over water only
(``check``); ``sf_gfdl`` takes the soil moisture each point would read.
"""

from __future__ import annotations

import math

from . import constants as C

# module_gfs_physcons (kind_phys = double)
con_rv = 4.6150e2
con_cvap = 1.8460e3
con_cliq = 4.1855e3
con_csol = 2.1060e3
con_hvap = 2.5000e6
con_hfus = 3.3358e5
con_psat = 6.1078e2
con_ttp = 2.7316e2
con_g = 9.80665e0

# MFLUX2 vapor-pressure tables (tab = tab11 // tab22 // tab3, table =
# table1 // table2 // table3 // table4: EQUIVALENCE in the Fortran)
_TAB11 = [0.01403] * 21 + [
    0.01719, 0.02101, 0.02561, 0.03117, 0.03784, 0.04584, 0.05542, 0.06685,
    0.08049, 0.09672, 0.1160, 0.1388, 0.1658, 0.1977, 0.2353, 0.2796, 0.3316,
    0.3925, 0.4638, 0.5472, 0.6444, 0.7577, 0.8894, 1.042, 1.220, 1.425, 1.662,
    1.936, 2.252, 2.615, 3.032, 3.511, 4.060, 4.688, 5.406, 6.225, 7.159, 8.223,
    9.432, 10.80, 12.36, 14.13, 16.12, 18.38, 20.92, 23.80, 27.03, 30.67, 34.76,
    39.35, 44.49, 50.26, 56.71, 63.93, 71.98, 80.97, 90.98, 102.1, 114.5, 128.3,
    143.6, 160.6, 179.4, 200.2, 223.3, 248.8, 276.9, 307.9, 342.1, 379.8, 421.3,
    466.9, 517.0, 572.0, 632.3, 698.5, 770.9, 850.2, 937.0, 1032.0,
]  # fmt: skip
_TAB22 = [
    1146.6, 1272.0, 1408.1, 1556.7, 1716.9, 1890.3, 2077.6, 2279.6, 2496.7,
    2729.8, 2980.0, 3247.8, 3534.1, 3839.8, 4164.8, 4510.5, 4876.9, 5265.1,
    5675.2, 6107.8, 6566.2, 7054.7, 7575.3, 8129.4, 8719.2, 9346.5, 10013.0,
    10722.0, 11474.0, 12272.0, 13119.0, 14017.0, 14969.0, 15977.0, 17044.0,
    18173.0, 19367.0, 20630.0, 21964.0, 23373.0, 24861.0, 26430.0, 28086.0,
    29831.0, 31671.0, 33608.0, 35649.0, 37796.0, 40055.0, 42430.0, 44927.0,
    47551.0, 50307.0, 53200.0, 56236.0, 59422.0, 62762.0, 66264.0, 69934.0,
    73777.0, 77802.0, 82015.0, 86423.0, 91034.0, 95855.0, 100890.0, 106160.0,
    111660.0, 117400.0, 123400.0, 129650.0, 136170.0, 142980.0, 150070.0,
    157460.0, 165160.0, 173180.0, 181530.0, 190220.0, 199260.0,
]  # fmt: skip
_TAB3 = [
    208670.0, 218450.0, 228610.0, 239180.0, 250160.0, 261560.0, 273400.0,
    285700.0, 298450.0, 311690.0, 325420.0, 339650.0, 354410.0, 369710.0,
    385560.0, 401980.0, 418980.0, 436590.0, 454810.0, 473670.0, 493170.0,
    513350.0, 534220.0, 555800.0, 578090.0, 601130.0, 624940.0, 649530.0,
    674920.0, 701130.0, 728190.0, 756110.0, 784920.0, 814630.0, 845280.0,
    876880.0, 909450.0, 943020.0, 977610.0, 1013250.0, 1049940.0, 1087740.0,
]  # fmt: skip
_TABLE1 = [0.0] * 20 + [
    0.3160e-02, 0.3820e-02, 0.4600e-02, 0.5560e-02, 0.6670e-02, 0.8000e-02,
    0.9580e-02, 0.1143e-01, 0.1364e-01, 0.1623e-01, 0.1928e-01, 0.2280e-01,
    0.2700e-01, 0.3190e-01, 0.3760e-01, 0.4430e-01, 0.5200e-01, 0.6090e-01,
    0.7130e-01, 0.8340e-01, 0.9720e-01, 0.1133e00, 0.1317e-00, 0.1526e-00,
    0.1780e-00, 0.2050e-00, 0.2370e-00, 0.2740e-00, 0.3160e-00, 0.3630e-00,
    0.4170e-00, 0.4790e-00, 0.5490e-00, 0.6280e-00, 0.7180e-00, 0.8190e-00,
    0.9340e-00, 0.1064e01, 0.1209e01, 0.1368e01, 0.1560e01, 0.1770e01,
    0.1990e01, 0.2260e01, 0.2540e01, 0.2880e01, 0.3230e01, 0.3640e01,
    0.4090e01, 0.4590e01, 0.5140e01, 0.5770e01, 0.6450e01, 0.7220e01,
]  # fmt: skip
_TABLE2 = [
    0.8050e01, 0.8990e01, 0.1001e02, 0.1112e02, 0.1240e02, 0.1380e02,
    0.1530e02, 0.1700e02, 0.1880e02, 0.2080e02, 0.2310e02, 0.2550e02,
    0.2810e02, 0.3100e02, 0.3420e02, 0.3770e02, 0.4150e02, 0.4560e02,
    0.5010e02, 0.5500e02, 0.6030e02, 0.6620e02, 0.7240e02, 0.7930e02,
    0.8680e02, 0.9500e02, 0.1146e03, 0.1254e03, 0.1361e03, 0.1486e03,
    0.1602e03, 0.1734e03, 0.1873e03, 0.2020e03, 0.2171e03, 0.2331e03,
    0.2502e03, 0.2678e03, 0.2863e03, 0.3057e03, 0.3250e03, 0.3457e03,
    0.3664e03, 0.3882e03, 0.4101e03, 0.4326e03, 0.4584e03, 0.4885e03,
    0.5206e03, 0.5541e03, 0.5898e03, 0.6273e03, 0.6665e03, 0.7090e03,
]  # fmt: skip
_TABLE3 = [
    0.7520e03, 0.7980e03, 0.8470e03, 0.8980e03, 0.9520e03, 0.1008e04,
    0.1067e04, 0.1129e04, 0.1194e04, 0.1263e04, 0.1334e04, 0.1409e04,
    0.1488e04, 0.1569e04, 0.1656e04, 0.1745e04, 0.1840e04, 0.1937e04,
    0.2041e04, 0.2147e04, 0.2259e04, 0.2375e04, 0.2497e04, 0.2624e04,
    0.2756e04, 0.2893e04, 0.3036e04, 0.3186e04, 0.3340e04, 0.3502e04,
    0.3670e04, 0.3843e04, 0.4025e04, 0.4213e04, 0.4408e04, 0.4611e04,
    0.4821e04, 0.5035e04, 0.5270e04, 0.5500e04, 0.5740e04, 0.6000e04,
    0.6250e04, 0.6520e04, 0.6810e04, 0.7090e04, 0.7390e04, 0.7700e04,
    0.8020e04, 0.8350e04, 0.8690e04, 0.9040e04, 0.9410e04, 0.9780e04,
]  # fmt: skip
_TABLE4 = [
    0.1016e05, 0.1057e05, 0.1098e05, 0.1140e05, 0.1184e05, 0.1230e05,
    0.1275e05, 0.1324e05, 0.1373e05, 0.1423e05, 0.1476e05, 0.1530e05,
    0.1585e05, 0.1642e05, 0.1700e05, 0.1761e05, 0.1822e05, 0.1886e05,
    0.1950e05, 0.2018e05, 0.2087e05, 0.2158e05, 0.2229e05, 0.2304e05,
    0.2381e05, 0.2459e05, 0.2539e05, 0.2621e05, 0.2706e05, 0.2792e05,
    0.2881e05, 0.2971e05, 0.3065e05, 0.3160e05, 0.3257e05, 0.3357e05,
    0.3459e05, 0.3564e05, 0.3669e05, 0.3780e05, 0.0000e00,
]  # fmt: skip
TAB = _TAB11 + _TAB22 + _TAB3
TABLE = _TABLE1 + _TABLE2 + _TABLE3 + _TABLE4
assert len(TAB) == 223 and len(TABLE) == 223

# SF_GFDL's soil tables (index isltyp)
MAXSMC = [
    0.339, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406,
    0.468, 0.468, 0.439, 1.000, 0.200, 0.421, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
]  # fmt: skip
DRYSMC = [
    0.010, 0.028, 0.047, 0.084, 0.084, 0.066, 0.067, 0.120, 0.103, 0.100,
    0.126, 0.138, 0.066, 0.000, 0.006, 0.028, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
]  # fmt: skip

NXPVS = 7501


def _mx(xp, a, b):
    if isinstance(a, (int, float)):
        a, b = b, a
    if isinstance(b, (int, float)):
        return xp.clip(a, min=b)
    return xp.maximum(a, b)


def _mn(xp, a, b):
    if isinstance(a, (int, float)):
        a, b = b, a
    if isinstance(b, (int, float)):
        return xp.clip(a, max=b)
    return xp.minimum(a, b)


def _rdiv(xp, s, a):
    """s / a for a Python scalar s (torch's scalar/tensor is s * (1/a))."""
    return xp.full_like(a, s) / a


# ---------------------------------------------------------------------------
# module_gfs_funcphys
def fpvsx(t: float) -> float:
    """fpvsx (scalar; builds the table with libm's pow/exp as the Fortran)."""
    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)
    tr = con_ttp / t
    if t >= tliq:
        return con_psat * (math.pow(tr, xponal)) * math.exp(xponbl * (1.0 - tr))
    if t < tice:
        return con_psat * (math.pow(tr, xponai)) * math.exp(xponbi * (1.0 - tr))
    w = (t - tice) / (tliq - tice)
    pvl = con_psat * (math.pow(tr, xponal)) * math.exp(xponbl * (1.0 - tr))
    pvi = con_psat * (math.pow(tr, xponai)) * math.exp(xponbi * (1.0 - tr))
    return w * pvl + (1.0 - w) * pvi


_GPVS = None


def gpvs():
    """gpvs: (c1xpvs, c2xpvs, tbpvs list) on 180..330 K."""
    global _GPVS
    if _GPVS is None:
        xmin, xmax = 180.0, 330.0
        xinc = (xmax - xmin) / (NXPVS - 1)
        c2xpvs = 1.0 / xinc
        c1xpvs = 1.0 - xmin * c2xpvs
        tb = [fpvsx(xmin + (jx - 1) * xinc) for jx in range(1, NXPVS + 1)]
        _GPVS = (c1xpvs, c2xpvs, tb)
    return _GPVS


def _lookup(bk, table, idx):
    """table[idx] for an integer array idx (0-based)."""
    xp = bk.xp
    tb = bk.asarray(table)
    return xp.reshape(xp.take(tb, xp.reshape(idx, (-1,))), idx.shape)


def fpvs(bk, t):
    """fpvs: saturation vapor pressure (Pa), linear in the gpvs table."""
    xp = bk.xp
    c1xpvs, c2xpvs, tb = gpvs()
    xj = _mn(xp, _mx(xp, c1xpvs + c2xpvs * t, 1.0), float(NXPVS))
    jx = xp.astype(_mn(xp, xj, NXPVS - 1.0), xp.int64)
    t0 = _lookup(bk, tb, jx - 1)
    t1 = _lookup(bk, tb, jx)
    return t0 + (xj - xp.astype(jx, xj.dtype)) * (t1 - t0)


# ---------------------------------------------------------------------------
# module_sf_exchcoef
def znot_m_v6(bk, uref):
    """znot_m_v6 (Bin Liu 2017): momentum roughness (m) from the 10-m wind."""
    xp = bk.xp
    W = xp.where
    p13, p12 = -1.296521881682694e-02, 2.855780863283819e-01
    p11, p10 = -1.597898515251717e00, -8.396975715683501e00
    p25, p24 = 3.790846746036765e-10, 3.281964357650687e-09
    p23, p22 = 1.962282433562894e-07, -1.240239171056262e-06
    p21, p20 = 1.739759082358234e-07, 2.147264020369413e-05
    p35, p34 = 1.840430200185075e-07, -2.793849676757154e-05
    p33, p32 = 1.735308193700643e-03, -6.139315534216305e-02
    p31, p30 = 1.255457892775006e00, -1.663993561652530e01
    p40 = 4.579369142033410e-04
    u2 = uref * uref
    u3 = u2 * uref
    u4 = u2 * u2
    u5 = u3 * u2
    # (the exp arguments on their own branch's range, so that the branch not
    # taken stays finite)
    ua = _mn(xp, uref, 6.5)
    a = xp.exp(p10 + p11 * ua + p12 * (ua * ua) + p13 * (ua * ua * ua))
    b = p25 * u5 + p24 * u4 + p23 * u3 + p22 * u2 + p21 * uref + p20
    uc = _mn(xp, _mx(xp, uref, 15.7), 53.0)
    c2 = uc * uc
    c3 = c2 * uc
    c = xp.exp(p35 * (c3 * c2) + p34 * (c2 * c2) + p33 * c3 + p32 * c2 + p31 * uc + p30)
    out = W(uref > 53.0, p40, 0.0 * uref)
    out = W((uref > 15.7) & (uref <= 53.0), c, out)
    out = W((uref > 6.5) & (uref <= 15.7), b, out)
    return W((uref >= 0.0) & (uref <= 6.5), a, out)


def znot_t_v6(bk, uref):
    """znot_t_v6 (Bin Liu 2017): thermal roughness (m) from the 10-m wind."""
    xp = bk.xp
    W = xp.where
    p00 = 1.100000000000000e-04
    p15, p14 = -9.144581627678278e-10, 7.020346616456421e-08
    p13, p12 = -2.155602086883837e-06, 3.333848806567684e-05
    p11, p10 = -2.628501274963990e-04, 8.634221567969181e-04
    p25, p24 = -8.654513012535990e-12, 1.232380050058077e-09
    p23, p22 = -6.837922749505057e-08, 1.871407733439947e-06
    p21, p20 = -2.552246987137160e-05, 1.428968311457630e-04
    p35, p34 = 3.207515102100162e-12, -2.945761895342535e-10
    p33, p32 = 8.788972147364181e-09, -3.814457439412957e-08
    p31, p30 = -2.448983648874671e-06, 3.436721779020359e-05
    p45, p44 = -3.530687797132211e-11, 3.939867958963747e-09
    p43, p42 = -1.227668406985956e-08, -1.367469811838390e-05
    p41, p40 = 5.988240863928883e-04, -7.746288511324971e-03
    p56, p55 = -1.187982453329086e-13, 4.801984186231693e-11
    p54, p53 = -8.049200462388188e-09, 7.169872601310186e-07
    p52, p51 = -3.581694433758150e-05, 9.503919224192534e-04
    p50 = -1.036679430885215e-02
    p60 = 4.751256171799112e-05
    u2 = uref * uref
    u3 = u2 * uref
    u4 = u2 * u2
    u5 = u3 * u2
    u6 = u3 * u3
    f1 = p15 * u5 + p14 * u4 + p13 * u3 + p12 * u2 + p11 * uref + p10
    f2 = p25 * u5 + p24 * u4 + p23 * u3 + p22 * u2 + p21 * uref + p20
    f3 = p35 * u5 + p34 * u4 + p33 * u3 + p32 * u2 + p31 * uref + p30
    f4 = p45 * u5 + p44 * u4 + p43 * u3 + p42 * u2 + p41 * uref + p40
    f5 = p56 * u6 + p55 * u5 + p54 * u4 + p53 * u3 + p52 * u2 + p51 * uref + p50
    out = W(uref > 80.0, p60, 0.0 * uref)
    out = W((uref > 53.3) & (uref <= 80.0), f5, out)
    out = W((uref > 42.2) & (uref <= 53.3), f4, out)
    out = W((uref > 21.6) & (uref <= 42.2), f3, out)
    out = W((uref > 15.4) & (uref <= 21.6), f2, out)
    out = W((uref >= 5.9) & (uref <= 15.4), f1, out)
    return W((uref >= 0.0) & (uref < 5.9), p00 + 0.0 * uref, out)


def znot_wind10m(bk, w10m, icoef_sf=6):
    """znot_wind10m: (znott, znotm) in m; only icoef_sf=6 is ported."""
    if icoef_sf != 6:
        raise NotImplementedError(f"znot_wind10m: icoef_sf={icoef_sf} is not ported")
    xp = bk.xp
    windmks = _mx(xp, _mn(xp, w10m, 85.0), 1.0)
    return znot_t_v6(bk, windmks), znot_m_v6(bk, windmks)


# ---------------------------------------------------------------------------
# MFLUX2
_cpc = 1.00464e7
_g = 980.6
_rgas = 2.87e6
_og = 1.0 / _g
_amask = -98.0
_c = 0.76
_a = 5.0
_ca = 0.4
_cmo1 = 0.5 * _a - 1.648
_cmo2 = 17.193 + 0.5 * _a - 10.0 * _c
_boycon = 0.61
_rovcp = _rgas / _cpc
_rca = 1.0 / _ca
_enrca = 0.0
_icntx = 30


def _estab(bk, t):
    """MFLUX2's tab/table lookup (saturation vapor pressure, cgs)."""
    xp = bk.xp
    tab1 = t - 153.16
    it = xp.astype(tab1, xp.int64)  # truncation, as int()
    tab2 = tab1 - xp.astype(it, t.dtype)
    idx = xp.clip(it + 1, 1, 223) - 1
    return _lookup(bk, TAB, idx) + tab2 * _lookup(bk, TABLE, idx)


def _hicks_f(xp, z, hi):
    W = xp.where
    zal = xp.log(z)
    r = 1.0 / z
    lo = (zal + _a * z) * _rca
    mid = (8.0 * zal + 4.25 * r - 0.5 * r * r + _cmo1) * _rca
    up = (_c * z + _cmo2) * _rca
    return W(z <= 0.5, lo, W(z <= hi, mid, up))


def _hicks_fs(xp, z):
    W = xp.where
    zal = xp.log(z)
    r = 1.0 / z
    lo = (zal + _a * z) * _rca
    mid = (8.0 * zal + 4.25 * r - 0.5 * r * r + _cmo1) * _rca
    up = (_c * z + _cmo2) * _rca
    s_lo = 1.0 + _a * z
    s_mid = 8.0 - 4.25 * r + r * r
    s_up = _c * z
    return (
        W(z <= 0.5, lo, W(z <= 10.0, mid, up)),
        W(z <= 0.5, s_lo, W(z <= 10.0, s_mid, s_up)),
    )


def _unstable_m(xp, ugz, ux11, ux12):
    y = xp.sqrt(ux11)
    yo = xp.sqrt(ux12)
    x = xp.sqrt(y)
    xo = xp.sqrt(yo)
    xnum = (x * x + 1.0) * ((x + 1.0) * (x + 1.0))
    xden = (xo * xo + 1.0) * ((xo + 1.0) * (xo + 1.0))
    xtan = xp.atan(x) - xp.atan(xo)
    ux3 = xp.log(xnum / xden)
    return ugz - ux3 + 2.0 * xtan


def mflux2(
    bk,
    *,
    zoc,
    tstrc,
    pspc,
    pkmax,
    wetc,
    slwdc,
    upc,
    vpc,
    tpc,
    rpc,
    dt,
    wind10,
    zkmaxin,
    ntsflg=1,
    sfenth=0.0,
    icoef_sf=6,
):
    """MFLUX2 (iwavecpl=0, lcurr_sf=.false., pert_Cd=.false.) on k=1
    inputs in cgs (upc, vpc cm/s, tpc K, rpc kg/kg, pspc/pkmax dyn/cm2,
    zoc cm < 0 over water, wind10 cm/s).  Returns fxh, fxe, fxmx, fxmy, cdm,
    rib, xxfh, xxfh2, zoc, mzoc, tstrc, wind10, tzot, zetaout."""
    xp = bk.xp
    W = xp.where
    z10, z2 = 1000.0, 200.0
    pss = pspc
    tstar = tstrc
    one = xp.ones_like(pss)
    ubot, vbot = upc, vpc
    uvs1 = _mx(xp, xp.sqrt(ubot * ubot + vbot * vbot), 100.0)
    ukmax, vkmax = ubot, vbot
    tkmax, rkmax = tpc, rpc
    windp = xp.sqrt(ukmax * ukmax + vkmax * vkmax)
    wind = _mx(xp, windp, 100.0)
    wind10p = _mx(xp, wind10, 100.0)
    zoc = W(zoc < _amask, -0.0185 * 0.001 * wind10p * wind10p * _og, zoc)
    lnd = zoc > 0.0
    ecof = wetc
    windmks = wind10p * 0.01
    znott, znotm = znot_wind10m(bk, windmks, icoef_sf)
    zot = W(lnd, zoc, -100 * znott)
    zoc = W(lnd, zoc, -100.0 * znotm)
    mzoc = zoc
    tzot = zot

    theta = tkmax / (xp.pow(pkmax / pspc, _rovcp))
    vrtkx = 1.0 + _boycon * rkmax
    zkmax = 100.0 * zkmaxin
    # saturation mixing ratio at the surface
    tsg = tstar
    estso = _estab(bk, tstar)
    psps1 = pss - estso
    psps1 = W(psps1 == 0.0, 0.1, psps1)
    rstso = 0.622 * estso / psps1
    vrts = 1.0 + _boycon * ecof * rstso
    tempa1 = theta * vrtkx - tstar * vrts
    tempa2 = tempa1 * (theta - tstar)
    tempa1 = W(tempa2 < 0.0, 1.0e-4, tempa1)
    tempa1 = W(xp.abs(tempa1) < 1.0e-4, 1.0e-4, tempa1)
    rib = _g * zkmax * tempa1 / (tkmax * vrtkx * wind * wind)
    tab2 = xp.abs(zoc)
    tab1 = _rdiv(xp, 0.95, _c * (1.0 - tab2 / zkmax))
    rib = W(rib > tab1, tab1, rib)
    zeta = _ca * rib / 0.03
    stable = zeta >= 0.0
    azoc = xp.abs(zoc)
    azot = xp.abs(zot)

    # stable points: Wegstein iteration for zeta (Hicks 1976)
    szeta = W(stable, zeta, one)
    szetam = 1.0e30 * one
    sgzm = 0.0 * one
    ifz = stable
    zz = 0.0 * one
    sfm = sfh = sf10 = sf2 = sfzo = zz
    for _ in range(_icntx):
        fmz1 = _hicks_f(xp, szeta, 10.0)
        szetao = azoc / zkmax * szeta
        fmzo1, _s = _hicks_fs(xp, szetao)
        szetho = azot / zkmax * szeta
        fhzo1, sfzo_n = _hicks_fs(xp, szetho)
        szet10 = _rdiv(xp, abs(z10), zkmax) * szeta
        fmz10 = _hicks_f(xp, szet10, 10.0)
        sf10_n = fmz10 - fmzo1
        szet2 = _rdiv(xp, abs(z2), zkmax) * szeta
        fmz2 = _hicks_f(xp, szet2, 2.0)
        sf2_n = fmz2 - fmzo1
        sfm_n = fmz1 - fmzo1
        sfh_n = fmz1 - fhzo1
        sgz = _ca * rib * sfm_n * sfm_n / (sfh_n + _enrca * sfzo_n)
        fmz = (sgz - szeta) / szeta
        go = xp.abs(fmz) >= 5.0e-5
        sfm, sfh, sf10, sf2 = (
            W(ifz, sfm_n, sfm),
            W(ifz, sfh_n, sfh),
            W(ifz, sf10_n, sf10),
            W(ifz, sf2_n, sf2),
        )
        sfzo = W(ifz, sfzo_n, sfzo)
        upd = ifz & go
        sq = (sgz - sgzm) / (szeta - szetam)
        szetam_n = szeta
        szeta_n = (sgz - szeta * sq) / (1.0 - sq)
        szetam = W(upd, szetam_n, szetam)
        szeta = W(upd, szeta_n, szeta)
        sgzm = W(upd, sgz, sgzm)
        ifz = upd

    # unstable points: Wegstein iteration with the Paulson functions
    uzeta = W(stable, -one, zeta)
    uzetam = 1.0e30 * one
    ugzm = 0.0 * one
    ifz = ~stable
    ufzo = ufh = ufm = uf10 = uf2 = zz
    for _ in range(_icntx):
        ugzzo = xp.log(zkmax / azot)
        uzetao = azot / zkmax * uzeta
        ux11 = 1.0 - 16.0 * uzeta
        ux12 = 1.0 - 16.0 * uzetao
        y = xp.sqrt(ux11)
        yo = xp.sqrt(ux12)
        ufzo_n = 1.0 / yo
        ux13 = (1.0 + y) / (1.0 + yo)
        ux21 = xp.log(ux13)
        ufh_n = (ugzzo - 2.0 * ux21) * _rca
        ugzzo = xp.log(zkmax / azoc)
        uzetao = azoc / zkmax * uzeta
        ux12 = 1.0 - 16.0 * uzetao
        ufm_n = _unstable_m(xp, ugzzo, ux11, ux12) * _rca
        ugz10 = xp.log(_rdiv(xp, z10, azoc))
        uzet1o = _rdiv(xp, abs(z10), zkmax) * uzeta
        uzo10 = azoc / zkmax * uzeta
        uf10_n = _unstable_m(xp, ugz10, 1.0 - 16.0 * uzet1o, 1.0 - 16.0 * uzo10) * _rca
        uzet1o = _rdiv(xp, abs(z2), zkmax) * uzeta
        uzo2 = azoc / zkmax * uzeta
        y = xp.sqrt(1.0 - 16.0 * uzet1o)
        yo = xp.sqrt(1.0 - 16.0 * uzo2)
        ux13 = (1.0 + y) / (1.0 + yo)
        ux21 = xp.log(ux13)
        uf2_n = (ugzzo - 2.0 * ux21) * _rca
        ugz = _ca * rib * ufm_n * ufm_n / (ufh_n + _enrca * ufzo_n)
        ux1 = (ugz - uzeta) / uzeta
        go = xp.abs(ux1) >= 5.0e-5
        ufzo, ufh, ufm = W(ifz, ufzo_n, ufzo), W(ifz, ufh_n, ufh), W(ifz, ufm_n, ufm)
        uf10, uf2 = W(ifz, uf10_n, uf10), W(ifz, uf2_n, uf2)
        upd = ifz & go
        uq = (ugz - ugzm) / (uzeta - uzetam)
        uzetam_n = uzeta
        uzeta_n = (ugz - uzeta * uq) / (1.0 - uq)
        uzetam = W(upd, uzetam_n, uzetam)
        uzeta = W(upd, uzeta_n, uzeta)
        ugzm = W(upd, ugz, ugzm)
        ifz = upd

    xxfm = W(stable, sfm, ufm)
    xxfh = W(stable, sfh, ufh)
    xxfh2 = W(stable, sf2, uf2)
    xxsh = W(stable, sfzo, ufzo)
    zetaout = W(stable, szeta, uzeta)
    f10 = W(stable, sf10, uf10)
    # update zo over the ocean
    wndm = wind * 0.01
    ckg = W(
        wndm < 15.0,
        0.0185 * _og * one,
        (sfenth * (4 * 0.000308 * wndm) + (1.0 - sfenth) * 0.0185) * _og,
    )
    zo_new = -ckg * wind * wind / (xxfm * xxfm)
    sea = zoc < 0.0
    zot = W(sea, zo_new, zoc)
    zoc = W(sea, zo_new, zoc)
    # 10 m wind
    wind10 = f10 * uvs1 / xxfm
    wind10 = wind10 * 1.944
    wind10 = wind10 / 1.944  # (+ wind10*cor1 - cor2 above 6000 with cor1 = cor2 = 0)
    # (290)
    calm = windp == 0.0
    windp = W(calm, 100.0, windp)
    ucom = W(calm, 100.0 / math.sqrt(2.0), ukmax)
    vcom = W(calm, 100.0 / math.sqrt(2.0), vkmax)
    rho = pss / (_rgas * (tsg + _enrca * (theta - tsg) * xxsh / (xxfh + _enrca * xxsh)))
    bq1 = wind * rho / (xxfm * (xxfh + _enrca * xxsh))

    # land T* prediction
    if ntsflg != 0:
        alll = 600.0
        xks = 0.01
        hcap = 0.5 / 2.39e-8
        alfus = alll / 2.39e-8
        teps = 0.1
        pith = math.sqrt(4.0 * math.atan(1.0))
        slwa = slwdc / (2.39e-8 * 60.0)
        tss = tstar
        thetap = theta
        rkmaxp = rkmax
        aap = 5.673e-5
        pssp = pss
        ecofp = ecof
        bq1p = _mx(xp, bq1, 0.1e-3)
        delsrad = dt * pith / (hcap * math.sqrt(3600.0 * 24.0 * xks))
        tsm = tss
        rdiff = _mn(xp, rkmaxp - rstso, 0.0)
        tsm2 = tsm * tsm
        foftm = tss + delsrad * (
            slwa
            - aap * (tsm2 * tsm2)
            - _cpc * bq1p * (tsm - thetap)
            + ecofp * alfus * bq1p * rdiff
        )
        tsp = foftm
        ifz = lnd
        for _ in range(_icntx):
            estsop = _estab(bk, tsp)
            psps2 = pssp - estsop
            psps2 = W(psps2 == 0.0, 0.1, psps2)
            rstsop = 0.622 * estsop / psps2
            rdiff = _mn(xp, rkmaxp - rstsop, 0.0)
            tsp2 = tsp * tsp
            foft = tss + delsrad * (
                slwa
                - aap * (tsp2 * tsp2)
                - _cpc * bq1p * (tsp - thetap)
                + ecofp * alfus * bq1p * rdiff
            )
            frac = xp.abs((foft - tsp) / tsp)
            upd = ifz & (frac >= teps)
            den = tsp - tsm
            qf = (foft - foftm) / W(upd, den, one)
            tsm_n = tsp
            tsp_n = (foft - tsp * qf) / (1.0 - qf)
            tsm = W(upd, tsm_n, tsm)
            tsp = W(upd, tsp_n, tsp)
            foftm = W(upd, foft, foftm)
            ifz = upd
        tstrc = W(lnd, tsp, tstrc)
    # (370) fluxes and momentum drag coefficient
    fxh = bq1 * (theta - tsg)
    fxe = ecof * bq1 * (rkmax - rstso)
    fxe = W(fxe > 0.0, 0.0, fxe)
    fxmx = rho / (xxfm * xxfm) * wind * wind * ucom / windp
    fxmy = rho / (xxfm * xxfm) * wind * wind * vcom / windp
    cdm = 1.0 / (xxfm * xxfm)
    return {
        "fxh": fxh,
        "fxe": fxe,
        "fxmx": fxmx,
        "fxmy": fxmy,
        "cdm": cdm,
        "rib": rib,
        "xxfh": xxfh,
        "xxfh2": xxfh2,
        "zoc": zoc,
        "mzoc": mzoc,
        "tstrc": tstrc,
        "wind10": wind10,
        "tzot": tzot,
        "zetaout": zetaout,
    }


# ---------------------------------------------------------------------------
def sf_gfdl(
    bk,
    *,
    u3d,
    v3d,
    t3d,
    qv3d,
    p3d,
    psfc,
    xland,
    gsw,
    glw,
    tsk,
    znt,
    u10,
    v10,
    zkmax,
    dt,
    smois=None,
    cp_=C.cp,
    r=C.rd,
    xlv=C.xlv,
    ep1=C.rv / C.rd - 1.0,
    ep2=C.rd / C.rv,
    karman=C.karman,
    ntsflg=1,
    sfenth=0.0,
    icoef_sf=6,
    isfflx=1,
    isltyp=14,
):
    """SF_GFDL on the interior points: k=1 values of u3d, v3d, t3d, qv3d,
    p3d and the 2-D inputs, all shaped (nj, ni) with [j-1, i-1] the Fortran
    (i, j) (ids = jds = 1: the tsk patch copies row/column 2 into row j=1
    and column i=1, so ni, nj >= 2 with ntsflg=1).  ``smois``: the soil
    moisture each land point reads (needed only over land).  Returns tsk,
    znt, mznt, ust, u10, v10, chs, chs2, cqs2, cpm, psim, psih, hfx, qfx,
    lh, taux, tauy, flhc, flqc, qgh, qsfc, gz1oz0, wspd, br, cd_out, ch_out,
    mol, zol, z0t, fm, fh."""
    xp = bk.xp
    W = xp.where
    prsl1 = p3d * 0.001
    if smois is None:
        wetc = xp.ones_like(psfc)
    else:
        smcdry, smcmax = DRYSMC[isltyp - 1], MAXSMC[isltyp - 1]
        wetc = (smois - smcdry) / (smcmax - smcdry)
        wetc = _mn(xp, _mx(xp, wetc, 0.0), 1.0)
        wetc = W(xland < 1.99, wetc, 1.0)
    pspc = psfc * 10.0
    pkmax = p3d * 10.0
    ps = psfc * 0.001
    q1 = qv3d
    tskin = tsk
    t1 = t3d
    u1, v1 = u3d, v3d
    upc = u3d * 100.0
    vpc = v3d * 100.0
    zoc = znt * 100.0
    zoc = W(xland > 1.99, -zoc, zoc)
    slwdc = gsw + glw
    slwdc = 0.239 * 60.0 * slwdc * 1.0e-4
    wind10 = xp.sqrt(u10 * u10 + v10 * v10)
    first = (wind10 <= 1.0e-10) | (wind10 > 150.0)
    zhalf = -r * t3d * xp.log(pkmax / pspc) / con_g
    w10l = (
        xp.sqrt(u1 * u1 + v1 * v1) * xp.log(_rdiv(xp, 10.0, znt)) / xp.log(zhalf / znt)
    )
    wind10 = W(first, w10l, wind10)
    wind10 = wind10 * 100.0
    rho1 = prsl1 * 1000.0 / (r * t1 * (1.0 + ep1 * q1))
    m = mflux2(
        bk,
        zoc=zoc,
        tstrc=tsk,
        pspc=pspc,
        pkmax=pkmax,
        wetc=wetc,
        slwdc=slwdc,
        upc=upc,
        vpc=vpc,
        tpc=t3d,
        rpc=qv3d,
        dt=dt,
        wind10=wind10,
        zkmaxin=zkmax,
        ntsflg=ntsflg,
        sfenth=sfenth,
        icoef_sf=icoef_sf,
    )
    out = {}
    if ntsflg == 1:
        if tsk.shape[-1] < 2 or tsk.shape[-2] < 2:
            raise NotImplementedError(
                "sf_gfdl: the ntsflg=1 tsk patch needs ni, nj >= 2"
            )
        ts = m["tstrc"]
        # tsk(i,1) = tsk(i,2), then tsk(1,j) = tsk(2,j), with tsk before the call
        ts = xp.concat([tsk[1:2, :], ts[1:, :]], axis=0)
        ts = xp.concat([tsk[:, 1:2], ts[:, 1:]], axis=1)
        out["tsk"] = ts
    else:
        out["tsk"] = tsk
    zoc, mzoc, tzot = m["zoc"], m["mzoc"], m["tzot"]
    wind10 = m["wind10"]
    cdm = m["cdm"]
    out["znt"] = 0.01 * xp.abs(zoc)
    mznt = 0.01 * xp.abs(mzoc)
    out["mznt"] = mznt
    wspd = xp.sqrt(upc * upc + vpc * vpc)
    wspd = _mx(xp, wspd, 100.0) / 100.0
    out["wspd"] = wspd
    u10m = u1 * (wind10 / wspd) / 100.0
    v10m = v1 * (wind10 / wspd) / 100.0
    zk = zkmax
    gz1oz0 = xp.log(zk / mznt)
    out["gz1oz0"] = gz1oz0
    ustar = 0.01 * xp.sqrt(cdm * (upc * upc + vpc * vpc))
    out["taux"] = m["fxmx"] / 10.0
    out["tauy"] = m["fxmy"] / 10.0
    fm = _rdiv(xp, karman, xp.sqrt(cdm))
    fh = karman * m["xxfh"]
    out["psim"] = gz1oz0 - fm
    z0t = 0.01 * xp.abs(tzot)
    out["psih"] = xp.log(zk / z0t) - fh
    out["fm"] = fm
    out["fh"] = fh
    fh2 = karman * m["xxfh2"]
    ch = _rdiv(xp, karman * karman, fm * fh)
    cm = cdm
    rw = wspd / (0.01 * wind10)
    w10ok = wind10 >= 0.1
    out["cd_out"] = W(w10ok, cm * (rw * rw), cm)
    out["ch_out"] = W(
        w10ok,
        ch
        * (wspd / (0.01 * wind10))
        * (xp.log(zk / z0t) / xp.log(_rdiv(xp, 10.0, z0t))),
        ch,
    )
    out["u10"] = u10m
    out["v10"] = v10m
    out["br"] = m["rib"]
    chs = ch * wspd
    chs2 = ustar * karman / fh2
    lnd = xland < 1.9
    chs = W(lnd, _mn(xp, chs, 0.05), chs)
    chs2c = _mn(xp, chs2, 0.05)
    chs2 = W(lnd, W(chs2c < 0, 1.0e-6, chs2c), chs2)
    out["chs"] = chs
    out["chs2"] = chs2
    cpm = cp_ * (1.0 + 0.8 * qv3d)
    out["cpm"] = cpm
    esat = fpvs(bk, t1)
    out["qgh"] = ep2 * esat / (1000.0 * ps - esat)
    esat = fpvs(bk, tskin)
    out["qsfc"] = ep2 * esat / (1000.0 * ps - esat)
    out["ust"] = ustar
    zetaout = m["zetaout"]
    out["zol"] = zetaout
    out["mol"] = zk / (1.0e-20 + zetaout)
    out["z0t"] = z0t
    out["flhc"] = cpm * rho1 * chs
    out["flqc"] = rho1 * chs
    out["cqs2"] = chs2
    if isfflx == 0:
        z = xp.zeros_like(psfc)
        out["hfx"], out["lh"], out["qfx"] = z, z, z
    else:
        hfx = -10.0 * cp_ * m["fxh"]
        hfx = W(xland - 1.5 < 0.0, _mx(xp, hfx, -250.0), hfx)
        qfx = -10.0 * m["fxe"]
        qfx = _mx(xp, qfx, 0.0)
        out["hfx"] = hfx
        out["qfx"] = qfx
        out["lh"] = xlv * qfx
    return out


# ---------------------------------------------------------------------------
# CM1 glue (registered as sfcmodel=4 in cm1x.pbl)
WRF_INPUTS = True
DUMP_KEYS = ("mznt", "mol", "rmol", "zol", "br", "gz1oz0", "psim", "psih")
_OWN = [
    "mznt",
    "chs",
    "chs2",
    "cqs2",
    "cpmm",
    "lh",
    "taux",
    "tauy",
    "flhc",
    "flqc",
    "qgh",
    "swspd",
    "zkmax",
    "cd_out",
    "ch_out",
    "mol",
    "zol",
    "z0t",
    "fm",
    "fh",
    "psim",
    "psih",
    "gz1oz0",
    "br",
    "rmol",
]


def _cfg(cfg, name, default):
    get = getattr(cfg, "get", None)
    return get(name, default) if get else getattr(cfg, name, default)


def check(cfg):
    if int(_cfg(cfg, "ipbl", 0)) not in (0, 3):
        raise NotImplementedError("sfcmodel=4 is ported with ipbl=3 (GFS-EDMF) only")
    if float(_cfg(cfg, "xland0", 2.0)) < 1.5:
        raise NotImplementedError(
            "sfcmodel=4 over land is not ported (SF_GFDL reads smois out of bounds)"
        )
    if int(_cfg(cfg, "imove", 0)) == 1:
        raise NotImplementedError("sfcmodel=4 with imove=1 is not ported")


def init_state(ctx):
    """The 2-D arrays only SF_GFDL uses (cm1.F allocates them with zeros);
    over land SF_GFDL is undefined in CM1, so every point must be water."""
    bk = ctx.bk
    xl = bk.to_numpy(ctx.s["xland"])
    if (xl < 1.5).any():
        raise NotImplementedError("sfcmodel=4 over land is not ported")
    z = bk.zeros(ctx.g.s_shape[1:])
    return {k: bk.copy(z) for k in _OWN if k not in ctx.s}


def surface_layer(ctx):
    """turb.F sfc_and_turb, sfcmodel=4: cd_out = ch_out = mol = zol = 0,
    zkmax = zh(:,:,1), SF_GFDL, u* = max(u*, 1e-8), wspd = swspd; then (all
    water: SLAB only diagnoses qsfc there, like the update_sfc=.false.
    branch) qsfc = qv1 + qfx/flqc; cd = cd_out, ch = cq = ch_out,
    s10 = |(u10, v10)|, rmol = 1/mol (+-1e20 where |mol| <= smeps), and
    thflux/qvflux with dosfcflx or output_sfcflx."""
    bk, g, cfg, s, w = ctx.bk, ctx.g, ctx.cfg, ctx.s, ctx.w
    xp = bk.xp
    ni, nj, n, z = g.ni, g.nj, g.ngxy, g.ngz
    J, I = slice(n, n + nj), slice(n, n + ni)

    def c2(a):
        return a[J, I]

    def c1(a):
        return a[z, J, I]

    zkmax = ctx.b["zh"][z, J, I]
    gz = bk.zeros(g.s_shape[1:])
    r = sf_gfdl(
        bk,
        u3d=c1(w["u"]),
        v3d=c1(w["v"]),
        t3d=c1(w["t"]),
        qv3d=c1(w["qv"]),
        p3d=c1(ctx.prs),
        psfc=c2(s["psfc"]),
        xland=c2(s["xland"]),
        gsw=c2(s.get("gsw", gz)),
        glw=c2(s.get("glw", gz)),
        tsk=c2(s["tsk"]),
        znt=c2(s["znt"]),
        u10=c2(s["u10"]),
        v10=c2(s["v10"]),
        zkmax=zkmax,
        dt=ctx.dt,
    )
    ust = _mx(xp, r["ust"], 1.0e-8)
    qv1 = c1(w["qv"])
    flqc = r["flqc"]
    qsfc = xp.where(flqc != 0.0, qv1 + r["qfx"] / xp.where(flqc != 0.0, flqc, 1.0), qv1)
    mol = r["mol"]
    smeps = float(_cfg(cfg, "smeps", 1.0e-30))
    rmol = xp.where(
        xp.abs(mol) <= smeps,
        xp.where(mol < 0.0, -1.0e20, 1.0e20) + 0.0 * mol,
        1.0 / xp.where(xp.abs(mol) <= smeps, 1.0, mol),
    )
    u10, v10 = r["u10"], r["v10"]
    vals = {
        "tsk": r["tsk"],
        "znt": r["znt"],
        "mznt": r["mznt"],
        "ust": ust,
        "u10": u10,
        "v10": v10,
        "chs": r["chs"],
        "chs2": r["chs2"],
        "cqs2": r["cqs2"],
        "cpmm": r["cpm"],
        "psim": r["psim"],
        "psih": r["psih"],
        "hfx": r["hfx"],
        "qfx": r["qfx"],
        "lh": r["lh"],
        "taux": r["taux"],
        "tauy": r["tauy"],
        "flhc": r["flhc"],
        "flqc": flqc,
        "qgh": r["qgh"],
        "qsfc": qsfc,
        "gz1oz0": r["gz1oz0"],
        "swspd": r["wspd"],
        "wspd": r["wspd"],
        "br": r["br"],
        "cd_out": r["cd_out"],
        "ch_out": r["ch_out"],
        "mol": mol,
        "zol": r["zol"],
        "z0t": r["z0t"],
        "fm": r["fm"],
        "fh": r["fh"],
        "zkmax": zkmax,
        "cd": r["cd_out"],
        "ch": r["ch_out"],
        "cq": r["ch_out"],
        "s10": xp.sqrt(u10 * u10 + v10 * v10),
        "rmol": rmol,
    }
    if bool(_cfg(cfg, "dosfcflx", False)) or int(_cfg(cfg, "output_sfcflx", 0)) == 1:
        rf1 = ctx.rf[z, J, I]
        vals["thflux"] = r["hfx"] / (r["cpm"] * rf1)
        vals["qvflux"] = r["qfx"] / rf1
    out = {}
    for k, v in vals.items():
        base = s.get(k, gz)
        out[k] = bk.set(bk.copy(base), (J, I), v)
    return out
