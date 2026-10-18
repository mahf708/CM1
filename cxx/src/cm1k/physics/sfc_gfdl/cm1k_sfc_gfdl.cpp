// Port of the GFDL surface layer (sfcmodel=4). See cm1k_sfc_gfdl.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// real**real is pow, x**2 is x*x, integer powers follow GCC's powi chains
// (x**3 = x2*x, x**4 = x2*x2, x**5 = x3*x2, x**6 = x3*x3 with x2 = x*x,
// x3 = x2*x), and Fortran's max/min (amax1/amin1) of two numbers is
// fmax2/fmin2. MFLUX2's i-loops only couple a point with itself (the
// stable/unstable packing gathers and scatters the same point), so each
// (i,j) point runs MFLUX2 and the rest of SF_GFDL in order.
#include "physics/sfc_gfdl/cm1k_sfc_gfdl.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cm1k::sfc_gfdl {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// MFLUX2 vapor-pressure tables: tab = tab11 // tab22 // tab3, table =
// table1 // table2 // table3 // table4 (EQUIVALENCE in the Fortran).
constexpr Real tab_data[223] = {
    0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403,
    0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01403,
    0.01403, 0.01403, 0.01403, 0.01403, 0.01403, 0.01719, 0.02101, 0.02561,
    0.03117, 0.03784, 0.04584, 0.05542, 0.06685, 0.08049, 0.09672, 0.1160,
    0.1388, 0.1658, 0.1977, 0.2353, 0.2796, 0.3316, 0.3925, 0.4638,
    0.5472, 0.6444, 0.7577, 0.8894, 1.042, 1.220, 1.425, 1.662,
    1.936, 2.252, 2.615, 3.032, 3.511, 4.060, 4.688, 5.406,
    6.225, 7.159, 8.223, 9.432, 10.80, 12.36, 14.13, 16.12,
    18.38, 20.92, 23.80, 27.03, 30.67, 34.76, 39.35, 44.49,
    50.26, 56.71, 63.93, 71.98, 80.97, 90.98, 102.1, 114.5,
    128.3, 143.6, 160.6, 179.4, 200.2, 223.3, 248.8, 276.9,
    307.9, 342.1, 379.8, 421.3, 466.9, 517.0, 572.0, 632.3,
    698.5, 770.9, 850.2, 937.0, 1032.0, 1146.6, 1272.0, 1408.1,
    1556.7, 1716.9, 1890.3, 2077.6, 2279.6, 2496.7, 2729.8, 2980.0,
    3247.8, 3534.1, 3839.8, 4164.8, 4510.5, 4876.9, 5265.1, 5675.2,
    6107.8, 6566.2, 7054.7, 7575.3, 8129.4, 8719.2, 9346.5, 10013.0,
    10722.0, 11474.0, 12272.0, 13119.0, 14017.0, 14969.0, 15977.0, 17044.0,
    18173.0, 19367.0, 20630.0, 21964.0, 23373.0, 24861.0, 26430.0, 28086.0,
    29831.0, 31671.0, 33608.0, 35649.0, 37796.0, 40055.0, 42430.0, 44927.0,
    47551.0, 50307.0, 53200.0, 56236.0, 59422.0, 62762.0, 66264.0, 69934.0,
    73777.0, 77802.0, 82015.0, 86423.0, 91034.0, 95855.0, 100890.0, 106160.0,
    111660.0, 117400.0, 123400.0, 129650.0, 136170.0, 142980.0, 150070.0, 157460.0,
    165160.0, 173180.0, 181530.0, 190220.0, 199260.0, 208670.0, 218450.0, 228610.0,
    239180.0, 250160.0, 261560.0, 273400.0, 285700.0, 298450.0, 311690.0, 325420.0,
    339650.0, 354410.0, 369710.0, 385560.0, 401980.0, 418980.0, 436590.0, 454810.0,
    473670.0, 493170.0, 513350.0, 534220.0, 555800.0, 578090.0, 601130.0, 624940.0,
    649530.0, 674920.0, 701130.0, 728190.0, 756110.0, 784920.0, 814630.0, 845280.0,
    876880.0, 909450.0, 943020.0, 977610.0, 1013250.0, 1049940.0, 1087740.0,
};
constexpr Real table_data[223] = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.3160e-02, 0.3820e-02, 0.4600e-02, 0.5560e-02,
    0.6670e-02, 0.8000e-02, 0.9580e-02, 0.1143e-01, 0.1364e-01, 0.1623e-01, 0.1928e-01, 0.2280e-01,
    0.2700e-01, 0.3190e-01, 0.3760e-01, 0.4430e-01, 0.5200e-01, 0.6090e-01, 0.7130e-01, 0.8340e-01,
    0.9720e-01, 0.1133e+00, 0.1317e-00, 0.1526e-00, 0.1780e-00, 0.2050e-00, 0.2370e-00, 0.2740e-00,
    0.3160e-00, 0.3630e-00, 0.4170e-00, 0.4790e-00, 0.5490e-00, 0.6280e-00, 0.7180e-00, 0.8190e-00,
    0.9340e-00, 0.1064e+01, 0.1209e+01, 0.1368e+01, 0.1560e+01, 0.1770e+01, 0.1990e+01, 0.2260e+01,
    0.2540e+01, 0.2880e+01, 0.3230e+01, 0.3640e+01, 0.4090e+01, 0.4590e+01, 0.5140e+01, 0.5770e+01,
    0.6450e+01, 0.7220e+01, 0.8050e+01, 0.8990e+01, 0.1001e+02, 0.1112e+02, 0.1240e+02, 0.1380e+02,
    0.1530e+02, 0.1700e+02, 0.1880e+02, 0.2080e+02, 0.2310e+02, 0.2550e+02, 0.2810e+02, 0.3100e+02,
    0.3420e+02, 0.3770e+02, 0.4150e+02, 0.4560e+02, 0.5010e+02, 0.5500e+02, 0.6030e+02, 0.6620e+02,
    0.7240e+02, 0.7930e+02, 0.8680e+02, 0.9500e+02, 0.1146e+03, 0.1254e+03, 0.1361e+03, 0.1486e+03,
    0.1602e+03, 0.1734e+03, 0.1873e+03, 0.2020e+03, 0.2171e+03, 0.2331e+03, 0.2502e+03, 0.2678e+03,
    0.2863e+03, 0.3057e+03, 0.3250e+03, 0.3457e+03, 0.3664e+03, 0.3882e+03, 0.4101e+03, 0.4326e+03,
    0.4584e+03, 0.4885e+03, 0.5206e+03, 0.5541e+03, 0.5898e+03, 0.6273e+03, 0.6665e+03, 0.7090e+03,
    0.7520e+03, 0.7980e+03, 0.8470e+03, 0.8980e+03, 0.9520e+03, 0.1008e+04, 0.1067e+04, 0.1129e+04,
    0.1194e+04, 0.1263e+04, 0.1334e+04, 0.1409e+04, 0.1488e+04, 0.1569e+04, 0.1656e+04, 0.1745e+04,
    0.1840e+04, 0.1937e+04, 0.2041e+04, 0.2147e+04, 0.2259e+04, 0.2375e+04, 0.2497e+04, 0.2624e+04,
    0.2756e+04, 0.2893e+04, 0.3036e+04, 0.3186e+04, 0.3340e+04, 0.3502e+04, 0.3670e+04, 0.3843e+04,
    0.4025e+04, 0.4213e+04, 0.4408e+04, 0.4611e+04, 0.4821e+04, 0.5035e+04, 0.5270e+04, 0.5500e+04,
    0.5740e+04, 0.6000e+04, 0.6250e+04, 0.6520e+04, 0.6810e+04, 0.7090e+04, 0.7390e+04, 0.7700e+04,
    0.8020e+04, 0.8350e+04, 0.8690e+04, 0.9040e+04, 0.9410e+04, 0.9780e+04, 0.1016e+05, 0.1057e+05,
    0.1098e+05, 0.1140e+05, 0.1184e+05, 0.1230e+05, 0.1275e+05, 0.1324e+05, 0.1373e+05, 0.1423e+05,
    0.1476e+05, 0.1530e+05, 0.1585e+05, 0.1642e+05, 0.1700e+05, 0.1761e+05, 0.1822e+05, 0.1886e+05,
    0.1950e+05, 0.2018e+05, 0.2087e+05, 0.2158e+05, 0.2229e+05, 0.2304e+05, 0.2381e+05, 0.2459e+05,
    0.2539e+05, 0.2621e+05, 0.2706e+05, 0.2792e+05, 0.2881e+05, 0.2971e+05, 0.3065e+05, 0.3160e+05,
    0.3257e+05, 0.3357e+05, 0.3459e+05, 0.3564e+05, 0.3669e+05, 0.3780e+05, 0.0000e+00,
};

// module_gfs_physcons (kind_phys = double)
constexpr Real con_rv = 4.6150e+2, con_cvap = 1.8460e+3, con_cliq = 4.1855e+3, con_csol = 2.1060e+3;
constexpr Real con_hvap = 2.5000e+6, con_hfus = 3.3358e+5, con_psat = 6.1078e+2, con_ttp = 2.7316e+2;
constexpr Real con_g = 9.80665e+0;

// module_gfs_funcphys fpvsx (host only: builds tbpvs)
Real fpvsx(Real t) {
  constexpr Real tliq = con_ttp;
  constexpr Real tice = con_ttp - 20.0;
  constexpr Real dldtl = con_cvap - con_cliq;
  constexpr Real heatl = con_hvap;
  constexpr Real xponal = -dldtl / con_rv;
  constexpr Real xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp);
  constexpr Real dldti = con_cvap - con_csol;
  constexpr Real heati = con_hvap + con_hfus;
  constexpr Real xponai = -dldti / con_rv;
  constexpr Real xponbi = -dldti / con_rv + heati / (con_rv * con_ttp);
  const Real tr = con_ttp / t;
  if (t >= tliq) return con_psat * (std::pow(tr, xponal)) * std::exp(xponbl * (1. - tr));
  if (t < tice) return con_psat * (std::pow(tr, xponai)) * std::exp(xponbi * (1. - tr));
  const Real w = (t - tice) / (tliq - tice);
  const Real pvl = con_psat * (std::pow(tr, xponal)) * std::exp(xponbl * (1. - tr));
  const Real pvi = con_psat * (std::pow(tr, xponai)) * std::exp(xponbi * (1. - tr));
  return w * pvl + (1. - w) * pvi;
}

// module_gfs_funcphys fpvs: linear interpolation in tbpvs
KOKKOS_INLINE_FUNCTION Real fpvs(const Kokkos::View<Real*, Device>& tb, Real c1xpvs, Real c2xpvs, Real t) {
  constexpr Real nx = Tables::nxpvs;
  const Real xj = fmin2(fmax2(c1xpvs + c2xpvs * t, 1.0), nx);
  const int jx = int(fmin2(xj, nx - 1.0));
  return tb(jx - 1) + (xj - jx) * (tb(jx) - tb(jx - 1));
}

// module_sf_exchcoef znot_m_v6 / znot_t_v6 (Bin Liu 2017)
KOKKOS_INLINE_FUNCTION Real znot_m_v6(Real uref) {
  constexpr Real p13 = -1.296521881682694e-02, p12 = 2.855780863283819e-01, p11 = -1.597898515251717e+00,
                 p10 = -8.396975715683501e+00;
  constexpr Real p25 = 3.790846746036765e-10, p24 = 3.281964357650687e-09, p23 = 1.962282433562894e-07,
                 p22 = -1.240239171056262e-06, p21 = 1.739759082358234e-07, p20 = 2.147264020369413e-05;
  constexpr Real p35 = 1.840430200185075e-07, p34 = -2.793849676757154e-05, p33 = 1.735308193700643e-03,
                 p32 = -6.139315534216305e-02, p31 = 1.255457892775006e+00, p30 = -1.663993561652530e+01;
  constexpr Real p40 = 4.579369142033410e-04;
  const Real u2 = uref * uref, u3 = u2 * uref, u4 = u2 * u2, u5 = u3 * u2;
  Real znotm = 0.0;   // the Fortran leaves it undefined for uref < 0 (not reachable: uref >= 1)
  if (uref >= 0.0 && uref <= 6.5)
    znotm = Kokkos::exp(p10 + p11 * uref + p12 * u2 + p13 * u3);
  else if (uref > 6.5 && uref <= 15.7)
    znotm = p25 * u5 + p24 * u4 + p23 * u3 + p22 * u2 + p21 * uref + p20;
  else if (uref > 15.7 && uref <= 53.0)
    znotm = Kokkos::exp(p35 * u5 + p34 * u4 + p33 * u3 + p32 * u2 + p31 * uref + p30);
  else if (uref > 53.0)
    znotm = p40;
  return znotm;
}

KOKKOS_INLINE_FUNCTION Real znot_t_v6(Real uref) {
  constexpr Real p00 = 1.100000000000000e-04;
  constexpr Real p15 = -9.144581627678278e-10, p14 = 7.020346616456421e-08, p13 = -2.155602086883837e-06,
                 p12 = 3.333848806567684e-05, p11 = -2.628501274963990e-04, p10 = 8.634221567969181e-04;
  constexpr Real p25 = -8.654513012535990e-12, p24 = 1.232380050058077e-09, p23 = -6.837922749505057e-08,
                 p22 = 1.871407733439947e-06, p21 = -2.552246987137160e-05, p20 = 1.428968311457630e-04;
  constexpr Real p35 = 3.207515102100162e-12, p34 = -2.945761895342535e-10, p33 = 8.788972147364181e-09,
                 p32 = -3.814457439412957e-08, p31 = -2.448983648874671e-06, p30 = 3.436721779020359e-05;
  constexpr Real p45 = -3.530687797132211e-11, p44 = 3.939867958963747e-09, p43 = -1.227668406985956e-08,
                 p42 = -1.367469811838390e-05, p41 = 5.988240863928883e-04, p40 = -7.746288511324971e-03;
  constexpr Real p56 = -1.187982453329086e-13, p55 = 4.801984186231693e-11, p54 = -8.049200462388188e-09,
                 p53 = 7.169872601310186e-07, p52 = -3.581694433758150e-05, p51 = 9.503919224192534e-04,
                 p50 = -1.036679430885215e-02;
  constexpr Real p60 = 4.751256171799112e-05;
  const Real u2 = uref * uref, u3 = u2 * uref, u4 = u2 * u2, u5 = u3 * u2, u6 = u3 * u3;
  Real znott = 0.0;   // undefined for uref < 0 in the Fortran (not reachable)
  if (uref >= 0.0 && uref < 5.9)
    znott = p00;
  else if (uref >= 5.9 && uref <= 15.4)
    znott = p15 * u5 + p14 * u4 + p13 * u3 + p12 * u2 + p11 * uref + p10;
  else if (uref > 15.4 && uref <= 21.6)
    znott = p25 * u5 + p24 * u4 + p23 * u3 + p22 * u2 + p21 * uref + p20;
  else if (uref > 21.6 && uref <= 42.2)
    znott = p35 * u5 + p34 * u4 + p33 * u3 + p32 * u2 + p31 * uref + p30;
  else if (uref > 42.2 && uref <= 53.3)
    znott = p45 * u5 + p44 * u4 + p43 * u3 + p42 * u2 + p41 * uref + p40;
  else if (uref > 53.3 && uref <= 80.0)
    znott = p56 * u6 + p55 * u5 + p54 * u4 + p53 * u3 + p52 * u2 + p51 * uref + p50;
  else if (uref > 80.0)
    znott = p60;
  return znott;
}

// znot_wind10m with icoef_sf = 6
KOKKOS_INLINE_FUNCTION void znot_wind10m(Real w10m, Real& znott, Real& znotm) {
  Real windmks = w10m;
  if (windmks > 85.0) windmks = 85.0;
  if (windmks < 1.0) windmks = 1.0;
  znotm = znot_m_v6(windmks);
  znott = znot_t_v6(windmks);
}

// MFLUX2's Hicks stable universal function: (8 ln z + 4.25/z - 0.5/z^2 + cmo1)
// between lo and hi, the log-linear form below, the linear one above.
struct Stable {
  Real a, c, rca, cmo1, cmo2;
  // fmz-type value for zeta (returns false where the Fortran leaves it unset)
  KOKKOS_INLINE_FUNCTION bool f(Real z, Real hi, Real& out) const {
    const Real zal = Kokkos::log(z);
    if (z <= 0.5) {
      out = (zal + a * z) * rca;
    } else if (z > 0.5 && z <= hi) {
      const Real r = 1. / z;
      out = (8. * zal + 4.25 * r - 0.5 * r * r + cmo1) * rca;
    } else if (z > hi) {
      out = (c * z + cmo2) * rca;
    } else {
      return false;
    }
    return true;
  }
  // same with the sfzo companion value
  KOKKOS_INLINE_FUNCTION void fs(Real z, Real& out, Real& sfzo) const {
    const Real zal = Kokkos::log(z);
    if (z <= 0.5) {
      out = (zal + a * z) * rca;
      sfzo = 1. + a * z;
    } else if (z > 0.5 && z <= 10.) {
      const Real r = 1. / z;
      out = (8. * zal + 4.25 * r - 0.5 * r * r + cmo1) * rca;
      sfzo = 8.0 - 4.25 * r + r * r;
    } else if (z > 10.) {
      out = (c * z + cmo2) * rca;
      sfzo = c * z;
    }
  }
};

// MFLUX2's unstable (Paulson) momentum integral term: ln(zz/zo) - ux3 + 2 xtan
KOKKOS_INLINE_FUNCTION Real unstable_m(Real ugz, Real ux11, Real ux12) {
  const Real y = Kokkos::sqrt(ux11);
  const Real yo = Kokkos::sqrt(ux12);
  const Real x = Kokkos::sqrt(y);
  const Real xo = Kokkos::sqrt(yo);
  const Real xnum = (x * x + 1.) * ((x + 1.) * (x + 1.));
  const Real xden = (xo * xo + 1.) * ((xo + 1.) * (xo + 1.));
  const Real xtan = Kokkos::atan(x) - Kokkos::atan(xo);
  const Real ux3 = Kokkos::log(xnum / xden);
  return ugz - ux3 + 2. * xtan;
}

// the tab/table lookup of MFLUX2 (saturation vapor pressure in cgs)
KOKKOS_INLINE_FUNCTION Real estab(const Kokkos::View<Real*, Device>& tab, const Kokkos::View<Real*, Device>& table,
                                  Real t) {
  const Real tab1 = t - 153.16;
  const int it = int(tab1);
  const Real tab2 = tab1 - Real(it);
  int idx = it + 1;
  idx = idx < 1 ? 1 : idx;
  idx = idx > 223 ? 223 : idx;
  const Real t1 = tab(idx - 1);
  const Real t2 = table(idx - 1);
  return t1 + tab2 * t2;
}

// SF_GFDL's soil tables (index isltyp)
constexpr Real maxsmc_data[30] = {0.339, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406,
                                  0.468, 0.468, 0.439, 1.000, 0.200, 0.421, 0.000, 0.000, 0.000, 0.000,
                                  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};
constexpr Real drysmc_data[30] = {0.010, 0.028, 0.047, 0.084, 0.084, 0.066, 0.067, 0.120, 0.103, 0.100,
                                  0.126, 0.138, 0.066, 0.000, 0.006, 0.028, 0.000, 0.000, 0.000, 0.000,
                                  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};

Kokkos::View<Real*, Device> to_device(const char* name, const Real* h, int n) {
  Kokkos::View<Real*, Device> d(std::string(name), n);
  auto m = Kokkos::create_mirror_view(d);
  for (int k = 0; k < n; ++k) m(k) = h[k];
  Kokkos::deep_copy(d, m);
  return d;
}

} // namespace

void Options::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k sfc_gfdl: " + m); };
  if (icoef_sf != 6) fail("icoef_sf=" + std::to_string(icoef_sf) + " is not ported (only 6, as CM1 sets it)");
  if (ntsflg != 0 && ntsflg != 1) fail("ntsflg must be 0 or 1");
  if (isfflx != 0 && isfflx != 1) fail("isfflx must be 0 or 1");
  if (isltyp < 1 || isltyp > 30) fail("isltyp must be 1..30");
}

Tables Tables::make() {
  Tables t;
  t.tab = to_device("cm1k::sfc_gfdl::tab", tab_data, ntab);
  t.table = to_device("cm1k::sfc_gfdl::table", table_data, ntab);
  // gpvs
  const Real xmin = 180.0, xmax = 330.0;
  const Real xinc = (xmax - xmin) / (nxpvs - 1);
  t.c2xpvs = 1. / xinc;
  t.c1xpvs = 1. - xmin * t.c2xpvs;
  std::vector<Real> tb(nxpvs);
  for (int jx = 1; jx <= nxpvs; ++jx) {
    const Real x = xmin + (jx - 1) * xinc;
    tb[jx - 1] = fpvsx(x);
  }
  t.tbpvs = to_device("cm1k::sfc_gfdl::tbpvs", tb.data(), nxpvs);
  return t;
}

void sf_gfdl(const Options& o, const WrfConst& c, const Tables& t, Real dt, int ni, int nj, const Inputs& in,
             const Fields& f) {
  o.check();
  const Inputs I = in;
  const Fields F = f;
  const auto tab = t.tab, table = t.table, tbpvs = t.tbpvs;
  const Real c1xpvs = t.c1xpvs, c2xpvs = t.c2xpvs;
  const Real CP = c.cp, R = c.rd, XLV = c.xlv, EP1 = c.ep1, EP2 = c.ep2, karman = c.karman;
  const int ntsflg = o.ntsflg, isfflx = o.isfflx;
  const Real sfenth = o.sfenth;
  const Real smcdry = drysmc_data[o.isltyp - 1], smcmax = maxsmc_data[o.isltyp - 1];
  // MFLUX2's slab constants: pith = sqrt(4*atan(1.)) (folded by gfortran)
  const Real pith = std::sqrt(4. * std::atan(1.0));

  Kokkos::parallel_for("cm1k::sfc_gfdl::sf_gfdl", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    //------------------------------------------------------------------
    // SF_GFDL: inputs in cgs
    const Real prsl1 = I.p3d(i, j, 1) * .001;
    Real wetc = 1.0;
    if (I.xland(i, j, 1) < 1.99) {
      const Real smc = I.smois(i, j, 1);
      wetc = (smc - smcdry) / (smcmax - smcdry);
      wetc = fmin2(1., fmax2(wetc, 0.));
    }
    const Real pspc = I.psfc(i, j, 1) * 10.;
    const Real pkmax = I.p3d(i, j, 1) * 10.;
    const Real ps = I.psfc(i, j, 1) * .001;
    const Real q1 = I.qv3d(i, j, 1);
    const Real rpc = I.qv3d(i, j, 1);
    const Real tskin = F.tsk(i, j, 1);
    const Real tstrc0 = F.tsk(i, j, 1);
    const Real t1 = I.t3d(i, j, 1);
    const Real tpc = I.t3d(i, j, 1);
    const Real u1 = I.u3d(i, j, 1);
    const Real upc = I.u3d(i, j, 1) * 100.;
    const Real v1 = I.v3d(i, j, 1);
    const Real vpc = I.v3d(i, j, 1) * 100.;
    Real zoc = F.znt(i, j, 1) * 100.;
    if (I.xland(i, j, 1) > 1.99) zoc = -zoc;
    Real slwdc = I.gsw(i, j, 1) + I.glw(i, j, 1);
    slwdc = 0.239 * 60. * slwdc * 1.e-4;
    // previous u10, v10 -> wind10 (first time step: log profile)
    Real wind10 = Kokkos::sqrt(F.u10(i, j, 1) * F.u10(i, j, 1) + F.v10(i, j, 1) * F.v10(i, j, 1));
    if (wind10 <= 1.0e-10 || wind10 > 150.0) {
      const Real zhalf = -R * tpc * Kokkos::log(pkmax / pspc) / con_g;
      wind10 = Kokkos::sqrt(u1 * u1 + v1 * v1) * Kokkos::log(10.0 / F.znt(i, j, 1)) /
               Kokkos::log(zhalf / F.znt(i, j, 1));
    }
    wind10 = wind10 * 100.0;
    const Real rho1 = prsl1 * 1000. / (R * t1 * (1. + EP1 * q1));
    const Real zkmaxin = I.zkmax(i, j, 1);

    //------------------------------------------------------------------
    // MFLUX2
    constexpr Real cpc = 1.00464e7, g = 980.6, rgas = 2.87e6, og = 1. / g;
    constexpr Real amask = -98.0;
    constexpr Real cor1 = 0., cor2 = 0.;
    const Real z10 = 1000., z2 = 200.;
    const Real pss = pspc;
    const Real tstar = tstrc0;
    const Real ubot = upc, vbot = vpc;
    const Real uvs1 = fmax2(Kokkos::sqrt(ubot * ubot + vbot * vbot), 100.0);
    const Real ukmax = ubot, vkmax = vbot;
    const Real tkmax = tpc, rkmax = rpc;
    Real windp = Kokkos::sqrt(ukmax * ukmax + vkmax * vkmax);
    const Real wind = fmax2(windp, 100.);
    Real wind10p = wind10;
    wind10p = fmax2(wind10p, 100.);
    if (zoc < amask) zoc = -0.0185 * 0.001 * wind10p * wind10p * og;
    Real ecof, land, zot;
    if (zoc > 0.0) {
      ecof = wetc;
      land = 1.0;
      zot = zoc;
    } else {
      ecof = wetc;
      land = 0.0;
      const Real windmks = wind10p * .01;
      Real znott, znotm;
      znot_wind10m(windmks, znott, znotm);
      zoc = -100. * znotm;
      zot = -100 * znott;
    }
    const Real mzoc = zoc;
    Real tzot = zot;

    constexpr Real c_ = .76, a = 5., ca = .4;
    constexpr Real cmo1 = .5 * a - 1.648;
    constexpr Real cmo2 = 17.193 + .5 * a - 10. * c_;
    constexpr Real boycon = .61;
    constexpr Real rovcp = rgas / cpc;
    const Real theta = tkmax / (Kokkos::pow(pkmax / pspc, rovcp));
    const Real vrtkx = 1.0 + boycon * rkmax;
    const Real zkmax = 100.0 * zkmaxin;

    // saturation mixing ratio at the surface
    const Real tsg = tstar;
    const Real estso = estab(tab, table, tstar);
    Real psps1 = (pss - estso);
    if (psps1 == 0.0) psps1 = .1;
    const Real rstso = 0.622 * estso / psps1;
    const Real vrts = 1. + boycon * ecof * rstso;

    Real tempa1 = theta * vrtkx - tstar * vrts;
    const Real tempa2 = tempa1 * (theta - tstar);
    if (tempa2 < 0.) tempa1 = 1.0e-4;
    if (Kokkos::fabs(tempa1) < 1.0e-4) tempa1 = 1.0e-4;
    Real rib = g * zkmax * tempa1 / (tkmax * vrtkx * wind * wind);
    {
      const Real tab2 = Kokkos::fabs(zoc);
      const Real tab1 = 0.95 / (c_ * (1. - tab2 / zkmax));
      if (rib > tab1) rib = tab1;
    }
    const Real zeta = ca * rib / 0.03;

    constexpr Real rca = 1. / ca;
    constexpr Real enrca = 0.0;   // interfacial layer turned off
    constexpr Real zog = .0185 * og;
    constexpr int icntx = 30;
    const Stable hicks{a, c_, rca, cmo1, cmo2};

    Real xxfm = 0, xxfh = 0, xxfh2 = 0, xxsh = 0, zetaout = 0;
    if (zeta >= 0.0) {
      // stable point: Wegstein iteration for zeta (Hicks 1976)
      Real szetam = 1.0e+30, sgzm = 0.0e+00, szeta = zeta;
      int ifz = 1;
      Real fmz1 = 0, fmzo1 = 0, sfzo = 0, fhzo1 = 0, fmz10 = 0, fmz2 = 0, sf10 = 0, sf2 = 0, sfm = 0, sfh = 0;
      for (int icnt = 1; icnt <= icntx; ++icnt) {
        if (ifz == 0) continue;
        hicks.f(szeta, 10., fmz1);
        const Real szetao = Kokkos::fabs(zoc) / zkmax * szeta;
        hicks.fs(szetao, fmzo1, sfzo);
        // heat and moisture parts of zot
        const Real szetho = Kokkos::fabs(zot) / zkmax * szeta;
        hicks.fs(szetho, fhzo1, sfzo);
        // 10 m and 2 m (diagnostics)
        const Real szet10 = Kokkos::fabs(z10) / zkmax * szeta;
        hicks.f(szet10, 10., fmz10);
        sf10 = fmz10 - fmzo1;
        const Real szet2 = Kokkos::fabs(z2) / zkmax * szeta;
        hicks.f(szet2, 2., fmz2);
        sf2 = fmz2 - fmzo1;
        sfm = fmz1 - fmzo1;
        sfh = fmz1 - fhzo1;
        const Real sgz = ca * rib * sfm * sfm / (sfh + enrca * sfzo);
        const Real fmz = (sgz - szeta) / szeta;
        const Real fmzo = Kokkos::fabs(fmz);
        if (fmzo >= 5.0e-5) {
          const Real sq = (sgz - sgzm) / (szeta - szetam);
          szetam = szeta;
          szeta = (sgz - szeta * sq) / (1.0 - sq);
          sgzm = sgz;
        } else {
          ifz = 0;
        }
      }
      // update zo over the ocean
      Real szo = zoc;
      if (szo < 0.0) {
        const Real wndm = wind * 0.01;
        Real ckg;
        if (wndm < 15.0) ckg = 0.0185 * og;
        else ckg = (sfenth * (4 * 0.000308 * wndm) + (1. - sfenth) * 0.0185) * og;
        szo = -ckg * wind * wind / (sfm * sfm);
        zot = szo;   // szo*exp(-rat) with rat = 0
      } else {
        zot = zoc;
      }
      zoc = szo;
      xxfm = sfm;
      xxfh = sfh;
      xxfh2 = sf2;
      xxsh = sfzo;
      zetaout = szeta;
      // 10 m wind
      wind10 = sf10 * uvs1 / sfm;
      wind10 = wind10 * 1.944;
      if (wind10 > 6000.0) wind10 = wind10 + wind10 * cor1 - cor2;
      wind10 = wind10 / 1.944;
    } else {
      // unstable point: Wegstein iteration with the Paulson functions
      Real uzeta = zeta, uzetam = 1.0e+30, ugzm = 0.0e+00;
      int ifz = 1;
      Real ufzo = 0, ufh = 0, ufm = 0, uf10 = 0, uf2 = 0;
      for (int icnt = 1; icnt <= icntx; ++icnt) {
        if (ifz == 0) continue;
        Real ugzzo = Kokkos::log(zkmax / Kokkos::fabs(zot));
        Real uzetao = Kokkos::fabs(zot) / zkmax * uzeta;
        Real ux11 = 1. - 16. * uzeta;
        Real ux12 = 1. - 16. * uzetao;
        Real y = Kokkos::sqrt(ux11);
        Real yo = Kokkos::sqrt(ux12);
        ufzo = 1. / yo;
        Real ux13 = (1. + y) / (1. + yo);
        Real ux21 = Kokkos::log(ux13);
        ufh = (ugzzo - 2. * ux21) * rca;
        // momentum, with the momentum znot zoc
        ugzzo = Kokkos::log(zkmax / Kokkos::fabs(zoc));
        uzetao = Kokkos::fabs(zoc) / zkmax * uzeta;
        ux11 = 1. - 16. * uzeta;
        ux12 = 1. - 16. * uzetao;
        ufm = unstable_m(ugzzo, ux11, ux12) * rca;
        // 10 m
        {
          const Real ugz10 = Kokkos::log(z10 / Kokkos::fabs(zoc));
          const Real uzet1o = Kokkos::fabs(z10) / zkmax * uzeta;
          const Real uzo10 = Kokkos::fabs(zoc) / zkmax * uzeta;
          uf10 = unstable_m(ugz10, 1. - 16. * uzet1o, 1. - 16. * uzo10) * rca;
        }
        // 2 m (with the momentum ugzzo, as in the Fortran)
        {
          const Real uzet1o = Kokkos::fabs(z2) / zkmax * uzeta;
          const Real uzo2 = Kokkos::fabs(zoc) / zkmax * uzeta;
          y = Kokkos::sqrt(1. - 16. * uzet1o);
          yo = Kokkos::sqrt(1. - 16. * uzo2);
          ux13 = (1. + y) / (1. + yo);
          ux21 = Kokkos::log(ux13);
          uf2 = (ugzzo - 2. * ux21) * rca;
        }
        const Real ugz = ca * rib * ufm * ufm / (ufh + enrca * ufzo);
        const Real ux1 = (ugz - uzeta) / uzeta;
        const Real ux2 = Kokkos::fabs(ux1);
        if (ux2 >= 5.0e-5) {
          const Real uq = (ugz - ugzm) / (uzeta - uzetam);
          uzetam = uzeta;
          uzeta = (ugz - uzeta * uq) / (1.0 - uq);
          ugzm = ugz;
        } else {
          ifz = 0;
        }
      }
      // update zo over the ocean
      Real uzo = zoc;
      if (zoc < 0.0) {
        const Real wndm = wind * 0.01;
        Real ckg;
        if (wndm < 15.0) ckg = 0.0185 * og;
        else ckg = (sfenth * (4 * 0.000308 * wndm) + (1. - sfenth) * 0.0185) * og;
        uzo = -ckg * wind * wind / (ufm * ufm);
        zot = uzo;   // uzo*exp(-rat) with rat = 0
      } else {
        zot = zoc;
      }
      zoc = uzo;
      // 10 m wind
      wind10 = uf10 * uvs1 / ufm;
      wind10 = wind10 * 1.944;
      if (wind10 > 6000.0) wind10 = wind10 + wind10 * cor1 - cor2;
      wind10 = wind10 / 1.944;
      xxfm = ufm;
      xxfh = ufh;
      xxfh2 = uf2;
      xxsh = ufzo;
      zetaout = uzeta;
    }
    // (290)
    Real ucom = ukmax, vcom = vkmax;
    if (windp == 0.0) {
      windp = 100.0;
      ucom = 100.0 / Kokkos::sqrt(2.0);
      vcom = 100.0 / Kokkos::sqrt(2.0);
    }
    const Real rho = pss / (rgas * (tsg + enrca * (theta - tsg) * xxsh / (xxfh + enrca * xxsh)));
    const Real bq1 = wind * rho / (xxfm * (xxfh + enrca * xxsh));

    // land T* prediction (ntsflg = 1)
    Real tstrc = tstrc0;
    if (ntsflg != 0 && land == 1) {
      constexpr Real alll = 600., xks = 0.01, hcap = .5 / 2.39e-8, alfus = alll / 2.39e-8, teps = 0.1;
      const Real slwa = slwdc / (2.39e-8 * 60.);
      const Real tss = tstar, thetap = theta, rkmaxp = rkmax, aap = 5.673e-5, pssp = pss, ecofp = ecof;
      Real rstsop = rstso;
      const Real bq1p = fmax2(bq1, 0.1e-3);
      const Real delsrad = dt * pith / (hcap * Kokkos::sqrt(3600. * 24. * xks));
      int ifz = 1;
      Real tsm = tss;
      Real rdiff = fmin2(0.0, (rkmaxp - rstsop));
      const Real tsm2 = tsm * tsm;
      Real foftm = tss + delsrad * (slwa - aap * (tsm2 * tsm2) - cpc * bq1p * (tsm - thetap) +
                                    ecofp * alfus * bq1p * rdiff);
      Real tsp = foftm;
      for (int icnt = 1; icnt <= icntx; ++icnt) {
        if (ifz == 0) continue;
        const Real estsop = estab(tab, table, tsp);
        Real psps2 = (pssp - estsop);
        if (psps2 == 0.0) psps2 = .1;
        rstsop = 0.622 * estsop / psps2;
        rdiff = fmin2(0.0, (rkmaxp - rstsop));
        const Real tsp2 = tsp * tsp;
        const Real foft = tss + delsrad * (slwa - aap * (tsp2 * tsp2) - cpc * bq1p * (tsp - thetap) +
                                           ecofp * alfus * bq1p * rdiff);
        const Real frac = Kokkos::fabs((foft - tsp) / tsp);
        if (frac >= teps) {
          const Real qf = (foft - foftm) / (tsp - tsm);
          tsm = tsp;
          tsp = (foft - tsp * qf) / (1. - qf);
          foftm = foft;
        } else {
          ifz = 0;
        }
      }
      tstrc = tsp;
    }
    // (370) fluxes and momentum drag coefficient
    const Real fxh = bq1 * (theta - tsg);
    Real fxe = ecof * bq1 * (rkmax - rstso);
    if (fxe > 0.0) fxe = 0.0;
    const Real fxmx = rho / (xxfm * xxfm) * wind * wind * ucom / windp;
    const Real fxmy = rho / (xxfm * xxfm) * wind * wind * vcom / windp;
    const Real cdm = 1. / (xxfm * xxfm);

    //------------------------------------------------------------------
    // SF_GFDL after MFLUX2
    if (ntsflg == 1) F.tstrc(i, j, 1) = tstrc;   // tsk(i,j) = tstrc(i), patched below
    F.znt(i, j, 1) = 0.01 * Kokkos::fabs(zoc);
    const Real mznt = 0.01 * Kokkos::fabs(mzoc);
    F.mznt(i, j, 1) = mznt;
    Real wspd = Kokkos::sqrt(upc * upc + vpc * vpc);
    wspd = fmax2(wspd, 100.) / 100.;
    F.wspd(i, j, 1) = wspd;
    const Real u10m = u1 * (wind10 / wspd) / 100.;
    const Real v10m = v1 * (wind10 / wspd) / 100.;
    const Real zk = zkmaxin;
    const Real gz1oz0 = Kokkos::log(zk / mznt);
    F.gz1oz0(i, j, 1) = gz1oz0;
    const Real ustar = 0.01 * Kokkos::sqrt(cdm * (upc * upc + vpc * vpc));
    F.taux(i, j, 1) = fxmx / 10.;
    F.tauy(i, j, 1) = fxmy / 10.;
    const Real fm = karman / Kokkos::sqrt(cdm);
    const Real fh = karman * xxfh;
    F.psim(i, j, 1) = gz1oz0 - fm;
    F.psih(i, j, 1) = Kokkos::log(zk / (0.01 * Kokkos::fabs(tzot))) - fh;
    F.fm(i, j, 1) = fm;
    F.fh(i, j, 1) = fh;
    const Real fh2 = karman * xxfh2;
    const Real ch = karman * karman / (fm * fh);
    const Real cm = cdm;
    Real cd_out = cm, ch_out = ch;
    // cd, ch at 10 m (output)
    if (wind10 >= 0.1) {
      const Real r = wspd / (0.01 * wind10);
      cd_out = cm * (r * r);
      const Real tmp9 = 0.01 * Kokkos::fabs(tzot);
      ch_out = ch * (wspd / (0.01 * wind10)) * (Kokkos::log(zk / tmp9) / Kokkos::log(10.0 / tmp9));
    }
    F.cd_out(i, j, 1) = cd_out;
    F.ch_out(i, j, 1) = ch_out;
    F.u10(i, j, 1) = u10m;
    F.v10(i, j, 1) = v10m;
    F.br(i, j, 1) = rib;
    Real chs = ch * wspd;
    Real chs2 = ustar * karman / fh2;
    // cap chs over land points
    if (I.xland(i, j, 1) < 1.9) {
      chs = fmin2(chs, 0.05);
      chs2 = fmin2(chs2, 0.05);
      if (chs2 < 0) chs2 = 1.0e-6;
    }
    F.chs(i, j, 1) = chs;
    F.chs2(i, j, 1) = chs2;
    const Real cpm = CP * (1. + 0.8 * I.qv3d(i, j, 1));
    F.cpm(i, j, 1) = cpm;
    Real esat = fpvs(tbpvs, c1xpvs, c2xpvs, t1);
    F.qgh(i, j, 1) = EP2 * esat / (1000. * ps - esat);
    esat = fpvs(tbpvs, c1xpvs, c2xpvs, tskin);
    F.qsfc(i, j, 1) = EP2 * esat / (1000. * ps - esat);
    F.ust(i, j, 1) = ustar;
    F.zol(i, j, 1) = zetaout;
    F.mol(i, j, 1) = zk / (1.0e-20 + zetaout);
    F.z0t(i, j, 1) = 0.01 * Kokkos::fabs(tzot);
    F.flhc(i, j, 1) = cpm * rho1 * chs;
    F.flqc(i, j, 1) = rho1 * chs;
    F.cqs2(i, j, 1) = chs2;
    if (isfflx == 0) {
      F.hfx(i, j, 1) = 0.;
      F.lh(i, j, 1) = 0.;
      F.qfx(i, j, 1) = 0.;
    } else {
      Real hfx = -10. * CP * fxh;   // (also the value before the ISFFLX block)
      if (I.xland(i, j, 1) - 1.5 < 0.) hfx = fmax2(hfx, -250.);
      F.hfx(i, j, 1) = hfx;
      Real qfx = -10. * fxe;
      qfx = fmax2(qfx, 0.);
      F.qfx(i, j, 1) = qfx;
      F.lh(i, j, 1) = XLV * qfx;
    }
  });

  if (ntsflg == 1) {
    // SF_GFDL: tsk(i,j) = tstrc(i), then (ids = jds = 1, ide = ni+1 and
    // jde = nj+1 never reached) tsk(i,1) = tsk(i,2) and tsk(1,j) = tsk(2,j)
    // with the values of tsk before this call (row j=2 and column i=2 are
    // updated after rows/points 1 in the Fortran's loop order).
    const auto tsk = f.tsk, tstrc = f.tstrc;
    Kokkos::parallel_for("cm1k::sfc_gfdl::tsk_patch", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      if (i == 1) tstrc(i, j, 1) = tsk(2, j, 1);
      else if (j == 1) tstrc(i, j, 1) = tsk(i, 2, 1);
    });
    Kokkos::parallel_for("cm1k::sfc_gfdl::tsk_set", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      tsk(i, j, 1) = tstrc(i, j, 1);
    });
  }
}

} // namespace cm1k::sfc_gfdl
