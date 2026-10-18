// Port of src/module_sf_myjsfc.F and src/module_bl_myjpbl.F (MYJ surface
// layer and PBL). See the header for scope.
//
// Expressions keep the Fortran's operation order and parenthesization
// (Fortran evaluates a*b*c as (a*b)*c, a-b+c as (a-b)+c, unary minus binds
// weaker than * and /); x**2 is x*x, x**3 is (x*x)*x (gfortran expands
// integer powers), real exponents are pow. MAX/MIN/AMAX1/AMIN1 of two
// non-NaN values are fmax2/fmin2. The module PARAMETERs are constant
// expressions that gfortran folds with correctly rounded double arithmetic,
// as constexpr does here.
#include "physics/myj/cm1k_myj.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cm1k::myj {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// module_model_constants (src/module_ra_etc.F)
namespace mc {
constexpr Real r_d = 287.04;
constexpr Real cp = 1005.7;
constexpr Real g = 9.81;
constexpr Real r_v = 461.5;
constexpr Real xls = 2.834E6;
constexpr Real xlv = 2.501E6;
constexpr Real rvovrd = r_v / r_d;
constexpr Real pq0 = 379.90516;
constexpr Real epsq2 = 0.2;
constexpr Real a2 = 17.2693882;
constexpr Real a3 = 273.16;
constexpr Real a4 = 35.86;
constexpr Real p608 = rvovrd - 1.;
constexpr Real ep_1 = r_v / r_d - 1.;
constexpr Real p1000mb = 100000.;
} // namespace mc

// module_sf_myjsfc parameters
namespace sf {
using namespace mc;
constexpr int itrmx = 5;
constexpr Real vkarman = 0.4;
constexpr Real capa = r_d / cp, elocp = 2.72E6 / cp, rcap = 1. / capa;
constexpr Real gocp02 = g / cp * 2., gocp10 = g / cp * 10.;
constexpr Real epsu2 = 1.E-6, epsust = 1.e-9, epszt = 1.E-28;
constexpr Real a2s = 17.2693882, a3s = 273.16, a4s = 35.86;
constexpr Real seafc = 0.98, pq0sea = pq0 * seafc;
constexpr Real beta = 1. / 273., excml = 0.0001, excms = 0.0001, glkbr = 10., glkbs = 30., pi = 3.1415926,
               qvisc = 2.1E-5, ric = 0.505, small = 0.35, sqpr = 0.84, sqsc = 0.84, sqvisc = 258.2,
               tvisc = 2.1E-5, ustc = 0.7, ustr = 0.225, visc = 1.5E-5, fh = 1.01, wwst = 1.2, ztfc = 1.;
constexpr Real btg = beta * g, cziv = small * glkbs, grrs = glkbr / glkbs, rtvisc = 1. / tvisc, rvisc = 1. / visc,
               zqrzt = sqsc / sqpr, fzq1 = rtvisc * qvisc * zqrzt, fzq2 = rtvisc * qvisc * zqrzt,
               fzt1 = rvisc * tvisc * sqpr, fzt2 = cziv * grrs * tvisc * sqpr, fzu1 = cziv * visc, pihf = 0.5 * pi,
               rqvisc = 1. / qvisc, ustfc = 0.018 / g, wwst2 = wwst * wwst;
constexpr Real ozo = 1.59E-5;
constexpr int kztm = Tables::kztm, kztm2 = kztm - 2;
} // namespace sf

// module_bl_myjpbl parameters
namespace bl {
using namespace mc;
constexpr Real pi = 3.1415926, vkarman = 0.4;
constexpr Real capa = r_d / cp;
constexpr Real rlivwv = xls / xlv, elocp = 2.72E6 / cp;
constexpr Real eps1 = 1.E-12, eps2 = 0.;
constexpr Real epsl = 0.32, epsru = 1.E-7, epsrs = 1.E-7, epstrb = 1.E-24;
constexpr Real epsa = 1.E-8, epsit = 1.E-4, epsu2 = 1.E-4, epsust = 0.07, fh = 1.01;
constexpr Real alph = 0.30, beta = 1. / 273., el0max = 1000., el0min = 1., elfc = 0.23 * 0.5,
               gam1 = 0.2222222222222222222, prt = 1.;
constexpr Real a1 = 0.659888514560862645, a2x = 0.6574209922667784586, b1 = 11.87799326209552761,
               b2 = 7.226971804046074028, c1 = 0.000830955950095854396;
constexpr Real a2s = 17.2693882, a3s = 273.16, a4s = 35.86;
constexpr Real elz0 = 0., esq = 5.0, excm = 0.001, fhneu = 0.8, glkbr = 10., glkbs = 30., qvisc = 2.1E-5,
               rfc = 0.191, ric = 0.505, small = 0.35, sqpr = 0.84, sqsc = 0.84, sqvisc = 258.2, tvisc = 2.1E-5,
               ustc = 0.7, ustr = 0.225, visc = 1.5E-5, wold = 0.15, wwst = 1.2, ztmax = 1., ztfc = 1., ztmin = -5.;
constexpr Real seafc = 0.98, pq0sea = pq0 * seafc;
constexpr Real btg = beta * g, cziv = small * glkbs, esqhf = 0.5 * 5.0, grrs = glkbr / glkbs, rb1 = 1. / b1,
               rtvisc = 1. / tvisc, rvisc = 1. / visc, zqrzt = sqsc / sqpr;
constexpr Real adnh = 9. * a1 * a2x * a2x * (12. * a1 + 3. * b2) * btg * btg,
               adnm = 18. * a1 * a1 * a2x * (b2 - 3. * a2x) * btg, anmh = -9. * a1 * a2x * a2x * btg * btg,
               anmm = -3. * a1 * a2x * (3. * a2x + 3. * b2 * c1 + 18. * a1 * c1 - b2) * btg,
               bdnh = 3. * a2x * (7. * a1 + b2) * btg, bdnm = 6. * a1 * a1,
               beqh = a2x * b1 * btg + 3. * a2x * (7. * a1 + b2) * btg, beqm = -a1 * b1 * (1. - 3. * c1) + 6. * a1 * a1,
               bnmh = -a2x * btg, bnmm = a1 * (1. - 3. * c1), bshh = 9. * a1 * a2x * a2x * btg,
               bshm = 18. * a1 * a1 * a2x * c1,
               bsmh = -3. * a1 * a2x * (3. * a2x + 3. * b2 * c1 + 12. * a1 * c1 - b2) * btg, cesh = a2x,
               cesm = a1 * (1. - 3. * c1), cnv = ep_1 * g / btg, elfcs = vkarman * btg;
// free term in the equilibrium equation for (L/Q)**2
constexpr Real aeqh = 9. * a1 * a2x * a2x * b1 * btg * btg + 9. * a1 * a2x * a2x * (12. * a1 + 3. * b2) * btg * btg,
               aeqm = 3. * a1 * a2x * b1 * (3. * a2x + 3. * b2 * c1 + 18. * a1 * c1 - b2) * btg +
                      18. * a1 * a1 * a2x * (b2 - 3. * a2x) * btg;
// forbidden turbulence area
constexpr Real requ = -aeqh / aeqm, epsgh = 1.E-9, epsgm = requ * epsgh;
// near isotropy for shear turbulence, ww/q2 lower limit
constexpr Real ubryl = (18. * requ * a1 * a1 * a2x * b2 * c1 * btg + 9. * a1 * a2x * a2x * b2 * btg * btg) /
                       (requ * adnm + adnh),
               ubry = (1. + epsrs) * ubryl, ubry3 = 3. * ubry;
constexpr Real aubh = 27. * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3,
               aubm = 54. * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3,
               bubh = (9. * a1 * a2x + 3. * a2x * b2) * btg - bdnh * ubry3,
               bubm = 18. * a1 * a1 * c1 - bdnm * ubry3, cubr = 1. - ubry3, rcubr = 1. / cubr;
// PRODQ2: B1**(2./3.), a constant expression gfortran folds (MPFR,
// correctly rounded); glibc's pow gives the same double
constexpr Real b1p23 = 5.205894744336633;
} // namespace bl

// A column array with Fortran indexing a(k), k = 0..L-1, in the workspace.
struct Arr {
  Real* p = nullptr;
  int s = 1;
  KOKKOS_INLINE_FUNCTION Real& operator()(int k) const { return p[k * s]; }
};

// Slot n of the column's workspace row.
struct ColWs {
  Real* base;
  int s, L;
  KOKKOS_INLINE_FUNCTION Arr slot(int n) const { return Arr{base + n * L * s, s}; }
};

// The psi tables, captured by value in kernels.
struct Psi {
  Kokkos::View<const Real*, Device> m1, h1, m2, h2;
  Real dzeta1, dzeta2, fh01, fh02, ztmax1, ztmax2, ztmin1, ztmin2;
  explicit Psi(const Tables& t)
      : m1(t.psim1), h1(t.psih1), m2(t.psim2), h2(t.psih2), dzeta1(t.dzeta1), dzeta2(t.dzeta2), fh01(t.fh01),
        fh02(t.fh02), ztmax1(t.ztmax1), ztmax2(t.ztmax2), ztmin1(t.ztmin1), ztmin2(t.ztmin2) {}
};

// RZ=(ZETA-ZTMIN)/DZETA; K=INT(RZ); RDZT=RZ-REAL(K); K=MIN(K,KZTM2);
// K=MAX(K,0); PSI=(TAB(K+2)-TAB(K+1))*RDZT+TAB(K+1)  (TAB(K) is tab(K-1))
KOKKOS_INLINE_FUNCTION Real lookup(const Kokkos::View<const Real*, Device>& tab, Real zeta, Real ztmin,
                                   Real dzeta) {
  const Real rz = (zeta - ztmin) / dzeta;
  int k = int(rz);
  const Real rdzt = rz - Real(k);
  k = k < sf::kztm2 ? k : sf::kztm2;
  k = k > 0 ? k : 0;
  return (tab(k + 1) - tab(k)) * rdzt + tab(k);
}

// ---------------------------------------------------------------------------
// SFCDIF (one point). Arguments in the Fortran order; scalars by reference
// for INOUT/OUT.
struct SfcdifOut {
  Real chs, chs2, cpm, cqs2, ct, flhc, flqc, flx_lh, hfx, pshltr, q02, q10, qfx, qgh, rib, rlmo, th02, th10,
      u10, v10, u10e, v10e, psim, psih, fm, fhh, z0out;
};

KOKKOS_INLINE_FUNCTION void sfcdif(const Psi& P, int ntsd, Real seamask, Real ths, Real& qs, Real psfc, Real& uz0,
                                   Real& vz0, Real tz0, Real tsk, Real& thz0, Real& qz0, Real& ustar, Real& z0,
                                   Real z0base, Real& akms, Real& akhs, Real pblh, Real wetm, Real ulow,
                                   Real vlow, Real tlow, Real thlow, Real thelow, Real qlow, Real cwmlow, Real zsl,
                                   Real plow, Real rhosfc, int isftcflx, SfcdifOut& o) {
  using namespace sf;
  Real zu = 0, zt = 0, zq = 0, wstar2 = 0, ustark = 0, psmz = 0, pshz = 0, simm, simh, rlmo = 0;
  Real zetalu, zetalt, zetau, zetat, rz, rzsu, rzst, rlogu, rlogt, zslu, zslt, psmzl, pshzl;
  Real dthv, du2, rib;

  const Real rdz = 1. / zsl;
  const Real cxchl = excml * rdz;
  const Real cxchs = excms * rdz;

  const Real btgx = g / thlow;
  const Real elfc = vkarman * btgx;

  Real btgh;
  if (pblh > 1000.) btgh = btgx * pblh;
  else              btgh = btgx * 1000.;

  if (seamask > 0.5) {
    // sea points
    for (int itr = 1; itr <= itrmx; ++itr) {
      z0 = fmax2(ustfc * ustar * ustar, 1.59E-5);
      // GHB, 200621: level-off roughness length at high wind speeds
      if (isftcflx != 0) {
        const Real zw = fmin2(Kokkos::pow(ustar / 1.06, 0.3), 1.0);
        const Real zn1 = 0.011 * ustar * ustar / g + ozo;
        const Real zn2 = 10. * Kokkos::exp(-9.5 * Kokkos::pow(ustar, -.3333)) + 0.11 * 1.5E-5 / fmax2(ustar, 0.01);
        z0 = (1.0 - zw) * zn1 + zw * zn2;
        z0 = fmin2(z0, 2.85e-3);
        z0 = fmax2(z0, 1.59E-5);
      }
      // viscous sublayer, Janjic MWR 1994
      if (ustar < ustc) {
        if (ustar < ustr) {
          if (ntsd == 1) {
            akms = cxchs;
            akhs = cxchs;
            qs = qlow;
          }
          zu = fzu1 * Kokkos::sqrt(Kokkos::sqrt(z0 * ustar * rvisc)) / ustar;
          const Real wght = akms * zu * rvisc;
          const Real rwgh = wght / (wght + 1.);
          uz0 = (ulow * rwgh + uz0) * 0.5;
          vz0 = (vlow * rwgh + vz0) * 0.5;
          zt = fzt1 * zu;
          zq = fzq1 * zt;
          const Real wghtt = akhs * zt * rtvisc;
          const Real wghtq = akhs * zq * rqvisc;
          if (ntsd > 1) {
            thz0 = ((wghtt * thlow + ths) / (wghtt + 1.) + thz0) * 0.5;
            qz0 = ((wghtq * qlow + qs) / (wghtq + 1.) + qz0) * 0.5;
          } else {
            thz0 = (wghtt * thlow + ths) / (wghtt + 1.);
            qz0 = (wghtq * qlow + qs) / (wghtq + 1.);
          }
        }
        if (ustar >= ustr && ustar < ustc) {
          zu = z0;
          uz0 = 0.;
          vz0 = 0.;
          zt = fzt2 * Kokkos::sqrt(Kokkos::sqrt(z0 * ustar * rvisc)) / ustar;
          zq = fzq2 * zt;
          const Real wghtt = akhs * zt * rtvisc;
          const Real wghtq = akhs * zq * rqvisc;
          if (ntsd > 1) {
            thz0 = ((wghtt * thlow + ths) / (wghtt + 1.) + thz0) * 0.5;
            qz0 = ((wghtq * qlow + qs) / (wghtq + 1.) + qz0) * 0.5;
          } else {
            thz0 = (wghtt * thlow + ths) / (wghtt + 1.);
            qz0 = (wghtq * qlow + qs) / (wghtq + 1.);
          }
        }
      } else {
        zu = z0;
        uz0 = 0.;
        vz0 = 0.;
        zt = z0;
        thz0 = ths;
        zq = z0;
        // ghb, 200619:
        qs = pq0sea / psfc * Kokkos::exp(a2s * (tsk - a3s) / (tsk - a4s));
        qz0 = qs;
      }
      const Real tem = (tlow + tz0) * 0.5;
      const Real thm = (thelow + thz0) * 0.5;
      const Real a = thm * p608;
      const Real b = (elocp / tem - 1. - p608) * thm;
      dthv = ((thelow - thz0) * ((qlow + qz0 + cwmlow) * (0.5 * p608) + 1.) + (qlow - qz0 + cwmlow) * a +
              cwmlow * b);
      du2 = fmax2((ulow - uz0) * (ulow - uz0) + (vlow - vz0) * (vlow - vz0), epsu2);
      rib = btgx * dthv * zsl / du2;
      // turbulent branch
      zslu = zsl + zu;
      zslt = zsl + zt;
      rzsu = zslu / zu;
      rzst = zslt / zt;
      rlogu = Kokkos::log(rzsu);
      rlogt = Kokkos::log(rzst);
      // 1./Monin-Obukhov length
      rlmo = elfc * akhs * dthv / ((ustar * ustar) * ustar);
      zetalu = zslu * rlmo;
      zetalt = zslt * rlmo;
      zetau = zu * rlmo;
      zetat = zt * rlmo;
      zetalu = fmin2(fmax2(zetalu, P.ztmin1), P.ztmax1);
      zetalt = fmin2(fmax2(zetalt, P.ztmin1), P.ztmax1);
      zetau = fmin2(fmax2(zetau, P.ztmin1 / rzsu), P.ztmax1 / rzsu);
      zetat = fmin2(fmax2(zetat, P.ztmin1 / rzst), P.ztmax1 / rzst);
      // water functions
      psmz = lookup(P.m1, zetau, P.ztmin1, P.dzeta1);
      psmzl = lookup(P.m1, zetalu, P.ztmin1, P.dzeta1);
      o.psim = -psmzl;
      simm = psmzl - psmz + rlogu;
      o.fm = psmzl + rlogu;
      pshz = lookup(P.h1, zetat, P.ztmin1, P.dzeta1);
      pshzl = lookup(P.h1, zetalt, P.ztmin1, P.dzeta1);
      o.psih = -pshzl;
      simh = (pshzl - pshz + rlogt) * P.fh01;
      o.fhh = (pshzl + rlogt) * P.fh01;
      ustark = ustar * vkarman;
      akms = fmax2(ustark / simm, cxchs);
      akhs = fmax2(ustark / simh, cxchs);
      // Beljaars correction for ustar
      if (dthv <= 0.) wstar2 = wwst2 * Kokkos::pow(Kokkos::fabs(btgh * akhs * dthv), 2. / 3.);
      else            wstar2 = 0.;
      ustar = fmax2(Kokkos::sqrt(akms * Kokkos::sqrt(du2 + wstar2)), epsust);
    }
  } else {
    // land points
    if (ntsd == 1) qs = qlow;
    zu = z0;
    uz0 = 0.;
    vz0 = 0.;
    zt = zu * ztfc;
    thz0 = ths;
    zq = zt;
    qz0 = qs;
    const Real tem = (tlow + tz0) * 0.5;
    const Real thm = (thelow + thz0) * 0.5;
    const Real a = thm * p608;
    const Real b = (elocp / tem - 1. - p608) * thm;
    dthv = ((thelow - thz0) * ((qlow + qz0 + cwmlow) * (0.5 * p608) + 1.) + (qlow - qz0 + cwmlow) * a + cwmlow * b);
    du2 = fmax2((ulow - uz0) * (ulow - uz0) + (vlow - vz0) * (vlow - vz0), epsu2);
    rib = btgx * dthv * zsl / du2;
    zslu = zsl + zu;
    rzsu = zslu / zu;
    rlogu = Kokkos::log(rzsu);
    zslt = zsl + zu;   // u,v and t are at the same level
    const Real czil = 0.1;
    const Real zilfc = -czil * vkarman * sqvisc;
    const Real czetmax = 10.;
    Real zzil;
    if (dthv > 0.) {
      // stable
      if (rib < ric) zzil = zilfc * (1.0 + (rib / ric) * (rib / ric) * czetmax);
      else           zzil = zilfc * (1.0 + czetmax);
    } else {
      // unstable
      zzil = zilfc;
    }
    for (int itr = 1; itr <= itrmx; ++itr) {
      // Zilitinkevitch fix for zt
      zt = fmax2(Kokkos::exp(zzil * Kokkos::sqrt(ustar * z0base)) * z0base, epszt);
      rzst = zslt / zt;
      rlogt = Kokkos::log(rzst);
      // 1./Monin-Obukhov length-scale
      rlmo = elfc * akhs * dthv / ((ustar * ustar) * ustar);
      zetalu = zslu * rlmo;
      zetalt = zslt * rlmo;
      zetau = zu * rlmo;
      zetat = zt * rlmo;
      zetalu = fmin2(fmax2(zetalu, P.ztmin2), P.ztmax2);
      zetalt = fmin2(fmax2(zetalt, P.ztmin2), P.ztmax2);
      zetau = fmin2(fmax2(zetau, P.ztmin2 / rzsu), P.ztmax2 / rzsu);
      zetat = fmin2(fmax2(zetat, P.ztmin2 / rzst), P.ztmax2 / rzst);
      // land functions
      psmz = lookup(P.m2, zetau, P.ztmin2, P.dzeta2);
      psmzl = lookup(P.m2, zetalu, P.ztmin2, P.dzeta2);
      o.psim = -psmzl;
      simm = psmzl - psmz + rlogu;
      o.fm = psmzl + rlogu;
      pshz = lookup(P.h2, zetat, P.ztmin2, P.dzeta2);
      pshzl = lookup(P.h2, zetalt, P.ztmin2, P.dzeta2);
      o.psih = -pshzl;
      simh = (pshzl - pshz + rlogt) * P.fh02;
      o.fhh = (pshzl + rlogt) * P.fh02;
      ustark = ustar * vkarman;
      akms = fmax2(ustark / simm, cxchl);
      akhs = fmax2(ustark / simh, cxchl);
      // Beljaars correction for ustar
      if (dthv <= 0.) wstar2 = wwst2 * Kokkos::pow(Kokkos::fabs(btgh * akhs * dthv), 2. / 3.);
      else            wstar2 = 0.;
      ustar = fmax2(Kokkos::sqrt(akms * Kokkos::sqrt(du2 + wstar2)), epsust);
    }
  }
  o.rib = rib;
  o.rlmo = rlmo;
  // countergradient fix (commented out in the Fortran)
  o.ct = 0.;
  // 2-m and 10-m diagnostics
  const Real umflx = akms * (ulow - uz0);
  const Real vmflx = akms * (vlow - vz0);
  const Real hsflx = akhs * (thlow - thz0);
  const Real hlflx = akhs * (qlow - qz0);
  Real zu10 = zu + 10.;
  const Real zt02 = zt + 02.;
  const Real zt10 = zt + 10.;
  Real rlnu10 = Kokkos::log(zu10 / zu);
  const Real rlnt02 = Kokkos::log(zt02 / zt);
  const Real rlnt10 = Kokkos::log(zt10 / zt);
  Real ztau10 = zu10 * rlmo;
  Real ztat02 = zt02 * rlmo;
  Real ztat10 = zt10 * rlmo;
  Real akms10, akhs02, akhs10;
  if (seamask > 0.5) {
    ztau10 = fmin2(fmax2(ztau10, P.ztmin1), P.ztmax1);
    ztat02 = fmin2(fmax2(ztat02, P.ztmin1), P.ztmax1);
    ztat10 = fmin2(fmax2(ztat10, P.ztmin1), P.ztmax1);
    const Real psm10 = lookup(P.m1, ztau10, P.ztmin1, P.dzeta1);
    const Real simm10 = psm10 - psmz + rlnu10;
    const Real psh02 = lookup(P.h1, ztat02, P.ztmin1, P.dzeta1);
    const Real simh02 = (psh02 - pshz + rlnt02) * P.fh01;
    const Real psh10 = lookup(P.h1, ztat10, P.ztmin1, P.dzeta1);
    const Real simh10 = (psh10 - pshz + rlnt10) * P.fh01;
    akms10 = fmax2(ustark / simm10, cxchs);
    akhs02 = fmax2(ustark / simh02, cxchs);
    akhs10 = fmax2(ustark / simh10, cxchs);
  } else {
    ztau10 = fmin2(fmax2(ztau10, P.ztmin2), P.ztmax2);
    ztat02 = fmin2(fmax2(ztat02, P.ztmin2), P.ztmax2);
    ztat10 = fmin2(fmax2(ztat10, P.ztmin2), P.ztmax2);
    const Real psm10 = lookup(P.m2, ztau10, P.ztmin2, P.dzeta2);
    const Real simm10 = psm10 - psmz + rlnu10;
    const Real psh02 = lookup(P.h2, ztat02, P.ztmin2, P.dzeta2);
    const Real simh02 = (psh02 - pshz + rlnt02) * P.fh02;
    const Real psh10 = lookup(P.h2, ztat10, P.ztmin2, P.dzeta2);
    const Real simh10 = (psh10 - pshz + rlnt10) * P.fh02;
    akms10 = fmax2(ustark / simm10, cxchl);
    akhs02 = fmax2(ustark / simh02, cxchl);
    akhs10 = fmax2(ustark / simh10, cxchl);
  }
  o.u10 = umflx / akms10 + uz0;
  o.v10 = vmflx / akms10 + vz0;
  Real th02 = hsflx / akhs02 + thz0;
  // be certain that the 2-m and 10-m theta are bracketed by thz0 and thlow
  if ((thlow > thz0 && (th02 < thz0 || th02 > thlow)) || (thlow < thz0 && (th02 > thz0 || th02 < thlow)))
    th02 = thz0 + 2. * rdz * (thlow - thz0);
  Real th10 = hsflx / akhs10 + thz0;
  if ((thlow > thz0 && (th10 < thz0 || th10 > thlow)) || (thlow < thz0 && (th10 > thz0 || th10 < thlow)))
    th10 = thz0 + 10. * rdz * (thlow - thz0);
  o.th02 = th02;
  o.th10 = th10;
  o.q02 = hlflx / akhs02 + qz0;
  o.q10 = hlflx / akhs10 + qz0;
  const Real term1 = -0.068283 / tlow;
  o.pshltr = psfc * Kokkos::exp(term1);
  o.z0out = zu;
  // "equivalent" z0 to approximate local shelter readings
  o.u10e = o.u10;
  o.v10e = o.v10;
  if (seamask < 0.5) {
    const Real zuuz = fmin2(zu * 0.50, 0.18);
    zu = fmax2(zu * 0.35, zuuz);
    zu10 = zu + 10.;
    rzsu = zu10 / zu;
    rlnu10 = Kokkos::log(rzsu);
    ztau10 = zu10 * rlmo;
    ztau10 = fmin2(fmax2(ztau10, P.ztmin2), P.ztmax2);
    // (ZETAU is recomputed and clamped here but no longer used)
    const Real psm10 = lookup(P.m2, ztau10, P.ztmin2, P.dzeta2);
    const Real simm10 = psm10 - psmz + rlnu10;
    const Real ekms10 = fmax2(ustark / simm10, cxchl);
    o.u10e = umflx / ekms10 + uz0;
    o.v10e = vmflx / ekms10 + vz0;
  }
  // other WRF driver arrays
  const Real rlow = rhosfc;
  o.chs = akhs;
  o.chs2 = akhs02;
  o.cqs2 = akhs02;
  o.hfx = -rlow * cp * hsflx;
  o.qfx = -rlow * hlflx * wetm;
  o.flx_lh = xlv * o.qfx;
  o.flhc = rlow * cp * akhs;
  o.flqc = rlow * akhs * wetm;
  Real qgh = ((1. - seamask) * pq0 + seamask * pq0sea) / plow * Kokkos::exp(a2s * (tlow - a3s) / (tlow - a4s));
  qgh = qgh / (1. - qgh);   // convert to mixing ratio
  o.qgh = qgh;
  o.cpm = cp * (1. + 0.8 * qlow);
}

// ---------------------------------------------------------------------------
// MYJPBL subroutines (column arrays with the Fortran's indices)

// MIXLEN: level 2.5 mixing length
KOKKOS_INLINE_FUNCTION void mixlen(int lmh, Arr u, Arr v, Arr t, Arr the, Arr q, Arr cwm, Arr q2, Arr z, Arr gm,
                                   Arr gh, Arr el, Real& pblh, int& lpbl, int& lmxl, Real& ct, Real& mixht, Arr q1,
                                   Arr dth, Arr elm, Arr rel) {
  using namespace bl;
  // the height of the PBL
  lpbl = 1;
  for (int k = lmh - 1; k >= 1; --k) {
    if (q2(k) <= epsq2 * fh) {
      lpbl = k;
      break;
    }
  }
  // (LPBL=LMH before the loop is overwritten in both outcomes)
  pblh = z(lpbl + 1) - z(lmh + 1);

  for (int k = 1; k <= lmh; ++k) q1(k) = 0.;
  for (int k = 1; k <= lmh - 1; ++k) dth(k) = the(k) - the(k + 1);
  for (int k = lmh - 2; k >= 1; --k) {
    if (dth(k) > 0. && dth(k + 1) <= 0.) {
      dth(k) = dth(k) + ct;
      break;
    }
  }
  ct = 0.;
  for (int k = 1; k <= lmh - 1; ++k) {
    const Real rdz = 2. / (z(k) - z(k + 2));
    const Real du = u(k) - u(k + 1), dv = v(k) - v(k + 1);
    const Real gml = (du * du + dv * dv) * rdz * rdz;
    gm(k) = fmax2(gml, epsgm);
    const Real tem = (t(k) + t(k + 1)) * 0.5;
    const Real thm = (the(k) + the(k + 1)) * 0.5;
    const Real a = thm * p608;
    const Real b = (elocp / tem - 1. - p608) * thm;
    Real ghl = (dth(k) * ((q(k) + q(k + 1) + cwm(k) + cwm(k + 1)) * (0.5 * p608) + 1.) +
                (q(k) - q(k + 1) + cwm(k) - cwm(k + 1)) * a + (cwm(k) - cwm(k + 1)) * b) *
               rdz;
    if (Kokkos::fabs(ghl) <= epsgh) ghl = epsgh;
    gh(k) = ghl;
  }
  // maximum mixing lengths and the level of the PBL top
  lmxl = lmh;
  for (int k = 1; k <= lmh - 1; ++k) {
    const Real gml = gm(k);
    const Real ghl = gh(k);
    if (ghl >= epsgh) {
      if (gml / ghl <= requ) {
        elm(k) = epsl;
        lmxl = k;
      } else {
        const Real aubr = (aubm * gml + aubh * ghl) * ghl;
        const Real bubr = bubm * gml + bubh * ghl;
        const Real qol2st = (-0.5 * bubr + Kokkos::sqrt(bubr * bubr * 0.25 - aubr * cubr)) * rcubr;
        const Real eloq2x = 1. / qol2st;
        elm(k) = fmax2(Kokkos::sqrt(eloq2x * q2(k)), epsl);
      }
    } else {
      const Real aden = (adnm * gml + adnh * ghl) * ghl;
      const Real bden = bdnm * gml + bdnh * ghl;
      const Real qol2un = -0.5 * bden + Kokkos::sqrt(bden * bden * 0.25 - aden);
      const Real eloq2x = 1. / (qol2un + epsru);
      elm(k) = fmax2(Kokkos::sqrt(eloq2x * q2(k)), epsl);
    }
  }
  if (elm(lmh - 1) == epsl) lmxl = lmh;
  // the height of the mixed layer
  const Real blmx = z(lmxl) - z(lmh + 1);
  mixht = blmx;
  for (int k = lpbl; k <= lmh; ++k) q1(k) = Kokkos::sqrt(q2(k));
  Real szq = 0., sq = 0.;
  for (int k = 1; k <= lmh - 1; ++k) {
    const Real qdzl = (q1(k) + q1(k + 1)) * (z(k + 1) - z(k + 2));
    szq = (z(k + 1) + z(k + 2) - z(lmh + 1) - z(lmh + 1)) * qdzl + szq;
    sq = qdzl + sq;
  }
  // asymptotic l in the Blackadar formula
  Real el0 = fmin2(alph * szq * 0.5 / sq, el0max);
  el0 = fmax2(el0, el0min);
  // above the PBL top
  const int lpblm = lpbl - 1 > 1 ? lpbl - 1 : 1;
  for (int k = 1; k <= lpblm; ++k) {
    el(k) = fmin2((z(k) - z(k + 2)) * elfc, elm(k));
    rel(k) = el(k) / elm(k);
  }
  // inside the PBL
  if (lpbl < lmh) {
    for (int k = lpbl; k <= lmh - 1; ++k) {
      const Real vkrmz = (z(k + 1) - z(lmh + 1)) * vkarman;
      el(k) = fmin2(vkrmz / (vkrmz / el0 + 1.), elm(k));
      rel(k) = el(k) / elm(k);
    }
  }
  for (int k = lpbl + 1; k <= lmh - 2; ++k) {
    const Real srel = fmin2(((rel(k - 1) + rel(k + 1)) * 0.5 + rel(k)) * 0.5, rel(k));
    el(k) = fmax2(srel * elm(k), epsl);
  }
}

// PRODQ2: level 2.5 q2 production/dissipation
KOKKOS_INLINE_FUNCTION void prodq2(int lmh, Real dtturbl, Real ustar, Arr gm, Arr gh, Arr el, Arr q2) {
  using namespace bl;
  for (int k = 1; k <= lmh - 1; ++k) {
    const Real gml = gm(k);
    const Real ghl = gh(k);
    // coefficients of the equilibrium equation
    const Real aequ = (aeqm * gml + aeqh * ghl) * ghl;
    const Real bequ = beqm * gml + beqh * ghl;
    // equilibrium solution for l/q
    const Real eqol2 = -0.5 * bequ + Kokkos::sqrt(bequ * bequ * 0.25 - aequ);
    // is there production/dissipation?
    if ((gml + ghl * ghl <= epstrb) || (ghl >= epsgh && gml / ghl <= requ) || (eqol2 <= eps2)) {
      // no turbulence
      q2(k) = epsq2;
      el(k) = epsl;
    } else {
      // turbulence: coefficients of the numerator, denominator
      const Real anum = (anmm * gml + anmh * ghl) * ghl;
      const Real bnum = bnmm * gml + bnmh * ghl;
      const Real aden = (adnm * gml + adnh * ghl) * ghl;
      const Real bden = bdnm * gml + bdnh * ghl;
      const Real cden = 1.;
      // coefficients of the numerator of the linearized eq.
      const Real arhs = -(anum * bden - bnum * aden) * 2.;
      const Real brhs = -anum * 4.;
      const Real crhs = -bnum * 2.;
      // initial value of l/q
      const Real dloq1 = el(k) / Kokkos::sqrt(q2(k));
      // first iteration for l/q, rhs=0
      const Real eloq21 = 1. / eqol2;
      const Real eloq11 = Kokkos::sqrt(eloq21);
      const Real eloq31 = eloq21 * eloq11;
      const Real eloq41 = eloq21 * eloq21;
      const Real eloq51 = eloq21 * eloq31;
      const Real rden1 = 1. / (aden * eloq41 + bden * eloq21 + cden);
      const Real rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1;
      // first-guess solution
      Real eloq12 = eloq11 + (dloq1 - eloq11) * Kokkos::exp(rhsp1 * dtturbl);
      eloq12 = fmax2(eloq12, eps1);
      // second iteration for l/q
      const Real eloq22 = eloq12 * eloq12;
      const Real eloq32 = eloq22 * eloq12;
      const Real eloq42 = eloq22 * eloq22;
      const Real eloq52 = eloq22 * eloq32;
      const Real rden2 = 1. / (aden * eloq42 + bden * eloq22 + cden);
      const Real rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + rb1;
      const Real rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2;
      const Real rhst2 = rhs2 / rhsp2;
      // corrected solution
      Real eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * Kokkos::exp(rhsp2 * dtturbl);
      eloq13 = fmax2(eloq13, eps1);
      // two iterations is enough in most cases ...
      const Real eloqn = eloq13;
      if (eloqn > eps1) {
        q2(k) = el(k) * el(k) / (eloqn * eloqn);
        q2(k) = fmax2(q2(k), epsq2);
        if (q2(k) == epsq2) el(k) = epsl;
      } else {
        q2(k) = epsq2;
        el(k) = epsl;
      }
    }
  }
  // lower boundary condition for q2
  q2(lmh) = fmax2(b1p23 * ustar * ustar, epsq2);
}

// DIFCOF: level 2.5 diffusion coefficients
KOKKOS_INLINE_FUNCTION void difcof(int lmh, Arr gm, Arr gh, Arr el, Arr q2, Arr z, Arr akm, Arr akh) {
  using namespace bl;
  for (int k = 1; k <= lmh - 1; ++k) {
    const Real ell = el(k);
    const Real eloq2 = ell * ell / q2(k);
    const Real eloq4 = eloq2 * eloq2;
    const Real gml = gm(k);
    const Real ghl = gh(k);
    const Real aden = (adnm * gml + adnh * ghl) * ghl;
    const Real bden = bdnm * gml + bdnh * ghl;
    const Real cden = 1.;
    const Real besm = bsmh * ghl;
    const Real besh = bshm * gml + bshh * ghl;
    const Real rden = 1. / (aden * eloq4 + bden * eloq2 + cden);
    const Real esm = (besm * eloq2 + cesm) * rden;
    const Real esh = (besh * eloq2 + cesh) * rden;
    const Real rdz = 2. / (z(k) - z(k + 2));
    const Real q1l = Kokkos::sqrt(q2(k));
    const Real elqdz = ell * q1l * rdz;
    akm(k) = elqdz * esm;
    akh(k) = elqdz * esh;
  }
}

// VDIFQ: vertical diffusion of q2 (TKE)
KOKKOS_INLINE_FUNCTION void vdifq(int lmh, Real dtdif, Arr q2, Arr el, Arr z, Arr akq, Arr cm, Arr cr, Arr dtoz,
                                  Arr rsq2) {
  using namespace bl;
  const Real esqhf_ = 0.5 * esq;
  for (int k = 1; k <= lmh - 2; ++k) {
    dtoz(k) = (dtdif + dtdif) / (z(k) - z(k + 2));
    akq(k) = Kokkos::sqrt((q2(k) + q2(k + 1)) * 0.5) * (el(k) + el(k + 1)) * esqhf_ / (z(k + 1) - z(k + 2));
    cr(k) = -dtoz(k) * akq(k);
  }
  cm(1) = dtoz(1) * akq(1) + 1.;
  rsq2(1) = q2(1);
  for (int k = 2; k <= lmh - 2; ++k) {
    const Real cf = -dtoz(k) * akq(k - 1) / cm(k - 1);
    cm(k) = -cr(k - 1) * cf + (akq(k - 1) + akq(k)) * dtoz(k) + 1.;
    rsq2(k) = -rsq2(k - 1) * cf + q2(k);
  }
  const Real dtozs = (dtdif + dtdif) / (z(lmh - 1) - z(lmh + 1));
  const Real akqs = Kokkos::sqrt((q2(lmh - 1) + q2(lmh)) * 0.5) * (el(lmh - 1) + elz0) * esqhf_ / (z(lmh) - z(lmh + 1));
  const Real cf = -dtozs * akq(lmh - 2) / cm(lmh - 2);
  q2(lmh - 1) = (dtozs * akqs * q2(lmh) - rsq2(lmh - 2) * cf + q2(lmh - 1)) /
                ((akq(lmh - 2) + akqs) * dtozs - cr(lmh - 2) * cf + 1.);
  for (int k = lmh - 2; k >= 1; --k) q2(k) = (-cr(k) * q2(k + 1) + rsq2(k)) / cm(k);
}

constexpr int nspec = 4;   // theta_e, q, qc, qi (flQI)

// VDIFH: vertical diffusion of the mass variables (MSS = 1, NSPEC = 4)
KOKKOS_INLINE_FUNCTION void vdifh(Real dtdif, int lmh, int lpbl, const Real* sz0, Real rkhs, const Real* clow,
                                  const Real* cts, const Arr* species, Arr rkh, Arr zhk, Arr rho, Arr cm, Arr cr,
                                  Arr dtoz, const Arr* rkct, const Arr* rss) {
  for (int k = 1; k <= lmh - 1; ++k) {
    dtoz(k) = dtdif / (zhk(k) - zhk(k + 1));
    cr(k) = -dtoz(k) * rkh(k);
    if (k < lpbl) {
      for (int m = 0; m < nspec; ++m) rkct[m](k) = 0.;
    } else {
      const Real rkhz = rkh(k) * (zhk(k) - zhk(k + 2));
      for (int m = 0; m < nspec; ++m) rkct[m](k) = rkhz * cts[m] * 0.5;
    }
  }
  // top level
  Real rhok = rho(1);
  cm(1) = dtoz(1) * rkh(1) + rhok;
  for (int m = 0; m < nspec; ++m) rss[m](1) = -rkct[m](1) * dtoz(1) + species[m](1) * rhok;
  // intermediate levels
  for (int k = 2; k <= lmh - 1; ++k) {
    const Real dtozl = dtoz(k);
    const Real cf = -dtozl * rkh(k - 1) / cm(k - 1);
    rhok = rho(k);
    cm(k) = -cr(k - 1) * cf + (rkh(k - 1) + rkh(k)) * dtozl + rhok;
    for (int m = 0; m < nspec; ++m)
      rss[m](k) = -rss[m](k - 1) * cf + (rkct[m](k - 1) - rkct[m](k)) * dtozl + species[m](k) * rhok;
  }
  // bottom level
  const Real dtozs = dtdif / (zhk(lmh) - zhk(lmh + 1));
  const Real rkhh = rkh(lmh - 1);
  const Real cf = -dtozs * rkhh / cm(lmh - 1);
  const Real cmb = cr(lmh - 1) * cf;
  rhok = rho(lmh);
  for (int m = 0; m < nspec; ++m) {
    const Real rkss = rkhs * clow[m];
    const Real cmsb = -cmb + (rkhh + rkss) * dtozs + rhok;
    const Real rssb = -rss[m](lmh - 1) * cf + rkct[m](lmh - 1) * dtozs + species[m](lmh) * rhok;
    species[m](lmh) = (dtozs * rkss * sz0[m] + rssb) / cmsb;
  }
  // backsubstitution
  for (int k = lmh - 1; k >= 1; --k) {
    const Real rcml = 1. / cm(k);
    for (int m = 0; m < nspec; ++m) species[m](k) = (-cr(k) * species[m](k + 1) + rss[m](k)) * rcml;
  }
}

// VDIFV: vertical diffusion of the velocity components
KOKKOS_INLINE_FUNCTION void vdifv(int lmh, Real dtdif, Real uz0, Real vz0, Real rkms, Arr u, Arr v, Arr rkm, Arr z,
                                  Arr rho, Arr cm, Arr cr, Arr dtoz, Arr rsu, Arr rsv) {
  for (int k = 1; k <= lmh - 1; ++k) {
    dtoz(k) = dtdif / (z(k) - z(k + 1));
    cr(k) = -dtoz(k) * rkm(k);
  }
  Real rhok = rho(1);
  cm(1) = dtoz(1) * rkm(1) + rhok;
  rsu(1) = u(1) * rhok;
  rsv(1) = v(1) * rhok;
  for (int k = 2; k <= lmh - 1; ++k) {
    const Real dtozl = dtoz(k);
    const Real cf = -dtozl * rkm(k - 1) / cm(k - 1);
    rhok = rho(k);
    cm(k) = -cr(k - 1) * cf + (rkm(k - 1) + rkm(k)) * dtozl + rhok;
    rsu(k) = -rsu(k - 1) * cf + u(k) * rhok;
    rsv(k) = -rsv(k - 1) * cf + v(k) * rhok;
  }
  const Real dtozs = dtdif / (z(lmh) - z(lmh + 1));
  const Real rkmh = rkm(lmh - 1);
  const Real cf = -dtozs * rkmh / cm(lmh - 1);
  rhok = rho(lmh);
  const Real rcmvb = 1. / ((rkmh + rkms) * dtozs - cr(lmh - 1) * cf + rhok);
  const Real dtozak = dtozs * rkms;
  u(lmh) = (dtozak * uz0 - rsu(lmh - 1) * cf + u(lmh) * rhok) * rcmvb;
  v(lmh) = (dtozak * vz0 - rsv(lmh - 1) * cf + v(lmh) * rhok) * rcmvb;
  for (int k = lmh - 1; k >= 1; --k) {
    const Real rcml = 1. / cm(k);
    u(k) = (-cr(k) * u(k + 1) + rsu(k)) * rcml;
    v(k) = (-cr(k) * v(k + 1) + rsv(k)) * rcml;
  }
}

constexpr int nslots = 52;

} // namespace

// ---------------------------------------------------------------------------
Tables Tables::make() {
  using sf::kztm;
  Tables t;
  t.psim1 = Kokkos::View<Real*, Device>("myj_psim1", kztm);
  t.psih1 = Kokkos::View<Real*, Device>("myj_psih1", kztm);
  t.psim2 = Kokkos::View<Real*, Device>("myj_psim2", kztm);
  t.psih2 = Kokkos::View<Real*, Device>("myj_psih2", kztm);
  auto m1 = Kokkos::create_mirror_view(t.psim1);
  auto h1 = Kokkos::create_mirror_view(t.psih1);
  auto m2 = Kokkos::create_mirror_view(t.psim2);
  auto h2 = Kokkos::create_mirror_view(t.psih2);
  // MYJSFCINIT (on the host, with libm as the Fortran; the loop is not
  // vectorized by gfortran: ZETA1/ZETA2 are floating-point recurrences)
  const Real pihf = 3.1415926 / 2., eps = 1.E-6;   // local PIHF, EPS
  t.fh01 = 1.;
  t.fh02 = 1.;
  t.ztmin1 = -5.0;
  t.ztmax1 = 1.0;
  t.ztmin2 = -5.0;
  t.ztmax2 = 1.0;
  const Real zrng1 = t.ztmax1 - t.ztmin1;
  const Real zrng2 = t.ztmax2 - t.ztmin2;
  t.dzeta1 = zrng1 / (kztm - 1);
  t.dzeta2 = zrng2 / (kztm - 1);
  Real zeta1 = t.ztmin1;
  Real zeta2 = t.ztmin2;
  for (int k = 1; k <= kztm; ++k) {
    if (zeta1 < 0.) {
      // Paulson 1970
      const Real x = std::sqrt(std::sqrt(1. - 16. * zeta1));
      m1(k - 1) = -2. * std::log((x + 1.) / 2.) - std::log((x * x + 1.) / 2.) + 2. * std::atan(x) - pihf;
      h1(k - 1) = -2. * std::log((x * x + 1.) / 2.);
    } else {
      // Holtslag and de Bruin 1988
      m1(k - 1) = 0.7 * zeta1 + 0.75 * zeta1 * (6. - 0.35 * zeta1) * std::exp(-0.35 * zeta1);
      h1(k - 1) = 0.7 * zeta1 + 0.75 * zeta1 * (6. - 0.35 * zeta1) * std::exp(-0.35 * zeta1);
    }
    if (zeta2 < 0.) {
      const Real x = std::sqrt(std::sqrt(1. - 16. * zeta2));
      m2(k - 1) = -2. * std::log((x + 1.) / 2.) - std::log((x * x + 1.) / 2.) + 2. * std::atan(x) - pihf;
      h2(k - 1) = -2. * std::log((x * x + 1.) / 2.);
    } else {
      m2(k - 1) = 0.7 * zeta2 + 0.75 * zeta2 * (6. - 0.35 * zeta2) * std::exp(-0.35 * zeta2);
      h2(k - 1) = 0.7 * zeta2 + 0.75 * zeta2 * (6. - 0.35 * zeta2) * std::exp(-0.35 * zeta2);
    }
    if (k == kztm) {
      t.ztmax1 = zeta1;
      t.ztmax2 = zeta2;
    }
    zeta1 = zeta1 + t.dzeta1;
    zeta2 = zeta2 + t.dzeta2;
  }
  t.ztmax1 = t.ztmax1 - eps;
  t.ztmax2 = t.ztmax2 - eps;
  Kokkos::deep_copy(t.psim1, m1);
  Kokkos::deep_copy(t.psih1, h1);
  Kokkos::deep_copy(t.psim2, m2);
  Kokkos::deep_copy(t.psih2, h2);
  return t;
}

void Workspace::allocate(int ni_, int nj_, int nk_) {
  ni = ni_; nj = nj_; nk = nk_;
  w = Kokkos::View<Real**, Kokkos::LayoutLeft, Device>("myj_ws", ni * nj, nslots * (nk + 2));
}

// ---------------------------------------------------------------------------
void myjsfc(const Tables& t, const Workspace& ws, int ni, int nj, int nk, int itimestep, bool usemyjpbl,
            int isftcflx, const SfcInputs& in, const SfcFields& f) {
  if (ws.nk != nk || ws.ni < ni || ws.nj < nj) throw std::runtime_error("cm1k myj::myjsfc: workspace size mismatch");
  const Psi P(t);
  const auto W = ws.w;
  const int L = nk + 2;
  const int ntsd = itimestep;
  const int kte = nk;
  Kokkos::parallel_for("myjsfc", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    using namespace sf;
    const int col = (i - 1) + ni * (j - 1);
    const ColWs cw{&W(col, 0), int(W.stride(1)), L};
    const Arr zint = cw.slot(0), thk = cw.slot(1), tk = cw.slot(2), qk = cw.slot(3), pk = cw.slot(4),
              cwmk = cw.slot(5), thek = cw.slot(6), q2k = cw.slot(7), zhk = cw.slot(8), uk = cw.slot(9),
              vk = cw.slot(10);
    // make preparations
    for (int k = 1; k <= kte + 1; ++k) zint(k) = 0.;
    zint(kte + 1) = in.ht(i, j, 1);   // Z at bottom of lowest sigma layer
    const Real pblsave = f.pblh(i, j, 1);
    f.pblh(i, j, 1) = -1.;
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      zint(k) = zint(k + 1) + in.dz(i, j, kflip);
    }
    if (ntsd == 1) f.ust(i, j, 1) = 0.1;
    f.ct(i, j, 1) = 0.;
    // setup_integration (LOWLYR = 1)
    const int lmh = kte;
    const Real psfc = in.pint(i, j, 1);
    const Real thsk = f.tsk(i, j, 1) / Kokkos::pow(psfc / p1000mb, capa);
    const Real seamask = f.xland(i, j, 1) - 1.;
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      thk(k) = in.th(i, j, kflip);
      tk(k) = in.t(i, j, kflip);
      const Real ratiomx = in.qv(i, j, kflip);
      qk(k) = ratiomx / (1. + ratiomx);
      pk(k) = in.pmid(i, j, kflip);
      cwmk(k) = in.qc(i, j, kflip);
      thek(k) = (cwmk(k) * (-elocp / tk(k)) + 1.) * thk(k);
      q2k(k) = 2. * in.q2(i, j, kflip);
      zhk(k) = zint(k);
    }
    zhk(kte + 1) = in.ht(i, j, 1);
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      uk(k) = in.u(i, j, kflip);
      vk(k) = in.v(i, j, kflip);
    }
    // the height of the PBL
    if (usemyjpbl) {
      int lpbl = 1;
      for (int k = lmh - 1; k >= 1; --k) {
        if (q2k(k) <= epsq2 * fh) {
          lpbl = k;
          break;
        }
      }
      f.pblh(i, j, 1) = zhk(lpbl) - zhk(lmh + 1);
    } else {
      f.pblh(i, j, 1) = pblsave;
    }
    // the surface exchange coefficients
    const Real plow = pk(lmh);
    const Real tlow = tk(lmh);
    const Real thlow = thk(lmh);
    const Real thelow = thek(lmh);
    const Real qlow = qk(lmh);
    const Real cwmlow = cwmk(lmh);
    const Real ulow = uk(lmh);
    const Real vlow = vk(lmh);
    const Real zsl = (zhk(lmh) - zhk(lmh + 1)) * 0.5;
    const Real apesfc = Kokkos::pow(psfc / p1000mb, capa);
    Real tz0;
    if (ntsd == 1) tz0 = f.tsk(i, j, 1);
    else           tz0 = f.thz0(i, j, 1) * apesfc;

    Real qs = f.qsfc(i, j, 1), uz0 = f.uz0(i, j, 1), vz0 = f.vz0(i, j, 1), thz0 = f.thz0(i, j, 1),
         qz0 = f.qz0(i, j, 1), ustar = f.ust(i, j, 1), z0 = f.znt(i, j, 1), akms = f.akms(i, j, 1),
         akhs = f.akhs(i, j, 1);
    SfcdifOut o{};
    sfcdif(P, ntsd, seamask, thsk, qs, psfc, uz0, vz0, tz0, f.tsk(i, j, 1), thz0, qz0, ustar, z0,
           f.z0base(i, j, 1), akms, akhs, f.pblh(i, j, 1), f.mavail(i, j, 1), ulow, vlow, tlow, thlow, thelow,
           qlow, cwmlow, zsl, plow, in.rhosfc(i, j, 1), isftcflx, o);
    f.qsfc(i, j, 1) = qs; f.uz0(i, j, 1) = uz0; f.vz0(i, j, 1) = vz0; f.thz0(i, j, 1) = thz0;
    f.qz0(i, j, 1) = qz0; f.ust(i, j, 1) = ustar; f.znt(i, j, 1) = z0; f.akms(i, j, 1) = akms;
    f.akhs(i, j, 1) = akhs;
    f.ct(i, j, 1) = o.ct; f.rmol(i, j, 1) = o.rlmo; f.rib(i, j, 1) = o.rib;
    f.chs(i, j, 1) = o.chs; f.chs2(i, j, 1) = o.chs2; f.cqs2(i, j, 1) = o.cqs2;
    f.hfx(i, j, 1) = o.hfx; f.qfx(i, j, 1) = o.qfx; f.lh(i, j, 1) = o.flx_lh;
    f.flhc(i, j, 1) = o.flhc; f.flqc(i, j, 1) = o.flqc; f.qgh(i, j, 1) = o.qgh; f.cpm(i, j, 1) = o.cpm;
    f.u10(i, j, 1) = o.u10; f.v10(i, j, 1) = o.v10;
    Real tshltr = o.th02, th10 = o.th10, qshltr = o.q02, q10 = o.q10;
    f.pshltr(i, j, 1) = o.pshltr; f.u10e(i, j, 1) = o.u10e; f.v10e(i, j, 1) = o.v10e;
    f.psim(i, j, 1) = o.psim; f.psih(i, j, 1) = o.psih; f.fm(i, j, 1) = o.fm; f.fhh(i, j, 1) = o.fhh;
    f.z0out(i, j, 1) = o.z0out;
    // remove supersaturation at 2 m and 10 m
    const Real rapa = apesfc;
    const Real th02p = tshltr;
    const Real th10p = th10;
    f.th02(i, j, 1) = tshltr;
    const Real rapa02 = rapa - gocp02 / th02p;
    const Real rapa10 = rapa - gocp10 / th10p;
    const Real t02p = th02p * rapa02;
    const Real t10p = th10p * rapa10;
    f.t02(i, j, 1) = f.th02(i, j, 1) * apesfc;
    const Real p02p = Kokkos::pow(rapa02, rcap) * p1000mb;
    const Real p10p = Kokkos::pow(rapa10, rcap) * p1000mb;
    const Real qs02 = pq0 / p02p * Kokkos::exp(a2 * (t02p - a3) / (t02p - a4));
    const Real qs10 = pq0 / p10p * Kokkos::exp(a2 * (t10p - a3) / (t10p - a4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    f.tshltr(i, j, 1) = tshltr;
    f.th10(i, j, 1) = th10;
    f.qshltr(i, j, 1) = qshltr;
    f.q10(i, j, 1) = q10;
    f.q02(i, j, 1) = qshltr / (1. - qshltr);
  });
}

// ---------------------------------------------------------------------------
void myjpbl(const Workspace& ws, int ni, int nj, int nk, Real dt, const PblInputs& in, const PblFields& f) {
  if (ws.nk != nk || ws.ni < ni || ws.nj < nj) throw std::runtime_error("cm1k myj::myjpbl: workspace size mismatch");
  const auto W = ws.w;
  const int L = nk + 2;
  const int kte = nk;
  Kokkos::parallel_for("myjpbl", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    using namespace bl;
    const int col = (i - 1) + ni * (j - 1);
    const ColWs cw{&W(col, 0), int(W.stride(1)), L};
    int n = 0;
    const Arr zint = cw.slot(n++), ape = cw.slot(n++), the = cw.slot(n++), cwm = cw.slot(n++), tk = cw.slot(n++),
              thek = cw.slot(n++), qk = cw.slot(n++), cwmk = cw.slot(n++), uk = cw.slot(n++), vk = cw.slot(n++),
              q2k = cw.slot(n++), zhk = cw.slot(n++), gm = cw.slot(n++), gh = cw.slot(n++), el = cw.slot(n++),
              akhk = cw.slot(n++), akmk = cw.slot(n++), akh = cw.slot(n++), akm = cw.slot(n++),
              qcwk = cw.slot(n++), qcik = cw.slot(n++), rhok = cw.slot(n++), q1 = cw.slot(n++), dth = cw.slot(n++),
              elm = cw.slot(n++), rel = cw.slot(n++), akq = cw.slot(n++), cm = cw.slot(n++), cr = cw.slot(n++),
              dtoz = cw.slot(n++), rsq2 = cw.slot(n++), rsu = cw.slot(n++), rsv = cw.slot(n++);
    Arr species[nspec], rkct[nspec], rss[nspec];
    for (int m = 0; m < nspec; ++m) species[m] = cw.slot(n++);
    for (int m = 0; m < nspec; ++m) rkct[m] = cw.slot(n++);
    for (int m = 0; m < nspec; ++m) rss[m] = cw.slot(n++);

    // make preparations (STEPBL = 1)
    const Real dtturbl = dt * 1;
    const Real rdtturbl = 1. / dtturbl;
    const Real dtdif = dtturbl;
    for (int k = 1; k <= kte - 1; ++k) akm(k) = 0.;
    for (int k = 1; k <= kte + 1; ++k) zint(k) = 0.;
    zint(kte + 1) = in.ht(i, j, 1);   // Z at bottom of lowest sigma layer
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      zint(k) = zint(k + 1) + in.dz(i, j, kflip);
      const Real apex = 1. / in.exner(i, j, k);
      ape(k) = apex;
      const Real tx = in.t(i, j, k);
      cwm(k) = in.qcw(i, j, k);
      cwm(k) = cwm(k) + in.qci(i, j, k);   // flQI
      the(k) = (cwm(k) * (-elocp / tx) + 1.) * in.th(i, j, k);
    }
    for (int k = 1; k <= kte; ++k) f.el_myj(i, j, k) = 0.;

    // setup_integration (LOWLYR = 1)
    int lmh = kte;
    const Real psfc = in.pint(i, j, 1);
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      tk(k) = in.t(i, j, kflip);
      thek(k) = the(kflip);
      const Real ratiomx = in.qv(i, j, kflip);
      qk(k) = ratiomx / (1. + ratiomx);
      cwmk(k) = cwm(kflip);
      uk(k) = in.u(i, j, kflip);
      vk(k) = in.v(i, j, kflip);
      // TKE=0.5*(q**2) ==> q**2=2.*TKE
      q2k(k) = 2. * f.tke_myj(i, j, kflip);
      zhk(k) = zint(k);
    }
    zhk(kte + 1) = in.ht(i, j, 1);
    // the mixing length
    Real pblh = 0, ct = f.ct(i, j, 1), mixht = 0;
    int lpbl = 0, lmxl = 0;
    mixlen(lmh, uk, vk, tk, thek, qk, cwmk, q2k, zhk, gm, gh, el, pblh, lpbl, lmxl, ct, mixht, q1, dth, elm, rel);
    f.pblh(i, j, 1) = pblh;
    f.ct(i, j, 1) = ct;
    f.mixht(i, j, 1) = mixht;
    // production/dissipation of TKE
    prodq2(lmh, dtturbl, f.ust(i, j, 1), gm, gh, el, q2k);
    // the model layer (counting upward) containing the top of the PBL
    f.kpbl(i, j, 1) = Real(kte - lpbl + 1);
    // exchange coefficients in the free atmosphere
    difcof(lmh, gm, gh, el, q2k, zhk, akmk, akhk);
    for (int k = 1; k <= kte - 1; ++k) {
      const int kflip = kte - k;
      akh(k) = akhk(k);
      akm(k) = akmk(k);
      const Real deltaz = 0.5 * (zhk(kflip) - zhk(kflip + 2));
      f.exch_h(i, j, k) = akhk(kflip) * deltaz;
      f.exch_m(i, j, k) = akmk(kflip) * deltaz;
    }
    // vertical diffusion of TKE
    vdifq(lmh, dtdif, q2k, el, zhk, akq, cm, cr, dtoz, rsq2);
    // the new TKE and mixing length
    for (int k = 1; k <= kte; ++k) {
      const int kflip = kte + 1 - k;
      q2k(kflip) = fmax2(q2k(kflip), epsq2);
      f.tke_myj(i, j, k) = 0.5 * q2k(kflip);
      if (kflip < kte) f.el_myj(i, j, k) = el(kflip);   // EL is not defined at KTE (ground surface)
    }

    // convert surface sensible temperature to potential temperature
    const Real thsk = f.tsk(i, j, 1) * Kokkos::pow(1.E5 / psfc, capa);

    // main_integration, first I loop: theta, q, qc, qi
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      thek(k) = the(kflip);
      const Real ratiomx = in.qv(i, j, kflip);
      qk(k) = ratiomx / (1. + ratiomx);
      cwmk(k) = cwm(kflip);
      qcwk(k) = in.qcw(i, j, kflip);
      species[0](k) = thek(k);
      species[1](k) = qk(k);
      species[2](k) = qcwk(k);
      qcik(k) = in.qci(i, j, kflip);
      species[3](k) = qcik(k);
      zhk(k) = zint(k);
      rhok(k) = in.pmid(i, j, kflip) / (r_d * in.t(i, j, kflip) * (1. + p608 * qk(k) - cwmk(k)));
    }
    // exchange coefficients times the density at the bottom interface level
    for (int k = 1; k <= kte - 1; ++k) akhk(k) = akh(k) * 0.5 * (rhok(k) + rhok(k + 1));
    zhk(kte + 1) = zint(kte + 1);
    const Real seamask = f.xland(i, j, 1) - 1.;
    f.thz0(i, j, 1) = (1. - seamask) * thsk + seamask * f.thz0(i, j, 1);
    const int llow = 1;
    const Real akhs_dens = f.akhs(i, j, 1) * rhok(kte + 1 - llow);
    if (seamask < 0.5) {
      const Real qfc1 = xlv * f.chklowq(i, j, 1) * akhs_dens;
      // (SNOW > 0 or SICE > 0.5: QFC1*RLIVWV; CM1 has snow = sice = 0)
      if (qfc1 > 0.) {
        const Real qlow = qk(kte + 1 - llow);
        f.qsfc(i, j, 1) = qlow + f.elflx(i, j, 1) / qfc1;
      }
    } else {
      const Real psfc_ = in.pint(i, j, llow);
      const Real exnsfc = Kokkos::pow(1.E5 / psfc_, capa);
      f.qsfc(i, j, 1) = pq0sea / psfc_ * Kokkos::exp(a2 * (thsk - a3 * exnsfc) / (thsk - a4 * exnsfc));
    }
    f.qz0(i, j, 1) = (1. - seamask) * f.qsfc(i, j, 1) + seamask * f.qz0(i, j, 1);
    const Real sz0[nspec] = {f.thz0(i, j, 1), f.qz0(i, j, 1), 0., 0.};
    const Real clow[nspec] = {1., f.chklowq(i, j, 1), 0., 0.};
    const Real cts[nspec] = {f.ct(i, j, 1), 0., 0., 0.};
    lmh = kte + 1 - llow;
    // vertical diffusion of temperature and water
    vdifh(dtdif, lmh, lpbl, sz0, akhs_dens, clow, cts, species, akhk, zhk, rhok, cm, cr, dtoz, rkct, rss);
    // primary variable tendencies
    for (int k = 1; k <= kte; ++k) {
      thek(k) = species[0](k);
      qk(k) = species[1](k);
      qcwk(k) = species[2](k);
      cwmk(k) = qcwk(k);
      qcik(k) = species[3](k);
      cwmk(k) = cwmk(k) + qcik(k);
    }
    for (int k = 1; k <= kte; ++k) {
      const int kflip = kte + 1 - k;
      const Real thold = in.th(i, j, k);
      const Real thnew = thek(kflip) + cwmk(kflip) * elocp * ape(k);
      const Real dtdt = (thnew - thold) * rdtturbl;
      const Real qold = in.qv(i, j, k) / (1. + in.qv(i, j, k));
      const Real dqdt = (qk(kflip) - qold) * rdtturbl;
      f.rthblten(i, j, k) = dtdt;
      const Real omq = 1. - qk(kflip);
      f.rqvblten(i, j, k) = dqdt / (omq * omq);
      f.rqcblten(i, j, k) = (qcwk(kflip) - in.qcw(i, j, k)) * rdtturbl;
      f.rqiblten(i, j, k) = (qcik(kflip) - in.qci(i, j, k)) * rdtturbl;
    }

    // main_integration, second I loop: velocity components
    for (int k = 1; k <= kte - 1; ++k) {
      akmk(k) = akm(k);
      akmk(k) = akmk(k) * (rhok(k) + rhok(k + 1)) * 0.5;
    }
    const Real akms_dens = f.akms(i, j, 1) * rhok(kte + 1 - llow);
    for (int k = kte; k >= 1; --k) {
      const int kflip = kte + 1 - k;
      uk(k) = in.u(i, j, kflip);
      vk(k) = in.v(i, j, kflip);
      zhk(k) = zint(k);
    }
    zhk(kte + 1) = zint(kte + 1);
    vdifv(lmh, dtdif, f.uz0(i, j, 1), f.vz0(i, j, 1), akms_dens, uk, vk, akmk, zhk, rhok, cm, cr, dtoz, rsu, rsv);
    for (int k = 1; k <= kte; ++k) {
      const int kflip = kte + 1 - k;
      f.rublten(i, j, k) = (uk(kflip) - in.u(i, j, k)) * rdtturbl;
      f.rvblten(i, j, k) = (vk(kflip) - in.v(i, j, k)) * rdtturbl;
    }
  });
}

} // namespace cm1k::myj
