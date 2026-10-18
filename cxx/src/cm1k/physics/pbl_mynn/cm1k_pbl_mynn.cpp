// Port of the MYNN-EDMF PBL (src/module_bl_mynn*.F) as CM1 calls it. See
// the header for scope. Every routine keeps the Fortran order of operations:
// `x**2` is x*x, `x**3` is (x*x)*x, other real exponents are pow; Fortran's
// unary minus (which binds looser than * and /) is written where it gives
// the same bits (negation is exact). Terms that are identically zero in
// CM1's configuration but appear in a sum (the downdraft fluxes sd_*, the
// environmental subsidence/detrainment sub_*/det_*, TKEprodTD, SPP) are
// kept as explicit zeros so the sums round the same way.
#include "physics/pbl_mynn/cm1k_pbl_mynn.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cm1k::pbl_mynn {

namespace {

KOKKOS_FORCEINLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_FORCEINLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// ---------------------------------------------------------------------------
// constants: module_model_constants (src/module_ra_etc.F),
// module_bl_mynn_common, module_bl_mynn

constexpr Real r_d = 287.04, cp = 1005.7, karman = 0.4, g = 9.81, p1000mb = 100000., r_v = 461.5;
constexpr Real rcp = r_d / cp;
constexpr Real xls = 2.834E6, xlv = 2.501E6, xlf = xls - xlv;
constexpr Real svpt0 = 273.15;
constexpr Real rvovrd = r_v / r_d, cpv = 4. * r_v, cliq = 4190., cice = 2106.;
constexpr Real p608 = rvovrd - 1.;

constexpr Real onethird = 1. / 3., twothirds = 2. / 3.;
constexpr Real tref = 300.0, tice = 240.0, grav = g, t0c = svpt0;
constexpr Real gtr = grav / tref;
constexpr Real tv0 = p608 * tref;
constexpr Real xlscp = (xlv + xlf) / cp, xlvcp = xlv / cp;

constexpr Real pr = 0.74, g1 = 0.235, b1 = 24.0, b2 = 15.0, c2 = 0.729, c3 = 0.340, c5 = 0.2;
constexpr Real a1 = b1 * (1.0 - 3.0 * g1) / 6.0;
constexpr Real c1 = g1 - 1.0 / (3.0 * a1 * 2.88449914061481660);
constexpr Real a2 = a1 * (g1 - c1) / (g1 * pr);
constexpr Real g2 = b2 / b1 * (1.0 - c3) + 2.0 * a1 / b1 * (3.0 - 2.0 * c2);
constexpr Real cc2 = 1.0 - c2, cc3 = 1.0 - c3;
constexpr Real e1c = 3.0 * a2 * b2 * cc3, e2c = 9.0 * a1 * a2 * cc2, e3c = 9.0 * a2 * a2 * cc2 * (1.0 - c5),
               e4c = 12.0 * a1 * a2 * cc2, e5c = 6.0 * a1 * a1;
constexpr Real qmin = 0.0, zmax = 1.0, Sqfac = 3.0;
constexpr Real qkemin = 1.e-12;
constexpr Real tliq = 269.;
constexpr Real cphm_unst = 16.0, cphh_unst = 16.0;

constexpr Real J0 = .611583699E03, J1 = .444606896E02, J2 = .143177157E01, J3 = .264224321E-1,
               J4 = .299291081E-3, J5 = .203154182E-5, J6 = .702620698E-8, J7 = .379534310E-11,
               J8 = -.321582393E-13;
constexpr Real K0 = .609868993E03, K1 = .499320233E02, K2 = .184672631E01, K3 = .402737184E-1,
               K4 = .565392987E-3, K5 = .521693933E-5, K6 = .307839583E-7, K7 = .105785160E-9,
               K8 = .161444444E-12;

// EXP(-1.), which gfortran folds at compile time: the correctly rounded value
constexpr Real exp_m1 = 0.36787944117144233;

// Terms that are zero in CM1's configuration (see the file comment).
constexpr Real ZERO = 0.0;

constexpr int NUP = 10;      // DMP_mf's number of plumes
constexpr int NSLOTS = 160;  // column arrays per column in the workspace

// ---------------------------------------------------------------------------
// per-column scratch

// A column array a(k), k = 0..nk+1 (Fortran kts = 1).
struct Col {
  Real* p;
  int s;
  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int k) const { return p[k * s]; }
};
// A plume array a(k,i), k = 0..nk+1, i = 1..NUP.
struct Col2 {
  Real* p;
  int s, L;
  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int k, int i) const { return p[((i - 1) * L + k) * s]; }
};
// Stack allocator over one column's workspace; pass it by value so a
// callee's arrays are released when it returns.
struct Ws {
  Real* base;
  int s, L, next;
  KOKKOS_INLINE_FUNCTION Col get() {
    if (next + 1 > NSLOTS) Kokkos::abort("pbl_mynn: column workspace exhausted");
    Col c{base + std::size_t(next) * L * s, s};
    next += 1;
    return c;
  }
  KOKKOS_INLINE_FUNCTION Col2 get2() {
    if (next + NUP > NSLOTS) Kokkos::abort("pbl_mynn: column workspace exhausted");
    Col2 c{base + std::size_t(next) * L * s, s, L};
    next += NUP;
    return c;
  }
};

// ---------------------------------------------------------------------------
// thermodynamics (module_bl_mynn)

KOKKOS_INLINE_FUNCTION Real qsat_blend(Real t, Real P) {
  const Real XC = fmax2(-80., t - t0c);
  if (t >= t0c) {
    const Real ESL = J0 + XC * (J1 + XC * (J2 + XC * (J3 + XC * (J4 + XC * (J5 + XC * (J6 + XC * (J7 + XC * J8)))))));
    return 0.622 * ESL / fmax2(P - ESL, 1e-5);
  } else if (t <= tice) {
    const Real ESI = K0 + XC * (K1 + XC * (K2 + XC * (K3 + XC * (K4 + XC * (K5 + XC * (K6 + XC * (K7 + XC * K8)))))));
    return 0.622 * ESI / fmax2(P - ESI, 1e-5);
  } else {
    const Real ESL = J0 + XC * (J1 + XC * (J2 + XC * (J3 + XC * (J4 + XC * (J5 + XC * (J6 + XC * (J7 + XC * J8)))))));
    const Real ESI = K0 + XC * (K1 + XC * (K2 + XC * (K3 + XC * (K4 + XC * (K5 + XC * (K6 + XC * (K7 + XC * K8)))))));
    const Real RSLF = 0.622 * ESL / fmax2(P - ESL, 1e-5);
    const Real RSIF = 0.622 * ESI / fmax2(P - ESI, 1e-5);
    const Real chi = (t0c - t) / (t0c - tice);
    return (1. - chi) * RSLF + chi * RSIF;
  }
}

KOKKOS_INLINE_FUNCTION Real xl_blend(Real t) {
  if (t >= t0c) {
    return xlv + (cpv - cliq) * (t - t0c);
  } else if (t <= tice) {
    return xls + (cpv - cice) * (t - t0c);
  } else {
    const Real xlvt = xlv + (cpv - cliq) * (t - t0c);
    const Real xlst = xls + (cpv - cice) * (t - t0c);
    const Real chi = (t0c - t) / (t0c - tice);
    return (1. - chi) * xlvt + chi * xlst;
  }
}

// ---------------------------------------------------------------------------
// similarity functions (bl_mynn_stfunc = 1)

constexpr Real am_st = 6.1, bm_st = 2.5, rbm_st = 1. / bm_st;
constexpr Real ah_st = 5.3, bh_st = 1.1, rbh_st = 1. / bh_st;
constexpr Real am_unst = 10., ah_unst = 34.;

KOKKOS_INLINE_FUNCTION Real phim(Real zet) {
  Real phi_m;
  if (zet >= 0.0) {
    const Real dummy_0 = 1 + Kokkos::pow(zet, bm_st);
    const Real dummy_1 = zet + Kokkos::pow(dummy_0, rbm_st);
    const Real dummy_11 = 1 + Kokkos::pow(dummy_0, rbm_st - 1) * Kokkos::pow(zet, bm_st - 1);
    const Real dummy_2 = (-am_st / dummy_1) * dummy_11;
    phi_m = 1 - zet * dummy_2;
  } else {
    Real dummy_0 = Kokkos::pow(1.0 - cphm_unst * zet, 0.25);
    phi_m = 1. / dummy_0;
    const Real dummy_psi = 2. * Kokkos::log(0.5 * (1. + dummy_0)) + Kokkos::log(0.5 * (1. + dummy_0 * dummy_0)) -
                           2. * Kokkos::atan(dummy_0) + 1.570796;
    dummy_0 = (1. - am_unst * zet);
    Real dummy_1 = Kokkos::pow(dummy_0, 0.333333);
    Real dummy_11 = -0.33333 * am_unst * Kokkos::pow(dummy_0, -0.6666667);
    Real dummy_2 = 0.33333 * (dummy_1 * dummy_1 + dummy_1 + 1.);
    Real dummy_22 = 0.3333 * dummy_11 * (2. * dummy_1 + 1.);
    const Real dummy_3 = 0.57735 * (2. * dummy_1 + 1.);
    const Real dummy_33 = 1.1547 * dummy_11;
    const Real dummy_4 = 1.5 * Kokkos::log(dummy_2) - 1.73205 * Kokkos::atan(dummy_3) + 1.813799364;
    const Real dummy_44 = (1.5 / dummy_2) * dummy_22 - 1.73205 * dummy_33 / (1. + dummy_3 * dummy_3);
    dummy_0 = zet * zet;
    dummy_1 = 1. / (1. + dummy_0);
    dummy_11 = 2. * zet;
    dummy_2 = ((1 - phi_m) / zet + dummy_11 * dummy_4 + dummy_0 * dummy_44) * dummy_1;
    dummy_22 = -dummy_11 * (dummy_psi + dummy_0 * dummy_4) * (dummy_1 * dummy_1);
    phi_m = 1. - zet * (dummy_2 + dummy_22);
  }
  return phi_m;
}

KOKKOS_INLINE_FUNCTION Real phih(Real zet) {
  if (zet >= 0.0) {
    const Real dummy_0 = 1 + Kokkos::pow(zet, bh_st);
    const Real dummy_1 = zet + Kokkos::pow(dummy_0, rbh_st);
    const Real dummy_11 = 1 + Kokkos::pow(dummy_0, rbh_st - 1) * Kokkos::pow(zet, bh_st - 1);
    const Real dummy_2 = (-ah_st / dummy_1) * dummy_11;
    return 1 - zet * dummy_2;
  } else {
    Real dummy_0 = Kokkos::pow(1.0 - cphh_unst * zet, 0.5);
    const Real phh = 1. / dummy_0;
    const Real dummy_psi = 2. * Kokkos::log(0.5 * (1. + dummy_0));
    dummy_0 = (1. - ah_unst * zet);
    Real dummy_1 = Kokkos::pow(dummy_0, 0.333333);
    Real dummy_11 = -0.33333 * ah_unst * Kokkos::pow(dummy_0, -0.6666667);
    Real dummy_2 = 0.33333 * (dummy_1 * dummy_1 + dummy_1 + 1.);
    Real dummy_22 = 0.3333 * dummy_11 * (2. * dummy_1 + 1.);
    const Real dummy_3 = 0.57735 * (2. * dummy_1 + 1.);
    const Real dummy_33 = 1.1547 * dummy_11;
    const Real dummy_4 = 1.5 * Kokkos::log(dummy_2) - 1.73205 * Kokkos::atan(dummy_3) + 1.813799364;
    const Real dummy_44 = (1.5 / dummy_2) * dummy_22 - 1.73205 * dummy_33 / (1. + dummy_3 * dummy_3);
    dummy_0 = zet * zet;
    dummy_1 = 1. / (1. + dummy_0);
    dummy_11 = 2. * zet;
    dummy_2 = ((1 - phh) / zet + dummy_11 * dummy_4 + dummy_0 * dummy_44) * dummy_1;
    dummy_22 = -dummy_11 * (dummy_psi + dummy_0 * dummy_4) * (dummy_1 * dummy_1);
    return 1. - zet * (dummy_2 + dummy_22);
  }
}

// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION void tridiag2(Ws ws, int n, Col a, Col b, Col c, Col d, Col x) {
  Col cp = ws.get(), dp = ws.get();
  cp(1) = c(1) / b(1);
  dp(1) = d(1) / b(1);
  for (int i = 2; i <= n; ++i) {
    const Real m = b(i) - cp(i - 1) * a(i);
    cp(i) = c(i) / m;
    dp(i) = (d(i) - dp(i - 1) * a(i)) / m;
  }
  x(n) = dp(n);
  for (int i = n - 1; i >= 1; --i) x(i) = dp(i) - cp(i) * x(i + 1);
}

// GET_PBLH: zi (PBLH) and kzi (KPBL)
KOKKOS_INLINE_FUNCTION void get_pblh(int nk, Real& zi, Col thetav, Col qke, Col zw, Col dz, Real landsea,
                                     int& kzi) {
  constexpr Real sbl_lim = 200., sbl_damp = 400.;
  kzi = 2;
  int k = 2;
  Real minthv = 9.E9;   // (the Fortran also records its level, kthv, unused)
  while (k <= nk + 1 && zw(k) <= 200.) {
    if (minthv > thetav(k)) minthv = thetav(k);
    k = k + 1;
  }
  const Real delt_thv = ((landsea - 1.5) >= 0) ? 1.0 : 1.25;
  zi = 0.;
  for (k = 2; k <= nk - 1; ++k) {
    if (thetav(k) >= (minthv + delt_thv)) {
      zi = zw(k) - dz(k - 1) * fmin2((thetav(k) - (minthv + delt_thv)) / fmax2(thetav(k) - thetav(k - 1), 1E-6), 1.0);
    }
    if (k == nk - 1) zi = zw(2);
    if (zi != 0.0) break;
  }
  const Real maxqke = fmax2(qke(1), 0.);
  Real TKEeps = maxqke / 40.;
  TKEeps = fmax2(TKEeps, 0.02);
  Real PBLH_TKE = 0.;
  for (k = 2; k <= nk - 1; ++k) {
    const Real qtke = fmax2(qke(k) / 2., 0.);
    const Real qtkem1 = fmax2(qke(k - 1) / 2., 0.);
    if (qtke <= TKEeps) {
      PBLH_TKE = zw(k) - dz(k - 1) * fmin2((TKEeps - qtke) / fmax2(qtkem1 - qtke, 1E-6), 1.0);
      PBLH_TKE = fmax2(PBLH_TKE, zw(2));
    }
    if (k == nk - 1) PBLH_TKE = zw(2);
    if (PBLH_TKE != 0.) break;
  }
  PBLH_TKE = fmin2(PBLH_TKE, zi + 350.);
  PBLH_TKE = fmax2(PBLH_TKE, fmax2(zi - 350., 10.));
  const Real wt = .5 * Kokkos::tanh((zi - sbl_lim) / sbl_damp) + .5;
  if (maxqke <= 0.05) {
  } else {
    zi = PBLH_TKE * (1. - wt) + zi * wt;
  }
  for (k = 2; k <= nk - 1; ++k) {
    if (zw(k) >= zi) {
      kzi = k - 1;
      break;
    }
  }
}

KOKKOS_INLINE_FUNCTION void scale_aware(Real dx, Real PBL1, Real& Psig_bl, Real& Psig_shcu) {
  Real dxdh = fmax2(2.5 * dx, 10.) / fmin2(PBL1, 3000.);
  Psig_bl = ((dxdh * dxdh) + 0.106 * Kokkos::pow(dxdh, 0.667)) /
            ((dxdh * dxdh) + 0.066 * Kokkos::pow(dxdh, 0.667) + 0.071);
  dxdh = fmax2(2.5 * dx, 10.) / fmin2(PBL1 + 500., 3500.);
  Psig_shcu = ((dxdh * dxdh) + 0.145 * Kokkos::pow(dxdh, 0.667)) /
              ((dxdh * dxdh) + 0.172 * Kokkos::pow(dxdh, 0.667) + 0.170);
  if (Psig_bl > 1.0) Psig_bl = 1.0;
  if (Psig_bl < 0.0) Psig_bl = 0.0;
  if (Psig_shcu > 1.0) Psig_shcu = 1.0;
  if (Psig_shcu < 0.0) Psig_shcu = 0.0;
}

// mym_level2 (ql, thetav, thlsg, qwsg are not used by it)
KOKKOS_INLINE_FUNCTION void mym_level2(int nk, Col dz, Col u, Col v, Col thl, Col qw, Col vt, Col vq, Col dtl,
                                       Col dqw, Col dtv, Col gm, Col gh, Col sm, Col sh) {
  for (int k = 2; k <= nk; ++k) {
    const Real dzk = 0.5 * (dz(k) + dz(k - 1));
    const Real afk = dz(k) / (dz(k) + dz(k - 1));
    const Real abk = 1.0 - afk;
    const Real du = u(k) - u(k - 1), dv = v(k) - v(k - 1);
    Real duz = du * du + dv * dv;
    duz = duz / (dzk * dzk);
    const Real dtz = (thl(k) - thl(k - 1)) / (dzk);
    const Real dqz = (qw(k) - qw(k - 1)) / (dzk);
    const Real vtt = 1.0 + vt(k) * abk + vt(k - 1) * afk;
    const Real vqq = tv0 + vq(k) * abk + vq(k - 1) * afk;
    const Real dtq = vtt * dtz + vqq * dqz;
    dtl(k) = dtz;
    dqw(k) = dqz;
    dtv(k) = dtq;
    gm(k) = duz;
    gh(k) = -dtq * gtr;
    const Real ri = -gh(k) / fmax2(duz, 1.0e-10);
    const Real a2fac = 1. / (1. + fmax2(ri, 0.0));   // CKmod = 1
    const Real rfc = g1 / (g1 + g2);
    const Real f1 = b1 * (g1 - c1) + 3.0 * a2 * a2fac * (1.0 - c2) * (1.0 - c5) + 2.0 * a1 * (3.0 - 2.0 * c2);
    const Real f2 = b1 * (g1 + g2) - 3.0 * a1 * (1.0 - c2);
    const Real rf1 = b1 * (g1 - c1) / f1;
    const Real rf2 = b1 * g1 / f2;
    const Real smc = a1 / (a2 * a2fac) * f1 / f2;
    const Real shc = 3.0 * (a2 * a2fac) * (g1 + g2);
    const Real ri1 = 0.5 / smc;
    const Real ri2 = rf1 * smc;
    const Real ri3 = 4.0 * rf2 * smc - 2.0 * ri2;
    const Real ri4 = ri2 * ri2;
    const Real rf = fmin2(ri1 * (ri + ri2 - Kokkos::sqrt(ri * ri - ri3 * ri + ri4)), rfc);
    sh(k) = shc * (rfc - rf) / (1.0 - rf);
    sm(k) = smc * (rf1 - rf) / (rf2 - rf) * sh(k);
  }
}

// boulac_length: only lb2 (elBLavg) is used by mym_length
KOKKOS_INLINE_FUNCTION void boulac_length(int nk, Col zw, Col dz, Col qtke, Col theta, Col lb2) {
  constexpr Real Lmax = 2000.;
  for (int iz = 1; iz <= nk; ++iz) {
    Real zup = 0.;
    Real dlu = zw(nk + 1) - zw(iz) - dz(iz) * 0.5;
    Real zzz = 0.;
    Real zup_inf = 0.;
    const Real beta = gtr;
    Real dzt, bbb, tl;
    if (iz < nk) {
      int found = 0;
      int izz = iz;
      while (found == 0) {
        if (izz < nk) {
          dzt = dz(izz);
          zup = zup - beta * theta(iz) * dzt;
          zup = zup + beta * (theta(izz + 1) + theta(izz)) * dzt * 0.5;
          zzz = zzz + dzt;
          if (qtke(iz) < zup && qtke(iz) >= zup_inf) {
            bbb = (theta(izz + 1) - theta(izz)) / dzt;
            if (bbb != 0.) {
              const Real x = beta * (theta(izz) - theta(iz));
              tl = (-beta * (theta(izz) - theta(iz)) +
                    Kokkos::sqrt(fmax2(0., x * x + 2. * bbb * beta * (qtke(iz) - zup_inf)))) /
                   bbb / beta;
            } else {
              if (theta(izz) != theta(iz)) {
                tl = (qtke(iz) - zup_inf) / (beta * (theta(izz) - theta(iz)));
              } else {
                tl = 0.;
              }
            }
            dlu = zzz - dzt + tl;
            found = 1;
          }
          zup_inf = zup;
          izz = izz + 1;
        } else {
          found = 1;
        }
      }
    }
    Real zdo = 0.;
    Real zdo_sup = 0.;
    Real dld = zw(iz);
    zzz = 0.;
    if (iz > 1) {
      int found = 0;
      int izz = iz;
      while (found == 0) {
        if (izz > 1) {
          dzt = dz(izz - 1);
          zdo = zdo + beta * theta(iz) * dzt;
          zdo = zdo - beta * (theta(izz - 1) + theta(izz)) * dzt * 0.5;
          zzz = zzz + dzt;
          if (qtke(iz) < zdo && qtke(iz) >= zdo_sup) {
            bbb = (theta(izz) - theta(izz - 1)) / dzt;
            if (bbb != 0.) {
              const Real x = beta * (theta(izz) - theta(iz));
              tl = (beta * (theta(izz) - theta(iz)) +
                    Kokkos::sqrt(fmax2(0., x * x + 2. * bbb * beta * (qtke(iz) - zdo_sup)))) /
                   bbb / beta;
            } else {
              if (theta(izz) != theta(iz)) {
                tl = (qtke(iz) - zdo_sup) / (beta * (theta(izz) - theta(iz)));
              } else {
                tl = 0.;
              }
            }
            dld = zzz - dzt + tl;
            found = 1;
          }
          zdo_sup = zdo;
          izz = izz - 1;
        } else {
          found = 1;
        }
      }
    }
    dld = fmin2(dld, zw(iz + 1));
    dlu = fmax2(0.1, fmin2(dlu, 1000.));
    dld = fmax2(0.1, fmin2(dld, 1000.));
    lb2(iz) = Kokkos::sqrt(dlu * dld);
    lb2(iz) = lb2(iz) / (1. + (lb2(iz) / Lmax));
    if (iz == nk) lb2(nk) = lb2(nk - 1);
  }
}

// mym_length, bl_mynn_mixlength = 1
KOKKOS_INLINE_FUNCTION void mym_length(Ws ws, int nk, Col dz, Col zw, Real rmo, Real flt, Col u1, Col v1,
                                       Col qke, Col dtv, Col el, Real zi, Col theta, Col qkw, Real Psig_bl,
                                       Col edmf_w1, Col edmf_a1) {
  Col qtke = ws.get(), elBLavg = ws.get(), thetaw = ws.get();
  const Real ugrid = Kokkos::sqrt(u1(1) * u1(1) + v1(1) * v1(1));
  const Real uonset = 15.;
  const Real wt_u = (1.0 - fmin2(fmax2(ugrid - uonset, 0.0) / 30.0, 0.5));
  const Real cns = 3.5, alp1 = 0.22, alp2 = 0.25, alp3 = 2.0 * wt_u, alp4 = 5.0, alp5 = 0.3, alp6 = 50.;
  const Real zi2 = fmax2(zi, 200.);
  Real h1 = fmax2(0.3 * zi2, 200.);
  h1 = fmin2(h1, 500.);
  const Real h2 = h1 / 2.0;
  qtke(1) = fmax2(0.5 * qke(1), 0.01);
  thetaw(1) = theta(1);
  qkw(1) = Kokkos::sqrt(fmax2(qke(1), 1.0e-10));
  for (int k = 2; k <= nk; ++k) {
    const Real afk = dz(k) / (dz(k) + dz(k - 1));
    const Real abk = 1.0 - afk;
    qkw(k) = Kokkos::sqrt(fmax2(qke(k) * abk + qke(k - 1) * afk, 1.0e-3));
    qtke(k) = 0.5 * (qkw(k) * qkw(k));
    thetaw(k) = theta(k) * abk + theta(k - 1) * afk;
  }
  Real elt = 1.0e-5;
  Real vsc = 1.0e-5;
  int k = 2;
  Real zwk = zw(k);
  while (zwk <= zi2 + h1 && k <= nk) {
    const Real dzk = 0.5 * (dz(k) + dz(k - 1));
    const Real qdz = fmin2(fmax2(qkw(k) - qmin, 0.03), 30.0) * dzk;
    elt = elt + qdz * zwk;
    vsc = vsc + qdz;
    k = k + 1;
    zwk = zw(k);
  }
  elt = fmin2(fmax2(alp1 * elt / vsc, 10.), 400.);
  const Real vflx = flt;
  vsc = Kokkos::pow(gtr * elt * fmax2(vflx, 0.0), onethird);
  el(1) = 0.0;
  boulac_length(nk, zw, dz, qtke, thetaw, elBLavg);
  for (k = 2; k <= nk; ++k) {
    zwk = zw(k);
    Real elb, elf;
    if (dtv(k) > 0.0) {
      const Real bv = fmax2(Kokkos::sqrt(gtr * dtv(k)), 0.001);
      elb = fmax2(alp2 * qkw(k), alp6 * edmf_a1(k - 1) * edmf_w1(k - 1)) / bv *
            (1.0 + alp3 * Kokkos::sqrt(vsc / (bv * elt)));
      elb = fmin2(elb, zwk);
      elf = 0.65 * qkw(k) / bv;
      elBLavg(k) = fmax2(elBLavg(k), alp6 * edmf_a1(k - 1) * edmf_w1(k - 1) / bv);
    } else {
      elb = 1.0e10;
      elf = elb;
    }
    Real els;
    if (rmo > 0.0) {
      els = karman * zwk / (1.0 + cns * fmin2(zwk * rmo, zmax));
    } else {
      els = karman * zwk * Kokkos::pow(1.0 - alp4 * zwk * rmo, 0.2);
    }
    const Real wt = .5 * Kokkos::tanh((zwk - (zi2 + h1)) / h2) + .5;
    el(k) = Kokkos::sqrt((els * els) / (1. + ((els * els) / (elt * elt)) + ((els * els) / (elb * elb))));
    el(k) = fmin2(el(k), elf);
    el(k) = el(k) * (1. - wt) + alp5 * elBLavg(k) * wt;
    el(k) = el(k) * Psig_bl;
  }
}

// mym_turbulence (closure 2.5 / 3.0; spp_pbl = 0, TKEprodTD = 0)
KOKKOS_INLINE_FUNCTION void mym_turbulence(Ws ws, int nk, Real closure, Col dz, Col zw, Col u, Col v, Col thl,
                                           Col qw, Col qke, Col tsq, Col qsq, Col cov, Col vt, Col vq, Real rmo,
                                           Real flt, Real zi, Col theta, Col sh, Col sm, Col el, Col dfm,
                                           Col dfh, Col dfq, Col tcd, Col qcd, Col pdk, Col pdt, Col pdq, Col pdc,
                                           Col qSHEAR1D, Col qBUOY1D, Real Psig_bl, Col cldfra_bl1D, Col edmf_w1,
                                           Col edmf_a1) {
  Col qkw = ws.get(), dtl = ws.get(), dqw = ws.get(), dtv = ws.get(), gm = ws.get(), gh = ws.get();
  constexpr Real Prlimit = 5.0;
  mym_level2(nk, dz, u, v, thl, qw, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
  mym_length(ws, nk, dz, zw, rmo, flt, u, v, qke, dtv, el, zi, theta, qkw, Psig_bl, edmf_w1, edmf_a1);
  for (int k = 2; k <= nk; ++k) {
    const Real dzk = 0.5 * (dz(k) + dz(k - 1));
    const Real afk = dz(k) / (dz(k) + dz(k - 1));
    const Real abk = 1.0 - afk;
    const Real elsq = el(k) * el(k);
    Real q3sq = qkw(k) * qkw(k);
    const Real q2sq = b1 * elsq * (sm(k) * gm(k) + sh(k) * gh(k));
    sh(k) = fmax2(sh(k), 1e-5);
    const Real du = u(k) - u(k - 1), dv = v(k) - v(k - 1);
    Real duz = du * du + dv * dv;
    duz = duz / (dzk * dzk);
    const Real ri = -gh(k) / fmax2(duz, 1.0e-10);
    const Real a2fac = 1. / (1. + fmax2(ri, 0.0));   // CKmod = 1
    const Real gmel = gm(k) * elsq;
    const Real ghel = gh(k) * elsq;
    Real dlsq = elsq;
    if (q3sq / dlsq < -gh(k)) q3sq = -dlsq * gh(k);
    Real qdiv, e1, e2, e3, e4, eden;
    if (q3sq < q2sq) {
      qdiv = Kokkos::sqrt(q3sq / q2sq);
      sh(k) = sh(k) * qdiv;
      sm(k) = sm(k) * qdiv;
      e1 = q3sq - e1c * ghel * a2fac * (qdiv * qdiv);
      e2 = q3sq - e2c * ghel * a2fac * (qdiv * qdiv);
      e3 = e1 + e3c * ghel * (a2fac * a2fac) * (qdiv * qdiv);
      e4 = e1 - e4c * ghel * a2fac * (qdiv * qdiv);
      eden = e2 * e4 + e3 * e5c * gmel * (qdiv * qdiv);
      eden = fmax2(eden, 1.0e-20);
    } else {
      e1 = q3sq - e1c * ghel * a2fac;
      e2 = q3sq - e2c * ghel * a2fac;
      e3 = e1 + e3c * ghel * (a2fac * a2fac);
      e4 = e1 - e4c * ghel * a2fac;
      eden = e2 * e4 + e3 * e5c * gmel;
      eden = fmax2(eden, 1.0e-20);
      qdiv = 1.0;
      sm(k) = q3sq * a1 * (e3 - 3.0 * c1 * e4) / eden;
      sh(k) = q3sq * (a2 * a2fac) * (e2 + 3.0 * c1 * e5c * gmel) / eden;
    }
    constexpr Real sm25max = 4., sh25max = 4., sh25min = 0.0;
    if (sh(k) > sh25max) sh(k) = sh25max;
    if (sh(k) < sh25min) sh(k) = sh25min;
    sm(k) = fmin2(sm(k), Prlimit * sh(k));
    (void)sm25max;
    Real gamt, gamq, gamv;
    if (closure >= 3.0) {
      Real t2sq = qdiv * b2 * elsq * sh(k) * (dtl(k) * dtl(k));
      Real r2sq = qdiv * b2 * elsq * sh(k) * (dqw(k) * dqw(k));
      Real c2sq = qdiv * b2 * elsq * sh(k) * dtl(k) * dqw(k);
      Real t3sq = fmax2(tsq(k) * abk + tsq(k - 1) * afk, 0.0);
      Real r3sq = fmax2(qsq(k) * abk + qsq(k - 1) * afk, 0.0);
      Real c3sq = cov(k) * abk + cov(k - 1) * afk;
      c3sq = Kokkos::copysign(fmin2(Kokkos::fabs(c3sq), Kokkos::sqrt(t3sq * r3sq)), c3sq);
      const Real vtt = 1.0 + vt(k) * abk + vt(k - 1) * afk;
      const Real vqq = tv0 + vq(k) * abk + vq(k - 1) * afk;
      t2sq = vtt * t2sq + vqq * c2sq;
      r2sq = vtt * c2sq + vqq * r2sq;
      c2sq = fmax2(vtt * t2sq + vqq * r2sq, 0.0);
      t3sq = vtt * t3sq + vqq * c3sq;
      r3sq = vtt * c3sq + vqq * r3sq;
      c3sq = fmax2(vtt * t3sq + vqq * r3sq, 0.0);
      const Real cw25 = e1 * (e2 + 3.0 * c1 * e5c * gmel * (qdiv * qdiv)) / (3.0 * eden);
      dlsq = elsq;
      if (q3sq / dlsq < -gh(k)) q3sq = -dlsq * gh(k);
      e2 = q3sq - e2c * ghel * a2fac * (qdiv * qdiv);
      e3 = q3sq + e3c * ghel * (a2fac * a2fac) * (qdiv * qdiv);
      e4 = q3sq - e4c * ghel * a2fac * (qdiv * qdiv);
      eden = e2 * e4 + e3 * e5c * gmel * (qdiv * qdiv);
      constexpr Real cc3gtr2 = cc3 * (gtr * gtr);
      const Real wden = cc3gtr2 * (dlsq * dlsq) / elsq * (qdiv * qdiv) *
                        (e2 * e4c * a2fac - e3c * e5c * gmel * (a2fac * a2fac) * (qdiv * qdiv));
      if (wden != 0.0) {
        const Real clow = q3sq * (0.12 - cw25) * eden / wden;
        const Real cupp = q3sq * (0.76 - cw25) * eden / wden;
        if (wden > 0.0) {
          c3sq = fmin2(fmax2(c3sq, c2sq + clow), c2sq + cupp);
        } else {
          c3sq = fmax2(fmin2(c3sq, c2sq + clow), c2sq + cupp);
        }
      }
      e1 = e2 + e5c * gmel * (qdiv * qdiv);
      eden = fmax2(eden, 1.0e-20);
      const Real e6c = 3.0 * (a2 * a2fac) * cc3 * gtr * dlsq / elsq;
      Real enumr;
      if (t2sq >= 0.0) {
        enumr = fmax2(qdiv * e6c * (t3sq - t2sq), 0.0);
      } else {
        enumr = fmin2(qdiv * e6c * (t3sq - t2sq), 0.0);
      }
      gamt = -e1 * enumr / eden;
      if (r2sq >= 0.0) {
        enumr = fmax2(qdiv * e6c * (r3sq - r2sq), 0.0);
      } else {
        enumr = fmin2(qdiv * e6c * (r3sq - r2sq), 0.0);
      }
      gamq = -e1 * enumr / eden;
      enumr = fmax2(qdiv * e6c * (c3sq - c2sq), 0.0);
      const Real smd = dlsq * enumr * gtr / eden * (qdiv * qdiv) * (e3c * (a2fac * a2fac) + e4c * a2fac) * a1 /
                       (a2 * a2fac);
      gamv = e1 * enumr * gtr / eden;
      sm(k) = sm(k) + smd;
      qdiv = 1.0;
    } else {
      gamt = 0.0;
      gamq = 0.0;
      gamv = 0.0;
    }
    const Real cldavg = 0.5 * (cldfra_bl1D(k - 1) + cldfra_bl1D(k));
    if (edmf_a1(k) > 0.001 || cldavg > 0.02) {
      sm(k) = fmax2(sm(k), 0.03 * fmin2(10. * edmf_a1(k) * edmf_w1(k), 1.0));
      sh(k) = fmax2(sh(k), 0.03 * fmin2(10. * edmf_a1(k) * edmf_w1(k), 1.0));
      sm(k) = fmax2(sm(k), 0.05 * fmin2(cldavg, 1.0));
      sh(k) = fmax2(sh(k), 0.05 * fmin2(cldavg, 1.0));
    }
    const Real elq = el(k) * qkw(k);
    const Real elh = elq * qdiv;
    pdk(k) = elq * (sm(k) * gm(k) + sh(k) * gh(k) + gamv) + ZERO;
    pdt(k) = elh * (sh(k) * dtl(k) + gamt) * dtl(k);
    pdq(k) = elh * (sh(k) * dqw(k) + gamq) * dqw(k);
    pdc(k) = elh * (sh(k) * dtl(k) + gamt) * dqw(k) * 0.5 + elh * (sh(k) * dqw(k) + gamq) * dtl(k) * 0.5;
    tcd(k) = elq * gamt;
    qcd(k) = elq * gamq;
    dfm(k) = elq * sm(k) / dzk;
    dfh(k) = elq * sh(k) / dzk;
    dfq(k) = dfm(k);
    // tke_budget = 1
    qSHEAR1D(k) = elq * sm(k) * gm(k);
    qBUOY1D(k) = elq * (sh(k) * gh(k) + gamv) + ZERO;
  }
  dfm(1) = 0.0;
  dfh(1) = 0.0;
  dfq(1) = 0.0;
  tcd(1) = 0.0;
  qcd(1) = 0.0;
  tcd(nk) = 0.0;
  qcd(nk) = 0.0;
  for (int k = 1; k <= nk - 1; ++k) {
    const Real dzk = dz(k);
    tcd(k) = (tcd(k + 1) - tcd(k)) / (dzk);
    qcd(k) = (qcd(k + 1) - qcd(k)) / (dzk);
  }
}

// mym_predict (bl_mynn_edmf_tke = 0: s_awqke = 0, onoff = 0; tke_budget = 1)
KOKKOS_INLINE_FUNCTION void mym_predict(Ws ws, int nk, Real closure, Real delt, Col dz, Real ust, Real pmz,
                                        Real phh, Col el, Col dfq, Col rho, Col pdk, Col pdt, Col pdq, Col pdc,
                                        Col qke, Col tsq, Col qsq, Col cov, Col s_aw, Col qWT1D, Col qDISS1D) {
  Col qkw = ws.get(), df3q = ws.get(), dtz = ws.get(), rhoz = ws.get(), rhoinv = ws.get(), kqdz = ws.get(),
      kmdz = ws.get(), bp = ws.get(), rp = ws.get(), a = ws.get(), b = ws.get(), c = ws.get(), d = ws.get(),
      x = ws.get(), tke_up = ws.get();
  const Real onoff = 0.0;
  const Real s_awqke = 0.0;   // s_awqke(k) for every k
  const Real vkz = karman * 0.5 * dz(1);
  for (int k = 1; k <= nk; ++k) {
    qkw(k) = Kokkos::sqrt(fmax2(qke(k), 0.0));
    df3q(k) = Sqfac * dfq(k);
    dtz(k) = delt / dz(k);
  }
  rhoz(1) = rho(1);
  rhoinv(1) = 1. / rho(1);
  kqdz(1) = rhoz(1) * df3q(1);
  kmdz(1) = rhoz(1) * dfq(1);
  for (int k = 2; k <= nk; ++k) {
    rhoz(k) = (rho(k) * dz(k - 1) + rho(k - 1) * dz(k)) / (dz(k - 1) + dz(k));
    rhoz(k) = fmax2(rhoz(k), 1E-4);
    rhoinv(k) = 1. / fmax2(rho(k), 1E-4);
    kqdz(k) = rhoz(k) * df3q(k);
    kmdz(k) = rhoz(k) * dfq(k);
  }
  rhoz(nk + 1) = rhoz(nk);
  kqdz(nk + 1) = rhoz(nk + 1) * df3q(nk);
  kmdz(nk + 1) = rhoz(nk + 1) * dfq(nk);
  for (int k = 2; k <= nk - 1; ++k) {
    kqdz(k) = fmax2(kqdz(k), 0.5 * s_aw(k));
    kqdz(k) = fmax2(kqdz(k), -0.5 * (s_aw(k) - s_aw(k + 1)));
    kmdz(k) = fmax2(kmdz(k), 0.5 * s_aw(k));
    kmdz(k) = fmax2(kmdz(k), -0.5 * (s_aw(k) - s_aw(k + 1)));
  }
  const Real pdk1 = 2.0 * ((ust * ust) * ust) * pmz / (vkz);
  pdk(1) = pdk1 - pdk(2);
  pdt(1) = pdt(2);
  pdq(1) = pdq(2);
  pdc(1) = pdc(2);
  (void)phh;
  for (int k = 1; k <= nk - 1; ++k) {
    const Real b1l = b1 * 0.5 * (el(k + 1) + el(k));
    bp(k) = 2. * qkw(k) / b1l;
    rp(k) = pdk(k + 1) + pdk(k);
  }
  for (int k = 1; k <= nk - 1; ++k) {
    a(k) = -dtz(k) * kqdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) * onoff;
    b(k) = 1. + dtz(k) * (kqdz(k) + kqdz(k + 1)) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) * onoff +
           bp(k) * delt;
    c(k) = -dtz(k) * kqdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) * onoff;
    d(k) = rp(k) * delt + qke(k) + dtz(k) * rhoinv(k) * (s_awqke - s_awqke) * onoff;
  }
  a(nk) = 0.;
  b(nk) = 1.;
  c(nk) = 0.;
  d(nk) = qke(nk);
  tridiag2(ws, nk, a, b, c, d, x);
  for (int k = 1; k <= nk; ++k) {
    qke(k) = fmax2(x(k), 1.e-4);
    qke(k) = fmin2(qke(k), 150.);
  }
  // tke_budget = 1
  {
    for (int k = 1; k <= nk; ++k) tke_up(k) = 0.5 * qke(k);
    int k = 1;
    qWT1D(k) = (1. / dz(k)) *
               ((kqdz(k + 1) * (tke_up(k + 1) - tke_up(k)) - kqdz(k) * tke_up(k)) +
                0.5 * rhoinv(k) *
                    (s_aw(k + 1) * tke_up(k + 1) + (s_aw(k + 1) - s_aw(k)) * tke_up(k) + (s_awqke - s_awqke)) * onoff);
    for (k = 2; k <= nk - 1; ++k) {
      qWT1D(k) = (1. / dz(k)) * ((kqdz(k + 1) * (tke_up(k + 1) - tke_up(k)) - kqdz(k) * (tke_up(k) - tke_up(k - 1))) +
                                 0.5 * rhoinv(k) *
                                     (s_aw(k + 1) * tke_up(k + 1) + (s_aw(k + 1) - s_aw(k)) * tke_up(k) -
                                      s_aw(k) * tke_up(k - 1) + (s_awqke - s_awqke)) *
                                     onoff);
    }
    k = nk;
    qWT1D(k) = (1. / dz(k)) * (-kqdz(k) * (tke_up(k) - tke_up(k - 1)) +
                               0.5 * rhoinv(k) * (-s_aw(k) * tke_up(k) - s_aw(k) * tke_up(k - 1) + s_awqke) * onoff);
    for (k = 1; k <= nk - 1; ++k) qDISS1D(k) = bp(k) * tke_up(k);   // (k = nk is undefined in the Fortran)
  }
  if (closure > 2.5) {
    for (int k = 1; k <= nk - 1; ++k) {
      const Real b2l = b2 * 0.5 * (el(k + 1) + el(k));
      bp(k) = 2. * qkw(k) / b2l;
      rp(k) = pdq(k + 1) + pdq(k);
    }
    for (int k = 1; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * kmdz(k) * rhoinv(k);
      b(k) = 1. + dtz(k) * (kmdz(k) + kmdz(k + 1)) * rhoinv(k) + bp(k) * delt;
      c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k);
      d(k) = rp(k) * delt + qsq(k);
    }
    a(nk) = -1.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = 0.;
    tridiag2(ws, nk, a, b, c, d, x);
    for (int k = 1; k <= nk; ++k) qsq(k) = fmax2(x(k), 1e-17);
  } else {
    for (int k = 1; k <= nk - 1; ++k) {
      Real b2l;
      if (qkw(k) <= 0.0) {
        b2l = 0.0;
      } else {
        b2l = b2 * 0.25 * (el(k + 1) + el(k)) / qkw(k);
      }
      qsq(k) = b2l * (pdq(k + 1) + pdq(k));
    }
    qsq(nk) = qsq(nk - 1);
  }
  if (closure >= 3.0) {
    for (int k = 1; k <= nk - 1; ++k) {
      const Real b2l = b2 * 0.5 * (el(k + 1) + el(k));
      bp(k) = 2. * qkw(k) / b2l;
      rp(k) = pdt(k + 1) + pdt(k);
    }
    for (int k = 1; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * kmdz(k) * rhoinv(k);
      b(k) = 1. + dtz(k) * (kmdz(k) + kmdz(k + 1)) * rhoinv(k) + bp(k) * delt;
      c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k);
      d(k) = rp(k) * delt + tsq(k);
    }
    a(nk) = -1.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = 0.;
    tridiag2(ws, nk, a, b, c, d, x);
    for (int k = 1; k <= nk; ++k) tsq(k) = x(k);
    for (int k = 1; k <= nk - 1; ++k) {
      const Real b2l = b2 * 0.5 * (el(k + 1) + el(k));
      bp(k) = 2. * qkw(k) / b2l;
      rp(k) = pdc(k + 1) + pdc(k);
    }
    for (int k = 1; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * kmdz(k) * rhoinv(k);
      b(k) = 1. + dtz(k) * (kmdz(k) + kmdz(k + 1)) * rhoinv(k) + bp(k) * delt;
      c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k);
      d(k) = rp(k) * delt + cov(k);
    }
    a(nk) = -1.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = 0.;
    tridiag2(ws, nk, a, b, c, d, x);
    for (int k = 1; k <= nk; ++k) cov(k) = x(k);
  } else {
    for (int k = 1; k <= nk - 1; ++k) {
      Real b2l;
      if (qkw(k) <= 0.0) {
        b2l = 0.0;
      } else {
        b2l = b2 * 0.25 * (el(k + 1) + el(k)) / qkw(k);
      }
      tsq(k) = b2l * (pdt(k + 1) + pdt(k));
      cov(k) = b2l * (pdc(k + 1) + pdc(k));
    }
    tsq(nk) = tsq(nk - 1);
    cov(nk) = cov(nk - 1);
  }
}

// mym_condensation, bl_mynn_cloudpdf = 2 (spp_pbl = 0)
KOKKOS_INLINE_FUNCTION void mym_condensation(int nk, Col dz, Col qw, Col qc, Col p, Col exner, Col qsq,
                                             Col qc_bl1D, Col qi_bl1D, Col cldfra_bl1D, Col vt, Col vq, Col th,
                                             Col sgm) {
  int k;
  for (k = nk - 3; k >= 1; --k) {
    const Real theta1 = th(k);
    const Real theta2 = th(k + 2);
    const Real ht1 = 44307.692 * (1.0 - Kokkos::pow(p(k) / 101325., 0.190));
    const Real ht2 = 44307.692 * (1.0 - Kokkos::pow(p(k + 2) / 101325., 0.190));
    if ((((theta2 - theta1) / (ht2 - ht1)) < 10. / 1500.) && (ht1 < 19000.) && (ht1 > 4000.)) break;
  }
  const int k_tropo = k + 2 > 3 ? k + 2 : 3;   // MAX(kts+2, k+2)
  Real zagl = 0.;
  for (k = 1; k <= nk - 1; ++k) {
    zagl = zagl + dz(k);
    const Real t = th(k) * exner(k);
    const Real xl = xl_blend(t);
    const Real qsat_tk = qsat_blend(t, p(k));
    const Real rh = fmax2(fmin2(1.0, qw(k) / fmax2(1.E-8, qsat_tk)), 0.001);
    const Real rsl = xl * qsat_tk / (r_v * (t * t));
    const Real cpm = cp + qw(k) * cpv;
    const Real a = 1. / (1. + xl * rsl / cpm);
    const Real b = a * rsl;
    const Real qw_pert = qw(k) + qw(k) * 0.5 * ZERO * Real(0);   // rstoch_col*real(spp_pbl)
    const Real qmq = qw_pert - qsat_tk;
    const Real r3sq = fmax2(qsq(k), 0.0);
    sgm(k) = Kokkos::sqrt(r3sq);
    sgm(k) = fmin2(sgm(k), qsat_tk * 0.666);
    sgm(k) = fmax2(sgm(k), qsat_tk * 0.035);
    const Real q1 = qmq / sgm(k);
    Real q1k = q1;
    cldfra_bl1D(k) = fmax2(0., fmin2(1., 0.5 + 0.36 * Kokkos::atan(1.8 * (q1 + 0.2))));
    Real ql_water, ql_ice;
    if (q1k < 0.) {
      ql_water = sgm(k) * Kokkos::exp(1.2 * q1k - 1);
      ql_ice = sgm(k) * Kokkos::exp(1.2 * q1k - 1.);
    } else if (q1k > 2.) {
      ql_water = sgm(k) * q1k;
      ql_ice = sgm(k) * q1k;
    } else {
      ql_water = sgm(k) * (exp_m1 + 0.66 * q1k + 0.086 * (q1k * q1k));
      ql_ice = sgm(k) * (exp_m1 + 0.66 * q1k + 0.086 * (q1k * q1k));
    }
    if (qc(k) > 1.e-6) ql_water = 0.5 * (ql_water + qc(k));
    if (cldfra_bl1D(k) < 0.01) {
      ql_ice = 0.0;
      ql_water = 0.0;
      cldfra_bl1D(k) = 0.0;
    }
    const Real liq_frac = fmin2(1.0, fmax2(0.0, (t - tice) / (tliq - tice)));
    qc_bl1D(k) = liq_frac * ql_water;
    qi_bl1D(k) = (1.0 - liq_frac) * ql_ice;
    if (k >= k_tropo - 1) {
      cldfra_bl1D(k) = 0.;
      qc_bl1D(k) = 0.;
      qi_bl1D(k) = 0.;
    }
    q1k = fmax2(q1, -2.0);
    Real Fng;
    if (q1k >= 1.0) {
      Fng = 1.0;
    } else if (q1k >= -1.7 && q1k < 1.0) {
      Fng = Kokkos::exp(-0.4 * (q1k - 1.0));
    } else if (q1k >= -2.5 && q1k < -1.7) {
      Fng = 3.0 + Kokkos::exp(-3.8 * (q1k + 1.7));
    } else {
      Fng = fmin2(23.9 + Kokkos::exp(-1.6 * (q1k + 2.5)), 60.);
    }
    const Real cfmax = fmin2(cldfra_bl1D(k), 0.5);
    const Real bb = b * t / th(k);
    const Real qww = 1. + 0.61 * qw(k);
    const Real alpha = 0.61 * th(k);
    const Real beta = (th(k) / t) * (xl / cp) - 1.61 * th(k);
    vt(k) = qww - cfmax * beta * bb * Fng - 1.;
    vq(k) = alpha + cfmax * beta * a * Fng - tv0;
    const Real fac_damp = fmin2(zagl * 0.0025, 1.0);
    const Real rhx = fmax2(0.0, (rh - 0.92)) / 0.145;
    const Real cld_factor = 1.0 + fac_damp * fmin2(rhx * rhx, 0.4);
    cldfra_bl1D(k) = fmin2(1., cld_factor * cldfra_bl1D(k));
  }
  vt(nk) = vt(nk - 1);
  vq(nk) = vq(nk - 1);
  qc_bl1D(nk) = 0.;
  qi_bl1D(nk) = 0.;
  cldfra_bl1D(nk) = 0.;
}

KOKKOS_INLINE_FUNCTION void condensation_edmf(Real QT, Real THL, Real P, Real zagl, Real& THV, Real& QC) {
  constexpr int niter = 50;
  constexpr Real diff = 1.e-6;
  const Real EXN = Kokkos::pow(P / p1000mb, rcp);
  Real T, QS;
  for (int i = 1; i <= niter; ++i) {
    T = EXN * THL + xlvcp * QC;
    QS = qsat_blend(T, P);
    const Real QCOLD = QC;
    QC = 0.5 * QC + 0.5 * fmax2((QT - QS), 0.);
    if (Kokkos::fabs(QC - QCOLD) < diff) break;
  }
  T = EXN * THL + xlvcp * QC;
  QS = qsat_blend(T, P);
  QC = fmax2(QT - QS, 0.);
  if (zagl < 100.) QC = 0.;
  THV = (THL + xlvcp * QC) * (1. + QT * (rvovrd - 1.) - rvovrd * QC);
}

struct MfOut {
  Col edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv;
};

// DMP_mf (env_subs = .false., tke_opt = 0, scalar_opt = 0, no chem, spp_pbl = 0)
KOKKOS_INLINE_FUNCTION void dmp_mf(Ws ws, int nk, Col zw, Col dz, Col p, Col rho, int momentum_opt, Col u, Col v,
                                   Col w, Col th, Col thl, Col thv, Col tk, Col qt, Col qc, Col exner, Col vt,
                                   Col vq, Real flt, Real fltv, Real flq, Real pblh, Real dx, Real landsea, Real ts,
                                   const MfOut& o, Col qc_bl1d, Col cldfra_bl1d, Real Psig_shcu, int& nup2,
                                   int& ktop, Real& maxmf) {
  constexpr Real Atot = 0.10, dl = 100., dcut = 1.2, cf_thresh = 0.5, fluxportion = 0.75, pgfac = 0.00;
  Col2 UPW = ws.get2(), UPTHL = ws.get2(), UPQT = ws.get2(), UPQC = ws.get2(), UPA = ws.get2(),
       UPU = ws.get2(), UPV = ws.get2(), ENT = ws.get2();
  Col edmf_th = ws.get(), dzi = ws.get();
  for (int i = 1; i <= NUP; ++i)
    for (int k = 1; k <= nk + 1; ++k) {
      UPW(k, i) = 0.;
      UPTHL(k, i) = 0.;
      UPQT(k, i) = 0.;
      UPA(k, i) = 0.;
      UPU(k, i) = 0.;
      UPV(k, i) = 0.;
      UPQC(k, i) = 0.;
      ENT(k, i) = 0.001;
    }
  for (int k = 1; k <= nk; ++k) {
    o.edmf_a(k) = 0.;
    o.edmf_w(k) = 0.;
    o.edmf_qt(k) = 0.;
    o.edmf_thl(k) = 0.;
    o.edmf_ent(k) = 0.;
    o.edmf_qc(k) = 0.;
  }
  for (int k = 1; k <= nk + 1; ++k) {
    o.s_aw(k) = 0.;
    o.s_awthl(k) = 0.;
    o.s_awqt(k) = 0.;
    o.s_awqv(k) = 0.;
    o.s_awqc(k) = 0.;
    o.s_awu(k) = 0.;
    o.s_awv(k) = 0.;
  }
  Real maxw = 0.0;
  Real cloud_base = 9000.0;
  int k50 = 1;   // always set below (zw(1) = 0)
  for (int k = 1; k <= nk - 1; ++k) {
    if (zw(k) > pblh + 500.) break;
    Real wpbl = w(k);
    if (w(k) < 0.) wpbl = 2. * w(k);
    maxw = fmax2(maxw, Kokkos::fabs(wpbl));
    if (zw(k) <= 50.) k50 = k;
    const Real qc_sgs = fmax2(qc(k), qc_bl1d(k) * cldfra_bl1d(k));
    if (qc_sgs > 1E-5 && cloud_base == 9000.0) cloud_base = 0.5 * (zw(k) + zw(k + 1));
  }
  maxw = fmax2(0., maxw - 1.0);
  Real Psig_w = fmax2(0.0, 1.0 - maxw);
  Psig_w = fmin2(Psig_w, Psig_shcu);
  Real fltv2 = fltv;
  if (Psig_w == 0.0 && fltv > 0.0) fltv2 = -1. * fltv;
  bool superadiabatic = false;
  const Real hux = ((landsea - 1.5) >= 0) ? -0.001 : -0.005;
  for (int k = 1; k <= (k50 - 1 > 1 ? k50 - 1 : 1); ++k) {
    if (k == 1) {
      if ((th(k) - ts) / (0.5 * dz(k)) < hux) {
        superadiabatic = true;
      } else {
        superadiabatic = false;
        break;
      }
    } else {
      if ((th(k) - th(k - 1)) / (0.5 * (dz(k) + dz(k - 1))) < hux) {
        superadiabatic = true;
      } else {
        superadiabatic = false;
        break;
      }
    }
  }
  {
    const int n = int(dx * dcut / dl);
    nup2 = n < NUP ? n : NUP;
    if (nup2 < 1) nup2 = 1;
  }
  Real maxwidth = 1.1 * pblh;
  maxwidth = fmin2(maxwidth, 0.5 * cloud_base);
  const Real wspd_pbl = Kokkos::sqrt(fmax2(u(1) * u(1) + v(1) * v(1), 0.01));
  Real width_flx;
  if ((landsea - 1.5) < 0) {
    width_flx = fmax2(fmin2(1000. * (0.6 * Kokkos::tanh((flt - 0.040) / 0.03) + .5), 1000.), 0.);
  } else {
    width_flx = fmax2(fmin2(1000. * (0.6 * Kokkos::tanh((flt - 0.003) / 0.01) + .5), 1000.), 0.);
  }
  maxwidth = fmin2(maxwidth, width_flx);
  {
    int n = int((maxwidth - Kokkos::fmod(maxwidth, 100.)) / 100);
    if (n < 0) n = 0;
    if (n < nup2) nup2 = n;
  }
  ktop = 0;
  maxmf = 0.0;
  if (fltv2 > 0.002 && nup2 >= 1 && superadiabatic) {
    Real cn = 0.;
    const Real d = -1.9;
    for (int I = 1; I <= NUP; ++I) {
      if (I > nup2) break;
      const Real l = dl * I;
      cn = cn + Kokkos::pow(l, d) * (l * l) / (dx * dx) * dl;
    }
    const Real C = Atot / cn;
    Real acfac;
    if ((landsea - 1.5) < 0) {
      acfac = .5 * Kokkos::tanh((fltv2 - 0.02) / 0.05) + .5;
    } else {
      acfac = .5 * Kokkos::tanh((fltv2 - 0.01) / 0.03) + .5;
    }
    const Real ac_wsp = 1.0 - fmin2(fmax2(wspd_pbl - 20.0, 0.0), 10.0) / 10.0;
    const Real ac_cld = fmin2(cloud_base / 1200., 1.0);
    acfac = acfac * fmin2(ac_wsp, ac_cld);
    for (int I = 1; I <= NUP; ++I) {
      if (I > nup2) break;
      const Real l = dl * I;
      const Real N = C * Kokkos::pow(l, d);
      UPA(1, I) = N * l * l / (dx * dx) * dl;
      UPA(1, I) = UPA(1, I) * acfac;
    }
    const Real z0 = 50., pwmin = 0.1, pwmax = 0.4;
    const Real wstar = fmax2(1.E-2, Kokkos::pow(gtr * fltv2 * pblh, onethird));
    const Real qstar = fmax2(flq, 1.0E-5) / wstar;
    const Real thstar = flt / wstar;
    const Real csigma = 1.34;
    Real exc_fac;
    if ((landsea - 1.5) >= 0) {
      exc_fac = 0.58 * 4.0 * fmin2(cloud_base / 1000., 1.0);
    } else {
      exc_fac = 0.58;
    }
    const Real sigmaW = csigma * wstar * Kokkos::pow(z0 / pblh, onethird) * (1 - 0.8 * z0 / pblh);
    const Real sigmaQT = csigma * qstar * Kokkos::pow(z0 / pblh, onethird);
    const Real sigmaTH = csigma * thstar * Kokkos::pow(z0 / pblh, onethird);
    const Real wmin = fmin2(sigmaW * pwmin, 0.1);
    const Real wmax = fmin2(sigmaW * pwmax, 0.5);
    for (int I = 1; I <= NUP; ++I) {
      if (I > nup2) break;
      UPW(1, I) = wmin + Real(I) / Real(NUP) * (wmax - wmin);
      UPU(1, I) = (u(1) * dz(2) + u(2) * dz(1)) / (dz(1) + dz(2));
      UPV(1, I) = (v(1) * dz(2) + v(2) * dz(1)) / (dz(1) + dz(2));
      UPQC(1, I) = 0.0;
      const Real exc_heat = exc_fac * UPW(1, I) * sigmaTH / sigmaW;
      UPTHL(1, I) = (thl(1) * dz(2) + thl(2) * dz(1)) / (dz(1) + dz(2)) + exc_heat;
      const Real exc_moist = exc_fac * UPW(1, I) * sigmaQT / sigmaW;
      UPQT(1, I) = (qt(1) * dz(2) + qt(2) * dz(1)) / (dz(1) + dz(2)) + exc_moist;
    }
    const Real dxsa = 1. - fmin2(fmax2((12000.0 - dx) / (12000.0 - 3000.0), 0.), 1.);
    for (int I = 1; I <= NUP; ++I) {
      if (I > nup2) break;
      Real QCn = 0.;
      const Real l = dl * I;
      for (int k = 2; k <= nk - 1; ++k) {
        const Real wminl = 0.3 + l * 0.0005;
        ENT(k, I) = 0.33 / (fmin2(fmax2(UPW(k - 1, I), wminl), 0.9) * l);
        ENT(k, I) = fmax2(ENT(k, I), 0.0003);
        if (zw(k) >= fmin2(pblh + 1500., 4000.)) {
          ENT(k, I) = ENT(k, I) + (zw(k) - fmin2(pblh + 1500., 4000.)) * 5.0E-6;
        }
        ENT(k, I) = ENT(k, I) * (1.0 - ZERO);   // (1 - rstoch_col)
        ENT(k, I) = fmin2(ENT(k, I), 0.9 / (zw(k + 1) - zw(k)));
        const Real Uk = (u(k) * dz(k + 1) + u(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
        const Real Ukm1 = (u(k - 1) * dz(k) + u(k) * dz(k - 1)) / (dz(k - 1) + dz(k));
        const Real Vk = (v(k) * dz(k + 1) + v(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
        const Real Vkm1 = (v(k - 1) * dz(k) + v(k) * dz(k - 1)) / (dz(k - 1) + dz(k));
        const Real EntExp = ENT(k, I) * (zw(k + 1) - zw(k));
        const Real EntExm = EntExp * 0.3333;
        const Real QTn = UPQT(k - 1, I) * (1. - EntExp) + qt(k) * EntExp;
        const Real THLn = UPTHL(k - 1, I) * (1. - EntExp) + thl(k) * EntExp;
        const Real Un = UPU(k - 1, I) * (1. - EntExm) + u(k) * EntExm + dxsa * pgfac * (Uk - Ukm1);
        const Real Vn = UPV(k - 1, I) * (1. - EntExm) + v(k) * EntExm + dxsa * pgfac * (Vk - Vkm1);
        const Real Pk = (p(k) * dz(k + 1) + p(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
        Real THVn;
        condensation_edmf(QTn, THLn, Pk, zw(k + 1), THVn, QCn);
        const Real THVk = (thv(k) * dz(k + 1) + thv(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
        const Real B = grav * (THVn / THVk - 1.0);
        const Real BCOEFF = (B > 0.) ? 0.15 : 0.2;
        Real Wn;
        if (UPW(k - 1, I) < 0.2) {
          Wn = UPW(k - 1, I) + (-2. * ENT(k, I) * UPW(k - 1, I) + BCOEFF * B / fmax2(UPW(k - 1, I), 0.2)) *
                                   fmin2(zw(k) - zw(k - 1), 250.);
        } else {
          Wn = UPW(k - 1, I) +
               (-2. * ENT(k, I) * UPW(k - 1, I) + BCOEFF * B / UPW(k - 1, I)) * fmin2(zw(k) - zw(k - 1), 250.);
        }
        if (Wn > UPW(k - 1, I) + fmin2(1.25 * (zw(k) - zw(k - 1)) / 200., 2.0)) {
          Wn = UPW(k - 1, I) + fmin2(1.25 * (zw(k) - zw(k - 1)) / 200., 2.0);
        }
        if (Wn < UPW(k - 1, I) - fmin2(1.25 * (zw(k) - zw(k - 1)) / 200., 2.0)) {
          Wn = UPW(k - 1, I) - fmin2(1.25 * (zw(k) - zw(k - 1)) / 200., 2.0);
        }
        Wn = fmin2(fmax2(Wn, 0.0), 3.0);
        if (k == 2 && Wn == 0.) {
          nup2 = 0;
          break;
        }
        // (the overshoot / detrainment bookkeeping only feeds env_subs)
        Wn = Wn * Kokkos::exp(-fmax2(zw(k + 1) - fmin2(pblh + 2000., 3500.), 0.0) / 1000.);
        if (Wn > 0.) {
          UPW(k, I) = Wn;
          UPTHL(k, I) = THLn;
          UPQT(k, I) = QTn;
          UPQC(k, I) = QCn;
          UPU(k, I) = Un;
          UPV(k, I) = Vn;
          UPA(k, I) = UPA(k - 1, I);
          ktop = ktop > k ? ktop : k;
        } else {
          break;
        }
      }
    }
  } else {
    nup2 = 0;
  }
  ktop = ktop < nk - 1 ? ktop : nk - 1;
  if (nup2 > 0) {
    for (int i = 1; i <= NUP; ++i) {
      if (i > nup2) break;
      for (int k = 1; k <= nk - 1; ++k) {
        if (k > ktop) break;
        const Real rho_int = (rho(k) * dz(k + 1) + rho(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
        o.s_aw(k + 1) = o.s_aw(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * Psig_w;
        o.s_awthl(k + 1) = o.s_awthl(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * UPTHL(k, i) * Psig_w;
        o.s_awqt(k + 1) = o.s_awqt(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * UPQT(k, i) * Psig_w;
        Real qc_plume;
        if (qc(k) > 1e-12 || qc(k + 1) > 1e-12) {
          qc_plume = UPQC(k, i);
        } else {
          qc_plume = 0.0;
        }
        o.s_awqc(k + 1) = o.s_awqc(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * qc_plume * Psig_w;
        if (momentum_opt > 0) {
          o.s_awu(k + 1) = o.s_awu(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * UPU(k, i) * Psig_w;
          o.s_awv(k + 1) = o.s_awv(k + 1) + rho_int * UPA(k, i) * UPW(k, i) * UPV(k, i) * Psig_w;
        }
        o.s_awqv(k + 1) = o.s_awqt(k + 1) - o.s_awqc(k + 1);
      }
    }
    Real flx1;
    if (o.s_aw(2) != 0.) {
      dzi(1) = 0.5 * (dz(1) + dz(2));
      flx1 = fmax2(o.s_aw(2) * (th(1) - th(2)) / dzi(1), 1.0e-5);
    } else {
      flx1 = 0.0;
    }
    if (flx1 > fluxportion * flt / dz(1) && flx1 > 0.0) {
      const Real adjustment = fluxportion * flt / dz(1) / flx1;
      for (int k = 1; k <= nk + 1; ++k) {
        o.s_aw(k) = o.s_aw(k) * adjustment;
        o.s_awthl(k) = o.s_awthl(k) * adjustment;
        o.s_awqt(k) = o.s_awqt(k) * adjustment;
        o.s_awqc(k) = o.s_awqc(k) * adjustment;
        o.s_awqv(k) = o.s_awqv(k) * adjustment;
        if (momentum_opt > 0) {
          o.s_awu(k) = o.s_awu(k) * adjustment;
          o.s_awv(k) = o.s_awv(k) * adjustment;
        }
      }
      for (int i = 1; i <= NUP; ++i)
        for (int k = 1; k <= nk + 1; ++k) UPA(k, i) = UPA(k, i) * adjustment;
    }
    for (int k = 1; k <= nk - 1; ++k) {
      if (k > ktop) break;
      const Real rho_int = (rho(k) * dz(k + 1) + rho(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
      for (int i = 1; i <= NUP; ++i) {
        if (i > nup2) break;
        o.edmf_a(k) = o.edmf_a(k) + UPA(k, i);
        o.edmf_w(k) = o.edmf_w(k) + rho_int * UPA(k, i) * UPW(k, i);
        o.edmf_qt(k) = o.edmf_qt(k) + rho_int * UPA(k, i) * UPQT(k, i);
        o.edmf_thl(k) = o.edmf_thl(k) + rho_int * UPA(k, i) * UPTHL(k, i);
        o.edmf_ent(k) = o.edmf_ent(k) + rho_int * UPA(k, i) * ENT(k, i);
        o.edmf_qc(k) = o.edmf_qc(k) + rho_int * UPA(k, i) * UPQC(k, i);
      }
      if (o.edmf_a(k) > 0.) {
        o.edmf_w(k) = o.edmf_w(k) / o.edmf_a(k);
        o.edmf_qt(k) = o.edmf_qt(k) / o.edmf_a(k);
        o.edmf_thl(k) = o.edmf_thl(k) / o.edmf_a(k);
        o.edmf_ent(k) = o.edmf_ent(k) / o.edmf_a(k);
        o.edmf_qc(k) = o.edmf_qc(k) / o.edmf_a(k);
        o.edmf_a(k) = o.edmf_a(k) * Psig_w;
        if (o.edmf_a(k) * o.edmf_w(k) > maxmf) maxmf = o.edmf_a(k) * o.edmf_w(k);
      }
    }
    for (int k = 1; k <= nk - 1; ++k) {
      const Real exneri = (exner(k) * dz(k + 1) + exner(k + 1) * dz(k)) / (dz(k + 1) + dz(k));
      edmf_th(k) = o.edmf_thl(k) + xlvcp / exneri * o.edmf_qc(k);
      dzi(k) = 0.5 * (dz(k) + dz(k + 1));
    }
    for (int k = 2; k <= nk - 2; ++k) {
      if (k > ktop) break;
      if (0.5 * (o.edmf_qc(k) + o.edmf_qc(k - 1)) > 0.0) {
        const Real Aup = (o.edmf_a(k) * dzi(k - 1) + o.edmf_a(k - 1) * dzi(k)) / (dzi(k - 1) + dzi(k));
        const Real QTp = (o.edmf_qt(k) * dzi(k - 1) + o.edmf_qt(k - 1) * dzi(k)) / (dzi(k - 1) + dzi(k));
        Real QCp;
        if (o.edmf_qc(k) > 0.0 && o.edmf_qc(k - 1) > 0.0) {
          QCp = (o.edmf_qc(k) * dzi(k - 1) + o.edmf_qc(k - 1) * dzi(k)) / (dzi(k - 1) + dzi(k));
        } else {
          QCp = fmax2(o.edmf_qc(k), o.edmf_qc(k - 1));
        }
        const Real xl = xl_blend(tk(k));
        const Real qsat_tk = qsat_blend(tk(k), p(k));
        const Real rsl = xl * qsat_tk / (r_v * (tk(k) * tk(k)));
        const Real cpm = cp + qt(k) * cpv;
        const Real a = 1. / (1. + xl * rsl / cpm);
        const Real b9 = a * rsl;
        const Real q2p = xlvcp / exner(k);
        const Real pt = thl(k) + q2p * QCp * Aup;
        const Real bb = b9 * tk(k) / pt;
        const Real qww = 1. + 0.61 * qt(k);
        const Real alpha = 0.61 * pt;
        const Real beta = pt * xl / (tk(k) * cp) - 1.61 * pt;
        Real sigq = 10. * Aup * (QTp - qt(k));
        sigq = fmax2(sigq, qsat_tk * 0.02);
        sigq = fmin2(sigq, qsat_tk * 0.25);
        const Real qmq = a * (qt(k) - qsat_tk);
        Real Q1 = qmq / sigq;
        Real mf_cf;
        if ((landsea - 1.5) >= 0) {
          mf_cf = fmin2(fmax2(0.5 + 0.36 * Kokkos::atan(1.55 * Q1), 0.01), 0.6);
          mf_cf = fmax2(mf_cf, 1.2 * Aup);
          mf_cf = fmin2(mf_cf, 5.0 * Aup);
        } else {
          mf_cf = fmin2(fmax2(0.5 + 0.36 * Kokkos::atan(1.55 * Q1), 0.01), 0.6);
          mf_cf = fmax2(mf_cf, 1.75 * Aup);
          mf_cf = fmin2(mf_cf, 5.0 * Aup);
        }
        if ((landsea - 1.5) >= 0) {
          if (cldfra_bl1d(k) < cf_thresh) {
            if (QCp * Aup > 5e-5) {
              qc_bl1d(k) = 1.86 * (QCp * Aup) - 2.2e-5;
            } else {
              qc_bl1d(k) = 1.18 * (QCp * Aup);
            }
            if (mf_cf >= Aup) qc_bl1d(k) = qc_bl1d(k) / mf_cf;
            cldfra_bl1d(k) = mf_cf;
          }
        } else {
          if (QCp * Aup > 5e-5) {
            qc_bl1d(k) = 1.86 * (QCp * Aup) - 2.2e-5;
          } else {
            qc_bl1d(k) = 1.18 * (QCp * Aup);
          }
          if (mf_cf >= Aup) qc_bl1d(k) = qc_bl1d(k) / mf_cf;
          cldfra_bl1d(k) = mf_cf;
        }
        if (cldfra_bl1d(k) < cf_thresh) {
          Q1 = fmax2(Q1, -2.25);
          Real Fng;
          if (Q1 >= 1.0) {
            Fng = 1.0;
          } else if (Q1 >= -1.7 && Q1 < 1.0) {
            Fng = Kokkos::exp(-0.4 * (Q1 - 1.0));
          } else if (Q1 >= -2.5 && Q1 < -1.7) {
            Fng = 3.0 + Kokkos::exp(-3.8 * (Q1 + 1.7));
          } else {
            Fng = fmin2(23.9 + Kokkos::exp(-1.6 * (Q1 + 2.5)), 60.);
          }
          vt(k) = qww - (1.5 * Aup) * beta * bb * Fng - 1.;
          vq(k) = alpha + (1.5 * Aup) * beta * a * Fng - tv0;
        }
      }
    }
  }
  if (ktop > 0) {
    Real maxqc = -Kokkos::Experimental::finite_max<Real>::value;
    for (int k = 1; k <= ktop; ++k)
      if (o.edmf_qc(k) > maxqc) maxqc = o.edmf_qc(k);
    if (maxqc < 1.E-8) maxmf = -1.0 * maxmf;
  }
}

// moisture_check (the dth it also updates is overwritten by its caller)
KOKKOS_INLINE_FUNCTION void moisture_check(int nk, Real delt, Col dp, Col exner, Col qv, Col qc, Col qi, Col th,
                                           Col dqv, Col dqc, Col dqi) {
  constexpr Real qvmin = 1e-20, qcmin = 0.0, qimin = 0.0;
  Real dqv2 = 0.0;
  for (int k = nk; k >= 1; --k) {
    const Real dqc2 = fmax2(0.0, qcmin - qc(k));
    const Real dqi2 = fmax2(0.0, qimin - qi(k));
    dqc(k) = dqc(k) + dqc2 / delt;
    dqi(k) = dqi(k) + dqi2 / delt;
    dqv(k) = dqv(k) - (dqc2 + dqi2) / delt;
    qc(k) = qc(k) + dqc2;
    qi(k) = qi(k) + dqi2;
    qv(k) = qv(k) - dqc2 - dqi2;
    th(k) = th(k) + xlvcp / exner(k) * dqc2 + xlscp / exner(k) * dqi2;
    dqv2 = fmax2(0.0, qvmin - qv(k));
    dqv(k) = dqv(k) + dqv2 / delt;
    qv(k) = qv(k) + dqv2;
    if (k != 1) {
      qv(k - 1) = qv(k - 1) - dqv2 * dp(k) / dp(k - 1);
      dqv(k - 1) = dqv(k - 1) - dqv2 * dp(k) / dp(k - 1) / delt;
    }
    qv(k) = fmax2(qv(k), qvmin);
    qc(k) = fmax2(qc(k), qcmin);
    qi(k) = fmax2(qi(k), qimin);
  }
  if (dqv2 > 1.e-20) {
    Real sum = 0.0;
    for (int k = 1; k <= nk; ++k)
      if (qv(k) > 2.0 * qvmin) sum = sum + qv(k) * dp(k);
    const Real aa = dqv2 * dp(1) / fmax2(1.e-20, sum);
    if (aa < 0.5) {
      for (int k = 1; k <= nk; ++k) {
        if (qv(k) > 2.0 * qvmin) {
          const Real dum = aa * qv(k);
          qv(k) = qv(k) - dum;
          dqv(k) = dqv(k) - dum / delt;
        }
      }
    }
  }
}

struct TendIn {
  Col dz, rho, u, v, th, tk, qv, qc, qi, p, exner, thl, sqv, sqc, sqi, tcd, qcd, dfm, dfh, diss_heat, s_aw,
      s_awthl, s_awqv, s_awqc, s_awu, s_awv;
};
struct TendOut {
  Col du, dv, dth, dqv, dqc, dqi;
};

// mynn_tendencies: bl_mynn_mixqt = 0, bl_mynn_cloudmix = 1, FLAG_QC = FLAG_QI
// = .true., no scalars, uoce = voce = 0, sd_* = sub_* = det_* = 0
KOKKOS_INLINE_FUNCTION void mynn_tendencies(Ws ws, int nk, Real delt, Real psfc, Real ust, Real flt, Real flqv,
                                            Real flqc, Real wspd, int edmf_mom, const TendIn& t,
                                            const TendOut& o) {
  Col dtz = ws.get(), rhoz = ws.get(), rhoinv = ws.get(), khdz = ws.get(), kmdz = ws.get(), delp = ws.get(),
      a = ws.get(), b = ws.get(), c = ws.get(), d = ws.get(), x = ws.get(), sqv2 = ws.get(), sqc2 = ws.get(),
      sqi2 = ws.get();
  const Col dz = t.dz, rho = t.rho, p = t.p, dfh = t.dfh, dfm = t.dfm, s_aw = t.s_aw;
  const Real uoce = 0.0, voce = 0.0;
  const Real onoff = (edmf_mom == 0) ? 0.0 : 1.0;
  const Real rhosfc = psfc / (r_d * (t.tk(1) + p608 * t.qv(1)));
  dtz(1) = delt / dz(1);
  rhoz(1) = rho(1);
  rhoinv(1) = 1. / rho(1);
  khdz(1) = rhoz(1) * dfh(1);
  kmdz(1) = rhoz(1) * dfm(1);
  delp(1) = psfc - (p(2) * dz(1) + p(1) * dz(2)) / (dz(1) + dz(2));
  for (int k = 2; k <= nk; ++k) {
    dtz(k) = delt / dz(k);
    rhoz(k) = (rho(k) * dz(k - 1) + rho(k - 1) * dz(k)) / (dz(k - 1) + dz(k));
    rhoz(k) = fmax2(rhoz(k), 1E-4);
    rhoinv(k) = 1. / fmax2(rho(k), 1E-4);
    khdz(k) = rhoz(k) * dfh(k);
    kmdz(k) = rhoz(k) * dfm(k);
  }
  for (int k = 2; k <= nk - 1; ++k) {
    delp(k) = (p(k) * dz(k - 1) + p(k - 1) * dz(k)) / (dz(k) + dz(k - 1)) -
              (p(k + 1) * dz(k) + p(k) * dz(k + 1)) / (dz(k) + dz(k + 1));
  }
  delp(nk) = delp(nk - 1);
  rhoz(nk + 1) = rhoz(nk);
  khdz(nk + 1) = rhoz(nk + 1) * dfh(nk);
  kmdz(nk + 1) = rhoz(nk + 1) * dfm(nk);
  for (int k = 2; k <= nk - 1; ++k) {
    khdz(k) = fmax2(khdz(k), 0.5 * s_aw(k));
    khdz(k) = fmax2(khdz(k), -0.5 * (s_aw(k) - s_aw(k + 1)));
    kmdz(k) = fmax2(kmdz(k), 0.5 * s_aw(k));
    kmdz(k) = fmax2(kmdz(k), -0.5 * (s_aw(k) - s_aw(k + 1)));
  }
  const Real ust2 = ust * ust;

  // u
  {
    int k = 1;
    a(k) = -dtz(k) * kmdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (kmdz(k + 1) + rhosfc * ust2 / wspd) * rhoinv(k) - 0.5 * dtz(k) * s_aw(k + 1) * onoff -
           0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
    c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * s_aw(k + 1) * onoff -
           0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
    d(k) = t.u(k) + dtz(k) * uoce * ust2 / wspd - dtz(k) * t.s_awu(k + 1) * onoff -
           dtz(k) * rhoinv(k) * ZERO * onoff + ZERO * delt + ZERO * delt;
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * kmdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) * onoff +
             0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
      b(k) = 1. + dtz(k) * (kmdz(k) + kmdz(k + 1)) * rhoinv(k) +
             0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) * onoff +
             0.5 * dtz(k) * rhoinv(k) * (ZERO - ZERO) * onoff;
      c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) * onoff -
             0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
      d(k) = t.u(k) + dtz(k) * rhoinv(k) * (t.s_awu(k) - t.s_awu(k + 1)) * onoff +
             dtz(k) * rhoinv(k) * (ZERO - ZERO) * onoff + ZERO * delt + ZERO * delt;
    }
    a(nk) = 0;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = t.u(nk);
    tridiag2(ws, nk, a, b, c, d, x);
    for (k = 1; k <= nk; ++k) o.du(k) = (x(k) - t.u(k)) / delt;
  }
  // v
  {
    int k = 1;
    a(k) = -dtz(k) * kmdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (kmdz(k + 1) + rhosfc * ust2 / wspd) * rhoinv(k) - 0.5 * dtz(k) * s_aw(k + 1) * onoff -
           0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
    c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * s_aw(k + 1) * onoff -
           0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
    d(k) = t.v(k) + dtz(k) * voce * ust2 / wspd - dtz(k) * t.s_awv(k + 1) * onoff -
           dtz(k) * rhoinv(k) * ZERO * onoff + ZERO * delt + ZERO * delt;
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * kmdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) * onoff +
             0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
      b(k) = 1. + dtz(k) * (kmdz(k) + kmdz(k + 1)) * rhoinv(k) +
             0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) * onoff +
             0.5 * dtz(k) * rhoinv(k) * (ZERO - ZERO) * onoff;
      c(k) = -dtz(k) * kmdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) * onoff -
             0.5 * dtz(k) * rhoinv(k) * ZERO * onoff;
      d(k) = t.v(k) + dtz(k) * rhoinv(k) * (t.s_awv(k) - t.s_awv(k + 1)) * onoff +
             dtz(k) * rhoinv(k) * (ZERO - ZERO) * onoff + ZERO * delt + ZERO * delt;
    }
    a(nk) = 0;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = t.v(nk);
    tridiag2(ws, nk, a, b, c, d, x);
    for (k = 1; k <= nk; ++k) o.dv(k) = (x(k) - t.v(k)) / delt;
  }
  // thl
  const Col thl = t.thl;
  {
    int k = 1;
    a(k) = -dtz(k) * khdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (khdz(k + 1) + khdz(k)) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    d(k) = thl(k) + dtz(k) * rhosfc * flt * rhoinv(k) + t.tcd(k) * delt - dtz(k) * rhoinv(k) * t.s_awthl(k + 1) -
           dtz(k) * rhoinv(k) * ZERO + t.diss_heat(k) * delt + ZERO * delt + ZERO * delt;
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * khdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) + 0.5 * dtz(k) * rhoinv(k) * ZERO;
      b(k) = 1. + dtz(k) * (khdz(k) + khdz(k + 1)) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) +
             0.5 * dtz(k) * rhoinv(k) * (ZERO - ZERO);
      c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
             0.5 * dtz(k) * rhoinv(k) * ZERO;
      d(k) = thl(k) + t.tcd(k) * delt + dtz(k) * rhoinv(k) * (t.s_awthl(k) - t.s_awthl(k + 1)) +
             dtz(k) * rhoinv(k) * (ZERO - ZERO) + t.diss_heat(k) * delt + ZERO * delt + ZERO * delt;
    }
    a(nk) = 0.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = thl(nk);
    tridiag2(ws, nk, a, b, c, d, x);
    for (k = 1; k <= nk; ++k) thl(k) = x(k);
  }
  // sqc (bl_mynn_mixqt = 0, cloudmix > 0, FLAG_QC)
  {
    int k = 1;
    a(k) = -dtz(k) * khdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (khdz(k + 1) + khdz(k)) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    d(k) = t.sqc(k) + dtz(k) * rhosfc * flqc * rhoinv(k) + t.qcd(k) * delt - dtz(k) * rhoinv(k) * t.s_awqc(k + 1) -
           dtz(k) * rhoinv(k) * ZERO + ZERO * delt;
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * khdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) + 0.5 * dtz(k) * rhoinv(k) * ZERO;
      b(k) = 1. + dtz(k) * (khdz(k) + khdz(k + 1)) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) +
             0.5 * dtz(k) * rhoinv(k) * (ZERO - ZERO);
      c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
             0.5 * dtz(k) * rhoinv(k) * ZERO;
      d(k) = t.sqc(k) + t.qcd(k) * delt + dtz(k) * rhoinv(k) * (t.s_awqc(k) - t.s_awqc(k + 1)) +
             dtz(k) * rhoinv(k) * (ZERO - ZERO) + ZERO * delt;
    }
    a(nk) = 0.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = t.sqc(nk);
    tridiag2(ws, nk, a, b, c, d, sqc2);
  }
  // sqv
  {
    int k = 1;
    Real qvflux = flqv;
    if (qvflux < 0.0) qvflux = fmax2(qvflux, (fmin2(0.9 * t.sqv(1) - 1e-8, 0.0) / dtz(1)));
    a(k) = -dtz(k) * khdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (khdz(k + 1) + khdz(k)) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
           0.5 * dtz(k) * rhoinv(k) * ZERO;
    d(k) = t.sqv(k) + dtz(k) * rhosfc * qvflux * rhoinv(k) + t.qcd(k) * delt - dtz(k) * rhoinv(k) * t.s_awqv(k + 1) -
           dtz(k) * rhoinv(k) * ZERO + ZERO * delt + ZERO * delt;
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * khdz(k) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * s_aw(k) + 0.5 * dtz(k) * rhoinv(k) * ZERO;
      b(k) = 1. + dtz(k) * (khdz(k) + khdz(k + 1)) * rhoinv(k) + 0.5 * dtz(k) * rhoinv(k) * (s_aw(k) - s_aw(k + 1)) +
             0.5 * dtz(k) * rhoinv(k) * (ZERO - ZERO);
      c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k) - 0.5 * dtz(k) * rhoinv(k) * s_aw(k + 1) -
             0.5 * dtz(k) * rhoinv(k) * ZERO;
      d(k) = t.sqv(k) + t.qcd(k) * delt + dtz(k) * rhoinv(k) * (t.s_awqv(k) - t.s_awqv(k + 1)) +
             dtz(k) * rhoinv(k) * (ZERO - ZERO) + ZERO * delt + ZERO * delt;
    }
    a(nk) = 0.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = t.sqv(nk);
    tridiag2(ws, nk, a, b, c, d, sqv2);
  }
  // sqi (cloudmix > 0, FLAG_QI)
  {
    int k = 1;
    a(k) = -dtz(k) * khdz(k) * rhoinv(k);
    b(k) = 1. + dtz(k) * (khdz(k + 1) + khdz(k)) * rhoinv(k);
    c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k);
    d(k) = t.sqi(k);
    for (k = 2; k <= nk - 1; ++k) {
      a(k) = -dtz(k) * khdz(k) * rhoinv(k);
      b(k) = 1. + dtz(k) * (khdz(k) + khdz(k + 1)) * rhoinv(k);
      c(k) = -dtz(k) * khdz(k + 1) * rhoinv(k);
      d(k) = t.sqi(k);
    }
    a(nk) = 0.;
    b(nk) = 1.;
    c(nk) = 0.;
    d(nk) = t.sqi(nk);
    tridiag2(ws, nk, a, b, c, d, sqi2);
  }
  for (int k = 1; k <= nk; ++k) o.dqv(k) = (sqv2(k) / (1. - sqv2(k)) - t.qv(k)) / delt;
  for (int k = 1; k <= nk; ++k) o.dqc(k) = (sqc2(k) / (1. - sqv2(k)) - t.qc(k)) / delt;
  for (int k = 1; k <= nk; ++k) o.dqi(k) = (sqi2(k) / (1. - sqv2(k)) - t.qi(k)) / delt;
  moisture_check(nk, delt, delp, t.exner, sqv2, sqc2, sqi2, thl, o.dqv, o.dqc, o.dqi);
  for (int k = 1; k <= nk; ++k)
    o.dth(k) = (thl(k) + xlvcp / t.exner(k) * sqc2(k) + xlscp / t.exner(k) * sqi2(k) - t.th(k)) / delt;
}

// mym_initialize (bl_mynn_mixlength = 1, INITIALIZE_QKE = .true.)
KOKKOS_INLINE_FUNCTION void mym_initialize(Ws ws, int nk, Col dz, Col zw, Col u, Col v, Col thl, Col qw, Real zi,
                                           Col theta, Col sh, Col sm, Real ust, Real rmo, Col el, Col qke, Col tsq,
                                           Col qsq, Col cov, Real Psig_bl, Col edmf_w1, Col edmf_a1) {
  Col vt = ws.get(), vq = ws.get(), pdk = ws.get(), pdt = ws.get(), pdq = ws.get(), pdc = ws.get(), dtl = ws.get(),
      dqw = ws.get(), dtv = ws.get(), gm = ws.get(), gh = ws.get(), qkw = ws.get();
  constexpr Real pmz = 1., phh = 1., flt = 0., flq = 0.;
  for (int k = 1; k <= nk; ++k) {
    vt(k) = 0.0;
    vq(k) = 0.0;
  }
  mym_level2(nk, dz, u, v, thl, qw, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
  el(1) = 0.0;
  qke(1) = 1.5 * (ust * ust) * Kokkos::pow(b1 * pmz, 2.0 / 3.0);
  for (int k = 2; k <= nk; ++k) qke(k) = qke(1) * fmax2((ust * 700. - zw(k)) / (fmax2(ust, 0.01) * 700.), 0.01);
  Real phm = phh * b2 / Kokkos::pow(b1 * pmz, 1.0 / 3.0);
  tsq(1) = phm * ((flt / ust) * (flt / ust));
  qsq(1) = phm * ((flq / ust) * (flq / ust));
  cov(1) = phm * (flt / ust) * (flq / ust);
  for (int k = 2; k <= nk; ++k) {
    const Real vkz = karman * zw(k);
    el(k) = vkz / (1.0 + vkz / 100.0);
    tsq(k) = 0.0;
    qsq(k) = 0.0;
    cov(k) = 0.0;
  }
  constexpr int lmax = 5;
  for (int l = 1; l <= lmax; ++l) {
    mym_length(ws, nk, dz, zw, rmo, flt, u, v, qke, dtv, el, zi, theta, qkw, Psig_bl, edmf_w1, edmf_a1);
    for (int k = 2; k <= nk; ++k) {
      const Real elq = el(k) * qkw(k);
      pdk(k) = elq * (sm(k) * gm(k) + sh(k) * gh(k));
      pdt(k) = elq * sh(k) * (dtl(k) * dtl(k));
      pdq(k) = elq * sh(k) * (dqw(k) * dqw(k));
      pdc(k) = elq * sh(k) * dtl(k) * dqw(k);
    }
    const Real vkz = karman * 0.5 * dz(1);
    const Real elv = 0.5 * (el(2) + el(1)) / vkz;
    const Real ustm = fmax2(ust, 0.02);
    qke(1) = 1.0 * (ustm * ustm) * Kokkos::pow(b1 * pmz * elv, 2.0 / 3.0);
    phm = phh * b2 / Kokkos::pow(b1 * pmz / (elv * elv), 1.0 / 3.0);
    tsq(1) = phm * ((flt / ust) * (flt / ust));
    qsq(1) = phm * ((flq / ust) * (flq / ust));
    cov(1) = phm * (flt / ust) * (flq / ust);
    for (int k = 2; k <= nk - 1; ++k) {
      const Real b1l = b1 * 0.25 * (el(k + 1) + el(k));
      const Real tmpq = fmin2(fmax2(b1l * (pdk(k + 1) + pdk(k)), qkemin), 125.);
      qke(k) = Kokkos::pow(tmpq, twothirds);
      Real b2l;
      if (qke(k) <= 0.0) {
        b2l = 0.0;
      } else {
        b2l = b2 * (b1l / b1) / Kokkos::sqrt(qke(k));
      }
      tsq(k) = b2l * (pdt(k + 1) + pdt(k));
      qsq(k) = b2l * (pdq(k + 1) + pdq(k));
      cov(k) = b2l * (pdc(k + 1) + pdc(k));
    }
  }
  qke(1) = 0.5 * (qke(1) + qke(2));
  qke(nk) = qke(nk - 1);
  tsq(nk) = tsq(nk - 1);
  qsq(nk) = qsq(nk - 1);
  cov(nk) = cov(nk - 1);
}

// ---------------------------------------------------------------------------
// The column kernel: mynnedmf_wrapper_run + mynn_bl_driver for one (i,j).

struct ColumnKernel {
  int ni, nk, col0, initflag, edmf, edmf_mom;
  Real delt, dx, closure;
  Inputs in;
  Surface sfc;
  State st;
  Tendencies td;
  Kokkos::View<Real**, Device> work;

  KOKKOS_FUNCTION void operator()(int n) const {
    const int c = col0 + n;
    const int i = c % ni + 1, j = c / ni + 1;
    const int L = nk + 2;
    Ws ws{&work(n, 0), int(work.stride(1)), L, 0};

    Col dz1 = ws.get(), u1 = ws.get(), v1 = ws.get(), w1 = ws.get(), th1 = ws.get(), tk1 = ws.get(),
        p1 = ws.get(), ex1 = ws.get(), rho1 = ws.get(), sqv = ws.get(), sqc = ws.get(), sqi = ws.get(),
        qv1 = ws.get(), qc1 = ws.get(), qi1 = ws.get(), sqw = ws.get(), thl = ws.get(), thetav = ws.get(),
        el = ws.get(), qke1 = ws.get(), sh = ws.get(), sm = ws.get(), tsq1 = ws.get(), qsq1 = ws.get(),
        cov1 = ws.get(), cldfra = ws.get(), qcbl = ws.get(), qibl = ws.get(), zw = ws.get(), vt = ws.get(),
        vq = ws.get(), sgm = ws.get(), dfm = ws.get(), dfh = ws.get(), dfq = ws.get(), tcd = ws.get(),
        qcd = ws.get(), pdk = ws.get(), pdt = ws.get(), pdq = ws.get(), pdc = ws.get(), qWT1 = ws.get(),
        qSHEAR1 = ws.get(), qBUOY1 = ws.get(), qDISS1 = ws.get(), du1 = ws.get(), dv1 = ws.get(),
        dth1 = ws.get(), dqv1 = ws.get(), dqc1 = ws.get(), dqi1 = ws.get(), diss_heat = ws.get();
    MfOut mf{ws.get(), ws.get(), ws.get(), ws.get(), ws.get(), ws.get(), ws.get(),
             ws.get(), ws.get(), ws.get(), ws.get(), ws.get(), ws.get()};

    const Real ust = sfc.ust(i, j, 1), xland = sfc.xland(i, j, 1);

    // mynnedmf_wrapper_run: specific humidities
    for (int k = 1; k <= nk; ++k) {
      const Real qv = in.qv(i, j, k);
      sqv(k) = qv / (1.0 + qv);
      sqc(k) = in.qc(i, j, k) / (1.0 + qv);
      sqi(k) = in.qi(i, j, k) / (1.0 + qv);
    }

    // mynn_bl_driver
    for (int k = 1; k <= nk; ++k) {
      st.edmf_a(i, j, k) = 0.;
      st.edmf_w(i, j, k) = 0.;
      st.edmf_qt(i, j, k) = 0.;
      st.edmf_thl(i, j, k) = 0.;
      st.edmf_ent(i, j, k) = 0.;
      st.edmf_qc(i, j, k) = 0.;
      st.sub_thl3d(i, j, k) = 0.;
      st.sub_sqv3d(i, j, k) = 0.;
      st.det_thl3d(i, j, k) = 0.;
      st.det_sqv3d(i, j, k) = 0.;
    }
    int ktop_plume = 0, nupdraft = 0;
    Real maxmf = 0.;
    Real pblh = sfc.hpbl(i, j, 1);
    int kpbl = int(sfc.kpbl(i, j, 1));
    Real Psig_bl, Psig_shcu;

    if (initflag > 0) {
      // (INITIALIZE_QKE = .true.: not a restart, not cycling)
      for (int k = 1; k <= nk; ++k) {
        st.sh3d(i, j, k) = 0.;
        st.sm3d(i, j, k) = 0.;
        st.el_pbl(i, j, k) = 0.;
        st.tsq(i, j, k) = 0.;
        st.qsq(i, j, k) = 0.;
        st.cov(i, j, k) = 0.;
        st.cldfra_bl(i, j, k) = 0.;
        st.qc_bl(i, j, k) = 0.;
        st.qke(i, j, k) = 0.;
      }
      for (int k = 1; k <= nk; ++k) {
        dz1(k) = in.dz(i, j, k);
        u1(k) = in.u(i, j, k);
        v1(k) = in.v(i, j, k);
        th1(k) = in.th(i, j, k);
        ex1(k) = in.exner(i, j, k);
        thetav(k) = in.th(i, j, k) * (1. + 0.608 * sqv(k));
        // FLAG_QI: thl, sqw (thlsg/sqwsg are not used)
        sqw(k) = sqv(k) + sqc(k) + sqi(k);
        thl(k) = th1(k) - xlvcp / ex1(k) * sqc(k) - xlscp / ex1(k) * sqi(k);
        if (k == 1) {
          zw(k) = 0.;
        } else {
          zw(k) = zw(k - 1) + in.dz(i, j, k - 1);
        }
        qke1(k) = 5. * ust * fmax2((ust * 700. - zw(k)) / (fmax2(ust, 0.01) * 700.), 0.01);
        el(k) = st.el_pbl(i, j, k);
        sh(k) = st.sh3d(i, j, k);
        sm(k) = st.sm3d(i, j, k);
        tsq1(k) = st.tsq(i, j, k);
        qsq1(k) = st.qsq(i, j, k);
        cov1(k) = st.cov(i, j, k);
        mf.edmf_a(k) = 0.0;
        mf.edmf_w(k) = 0.0;
      }
      zw(nk + 1) = zw(nk) + in.dz(i, j, nk);
      get_pblh(nk, pblh, thetav, qke1, zw, dz1, xland, kpbl);
      scale_aware(dx, pblh, Psig_bl, Psig_shcu);
      mym_initialize(ws, nk, dz1, zw, u1, v1, thl, sqv, pblh, th1, sh, sm, ust, sfc.rmol(i, j, 1), el, qke1, tsq1,
                     qsq1, cov1, Psig_bl, mf.edmf_w, mf.edmf_a);
      for (int k = 1; k <= nk; ++k) {
        st.el_pbl(i, j, k) = el(k);
        st.sh3d(i, j, k) = sh(k);
        st.sm3d(i, j, k) = sm(k);
        st.qke(i, j, k) = qke1(k);
        st.tsq(i, j, k) = tsq1(k);
        st.qsq(i, j, k) = qsq1(k);
        st.cov(i, j, k) = cov1(k);
        st.qke_adv(i, j, k) = qke1(k);   // bl_mynn_tkeadvect
      }
    }
    // bl_mynn_tkeadvect: qke = qke_adv
    for (int k = 1; k <= nk; ++k) st.qke(i, j, k) = st.qke_adv(i, j, k);

    for (int k = 1; k <= nk; ++k) {
      st.dqke(i, j, k) = st.qke(i, j, k);
      cldfra(k) = st.cldfra_bl(i, j, k);
      qcbl(k) = st.qc_bl(i, j, k);
      qibl(k) = st.qi_bl(i, j, k);
      dz1(k) = in.dz(i, j, k);
      u1(k) = in.u(i, j, k);
      v1(k) = in.v(i, j, k);
      w1(k) = in.w(i, j, k);
      th1(k) = in.th(i, j, k);
      tk1(k) = in.t3d(i, j, k);
      p1(k) = in.p(i, j, k);
      ex1(k) = in.exner(i, j, k);
      rho1(k) = in.rho(i, j, k);
      qv1(k) = sqv(k) / (1. - sqv(k));
      qc1(k) = sqc(k) / (1. - sqv(k));
      qi1(k) = sqi(k) / (1. - sqv(k));
      sqw(k) = sqv(k) + sqc(k) + sqi(k);
      thl(k) = th1(k) - xlvcp / ex1(k) * sqc(k) - xlscp / ex1(k) * sqi(k);
      thetav(k) = th1(k) * (1. + 0.608 * sqv(k));
      el(k) = st.el_pbl(i, j, k);
      qke1(k) = st.qke(i, j, k);
      sh(k) = st.sh3d(i, j, k);
      sm(k) = st.sm3d(i, j, k);
      tsq1(k) = st.tsq(i, j, k);
      qsq1(k) = st.qsq(i, j, k);
      cov1(k) = st.cov(i, j, k);
      mf.edmf_a(k) = 0.0;
      mf.edmf_w(k) = 0.0;
      mf.edmf_qc(k) = 0.0;
      if (k == 1) {
        zw(k) = 0.;
      } else {
        zw(k) = zw(k - 1) + in.dz(i, j, k - 1);
      }
    }
    zw(nk + 1) = zw(nk) + in.dz(i, j, nk);
    for (int k = 1; k <= nk + 1; ++k) {
      mf.s_aw(k) = 0.;
      mf.s_awthl(k) = 0.;
      mf.s_awqt(k) = 0.;
      mf.s_awqv(k) = 0.;
      mf.s_awqc(k) = 0.;
      mf.s_awu(k) = 0.;
      mf.s_awv(k) = 0.;
    }
    get_pblh(nk, pblh, thetav, qke1, zw, dz1, xland, kpbl);
    scale_aware(dx, pblh, Psig_bl, Psig_shcu);
    const Real cpm = cp * (1. + 0.84 * qv1(1));
    const Real flqv = sfc.qfx(i, j, 1) / rho1(1);
    const Real flqc = 0.0;
    const Real th_sfc = sfc.tsk(i, j, 1) / ex1(1);
    const Real flq = flqv + flqc;
    const Real flt = sfc.hfx(i, j, 1) / (rho1(1) * cpm) - xlvcp * flqc / ex1(1);
    const Real fltv = flt + flqv * p608 * th_sfc;
    const Real rmol = -karman * gtr * fltv / fmax2((ust * ust) * ust, 1.0e-6);
    sfc.rmol(i, j, 1) = rmol;
    Real zet = 0.5 * in.dz(i, j, 1) * rmol;
    zet = fmax2(zet, -20.);
    zet = fmin2(zet, 20.);
    const Real phi_m = phim(zet);
    const Real pmz = phi_m - zet;
    const Real phh = phih(zet);
    mym_condensation(nk, dz1, sqw, sqc, p1, ex1, qsq1, qcbl, qibl, cldfra, vt, vq, th1, sgm);
    if (edmf > 0) {
      dmp_mf(ws, nk, zw, dz1, p1, rho1, edmf_mom, u1, v1, w1, th1, thl, thetav, tk1, sqw, sqc, ex1, vt, vq, flt,
             fltv, flq, pblh, dx, xland, th_sfc, mf, qcbl, cldfra, Psig_shcu, nupdraft, ktop_plume, maxmf);
    }
    mym_turbulence(ws, nk, closure, dz1, zw, u1, v1, thl, sqw, qke1, tsq1, qsq1, cov1, vt, vq, rmol, flt, pblh,
                   th1, sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc, qSHEAR1, qBUOY1, Psig_bl, cldfra,
                   mf.edmf_w, mf.edmf_a);
    mym_predict(ws, nk, closure, delt, dz1, ust, pmz, phh, el, dfq, rho1, pdk, pdt, pdq, pdc, qke1, tsq1, qsq1,
                cov1, mf.s_aw, qWT1, qDISS1);
    // dheat_opt = 1
    for (int k = 1; k <= nk - 1; ++k) {
      diss_heat(k) = fmin2(
          fmax2(1.0 * (Kokkos::pow(qke1(k), 1.5)) / (b1 * fmax2(0.5 * (el(k) + el(k + 1)), 1.)) / cp, 0.0), 0.002);
      diss_heat(k) = diss_heat(k) * Kokkos::exp(-10000. / fmax2(p1(k), 1.));
    }
    diss_heat(nk) = 0.;
    for (int k = 1; k <= nk; ++k) td.dheat(i, j, k) = diss_heat(k);
    const TendIn tin{dz1,  rho1, u1,  v1,  th1, tk1, qv1, qc1,       qi1,     p1,          ex1,         thl,
                     sqv,  sqc,  sqi, tcd, qcd, dfm, dfh, diss_heat, mf.s_aw, mf.s_awthl,  mf.s_awqv,   mf.s_awqc,
                     mf.s_awu, mf.s_awv};
    const TendOut tout{du1, dv1, dth1, dqv1, dqc1, dqi1};
    mynn_tendencies(ws, nk, delt, sfc.psfc(i, j, 1), ust, flt, flqv, flqc, sfc.wspd(i, j, 1), edmf_mom, tin, tout);
    // retrieve_exchange_coeffs and the driver's / wrapper's / turb.F's copies
    for (int k = 1; k <= nk; ++k) {
      Real K_m, K_h;
      if (k == 1) {
        K_m = 0.;
        K_h = 0.;
      } else {
        const Real dzk = 0.5 * (dz1(k) + dz1(k - 1));
        K_m = dfm(k) * dzk;
        K_h = dfh(k) * dzk;
      }
      td.xkzm(i, j, k) = K_m;
      td.xkzh(i, j, k) = K_h;
      td.upten(i, j, k) = du1(k);
      td.vpten(i, j, k) = dv1(k);
      td.thpten(i, j, k) = dth1(k);
      td.qvpten(i, j, k) = dqv1(k) / (1.0 - sqv(k));
      td.qcpten(i, j, k) = dqc1(k) / (1.0 - sqv(k));
      td.qipten(i, j, k) = dqi1(k) / (1.0 - sqv(k));
      st.qc_bl(i, j, k) = qcbl(k);
      st.qi_bl(i, j, k) = qibl(k);
      st.cldfra_bl(i, j, k) = cldfra(k);
      st.el_pbl(i, j, k) = el(k);
      st.qke(i, j, k) = qke1(k);
      st.qke_adv(i, j, k) = qke1(k);
      st.tsq(i, j, k) = tsq1(k);
      st.qsq(i, j, k) = qsq1(k);
      st.cov(i, j, k) = cov1(k);
      st.sh3d(i, j, k) = sh(k);
      st.sm3d(i, j, k) = sm(k);
    }
    // tke_budget = 1
    {
      const int k = 1;
      const Real u3 = (ust * ust) * ust;
      qSHEAR1(k) = 4. * (u3 * phi_m / (karman * in.dz(i, j, k))) - qSHEAR1(k + 1);
      qBUOY1(k) = 4. * (-u3 * zet / (karman * in.dz(i, j, k))) - qBUOY1(k + 1);
      for (int kk = 1; kk <= nk - 1; ++kk) {
        st.qshear(i, j, kk) = 0.5 * (qSHEAR1(kk) + qSHEAR1(kk + 1));
        st.qbuoy(i, j, kk) = 0.5 * (qBUOY1(kk) + qBUOY1(kk + 1));
        st.qwt(i, j, kk) = qWT1(kk);
        st.qdiss(i, j, kk) = qDISS1(kk);
        st.dqke(i, j, kk) = (qke1(kk) - st.dqke(i, j, kk)) * 0.5 / delt;
      }
      st.qshear(i, j, nk) = 0.;
      st.qbuoy(i, j, nk) = 0.;
      st.qwt(i, j, nk) = 0.;
      st.qdiss(i, j, nk) = 0.;
      st.dqke(i, j, nk) = 0.;
    }
    if (edmf > 0) {
      for (int k = 1; k <= nk; ++k) {
        st.edmf_a(i, j, k) = mf.edmf_a(k);
        st.edmf_w(i, j, k) = mf.edmf_w(k);
        st.edmf_qt(i, j, k) = mf.edmf_qt(k);
        st.edmf_thl(i, j, k) = mf.edmf_thl(k);
        st.edmf_ent(i, j, k) = mf.edmf_ent(k);
        st.edmf_qc(i, j, k) = mf.edmf_qc(k);
        // sub_*/det_* stay 0 (env_subs = .false.)
      }
    }
    sfc.hpbl(i, j, 1) = pblh;
    sfc.kpbl(i, j, 1) = kpbl;
    sfc.nupdraft(i, j, 1) = nupdraft;
    sfc.ktop_plume(i, j, 1) = ktop_plume;
    sfc.maxmf(i, j, 1) = maxmf;
  }
};

} // namespace

// ---------------------------------------------------------------------------

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.ipbl = nl.geti("ipbl", o.ipbl);
  o.imoist = nl.geti("imoist", o.imoist);
  o.ptype = nl.geti("ptype", o.ptype);
  o.cm1setup = nl.geti("cm1setup", o.cm1setup);
  return o;
}

void Options::check() const {
  if (!active()) return;
  auto need = [](bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("pbl_mynn: " + what + " not supported (CM1's param.F fixes it)");
  };
  need(bl_mynn_cloudpdf == 2, "bl_mynn_cloudpdf=" + std::to_string(bl_mynn_cloudpdf));
  need(icloud_bl == 1, "icloud_bl=" + std::to_string(icloud_bl));
  need(spp_pbl == 0, "spp_pbl=" + std::to_string(spp_pbl));
  need(bl_mynn_tkebudget == 1, "bl_mynn_tkebudget=" + std::to_string(bl_mynn_tkebudget));
  need(bl_mynn_mixlength == 1, "bl_mynn_mixlength=" + std::to_string(bl_mynn_mixlength));
  need(bl_mynn_edmf == 0 || bl_mynn_edmf == 1, "bl_mynn_edmf=" + std::to_string(bl_mynn_edmf));
  need(bl_mynn_edmf_mom == 0 || bl_mynn_edmf_mom == 1, "bl_mynn_edmf_mom=" + std::to_string(bl_mynn_edmf_mom));
  need(bl_mynn_edmf_tke == 0, "bl_mynn_edmf_tke=" + std::to_string(bl_mynn_edmf_tke));
  need(bl_mynn_cloudmix == 1, "bl_mynn_cloudmix=" + std::to_string(bl_mynn_cloudmix));
  need(bl_mynn_mixscalars == 0, "bl_mynn_mixscalars=" + std::to_string(bl_mynn_mixscalars));
  need(bl_mynn_mixqt == 0, "bl_mynn_mixqt=" + std::to_string(bl_mynn_mixqt));
  need(bl_mynn_output == 1, "bl_mynn_output=" + std::to_string(bl_mynn_output));
  need(bl_mynn_tkeadvect, "bl_mynn_tkeadvect=.false.");
  if (cm1setup == 4)
    throw std::runtime_error("pbl_mynn: cm1setup=4 (qke zeroed in the LES subdomain) not ported");
}

void Surface::allocate(const Bounds& b) {
  auto a = [&](const char* n) { return FArray3(n, {b.ib, b.jb, 1}, {b.ie, b.je, 1}); };
  znt = a("znt"); xland = a("xland"); tsk = a("tsk"); qsfc = a("qsfc"); psfc = a("psfc"); ust = a("ust");
  ch = a("ch"); hfx = a("hfx"); qfx = a("qfx"); wspd = a("wspd"); rmol = a("rmol"); hpbl = a("hpbl");
  kpbl = a("kpbl2d"); nupdraft = a("nupdraft"); maxmf = a("maxmf"); ktop_plume = a("ktop_plume");
}

void State::allocate(const Bounds& b) {
  auto a = [&](const char* n) { return FArray3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
  qke = a("qke"); qke_adv = a("qke_adv"); sh3d = a("sh3d"); sm3d = a("sm3d"); tsq = a("tsq"); qsq = a("qsq");
  cov = a("cov"); el_pbl = a("el_pbl"); qc_bl = a("qc_bl"); qi_bl = a("qi_bl"); cldfra_bl = a("cldfra_bl");
  edmf_a = a("edmf_a"); edmf_w = a("edmf_w"); edmf_qt = a("edmf_qt"); edmf_thl = a("edmf_thl");
  edmf_ent = a("edmf_ent"); edmf_qc = a("edmf_qc"); sub_thl3d = a("sub_thl3d"); sub_sqv3d = a("sub_sqv3d");
  det_thl3d = a("det_thl3d"); det_sqv3d = a("det_sqv3d"); dqke = a("dqke"); qwt = a("qwt");
  qshear = a("qshear"); qbuoy = a("qbuoy"); qdiss = a("qdiss");
}

void Tendencies::allocate(const Bounds& b) {
  auto a = [&](const char* n) { return FArray3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
  upten = a("upten"); vpten = a("vpten"); thpten = a("thpten"); qvpten = a("qvpten"); qcpten = a("qcpten");
  qipten = a("qipten"); xkzh = a("xkzh"); xkzm = a("xkzm"); dheat = a("dheat");
}

void prepare_inputs(const Options& o, const Grid& g, const FArray3& ugr, const FArray3& vgr,
                    const FArray3& wa, const FArray3& th0, const FArray3& tha, const FArray3& pi0,
                    const FArray3& ppi, const FArray3& prs, const FArray3& rho, const FArray4& qa, int nqv,
                    int nqc, int nqi, const Inputs& in) {
  const Bounds& b = g.b;
  const bool hasqv = (o.imoist == 1 && nqv >= 1), hasqc = (nqc >= 1), hasqi = (nqi >= 1);
  const Real dz = g.dz;
  const auto rmh = g.rmh;
  const Inputs I = in;
  Kokkos::parallel_for("pbl_mynn_prep", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    I.dz(i, j, k) = dz * rmh(i, j, k);
    I.u(i, j, k) = 0.5 * (ugr(i, j, k) + ugr(i + 1, j, k));
    I.v(i, j, k) = 0.5 * (vgr(i, j, k) + vgr(i, j + 1, k));
    I.w(i, j, k) = 0.5 * (wa(i, j, k) + wa(i, j, k + 1));
    const Real th = th0(i, j, k) + tha(i, j, k);
    const Real pi = pi0(i, j, k) + ppi(i, j, k);
    I.th(i, j, k) = th;
    I.exner(i, j, k) = pi;
    I.t3d(i, j, k) = th * pi;
    I.qv(i, j, k) = hasqv ? qa(i, j, k, nqv) : 0.0;
    I.qc(i, j, k) = hasqc ? qa(i, j, k, nqc) : 0.0;
    I.qi(i, j, k) = hasqi ? qa(i, j, k, nqi) : 0.0;
    I.p(i, j, k) = prs(i, j, k);
    I.rho(i, j, k) = rho(i, j, k);
  });
}

void run(const Options& o, int ni, int nj, int nk, Real dt, Real dx, int nstep, const Inputs& in,
         const Surface& sfc, const State& st, const Tendencies& tend, std::size_t max_work_bytes) {
  o.check();
  if (!o.active()) return;
  if (nk < 4) throw std::runtime_error("pbl_mynn: needs nk >= 4");
  const int ncol = ni * nj;
  const std::size_t per_col = std::size_t(NSLOTS) * std::size_t(nk + 2) * sizeof(Real);
  int chunk = int(std::max<std::size_t>(1, max_work_bytes / per_col));
  chunk = std::min(chunk, ncol);
  Kokkos::View<Real**, Device> work("pbl_mynn_work", chunk, std::size_t(NSLOTS) * (nk + 2));
  ColumnKernel kern;
  kern.ni = ni;
  kern.nk = nk;
  kern.initflag = (nstep == 0) ? 1 : 0;
  kern.edmf = o.bl_mynn_edmf;
  kern.edmf_mom = o.bl_mynn_edmf_mom;
  kern.delt = dt;
  kern.dx = dx;
  kern.closure = o.closure();
  kern.in = in;
  kern.sfc = sfc;
  kern.st = st;
  kern.td = tend;
  kern.work = work;
  for (int c0 = 0; c0 < ncol; c0 += chunk) {
    kern.col0 = c0;
    const int n = std::min(chunk, ncol - c0);
    Kokkos::parallel_for("pbl_mynn_columns", Kokkos::RangePolicy<ExeSpace>(0, n), kern);
  }
}

} // namespace cm1k::pbl_mynn
