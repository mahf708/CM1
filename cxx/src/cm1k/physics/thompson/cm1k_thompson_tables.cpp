// Port of thompson_init and its table builders (src/thompson.F):
// qr_acr_qg, qr_acr_qs, freezeH2O, qi_aut_qs, table_Efrw, table_Efsw,
// table_dropEvap, and the GAMMLN/GAMMP/WGAMMA helpers.
//
// Bitwise parity with the double-precision Fortran:
//  - every expression keeps the Fortran's left-to-right evaluation; Fortran
//    x**3.0 is pow(x,3.0), x**2.0 is x*x, x**0.0 is 1.0, x**n (integer n)
//    is libgcc's __powidf2 (powi below), NINT is lround, INT truncates;
//  - the reference build keeps the table loops scalar (`!GCC$ NOVECTOR` in
//    thompson.F), so glibc's scalar exp/pow/log are used on both sides;
//  - the collection integrals (qr_acr_qg/qs) are sums over 100x100 bin
//    pairs per table entry, accumulated in the Fortran's order. Here the
//    factors that do not depend on the entry are computed once per bin pair
//    (the same products, rounded identically), terms that are exactly zero
//    (dvg/dvr = 0, so the term is +0 and adding it is a no-op) are skipped,
//    and neighbouring entries are accumulated side by side (each entry's sum
//    stays sequential). This is the same arithmetic, only faster.
//
// Build these with optimization even in Debug builds: the tables take minutes
// unoptimized. Optimization does not change results (no FMA contraction,
// no reassociation).
#if defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__) && !defined(__NVCOMPILER)
#pragma GCC optimize("O3")
#endif

#include "cm1k_thompson_tables.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

namespace cm1k::thompson {

using namespace par;

namespace {

// libgcc __powidf2: Fortran real**integer with a non-constant exponent.
inline double powi(double x, int m) {
  unsigned n = m < 0 ? -(unsigned)m : (unsigned)m;
  double y = (n % 2) ? x : 1.0;
  while (n >>= 1) {
    x = x * x;
    if (n % 2) y = y * x;
  }
  return m < 0 ? 1.0 / y : y;
}
inline int nint(double x) { return int(std::lround(x)); }
inline double fmax2(double a, double b) { return a > b ? a : b; }
inline double fmin2(double a, double b) { return a < b ? a : b; }

// Lookup-table abscissae (thompson.F declarations)
constexpr double r_c[ntb_c + 1] = {0, 1.e-6, 2.e-6, 3.e-6, 4.e-6, 5.e-6, 6.e-6, 7.e-6, 8.e-6, 9.e-6,
    1.e-5, 2.e-5, 3.e-5, 4.e-5, 5.e-5, 6.e-5, 7.e-5, 8.e-5, 9.e-5,
    1.e-4, 2.e-4, 3.e-4, 4.e-4, 5.e-4, 6.e-4, 7.e-4, 8.e-4, 9.e-4,
    1.e-3, 2.e-3, 3.e-3, 4.e-3, 5.e-3, 6.e-3, 7.e-3, 8.e-3, 9.e-3, 1.e-2};
constexpr double r_i[ntb_i + 1] = {0, 1.e-10, 2.e-10, 3.e-10, 4.e-10, 5.e-10, 6.e-10, 7.e-10, 8.e-10, 9.e-10,
    1.e-9, 2.e-9, 3.e-9, 4.e-9, 5.e-9, 6.e-9, 7.e-9, 8.e-9, 9.e-9,
    1.e-8, 2.e-8, 3.e-8, 4.e-8, 5.e-8, 6.e-8, 7.e-8, 8.e-8, 9.e-8,
    1.e-7, 2.e-7, 3.e-7, 4.e-7, 5.e-7, 6.e-7, 7.e-7, 8.e-7, 9.e-7,
    1.e-6, 2.e-6, 3.e-6, 4.e-6, 5.e-6, 6.e-6, 7.e-6, 8.e-6, 9.e-6,
    1.e-5, 2.e-5, 3.e-5, 4.e-5, 5.e-5, 6.e-5, 7.e-5, 8.e-5, 9.e-5,
    1.e-4, 2.e-4, 3.e-4, 4.e-4, 5.e-4, 6.e-4, 7.e-4, 8.e-4, 9.e-4, 1.e-3};
constexpr const double* r_r = r_c;   // same values (1e-6 .. 1e-2)
constexpr const double* r_g = r_c;
constexpr const double* r_s = r_c;
constexpr double N0r_exp[ntb_r1 + 1] = {0, 1.e6, 2.e6, 3.e6, 4.e6, 5.e6, 6.e6, 7.e6, 8.e6, 9.e6,
    1.e7, 2.e7, 3.e7, 4.e7, 5.e7, 6.e7, 7.e7, 8.e7, 9.e7,
    1.e8, 2.e8, 3.e8, 4.e8, 5.e8, 6.e8, 7.e8, 8.e8, 9.e8,
    1.e9, 2.e9, 3.e9, 4.e9, 5.e9, 6.e9, 7.e9, 8.e9, 9.e9, 1.e10};
constexpr double N0g_exp[ntb_g1 + 1] = {0, 1.e2, 2.e2, 3.e2, 4.e2, 5.e2, 6.e2, 7.e2, 8.e2, 9.e2,
    1.e3, 2.e3, 3.e3, 4.e3, 5.e3, 6.e3, 7.e3, 8.e3, 9.e3,
    1.e4, 2.e4, 3.e4, 4.e4, 5.e4, 6.e4, 7.e4, 8.e4, 9.e4,
    1.e5, 2.e5, 3.e5, 4.e5, 5.e5, 6.e5, 7.e5, 8.e5, 9.e5, 1.e6};
constexpr double Nt_i[ntb_i1 + 1] = {0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    1.e1, 2.e1, 3.e1, 4.e1, 5.e1, 6.e1, 7.e1, 8.e1, 9.e1,
    1.e2, 2.e2, 3.e2, 4.e2, 5.e2, 6.e2, 7.e2, 8.e2, 9.e2,
    1.e3, 2.e3, 3.e3, 4.e3, 5.e3, 6.e3, 7.e3, 8.e3, 9.e3,
    1.e4, 2.e4, 3.e4, 4.e4, 5.e4, 6.e4, 7.e4, 8.e4, 9.e4,
    1.e5, 2.e5, 3.e5, 4.e5, 5.e5, 6.e5, 7.e5, 8.e5, 9.e5, 1.e6};
constexpr const double* Nt_IN = Nt_i;   // same values (1 .. 1e6)
constexpr double sa[11] = {0, 5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
                           0.31255, 0.000204, 0.003199, 0.0, -0.015952};
constexpr double sb[11] = {0, 0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
                           0.060366, 0.000079, 0.000594, 0.0, -0.003577};
constexpr double Tc[ntb_t + 1] = {0, -0.01, -5., -10., -15., -20., -25., -30., -35., -40.};

// Rain fall speed polynomial used by the collection tables and table_Efrw.
inline double vt_rain_poly(double D) {
  return -0.1021 + 4.932E3 * D - 0.9551E6 * D * D + 0.07934E9 * D * D * D
       - 0.002362E12 * D * D * D * D;
}

// Bin edges: exp((n-1)/nb*log(xmax/x1) + log(x1)), centres sqrt(x(n)*x(n+1)).
void make_bins(int nb, double x1, double xmax, std::vector<double>& D, std::vector<double>& dt) {
  std::vector<double> xDx(nb + 2);
  xDx[1] = x1;
  xDx[nb + 1] = xmax;
  for (int n = 2; n <= nb; ++n)
    xDx[n] = std::exp(double(n - 1) / double(nb) * std::log(xDx[nb + 1] / xDx[1]) + std::log(xDx[1]));
  D.assign(nb + 1, 0.0);
  dt.assign(nb + 1, 0.0);
  for (int n = 1; n <= nb; ++n) {
    D[n] = std::sqrt(xDx[n] * xDx[n + 1]);
    dt[n] = xDx[n + 1] - xDx[n];
  }
}

// Run f(km) for km in [s, e] on the host execution space (each km writes its
// own table entries, so the order does not matter).
template <class F>
void host_for(int s, int e, const F& f) {
  if (e < s) return;
  Kokkos::parallel_for("thompson_tables",
                       Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(s, e + 1), f);
  Kokkos::DefaultHostExecutionSpace().fence();
}

// Snow moment helper of qr_acr_qs / mp_thompson: polynomial in tc and x.
inline double sm_loga(double tc, double x) {
  return sa[1] + sa[2] * tc + sa[3] * x + sa[4] * tc * x + sa[5] * tc * tc + sa[6] * x * x
       + sa[7] * tc * tc * x + sa[8] * tc * x * x + sa[9] * tc * tc * tc + sa[10] * x * x * x;
}
inline double sm_b(double tc, double x) {
  return sb[1] + sb[2] * tc + sb[3] * x + sb[4] * tc * x + sb[5] * tc * tc + sb[6] * x * x
       + sb[7] * tc * tc * x + sb[8] * tc * x * x + sb[9] * tc * tc * tc + sb[10] * x * x * x;
}

} // namespace

//---------------------------------------------------------------------------
double gammln(double xx) {
  constexpr double STP = 2.5066282746310005;
  constexpr double COF[7] = {0, 76.18009172947146, -86.50532032941677, 24.01409824083091,
                             -1.231739572450155, .1208650973866179E-2, -.5395239384953E-5};
  double x = xx, y = x, tmp = x + 5.5;
  tmp = (x + 0.5) * std::log(tmp) - tmp;
  double ser = 1.000000000190015;
  for (int j = 1; j <= 6; ++j) {
    y = y + 1.0;
    ser = ser + COF[j] / y;
  }
  return tmp + std::log(STP * ser / x);
}

double wgamma(double y) { return std::exp(gammln(y)); }

namespace {
constexpr int ITMAX = 100;
constexpr double gEPS = 3.E-7, FPMIN = 1.E-30;

double gser(double a, double x) {
  const double gln = gammln(a);
  if (x <= 0.) return 0.;
  double ap = a, sum = 1. / a, del = sum;
  for (int n = 1; n <= ITMAX; ++n) {
    ap = ap + 1.;
    del = del * x / ap;
    sum = sum + del;
    if (std::fabs(del) < std::fabs(sum) * gEPS) break;
  }
  return sum * std::exp(-x + a * std::log(x) - gln);
}

double gcf(double a, double x) {
  const double gln = gammln(a);
  double b = x + 1. - a, c = 1. / FPMIN, d = 1. / b, h = d;
  for (int i = 1; i <= ITMAX; ++i) {
    const double an = -(double(i) * (double(i) - a));
    b = b + 2.;
    d = an * d + b;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (std::fabs(c) < FPMIN) c = FPMIN;
    d = 1. / d;
    const double del = d * c;
    h = h * del;
    if (std::fabs(del - 1.) < gEPS) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}
} // namespace

double gammp(double a, double x) {
  if (x < 0. || a <= 0.) return 0.;
  if (x < a + 1.) return gser(a, x);
  return 1. - gcf(a, x);
}

int default_idx_IN(const Consts& c) {
  // mp_thompson, not aerosol aware: xni = 1.0*1000. (1 per liter)
  const double xni = 1.0 * 1000.;
  if (!(xni > Nt_IN[1])) return 1;
  const int niin = nint(std::log10(xni));
  int n = niin + 1;
  for (int nn = niin - 1; nn <= niin + 1; ++nn) {
    n = nn;
    if ((xni / powi(10., nn)) >= 1.0 && (xni / powi(10., nn)) < 10.0) break;
  }
  int idx = int(xni / powi(10., n)) + 10 * (n - c.niIN2) - (n - c.niIN2);
  return std::max(1, std::min(idx, ntb_IN));
}

//---------------------------------------------------------------------------
void init_constants(HostTables& t, double nt_c_in, std::vector<int> in_slices) {
  Consts& c = t.c;
  c = Consts{};
  c.Nt_c = 1.0e6 * nt_c_in;
  c.mu_c = fmin2(15., (1000.E6 / c.Nt_c + 2.));
  c.Sc3 = std::pow(Sc, 1. / 3.);
  // constant expressions in the Fortran (folded by the compiler)
  c.D0i = std::pow(xm0i / am_i, 1. / bm_i);
  c.xm0s = am_s * (D0s * D0s);
  c.xm0g = am_g * std::pow(D0g, bm_g);

  for (int n = 1; n <= 15; ++n) {
    c.cce[1][n] = n + 1.;
    c.cce[2][n] = bm_r + n + 1.;
    c.cce[3][n] = bm_r + n + 4.;
    c.cce[4][n] = n + bv_c + 1.;
    c.cce[5][n] = bm_r + n + bv_c + 1.;
    for (int l = 1; l <= 5; ++l) c.ccg[l][n] = wgamma(c.cce[l][n]);
    c.ocg1[n] = 1. / c.ccg[1][n];
    c.ocg2[n] = 1. / c.ccg[2][n];
  }

  c.cie[1] = mu_i + 1.;
  c.cie[2] = bm_i + mu_i + 1.;
  c.cie[3] = bm_i + mu_i + bv_i + 1.;
  c.cie[4] = mu_i + bv_i + 1.;
  c.cie[5] = mu_i + 2.;
  c.cie[6] = bm_i * 0.5 + mu_i + bv_i + 1.;
  c.cie[7] = bm_i * 0.5 + mu_i + 1.;
  for (int n = 1; n <= 7; ++n) c.cig[n] = wgamma(c.cie[n]);
  c.oig1 = 1. / c.cig[1];
  c.oig2 = 1. / c.cig[2];
  c.obmi = 1. / bm_i;

  c.cre[1] = bm_r + 1.;
  c.cre[2] = mu_r + 1.;
  c.cre[3] = bm_r + mu_r + 1.;
  c.cre[4] = bm_r * 2. + mu_r + 1.;
  c.cre[5] = mu_r + bv_r + 1.;
  c.cre[6] = bm_r + mu_r + bv_r + 1.;
  c.cre[7] = bm_r * 0.5 + mu_r + bv_r + 1.;
  c.cre[8] = bm_r + mu_r + bv_r + 3.;
  c.cre[9] = mu_r + bv_r + 3.;
  c.cre[10] = mu_r + 2.;
  c.cre[11] = 0.5 * (bv_r + 5. + 2. * mu_r);
  c.cre[12] = bm_r * 0.5 + mu_r + 1.;
  c.cre[13] = bm_r * 2. + mu_r + bv_r + 1.;
  for (int n = 1; n <= 13; ++n) c.crg[n] = wgamma(c.cre[n]);
  c.obmr = 1. / bm_r;
  c.ore1 = 1. / c.cre[1];
  c.org1 = 1. / c.crg[1];
  c.org2 = 1. / c.crg[2];
  c.org3 = 1. / c.crg[3];

  c.cse[1] = bm_s + 1.;
  c.cse[2] = bm_s + 2.;
  c.cse[3] = bm_s * 2.;
  c.cse[4] = bm_s + bv_s + 1.;
  c.cse[5] = bm_s * 2. + bv_s + 1.;
  c.cse[6] = bm_s * 2. + 1.;
  c.cse[7] = bm_s + mu_s + 1.;
  c.cse[8] = bm_s + mu_s + 2.;
  c.cse[9] = bm_s + mu_s + 3.;
  c.cse[10] = bm_s + mu_s + bv_s + 1.;
  c.cse[11] = bm_s * 2. + mu_s + bv_s + 1.;
  c.cse[12] = bm_s * 2. + mu_s + 1.;
  c.cse[13] = bv_s + 2.;
  c.cse[14] = bm_s + bv_s;
  c.cse[15] = mu_s + 1.;
  c.cse[16] = 1.0 + (1.0 + bv_s) / 2.;
  c.cse[17] = c.cse[16] + mu_s + 1.;
  c.cse[18] = bv_s + mu_s + 3.;
  for (int n = 1; n <= 18; ++n) c.csg[n] = wgamma(c.cse[n]);
  c.oams = 1. / am_s;
  c.obms = 1. / bm_s;
  c.ocms = std::pow(c.oams, c.obms);

  c.cge[1] = bm_g + 1.;
  c.cge[2] = mu_g + 1.;
  c.cge[3] = bm_g + mu_g + 1.;
  c.cge[4] = bm_g * 2. + mu_g + 1.;
  c.cge[5] = bm_g * 2. + mu_g + bv_g + 1.;
  c.cge[6] = bm_g + mu_g + bv_g + 1.;
  c.cge[7] = bm_g + mu_g + bv_g + 2.;
  c.cge[8] = bm_g + mu_g + bv_g + 3.;
  c.cge[9] = mu_g + bv_g + 3.;
  c.cge[10] = mu_g + 2.;
  c.cge[11] = 0.5 * (bv_g + 5. + 2. * mu_g);
  c.cge[12] = 0.5 * (bv_g + 5.) + mu_g;
  for (int n = 1; n <= 12; ++n) c.cgg[n] = wgamma(c.cge[n]);
  c.oamg = 1. / am_g;
  c.obmg = 1. / bm_g;
  c.ocmg = std::pow(c.oamg, c.obmg);
  c.oge1 = 1. / c.cge[1];
  c.ogg1 = 1. / c.cgg[1];
  c.ogg2 = 1. / c.cgg[2];
  c.ogg3 = 1. / c.cgg[3];

  c.t1_qr_qc = PI * .25 * av_r * c.crg[9];
  c.t1_qr_qi = PI * .25 * av_r * c.crg[9];
  c.t2_qr_qi = PI * .25 * am_r * av_r * c.crg[8];
  c.t1_qg_qc = PI * .25 * av_g * c.cgg[9];
  c.t1_qs_qc = PI * .25 * av_s;
  c.t1_qs_qi = PI * .25 * av_s;
  c.t1_qr_ev = 0.78 * c.crg[10];
  c.t2_qr_ev = 0.308 * c.Sc3 * std::sqrt(av_r) * c.crg[11];
  c.t1_qs_sd = 0.86;
  c.t2_qs_sd = 0.28 * c.Sc3 * std::sqrt(av_s);
  c.t1_qs_me = PI * 4. * C_sqrd * olfus * 0.86;
  c.t2_qs_me = PI * 4. * C_sqrd * olfus * 0.28 * c.Sc3 * std::sqrt(av_s);
  c.t1_qg_sd = 0.86 * c.cgg[10];
  c.t2_qg_sd = 0.28 * c.Sc3 * std::sqrt(av_g) * c.cgg[11];
  c.t1_qg_me = PI * 4. * C_cube * olfus * 0.86 * c.cgg[10];
  c.t2_qg_me = PI * 4. * C_cube * olfus * 0.28 * c.Sc3 * std::sqrt(av_g) * c.cgg[11];

  c.nic2 = nint(std::log10(r_c[1]));
  c.nii2 = nint(std::log10(r_i[1]));
  c.nii3 = nint(std::log10(Nt_i[1]));
  c.nir2 = nint(std::log10(r_r[1]));
  c.nir3 = nint(std::log10(N0r_exp[1]));
  c.nis2 = nint(std::log10(r_s[1]));
  c.nig2 = nint(std::log10(r_g[1]));
  c.nig3 = nint(std::log10(N0g_exp[1]));
  c.niIN2 = nint(std::log10(Nt_IN[1]));

  // cloud water bins (1 um steps)
  t.Dc.assign(nbc + 1, 0.0);
  t.dtc.assign(nbc + 1, 0.0);
  t.Dc[1] = D0c * 1.0;
  t.dtc[1] = D0c * 1.0;
  for (int n = 2; n <= nbc; ++n) {
    t.Dc[n] = t.Dc[n - 1] + 1.0E-6;
    t.dtc[n] = (t.Dc[n] - t.Dc[n - 1]);
  }
  make_bins(nbi, c.D0i * 1.0, 5.0 * D0s, t.Di, t.dti);
  make_bins(nbr, D0r * 1.0, 0.005, t.Dr, t.dtr);
  make_bins(nbs, D0s * 1.0, 0.02, t.Ds, t.dts);
  make_bins(nbg, D0g * 1.0, 0.05, t.Dg, t.dtg);
  {
    std::vector<double> xDx(nbc + 2);
    xDx[1] = 1.0;
    xDx[nbc + 1] = 3000.0;
    for (int n = 2; n <= nbc; ++n)
      xDx[n] = std::exp(double(n - 1) / double(nbc) * std::log(xDx[nbc + 1] / xDx[1]) + std::log(xDx[1]));
    t.t_Nc.assign(nbc + 1, 0.0);
    for (int n = 1; n <= nbc; ++n) t.t_Nc[n] = std::sqrt(xDx[n] * xDx[n + 1]) * 1.E6;
    c.nic1 = int(std::log(t.t_Nc[nbc] / t.t_Nc[1]));   // INTEGER = DLOG(...): truncation
  }
  c.Dr1 = t.Dr[1]; c.Drn = t.Dr[nbr];
  c.Ds1 = t.Ds[1]; c.Dsn = t.Ds[nbs];
  c.t_Nc1 = t.t_Nc[1];

  // tables, zeroed
  const size_t nrg = size_t(ntb_g1) * ntb_g * ntb_r1 * ntb_r;
  for (auto* v : {&t.tcg_racg, &t.tmr_racg, &t.tcr_gacr, &t.tmg_gacr, &t.tnr_racg, &t.tnr_gacr}) v->assign(nrg, 0.0);
  const size_t nrs = size_t(ntb_s) * ntb_t * ntb_r1 * ntb_r;
  for (auto* v : {&t.tcs_racs1, &t.tmr_racs1, &t.tcs_racs2, &t.tmr_racs2, &t.tcr_sacr1, &t.tms_sacr1,
                  &t.tcr_sacr2, &t.tms_sacr2, &t.tnr_racs1, &t.tnr_racs2, &t.tnr_sacr1, &t.tnr_sacr2})
    v->assign(nrs, 0.0);
  for (int m : in_slices)
    if (m < 1 || m > ntb_IN) throw std::runtime_error("thompson: bad ice-nuclei slice");
  t.in_slices = std::move(in_slices);
  const size_t ns = t.in_slices.size();
  for (auto* v : {&t.tpi_qrfz, &t.tni_qrfz, &t.tpg_qrfz, &t.tnr_qrfz}) v->assign(size_t(ntb_r) * ntb_r1 * 45 * ns, 0.0);
  for (auto* v : {&t.tpi_qcfz, &t.tni_qcfz}) v->assign(size_t(ntb_c) * nbc * 45 * ns, 0.0);
  for (auto* v : {&t.tps_iaus, &t.tni_iaus, &t.tpi_ide}) v->assign(size_t(ntb_i) * ntb_i1, 0.0);
  t.t_Efrw.assign(size_t(nbr) * nbc, 0.0);
  t.t_Efsw.assign(size_t(nbs) * nbc, 0.0);
  for (auto* v : {&t.tpc_wev, &t.tnc_wev}) v->assign(size_t(nbc) * ntb_c * nbc, 0.0);
}

//---------------------------------------------------------------------------
void table_Efrw(HostTables& t) {
  const auto& Dc = t.Dc;
  const auto& Dr = t.Dr;
  for (int j = 1; j <= nbc; ++j)
    for (int i = 1; i <= nbr; ++i) {
      double Ef_rw = 0.0;
      const double p = Dc[j] / Dr[i];
      double& out = t.t_Efrw[HostTables::i2(i, j, nbr)];
      if (Dr[i] < 50.E-6 || Dc[j] < 3.E-6) {
        out = 0.0;
      } else if (p > 0.25) {
        const double X = Dc[j] * 1.E6;
        if (Dr[i] < 75.e-6) {
          Ef_rw = 0.026794 * X - 0.20604;
        } else if (Dr[i] < 125.e-6) {
          Ef_rw = -0.00066842 * X * X + 0.061542 * X - 0.37089;
        } else if (Dr[i] < 175.e-6) {
          Ef_rw = 4.091e-06 * X * X * X * X - 0.00030908 * X * X * X + 0.0066237 * X * X - 0.0013687 * X - 0.073022;
        } else if (Dr[i] < 250.e-6) {
          Ef_rw = 9.6719e-5 * X * X * X - 0.0068901 * X * X + 0.17305 * X - 0.65988;
        } else if (Dr[i] < 350.e-6) {
          Ef_rw = 9.0488e-5 * X * X * X - 0.006585 * X * X + 0.16606 * X - 0.56125;
        } else {
          Ef_rw = 0.00010721 * X * X * X - 0.0072962 * X * X + 0.1704 * X - 0.46929;
        }
      } else {
        const double vtr = vt_rain_poly(Dr[i]);
        const double stokes = Dc[j] * Dc[j] * vtr * rho_w / (9. * 1.718E-5 * Dr[i]);
        const double reynolds = 9. * stokes / (p * p * rho_w);
        const double F = std::log(reynolds);
        const double G = -0.1007 - 0.358 * F + 0.0261 * F * F;
        const double K0 = std::exp(G);
        const double z = std::log(stokes / (K0 + 1.E-15));
        const double H = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
        const double yc0 = 2.0 / PI * std::atan(H);
        Ef_rw = (yc0 + p) * (yc0 + p) / ((1. + p) * (1. + p));
      }
      out = fmax2(0.0, fmin2(Ef_rw, 0.95));
    }
}

void table_Efsw(HostTables& t) {
  const auto& Dc = t.Dc;
  const auto& Ds = t.Ds;
  for (int j = 1; j <= nbc; ++j) {
    const double vtc = 1.19E4 * (1.0E4 * Dc[j] * Dc[j] * 0.25);
    for (int i = 1; i <= nbs; ++i) {
      const double vts = av_s * std::pow(Ds[i], bv_s) * std::exp(-fv_s * Ds[i]) - vtc;
      const double Ds_m = std::pow(am_s * (Ds[i] * Ds[i]) / am_r, t.c.obmr);
      const double p = Dc[j] / Ds_m;
      double& out = t.t_Efsw[HostTables::i2(i, j, nbs)];
      if (p > 0.25 || Ds[i] < D0s || Dc[j] < 6.E-6 || vts < 1.E-3) {
        out = 0.0;
      } else {
        const double stokes = Dc[j] * Dc[j] * vts * rho_w / (9. * 1.718E-5 * Ds_m);
        const double reynolds = 9. * stokes / (p * p * rho_w);
        const double F = std::log(reynolds);
        const double G = -0.1007 - 0.358 * F + 0.0261 * F * F;
        const double K0 = std::exp(G);
        const double z = std::log(stokes / (K0 + 1.E-15));
        const double H = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
        const double yc0 = 2.0 / PI * std::atan(H);
        const double Ef_sw = (yc0 + p) * (yc0 + p) / ((1. + p) * (1. + p));
        out = fmax2(0.0, fmin2(Ef_sw, 0.95));
      }
    }
  }
}

//---------------------------------------------------------------------------
void table_dropEvap(HostTables& t) {
  const Consts& c = t.c;
  const auto& Dc = t.Dc;
  std::vector<double> massc(nbc + 1), N_c(nbc + 1);
  for (int n = 1; n <= nbc; ++n) massc[n] = am_r * std::pow(Dc[n], bm_r);
  for (int k = 1; k <= nbc; ++k) {
    const int nu_c = std::min(15, nint(1000.E6 / t.t_Nc[k]) + 2);
    for (int j = 1; j <= ntb_c; ++j) {
      const double lamc = std::pow(t.t_Nc[k] * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / r_c[j], c.obmr);
      const double N0_c = t.t_Nc[k] * c.ocg1[nu_c] * std::pow(lamc, c.cce[1][nu_c]);
      for (int i = 1; i <= nbc; ++i) {
        N_c[i] = N0_c * powi(Dc[i], nu_c) * std::exp(-lamc * Dc[i]) * t.dtc[i];
        double summ = 0., summ2 = 0.;
        for (int n = 1; n <= i; ++n) {
          summ = summ + massc[n] * N_c[n];
          summ2 = summ2 + N_c[n];
        }
        t.tpc_wev[HostTables::iwev(i, j, k)] = summ;
        t.tnc_wev[HostTables::iwev(i, j, k)] = summ2;
      }
    }
  }
}

//---------------------------------------------------------------------------
// qr_acr_qg: rain collecting graupel and graupel collecting rain.
//
// Fortran, per entry (i,j,k,m), with c0 = PI*.25*Ef_rg (folded):
//   do n2: massr = am_r*Dr**bm_r
//    do n: massg = am_g*Dg**bm_g; dvg, dvr; P = c0*(Dg+Dr)*(Dg+Dr)
//      t1 += P*dvg*massg*N_g*N_r ; z1 += P*dvg*massr*N_g*N_r ; y1 += P*dvg*N_g*N_r
//      t2 += P*dvr*massr*N_g*N_r ; y2 += P*dvr*N_g*N_r ; z2 += P*dvr*massg*N_g*N_r
// (all left-associative). The prefixes P*dvg*massg etc. depend on (n2,n)
// only; N_g on (i,j,n), N_r on (k,m,n2).
void qr_acr_qg(HostTables& t, int km_s, int km_e) {
  const Consts& c = t.c;
  const auto& Dr = t.Dr;
  const auto& Dg = t.Dg;
  std::vector<double> vr(nbr + 1), vg(nbg + 1);
  for (int n2 = 1; n2 <= nbr; ++n2) vr[n2] = vt_rain_poly(Dr[n2]);
  for (int n = 1; n <= nbg; ++n) vg[n] = av_g * std::pow(Dg[n], bv_g);

  // Per bin pair: the entry-independent prefixes, for the pairs whose term is
  // nonzero (dvg > 0: t1, z1, y1; dvr > 0: t2, y2, z2), in Fortran order.
  struct Pair { int n2, n; double a, am1, am2; };   // a = P*dv, am1/am2 = a*mass
  std::vector<Pair> pg, pr;
  const double c0 = PI * .25 * Ef_rg;
  for (int n2 = 1; n2 <= nbr; ++n2) {
    const double massr = am_r * std::pow(Dr[n2], bm_r);
    for (int n = 1; n <= nbg; ++n) {
      const double massg = am_g * std::pow(Dg[n], bm_g);
      const double dvg = 0.5 * ((vr[n2] - vg[n]) + std::fabs(vr[n2] - vg[n]));
      const double dvr = 0.5 * ((vg[n] - vr[n2]) + std::fabs(vg[n] - vr[n2]));
      const double P = c0 * (Dg[n] + Dr[n2]) * (Dg[n] + Dr[n2]);
      if (dvg != 0.0) { const double a = P * dvg; pg.push_back({n2, n, a, a * massg, a * massr}); }
      if (dvr != 0.0) { const double a = P * dvr; pr.push_back({n2, n, a, a * massr, a * massg}); }
    }
  }

  // N_g(n) for every (i,j): (N0_g*Dg**mu_g)*DEXP(-lamg*Dg)*dtg, Dg**0 = 1
  constexpr int L = ntb_g1;   // lanes: i = 1..ntb_g1
  std::vector<double> Ngv(size_t(ntb_g) * (nbg + 1) * L);   // [j][n][i]
  for (int j = 1; j <= ntb_g; ++j)
    for (int i = 1; i <= ntb_g1; ++i) {
      const double lam_exp = std::pow(N0g_exp[i] * am_g * c.cgg[1] / r_g[j], c.oge1);
      const double lamg = lam_exp * std::pow(c.cgg[3] * c.ogg2 * c.ogg1, c.obmg);
      const double N0_g = N0g_exp[i] / (c.cgg[2] * lam_exp) * std::pow(lamg, c.cge[2]);
      for (int n = 1; n <= nbg; ++n)
        Ngv[(size_t(j - 1) * (nbg + 1) + n) * L + (i - 1)] = N0_g * 1.0 * std::exp(-lamg * Dg[n]) * t.dtg[n];
    }

  host_for(km_s, km_e, [&](int km) {
    const int m = km / ntb_r1 + 1;
    const int k = km % ntb_r1 + 1;
    const double lam_exp = std::pow(N0r_exp[k] * am_r * c.crg[1] / r_r[m], c.ore1);
    const double lamr = lam_exp * std::pow(c.crg[3] * c.org2 * c.org1, c.obmr);
    const double N0_r = N0r_exp[k] / (c.crg[2] * lam_exp) * std::pow(lamr, c.cre[2]);
    double N_r[nbr + 1];
    for (int n2 = 1; n2 <= nbr; ++n2) N_r[n2] = N0_r * 1.0 * std::exp(-lamr * Dr[n2]) * t.dtr[n2];

    for (int j = 1; j <= ntb_g; ++j) {
      const double* Ng = &Ngv[size_t(j - 1) * (nbg + 1) * L];
      alignas(64) double t1[L] = {}, z1[L] = {}, y1[L] = {}, t2[L] = {}, y2[L] = {}, z2[L] = {};
      for (const Pair& p : pg) {
        const double* g = Ng + size_t(p.n) * L;
        const double nr = N_r[p.n2];
        for (int l = 0; l < L; ++l) {
          t1[l] = t1[l] + p.am1 * g[l] * nr;
          z1[l] = z1[l] + p.am2 * g[l] * nr;
          y1[l] = y1[l] + p.a * g[l] * nr;
        }
      }
      for (const Pair& p : pr) {
        const double* g = Ng + size_t(p.n) * L;
        const double nr = N_r[p.n2];
        for (int l = 0; l < L; ++l) {
          t2[l] = t2[l] + p.am1 * g[l] * nr;
          y2[l] = y2[l] + p.a * g[l] * nr;
          z2[l] = z2[l] + p.am2 * g[l] * nr;
        }
      }
      for (int i = 1; i <= ntb_g1; ++i) {
        const size_t id = HostTables::ig(i, j, k, m);
        t.tcg_racg[id] = t1[i - 1];
        t.tmr_racg[id] = fmin2(z1[i - 1], r_r[m] * 1.0);
        t.tcr_gacr[id] = t2[i - 1];
        t.tmg_gacr[id] = fmin2(z2[i - 1], r_g[j] * 1.0);
        t.tnr_racg[id] = y1[i - 1];
        t.tnr_gacr[id] = y2[i - 1];
      }
    }
  });
}

//---------------------------------------------------------------------------
// qr_acr_qs: rain collecting snow and snow collecting rain (same structure;
// which sums a term goes to also depends on massr > 1.5*masss).
void qr_acr_qs(HostTables& t, int km_s, int km_e) {
  const Consts& c = t.c;
  const auto& Dr = t.Dr;
  const auto& Ds = t.Ds;
  std::vector<double> vr(nbr + 1), vs(nbs + 1);
  for (int n2 = 1; n2 <= nbr; ++n2) vr[n2] = vt_rain_poly(Dr[n2]);
  for (int n = 1; n <= nbs; ++n) vs[n] = 1.5 * av_s * std::pow(Ds[n], bv_s) * std::exp(-fv_s * Ds[n]);

  // sum ids: 0 (t1,z1,y1), 1 (t3,z3,y3) for dvs; 2 (t2,y2,z2), 3 (t4,y4,z4) for dvr
  struct Pair { int n2, n, which; double a, am1, am2; };
  std::vector<Pair> ps, prr;
  const double c0 = PI * .25 * Ef_rs;
  for (int n2 = 1; n2 <= nbr; ++n2) {
    const double massr = am_r * std::pow(Dr[n2], bm_r);
    for (int n = 1; n <= nbs; ++n) {
      const double masss = am_s * (Ds[n] * Ds[n]);
      const double dvs = 0.5 * ((vr[n2] - vs[n]) + std::fabs(vr[n2] - vs[n]));
      const double dvr = 0.5 * ((vs[n] - vr[n2]) + std::fabs(vs[n] - vr[n2]));
      const double P = c0 * (Ds[n] + Dr[n2]) * (Ds[n] + Dr[n2]);
      const bool big = massr > 1.5 * masss;
      // t1/t3 = P*dvs*masss..., z1/z3 = P*dvs*massr..., y = P*dvs...
      if (dvs != 0.0) { const double a = P * dvs; ps.push_back({n2, n, big ? 0 : 1, a, a * masss, a * massr}); }
      // t2/t4 = P*dvr*massr..., z2/z4 = P*dvr*masss...
      if (dvr != 0.0) { const double a = P * dvr; prr.push_back({n2, n, big ? 2 : 3, a, a * massr, a * masss}); }
    }
  }

  // N_s(n) for every (i,j): lanes over (i,j), i fastest
  constexpr int L = ntb_s * ntb_t;
  std::vector<double> Nsv(size_t(nbs + 1) * L);   // [n][lane]
  for (int j = 1; j <= ntb_t; ++j)
    for (int i = 1; i <= ntb_s; ++i) {
      const double M2 = r_s[i] * c.oams * 1.0;
      double second;
      if (bm_s > 2.0 - 1.E-3 && bm_s < 2.0 + 1.E-3) {
        const double loga_ = sm_loga(Tc[j], bm_s);
        const double a_ = std::pow(10.0, loga_);
        const double b_ = sm_b(Tc[j], bm_s);
        second = std::pow(M2 / a_, 1. / b_);
      } else {
        second = M2;
      }
      const double loga_ = sm_loga(Tc[j], c.cse[1]);
      const double a_ = std::pow(10.0, loga_);
      const double b_ = sm_b(Tc[j], c.cse[1]);
      const double M3 = a_ * std::pow(second, b_);
      const double oM3 = 1. / M3;
      const double Mrat = M2 * (M2 * oM3) * (M2 * oM3) * (M2 * oM3);
      const double M0 = std::pow(M2 * oM3, mu_s);
      const double slam1 = M2 * oM3 * Lam0;
      const double slam2 = M2 * oM3 * Lam1;
      const int lane = (i - 1) + ntb_s * (j - 1);
      for (int n = 1; n <= nbs; ++n)
        Nsv[size_t(n) * L + lane] = Mrat * (Kap0 * std::exp(-slam1 * Ds[n])
                                    + Kap1 * M0 * std::pow(Ds[n], mu_s) * std::exp(-slam2 * Ds[n])) * t.dts[n];
    }

  host_for(km_s, km_e, [&](int km) {
    const int m = km / ntb_r1 + 1;
    const int k = km % ntb_r1 + 1;
    const double lam_exp = std::pow(N0r_exp[k] * am_r * c.crg[1] / r_r[m], c.ore1);
    const double lamr = lam_exp * std::pow(c.crg[3] * c.org2 * c.org1, c.obmr);
    const double N0_r = N0r_exp[k] / (c.crg[2] * lam_exp) * std::pow(lamr, c.cre[2]);
    double N_r[nbr + 1];
    for (int n2 = 1; n2 <= nbr; ++n2) N_r[n2] = N0_r * 1.0 * std::exp(-lamr * Dr[n2]) * t.dtr[n2];

    // acc[s][0..2][lane]: (t,z,y) for s = 0 (t1,z1,y1), 1 (t3,z3,y3), 2 (t2,z2,y2), 3 (t4,z4,y4)
    std::vector<double> acc(size_t(4) * 3 * L, 0.0);
    for (const auto* list : {&ps, &prr})
      for (const Pair& p : *list) {
        const double* s = &Nsv[size_t(p.n) * L];
        const double nr = N_r[p.n2];
        double* tt = &acc[(size_t(p.which) * 3 + 0) * L];
        double* zz = &acc[(size_t(p.which) * 3 + 1) * L];
        double* yy = &acc[(size_t(p.which) * 3 + 2) * L];
        for (int l = 0; l < L; ++l) {
          tt[l] = tt[l] + p.am1 * s[l] * nr;
          zz[l] = zz[l] + p.am2 * s[l] * nr;
          yy[l] = yy[l] + p.a * s[l] * nr;
        }
      }
    auto A = [&](int s, int q, int lane) { return acc[(size_t(s) * 3 + q) * L + lane]; };
    for (int j = 1; j <= ntb_t; ++j)
      for (int i = 1; i <= ntb_s; ++i) {
        const int lane = (i - 1) + ntb_s * (j - 1);
        const size_t id = HostTables::is(i, j, k, m);
        t.tcs_racs1[id] = A(0, 0, lane);
        t.tmr_racs1[id] = fmin2(A(0, 1, lane), r_r[m] * 1.0);
        t.tcs_racs2[id] = A(1, 0, lane);
        t.tmr_racs2[id] = A(1, 1, lane);
        t.tcr_sacr1[id] = A(2, 0, lane);
        t.tms_sacr1[id] = A(2, 1, lane);
        t.tcr_sacr2[id] = A(3, 0, lane);
        t.tms_sacr2[id] = A(3, 1, lane);
        t.tnr_racs1[id] = A(0, 2, lane);
        t.tnr_racs2[id] = A(1, 2, lane);
        t.tnr_sacr1[id] = A(2, 2, lane);
        t.tnr_sacr2[id] = A(3, 2, lane);
      }
  });
}

//---------------------------------------------------------------------------
// freezeH2O (Bigg 1953), for temperature indices k_s..k_e and every stored
// ice-nuclei slice.
void freezeH2O(HostTables& t, int k_s, int k_e) {
  const Consts& c = t.c;
  const auto& Dr = t.Dr;
  const auto& Dc = t.Dc;
  const double orho_w = 1. / rho_w;
  std::vector<double> massr(nbr + 1), massc(nbc + 1);
  for (int n2 = 1; n2 <= nbr; ++n2) massr[n2] = am_r * std::pow(Dr[n2], bm_r);
  for (int n = 1; n <= nbc; ++n) massc[n] = am_r * std::pow(Dc[n], bm_r);
  const int ns = int(t.in_slices.size());
  const int nk = k_e - k_s + 1;
  if (nk <= 0) return;

  host_for(0, ns * nk - 1, [&](int idx) {
    const int s = idx / nk;
    const int k = k_s + idx % nk;
    const int m = t.in_slices[s];
    const double T_adjust = fmax2(-3.0, fmin2(3.0 - std::log10(Nt_IN[m]), 3.0));
    const double Texp = std::exp(double(k) - T_adjust * 1.0) - 1.0;
    for (int j = 1; j <= ntb_r1; ++j)
      for (int i = 1; i <= ntb_r; ++i) {
        const double lam_exp = std::pow(N0r_exp[j] * am_r * c.crg[1] / r_r[i], c.ore1);
        const double lamr = lam_exp * std::pow(c.crg[3] * c.org2 * c.org1, c.obmr);
        const double N0_r = N0r_exp[j] / (c.crg[2] * lam_exp) * std::pow(lamr, c.cre[2]);
        double sum1 = 0.0, sum2 = 0.0, sumn1 = 0.0, sumn2 = 0.0;
        for (int n2 = nbr; n2 >= 1; --n2) {
          const double N_r = N0_r * 1.0 * std::exp(-lamr * Dr[n2]) * t.dtr[n2];
          const double vol = massr[n2] * orho_w;
          const double prob = fmax2(0.0, 1.0 - std::exp(-120.0 * vol * 5.2E-4 * Texp));
          if (massr[n2] < c.xm0g) {
            sumn1 = sumn1 + prob * N_r;
            sum1 = sum1 + prob * N_r * massr[n2];
          } else {
            sumn2 = sumn2 + prob * N_r;
            sum2 = sum2 + prob * N_r * massr[n2];
          }
          if ((sum1 + sum2) >= r_r[i]) break;
        }
        const size_t id = HostTables::irfz(i, j, k, s);
        t.tpi_qrfz[id] = sum1;
        t.tni_qrfz[id] = sumn1;
        t.tpg_qrfz[id] = sum2;
        t.tnr_qrfz[id] = sumn2;
      }
    for (int j = 1; j <= nbc; ++j) {
      const int nu_c = std::min(15, nint(1000.E6 / t.t_Nc[j]) + 2);
      for (int i = 1; i <= ntb_c; ++i) {
        const double lamc = std::pow(t.t_Nc[j] * am_r * c.ccg[2][nu_c] * c.ocg1[nu_c] / r_c[i], c.obmr);
        const double N0_c = t.t_Nc[j] * c.ocg1[nu_c] * std::pow(lamc, c.cce[1][nu_c]);
        double sum1 = 0.0, sumn2 = 0.0;
        for (int n = nbc; n >= 1; --n) {
          const double vol = massc[n] * orho_w;
          const double prob = fmax2(0.0, 1.0 - std::exp(-120.0 * vol * 5.2E-4 * Texp));
          const double N_c = N0_c * powi(Dc[n], nu_c) * std::exp(-lamc * Dc[n]) * t.dtc[n];
          sumn2 = fmin2(t.t_Nc[j], sumn2 + prob * N_c);
          sum1 = sum1 + prob * N_c * massc[n];
          if (sum1 >= r_c[i]) break;
        }
        const size_t id = HostTables::icfz(i, j, k, s);
        t.tpi_qcfz[id] = sum1;
        t.tni_qcfz[id] = sumn2;
      }
    }
  });
}

//---------------------------------------------------------------------------
void qi_aut_qs(HostTables& t) {
  const Consts& c = t.c;
  for (int j = 1; j <= ntb_i1; ++j)
    for (int i = 1; i <= ntb_i; ++i) {
      const double lami = std::pow(am_i * c.cig[2] * c.oig1 * Nt_i[j] / r_i[i], c.obmi);
      const double Di_mean = (bm_i + mu_i + 1.) / lami;
      const double N0_i = Nt_i[j] * c.oig1 * std::pow(lami, c.cie[1]);
      double t1 = 0.0, t2 = 0.0;
      const size_t id = HostTables::i2(i, j, ntb_i);
      if (Di_mean > 5. * D0s) {
        t1 = r_i[i];
        t2 = Nt_i[j];
        t.tpi_ide[id] = 0.0;
      } else if (Di_mean < c.D0i) {
        t1 = 0.0;
        t2 = 0.0;
        t.tpi_ide[id] = 1.0;
      } else {
        const double xlimit_intg = lami * D0s;
        t.tpi_ide[id] = gammp(mu_i + 2.0, xlimit_intg) * 1.0;
        for (int n2 = 1; n2 <= nbi; ++n2) {
          const double N_i = N0_i * 1.0 * std::exp(-lami * t.Di[n2]) * t.dti[n2];
          if (t.Di[n2] >= D0s) {
            t1 = t1 + N_i * am_i * std::pow(t.Di[n2], bm_i);
            t2 = t2 + N_i;
          }
        }
      }
      t.tps_iaus[id] = t1;
      t.tni_iaus[id] = t2;
    }
}

//---------------------------------------------------------------------------
std::shared_ptr<const HostTables> build(double nt_c_in) {
  auto t = std::make_shared<HostTables>();
  init_constants(*t, nt_c_in, {});
  t->in_slices = {default_idx_IN(t->c)};
  init_constants(*t, nt_c_in, t->in_slices);
  table_Efrw(*t);
  table_Efsw(*t);
  table_dropEvap(*t);
  qr_acr_qg(*t, 0, ntb_r * ntb_r1 - 1);
  qr_acr_qs(*t, 0, ntb_r * ntb_r1 - 1);
  freezeH2O(*t, 1, 45);
  qi_aut_qs(*t);
  return t;
}

//---------------------------------------------------------------------------
// Disk cache
//---------------------------------------------------------------------------
namespace {

constexpr const char* kMagic = "CM1K-THOMPSON-TABLES 1\n";

std::vector<std::vector<double>*> big_tables(HostTables& t) {
  return {&t.tcg_racg, &t.tmr_racg, &t.tcr_gacr, &t.tmg_gacr, &t.tnr_racg, &t.tnr_gacr,
          &t.tcs_racs1, &t.tmr_racs1, &t.tcs_racs2, &t.tmr_racs2, &t.tcr_sacr1, &t.tms_sacr1,
          &t.tcr_sacr2, &t.tms_sacr2, &t.tnr_racs1, &t.tnr_racs2, &t.tnr_sacr1, &t.tnr_sacr2,
          &t.tpi_qrfz, &t.tni_qrfz, &t.tpg_qrfz, &t.tnr_qrfz, &t.tpi_qcfz, &t.tni_qcfz};
}

bool read_cache(const std::string& path, HostTables& t) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string magic(std::strlen(kMagic), '\0');
  f.read(magic.data(), magic.size());
  if (!f || magic != kMagic) return false;
  for (auto* v : big_tables(t)) {
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof n);
    if (!f || n != v->size()) return false;
    f.read(reinterpret_cast<char*>(v->data()), std::streamsize(n * sizeof(double)));
    if (!f) return false;
  }
  return true;
}

void write_cache(const std::string& dir, const std::string& path, HostTables& t) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return;
    f.write(kMagic, std::strlen(kMagic));
    for (auto* v : big_tables(t)) {
      const uint64_t n = v->size();
      f.write(reinterpret_cast<const char*>(&n), sizeof n);
      f.write(reinterpret_cast<const char*>(v->data()), std::streamsize(n * sizeof(double)));
    }
    if (!f) { std::filesystem::remove(tmp, ec); return; }
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) std::filesystem::remove(tmp, ec);
}

// Recompute a sample of the expensive entries and compare bit for bit.
bool validate(const HostTables& cached, double nt_c_in) {
  HostTables s;
  init_constants(s, nt_c_in, cached.in_slices);
  const int samples[] = {0, 700, ntb_r * ntb_r1 - 1};
  for (int km : samples) {
    qr_acr_qg(s, km, km);
    qr_acr_qs(s, km, km);
  }
  freezeH2O(s, 1, 1);
  freezeH2O(s, 30, 30);
  auto eq = [](const std::vector<double>& a, const std::vector<double>& b, size_t i) {
    return std::memcmp(&a[i], &b[i], sizeof(double)) == 0;
  };
  HostTables& sm = s;
  auto& cm = const_cast<HostTables&>(cached);
  const auto A = big_tables(sm), B = big_tables(cm);
  for (int km : samples) {
    const int m = km / ntb_r1 + 1, k = km % ntb_r1 + 1;
    for (int j = 1; j <= ntb_g; ++j)
      for (int i = 1; i <= ntb_g1; ++i)
        for (int q = 0; q < 6; ++q)
          if (!eq(*A[q], *B[q], HostTables::ig(i, j, k, m))) return false;
    for (int j = 1; j <= ntb_t; ++j)
      for (int i = 1; i <= ntb_s; ++i)
        for (int q = 6; q < 18; ++q)
          if (!eq(*A[q], *B[q], HostTables::is(i, j, k, m))) return false;
  }
  for (int sl = 0; sl < int(cached.in_slices.size()); ++sl)
    for (int k : {1, 30}) {
      for (int j = 1; j <= ntb_r1; ++j)
        for (int i = 1; i <= ntb_r; ++i)
          for (int q = 18; q < 22; ++q)
            if (!eq(*A[q], *B[q], HostTables::irfz(i, j, k, sl))) return false;
      for (int j = 1; j <= nbc; ++j)
        for (int i = 1; i <= ntb_c; ++i)
          for (int q = 22; q < 24; ++q)
            if (!eq(*A[q], *B[q], HostTables::icfz(i, j, k, sl))) return false;
    }
  return true;
}

} // namespace

std::string cache_dir() {
  if (const char* e = std::getenv("CM1K_THOMPSON_CACHE")) {
    const std::string s = e;
    if (s.empty() || s == "0" || s == "off" || s == "OFF") return "";
    return s;
  }
  if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/cm1k";
  if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/cm1k";
  return "";
}

std::shared_ptr<const HostTables> get(double nt_c_in) {
  static std::mutex mtx;
  static std::map<double, std::shared_ptr<const HostTables>> memo;
  std::lock_guard<std::mutex> lock(mtx);
  if (auto it = memo.find(nt_c_in); it != memo.end()) return it->second;

  // cheap part (constants and small tables) always computed
  auto t = std::make_shared<HostTables>();
  init_constants(*t, nt_c_in, {});
  t->in_slices = {default_idx_IN(t->c)};
  init_constants(*t, nt_c_in, t->in_slices);
  table_Efrw(*t);
  table_Efsw(*t);
  table_dropEvap(*t);
  qi_aut_qs(*t);

  // The expensive tables do not depend on nt_c (only on module constants).
  const std::string dir = cache_dir();
  const std::string path = dir.empty() ? "" : dir + "/thompson_tables_v1_in" + std::to_string(t->in_slices[0]) + ".bin";
  bool ok = !path.empty() && read_cache(path, *t) && validate(*t, nt_c_in);
  if (!ok) {
    qr_acr_qg(*t, 0, ntb_r * ntb_r1 - 1);
    qr_acr_qs(*t, 0, ntb_r * ntb_r1 - 1);
    freezeH2O(*t, 1, 45);
    if (!path.empty()) write_cache(dir, path, *t);
  }
  memo[nt_c_in] = t;
  return t;
}

//---------------------------------------------------------------------------
DevTables DevTables::upload(const HostTables& h) {
  DevTables d;
  d.c = h.c;
  if (h.in_slices.empty()) throw std::runtime_error("thompson: no freezeH2O slice");
  d.idx_IN0 = h.in_slices[0];
  auto up4 = [](const std::vector<double>& v, const char* name, int n0, int n1, int n2, int n3) {
    LView<double****> a(std::string(name), n0, n1, n2, n3);
    auto hm = Kokkos::create_mirror_view(a);
    std::memcpy(hm.data(), v.data(), sizeof(double) * size_t(n0) * n1 * n2 * n3);
    Kokkos::deep_copy(a, hm);
    return a;
  };
  auto up3 = [](const std::vector<double>& v, size_t off, const char* name, int n0, int n1, int n2) {
    LView<double***> a(std::string(name), n0, n1, n2);
    auto hm = Kokkos::create_mirror_view(a);
    std::memcpy(hm.data(), v.data() + off, sizeof(double) * size_t(n0) * n1 * n2);
    Kokkos::deep_copy(a, hm);
    return a;
  };
  auto up2 = [](const std::vector<double>& v, const char* name, int n0, int n1) {
    LView<double**> a(std::string(name), n0, n1);
    auto hm = Kokkos::create_mirror_view(a);
    std::memcpy(hm.data(), v.data(), sizeof(double) * size_t(n0) * n1);
    Kokkos::deep_copy(a, hm);
    return a;
  };
  d.tcg_racg = up4(h.tcg_racg, "tcg_racg", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tmr_racg = up4(h.tmr_racg, "tmr_racg", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tcr_gacr = up4(h.tcr_gacr, "tcr_gacr", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tmg_gacr = up4(h.tmg_gacr, "tmg_gacr", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tnr_racg = up4(h.tnr_racg, "tnr_racg", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tnr_gacr = up4(h.tnr_gacr, "tnr_gacr", ntb_g1, ntb_g, ntb_r1, ntb_r);
  d.tcs_racs1 = up4(h.tcs_racs1, "tcs_racs1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tmr_racs1 = up4(h.tmr_racs1, "tmr_racs1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tcs_racs2 = up4(h.tcs_racs2, "tcs_racs2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tmr_racs2 = up4(h.tmr_racs2, "tmr_racs2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tcr_sacr1 = up4(h.tcr_sacr1, "tcr_sacr1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tms_sacr1 = up4(h.tms_sacr1, "tms_sacr1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tcr_sacr2 = up4(h.tcr_sacr2, "tcr_sacr2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tms_sacr2 = up4(h.tms_sacr2, "tms_sacr2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tnr_racs1 = up4(h.tnr_racs1, "tnr_racs1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tnr_racs2 = up4(h.tnr_racs2, "tnr_racs2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tnr_sacr1 = up4(h.tnr_sacr1, "tnr_sacr1", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tnr_sacr2 = up4(h.tnr_sacr2, "tnr_sacr2", ntb_s, ntb_t, ntb_r1, ntb_r);
  d.tpi_qrfz = up3(h.tpi_qrfz, 0, "tpi_qrfz", ntb_r, ntb_r1, 45);
  d.tni_qrfz = up3(h.tni_qrfz, 0, "tni_qrfz", ntb_r, ntb_r1, 45);
  d.tpg_qrfz = up3(h.tpg_qrfz, 0, "tpg_qrfz", ntb_r, ntb_r1, 45);
  d.tnr_qrfz = up3(h.tnr_qrfz, 0, "tnr_qrfz", ntb_r, ntb_r1, 45);
  d.tpi_qcfz = up3(h.tpi_qcfz, 0, "tpi_qcfz", ntb_c, nbc, 45);
  d.tni_qcfz = up3(h.tni_qcfz, 0, "tni_qcfz", ntb_c, nbc, 45);
  d.tps_iaus = up2(h.tps_iaus, "tps_iaus", ntb_i, ntb_i1);
  d.tni_iaus = up2(h.tni_iaus, "tni_iaus", ntb_i, ntb_i1);
  d.tpi_ide = up2(h.tpi_ide, "tpi_ide", ntb_i, ntb_i1);
  d.t_Efrw = up2(h.t_Efrw, "t_Efrw", nbr, nbc);
  d.t_Efsw = up2(h.t_Efsw, "t_Efsw", nbs, nbc);
  d.tnc_wev = up3(h.tnc_wev, 0, "tnc_wev", nbc, ntb_c, nbc);
  return d;
}

} // namespace cm1k::thompson
