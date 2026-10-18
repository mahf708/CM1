// Component tests: Thompson microphysics (ptype=3) against the Fortran
// (a copy of module_mp_thompson and mp_driver_module, see
// cm1f_bridge_thomcopy.F90), bit for bit:
//   1. thompson_init: the derived constants, bins and the cheap tables in
//      full, and the three expensive tables (qr_acr_qg, qr_acr_qs,
//      freezeH2O) on a subset of their entries (the Fortran computes only
//      that subset, through its MPI decomposition hook);
//   2. mp_gt_driver (all of mp_thompson) on random columns, with the
//      Fortran's tables replaced by cm1k's (identical where both computed);
//   3. the ptype=3 branch of mp_driver, on random and reference states.
#include "physics/thompson/cm1k_thompson.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>

extern "C" {
void cm1f_thompson_setup(int ni, int nj, int nk, int ngxy, int ngz, int nrain, int testcase, int eqtset,
                         double dx, double dy, double dz);
int cm1f_thompson_init(double nt_c, int kg_s, int kg_e, int ks_s, int ks_e, int kf_s, int kf_e);
void cm1f_thompson_recompute(int which, int km_s, int km_e);
int cm1f_thompson_consts(double* buf);
int cm1f_thompson_table(int id, double* buf, int dir);
void cm1f_thompson_mp_gt_driver(double dt, double* t3d, double* p, double* w, double* dz, double* q3d,
                                double* rain, double* prate, double* vtc, double* vtr, double* vti,
                                double* vts, double* vtg);
void cm1f_thompson_mp_driver(double dt, int getvt, double* mh, double* rmh, double* mf, double* pi0,
                             double* th0, double* w3d, double* rho, double* prs, double* pp3d, double* th3d,
                             double* q3d, double* rain, double* prate, double* vt);
}

using namespace cm1k;
namespace th = cm1k::thompson;

namespace {

using H2 = FArray<2, HostSpace>;
using H3 = FArray<3, HostSpace>;
using H4 = FArray<4, HostSpace>;

constexpr double kNtc = 100.0;   // namelist nt_c (the default)

// Fortran km subsets for the expensive tables (zero-based, inclusive):
// qr_acr_qg/qs: km = (m-1)*37 + (k-1) (m: rain content, k: rain
// intercept); freezeH2O: km = (m-1)*45 + (k-1) (m: ice nuclei, k:
// temperature), here in the slice the scheme uses (idx_IN0) and across its
// boundaries with the neighbouring slices.
struct Range { int s, e; };
struct Subsets {
  std::vector<Range> qg{{0, 2}, {500, 502}, {700, 702}, {1366, 1368}};
  std::vector<Range> qs{{0, 2}, {200, 202}, {900, 902}, {1366, 1368}};
  std::vector<Range> fz;                       // set from in0
  std::vector<Range> fz_k{{1, 3}, {15, 17}, {30, 32}, {44, 45}};   // C++ temperature ranges covering fz
  int in0 = 0;                                 // idx_IN0 (from the constants)
};

// The Fortran tables are built once per process (thompson_init's guard),
// over the first ranges; the other ranges are recomputed by calling the
// table routines again.
const Subsets& fortran_init() {
  static Subsets s = [] {
    Subsets x;
    th::HostTables t;
    th::init_constants(t, kNtc, {});
    x.in0 = th::default_idx_IN(t.c);
    const int b = (x.in0 - 1) * 45;   // km of (idx_IN0, k=1)
    x.fz = {{b - 2, b + 2}, {b + 14, b + 16}, {b + 29, b + 31}, {b + 43, b + 45}};
    const int built = cm1f_thompson_init(kNtc, x.qg[0].s, x.qg[0].e, x.qs[0].s, x.qs[0].e, x.fz[0].s, x.fz[0].e);
    if (!built) throw std::runtime_error("thompson tables were already built in this process");
    for (size_t r = 1; r < x.qg.size(); ++r) cm1f_thompson_recompute(1, x.qg[r].s, x.qg[r].e);
    for (size_t r = 1; r < x.qs.size(); ++r) cm1f_thompson_recompute(2, x.qs[r].s, x.qs[r].e);
    for (size_t r = 1; r < x.fz.size(); ++r) cm1f_thompson_recompute(3, x.fz[r].s, x.fz[r].e);
    return x;
  }();
  return s;
}

// cm1k's constants packed in cm1f_thompson_consts' order.
std::vector<double> pack_consts(const th::HostTables& t) {
  const auto& c = t.c;
  std::vector<double> v;
  auto p = [&](double x) { v.push_back(x); };
  p(c.Nt_c); p(c.mu_c); p(c.Sc3); p(c.D0i); p(c.xm0s); p(c.xm0g);
  for (int j = 1; j <= 15; ++j) for (int i = 1; i <= 5; ++i) p(c.cce[i][j]);
  for (int j = 1; j <= 15; ++j) for (int i = 1; i <= 5; ++i) p(c.ccg[i][j]);
  for (int i = 1; i <= 15; ++i) p(c.ocg1[i]);
  for (int i = 1; i <= 15; ++i) p(c.ocg2[i]);
  for (int i = 1; i <= 7; ++i) p(c.cie[i]);
  for (int i = 1; i <= 7; ++i) p(c.cig[i]);
  p(c.oig1); p(c.oig2); p(c.obmi);
  for (int i = 1; i <= 13; ++i) p(c.cre[i]);
  for (int i = 1; i <= 13; ++i) p(c.crg[i]);
  p(c.ore1); p(c.org1); p(c.org2); p(c.org3); p(c.obmr);
  for (int i = 1; i <= 18; ++i) p(c.cse[i]);
  for (int i = 1; i <= 18; ++i) p(c.csg[i]);
  p(c.oams); p(c.obms); p(c.ocms);
  for (int i = 1; i <= 12; ++i) p(c.cge[i]);
  for (int i = 1; i <= 12; ++i) p(c.cgg[i]);
  p(c.oge1); p(c.ogg1); p(c.ogg2); p(c.ogg3); p(c.oamg); p(c.obmg); p(c.ocmg);
  p(c.t1_qr_qc); p(c.t1_qr_qi); p(c.t2_qr_qi); p(c.t1_qg_qc); p(c.t1_qs_qc); p(c.t1_qs_qi);
  p(c.t1_qr_ev); p(c.t2_qr_ev); p(c.t1_qs_sd); p(c.t2_qs_sd); p(c.t1_qg_sd); p(c.t2_qg_sd);
  p(c.t1_qs_me); p(c.t2_qs_me); p(c.t1_qg_me); p(c.t2_qg_me);
  for (int n : {c.nic1, c.nic2, c.nii2, c.nii3, c.nir2, c.nir3, c.nis2, c.nig2, c.nig3, c.niIN2}) p(double(n));
  for (const auto* a : {&t.Dc, &t.dtc, &t.Di, &t.dti, &t.Dr, &t.dtr, &t.Ds, &t.dts, &t.Dg, &t.dtg, &t.t_Nc})
    for (int i = 1; i <= 100; ++i) p((*a)[i]);
  return v;
}

std::vector<double> fortran_table(int id) {
  static std::vector<double> buf(37 * 100 * 45 * 55);
  const int n = cm1f_thompson_table(id, buf.data(), 0);
  return std::vector<double>(buf.begin(), buf.begin() + n);
}

bool same(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Number of entries (halos included) that are not bit-identical.
template <int R>
size_t count_diff(const std::string& name, const FArray<R, HostSpace>& a, const FArray<R, HostSpace>& f) {
  size_t n = 0;
  double maxabs = 0;
  for (size_t m = 0; m < a.size(); ++m) {
    const double x = a.data()[m], y = f.data()[m];
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) {
      ++n;
      maxabs = std::max(maxabs, std::fabs(x - y));
    }
  }
  if (n) UNSCOPED_INFO(name << ": " << n << " entries differ, max abs diff " << maxabs);
  return n;
}

template <int R>
FArray<R> to_dev(const FArray<R, HostSpace>& h) {
  std::array<int, R> lb, ub;
  for (int d = 0; d < R; ++d) { lb[d] = h.lb(d); ub[d] = h.ub(d); }
  FArray<R> a(h.label(), lb, ub);
  a.deep_copy_from(h);
  return a;
}

// Replace the Fortran copy's expensive tables (only partly computed there)
// by cm1k's, so mp_thompson runs on identical tables in both languages.
void install_tables_in_fortran(const th::HostTables& h) {
  static bool done = false;
  if (done) return;
  fortran_init();
  const std::vector<double>* racg[6] = {&h.tcg_racg, &h.tmr_racg, &h.tcr_gacr, &h.tmg_gacr, &h.tnr_racg, &h.tnr_gacr};
  const std::vector<double>* racs[12] = {&h.tcs_racs1, &h.tmr_racs1, &h.tcs_racs2, &h.tmr_racs2,
                                         &h.tcr_sacr1, &h.tms_sacr1, &h.tcr_sacr2, &h.tms_sacr2,
                                         &h.tnr_racs1, &h.tnr_racs2, &h.tnr_sacr1, &h.tnr_sacr2};
  const std::vector<double>* fz[6] = {&h.tpi_qrfz, &h.tni_qrfz, &h.tpg_qrfz, &h.tnr_qrfz, &h.tpi_qcfz, &h.tni_qcfz};
  for (int n = 0; n < 6; ++n) cm1f_thompson_table(1 + n, const_cast<double*>(racg[n]->data()), 1);
  for (int n = 0; n < 12; ++n) cm1f_thompson_table(7 + n, const_cast<double*>(racs[n]->data()), 1);
  for (int n = 0; n < 6; ++n) {
    // freezeH2O: cm1k keeps only the slices in h.in_slices
    auto f = fortran_table(19 + n);
    const size_t slice = f.size() / 55;
    for (size_t s = 0; s < h.in_slices.size(); ++s)
      std::memcpy(f.data() + slice * (h.in_slices[s] - 1), fz[n]->data() + slice * s, slice * sizeof(double));
    cm1f_thompson_table(19 + n, f.data(), 1);
  }
  done = true;
}

Bounds make_bounds(int ni, int nj, int nk, int ngxy) {
  Bounds b;
  b.nx = b.ni = ni; b.ny = b.nj = nj; b.nz = b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = ngxy; b.ngz = 1;
  b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.numq = 8; b.nrain = 2;
  return b;
}

// Everything mp_driver reads or writes, on the host.
struct ThState {
  Bounds b;
  Real dx = 1000, dy = 1000, dz = 300, dt = 10;
  H3 mh, rmh, mf, pi0, th0, w3d, rho, prs, pp3d, th3d, rain;
  H4 q;

  void allocate() {
    auto s3 = [&](const char* n) { return H3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
    auto f3 = [&](const char* n) { return H3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1}); };
    mh = s3("mh"); rmh = s3("rmh"); pi0 = s3("pi0"); th0 = s3("th0"); rho = s3("rho"); prs = s3("prs");
    pp3d = s3("pp3d"); th3d = s3("th3d");
    mf = f3("mf"); w3d = f3("w3d");
    rain = H3("rain", {b.ib, b.jb, 1}, {b.ie, b.je, b.nrain});
    q = H4("q", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, b.numq});
  }

  ThState clone() const {
    ThState c = *this;
    c.allocate();
    for (auto [d, s] : {std::pair{&c.mh, &mh}, {&c.rmh, &rmh}, {&c.mf, &mf}, {&c.pi0, &pi0}, {&c.th0, &th0},
                        {&c.w3d, &w3d}, {&c.rho, &rho}, {&c.prs, &prs}, {&c.pp3d, &pp3d}, {&c.th3d, &th3d},
                        {&c.rain, &rain}})
      Kokkos::deep_copy(d->view(), s->view());
    Kokkos::deep_copy(c.q.view(), q.view());
    return c;
  }
};

// Random but physically plausible state: a stable sounding (300 K at the
// ground, about 200 K at 15 km) with random perturbations; each column picks
// a regime (clear, warm rain, deep convection with all species, stratiform
// snow over a melting layer, ice-supersaturated without ice), and some
// values are zero, tiny (below the scheme's R1/R2 thresholds), without
// their number concentration, or (when `negatives`) slightly negative for
// pdefq.
ThState random_state(int ni, int nj, int nk, Real dz, Real dt, unsigned seed, bool negatives) {
  ThState s;
  s.b = make_bounds(ni, nj, nk, 2);
  s.dz = dz; s.dt = dt;
  s.allocate();
  const auto c = Constants::make(0);
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  auto logu = [&](double a, double b) { return a * std::pow(b / a, U(gen)); };
  const auto& b = s.b;
  for (int k = b.kb; k <= b.ke + 1; ++k)
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        s.mf(i, j, k) = 0.95 + 0.1 * U(gen);
        s.w3d(i, j, k) = -4.0 + 14.0 * U(gen);
      }
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) {
      const int regime = int(5 * U(gen));
      for (int nr = 1; nr <= b.nrain; ++nr) s.rain(i, j, nr) = U(gen) < 0.5 ? 0.0 : 0.3 * U(gen);
      const double wscale = regime == 2 ? 1.0 : 0.2;
      for (int k = b.kb; k <= b.ke + 1; ++k) s.w3d(i, j, k) *= wscale;
      for (int k = b.kb; k <= b.ke; ++k) {
        const double z = (k - 0.5) * dz;
        s.mh(i, j, k) = 0.95 + 0.1 * U(gen);
        s.rmh(i, j, k) = 1.0 / s.mh(i, j, k);
        s.th0(i, j, k) = 300.0 + 0.004 * z;
        s.pi0(i, j, k) = 1.0 - c.g * z / (c.cp * (300.0 + 0.002 * z));
        s.th3d(i, j, k) = 4.0 * (U(gen) - 0.5);
        s.pp3d(i, j, k) = 2.0e-3 * (U(gen) - 0.5);
        const double pii = s.pi0(i, j, k) + s.pp3d(i, j, k);
        const double t = (s.th0(i, j, k) + s.th3d(i, j, k)) * pii;
        const double p = phys::p00 * std::pow(pii, c.cpdrd);
        const double tc = t - 273.15;
        double esl = 611.2 * std::exp(17.67 * tc / (t - 29.65));
        esl = std::min(esl, 0.5 * p);
        const double qvs = c.eps * esl / (p - esl);
        const double esi = std::min(611.2 * std::exp(21.8745584 * tc / (t - 7.66)), 0.5 * p);
        const double qvi = c.eps * esi / (p - esi);
        double qv = 0, qc = 0, qr = 0, qi = 0, qs = 0, qg = 0, nci = 0, ncr = 0;
        switch (regime) {
          case 0: qv = (0.2 + 0.7 * U(gen)) * qvs; break;
          case 1:
            qv = (0.9 + 0.15 * U(gen)) * qvs;
            if (tc > -10 && U(gen) < 0.7) qc = 2.0e-3 * U(gen);
            if (U(gen) < 0.6) qr = logu(1e-7, 5e-3);
            break;
          case 2:
            qv = (0.95 + 0.15 * U(gen)) * qvs;
            if (tc > -38 && U(gen) < 0.7) qc = 2.5e-3 * U(gen);
            if (U(gen) < 0.7) qr = logu(1e-7, 6e-3);
            if (tc < 0 && U(gen) < 0.6) qi = logu(1e-8, 1e-3);
            if (tc < 5 && U(gen) < 0.6) qs = logu(1e-8, 3e-3);
            if (U(gen) < 0.6) qg = logu(1e-8, 6e-3);
            break;
          case 3:
            qv = (0.8 + 0.25 * U(gen)) * (tc > 0 ? qvs : qvi);
            if (tc < 0 && U(gen) < 0.5) qi = logu(1e-8, 3e-4);
            if (tc < 6 && U(gen) < 0.8) qs = logu(1e-7, 2e-3);
            if (tc > -2 && U(gen) < 0.8) qr = logu(1e-7, 2e-3);
            if (tc > -15 && tc < 0 && U(gen) < 0.3) qc = 3e-4 * U(gen);
            break;
          case 4:
            qv = (tc < -5 ? (1.0 + 0.3 * U(gen)) * qvi : (0.7 + 0.3 * U(gen)) * qvs);
            qv = std::min(qv, 1.02 * qvs);
            if (tc < -40 && U(gen) < 0.3) qi = logu(1e-9, 1e-5);
            break;
        }
        if (qi > 0) nci = U(gen) < 0.1 ? 0.0 : logu(1e2, 1e7);
        if (qr > 0) ncr = U(gen) < 0.1 ? 0.0 : logu(1e1, 1e6);
        const double r = U(gen);
        if (r < 0.03) qc = 1e-13 * U(gen);
        else if (r < 0.06) { qr = 1e-13 * U(gen); ncr = logu(1e-3, 1e3); }
        else if (r < 0.09) { qi = 1e-13 * U(gen); nci = logu(1e-3, 1e3); }
        else if (r < 0.11) qs = 1e-13 * U(gen);
        else if (r < 0.13) qg = 1e-13 * U(gen);
        else if (r < 0.15) { ncr = 0.0; nci = 0.0; }
        if (negatives) {
          const double r2 = U(gen);
          if (r2 < 0.03) qc = -1e-10 * U(gen);
          else if (r2 < 0.06) qr = -1e-10 * U(gen);
          else if (r2 < 0.08) qi = -1e-10 * U(gen);
          else if (r2 < 0.10) qs = 1e-26 * U(gen);
          else if (r2 < 0.12) qg = -1e-26 * U(gen);
        }
        const double qq[8] = {qv, qc, qr, qi, qs, qg, nci, ncr};
        for (int n = 0; n < 8; ++n) s.q(i, j, k, n + 1) = qq[n];
        s.prs(i, j, k) = p;
        s.rho(i, j, k) = p / (c.rd * t * (1.0 + qv * c.reps));
      }
    }
  return s;
}

void setup_fortran(const ThState& s, int eqtset) {
  cm1f_thompson_setup(s.b.ni, s.b.nj, s.b.nk, s.b.ngxy, s.b.ngz, s.b.nrain, 0, eqtset, s.dx, s.dy, s.dz);
}

struct Changed {
  size_t q[9] = {};   // per species, 1-based
  double rain_max = 0;
};

// The full ptype=3 mp_driver path in both languages; returns the number of
// differing entries.
size_t run_mp_driver(const ThState& init, int eqtset, bool getvt, Changed* ch = nullptr) {
  const auto tab = th::get(kNtc);
  install_tables_in_fortran(*tab);
  ThState f = init.clone(), x = init.clone();
  setup_fortran(f, eqtset);
  const auto& b = f.b;
  H2 pratef("prate", {b.ib, b.jb}, {b.ie, b.je});
  H4 vtf("vt", {b.ib, b.jb, b.kb, 1}, {b.ie, b.je, b.ke, 5});
  cm1f_thompson_mp_driver(f.dt, getvt ? 1 : 0, f.mh.data(), f.rmh.data(), f.mf.data(), f.pi0.data(),
                          f.th0.data(), f.w3d.data(), f.rho.data(), f.prs.data(), f.pp3d.data(),
                          f.th3d.data(), f.q.data(), f.rain.data(), pratef.data(), vtf.data());

  th::Options opt;
  opt.eqtset = eqtset;
  opt.nt_c = kNtc;
  th::Work w;
  w.allocate(x.b, opt);
  auto rho = to_dev(x.rho), prs = to_dev(x.prs), pp3d = to_dev(x.pp3d), th3d = to_dev(x.th3d);
  auto q = to_dev(x.q);
  auto rain = to_dev(x.rain);
  th::mp_driver(opt, Constants::make(0), x.b, x.dt, x.dz, to_dev(x.rmh), to_dev(x.pi0), to_dev(x.th0),
                to_dev(x.w3d), rho, prs, pp3d, th3d, q, rain, w);
  x.rho = rho.to_host(); x.prs = prs.to_host(); x.pp3d = pp3d.to_host(); x.th3d = th3d.to_host();
  x.q = q.to_host(); x.rain = rain.to_host();

  size_t n = 0;
  n += count_diff("th3d", x.th3d, f.th3d);
  n += count_diff("pp3d", x.pp3d, f.pp3d);
  n += count_diff("prs", x.prs, f.prs);
  n += count_diff("rho", x.rho, f.rho);
  n += count_diff("q", x.q, f.q);
  n += count_diff("rain", x.rain, f.rain);
  // prate and the fall speeds: interior only (the Fortran leaves halos alone)
  const auto prate = w.prate.to_host();
  const FArray3 vt[5] = {w.vtc, w.vtr, w.vti, w.vts, w.vtg};
  std::vector<H3> vth;
  for (int m = 0; m < 5; ++m) vth.push_back(vt[m].to_host());
  size_t np = 0, nv = 0;
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i) {
      np += !same(prate(i, j), pratef(i, j));
      if (getvt)
        for (int k = 1; k <= b.nk; ++k)
          for (int m = 0; m < 5; ++m) nv += !same(vth[m](i, j, k), vtf(i, j, k, m + 1));
    }
  if (np) UNSCOPED_INFO("prate: " << np << " entries differ");
  if (nv) UNSCOPED_INFO("vt: " << nv << " entries differ");
  n += np + nv;

  if (ch) {
    for (int m = 1; m <= 8; ++m)
      for (int k = 1; k <= b.nk; ++k)
        for (int j = 1; j <= b.nj; ++j)
          for (int i = 1; i <= b.ni; ++i) ch->q[m] += !same(x.q(i, j, k, m), init.q(i, j, k, m));
    for (int j = 1; j <= b.nj; ++j)
      for (int i = 1; i <= b.ni; ++i) ch->rain_max = std::max(ch->rain_max, x.rain(i, j, 1) - init.rain(i, j, 1));
  }
  return n;
}

} // namespace

TEST_CASE("thompson_init constants and tables match thompson.F", "[bridge][thompson]") {
  const Subsets& sub = fortran_init();

  th::HostTables t;
  th::init_constants(t, kNtc, {sub.in0 - 1, sub.in0, sub.in0 + 1});
  REQUIRE(th::default_idx_IN(t.c) == sub.in0);

  SECTION("constants and bins") {
    std::vector<double> f(4096);
    const int n = cm1f_thompson_consts(f.data());
    const auto x = pack_consts(t);
    REQUIRE(size_t(n) == x.size());
    size_t nd = 0;
    for (int m = 0; m < n; ++m)
      if (!same(x[m], f[m])) {
        ++nd;
        UNSCOPED_INFO("constant #" << m << ": cm1k " << x[m] << " Fortran " << f[m]);
      }
    REQUIRE(nd == 0);
  }

  SECTION("table_Efrw, table_Efsw, table_dropEvap, qi_aut_qs (whole tables)") {
    th::table_Efrw(t);
    th::table_Efsw(t);
    th::table_dropEvap(t);
    th::qi_aut_qs(t);
    struct T { int id; const std::vector<double>* v; const char* name; };
    for (const T& e : {T{25, &t.tps_iaus, "tps_iaus"}, T{26, &t.tni_iaus, "tni_iaus"}, T{27, &t.tpi_ide, "tpi_ide"},
                       T{28, &t.t_Efrw, "t_Efrw"}, T{29, &t.t_Efsw, "t_Efsw"}, T{30, &t.tpc_wev, "tpc_wev"},
                       T{31, &t.tnc_wev, "tnc_wev"}}) {
      const auto f = fortran_table(e.id);
      REQUIRE(f.size() == e.v->size());
      size_t nd = 0, nz = 0;
      for (size_t m = 0; m < f.size(); ++m) {
        nd += !same((*e.v)[m], f[m]);
        nz += f[m] != 0.0;
      }
      INFO(e.name << ": " << nd << " of " << f.size() << " differ");
      CHECK(nd == 0);
      CHECK(nz > 0);
    }
  }

  SECTION("qr_acr_qg, qr_acr_qs, freezeH2O (Fortran-computed subset)") {
    th::table_Efrw(t);
    th::table_Efsw(t);
    for (const Range& r : sub.qg) th::qr_acr_qg(t, r.s, r.e);
    for (const Range& r : sub.qs) th::qr_acr_qs(t, r.s, r.e);
    for (const Range& r : sub.fz_k) th::freezeH2O(t, r.s, r.e);

    using th::HostTables;
    size_t ncmp = 0, nnz = 0;
    auto cmp = [&](int id, const std::vector<double>& v, auto index_c, auto index_f, const char* name,
                   const std::vector<Range>& ranges, int n1, int n2, int div) {
      const auto f = fortran_table(id);
      size_t nd = 0;
      for (const Range& r : ranges)
      for (int km = r.s; km <= r.e; ++km) {
        const int m = km / div + 1, k = km % div + 1;
        for (int j = 1; j <= n2; ++j)
          for (int i = 1; i <= n1; ++i) {
            const double x = v[index_c(i, j, k, m)], y = f[index_f(i, j, k, m)];
            nd += !same(x, y);
            nnz += y != 0.0;
            ++ncmp;
          }
      }
      INFO(name << ": " << nd << " differ");
      CHECK(nd == 0);
    };
    auto ig = [](int i, int j, int k, int m) { return HostTables::ig(i, j, k, m); };
    auto is = [](int i, int j, int k, int m) { return HostTables::is(i, j, k, m); };
    const auto& sl = t.in_slices;
    auto slot = [&](int m) { return int(std::find(sl.begin(), sl.end(), m) - sl.begin()); };
    auto irc = [&](int i, int j, int k, int m) { return HostTables::irfz(i, j, k, slot(m)); };
    auto irf = [](int i, int j, int k, int m) {
      return size_t(i - 1) + 37 * (size_t(j - 1) + 37 * (size_t(k - 1) + 45 * size_t(m - 1)));
    };
    auto icc = [&](int i, int j, int k, int m) { return HostTables::icfz(i, j, k, slot(m)); };
    auto icf = [](int i, int j, int k, int m) {
      return size_t(i - 1) + 37 * (size_t(j - 1) + 100 * (size_t(k - 1) + 45 * size_t(m - 1)));
    };
    const std::vector<double>* racg[6] = {&t.tcg_racg, &t.tmr_racg, &t.tcr_gacr, &t.tmg_gacr, &t.tnr_racg, &t.tnr_gacr};
    const char* racg_n[6] = {"tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg", "tnr_gacr"};
    for (int n = 0; n < 6; ++n) cmp(1 + n, *racg[n], ig, ig, racg_n[n], sub.qg, 37, 37, 37);
    const std::vector<double>* racs[12] = {&t.tcs_racs1, &t.tmr_racs1, &t.tcs_racs2, &t.tmr_racs2,
                                           &t.tcr_sacr1, &t.tms_sacr1, &t.tcr_sacr2, &t.tms_sacr2,
                                           &t.tnr_racs1, &t.tnr_racs2, &t.tnr_sacr1, &t.tnr_sacr2};
    const char* racs_n[12] = {"tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1", "tms_sacr1",
                              "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"};
    for (int n = 0; n < 12; ++n) cmp(7 + n, *racs[n], is, is, racs_n[n], sub.qs, 37, 9, 37);
    cmp(19, t.tpi_qrfz, irc, irf, "tpi_qrfz", sub.fz, 37, 37, 45);
    cmp(20, t.tni_qrfz, irc, irf, "tni_qrfz", sub.fz, 37, 37, 45);
    cmp(21, t.tpg_qrfz, irc, irf, "tpg_qrfz", sub.fz, 37, 37, 45);
    cmp(22, t.tnr_qrfz, irc, irf, "tnr_qrfz", sub.fz, 37, 37, 45);
    cmp(23, t.tpi_qcfz, icc, icf, "tpi_qcfz", sub.fz, 37, 100, 45);
    cmp(24, t.tni_qcfz, icc, icf, "tni_qcfz", sub.fz, 37, 100, 45);
    INFO(ncmp << " entries compared, " << nnz << " nonzero");
    CHECK(nnz > ncmp / 4);
  }

  SECTION("cached tables equal a fresh build") {
    // th::get() (memory/disk cache) against the recomputed pieces above
    const auto g = th::get(kNtc);
    th::table_Efrw(t);
    th::table_Efsw(t);
    th::table_dropEvap(t);
    th::qi_aut_qs(t);
    th::qr_acr_qg(t, 0, 2);
    th::qr_acr_qg(t, 1366, 1368);
    th::qr_acr_qs(t, 0, 2);
    th::qr_acr_qs(t, 1366, 1368);
    th::freezeH2O(t, 1, 2);
    th::freezeH2O(t, 44, 45);
    REQUIRE(g->in_slices.size() == 1);
    REQUIRE(g->in_slices[0] == sub.in0);
    REQUIRE(std::memcmp(g->t_Efrw.data(), t.t_Efrw.data(), t.t_Efrw.size() * 8) == 0);
    REQUIRE(std::memcmp(g->tnc_wev.data(), t.tnc_wev.data(), t.tnc_wev.size() * 8) == 0);
    REQUIRE(std::memcmp(g->tps_iaus.data(), t.tps_iaus.data(), t.tps_iaus.size() * 8) == 0);
    for (int km : {0, 1, 2, 1366, 1367, 1368}) {
      const size_t o = size_t(km) * 37 * 37, os = size_t(km) * 37 * 9;
      CHECK(std::memcmp(g->tcg_racg.data() + o, t.tcg_racg.data() + o, 37 * 37 * 8) == 0);
      CHECK(std::memcmp(g->tnr_gacr.data() + o, t.tnr_gacr.data() + o, 37 * 37 * 8) == 0);
      CHECK(std::memcmp(g->tcs_racs2.data() + os, t.tcs_racs2.data() + os, 37 * 9 * 8) == 0);
      CHECK(std::memcmp(g->tnr_sacr1.data() + os, t.tnr_sacr1.data() + os, 37 * 9 * 8) == 0);
    }
    const size_t sl = 37 * 37 * 45, s1 = 1;   // slot of in0 in t.in_slices (in0-1, in0, in0+1)
    CHECK(std::memcmp(g->tpi_qrfz.data(), t.tpi_qrfz.data() + sl * s1, 37 * 37 * 2 * 8) == 0);
  }
}

TEST_CASE("thompson mp_gt_driver matches thompson.F on random columns", "[bridge][thompson]") {
  const auto tab = th::get(kNtc);
  install_tables_in_fortran(*tab);
  unsigned seed = 7;
  size_t changed[9] = {};
  for (int eqtset : {1, 2})
    for (Real dt : {6.0, 30.0, 90.0}) {   // 90 s: sedimentation substeps
      INFO("eqtset " << eqtset << " dt " << dt);
      const ThState s = random_state(9, 7, 50, 300.0, dt, seed++, false);
      setup_fortran(s, eqtset);
      const auto& b = s.b;
      auto mk3 = [&](const char* n) { return H3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
      H3 t = mk3("t"), dzq = mk3("dz");
      for (int k = b.kb; k <= b.ke; ++k)
        for (int j = b.jb; j <= b.je; ++j)
          for (int i = b.ib; i <= b.ie; ++i) {
            t(i, j, k) = (s.th0(i, j, k) + s.th3d(i, j, k)) * (s.pi0(i, j, k) + s.pp3d(i, j, k));
            dzq(i, j, k) = s.dz * s.rmh(i, j, k);
          }
      // Fortran
      ThState f = s.clone();
      H3 tf = mk3("tf");
      Kokkos::deep_copy(tf.view(), t.view());
      H2 pratef("prate", {b.ib, b.jb}, {b.ie, b.je});
      H3 vf[5] = {mk3("vtc"), mk3("vtr"), mk3("vti"), mk3("vts"), mk3("vtg")};
      cm1f_thompson_mp_gt_driver(dt, tf.data(), f.prs.data(), f.w3d.data(), dzq.data(), f.q.data(), f.rain.data(),
                                 pratef.data(), vf[0].data(), vf[1].data(), vf[2].data(), vf[3].data(),
                                 vf[4].data());
      // cm1k
      th::Options opt;
      opt.eqtset = eqtset;
      opt.nt_c = kNtc;
      th::Work w;
      w.allocate(b, opt);
      auto td = to_dev(t), rain = to_dev(s.rain);
      auto q = to_dev(s.q);
      th::mp_gt_driver(opt, b, dt, td, to_dev(s.prs), to_dev(s.w3d), to_dev(dzq), q, rain, w);
      const auto qx = q.to_host();
      const auto tx = td.to_host(), rx = rain.to_host();
      const auto px = w.prate.to_host();
      const FArray3 vt[5] = {w.vtc, w.vtr, w.vti, w.vts, w.vtg};

      size_t nd = 0;
      nd += count_diff("q", qx, f.q);
      nd += count_diff("t", tx, tf);
      nd += count_diff("rain", rx, f.rain);
      for (int m = 0; m < 5; ++m) {
        const auto vh = vt[m].to_host();
        for (int k = 1; k <= b.nk; ++k)
          for (int j = 1; j <= b.nj; ++j)
            for (int i = 1; i <= b.ni; ++i) nd += !same(vh(i, j, k), vf[m](i, j, k));
      }
      for (int j = 1; j <= b.nj; ++j)
        for (int i = 1; i <= b.ni; ++i) nd += !same(px(i, j), pratef(i, j));
      REQUIRE(nd == 0);
      for (int m = 1; m <= 8; ++m)
        for (int k = 1; k <= b.nk; ++k)
          for (int j = 1; j <= b.nj; ++j)
            for (int i = 1; i <= b.ni; ++i) changed[m] += !same(qx(i, j, k, m), s.q(i, j, k, m));
    }
  // every species is exercised
  for (int m = 1; m <= 8; ++m) {
    INFO("species " << m);
    CHECK(changed[m] > 100);
  }
}

TEST_CASE("thompson mp_driver matches mp_driver.F on random states", "[bridge][thompson]") {
  unsigned seed = 1000;
  for (int eqtset : {1, 2})
    for (bool getvt : {false, true})
      for (Real dt : {10.0, 60.0}) {
        INFO("eqtset " << eqtset << " getvt " << getvt << " dt " << dt);
        const ThState s = random_state(8, 6, 50, 300.0, dt, seed++, true);
        Changed ch;
        REQUIRE(run_mp_driver(s, eqtset, getvt, &ch) == 0);
        for (int m = 1; m <= 8; ++m) CHECK(ch.q[m] > 0);
        CHECK(ch.rain_max > 0);
      }
}

// States from the Fortran reference runs: tha/ppi/qa as th3d/pp3d/q3d (after
// a full time step, so realistic inputs for the next microphysics call),
// plus rho, prs, rain, w and the grid/base state.
TEST_CASE("thompson mp_driver matches mp_driver.F on reference states", "[bridge][thompson][ref]") {
  struct C { const char* name; std::vector<int> steps; };
  for (const C& cs : {C{"thompson_bubble", {0, 30, 60, 90, 120}}}) {
    const auto ref = testing::case_ref(cs.name);
    if (ref.empty()) continue;
    const auto nl = Namelist::from_file(testing::case_namelist(cs.name));
    size_t nchanged_total = 0;
    for (int step : cs.steps) {
      const auto dir = DumpReader::step_dir(ref, step);
      if (!std::filesystem::exists(dir)) { WARN("missing " << dir); continue; }
      DumpReader d(dir);
      ThState s;
      s.b = Bounds::from_namelist(nl);
      s.dx = nl.getr("dx", 0); s.dy = nl.getr("dy", 0); s.dz = nl.getr("dz", 0); s.dt = nl.getr("dtl", 0);
      s.allocate();
      auto rd = [&](const std::string& n, H3& a) {
        auto h = d.read_host<3>(n);
        REQUIRE(h.size() == a.size());
        Kokkos::deep_copy(a.view(), h.view());
      };
      rd("mh", s.mh); rd("rmh", s.rmh); rd("mf", s.mf); rd("pi0", s.pi0); rd("th0", s.th0); rd("wa", s.w3d);
      rd("rho", s.rho); rd("prs", s.prs); rd("ppi", s.pp3d); rd("tha", s.th3d); rd("rain", s.rain);
      const char* qn[8] = {"qa_qv", "qa_qc", "qa_qr", "qa_qi", "qa_qs", "qa_qg", "qa_nci", "qa_ncr"};
      for (int n = 0; n < 8; ++n) {
        auto h = d.read_host<3>(qn[n]);
        Kokkos::deep_copy(Kokkos::subview(s.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), h.view());
      }
      const auto opt = th::Options::from_namelist(nl);
      INFO(cs.name << " step " << step << " eqtset " << opt.eqtset);
      Changed ch;
      REQUIRE(run_mp_driver(s, opt.eqtset, true, &ch) == 0);
      for (int m = 1; m <= 8; ++m) nchanged_total += ch.q[m];
      REQUIRE(run_mp_driver(s, 3 - opt.eqtset, false) == 0);
    }
    CHECK(nchanged_total > 0);
  }
}
