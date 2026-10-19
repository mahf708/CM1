// Component tests: Morrison two-moment microphysics (ptype=5) against the
// Fortran, bit for bit: GRAUPEL_INIT's constants, GAMMA, DERF1, satadj2 and
// MP_GRAUPEL (through a public copy of module_mp_GRAUPEL, which also runs
// the INUM=0 path), and the full mp_driver (real module, INUM=1) on random
// columns and on states from the morrison_* reference runs.
#include "physics/morrison/cm1k_morrison.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <random>

extern "C" {
void cm1f_morrison_setup(int ni, int nj, int nk, int ngxy, int ngz, int nrain, int eqtset, int efall, int radopt,
                         int ihail, double ndcnst, int inum, double dx, double dy, double dz);
void cm1f_morrison_mp_driver(double dt, int dorad, double* mh, double* rmh, double* mf, double* pi0, double* th0,
                             double* rho, double* prs, double* pp3d, double* th3d, double* w3d, double* q3d,
                             double* rain, double* prate, double* vt, double* eff);
void cm1f_morrison_params(double* v);
double cm1f_morrison_gamma(double x);
double cm1f_morrison_derf1(double x);
void cm1f_morrison_satadj2(double* tnew, double* prs, double* qv, double* qc, double qr, double qi, double qs,
                           double qg, double rho, int eqtset, double tsmall);
void cm1f_morrison_mp_graupel(double dt, int dorad, double* t, double* q3d, double* p, double* rho, double* dzq,
                              double* w, double* rain, double* prate, double* vt, double* eff);
}

using namespace cm1k;
namespace mo = cm1k::morrison;

namespace {

using H2 = FArray<2, HostSpace>;
using H3 = FArray<3, HostSpace>;
using H4 = FArray<4, HostSpace>;

std::vector<Real> params_vec(const mo::Params& p) {
  std::vector<Real> v = {p.ai, p.ac, p.as, p.ar, p.ag, p.bi, p.bc, p.bs, p.br, p.bg,
                         p.r, p.rv, p.cp, p.ep_2, p.cv, p.cvv, p.cpv, p.cpl, p.cpi, p.xlv0, p.xls0, p.lv1, p.lv2,
                         p.ls1, p.ls2,
                         p.tmelt, p.rhosu, p.rhow, p.rhoi, p.rhosn, p.rhog, p.aimm, p.bimm, p.ecr, p.dcs, p.mi0,
                         p.mg0,
                         p.f1s, p.f2s, p.f1r, p.f2r, p.g, p.qsmall, p.qthr, p.eii, p.eci, p.cpw,
                         p.ci, p.di, p.cs, p.ds, p.cg, p.dg, p.rin, p.mmult,
                         p.lammaxi, p.lammini, p.lammaxr, p.lamminr, p.d_b, p.lammaxs, p.lammins, p.lammaxg,
                         p.lamming,
                         p.k1, p.c1, p.mw, p.osm, p.vi, p.epsm, p.rhoa, p.map, p.ma, p.rr, p.bact,
                         p.rm1, p.sig1, p.nanew1, p.f11, p.f21, p.rm2, p.sig2, p.nanew2, p.f12, p.f22};
  for (int n = 1; n <= 41; ++n) v.push_back(p.cons[n]);
  v.push_back(p.inum);
  v.push_back(p.ndcnst);
  v.push_back(p.ihail);
  return v;
}

// Everything mp_driver / MP_GRAUPEL reads or writes, on the host.
struct State {
  Bounds b;
  Real dx = 1000, dy = 1000, dz = 400, dt = 6;
  H3 mh, rmh, mf, pi0, th0, rho, prs, pp3d, th3d, w3d, t, dzq;
  H4 q, vt, eff;
  H3 rain;
  H2 prate;

  void allocate() {
    auto s3 = [&](const char* n) { return H3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
    mh = s3("mh"); rmh = s3("rmh"); pi0 = s3("pi0"); th0 = s3("th0"); rho = s3("rho"); prs = s3("prs");
    pp3d = s3("pp3d"); th3d = s3("th3d"); t = s3("t"); dzq = s3("dzq");
    mf = H3("mf", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
    w3d = H3("w3d", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
    rain = H3("rain", {b.ib, b.jb, 1}, {b.ie, b.je, b.nrain});
    q = H4("q", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, b.numq});
    vt = H4("vt", {b.ib, b.jb, b.kb, 1}, {b.ie, b.je, b.ke, 5});
    eff = H4("eff", {b.ib, b.jb, b.kb, 1}, {b.ie, b.je, b.ke, 6});
    prate = H2("prate", {b.ib, b.jb}, {b.ie, b.je});
  }

  State clone() const {
    State c = *this;
    c.allocate();
    for (auto [d, s] : {std::pair{&c.mh, &mh}, {&c.rmh, &rmh}, {&c.mf, &mf}, {&c.pi0, &pi0}, {&c.th0, &th0},
                        {&c.rho, &rho}, {&c.prs, &prs}, {&c.pp3d, &pp3d}, {&c.th3d, &th3d}, {&c.w3d, &w3d},
                        {&c.t, &t}, {&c.dzq, &dzq}, {&c.rain, &rain}})
      Kokkos::deep_copy(d->view(), s->view());
    Kokkos::deep_copy(c.q.view(), q.view());
    Kokkos::deep_copy(c.vt.view(), vt.view());
    Kokkos::deep_copy(c.eff.view(), eff.view());
    Kokkos::deep_copy(c.prate.view(), prate.view());
    return c;
  }
};

Bounds make_bounds(int ni, int nj, int nk, int ngxy, int numq) {
  Bounds b;
  b.nx = b.ni = ni; b.ny = b.nj = nj; b.nz = b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = ngxy; b.ngz = 1;
  b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.numq = numq; b.nrain = 1;
  return b;
}

// Random, physically plausible columns through the depth of a deep
// convective storm (surface to ~16 km, 300 K to ~210 K). Each column picks
// a regime: clear, cloudy (liquid only), mixed phase, ice aloft, heavy
// precipitation (rain + snow + graupel), or all species everywhere; numbers
// are drawn around the masses (sometimes zero with mass, or with no mass),
// and a few values are slightly negative or below qsmall so pdefq and the
// scheme's limiters have work to do.
State random_state(int ni, int nj, int nk, Real dz, Real dt, int numq, unsigned seed) {
  State s;
  s.b = make_bounds(ni, nj, nk, 3, numq);
  s.dz = dz; s.dt = dt;
  s.allocate();
  const auto c = Constants::make(0);
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  auto logu = [&](double lo, double hi) { return lo * std::pow(hi / lo, U(gen)); };
  const auto& b = s.b;
  for (int k = b.kb; k <= b.ke + 1; ++k)
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        s.mf(i, j, k) = 0.9 + 0.2 * U(gen);
        s.w3d(i, j, k) = -4.0 + 16.0 * U(gen);
      }
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) {
      const int regime = int(6 * U(gen));
      for (int nr = 1; nr <= b.nrain; ++nr) s.rain(i, j, nr) = U(gen) < 0.5 ? 0.0 : 0.3 * U(gen);
      for (int k = b.kb; k <= b.ke; ++k) {
        const double z = (k - 0.5) * dz;
        s.mh(i, j, k) = 0.9 + 0.2 * U(gen);
        s.rmh(i, j, k) = 1.0 / s.mh(i, j, k);
        // ~6.5 K/km lapse rate to 12 km, isothermal above
        const double tz = 300.0 - 0.0065 * std::min(z, 12000.0);
        s.th0(i, j, k) = 300.0 + 0.0035 * z;
        s.pi0(i, j, k) = tz / s.th0(i, j, k);
        s.th3d(i, j, k) = 4.0 * (U(gen) - 0.5);
        s.pp3d(i, j, k) = 2.0e-3 * (U(gen) - 0.5);
        const double pii = s.pi0(i, j, k) + s.pp3d(i, j, k);
        const double t = (s.th0(i, j, k) + s.th3d(i, j, k)) * pii;
        const double p = phys::p00 * std::pow(pii, c.cpdrd);
        double esl = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
        esl = std::min(esl, 0.5 * p);
        const double qvs = c.eps * esl / (p - esl);
        const bool cold = t < 268.0, warm = t > 273.15;
        double rh = 0.3 + 0.6 * U(gen);
        double qc = 0, qr = 0, qi = 0, qs = 0, qg = 0;
        switch (regime) {
          case 0: if (U(gen) < 0.3) rh = 0.97 + 0.06 * U(gen); break;
          case 1: rh = 0.95 + 0.1 * U(gen); qc = 2.0e-3 * U(gen); if (U(gen) < 0.3) qr = 1.0e-4 * U(gen); break;
          case 2:
            rh = 0.9 + 0.15 * U(gen);
            qc = 1.5e-3 * U(gen);
            if (!warm) { qi = 5.0e-4 * U(gen); qs = 1.0e-3 * U(gen); }
            qr = U(gen) < 0.5 ? 1.0e-3 * U(gen) : 0.0;
            qg = U(gen) < 0.5 ? 2.0e-3 * U(gen) : 0.0;
            break;
          case 3:
            rh = 0.6 + 0.5 * U(gen);
            if (cold) { qi = 1.0e-3 * U(gen); qs = 2.0e-3 * U(gen); }
            else qs = 2.0e-4 * U(gen);   // melting snow
            break;
          case 4:
            rh = 0.5 + 0.5 * U(gen);
            qr = z < 6000 ? 6.0e-3 * U(gen) : 0.0;
            qs = z > 3000 ? 3.0e-3 * U(gen) : 1.0e-4 * U(gen);
            qg = 8.0e-3 * U(gen);
            if (U(gen) < 0.4) qc = 1.0e-3 * U(gen);
            break;
          default:
            rh = 0.7 + 0.4 * U(gen);
            qc = 1.0e-3 * U(gen); qr = 2.0e-3 * U(gen); qi = 5.0e-4 * U(gen); qs = 1.0e-3 * U(gen);
            qg = 2.0e-3 * U(gen);
            break;
        }
        double* qq[5] = {&qc, &qr, &qi, &qs, &qg};
        const double r = U(gen);
        if (r < 0.04) *qq[int(5 * U(gen))] = -1.0e-10 * U(gen);
        else if (r < 0.08) *qq[int(5 * U(gen))] = 1.0e-25 * U(gen);
        else if (r < 0.11) *qq[int(5 * U(gen))] = 1.0e-15 * U(gen);
        s.q(i, j, k, mo::nqv) = rh * qvs;
        s.q(i, j, k, mo::nqc) = qc;
        s.q(i, j, k, mo::nqr) = qr;
        s.q(i, j, k, mo::nqi) = qi;
        s.q(i, j, k, mo::nqs) = qs;
        s.q(i, j, k, mo::nqg) = qg;
        auto num = [&](double qx, double lo, double hi) {
          const double u = U(gen);
          if (u < 0.05) return 0.0;                // no number
          if (u < 0.08) return logu(lo, hi);       // number without mass
          return qx > 0 ? logu(lo, hi) : 0.0;
        };
        s.q(i, j, k, mo::nnci) = num(qi, 1.0e2, 1.0e6);
        s.q(i, j, k, mo::nncs) = num(qs, 1.0e2, 1.0e5);
        s.q(i, j, k, mo::nncr) = num(qr, 1.0e1, 1.0e5);
        s.q(i, j, k, mo::nncg) = num(qg, 1.0e1, 1.0e4);
        if (numq >= 11) s.q(i, j, k, mo::nncc) = num(qc, 1.0e6, 5.0e8);
        s.prs(i, j, k) = p;
        s.rho(i, j, k) = p / (c.rd * t * (1.0 + s.q(i, j, k, 1) * c.reps));
        s.t(i, j, k) = t;
        s.dzq(i, j, k) = dz * s.rmh(i, j, k);
      }
    }
  return s;
}

template <int R>
FArray<R> to_dev(const FArray<R, HostSpace>& h) {
  std::array<int, R> lb, ub;
  for (int d = 0; d < R; ++d) { lb[d] = h.lb(d); ub[d] = h.ub(d); }
  FArray<R> a(h.label(), lb, ub);
  a.deep_copy_from(h);
  return a;
}

template <int R>
size_t count_diff(const std::string& name, const FArray<R, HostSpace>& a, const FArray<R, HostSpace>& f) {
  REQUIRE(a.size() == f.size());
  size_t n = 0;
  double maxabs = 0;
  const Real* pa = a.data();
  const Real* pf = f.data();
  for (size_t m = 0; m < a.size(); ++m) {
    if (std::memcmp(&pa[m], &pf[m], sizeof(Real)) != 0 && !(std::isnan(pa[m]) && std::isnan(pf[m]))) {
      ++n;
      maxabs = std::max(maxabs, std::fabs(pa[m] - pf[m]));
    }
  }
  if (n) UNSCOPED_INFO(name << ": " << n << " entries differ, max abs diff " << maxabs);
  return n;
}

size_t compare(const State& c, const State& f) {
  size_t n = 0;
  n += count_diff("th3d", c.th3d, f.th3d);
  n += count_diff("pp3d", c.pp3d, f.pp3d);
  n += count_diff("prs", c.prs, f.prs);
  n += count_diff("rho", c.rho, f.rho);
  n += count_diff("t", c.t, f.t);
  n += count_diff("q", c.q, f.q);
  n += count_diff("rain", c.rain, f.rain);
  n += count_diff("prate", c.prate, f.prate);
  n += count_diff("vt", c.vt, f.vt);
  n += count_diff("eff", c.eff, f.eff);
  return n;
}

struct Opts {
  int eqtset = 2, efall = 0, radopt = 0, ihail = 1, inum = 1;
  bool dorad = false;
  Real ndcnst = 250.0;
};

void setup_fortran(const State& s, const Opts& o) {
  cm1f_morrison_setup(s.b.ni, s.b.nj, s.b.nk, s.b.ngxy, s.b.ngz, s.b.nrain, o.eqtset, o.efall, o.radopt, o.ihail,
                      o.ndcnst, o.inum, s.dx, s.dy, s.dz);
}

void work_to_host(const mo::Work& w, State& x) {
  x.prate = w.prate.to_host();
  const FArray3* vts[5] = {&w.vtc, &w.vtr, &w.vti, &w.vts, &w.vtg};
  for (int n = 0; n < 5; ++n)
    Kokkos::deep_copy(Kokkos::subview(x.vt.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), vts[n]->to_host().view());
  const FArray3* effs[6] = {&w.effc, &w.effi, &w.effs, &w.effr, &w.effg, &w.effis};
  for (int n = 0; n < 6; ++n)
    Kokkos::deep_copy(Kokkos::subview(x.eff.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), effs[n]->to_host().view());
}

// MP_GRAUPEL alone (the Fortran copy, so INUM=0 is reachable).
size_t run_mp_graupel(const State& init, const Opts& o, size_t* nchanged = nullptr) {
  State f = init.clone(), x = init.clone();
  setup_fortran(f, o);
  cm1f_morrison_mp_graupel(f.dt, o.dorad ? 1 : 0, f.t.data(), f.q.data(), f.prs.data(), f.rho.data(), f.dzq.data(),
                           f.w3d.data(), f.rain.data(), f.prate.data(), f.vt.data(), f.eff.data());

  mo::Work w;
  w.allocate(x.b, mo::Params::init(o.ihail, o.ndcnst, o.inum));
  auto t = to_dev(x.t), q = to_dev(x.q), rain = to_dev(x.rain);
  mo::mp_graupel(w.p, x.b, o.eqtset, o.radopt, x.dt, 1.0e-12, o.dorad, t, q, to_dev(x.prs), to_dev(x.rho),
                 to_dev(x.dzq), to_dev(x.w3d), rain, w);
  x.t = t.to_host(); x.q = q.to_host(); x.rain = rain.to_host();
  work_to_host(w, x);
  if (nchanged) *nchanged = count_diff("changed q", x.q, init.q);
  return compare(x, f);
}

// The full mp_driver path (real module, INUM=1).
size_t run_mp_driver(const State& init, const Opts& o, size_t* nchanged = nullptr) {
  State f = init.clone(), x = init.clone();
  setup_fortran(f, o);
  cm1f_morrison_mp_driver(f.dt, o.dorad ? 1 : 0, f.mh.data(), f.rmh.data(), f.mf.data(), f.pi0.data(),
                          f.th0.data(), f.rho.data(), f.prs.data(), f.pp3d.data(), f.th3d.data(), f.w3d.data(),
                          f.q.data(), f.rain.data(), f.prate.data(), f.vt.data(), f.eff.data());

  mo::Options opt;
  opt.ihail = o.ihail; opt.ndcnst = o.ndcnst; opt.inum = o.inum;
  opt.eqtset = o.eqtset; opt.efall = o.efall; opt.radopt = o.radopt;
  mo::Work w;
  w.allocate(x.b, mo::Params::init(o.ihail, o.ndcnst, o.inum));
  auto rho = to_dev(x.rho), prs = to_dev(x.prs), pp3d = to_dev(x.pp3d), th3d = to_dev(x.th3d);
  auto q = to_dev(x.q), rain = to_dev(x.rain);
  mo::mp_driver(opt, Constants::make(0), x.b, x.dt, x.dz, 1.0 / x.dz, to_dev(x.rmh), to_dev(x.mf), to_dev(x.pi0),
                to_dev(x.th0), to_dev(x.w3d), rho, prs, pp3d, th3d, q, rain, w, o.dorad);
  x.rho = rho.to_host(); x.prs = prs.to_host(); x.pp3d = pp3d.to_host(); x.th3d = th3d.to_host();
  x.q = q.to_host(); x.rain = rain.to_host();
  work_to_host(w, x);
  if (nchanged) *nchanged = count_diff("changed q", x.q, init.q);
  return compare(x, f);
}

} // namespace

TEST_CASE("morrison GRAUPEL_INIT constants match morrison.F", "[bridge][morrison]") {
  const State s = random_state(2, 2, 4, 400.0, 6.0, 10, 1u);
  for (int ihail : {0, 1})
    for (Real ndcnst : {250.0, 100.0}) {
      Opts o;
      o.ihail = ihail; o.ndcnst = ndcnst;
      setup_fortran(s, o);
      std::vector<Real> f(129);
      cm1f_morrison_params(f.data());
      const auto x = params_vec(mo::Params::init(ihail, ndcnst, 1));
      REQUIRE(x.size() == f.size());
      for (size_t n = 0; n < f.size(); ++n) {
        INFO("ihail " << ihail << " ndcnst " << ndcnst << " value #" << n + 1 << ": " << x[n] << " vs " << f[n]);
        CHECK(std::memcmp(&x[n], &f[n], sizeof(Real)) == 0);
      }
    }
}

TEST_CASE("morrison GAMMA and DERF1 match morrison.F", "[bridge][morrison]") {
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<double> xs = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.41, 4.8, 4.37, 4.5, 1.41, 1.8, 1.37, 2.705, 2.9,
                            2.685, 3.41, 3.8, 3.37, 6.8, 2.82, 7.0, 12.0, 13.0, 30.0};
  for (int n = 0; n < 2000; ++n) xs.push_back(1.0e-3 + 40.0 * U(gen));
  for (int n = 0; n < 200; ++n) xs.push_back(1.0e-6 + 1.0e-2 * U(gen));
  size_t nbad = 0;
  for (double x : xs) {
    const double a = mo::gamma_host(x), b = cm1f_morrison_gamma(x);
    if (std::memcmp(&a, &b, sizeof a) != 0) { ++nbad; UNSCOPED_INFO("gamma(" << x << "): " << a << " vs " << b); }
  }
  for (int n = 0; n < 2000; ++n) {
    const double x = -6.0 + 12.0 * U(gen);
    const double a = mo::derf1_host(x), b = cm1f_morrison_derf1(x);
    if (std::memcmp(&a, &b, sizeof a) != 0) { ++nbad; UNSCOPED_INFO("derf1(" << x << "): " << a << " vs " << b); }
  }
  for (double x : {-10.0, -5.0, -1.0, 0.0, 1.0e-3, 0.5, 1.0, 2.0, 5.0, 10.0}) {
    const double a = mo::derf1_host(x), b = cm1f_morrison_derf1(x);
    if (std::memcmp(&a, &b, sizeof a) != 0) { ++nbad; UNSCOPED_INFO("derf1(" << x << "): " << a << " vs " << b); }
  }
  REQUIRE(nbad == 0);
}

TEST_CASE("morrison satadj2 matches morrison.F", "[bridge][morrison]") {
  const State s = random_state(2, 2, 4, 400.0, 6.0, 10, 2u);
  Opts o;
  setup_fortran(s, o);
  const auto p = mo::Params::init(1, 250.0, 1);
  const auto c = Constants::make(0);
  std::mt19937_64 gen(11);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  size_t nbad = 0, nadj = 0;
  for (int eqtset : {1, 2})
    for (int n = 0; n < 4000; ++n) {
      const double t0 = 220.0 + 85.0 * U(gen);
      const double prs0 = 2.0e4 + 8.5e4 * U(gen);
      double esl = 611.2 * std::exp(17.67 * (t0 - 273.15) / (t0 - 29.65));
      esl = std::min(esl, 0.5 * prs0);
      const double qvs = c.eps * esl / (prs0 - esl);
      const double qv0 = qvs * (0.8 + 0.4 * U(gen));
      const double qc0 = U(gen) < 0.3 ? 0.0 : 3.0e-3 * U(gen);
      const double qr = 2.0e-3 * U(gen), qi = 1.0e-3 * U(gen), qs = 1.0e-3 * U(gen), qg = 2.0e-3 * U(gen);
      const double rho = prs0 / (c.rd * t0 * (1.0 + qv0 * c.reps));
      double tf = t0, pf = prs0, qvf = qv0, qcf = qc0;
      double tx = t0, px = prs0, qvx = qv0, qcx = qc0;
      cm1f_morrison_satadj2(&tf, &pf, &qvf, &qcf, qr, qi, qs, qg, rho, eqtset, 1.0e-12);
      const int iflag = mo::satadj2_host(p, tx, px, qvx, qcx, qr, qi, qs, qg, rho, eqtset, 1.0e-12);
      CHECK(iflag == 0);
      if (qvx != qv0) ++nadj;
      if (std::memcmp(&tf, &tx, 8) || std::memcmp(&pf, &px, 8) || std::memcmp(&qvf, &qvx, 8) ||
          std::memcmp(&qcf, &qcx, 8)) {
        ++nbad;
        UNSCOPED_INFO("eqtset " << eqtset << " t " << t0 << " p " << prs0 << ": t " << tx << "/" << tf << " qv "
                                << qvx << "/" << qvf);
      }
    }
  REQUIRE(nbad == 0);
  CHECK(nadj > 1000);
}

TEST_CASE("morrison MP_GRAUPEL matches morrison.F on random columns (INUM=1 and 0)", "[bridge][morrison]") {
  unsigned seed = 100;
  for (int inum : {1, 0})
    for (int eqtset : {1, 2})
      for (Real dt : {6.0, 60.0}) {
        Opts o;
        o.inum = inum; o.eqtset = eqtset;
        o.ihail = (seed % 2 == 0) ? 1 : 0;
        o.dorad = (dt > 10.0);
        o.radopt = eqtset;   // effective radii in both unit conventions
        INFO("inum " << inum << " eqtset " << eqtset << " dt " << dt << " ihail " << o.ihail << " dorad " << o.dorad);
        const State s = random_state(8, 6, 40, 400.0, dt, inum == 0 ? 11 : 10, seed++);
        size_t nchanged = 0;
        REQUIRE(run_mp_graupel(s, o, &nchanged) == 0);
        CHECK(nchanged > 0);
      }
}

TEST_CASE("morrison mp_driver matches mp_driver.F on random columns", "[bridge][morrison]") {
  unsigned seed = 200;
  for (int eqtset : {1, 2})
    for (int efall : {0, 1})
      for (Real dt : {6.0, 30.0}) {
        Opts o;
        o.eqtset = eqtset; o.efall = efall;
        o.ihail = efall;
        o.dorad = (efall == 1);
        o.radopt = (dt > 10.0) ? 2 : 1;
        INFO("eqtset " << eqtset << " efall " << efall << " dt " << dt);
        const State s = random_state(8, 6, 40, 400.0, dt, 10, seed++);
        size_t nchanged = 0;
        REQUIRE(run_mp_driver(s, o, &nchanged) == 0);
        CHECK(nchanged > 0);
      }
}

// States from the Fortran reference runs: tha/ppi/qa/wa as th3d/pp3d/q3d/w3d,
// plus rho, prs, rain and the grid/base state.
TEST_CASE("morrison mp_driver matches mp_driver.F on reference states", "[bridge][morrison][ref]") {
  struct C { const char* name; std::vector<int> steps; };
  for (const C& cs : {C{"morrison_saturated", {0, 1, 10, 30}}, C{"morrison_bubble", {150, 240, 300}}}) {
    const auto ref = testing::case_ref(cs.name);
    if (ref.empty()) continue;
    const auto nl = Namelist::from_file(testing::case_namelist(cs.name));
    const auto mopt = mo::Options::from_namelist(nl);
    size_t nchanged_total = 0;
    for (int step : cs.steps) {
      const auto dir = DumpReader::step_dir(ref, step);
      if (!std::filesystem::exists(dir)) { WARN("missing " << dir); continue; }
      DumpReader d(dir);
      State s;
      s.b = Bounds::from_namelist(nl);
      s.dx = nl.getr("dx", 0); s.dy = nl.getr("dy", 0); s.dz = nl.getr("dz", 0); s.dt = nl.getr("dtl", 0);
      s.allocate();
      auto rd = [&](const std::string& n, H3& a) {
        auto h = d.read_host<3>(n);
        REQUIRE(h.size() == a.size());
        Kokkos::deep_copy(a.view(), h.view());
      };
      rd("mh", s.mh); rd("rmh", s.rmh); rd("mf", s.mf); rd("pi0", s.pi0); rd("th0", s.th0);
      rd("rho", s.rho); rd("prs", s.prs); rd("ppi", s.pp3d); rd("tha", s.th3d); rd("rain", s.rain);
      rd("wa", s.w3d);
      const char* qn[10] = {"qa_qv", "qa_qc", "qa_qr", "qa_qi", "qa_qs", "qa_qg", "qa_nci", "qa_ncs", "qa_ncr", "qa_ncg"};
      for (int n = 0; n < 10; ++n) {
        auto h = d.read_host<3>(qn[n]);
        Kokkos::deep_copy(Kokkos::subview(s.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), h.view());
      }
      Opts o;
      o.eqtset = mopt.eqtset; o.efall = mopt.efall; o.ihail = mopt.ihail; o.ndcnst = mopt.ndcnst;
      INFO(cs.name << " step " << step);
      size_t nchanged = 0;
      REQUIRE(run_mp_driver(s, o, &nchanged) == 0);
      nchanged_total += nchanged;
      // the other options on the same realistic states
      o.eqtset = 3 - o.eqtset; o.efall = 1 - o.efall; o.ihail = 1 - o.ihail; o.dorad = true; o.radopt = 1 + step % 2;
      REQUIRE(run_mp_driver(s, o) == 0);
    }
    CHECK(nchanged_total > 0);
  }
}
