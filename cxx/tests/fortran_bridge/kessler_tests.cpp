// Component tests: Kessler microphysics (ptype=1) against the Fortran
// (mp_driver_module::mp_driver and kessler_module routines), bit for bit.
#include "physics/kessler/cm1k_kessler.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <random>

extern "C" {
void cm1f_kessler_setup(int ni, int nj, int nk, int ngxy, int ngz, int nrain, int testcase, int eqtset,
                        int efall, double dx, double dy, double dz);
void cm1f_kessler_mp_driver(double dt, double* mh, double* mf, double* pi0, double* th0, double* rho,
                            double* prs, double* pp3d, double* th3d, double* q3d, double* rain,
                            double* prate, double* vr);
void cm1f_kessler_pdefq(double rmax, double* rho, double* q);
void cm1f_kessler_k_fallout(double* rho, double* qr, double* vr);
void cm1f_kessler_fallout(double dt, double* mh, double* mf, double* rho, double* qr, double* vr,
                          double* rain, double* prate);
void cm1f_kessler_kessler(double dt, double* pi0, double* th0, double* tmp, double* rho, double* pp3d,
                          double* th3d, double* prs, double* qv, double* qc, double* qr);
void cm1f_kessler_satadj(int nrk, double dt, double* pi0, double* th0, double* rho, double* pp3d,
                         double* prs, double* th3d, double* q3d);
}

using namespace cm1k;

namespace {

using H3 = FArray<3, HostSpace>;
using H4 = FArray<4, HostSpace>;
using H2 = FArray<2, HostSpace>;

// Everything mp_driver reads or writes, on the host.
struct Column {
  Bounds b;
  Real dx = 1000, dy = 1000, dz = 500, dt = 6;
  H3 mh, mf, pi0, th0, rho, prs, pp3d, th3d, rain, vr;
  H4 q;
  H2 prate;

  void allocate() {
    auto s3 = [&](const char* n) { return H3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
    mh = s3("mh"); pi0 = s3("pi0"); th0 = s3("th0"); rho = s3("rho"); prs = s3("prs");
    pp3d = s3("pp3d"); th3d = s3("th3d"); vr = s3("vr");
    mf = H3("mf", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
    rain = H3("rain", {b.ib, b.jb, 1}, {b.ie, b.je, b.nrain});
    q = H4("q", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, b.numq});
    prate = H2("prate", {b.ib, b.jb}, {b.ie, b.je});
  }

  Column clone() const {
    Column c = *this;
    c.allocate();
    Kokkos::deep_copy(c.mh.view(), mh.view()); Kokkos::deep_copy(c.mf.view(), mf.view());
    Kokkos::deep_copy(c.pi0.view(), pi0.view()); Kokkos::deep_copy(c.th0.view(), th0.view());
    Kokkos::deep_copy(c.rho.view(), rho.view()); Kokkos::deep_copy(c.prs.view(), prs.view());
    Kokkos::deep_copy(c.pp3d.view(), pp3d.view()); Kokkos::deep_copy(c.th3d.view(), th3d.view());
    Kokkos::deep_copy(c.rain.view(), rain.view()); Kokkos::deep_copy(c.vr.view(), vr.view());
    Kokkos::deep_copy(c.q.view(), q.view()); Kokkos::deep_copy(c.prate.view(), prate.view());
    return c;
  }
};

Bounds make_bounds(int ni, int nj, int nk, int ngxy) {
  Bounds b;
  b.nx = b.ni = ni; b.ny = b.nj = nj; b.nz = b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = ngxy; b.ngz = 1;
  b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.numq = 3; b.nrain = 1;
  return b;
}

// Random but physically plausible state: a stable sounding with random
// perturbations; each column picks a regime (subsaturated, near saturation,
// cloudy, rainy, heavy rain aloft) and some values are slightly negative or
// below qsmall so pdefq has work to do.
Column random_column(int ni, int nj, int nk, Real dz, Real dt, unsigned seed) {
  Column s;
  s.b = make_bounds(ni, nj, nk, 3);
  s.dz = dz; s.dt = dt;
  s.allocate();
  const auto c = Constants::make(0);
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const auto& b = s.b;
  for (int k = b.kb; k <= b.ke + 1; ++k)
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) s.mf(i, j, k) = 0.9 + 0.2 * U(gen);
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) {
      const int regime = int(5 * U(gen));
      for (int nr = 1; nr <= b.nrain; ++nr) s.rain(i, j, nr) = U(gen) < 0.5 ? 0.0 : 0.3 * U(gen);
      for (int k = b.kb; k <= b.ke; ++k) {
        const double z = (k - 0.5) * dz;
        s.mh(i, j, k) = 0.9 + 0.2 * U(gen);
        s.th0(i, j, k) = 300.0 + 0.004 * z;
        s.pi0(i, j, k) = 1.0 - c.g * z / (c.cp * (300.0 + 0.002 * z));
        s.th3d(i, j, k) = 4.0 * (U(gen) - 0.5);
        s.pp3d(i, j, k) = 2.0e-3 * (U(gen) - 0.5);
        const double pii = s.pi0(i, j, k) + s.pp3d(i, j, k);
        const double t = (s.th0(i, j, k) + s.th3d(i, j, k)) * pii;
        const double p = phys::p00 * std::pow(pii, c.cpdrd);
        double esl = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
        esl = std::min(esl, 0.5 * p);
        const double qvs = c.eps * esl / (p - esl);
        double rh = 0.3 + 0.6 * U(gen), qc = 0, qr = 0;
        if (regime == 1) rh = 0.97 + 0.06 * U(gen);
        if (regime == 2) { rh = 0.95 + 0.1 * U(gen); qc = 2.0e-3 * U(gen); }
        if (regime == 3) { rh = 0.6 + 0.45 * U(gen); qc = U(gen) < 0.5 ? 1.5e-3 * U(gen) : 0.0; qr = 3.0e-3 * U(gen); }
        if (regime == 4) { rh = 0.4 + 0.5 * U(gen); qr = z > 3000 ? 8.0e-3 * U(gen) : 1.0e-9 * U(gen); }
        const double r = U(gen);
        if (r < 0.05) qc = -1.0e-10 * U(gen);
        else if (r < 0.10) qr = -1.0e-10 * U(gen);
        else if (r < 0.13) qr = 1.0e-25 * U(gen);
        s.q(i, j, k, 1) = rh * qvs;
        s.q(i, j, k, 2) = qc;
        s.q(i, j, k, 3) = qr;
        s.prs(i, j, k) = p;
        s.rho(i, j, k) = p / (c.rd * t * (1.0 + s.q(i, j, k, 1) * c.reps));
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

// Number of entries (halos included) that are not bit-identical.
template <int R>
size_t count_diff(const std::string& name, const FArray<R, HostSpace>& a, const FArray<R, HostSpace>& f) {
  size_t n = 0;
  double maxabs = 0;
  const Real* pa = a.data();
  const Real* pf = f.data();
  for (size_t m = 0; m < a.size(); ++m) {
    if (!(pa[m] == pf[m]) && !(std::isnan(pa[m]) && std::isnan(pf[m]))) {
      ++n;
      maxabs = std::max(maxabs, std::fabs(pa[m] - pf[m]));
    }
  }
  if (n) UNSCOPED_INFO(name << ": " << n << " entries differ, max abs diff " << maxabs);
  return n;
}

size_t compare(const Column& c, const Column& f, bool with_vr = true) {
  size_t n = 0;
  n += count_diff("th3d", c.th3d, f.th3d);
  n += count_diff("pp3d", c.pp3d, f.pp3d);
  n += count_diff("prs", c.prs, f.prs);
  n += count_diff("rho", c.rho, f.rho);
  n += count_diff("q", c.q, f.q);
  n += count_diff("rain", c.rain, f.rain);
  n += count_diff("prate", c.prate, f.prate);
  if (with_vr) n += count_diff("vr", c.vr, f.vr);
  return n;
}

void setup_fortran(const Column& s, int eqtset, int efall) {
  cm1f_kessler_setup(s.b.ni, s.b.nj, s.b.nk, s.b.ngxy, s.b.ngz, s.b.nrain, 0, eqtset, efall, s.dx, s.dy, s.dz);
}

// Run the full mp_driver path in both languages and compare everything.
size_t run_mp_driver(const Column& init, int eqtset, int efall, Real* rain_max = nullptr,
                     size_t* nchanged = nullptr) {
  Column f = init.clone(), x = init.clone();
  setup_fortran(f, eqtset, efall);
  cm1f_kessler_mp_driver(f.dt, f.mh.data(), f.mf.data(), f.pi0.data(), f.th0.data(), f.rho.data(),
                         f.prs.data(), f.pp3d.data(), f.th3d.data(), f.q.data(), f.rain.data(),
                         f.prate.data(), f.vr.data());

  kessler::Options opt;
  opt.eqtset = eqtset;
  opt.efall = efall;
  kessler::Work w;
  w.allocate(x.b);
  w.prate.deep_copy_from(x.prate);
  w.vr.deep_copy_from(x.vr);
  auto rho = to_dev(x.rho), prs = to_dev(x.prs), pp3d = to_dev(x.pp3d), th3d = to_dev(x.th3d);
  auto q = to_dev(x.q);
  auto rain = to_dev(x.rain);
  kessler::mp_driver(opt, Constants::make(0), x.b, x.dt, 1.0 / x.dz, to_dev(x.mh), to_dev(x.mf),
                     to_dev(x.pi0), to_dev(x.th0), rho, prs, pp3d, th3d, q, rain, w);
  x.rho = rho.to_host(); x.prs = prs.to_host(); x.pp3d = pp3d.to_host(); x.th3d = th3d.to_host();
  x.q = q.to_host(); x.rain = rain.to_host(); x.prate = w.prate.to_host(); x.vr = w.vr.to_host();

  // how many moisture values the microphysics changed (tests check it is exercised)
  if (nchanged) *nchanged = count_diff("changed q", x.q, init.q);
  if (rain_max) {
    *rain_max = 0;
    for (int j = 1; j <= x.b.nj; ++j)
      for (int i = 1; i <= x.b.ni; ++i) *rain_max = std::max(*rain_max, x.rain(i, j, 1) - init.rain(i, j, 1));
  }
  return compare(x, f);
}

} // namespace

TEST_CASE("kessler routines match kessler.F", "[bridge][kessler]") {
  for (int eqtset : {1, 2}) {
    const Column s = random_column(9, 7, 30, 250.0, 10.0, 1234u + eqtset);
    setup_fortran(s, eqtset, 0);
    const auto c = Constants::make(0);
    const auto& b = s.b;
    INFO("eqtset " << eqtset);

    DYNAMIC_SECTION("pdefq, eqtset " << eqtset) {
      Column f = s.clone(), x = s.clone();
      for (int n = 1; n <= 3; ++n) {
        const Real rmax = n == 1 ? 0.0 : 1.0e-24;
        // Fortran gets a contiguous 3D slice
        H3 qf("qf", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
        Kokkos::deep_copy(qf.view(), Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1));
        cm1f_kessler_pdefq(rmax, f.rho.data(), qf.data());
        Kokkos::deep_copy(Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1), qf.view());
        auto q = to_dev(x.q);
        kessler::pdefq(b, rmax, to_dev(x.rho), q, n);
        x.q = q.to_host();
      }
      REQUIRE(count_diff("q", x.q, f.q) == 0);
      REQUIRE(count_diff("pdefq changed q", x.q, s.q) > 0);
    }

    DYNAMIC_SECTION("k_fallout + fallout, eqtset " << eqtset) {
      for (Real dt : {10.0, 120.0}) {   // 120 s: several fallout substeps (nfall > 1)
        Column f = s.clone(), x = s.clone();
        H3 qf("qf", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
        Kokkos::deep_copy(qf.view(), Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 2));
        cm1f_kessler_k_fallout(f.rho.data(), qf.data(), f.vr.data());
        cm1f_kessler_fallout(dt, f.mh.data(), f.mf.data(), f.rho.data(), qf.data(), f.vr.data(),
                             f.rain.data(), f.prate.data());
        Kokkos::deep_copy(Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, 2), qf.view());

        auto q = to_dev(x.q);
        auto vr = to_dev(x.vr), rain = to_dev(x.rain);
        auto prate = to_dev(x.prate);
        auto rho = to_dev(x.rho);
        kessler::k_fallout(b, rho, q, 3, vr);
        kessler::fallout(b, dt, 1.0 / s.dz, to_dev(x.mh), rho, q, 3, vr, rain, prate);
        x.q = q.to_host(); x.vr = vr.to_host(); x.rain = rain.to_host(); x.prate = prate.to_host();
        INFO("dt " << dt);
        REQUIRE(compare(x, f) == 0);
      }
    }

    DYNAMIC_SECTION("kessler, eqtset " << eqtset) {
      Column f = s.clone(), x = s.clone();
      H3 tmpf("tmp", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
      std::vector<H3> qf;
      for (int n = 0; n < 3; ++n) {
        qf.emplace_back("qf", std::array<int, 3>{b.ib, b.jb, b.kb}, std::array<int, 3>{b.ie, b.je, b.ke});
        Kokkos::deep_copy(qf[n].view(), Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n));
      }
      cm1f_kessler_kessler(s.dt, f.pi0.data(), f.th0.data(), tmpf.data(), f.rho.data(), f.pp3d.data(),
                           f.th3d.data(), f.prs.data(), qf[0].data(), qf[1].data(), qf[2].data());
      for (int n = 0; n < 3; ++n)
        Kokkos::deep_copy(Kokkos::subview(f.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), qf[n].view());

      auto rho = to_dev(x.rho), prs = to_dev(x.prs), pp3d = to_dev(x.pp3d), th3d = to_dev(x.th3d);
      auto q = to_dev(x.q);
      FArray3 tmp("tmp", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
      kessler::kessler(b, c, eqtset, s.dt, to_dev(x.pi0), to_dev(x.th0), tmp, rho, pp3d, th3d, prs, q);
      x.rho = rho.to_host(); x.prs = prs.to_host(); x.pp3d = pp3d.to_host(); x.th3d = th3d.to_host();
      x.q = q.to_host();
      REQUIRE(compare(x, f, false) == 0);
      REQUIRE(count_diff("tmp", tmp.to_host(), tmpf) == 0);
    }

    DYNAMIC_SECTION("satadj, eqtset " << eqtset) {
      Column f = s.clone(), x = s.clone();
      cm1f_kessler_satadj(4, s.dt, f.pi0.data(), f.th0.data(), f.rho.data(), f.pp3d.data(), f.prs.data(),
                          f.th3d.data(), f.q.data());
      auto rho = to_dev(x.rho), prs = to_dev(x.prs), pp3d = to_dev(x.pp3d), th3d = to_dev(x.th3d);
      auto q = to_dev(x.q);
      kessler::satadj(b, c, eqtset, 1.0e-12, 4, s.dt, to_dev(x.pi0), to_dev(x.th0), rho, pp3d, prs, th3d, q);
      x.rho = rho.to_host(); x.prs = prs.to_host(); x.pp3d = pp3d.to_host(); x.th3d = th3d.to_host();
      x.q = q.to_host();
      REQUIRE(compare(x, f, false) == 0);
    }
  }
}

TEST_CASE("kessler mp_driver matches mp_driver.F on random columns", "[bridge][kessler]") {
  unsigned seed = 42;
  for (int eqtset : {1, 2})
    for (int efall : {0, 1})
      for (Real dt : {6.0, 60.0}) {
        INFO("eqtset " << eqtset << " efall " << efall << " dt " << dt);
        const Column s = random_column(8, 6, 40, 200.0, dt, seed++);
        Real rain_max = 0;
        size_t nchanged = 0;
        REQUIRE(run_mp_driver(s, eqtset, efall, &rain_max, &nchanged) == 0);
        CHECK(nchanged > 0);
        CHECK(rain_max > 0);
      }
}

// States from the Fortran reference runs (after the full time step, so they
// are realistic inputs for the next microphysics call): tha/ppi/qa as
// th3d/pp3d/q3d, plus rho, prs, rain and the grid/base state.
TEST_CASE("kessler mp_driver matches mp_driver.F on reference states", "[bridge][kessler][ref]") {
  struct C { const char* name; std::vector<int> steps; };
  for (const C& cs : {C{"kessler_saturated", {0, 1, 5, 15, 30}}, C{"kessler_bubble", {0, 50, 100, 150}}}) {
    const auto ref = testing::case_ref(cs.name);
    if (ref.empty()) continue;
    const auto nl = Namelist::from_file(testing::case_namelist(cs.name));
    size_t nchanged_total = 0;
    for (int step : cs.steps) {
      const auto dir = DumpReader::step_dir(ref, step);
      if (!std::filesystem::exists(dir)) { WARN("missing " << dir); continue; }
      DumpReader d(dir);
      Column s;
      s.b = Bounds::from_namelist(nl);
      s.dx = nl.getr("dx", 0); s.dy = nl.getr("dy", 0); s.dz = nl.getr("dz", 0); s.dt = nl.getr("dtl", 0);
      s.allocate();
      auto rd = [&](const std::string& n, H3& a) {
        auto h = d.read_host<3>(n);
        REQUIRE(h.size() == a.size());
        Kokkos::deep_copy(a.view(), h.view());
      };
      rd("mh", s.mh); rd("mf", s.mf); rd("pi0", s.pi0); rd("th0", s.th0);
      rd("rho", s.rho); rd("prs", s.prs); rd("ppi", s.pp3d); rd("tha", s.th3d); rd("rain", s.rain);
      const char* qn[3] = {"qa_qv", "qa_qc", "qa_qr"};
      for (int n = 0; n < 3; ++n) {
        auto h = d.read_host<3>(qn[n]);
        Kokkos::deep_copy(Kokkos::subview(s.q.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n), h.view());
      }
      const auto opt = kessler::Options::from_namelist(nl);
      INFO(cs.name << " step " << step << " eqtset " << opt.eqtset);
      size_t nchanged = 0;
      REQUIRE(run_mp_driver(s, opt.eqtset, opt.efall, nullptr, &nchanged) == 0);
      nchanged_total += nchanged;
      // also with the other options, on the same realistic states
      REQUIRE(run_mp_driver(s, 3 - opt.eqtset, 1) == 0);
    }
    // the dumped states must exercise the scheme (clouds/rain appear in both cases)
    CHECK(nchanged_total > 0);
  }
}
