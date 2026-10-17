// Kessler (ptype=1) properties that hold independently of the Fortran.
#include "physics/kessler/cm1k_kessler.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <random>

using namespace cm1k;

namespace {

Bounds make_bounds(int ni, int nj, int nk) {
  Bounds b;
  b.nx = b.ni = ni; b.ny = b.nj = nj; b.nz = b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = 2; b.ngz = 1;
  b.ib = -1; b.ie = ni + 2; b.jb = -1; b.je = nj + 2; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.numq = 3; b.nrain = 1;
  return b;
}

} // namespace

TEST_CASE("kessler options from namelist", "[kessler]") {
  auto nl = Namelist::from_file(testing::case_namelist("kessler_bubble"));
  auto o = kessler::Options::from_namelist(nl);
  REQUIRE(o.eqtset == nl.geti("eqtset", 0));
  REQUIRE(o.efall == nl.geti("efall", -1));
  nl.set("eqtset", "7");
  REQUIRE(kessler::Options::from_namelist(nl).eqtset == 2);   // param.F clamps to [1,2]
  nl.set("psolver", "6");
  REQUIRE(kessler::Options::from_namelist(nl).eqtset == 1);   // psolver 4..7 needs eqtset=1
  nl.set("ptype", "2");
  REQUIRE_THROWS(kessler::Options::from_namelist(nl));
}

// With eqtset=2 rho is not modified and, with mh=1, every step of mp_driver
// conserves sum_k rho*(qv+qc+qr)*dz + 10*rain (kg/m2; rain is in cm):
// pdefq rescales to keep sum rho*q, fallout is in flux form, and
// kessler/satadj only move water between species.
TEST_CASE("kessler mp_driver conserves column water (eqtset=2)", "[kessler]") {
  const int ni = 6, nj = 5, nk = 40;
  const Real dz = 200.0, dt = 30.0;
  const auto B = make_bounds(ni, nj, nk);
  const auto c = Constants::make(0);
  Grid g;
  g.allocate(B);
  g.dz = dz; g.rdz = 1.0 / dz;
  g.mh.fill(1.0); g.mf.fill(1.0);
  BaseState base;
  base.allocate(B);
  State st;
  st.allocate(B);
  FArray3 th3d("th3d", {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  FArray3 pp3d("pp3d", {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});

  auto pi0 = base.pi0.to_host(), th0 = base.th0.to_host(), rho = st.rho.to_host(), prs = st.prs.to_host();
  auto th = th3d.to_host(), pp = pp3d.to_host();
  auto q = st.qa.to_host();
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (int k = B.kb; k <= B.ke; ++k)
    for (int j = B.jb; j <= B.je; ++j)
      for (int i = B.ib; i <= B.ie; ++i) {
        const double z = (k - 0.5) * dz;
        th0(i, j, k) = 300.0 + 0.004 * z;
        pi0(i, j, k) = 1.0 - c.g * z / (c.cp * 300.0);
        th(i, j, k) = 2.0 * (U(gen) - 0.5);
        pp(i, j, k) = 0.0;
        const double t = (th0(i, j, k) + th(i, j, k)) * pi0(i, j, k);
        prs(i, j, k) = phys::p00 * std::pow(pi0(i, j, k), c.cpdrd);
        q(i, j, k, 1) = (0.6 + 0.5 * U(gen)) * 0.014 * std::exp(-z / 2500.0);
        q(i, j, k, 2) = U(gen) < 0.5 ? 1.5e-3 * U(gen) : 0.0;
        q(i, j, k, 3) = z > 2000 ? 5.0e-3 * U(gen) : 0.0;
        rho(i, j, k) = prs(i, j, k) / (c.rd * t * (1.0 + q(i, j, k, 1) * c.reps));
      }
  base.pi0.deep_copy_from(pi0); base.th0.deep_copy_from(th0);
  st.rho.deep_copy_from(rho); st.prs.deep_copy_from(prs);
  th3d.deep_copy_from(th); pp3d.deep_copy_from(pp);
  st.qa.deep_copy_from(q);
  st.rain.fill(0.0);

  auto column_water = [&](int i, int j) {
    auto r = st.rho.to_host();
    auto qq = st.qa.to_host();
    auto rn = st.rain.to_host();
    double s = 0;
    for (int k = 1; k <= nk; ++k) s += r(i, j, k) * (qq(i, j, k, 1) + qq(i, j, k, 2) + qq(i, j, k, 3)) * dz;
    return s + 10.0 * rn(i, j, 1);
  };
  std::vector<double> w0;
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) w0.push_back(column_water(i, j));

  kessler::Options opt;
  opt.eqtset = 2;
  kessler::Work w;
  w.allocate(B);
  for (int n = 0; n < 5; ++n)
    kessler::mp_driver(opt, c, g, base, dt, st.rho, st.prs, pp3d, th3d, st.qa, st.rain, w);

  auto rn = st.rain.to_host();
  double rain_max = 0;
  size_t m = 0;
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i, ++m) {
      const double w1 = column_water(i, j);
      INFO("column " << i << "," << j << " before " << w0[m] << " after " << w1);
      REQUIRE(std::fabs(w1 - w0[m]) <= 1e-12 * w0[m]);
      rain_max = std::max(rain_max, rn(i, j, 1));
    }
  REQUIRE(rain_max > 0);   // it rained
  // all species non-negative after the scheme (pdefq + positive-definite fallout)
  auto qq = st.qa.to_host();
  for (int n = 1; n <= 3; ++n)
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) REQUIRE(qq(i, j, k, n) >= 0.0);
}
