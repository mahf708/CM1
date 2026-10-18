// Component tests of the MYNN surface layer port (physics/sfc_mynn) against
// the unmodified src/module_sf_mynn.F, through cm1f_bridge_mynn_sfc.F90
// (SFCLAY_mynn called as turb.F calls it).
//
// Synthetic but physical lowest-level columns: water and land (some snow
// covered), stable/neutral/convective (tsk vs. t), moist and dry, calm to
// strong winds, layer depths that exercise every u10 branch. Several steps
// are run on both sides with the persistent arrays (ust, mol, qsfc, znt,
// ustm, hfx, qfx, zol, rmol, wspd) carried over and the inputs drifting, and
// every output is compared bit for bit after every step.
#include "physics/sfc_mynn/cm1k_sfc_mynn.hpp"
#include "share/cm1k_constants.hpp"

#include <catch2/catch.hpp>

#include <cmath>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void cm1f_mynn_sf_init();
void cm1f_mynn_psi_tables(double* ms, double* mu, double* hs, double* hu);
void cm1f_mynn_sfclay(int ni, int nj, int nk, int itimestep, int isfflx, int isftcflx, int iz0tlnd, double dx,
                      const double* u3d, const double* v3d, const double* dz8w, const double* t3d,
                      const double* qv3d, const double* p3d, const double* rho3d, const double* psfc,
                      const double* hpbl, const double* mavail, const double* xland, const double* tsk,
                      const double* snowh, double* ust, double* mol, double* qsfc, double* znt, double* ustm,
                      double* hfx, double* qfx, double* zol, double* rmol, double* wspd, double* chs,
                      double* chs2, double* cqs2, double* cpmm, double* regime, double* psim, double* psih,
                      double* lh, double* flhc, double* flqc, double* qgh, double* u10, double* v10,
                      double* th2, double* t2, double* q2, double* gz1oz0, double* br, double* ch, double* ck,
                      double* cka, double* cd, double* cda);
void cm1f_mynn_sfc_finish(int ni, int nj, int dosfcflx, const double* qv1, const double* rf1, double* qsfc,
                          const double* qfx, const double* flqc, const double* chs2, const double* cqs2,
                          const double* ck, const double* u10, const double* v10, const double* hfx,
                          const double* cpmm, double* ch, double* cq, double* s10, double* thflux,
                          double* qvflux);
}

using namespace cm1k;
namespace M = cm1k::sfc_mynn;

namespace {

using H3 = FArray<3, HostSpace>;

void require_same_ptr(const double* a, const double* ref, size_t n, const std::string& what) {
  size_t nbad = 0;
  double maxd = 0;
  for (size_t m = 0; m < n; ++m) {
    const double x = a[m], y = ref[m];
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
  }
  INFO(what << ": " << nbad << " differing points, max |diff| = " << maxd);
  REQUIRE(nbad == 0);
}

struct Case {
  int isftcflx = 0, iz0tlnd = 0, isfflx = 1;
  bool moist = true;
  double dx = 1000.;
  std::string str() const {
    std::ostringstream s;
    s << "isftcflx=" << isftcflx << " iz0tlnd=" << iz0tlnd << " isfflx=" << isfflx << " moist=" << moist
      << " dx=" << dx;
    return s.str();
  }
};

// the persistent and output 2D arrays, by name (Fortran argument order)
const std::vector<std::string> state_names = {
    "ust", "mol", "qsfc", "znt", "ustm", "hfx", "qfx", "zol", "rmol", "wspd", "chs", "chs2", "cqs2", "cpmm",
    "regime", "psim", "psih", "lh", "flhc", "flqc", "qgh", "u10", "v10", "th2", "t2", "q2", "gz1oz0", "br",
    "ch", "ck", "cka", "cd", "cda"};

// a host array with the bounds of `a` that never aliases it (to_host() returns
// the array itself when the device is the host)
H3 hnew(const FArray3& a) {
  H3 h(a.label() + "_f", {a.lb(0), a.lb(1), a.lb(2)}, {a.ub(0), a.ub(1), a.ub(2)});
  Kokkos::deep_copy(h.view(), 0.0);
  return h;
}

std::map<std::string, FArray3> field_map(const M::Fields& f) {
  return {{"psfc", f.psfc}, {"hpbl", f.hpbl}, {"mavail", f.mavail}, {"xland", f.xland}, {"tsk", f.tsk},
          {"snowh", f.snowh}, {"ust", f.ust}, {"mol", f.mol}, {"qsfc", f.qsfc}, {"znt", f.znt},
          {"ustm", f.ustm}, {"hfx", f.hfx}, {"qfx", f.qfx}, {"zol", f.zol}, {"rmol", f.rmol},
          {"wspd", f.wspd}, {"chs", f.chs}, {"chs2", f.chs2}, {"cqs2", f.cqs2}, {"cpmm", f.cpmm},
          {"regime", f.regime}, {"psim", f.psim}, {"psih", f.psih}, {"lh", f.lh}, {"flhc", f.flhc},
          {"flqc", f.flqc}, {"qgh", f.qgh}, {"u10", f.u10}, {"v10", f.v10}, {"th2", f.th2}, {"t2", f.t2},
          {"q2", f.q2}, {"gz1oz0", f.gz1oz0}, {"br", f.br}, {"ch", f.ch}, {"ck", f.ck}, {"cka", f.cka},
          {"cd", f.cd}, {"cda", f.cda}};
}

// Saturation mixing ratio (only to build realistic inputs).
double qsat(double t, double p) {
  const double es = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
  return 0.622 * es / (p - es);
}

void run_case(const Case& c, unsigned seed) {
  INFO(c.str());
  const int ni = 10, nj = 6, nk = 4, nsteps = 5;
  Bounds b;
  b.ni = ni; b.nj = nj; b.nk = nk;
  b.ib = 1; b.ie = ni; b.jb = 1; b.je = nj; b.kb = 1; b.ke = nk;
  const auto C = Constants::make(0);
  const auto tables = M::PsiTables::make();

  M::Options o;
  o.sfcmodel = 6; o.isftcflx = c.isftcflx; o.iz0tlnd = c.iz0tlnd; o.imoist = c.moist ? 1 : 0;
  REQUIRE_NOTHROW(o.check());

  M::Fields f;
  f.allocate(b);
  auto fm = field_map(f);
  std::map<std::string, H3> hf;   // Fortran-side copies
  for (auto& [n, a] : fm) { hf[n] = hnew(a); }

  M::Inputs in;
  auto a3 = [&](const char* n) { return FArray3(n, {1, 1, 1}, {ni, nj, 2}); };
  in.u3d = a3("u3d"); in.v3d = a3("v3d"); in.dz8w = a3("dz8w"); in.t3d = a3("t3d"); in.qv3d = a3("qv3d");
  in.p3d = a3("p3d"); in.rho3d = a3("rho3d");
  H3 u = hnew(in.u3d), v = hnew(in.v3d), dz = hnew(in.dz8w), t = hnew(in.t3d), qv = hnew(in.qv3d),
     p = hnew(in.p3d), rho = hnew(in.rho3d);

  std::mt19937_64 gen(seed);
  auto U = [&](double a, double bb) { return std::uniform_real_distribution<double>(a, bb)(gen); };

  // fixed surface properties and the initial state (CM1's allocation values)
  const double dzs[5] = {8.0, 12.0, 20.0, 40.0, 100.0};
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      const bool water = ((i + j) % 2 == 0);
      hf["xland"](i, j, 1) = water ? 2.0 : 1.0;
      hf["snowh"](i, j, 1) = (!water && j == nj) ? 0.2 : 0.0;
      hf["mavail"](i, j, 1) = water ? 1.0 : U(0.05, 0.6);
      hf["znt"](i, j, 1) = water ? 1.0e-4 : U(0.01, 0.5);
      hf["ust"](i, j, 1) = 1.0e-6;
      hf["qsfc"](i, j, 1) = 1.0e-5;
      hf["hpbl"](i, j, 1) = U(100.0, 1500.0);
      hf["psfc"](i, j, 1) = 1.0e5 + U(-2000.0, 2000.0);
      dz(i, j, 1) = dzs[(i - 1) % 5];
      dz(i, j, 2) = dzs[(i + j) % 5];
    }

  for (int step = 1; step <= nsteps; ++step) {
    INFO("step " << step);
    // inputs: lowest-level state drifting from step to step
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        const int regime = (i + 2 * j + step) % 4;   // 0 stable, 1 neutral-ish, 2/3 convective
        const double psfc = hf["psfc"](i, j, 1);
        const double t1 = 285.0 + 8.0 * std::sin(0.3 * i + 0.7 * j) + U(-1.0, 1.0) - (j == 1 ? 25.0 : 0.0);
        const double dts = regime == 0 ? U(-6.0, -0.5) : regime == 1 ? 0.0 : U(0.5, 8.0);
        hf["tsk"](i, j, 1) = t1 + dts + 0.5;
        const double wind = (i == 1) ? U(0.0, 0.05) : (i == 2 ? U(0.5, 2.0) : U(2.0, 25.0));
        const double dir = U(0.0, 6.283185307179586);
        for (int k = 1; k <= 2; ++k) {
          u(i, j, k) = wind * std::cos(dir) * (k == 2 ? 1.15 : 1.0);
          v(i, j, k) = wind * std::sin(dir) * (k == 2 ? 1.15 : 1.0);
        }
        p(i, j, 1) = psfc - 1.16 * 9.81 * 0.5 * dz(i, j, 1);
        t(i, j, 1) = t1;
        qv(i, j, 1) = c.moist ? U(0.3, 0.95) * qsat(t1, p(i, j, 1)) : 0.0;
        rho(i, j, 1) = p(i, j, 1) / (C.rd * t1 * (1.0 + 0.61 * qv(i, j, 1)));
        hf["hpbl"](i, j, 1) *= U(0.9, 1.1);
      }
    // device side gets the same inputs and the same state
    in.u3d.deep_copy_from(u); in.v3d.deep_copy_from(v); in.dz8w.deep_copy_from(dz); in.t3d.deep_copy_from(t);
    in.qv3d.deep_copy_from(qv); in.p3d.deep_copy_from(p); in.rho3d.deep_copy_from(rho);
    for (auto n : {"psfc", "hpbl", "mavail", "xland", "tsk", "snowh"}) fm[n].deep_copy_from(hf[n]);
    if (step == 1)
      for (auto& n : state_names) fm[n].deep_copy_from(hf[n]);

    M::sfclay(o, C, tables, ni, nj, c.dx, step, c.isfflx, in, f);

    // t3d etc. are (ni,nj,2) here: the shim reads their k=1 plane
    cm1f_mynn_sfclay(ni, nj, nk, step, c.isfflx, c.isftcflx, c.iz0tlnd, c.dx, u.data(), v.data(), dz.data(),
                     t.data(), qv.data(), p.data(), rho.data(), hf["psfc"].data(), hf["hpbl"].data(),
                     hf["mavail"].data(), hf["xland"].data(), hf["tsk"].data(), hf["snowh"].data(),
                     hf["ust"].data(), hf["mol"].data(), hf["qsfc"].data(), hf["znt"].data(), hf["ustm"].data(),
                     hf["hfx"].data(), hf["qfx"].data(), hf["zol"].data(), hf["rmol"].data(), hf["wspd"].data(),
                     hf["chs"].data(), hf["chs2"].data(), hf["cqs2"].data(), hf["cpmm"].data(),
                     hf["regime"].data(), hf["psim"].data(), hf["psih"].data(), hf["lh"].data(),
                     hf["flhc"].data(), hf["flqc"].data(), hf["qgh"].data(), hf["u10"].data(), hf["v10"].data(),
                     hf["th2"].data(), hf["t2"].data(), hf["q2"].data(), hf["gz1oz0"].data(), hf["br"].data(),
                     hf["ch"].data(), hf["ck"].data(), hf["cka"].data(), hf["cd"].data(), hf["cda"].data());

    for (auto& n : state_names) {
      const H3 h = fm[n].to_host();
      require_same_ptr(h.data(), hf[n].data(), h.size(), n);
    }
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        REQUIRE(std::isfinite(hf["ust"](i, j, 1)));
        REQUIRE(std::isfinite(hf["hfx"](i, j, 1)));
        REQUIRE(std::isfinite(hf["q2"](i, j, 1)));
      }
    // the regimes seen (sanity: the inputs do reach every branch)
    if (step == nsteps) {
      int nreg[5] = {0, 0, 0, 0, 0};
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) nreg[int(hf["regime"](i, j, 1))]++;
      CHECK(nreg[1] + nreg[2] > 0);
      CHECK(nreg[4] > 0);
    }
  }

  // turb.F's post-processing
  for (int dosfcflx : {0, 1}) {
    auto a1 = [&](const char* n) { return FArray3(n, {1, 1, 1}, {ni, nj, 1}); };
    FArray3 cq = a1("cq"), s10 = a1("s10"), thflux = a1("thflux"), qvflux = a1("qvflux"), rf = a1("rf");
    H3 rfh = hnew(rf);
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) rfh(i, j, 1) = 1.1 + 0.01 * i - 0.02 * j;
    rf.deep_copy_from(rfh);
    H3 cqh = hnew(cq), s10h = hnew(s10), thh = hnew(thflux), qvh = hnew(qvflux);
    cq.fill(0.0); s10.fill(0.0); thflux.fill(0.0); qvflux.fill(0.0);
    M::finish(ni, nj, dosfcflx == 1, in, rf, f, cq, s10, thflux, qvflux);
    cm1f_mynn_sfc_finish(ni, nj, dosfcflx, qv.data(), rfh.data(), hf["qsfc"].data(), hf["qfx"].data(),
                         hf["flqc"].data(), hf["chs2"].data(), hf["cqs2"].data(), hf["ck"].data(),
                         hf["u10"].data(), hf["v10"].data(), hf["hfx"].data(), hf["cpmm"].data(),
                         hf["ch"].data(), cqh.data(), s10h.data(), thh.data(), qvh.data());
    for (auto n : {"qsfc", "ch"}) {
      const H3 h = fm[n].to_host();
      require_same_ptr(h.data(), hf[n].data(), h.size(), std::string("finish ") + n);
    }
    std::pair<FArray3, H3*> outs[] = {{cq, &cqh}, {s10, &s10h}, {thflux, &thh}, {qvflux, &qvh}};
    for (auto& pr : outs) {
      const H3 h = pr.first.to_host();
      require_same_ptr(h.data(), pr.second->data(), h.size(), "finish " + pr.first.label());
    }
  }
}

} // namespace

TEST_CASE("mynn sfc psi tables match psi_init", "[bridge][sfc_mynn]") {
  cm1f_mynn_sf_init();
  std::vector<double> ms(1001), mu(1001), hs(1001), hu(1001);
  cm1f_mynn_psi_tables(ms.data(), mu.data(), hs.data(), hu.data());
  const auto t = M::PsiTables::make();
  auto h = [](const Kokkos::View<Real*, Device>& v) { return Kokkos::create_mirror_view_and_copy(HostSpace(), v); };
  require_same_ptr(h(t.psim_stab).data(), ms.data(), 1001, "psim_stab");
  require_same_ptr(h(t.psim_unstab).data(), mu.data(), 1001, "psim_unstab");
  require_same_ptr(h(t.psih_stab).data(), hs.data(), 1001, "psih_stab");
  require_same_ptr(h(t.psih_unstab).data(), hu.data(), 1001, "psih_unstab");
}

TEST_CASE("mynn sfc SFCLAY_mynn matches module_sf_mynn.F", "[bridge][sfc_mynn]") {
  cm1f_mynn_sf_init();
  unsigned seed = 7001;
  for (int isftcflx : {0, 1, 2, 3})
    for (int iz0tlnd : {0, 1, 3}) {
      Case c; c.isftcflx = isftcflx; c.iz0tlnd = iz0tlnd;
      run_case(c, seed++);
    }
  Case dry; dry.moist = false; run_case(dry, seed++);
  Case nof; nof.isfflx = 0; run_case(nof, seed++);
  Case coarse; coarse.dx = 12000.; run_case(coarse, seed++);
}

TEST_CASE("mynn sfc options: unsupported values throw", "[sfc_mynn]") {
  M::Options o;
  o.sfcmodel = 6;
  REQUIRE_NOTHROW(o.check());
  auto bad = [&](auto fn) { M::Options x = o; fn(x); REQUIRE_THROWS(x.check()); };
  bad([](M::Options& x) { x.isftcflx = 4; });
  bad([](M::Options& x) { x.isftcflx = -1; });
  bad([](M::Options& x) { x.iz0tlnd = 2; });
  bad([](M::Options& x) { x.iz0tlnd = 4; });
  M::Options off;   // sfcmodel /= 6: not checked
  off.iz0tlnd = 2;
  REQUIRE_NOTHROW(off.check());
  auto nl = Namelist::from_string("&param2\n sfcmodel = 6 /\n&param12\n isftcflx = 2, iz0tlnd = 1 /\n");
  const auto from = M::Options::from_namelist(nl);
  REQUIRE(from.active());
  REQUIRE(from.isftcflx == 2);
  REQUIRE(from.iz0tlnd == 1);
}
