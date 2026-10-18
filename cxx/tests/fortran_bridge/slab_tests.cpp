// Component tests of physics/slab (the SLAB land model, slab.F, and the
// soil-layer setup of init_surface.F) and of the LANDUSE.TBL emissivity /
// thermal inertia of physics/sfc, against the original Fortran through
// cm1f_bridge_slab.F90.
//
// Inputs: a small grid of random but plausible surface states (land and
// water points, flqc = 0 at some points, snow cover at some), stepped
// several times with tsk and tslb carried from one call to the next as
// across time steps; every array is compared bit for bit (halos included).
#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/slab/cm1k_slab.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <filesystem>
#include <random>
#include <sstream>

extern "C" {
void cm1f_slab(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int nsl, int ifsnow, int irad,
               double deltsm, const double* rc, const double* zs, const double* dzs, const double* t3d,
               const double* qv3d, const double* p3d, double* s2d, double* tslb);
void cm1f_slab_soil(int nsl, int ib, int ie, int jb, int je, const double* tsk, const double* tmn, double* zs,
                    double* dzs, double* tslb);
void cm1f_getlanduse(int season, int lu, double* emiss, double* thc);
}

using namespace cm1k;
using namespace cm1k::wrftest;

namespace {

const Constants kC = Constants::make(0);
const wrfphys::WrfConst kW = wrfphys::WrfConst::make(kC);

std::vector<double> rc19(const wrfphys::WrfConst& w) {
  return {w.cp, w.g, w.rovcp, w.rd, w.xlv, w.lv1, w.lv2, w.svp1, w.svp2, w.svp3,
          w.svpt0, w.ep1, w.ep2, w.karman, w.eomeg, w.stbolt, w.p1000mb, w.rv, w.rovg};
}

// cm1f_bridge_slab.F90's packed order
constexpr int NS = 18;
const char* const kNames[NS] = {"flhc", "flqc", "psfc", "xland", "tmn", "hfx", "qfx", "lh", "tsk",
                                "qsfc", "chklowq", "gsw", "glw", "capg", "thc", "snowc", "emiss", "mavail"};

struct SlabCase {
  int nsl = 5, ifsnow = 0, irad = 1;
  double dt = 6.0;
  bool sun = false;   // gsw, glw > 0 (CM1 passes 0 without radiation)
  std::string str() const {
    std::ostringstream s;
    s << "nsl=" << nsl << " ifsnow=" << ifsnow << " radiation=" << irad << " dt=" << dt << " sun=" << sun;
    return s.str();
  }
};

void check_slab(const SlabCase& c, unsigned seed, int ncalls) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  auto U = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(gen); };
  const auto b = box(7, 5, 4, 2);
  const std::array<int, 3> lo3{b.ib, b.jb, b.kb}, hi3{b.ie, b.je, b.ke};
  // 3D inputs (only k=1 is read)
  H3 t3d = h3("t3d", lo3, hi3), qv3d = h3("qv3d", lo3, hi3), p3d = h3("p3d", lo3, hi3);
  for (int k = b.kb; k <= b.ke; ++k)
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        t3d(i, j, k) = U(275.0, 305.0) - 6.5e-3 * 50.0 * k;
        qv3d(i, j, k) = (i + j) % 5 == 0 ? 0.0 : U(0.0, 0.02);
        p3d(i, j, k) = U(95000.0, 100500.0) - 600.0 * k;
      }
  // 2D state, packed
  H3 s = h3("s2d", {b.ib, b.jb, 1}, {b.ie, b.je, NS});
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) {
      auto v = [&](int n) -> double& { return s(i, j, n); };
      const bool water = ((i * 3 + j) % 4 == 0);
      v(1) = (i == 2 && j == 3) ? 0.0 : U(0.0, 45.0);                       // flhc
      v(2) = ((i + 2 * j) % 6 == 0) ? 0.0 : U(0.0, 0.05);                   // flqc
      v(3) = p3d(i, j, 1) + U(100.0, 700.0);                               // psfc
      v(4) = water ? 2.0 : 1.0;                                            // xland
      v(5) = U(280.0, 300.0);                                              // tmn
      v(6) = U(-50.0, 400.0);                                              // hfx
      v(7) = U(0.0, 3.0e-4);                                               // qfx
      v(8) = -7.25;                                                        // lh
      v(9) = U(262.0, 322.0);                                              // tsk
      v(10) = -7.25; v(11) = -7.25;                                        // qsfc, chklowq
      v(12) = c.sun ? U(0.0, 900.0) : 0.0;                                 // gsw
      v(13) = c.sun ? U(250.0, 420.0) : 0.0;                               // glw
      v(14) = -7.25;                                                       // capg
      v(15) = U(0.02, 0.06);                                               // thc
      v(16) = (c.ifsnow == 1 && (i + j) % 3 == 0) ? 1.0 : 0.0;            // snowc
      v(17) = U(0.88, 0.99);                                               // emiss
      v(18) = water ? 1.0 : U(0.02, 1.0);                                  // mavail
    }
  // soil layers (both sides from init_surface.F's setup) and tslb
  const auto sl = slab::SoilLayers::make(c.nsl);
  std::vector<double> zs(c.nsl, 0.0), dzs(c.nsl, 0.0);
  for (int l = 0; l < c.nsl; ++l) { zs[l] = sl.zs[l]; dzs[l] = sl.dzs[l]; }
  H3 tslb = h3("tslb", {b.ib, b.jb, 1}, {b.ie, b.je, c.nsl});
  for (int l = 1; l <= c.nsl; ++l)
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i)
        tslb(i, j, l) = s(i, j, 9) + (s(i, j, 5) - s(i, j, 9)) * (l - 1) / double(c.nsl) + U(-0.5, 0.5);

  // C++ state
  const std::array<int, 3> lo2{b.ib, b.jb, 1}, hi2{b.ie, b.je, 1};
  std::vector<FArray3> d(NS);
  for (int n = 0; n < NS; ++n) {
    H3 h = h3(kNames[n], lo2, hi2);
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) h(i, j, 1) = s(i, j, n + 1);
    d[n] = FArray3(kNames[n], lo2, hi2);
    d[n].deep_copy_from(h);
  }
  slab::Arrays a;
  a.flhc = d[0]; a.flqc = d[1]; a.psfc = d[2]; a.xland = d[3]; a.tmn = d[4]; a.hfx = d[5]; a.qfx = d[6];
  a.lh = d[7]; a.tsk = d[8]; a.qsfc = d[9]; a.chklowq = d[10]; a.gsw = d[11]; a.glw = d[12]; a.capg = d[13];
  a.thc = d[14]; a.snowc = d[15]; a.emiss = d[16]; a.mavail = d[17];
  a.tslb = FArray3("tslb", lo2, {b.ie, b.je, c.nsl});
  a.tslb.deep_copy_from(tslb);
  auto dev = [](const H3& h) {
    FArray3 x(h.label(), {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
    x.deep_copy_from(h);
    return x;
  };
  const FArray3 T = dev(t3d), Q = dev(qv3d), P = dev(p3d);
  const auto rc = rc19(kW);
  const H3 tsk_start = hcopy(s);

  for (int call = 1; call <= ncalls; ++call) {
    slab::slab(kW, c.dt, c.ifsnow, c.irad == 1, sl, b.ni, b.nj, T, Q, P, a);
    cm1f_slab(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.nsl, c.ifsnow, c.irad, c.dt, rc.data(),
              zs.data(), dzs.data(), t3d.data(), qv3d.data(), p3d.data(), s.data(), tslb.data());
    for (int n = 0; n < NS; ++n) {
      auto h = d[n].to_host();
      size_t nbad = 0;
      double maxd = 0;
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          const double x = h(i, j, 1), y = s(i, j, n + 1);
          if (!(x == y)) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
        }
      INFO("call " << call << " " << kNames[n] << ": " << nbad << " differing points, max |diff| = " << maxd);
      REQUIRE(nbad == 0);
    }
    double maxd;
    const size_t nbad = count_diff(a.tslb.to_host(), tslb, maxd, [](int, int) { return true; });
    INFO("call " << call << " tslb: " << nbad << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }
  // land points did change (the test exercises the land branch)
  auto tsk = a.tsk.to_host();
  int nchanged = 0;
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i)
      if (s(i, j, 4) < 1.5 && tsk(i, j, 1) != tsk_start(i, j, 9)) ++nchanged;
  REQUIRE(nchanged > 0);
}

// getlanduse reads ./LANDUSE.TBL: run the Fortran in a case directory that has it.
struct InCaseDir {
  std::filesystem::path old;
  explicit InCaseDir(const std::string& case_name) : old(std::filesystem::current_path()) {
    const auto dir = std::filesystem::path(testing::case_namelist(case_name)).parent_path();
    REQUIRE(std::filesystem::exists(dir / "LANDUSE.TBL"));
    std::filesystem::current_path(dir);
  }
  ~InCaseDir() { std::filesystem::current_path(old); }
};

} // namespace

TEST_CASE("slab matches slab.F (SLAB/SLAB1D)", "[fortran_bridge][slab]") {
  unsigned seed = 7001;
  // CM1: 5 soil layers, radiation=.true., ifsnow=0, gsw=glw=0; dt=6 s gives
  // one soil sub-step, 60 s two, 200 s six
  for (double dt : {6.0, 60.0, 200.0}) {
    SlabCase c; c.dt = dt;
    check_slab(c, seed++, 4);
  }
  SlabCase sun; sun.sun = true; check_slab(sun, seed++, 4);
  SlabCase snow; snow.ifsnow = 1; snow.dt = 60.0; check_slab(snow, seed++, 4);
  SlabCase norad; norad.irad = 0; norad.sun = true; check_slab(norad, seed++, 3);
  SlabCase one; one.nsl = 1; one.dt = 30.0; check_slab(one, seed++, 4);   // single-layer slab (no soil)
  SlabCase one_snow; one_snow.nsl = 1; one_snow.ifsnow = 1; check_slab(one_snow, seed++, 3);
  SlabCase three; three.nsl = 3; three.dt = 120.0; check_slab(three, seed++, 3);
}

TEST_CASE("slab soil layers and initial tslb match init_surface.F", "[fortran_bridge][slab]") {
  std::mt19937_64 gen(7101);
  auto U = [&](double a, double b) { return std::uniform_real_distribution<double>(a, b)(gen); };
  const auto b = box(6, 4, 3, 2);
  const std::array<int, 3> lo2{b.ib, b.jb, 1}, hi2{b.ie, b.je, 1};
  H3 tsk = h3("tsk", lo2, hi2), tmn = h3("tmn", lo2, hi2);
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) { tsk(i, j, 1) = U(270.0, 320.0); tmn(i, j, 1) = U(280.0, 300.0); }
  FArray3 tskd("tsk", lo2, hi2), tmnd("tmn", lo2, hi2);
  tskd.deep_copy_from(tsk);
  tmnd.deep_copy_from(tmn);
  for (int nsl : {1, 2, 3, 5, 10}) {
    INFO("num_soil_layers = " << nsl);
    const auto sl = slab::SoilLayers::make(nsl);
    std::vector<double> zs(nsl, 0.0), dzs(nsl, 0.0);
    H3 tslb = h3("tslb", lo2, {b.ie, b.je, nsl});
    for (size_t n = 0; n < tslb.size(); ++n) tslb.data()[n] = -7.25;
    FArray3 tslbd("tslb", lo2, {b.ie, b.je, nsl});
    tslbd.fill(-7.25);
    cm1f_slab_soil(nsl, b.ib, b.ie, b.jb, b.je, tsk.data(), tmn.data(), zs.data(), dzs.data(), tslb.data());
    slab::init_tslb(sl, b, tskd, tmnd, tslbd);
    REQUIRE(sl.n == nsl);
    for (int l = 0; l < nsl; ++l) {
      REQUIRE(sl.zs[l] == zs[l]);
      REQUIRE(sl.dzs[l] == dzs[l]);
    }
    double maxd;
    const size_t nbad = count_diff(tslbd.to_host(), tslb, maxd, [](int, int) { return true; });
    INFO("tslb: " << nbad << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }
  REQUIRE_THROWS(slab::SoilLayers::make(0));
  REQUIRE_THROWS(slab::SoilLayers::make(slab::max_soil_layers + 1));
}

TEST_CASE("sfc LANDUSE.TBL emissivity and thermal inertia match getlanduse", "[fortran_bridge][slab][sfc]") {
  InCaseDir in("ysu_dry");
  for (int season : {1, 2})
    for (int lu = 1; lu <= 33; ++lu) {
      double emiss = -1, thc = -1;
      cm1f_getlanduse(season, lu, &emiss, &thc);
      INFO("season " << season << " category " << lu);
      REQUIRE(sfc::landuse_sfem(lu, season) == emiss);
      REQUIRE(sfc::landuse_therin(lu, season) / 100. == thc);
    }
  REQUIRE_THROWS(sfc::landuse_sfem(0, 1));
  REQUIRE_THROWS(sfc::landuse_therin(34, 1));
}
