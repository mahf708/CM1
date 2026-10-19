// base.F's soundings (isnd) and wind profiles (iwnd) vs the Fortran base
// routine (cm1f_base shim), bit for bit, on every array base sets and on
// its full bounds: every isnd/iwnd of base.F, with the namelist settings
// each depends on (imoist, ptype/iice, eqtset, tsk0, testcase, hurr_vg,
// hurr_rotate, imove, lateral boundaries), on a stretched deep grid and a
// uniform shallow one. The cases in run/config_files use isnd 1, 7, 8, 9, 5,
// 14, 15, 18, 19, 20, 22, 23 and iwnd 0, 1, 2, 6, 8, 9, 10, 11; each is here
// with that case's settings.
#include "init/cm1k_base_snd.hpp"
#include "init/cm1k_init.hpp"
#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void cm1f_base(const int* iparams, const double* rparams, const double* zh, const double* mh,
                          const double* rmh, const double* c1, const double* c2, const double* zf,
                          const double* mf, double* pi0, double* prs0, double* rho0, double* thv0, double* th0,
                          double* rth0, double* qv0, double* qc0, double* qi0, double* rr0, double* rf0,
                          double* rrf0, double* u0, double* v0);

using namespace cm1k;

namespace {

// a small 3D domain; grid "deep" is stretched (stretch_z = 1, 100 m at the
// bottom to 900 m at the top, ztop = 19.5 km), "shallow" uniform 60 m to
// 2.4 km, "pbl" uniform 25 m to 1.5 km (the boundary-layer soundings)
Namelist base_namelist(const std::string& grid) {
  std::string s = R"(
 &param0 nx = 5, ny = 4, nz = 35, terrain_flag = .false. /
 &param1 dx = 1000.0, dy = 1000.0, dz = 500.0, dtl = 1.0 /
 &param2 cm1setup = 1, testcase = 0, imoist = 0, ptype = 0, psolver = 3, eqtset = 2,
         isnd = 1, iwnd = 0, imove = 0, iorigin = 1, sgsmodel = 0, sfcmodel = 0,
         wbc = 1, ebc = 1, sbc = 1, nbc = 1 /
 &param3 umove = 0.0, vmove = 0.0 /
 &param6 stretch_z = 0, ztop = 19500.0, str_bot = 500.0, str_top = 10500.0, dz_bot = 100.0, dz_top = 900.0 /
 &param12 tsk0 = 299.28 /
 &param21 hurr_vg = 40.0, hurr_rad = 40000.0, hurr_vgpl = -0.70, hurr_rotate = 0.0 /
)";
  auto nl = Namelist::from_string(s);
  if (grid == "deep") {
    nl.set("stretch_z", "1");
  } else if (grid == "shallow") {
    nl.set("nz", "40");
    nl.set("dz", "60.0");
  } else if (grid == "pbl") {
    nl.set("nz", "60");
    nl.set("dz", "25.0");
  } else {
    throw std::runtime_error("unknown grid " + grid);
  }
  return nl;
}

void write_sounding_files() {
  // the shipped soundings: hurricane cases (15 levels, qv = 0 at the top,
  // which base.F replaces by 0.01% RH) and sea_breeze (80 levels, E-format)
  for (const char* cs : {"hurricane_axisymmetric", "sea_breeze"}) {
    std::ifstream src(std::string(CM1K_SOURCE_RUN_DIR) + "/config_files/" + cs + "/input_sounding");
    REQUIRE(src.good());
    std::ofstream dst(std::string("input_sounding_") + cs);
    dst << src.rdbuf();
  }
  // winds varying with height, uneven levels, comma separators, d exponents
  std::ofstream w("input_sounding_wind");
  w << "  1000.0 , 300.0, 14.0\n";
  for (int k = 1; k <= 60; ++k) {
    const double z = 37.0 * k + 3.1 * k * k + 0.4 * k * k * k;
    const double th = 300.0 + 0.004 * z + 1.0e-7 * z * z;
    const double qv = 14.0 * std::exp(-z / 2500.0);
    const double u = 3.0 + 12.0 * std::tanh(z / 3000.0), v = -2.0 + 1.0e-3 * z - 4.0e-8 * z * z;
    char buf[200];
    std::snprintf(buf, sizeof buf, "%.4f, %.5f %.6fd0 %.3E   %.4f\n", z, th, qv, u, v);
    w << buf;
  }
}

struct Run {
  std::string label, grid;
  std::map<std::string, std::string> set;
  std::string sounding;   // input_sounding source for isnd = 7, 17
};

void compare(const Run& r) {
  auto nl = base_namelist(r.grid);
  for (const auto& kv : r.set) nl.set(kv.first, kv.second);
  if (!r.sounding.empty()) {
    std::ifstream src(r.sounding);
    std::ofstream dst("input_sounding");
    dst << src.rdbuf();
  }
  const int testcase = nl.geti("testcase", 0);
  const auto c = Constants::make(testcase);
  const auto b = Bounds::from_namelist(nl);
  Grid g;
  g.allocate(b);
  BaseState bs;
  bs.allocate(b);
  auto info = init::init_grid(nl, g);
  init::init_base_state(nl, c, g, info, bs);

  // Fortran
  const int iparams[16] = {b.ni, b.nj, b.nk, nl.geti("isnd", 1), nl.geti("iwnd", 0), nl.geti("imoist", 0),
                           nl.geti("ptype", 0), nl.geti("psolver", 3), nl.geti("imove", 0), testcase,
                           init::param_eqtset(nl), init::param_iice(nl), nl.geti("wbc", 1), nl.geti("ebc", 1),
                           nl.geti("sbc", 1), nl.geti("nbc", 1)};
  const auto& ex = info.ex;
  const int imove = nl.geti("imove", 0);
  const double rparams[17] = {g.dx, g.dy, g.dz,
                              imove ? nl.getr("umove", 0.0) : 0.0, imove ? nl.getr("vmove", 0.0) : 0.0,
                              nl.getr("tsk0", 299.28), nl.getr("hurr_vg", 40.0), nl.getr("hurr_rotate", 0.0),
                              nl.getr("hurr_rad", 40000.0), nl.getr("hurr_vgpl", -0.7),
                              ex.cgs1, ex.cgs2, ex.cgs3, ex.cgt1, ex.cgt2, ex.cgt3, 0.0001};
  auto zh = g.zh.to_host(), mh = g.mh.to_host(), rmh = g.rmh.to_host(), c1 = g.c1.to_host(), c2 = g.c2.to_host();
  auto zf = g.zf.to_host(), mf = g.mf.to_host();
  const std::vector<std::string> names = {"pi0", "prs0", "rho0", "thv0", "th0", "rth0", "qv0",
                                          "qc0", "qi0",  "rr0",  "rf0",  "rrf0", "u0",  "v0"};
  const std::vector<FArray3*> cxx = {&bs.pi0, &bs.prs0, &bs.rho0, &bs.thv0, &bs.th0, &bs.rth0, &bs.qv0,
                                     &bs.qc0, &bs.qi0,  &bs.rr0,  &bs.rf0,  &bs.rrf0, &bs.u0,  &bs.v0};
  std::vector<std::vector<double>> f(names.size());
  for (size_t n = 0; n < names.size(); ++n) f[n].assign(cxx[n]->view().size(), -12345.0);
  cm1f_base(iparams, rparams, zh.data(), mh.data(), rmh.data(), c1.data(), c2.data(), zf.data(), mf.data(),
            f[0].data(), f[1].data(), f[2].data(), f[3].data(), f[4].data(), f[5].data(), f[6].data(),
            f[7].data(), f[8].data(), f[9].data(), f[10].data(), f[11].data(), f[12].data(), f[13].data());

  for (size_t n = 0; n < names.size(); ++n) {
    auto h = cxx[n]->to_host();
    const double* a = h.data();
    double maxabs = 0.0;
    size_t where = 0, nbad = 0;
    for (size_t m = 0; m < f[n].size(); ++m) {
      const double d = std::abs(a[m] - f[n][m]);
      if (!(a[m] == f[n][m]) && !(std::isnan(a[m]) && std::isnan(f[n][m]))) {
        ++nbad;
        if (!(d <= maxabs)) { maxabs = d; where = m; }
      }
    }
    INFO(r.label << ": " << names[n] << " differs at " << nbad << " points, max |diff| " << maxabs
                 << " at flat index " << where << " (" << (nbad ? a[where] : 0.0) << " vs "
                 << (nbad ? f[n][where] : 0.0) << ")");
    CHECK(nbad == 0);
  }
}

} // namespace

TEST_CASE("base.F isnd/iwnd vs Fortran", "[bridge][init][base]") {
  write_sounding_files();
  using M = std::map<std::string, std::string>;
  const M moist1 = {{"imoist", "1"}, {"ptype", "1"}};
  auto with = [](M a, const M& b) {
    for (const auto& kv : b) a[kv.first] = kv.second;
    return a;
  };
  std::vector<Run> runs = {
      // ---- soundings (zero wind unless noted)
      {"isnd1", "deep", {{"isnd", "1"}}, ""},
      {"isnd1 les_ib_windtunnel (iwnd6 tc12, psolver6)", "pbl",
       {{"isnd", "1"}, {"iwnd", "6"}, {"testcase", "12"}, {"psolver", "6"}}, ""},
      {"isnd2", "deep", {{"isnd", "2"}}, ""},
      {"isnd3", "deep", {{"isnd", "3"}}, ""},
      {"isnd4 kessler", "deep", with(moist1, {{"isnd", "4"}}), ""},
      {"isnd5 moist", "deep", with(moist1, {{"isnd", "5"}, {"iwnd", "2"}}), ""},
      {"isnd5 dry", "deep", {{"isnd", "5"}, {"iwnd", "1"}}, ""},
      {"isnd5 ptype4", "deep", {{"isnd", "5"}, {"imoist", "1"}, {"ptype", "4"}}, ""},
      {"isnd7 hurricane", "deep", with(moist1, {{"isnd", "7"}}), "input_sounding_hurricane_axisymmetric"},
      {"isnd7 sea_breeze", "shallow", with(moist1, {{"isnd", "7"}, {"wbc", "2"}, {"ebc", "2"}}),
       "input_sounding_sea_breeze"},
      {"isnd7 wind moist", "deep", with(moist1, {{"isnd", "7"}}), "input_sounding_wind"},
      {"isnd7 wind dry imove", "deep", {{"isnd", "7"}, {"imove", "1"}, {"umove", "3.5"}, {"vmove", "-1.25"}},
       "input_sounding_wind"},
      {"isnd7 wind open bc", "deep", with(moist1, {{"isnd", "7"}, {"wbc", "2"}, {"ebc", "2"}, {"sbc", "3"}, {"nbc", "4"}}),
       "input_sounding_wind"},
      {"isnd17 + iwnd2", "deep", with(moist1, {{"isnd", "17"}, {"iwnd", "2"}}), "input_sounding_wind"},
      {"isnd17 dry + iwnd5", "deep", {{"isnd", "17"}, {"iwnd", "5"}}, "input_sounding_hurricane_axisymmetric"},
      {"isnd8 dns_RayleighBenard", "pbl", {{"isnd", "8"}}, ""},
      {"isnd8 les_HurrBoundLayer (tc6, iwnd8)", "pbl", {{"isnd", "8"}, {"iwnd", "8"}, {"testcase", "6"}}, ""},
      {"isnd8 scm_HurrBoundLayer_tqnudge (tc10, iwnd8)", "pbl",
       {{"isnd", "8"}, {"iwnd", "8"}, {"testcase", "10"}, {"imoist", "1"}, {"ptype", "6"}}, ""},
      {"isnd8 les_HurrCoast (tc15, iwnd8, rotate -90)", "pbl",
       {{"isnd", "8"}, {"iwnd", "8"}, {"testcase", "15"}, {"imoist", "1"}, {"ptype", "0"}, {"eqtset", "1"},
        {"psolver", "2"}, {"hurr_rotate", "-90.0"}, {"hurr_vg", "35.0"}}, ""},
      {"isnd9", "deep", {{"isnd", "9"}, {"iwnd", "6"}}, ""},
      {"isnd9 moist", "deep", with(moist1, {{"isnd", "9"}}), ""},
      {"isnd10 kessler eqtset2", "deep", with(moist1, {{"isnd", "10"}}), ""},
      {"isnd10 kessler eqtset1", "deep", with(moist1, {{"isnd", "10"}, {"eqtset", "1"}}), ""},
      {"isnd10 ice (ptype5)", "deep", {{"isnd", "10"}, {"imoist", "1"}, {"ptype", "5"}}, ""},
      {"isnd10 ice (ptype2, goddard t0k)", "deep", {{"isnd", "10"}, {"imoist", "1"}, {"ptype", "2"}}, ""},
      {"isnd11", "deep", with(moist1, {{"isnd", "11"}, {"tsk0", "301.15"}}), ""},
      {"isnd12", "pbl", {{"isnd", "12"}}, ""},
      {"isnd13", "deep", {{"isnd", "13"}, {"iwnd", "3"}}, ""},
      {"isnd14 les_ConvBoundLayer (tc1, iwnd6)", "pbl", {{"isnd", "14"}, {"iwnd", "6"}, {"testcase", "1"}}, ""},
      {"isnd15 les_StratoCuNoPrecip (tc4, iwnd6)", "pbl",
       {{"isnd", "15"}, {"iwnd", "6"}, {"testcase", "4"}, {"imoist", "1"}, {"ptype", "6"}}, ""},
      {"isnd15 les_StratoCuDrizzle (tc5, iwnd10)", "pbl",
       {{"isnd", "15"}, {"iwnd", "10"}, {"testcase", "5"}, {"imoist", "1"}, {"ptype", "5"}}, ""},
      {"isnd18 les_ShearBoundLayer (tc2, iwnd6)", "pbl", {{"isnd", "18"}, {"iwnd", "6"}, {"testcase", "2"}}, ""},
      {"isnd19 les_ShallowCu (tc3, iwnd9)", "shallow",
       {{"isnd", "19"}, {"iwnd", "9"}, {"testcase", "3"}, {"imoist", "1"}, {"ptype", "5"}}, ""},
      {"isnd19 les_ConvPBL_moisture (tc11, iwnd6)", "deep",
       {{"isnd", "19"}, {"iwnd", "6"}, {"testcase", "11"}, {"imoist", "1"}, {"ptype", "0"}, {"eqtset", "1"},
        {"psolver", "7"}}, ""},
      {"isnd19 dry", "shallow", {{"isnd", "19"}, {"iwnd", "9"}}, ""},
      {"isnd20 les_ShallowCuPrecip (tc7, iwnd11)", "shallow",
       {{"isnd", "20"}, {"iwnd", "11"}, {"testcase", "7"}, {"imoist", "1"}, {"ptype", "5"}}, ""},
      {"isnd21", "deep", with(moist1, {{"isnd", "21"}, {"tsk0", "300.0"}}), ""},
      {"isnd22 les_StableBoundLayer (tc9, iwnd6)", "pbl",
       {{"isnd", "22"}, {"iwnd", "6"}, {"testcase", "9"}, {"eqtset", "1"}, {"psolver", "2"}}, ""},
      {"isnd23 les_ShallowCuLand (tc14, iwnd6)", "shallow",
       {{"isnd", "23"}, {"iwnd", "6"}, {"testcase", "14"}, {"imoist", "1"}, {"ptype", "6"}, {"eqtset", "1"}}, ""},
  };
  // ---- every iwnd on the deep grid, with imove and open boundaries
  for (int iwnd = 0; iwnd <= 11; ++iwnd) {
    const auto w = std::to_string(iwnd);
    runs.push_back({"iwnd" + w, "deep", {{"isnd", "5"}, {"iwnd", w}}, ""});
    runs.push_back({"iwnd" + w + " imove open", "deep",
                    {{"isnd", "1"}, {"iwnd", w}, {"imove", "1"}, {"umove", "7.5"}, {"vmove", "2.25"},
                     {"wbc", "2"}, {"ebc", "3"}, {"sbc", "2"}, {"nbc", "4"}, {"hurr_rotate", "30.0"}}, ""});
  }
  for (const char* tc : {"0", "1", "2", "4", "9", "11", "12", "14", "8"})
    runs.push_back({std::string("iwnd6 testcase ") + tc, "pbl", {{"isnd", "1"}, {"iwnd", "6"}, {"testcase", tc}}, ""});
  for (const auto& r : runs) {
    INFO(r.label);
    compare(r);
  }
}

TEST_CASE("base.F options that stop", "[bridge][init][base]") {
  using M = std::map<std::string, std::string>;
  const std::vector<M> bad = {
      {{"isnd", "6"}},
      {{"isnd", "16"}},
      {{"isnd", "24"}},
      {{"isnd", "2"}, {"imoist", "1"}, {"ptype", "1"}},
      {{"isnd", "11"}},                         // requires imoist = 1
      {{"isnd", "15"}, {"testcase", "0"}},      // requires testcase 4 or 5
      {{"isnd", "5"}, {"iwnd", "12"}},
      {{"isnd", "7"}},                          // input_sounding missing
  };
  for (const auto& m : bad) {
    auto nl = base_namelist("deep");
    for (const auto& kv : m) nl.set(kv.first, kv.second);
    std::string label;
    for (const auto& kv : m) label += kv.first + "=" + kv.second + " ";
    INFO(label);
    setenv("CM1K_INPUT_SOUNDING", "no_such_input_sounding", 1);
    const auto b = Bounds::from_namelist(nl);
    Grid g;
    g.allocate(b);
    BaseState bs;
    bs.allocate(b);
    auto info = init::init_grid(nl, g);
    REQUIRE_THROWS_AS(init::init_base_state(nl, Constants::make(nl.geti("testcase", 0)), g, info, bs),
                      std::runtime_error);
    unsetenv("CM1K_INPUT_SOUNDING");
  }
}
