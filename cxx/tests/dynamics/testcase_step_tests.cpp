// Step-level tests of the pre-configured test cases (testcase = 1..15) and
// lspgrad = 2, 3, 4 (physics/testcase, the testcase branches of init/,
// physics/sfc, dynamics/cm1k_adv wsub, physics/rrtmg): start from the
// Fortran reference dump step_000000 (or from Model::init where cm1k has the
// case's sounding), advance with Model::step, and compare every array both
// write (the prognostics, turbulence, surface, PBL, radiation arrays, the
// forcing profiles wprof .. vlspg, frad, thraten, zir, hpbl), halos
// included, bit for bit at every dump.
#include "driver/cm1k_model.hpp"
#include "physics/testcase/cm1k_testcase.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

using namespace cm1k;

namespace {

// Largest |a - b| over every array of `m` that the dump `r` holds.
double compare_all(const Model& m, const DumpReader& r, const std::string& what) {
  double worst = 0;
  std::string worst_name;
  for (auto& n : m.field_names()) {
    if (!r.has(n)) continue;
    DiffStats st;
    if (m.field_rank(n) == 1) {
      auto a = m.field1(n).to_host();
      auto b = r.read_host<1>(n);
      st = diff<1>(a, b, {b.lb(0)}, {b.ub(0)});
    } else {
      auto a = m.field3(n).to_host();
      auto b = r.read_host<3>(n);
      std::array<int, 3> lo, hi;
      for (int d = 0; d < 3; ++d) { lo[d] = b.lb(d); hi[d] = b.ub(d); }
      st = diff<3>(a, b, lo, hi);
    }
    if (st.max_abs > worst) { worst = st.max_abs; worst_name = n; }
    INFO(what << " field " << n << ": max abs " << st.max_abs << " rel " << st.rel);
    CHECK(st.max_abs == 0.0);
  }
  if (worst > 0) std::printf("  %s: worst %s %.3e\n", what.c_str(), worst_name.c_str(), worst);
  return worst;
}

void run_case(const std::string& case_name, bool native_init = false) {
  const auto ref = cm1k::testing::case_ref(case_name);
  if (ref.empty()) return;
  if (!std::filesystem::exists(DumpReader::step_dir(ref, 0))) {
    WARN("missing " << DumpReader::step_dir(ref, 0));
    return;
  }
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  REQUIRE(m.testcase_forcing().allocated());
  const std::string label = case_name + (native_init ? " (Model::init)" : "");
  double worst = 0;
  if (native_init) {
    m.init();
    worst = compare_all(m, DumpReader(DumpReader::step_dir(ref, 0)), label + " step 0");
  } else {
    m.init_from_dump(DumpReader::step_dir(ref, 0));
  }
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > 0) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  cm1k::testing::limit_steps(steps);
  REQUIRE(!steps.empty());
  for (int step : steps) {
    while (m.nstep() < step) m.step();
    DumpReader r(DumpReader::step_dir(ref, step));
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    worst = std::max(worst, compare_all(m, r, label + " step " + std::to_string(step)));
  }
  std::printf("%s: max abs difference over %zu dumps (to step %d), all arrays: %.3e\n", label.c_str(),
              steps.size(), steps.back(), worst);
}

Namelist with(const std::string& case_name, std::initializer_list<std::pair<const char*, const char*>> kv) {
  auto nl = Namelist::from_file(cm1k::testing::case_namelist(case_name));
  for (auto& [k, v] : kv) nl.set(k, v);
  return nl;
}

} // namespace

TEST_CASE("testcase 1 (CBL, use_avg_sfc, max-gradient hpbl, lspgrad=2) vs Fortran", "[dyn][step][testcase]") {
  run_case("tc1_cbl");
}
TEST_CASE("testcase 1 from Model::init", "[dyn][step][init][testcase]") { run_case("tc1_cbl", true); }

TEST_CASE("testcase 2 (shear PBL, heat flux off after 3000 s) vs Fortran", "[dyn][step][testcase][long]") {
  run_case("tc2_shear_long");
}
TEST_CASE("testcase 2 from Model::init", "[dyn][step][init][testcase][long]") {
  run_case("tc2_shear_long", true);
}

TEST_CASE("testcase 3 (shallow Cu: dolsw, thfrc, qvfrc, ug/vg, TKE init) + Kessler vs Fortran",
          "[dyn][step][testcase][kessler]") {
  run_case("tc3_shcu");
}
TEST_CASE("testcase 3 from Model::init", "[dyn][step][init][testcase][kessler]") { run_case("tc3_shcu", true); }

TEST_CASE("testcase 4 (simple radiation, constants) + Kessler vs Fortran", "[dyn][step][testcase][kessler]") {
  run_case("tc4_simplerad");
}
TEST_CASE("testcase 4 from Model::init", "[dyn][step][init][testcase][kessler]") {
  run_case("tc4_simplerad", true);
}

TEST_CASE("testcase 5 (simple radiation, no surface layer) + Kessler vs Fortran", "[dyn][step][testcase][kessler]") {
  run_case("tc5_simplerad_nosfc");
}
TEST_CASE("testcase 5 from Model::init", "[dyn][step][init][testcase][kessler]") {
  run_case("tc5_simplerad_nosfc", true);
}

TEST_CASE("testcase 6 (hurricane BL: lspgrad=3, rotated, heat flux off after 3000 s) vs Fortran",
          "[dyn][step][testcase][long]") {
  run_case("tc6_hurr_long");
}
TEST_CASE("testcase 6 from Model::init", "[dyn][step][init][testcase][long]") {
  run_case("tc6_hurr_long", true);
}

TEST_CASE("testcase 7 (DYCOMS surface stress and fluxes) + Kessler vs Fortran", "[dyn][step][testcase][kessler]") {
  run_case("tc7_dycoms");
}
TEST_CASE("testcase 7 from Model::init", "[dyn][step][init][testcase][kessler]") { run_case("tc7_dycoms", true); }

TEST_CASE("testcase 8 (RRTMG with fixed solcon, coszen) + YSU + Kessler vs Fortran",
          "[dyn][step][testcase][rrtmg]") {
  run_case("tc8_rrtmg");
}
TEST_CASE("testcase 8 from Model::init", "[dyn][step][init][testcase][rrtmg]") { run_case("tc8_rrtmg", true); }

TEST_CASE("testcase 9 (stable BL: TKE init, use_avg_sfc) vs Fortran", "[dyn][step][testcase]") {
  run_case("tc9_sbl");
}
TEST_CASE("testcase 9 from Model::init", "[dyn][step][init][testcase]") { run_case("tc9_sbl", true); }

TEST_CASE("testcase 10 (theta/qv nudging, lspgrad=3, simple PBL) vs Fortran", "[dyn][step][testcase][pbl]") {
  run_case("tc10_nudge");
}
TEST_CASE("testcase 10 from Model::init", "[dyn][step][init][testcase][pbl]") { run_case("tc10_nudge", true); }

TEST_CASE("testcase 11 (SAS: TKE init, sfcflux) + Kessler vs Fortran", "[dyn][step][testcase][kessler]") {
  run_case("tc11_sas");
}
TEST_CASE("testcase 11 from Model::init", "[dyn][step][init][testcase][kessler]") { run_case("tc11_sas", true); }

TEST_CASE("testcase 12 (wind tunnel: lspgrad=4, velocity perturbations) vs Fortran", "[dyn][step][testcase]") {
  run_case("tc12_windtunnel");
}
TEST_CASE("testcase 12 from Model::init", "[dyn][step][init][testcase]") { run_case("tc12_windtunnel", true); }

TEST_CASE("testcase 13 (BL wind tunnel: u0, upstream znt, lspgrad=4) vs Fortran", "[dyn][step][testcase]") {
  run_case("tc13_blwt");
}
TEST_CASE("testcase 13 from Model::init", "[dyn][step][init][testcase]") { run_case("tc13_blwt", true); }

TEST_CASE("testcase 14 (SGP: time-dependent forcing, use_avg_sfc) + Kessler vs Fortran",
          "[dyn][step][testcase][kessler]") {
  run_case("tc14_sgp");
}
TEST_CASE("testcase 14 from Model::init", "[dyn][step][init][testcase][kessler]") { run_case("tc14_sgp", true); }

TEST_CASE("testcase 15 (hurricane coast: lspgrad=3, perturbations) vs Fortran", "[dyn][step][testcase]") {
  run_case("tc15_hurr");
}
TEST_CASE("testcase 15 from Model::init", "[dyn][step][init][testcase]") { run_case("tc15_hurr", true); }

// the same test cases without a surface layer (sfcmodel = 0, bbc = 1): the
// configurations the array port (cm1x) runs too
TEST_CASE("testcases without a surface layer vs Fortran, from the dump and from Model::init",
          "[dyn][step][init][testcase][nosfc]") {
  for (const char* c : {"tc3_nosfc", "tc4_nosfc", "tc6_nosfc", "tc7_nosfc", "tc9_nosfc", "tc10_les", "tc11_nosfc",
                        "tc13_nosfc", "tc14_nosfc"}) {
    run_case(c);
    run_case(c, true);
  }
}

TEST_CASE("testcase options: unsupported combinations throw", "[testcase]") {
  const std::string c = "tc3_shcu";
  // sfcmodel=5 (cm1most) has its own testcase branches
  CHECK_THROWS_AS(Model(with(c, {{"sfcmodel", "5"}})), std::runtime_error);
  // testcase 1..7 with the WRF surface layers
  CHECK_THROWS_AS(Model(with(c, {{"sfcmodel", "2"}})), std::runtime_error);
  // simple radiation with RRTMG (param.F stops) or a PBL scheme
  CHECK_THROWS_AS(Model(with("tc4_simplerad", {{"radopt", "2"}})), std::runtime_error);
  CHECK_THROWS_AS(Model(with("tc4_simplerad", {{"cm1setup", "2"}, {"ipbl", "2"}, {"sgsmodel", "0"}})),
                  std::runtime_error);
  // simple radiation without liquid water
  CHECK_THROWS_AS(Model(with("tc5_simplerad_nosfc", {{"ptype", "0"}})), std::runtime_error);
  // dolsw with terrain (param.F stops)
  CHECK_THROWS_AS(Model(with(c, {{"terrain_flag", ".true."}, {"itern", "1"}})), std::runtime_error);
  // no such testcase / lspgrad
  CHECK_THROWS_AS(Model(with(c, {{"testcase", "16"}})), std::runtime_error);
  CHECK_THROWS_AS(Model(with(c, {{"lspgrad", "5"}})), std::runtime_error);
  // lspgrad=3 without hurr_rad
  CHECK_THROWS_AS(Model(with("tc15_hurr", {{"hurr_rad", "0.0"}})), std::runtime_error);
}

TEST_CASE("testcase options: param.F's imoist override", "[testcase]") {
  // testcase 3 is moist even with imoist = 0 in the namelist; 1 is dry
  auto nl = testcase::param_overrides(with("tc3_shcu", {{"imoist", "0"}}));
  CHECK(nl.geti("imoist", 0) == 1);
  nl = testcase::param_overrides(with("tc1_cbl", {{"imoist", "1"}}));
  CHECK(nl.geti("imoist", 1) == 0);
}
