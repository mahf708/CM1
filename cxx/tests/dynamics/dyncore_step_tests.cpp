// Step-level tests of the dynamical core: start from the Fortran reference
// dump step_000000, advance with Model::step, and compare the prognostic
// arrays (halos included) against every dumped step.
#include "driver/cm1k_model.hpp"
#include "dynamics/cm1k_dyncore.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace cm1k;

namespace {

// Largest relative difference (|a-b| / max|ref|) over all compared fields.
double run_case(const std::string& case_name, double tol, bool native_init = false) {
  const auto ref = cm1k::testing::case_ref(case_name);
  if (ref.empty()) return 0;
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  if (native_init) m.init();   // init/ port: param.F, base.F, init3d.F
  else             m.init_from_dump(DumpReader::step_dir(ref, 0));

  std::vector<std::string> fields = {"ua", "va", "wa", "ppi", "tha", "rho", "prs"};
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  // dumped steps (every step, or every N steps for some cases)
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && n != "step_000000") steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  REQUIRE(!steps.empty());
  const int nsteps = int(steps.size());
  double worst = 0;
  for (int step : steps) {
    const auto dir = DumpReader::step_dir(ref, step);
    while (m.nstep() < step) m.step();
    DumpReader r(dir);
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    std::string line = case_name + " step " + std::to_string(step) + ":";
    for (auto& n : fields) {
      auto a = m.field3(n).to_host();
      auto b = r.read_host<3>(n);
      std::array<int, 3> lo, hi;
      for (int d = 0; d < 3; ++d) { lo[d] = b.lb(d); hi[d] = b.ub(d); }
      const auto st = diff<3>(a, b, lo, hi);
      char buf[128];
      std::snprintf(buf, sizeof buf, " %s %.2e/%.2e", n.c_str(), st.max_abs, st.rel);
      line += buf;
      worst = std::max(worst, st.rel);
      INFO(case_name << " step " << step << " field " << n << ": max abs " << st.max_abs << " rel " << st.rel);
      CHECK(st.rel <= tol);
    }
    std::printf("%s\n", line.c_str());
  }
  std::printf("%s: worst relative difference over %d dumps (to step %d): %.3e\n", case_name.c_str(), nsteps,
              steps.back(), worst);
  return worst;
}

} // namespace

TEST_CASE("dyncore: dry_bubble vs Fortran, every step", "[dyn][step]") {
  run_case("dry_bubble", 1e-12);
}

TEST_CASE("dyncore: dry_shear vs Fortran, every step", "[dyn][step]") {
  run_case("dry_shear", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_saturated vs Fortran, every step", "[dyn][step][kessler]") {
  run_case("kessler_saturated", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_bubble vs Fortran", "[dyn][step][kessler]") {
  run_case("kessler_bubble", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_periodic vs Fortran", "[dyn][step][kessler]") {
  run_case("kessler_periodic", 1e-12);
}

TEST_CASE("dyncore: dry_bubble from Model::init (no dump), every step", "[dyn][step][init]") {
  run_case("dry_bubble", 1e-12, true);
}

TEST_CASE("dyncore + Kessler: kessler_saturated from Model::init (no dump)", "[dyn][step][init][kessler]") {
  run_case("kessler_saturated", 1e-12, true);
}

TEST_CASE("dyncore: option checks", "[dyn]") {
  auto nl = Namelist::from_file(cm1k::testing::case_namelist("dry_bubble"));
  nl.set("psolver", "2");
  REQUIRE_THROWS(DynConfig::from_namelist(nl).check());
  nl.set("psolver", "3");
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());
  nl.set("terrain_flag", ".true.");
  REQUIRE_THROWS(DynConfig::from_namelist(nl).check());
  nl.set("terrain_flag", ".false.");
  nl.set("wbc", "2");
  REQUIRE_THROWS(DynConfig::from_namelist(nl).check());
}
