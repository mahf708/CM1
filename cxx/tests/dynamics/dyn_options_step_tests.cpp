// Step-level tests of the dyncore options Coriolis (icor=1), Rayleigh
// damping (irdamp=1 with zd below the model top, hrdamp=1) and explicit
// diffusion (idiff=1 with difforder=6 or 2): start from the Fortran reference
// dump step_000000 (or from Model::init), advance with Model::step, and
// compare every prognostic array (halos included) bit for bit at every dump.
#include "driver/cm1k_model.hpp"
#include "dynamics/cm1k_dyncore.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace cm1k;

namespace {

// Advances `m` through every dump of `ref` and returns the largest absolute
// difference over all prognostic fields and dumps (CHECKs each one is 0).
double run_bitwise(const std::string& case_name, Model& m, const std::string& ref) {
  std::vector<std::string> fields = {"ua", "va", "wa", "ppi", "tha", "rho", "prs"};
  if (DynConfig::from_namelist(m.namelist()).turb)
    for (const char* n : {"tkea", "kmh", "kmv", "khh", "khv"}) fields.push_back(n);
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > m.nstep()) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  cm1k::testing::limit_steps(steps);
  REQUIRE(!steps.empty());
  double worst = 0;
  for (int step : steps) {
    while (m.nstep() < step) m.step();
    DumpReader r(DumpReader::step_dir(ref, step));
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    for (auto& n : fields) {
      auto a = m.field3(n).to_host();
      auto b = r.read_host<3>(n);
      std::array<int, 3> lo, hi;
      for (int d = 0; d < 3; ++d) { lo[d] = b.lb(d); hi[d] = b.ub(d); }
      const auto st = diff<3>(a, b, lo, hi);
      worst = std::max(worst, st.max_abs);
      INFO(case_name << " step " << step << " field " << n << ": max abs " << st.max_abs << " rel " << st.rel);
      CHECK(st.max_abs == 0.0);
    }
  }
  std::printf("%s: max abs difference over %zu dumps (to step %d): %.3e\n", case_name.c_str(), steps.size(),
              steps.back(), worst);
  return worst;
}

void run_case(const std::string& case_name, bool native_init = false) {
  const auto ref = cm1k::testing::case_ref(case_name);
  if (ref.empty()) return;
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  if (native_init) m.init();
  else             m.init_from_dump(DumpReader::step_dir(ref, 0));
  run_bitwise(case_name + (native_init ? " (Model::init)" : ""), m, ref);
}

// Largest |a - b| over field `name` of two models.
double field_diff(const Model& a, const Model& b, const std::string& name) {
  auto x = a.field3(name).to_host();
  auto y = b.field3(name).to_host();
  std::array<int, 3> lo, hi;
  for (int d = 0; d < 3; ++d) { lo[d] = x.lb(d); hi[d] = x.ub(d); }
  return diff<3>(x, y, lo, hi).max_abs;
}

} // namespace

TEST_CASE("dyncore options: kessler_cor_diff6 (icor, irdamp+hrdamp, diff6, TKE) vs Fortran",
          "[dyn][step][kessler][turb][cor][rdamp][diff]") {
  run_case("kessler_cor_diff6");
}

TEST_CASE("dyncore options: dry_cor_damp (icor with imove, strong irdamp+hrdamp, diff6 mdiff=1) vs Fortran",
          "[dyn][step][cor][rdamp][diff]") {
  run_case("dry_cor_damp");
}

TEST_CASE("dyncore options: dry_cor_damp from Model::init (no dump)", "[dyn][step][init][cor][rdamp][diff]") {
  run_case("dry_cor_damp", true);
}

TEST_CASE("dyncore options: dry_diff2 (cm1setup=0, idiff=1 difforder=2, icor, irdamp) vs Fortran",
          "[dyn][step][cor][rdamp][diff]") {
  run_case("dry_diff2");
}

TEST_CASE("dyncore options: kessler_cor_diff6 from Model::init (no dump)",
          "[dyn][step][init][kessler][turb][cor][rdamp][diff]") {
  run_case("kessler_cor_diff6", true);
}

// Each option of the new cases must actually change the solution (so the
// bitwise step tests above exercise it): switching it off gives a different
// state after 30 steps (fewer when CM1K_MAX_STEP bounds the run).
TEST_CASE("dyncore options: every option of the new cases is active", "[dyn][cor][rdamp][diff]") {
  struct Off { const char* case_name; const char* key; const char* value; const char* field; };
  for (const auto& o : {Off{"kessler_cor_diff6", "icor", "0", "ua"}, Off{"kessler_cor_diff6", "irdamp", "0", "wa"},
                        Off{"kessler_cor_diff6", "hrdamp", "0", "wa"}, Off{"kessler_cor_diff6", "idiff", "0", "tha"},
                        Off{"dry_cor_damp", "icor", "0", "va"}, Off{"dry_cor_damp", "irdamp", "0", "tha"},
                        Off{"dry_cor_damp", "hrdamp", "0", "wa"}, Off{"dry_diff2", "idiff", "0", "ua"},
                        Off{"dry_diff2", "irdamp", "0", "ua"}}) {
    const int nsteps = std::min(30, cm1k::testing::max_step());
    const auto ref = cm1k::testing::case_ref(o.case_name);
    if (ref.empty()) return;
    auto nl = Namelist::from_file(cm1k::testing::case_namelist(o.case_name));
    Model on(nl);
    nl.set(o.key, o.value);
    Model off(nl);
    on.init_from_dump(DumpReader::step_dir(ref, 0));
    off.init_from_dump(DumpReader::step_dir(ref, 0));
    for (int n = 0; n < nsteps; ++n) { on.step(); off.step(); }
    const double d = field_diff(on, off, o.field);
    INFO(o.case_name << ": " << o.key << " = " << o.value << " changes " << o.field << " by " << d);
    std::printf("%s: %s=%s changes %s by up to %.3e after %d steps\n", o.case_name, o.key, o.value, o.field, d,
                nsteps);
    CHECK(d > 0.0);
  }
}

TEST_CASE("dyncore options: checks", "[dyn][cor][rdamp][diff]") {
  auto nl = Namelist::from_file(cm1k::testing::case_namelist("kessler_cor_diff6"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());
  {
    const auto c = DynConfig::from_namelist(nl);
    REQUIRE(c.diff6());
    REQUIRE(c.fcor == 0.0001);
    REQUIRE(c.umove == 0.0);   // imove = 0
  }
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"betaplane", "1"}, {"lspgrad", "1"}, {"irdamp", "2"}, {"idiff", "2"},
         {"difforder", "4"}, {"do_adapt_move", ".true."}}) {
    auto x = nl;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  // hrdamp follows irdamp when both are set (param.F); alone, only 0/1 are ported
  {
    auto x = nl;
    x.set("hrdamp", "2");
    REQUIRE(DynConfig::from_namelist(x).hrdamp == 1);
    x.set("irdamp", "0");
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  // icor=0 zeroes fcor (param.F); difforder=2 needs cm1setup=0 (else idiff -> 0)
  {
    auto x = nl;
    x.set("icor", "0");
    REQUIRE(DynConfig::from_namelist(x).fcor == 0.0);
    x.set("difforder", "2");
    REQUIRE(DynConfig::from_namelist(x).idiff == 0);
  }
  auto d2 = Namelist::from_file(cm1k::testing::case_namelist("dry_diff2"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(d2).check());
  REQUIRE(DynConfig::from_namelist(d2).diff2());
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{{"idiss", "1"}, {"bbc", "3"}, {"tbc", "3"}}) {
    auto x = d2;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  // the damping layer / lateral damping zone may not cover half the domain
  {
    auto x = Namelist::from_file(cm1k::testing::case_namelist("dry_cor_damp"));
    x.set("zd", "4000.0");
    Model m(x);
    m.init();
    REQUIRE_THROWS(m.step());
    x.set("zd", "5000.0");
    x.set("xhd", "40000.0");
    Model m2(x);
    m2.init();
    REQUIRE_THROWS(m2.step());
  }
}
