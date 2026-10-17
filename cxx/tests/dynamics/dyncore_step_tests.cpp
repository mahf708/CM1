// Step-level tests of the dynamical core: start from the Fortran reference
// dump step_000000, advance with Model::step, and compare the prognostic
// arrays (halos included) against every dumped step.
#include "driver/cm1k_model.hpp"
#include "dynamics/cm1k_dyncore.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace cm1k;

namespace {

// Largest relative difference (|a-b| / max|ref|) over all compared fields.
// start_step > 0 restarts from that (mid-run) dump instead of step_000000.
double run_case(const std::string& case_name, double tol, bool native_init = false, bool turb = false,
                int start_step = 0, const std::string& ref_root = "") {
  const auto ref = ref_root.empty() ? cm1k::testing::case_ref(case_name) : ref_root + "/" + case_name;
  if (ref.empty()) return 0;
  if (!std::filesystem::exists(DumpReader::step_dir(ref, start_step))) {
    WARN("missing " << DumpReader::step_dir(ref, start_step));
    return 0;
  }
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  if (native_init) m.init();   // init/ port: param.F, base.F, init3d.F
  else             m.init_from_dump(DumpReader::step_dir(ref, start_step));

  std::vector<std::string> fields = {"ua", "va", "wa", "ppi", "tha", "rho", "prs"};
  if (turb) for (const char* n : {"tkea", "kmh", "kmv", "khh", "khv"}) fields.push_back(n);
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  // dumped steps (every step, or every N steps for some cases)
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > start_step) steps.push_back(std::stoi(n.substr(5)));
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
    std::string line = case_name + (start_step > 0 ? " (from " + std::to_string(start_step) + ")" : std::string()) +
                       " step " + std::to_string(step) + ":";
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

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke vs Fortran", "[dyn][step][kessler][turb]") {
  // sgsmodel=1: also compares tkea and the eddy viscosities (dump N holds
  // the K computed from state N-1, as the model state between steps does)
  run_case("kessler_tke", 0.0, false, true);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke from Model::init (no dump)", "[dyn][step][init][kessler][turb]") {
  run_case("kessler_tke", 0.0, true, true);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke restarted from a mid-run dump", "[dyn][step][kessler][turb]") {
  // Needs consecutive dumps (e.g. local every-step references of steps
  // 140..151 in CM1K_EXTRA_REF_DIR/kessler_tke). Restarting from dump N, the
  // first step must recompute the eddy viscosities from state N exactly as
  // the Fortran did (dump N+1 holds them). The other fields of the restarted
  // run are only approximate: ppx and the diabatic tendencies (qpten, ...)
  // of the previous step are not in the dumps (see Dyncore::start_from_state).
  const char* e = std::getenv("CM1K_EXTRA_REF_DIR");
  if (!e) return;
  const std::string ref = std::string(e) + "/kessler_tke";
  if (!std::filesystem::exists(ref)) return;
  int nchecked = 0;
  for (auto& d : std::filesystem::directory_iterator(ref)) {
    const auto n = d.path().filename().string();
    if (n.rfind("step_", 0) != 0) continue;
    const int step = std::stoi(n.substr(5));
    if (!std::filesystem::exists(DumpReader::step_dir(ref, step + 1))) continue;
    Model m(Namelist::from_file(cm1k::testing::case_namelist("kessler_tke")));
    m.init_from_dump(DumpReader::step_dir(ref, step));
    m.step();
    DumpReader r(DumpReader::step_dir(ref, step + 1));
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    for (const char* f : {"kmh", "kmv", "khh", "khv"}) {
      auto a = m.field3(f).to_host();
      auto b = r.read_host<3>(f);
      std::array<int, 3> lo, hi;
      for (int k = 0; k < 3; ++k) { lo[k] = b.lb(k); hi[k] = b.ub(k); }
      const auto st = diff<3>(a, b, lo, hi);
      INFO("restart from step " << step << ": " << f << " max abs " << st.max_abs);
      CHECK(st.max_abs == 0.0);
    }
    ++nchecked;
  }
  std::printf("kessler_tke: restart K checked from %d dumps\n", nchecked);
}

TEST_CASE("dyncore + Kessler: kessler_openbc (open-radiative bc, irbc=4) vs Fortran", "[dyn][step][kessler][openbc]") {
  // wbc=ebc=sbc=nbc=2: radbcew4/radbcns4 phase speeds carried between RK
  // stages and steps, ssopenbc* in the acoustic steps, inflow-zeroed fluxes
  // in advection and pdef
  run_case("kessler_openbc", 0.0);
}

TEST_CASE("dyncore: dry_openbc (open bc, irbc=2, roflux=1, nudgeobc=1) vs Fortran", "[dyn][step][openbc]") {
  run_case("dry_openbc", 0.0);
}

TEST_CASE("dyncore: dry_openbc_walls (open E/W, walls S/N, irbc=1) vs Fortran", "[dyn][step][openbc]") {
  run_case("dry_openbc_walls", 0.0);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke_openbc (open E/W, walls S/N) vs Fortran",
          "[dyn][step][kessler][turb][openbc]") {
  run_case("kessler_tke_openbc", 0.0, false, true);
}

TEST_CASE("dyncore + Kessler: kessler_openbc from Model::init (no dump)", "[dyn][step][init][kessler][openbc]") {
  run_case("kessler_openbc", 0.0, true);
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
  // lateral boundaries: 1..4 are ported (2 with irbc = 1, 2, 4)
  nl.set("wbc", "2"); nl.set("ebc", "2");
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"irbc", "3"}, {"irbc", "0"}, {"roflux", "2"}, {"nudgeobc", "2"}, {"wbc", "5"}}) {
    auto x = nl;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  nl.set("wbc", "1"); nl.set("ebc", "1");
  nl.set("irbc", "3");   // irbc only matters for open boundaries
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());

  // subgrid turbulence: sgsmodel=1 (LES TKE) is ported, other options throw
  auto tk = Namelist::from_file(cm1k::testing::case_namelist("kessler_tke"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(tk).check());
  REQUIRE(DynConfig::from_namelist(tk).turb);
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"sgsmodel", "2"}, {"sgsmodel", "3"}, {"horizturb", "1"}, {"bbc", "2"}, {"tbc", "2"},
         {"isfcflx", "1"}, {"ipbl", "1"}, {"tconfig", "3"}}) {
    auto x = tk;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
}
