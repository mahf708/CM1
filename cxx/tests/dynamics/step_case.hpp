// run_case: shared by the step-test executables of this group. Start from
// the Fortran reference dump step_000000 (or a mid-run dump), advance with
// Model::step, and compare the prognostic arrays (halos included) against
// every dumped step.
#ifndef CM1K_STEP_CASE_HPP
#define CM1K_STEP_CASE_HPP
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
  if (m.has_surface())   // sfcmodel>=1: every dumped surface array (tsk, znt, ust, thflux, ...)
    for (auto& [n, a] : m.surface().dumped()) fields.push_back(n);
  if (m.has_wrf_surface())   // sfcmodel=2,3: mol, and the SLAB soil temperatures tslb
    for (auto& [n, a] : m.wrf_surface().dumped()) fields.push_back(n);
  for (auto& n : m.pbl_field_names()) fields.push_back(n);   // ipbl=1,2: hpbl, thpten, ..., vpten
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  // dumped steps (every step, or every N steps for some cases)
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > start_step) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  cm1k::testing::limit_steps(steps);
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

#endif
