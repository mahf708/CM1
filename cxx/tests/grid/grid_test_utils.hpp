// Shared helpers of the stretched-grid and adaptive-time-step step tests:
// compare every array a Model registers with a reference dump.
#ifndef CM1K_GRID_TEST_UTILS_HPP
#define CM1K_GRID_TEST_UTILS_HPP

#include "driver/cm1k_model.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace cm1k::testing {

// Largest relative difference (|a-b| / max|ref|) over every field the model
// registers that the dump has (grid, base state, prognostics, turbulence,
// surface, PBL, ...); CHECKs each against tol.
inline double compare_all(Model& m, const DumpReader& r, double tol, const std::string& what) {
  double worst = 0;
  std::string line = what + ":";
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
    worst = std::max(worst, st.rel);
    if (st.max_abs > 0) {
      char buf[160];
      std::snprintf(buf, sizeof buf, " %s %.2e/%.2e", n.c_str(), st.max_abs, st.rel);
      line += buf;
    }
    INFO(what << " field " << n << ": max abs " << st.max_abs << " rel " << st.rel);
    CHECK(st.rel <= tol);
  }
  std::printf("%s (worst rel %.3e)\n", line.c_str(), worst);
  return worst;
}

// Dumped steps of a reference case after `after`.
inline std::vector<int> dumped_steps(const std::string& ref, int after) {
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > after) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  limit_steps(steps);
  return steps;
}

// Runs a reference case from its step_000000 dump (or Model::init, which is
// then compared with step_000000 too) through every dump; returns the worst
// relative difference.
inline double run_all_fields(const std::string& case_name, double tol, bool native_init) {
  const auto ref = case_ref(case_name);
  if (ref.empty()) return 0;
  Model m(Namelist::from_file(case_namelist(case_name)));
  double worst = 0;
  if (native_init) {
    m.init();
    DumpReader r0(DumpReader::step_dir(ref, 0));
    worst = compare_all(m, r0, tol, case_name + " (Model::init) step 0");
  } else {
    m.init_from_dump(DumpReader::step_dir(ref, 0));
  }
  const auto steps = dumped_steps(ref, 0);
  REQUIRE(!steps.empty());
  for (int step : steps) {
    while (m.nstep() < step) m.step();
    DumpReader r(DumpReader::step_dir(ref, step));
    REQUIRE(r.nstep() == m.nstep());
    INFO(case_name << " step " << step << ": mtime " << m.mtime() << " vs " << r.mtime());
    CHECK(r.mtime() == m.mtime());
    worst = std::max(worst, compare_all(m, r, tol, case_name + (native_init ? " (Model::init)" : "") + " step " +
                                                       std::to_string(step)));
  }
  std::printf("%s%s: worst relative difference over %zu dumps (to step %d): %.3e\n", case_name.c_str(),
              native_init ? " (Model::init)" : "", steps.size(), steps.back(), worst);
  return worst;
}

} // namespace cm1k::testing

#endif
