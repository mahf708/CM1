// Adaptive time step (adapt_dt = 1: cm1.F's initial getnewdt, misclibs.F
// calccflquick, calcksquick, getnewdt and dtsmall, the output-time ramp-
// down): step tests from the Fortran reference dumps and from Model::init,
// comparing every registered array bit for bit and the time step, nsound
// and output timers the Fortran recorded (src/cm1dump.F dump_adapt).
#include "grid_test_utils.hpp"

#include "dynamics/cm1k_dyncore.hpp"

#include <set>

using namespace cm1k;

namespace {

void run_adapt(const std::string& case_name, bool native_init) {
  const auto ref = cm1k::testing::case_ref(case_name);
  if (ref.empty()) return;
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  if (native_init) {
    m.init();
    DumpReader r0(DumpReader::step_dir(ref, 0));
    cm1k::testing::compare_all(m, r0, 0.0, case_name + " (Model::init) step 0");
  } else {
    m.init_from_dump(DumpReader::step_dir(ref, 0));
  }
  const auto steps = cm1k::testing::dumped_steps(ref, 0);
  REQUIRE(!steps.empty());
  std::set<double> dts;
  std::set<int> nsounds;
  double worst = 0;
  for (int step : steps) {
    while (m.nstep() < step) {
      m.step();
      dts.insert(m.dt());
      nsounds.insert(m.dyncore().nsound());
    }
    DumpReader r(DumpReader::step_dir(ref, step));
    REQUIRE(r.nstep() == m.nstep());
    const auto& a = r.adapt();
    REQUIRE(a.size() >= 10);
    INFO(case_name << " step " << step);
    CHECK(r.mtime() == m.mtime());
    CHECK(a[0] == double(m.dt()));
    CHECK(int(a[1]) == m.dyncore().nsound());
    CHECK(a[2] == double(m.adapt_dt().cflmax));
    CHECK(a[4] == m.adapt_dt().taptim);
    CHECK(a[5] == m.adapt_dt().stattim);
    CHECK(a[6] == m.adapt_dt().rsttim);
    CHECK((a[7] != 0) == m.adapt_dt().dowriteout);
    CHECK((a[8] != 0) == m.adapt_dt().dostat);
    std::printf("%s step %d: mtime %.10f dt %.10f nsound %d cflmax %.6f\n", case_name.c_str(), step, m.mtime(),
                double(m.dt()), m.dyncore().nsound(), double(m.adapt_dt().cflmax));
    worst = std::max(worst, cm1k::testing::compare_all(m, r, 0.0, case_name + " step " + std::to_string(step)));
  }
  // the case must exercise the adaptation: dt and nsound change
  CHECK(dts.size() > 2);
  CHECK(nsounds.size() > 1);
  std::printf("%s%s: %zu distinct dt, %zu distinct nsound; worst relative difference %.3e\n", case_name.c_str(),
              native_init ? " (Model::init)" : "", dts.size(), nsounds.size(), worst);
}

} // namespace

TEST_CASE("adapt_dt: adapt_dry (cm1setup=0, CFL-only dt, output/stat ramp-down) vs Fortran", "[grid][adapt][step]") {
  run_adapt("adapt_dry", false);
}

TEST_CASE("adapt_dt: adapt_dry from Model::init (initial getnewdt)", "[grid][adapt][step][init]") {
  run_adapt("adapt_dry", true);
}

TEST_CASE("adapt_dt: adapt_kessler_tke (LES TKE ksmax, Kessler) vs Fortran", "[grid][adapt][step][kessler][turb]") {
  run_adapt("adapt_kessler_tke", false);
}

TEST_CASE("adapt_dt: adapt_kessler_tke from Model::init", "[grid][adapt][step][init][kessler][turb]") {
  run_adapt("adapt_kessler_tke", true);
}

TEST_CASE("adapt_dt: unsupported combinations throw", "[grid][adapt]") {
  auto nl = Namelist::from_file(cm1k::testing::case_namelist("adapt_dry"));
  nl.set("dodomaindiag", ".true.");   // diagtim ramp-down: not ported
  CHECK_THROWS(Model(nl));
  auto nl2 = Namelist::from_file(cm1k::testing::case_namelist("adapt_kessler_tke"));
  nl2.set("radopt", "2");
  CHECK_THROWS(Model(nl2));
  auto nl3 = Namelist::from_file(cm1k::testing::case_namelist("adapt_kessler_tke"));
  nl3.set("adapt_dt", "2");
  Model m3(nl3);
  CHECK_THROWS([&] { m3.init(); m3.step(); }());
}
