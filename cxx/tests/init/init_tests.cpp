// Native initialization (Model::init) vs the Fortran initial state
// (reference dump step_000000), for every array in the dump, on its full
// bounds (halos included).
#include "driver/cm1k_model.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cm1k;

namespace {

// Every array is expected to be bit-for-bit: same operation order as the
// Fortran, FMA contraction off in both builds, and gfortran and the C++ code
// call the same libm exp/pow/sin/cos/sqrt.
void check_case(const std::string& case_name) {
  const auto ref = testing::case_ref(case_name);
  if (ref.empty()) return;
  testing::use_case_inputs(case_name);
  Model m(Namelist::from_file(testing::case_namelist(case_name)));
  m.init();
  DumpReader r(DumpReader::step_dir(ref, 0));
  REQUIRE(r.nstep() == 0);
  int n = 0;
  for (auto& name : r.names()) {
    INFO(case_name << ": " << name);
    REQUIRE(m.has_field(name));
    const auto& e = r.entry(name);
    DiffStats s;
    if (e.lb.size() == 1) {
      auto a = m.field1(name).to_host();
      REQUIRE(a.lb(0) == e.lb[0]);
      REQUIRE(a.ub(0) == e.ub[0]);
      s = diff<1>(a, r.read_host<1>(name), {e.lb[0]}, {e.ub[0]});
    } else {
      REQUIRE(e.lb.size() == 3);
      auto a = m.field3(name).to_host();
      for (int d = 0; d < 3; ++d) {
        REQUIRE(a.lb(d) == e.lb[d]);
        REQUIRE(a.ub(d) == e.ub[d]);
      }
      s = diff<3>(a, r.read_host<3>(name), {e.lb[0], e.lb[1], e.lb[2]}, {e.ub[0], e.ub[1], e.ub[2]});
    }
    if (s.max_abs != 0) {
      std::printf("  %s %-8s max_abs %.3e rel %.3e at", case_name.c_str(), name.c_str(), s.max_abs, s.rel);
      for (int w : s.where) std::printf(" %d", w);
      std::printf("\n");
    }
    CHECK(s.max_abs == 0.0);
    ++n;
  }
  std::printf("  %s: %d arrays compared\n", case_name.c_str(), n);
  REQUIRE(n > 0);
}

} // namespace

TEST_CASE("init: dry_bubble", "[init][ref]") { check_case("dry_bubble"); }
TEST_CASE("init: dry_shear", "[init][ref]") { check_case("dry_shear"); }
TEST_CASE("init: kessler_bubble", "[init][ref]") { check_case("kessler_bubble"); }
TEST_CASE("init: kessler_tke", "[init][ref]") { check_case("kessler_tke"); }
TEST_CASE("init: kessler_saturated", "[init][ref]") { check_case("kessler_saturated"); }
TEST_CASE("init: kessler_periodic", "[init][ref]") { check_case("kessler_periodic"); }
TEST_CASE("init: kessler_periodic_long", "[init][ref]") { check_case("kessler_periodic_long"); }
// option coverage for the initialization only (1-step runs):
//   init_variants:     stretch_z=1, rigid E/W + open N/S walls, iwnd=1, iorigin=1, ibalance=1, moist
//   init_variants_dry: stretch_z=1, open E/W walls, iwnd=2, imove=1, ibalance=1, dry
TEST_CASE("init: init_variants", "[init][ref]") { check_case("init_variants"); }
TEST_CASE("init: init_variants_dry", "[init][ref]") { check_case("init_variants_dry"); }
// irandp=1 (gfortran random_number, +-0.25 K on theta) and the surface layer
// (init_surface.F with LANDUSE.TBL: land lu0=3 with set_znt=1; water lu0=16)
TEST_CASE("init: cbl_sfc", "[init][ref][sfc]") { check_case("cbl_sfc"); }
TEST_CASE("init: cbl_sfc_moist", "[init][ref][sfc]") { check_case("cbl_sfc_moist"); }

// terrain-following coordinate: itern=1 (2D bell, open E/W, isnd=9, iwnd=6)
// and itern=3 (3D hill, Kessler + TKE); every metric (zs, gz, rgzu, dzdx,
// gx, gyv, ...) and the terrain-dependent base state are compared
TEST_CASE("init: nh_mountain_waves", "[init][ref][terrain]") { check_case("nh_mountain_waves"); }
TEST_CASE("init: kessler_terrain_tke", "[init][ref][terrain]") { check_case("kessler_terrain_tke"); }
TEST_CASE("init: nh_mountain_waves_long", "[init][ref][terrain]") { check_case("nh_mountain_waves_long"); }
TEST_CASE("init: kessler_terrain_tke_openbc", "[init][ref][terrain]") { check_case("kessler_terrain_tke_openbc"); }

// base-state soundings and wind profiles (base.F isnd / iwnd), small grids,
// testcase = 0 (see cases/README.md):
TEST_CASE("init: snd_les_shcu", "[init][ref][snd]") { check_case("snd_les_shcu"); }
TEST_CASE("init: snd_les_shcu_precip", "[init][ref][snd]") { check_case("snd_les_shcu_precip"); }
TEST_CASE("init: snd_les_shcu_land", "[init][ref][snd]") { check_case("snd_les_shcu_land"); }
TEST_CASE("init: snd_les_sbl", "[init][ref][snd]") { check_case("snd_les_sbl"); }
TEST_CASE("init: snd_les_cbl", "[init][ref][snd]") { check_case("snd_les_cbl"); }
TEST_CASE("init: snd_les_shear", "[init][ref][snd]") { check_case("snd_les_shear"); }
TEST_CASE("init: snd_les_hbl", "[init][ref][snd]") { check_case("snd_les_hbl"); }
TEST_CASE("init: snd_input_sounding", "[init][ref][snd]") { check_case("snd_input_sounding"); }
TEST_CASE("init: snd_hurricane17", "[init][ref][snd]") { check_case("snd_hurricane17"); }
TEST_CASE("init: snd_tropical", "[init][ref][snd]") { check_case("snd_tropical"); }
TEST_CASE("init: snd_moist_neutral", "[init][ref][snd]") { check_case("snd_moist_neutral"); }

TEST_CASE("init: unsupported options throw", "[init]") {
  const auto nl0 = Namelist::from_file(testing::case_namelist("dry_bubble"));
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"isnd", "6"},      {"iwnd", "12"},          {"iinit", "4"},  {"stretch_z", "2"},
      {"stretch_x", "1"}, {"irandp", "2"}, {"ibalance", "2"},
      {"iorigin", "3"},   {"wbc", "5"}};
  for (const auto& kv : bad) {
    INFO(kv.first << " = " << kv.second);
    auto nl = nl0;
    nl.set(kv.first, kv.second);
    Model m(nl);
    REQUIRE_THROWS_AS(m.init(), std::runtime_error);
  }
  {   // no such test case: the Model constructor already throws
    auto nl = nl0;
    nl.set("testcase", "16");
    REQUIRE_THROWS_AS(Model(nl), std::runtime_error);
  }
  // terrain: itern=1,2,3 are ported; 0 and 4 (GrADS perts.dat) throw
  for (const char* it : {"0", "4"}) {
    INFO("terrain_flag with itern = " << it);
    auto nl = nl0;
    nl.set("terrain_flag", ".true.");
    nl.set("itern", it);
    Model m(nl);
    REQUIRE_THROWS_AS(m.init(), std::runtime_error);
  }
}
