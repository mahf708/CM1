#include "share/cm1k_namelist.hpp"
#include "share/cm1k_grid.hpp"

#include "cm1k_test_utils.hpp"

using namespace cm1k;

TEST_CASE("namelist parses CM1 syntax", "[share]") {
  const auto nl = Namelist::from_string(R"(
 &param0
 nx           =      60,
 terrain_flag = .false.,  ! comment
 /
 &param1
 dx     =  2000.0,  dtl = 7.5d0,
 run_name = 'a, b!',
 arr = 1, 2, 3,
 /
)");
  REQUIRE(nl.geti("nx", 0) == 60);
  REQUIRE(nl.getb("terrain_flag", true) == false);
  REQUIRE(nl.getr("dx", 0) == 2000.0);
  REQUIRE(nl.getr("DTL", 0) == 7.5);
  REQUIRE(nl.gets("run_name", "") == "a, b!");
  REQUIRE(nl.getv("arr").size() == 3);
  REQUIRE(nl.geti("missing", 42) == 42);
  REQUIRE(nl.group_of("dx") == "param1");
}

TEST_CASE("bounds for the dry bubble case", "[share]") {
  const auto nl = Namelist::from_file(testing::case_namelist("dry_bubble"));
  const auto b = Bounds::from_namelist(nl);
  REQUIRE(b.ngxy == 3);
  REQUIRE(b.ib == -2);
  REQUIRE(b.ie == b.nx + 3);
  REQUIRE(b.kb == 0);
  REQUIRE(b.ke == b.nz + 1);
}
