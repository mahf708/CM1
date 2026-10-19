#include "share/cm1k_namelist.hpp"
#include "share/cm1k_grid.hpp"

#include "cm1k_test_utils.hpp"

#include <filesystem>
#include <fstream>

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

namespace {
std::string write_tmp(const std::string& name, const std::string& text) {
  const auto p = std::filesystem::temp_directory_path() / name;
  std::ofstream(p) << text;
  return p.string();
}
} // namespace

TEST_CASE("namelist files: names CM1 does not read throw", "[share]") {
  // every shipped namelist (cxx/cases and run/config_files) reads cleanly
  const std::filesystem::path cases = CM1K_DEFAULT_CASES_DIR;
  int n = 0;
  for (const auto& dir : {cases, cases / ".." / ".." / "run" / "config_files"})
    for (const auto& e : std::filesystem::directory_iterator(dir))
      if (std::filesystem::exists(e.path() / "namelist.input")) {
        INFO(e.path());
        REQUIRE_NOTHROW(Namelist::from_file((e.path() / "namelist.input").string()));
        ++n;
      }
  REQUIRE(n > 30);
  // a misspelled variable, a variable in the wrong group, an unknown group
  REQUIRE_THROWS_WITH(Namelist::from_file(write_tmp("cm1k_nl_a.input", "&param2\n isnf = 5,\n/\n")),
                      Catch::Matchers::Contains("isnf"));
  REQUIRE_THROWS_WITH(Namelist::from_file(write_tmp("cm1k_nl_b.input", "&param1\n isnd = 5,\n/\n")),
                      Catch::Matchers::Contains("belongs to &param2"));
  REQUIRE_THROWS_WITH(Namelist::from_file(write_tmp("cm1k_nl_c.input", "&param99\n nx = 5,\n/\n")),
                      Catch::Matchers::Contains("&param99"));
  REQUIRE_NOTHROW(Namelist::from_file(write_tmp("cm1k_nl_d.input", "&param0\n nx = 5,\n/\n&param2\n isnd = 5,\n/\n")));
  // from_string stays a plain parser
  REQUIRE_NOTHROW(Namelist::from_string("&param99\n anything = 1 /\n"));
}
