#include "share/cm1k_dump_io.hpp"
#include "driver/cm1k_model.hpp"

#include "cm1k_test_utils.hpp"

#include <filesystem>

using namespace cm1k;

TEST_CASE("dump round trip", "[share]") {
  const auto tmp = std::filesystem::temp_directory_path() / "cm1k_dump_rt";
  FArray3 a("a", {0, -1, 1}, {3, 2, 2});
  Kokkos::parallel_for(box3(0, 3, -1, 2, 1, 2), KOKKOS_LAMBDA(int i, int j, int k) { a(i, j, k) = i * 0.5 + j * 1e-3 + k; });
  {
    DumpWriter w(DumpReader::step_dir(tmp.string(), 7), 7, 42.0);
    w.write<3>("a", a);
  }
  DumpReader r(DumpReader::step_dir(tmp.string(), 7));
  REQUIRE(r.nstep() == 7);
  REQUIRE(r.mtime() == 42.0);
  auto h = r.read_host<3>("a");
  auto s = diff<3>(h, a.to_host(), {0, -1, 1}, {3, 2, 2});
  REQUIRE(s.max_abs == 0.0);
}

TEST_CASE("model loads the Fortran initial state", "[share][ref]") {
  const auto ref = testing::case_ref("dry_bubble");
  if (ref.empty()) return;
  Model m(Namelist::from_file(testing::case_namelist("dry_bubble")));
  m.init_from_dump(DumpReader::step_dir(ref, 0));
  REQUIRE(m.nstep() == 0);
  DumpReader r(DumpReader::step_dir(ref, 0));
  // every array in the dump must be known to the model, with the same bounds
  for (auto& name : r.names()) {
    INFO(name);
    REQUIRE(m.has_field(name));
  }
  auto tha = m.field3("tha").to_host();
  auto ref_tha = r.read_host<3>("tha");
  const auto& b = m.bounds();
  REQUIRE(diff<3>(tha, ref_tha, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}).max_abs == 0.0);
}
