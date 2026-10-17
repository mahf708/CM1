#include "share/cm1k_farray.hpp"

#include "cm1k_test_utils.hpp"

using namespace cm1k;

TEST_CASE("FArray uses Fortran bounds and order", "[share]") {
  FArray3 a("a", {-2, -2, 0}, {5, 4, 3});
  REQUIRE(a.extent(0) == 8);
  REQUIRE(a.extent(1) == 7);
  REQUIRE(a.extent(2) == 4);
  Kokkos::parallel_for(box3(-2, 5, -2, 4, 0, 3), KOKKOS_LAMBDA(int i, int j, int k) {
    a(i, j, k) = 100 * i + 10 * j + k;
  });
  auto h = a.to_host();
  REQUIRE(h(-2, -2, 0) == -220);
  REQUIRE(h(5, 4, 3) == 543);
  // i fastest in memory, as in Fortran
  REQUIRE(&h(-1, -2, 0) - &h(-2, -2, 0) == 1);
  REQUIRE(h.data()[0] == -220);
}
