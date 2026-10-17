#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

extern "C" void cm1f_set_constants(int testcase, double* out);

using namespace cm1k;

TEST_CASE("constants match constants.F", "[bridge]") {
  for (int tc : {0, 4, 5}) {
    double f[8];
    cm1f_set_constants(tc, f);
    const auto c = Constants::make(tc);
    const double cxx[8] = {c.g, c.rd, c.cp, c.cv, c.cpdrd, c.lv1, c.c_s, c.condc};
    for (int n = 0; n < 8; ++n) {
      INFO("testcase " << tc << " entry " << n);
      REQUIRE(cxx[n] == f[n]);   // bit-for-bit
    }
  }
}
