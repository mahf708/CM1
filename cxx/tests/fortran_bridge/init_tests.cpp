#include "init/cm1k_cm1libs.hpp"
#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

#include <vector>

extern "C" void cm1f_rslf_rsif(int ptype, int testcase, int n, const double* p, const double* t,
                               double* rsl, double* rsi);

using namespace cm1k;

TEST_CASE("rslf/rsif match cm1libs.F", "[bridge][init]") {
  // p from 1 Pa to 1100 hPa, T from 150 K to 350 K (includes the very cold
  // range where esl is capped at p/2)
  std::vector<double> p, t;
  for (int a = 0; a <= 200; ++a)
    for (int b = 0; b <= 200; ++b) {
      p.push_back(1.0 + a * (110000.0 - 1.0) / 200.0);
      t.push_back(150.0 + b * 1.0 + 0.123 * (a % 7));
    }
  const int n = int(p.size());
  for (int tc : {0, 4, 5}) {
    const auto c = Constants::make(tc);
    for (int ptype : {0, 1, 2, 3, 4, 5, 6, 27}) {
      std::vector<double> fl(n), fi(n);
      cm1f_rslf_rsif(ptype, tc, n, p.data(), t.data(), fl.data(), fi.data());
      int bad = 0;
      for (int i = 0; i < n; ++i) {
        const double cl = cm1libs::rslf(ptype, c.eps, p[i], t[i]);
        const double ci = cm1libs::rsif(ptype, c.eps, p[i], t[i]);
        if (cl != fl[i] || ci != fi[i]) {
          if (bad < 5) UNSCOPED_INFO("ptype " << ptype << " p " << p[i] << " t " << t[i] << " rslf " << cl
                                     << " vs " << fl[i] << " rsif " << ci << " vs " << fi[i]);
          ++bad;
        }
      }
      INFO("testcase " << tc << " ptype " << ptype);
      REQUIRE(bad == 0);   // bit-for-bit
    }
  }
}
