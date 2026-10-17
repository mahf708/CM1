#ifndef CM1K_DYN_BRIDGE_UTILS_HPP
#define CM1K_DYN_BRIDGE_UTILS_HPP

// Helpers shared by the dynamical-core bridge tests (bc_tests.cpp,
// sound_tests.cpp): paired host/device arrays with Fortran bounds.

#include "share/cm1k_farray.hpp"

#include <catch2/catch.hpp>

#include <random>
#include <string>

extern "C" {
void cm1f_dyn_set_input(int nx, int ny, int nz, int ngxy, int wbc, int ebc, int sbc, int nbc,
                        double dx, double dy, double dz, double alph, double kdiv, int nsound);
}

namespace cm1k::testing {

// A host array (passed to the Fortran) and a device array (passed to the C++),
// both with the same Fortran bounds and initial values.
template <int R>
struct Pair {
  FArray<R, HostSpace> f;   // Fortran side
  FArray<R> d;              // C++ side
  Pair(const std::string& n, const std::array<int, R>& lb, const std::array<int, R>& ub) {
    d = FArray<R>(n, lb, ub);
    f = d.to_host();
  }
  double* fp() { return f.data(); }
  template <typename Fn> void init(Fn fn) {
    auto* p = f.data();
    for (size_t n = 0; n < f.size(); ++n) p[n] = fn(n);
    d.deep_copy_from(f);
  }
  void constant(double v) { init([v](size_t) { return v; }); }
  void random(std::mt19937_64& g, double lo, double hi) {
    std::uniform_real_distribution<double> u(lo, hi);
    init([&](size_t) { return u(g); });
  }
  // Number of elements that differ (bitwise) between C++ and Fortran, and max abs diff.
  void compare(const char* what) const {
    auto h = d.to_host();
    size_t nbad = 0;
    double maxd = 0;
    for (size_t n = 0; n < h.size(); ++n) {
      const double a = h.data()[n], b = f.data()[n];
      if (!(a == b) && !(a != a && b != b)) { ++nbad; maxd = std::max(maxd, std::abs(a - b)); }
    }
    INFO(what << ": " << nbad << " of " << h.size() << " elements differ, max abs diff " << maxd);
    CHECK(nbad == 0);
  }
};

} // namespace cm1k::testing

#endif
