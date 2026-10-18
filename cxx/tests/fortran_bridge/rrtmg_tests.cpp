// Component tests: RRTMG radiation (radopt=2) against module_ra_rrtmg_lw.F /
// module_ra_rrtmg_sw.F, bit for bit.
#include "physics/rrtmg/cm1k_rrtmg_lw_tables.hpp"

#include "cm1k_test_utils.hpp"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
void cm1f_rrtmg_lwinit(double p_top, int kme, int* nlayers);
void cm1f_rrtmg_lw_scalars(double* v);
int cm1f_rrtmg_lw_table(const char* name, double* out, int n);
}

using namespace cm1k;

namespace {

// The Fortran opens RRTMG_LW_DATA / RRTMG_SW_DATA in the working directory.
struct InRunDir {
  std::filesystem::path old;
  InRunDir() : old(std::filesystem::current_path()) { std::filesystem::current_path(CM1K_SOURCE_RUN_DIR); }
  ~InRunDir() { std::filesystem::current_path(old); }
};

void fortran_init_once() {
  static bool done = false;
  if (done) return;
  InRunDir rd;
  int nl = 0;
  cm1f_rrtmg_lwinit(5000.0, 42, &nl);
  done = true;
}

// Bitwise equality, reporting the first difference.
void require_same(const std::string& what, const double* a, const double* b, size_t n) {
  size_t bad = 0, first = n;
  for (size_t i = 0; i < n; ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0 && !(a[i] == 0 && b[i] == 0)) {
      if (first == n) first = i;
      ++bad;
    }
  INFO(what << ": " << bad << " of " << n << " differ"
            << (first < n ? ", first at " + std::to_string(first) + ": " + std::to_string(a[first]) + " vs " +
                                std::to_string(b[first])
                          : std::string()));
  REQUIRE(bad == 0);
}

} // namespace

TEST_CASE("rrtmg lw tables match rrtmg_lwinit", "[fortran_bridge][rrtmg]") {
  fortran_init_once();
  auto t = rrtmg::lw_tables(1005.7, 9.81, rrtmg::find_data_file("RRTMG_LW_DATA", CM1K_SOURCE_RUN_DIR));
  for (auto& [name, a] : t->real_arrays) {
    std::vector<double> f(a.second);
    REQUIRE(cm1f_rrtmg_lw_table(name.c_str(), f.data(), int(f.size())) == int(f.size()));
    require_same(name, a.first, f.data(), f.size());
  }
  for (auto& [name, a] : t->int_arrays) {
    std::vector<double> f(a.second), c(a.second);
    REQUIRE(cm1f_rrtmg_lw_table(name.c_str(), f.data(), int(f.size())) == int(f.size()));
    for (size_t i = 0; i < c.size(); ++i) c[i] = a.first[i];
    require_same(name, c.data(), f.data(), f.size());
  }
  double v[21];
  cm1f_rrtmg_lw_scalars(v);
  const auto& h = t->host;
  const double c[21] = {h.fluxfac, h.heatfac, h.oneminus, h.pi,     h.grav,   h.planck, h.boltz,
                        h.clight,  h.avogad,  h.alosmt,   h.gascon, h.radcn1, h.radcn2, h.sbcnst,
                        h.secdy,   h.abscld1, h.absliq0,  h.bpade,  h.wtdiff, h.rec_6,  double(h.nxmol)};
  require_same("lw scalars", c, v, 21);
}
