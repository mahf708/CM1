// Component tests: RRTMG radiation (radopt=2) against module_ra_rrtmg_lw.F /
// module_ra_rrtmg_sw.F, bit for bit.
#include "physics/rrtmg/cm1k_rrtmg_lw.hpp"
#include "physics/rrtmg/cm1k_rrtmg_lw_tables.hpp"
#include "physics/rrtmg/cm1k_rrtmg_sw_tables.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
void cm1f_rrtmg_swinit();
void cm1f_rrtmg_sw_scalars(double* v);
double cm1f_rrtmg_sw_scalar(const char* name);
int cm1f_rrtmg_sw_table(const char* name, double* out, int n);
void cm1f_rrtmg_lwinit(double p_top, int kme, int* nlayers);
void cm1f_rrtmg_lw_scalars(double* v);
int cm1f_rrtmg_lw_table(const char* name, double* out, int n);
void cm1f_rrtmg_lwrad(int ni, int nj, int nk, int ngxy, int cldovrlp, int yr, double julian, double r, double g,
                      const int* fq, double* p8w, double* p3d, double* pi3d, double* dz8w, double* t3d, double* t8w,
                      double* rho3d, double* cldfra, double* qv, double* qc, double* qr, double* qi, double* qs,
                      double* qg, double* o30, double* tsk, double* emiss, double* xland, double* xice,
                      double* xsnow, double* lwten, double* lwtenc, double* glw, double* olr, double* lwcf,
                      double* lwupt, double* lwuptc, double* lwdnt, double* lwdntc, double* lwupb, double* lwupbc,
                      double* lwdnb, double* lwdnbc, double* co2out);
}

using namespace cm1k;

namespace {

// The Fortran opens RRTMG_LW_DATA / RRTMG_SW_DATA in the working directory.
struct InRunDir {
  std::filesystem::path old;
  InRunDir() : old(std::filesystem::current_path()) { std::filesystem::current_path(CM1K_SOURCE_RUN_DIR); }
  ~InRunDir() { std::filesystem::current_path(old); }
};

// rrtmg_lwinit for (p_top, kme); returns the Fortran's nlayers.
int fortran_lwinit(double p_top, int kme) {
  static double last_p = -1;
  static int last_k = -1, last_n = 0;
  if (p_top == last_p && kme == last_k) return last_n;
  InRunDir rd;
  cm1f_rrtmg_lwinit(p_top, kme, &last_n);
  last_p = p_top;
  last_k = kme;
  return last_n;
}
void fortran_init_once() { fortran_lwinit(5000.0, 42); }

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
    // local DATA statements of RRTMG_LWRAD / O3DATA: covered by the column tests
    if (name.rfind("rad.", 0) == 0 && name != "rad.retab") continue;
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

TEST_CASE("rrtmg sw tables match rrtmg_swinit", "[fortran_bridge][rrtmg]") {
  {
    InRunDir rd;
    cm1f_rrtmg_swinit();
  }
  auto t = rrtmg::sw_tables(1005.7, 9.81, rrtmg::find_data_file("RRTMG_SW_DATA", CM1K_SOURCE_RUN_DIR));
  for (auto& [name, a] : t->real_arrays) {
    if (name.rfind("rad.", 0) == 0 && name != "rad.retab") continue;
    std::vector<double> f(a.second);
    REQUIRE(cm1f_rrtmg_sw_table(name.c_str(), f.data(), int(f.size())) == int(f.size()));
    require_same(name, a.first, f.data(), f.size());
  }
  for (auto& [name, a] : t->int_arrays) {
    std::vector<double> f(a.second), c(a.second);
    REQUIRE(cm1f_rrtmg_sw_table(name.c_str(), f.data(), int(f.size())) == int(f.size()));
    for (size_t i = 0; i < c.size(); ++i) c[i] = a.first[i];
    require_same(name, c.data(), f.data(), f.size());
  }
  for (auto& [name, v] : t->scalars) {
    const double f = cm1f_rrtmg_sw_scalar(name.c_str());
    require_same(name, &v, &f, 1);
  }
  double v[16];
  cm1f_rrtmg_sw_scalars(v);
  const auto& h = t->host;
  const double c[16] = {h.heatfac, h.oneminus, h.pi,     h.grav,   h.planck, h.boltz, h.clight,  h.avogad,
                        h.alosmt,  h.gascon,   h.radcn1, h.radcn2, h.sbcnst, h.secdy, h.bpade, h.tau_tbl};
  require_same("sw scalars", c, v, 16);
}

namespace {

using H3 = FArray<3, HostSpace>;

// A small domain of radiation columns (ni x nj, nk levels, ngxy = 1) on
// CM1's grid layout: 3-D (ib:ie,jb:je,0:nk+1), 2-D (ib:ie,jb:je,1:1).
struct RadDomain {
  int ni = 4, nj = 3, nk = 30, ngxy = 1;
  Real dz = 500.0, r = 287.04, g = 9.81, cp = 1005.7;
  int yr = 2009, cldovrlp = 2;
  Real julian = 172.4;
  int fq[5] = {1, 1, 0, 0, 0};
  std::map<std::string, H3> f;   // every array by its radiation_driver.F name
  const char* names3[17] = {"p8w", "p3d", "pi3d", "dz8w", "t3d", "t8w", "rho3d", "cldfra", "qv", "qc",
                            "qr",  "qi",  "qs",   "qg",   "o30", "lwten", "lwtenc"};
  const char* names2[16] = {"tsk",  "emiss", "xland", "xice",   "xsnow", "glw",   "olr",   "lwcf",
                            "lwupt", "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc", "lwdnb", "lwdnbc"};
  H3& operator[](const std::string& n) { return f.at(n); }

  void allocate() {
    const int ib = 1 - ngxy, ie = ni + ngxy, jb = 1 - ngxy, je = nj + ngxy;
    for (auto n : names3) f.emplace(n, H3(n, {ib, jb, 0}, {ie, je, nk + 1}));
    for (auto n : names2) f.emplace(n, H3(n, {ib, jb, 1}, {ie, je, 1}));
    for (auto& [n, a] : f) Kokkos::deep_copy(a.view(), 0.0);
  }

  // Column regimes: 0 clear moist, 1 warm overcast cloud + rain, 2 fractional
  // clouds at several levels, 3 cold column (clouds below 273.15 K become
  // ice and snow), 4 dry, high surface, 5 deep cloud with tiny fractions.
  void fill(unsigned seed) {
    allocate();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    for (int j = 1 - ngxy; j <= nj + ngxy; ++j)
      for (int i = 1 - ngxy; i <= ni + ngxy; ++i) {
        const int reg = ((i + ngxy) + 3 * (j + ngxy)) % 6;
        const Real T0 = reg == 3 ? 262.0 + 4 * u(gen) : 288.0 + 14 * u(gen);
        const Real q0 = reg == 4 ? 0.002 : (reg == 3 ? 0.0015 : 0.012 + 0.006 * u(gen));
        const Real ps = reg == 4 ? 85000.0 : 100000.0 + 1500 * u(gen);
        std::vector<Real> T(nk + 2), p(nk + 2);
        for (int k = 1; k <= nk; ++k) {
          const Real z = (k - 0.5) * dz;
          T[k] = std::max(T0 - 0.0065 * z, reg == 3 ? 210.0 : 215.0) + 0.3 * (u(gen) - 0.5);
        }
        p[1] = ps * std::exp(-g * 0.5 * dz / (r * T[1]));
        for (int k = 2; k <= nk; ++k) p[k] = p[k - 1] * std::exp(-g * dz / (r * 0.5 * (T[k] + T[k - 1])));
        for (int k = 1; k <= nk; ++k) {
          const Real z = (k - 0.5) * dz;
          f.at("t3d")(i, j, k) = T[k];
          f.at("p3d")(i, j, k) = p[k];
          f.at("pi3d")(i, j, k) = std::pow(p[k] / 1.0e5, r / cp);
          f.at("rho3d")(i, j, k) = p[k] / (r * T[k]);
          f.at("dz8w")(i, j, k) = dz;
          f.at("qv")(i, j, k) = q0 * std::exp(-z / 2500.0) * (0.8 + 0.4 * u(gen));
          f.at("o30")(i, j, k) = 1.0e-12;
          Real qc = 0, qr = 0, cf = 0;
          if (reg == 1 && k >= 3 && k <= 8) { qc = 4e-4 * u(gen) + 1e-4; qr = k <= 5 ? 2e-4 * u(gen) : 0; cf = 1.0; }
          if (reg == 2 && (k == 4 || k == 9 || k == 10 || k == 15)) { qc = 3e-4 * u(gen); cf = 0.2 + 0.7 * u(gen); }
          if (reg == 3 && k >= 2 && k <= 12) { qc = 2e-4 * u(gen); qr = 1e-4 * u(gen); cf = u(gen); }
          if (reg == 5 && k >= 2 && k <= 20) { qc = 5e-4 * u(gen); qr = 1e-4 * u(gen); cf = k % 4 == 0 ? 1e-25 : u(gen); }
          if (reg == 0 && k == 6) { qc = -1e-12; }   // negative: max(0., qc)
          f.at("qc")(i, j, k) = qc;
          f.at("qr")(i, j, k) = qr;
          f.at("cldfra")(i, j, k) = cf;
        }
        const Real tsk = T0 + 2 * (u(gen) - 0.5);
        for (int k = 2; k <= nk; ++k) {
          f.at("p8w")(i, j, k) = 0.5 * (p[k - 1] + p[k]);
          f.at("t8w")(i, j, k) = 0.5 * (T[k - 1] + T[k]);
        }
        f.at("p8w")(i, j, 1) = 1.5 * p[1] - 0.5 * p[2];
        f.at("t8w")(i, j, 1) = tsk;
        f.at("p8w")(i, j, nk + 1) = 1.5 * p[nk] - 0.5 * p[nk - 1];
        f.at("t8w")(i, j, nk + 1) = 1.5 * T[nk] - 0.5 * T[nk - 1];
        f.at("tsk")(i, j, 1) = tsk;
        f.at("emiss")(i, j, 1) = reg % 2 ? 0.98 : 0.95;
        f.at("xland")(i, j, 1) = reg == 2 ? 2.0 : 1.0;
        f.at("xice")(i, j, 1) = reg == 3 ? 0.3 : 0.0;
        f.at("xsnow")(i, j, 1) = reg == 3 ? 40.0 : 0.0;
      }
  }
  // rrtmg_lwinit's p_top: the buffer layers above the model top are 4 hPa
  // thick down to p = 0, so use the lowest model-top pressure.
  Real p_top() {
    Real p = 1e30;
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) p = std::min(p, f.at("p8w")(i, j, nk + 1));
    return p;
  }

  RadDomain clone() const {
    RadDomain d = *this;
    d.f.clear();
    for (auto& [n, a] : f) {
      H3 b(n, {a.lb(0), a.lb(1), a.lb(2)}, {a.ub(0), a.ub(1), a.ub(2)});
      Kokkos::deep_copy(b.view(), a.view());
      d.f.emplace(n, b);
    }
    return d;
  }

  double run_fortran() {
    double co2 = 0;
    auto P = [&](const char* n) { return f.at(n).data(); };
    cm1f_rrtmg_lwrad(ni, nj, nk, ngxy, cldovrlp, yr, julian, r, g, fq, P("p8w"), P("p3d"), P("pi3d"), P("dz8w"),
                     P("t3d"), P("t8w"), P("rho3d"), P("cldfra"), P("qv"), P("qc"), P("qr"), P("qi"), P("qs"),
                     P("qg"), P("o30"), P("tsk"), P("emiss"), P("xland"), P("xice"), P("xsnow"), P("lwten"),
                     P("lwtenc"), P("glw"), P("olr"), P("lwcf"), P("lwupt"), P("lwuptc"), P("lwdnt"),
                     P("lwdntc"), P("lwupb"), P("lwupbc"), P("lwdnb"), P("lwdnbc"), &co2);
    return co2;
  }

  double run_cxx(int nlayers, int chunk) {
    std::map<std::string, FArray3> d;
    for (auto& [n, a] : f) {
      FArray3 b(n, {a.lb(0), a.lb(1), a.lb(2)}, {a.ub(0), a.ub(1), a.ub(2)});
      b.deep_copy_from(a);
      d.emplace(n, b);
    }
    rrtmg::LwRadArgs a;
    a.rthratenlw = d["lwten"]; a.lwtenc = d["lwtenc"];
    a.lwupt = d["lwupt"]; a.lwuptc = d["lwuptc"]; a.lwdnt = d["lwdnt"]; a.lwdntc = d["lwdntc"];
    a.lwupb = d["lwupb"]; a.lwupbc = d["lwupbc"]; a.lwdnb = d["lwdnb"]; a.lwdnbc = d["lwdnbc"];
    a.glw = d["glw"]; a.olr = d["olr"]; a.lwcf = d["lwcf"];
    a.emiss = d["emiss"]; a.tsk = d["tsk"]; a.xland = d["xland"]; a.xice = d["xice"]; a.snow = d["xsnow"];
    a.p8w = d["p8w"]; a.p3d = d["p3d"]; a.pi3d = d["pi3d"]; a.dz8w = d["dz8w"]; a.t3d = d["t3d"];
    a.t8w = d["t8w"]; a.rho3d = d["rho3d"]; a.cldfra3d = d["cldfra"];
    a.qv3d = d["qv"]; a.qc3d = d["qc"]; a.qr3d = d["qr"]; a.qi3d = d["qi"]; a.qs3d = d["qs"]; a.qg3d = d["qg"];
    a.o33d = d["o30"];
    a.r = r; a.g = g; a.cldovrlp = cldovrlp;
    a.f_qc = fq[0]; a.f_qr = fq[1]; a.f_qi = fq[2]; a.f_qs = fq[3]; a.f_qg = fq[4];
    a.yr = yr; a.julian = julian;
    a.its = 1; a.ite = ni; a.jts = 1; a.jte = nj; a.kts = 1; a.kte = nk;
    a.nlayers = nlayers;
    a.chunk = chunk;
    auto t = rrtmg::lw_tables(1005.7, 9.81, rrtmg::find_data_file("RRTMG_LW_DATA", CM1K_SOURCE_RUN_DIR));
    const double co2 = rrtmg::rrtmg_lwrad(*t, a);
    for (auto& [n, b] : d) Kokkos::deep_copy(f.at(n).view(), b.view());
    return co2;
  }
};

void require_same_arrays(RadDomain& c, RadDomain& f, const std::vector<std::string>& names) {
  for (auto& n : names) {
    auto& a = c[n];
    auto& b = f[n];
    require_same(n, a.data(), b.data(), a.size());
  }
}

} // namespace

TEST_CASE("rrtmg_lwrad matches module_ra_rrtmg_lw.F", "[fortran_bridge][rrtmg]") {
  for (int ovl : {2, 1, 3, 4}) {
    for (unsigned seed : {1u, 2u}) {
      RadDomain f;
      f.cldovrlp = ovl;
      f.fill(seed + 10 * ovl);
      RadDomain c = f.clone();
      const int nl_f = fortran_lwinit(f.p_top(), f.nk + 1);
      const int nl_c = rrtmg::lw_nlayers(f.nk + 1, f.p_top());
      REQUIRE(nl_f == nl_c);
      const double co2c = c.run_cxx(nl_c, seed == 1 ? 0 : 5);
      const double co2f = f.run_fortran();
      INFO("cldovrlp " << ovl << " seed " << seed);
      REQUIRE(co2f == co2c);
      require_same_arrays(c, f, {"lwten", "lwtenc", "glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc",
                                 "lwupb", "lwupbc", "lwdnb", "lwdnbc", "o30"});
      // the heating is not trivially zero
      REQUIRE(std::abs(f["lwten"](2, 2, 3)) > 0);
    }
  }
}
