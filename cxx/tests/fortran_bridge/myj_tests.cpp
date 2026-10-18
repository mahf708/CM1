// Component tests of the MYJ surface layer and PBL port (physics/myj)
// against the unmodified src/module_sf_myjsfc.F and src/module_bl_myjpbl.F,
// through cm1f_bridge_myj.F90 (MYJSFCINIT, MYJSFC and MYJPBL called as
// init_physics.F and turb.F call them).
//
// Synthetic columns (nk = 20 stretched levels): stable / neutral / convective
// x land / water x moist (with cloud water and ice in some) / dry, light to
// strong winds, terrain height 0 or not, random roughness, u*, TKE, PBL
// depth. One land column has zero roughness, as CM1 runs MYJ over land
// (zntmyj stays 0), which gives infinite fm (NaN are compared as equal). Several steps
// of MYJSFC then MYJPBL (itimestep = 0, 1, 2, ...) are run on both sides with
// the carried arrays passed from one call to the next as in turb.F, and
// every output is compared bit for bit after every call.
#include "physics/myj/cm1k_myj.hpp"
#include "share/cm1k_constants.hpp"

#include <catch2/catch.hpp>

#include <cmath>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void cm1f_myjsfcinit();
void cm1f_myjsfc(int ni, int nj, int nk, int itimestep, int iusemyjpbl, int isftcflx, const double* a3,
                 double* s2);
void cm1f_myjpbl(int ni, int nj, int nk, double dt, const double* a3, double* o3, double* s2);
}

using namespace cm1k;

namespace {

using Vec = std::vector<double>;

struct Grid3 {
  int ni, nj, nk;
  size_t n2() const { return size_t(ni) * nj; }
  size_t n3() const { return size_t(ni) * nj * (nk + 1); }
  // C++ layout (i,j,k), k = 1..nk+1
  size_t c(int i, int j, int k) const { return (i - 1) + size_t(ni) * ((j - 1) + size_t(nj) * (k - 1)); }
  // Fortran packed layout (i,k,j)
  size_t f(int i, int j, int k) const { return (i - 1) + size_t(ni) * ((k - 1) + size_t(nk + 1) * (j - 1)); }
};

FArray3 to_dev(const Vec& v, int ni, int nj, int nk3, const std::string& name) {
  FArray3 d(name, {1, 1, 1}, {ni, nj, nk3});
  Kokkos::View<Real***, Kokkos::LayoutLeft, HostSpace> h("h", ni, nj, nk3);
  for (size_t n = 0; n < v.size(); ++n) h.data()[n] = v[n];
  Kokkos::deep_copy(d.view(), h);
  return d;
}

Vec from_dev(const FArray3& d) {
  auto h = d.to_host();
  return Vec(h.data(), h.data() + h.size());
}

size_t ndiff(const Vec& a, const Vec& ref, double& maxd) {
  size_t nbad = 0;
  maxd = 0;
  for (size_t m = 0; m < a.size(); ++m) {
    const double x = a[m], y = ref[m];
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
  }
  return nbad;
}

// the 2D arrays, by name (MYJSFC slots 1..46 of the shim)
const std::vector<std::string> sfc_names = {
    "ht",   "tsk",  "qsfc", "thz0", "qz0",  "uz0",   "vz0",    "xland", "ust",    "znt",  "z0base", "pblh",
    "mavail", "rmol", "akhs", "akms", "rib", "chs",  "chs2",   "cqs2",  "hfx",    "qfx",  "lh",     "flhc",
    "flqc", "qgh",  "cpm",  "ct",   "u10",  "v10",   "t02",    "th02",  "tshltr", "th10", "q02",    "qshltr",
    "q10",  "pshltr", "u10e", "v10e", "psim", "psih", "fm",    "fhh",   "z0out",  "rhosfc"};
// MYJPBL's 2D slots 1..18 (in the sfc names where shared; chklowq, kpbl,
// mixht are the PBL's own, elflx = lh, znt = z0out as turb.F passes it)
const std::vector<std::string> pbl_names = {"ht",   "tsk",  "qsfc", "chklowq", "thz0", "qz0",
                                            "uz0",  "vz0",  "xland", "ust",    "z0out", "pblh",
                                            "kpbl", "ct",   "akhs", "akms",    "lh",   "mixht"};
const std::vector<std::string> in3_names = {"dz", "pmid", "pint", "th", "t", "exner", "qv", "qc", "qi", "u", "v",
                                            "rho"};
const std::vector<std::string> out3_names = {"tke", "exch_h", "exch_m", "el", "rub", "rvb", "rth", "rqv", "rqc",
                                             "rqi"};

struct Columns {
  Grid3 g;
  std::map<std::string, Vec> a2, a3;   // a3 in C++ layout (i,j,k)
};

Columns make_columns(int ni, int nj, int nk, unsigned seed) {
  Columns C;
  C.g = Grid3{ni, nj, nk};
  const auto& g = C.g;
  std::mt19937_64 gen(seed);
  auto U01 = [&]() { return std::uniform_real_distribution<double>(0.0, 1.0)(gen); };
  for (auto& n : sfc_names) C.a2[n] = Vec(g.n2(), 0.0);
  for (auto& n : {"chklowq", "kpbl", "mixht"}) C.a2[n] = Vec(g.n2(), 0.0);
  for (auto& n : in3_names) C.a3[n] = Vec(g.n3(), 0.0);
  for (auto& n : out3_names) C.a3[n] = Vec(g.n3(), 0.0);
  const Constants kc = Constants::make(0);
  const double grav = kc.g, cp = kc.cp, rd = kc.rd, kap = rd / cp;
  const double speeds[4] = {0.3, 4.0, 11.0, 27.0};
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      const size_t m = (i - 1) + size_t(ni) * (j - 1);
      auto& A = C.a2;
      const int regime = (i - 1) % 3;   // 0 stable, 1 neutral, 2 convective
      const bool water = ((i - 1) / 3) % 2 == 1;
      const bool moist = ((i - 1) / 6) % 2 == 0;
      const double S0 = speeds[(j - 1) % 4] * (0.7 + 0.6 * U01());
      const double h = regime == 0 ? 100 + 250 * U01() : (regime == 1 ? 400 + 500 * U01() : 600 + 900 * U01());
      const double ths = 296.0 + 6.0 * U01(), zfac = 0.7 + 0.6 * U01(), dir0 = 6.283 * U01();
      const double ht = (j == 2) ? 150.0 * U01() : 0.0;
      const double psfc = 101000.0 - 2500.0 * U01() - 11.0 * ht;
      // levels
      std::vector<double> zw(nk + 2), th(nk + 1);
      zw[1] = ht;
      for (int k = 1; k <= nk; ++k) {
        const double dz = 40.0 * zfac * std::pow(1.12, k - 1);
        C.a3["dz"][g.c(i, j, k)] = dz;
        zw[k + 1] = zw[k] + dz;
      }
      for (int k = 1; k <= nk; ++k) {
        const double z = 0.5 * (zw[k] + zw[k + 1]) - ht;
        double t;
        if (regime == 0) t = ths + (2.0 + 4.0 * U01() * 0.2) * (1.0 - std::exp(-z / 150.0)) + 0.004 * z;
        else if (regime == 1) t = ths + 0.003 * std::fmax(0.0, z - h);
        else t = ths - 0.6 * std::exp(-z / 60.0) + 0.005 * std::fmax(0.0, z - h);
        th[k] = t + 0.02 * (U01() - 0.5);
        C.a3["th"][g.c(i, j, k)] = th[k];
      }
      // hydrostatic Exner on w levels, pressures, T
      std::vector<double> piw(nk + 2);
      piw[1] = std::pow(psfc / 1.0e5, kap);
      for (int k = 1; k <= nk; ++k) piw[k + 1] = piw[k] - grav * (zw[k + 1] - zw[k]) / (cp * th[k]);
      for (int k = 1; k <= nk + 1; ++k) C.a3["pint"][g.c(i, j, k)] = 1.0e5 * std::pow(piw[k], 1.0 / kap);
      C.a3["pint"][g.c(i, j, 1)] = psfc;
      double qsfc0 = 0;
      for (int k = 1; k <= nk; ++k) {
        const size_t c = g.c(i, j, k);
        const double z = 0.5 * (zw[k] + zw[k + 1]) - ht;
        const double pim = 0.5 * (piw[k] + piw[k + 1]);
        const double p = 1.0e5 * std::pow(pim, 1.0 / kap);
        C.a3["pmid"][c] = p;
        C.a3["exner"][c] = pim;
        C.a3["t"][c] = th[k] * pim;
        const double qv = moist ? (0.006 + 0.01 * U01() * 0.3 + 0.004 * (1 - regime * 0.3)) * std::exp(-z / 2200.0) : 0.0;
        C.a3["qv"][c] = qv;
        if (k == 1) qsfc0 = qv;
        C.a3["qc"][c] = (moist && regime == 2 && j % 2 == 1 && z > h - 250 && z < h) ? 3.0e-4 * U01() : 0.0;
        C.a3["qi"][c] = (moist && i % 4 == 0 && z > 1200 && z < 2600) ? 2.0e-5 * U01() : 0.0;
        const double sp = S0 * (0.55 + 0.45 * std::fmin(1.0, z / h)) + 0.3 * (U01() - 0.5);
        const double dir = dir0 + 0.4 * z / h;
        C.a3["u"][c] = sp * std::cos(dir);
        C.a3["v"][c] = sp * std::sin(dir);
        C.a3["rho"][c] = p / (rd * th[k] * pim);
        const double tke0 = regime == 0 ? 0.05 * U01() : (regime == 1 ? 0.4 * U01() : 1.5 * U01());
        C.a3["tke"][c] = 0.01 + tke0 * std::fmax(0.0, 1.0 - z / h) + 0.002 * U01();
      }
      // 2D state
      const double pis = std::pow(psfc / 1.0e5, kap);
      const double dthsfc = regime == 0 ? -(0.5 + 4.0 * U01()) : (regime == 1 ? 0.2 * (U01() - 0.5) : 1.0 + 5.0 * U01());
      A["ht"][m] = ht;
      A["tsk"][m] = (th[1] + dthsfc) * pis;
      A["xland"][m] = water ? 2.0 : 1.0;
      A["mavail"][m] = water ? 1.0 : 0.15 + 0.7 * U01();
      A["qsfc"][m] = moist ? qsfc0 * (1.0 + 0.6 * U01()) : 0.002 * U01();
      A["thz0"][m] = th[1] + 0.5 * dthsfc;
      A["qz0"][m] = A["qsfc"][m] * 0.95;
      A["uz0"][m] = 0.05 * (U01() - 0.5);
      A["vz0"][m] = 0.05 * (U01() - 0.5);
      A["ust"][m] = 0.1;
      A["znt"][m] = water ? 1.0e-4 + 2.0e-3 * U01() : 0.02 + 0.5 * U01();
      if (i == 1 && j == nj) A["znt"][m] = 0.0;   // CM1 over land: zntmyj = 0
      A["z0base"][m] = water ? A["znt"][m] : 0.01 + 0.4 * U01();
      A["pblh"][m] = h;
      A["akhs"][m] = 0.005 + 0.04 * U01();
      A["akms"][m] = 0.005 + 0.04 * U01();
      A["rhosfc"][m] = psfc / (rd * A["tsk"][m]);
      A["chklowq"][m] = 0.0;
    }
  return C;
}

// the 2D C++ arrays, by name, sharing one array where turb.F passes one
struct Dev {
  std::map<std::string, FArray3> a2, a3;
};

Dev upload(const Columns& C) {
  Dev D;
  const auto& g = C.g;
  for (auto& [n, v] : C.a2) D.a2[n] = to_dev(v, g.ni, g.nj, 1, n);
  for (auto& [n, v] : C.a3) D.a3[n] = to_dev(v, g.ni, g.nj, g.nk + 1, n);
  return D;
}

void run_fortran_sfc(Columns& C, int itimestep, bool usemyjpbl, int isftcflx) {
  const auto& g = C.g;
  Vec a3(g.n3() * 10, 0.0), s2(g.n2() * sfc_names.size(), 0.0);
  const std::vector<std::string> in = {"dz", "pmid", "pint", "th", "t", "qv", "qc", "u", "v", "tke"};
  for (size_t n = 0; n < in.size(); ++n)
    for (int k = 1; k <= g.nk + 1; ++k)
      for (int j = 1; j <= g.nj; ++j)
        for (int i = 1; i <= g.ni; ++i) a3[n * g.n3() + g.f(i, j, k)] = C.a3.at(in[n])[g.c(i, j, k)];
  for (size_t n = 0; n < sfc_names.size(); ++n)
    for (size_t m = 0; m < g.n2(); ++m) s2[n * g.n2() + m] = C.a2.at(sfc_names[n])[m];
  cm1f_myjsfc(g.ni, g.nj, g.nk, itimestep, usemyjpbl ? 1 : 0, isftcflx, a3.data(), s2.data());
  for (size_t n = 0; n < sfc_names.size(); ++n)
    for (size_t m = 0; m < g.n2(); ++m) C.a2[sfc_names[n]][m] = s2[n * g.n2() + m];
}

void run_fortran_pbl(Columns& C, double dt) {
  const auto& g = C.g;
  Vec a3(g.n3() * in3_names.size(), 0.0), o3(g.n3() * out3_names.size(), 0.0),
      s2(g.n2() * pbl_names.size(), 0.0);
  for (size_t n = 0; n < in3_names.size(); ++n)
    for (int k = 1; k <= g.nk + 1; ++k)
      for (int j = 1; j <= g.nj; ++j)
        for (int i = 1; i <= g.ni; ++i) a3[n * g.n3() + g.f(i, j, k)] = C.a3.at(in3_names[n])[g.c(i, j, k)];
  for (size_t n = 0; n < out3_names.size(); ++n)
    for (int k = 1; k <= g.nk + 1; ++k)
      for (int j = 1; j <= g.nj; ++j)
        for (int i = 1; i <= g.ni; ++i) o3[n * g.n3() + g.f(i, j, k)] = C.a3.at(out3_names[n])[g.c(i, j, k)];
  for (size_t n = 0; n < pbl_names.size(); ++n)
    for (size_t m = 0; m < g.n2(); ++m) s2[n * g.n2() + m] = C.a2.at(pbl_names[n])[m];
  cm1f_myjpbl(g.ni, g.nj, g.nk, dt, a3.data(), o3.data(), s2.data());
  for (size_t n = 0; n < out3_names.size(); ++n)
    for (int k = 1; k <= g.nk + 1; ++k)
      for (int j = 1; j <= g.nj; ++j)
        for (int i = 1; i <= g.ni; ++i) C.a3[out3_names[n]][g.c(i, j, k)] = o3[n * g.n3() + g.f(i, j, k)];
  for (size_t n = 0; n < pbl_names.size(); ++n)
    for (size_t m = 0; m < g.n2(); ++m) C.a2[pbl_names[n]][m] = s2[n * g.n2() + m];
}

void run_cxx_sfc(const myj::Tables& tab, const myj::Workspace& ws, const Grid3& g, Dev& D, int itimestep,
                 bool usemyjpbl, int isftcflx) {
  myj::SfcInputs in;
  in.dz = D.a3["dz"]; in.pmid = D.a3["pmid"]; in.pint = D.a3["pint"]; in.th = D.a3["th"]; in.t = D.a3["t"];
  in.qv = D.a3["qv"]; in.qc = D.a3["qc"]; in.u = D.a3["u"]; in.v = D.a3["v"]; in.q2 = D.a3["tke"];
  in.ht = D.a2["ht"]; in.rhosfc = D.a2["rhosfc"];
  myj::SfcFields f;
  auto& A = D.a2;
  f.tsk = A["tsk"]; f.xland = A["xland"]; f.z0base = A["z0base"]; f.mavail = A["mavail"];
  f.qsfc = A["qsfc"]; f.thz0 = A["thz0"]; f.qz0 = A["qz0"]; f.uz0 = A["uz0"]; f.vz0 = A["vz0"]; f.ust = A["ust"];
  f.znt = A["znt"]; f.pblh = A["pblh"]; f.akhs = A["akhs"]; f.akms = A["akms"];
  f.rmol = A["rmol"]; f.rib = A["rib"]; f.chs = A["chs"]; f.chs2 = A["chs2"]; f.cqs2 = A["cqs2"]; f.hfx = A["hfx"];
  f.qfx = A["qfx"]; f.lh = A["lh"]; f.flhc = A["flhc"]; f.flqc = A["flqc"]; f.qgh = A["qgh"]; f.cpm = A["cpm"];
  f.ct = A["ct"]; f.u10 = A["u10"]; f.v10 = A["v10"]; f.t02 = A["t02"]; f.th02 = A["th02"]; f.tshltr = A["tshltr"];
  f.th10 = A["th10"]; f.q02 = A["q02"]; f.qshltr = A["qshltr"]; f.q10 = A["q10"]; f.pshltr = A["pshltr"];
  f.u10e = A["u10e"]; f.v10e = A["v10e"]; f.psim = A["psim"]; f.psih = A["psih"]; f.fm = A["fm"]; f.fhh = A["fhh"];
  f.z0out = A["z0out"];
  myj::myjsfc(tab, ws, g.ni, g.nj, g.nk, itimestep, usemyjpbl, isftcflx, in, f);
  Kokkos::fence();
}

void run_cxx_pbl(const myj::Workspace& ws, const Grid3& g, Dev& D, double dt) {
  myj::PblInputs in;
  in.dz = D.a3["dz"]; in.pmid = D.a3["pmid"]; in.pint = D.a3["pint"]; in.th = D.a3["th"]; in.t = D.a3["t"];
  in.exner = D.a3["exner"]; in.qv = D.a3["qv"]; in.qcw = D.a3["qc"]; in.qci = D.a3["qi"]; in.u = D.a3["u"];
  in.v = D.a3["v"]; in.ht = D.a2["ht"];
  myj::PblFields f;
  auto& A = D.a2;
  f.tsk = A["tsk"]; f.xland = A["xland"]; f.chklowq = A["chklowq"]; f.elflx = A["lh"];
  f.qsfc = A["qsfc"]; f.thz0 = A["thz0"]; f.qz0 = A["qz0"]; f.uz0 = A["uz0"]; f.vz0 = A["vz0"]; f.ust = A["ust"];
  f.ct = A["ct"]; f.akhs = A["akhs"]; f.akms = A["akms"]; f.tke_myj = D.a3["tke"];
  f.pblh = A["pblh"]; f.kpbl = A["kpbl"]; f.mixht = A["mixht"]; f.el_myj = D.a3["el"];
  f.exch_h = D.a3["exch_h"]; f.exch_m = D.a3["exch_m"];
  f.rublten = D.a3["rub"]; f.rvblten = D.a3["rvb"]; f.rthblten = D.a3["rth"]; f.rqvblten = D.a3["rqv"];
  f.rqcblten = D.a3["rqc"]; f.rqiblten = D.a3["rqi"];
  myj::myjpbl(ws, g.ni, g.nj, g.nk, dt, in, f);
  Kokkos::fence();
}

void compare(const Columns& C, Dev& D, const std::vector<std::string>& n2, const std::vector<std::string>& n3,
             const std::string& what) {
  for (auto& n : n2) {
    double maxd = 0;
    const size_t nb = ndiff(from_dev(D.a2[n]), C.a2.at(n), maxd);
    INFO(what << " " << n << ": " << nb << " differing points, max |diff| = " << maxd);
    CHECK(nb == 0);
  }
  for (auto& n : n3) {
    // k = 1..nk (k = nk+1 is not an MYJ level)
    const auto a = from_dev(D.a3[n]);
    const auto& r = C.a3.at(n);
    const size_t nk2 = C.g.n2() * C.g.nk;
    double maxd = 0;
    const size_t nb = ndiff(Vec(a.begin(), a.begin() + nk2), Vec(r.begin(), r.begin() + nk2), maxd);
    INFO(what << " " << n << ": " << nb << " differing points, max |diff| = " << maxd);
    CHECK(nb == 0);
  }
}

// After each step the 3D inputs drift a little (as the model state would),
// on both sides alike.
void drift(Columns& C, Dev& D, int step) {
  for (const char* n : {"th", "u", "v"}) {
    auto& v = C.a3[n];
    for (size_t m = 0; m < v.size(); ++m) v[m] += 0.01 * std::sin(0.37 * double(m) + step);
    D.a3[n] = to_dev(v, C.g.ni, C.g.nj, C.g.nk + 1, n);
  }
}

} // namespace

TEST_CASE("MYJ: MYJSFCINIT tables vs Fortran", "[myj][fortran]") {
  cm1f_myjsfcinit();
  const auto t = myj::Tables::make();
  REQUIRE(t.psim1.extent(0) == size_t(myj::Tables::kztm));
  // the tables are module variables of module_sf_myjsfc: checked through
  // MYJSFC's results below; here only that they are finite
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), t.psih2);
  for (size_t k = 0; k < h.extent(0); ++k) REQUIRE(std::isfinite(h(k)));
}

TEST_CASE("MYJ: MYJSFC + MYJPBL vs Fortran on synthetic columns, several steps", "[myj][fortran]") {
  cm1f_myjsfcinit();
  const auto tab = myj::Tables::make();
  const int ni = 12, nj = 4, nk = 20;
  myj::Workspace ws;
  ws.allocate(ni, nj, nk);
  struct Cfg {
    int isftcflx;
    bool usemyjpbl;
    unsigned seed;
  };
  for (const Cfg cfg : {Cfg{0, true, 11u}, Cfg{1, true, 12u}, Cfg{2, true, 13u}, Cfg{0, false, 14u}}) {
    std::ostringstream os;
    os << "isftcflx=" << cfg.isftcflx << " usemyjpbl=" << cfg.usemyjpbl;
    INFO(os.str());
    Columns C = make_columns(ni, nj, nk, cfg.seed);
    Dev D = upload(C);
    const double dt = 6.0;
    for (int step = 0; step <= 4; ++step) {
      run_fortran_sfc(C, step, cfg.usemyjpbl, cfg.isftcflx);
      run_cxx_sfc(tab, ws, C.g, D, step, cfg.usemyjpbl, cfg.isftcflx);
      compare(C, D, sfc_names, {}, os.str() + " MYJSFC step " + std::to_string(step));
      if (cfg.usemyjpbl) {
        // turb.F: CHKLOWQ = mavail once SLAB has run (nstep >= 1), else 0
        if (step >= 1) {
          C.a2["chklowq"] = C.a2["mavail"];
          D.a2["chklowq"] = to_dev(C.a2["chklowq"], ni, nj, 1, "chklowq");
        }
        run_fortran_pbl(C, dt);
        run_cxx_pbl(ws, C.g, D, dt);
        compare(C, D, pbl_names, out3_names, os.str() + " MYJPBL step " + std::to_string(step));
      }
      drift(C, D, step);
    }
    // the columns cover what they should
    size_t nnan = 0, nland = 0, nconv = 0;
    for (size_t m = 0; m < C.g.n2(); ++m) {
      if (!std::isfinite(C.a2["fm"][m])) ++nnan;
      if (C.a2["xland"][m] < 1.5) ++nland;
      if (C.a2["hfx"][m] > 0) ++nconv;
    }
    auto range = [&](const Vec& v) {
      double lo = 1e300, hi = -1e300;
      for (double x : v)
        if (!std::isnan(x)) { lo = std::min(lo, x); hi = std::max(hi, x); }
      return std::make_pair(lo, hi);
    };
    const auto rib = range(C.a2["rib"]), hfx = range(C.a2["hfx"]), ust = range(C.a2["ust"]);
    const auto pblh = range(C.a2["pblh"]), tke = range(C.a3["tke"]);
    std::printf("MYJ %s: rib %.3g..%.3g hfx %.3g..%.3g ust %.3g..%.3g pblh %.3g..%.3g tke %.3g..%.3g, "
                "%zu columns with non-finite fm\n",
                os.str().c_str(), rib.first, rib.second, hfx.first, hfx.second, ust.first, ust.second,
                pblh.first, pblh.second, tke.first, tke.second, nnan);
    CHECK(nland == C.g.n2() / 2);
    CHECK(nconv > 0);
    CHECK(rib.first < 0);
    CHECK(rib.second > 0);
    // the zero-roughness land column (CM1's zntmyj = 0 over land) only: fm =
    // log(z/z0) - psim is infinite there (CM1's wspd then is too)
    CHECK(nnan == 1);
  }
}
