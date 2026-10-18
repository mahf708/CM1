// Component tests of physics/sfc_gfdl (module_sf_gfdl.F SF_GFDL + MFLUX2,
// module_sf_exchcoef.F znot_wind10m with icoef_sf=6, the fpvs table of
// module_gfs_funcphys.F) against the original Fortran through
// cm1f_bridge_gfdl.F90.
//
// The inputs are the synthetic columns of wrfphys_test_columns.hpp
// (stable / neutral / convective, land / water, moist / dry, calm to
// strong winds), prepared by wrfphys::pbl_prep / pbl_moist as turb.F does,
// with some columns pushed to hurricane-force winds (the high-wind znot
// fits), one calm column (wind exactly 0) and one water column with a
// roughness length above the 0.98 m mask. Land columns get random soil
// moisture and the GFDL slab T* prediction runs (ntsflg=1, nonzero
// radiation). Several consecutive calls carry tsk, znt, mznt, u10, v10 (the
// first call starts from u10 = v10 = 0: the log-profile first guess of the
// 10-m wind). Every output array is compared bit for bit, halos included.
#include "physics/sfc_gfdl/cm1k_sfc_gfdl.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <sstream>

extern "C" {
void cm1f_sf_gfdl(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, double dt, double sfenth,
                  const double* rc, const int* ic, const double* a3d, double* s2d);
}

using namespace cm1k;
using namespace cm1k::wrftest;
namespace G = cm1k::sfc_gfdl;

namespace {

const Constants kC = Constants::make(0);
const wrfphys::WrfConst kW = wrfphys::WrfConst::make(kC);

FArray3 dev(const H3& h) {
  FArray3 d(h.label(), {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  d.deep_copy_from(h);
  return d;
}

struct GfdlCase {
  int ntsflg = 1, isfflx = 1, isltyp = 14, imoist = 1, nsteps = 4;
  double dt = 6.0, sfenth = 0.0;
  std::string str() const {
    std::ostringstream s;
    s << "ntsflg=" << ntsflg << " isfflx=" << isfflx << " isltyp=" << isltyp << " imoist=" << imoist
      << " dt=" << dt << " sfenth=" << sfenth;
    return s.str();
  }
};

constexpr int NS2 = 37;
const char* kS2n[NS2] = {"psfc", "xland", "gsw",  "glw",  "zkmax", "smois", "tsk",   "znt",  "mznt", "ust",
                         "u10",  "v10",   "chs",  "chs2", "cqs2",  "cpmm",  "psim",  "psih", "hfx",  "qfx",
                         "lh",   "taux",  "tauy", "flhc", "flqc",  "qgh",   "qsfc",  "gz1oz0", "swspd", "br",
                         "cd_out", "ch_out", "mol", "zol", "z0t", "fm", "fh"};

void check_gfdl(const GfdlCase& c, unsigned seed) {
  INFO(c.str());
  const Columns C = make_columns(12, 8, 12, seed);
  const auto& b = C.b;
  std::mt19937_64 gen(seed + 77);
  auto U01 = [&]() { return std::uniform_real_distribution<double>(0.0, 1.0)(gen); };
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
  const std::array<int, 3> lo2{b.ib, b.jb, 1}, hi2{b.ie, b.je, 1};

  // turb.F's preparation
  FArray3 dum1("dum1", lo, hi), dum2("dum2", lo, hi), dum3("dum3", lo, hi), dum4("dum4", lo, hi),
      dum5("dum5", lo, hi), dum6("dum6", lo, hi), dum7("dum7", lo, hi), divx("divx", lo, hi),
      ppten("ppten", lo, hi), dum8("dum8", lo, hi);
  wrfphys::PrepIn in;
  in.ni = b.ni; in.nj = b.nj; in.nk = b.nk; in.pertflx = 0;
  in.dz = C.dz; in.cgt1 = C.cgt1; in.cgt2 = C.cgt2; in.cgt3 = C.cgt3;
  in.ua = dev(C.ua); in.va = dev(C.va); in.u0 = dev(C.u0); in.v0 = dev(C.v0); in.th0 = dev(C.th0);
  in.tha = dev(C.tha); in.pi0 = dev(C.pi0); in.ppi = dev(C.ppi); in.prs = dev(C.prs); in.rmh = dev(C.rmh);
  in.c1 = dev(C.c1); in.c2 = dev(C.c2); in.psfc = dev(C.psfc);
  wrfphys::pbl_prep(in, dum1, dum2, dum3, dum4, dum5, dum6, dum7);
  FArray4 qa("qa", {b.ib, b.jb, b.kb, 1}, {b.ie, b.je, b.ke, 3});
  qa.deep_copy_from(C.qa);
  wrfphys::pbl_moist(b.ni, b.nj, b.nk, c.imoist, 1, 2, 3, qa, divx, ppten, dum8);

  // hurricane-force winds in the last row, a calm column, a rough water column
  auto h1 = dum1.to_host(), h2 = dum2.to_host();
  for (int i = 1; i <= b.ni; ++i) {
    const double fac = 2.0 + 0.5 * i;
    h1(i, b.nj, 1) *= fac;
    h2(i, b.nj, 1) *= fac;
  }
  h1(3, 2, 1) = 0.0;
  h2(3, 2, 1) = 0.0;
  dum1.deep_copy_from(h1);
  dum2.deep_copy_from(h2);
  auto znt0 = hcopy(C.znt);
  auto xland0 = hcopy(C.xland);
  xland0(5, 3, 1) = 2.0;
  znt0(5, 3, 1) = 1.5;

  // 2D state
  std::vector<FArray3> s2(NS2);
  for (int n = 0; n < NS2; ++n) {
    s2[n] = FArray3(kS2n[n], lo2, hi2);
    s2[n].fill(n >= 12 ? -7.25 : 0.0);
  }
  s2[0].deep_copy_from(C.psfc);
  s2[1].deep_copy_from(xland0);
  {
    auto gsw = h3("gsw", lo2, hi2), glw = h3("glw", lo2, hi2), zk = h3("zkmax", lo2, hi2),
         sm = h3("smois", lo2, hi2);
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        gsw(i, j, 1) = 600.0 * U01();
        glw(i, j, 1) = 300.0 + 100.0 * U01();
        zk(i, j, 1) = 0.5 * C.dz * C.rmh(i, j, 1);
        sm(i, j, 1) = 0.5 * U01();
      }
    s2[2].deep_copy_from(gsw);
    s2[3].deep_copy_from(glw);
    s2[4].deep_copy_from(zk);
    s2[5].deep_copy_from(sm);
  }
  s2[6].deep_copy_from(C.tsk);
  s2[7].deep_copy_from(znt0);
  s2[8].deep_copy_from(znt0);
  s2[9].deep_copy_from(C.ust);

  G::Inputs gi;
  gi.u3d = dum1; gi.v3d = dum2; gi.t3d = dum4; gi.qv3d = divx; gi.p3d = in.prs;
  gi.psfc = s2[0]; gi.xland = s2[1]; gi.gsw = s2[2]; gi.glw = s2[3]; gi.zkmax = s2[4]; gi.smois = s2[5];
  G::Fields gf;
  gf.tsk = s2[6]; gf.znt = s2[7]; gf.mznt = s2[8]; gf.ust = s2[9]; gf.u10 = s2[10]; gf.v10 = s2[11];
  gf.chs = s2[12]; gf.chs2 = s2[13]; gf.cqs2 = s2[14]; gf.cpm = s2[15]; gf.psim = s2[16]; gf.psih = s2[17];
  gf.hfx = s2[18]; gf.qfx = s2[19]; gf.lh = s2[20]; gf.taux = s2[21]; gf.tauy = s2[22]; gf.flhc = s2[23];
  gf.flqc = s2[24]; gf.qgh = s2[25]; gf.qsfc = s2[26]; gf.gz1oz0 = s2[27]; gf.wspd = s2[28]; gf.br = s2[29];
  gf.cd_out = s2[30]; gf.ch_out = s2[31]; gf.mol = s2[32]; gf.zol = s2[33]; gf.z0t = s2[34]; gf.fm = s2[35];
  gf.fh = s2[36];
  gf.tstrc = FArray3("tstrc", lo2, hi2);

  // the Fortran's packed copies
  H4 a3d(Kokkos::View<Real****, Kokkos::LayoutLeft, HostSpace>("a3d", hi[0] - lo[0] + 1, hi[1] - lo[1] + 1,
                                                               hi[2] - lo[2] + 1, 5),
         {lo[0], lo[1], lo[2], 1});
  {
    const FArray3 a3v[5] = {dum1, dum2, dum4, divx, in.prs};
    for (int n = 0; n < 5; ++n) {
      auto h = a3v[n].to_host();
      for (int k = lo[2]; k <= hi[2]; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
          for (int i = lo[0]; i <= hi[0]; ++i) a3d(i, j, k, n + 1) = h(i, j, k);
    }
  }
  H3 s2d = h3("s2d", {b.ib, b.jb, 1}, {b.ie, b.je, NS2});
  for (int n = 0; n < NS2; ++n) {
    auto h = s2[n].to_host();
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) s2d(i, j, n + 1) = h(i, j, 1);
  }

  G::Options o;
  o.ntsflg = c.ntsflg; o.isfflx = c.isfflx; o.isltyp = c.isltyp; o.sfenth = c.sfenth;
  const G::Tables T = G::Tables::make();
  const double rc[7] = {kW.cp, kW.rovcp, kW.rd, kW.xlv, kW.ep1, kW.ep2, kW.karman};
  const int ic[4] = {c.ntsflg, c.isfflx, 6, c.isltyp};
  for (int step = 1; step <= c.nsteps; ++step) {
    INFO("step " << step);
    G::sf_gfdl(o, kW, T, c.dt, b.ni, b.nj, gi, gf);
    cm1f_sf_gfdl(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.dt, c.sfenth, rc, ic, a3d.data(),
                 s2d.data());
    Kokkos::fence();
    for (int n = 0; n < NS2; ++n) {
      auto h = s2[n].to_host();
      size_t nbad = 0;
      double maxd = 0;
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          const double x = h(i, j, 1), y = s2d(i, j, n + 1);
          if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
        }
      INFO(kS2n[n] << ": " << nbad << " differing points, max |diff| = " << maxd);
      REQUIRE(nbad == 0);
    }
  }
  // coverage: stable and unstable points, land and water
  auto zol = s2[33].to_host();
  auto xl = s2[1].to_host();
  int nst = 0, nun = 0, nland = 0;
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i) {
      (zol(i, j, 1) >= 0 ? nst : nun)++;
      if (xl(i, j, 1) < 1.5) ++nland;
    }
  INFO("stable " << nst << " unstable " << nun << " land " << nland);
  CHECK(nst > 0);
  CHECK(nun > 0);
  CHECK(nland > 0);
}

} // namespace

TEST_CASE("gfdl_sfc", "[fortran_bridge][gfdl]") {
  unsigned seed = 700;
  for (double dt : {3.0, 30.0}) {
    GfdlCase c;
    c.dt = dt;
    check_gfdl(c, seed++);
  }
  GfdlCase dry;
  dry.imoist = 0;
  check_gfdl(dry, seed++);
  GfdlCase notsflg;
  notsflg.ntsflg = 0;
  notsflg.isfflx = 0;
  notsflg.isltyp = 3;
  check_gfdl(notsflg, seed++);
  GfdlCase spray;
  spray.sfenth = 0.5;
  spray.isltyp = 7;
  check_gfdl(spray, seed++);
}

TEST_CASE("gfdl_options", "[fortran_bridge][gfdl]") {
  G::Options o;
  REQUIRE_NOTHROW(o.check());
  o.icoef_sf = 2;
  REQUIRE_THROWS(o.check());
  o.icoef_sf = 6;
  o.ntsflg = 2;
  REQUIRE_THROWS(o.check());
}
