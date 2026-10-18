// Component tests of physics/pbl_ysu (bl_ysu.F through turb.F's ysu_j_loop
// and mmm_physics_wrapper's ysu_wrapper; turb.F's ysudiss) against the
// original Fortran through cm1f_bridge_ysu.F90.
//
// The inputs are what sfc_and_turb hands the scheme: the synthetic columns
// of wrfphys_test_columns.hpp, prepared by wrfphys::pbl_prep/pbl_moist, with
// the surface fields (hfx, qfx, br, wspd, fm, fh, u*, z0, u10, v10) from a
// surface-layer call (sfcmodel=2 or 3; those are verified against the
// Fortran in sfclay_tests.cpp). Every output array is compared bit for bit.
#include "physics/pbl_ysu/cm1k_pbl_ysu.hpp"
#include "physics/sfclay/cm1k_sfclay.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <sstream>

extern "C" {
void cm1f_ysu(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int itopdown, double dt,
              const double* rc, const double* a3d, double* s2d, double* o3d);
void cm1f_ysu_dissten(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, double dz,
                      double rdz, const double* rmh, const double* mf, const double* upten, const double* vpten,
                      const double* dum1, const double* dum2, double* thten, double* thten1, double* dissten);
}

using namespace cm1k;
using namespace cm1k::wrftest;
namespace SL = cm1k::sfclay;
namespace Y = cm1k::pbl_ysu;

namespace {

const Constants kC = Constants::make(0);
const wrfphys::WrfConst kW = wrfphys::WrfConst::make(kC);

FArray3 dev(const H3& h) {
  FArray3 d(h.label(), {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  d.deep_copy_from(h);
  return d;
}

struct YsuCase {
  int sfcmodel = 2;
  bool topdown = false;
  double dt = 6.0;
  int imoist = 1;
  std::string str() const {
    std::ostringstream s;
    s << "sfcmodel=" << sfcmodel << " topdown=" << topdown << " dt=" << dt << " imoist=" << imoist;
    return s.str();
  }
};

void check_ysu(const YsuCase& c, unsigned seed) {
  INFO(c.str());
  const Columns C = make_columns(12, 8, 40, seed);
  const auto& b = C.b;
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
  const FArray3 rthraten = dev(C.rthraten), rf = dev(C.rf);

  // a surface-layer call for consistent surface fields
  SL::SfcLayerState s;
  s.allocate(b);
  s.psfc.deep_copy_from(C.psfc); s.tsk.deep_copy_from(C.tsk); s.xland.deep_copy_from(C.xland);
  s.znt.deep_copy_from(C.znt); s.mavail.deep_copy_from(C.mavail); s.ust.deep_copy_from(C.ust);
  s.mol.deep_copy_from(C.mol); s.qsfc.deep_copy_from(C.qsfc); s.hpbl.deep_copy_from(C.hpbl);
  s.dsxy.deep_copy_from(C.dsxy); s.hfx.deep_copy_from(C.hfx); s.qfx.deep_copy_from(C.qfx);
  s.zol.deep_copy_from(C.zol); s.ustm.deep_copy_from(C.ust);
  SL::Options so;
  so.sfcmodel = c.sfcmodel;
  if (c.sfcmodel == 2) SL::sfclay(so, kW, SL::sfclayinit(), b.ni, b.nj, dum1, dum2, dum4, divx, in.prs, dum5, rf, s);
  else SL::sfclayrev(so, kW, SL::sf_sfclayrev_init(), b.ni, b.nj, dum1, dum2, dum4, divx, in.prs, dum5, s);

  // YSU, C++
  Y::Inputs yi;
  yi.ux = dum1; yi.vx = dum2; yi.tx = dum4; yi.qvx = divx; yi.qcx = ppten; yi.qix = dum8; yi.p2d = in.prs;
  yi.p2di = dum6; yi.pi2d = dum7; yi.dz8w2d = dum5; yi.rthraten = rthraten;
  yi.psfcpa = s.psfc; yi.znt = s.znt; yi.ust = s.ust; yi.psim = s.fm; yi.psih = s.fh; yi.xland = s.xland;
  yi.hfx = s.hfx; yi.qfx = s.qfx; yi.wspd = s.wspd; yi.br = s.br;
  Y::Outputs yo;
  const char* o3n[8] = {"upten", "vpten", "thpten", "qvpten", "qcpten", "qipten", "xkzh", "xkzm"};
  FArray3* o3[8] = {&yo.utnp, &yo.vtnp, &yo.ttnp, &yo.qvtnp, &yo.qctnp, &yo.qitnp, &yo.exch_hx, &yo.exch_mx};
  for (int n = 0; n < 8; ++n) { *o3[n] = FArray3(o3n[n], lo, hi); o3[n]->fill(-7.25); }
  yo.hpbl = s.hpbl;
  yo.kpbl2d = FArray3("kpbl2d", lo2, hi2);
  yo.wstar = FArray3("wstar", lo2, hi2);
  yo.wstar.fill(-7.25);
  yo.delta = FArray3("delta", lo2, hi2);
  yo.delta.fill(-7.25);
  yo.u10 = s.u10;
  yo.v10 = s.v10;

  // the Fortran inputs, packed (cm1f_bridge_ysu.F90), taken before the call
  auto pack3 = [&](const std::vector<FArray3>& v, int nn) {
    H4 p(Kokkos::View<Real****, Kokkos::LayoutLeft, HostSpace>("p3", hi[0] - lo[0] + 1, hi[1] - lo[1] + 1,
                                                                hi[2] - lo[2] + 1, nn),
         {lo[0], lo[1], lo[2], 1});
    for (int n = 0; n < nn; ++n) {
      auto h = v[n].to_host();
      for (int k = lo[2]; k <= hi[2]; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
          for (int i = lo[0]; i <= hi[0]; ++i) p(i, j, k, n + 1) = h(i, j, k);
    }
    return p;
  };
  const std::vector<FArray3> a3v = {dum1, dum2, dum4, divx, ppten, dum8, in.prs, dum6, dum7, dum5, rthraten};
  const H4 a3d = pack3(a3v, 11);
  const std::vector<FArray3> s2v = {s.psfc, s.znt, s.ust, yo.hpbl, s.fm, s.fh, s.xland, s.hfx,
                                    s.qfx,  s.wspd, s.br, yo.kpbl2d, yo.wstar, yo.delta, yo.u10, yo.v10};
  H3 s2d = h3("s2d", {b.ib, b.jb, 1}, {b.ie, b.je, 16});
  for (int n = 0; n < 16; ++n) {
    auto h = s2v[n].to_host();
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) s2d(i, j, n + 1) = h(i, j, 1);
  }
  std::vector<FArray3> o3v;
  for (int n = 0; n < 8; ++n) o3v.push_back(*o3[n]);
  H4 o3d = pack3(o3v, 8);

  Y::Options yopt;
  yopt.ysu_topdown_pblmix = c.topdown;
  Y::Workspace ws;
  Y::ysu(yopt, kW, c.dt, b.ni, b.nj, b.nk, yi, yo, ws);
  const double rc[10] = {kW.cp, kW.g, kW.rovcp, kW.rd, kW.rovg, kW.ep1, kW.ep2, kW.karman, kW.xlv, kW.rv};
  cm1f_ysu(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.topdown ? 1 : 0, c.dt, rc, a3d.data(),
           s2d.data(), o3d.data());
  Kokkos::fence();

  for (int n = 0; n < 8; ++n) {
    auto h = o3[n]->to_host();
    size_t nbad = 0;
    double maxd = 0;
    for (int k = lo[2]; k <= hi[2]; ++k)
      for (int j = lo[1]; j <= hi[1]; ++j)
        for (int i = lo[0]; i <= hi[0]; ++i) {
          const double x = h(i, j, k), y = o3d(i, j, k, n + 1);
          if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
        }
    INFO(o3n[n] << ": " << nbad << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }
  const char* s2n[16] = {"psfc", "znt", "ust", "hpbl", "fm", "fh", "xland", "hfx",
                         "qfx", "wspd", "br", "kpbl2d", "wstar", "delta", "u10", "v10"};
  for (int n = 0; n < 16; ++n) {
    auto h = s2v[n].to_host();
    size_t nbad = 0;
    double maxd = 0;
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        const double x = h(i, j, 1), y = s2d(i, j, n + 1);
        if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
      }
    INFO(s2n[n] << ": " << nbad << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }

  // coverage: convective columns with a mixed layer, and stable ones
  auto kp = yo.kpbl2d.to_host();
  int nmix = 0, nsbl = 0;
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i) (kp(i, j, 1) > 1.5 ? nmix : nsbl)++;
  INFO("columns with kpbl>1: " << nmix << ", kpbl=1: " << nsbl);
  CHECK(nmix > 0);
}

} // namespace

TEST_CASE("ysu_pbl", "[fortran_bridge][ysu]") {
  unsigned seed = 500;
  for (int sfcmodel : {2, 3})
    for (bool topdown : {false, true})
      for (double dt : {2.0, 30.0}) {
        YsuCase c;
        c.sfcmodel = sfcmodel; c.topdown = topdown; c.dt = dt;
        check_ysu(c, seed++);
      }
  YsuCase dry;
  dry.imoist = 0;
  check_ysu(dry, 590);
}

TEST_CASE("ysu_dissten", "[fortran_bridge][ysu]") {
  const Columns C = make_columns(9, 7, 20, 600);
  const auto& b = C.b;
  std::mt19937_64 gen(601);
  auto U = [&]() { return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); };
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke}, hiw{b.ie, b.je, b.ke + 1};
  H3 up = h3("up", lo, hi), vp = h3("vp", lo, hi), d1 = h3("d1", lo, hi), d2 = h3("d2", lo, hi),
     t1 = h3("t1", lo, hi), t2 = h3("t2", lo, hi), ds = h3("ds", lo, hiw);
  for (auto* h : {&up, &vp}) for (size_t n = 0; n < h->size(); ++n) h->data()[n] = 1.0e-3 * U();
  for (auto* h : {&d1, &d2}) for (size_t n = 0; n < h->size(); ++n) h->data()[n] = 10.0 * U();
  for (auto* h : {&t1, &t2, &ds}) for (size_t n = 0; n < h->size(); ++n) h->data()[n] = U();
  const FArray3 rmh = dev(C.rmh), mf = dev(C.mf), upd = dev(up), vpd = dev(vp), d1d = dev(d1), d2d = dev(d2);
  FArray3 t1d = dev(t1), t2d = dev(t2), dsd = dev(ds);
  const double dz = C.dz, rdz = 1.0 / C.dz;
  Y::ysu_dissten(b.ni, b.nj, b.nk, dz, rdz, rmh, mf, upd, vpd, d1d, d2d, t1d, t2d, dsd);
  cm1f_ysu_dissten(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, dz, rdz, C.rmh.data(), C.mf.data(),
                   up.data(), vp.data(), d1.data(), d2.data(), t1.data(), t2.data(), ds.data());
  double maxd;
  auto all = [](int, int) { return true; };
  REQUIRE(count_diff(t1d.to_host(), t1, maxd, all) == 0);
  REQUIRE(count_diff(t2d.to_host(), t2, maxd, all) == 0);
  REQUIRE(count_diff(dsd.to_host(), ds, maxd, all) == 0);
}
