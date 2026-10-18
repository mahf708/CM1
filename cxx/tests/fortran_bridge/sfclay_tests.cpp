// Component tests of physics/sfclay (sfcmodel=2: sfclay.F, sfcmodel=3:
// sf_sfclayrev.F) and of the turb.F preparation/post-processing around them,
// against the original Fortran through cm1f_bridge_sfclay.F90.
//
// Inputs are the synthetic columns of wrfphys_test_columns.hpp (stable,
// neutral and convective; land and water; moist and dry; calm to 25 m/s;
// an exactly neutral column). Each scheme runs several consecutive calls
// (u*, z/L, qsfc, z0 and the fluxes carried from one call to the next, as
// across time steps) and every output array is compared bit for bit.
#include "physics/sfclay/cm1k_sfclay.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <sstream>

extern "C" {
void cm1f_sfclay_init(double* psimtb, double* psihtb);
void cm1f_sfclayrev_init();
void cm1f_sfclay(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int isftcflx, int iz0tlnd,
                 const double* rc, const double* dum1, const double* dum2, const double* dum4, const double* divx,
                 const double* prs, const double* dum5, const double* rf, double* s2d);
void cm1f_sfclayrev(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int isftcflx,
                    int iz0tlnd, const double* rc, const double* dum1, const double* dum2, const double* dum4,
                    const double* divx, const double* prs, const double* dum5, double* s2d);
void cm1f_sfclay_post(int ni, int nj, int ib, int ie, int jb, int je, int kb, int ke, int set_fluxes,
                      const double* divx, const double* rf, double* s2d);
void cm1f_wrf_prep(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int pertflx, double dz,
                   double cgt1, double cgt2, double cgt3, const double* ua, const double* va, const double* u0,
                   const double* v0, const double* th0, const double* tha, const double* pi0, const double* ppi,
                   const double* prs, const double* rmh, const double* c1, const double* c2, const double* psfc,
                   double* dum1, double* dum2, double* dum3, double* dum4, double* dum5, double* dum6, double* dum7,
                   int numq, int imoist, int nqv, int nqc, int nqi, const double* qa, double* divx, double* ppten,
                   double* dum8);
}

using namespace cm1k;
using namespace cm1k::wrftest;
namespace SL = cm1k::sfclay;

namespace {

const Constants kC = Constants::make(0);
const SL::WrfConst kW = SL::WrfConst::make(kC);

// rc(19) of cm1f_bridge_sfclay.F90
std::vector<double> rc19(const SL::WrfConst& w) {
  return {w.cp, w.g, w.rovcp, w.rd, w.xlv, w.lv1, w.lv2, w.svp1, w.svp2, w.svp3,
          w.svpt0, w.ep1, w.ep2, w.karman, w.eomeg, w.stbolt, w.p1000mb, w.rv, w.rovg};
}

// The packed order of cm1f_bridge_sfclay.F90's s2d(:,:,1:47).
std::vector<std::pair<std::string, FArray3*>> packed_fields(SL::SfcLayerState& s) {
  return {{"psfc", &s.psfc},   {"chs", &s.chs},       {"chs2", &s.chs2},     {"cqs2", &s.cqs2},
          {"cpmm", &s.cpmm},   {"znt", &s.znt},       {"ust", &s.ust},       {"hpbl", &s.hpbl},
          {"mavail", &s.mavail}, {"zol", &s.zol},     {"mol", &s.mol},       {"regime", &s.regime},
          {"psim", &s.psim},   {"psih", &s.psih},     {"fm", &s.fm},         {"fh", &s.fh},
          {"xland", &s.xland}, {"hfx", &s.hfx},       {"qfx", &s.qfx},       {"lh", &s.lh},
          {"tsk", &s.tsk},     {"flhc", &s.flhc},     {"flqc", &s.flqc},     {"qgh", &s.qgh},
          {"qsfc", &s.qsfc},   {"rmol", &s.rmol},     {"u10", &s.u10},       {"v10", &s.v10},
          {"th2", &s.th2},     {"t2", &s.t2},         {"q2", &s.q2},         {"gz1oz0", &s.gz1oz0},
          {"wspd", &s.wspd},   {"br", &s.br},         {"dsxy", &s.dsxy},     {"ustm", &s.ustm},
          {"ck", &s.ck},       {"cka", &s.cka},       {"cd", &s.cd},         {"cda", &s.cda},
          {"z0t", &s.z0t},     {"z0q", &s.z0q},       {"ch", &s.ch},         {"cq", &s.cq},
          {"s10", &s.s10},     {"thflux", &s.thflux}, {"qvflux", &s.qvflux}};
}
constexpr int NS = 47;

H3 pack(SL::SfcLayerState& s, const Bounds& b) {
  auto f = packed_fields(s);
  H3 p = h3("s2d", {b.ib, b.jb, 1}, {b.ie, b.je, NS});
  for (int n = 0; n < NS; ++n) {
    auto h = f[n].second->to_host();
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) p(i, j, n + 1) = h(i, j, 1);
  }
  return p;
}

// The surface state of the columns (both sides start from it).
void init_state(SL::SfcLayerState& s, const Columns& C) {
  s.allocate(C.b);
  auto set = [&](FArray3& d, const H3& h) { d.deep_copy_from(h); };
  set(s.psfc, C.psfc); set(s.tsk, C.tsk); set(s.xland, C.xland); set(s.znt, C.znt); set(s.mavail, C.mavail);
  set(s.ust, C.ust); set(s.mol, C.mol); set(s.qsfc, C.qsfc); set(s.hpbl, C.hpbl); set(s.dsxy, C.dsxy);
  set(s.hfx, C.hfx); set(s.qfx, C.qfx); set(s.zol, C.zol); set(s.regime, C.regime);
  s.ustm.deep_copy_from(C.ust);
  // arrays the schemes only write (or that sfclay_post sets): a marker value,
  // so that points a scheme leaves alone are compared too
  for (auto* p : {&s.chs, &s.chs2, &s.cqs2, &s.cpmm, &s.psim, &s.psih, &s.fm, &s.fh, &s.lh, &s.flhc, &s.flqc,
                  &s.qgh, &s.rmol, &s.u10, &s.v10, &s.th2, &s.t2, &s.q2, &s.gz1oz0, &s.wspd, &s.br, &s.ck,
                  &s.cka, &s.cd, &s.cda, &s.z0t, &s.z0q, &s.ch, &s.cq, &s.s10, &s.thflux, &s.qvflux})
    p->fill(-7.25);
}

// The prep arrays on the device (C++) and as host copies (Fortran inputs).
struct Prep {
  FArray3 dum1, dum2, dum3, dum4, dum5, dum6, dum7, divx, ppten, dum8;
  std::vector<std::pair<std::string, FArray3*>> all() {
    return {{"dum1", &dum1}, {"dum2", &dum2}, {"dum3", &dum3}, {"dum4", &dum4}, {"dum5", &dum5},
            {"dum6", &dum6}, {"dum7", &dum7}, {"divx", &divx}, {"ppten", &ppten}, {"dum8", &dum8}};
  }
};

// Run the C++ and Fortran prep on the columns; require equality; return the
// C++ result.
Prep run_prep(const Columns& C, int pertflx, int imoist) {
  const auto& b = C.b;
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
  Prep P;
  for (auto& [n, p] : P.all()) { *p = FArray3(n, lo, hi); p->fill(-3.5); }
  auto dev = [](const H3& h) {
    FArray3 d(h.label(), {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
    d.deep_copy_from(h);
    return d;
  };
  wrfphys::PrepIn in;
  in.ni = b.ni; in.nj = b.nj; in.nk = b.nk; in.pertflx = pertflx;
  in.dz = C.dz; in.cgt1 = C.cgt1; in.cgt2 = C.cgt2; in.cgt3 = C.cgt3;
  in.ua = dev(C.ua); in.va = dev(C.va); in.u0 = dev(C.u0); in.v0 = dev(C.v0); in.th0 = dev(C.th0);
  in.tha = dev(C.tha); in.pi0 = dev(C.pi0); in.ppi = dev(C.ppi); in.prs = dev(C.prs); in.rmh = dev(C.rmh);
  in.c1 = dev(C.c1); in.c2 = dev(C.c2); in.psfc = dev(C.psfc);
  wrfphys::pbl_prep(in, P.dum1, P.dum2, P.dum3, P.dum4, P.dum5, P.dum6, P.dum7);
  FArray4 qa("qa", {b.ib, b.jb, b.kb, 1}, {b.ie, b.je, b.ke, 3});
  qa.deep_copy_from(C.qa);
  wrfphys::pbl_moist(b.ni, b.nj, b.nk, imoist, 1, 2, 3, qa, P.divx, P.ppten, P.dum8);

  std::vector<H3> f;
  for (auto& [n, p] : P.all()) { H3 h = h3(n, lo, hi); for (size_t m = 0; m < h.size(); ++m) h.data()[m] = -3.5; f.push_back(h); }
  cm1f_wrf_prep(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, pertflx, C.dz, C.cgt1, C.cgt2, C.cgt3,
                C.ua.data(), C.va.data(), C.u0.data(), C.v0.data(), C.th0.data(), C.tha.data(), C.pi0.data(),
                C.ppi.data(), C.prs.data(), C.rmh.data(), C.c1.data(), C.c2.data(), C.psfc.data(), f[0].data(),
                f[1].data(), f[2].data(), f[3].data(), f[4].data(), f[5].data(), f[6].data(), 3, imoist, 1, 2, 3,
                C.qa.data(), f[7].data(), f[8].data(), f[9].data());
  auto all = P.all();
  for (size_t n = 0; n < all.size(); ++n) {
    double maxd;
    const size_t nbad = count_diff(all[n].second->to_host(), f[n], maxd, [](int, int) { return true; });
    INFO("prep " << all[n].first << " pertflx=" << pertflx << " imoist=" << imoist << ": " << nbad
                 << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }
  return P;
}

void compare_state(SL::SfcLayerState& s, const H3& fs2d, const Columns& C, bool z0_water_only,
                   const std::string& what) {
  const auto& b = C.b;
  H3 cs = pack(s, b);
  auto f = packed_fields(s);
  const H3 xl = C.xland;
  for (int n = 0; n < NS; ++n) {
    const std::string name = f[n].first;
    const bool water_only = z0_water_only && (name == "z0t" || name == "z0q");
    size_t nbad = 0;
    double maxd = 0;
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        if (water_only && !(xl(i, j, 1) > 1.5 && i >= 1 && i <= b.ni && j >= 1 && j <= b.nj)) continue;
        const double x = cs(i, j, n + 1), y = fs2d(i, j, n + 1);
        if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
      }
    INFO(what << " " << name << ": " << nbad << " differing points, max |diff| = " << maxd);
    REQUIRE(nbad == 0);
  }
}

struct SfcCase {
  int sfcmodel = 2, isftcflx = 0, iz0tlnd = 0, pertflx = 0, imoist = 1;
  std::string str() const {
    std::ostringstream s;
    s << "sfcmodel=" << sfcmodel << " isftcflx=" << isftcflx << " iz0tlnd=" << iz0tlnd << " pertflx=" << pertflx
      << " imoist=" << imoist;
    return s.str();
  }
};

void check_sfclay(const SfcCase& c, unsigned seed, int ncalls) {
  INFO(c.str());
  const Columns C = make_columns(12, 8, 40, seed);
  const auto& b = C.b;
  Prep P = run_prep(C, c.pertflx, c.imoist);
  auto h = [](const FArray3& d) { return d.to_host(); };
  const H3 dum1 = h(P.dum1), dum2 = h(P.dum2), dum4 = h(P.dum4), divx = h(P.divx), dum5 = h(P.dum5);
  FArray3 rf("rf", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  rf.deep_copy_from(C.rf);
  FArray3 prs("prs", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  prs.deep_copy_from(C.prs);

  SL::Options o;
  o.sfcmodel = c.sfcmodel; o.isftcflx = c.isftcflx; o.iz0tlnd = c.iz0tlnd;
  SL::Tables t = (c.sfcmodel == 2) ? SL::sfclayinit() : SL::sf_sfclayrev_init();
  std::vector<double> ptb(1001), phtb(1001);
  if (c.sfcmodel == 2) {
    cm1f_sfclay_init(ptb.data(), phtb.data());
    auto m = Kokkos::create_mirror_view_and_copy(HostSpace(), t.psimtb);
    auto hh = Kokkos::create_mirror_view_and_copy(HostSpace(), t.psihtb);
    for (int n = 0; n <= 1000; ++n) {
      INFO("psimtb/psihtb(" << n << ")");
      REQUIRE(m(n) == ptb[n]);
      REQUIRE(hh(n) == phtb[n]);
    }
  } else {
    cm1f_sfclayrev_init();
  }

  SL::SfcLayerState s;
  init_state(s, C);
  H3 fs = pack(s, b);
  const auto rc = rc19(kW);
  for (int call = 1; call <= ncalls; ++call) {
    INFO("call " << call);
    if (c.sfcmodel == 2) {
      SL::sfclay(o, kW, t, b.ni, b.nj, P.dum1, P.dum2, P.dum4, P.divx, prs, P.dum5, rf, s);
      cm1f_sfclay(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.isftcflx, c.iz0tlnd, rc.data(),
                  dum1.data(), dum2.data(), dum4.data(), divx.data(), C.prs.data(), dum5.data(), C.rf.data(),
                  fs.data());
    } else {
      SL::sfclayrev(o, kW, t, b.ni, b.nj, P.dum1, P.dum2, P.dum4, P.divx, prs, P.dum5, s);
      cm1f_sfclayrev(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.isftcflx, c.iz0tlnd, rc.data(),
                     dum1.data(), dum2.data(), dum4.data(), divx.data(), C.prs.data(), dum5.data(), fs.data());
    }
    Kokkos::fence();
    compare_state(s, fs, C, c.sfcmodel == 2, "sfc layer");
    // post-processing (alternating set_fluxes)
    const bool set_fluxes = (call % 2 == 1);
    SL::sfclay_post(b.ni, b.nj, set_fluxes, P.divx, rf, s);
    cm1f_sfclay_post(b.ni, b.nj, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, set_fluxes ? 1 : 0, divx.data(), C.rf.data(),
                     fs.data());
    Kokkos::fence();
    compare_state(s, fs, C, c.sfcmodel == 2, "sfclay_post");
  }

  // the stability regimes the calls went through (all four should be hit)
  auto reg = s.regime.to_host();
  int seen[5] = {0, 0, 0, 0, 0};
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i) {
      const int r = int(reg(i, j, 1));
      if (r >= 1 && r <= 4) seen[r]++;
    }
  INFO("regimes after the last call: 1:" << seen[1] << " 2:" << seen[2] << " 3:" << seen[3] << " 4:" << seen[4]);
  CHECK(seen[4] > 0);
  CHECK(seen[1] + seen[2] > 0);
}

} // namespace

TEST_CASE("wrfphys_prep", "[fortran_bridge][sfclay]") {
  const Columns C = make_columns(12, 8, 40, 101);
  for (int pertflx : {0, 1})
    for (int imoist : {0, 1}) run_prep(C, pertflx, imoist);
}

TEST_CASE("sfclay_mm5", "[fortran_bridge][sfclay]") {
  unsigned seed = 200;
  for (int isftcflx : {0, 1, 2})
    for (int iz0tlnd : {0, 1, 2}) {
      SfcCase c;
      c.sfcmodel = 2; c.isftcflx = isftcflx; c.iz0tlnd = iz0tlnd;
      c.pertflx = (isftcflx + iz0tlnd) % 2;
      check_sfclay(c, seed++, 4);
    }
  SfcCase dry;
  dry.sfcmodel = 2; dry.imoist = 0;
  check_sfclay(dry, 290, 3);
}

TEST_CASE("sfclay_revised", "[fortran_bridge][sfclay]") {
  unsigned seed = 300;
  for (int isftcflx : {0, 1, 2})
    for (int iz0tlnd : {0, 1, 2}) {
      SfcCase c;
      c.sfcmodel = 3; c.isftcflx = isftcflx; c.iz0tlnd = iz0tlnd;
      c.pertflx = (isftcflx + iz0tlnd) % 2;
      check_sfclay(c, seed++, 4);
    }
  SfcCase dry;
  dry.sfcmodel = 3; dry.imoist = 0;
  check_sfclay(dry, 390, 3);
}

TEST_CASE("sfclay_options", "[fortran_bridge][sfclay]") {
  SL::Options o;
  o.sfcmodel = 1;
  REQUIRE_THROWS(o.check());
  o.sfcmodel = 3;
  o.isftcflx = 3;
  REQUIRE_THROWS(o.check());
  o.isftcflx = 2;
  o.iz0tlnd = 2;
  REQUIRE_NOTHROW(o.check());
}
