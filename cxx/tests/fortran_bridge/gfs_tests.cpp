// Component tests of physics/pbl_gfs (module_bl_gfsedmf.F's BL_GFSEDMF as
// turb.F calls it for ipbl=3) against the original Fortran through
// cm1f_bridge_gfs.F90.
//
// The inputs are what sfc_and_turb hands the scheme: the synthetic columns
// of wrfphys_test_columns.hpp (stable / neutral / convective, land / water,
// moist / dry, cloud-topped, calm .. 25 m/s), prepared by
// wrfphys::pbl_prep/pbl_moist, with the surface fields (hfx, qfx, br, wspd,
// psim, psih, gz1oz0, u*, z0, u10, v10) from a surface-layer call (sfcmodel=2
// or 3). Each case runs a few calls in sequence (u*, wspd, hpbl, kpbl2d and
// the outputs carried over, as in the model); every output array is compared
// bit for bit, halos included.
#include "physics/pbl_gfs/cm1k_pbl_gfs.hpp"
#include "physics/sfclay/cm1k_sfclay.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <sstream>

extern "C" {
void cm1f_gfsedmf(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, double dt,
                  const double* rc, const double* rp, const int* ic, const double* a3d, double* s2d, double* o3d);
}

using namespace cm1k;
using namespace cm1k::wrftest;
namespace SL = cm1k::sfclay;
namespace GF = cm1k::pbl_gfs;

namespace {

const Constants kC = Constants::make(0);
const wrfphys::WrfConst kW = wrfphys::WrfConst::make(kC);

FArray3 dev(const H3& h) {
  FArray3 d(h.label(), {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  d.deep_copy_from(h);
  return d;
}

struct GfsCase {
  int sfcmodel = 2;
  double dt = 6.0;
  int imoist = 1;
  bool qi = true, disheat = true;
  double alpha = -1.0;
  int nsteps = 3;
  std::string str() const {
    std::ostringstream s;
    s << "sfcmodel=" << sfcmodel << " dt=" << dt << " imoist=" << imoist << " qi=" << qi << " disheat=" << disheat
      << " alpha=" << alpha;
    return s.str();
  }
};

void check_gfs(const GfsCase& c, unsigned seed) {
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
  wrfphys::pbl_moist(b.ni, b.nj, b.nk, c.imoist, 1, 2, c.qi ? 3 : 0, qa, divx, ppten, dum8);
  const FArray3 rthraten = dev(C.rthraten), rf = dev(C.rf);

  // zf: heights of the w levels (k = 1 the terrain height)
  H3 zfh = h3("zf", lo, hi);
  {
    std::mt19937_64 gen(seed + 77);
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        const double zs = 50.0 * std::uniform_real_distribution<double>(0.0, 1.0)(gen);
        zfh(i, j, 1) = zs;
        for (int k = 1; k < b.ke; ++k) zfh(i, j, k + 1) = zfh(i, j, k) + C.dz * C.rmh(i, j, k);
        zfh(i, j, 0) = zs - C.dz * C.rmh(i, j, 0);
      }
  }
  const FArray3 zf = dev(zfh);

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

  GF::Inputs gi;
  gi.u3d = dum1; gi.v3d = dum2; gi.t3d = dum4; gi.qv3d = divx; gi.qc3d = ppten; gi.qi3d = dum8; gi.p3d = in.prs;
  gi.pi3d = dum7; gi.dz8w = dum5; gi.z = zf; gi.rthraten = rthraten;
  gi.psfc = s.psfc; gi.psim = s.psim; gi.psih = s.psih; gi.hfx = s.hfx; gi.qfx = s.qfx; gi.tsk = s.tsk;
  gi.gz1oz0 = s.gz1oz0; gi.br = s.br; gi.u10 = s.u10; gi.v10 = s.v10; gi.znt = s.znt; gi.xland = s.xland;
  GF::Outputs go;
  const char* o3n[10] = {"upten", "vpten", "thpten", "qvpten", "qcpten", "qipten", "xkzm", "xkzh", "dissten", "dheat"};
  FArray3* o3[10] = {&go.rublten, &go.rvblten, &go.rthblten, &go.rqvblten, &go.rqcblten,
                     &go.rqiblten, &go.dku3d,  &go.dkt3d,    &go.dissten,  &go.dheat};
  for (int n = 0; n < 10; ++n) { *o3[n] = FArray3(o3n[n], lo, hi); o3[n]->fill(-7.25); }
  go.ust = s.ust; go.wspd = s.wspd; go.hpbl = s.hpbl;
  go.kpbl2d = FArray3("kpbl2d", lo2, hi2);
  go.kpbl2d.fill(3.0);

  // the Fortran inputs, packed (cm1f_bridge_gfs.F90), taken before the calls
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
  const std::vector<FArray3> a3v = {dum1, dum2, dum3, dum4, divx, ppten, dum8, in.prs, dum7, dum5, zf, rthraten};
  const H4 a3d = pack3(a3v, 12);
  const std::vector<FArray3> s2v = {s.psfc, go.ust, go.hpbl, s.psim, s.psih, s.hfx, s.qfx, s.tsk,
                                    s.gz1oz0, go.wspd, s.br, go.kpbl2d, s.u10, s.v10, s.znt, s.xland};
  H3 s2d = h3("s2d", {b.ib, b.jb, 1}, {b.ie, b.je, 16});
  for (int n = 0; n < 16; ++n) {
    auto h = s2v[n].to_host();
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) s2d(i, j, n + 1) = h(i, j, 1);
  }
  std::vector<FArray3> o3v;
  for (int n = 0; n < 10; ++n) o3v.push_back(*o3[n]);
  H4 o3d = pack3(o3v, 10);

  GF::Options gopt;
  gopt.qi = c.qi;
  gopt.disheat = c.disheat;
  gopt.alpha = c.alpha;
  GF::Workspace ws;
  const double rc[7] = {kW.cp, kW.g, kW.rovcp, kW.rd, kW.rovg, kW.ep1, kW.karman};
  const double rp[4] = {gopt.var_ric, gopt.coef_ric_l, gopt.coef_ric_s, gopt.alpha};
  const int ic[3] = {c.qi ? 3 : 0, 1, c.disheat ? 1 : 0};
  for (int step = 0; step < c.nsteps; ++step) {
    GF::gfsedmf(gopt, kW, c.dt, b.ni, b.nj, b.nk, gi, go, ws);
    cm1f_gfsedmf(b.ni, b.nj, b.nk, b.ib, b.ie, b.jb, b.je, b.kb, b.ke, c.dt, rc, rp, ic, a3d.data(), s2d.data(),
                 o3d.data());
  }
  Kokkos::fence();

  for (int n = 0; n < 10; ++n) {
    // disheat = .false. leaves moninedmf's diss uninitialized (dissten undefined)
    if (n == 8 && !c.disheat) continue;
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
  const char* s2n[16] = {"psfc", "ust", "hpbl", "psim", "psih", "hfx", "qfx", "tsk",
                         "gz1oz0", "wspd", "br", "kpbl2d", "u10", "v10", "znt", "xland"};
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

  // coverage: deep (convective) PBLs, shallow ones, land and water
  auto kp = go.kpbl2d.to_host();
  int ndeep = 0, nshallow = 0;
  for (int j = 1; j <= b.nj; ++j)
    for (int i = 1; i <= b.ni; ++i) (kp(i, j, 1) > 3.5 ? ndeep : nshallow)++;
  INFO("columns with kpbl>3: " << ndeep << ", kpbl<=3: " << nshallow);
  CHECK(ndeep > 0);
  CHECK(nshallow > 0);
}

} // namespace

TEST_CASE("gfs_pbl", "[fortran_bridge][gfs]") {
  unsigned seed = 700;
  for (int sfcmodel : {2, 3})
    for (double dt : {2.0, 30.0}) {
      GfsCase c;
      c.sfcmodel = sfcmodel; c.dt = dt;
      check_gfs(c, seed++);
    }
  GfsCase noqi;
  noqi.qi = false;
  check_gfs(noqi, 720);
  GfsCase dry;
  dry.imoist = 0; dry.qi = false;
  check_gfs(dry, 721);
  GfsCase a1;
  a1.alpha = 1.0; a1.disheat = false;
  check_gfs(a1, 722);
}

TEST_CASE("gfs_pbl_shallow_grid_throws", "[fortran_bridge][gfs]") {
  GF::Options o;
  GF::Workspace ws;
  GF::Inputs in;
  GF::Outputs out;
  CHECK_THROWS(GF::gfsedmf(o, kW, 1.0, 2, 2, 3, in, out, ws));
}
