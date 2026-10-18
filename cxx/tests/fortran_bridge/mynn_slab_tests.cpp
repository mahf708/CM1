// Component test of the sfcmodel=6 surface step over land: the MYNN surface
// layer (physics/sfc_mynn) followed by SLAB through slab::Land (the call
// pbl_mynn::surface_layer makes), against src/module_sf_mynn.F + src/slab.F
// through cm1f_bridge_mynn_sfc.F90 and cm1f_bridge_slab.F90, in turb.F's
// order: SFCLAY_mynn, then SLAB (update_sfc) or the qsfc diagnosis (startup
// pass), then the ch/cq/s10/thflux/qvflux block.
//
// Land and water points, moist, several steps with tsk, tslb and the MYNN
// persistent arrays carried over (SLAB's tsk feeds the next SFCLAY_mynn);
// every array is compared bit for bit after every step. Also checks
// slab::Land::setup / init_soil (tmn, emiss, thc from LANDUSE.TBL; tslb).
#include "physics/sfc_mynn/cm1k_sfc_mynn.hpp"
#include "physics/slab/cm1k_slab_land.hpp"

#include "cm1k_test_utils.hpp"
#include "wrfphys_test_columns.hpp"

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

extern "C" {
void cm1f_mynn_sf_init();
void cm1f_mynn_sfclay(int ni, int nj, int nk, int itimestep, int isfflx, int isftcflx, int iz0tlnd, double dx,
                      const double* u3d, const double* v3d, const double* dz8w, const double* t3d,
                      const double* qv3d, const double* p3d, const double* rho3d, const double* psfc,
                      const double* hpbl, const double* mavail, const double* xland, const double* tsk,
                      const double* snowh, double* ust, double* mol, double* qsfc, double* znt, double* ustm,
                      double* hfx, double* qfx, double* zol, double* rmol, double* wspd, double* chs,
                      double* chs2, double* cqs2, double* cpmm, double* regime, double* psim, double* psih,
                      double* lh, double* flhc, double* flqc, double* qgh, double* u10, double* v10,
                      double* th2, double* t2, double* q2, double* gz1oz0, double* br, double* ch, double* ck,
                      double* cka, double* cd, double* cda);
void cm1f_mynn_sfc_finish(int ni, int nj, int dosfcflx, const double* qv1, const double* rf1, double* qsfc,
                          const double* qfx, const double* flqc, const double* chs2, const double* cqs2,
                          const double* ck, const double* u10, const double* v10, const double* hfx,
                          const double* cpmm, double* ch, double* cq, double* s10, double* thflux,
                          double* qvflux);
void cm1f_slab(int ni, int nj, int nk, int ib, int ie, int jb, int je, int kb, int ke, int nsl, int ifsnow, int irad,
               double deltsm, const double* rc, const double* zs, const double* dzs, const double* t3d,
               const double* qv3d, const double* p3d, double* s2d, double* tslb);
void cm1f_slab_soil(int nsl, int ib, int ie, int jb, int je, const double* tsk, const double* tmn, double* zs,
                    double* dzs, double* tslb);
}

using namespace cm1k;
using namespace cm1k::wrftest;
namespace M = cm1k::sfc_mynn;

namespace {

void require_same(const H3& a, const H3& ref, const std::string& what) {
  double maxd;
  const size_t nbad = count_diff(a, ref, maxd, [](int, int) { return true; });
  INFO(what << ": " << nbad << " differing points, max |diff| = " << maxd);
  REQUIRE(nbad == 0);
}

H3 hnew(const FArray3& a) {
  H3 h = h3(a.label() + "_f", {a.lb(0), a.lb(1), a.lb(2)}, {a.ub(0), a.ub(1), a.ub(2)});
  Kokkos::deep_copy(h.view(), 0.0);
  return h;
}

double qsat(double t, double p) {
  const double es = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
  return 0.622 * es / (p - es);
}

void run_land(int iz0tlnd, int lu0, unsigned seed) {
  INFO("iz0tlnd=" << iz0tlnd << " lu0=" << lu0);
  const int ni = 9, nj = 6, nk = 2, nsteps = 6;
  const double dt = 6.0, dx = 2000.0;
  Bounds b;
  b.ni = ni; b.nj = nj; b.nk = nk;
  b.ib = 1; b.ie = ni; b.jb = 1; b.je = nj; b.kb = 1; b.ke = nk;
  const auto C = Constants::make(0);
  const auto W = wrfphys::WrfConst::make(C);
  const std::vector<double> rc = {W.cp, W.g, W.rovcp, W.rd, W.xlv, W.lv1, W.lv2, W.svp1, W.svp2, W.svp3,
                                  W.svpt0, W.ep1, W.ep2, W.karman, W.eomeg, W.stbolt, W.p1000mb, W.rv, W.rovg};
  const auto tables = M::PsiTables::make();
  M::Options o;
  o.sfcmodel = 6; o.iz0tlnd = iz0tlnd; o.imoist = 1;
  REQUIRE_NOTHROW(o.check());

  // C++ side: MYNN fields, SLAB's state (Land::setup as the Model does it)
  M::Fields f;
  f.allocate(b);
  sfc::Options so;
  so.sfcmodel = 6; so.lu0 = lu0; so.season = 1; so.tmn0 = 291.5;
  slab::Land land;
  land.setup(so, b);

  std::map<std::string, FArray3> fm = {
      {"psfc", f.psfc}, {"hpbl", f.hpbl}, {"mavail", f.mavail}, {"xland", f.xland}, {"tsk", f.tsk},
      {"snowh", f.snowh}, {"ust", f.ust}, {"mol", f.mol}, {"qsfc", f.qsfc}, {"znt", f.znt},
      {"ustm", f.ustm}, {"hfx", f.hfx}, {"qfx", f.qfx}, {"zol", f.zol}, {"rmol", f.rmol},
      {"wspd", f.wspd}, {"chs", f.chs}, {"chs2", f.chs2}, {"cqs2", f.cqs2}, {"cpmm", f.cpmm},
      {"regime", f.regime}, {"psim", f.psim}, {"psih", f.psih}, {"lh", f.lh}, {"flhc", f.flhc},
      {"flqc", f.flqc}, {"qgh", f.qgh}, {"u10", f.u10}, {"v10", f.v10}, {"th2", f.th2}, {"t2", f.t2},
      {"q2", f.q2}, {"gz1oz0", f.gz1oz0}, {"br", f.br}, {"ch", f.ch}, {"ck", f.ck}, {"cka", f.cka},
      {"cd", f.cd}, {"cda", f.cda},
      // SLAB's
      {"tmn", land.tmn}, {"emiss", land.emiss}, {"thc", land.thc}, {"capg", land.capg},
      {"chklowq", land.chklowq}, {"gsw", land.gsw}, {"glw", land.glw}, {"snowc", land.snowc}};
  std::map<std::string, H3> hf;   // Fortran side
  for (auto& [n, a] : fm) hf[n] = hnew(a);
  // tmn, emiss, thc: what init_surface.F/getlanduse give (LANDUSE.TBL, checked
  // against getlanduse in slab_tests.cpp)
  const int cat = lu0 == 0 ? 16 : lu0;
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      hf["tmn"](i, j, 1) = 291.5;
      hf["emiss"](i, j, 1) = sfc::landuse_sfem(cat, 1);
      hf["thc"](i, j, 1) = sfc::landuse_therin(cat, 1) / 100.;
    }
  for (auto n : {"tmn", "emiss", "thc", "gsw", "glw", "snowc"}) require_same(fm[n].to_host(), hf[n], std::string("setup ") + n);

  M::Inputs in;
  auto a3 = [&](const char* n) { return FArray3(n, {1, 1, 1}, {ni, nj, nk}); };
  in.u3d = a3("u3d"); in.v3d = a3("v3d"); in.dz8w = a3("dz8w"); in.t3d = a3("t3d"); in.qv3d = a3("qv3d");
  in.p3d = a3("p3d"); in.rho3d = a3("rho3d");
  H3 u = hnew(in.u3d), v = hnew(in.v3d), dz = hnew(in.dz8w), t = hnew(in.t3d), qv = hnew(in.qv3d),
     p = hnew(in.p3d), rho = hnew(in.rho3d);
  auto a1 = [&](const char* n) { return FArray3(n, {1, 1, 1}, {ni, nj, 1}); };
  FArray3 cq = a1("cq"), s10 = a1("s10"), thflux = a1("thflux"), qvflux = a1("qvflux"), rf = a1("rf");
  H3 rfh = hnew(rf), cqh = hnew(cq), s10h = hnew(s10), thh = hnew(thflux), qvh = hnew(qvflux);

  std::mt19937_64 gen(seed);
  auto U = [&](double a, double bb) { return std::uniform_real_distribution<double>(a, bb)(gen); };
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      const bool water = ((i + 2 * j) % 5 == 0);
      hf["xland"](i, j, 1) = water ? 2.0 : 1.0;
      hf["mavail"](i, j, 1) = water ? 1.0 : U(0.05, 0.6);
      hf["znt"](i, j, 1) = water ? 1.0e-4 : U(0.01, 0.5);
      hf["ust"](i, j, 1) = 1.0e-6;
      hf["qsfc"](i, j, 1) = 1.0e-5;
      hf["hpbl"](i, j, 1) = U(100.0, 1500.0);
      hf["psfc"](i, j, 1) = 1.0e5 + U(-2000.0, 2000.0);
      hf["tsk"](i, j, 1) = U(285.0, 310.0);
      dz(i, j, 1) = U(20.0, 60.0);
      dz(i, j, 2) = U(20.0, 60.0);
      rfh(i, j, 1) = 1.1 + 0.01 * i - 0.02 * j;
    }
  rf.deep_copy_from(rfh);
  for (auto& [n, a] : fm) a.deep_copy_from(hf[n]);

  // soil layers and the initial tslb (init_surface.F)
  land.init_soil(b, f.tsk);
  const int nsl = land.soil.n;
  std::vector<double> zs(nsl, 0.0), dzs(nsl, 0.0);
  H3 tslb = h3("tslb", {1, 1, 1}, {ni, nj, nsl});
  cm1f_slab_soil(nsl, 1, ni, 1, nj, hf["tsk"].data(), hf["tmn"].data(), zs.data(), dzs.data(), tslb.data());
  require_same(land.tslb.to_host(), tslb, "init tslb");

  // SLAB's packed 2D arrays (cm1f_bridge_slab.F90's order)
  const char* const pk[18] = {"flhc", "flqc", "psfc", "xland", "tmn", "hfx", "qfx", "lh", "tsk",
                              "qsfc", "chklowq", "gsw", "glw", "capg", "thc", "snowc", "emiss", "mavail"};
  H3 s2d = h3("s2d", {1, 1, 1}, {ni, nj, 18});

  slab::Land::Surface ls;
  ls.flhc = f.flhc; ls.flqc = f.flqc; ls.psfc = f.psfc; ls.xland = f.xland; ls.hfx = f.hfx; ls.qfx = f.qfx;
  ls.lh = f.lh; ls.tsk = f.tsk; ls.qsfc = f.qsfc; ls.mavail = f.mavail;

  const H3 tsk0 = hcopy(hf["tsk"]);
  for (int step = 0; step < nsteps; ++step) {
    INFO("step " << step);
    const bool update_sfc = step > 0;   // cm1.F: .false. in the startup pass
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        const double t1 = 288.0 + 6.0 * std::sin(0.4 * i + 0.9 * j) + U(-1.0, 1.0);
        const double wind = (i == 1) ? U(0.0, 0.1) : U(1.0, 15.0);
        const double dir = U(0.0, 6.283185307179586);
        for (int k = 1; k <= 2; ++k) {
          u(i, j, k) = wind * std::cos(dir) * (k == 2 ? 1.1 : 1.0);
          v(i, j, k) = wind * std::sin(dir) * (k == 2 ? 1.1 : 1.0);
        }
        p(i, j, 1) = hf["psfc"](i, j, 1) - 1.16 * 9.81 * 0.5 * dz(i, j, 1);
        t(i, j, 1) = t1;
        qv(i, j, 1) = U(0.3, 0.95) * qsat(t1, p(i, j, 1));
        rho(i, j, 1) = p(i, j, 1) / (C.rd * t1 * (1.0 + 0.61 * qv(i, j, 1)));
      }
    in.u3d.deep_copy_from(u); in.v3d.deep_copy_from(v); in.dz8w.deep_copy_from(dz); in.t3d.deep_copy_from(t);
    in.qv3d.deep_copy_from(qv); in.p3d.deep_copy_from(p); in.rho3d.deep_copy_from(rho);

    // C++: pbl_mynn::surface_layer's sequence
    M::sfclay(o, C, tables, ni, nj, dx, step, 1, in, f);
    if (update_sfc) land.run(W, dt, ni, nj, in.t3d, in.qv3d, in.p3d, ls);
    else            land.diag_qsfc(ni, nj, in.qv3d, ls);
    M::finish(ni, nj, true, in, rf, f, cq, s10, thflux, qvflux, false);

    // Fortran: turb.F's
    cm1f_mynn_sfclay(ni, nj, nk, step, 1, 0, iz0tlnd, dx, u.data(), v.data(), dz.data(), t.data(), qv.data(),
                     p.data(), rho.data(), hf["psfc"].data(), hf["hpbl"].data(), hf["mavail"].data(),
                     hf["xland"].data(), hf["tsk"].data(), hf["snowh"].data(), hf["ust"].data(), hf["mol"].data(),
                     hf["qsfc"].data(), hf["znt"].data(), hf["ustm"].data(), hf["hfx"].data(), hf["qfx"].data(),
                     hf["zol"].data(), hf["rmol"].data(), hf["wspd"].data(), hf["chs"].data(), hf["chs2"].data(),
                     hf["cqs2"].data(), hf["cpmm"].data(), hf["regime"].data(), hf["psim"].data(),
                     hf["psih"].data(), hf["lh"].data(), hf["flhc"].data(), hf["flqc"].data(), hf["qgh"].data(),
                     hf["u10"].data(), hf["v10"].data(), hf["th2"].data(), hf["t2"].data(), hf["q2"].data(),
                     hf["gz1oz0"].data(), hf["br"].data(), hf["ch"].data(), hf["ck"].data(), hf["cka"].data(),
                     hf["cd"].data(), hf["cda"].data());
    if (update_sfc) {
      for (int n = 0; n < 18; ++n)
        for (int j = 1; j <= nj; ++j)
          for (int i = 1; i <= ni; ++i) s2d(i, j, n + 1) = hf[pk[n]](i, j, 1);
      cm1f_slab(ni, nj, nk, 1, ni, 1, nj, 1, nk, nsl, 0, 1, dt, rc.data(), zs.data(), dzs.data(), t.data(),
                qv.data(), p.data(), s2d.data(), tslb.data());
      for (int n = 0; n < 18; ++n)
        for (int j = 1; j <= nj; ++j)
          for (int i = 1; i <= ni; ++i) hf[pk[n]](i, j, 1) = s2d(i, j, n + 1);
    } else {
      // turb.F's other branch also sets chklowq = mavail (the qsfc part is
      // in cm1f_mynn_sfc_finish)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) hf["chklowq"](i, j, 1) = hf["mavail"](i, j, 1);
    }
    // ch/cq/s10/thflux/qvflux; its qsfc = qv + qfx/flqc is what SLAB (or the
    // startup branch) just computed, with the same inputs
    cm1f_mynn_sfc_finish(ni, nj, 1, qv.data(), rfh.data(), hf["qsfc"].data(), hf["qfx"].data(), hf["flqc"].data(),
                         hf["chs2"].data(), hf["cqs2"].data(), hf["ck"].data(), hf["u10"].data(), hf["v10"].data(),
                         hf["hfx"].data(), hf["cpmm"].data(), hf["ch"].data(), cqh.data(), s10h.data(), thh.data(),
                         qvh.data());

    for (auto& [n, a] : fm) require_same(a.to_host(), hf[n], n);
    require_same(land.tslb.to_host(), tslb, "tslb");
    require_same(cq.to_host(), cqh, "cq");
    require_same(s10.to_host(), s10h, "s10");
    require_same(thflux.to_host(), thh, "thflux");
    require_same(qvflux.to_host(), qvh, "qvflux");
  }
  // SLAB changed tsk at land points only
  int nland = 0;
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      if (hf["xland"](i, j, 1) < 1.5) nland += (hf["tsk"](i, j, 1) != tsk0(i, j, 1));
      else REQUIRE(hf["tsk"](i, j, 1) == tsk0(i, j, 1));
    }
  REQUIRE(nland > 0);
}

} // namespace

TEST_CASE("mynn sfc + SLAB over land (sfcmodel=6 in turb.F) matches the Fortran", "[bridge][sfc_mynn][slab]") {
  cm1f_mynn_sf_init();
  unsigned seed = 8101;
  for (int iz0tlnd : {0, 1, 3})
    for (int lu0 : {3, 7, 19}) run_land(iz0tlnd, lu0, seed++);
}
