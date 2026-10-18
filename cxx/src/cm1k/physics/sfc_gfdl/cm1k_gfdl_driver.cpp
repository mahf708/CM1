// The GFDL surface layer in the time step. See cm1k_gfdl_driver.hpp.
#include "physics/sfc_gfdl/cm1k_gfdl_driver.hpp"

namespace cm1k::sfc_gfdl {

void GfdlSurface::setup(const sfc::Options& so, const Bounds& b, const sfc::SfcState& common) {
  o = Options{};
  o.check();
  tab = Tables::make();
  land.setup(so, b);   // cm1.F: num_soil_layers = 5
  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  // one array in cm1.F
  f.tsk = common.tsk; f.znt = common.znt; f.ust = common.ust; f.u10 = common.u10; f.v10 = common.v10;
  f.hfx = common.hfx; f.qfx = common.qfx; f.qsfc = common.qsfc; f.gz1oz0 = common.gz1oz0; f.br = common.br;
  psfc = common.psfc; xland = common.xland; mavail = common.mavail; hfx = common.hfx; qfx = common.qfx;
  cd = common.cd; ch = common.ch; cq = common.cq; s10 = common.s10; thflux = common.thflux;
  qvflux = common.qvflux; wspd = common.wspd;
  // its own
  f.mznt = mk("mznt"); f.chs = mk("chs"); f.chs2 = mk("chs2"); f.cqs2 = mk("cqs2"); f.cpm = mk("cpmm");
  f.psim = mk("psim"); f.psih = mk("psih"); f.lh = mk("lh"); f.taux = mk("taux"); f.tauy = mk("tauy");
  f.flhc = mk("flhc"); f.flqc = mk("flqc"); f.qgh = mk("qgh"); f.wspd = mk("swspd"); f.cd_out = mk("cd_out");
  f.ch_out = mk("ch_out"); f.mol = mk("mol"); f.zol = mk("zol"); f.z0t = mk("z0t"); f.fm = mk("fm");
  f.fh = mk("fh"); f.tstrc = mk("tstrc");
  rmol = mk("rmol"); zkmax = mk("zkmax"); smois = mk("smois");
}

void GfdlSurface::init_soil(const Bounds& b) const { land.init_soil(b, f.tsk); }

std::vector<std::pair<std::string, FArray3>> GfdlSurface::dumped() const {
  return {{"mznt", f.mznt}, {"mol", f.mol},       {"rmol", rmol},   {"zol", f.zol},
          {"br", f.br},     {"gz1oz0", f.gz1oz0}, {"psim", f.psim}, {"psih", f.psih}};
}

void gfdl_surface_layer(const sfc::Options& so, const WrfConst& c, const FArray3& zh, int ni, int nj, Real dt,
                        Real smeps, bool update_sfc, const FArray3& dum1, const FArray3& dum2, const FArray3& dum4,
                        const FArray3& divx, const FArray3& prs, const FArray3& rf, const GfdlSurface& w) {
  const Fields F = w.f;
  const auto zkmax = w.zkmax;
  // turb.F 842-851 (cd_out, ch_out, mol, zol: whole arrays)
  Kokkos::deep_copy(F.cd_out.view(), 0.0);
  Kokkos::deep_copy(F.ch_out.view(), 0.0);
  Kokkos::deep_copy(F.mol.view(), 0.0);
  Kokkos::deep_copy(F.zol.view(), 0.0);
  Kokkos::parallel_for("cm1k::sfc_gfdl::zkmax", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    zkmax(i, j, 1) = zh(i, j, 1);
  });
  Inputs in;
  in.u3d = dum1; in.v3d = dum2; in.t3d = dum4; in.qv3d = divx; in.p3d = prs;
  in.psfc = w.psfc; in.xland = w.xland; in.gsw = w.land.gsw; in.glw = w.land.glw; in.zkmax = zkmax;
  in.smois = w.smois;
  sf_gfdl(w.o, c, w.tab, dt, ni, nj, in, F);
  const auto wspd = w.wspd;
  Kokkos::parallel_for("cm1k::sfc_gfdl::ust_wspd", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const Real u = F.ust(i, j, 1);
    F.ust(i, j, 1) = u > 1.0e-8 ? u : 1.0e-8;
    wspd(i, j, 1) = F.wspd(i, j, 1);
  });

  // SLAB, or the qsfc diagnosis
  slab::Land::Surface ls;
  ls.flhc = F.flhc; ls.flqc = F.flqc; ls.psfc = w.psfc; ls.xland = w.xland; ls.hfx = w.hfx; ls.qfx = w.qfx;
  ls.lh = F.lh; ls.tsk = F.tsk; ls.qsfc = F.qsfc; ls.mavail = w.mavail;
  if (update_sfc && so.testcase != 11) w.land.run(c, dt, ni, nj, dum4, divx, prs, ls);
  else w.land.diag_qsfc(ni, nj, divx, ls);

  // turb.F 1062-1077, 1101-1110
  const auto cd = w.cd, ch = w.ch, cq = w.cq, s10 = w.s10, rmol = w.rmol, thflux = w.thflux,
             qvflux = w.qvflux, hfx = w.hfx, qfx = w.qfx;
  const bool set_fluxes = so.dosfcflx || so.output_sfcflx == 1;
  Kokkos::parallel_for("cm1k::sfc_gfdl::post", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    cd(i, j, 1) = F.cd_out(i, j, 1);
    ch(i, j, 1) = F.ch_out(i, j, 1);
    cq(i, j, 1) = F.ch_out(i, j, 1);
    s10(i, j, 1) = Kokkos::sqrt(F.u10(i, j, 1) * F.u10(i, j, 1) + F.v10(i, j, 1) * F.v10(i, j, 1));
    const Real mol = F.mol(i, j, 1);
    if (Kokkos::fabs(mol) <= smeps) rmol(i, j, 1) = Kokkos::copysign(1.0e20, mol);
    else                            rmol(i, j, 1) = 1.0 / mol;
    if (set_fluxes) {
      thflux(i, j, 1) = hfx(i, j, 1) / (F.cpm(i, j, 1) * rf(i, j, 1));
      qvflux(i, j, 1) = qfx(i, j, 1) / rf(i, j, 1);
    }
  });
}

void gfdl_bcs(const turb::Options& to, const Bounds& b, const GfdlSurface& w, const FArray3& zntmp) {
  turb::bc2d_level(to, b, w.f.mznt, 1);
  const auto mznt = w.f.mznt, z = zntmp;
  Kokkos::parallel_for("cm1k::sfc_gfdl::zntmp", box2(0, b.ni + 1, 0, b.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    z(i, j, 1) = mznt(i, j, 1);
  });
}

} // namespace cm1k::sfc_gfdl
