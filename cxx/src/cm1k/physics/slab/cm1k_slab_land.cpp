// SLAB's state and its turb.F call. See cm1k_slab_land.hpp.
#include "physics/slab/cm1k_slab_land.hpp"

namespace cm1k::slab {

void Land::setup(const sfc::Options& so, const Bounds& b, int num_soil_layers) {
  soil = SoilLayers::make(num_soil_layers);
  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  tmn = mk("tmn"); emiss = mk("emiss"); thc = mk("thc"); capg = mk("capg"); chklowq = mk("chklowq");
  gsw = mk("gsw"); glw = mk("glw"); snowc = mk("snowc");
  tslb = FArray3("tslb", lo, {b.ie, b.je, num_soil_layers});
  // init_surface.F (initsfc=1) + getlanduse (category 0 is water, 16)
  const int cat = (so.lu0 == 0) ? 16 : so.lu0;
  tmn.fill(so.tmn0);
  emiss.fill(sfc::landuse_sfem(cat, so.season));
  thc.fill(sfc::landuse_therin(cat, so.season) / 100.);
}

void Land::init_soil(const Bounds& b, const FArray3& tsk) const { init_tslb(soil, b, tsk, tmn, tslb); }

std::vector<std::pair<std::string, FArray3>> Land::dumped() const { return {{"tslb", tslb}}; }

void Land::run(const WrfConst& c, Real dt, int ni, int nj, const FArray3& t3d, const FArray3& qv3d,
               const FArray3& p3d, const Surface& s) const {
  Arrays a;
  a.flhc = s.flhc; a.flqc = s.flqc; a.psfc = s.psfc; a.xland = s.xland; a.tmn = tmn;
  a.hfx = s.hfx; a.qfx = s.qfx; a.lh = s.lh; a.tsk = s.tsk; a.qsfc = s.qsfc;
  a.chklowq = chklowq; a.gsw = gsw; a.glw = glw; a.capg = capg; a.thc = thc;
  a.snowc = snowc; a.emiss = emiss; a.mavail = s.mavail; a.tslb = tslb;
  // ifsnow = 0, radiation switch .true. (turb.F 1021)
  slab(c, dt, 0, true, soil, ni, nj, t3d, qv3d, p3d, a);
}

void Land::diag_qsfc(int ni, int nj, const FArray3& qv3d, const Surface& s) const {
  const auto flqc = s.flqc, qfx = s.qfx, qsfc = s.qsfc, mavail = s.mavail, chk = chklowq;
  Kokkos::parallel_for("cm1k::slab::diag_qsfc", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const Real qx = qv3d(i, j, 1);
    if (flqc(i, j, 1) != 0.) qsfc(i, j, 1) = qx + qfx(i, j, 1) / flqc(i, j, 1);
    else                     qsfc(i, j, 1) = qx;
    chk(i, j, 1) = mavail(i, j, 1);
  });
}

} // namespace cm1k::slab
