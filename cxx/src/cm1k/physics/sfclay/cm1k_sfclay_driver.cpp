// The WRF/MM5 surface layers in the time step. See cm1k_sfclay_driver.hpp.
#include "physics/sfclay/cm1k_sfclay_driver.hpp"

#include <stdexcept>

namespace cm1k::sfclay {

void WrfSurface::setup(const sfc::Options& so, const Options& o, const Bounds& b, const sfc::SfcState& common,
                       int num_soil_layers) {
  o.check();
  if (!so.wrf()) throw std::runtime_error("cm1k sfclay::WrfSurface: sfcmodel must be 2 or 3");
  s.allocate(b);
  // one array in cm1.F
  s.psfc = common.psfc; s.znt = common.znt; s.ust = common.ust; s.hpbl = common.hpbl;
  s.mavail = common.mavail; s.xland = common.xland; s.hfx = common.hfx; s.qfx = common.qfx;
  s.tsk = common.tsk; s.qsfc = common.qsfc; s.u10 = common.u10; s.v10 = common.v10;
  s.th2 = common.th2; s.t2 = common.t2; s.q2 = common.q2; s.gz1oz0 = common.gz1oz0;
  s.wspd = common.wspd; s.br = common.br; s.dsxy = common.dsxy; s.cd = common.cd; s.cda = common.cda;
  s.ch = common.ch; s.cq = common.cq; s.s10 = common.s10; s.thflux = common.thflux; s.qvflux = common.qvflux;

  tab = (o.sfcmodel == 2) ? sfclayinit() : sf_sfclayrev_init();
  soil = slab::SoilLayers::make(num_soil_layers);

  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  tmn = mk("tmn"); emiss = mk("emiss"); thc = mk("thc"); capg = mk("capg"); chklowq = mk("chklowq");
  gsw = mk("gsw"); glw = mk("glw"); snowc = mk("snowc");
  tslb = FArray3("tslb", lo, {b.ie, b.je, num_soil_layers});

  // init_surface.F (initsfc=1) + getlanduse
  tmn.fill(so.tmn0);
  emiss.fill(sfc::landuse_sfem(so.lu0, so.season));
  thc.fill(sfc::landuse_therin(so.lu0, so.season) / 100.);
}

std::vector<std::pair<std::string, FArray3>> WrfSurface::dumped() const {
  return {{"mol", s.mol}, {"tslb", tslb}};
}

void init_soil(const Bounds& b, const WrfSurface& w) { slab::init_tslb(w.soil, b, w.s.tsk, w.tmn, w.tslb); }

void wrf_surface_layer(const sfc::Options& so, const Options& o, const WrfConst& c, int ni, int nj, Real dt,
                       bool update_sfc, const FArray3& dum1, const FArray3& dum2, const FArray3& dum4,
                       const FArray3& dum5, const FArray3& divx, const FArray3& prs, const FArray3& rf,
                       const WrfSurface& w) {
  if (o.sfcmodel == 2)
    sfclay(o, c, w.tab, ni, nj, dum1, dum2, dum4, divx, prs, dum5, rf, w.s);
  else
    sfclayrev(o, c, w.tab, ni, nj, dum1, dum2, dum4, divx, prs, dum5, w.s);
  const bool run_slab = update_sfc && so.testcase != 11;
  if (run_slab) {
    slab::Arrays a;
    a.flhc = w.s.flhc; a.flqc = w.s.flqc; a.psfc = w.s.psfc; a.xland = w.s.xland; a.tmn = w.tmn;
    a.hfx = w.s.hfx; a.qfx = w.s.qfx; a.lh = w.s.lh; a.tsk = w.s.tsk; a.qsfc = w.s.qsfc;
    a.chklowq = w.chklowq; a.gsw = w.gsw; a.glw = w.glw; a.capg = w.capg; a.thc = w.thc;
    a.snowc = w.snowc; a.emiss = w.emiss; a.mavail = w.s.mavail; a.tslb = w.tslb;
    // ifsnow = 0, radiation switch .true. (turb.F 1021)
    slab::slab(c, dt, 0, true, w.soil, ni, nj, dum4, divx, prs, a);
  }
  sfclay_post(ni, nj, so.dosfcflx || so.output_sfcflx == 1, divx, rf, w.s, !run_slab);
}

} // namespace cm1k::sfclay
