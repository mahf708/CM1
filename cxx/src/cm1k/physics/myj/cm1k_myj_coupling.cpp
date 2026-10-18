// The MYJ surface layer + PBL inside the full model. See cm1k_myj_coupling.hpp.
#include "physics/myj/cm1k_myj_coupling.hpp"

#include "share/cm1k_constants.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::myj {

namespace {
// Fortran MAX(a, b) as gfortran evaluates it
KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return (a > b) ? a : b; }
} // namespace

Coupling Coupling::from_namelist(const Namelist& nl) {
  Coupling o;
  o.cm1setup = nl.geti("cm1setup", 1);
  o.ipbl = nl.geti("ipbl", 0);
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.isftcflx = nl.geti("isftcflx", 0);
  o.testcase = nl.geti("testcase", 0);
  o.output_sfcflx = nl.geti("output_sfcflx", 0);
  o.output_sfcflx = o.output_sfcflx < 0 ? 0 : (o.output_sfcflx > 1 ? 1 : o.output_sfcflx);
  // init_surface.F: dosfcflx (isfcflx = 1 is forced by param.F for sfcmodel=7)
  o.dosfcflx = false;
  if (o.cm1setup >= 1 || o.ipbl == 2) {
    if (o.sfc_active()) o.dosfcflx = true;
  }
  if (o.ipbl == 1 || o.ipbl == 3 || o.ipbl == 4 || o.ipbl == 5) o.dosfcflx = false;
  if (o.cm1setup == 4 && o.sfc_active()) o.dosfcflx = true;
  return o;
}

void Coupling::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k MYJ: " + m); };
  if (pbl_active() && !sfc_active())
    fail("ipbl=6 requires sfcmodel=7 (param.F stops)");
  if (sfc_active() && !pbl_active())
    fail("sfcmodel=7 is only ported with the MYJ PBL (cm1setup=2, ipbl=6); usemyjpbl=.false. is not ported");
  if (!active()) return;
  if (cm1setup != 2) fail("cm1setup=" + std::to_string(cm1setup) + " is not ported (only 2)");
}

void Carried::allocate(const Bounds& b, const sfc::Options& so, int num_soil_layers) {
  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  mol = mk("mol"); rmol = mk("rmol"); thz0 = mk("thz0"); qz0 = mk("qz0"); uz0 = mk("uz0"); vz0 = mk("vz0");
  akhs = mk("akhs"); akms = mk("akms"); zntmyj = mk("zntmyj"); z0base = mk("z0base");
  th10 = mk("th10"); q10 = mk("q10"); u10e = mk("u10e"); v10e = mk("v10e");
  tshltr = mk("tshltr"); qshltr = mk("qshltr"); pshltr = mk("pshltr");
  chs = mk("chs"); chs2 = mk("chs2"); cqs2 = mk("cqs2"); cpmm = mk("cpmm"); flhc = mk("flhc"); flqc = mk("flqc");
  qgh = mk("qgh"); lh = mk("lh"); psim = mk("psim"); psih = mk("psih"); fm = mk("fm"); fh = mk("fh");
  ct = mk("ct"); ustm = mk("ustm");
  soil = slab::SoilLayers::make(num_soil_layers);
  tmn = mk("tmn"); emiss = mk("emiss"); thc = mk("thc"); capg = mk("capg"); chklowq = mk("chklowq");
  gsw = mk("gsw"); glw = mk("glw"); snowc = mk("snowc");
  tslb = FArray3("tslb", lo, {b.ie, b.je, num_soil_layers});
  tke_myj = FArray3("tke_myj", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
  el_myj = FArray3("el_myj", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
  xkzh = FArray3("xkzh", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  xkzm = FArray3("xkzm", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  mixht = mk("mixht"); kpbl2d = mk("kpbl2d");
  // cm1.F
  tke_myj.fill(0.01);
  // init_surface.F (initsfc=1) + getlanduse
  if (so.sfcmodel == 7) {
    const int cat = (so.lu0 == 0) ? 16 : so.lu0;
    tmn.fill(so.tmn0);
    emiss.fill(sfc::landuse_sfem(cat, so.season));
    thc.fill(sfc::landuse_therin(cat, so.season) / 100.);
  }
}

std::vector<std::pair<std::string, FArray3>> Carried::dumped_sfc() const {
  return {{"mol", mol},       {"rmol", rmol},   {"tslb", tslb},     {"thz0", thz0}, {"qz0", qz0},
          {"uz0", uz0},       {"vz0", vz0},     {"akhs", akhs},     {"akms", akms}, {"zntmyj", zntmyj},
          {"z0base", z0base}, {"th10", th10},   {"q10", q10},       {"u10e", u10e}, {"v10e", v10e}};
}

std::vector<std::pair<std::string, FArray3>> Carried::dumped_pbl() const {
  return {{"xkzh", xkzh}, {"xkzm", xkzm}, {"tke_myj", tke_myj}, {"el_myj", el_myj}, {"mixht", mixht}};
}

void init(const Bounds& b, const sfc::SfcState& s, const Carried& cr) {
  const auto ust = s.ust, znt = s.znt, z0base = cr.z0base;
  // init_physics.F: MYJSFCINIT (RESTART = .false.) on 1..ni x 1..nj
  Kokkos::parallel_for("myj_init_ust", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    ust(i, j, 1) = 0.1;
  });
  // init_surface.F: z0base = znt on the whole plane
  Kokkos::parallel_for("myj_init_z0base", box2(b.ib, b.ie, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) {
    z0base(i, j, 1) = znt(i, j, 1);
  });
  slab::init_tslb(cr.soil, b, s.tsk, cr.tmn, cr.tslb);
}

void Work::allocate(const Bounds& b) {
  tab = Tables::make();
  ws.allocate(b.ni, b.nj, b.nk);
  ht = FArray3("myj_ht", {b.ib, b.jb, 1}, {b.ie, b.je, 1});
  exch_h = FArray3("myj_exch_h", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  exch_m = FArray3("myj_exch_m", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
}

void surface_layer(const Coupling& o, const wrfphys::WrfConst& c, const Grid& g, Real dt, int nstep,
                   bool update_sfc, const WrfArrays& wa, const FArray3& prs, const FArray3& rf,
                   const sfc::SfcState& s, const Carried& cr, const Work& w) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj;
  SfcInputs in;
  in.dz = wa.d5; in.pmid = prs; in.pint = wa.d6; in.th = wa.d3; in.t = wa.d4; in.qv = wa.qv; in.qc = wa.qc;
  in.u = wa.d1; in.v = wa.d2; in.q2 = cr.tke_myj; in.ht = w.ht; in.rhosfc = rf;
  SfcFields f;
  f.tsk = s.tsk; f.xland = s.xland; f.z0base = cr.z0base; f.mavail = s.mavail;
  f.qsfc = s.qsfc; f.thz0 = cr.thz0; f.qz0 = cr.qz0; f.uz0 = cr.uz0; f.vz0 = cr.vz0; f.ust = s.ust;
  f.znt = cr.zntmyj; f.pblh = s.hpbl; f.akhs = cr.akhs; f.akms = cr.akms;
  f.rmol = cr.rmol; f.rib = s.br; f.chs = cr.chs; f.chs2 = cr.chs2; f.cqs2 = cr.cqs2; f.hfx = s.hfx;
  f.qfx = s.qfx; f.lh = cr.lh; f.flhc = cr.flhc; f.flqc = cr.flqc; f.qgh = cr.qgh; f.cpm = cr.cpmm;
  f.ct = cr.ct; f.u10 = s.u10; f.v10 = s.v10; f.t02 = s.t2; f.th02 = s.th2; f.tshltr = cr.tshltr;
  f.th10 = cr.th10; f.q02 = s.q2; f.qshltr = cr.qshltr; f.q10 = cr.q10; f.pshltr = cr.pshltr;
  f.u10e = cr.u10e; f.v10e = cr.v10e; f.psim = cr.psim; f.psih = cr.psih; f.fm = cr.fm; f.fhh = cr.fh;
  f.z0out = s.znt;
  myjsfc(w.tab, w.ws, ni, nj, b.nk, nstep, o.pbl_active(), o.isftcflx, in, f);
  {
    const auto br = s.br, gz1oz0 = s.gz1oz0, znt = s.znt;
    const auto zh = g.zh;
    Kokkos::parallel_for("myj_br", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      br(i, j, 1) = fmax2(-10.0, br(i, j, 1));
      br(i, j, 1) = (10.0 < br(i, j, 1)) ? 10.0 : br(i, j, 1);   // min(10.0, br)
      gz1oz0(i, j, 1) = Kokkos::log(zh(i, j, 1) / znt(i, j, 1));
    });
  }
  // SLAB (ifsnow = 0, radiation switch .true.)
  if (update_sfc && o.testcase != 11) {
    slab::Arrays a;
    a.flhc = cr.flhc; a.flqc = cr.flqc; a.psfc = s.psfc; a.xland = s.xland; a.tmn = cr.tmn;
    a.hfx = s.hfx; a.qfx = s.qfx; a.lh = cr.lh; a.tsk = s.tsk; a.qsfc = s.qsfc;
    a.chklowq = cr.chklowq; a.gsw = cr.gsw; a.glw = cr.glw; a.capg = cr.capg; a.thc = cr.thc;
    a.snowc = cr.snowc; a.emiss = cr.emiss; a.mavail = s.mavail; a.tslb = cr.tslb;
    slab::slab(c, dt, 0, true, cr.soil, ni, nj, wa.d4, wa.qv, prs, a);
  }
  // WRF parameters into CM1 arrays, the sfcmodel=7 block, thflux/qvflux
  {
    const auto ch = s.ch, cq = s.cq, s10 = s.s10, u10 = s.u10, v10 = s.v10, chs2 = cr.chs2, cqs2 = cr.cqs2;
    const auto ustm = cr.ustm, ust = s.ust, wspd = s.wspd, d1 = wa.d1, d2 = wa.d2, rmol = cr.rmol, mol = cr.mol;
    const auto fm = cr.fm, cd = s.cd, thflux = s.thflux, qvflux = s.qvflux, hfx = s.hfx, qfx = s.qfx;
    const auto cpmm = cr.cpmm;
    const bool flx = o.dosfcflx || o.output_sfcflx == 1;
    const Real smeps = 1.0e-60;   // cm1.F, double precision
    const Real rkarman = phys::rkarman;
    Kokkos::parallel_for("myj_sfc_post", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      ch(i, j, 1) = chs2(i, j, 1);
      cq(i, j, 1) = cqs2(i, j, 1);
      s10(i, j, 1) = Kokkos::sqrt(u10(i, j, 1) * u10(i, j, 1) + v10(i, j, 1) * v10(i, j, 1));
      ustm(i, j, 1) = ust(i, j, 1);
      // from wrf surface_driver:
      wspd(i, j, 1) = fmax2(Kokkos::sqrt(d1(i, j, 1) * d1(i, j, 1) + d2(i, j, 1) * d2(i, j, 1)), 0.001);
      if (Kokkos::fabs(rmol(i, j, 1)) <= smeps)
        mol(i, j, 1) = Kokkos::copysign(1.0e20, rmol(i, j, 1));
      else
        mol(i, j, 1) = 1.0 / rmol(i, j, 1);
      const Real tem = fmax2(0.1, ust(i, j, 1) * rkarman * fm(i, j, 1));
      wspd(i, j, 1) = fmax2(wspd(i, j, 1), tem);
      cd(i, j, 1) = ust(i, j, 1) * ust(i, j, 1) / fmax2(1.0e-20, s10(i, j, 1) * s10(i, j, 1));
      if (flx) {
        thflux(i, j, 1) = hfx(i, j, 1) / (cpmm(i, j, 1) * rf(i, j, 1));
        qvflux(i, j, 1) = qfx(i, j, 1) / rf(i, j, 1);
      }
    });
  }
}

void pbl(const Coupling& /*o*/, const Bounds& b, Real dt, const WrfArrays& wa, const FArray3& prs,
         const sfc::SfcState& s, const pbl::Tendencies& pt, const Carried& cr, const Work& w) {
  PblInputs in;
  in.dz = wa.d5; in.pmid = prs; in.pint = wa.d6; in.th = wa.d3; in.t = wa.d4; in.exner = wa.d7;
  in.qv = wa.qv; in.qcw = wa.qc; in.qci = wa.qi; in.u = wa.d1; in.v = wa.d2; in.ht = w.ht;
  PblFields f;
  f.tsk = s.tsk; f.xland = s.xland; f.chklowq = cr.chklowq; f.elflx = cr.lh;
  f.qsfc = s.qsfc; f.thz0 = cr.thz0; f.qz0 = cr.qz0; f.uz0 = cr.uz0; f.vz0 = cr.vz0; f.ust = s.ust;
  f.ct = cr.ct; f.akhs = cr.akhs; f.akms = cr.akms; f.tke_myj = cr.tke_myj;
  f.pblh = s.hpbl; f.kpbl = cr.kpbl2d; f.mixht = cr.mixht; f.el_myj = cr.el_myj;
  f.exch_h = w.exch_h; f.exch_m = w.exch_m;
  f.rublten = pt.upten; f.rvblten = pt.vpten; f.rthblten = pt.thpten; f.rqvblten = pt.qvpten;
  f.rqcblten = pt.qcpten; f.rqiblten = pt.qipten;
  myjpbl(w.ws, b.ni, b.nj, b.nk, dt, in, f);
  // turb.F: xkzh(k+1) = EXCH_H(k), k = 1..nk (EXCH_H(nk) is the zeroed tmp2d)
  const int nk = b.nk;
  const auto xkzh = cr.xkzh, xkzm = cr.xkzm, eh = w.exch_h, em = w.exch_m;
  Kokkos::parallel_for("myj_xkz", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    for (int k = 1; k <= nk - 1; ++k) {
      xkzh(i, j, k + 1) = eh(i, j, k);
      xkzm(i, j, k + 1) = em(i, j, k);
    }
    xkzh(i, j, nk + 1) = 0.0;
    xkzm(i, j, nk + 1) = 0.0;
  });
}

} // namespace cm1k::myj
