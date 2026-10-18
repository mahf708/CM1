// The MYNN surface layer + PBL inside the full model. See cm1k_mynn_coupling.hpp.
#include "physics/pbl_mynn/cm1k_mynn_coupling.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::pbl_mynn {

Coupling Coupling::from_namelist(const Namelist& nl) {
  Coupling o;
  o.pbl = Options::from_namelist(nl);
  // param.F: the PBL scheme only runs with cm1setup = 2, 4 (idopbl)
  if (!(o.pbl.cm1setup == 2 || o.pbl.cm1setup == 4)) o.pbl.ipbl = 0;
  o.sfc = sfc_mynn::Options::from_namelist(nl);
  o.output_sfcflx = nl.geti("output_sfcflx", 0);
  o.lu0 = nl.geti("lu0", 16);
  o.initsfc = nl.geti("initsfc", 1);
  o.tmn0 = nl.getr("tmn0", 297.28);
  o.testcase = nl.geti("testcase", 0);
  // init_surface.F: dosfcflx (sfcmodel=6 sets it, ipbl=4,5 clear it again)
  const int cm1setup = nl.geti("cm1setup", 1), ipbl = nl.geti("ipbl", 0);
  const int isfcflx = 1;   // param.F: sfcmodel=2..7 require isfcflx=1 (and set it)
  o.dosfcflx = false;
  if (cm1setup >= 1 || ipbl == 2) {
    if (isfcflx == 1 && o.sfc.active()) o.dosfcflx = true;
  }
  if (ipbl == 1 || ipbl == 3 || ipbl == 4 || ipbl == 5) o.dosfcflx = false;
  if (cm1setup == 4 && isfcflx == 1) o.dosfcflx = true;
  return o;
}

void Coupling::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k MYNN: " + m); };
  if (pbl.active() && !sfc.active())
    fail("ipbl=" + std::to_string(pbl.ipbl) + " is only ported with the MYNN surface layer (sfcmodel=6)");
  if (sfc.active() && !pbl.active())
    fail("sfcmodel=6 is only ported with the MYNN PBL (cm1setup=2, ipbl=4 or 5)");
  if (!active()) return;
  pbl.check();
  sfc.check();
  if (initsfc != 1) fail("initsfc=" + std::to_string(initsfc) + " is not ported (only 1)");
  // init_surface.F (initsfc=1: lu_index = lu0 everywhere) stops for these
  // with sfcmodel=6. Category 16 is water (SLAB then only diagnoses qsfc),
  // anything else land (SLAB predicts tsk and tslb).
  if (lu0 < 1 || lu0 > 33) fail("lu_index (lu0=" + std::to_string(lu0) + ") must be between 1 and 33 with sfcmodel=6");
  if (tmn0 < 100.0) fail("tmn is less than 100 K");
}

void Carried::allocate(const Bounds& b, const sfc::SfcState& s, const pbl::Tendencies& pt,
                       const sfc::Options& so) {
  sf.allocate(b);
  // turb.F passes CM1's arrays to SFCLAY_mynn: the shared ones are aliases
  sf.psfc = s.psfc; sf.hpbl = s.hpbl; sf.mavail = s.mavail; sf.xland = s.xland; sf.tsk = s.tsk;
  sf.ust = s.ust; sf.qsfc = s.qsfc; sf.znt = s.znt; sf.hfx = s.hfx; sf.qfx = s.qfx; sf.wspd = s.wspd;
  sf.u10 = s.u10; sf.v10 = s.v10; sf.th2 = s.th2; sf.t2 = s.t2; sf.q2 = s.q2; sf.gz1oz0 = s.gz1oz0;
  sf.br = s.br; sf.ch = s.ch; sf.cd = s.cd; sf.cda = s.cda;
  sfc2d.allocate(b);
  sfc2d.znt = s.znt; sfc2d.xland = s.xland; sfc2d.tsk = s.tsk; sfc2d.qsfc = s.qsfc; sfc2d.psfc = s.psfc;
  sfc2d.ust = s.ust; sfc2d.ch = s.ch; sfc2d.hfx = s.hfx; sfc2d.qfx = s.qfx; sfc2d.wspd = s.wspd;
  sfc2d.hpbl = s.hpbl; sfc2d.rmol = sf.rmol;
  st.allocate(b);
  td.allocate(b);
  td.upten = pt.upten; td.vpten = pt.vpten; td.thpten = pt.thpten;
  td.qvpten = pt.qvpten; td.qcpten = pt.qcpten; td.qipten = pt.qipten;
  land.setup(so, b);   // cm1.F: num_soil_layers = 5
}

void Carried::init_soil(const Bounds& b, const sfc::SfcState& s) const { land.init_soil(b, s.tsk); }

std::vector<std::pair<std::string, FArray3>> Carried::dumped_sfc() const {
  return {{"mol", sf.mol}, {"rmol", sf.rmol}, {"zol", sf.zol}, {"ustm", sf.ustm}, {"tslb", land.tslb}};
}

std::vector<std::pair<std::string, FArray3>> Carried::dumped_pbl() const {
  return {{"xkzh", td.xkzh}, {"xkzm", td.xkzm}, {"qke", st.qke}, {"qke3d", st.qke_adv},
          {"tsq", st.tsq}, {"qsq", st.qsq}, {"cov", st.cov}, {"sh3d", st.sh3d}, {"sm3d", st.sm3d},
          {"el_pbl", st.el_pbl}, {"qc_bl", st.qc_bl}, {"qi_bl", st.qi_bl},
          {"cldfra_bl", st.cldfra_bl}, {"dqke", st.dqke}};
}

void Work::allocate(const Bounds& b) {
  psi = sfc_mynn::PsiTables::make();
  auto a = [&](const char* n) { return FArray3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
  sin.u3d = a("mynn_sfc_u3d"); sin.v3d = a("mynn_sfc_v3d"); sin.dz8w = a("mynn_sfc_dz8w");
  sin.t3d = a("mynn_sfc_t3d"); sin.qv3d = a("mynn_sfc_qv3d"); sin.p3d = a("mynn_sfc_p3d");
  sin.rho3d = a("mynn_sfc_rho3d");
  pin.dz = a("mynn_dz"); pin.u = a("mynn_u"); pin.v = a("mynn_v"); pin.w = a("mynn_w");
  pin.th = a("mynn_th"); pin.qv = a("mynn_qv"); pin.qc = a("mynn_qc"); pin.qi = a("mynn_qi");
  pin.p = a("mynn_p"); pin.exner = a("mynn_exner"); pin.rho = a("mynn_rho"); pin.t3d = a("mynn_t3d");
  qketen = a("qketen");
}

void surface_layer(const Coupling& o, const Constants& c, const Grid& g, const BaseState& base, Real cgs1,
                   Real cgs2, Real cgs3, int nstep, Real dt, bool update_sfc, const FArray3& ua,
                   const FArray3& va, const FArray3& tha, const FArray3& ppi, const FArray4& qa, int nqv,
                   const FArray3& prs, const FArray3& rho, const FArray3& rf, const sfc::SfcState& s,
                   const Carried& cr, const Work& w) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj;
  // turb.F: psfc, then (bbc=3, kval=1) u1, v1, s1, t1 from the ground-
  // relative winds (ugr = ua, vgr = va: imove=0, pertflx=0 with sfcmodel>=2)
  {
    const auto psfc = s.psfc, u1 = s.u1, v1 = s.v1, s1 = s.s1, t1 = s.t1;
    const auto th0 = base.th0;
    Kokkos::parallel_for("mynn_sfc_k1", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      psfc(i, j, 1) = cgs1 * prs(i, j, 1) + cgs2 * prs(i, j, 2) + cgs3 * prs(i, j, 3);
      u1(i, j, 1) = 0.5 * (ua(i, j, 1) + ua(i + 1, j, 1));
      v1(i, j, 1) = 0.5 * (va(i, j, 1) + va(i, j + 1, 1));
      s1(i, j, 1) = Kokkos::sqrt(u1(i, j, 1) * u1(i, j, 1) + v1(i, j, 1) * v1(i, j, 1));
      t1(i, j, 1) = th0(i, j, 1) + tha(i, j, 1);
    });
  }
  // the k2=2 "variables for wrf physics" and the mynn_sf_j_loop packing
  sfc_mynn::prepare_inputs(o.sfc, g, ua, va, base.th0, tha, base.pi0, ppi, prs, rho, qa, nqv, w.sin);
  // SFCLAY_mynn (isfflx = 1, itimestep = nstep, DX = dx)
  sfc_mynn::sfclay(o.sfc, c, w.psi, ni, nj, g.dx, nstep, 1, w.sin, cr.sf);
  // turb.F 1019-1047: SLAB (land: tsk, tslb, hfx, qfx, lh; qsfc), or the
  // qsfc diagnosis in the startup pass
  slab::Land::Surface ls;
  ls.flhc = cr.sf.flhc; ls.flqc = cr.sf.flqc; ls.psfc = cr.sf.psfc; ls.xland = cr.sf.xland;
  ls.hfx = cr.sf.hfx; ls.qfx = cr.sf.qfx; ls.lh = cr.sf.lh; ls.tsk = cr.sf.tsk; ls.qsfc = cr.sf.qsfc;
  ls.mavail = cr.sf.mavail;
  if (update_sfc && o.testcase != 11)
    cr.land.run(wrfphys::WrfConst::make(c), dt, ni, nj, w.sin.t3d, w.sin.qv3d, prs, ls);
  else
    cr.land.diag_qsfc(ni, nj, w.sin.qv3d, ls);
  // ch, cq, s10, thflux, qvflux
  sfc_mynn::finish(ni, nj, o.dosfcflx || o.output_sfcflx == 1, w.sin, rf, cr.sf, s.cq, s.s10, s.thflux,
                   s.qvflux, false);
}

void pbl(const Coupling& o, const Grid& g, const BaseState& base, Real dt, int nstep, const FArray3& ua,
         const FArray3& va, const FArray3& wa, const FArray3& tha, const FArray3& ppi, const FArray4& qa,
         int nqv, int nqc, int nqi, const FArray3& prs, const FArray3& rho, const Carried& cr, const Work& w) {
  const auto& b = g.b;
  prepare_inputs(o.pbl, g, ua, va, wa, base.th0, tha, base.pi0, ppi, prs, rho, qa, nqv, nqc, nqi, w.pin);
  run(o.pbl, b.ni, b.nj, b.nk, dt, g.dx, nstep, w.pin, cr.sfc2d, cr.st, cr.td);
}

} // namespace cm1k::pbl_mynn
