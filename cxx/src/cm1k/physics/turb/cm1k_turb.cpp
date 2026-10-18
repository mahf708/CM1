// Port of CM1's LES subgrid turbulence (sgsmodel=1). See cm1k_turb.hpp for
// what is supported and where the time step calls each routine.
//
// Every expression keeps the Fortran's operation order (cxx/README.md). Loops
// that the Fortran runs as one `do k` nest with several statements are split
// into consecutive kernels only where the statements are independent of each
// other's results at other points (e.g. flux, then boundary copy, then
// divergence), so the values are identical.
#include "physics/turb/cm1k_turb.hpp"

#include "init/cm1k_cm1libs.hpp"
#include "dynamics/cm1k_bc.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::turb {

namespace {

// Fortran max/min for non-NaN arguments.
KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

constexpr Real vialpha = phys::vialpha;
constexpr Real vibeta  = phys::vibeta;
constexpr Real th0r    = phys::th0r;

FArray3 slice(const FArray4& a, int n) {
  auto v = Kokkos::subview(a.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - a.lb(3));
  return FArray3(v, {a.lb(0), a.lb(1), a.lb(2)});
}

} // namespace

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.cm1setup = nl.geti("cm1setup", 1);
  o.sgsmodel = nl.geti("sgsmodel", 1);
  o.ipbl = nl.geti("ipbl", 0);
  o.horizturb = nl.geti("horizturb", 0);
  o.cuparam = nl.geti("cuparam", 0);
  o.tconfig = nl.geti("tconfig", 1);
  if (o.cm1setup == 2) o.tconfig = 2;   // param.F
  o.bcturbs = nl.geti("bcturbs", 1);
  o.doimpl = nl.geti("doimpl", 1);
  o.idiss = nl.geti("idiss", 1);
  o.ramp_sgs = nl.geti("ramp_sgs", 0);     // cm1.F default
  o.ramp_time = nl.getr("ramp_time", 0.0);
  o.bbc = nl.geti("bbc", 1);
  o.tbc = nl.geti("tbc", 1);
  o.wbc = nl.geti("wbc", 1); o.ebc = nl.geti("ebc", 1);
  o.sbc = nl.geti("sbc", 1); o.nbc = nl.geti("nbc", 1);
  o.terrain_flag = nl.getb("terrain_flag", false);
  o.axisymm = nl.geti("axisymm", 0);
  o.do_ib = nl.getb("do_ib", false);
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.isfcflx = nl.geti("isfcflx", 0);
  // param.F overrides
  if (o.sfcmodel >= 1 || nl.geti("oceanmodel", 1) == 2 || o.ipbl >= 1) o.bbc = 3;
  if ((o.ipbl == 3 || o.ipbl == 4 || o.ipbl == 5) && o.idiss >= 1 && o.cm1setup != 4) o.idiss = 0;
  if ((o.sgsmodel == 5 || o.sgsmodel == 6) && o.doimpl >= 1) o.doimpl = 0;
  if (o.do_ib && o.doimpl == 1) o.doimpl = 0;
  // init_surface.F: dosfcflx
  const int isfcflx = o.isfcflx, sfcmodel = o.sfcmodel;
  o.dosfcflx = false;
  if (o.cm1setup >= 1 || o.ipbl == 2) {
    if (isfcflx == 1) o.dosfcflx = true;
    if (sfcmodel == 2 || sfcmodel == 3 || sfcmodel == 4 || sfcmodel == 6 || sfcmodel == 7) o.dosfcflx = true;
  }
  if (o.ipbl == 1 || o.ipbl == 3 || o.ipbl == 4 || o.ipbl == 5) o.dosfcflx = false;
  if (o.cm1setup == 4 && isfcflx == 1) o.dosfcflx = true;
  // param.F: dohturb/dovturb
  const bool idoles = (o.cm1setup == 1 || o.cm1setup == 4);
  o.dohturb = idoles || o.cm1setup == 3 || (o.cm1setup == 2 && o.horizturb == 1);
  o.dovturb = idoles || o.cm1setup == 3;
  // moisture
  o.imoist = nl.geti("imoist", 0);
  o.ptype = nl.geti("ptype", 0);
  o.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  const int psolver = nl.geti("psolver", 3);
  if (psolver >= 4 && psolver <= 7 && o.eqtset == 2) o.eqtset = 1;
  if (o.imoist == 1) {
    if (o.ptype != 1)
      throw std::runtime_error("cm1k turb: imoist=1 with ptype=" + std::to_string(o.ptype) + " is not ported");
    // param.F, ptype=1 (Kessler)
    o.numq = 3; o.nqv = 1; o.nqc = 2; o.nql1 = 2; o.nql2 = 3; o.iice = 0; o.nqi = 0; o.nqs1 = -2; o.nqs2 = -2;
  } else {
    o.numq = 1; o.nqv = 0; o.nqc = 0; o.nql1 = 0; o.nql2 = 0; o.iice = 0;
  }
  o.set_boundary_flags();
  return o;
}

void Options::set_boundary_flags() {
  ibw = (wbc != 1) ? 1 : 0;
  ibe = (ebc != 1) ? 1 : 0;
  ibs = (sbc != 1) ? 1 : 0;
  ibn = (nbc != 1) ? 1 : 0;
}

void Options::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k turb: " + m); };
  // LES (cm1setup=1, sgsmodel=1, ipbl=0), or a PBL scheme (cm1setup=2 with
  // YSU, ipbl=1, or the simple scheme, ipbl=2; sgsmodel is 0 there, param.F)
  // with or without the horizontal Smagorinsky scheme (horizturb=1), see
  // physics/pbl
  const bool les = (cm1setup == 1);
  // (also GFS-EDMF, ipbl=3, the MYNN PBL, ipbl=4/5, and MYJ, ipbl=6, whose
  // turb.F path is the same here)
  const bool pbl2 = (cm1setup == 2 && ipbl >= 1 && ipbl <= 6);
  if (!les && !pbl2)
    fail("cm1setup=" + std::to_string(cm1setup) + " with ipbl=" + std::to_string(ipbl) +
         " is not ported (only cm1setup=1 LES, or cm1setup=2 with ipbl=1..6)");
  if (les && sgsmodel != 1) fail("sgsmodel=" + std::to_string(sgsmodel) + " is not ported (only 1, TKE)");
  if (les && ipbl != 0) fail("ipbl>0 with cm1setup=1 is not ported");
  if (les && horizturb != 0) fail("horizturb=1 with cm1setup=1 is not ported");
  if (pbl2 && sgsmodel != 0) fail("cm1setup=2 needs sgsmodel=0 (param.F)");
  if (pbl2 && horizturb != 0 && horizturb != 1) fail("horizturb must be 0 or 1");
  if (cuparam != 0) fail("cuparam>0 is not ported");
  if (tconfig != 1 && tconfig != 2) fail("tconfig must be 1 or 2");
  if (bcturbs != 1 && bcturbs != 2) fail("bcturbs must be 1 or 2");
  if (doimpl != 0 && doimpl != 1) fail("doimpl must be 0 or 1");
  if (idiss != 0 && idiss != 1) fail("idiss must be 0 or 1");
  if (ramp_sgs < 0) fail("ramp_sgs must be >= 0");
  // surface layers: sfcmodel=1 (physics/sfc), and with a PBL scheme also the
  // WRF/MM5 ones, sfcmodel=2,3 (physics/sfclay), GFDL, sfcmodel=4
  // (physics/sfc_gfdl), MYNN's, sfcmodel=6 (physics/sfc_mynn), and MYJ's,
  // sfcmodel=7 (physics/myj)
  if (!((sfcmodel >= 0 && sfcmodel <= 4) || sfcmodel == 6 || sfcmodel == 7))
    fail("sfcmodel=" + std::to_string(sfcmodel) + " is not ported (only 0..4, 6, 7)");
  if (les && sfcmodel > 1) fail("sfcmodel=" + std::to_string(sfcmodel) + " with the LES is not ported (only 0 and 1)");
  if (dosfcflx && sfcmodel < 1) fail("surface fluxes need a surface layer (sfcmodel>=1)");
  if (bbc != 1 && bbc != 3) fail("bbc=" + std::to_string(bbc) + " is not ported (only 1 free slip, 3 semi-slip)");
  if ((bbc == 3) != (sfcmodel >= 1)) fail("bbc=3 goes with sfcmodel>=1 (and bbc=1 with sfcmodel=0)");
  if (tbc != 1) fail("tbc=" + std::to_string(tbc) + " is not ported (only free slip, tbc=1)");
  for (int v : {wbc, ebc, sbc, nbc})
    if (v < 1 || v > 4) fail("lateral boundary condition " + std::to_string(v) + " is not ported");
  // terrain: the LES TKE scheme (sgsmodel=1) without a surface layer
  if (terrain_flag && !les) fail("terrain with cm1setup=2 (PBL schemes) is not ported");
  if (terrain_flag && sfcmodel >= 1) fail("terrain with a surface layer (sfcmodel>=1) is not ported");
  if (axisymm != 0) fail("axisymm=1 is not ported");
  if (do_ib) fail("immersed boundaries (do_ib) are not ported");
  if (les && (!dohturb || !dovturb)) fail("dohturb/dovturb must be true (LES)");
  if (pbl2 && (dovturb || dohturb != (horizturb == 1))) fail("cm1setup=2: dovturb=.false., dohturb=(horizturb=1)");
  if (imoist != 0 && imoist != 1) fail("imoist must be 0 or 1");
  if (imoist == 1 && (nqv < 1 || nqv > numq)) fail("moisture layout: nqv out of range");
}

//------------------------------------------------------------------------------
// State / work
//------------------------------------------------------------------------------

void TurbState::allocate(const Bounds& b) {
  cm0 = FArray2("cm0", {b.ib, b.jb}, {b.ie, b.je});
  const std::array<int, 3> lc{b.ibc, b.jbc, b.kbc}, uc{b.iec, b.jec, b.kec};
  cme = FArray3("cme", lc, uc); csm = FArray3("csm", lc, uc);
  ce1 = FArray3("ce1", lc, uc); ce2 = FArray3("ce2", lc, uc);
  const std::array<int, 3> lw{b.ib, b.jb, b.kb}, uw{b.ie, b.je, b.ke + 1}, us{b.ie, b.je, b.ke};
  nm = FArray3("nm", lw, uw); defv = FArray3("defv", lw, uw); defh = FArray3("defh", lw, uw);
  lenscl = FArray3("lenscl", lw, uw); dissten = FArray3("dissten", lw, uw);
  t11 = FArray3("t11", lw, us); t12 = FArray3("t12", lw, us); t13 = FArray3("t13", lw, us);
  t22 = FArray3("t22", lw, us); t23 = FArray3("t23", lw, us); t33 = FArray3("t33", lw, us);
}

void TurbState::init_constants(const Bounds& b) {
  // init3d.F: cm0 = c_m (cm1setup=1); cme/ce1/ce2/csm on kb..ke.
  auto cm0_ = cm0; auto cme_ = cme, csm_ = csm, ce1_ = ce1, ce2_ = ce2;
  Kokkos::parallel_for("turb_cm0", box2(b.ib, b.ie, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) {
    cm0_(i, j) = phys::c_m;
  });
  Kokkos::parallel_for("turb_init_constants", box3(b.ib, b.ie, b.jb, b.je, b.kb, b.ke),
                       KOKKOS_LAMBDA(int i, int j, int k) {
    using namespace phys;
    if (cm0_(i, j) > cmemin) {
      cme_(i, j, k) = cm0_(i, j);
      ce1_(i, j, k) = cme_(i, j, k) * c_l * c_l * (1.0 / ri_c - 1.0);
      ce2_(i, j, k) = fmax2(0.0, cme_(i, j, k) * pi * pi - ce1_(i, j, k));
      csm_(i, j, k) = Kokkos::pow(cme_(i, j, k) * cme_(i, j, k) * cme_(i, j, k) / (ce1_(i, j, k) + ce2_(i, j, k)), 0.25);
    }
  });
}

void Work::allocate(const Bounds& b) {
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
  d1 = FArray3("turb_d1", lo, hi); d2 = FArray3("turb_d2", lo, hi); d3 = FArray3("turb_d3", lo, hi);
  d4 = FArray3("turb_d4", lo, hi); d5 = FArray3("turb_d5", lo, hi); d6 = FArray3("turb_d6", lo, hi);
  d7 = FArray3("turb_d7", lo, hi); d8 = FArray3("turb_d8", lo, hi); d9 = FArray3("turb_d9", lo, hi);
  sadv = FArray3("turb_sadv", lo, hi);
}

//------------------------------------------------------------------------------
// bc.F (non-MPI): bcw and bc2d
//------------------------------------------------------------------------------

void bcw(const Options& o, const Bounds& b, const FArray3& w, int flag) {
  const int ni = b.ni, nj = b.nj, nk = b.nk, ng = b.ngxy;
  if (nk >= 2) {
    // west
    if (o.wbc == 1)
      Kokkos::parallel_for("bcw_w", box3(1, ng, b.jb, b.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(pwrap(ni + 1 - i, 1, ni), j, k); });
    else if (o.wbc == 2)
      Kokkos::parallel_for("bcw_w", box3(1 - ng, 0, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(1, j, k); });
    else
      Kokkos::parallel_for("bcw_w", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(i, j, k); });
    // east
    if (o.ebc == 1)
      Kokkos::parallel_for("bcw_e", box3(1, ng, b.jb, b.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(pwrap(i, 1, ni), j, k); });
    else if (o.ebc == 2)
      Kokkos::parallel_for("bcw_e", box3(ni + 1, ni + ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(ni, j, k); });
    else
      Kokkos::parallel_for("bcw_e", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(ni + 1 - i, j, k); });
    // south
    if (o.sbc == 1)
      Kokkos::parallel_for("bcw_s", box3(b.ib, b.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, pwrap(nj + 1 - j, 1, nj), k); });
    else if (o.sbc == 2)
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1 - ng, 0, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, 1, k); });
    else
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, j, k); });
    // north
    if (o.nbc == 1)
      Kokkos::parallel_for("bcw_n", box3(b.ib, b.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, pwrap(j, 1, nj), k); });
    else if (o.nbc == 2)
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, nj + 1, nj + ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, nj, k); });
    else
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, nj + 1 - j, k); });
  }
  if (flag == 1)
    Kokkos::parallel_for("bcw_tb", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      w(i, j, 1) = 0.0;
      w(i, j, nk + 1) = 0.0;
    });
}

void bc2d_level(const Options& o, const Bounds& b, const FArray3& s, int k) {
  const int ni = b.ni, nj = b.nj, ng = b.ngxy;
  // west
  if (o.wbc == 1)
    Kokkos::parallel_for("bc2d_w", box2(1, ng, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(1 - i, j, k) = s(pwrap(ni + 1 - i, 1, ni), j, k); });
  else if (o.wbc == 2)
    Kokkos::parallel_for("bc2d_w", box2(1 - ng, 0, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(1, j, k); });
  else
    Kokkos::parallel_for("bc2d_w", box2(1, ng, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(1 - i, j, k) = s(i, j, k); });
  // east
  if (o.ebc == 1)
    Kokkos::parallel_for("bc2d_e", box2(1, ng, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(ni + i, j, k) = s(pwrap(i, 1, ni), j, k); });
  else if (o.ebc == 2)
    Kokkos::parallel_for("bc2d_e", box2(ni + 1, ni + ng, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(ni, j, k); });
  else
    Kokkos::parallel_for("bc2d_e", box2(1, ng, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) { s(ni + i, j, k) = s(ni + 1 - i, j, k); });
  // south
  if (o.sbc == 1)
    Kokkos::parallel_for("bc2d_s", box2(b.ib, b.ie, 1, ng), KOKKOS_LAMBDA(int i, int j) { s(i, 1 - j, k) = s(i, pwrap(nj + 1 - j, 1, nj), k); });
  else if (o.sbc == 2)
    Kokkos::parallel_for("bc2d_s", box2(b.ib, b.ie, 1 - ng, 0), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(i, 1, k); });
  else
    Kokkos::parallel_for("bc2d_s", box2(b.ib, b.ie, 1, ng), KOKKOS_LAMBDA(int i, int j) { s(i, 1 - j, k) = s(i, j, k); });
  // north
  if (o.nbc == 1)
    Kokkos::parallel_for("bc2d_n", box2(b.ib, b.ie, 1, ng), KOKKOS_LAMBDA(int i, int j) { s(i, nj + j, k) = s(i, pwrap(j, 1, nj), k); });
  else if (o.nbc == 2)
    Kokkos::parallel_for("bc2d_n", box2(b.ib, b.ie, nj + 1, nj + ng), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(i, nj, k); });
  else
    Kokkos::parallel_for("bc2d_n", box2(b.ib, b.ie, 1, ng), KOKKOS_LAMBDA(int i, int j) { s(i, nj + j, k) = s(i, nj + 1 - j, k); });
}

//------------------------------------------------------------------------------
// sfc_and_turb pieces
//------------------------------------------------------------------------------

void update_constants(const Options& o, const Bounds& b, Real rtime, const TurbState& ts) {
  if (o.ramp_sgs < 1) return;
  Real turbfac = (rtime - 0.0) / o.ramp_time;
  turbfac = fmax2(0.0, turbfac);
  turbfac = fmin2(1.0, turbfac);
  auto cm0 = ts.cm0; auto cme = ts.cme, csm = ts.csm, ce1 = ts.ce1, ce2 = ts.ce2;
  Kokkos::parallel_for("turb_ramp", box3(1, b.ni, 1, b.nj, 1, b.nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    using namespace phys;
    if (cm0(i, j) > cmemin) {
      cme(i, j, k) = fmax2(0.001, turbfac * cm0(i, j));
      ce1(i, j, k) = cme(i, j, k) * c_l * c_l * (1.0 / ri_c - 1.0);
      ce2(i, j, k) = fmax2(0.0, cme(i, j, k) * pi * pi - ce1(i, j, k));
      csm(i, j, k) = Kokkos::pow(cme(i, j, k) * cme(i, j, k) * cme(i, j, k) / (ce1(i, j, k) + ce2(i, j, k)), 0.25);
    }
  });
}

void calcnm(const Options& o, const Constants& c, const Bounds& b, Real rdz,
            const FArray3& c1, const FArray3& c2, const FArray3& mf,
            const FArray3& pi0, const FArray3& th0, const FArray3& nm,
            const FArray3& t, const FArray3& qt, const FArray3& thv, const FArray3& qvci,
            const FArray3& prs, const FArray3& pp, const FArray3& th, const FArray4& qa) {
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real g = c.g;
  if (o.imoist == 0) {
    Kokkos::parallel_for("calcnm_dry", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      nm(i, j, k) = Kokkos::log((th0(i, j, k) + th(i, j, k)) / (th0(i, j, k - 1) + th(i, j, k - 1)))
                    * g * rdz * mf(i, j, k);
    });
  } else {
    const int ptype = o.ptype, iice = o.iice, numq = o.numq, nqv = o.nqv, nqc = o.nqc, nqi = o.nqi;
    const int nql1 = o.nql1, nql2 = o.nql2, nqs1 = o.nqs1, nqs2 = o.nqs2;
    const Real reps = c.reps;
    Kokkos::parallel_for("calcnm_q", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real qv_ci = (ptype == 0) ? qa(i, j, k, nqv) : qa(i, j, k, nqv) + qa(i, j, k, nqc);
      if (iice == 1) qv_ci = qv_ci + qa(i, j, k, nqi);
      qvci(i, j, k) = qv_ci;
      Real q = 0.0;
      for (int n = 1; n <= numq; ++n)
        if ((n == nqv) || (n >= nql1 && n <= nql2) || (n >= nqs1 && n <= nqs2 && iice == 1))
          q = q + qa(i, j, k, n);
      qt(i, j, k) = q;
      t(i, j, k) = (th0(i, j, k) + th(i, j, k)) * (pi0(i, j, k) + pp(i, j, k));
      thv(i, j, k) = (th0(i, j, k) + th(i, j, k)) * (1.0 + reps * qa(i, j, k, nqv)) / (1.0 + q);
    });
    const Real cp = c.cp, cpv = c.cpv, cpl = c.cpl, cpi = c.cpi, lv1 = c.lv1, lv2 = c.lv2,
               ls1 = c.ls1, ls2 = c.ls2, rd = c.rd, eps = c.eps;
    constexpr Real nmsat = 1.0e-10;
    Kokkos::parallel_for("calcnm_moist", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      // subsaturated formulation:
      nm(i, j, k) = g * Kokkos::log(thv(i, j, k) / thv(i, j, k - 1)) * rdz * mf(i, j, k);
      // saturated formulation (if necessary):
      const Real pavg = c1(i, j, k) * prs(i, j, k - 1) + c2(i, j, k) * prs(i, j, k);
      const Real tavg = c1(i, j, k) * t(i, j, k - 1) + c2(i, j, k) * t(i, j, k);
      const Real qtavg = c1(i, j, k) * qt(i, j, k - 1) + c2(i, j, k) * qt(i, j, k);
      const Real qvciavg = c1(i, j, k) * qvci(i, j, k - 1) + c2(i, j, k) * qvci(i, j, k);
      const Real dlogt = Kokkos::log(t(i, j, k) / t(i, j, k - 1));
      const Real dlogq = Kokkos::log((1.0 + qt(i, j, k)) / (1.0 + qt(i, j, k - 1)));
      auto nmsatf = [&](Real lh, Real drdt, Real qvs, Real cpml) {
        const Real gamma = g * (1.0 + qtavg) * (1.0 + lh * qvs / (rd * tavg)) / (cpml + lh * drdt);
        return g * ((dlogt * rdz * mf(i, j, k) + gamma / tavg) * (1.0 + tavg * drdt / (eps + qvs))
                    - dlogq * rdz * mf(i, j, k));
      };
      auto drdt_l = [&](Real qvs) { const Real d = tavg - 29.65; return 17.67 * (273.15 - 29.65) * qvs / (d * d); };
      auto drdt_i = [&](Real qvs) { const Real d = tavg - 7.66; return 21.8745584 * (273.15 - 7.66) * qvs / (d * d); };
      if (iice == 1) {
        if (tavg >= 273.15) {
          const Real qvs = cm1libs::rslf(ptype, eps, pavg, tavg);
          const Real ql = fmax2(qvciavg - qvs, 0.0);
          if (ql > nmsat) {
            const Real qv = fmax2(qvciavg - ql, 0.0);
            const Real cpml = cp + cpv * qv + cpl * ql;
            const Real lh = lv1 - lv2 * tavg;
            nm(i, j, k) = nmsatf(lh, drdt_l(qvs), qvs, cpml);
          }
        } else if (tavg <= 233.15) {
          const Real qvs = cm1libs::rsif(ptype, eps, pavg, tavg);
          const Real qi = fmax2(qvciavg - qvs, 0.0);
          if (qi > nmsat) {
            const Real qv = fmax2(qvciavg - qi, 0.0);
            const Real cpml = cp + cpv * qv + cpi * qi;
            const Real lh = ls1 - ls2 * tavg;
            nm(i, j, k) = nmsatf(lh, drdt_i(qvs), qvs, cpml);
          }
        } else {
          const Real ff = (tavg - 233.15) / (273.15 - 233.15);
          const Real qvsl = cm1libs::rslf(ptype, eps, pavg, tavg);
          const Real qvsi = cm1libs::rsif(ptype, eps, pavg, tavg);
          const Real qvs = ff * qvsl + (1.0 - ff) * qvsi;
          const Real qi = fmax2(0.0, (1.0 - ff) * (qvciavg - qvs));
          const Real ql = fmax2(0.0, ff * (qvciavg - qvs));
          if ((ql + qi) > nmsat) {
            const Real qv = fmax2(0.0, qvciavg - ql - qi);
            const Real cpml = cp + cpv * qv + cpl * ql + cpi * qi;
            const Real nml = nmsatf(lv1 - lv2 * tavg, drdt_l(qvsl), qvsl, cpml);
            const Real nmi = nmsatf(ls1 - ls2 * tavg, drdt_i(qvsi), qvsi, cpml);
            nm(i, j, k) = ff * nml + (1.0 - ff) * nmi;
          }
        }
      } else {
        const Real qvs = cm1libs::rslf(ptype, eps, pavg, tavg);
        const Real ql = fmax2(qvciavg - qvs, 0.0);
        if (ql > nmsat) {
          const Real qv = fmax2(qvciavg - ql, 0.0);
          const Real cpml = cp + cpv * qv + cpl * ql;
          const Real lh = lv1 - lv2 * tavg;
          nm(i, j, k) = nmsatf(lh, drdt_l(qvs), qvs, cpml);
        }
      }
    });
  }
  Kokkos::parallel_for("calcnm_tb", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    nm(i, j, 1) = 0.0;
    nm(i, j, nk + 1) = 0.0;
  });
}

// turb.F calcdef, "Cartesian with terrain": strain rates (times rho);
// dum1/dum2 are scratch (u, v at w points; then w at scalar points)
void calcdef_terrain(const Grid& g, const FArray3& rho, const FArray3& rf,
                     const FArray3& ua, const FArray3& va, const FArray3& wa,
                     const FArray3& s11, const FArray3& s12, const FArray3& s13,
                     const FArray3& s22, const FArray3& s23, const FArray3& s33,
                     const FArray3& dum1, const FArray3& dum2) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real rdx = g.rdx, rdy = g.rdy, zt = g.zt;
  const Real cgs1 = g.cgs1, cgs2 = g.cgs2, cgs3 = g.cgs3, cgt1 = g.cgt1, cgt2 = g.cgt2, cgt3 = g.cgt3;
  const auto uh = g.uh; const auto uf = g.uf; const auto vh = g.vh; const auto vf = g.vf;
  const auto rds = g.rds; const auto sigma = g.sigma; const auto rdsf = g.rdsf; const auto sigmaf = g.sigmaf;
  const auto gz = g.gz; const auto rgz = g.rgz; const auto rgzu = g.rgzu; const auto rgzv = g.rgzv;
  const auto gx = g.gx; const auto gy = g.gy;

  // dum1 stores u at w-pts, dum2 stores v at w-pts
  Kokkos::parallel_for("calcdef_t_uvw", box3(0, ni + 2, 0, nj + 2, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    if (k == 1) {
      dum1(i, j, 1) = cgs1 * ua(i, j, 1) + cgs2 * ua(i, j, 2) + cgs3 * ua(i, j, 3);
      dum2(i, j, 1) = cgs1 * va(i, j, 1) + cgs2 * va(i, j, 2) + cgs3 * va(i, j, 3);
    } else if (k == nk + 1) {
      dum1(i, j, nk + 1) = cgt1 * ua(i, j, nk) + cgt2 * ua(i, j, nk - 1) + cgt3 * ua(i, j, nk - 2);
      dum2(i, j, nk + 1) = cgt1 * va(i, j, nk) + cgt2 * va(i, j, nk - 1) + cgt3 * va(i, j, nk - 2);
    } else {
      const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
      const Real r1 = 1.0 - r2;
      dum1(i, j, k) = r1 * ua(i, j, k - 1) + r2 * ua(i, j, k);
      dum2(i, j, k) = r1 * va(i, j, k - 1) + r2 * va(i, j, k);
    }
  });
  Kokkos::parallel_for("calcdef_t_s", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    Real t = gz(i, j, 1) * (ua(i + 1, j, k) * rgzu(i + 1, j, 1) - ua(i, j, k) * rgzu(i, j, 1)) * rdx * uh(i)
             + (gx(i, j, k + 1) * (dum1(i, j, k + 1) + dum1(i + 1, j, k + 1))
                - gx(i, j, k) * (dum1(i, j, k) + dum1(i + 1, j, k))) * 0.5 * rdsf(k);
    s11(i, j, k) = t * rho(i, j, k);
    t = gz(i, j, 1) * (va(i, j + 1, k) * rgzv(i, j + 1, 1) - va(i, j, k) * rgzv(i, j, 1)) * rdy * vh(j)
        + (gy(i, j, k + 1) * (dum2(i, j, k + 1) + dum2(i, j + 1, k + 1))
           - gy(i, j, k) * (dum2(i, j, k) + dum2(i, j + 1, k))) * 0.5 * rdsf(k);
    s22(i, j, k) = t * rho(i, j, k);
    t = (wa(i, j, k + 1) - wa(i, j, k)) * rdsf(k) * gz(i, j, 1);
    s33(i, j, k) = t * rho(i, j, k);
  });
  Kokkos::parallel_for("calcdef_t_s12", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real r1 = 0.25 * ((rho(i - 1, j - 1, k) * gz(i - 1, j - 1, 1) + rho(i, j, k) * gz(i, j, 1))
                           + (rho(i - 1, j, k) * gz(i - 1, j, 1) + rho(i, j - 1, k) * gz(i, j - 1, 1)));
    s12(i, j, k) = 0.5 * ((r1 * (ua(i, j, k) * rgzu(i, j, 1) - ua(i, j - 1, k) * rgzu(i, j - 1, 1)) * rdy * vf(j)
                           + 0.5 * ((zt - sigmaf(k + 1)) * (dum1(i, j - 1, k + 1) + dum1(i, j, k + 1))
                                    - (zt - sigmaf(k)) * (dum1(i, j - 1, k) + dum1(i, j, k)))
                                 * rdsf(k) * r1 * (rgzu(i, j, 1) - rgzu(i, j - 1, 1)) * rdy * vf(j))
                          + (r1 * (va(i, j, k) * rgzv(i, j, 1) - va(i - 1, j, k) * rgzv(i - 1, j, 1)) * rdx * uf(i)
                             + 0.5 * ((zt - sigmaf(k + 1)) * (dum2(i - 1, j, k + 1) + dum2(i, j, k + 1))
                                      - (zt - sigmaf(k)) * (dum2(i - 1, j, k) + dum2(i, j, k)))
                                   * rdsf(k) * r1 * (rgzv(i, j, 1) - rgzv(i - 1, j, 1)) * rdx * uf(i)));
  });
  // now, dum1 stores w at scalar-pts
  Kokkos::parallel_for("calcdef_t_ws", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum1(i, j, k) = 0.5 * (wa(i, j, k) + wa(i, j, k + 1));
  });
  if (nk >= 2) {
    Kokkos::parallel_for("calcdef_t_s13", box3(1, ni + 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real t = 0.5 * ((ua(i, j, k) - ua(i, j, k - 1)) * rds(k)
                            + (wa(i, j, k) * rgz(i, j, 1) - wa(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                            + 0.5 * rds(k) * ((zt - sigma(k)) * (dum1(i, j, k) + dum1(i - 1, j, k))
                                              - (zt - sigma(k - 1)) * (dum1(i, j, k - 1) + dum1(i - 1, j, k - 1)))
                                  * (rgz(i, j, 1) - rgz(i - 1, j, 1)) * rdx * uf(i));
      s13(i, j, k) = t * 0.5 * (gz(i - 1, j, 1) * rf(i - 1, j, k) + gz(i, j, 1) * rf(i, j, k));
    });
    Kokkos::parallel_for("calcdef_t_s23", box3(1, ni, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real t = 0.5 * ((va(i, j, k) - va(i, j, k - 1)) * rds(k)
                            + (wa(i, j, k) * rgz(i, j, 1) - wa(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                            + 0.5 * rds(k) * ((zt - sigma(k)) * (dum1(i, j, k) + dum1(i, j - 1, k))
                                              - (zt - sigma(k - 1)) * (dum1(i, j, k - 1) + dum1(i, j - 1, k - 1)))
                                  * (rgz(i, j, 1) - rgz(i, j - 1, 1)) * rdy * vf(j));
      s23(i, j, k) = t * 0.5 * (gz(i, j - 1, 1) * rf(i, j - 1, k) + gz(i, j, 1) * rf(i, j, k));
    });
  }
}

void calcdef(const Options& o, const Grid& g, const FArray3& rho, const FArray3& rf,
             const FArray3& ua, const FArray3& va, const FArray3& wa,
             const FArray3& s11, const FArray3& s12, const FArray3& s13,
             const FArray3& s22, const FArray3& s23, const FArray3& s33,
             const FArray3& defv, const FArray3& defh,
             const FArray3& dum1, const FArray3& dum2) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real rdx = g.rdx, rdy = g.rdy, rdz = g.rdz;
  const auto uh = g.uh;
  const auto uf = g.uf;
  const auto vh = g.vh;
  const auto vf = g.vf;
  const auto mh = g.mh;
  const auto mf = g.mf;
  const auto c1 = g.c1;
  const auto c2 = g.c2;

  if (b.terrain_flag) {
    // Cartesian with terrain (no wall conditions in this branch of turb.F)
    if (!dum1.is_allocated() || !dum2.is_allocated())
      throw std::runtime_error("cm1k turb::calcdef: terrain needs the dum1/dum2 scratch arrays");
    calcdef_terrain(g, rho, rf, ua, va, wa, s11, s12, s13, s22, s23, s33, dum1, dum2);
  } else {
  Kokkos::parallel_for("calcdef_s", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real temz = rdz * mh(1, 1, k);
    s11(i, j, k) = rho(i, j, k) * (ua(i + 1, j, k) - ua(i, j, k)) * rdx * uh(i);
    s22(i, j, k) = rho(i, j, k) * (va(i, j + 1, k) - va(i, j, k)) * rdy * vh(j);
    s33(i, j, k) = rho(i, j, k) * (wa(i, j, k + 1) - wa(i, j, k)) * temz;
    s12(i, j, k) = 0.5 * ((ua(i, j, k) - ua(i, j - 1, k)) * rdy * vf(j)
                         + (va(i, j, k) - va(i - 1, j, k)) * rdx * uf(i))
                   * 0.25 * ((rho(i - 1, j - 1, k) + rho(i, j, k)) + (rho(i - 1, j, k) + rho(i, j - 1, k)));
  });
  auto rho4 = KOKKOS_LAMBDA(int i, int j, int k) {
    return 0.25 * ((rho(i - 1, j - 1, k) + rho(i, j, k)) + (rho(i - 1, j, k) + rho(i, j - 1, k)));
  };
  // lateral boundary conditions (walls)
  if (o.wbc == 3 && o.ibw == 1)
    Kokkos::parallel_for("calcdef_w", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s12(1, j, k) = s12(2, j, k); });
  else if (o.wbc == 4 && o.ibw == 1)
    Kokkos::parallel_for("calcdef_w", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      s12(1, j, k) = 2.0 * va(1, j, k) * rdx * uf(1) * rho4(1, j, k); });
  if (o.ebc == 3 && o.ibe == 1)
    Kokkos::parallel_for("calcdef_e", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s12(ni + 1, j, k) = s12(ni, j, k); });
  else if (o.ebc == 4 && o.ibe == 1)
    Kokkos::parallel_for("calcdef_e", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      s12(ni + 1, j, k) = -2.0 * va(ni, j, k) * rdx * uf(ni + 1) * rho4(ni + 1, j, k); });
  if (o.sbc == 3 && o.ibs == 1)
    Kokkos::parallel_for("calcdef_s", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s12(i, 1, k) = s12(i, 2, k); });
  else if (o.sbc == 4 && o.ibs == 1)
    Kokkos::parallel_for("calcdef_s", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      s12(i, 1, k) = 2.0 * ua(i, 1, k) * rdy * vf(1) * rho4(i, 1, k); });
  if (o.nbc == 3 && o.ibn == 1)
    Kokkos::parallel_for("calcdef_n", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s12(i, nj + 1, k) = s12(i, nj, k); });
  else if (o.nbc == 4 && o.ibn == 1)
    Kokkos::parallel_for("calcdef_n", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      s12(i, nj + 1, k) = -2.0 * ua(i, nj, k) * rdy * vf(nj + 1) * rho4(i, nj + 1, k); });

  if (nk >= 2)
    Kokkos::parallel_for("calcdef_s13", box3(1, ni + 1, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s13(i, j, k) = 0.5 * ((wa(i, j, k) - wa(i - 1, j, k)) * rdx * uf(i)
                           + (ua(i, j, k) - ua(i, j, k - 1)) * rdz * 0.5 * (mf(i - 1, j, k) + mf(i, j, k)))
                     * 0.5 * (rf(i - 1, j, k) + rf(i, j, k));
      s23(i, j, k) = 0.5 * ((wa(i, j, k) - wa(i, j - 1, k)) * rdy * vf(j)
                           + (va(i, j, k) - va(i, j, k - 1)) * rdz * 0.5 * (mf(i, j - 1, k) + mf(i, j, k)))
                     * 0.5 * (rf(i, j - 1, k) + rf(i, j, k));
    });
  if (nk >= 2) {
    if (o.wbc == 3 && o.ibw == 1)
      Kokkos::parallel_for("calcdef_w13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { s13(1, j, k) = s13(2, j, k); });
    else if (o.wbc == 4 && o.ibw == 1)
      Kokkos::parallel_for("calcdef_w13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { s13(1, j, k) = 2.0 * wa(1, j, k) * rdx * uf(1); });
    if (o.ebc == 3 && o.ibe == 1)
      Kokkos::parallel_for("calcdef_e13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { s13(ni + 1, j, k) = s13(ni, j, k); });
    else if (o.ebc == 4 && o.ibe == 1)
      Kokkos::parallel_for("calcdef_e13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { s13(ni + 1, j, k) = -2.0 * wa(ni, j, k) * rdx * uf(ni + 1); });
    if (o.sbc == 3 && o.ibs == 1)
      Kokkos::parallel_for("calcdef_s23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { s23(i, 1, k) = s23(i, 2, k); });
    else if (o.sbc == 4 && o.ibs == 1)
      Kokkos::parallel_for("calcdef_s23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { s23(i, 1, k) = 2.0 * wa(i, 1, k) * rdy * vf(1); });
    if (o.nbc == 3 && o.ibn == 1)
      Kokkos::parallel_for("calcdef_n23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { s23(i, nj + 1, k) = s23(i, nj, k); });
    else if (o.nbc == 4 && o.ibn == 1)
      Kokkos::parallel_for("calcdef_n23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { s23(i, nj + 1, k) = -2.0 * wa(i, nj, k) * rdy * vf(nj + 1); });
  }
  }  // terrain_flag

  // open boundary conditions
  if (o.wbc == 2 || o.ebc == 2 || o.sbc == 2 || o.nbc == 2) {
    const bool w2 = (o.wbc == 2 && o.ibw == 1), e2 = (o.ebc == 2 && o.ibe == 1);
    const bool s2 = (o.sbc == 2 && o.ibs == 1), n2 = (o.nbc == 2 && o.ibn == 1);
    if (w2) Kokkos::parallel_for("calcdef_ow", box2(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s11(0, j, k) = s11(1, j, k); });
    if (e2) Kokkos::parallel_for("calcdef_oe", box2(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s11(ni + 1, j, k) = s11(ni, j, k); });
    if (s2) Kokkos::parallel_for("calcdef_os", box2(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s22(i, 0, k) = s22(i, 1, k); });
    if (n2) Kokkos::parallel_for("calcdef_on", box2(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s22(i, nj + 1, k) = s22(i, nj, k); });
    if (w2) Kokkos::parallel_for("calcdef_ow12", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s12(1, j, k) = s12(2, j, k); });
    if (e2) Kokkos::parallel_for("calcdef_oe12", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s12(ni + 1, j, k) = s12(ni, j, k); });
    if (s2) Kokkos::parallel_for("calcdef_os12", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s12(i, 1, k) = s12(i, 2, k); });
    if (n2) Kokkos::parallel_for("calcdef_on12", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s12(i, nj + 1, k) = s12(i, nj, k); });
    const bool sw = s2 && w2, se = s2 && e2, nw = n2 && w2, ne = n2 && e2;
    if (sw || se || nw || ne)
      Kokkos::parallel_for("calcdef_corners", Kokkos::RangePolicy<ExeSpace>(1, nk + 1), KOKKOS_LAMBDA(int k) {
        if (sw) s12(1, 1, k) = s12(2, 2, k);
        if (se) s12(ni + 1, 1, k) = s12(ni, 2, k);
        if (nw) s12(1, nj + 1, k) = s12(2, nj, k);
        if (ne) s12(ni + 1, nj + 1, k) = s12(ni, nj, k);
      });
  }

  // deformation (w points)
  if (nk >= 2)
    Kokkos::parallel_for("calcdef_def", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      auto sq = [](Real x) { return x * x; };
      const Real tmp11 = (c1(i, j, k) * sq(s11(i, j, k - 1)) + c2(i, j, k) * sq(s11(i, j, k)));
      const Real tmp22 = (c1(i, j, k) * sq(s22(i, j, k - 1)) + c2(i, j, k) * sq(s22(i, j, k)));
      const Real tmp33 = (c1(i, j, k) * sq(s33(i, j, k - 1)) + c2(i, j, k) * sq(s33(i, j, k)));
      const Real tmp12 = 0.25 * (c1(i, j, k) * ((sq(s12(i, j, k - 1)) + sq(s12(i + 1, j + 1, k - 1)))
                                               + (sq(s12(i, j + 1, k - 1)) + sq(s12(i + 1, j, k - 1))))
                                + c2(i, j, k) * ((sq(s12(i, j, k)) + sq(s12(i + 1, j + 1, k)))
                                               + (sq(s12(i, j + 1, k)) + sq(s12(i + 1, j, k)))));
      const Real tmp13 = 0.5 * (sq(s13(i, j, k)) + sq(s13(i + 1, j, k)));
      const Real tmp23 = 0.5 * (sq(s23(i, j, k)) + sq(s23(i, j + 1, k)));
      const Real rrf = 1.0 / sq(rf(i, j, k));
      defv(i, j, k) = 4.0 * (tmp13 + tmp23) * rrf;
      defh(i, j, k) = (2.0 * ((tmp11 + tmp22) + tmp33) + 4.0 * tmp12) * rrf;
    });
}

void tkekm(const Options& o, const Grid& g, const FArray3& nm,
           const FArray3& tk, const FArray3& lenh, const FArray3& grdscl, const FArray3& rgrdscl,
           const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
           const FArray3& cme, const FArray3& ce1, const FArray3& ce2,
           const FArray3& tkea, const FArray3& lenscl, const FArray3& dissten) {
  const auto& b = g.b;
  const Real dx = g.dx, dy = g.dy, dz = g.dz;
  const auto ruh = g.ruh;
  const auto rvh = g.rvh;
  const auto rmf = g.rmf;
  const int tconfig = o.tconfig;
  constexpr Real tke_min = 1.0e-10, nm_min = 1.0e-6, small_len_frac = 0.001;
  if (b.nk >= 2)
    Kokkos::parallel_for("tkekm", box3(1, b.ni, 1, b.nj, 2, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (tconfig == 1) {
        grdscl(i, j, k) = Kokkos::pow(((dx * ruh(i)) * (dy * rvh(j))) * (dz * rmf(i, j, k)), 0.33333333);
      } else {
        lenh(i, j, k) = Kokkos::sqrt((dx * ruh(i)) * (dy * rvh(j)));
        grdscl(i, j, k) = dz * rmf(i, j, k);
      }
      rgrdscl(i, j, k) = 1.0 / grdscl(i, j, k);
      // turbulence length scale
      if (tkea(i, j, k) <= tke_min) {
        tk(i, j, k) = tke_min;
        lenscl(i, j, k) = small_len_frac * grdscl(i, j, k);
      } else {
        tk(i, j, k) = tkea(i, j, k);
        lenscl(i, j, k) = grdscl(i, j, k);
        if (nm(i, j, k) > nm_min) {
          lenscl(i, j, k) = phys::c_l * Kokkos::sqrt(tk(i, j, k) / nm(i, j, k));
          lenscl(i, j, k) = fmin2(lenscl(i, j, k), grdscl(i, j, k));
          lenscl(i, j, k) = fmax2(lenscl(i, j, k), small_len_frac * grdscl(i, j, k));
        }
      }
      // dissipation
      dissten(i, j, k) = dissten(i, j, k)
                       + (ce1(i, j, k) + ce2(i, j, k) * lenscl(i, j, k) * rgrdscl(i, j, k))
                         * tk(i, j, k) * Kokkos::sqrt(tk(i, j, k)) / lenscl(i, j, k);
      // km, kh
      Real prinv = 3.00;
      if (tconfig == 1) {
        kmh(i, j, k) = cme(i, j, k) * Kokkos::sqrt(tk(i, j, k)) * lenscl(i, j, k);
        kmv(i, j, k) = kmh(i, j, k);
        if (nm(i, j, k) > nm_min) prinv = fmin2(1.0 + 2.00 * lenscl(i, j, k) * rgrdscl(i, j, k), 3.00);
        khh(i, j, k) = kmh(i, j, k) * prinv;
        khv(i, j, k) = khh(i, j, k);
      } else {
        kmh(i, j, k) = cme(i, j, k) * Kokkos::sqrt(tk(i, j, k)) * lenh(i, j, k);
        kmv(i, j, k) = cme(i, j, k) * Kokkos::sqrt(tk(i, j, k)) * lenscl(i, j, k);
        if (nm(i, j, k) > nm_min) prinv = fmin2(1.0 + 2.00 * lenscl(i, j, k) * rgrdscl(i, j, k), 3.00);
        khh(i, j, k) = kmh(i, j, k) * prinv;
        khv(i, j, k) = kmv(i, j, k) * prinv;
      }
    });
  bcw(o, b, kmh, 1);
  bcw(o, b, kmv, 1);
  bcw(o, b, khh, 1);
  bcw(o, b, khv, 1);
}

void tkebc(const Options& o, const Bounds& b, const FArray3& tkea,
           const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
           const sfc::SfcState* sfc) {
  if ((o.bbc != 1 && o.bbc != 3) || o.tbc != 1)
    throw std::runtime_error("cm1k turb::tkebc: only bbc=1,3 and tbc=1 are ported");
  if (o.bbc == 3) {
    // semi-slip
    if (!sfc) throw std::runtime_error("cm1k turb::tkebc: bbc=3 needs the surface layer state");
    const auto zntmp = sfc->zntmp, ust = sfc->ust;
    Kokkos::parallel_for("tkebc_sfc3", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
      tkea(i, j, 1) = 0.0;
      kmh(i, j, 1) = phys::karman * zntmp(i, j, 1) * ust(i, j, 1);
    });
  } else {
    // free slip
    Kokkos::parallel_for("tkebc_sfc", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
      tkea(i, j, 1) = 0.0;
      kmh(i, j, 1) = 0.0;
    });
  }
  bc2d_level(o, b, tkea, 1);
  bc2d_level(o, b, kmh, 1);
  Kokkos::parallel_for("tkebc_k1", box2(0, b.ni + 1, 0, b.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    kmv(i, j, 1) = kmh(i, j, 1);
    khv(i, j, 1) = kmv(i, j, 1) * (khv(i, j, 2) / (1.0e-10 + kmv(i, j, 2)));
    khh(i, j, 1) = kmh(i, j, 1) * (khh(i, j, 2) / (1.0e-10 + kmh(i, j, 2)));
  });
}

void gettau(const Options& o, const Bounds& b, const FArray3& kmh, const FArray3& kmv,
            const FArray3& t11, const FArray3& t12, const FArray3& t13,
            const FArray3& t22, const FArray3& t23, const FArray3& t33,
            const FArray3& rf, const sfc::SfcState* sfc) {
  if ((o.bbc != 1 && o.bbc != 3) || o.tbc != 1)
    throw std::runtime_error("cm1k turb::gettau: only bbc=1,3 and tbc=1 are ported");
  if (o.bbc == 3 && (!sfc || !rf.is_allocated()))
    throw std::runtime_error("cm1k turb::gettau: bbc=3 needs rf and the surface layer state");
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  Kokkos::parallel_for("gettau_h", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real tem = (kmh(i, j, k) + kmh(i, j, k + 1));
    t11(i, j, k) = t11(i, j, k) * tem;
    t22(i, j, k) = t22(i, j, k) * tem;
    t33(i, j, k) = t33(i, j, k) * tem;
    t12(i, j, k) = t12(i, j, k) * 0.25
      * (((kmh(i - 1, j - 1, k) + kmh(i, j, k)) + (kmh(i - 1, j, k) + kmh(i, j - 1, k)))
         + ((kmh(i - 1, j - 1, k + 1) + kmh(i, j, k + 1)) + (kmh(i - 1, j, k + 1) + kmh(i, j - 1, k + 1))));
  });
  if (o.wbc == 3 && o.ibw == 1)
    Kokkos::parallel_for("gettau_w", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t12(1, j, k) = t12(2, j, k); });
  if (o.ebc == 3 && o.ibe == 1)
    Kokkos::parallel_for("gettau_e", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t12(ni + 1, j, k) = t12(ni, j, k); });
  if (o.sbc == 3 && o.ibs == 1)
    Kokkos::parallel_for("gettau_s", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t12(i, 1, k) = t12(i, 2, k); });
  if (o.nbc == 3 && o.ibn == 1)
    Kokkos::parallel_for("gettau_n", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t12(i, nj + 1, k) = t12(i, nj, k); });

  if (nk >= 2) {
    Kokkos::parallel_for("gettau_v", box3(1, ni + 1, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      t13(i, j, k) = t13(i, j, k) * (kmv(i - 1, j, k) + kmv(i, j, k));
      t23(i, j, k) = t23(i, j, k) * (kmv(i, j - 1, k) + kmv(i, j, k));
    });
    if (o.wbc == 3 && o.ibw == 1)
      Kokkos::parallel_for("gettau_w13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { t13(1, j, k) = t13(2, j, k); });
    if (o.ebc == 3 && o.ibe == 1)
      Kokkos::parallel_for("gettau_e13", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { t13(ni + 1, j, k) = t13(ni, j, k); });
    if (o.sbc == 3 && o.ibs == 1)
      Kokkos::parallel_for("gettau_s23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { t23(i, 1, k) = t23(i, 2, k); });
    if (o.nbc == 3 && o.ibn == 1)
      Kokkos::parallel_for("gettau_n23", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { t23(i, nj + 1, k) = t23(i, nj, k); });
  }

  // open boundary conditions
  if (o.wbc == 2 || o.ebc == 2 || o.sbc == 2 || o.nbc == 2) {
    const bool w2 = (o.wbc == 2 && o.ibw == 1), e2 = (o.ebc == 2 && o.ibe == 1);
    const bool s2 = (o.sbc == 2 && o.ibs == 1), n2 = (o.nbc == 2 && o.ibn == 1);
    if (w2) Kokkos::parallel_for("gettau_ow", box2(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t11(0, j, k) = t11(1, j, k); });
    if (e2) Kokkos::parallel_for("gettau_oe", box2(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t11(ni + 1, j, k) = t11(ni, j, k); });
    if (s2) Kokkos::parallel_for("gettau_os", box2(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t22(i, 0, k) = t22(i, 1, k); });
    if (n2) Kokkos::parallel_for("gettau_on", box2(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t22(i, nj + 1, k) = t22(i, nj, k); });
    if (w2) Kokkos::parallel_for("gettau_ow12", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t12(1, j, k) = t12(2, j, k); });
    if (e2) Kokkos::parallel_for("gettau_oe12", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { t12(ni + 1, j, k) = t12(ni, j, k); });
    if (s2) Kokkos::parallel_for("gettau_os12", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t12(i, 1, k) = t12(i, 2, k); });
    if (n2) Kokkos::parallel_for("gettau_on12", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { t12(i, nj + 1, k) = t12(i, nj, k); });
    const bool sw = s2 && w2, se = s2 && e2, nw = n2 && w2, ne = n2 && e2;
    if (sw || se || nw || ne)
      Kokkos::parallel_for("gettau_corners", Kokkos::RangePolicy<ExeSpace>(1, nk + 1), KOKKOS_LAMBDA(int k) {
        if (sw) t12(1, 1, k) = t12(2, 2, k);
        if (se) t12(ni + 1, 1, k) = t12(ni, 2, k);
        if (nw) t12(1, nj + 1, k) = t12(2, nj, k);
        if (ne) t12(ni + 1, nj + 1, k) = t12(ni, nj, k);
      });
  }

  // lower boundary (sfcstress: bbc=1 free slip, bbc=3 semi slip), then
  // upper (tbc=1: free slip)
  if (o.bbc == 3)
    sfc::sfcstress(b, rf, *sfc, t13, t23);
  else
    Kokkos::parallel_for("gettau_sfc", box2(1, ni + 1, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      t13(i, j, 1) = t13(i, j, 2);
      t23(i, j, 1) = t23(i, j, 2);
    });
  Kokkos::parallel_for("gettau_top", box2(1, ni + 1, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
    t13(i, j, nk + 1) = t13(i, j, nk);
    t23(i, j, nk + 1) = t23(i, j, nk);
  });
}

void sfc_and_turb(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
                  Real rtime, const FArray3& ua, const FArray3& va, const FArray3& wa,
                  const FArray3& ppi, const FArray3& tha, const FArray4& qa,
                  const FArray3& prs, const FArray3& rho, const FArray3& rf,
                  const FArray3& tkea, const FArray3& tke3d,
                  const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
                  const TurbState& ts, const Work& w, const sfc::SfcState* sfc) {
  o.check();
  if (o.cm1setup != 1) throw std::runtime_error("cm1k turb::sfc_and_turb: this is the LES path (cm1setup=1)");
  const auto& b = g.b;
  if (o.bbc == 3) {
    // surface layer (sfc::sfc_layer, called before) -> bc/comms block
    if (!sfc) throw std::runtime_error("cm1k turb::sfc_and_turb: bbc=3 needs the surface layer state");
    sfc_bcs(o, b, *sfc);
  }
  // facheck: ramp of the turbulence constants
  update_constants(o, b, rtime, ts);
  // dissten = 0 on w levels
  auto dissten = ts.dissten;
  Kokkos::parallel_for("turb_dissten0", box3(1, b.ni, 1, b.nj, 1, b.nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    dissten(i, j, k) = 0.0;
  });
  calcnm(o, c, b, g.rdz, g.c1, g.c2, g.mf, base.pi0, base.th0, ts.nm, w.d1, w.d2, w.d3, w.d6,
         prs, ppi, tha, qa);
  calcdef(o, g, rho, rf, ua, va, wa, ts.t11, ts.t12, ts.t13, ts.t22, ts.t23, ts.t33, ts.defv, ts.defh,
          w.d1, w.d2);
  tkekm(o, g, ts.nm, w.d1, w.d3, w.d4, w.d5, kmh, kmv, khh, khv, ts.cme, ts.ce1, ts.ce2,
        tkea, ts.lenscl, ts.dissten);
  tkebc(o, b, tkea, kmh, kmv, khh, khv, sfc);
  const int nk = b.nk;
  Kokkos::parallel_for("turb_tke3d_tb", box2(0, b.ni + 1, 0, b.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    tke3d(i, j, 1) = tkea(i, j, 1);
    tke3d(i, j, nk + 1) = tkea(i, j, nk + 1);
  });
  gettau(o, b, kmh, kmv, ts.t11, ts.t12, ts.t13, ts.t22, ts.t23, ts.t33, rf, sfc);
}

void sfc_bcs(const Options& o, const Bounds& b, const sfc::SfcState& s) {
  for (const FArray3* a : {&s.ust, &s.u1, &s.v1, &s.s1, &s.znt, &s.wspd}) bc2d_level(o, b, *a, 1);
  const auto zntmp = s.zntmp, znt = s.znt;
  Kokkos::parallel_for("sfc_zntmp", box2(0, b.ni + 1, 0, b.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    zntmp(i, j, 1) = znt(i, j, 1);
  });
}

//------------------------------------------------------------------------------
// solve1 pieces
//------------------------------------------------------------------------------

void tke_source(const Options& o, const Bounds& b, const FArray3& nm, const FArray3& defv,
                const FArray3& defh, const FArray3& dissten, const FArray3& kmh,
                const FArray3& kmv, const FArray3& khv, const FArray3& tketen) {
  if (b.nk < 2) return;
  const int tconfig = o.tconfig;
  Kokkos::parallel_for("tke_source", box3(1, b.ni, 1, b.nj, 2, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (tconfig == 1)
      tketen(i, j, k) = -(khv(i, j, k) * nm(i, j, k)) - dissten(i, j, k)
                        + kmv(i, j, k) * fmax2(0.0, (defv(i, j, k) + defh(i, j, k)));
    else
      tketen(i, j, k) = -(khv(i, j, k) * nm(i, j, k)) - dissten(i, j, k)
                        + kmv(i, j, k) * fmax2(0.0, defv(i, j, k))
                        + kmh(i, j, k) * fmax2(0.0, defh(i, j, k));
  });
}

void turbt(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& t, const FArray3& tten, const FArray3& kmh, const FArray3& kmv,
           const Work& w) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbt: nk >= 3 required");
  const Real rdx = g.rdx, rdy = g.rdy, rdz = g.rdz;
  const auto uh = g.uh;
  const auto uf = g.uf;
  const auto vh = g.vh;
  const auto vf = g.vf;
  const auto mh = g.mh;
  const auto mf = g.mf;
  const auto turbx = w.d1, turby = w.d2, turbz = w.d3, dumx = w.d4, dumy = w.d5, dumz = w.d6;

  if (b.terrain_flag) {
    // Cartesian with terrain
    const Real zt = g.zt;
    const auto rds = g.rds; const auto sigma = g.sigma;
    const auto gz = g.gz; const auto rgz = g.rgz; const auto rgzu = g.rgzu; const auto rgzv = g.rgzv;
    // turbz stores t at s-pts
    Kokkos::parallel_for("turbt_t_ts", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbz(i, j, k) = 0.5 * (t(i, j, k) + t(i, j, k + 1));
    });
    Kokkos::parallel_for("turbt_t_fx", box3(1, ni + 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      // note: K is multiplied by 2
      dumx(i, j, k) = -0.25 * (gz(i, j, 1) * rf(i, j, k) + gz(i - 1, j, 1) * rf(i - 1, j, k))
                      * 2.0 * (kmh(i, j, k) + kmh(i - 1, j, k)) * (
                          (t(i, j, k) * rgz(i, j, 1) - t(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                          + 0.5 * ((zt - sigma(k)) * (turbz(i - 1, j, k) + turbz(i, j, k))
                                   - (zt - sigma(k - 1)) * (turbz(i - 1, j, k - 1) + turbz(i, j, k - 1)))
                                * rds(k) * (rgz(i, j, 1) - rgz(i - 1, j, 1)) * rdx * uf(i));
    });
    Kokkos::parallel_for("turbt_t_fy", box3(1, ni, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumy(i, j, k) = -0.25 * (gz(i, j, 1) * rf(i, j, k) + gz(i, j - 1, 1) * rf(i, j - 1, k))
                      * 2.0 * (kmh(i, j, k) + kmh(i, j - 1, k)) * (
                          (t(i, j, k) * rgz(i, j, 1) - t(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                          + 0.5 * ((zt - sigma(k)) * (turbz(i, j - 1, k) + turbz(i, j, k))
                                   - (zt - sigma(k - 1)) * (turbz(i, j - 1, k - 1) + turbz(i, j, k - 1)))
                                * rds(k) * (rgz(i, j, 1) - rgz(i, j - 1, 1)) * rdy * vf(j));
    });
    Kokkos::parallel_for("turbt_t_tb", box2(1, ni + 1, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      dumx(i, j, 1) = 0.0;
      dumx(i, j, nk + 1) = 0.0;
      dumy(i, j, 1) = 0.0;
      dumy(i, j, nk + 1) = 0.0;
    });
    // turbz stores dumx at s-pts, dumz stores dumy at s-pts
    Kokkos::parallel_for("turbt_t_fs", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbz(i, j, k) = 0.5 * (dumx(i, j, k) + dumx(i, j, k + 1));
      dumz(i, j, k) = 0.5 * (dumy(i, j, k) + dumy(i, j, k + 1));
    });
    Kokkos::parallel_for("turbt_t_h", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = -(dumx(i + 1, j, k) * rgzu(i + 1, j, 1) - dumx(i, j, k) * rgzu(i, j, 1)) * gz(i, j, 1) * rdx * uh(i)
                       - 0.5 * ((zt - sigma(k)) * (turbz(i, j, k) + turbz(i + 1, j, k))
                                - (zt - sigma(k - 1)) * (turbz(i, j, k - 1) + turbz(i + 1, j, k - 1)))
                             * rds(k) * (rgzu(i + 1, j, 1) - rgzu(i, j, 1)) * gz(i, j, 1) * rdx * uh(i);
      turby(i, j, k) = -(dumy(i, j + 1, k) * rgzv(i, j + 1, 1) - dumy(i, j, k) * rgzv(i, j, 1)) * gz(i, j, 1) * rdy * vh(j)
                       - 0.5 * ((zt - sigma(k)) * (dumz(i, j, k) + dumz(i, j + 1, k))
                                - (zt - sigma(k - 1)) * (dumz(i, j, k - 1) + dumz(i, j + 1, k - 1)))
                             * rds(k) * (rgzv(i, j + 1, 1) - rgzv(i, j, 1)) * gz(i, j, 1) * rdy * vh(j);
    });
  } else {
  Kokkos::parallel_for("turbt_flux", box3(1, ni + 1, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    // note: K is multiplied by 2
    dumx(i, j, k) = -0.25 * (rf(i, j, k) + rf(i - 1, j, k))
                    * 2.0 * (kmh(i, j, k) + kmh(i - 1, j, k))
                    * (t(i, j, k) - t(i - 1, j, k)) * rdx * uf(i);
    dumy(i, j, k) = -0.25 * (rf(i, j, k) + rf(i, j - 1, k))
                    * 2.0 * (kmh(i, j, k) + kmh(i, j - 1, k))
                    * (t(i, j, k) - t(i, j - 1, k)) * rdy * vf(j);
  });
  if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbt_bw", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { dumx(1, j, k) = dumx(2, j, k); });
  if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbt_be", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { dumx(ni + 1, j, k) = dumx(ni, j, k); });
  if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbt_bs", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { dumy(i, 1, k) = dumy(i, 2, k); });
  if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbt_bn", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { dumy(i, nj + 1, k) = dumy(i, nj, k); });
  Kokkos::parallel_for("turbt_h", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    turbx(i, j, k) = -(dumx(i + 1, j, k) - dumx(i, j, k)) * rdx * uh(i);
    turby(i, j, k) = -(dumy(i, j + 1, k) - dumy(i, j, k)) * rdy * vh(j);
  });
  }  // terrain_flag
  if (o.wbc == 2 || o.ebc == 2 || o.sbc == 2 || o.nbc == 2) {
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbt_ow", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbt_oe", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbt_os", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbt_on", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj, k) = 0.0; });
  }

  if (o.doimpl == 0) {
    const int bbc = o.bbc;
    Kokkos::parallel_for("turbt_fz", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumz(i, j, k) = -(kmv(i, j, k) + kmv(i, j, k + 1)) * (t(i, j, k + 1) - t(i, j, k)) * rdz * mh(i, j, k) * rho(i, j, k);
      if (bbc == 3 && k == 1) dumz(i, j, 1) = 0.0;
    });
    Kokkos::parallel_for("turbt_ez", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbz(i, j, k) = -(dumz(i, j, k) - dumz(i, j, k - 1)) * rdz * mf(i, j, k);
      tten(i, j, k) = tten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) / rf(i, j, k);
    });
  } else {
    const Real rdt = 1.0 / dt;
    const Real tema = -1.0 * dt * vialpha * rdz * rdz;
    const Real temb = dt * vibeta * rdz * rdz;
    Kokkos::parallel_for("turbt_iz", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      {
        const int k = 2;
        const Real rrf = mf(i, j, k) / rf(i, j, k);
        const Real r2 = (kmv(i, j, k) + kmv(i, j, k + 1)) * mh(i, j, k) * rho(i, j, k) * rrf;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfc;
        const Real cfd = t(i, j, k) + temb * (r2 * t(i, j, k + 1) - r2 * t(i, j, k));
        const Real tem = 1.0 / cfb;
        dumx(i, j, k) = -cfc * tem;
        dumy(i, j, k) = cfd * tem;
      }
      for (int k = 3; k <= nk - 1; ++k) {
        const Real rrf = mf(i, j, k) / rf(i, j, k);
        const Real r1 = (kmv(i, j, k - 1) + kmv(i, j, k)) * mh(i, j, k - 1) * rho(i, j, k - 1) * rrf;
        const Real r2 = (kmv(i, j, k) + kmv(i, j, k + 1)) * mh(i, j, k) * rho(i, j, k) * rrf;
        const Real cfa = tema * r1;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfa - cfc;
        const Real cfd = t(i, j, k) + temb * (r2 * t(i, j, k + 1) - (r1 + r2) * t(i, j, k) + r1 * t(i, j, k - 1));
        const Real tem = 1.0 / (cfa * dumx(i, j, k - 1) + cfb);
        dumx(i, j, k) = -cfc * tem;
        dumy(i, j, k) = (cfd - cfa * dumy(i, j, k - 1)) * tem;
      }
      {
        const int k = nk;
        const Real rrf = mf(i, j, k) / rf(i, j, k);
        const Real r1 = (kmv(i, j, k - 1) + kmv(i, j, k)) * mh(i, j, k - 1) * rho(i, j, k - 1) * rrf;
        const Real cfa = tema * r1;
        const Real cfb = 1.0 - cfa;
        const Real cfd = t(i, j, k) + temb * (-r1 * t(i, j, k) + r1 * t(i, j, k - 1));
        const Real tem = 1.0 / (cfa * dumx(i, j, k - 1) + cfb);
        dumz(i, j, k) = (cfd - cfa * dumy(i, j, k - 1)) * tem;
        turbz(i, j, k) = rf(i, j, k) * (dumz(i, j, k) - t(i, j, k)) * rdt;
        tten(i, j, k) = tten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) / rf(i, j, k);
      }
      for (int k = nk - 1; k >= 2; --k) {
        dumz(i, j, k) = dumx(i, j, k) * dumz(i, j, k + 1) + dumy(i, j, k);
        turbz(i, j, k) = rf(i, j, k) * (dumz(i, j, k) - t(i, j, k)) * rdt;
        tten(i, j, k) = tten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) / rf(i, j, k);
      }
    });
  }
}

void dissheat(const Options& o, const Constants& c, const Bounds& b, const FArray3& pi0,
              const FArray3& ppi, const FArray4& qa, const FArray3& dissten,
              const FArray3& thten1, const FArray3& cvm, const FArray3& zf,
              const sfc::SfcState* sfc) {
  // cvm (solve1: dum1): needed for eqtset=2, imoist=1
  const bool moist2 = (o.eqtset == 2 && o.imoist == 1 && o.idiss == 1);
  {
    const int nql1 = o.nql1, nql2 = o.nql2, nqs1 = o.nqs1, nqs2 = o.nqs2, iice = o.iice, nqv = o.nqv;
    const Real cv = c.cv, cvv = c.cvv, cpl = c.cpl, cpi = c.cpi;
    Kokkos::parallel_for("turb_cvm", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (moist2) {
        Real ql = 0.0, qi = 0.0;
        if (nql1 >= 1) { ql = qa(i, j, k, nql1); for (int n = nql1 + 1; n <= nql2; ++n) ql = ql + qa(i, j, k, n); }
        if (iice == 1) { qi = qa(i, j, k, nqs1); for (int n = nqs1 + 1; n <= nqs2; ++n) qi = qi + qa(i, j, k, n); }
        cvm(i, j, k) = cv + cvv * qa(i, j, k, nqv) + cpl * ql + cpi * qi;
      } else {
        cvm(i, j, k) = cv;
      }
    });
  }
  if (o.idiss != 1) return;
  if (o.bbc == 3 && (!sfc || !zf.is_allocated()))
    throw std::runtime_error("cm1k turb::dissheat: bbc=3 needs zf and the surface layer state");
  const Real cpdcv = c.cpdcv, cp = c.cp;
  // note: dissten stores epsilon (dissipation rate) at w points
  const int k1 = (o.bbc == 3) ? 2 : 1;
  Kokkos::parallel_for("turb_dissheat", box3(1, b.ni, 1, b.nj, k1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real epsd = 0.5 * (dissten(i, j, k) + dissten(i, j, k + 1));
    Real dheat;
    if (moist2) dheat = epsd / (cpdcv * cvm(i, j, k) * (pi0(i, j, k) + ppi(i, j, k)));
    else        dheat = epsd / (cp * (pi0(i, j, k) + ppi(i, j, k)));
    thten1(i, j, k) = thten1(i, j, k) + dheat;
  });
  if (o.bbc == 3) {
    // surface layer: epsilon from u* and z0 at k=1
    const auto ust = sfc->ust, zntmp = sfc->zntmp;
    Kokkos::parallel_for("turb_dissheat_sfc", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
      constexpr Real karman = phys::karman;
      const int k = 1;
      const Real dz1 = zf(i, j, 2) - zf(i, j, 1);
      const Real epsd = (ust(i, j, 1) * ust(i, j, 1) * ust(i, j, 1))
                        * Kokkos::log((dz1 + zntmp(i, j, 1)) / zntmp(i, j, 1)) / (karman * dz1);
      Real dheat;
      if (moist2) dheat = epsd / (cpdcv * cvm(i, j, k) * (pi0(i, j, k) + ppi(i, j, k)));
      else        dheat = epsd / (cp * (pi0(i, j, k) + ppi(i, j, k)));
      thten1(i, j, k) = thten1(i, j, k) + dheat;
    });
  }
}

void turbuz(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
            const FArray3& u, const FArray3& w, const FArray3& t13, const FArray3& kmv,
            const FArray3& turbz, const FArray3& dum1, const FArray3& dum2, const FArray3& dum3,
            const FArray3& dum7,
            const FArray3& dum9) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbuz: nk >= 3 required");
  const Real rdx = g.rdx, rdz = g.rdz;
  const auto uf = g.uf;
  const auto mh = g.mh;
  const auto mf = g.mf;
  if (o.doimpl == 0) {
    const Real tem = rdz * 0.5;
    Kokkos::parallel_for("turbuz_e", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbz(i, j, k) = (t13(i, j, k + 1) - t13(i, j, k)) * tem * (mh(i - 1, j, k) + mh(i, j, k));
    });
  } else {
    const Real rdt = 0.5 / dt;
    const Real tema = -0.0625 * dt * vialpha * rdz * rdz;
    const Real temb = 0.0625 * dt * vibeta * rdz * rdz;
    const Real temc = 0.5 * dt * rdz;
    const bool terrain = b.terrain_flag;
    if (terrain && !dum9.is_allocated())
      throw std::runtime_error("cm1k turb::turbuz: terrain needs the dum9 scratch array");
    const Real zt = g.zt;
    const auto rds = g.rds; const auto sigma = g.sigma; const auto gz = g.gz; const auto rgz = g.rgz;
    if (terrain)
      // dum9 stores w at scalar-pts
      Kokkos::parallel_for("turbuz_ws", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dum9(i, j, k) = 0.5 * (w(i, j, k) + w(i, j, k + 1));
      });
    Kokkos::parallel_for("turbuz_i", box2(1, ni + 1, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      {
        const int k = 1;
        const Real rru0 = 1.0 / (0.5 * (rho(i - 1, j, k) + rho(i, j, k)));
        Real tem = (mh(i - 1, j, k) + mh(i, j, k)) * rru0;
        const Real r2 = (kmv(i - 1, j, k + 1) + kmv(i, j, k + 1)) * (mf(i - 1, j, k + 1) + mf(i, j, k + 1))
                        * (rf(i - 1, j, k + 1) + rf(i, j, k + 1)) * tem;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfc;
        const Real cfd = u(i, j, k) + temb * (r2 * u(i, j, k + 1) - r2 * u(i, j, k))
                         - temc * t13(i, j, 1) * (mh(i - 1, j, 1) + mh(i, j, 1)) * rru0;
        tem = 1.0 / cfb;
        dum1(i, j, 1) = -cfc * tem;
        dum2(i, j, 1) = cfd * tem;
        dum7(i, j, 1) = 0.0;
      }
      for (int k = 2; k <= nk - 1; ++k) {
        const Real rru0 = 1.0 / (0.5 * (rho(i - 1, j, k) + rho(i, j, k)));
        Real tem = (mh(i - 1, j, k) + mh(i, j, k)) * rru0;
        const Real r1 = (kmv(i - 1, j, k) + kmv(i, j, k)) * (mf(i - 1, j, k) + mf(i, j, k))
                        * (rf(i - 1, j, k) + rf(i, j, k)) * tem;
        const Real r2 = (kmv(i - 1, j, k + 1) + kmv(i, j, k + 1)) * (mf(i - 1, j, k + 1) + mf(i, j, k + 1))
                        * (rf(i - 1, j, k + 1) + rf(i, j, k + 1)) * tem;
        const Real cfa = tema * r1;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfa - cfc;
        const Real cfd = u(i, j, k) + temb * (r2 * u(i, j, k + 1) - (r1 + r2) * u(i, j, k) + r1 * u(i, j, k - 1));
        tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dum1(i, j, k) = -cfc * tem;
        dum2(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        // explicit piece ... dwdx term
        if (terrain) {
          const Real d7 = (w(i, j, k) * rgz(i, j, 1) - w(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                          + 0.5 * rds(k) * ((zt - sigma(k)) * (dum9(i, j, k) + dum9(i - 1, j, k))
                                            - (zt - sigma(k - 1)) * (dum9(i, j, k - 1) + dum9(i - 1, j, k - 1)))
                                * (rgz(i, j, 1) - rgz(i - 1, j, 1)) * rdx * uf(i);
          dum7(i, j, k) = d7 * 0.25 * (kmv(i - 1, j, k) + kmv(i, j, k))
                          * (gz(i - 1, j, 1) * rf(i - 1, j, k) + gz(i, j, 1) * rf(i, j, k));
        } else {
          dum7(i, j, k) = (w(i, j, k) - w(i - 1, j, k)) * rdx * uf(i)
                          * 0.25 * (kmv(i - 1, j, k) + kmv(i, j, k))
                          * (rf(i - 1, j, k) + rf(i, j, k));
        }
      }
      {
        const int k = nk;
        const Real rru0 = 1.0 / (0.5 * (rho(i - 1, j, k) + rho(i, j, k)));
        Real tem = (mh(i - 1, j, k) + mh(i, j, k)) * rru0;
        const Real r1 = (kmv(i - 1, j, k) + kmv(i, j, k)) * (mf(i - 1, j, k) + mf(i, j, k))
                        * (rf(i - 1, j, k) + rf(i, j, k)) * tem;
        const Real cfa = tema * r1;
        const Real cfb = 1.0 - cfa;
        const Real cfd = u(i, j, k) + temb * (-r1 * u(i, j, k) + r1 * u(i, j, k - 1))
                         + temc * t13(i, j, nk + 1) * (mh(i - 1, j, nk) + mh(i, j, nk)) * rru0;
        tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dum3(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        turbz(i, j, k) = (rho(i - 1, j, k) + rho(i, j, k)) * (dum3(i, j, k) - u(i, j, k)) * rdt;
        if (terrain) {
          const Real d7 = (w(i, j, k) * rgz(i, j, 1) - w(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                          + 0.5 * rds(k) * ((zt - sigma(k)) * (dum9(i, j, k) + dum9(i - 1, j, k))
                                            - (zt - sigma(k - 1)) * (dum9(i, j, k - 1) + dum9(i - 1, j, k - 1)))
                                * (rgz(i, j, 1) - rgz(i - 1, j, 1)) * rdx * uf(i);
          dum7(i, j, k) = d7 * 0.25 * (kmv(i - 1, j, k) + kmv(i, j, k))
                          * (gz(i - 1, j, 1) * rf(i - 1, j, k) + gz(i, j, 1) * rf(i, j, k));
        } else {
          dum7(i, j, k) = (w(i, j, k) - w(i - 1, j, k)) * rdx * uf(i)
                          * 0.25 * (kmv(i - 1, j, k) + kmv(i, j, k))
                          * (rf(i - 1, j, k) + rf(i, j, k));
        }
        turbz(i, j, k) = turbz(i, j, k) + (0.0 - dum7(i, j, k)) * rdz * 0.5 * (mh(i - 1, j, k) + mh(i, j, k));
      }
      for (int k = nk - 1; k >= 1; --k) {
        dum3(i, j, k) = dum1(i, j, k) * dum3(i, j, k + 1) + dum2(i, j, k);
        turbz(i, j, k) = (rho(i - 1, j, k) + rho(i, j, k)) * (dum3(i, j, k) - u(i, j, k)) * rdt;
        turbz(i, j, k) = turbz(i, j, k) + (dum7(i, j, k + 1) - dum7(i, j, k)) * rdz * 0.5 * (mh(i - 1, j, k) + mh(i, j, k));
      }
    });
  }
}

void turbu(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& u, const FArray3& uten, const FArray3& w,
           const FArray3& t11, const FArray3& t12, const FArray3& t13, const FArray3& kmv,
           const Work& wk) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbu: nk >= 3 required");
  const Real rdx = g.rdx, rdy = g.rdy;
  const auto uf = g.uf;
  const auto vh = g.vh;
  const auto turbx = wk.d1, turby = wk.d2, turbz = wk.d3, dum1 = wk.d4, dum2 = wk.d5, dum3 = wk.d6, dum7 = wk.d7;

  if (o.dohturb) {
    if (b.terrain_flag) {
      const Real zt = g.zt;
      const Real cgs1 = g.cgs1, cgs2 = g.cgs2, cgs3 = g.cgs3, cgt1 = g.cgt1, cgt2 = g.cgt2, cgt3 = g.cgt3;
      const auto rds = g.rds; const auto sigma = g.sigma; const auto rdsf = g.rdsf; const auto sigmaf = g.sigmaf;
      const auto gzu = g.gzu; const auto rgz = g.rgz; const auto gxu = g.gxu;
      // dum1 stores t11 at w-pts, dum2 stores t12 at w-pts
      Kokkos::parallel_for("turbu_t_w", box3(0, ni + 1, 1, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
        if (k == 1) {
          dum1(i, j, 1) = cgs1 * t11(i, j, 1) + cgs2 * t11(i, j, 2) + cgs3 * t11(i, j, 3);
          dum2(i, j, 1) = cgs1 * t12(i, j, 1) + cgs2 * t12(i, j, 2) + cgs3 * t12(i, j, 3);
        } else if (k == nk + 1) {
          dum1(i, j, nk + 1) = cgt1 * t11(i, j, nk) + cgt2 * t11(i, j, nk - 1) + cgt3 * t11(i, j, nk - 2);
          dum2(i, j, nk + 1) = cgt1 * t12(i, j, nk) + cgt2 * t12(i, j, nk - 1) + cgt3 * t12(i, j, nk - 2);
        } else {
          const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
          const Real r1 = 1.0 - r2;
          dum1(i, j, k) = r1 * t11(i, j, k - 1) + r2 * t11(i, j, k);
          dum2(i, j, k) = r1 * t12(i, j, k - 1) + r2 * t12(i, j, k);
        }
      });
      Kokkos::parallel_for("turbu_t_h", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        turbx(i, j, k) = gzu(i, j, 1) * (t11(i, j, k) * rgz(i, j, 1) - t11(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                         + 0.5 * (gxu(i, j, k + 1) * (dum1(i - 1, j, k + 1) + dum1(i, j, k + 1))
                                  - gxu(i, j, k) * (dum1(i - 1, j, k) + dum1(i, j, k))) * rdsf(k);
        const Real r1 = 0.25 * ((rgz(i - 1, j - 1, 1) + rgz(i, j, 1)) + (rgz(i - 1, j, 1) + rgz(i, j - 1, 1)));
        const Real r2 = 0.25 * ((rgz(i - 1, j + 1, 1) + rgz(i, j, 1)) + (rgz(i - 1, j, 1) + rgz(i, j + 1, 1)));
        turby(i, j, k) = gzu(i, j, 1) * (t12(i, j + 1, k) * r2 - t12(i, j, k) * r1) * rdy * vh(j)
                         + 0.5 * ((zt - sigmaf(k + 1)) * (dum2(i, j, k + 1) + dum2(i, j + 1, k + 1))
                                  - (zt - sigmaf(k)) * (dum2(i, j, k) + dum2(i, j + 1, k)))
                               * gzu(i, j, 1) * (r2 - r1) * rdy * vh(j) * rdsf(k);
      });
    } else
    Kokkos::parallel_for("turbu_h", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = (t11(i, j, k) - t11(i - 1, j, k)) * rdx * uf(i);
      turby(i, j, k) = (t12(i, j + 1, k) - t12(i, j, k)) * rdy * vh(j);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbu_ow", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbu_oe", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni + 1, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbu_os", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbu_on", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj, k) = 0.0; });
  } else {
    Kokkos::parallel_for("turbu_h0", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = 0.0;
      turby(i, j, k) = 0.0;
    });
  }

  // turbuz (z-direction)
  if (o.dovturb) {
    turbuz(o, g, dt, rho, rf, u, w, t13, kmv, turbz, dum1, dum2, dum3, dum7, wk.d9);
  } else {
    Kokkos::parallel_for("turbu_z0", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { turbz(i, j, k) = 0.0; });
  }

  Kokkos::parallel_for("turbu_ten", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real rru0 = 1.0 / (0.5 * (rho(i - 1, j, k) + rho(i, j, k)));
    uten(i, j, k) = uten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) * rru0;
  });
}

void turbvz(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
            const FArray3& v, const FArray3& w, const FArray3& t23, const FArray3& kmv,
            const FArray3& turbz, const FArray3& dum1, const FArray3& dum2, const FArray3& dum3,
            const FArray3& dum7,
            const FArray3& dum9) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbvz: nk >= 3 required");
  const Real rdy = g.rdy, rdz = g.rdz;
  const auto vf = g.vf;
  const auto mh = g.mh;
  const auto mf = g.mf;
  if (o.doimpl == 0) {
    const Real tem = rdz * 0.5;
    Kokkos::parallel_for("turbvz_e", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbz(i, j, k) = (t23(i, j, k + 1) - t23(i, j, k)) * tem * (mh(i, j - 1, k) + mh(i, j, k));
    });
  } else {
    const Real rdt = 0.5 / dt;
    const Real tema = -0.0625 * dt * vialpha * rdz * rdz;
    const Real temb = 0.0625 * dt * vibeta * rdz * rdz;
    const Real temc = 0.5 * dt * rdz;
    const bool terrain = b.terrain_flag;
    if (terrain && !dum9.is_allocated())
      throw std::runtime_error("cm1k turb::turbvz: terrain needs the dum9 scratch array");
    const Real zt = g.zt;
    const auto rds = g.rds; const auto sigma = g.sigma; const auto gz = g.gz; const auto rgz = g.rgz;
    if (terrain)
      // dum9 stores w at scalar-pts
      Kokkos::parallel_for("turbvz_ws", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dum9(i, j, k) = 0.5 * (w(i, j, k) + w(i, j, k + 1));
      });
    Kokkos::parallel_for("turbvz_i", box2(1, ni, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      {
        const int k = 1;
        const Real rrv0 = 1.0 / (0.5 * (rho(i, j - 1, k) + rho(i, j, k)));
        Real tem = (mh(i, j - 1, k) + mh(i, j, k)) * rrv0;
        const Real r2 = (kmv(i, j - 1, k + 1) + kmv(i, j, k + 1)) * (mf(i, j - 1, k + 1) + mf(i, j, k + 1))
                        * (rf(i, j - 1, k + 1) + rf(i, j, k + 1)) * tem;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfc;
        const Real cfd = v(i, j, k) + temb * (r2 * v(i, j, k + 1) - r2 * v(i, j, k))
                         - temc * t23(i, j, 1) * (mh(i, j - 1, 1) + mh(i, j, 1)) * rrv0;
        tem = 1.0 / cfb;
        dum1(i, j, 1) = -cfc * tem;
        dum2(i, j, 1) = cfd * tem;
        dum7(i, j, 1) = 0.0;
      }
      for (int k = 2; k <= nk - 1; ++k) {
        const Real rrv0 = 1.0 / (0.5 * (rho(i, j - 1, k) + rho(i, j, k)));
        Real tem = (mh(i, j - 1, k) + mh(i, j, k)) * rrv0;
        const Real r1 = (kmv(i, j - 1, k) + kmv(i, j, k)) * (mf(i, j - 1, k) + mf(i, j, k))
                        * (rf(i, j - 1, k) + rf(i, j, k)) * tem;
        const Real r2 = (kmv(i, j - 1, k + 1) + kmv(i, j, k + 1)) * (mf(i, j - 1, k + 1) + mf(i, j, k + 1))
                        * (rf(i, j - 1, k + 1) + rf(i, j, k + 1)) * tem;
        const Real cfa = tema * r1;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfa - cfc;
        const Real cfd = v(i, j, k) + temb * (r2 * v(i, j, k + 1) - (r1 + r2) * v(i, j, k) + r1 * v(i, j, k - 1));
        tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dum1(i, j, k) = -cfc * tem;
        dum2(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        // explicit piece ... dwdy term
        if (terrain) {
          const Real d7 = (w(i, j, k) * rgz(i, j, 1) - w(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                          + 0.5 * rds(k) * ((zt - sigma(k)) * (dum9(i, j, k) + dum9(i, j - 1, k))
                                            - (zt - sigma(k - 1)) * (dum9(i, j, k - 1) + dum9(i, j - 1, k - 1)))
                                * (rgz(i, j, 1) - rgz(i, j - 1, 1)) * rdy * vf(j);
          dum7(i, j, k) = d7 * 0.25 * (kmv(i, j - 1, k) + kmv(i, j, k))
                          * (gz(i, j - 1, 1) * rf(i, j - 1, k) + gz(i, j, 1) * rf(i, j, k));
        } else {
          dum7(i, j, k) = (w(i, j, k) - w(i, j - 1, k)) * rdy * vf(j)
                          * 0.25 * (kmv(i, j - 1, k) + kmv(i, j, k))
                          * (rf(i, j - 1, k) + rf(i, j, k));
        }
      }
      {
        const int k = nk;
        const Real rrv0 = 1.0 / (0.5 * (rho(i, j - 1, k) + rho(i, j, k)));
        Real tem = (mh(i, j - 1, k) + mh(i, j, k)) * rrv0;
        const Real r1 = (kmv(i, j - 1, k) + kmv(i, j, k)) * (mf(i, j - 1, k) + mf(i, j, k))
                        * (rf(i, j - 1, k) + rf(i, j, k)) * tem;
        const Real cfa = tema * r1;
        const Real cfb = 1.0 - cfa;
        const Real cfd = v(i, j, k) + temb * (-r1 * v(i, j, k) + r1 * v(i, j, k - 1))
                         + temc * t23(i, j, nk + 1) * (mh(i, j - 1, nk) + mh(i, j, nk)) * rrv0;
        tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dum3(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        turbz(i, j, k) = (rho(i, j - 1, k) + rho(i, j, k)) * (dum3(i, j, k) - v(i, j, k)) * rdt;
        if (terrain) {
          const Real d7 = (w(i, j, k) * rgz(i, j, 1) - w(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                          + 0.5 * rds(k) * ((zt - sigma(k)) * (dum9(i, j, k) + dum9(i, j - 1, k))
                                            - (zt - sigma(k - 1)) * (dum9(i, j, k - 1) + dum9(i, j - 1, k - 1)))
                                * (rgz(i, j, 1) - rgz(i, j - 1, 1)) * rdy * vf(j);
          dum7(i, j, k) = d7 * 0.25 * (kmv(i, j - 1, k) + kmv(i, j, k))
                          * (gz(i, j - 1, 1) * rf(i, j - 1, k) + gz(i, j, 1) * rf(i, j, k));
        } else {
          dum7(i, j, k) = (w(i, j, k) - w(i, j - 1, k)) * rdy * vf(j)
                          * 0.25 * (kmv(i, j - 1, k) + kmv(i, j, k))
                          * (rf(i, j - 1, k) + rf(i, j, k));
        }
        turbz(i, j, k) = turbz(i, j, k) + (0.0 - dum7(i, j, k)) * rdz * 0.5 * (mh(i, j - 1, k) + mh(i, j, k));
      }
      for (int k = nk - 1; k >= 1; --k) {
        dum3(i, j, k) = dum1(i, j, k) * dum3(i, j, k + 1) + dum2(i, j, k);
        turbz(i, j, k) = (rho(i, j - 1, k) + rho(i, j, k)) * (dum3(i, j, k) - v(i, j, k)) * rdt;
        turbz(i, j, k) = turbz(i, j, k) + (dum7(i, j, k + 1) - dum7(i, j, k)) * rdz * 0.5 * (mh(i, j - 1, k) + mh(i, j, k));
      }
    });
  }
}

void turbv(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& v, const FArray3& vten, const FArray3& w,
           const FArray3& t12, const FArray3& t22, const FArray3& t23, const FArray3& kmv,
           const Work& wk) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbv: nk >= 3 required");
  const Real rdx = g.rdx, rdy = g.rdy;
  const auto uh = g.uh;
  const auto vf = g.vf;
  const auto turbx = wk.d1, turby = wk.d2, turbz = wk.d3, dum1 = wk.d4, dum2 = wk.d5, dum3 = wk.d6, dum7 = wk.d7;

  if (o.dohturb) {
    if (b.terrain_flag) {
      const Real zt = g.zt;
      const Real cgs1 = g.cgs1, cgs2 = g.cgs2, cgs3 = g.cgs3, cgt1 = g.cgt1, cgt2 = g.cgt2, cgt3 = g.cgt3;
      const auto rds = g.rds; const auto sigma = g.sigma; const auto rdsf = g.rdsf; const auto sigmaf = g.sigmaf;
      const auto gzv = g.gzv; const auto rgz = g.rgz; const auto gyv = g.gyv;
      // dum1 stores t12 at w-pts, dum2 stores t22 at w-pts
      Kokkos::parallel_for("turbv_t_w", box3(1, ni + 1, 0, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
        if (k == 1) {
          dum1(i, j, 1) = cgs1 * t12(i, j, 1) + cgs2 * t12(i, j, 2) + cgs3 * t12(i, j, 3);
          dum2(i, j, 1) = cgs1 * t22(i, j, 1) + cgs2 * t22(i, j, 2) + cgs3 * t22(i, j, 3);
        } else if (k == nk + 1) {
          dum1(i, j, nk + 1) = cgt1 * t12(i, j, nk) + cgt2 * t12(i, j, nk - 1) + cgt3 * t12(i, j, nk - 2);
          dum2(i, j, nk + 1) = cgt1 * t22(i, j, nk) + cgt2 * t22(i, j, nk - 1) + cgt3 * t22(i, j, nk - 2);
        } else {
          const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
          const Real r1 = 1.0 - r2;
          dum1(i, j, k) = r1 * t12(i, j, k - 1) + r2 * t12(i, j, k);
          dum2(i, j, k) = r1 * t22(i, j, k - 1) + r2 * t22(i, j, k);
        }
      });
      Kokkos::parallel_for("turbv_t_h", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        const Real r1 = 0.25 * ((rgz(i - 1, j - 1, 1) + rgz(i, j, 1)) + (rgz(i - 1, j, 1) + rgz(i, j - 1, 1)));
        const Real r2 = 0.25 * ((rgz(i + 1, j - 1, 1) + rgz(i, j, 1)) + (rgz(i + 1, j, 1) + rgz(i, j - 1, 1)));
        turbx(i, j, k) = gzv(i, j, 1) * (t12(i + 1, j, k) * r2 - t12(i, j, k) * r1) * rdx * uh(i)
                         + 0.5 * ((zt - sigmaf(k + 1)) * (dum1(i, j, k + 1) + dum1(i + 1, j, k + 1))
                                  - (zt - sigmaf(k)) * (dum1(i, j, k) + dum1(i + 1, j, k)))
                               * gzv(i, j, 1) * (r2 - r1) * rdx * uh(i) * rdsf(k);
        turby(i, j, k) = gzv(i, j, 1) * (t22(i, j, k) * rgz(i, j, 1) - t22(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                         + 0.5 * (gyv(i, j, k + 1) * (dum2(i, j - 1, k + 1) + dum2(i, j, k + 1))
                                  - gyv(i, j, k) * (dum2(i, j - 1, k) + dum2(i, j, k))) * rdsf(k);
      });
    } else
    Kokkos::parallel_for("turbv_h", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = (t12(i + 1, j, k) - t12(i, j, k)) * rdx * uh(i);
      turby(i, j, k) = (t22(i, j, k) - t22(i, j - 1, k)) * rdy * vf(j);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbv_ow", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbv_oe", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbv_os", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbv_on", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj + 1, k) = 0.0; });
  } else {
    Kokkos::parallel_for("turbv_h0", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = 0.0;
      turby(i, j, k) = 0.0;
    });
  }

  // turbvz (z-direction)
  if (o.dovturb) {
    turbvz(o, g, dt, rho, rf, v, w, t23, kmv, turbz, dum1, dum2, dum3, dum7, wk.d9);
  } else {
    Kokkos::parallel_for("turbv_z0", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { turbz(i, j, k) = 0.0; });
  }

  Kokkos::parallel_for("turbv_ten", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real rrv0 = 1.0 / (0.5 * (rho(i, j - 1, k) + rho(i, j, k)));
    vten(i, j, k) = vten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) * rrv0;
  });
}

void turbw(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& w, const FArray3& wten, const FArray3& t13, const FArray3& t23,
           const FArray3& t33, const FArray3& kmh, const Work& wk) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 3) throw std::runtime_error("cm1k turb::turbw: nk >= 3 required");
  const Real rdx = g.rdx, rdy = g.rdy, rdz = g.rdz;
  const auto uh = g.uh;
  const auto vh = g.vh;
  const auto mh = g.mh;
  const auto mf = g.mf;
  const auto turbx = wk.d1, turby = wk.d2, turbz = wk.d3, dum1 = wk.d4, dum2 = wk.d5, dum3 = wk.d6;

  if (o.dohturb) {
    if (b.terrain_flag) {
      const Real zt = g.zt;
      const auto rds = g.rds; const auto sigma = g.sigma;
      const auto gz = g.gz; const auto rgzu = g.rgzu; const auto rgzv = g.rgzv;
      Kokkos::parallel_for("turbw_t_s", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dum1(i, j, k) = 0.25 * ((t13(i, j, k + 1) + t13(i + 1, j, k + 1)) + (t13(i, j, k) + t13(i + 1, j, k)));
        dum2(i, j, k) = 0.25 * ((t23(i, j, k + 1) + t23(i, j + 1, k + 1)) + (t23(i, j, k) + t23(i, j + 1, k)));
      });
      Kokkos::parallel_for("turbw_t_h", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        turbx(i, j, k) = gz(i, j, 1) * (t13(i + 1, j, k) * rgzu(i + 1, j, 1) - t13(i, j, k) * rgzu(i, j, 1)) * rdx * uh(i)
                         + ((zt - sigma(k)) * dum1(i, j, k) - (zt - sigma(k - 1)) * dum1(i, j, k - 1))
                               * gz(i, j, 1) * (rgzu(i + 1, j, 1) - rgzu(i, j, 1)) * rdx * uh(i) * rds(k);
        turby(i, j, k) = gz(i, j, 1) * (t23(i, j + 1, k) * rgzv(i, j + 1, 1) - t23(i, j, k) * rgzv(i, j, 1)) * rdy * vh(j)
                         + ((zt - sigma(k)) * dum2(i, j, k) - (zt - sigma(k - 1)) * dum2(i, j, k - 1))
                               * gz(i, j, 1) * (rgzv(i, j + 1, 1) - rgzv(i, j, 1)) * rdy * vh(j) * rds(k);
      });
    } else
    Kokkos::parallel_for("turbw_h", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = (t13(i + 1, j, k) - t13(i, j, k)) * rdx * uh(i);
      turby(i, j, k) = (t23(i, j + 1, k) - t23(i, j, k)) * rdy * vh(j);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbw_ow", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbw_oe", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbw_os", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbw_on", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj, k) = 0.0; });
  } else {
    Kokkos::parallel_for("turbw_h0", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = 0.0;
      turby(i, j, k) = 0.0;
    });
  }

  // z-direction (also for the simple PBL scheme, cm1setup=2 with ipbl=2)
  if (o.dovturb || (o.cm1setup == 2 && o.ipbl == 2)) {
    if (o.doimpl == 0) {
      Kokkos::parallel_for("turbw_e", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        turbz(i, j, k) = (t33(i, j, k) - t33(i, j, k - 1)) * rdz * mf(i, j, k);
      });
    } else {
      const Real rdt = 1.0 / dt;
      const Real tema = -1.0 * dt * vialpha * rdz * rdz;
      const Real temb = dt * vibeta * rdz * rdz;
      const Real temc = dt * rdz;
      Kokkos::parallel_for("turbw_i", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
        {
          const int k = 2;
          const Real rrf = mf(i, j, k) / rf(i, j, k);
          const Real r2 = (kmh(i, j, k) + kmh(i, j, k + 1)) * mh(i, j, k) * rho(i, j, k) * rrf;
          const Real cfc = tema * r2;
          const Real cfb = 1.0 - cfc;
          const Real cfd = w(i, j, k) + temb * (r2 * w(i, j, k + 1) - r2 * w(i, j, k))
                           - temc * t33(i, j, k - 1) * rrf;
          const Real tem = 1.0 / cfb;
          dum1(i, j, k) = -cfc * tem;
          dum2(i, j, k) = cfd * tem;
        }
        for (int k = 3; k <= nk - 1; ++k) {
          const Real rrf = mf(i, j, k) / rf(i, j, k);
          const Real r1 = (kmh(i, j, k - 1) + kmh(i, j, k)) * mh(i, j, k - 1) * rho(i, j, k - 1) * rrf;
          const Real r2 = (kmh(i, j, k) + kmh(i, j, k + 1)) * mh(i, j, k) * rho(i, j, k) * rrf;
          const Real cfa = tema * r1;
          const Real cfc = tema * r2;
          const Real cfb = 1.0 - cfa - cfc;
          const Real cfd = w(i, j, k) + temb * (r2 * w(i, j, k + 1) - (r1 + r2) * w(i, j, k) + r1 * w(i, j, k - 1));
          const Real tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
          dum1(i, j, k) = -cfc * tem;
          dum2(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        }
        {
          const int k = nk;
          const Real rrf = mf(i, j, k) / rf(i, j, k);
          const Real r1 = (kmh(i, j, k - 1) + kmh(i, j, k)) * mh(i, j, k - 1) * rho(i, j, k - 1) * rrf;
          const Real cfa = tema * r1;
          const Real cfb = 1.0 - cfa;
          const Real cfd = w(i, j, k) + temb * (-r1 * w(i, j, k) + r1 * w(i, j, k - 1))
                           + temc * t33(i, j, k) * rrf;
          const Real tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
          dum3(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
          turbz(i, j, k) = rf(i, j, k) * (dum3(i, j, k) - w(i, j, k)) * rdt;
        }
        for (int k = nk - 1; k >= 2; --k) {
          dum3(i, j, k) = dum1(i, j, k) * dum3(i, j, k + 1) + dum2(i, j, k);
          turbz(i, j, k) = rf(i, j, k) * (dum3(i, j, k) - w(i, j, k)) * rdt;
        }
      });
    }
  } else {
    Kokkos::parallel_for("turbw_z0", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) { turbz(i, j, k) = 0.0; });
  }

  Kokkos::parallel_for("turbw_ten", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    wten(i, j, k) = wten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) / rf(i, j, k);
  });
}

void vimpl_coefs(const Bounds& b, const FArray3& khv, const FArray3& mf, const FArray3& rf,
                 const FArray3& mh, const FArray3& rr, const FArray3& dum7, const FArray3& dum8) {
  Kokkos::parallel_for("turb_vimpl", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum7(i, j, k) = khv(i, j, k) * mf(i, j, k) * rf(i, j, k) * mh(i, j, k) * rr(i, j, k);
    dum8(i, j, k) = khv(i, j, k + 1) * mf(i, j, k + 1) * rf(i, j, k + 1) * mh(i, j, k) * rr(i, j, k);
  });
}

void turbsz(const Options& o, const Grid& g, int iflux, Real dt, const FArray3& rho,
            const FArray3& rr, const FArray3& rf, const FArray3& s, const FArray3& khv,
            const FArray3& dum7, const FArray3& dum8, const FArray3& turbz, const FArray3& dum1,
            const FArray3& dum2, const FArray3& dumz, const FArray3& sflux) {
  if (o.cm1setup != 1 && o.cm1setup != 2 && o.cm1setup != 4)
    throw std::runtime_error("cm1k turb::turbsz: only the LES/mesoscale boundary conditions (cm1setup=1,2,4) are ported");
  // surface flux of s (turbsz: iflux=1 .and. dosfcflx)
  const bool use_sflux = (iflux == 1 && o.dosfcflx);
  if (use_sflux && !sflux.is_allocated())
    throw std::runtime_error("cm1k turb::turbsz: dosfcflx with iflux=1 needs the surface flux");
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 2) throw std::runtime_error("cm1k turb::turbsz: nk >= 2 required");
  const Real rdz = g.rdz;
  const auto mh = g.mh;
  const auto mf = g.mf;
  const int bcturbs = o.bcturbs;
  if (o.doimpl == 0) {
    // the Fortran reuses dumz; a separate (k=1..nk+1) flux array is not needed
    // since all fluxes of a column are set before the divergence.
    Kokkos::parallel_for("turbsz_e", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      // dumz is (kb:ke): k=nk+1 is ke (ngz=1)
      for (int k = 2; k <= nk; ++k)
        dumz(i, j, k) = -khv(i, j, k) * (s(i, j, k) - s(i, j, k - 1)) * rdz * mf(i, j, k) * rf(i, j, k);
      if (bcturbs == 1) {
        dumz(i, j, 1) = 0.0;
        dumz(i, j, nk + 1) = 0.0;
      } else {
        dumz(i, j, 1) = dumz(i, j, 2);
        dumz(i, j, nk + 1) = dumz(i, j, nk);
      }
      if (use_sflux) dumz(i, j, 1) = sflux(i, j, 1) * rf(i, j, 1);
      for (int k = 1; k <= nk; ++k)
        turbz(i, j, k) = -(dumz(i, j, k + 1) - dumz(i, j, k)) * rdz * mh(i, j, k);
    });
  } else {
    const Real rdt = 1.0 / dt;
    const Real tema = -1.0 * dt * vialpha * rdz * rdz;
    const Real temb = dt * vibeta * rdz * rdz;
    const Real temc = dt * rdz;
    Kokkos::parallel_for("turbsz_i", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      if (bcturbs == 1) {
        dumz(i, j, 1) = 0.0;
        dumz(i, j, nk + 1) = 0.0;
      } else {
        dumz(i, j, 1) = -khv(i, j, 2) * (s(i, j, 2) - s(i, j, 1)) * rdz * mf(i, j, 2) * rf(i, j, 2);
        dumz(i, j, nk + 1) = -khv(i, j, nk) * (s(i, j, nk) - s(i, j, nk - 1)) * rdz * mf(i, j, nk) * rf(i, j, nk);
      }
      if (use_sflux) dumz(i, j, 1) = sflux(i, j, 1) * rf(i, j, 1);   // surface heat/moisture flux
      {
        const int k = 1;
        const Real r2 = dum8(i, j, k);
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfc;
        const Real cfd = s(i, j, k) + temb * (-r2 * s(i, j, k) + r2 * s(i, j, k + 1))
                         + temc * dumz(i, j, 1) * mh(i, j, 1) * rr(i, j, 1);
        const Real tem = 1.0 / cfb;
        dum1(i, j, 1) = -cfc * tem;
        dum2(i, j, 1) = cfd * tem;
      }
      for (int k = 2; k <= nk - 1; ++k) {
        const Real r1 = dum7(i, j, k);
        const Real r2 = dum8(i, j, k);
        const Real cfa = tema * r1;
        const Real cfc = tema * r2;
        const Real cfb = 1.0 - cfa - cfc;
        const Real cfd = s(i, j, k) + temb * (r1 * s(i, j, k - 1) - (r1 + r2) * s(i, j, k) + r2 * s(i, j, k + 1));
        const Real tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dum1(i, j, k) = -cfc * tem;
        dum2(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
      }
      {
        const int k = nk;
        const Real r1 = dum7(i, j, k);
        const Real cfa = tema * r1;
        const Real cfb = 1.0 - cfa;
        const Real cfd = s(i, j, k) + temb * (r1 * s(i, j, k - 1) - r1 * s(i, j, k))
                         - temc * dumz(i, j, nk + 1) * mh(i, j, nk) * rr(i, j, nk);
        const Real tem = 1.0 / (cfa * dum1(i, j, k - 1) + cfb);
        dumz(i, j, k) = (cfd - cfa * dum2(i, j, k - 1)) * tem;
        turbz(i, j, k) = rho(i, j, k) * (dumz(i, j, k) - s(i, j, k)) * rdt;
      }
      for (int k = nk - 1; k >= 1; --k) {
        dumz(i, j, k) = dum1(i, j, k) * dumz(i, j, k + 1) + dum2(i, j, k);
        turbz(i, j, k) = rho(i, j, k) * (dumz(i, j, k) - s(i, j, k)) * rdt;
      }
    });
  }
}

void turbs(const Options& o, const Grid& g, int iflux, Real dt, const FArray3& rho,
           const FArray3& rr, const FArray3& rf, const FArray3& s, const FArray3& sten,
           const FArray3& khh, const FArray3& khv, const FArray3& dum7, const FArray3& dum8,
           const Work& wk, const FArray3& sflux) {
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  if (nk < 2) throw std::runtime_error("cm1k turb::turbs: nk >= 2 required");
  const Real rdx = g.rdx, rdy = g.rdy;
  const auto uh = g.uh;
  const auto uf = g.uf;
  const auto vh = g.vh;
  const auto vf = g.vf;
  const auto turbx = wk.d1, turby = wk.d2, turbz = wk.d3, dumx = wk.d4, dumy = wk.d5, dumz = wk.d6;

  if (o.dohturb && b.terrain_flag) {
    // Cartesian with terrain (the open-bc flux copies of the flat branch are not applied here)
    if (nk < 3) throw std::runtime_error("cm1k turb::turbs: terrain needs nk >= 3");
    const Real cgs1 = g.cgs1, cgs2 = g.cgs2, cgs3 = g.cgs3, cgt1 = g.cgt1, cgt2 = g.cgt2, cgt3 = g.cgt3;
    const auto rds = g.rds; const auto sigma = g.sigma; const auto rdsf = g.rdsf; const auto sigmaf = g.sigmaf;
    const auto gz = g.gz; const auto rgz = g.rgz; const auto rgzu = g.rgzu; const auto rgzv = g.rgzv;
    const auto gx = g.gx; const auto gxu = g.gxu; const auto gy = g.gy; const auto gyv = g.gyv;
    // use turbz as a temporary array for s at w-pts
    Kokkos::parallel_for("turbs_t_sw", box3(0, ni + 1, 0, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
      if (k == 1) {
        turbz(i, j, 1) = cgs1 * s(i, j, 1) + cgs2 * s(i, j, 2) + cgs3 * s(i, j, 3);
      } else if (k == nk + 1) {
        turbz(i, j, nk + 1) = cgt1 * s(i, j, nk) + cgt2 * s(i, j, nk - 1) + cgt3 * s(i, j, nk - 2);
      } else {
        const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
        const Real r1 = 1.0 - r2;
        turbz(i, j, k) = r1 * s(i, j, k - 1) + r2 * s(i, j, k);
      }
    });
    Kokkos::parallel_for("turbs_t_fx", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumx(i, j, k) = -0.125 * (gz(i, j, 1) * rho(i, j, k) + gz(i - 1, j, 1) * rho(i - 1, j, k))
                      * ((khh(i, j, k) + khh(i - 1, j, k)) + (khh(i, j, k + 1) + khh(i - 1, j, k + 1))) * (
                          (s(i, j, k) * rgz(i, j, 1) - s(i - 1, j, k) * rgz(i - 1, j, 1)) * rdx * uf(i)
                          + 0.5 * (gxu(i, j, k + 1) * (turbz(i, j, k + 1) + turbz(i - 1, j, k + 1))
                                   - gxu(i, j, k) * (turbz(i, j, k) + turbz(i - 1, j, k)))
                                * rdsf(k) * rgzu(i, j, 1));
    });
    Kokkos::parallel_for("turbs_t_fy", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumy(i, j, k) = -0.125 * (gz(i, j, 1) * rho(i, j, k) + gz(i, j - 1, 1) * rho(i, j - 1, k))
                      * ((khh(i, j, k) + khh(i, j - 1, k)) + (khh(i, j, k + 1) + khh(i, j - 1, k + 1))) * (
                          (s(i, j, k) * rgz(i, j, 1) - s(i, j - 1, k) * rgz(i, j - 1, 1)) * rdy * vf(j)
                          + 0.5 * (gyv(i, j, k + 1) * (turbz(i, j, k + 1) + turbz(i, j - 1, k + 1))
                                   - gyv(i, j, k) * (turbz(i, j, k) + turbz(i, j - 1, k)))
                                * rdsf(k) * rgzv(i, j, 1));
    });
    // use turbz, dumz as temporary arrays for the fluxes at w-pts
    Kokkos::parallel_for("turbs_t_fw", box3(1, ni + 1, 1, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
      if (k == 1) {
        turbz(i, j, 1) = cgs1 * dumx(i, j, 1) + cgs2 * dumx(i, j, 2) + cgs3 * dumx(i, j, 3);
        dumz(i, j, 1) = cgs1 * dumy(i, j, 1) + cgs2 * dumy(i, j, 2) + cgs3 * dumy(i, j, 3);
      } else if (k == nk + 1) {
        turbz(i, j, nk + 1) = cgt1 * dumx(i, j, nk) + cgt2 * dumx(i, j, nk - 1) + cgt3 * dumx(i, j, nk - 2);
        dumz(i, j, nk + 1) = cgt1 * dumy(i, j, nk) + cgt2 * dumy(i, j, nk - 1) + cgt3 * dumy(i, j, nk - 2);
      } else {
        const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
        const Real r1 = 1.0 - r2;
        turbz(i, j, k) = r1 * dumx(i, j, k - 1) + r2 * dumx(i, j, k);
        dumz(i, j, k) = r1 * dumy(i, j, k - 1) + r2 * dumy(i, j, k);
      }
    });
    Kokkos::parallel_for("turbs_t_h", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = -gz(i, j, 1) * (dumx(i + 1, j, k) * rgzu(i + 1, j, 1) - dumx(i, j, k) * rgzu(i, j, 1)) * rdx * uh(i)
                       - ((gx(i, j, k + 1) * (turbz(i, j, k + 1) + turbz(i + 1, j, k + 1))
                           - gx(i, j, k) * (turbz(i, j, k) + turbz(i + 1, j, k)))) * 0.5 * rdsf(k);
      turby(i, j, k) = -gz(i, j, 1) * (dumy(i, j + 1, k) * rgzv(i, j + 1, 1) - dumy(i, j, k) * rgzv(i, j, 1)) * rdy * vh(j)
                       - ((gy(i, j, k + 1) * (dumz(i, j, k + 1) + dumz(i, j + 1, k + 1))
                           - gy(i, j, k) * (dumz(i, j, k) + dumz(i, j + 1, k)))) * 0.5 * rdsf(k);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbs_ow", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbs_oe", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbs_os", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbs_on", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj, k) = 0.0; });
  } else if (o.dohturb) {
    Kokkos::parallel_for("turbs_flux", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumx(i, j, k) = -0.125 * (rho(i, j, k) + rho(i - 1, j, k))
                      * ((khh(i, j, k) + khh(i - 1, j, k)) + (khh(i, j, k + 1) + khh(i - 1, j, k + 1)))
                      * (s(i, j, k) - s(i - 1, j, k)) * rdx * uf(i);
      dumy(i, j, k) = -0.125 * (rho(i, j, k) + rho(i, j - 1, k))
                      * ((khh(i, j, k) + khh(i, j - 1, k)) + (khh(i, j, k + 1) + khh(i, j - 1, k + 1)))
                      * (s(i, j, k) - s(i, j - 1, k)) * rdy * vf(j);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbs_bw", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { dumx(1, j, k) = dumx(2, j, k); });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbs_be", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { dumx(ni + 1, j, k) = dumx(ni, j, k); });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbs_bs", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { dumy(i, 1, k) = dumy(i, 2, k); });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbs_bn", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { dumy(i, nj + 1, k) = dumy(i, nj, k); });
    Kokkos::parallel_for("turbs_h", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = -(dumx(i + 1, j, k) - dumx(i, j, k)) * rdx * uh(i);
      turby(i, j, k) = -(dumy(i, j + 1, k) - dumy(i, j, k)) * rdy * vh(j);
    });
    if (o.wbc == 2 && o.ibw == 1) Kokkos::parallel_for("turbs_ow", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(1, j, k) = 0.0; });
    if (o.ebc == 2 && o.ibe == 1) Kokkos::parallel_for("turbs_oe", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) { turbx(ni, j, k) = 0.0; });
    if (o.sbc == 2 && o.ibs == 1) Kokkos::parallel_for("turbs_os", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, 1, k) = 0.0; });
    if (o.nbc == 2 && o.ibn == 1) Kokkos::parallel_for("turbs_on", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) { turby(i, nj, k) = 0.0; });
  } else {
    Kokkos::parallel_for("turbs_h0", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      turbx(i, j, k) = 0.0;
      turby(i, j, k) = 0.0;
    });
  }

  // turbsz (z-direction; dum1, dum2 = dumx, dumy as in turbs)
  if (o.dovturb) {
    turbsz(o, g, iflux, dt, rho, rr, rf, s, khv, dum7, dum8, turbz, dumx, dumy, dumz, sflux);
  } else {
    Kokkos::parallel_for("turbs_z0", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { turbz(i, j, k) = 0.0; });
  }

  Kokkos::parallel_for("turbs_ten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    sten(i, j, k) = sten(i, j, k) + ((turbx(i, j, k) + turby(i, j, k)) + turbz(i, j, k)) * rr(i, j, k);
  });
}

void turb_tendencies(const Options& o, const Grid& g, const BaseState& base, Real dt,
                     const FArray3& rho, const FArray3& rr, const FArray3& rf,
                     const FArray3& ua, const FArray3& va, const FArray3& wa,
                     const FArray3& tha, const FArray4& qa,
                     const FArray3& uten1, const FArray3& vten1, const FArray3& wten1,
                     const FArray3& thten1, const FArray4& qten,
                     const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
                     const TurbState& ts, const Work& wk, const sfc::SfcState* sfc) {
  o.check();
  if (o.dosfcflx && !sfc)
    throw std::runtime_error("cm1k turb::turb_tendencies: dosfcflx needs the surface layer state");
  const auto& b = g.b;
  const FArray3 thflux = sfc ? sfc->thflux : FArray3(), qvflux = sfc ? sfc->qvflux : FArray3();
  turbu(o, g, dt, rho, rf, ua, uten1, wa, ts.t11, ts.t12, ts.t13, kmv, wk);
  turbv(o, g, dt, rho, rf, va, vten1, wa, ts.t12, ts.t22, ts.t23, kmv, wk);
  turbw(o, g, dt, rho, rf, wa, wten1, ts.t13, ts.t23, ts.t33, kmh, wk);
  if ((o.doimpl == 1 && o.dovturb) || (o.cm1setup == 2 && o.ipbl == 2)) vimpl_coefs(b, khv, g.mf, rf, g.mh, rr, wk.d7, wk.d8);
  // cm1r18: subtract th0r from theta (as in advection scheme)
  const auto th0 = base.th0, sadv = wk.sadv;
  if (b.terrain_flag)
    Kokkos::parallel_for("turb_sadv", box3(0, b.ni + 1, 0, b.nj + 1, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      sadv(i, j, k) = (th0(i, j, k) - th0r) + tha(i, j, k);
    });
  else
  Kokkos::parallel_for("turb_sadv", box3(0, b.ni + 1, 0, b.nj + 1, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real tem = th0(1, 1, k) - th0r;
    sadv(i, j, k) = tem + tha(i, j, k);
  });
  turbs(o, g, 1, dt, rho, rr, rf, sadv, thten1, khh, khv, wk.d7, wk.d8, wk, thflux);
  if (o.imoist == 1)
    for (int n = 1; n <= o.numq; ++n)
      turbs(o, g, (n == o.nqv) ? 1 : 0, dt, rho, rr, rf, slice(qa, n), slice(qten, n), khh, khv, wk.d7, wk.d8, wk,
            qvflux);
}

//------------------------------------------------------------------------------
// solve2/solve3 helpers
//------------------------------------------------------------------------------

void tke_rk_update(const Options& o, const Bounds& b, Real dttmp, const FArray3& tkea,
                   const FArray3& tten, const FArray3& tke3d) {
  if (b.nk >= 2)
    Kokkos::parallel_for("tke_rk_update", box3(1, b.ni, 1, b.nj, 2, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      tke3d(i, j, k) = tkea(i, j, k) + dttmp * tten(i, j, k);
      if (tke3d(i, j, k) < 1.0e-10) tke3d(i, j, k) = 0.0;
    });
  bcw(o, b, tke3d, 1);
}

void tke_copy(const Bounds& b, const FArray3& tke3d, const FArray3& tkea) {
  Kokkos::parallel_for("tke_copy", box3(0, b.ni + 1, 0, b.nj + 1, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    tkea(i, j, k) = tke3d(i, j, k);
  });
}

} // namespace cm1k::turb
