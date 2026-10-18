// cm1.F radiation for radopt = 2 (see cm1k_radiation.hpp).

#include "physics/rrtmg/cm1k_radiation.hpp"

#include "init/cm1k_init.hpp"
#include "physics/kessler/cm1k_kessler.hpp"
#include "physics/rrtmg/cm1k_rrtmg_lw.hpp"
#include "physics/rrtmg/cm1k_rrtmg_sw.hpp"
#include "physics/rrtmg/cm1k_rrtmg_util.hpp"
#include "physics/sfc/cm1k_sfc.hpp"

#include <cmath>
#include <stdexcept>

namespace cm1k::rrtmg {

namespace {
[[noreturn]] void unsupported(const std::string& m) {
  throw std::runtime_error("cm1k radiation: " + m + " is not ported");
}

// module_model_constants (module_ra_etc.F): rrtmg_lwinit / rrtmg_swinit call
// rrtmg_lw_ini(cp, g), rrtmg_sw_ini(cp, g) with these
constexpr Real wrf_cp = 1005.7, wrf_g = 9.81;

// radiation_driver.F parameters
constexpr Real dpd = 360.0 / 365.0;
constexpr Real degrad = 3.1415926535897932384626433 / 180.0;
} // namespace

RadOptions RadOptions::from_namelist(const Namelist& nl) {
  RadOptions o;
  o.radopt = nl.geti("radopt", 0);
  o.dtrad = nl.getr("dtrad", o.dtrad);
  o.ctrlat = nl.getr("ctrlat", o.ctrlat);
  o.ctrlon = nl.getr("ctrlon", o.ctrlon);
  o.year = nl.geti("year", o.year);
  o.month = nl.geti("month", o.month);
  o.day = nl.geti("day", o.day);
  o.hour = nl.geti("hour", o.hour);
  o.minute = nl.geti("minute", o.minute);
  o.second = nl.geti("second", o.second);
  o.psolver = nl.geti("psolver", 3);
  o.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  if (o.psolver >= 4 && o.psolver <= 7) o.eqtset = 1;   // param.F
  o.imoist = nl.geti("imoist", 0);
  o.ptype = nl.geti("ptype", 0);
  o.testcase = nl.geti("testcase", 0);
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.ipbl = nl.geti("ipbl", 0);
  o.rterm = nl.geti("rterm", 0);
  o.lu0 = nl.geti("lu0", 16);
  o.season = nl.geti("season", 1);
  return o;
}

void RadOptions::check() const {
  if (radopt == 0) return;
  if (radopt == 1) unsupported("radopt=1 (NASA-GSFC)");
  if (radopt != 2) throw std::runtime_error("cm1k radiation: radopt must be 0, 1 or 2 (param.F)");
  // param.F stops for these
  if (imoist != 1) throw std::runtime_error("cm1k radiation: radopt=2 requires imoist=1 (param.F)");
  if (rterm != 0) throw std::runtime_error("cm1k radiation: radopt and rterm cannot be used together (param.F)");
  if (sfcmodel == 0) throw std::runtime_error("cm1k radiation: radopt=2 requires a surface model (param.F)");
  if (testcase == 4 || testcase == 5) throw std::runtime_error("cm1k radiation: testcase=4,5 require radopt=0");
  // ported subset
  if (ptype != 1) unsupported("radopt=2 with ptype=" + std::to_string(ptype) + " (only Kessler, ptype=1)");
  if (sfcmodel != 2 && sfcmodel != 3)
    unsupported("radopt=2 with sfcmodel=" + std::to_string(sfcmodel) + " (only 2, 3: gsw/glw/emiss of SLAB)");
  if (ipbl == 4 || ipbl == 5) unsupported("radopt=2 with the MYNN PBL (icloud_bl coupling)");
  if (testcase != 0) unsupported("radopt=2 with testcase=" + std::to_string(testcase));
  if (month < 1 || month > 12) throw std::runtime_error("cm1k radiation: month must be 1..12");
}

int julday(int year, int month, int day) {
  static const int mndys[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  const int lpyear = (year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)) ? 1 : 0;
  const int lp = month > 2 ? lpyear : 0;
  return mndys[month - 1] + day + lp;
}

void radconst(Real julian, Real degrad_, Real dpd_, Real& declin, Real& solcon) {
  const Real OBECL = 23.5 * degrad_;
  const Real SINOB = std::sin(OBECL);
  Real SXLONG = 0;
  if (julian >= 80.) SXLONG = dpd_ * (julian - 80.);
  if (julian < 80.) SXLONG = dpd_ * (julian + 285.);
  SXLONG = SXLONG * degrad_;
  const Real ARG = SINOB * std::sin(SXLONG);
  declin = std::asin(ARG);
  const Real DJUL = julian * 360. / 365.;
  const Real RJUL = DJUL * degrad_;
  const Real ECCFAC = 1.000110 + 0.034221 * std::cos(RJUL) + 0.001280 * std::sin(RJUL) +
                      0.000719 * std::cos(2 * RJUL) + 0.000077 * std::sin(2 * RJUL);
  solcon = 1370. * ECCFAC;
}

void calc_coszen(Real julian, Real xtime, Real gmt, Real declin, Real degrad_, Real xlon, Real xlat, Real& coszen,
                 Real& hrang) {
  const Real da = 6.2831853071795862 * (julian - 1) / 365.;
  const Real eot = (0.000075 + 0.001868 * std::cos(da) - 0.032077 * std::sin(da) - 0.014615 * std::cos(2 * da) -
                    0.04089 * std::sin(2 * da)) *
                   (229.18);
  const Real xt24 = std::fmod(xtime, 1440.) + eot;
  const Real tloctm = gmt + xt24 / 60. + xlon / 15.;
  hrang = 15. * (tloctm - 12.) * degrad_;
  const Real xxlat = xlat * degrad_;
  coszen = std::sin(xxlat) * std::sin(declin) + std::cos(xxlat) * std::cos(declin) * std::cos(hrang);
  coszen = std::min(std::max(coszen, -1.0), 1.0);
}

void Radiation::setup(const RadOptions& o, const Constants& c, const Bounds& b, const FArray3& tsk_,
                      const FArray3& xland_, const FArray3& gsw_, const FArray3& glw_, const FArray3& emiss_) {
  o.check();
  m_o = o;
  m_c = c;
  m_b = b;
  const std::array<int, 3> lo3{b.ib, b.jb, b.kb}, hi3{b.ie, b.je, b.ke};
  const std::array<int, 3> lo2{b.ib, b.jb, 1}, hi2{b.ie, b.je, 1};
  auto m3 = [&](const char* n) { return FArray3(n, lo3, hi3); };
  auto m2 = [&](const char* n) { return FArray3(n, lo2, hi2); };
  swten = m3("swten"); lwten = m3("lwten"); swtenc = m3("swtenc"); lwtenc = m3("lwtenc");
  cldfra = m3("cldfra"); o30 = m3("o30"); thraten = m3("thraten");
  lwupt = m2("lwupt"); lwuptc = m2("lwuptc"); lwdnt = m2("lwdnt"); lwdntc = m2("lwdntc");
  lwupb = m2("lwupb"); lwupbc = m2("lwupbc"); lwdnb = m2("lwdnb"); lwdnbc = m2("lwdnbc");
  swupt = m2("swupt"); swuptc = m2("swuptc"); swdnt = m2("swdnt"); swdntc = m2("swdntc");
  swupb = m2("swupb"); swupbc = m2("swupbc"); swdnb = m2("swdnb"); swdnbc = m2("swdnbc");
  lwcf = m2("lwcf"); swcf = m2("swcf"); coszr = m2("coszr"); xice = m2("xice"); xsnow = m2("xsnow");
  xlat = m2("xlat"); xlong = m2("xlong"); coszen = m2("coszen"); swddir = m2("swddir"); swddni = m2("swddni");
  swddif = m2("swddif"); hrang = m2("hrang");
  albd = m2("albd");
  olr = FArray3("olr", {1, 1, 1}, {b.ni, b.nj, 1});
  tsk = tsk_; xland = xland_; gsw = gsw_; glw = glw_; emiss = emiss_;
  dum1 = m3("rad_dum1"); dum2 = m3("rad_dum2"); dum3 = m3("rad_dum3"); dum4 = m3("rad_dum4");
  dum5 = m3("rad_dum5"); dum6 = m2("rad_dum6"); dum7 = m3("rad_dum7"); dum8 = m3("rad_dum8");
  dum9 = m3("rad_dum9"); dum10 = m3("rad_dum10"); dum11 = m3("rad_dum11");
  swdownc = m2("swdownc"); swddnic = m2("swddnic"); swddirc = m2("swddirc");
  // init_surface.F getlanduse (initsfc=1: lu_index = lu0 everywhere)
  albd.fill(sfc::landuse_albd(o.lu0, o.season) / 100.);
  // radiation_driver: dtrad = max(dtrad, dt) at every call (dt is constant)
  m_dtrad = o.dtrad;
  m_jday = julday(o.year, o.month, o.day);   // init_physics.F
  init_state();
}

void Radiation::init_state() {
  o30.fill(1.0e-12);   // init_physics.F
  m_radtim = 0.0;      // param.F
}

void Radiation::set_clock(int nstep, Real dt) { m_radtim = clock_at(nstep, dt, m_dtrad); }

double Radiation::clock_at(int nstep, Real dt, Real dtrad_in) {
  if (!(dt > 0)) throw std::runtime_error("cm1k radiation: dt must be > 0");
  const Real dtrad = std::max(dtrad_in, dt);
  const double dbldt = double(dt);
  double radtim = 0.0, mtime = 0.0;
  auto advance = [&] {
    do { radtim = radtim + dtrad; } while (!(radtim > mtime));
  };
  for (int n = 0; n < nstep; ++n) {
    if (n == 0) { advance(); continue; }   // startup pass
    // pass n: dorad from the time at its start, then mtime = mtime + dt
    const bool dorad = (mtime + dbldt) >= (radtim - 0.1 * dbldt);
    mtime = mtime + dbldt;
    if (dorad) advance();
  }
  return radtim;
}

void Radiation::start(const Grid& g, const BaseState& base) {
  if (m_started) return;
  const auto& b = m_b;
  const auto ex = init::extrap_coeffs(g.sigma.to_host(), g.sigmaf.to_host(), b.nk);
  m_cgs1 = ex.cgs1; m_cgs2 = ex.cgs2; m_cgs3 = ex.cgs3;
  m_cgt1 = ex.cgt1; m_cgt2 = ex.cgt2; m_cgt3 = ex.cgt3;
  // init_physics.F
  const auto prs0 = base.prs0.to_host();
  const int nk = b.nk;
  m_p_top = m_cgt1 * prs0(1, 1, nk) + m_cgt2 * prs0(1, 1, nk - 1) + m_cgt3 * prs0(1, 1, nk - 2);
  m_nlayers = lw_nlayers(b.ke, m_p_top);
  m_lw = lw_tables(wrf_cp, wrf_g, find_data_file("RRTMG_LW_DATA"));
  m_sw = sw_tables(wrf_cp, wrf_g, find_data_file("RRTMG_SW_DATA"));
  m_started = true;
}

void Radiation::pass(double mtime, int nstep, Real dt, const Grid& g, const BaseState& base, const State& s) {
  const double dbldt = double(dt);
  // cm1.F: dorad (startup pass, or the end of the pass reached radtim)
  const bool dorad = nstep == 0 || mtime >= (m_radtim - 0.1 * dbldt);
  if (!dorad) return;
  if (!(dt > 0)) throw std::runtime_error("cm1k radiation: dt must be > 0");
  driver(mtime, nstep, dt, g, base, s);
  const Real dtrad = std::max(m_dtrad, dt);
  do { m_radtim = m_radtim + dtrad; } while (!(m_radtim > mtime));
}

void Radiation::driver(double mtime, int /*nstep*/, Real dt, const Grid& g, const BaseState& base,
                       const State& s) {
  start(g, base);
  const auto& b = m_b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real dtrad = std::max(m_dtrad, dt);
  // Kessler's moisture indices (qv, qc, qr)
  constexpr int nqv = kessler::nqv, nqc = kessler::nqc, nqr = kessler::nqr;

  auto d1 = dum1, d2 = dum2, d3 = dum3, d4 = dum4, d5 = dum5, d7 = dum7, d8 = dum8, d9 = dum9, d10 = dum10,
       d11 = dum11;
  auto pi0 = base.pi0, th0 = base.th0;
  auto ppi = s.ppi, tha = s.tha, prs = s.prs, rho = s.rho;
  auto qa = s.qa;
  auto c1 = g.c1, c2 = g.c2, rmh = g.rmh;
  Kokkos::parallel_for("rad_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    d4(i, j, k) = (pi0(i, j, k) + ppi(i, j, k));
    d3(i, j, k) = (th0(i, j, k) + tha(i, j, k)) * d4(i, j, k);
    d7(i, j, k) = 0.0; d8(i, j, k) = 0.0; d9(i, j, k) = 0.0; d10(i, j, k) = 0.0; d11(i, j, k) = 0.0;
  });
  const Real dz = g.dz;
  // (k = 1 of dum1/dum2 is replaced below, so dum3(k = 0) is never used)
  Kokkos::parallel_for("rad_w", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    d1(i, j, k) = c1(i, j, k) * prs(i, j, k - 1) + c2(i, j, k) * prs(i, j, k);
    d2(i, j, k) = c1(i, j, k) * d3(i, j, k - 1) + c2(i, j, k) * d3(i, j, k);
    d5(i, j, k) = dz * rmh(i, j, k);
  });
  {
    const Real cgs1 = m_cgs1, cgs2 = m_cgs2, cgs3 = m_cgs3, cgt1 = m_cgt1, cgt2 = m_cgt2, cgt3 = m_cgt3;
    auto tsk_ = tsk;
    Kokkos::parallel_for("rad_bt", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      d1(i, j, 1) = cgs1 * prs(i, j, 1) + cgs2 * prs(i, j, 2) + cgs3 * prs(i, j, 3);
      d2(i, j, 1) = tsk_(i, j, 1);   // cm1r21.1: surface temperature
      d1(i, j, nk + 1) = cgt1 * prs(i, j, nk) + cgt2 * prs(i, j, nk - 1) + cgt3 * prs(i, j, nk - 2);
      d2(i, j, nk + 1) = cgt1 * d3(i, j, nk) + cgt2 * d3(i, j, nk - 1) + cgt3 * d3(i, j, nk - 2);
    });
  }
  // Kessler (nqv, nqc, nqr): qr -> dum11, qc -> dum10; no qs, qg, qi
  Kokkos::parallel_for("rad_q", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    d11(i, j, k) = qa(i, j, k, nqr);
    d10(i, j, k) = qa(i, j, k, nqc);
  });
  const bool f_qv = true, f_qc = true, f_qr = true, f_qi = false, f_qs = false, f_qg = false;
  const int mp_physics = 1, icloud = 1, o3input = 0;

  const auto& o = m_o;
  const Real julian = m_jday - 1.0 + o.hour / 24.0 + o.minute / (60.0 * 24.0) + (o.second + mtime) / (60.0 * 60.0 * 24.0);
  const int yr = o.year;
  const Real gmt = o.hour;
  const Real xtime = mtime / 60.0;
  xlong.fill(o.ctrlon);
  xlat.fill(o.ctrlat);
  Real declin = 0, solcon = 0;
  radconst(julian, degrad, dpd, declin, solcon);
  (void)xtime;   // radconst does not use it
  const Real tfoo = (mtime + 0.5 * dtrad) / 60.0;
  {
    // calc_coszen over 1..ni, 1..nj: xlat, xlong are the same everywhere
    Real cz = 0, hr = 0;
    calc_coszen(julian, tfoo, gmt, declin, degrad, Real(o.ctrlon), Real(o.ctrlat), cz, hr);
    auto coszen_ = coszen, hrang_ = hrang;
    Kokkos::parallel_for("rad_coszen", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      coszen_(i, j, 1) = cz;
      hrang_(i, j, 1) = hr;
    });
  }
  {
    auto gsw_ = gsw, glw_ = glw;
    Kokkos::parallel_for("rad_g0", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      gsw_(i, j, 1) = 0.0;
      glw_(i, j, 1) = 0.0;
    });
  }

  // cal_cldfra1 (QV = qa(nqv), QC = dum10, QI = dum9, QS = dum7, t = dum3,
  // p = prs), the F_QC only branch (mp_physics = 1; F_QI = F_QS = .false.)
  {
    auto cf = cldfra;
    Kokkos::parallel_for("cal_cldfra1", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      constexpr Real ALPHA0 = 100., GAMMA = 0.49, QCLDMIN = 1.E-12, PEXP = 0.25, RHGRID = 1.0;
      constexpr Real SVP1 = 0.6112, SVP2 = 17.67, SVPI2 = 21.8745584, SVP3 = 29.65, SVPI3 = 7.66, SVPT0 = 273.15;
      constexpr Real r_d = 287.04, r_v = 461.5, ep_2 = r_d / r_v;
      const Real t = d3(i, j, k), p = prs(i, j, k);
      const Real tc = t - SVPT0;
      const Real esw = 1000.0 * SVP1 * Kokkos::exp(SVP2 * tc / (t - SVP3));
      const Real esi = 1000.0 * SVP1 * Kokkos::exp(SVPI2 * tc / (t - SVPI3));
      const Real QVSW = ep_2 * esw / (p - esw);
      const Real QVSI = ep_2 * esi / (p - esi);
      const Real QCLD = d10(i, j, k);
      Real weight;
      if (QCLD < QCLDMIN) {
        weight = 0.;
      } else {
        weight = 0.;
        if (t > 273.15) weight = 0.;
        if (t <= 273.15) weight = 1.;
      }
      const Real QVS_WEIGHT = (1 - weight) * QVSW + weight * QVSI;
      Real RHUM = qa(i, j, k, nqv) / QVS_WEIGHT;
      if (QCLD < QCLDMIN) {
        cf(i, j, k) = 0.;
      } else if (RHUM >= RHGRID) {
        cf(i, j, k) = 1.;
      } else {
        const Real SUBSAT = Kokkos::max(1.E-10, RHGRID * QVS_WEIGHT - qa(i, j, k, nqv));
        const Real DENOM = Kokkos::pow(SUBSAT, GAMMA);
        const Real ARG = Kokkos::max(-6.9, -ALPHA0 * QCLD / DENOM);
        RHUM = Kokkos::max(1.E-10, RHUM);
        cf(i, j, k) = Kokkos::pow(RHUM / RHGRID, PEXP) * (1. - Kokkos::exp(ARG));
        if (cf(i, j, k) < .01) cf(i, j, k) = 0.;
      }
    });
  }
  (void)f_qi; (void)f_qs; (void)f_qg;

  auto qv3d = FArray3(Kokkos::subview(qa.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, nqv - 1),
                      {b.ibm, b.jbm, b.kbm});

  // RRTMG_LWRAD
  {
    LwRadArgs a;
    a.rthratenlw = lwten; a.lwtenc = lwtenc;
    a.lwupt = lwupt; a.lwuptc = lwuptc; a.lwdnt = lwdnt; a.lwdntc = lwdntc;
    a.lwupb = lwupb; a.lwupbc = lwupbc; a.lwdnb = lwdnb; a.lwdnbc = lwdnbc;
    a.glw = glw; a.olr = dum6; a.lwcf = lwcf;
    a.emiss = emiss; a.tsk = tsk; a.xland = xland; a.xice = xice; a.snow = xsnow;
    a.p8w = dum1; a.p3d = prs; a.pi3d = dum4; a.dz8w = dum5; a.t3d = dum3; a.t8w = dum2; a.rho3d = rho;
    a.cldfra3d = cldfra;
    a.qv3d = qv3d; a.qc3d = dum10; a.qr3d = dum11; a.qi3d = dum9; a.qs3d = dum7; a.qg3d = dum8;
    a.o33d = o30;
    a.r = m_c.rd; a.g = m_c.g;
    a.icloud = icloud; a.cldovrlp = 2; a.o3input = o3input;
    a.warm_rain = false;
    a.f_qv = f_qv; a.f_qc = f_qc; a.f_qr = f_qr; a.f_qi = f_qi; a.f_qs = f_qs; a.f_qg = f_qg;
    a.has_reqc = a.has_reqi = a.has_reqs = 0;
    a.is_cammgmp_used = false;
    a.calc_clean_atm_diag = 0;
    a.yr = yr; a.julian = julian;
    a.mp_physics = mp_physics;
    a.its = 1; a.ite = ni; a.jts = 1; a.jte = nj; a.kts = 1; a.kte = nk;
    a.nlayers = m_nlayers;
    rrtmg_lwrad(*m_lw, a);
  }
  {
    auto olr_ = olr, d6 = dum6;
    Kokkos::parallel_for("rad_olr", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) { olr_(i, j, 1) = d6(i, j, 1); });
  }

  // RRTMG_SWRAD
  {
    SwRadArgs a;
    a.rthratensw = swten; a.swtenc = swtenc;
    a.swupt = swupt; a.swuptc = swuptc; a.swdnt = swdnt; a.swdntc = swdntc;
    a.swupb = swupb; a.swupbc = swupbc; a.swdnb = swdnb; a.swdnbc = swdnbc;
    a.swcf = swcf; a.gsw = gsw; a.coszr = coszr;
    a.swddir = swddir; a.swddni = swddni; a.swddif = swddif;
    a.swdownc = swdownc; a.swddnic = swddnic; a.swddirc = swddirc;
    a.xcoszen = coszen; a.albedo = albd; a.tsk = tsk; a.xland = xland; a.xice = xice; a.snow = xsnow;
    a.t3d = dum3; a.t8w = dum2; a.p3d = prs; a.p8w = dum1; a.pi3d = dum4; a.rho3d = rho; a.dz8w = dum5;
    a.cldfra3d = cldfra;
    a.qv3d = qv3d; a.qc3d = dum10; a.qr3d = dum11; a.qi3d = dum9; a.qs3d = dum7; a.qg3d = dum8;
    a.o33d = o30;
    a.r = m_c.rd; a.g = m_c.g; a.solcon = solcon;
    a.icloud = icloud; a.cldovrlp = 2; a.o3input = o3input;
    a.warm_rain = false;
    a.f_qv = f_qv; a.f_qc = f_qc; a.f_qr = f_qr; a.f_qi = f_qi; a.f_qs = f_qs; a.f_qg = f_qg;
    a.has_reqc = a.has_reqi = a.has_reqs = 0;
    a.is_cammgmp_used = false;
    a.calc_clean_atm_diag = 0;
    a.aer_opt = 0; a.sf_surface_physics = 1;
    a.yr = yr;
    a.mp_physics = mp_physics;
    a.its = 1; a.ite = ni; a.jts = 1; a.jte = nj; a.kts = 1; a.kte = nk;
    rrtmg_swrad(*m_sw, a);
  }

  auto lwt = lwten, swt = swten, lwtc = lwtenc, swtc = swtenc, thr = thraten;
  if (o.eqtset == 2) {
    const Real cp = m_c.cp, cv = m_c.cv, cvv = m_c.cvv, cpl = m_c.cpl, cpi = m_c.cpi;
    Kokkos::parallel_for("rad_rcvm", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real rcvm = cp / (cv + cvv * Kokkos::max(0.0, qa(i, j, k, nqv)) +
                              cpl * Kokkos::max(0.0, (d10(i, j, k) + d11(i, j, k))) +
                              cpi * Kokkos::max(0.0, (d9(i, j, k) + d7(i, j, k) + d8(i, j, k))));
      lwt(i, j, k) = lwt(i, j, k) * rcvm;
      swt(i, j, k) = swt(i, j, k) * rcvm;
      lwtc(i, j, k) = lwtc(i, j, k) * rcvm;
      swtc(i, j, k) = swtc(i, j, k) * rcvm;
    });
    const Real cvdcp = m_c.cvdcp;
    Kokkos::parallel_for("rad_thraten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      thr(i, j, k) = (swt(i, j, k) + lwt(i, j, k)) * cvdcp / (pi0(i, j, k) + ppi(i, j, k));
    });
  } else {
    Kokkos::parallel_for("rad_thraten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      thr(i, j, k) = (swt(i, j, k) + lwt(i, j, k)) / (pi0(i, j, k) + ppi(i, j, k));
    });
  }
  Kokkos::fence();
}

std::vector<std::pair<std::string, FArray3>> Radiation::dumped() const {
  return {{"swten", swten},   {"lwten", lwten},   {"swtenc", swtenc}, {"lwtenc", lwtenc}, {"thraten", thraten},
          {"cldfra", cldfra}, {"o30", o30},       {"lwupt", lwupt},   {"lwuptc", lwuptc}, {"lwdnt", lwdnt},
          {"lwdntc", lwdntc}, {"lwupb", lwupb},   {"lwupbc", lwupbc}, {"lwdnb", lwdnb},   {"lwdnbc", lwdnbc},
          {"swupt", swupt},   {"swuptc", swuptc}, {"swdnt", swdnt},   {"swdntc", swdntc}, {"swupb", swupb},
          {"swupbc", swupbc}, {"swdnb", swdnb},   {"swdnbc", swdnbc}, {"lwcf", lwcf},     {"swcf", swcf},
          {"coszr", coszr},   {"coszen", coszen}, {"hrang", hrang},   {"swddir", swddir}, {"swddni", swddni},
          {"swddif", swddif}, {"olr", olr},       {"albd", albd},     {"emiss", emiss},   {"gsw", gsw},
          {"glw", glw}};
}

} // namespace cm1k::rrtmg
