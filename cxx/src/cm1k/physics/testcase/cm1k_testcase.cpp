// Port of the testcase forcing of CM1 (see cm1k_testcase.hpp).
#include "physics/testcase/cm1k_testcase.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cm1k::testcase {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("cm1k testcase: " + m); }

// testcase_simple_phys.F parameters (simple radiation)
constexpr Real bigd = 3.75e-6;
constexpr Real qcrit = 0.008;
constexpr Real f0 = 70.0;
constexpr Real f1 = 22.0;
constexpr Real kappa = 85.0;
constexpr Real alphaz = 1.0;
constexpr Real rhoi = 1.12;
} // namespace

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

Namelist param_overrides(const Namelist& nl) {
  Namelist out = nl;
  const int testcase = nl.geti("testcase", 0);
  // param.F: testcase 1, 2 are dry; 3, 4, 5, 7, 8 and 11 moist
  if ((testcase == 1 || testcase == 2) && nl.geti("imoist", 0) != 0) out.set("imoist", "0");
  if ((testcase == 3 || testcase == 4 || testcase == 5 || testcase == 7 || testcase == 8 || testcase == 11) &&
      nl.geti("imoist", 0) != 1)
    out.set("imoist", "1");
  return out;
}

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.testcase = nl.geti("testcase", 0);
  o.lspgrad = nl.geti("lspgrad", 0);
  o.imoist = std::min(std::max(nl.geti("imoist", 1), 0), 1);
  if (o.testcase == 1 || o.testcase == 2) o.imoist = 0;
  if (o.testcase == 3 || o.testcase == 4 || o.testcase == 5 || o.testcase == 7 || o.testcase == 8 ||
      o.testcase == 11)
    o.imoist = 1;
  o.ptype = nl.geti("ptype", 0);
  if (o.imoist == 1) {
    if (o.ptype == 0) { o.numq = 1; o.nqv = 1; o.nql1 = 0; o.nql2 = 0; }
    if (o.ptype == 1) { o.numq = 3; o.nqv = 1; o.nql1 = 2; o.nql2 = 3; }
  }
  o.isnd = nl.geti("isnd", 1);
  o.iwnd = nl.geti("iwnd", 0);
  o.cm1setup = nl.geti("cm1setup", 1);
  o.ipbl = nl.geti("ipbl", 0);
  o.sgsmodel = nl.geti("sgsmodel", 1);
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.radopt = nl.geti("radopt", 0);
  o.imove = nl.geti("imove", 0);
  o.icor = std::min(std::max(nl.geti("icor", 0), 0), 1);
  o.axisymm = nl.geti("axisymm", 0);
  o.terrain_flag = nl.getb("terrain_flag", false);
  o.fcor = (o.icor == 1) ? nl.getr("fcor", 0.0) : 0.0;
  o.umove = (o.imove == 1) ? nl.getr("umove", 0.0) : 0.0;
  o.vmove = (o.imove == 1) ? nl.getr("vmove", 0.0) : 0.0;
  o.hurr_vg = nl.getr("hurr_vg", 0.0);
  o.hurr_rad = nl.getr("hurr_rad", 0.0);
  o.has_hurr_rad = nl.has("hurr_rad");
  o.hurr_vgpl = nl.getr("hurr_vgpl", 0.0);
  o.hurr_rotate = nl.getr("hurr_rotate", 0.0);
  // param.F: hurr_angle = dble(hurr_rotate)*(pi_dp/180.0d0). With
  // -fdefault-real-8, input.F's kind dp is selected_real_kind(30), i.e. quad
  // precision: the product is formed in binary128 and rounded to double on
  // the assignment (for hurr_rotate = 30 that gives sin(-hurr_angle) = -0.5
  // exactly, not the -0.49999999999999994 of a double pi/6).
  if (o.lspgrad == 3) {
    const __float128 pi_qp = 3.14159265358979323846264338327950288Q;   // 4*atan(1) in binary128
    o.hurr_angle = double(__float128(double(o.hurr_rotate)) * (pi_qp / __float128(180.0)));
  }
  return o;
}

void Options::check() const {
  if (testcase < 0 || testcase > 15) fail("testcase=" + std::to_string(testcase) + " does not exist (0..15)");
  if (lspgrad < 0 || lspgrad > 4) fail("lspgrad=" + std::to_string(lspgrad) + " does not exist (0..4)");
  if (!active()) return;
  // param.F stops
  if (simple_rad() && radopt >= 1) fail("testcase=4,5 require radopt=0 (param.F stops)");
  if (dolsw() && (terrain_flag || axisymm == 1))
    fail("large-scale vertical velocity (testcase 3-7) cannot be used with terrain or axisymm (param.F stops)");
  // not ported
  const bool wrf_ok = testcase == 8 || testcase == 9 || testcase == 10 || testcase == 12 || testcase == 14 ||
                     testcase == 15;
  if (testcase >= 1 && sfcmodel != 0 && sfcmodel != 1 && !((sfcmodel == 2 || sfcmodel == 3) && wrf_ok))
    fail("testcase=" + std::to_string(testcase) + " with sfcmodel=" + std::to_string(sfcmodel) +
         " is not ported (sfcmodel=0, 1; 2, 3 only for testcase 8, 9, 10, 12, 14, 15; sfcmodel=5 (cm1most) has "
         "its own testcase branches)");
  if (simple_rad() && (cm1setup == 2 || cm1setup == 4))
    fail("testcase=4,5 with a PBL scheme (cm1setup=2, 4) is not ported (the PBL schemes read the simple-radiation thraten)");
  if (simple_rad() && (imoist != 1 || ptype < 1))
    fail("testcase=4,5 need liquid-water microphysics (simplerad sums qa(nql1:nql2)); ptype=" +
         std::to_string(ptype) + " is not supported");
  if (axisymm != 0) fail("axisymm=1 is not ported");
  if ((lspgrad == 3 || testcase == 6 || testcase == 10 || testcase == 15) && (!has_hurr_rad || hurr_rad == 0.0))
    fail("lspgrad=3 and testcase=6,10,15 need hurr_rad (namelist param21)");
}

//------------------------------------------------------------------------------
// Forcing arrays
//------------------------------------------------------------------------------

void Forcing::allocate(const Bounds& b, const Options& o) {
  auto k1 = [&](const char* n) { return FArray1(n, {b.kb}, {b.ke}); };
  wprof = k1("wprof"); ufrc = k1("ufrc"); vfrc = k1("vfrc"); thfrc = k1("thfrc"); qvfrc = k1("qvfrc");
  ug = k1("ug"); vg = k1("vg"); dvdr = k1("dvdr"); ulspg = k1("ulspg"); vlspg = k1("vlspg");
  if (o.simple_rad()) {
    frad = FArray3("frad", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    thraten = FArray3("thraten", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    zir = FArray3("zir", {b.ib, b.jb, 1}, {b.ie, b.je, 1});
    // simplerad scratch (solve1's dum1..dum4); zero at k = nk+1
    w_ql = FArray3("sr_ql", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    w_qt = FArray3("sr_qt", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    w_d1 = FArray3("sr_d1", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    w_d2 = FArray3("sr_d2", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  }
}

std::vector<std::pair<std::string, FArray1>> Forcing::dumped1() const {
  return {{"wprof", wprof}, {"ufrc", ufrc}, {"vfrc", vfrc}, {"thfrc", thfrc}, {"qvfrc", qvfrc},
          {"ug", ug},       {"vg", vg},     {"dvdr", dvdr}, {"ulspg", ulspg}, {"vlspg", vlspg}};
}

std::vector<std::pair<std::string, FArray3>> Forcing::dumped3() const {
  if (!frad.is_allocated()) return {};
  return {{"frad", frad}, {"thraten", thraten}, {"zir", zir}};
}

//------------------------------------------------------------------------------
// base.F
//------------------------------------------------------------------------------

void init_profiles(const Options& o, const Grid& g, const BaseState& bs, const Forcing& f) {
  o.check();
  const auto& B = g.b;
  const int nk = B.nk, kb = B.kb, ke = B.ke;
  auto zh = g.zh.to_host(), zf = g.zf.to_host();
  auto u0 = bs.u0.to_host(), v0 = bs.v0.to_host();
  auto wprof = f.wprof.to_host(), ufrc = f.ufrc.to_host(), vfrc = f.vfrc.to_host();
  auto thfrc = f.thfrc.to_host(), qvfrc = f.qvfrc.to_host(), ug = f.ug.to_host(), vg = f.vg.to_host();
  auto dvdr = f.dvdr.to_host(), ulspg = f.ulspg.to_host(), vlspg = f.vlspg.to_host();
  // cm1.F: every profile starts at zero
  for (int k = kb; k <= ke; ++k) {
    wprof(k) = 0.0; ufrc(k) = 0.0; vfrc(k) = 0.0; thfrc(k) = 0.0; qvfrc(k) = 0.0;
    ug(k) = 0.0; vg(k) = 0.0; dvdr(k) = 0.0; ulspg(k) = 0.0; vlspg(k) = 0.0;
  }
  const int tc = o.testcase;
  const Real umove = o.umove, vmove = o.vmove;

  // iwnd = 8 (hurricane boundary layer): gradient wind, constant with height
  if (o.iwnd == 8)
    for (int k = 1; k <= nk; ++k) {
      ug(k) = 0.0;
      vg(k) = o.hurr_vg;
    }

  if (tc == 3) {
    // shallow Cu case (Siebesma et al, 2003, JAS)
    Real wmin = -0.0065, z1 = 1500.0, z2 = 2100.0;
    for (int k = 1; k <= nk; ++k) {
      if (zf(1, 1, k) <= z1) wprof(k) = 0.0 + (wmin - 0.0) * (zf(1, 1, k) - 0.0) / (z1 - 0.0);
      else if (zf(1, 1, k) <= z2) wprof(k) = wmin + (0.0 - wmin) * (zf(1, 1, k) - z1) / (z2 - z1);
      else wprof(k) = 0.0;
    }
    const Real radsfc = -2.0 / (3600.0 * 24.0);
    z1 = 1500.0; z2 = 2100.0;
    for (int k = 1; k <= nk; ++k) {
      if (zh(1, 1, k) <= z1) thfrc(k) = radsfc;
      else if (zh(1, 1, k) <= z2) thfrc(k) = radsfc + (0.0 - radsfc) * (zh(1, 1, k) - z1) / (z2 - z1);
      else thfrc(k) = 0.0;
    }
    const Real qvsfc = -1.2e-8;
    z1 = 300.0; z2 = 500.0;
    for (int k = 1; k <= nk; ++k) {
      if (zh(1, 1, k) <= z1) qvfrc(k) = -1.2e-8;
      else if (zh(1, 1, k) <= z2) qvfrc(k) = qvsfc + (0.0 - qvsfc) * (zh(1, 1, k) - z1) / (z2 - z1);
      else qvfrc(k) = 0.0;
    }
    for (int k = 1; k <= nk; ++k) {
      ug(k) = -10.0 + 0.0018 * zh(1, 1, k);
      vg(k) = 0.0;
    }
  }

  if (tc == 4 || tc == 5) {
    // stratocumulus (Stevens et al 2005; Ackerman et al 2009)
    for (int k = 2; k <= nk; ++k) wprof(k) = -3.75e-6 * zf(1, 1, k);
  }

  if (tc == 6 || tc == 10 || tc == 15) {
    // hurricane boundary layer (Bryan et al, 2017, BLM)
    const Real wmin = -0.01, z1 = 1000.0, z2 = 4000.0;
    for (int k = 1; k <= nk; ++k) {
      dvdr(k) = o.hurr_vgpl * (vg(k) / o.hurr_rad);
      if (zf(1, 1, k) <= z1) wprof(k) = 0.0 + (wmin - 0.0) * (zf(1, 1, k) - 0.0) / (z1 - 0.0);
      else if (zf(1, 1, k) <= z2) wprof(k) = wmin + (0.0 - wmin) * (zf(1, 1, k) - z1) / (z2 - z1);
      else wprof(k) = 0.0;
    }
  }

  if (tc == 7) {
    // RICO, precipitating shallow Cu (VanZanten et al 2011, JAMES)
    Real wmin = -0.005, z1 = 2260.0;
    for (int k = 1; k <= nk; ++k) {
      if (zf(1, 1, k) <= z1) wprof(k) = 0.0 + (wmin - 0.0) * (zf(1, 1, k) - 0.0) / (z1 - 0.0);
      else wprof(k) = wmin;
    }
    const Real thsfc = -2.5 / (3600.0 * 24.0);
    for (int k = 1; k <= nk; ++k) thfrc(k) = thsfc;
    const Real qv1 = -1.0 / 1000.0 / (3600.0 * 24.0);
    const Real qv2 = 0.3456 / 1000.0 / (3600.0 * 24.0);
    z1 = 2980.0;
    for (int k = 1; k <= nk; ++k) {
      if (zh(1, 1, k) <= z1) qvfrc(k) = qv1 + (qv2 - qv1) * (zh(1, 1, k) - 0.0) / (z1 - 0.0);
      else qvfrc(k) = qv2;
    }
    for (int k = 1; k <= nk; ++k) {
      ug(k) = u0(1, 1, k);
      vg(k) = v0(1, 1, k);
    }
  }

  if (tc == 11) {
    // SAS NCAR PBL intercomparison (whole-array assignments)
    for (int k = kb; k <= ke; ++k) {
      ug(k) = u0(1, 1, 1) + umove;
      vg(k) = v0(1, 1, 1) + vmove;
    }
    for (int k = 2; k <= nk; ++k) wprof(k) = -9.0e-6 * zf(1, 1, k);
    for (int k = kb; k <= ke; ++k) {
      ufrc(k) = 0.0; vfrc(k) = 0.0; thfrc(k) = 0.0; qvfrc(k) = 0.0; wprof(k) = 0.0;
    }
  }

  if (tc == 12) {
    // Martinuzzi and Tropea (1993, JFE) wind tunnel case
    for (int k = kb; k <= ke; ++k) { ulspg(k) = 36.0; vlspg(k) = 0.0; }
  }

  if (tc == 13) {
    // UF Boundary Layer Wind Tunnel case (u0 = 18 m/s: init/cm1k_init.cpp)
    if (o.isnd != 7)
      for (int k = kb; k <= ke; ++k) { ug(k) = 18.0; vg(k) = 0.0; }
    for (int k = kb; k <= ke; ++k) { ulspg(k) = 0.000545; vlspg(k) = 0.0; }
  }

  for (int k = kb; k <= ke; ++k) {
    ug(k) = ug(k) - umove;
    vg(k) = vg(k) - vmove;
  }
  wprof(1) = 0.0;
  wprof(nk + 1) = 0.0;

  f.wprof.deep_copy_from(wprof); f.ufrc.deep_copy_from(ufrc); f.vfrc.deep_copy_from(vfrc);
  f.thfrc.deep_copy_from(thfrc); f.qvfrc.deep_copy_from(qvfrc); f.ug.deep_copy_from(ug);
  f.vg.deep_copy_from(vg); f.dvdr.deep_copy_from(dvdr); f.ulspg.deep_copy_from(ulspg);
  f.vlspg.deep_copy_from(vlspg);
  if (f.frad.is_allocated()) { f.frad.fill(0.0); f.thraten.fill(0.0); f.zir.fill(0.0); }
}

//------------------------------------------------------------------------------
// solve1.F: large-scale pressure gradient (lspgrad = 2, 3, 4)
//------------------------------------------------------------------------------

void lspgrad(const Options& o, const Bounds& b, const Forcing& f, const FArray3& uten1, const FArray3& vten1) {
  if (o.lspgrad < 2) return;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const auto ulspg = f.ulspg, vlspg = f.vlspg, ug = f.ug, vg = f.vg;
  const Real fcor = o.fcor, umove = o.umove, vmove = o.vmove, hurr_rad = o.hurr_rad;
  using Range = Kokkos::RangePolicy<ExeSpace, Kokkos::IndexType<int>>;
  if (o.lspgrad == 2) {
    // geostrophic balance with ug, vg
    Kokkos::parallel_for("lspgrad2", Range(1, nk + 1), KOKKOS_LAMBDA(int k) {
      ulspg(k) = -fcor * (vg(k) + vmove);
      vlspg(k) = fcor * (ug(k) + umove);
    });
  } else if (o.lspgrad == 3) {
    // gradient-wind balance (Bryan et al, 2017, BLM, eqn 10)
    const Real ca = std::cos(-o.hurr_angle), sa = std::sin(-o.hurr_angle);
    Kokkos::parallel_for("lspgrad3", Range(1, nk + 1), KOKKOS_LAMBDA(int k) {
      const Real fr = -fcor * vg(k) - vg(k) * vg(k) / hurr_rad;
      ulspg(k) = fr * ca;
      vlspg(k) = fr * sa;
    });
  }
  // lspgrad = 4: ulspg, vlspg as set in base.F
  Kokkos::parallel_for("lspgrad_add", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    uten1(i, j, k) = uten1(i, j, k) + ulspg(k);
    vten1(i, j, k) = vten1(i, j, k) + vlspg(k);
  });
}

//------------------------------------------------------------------------------
// testcase_simple_phys.F: domain averages
//------------------------------------------------------------------------------

void get_avg_uvt(const Options& o, const Grid& g, const FArray3& th0, const FArray3& ua, const FArray3& va,
                 const FArray3& tha, const FArray4* qa, Averages& a) {
  const auto& B = g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk, nkk = nk + 2;
  const int nq = qa ? o.numq : 0;
  Kokkos::fence();
  auto hu = ua.to_host(), hv = va.to_host(), ht = tha.to_host(), h0 = th0.to_host();
  auto ruh = g.ruh.to_host(), ruf = g.ruf.to_host(), rvh = g.rvh.to_host(), rvf = g.rvf.to_host();
  auto xf = g.xf.to_host(), yf = g.yf.to_host();
  a.uavg.assign(nkk, 0.0); a.vavg.assign(nkk, 0.0); a.thavg.assign(nkk, 0.0);
  if (qa) a.qavg.assign(size_t(nkk) * size_t(std::max(nq, 1)), 0.0);
  // 180612: area-weighted average
  const Real tem = g.dx * g.dy;
  std::vector<double> cq(std::max(nq, 1));
  FArray<4, HostSpace> hq;
  if (qa) hq = qa->to_host();
  for (int k = 1; k <= nk; ++k) {
    double c1 = 0.0, c2 = 0.0, c3 = 0.0;
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        c1 = c1 + hu(i, j, k) * tem * ruf(i) * rvh(j);
        c2 = c2 + hv(i, j, k) * tem * ruh(i) * rvf(j);
        c3 = c3 + (h0(i, j, k) + ht(i, j, k)) * tem * ruh(i) * rvh(j);
      }
    for (int n = 1; n <= nq; ++n) {
      double s = 0.0;
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) s = s + hq(i, j, k, n) * tem * ruh(i) * rvh(j);
      cq[n - 1] = s;
    }
    // (serial: maxx-minx = xf(nx+1)-xf(1))
    const Real dxx = xf(ni + 1) - xf(1), dyy = yf(nj + 1) - yf(1);
    const double temd = 1.0 / (double(dxx) * double(dyy));
    a.uavg[k] = c1 * temd;
    a.vavg[k] = c2 * temd;
    a.thavg[k] = c3 * temd;
    for (int n = 1; n <= nq; ++n) a.qavg[k + size_t(nkk) * (n - 1)] = cq[n - 1] * temd;
  }
}

//------------------------------------------------------------------------------
// solve1.F: testcase 10 nudging
//------------------------------------------------------------------------------

void tqnudge(const Options& o, const Grid& g, const BaseState& bs, double mtime, const FArray3& ua,
             const FArray3& va, const FArray3& tha, const FArray4& qa, const FArray3& thten1,
             const FArray4& qten) {
  const auto& B = g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk, nkk = nk + 2;
  // pre-RK: get_avg_uvtq (imoist = 1) or get_avg_uvt
  Averages a;
  get_avg_uvt(o, g, bs.th0, ua, va, tha, o.imoist == 1 ? &qa : nullptr, a);
  // ramp-up nudging at start of simulation
  Real nudgefac = (mtime - 0.0) / (900.0 - 0.0);
  nudgefac = std::max(0.0, nudgefac);
  nudgefac = std::min(1.0, nudgefac);
  const Real taunudge = 300.0;
  if (!(std::abs(nudgefac) > 0.001)) return;
  const Real tem = nudgefac / taunudge;
  auto th0 = bs.th0.to_host(), qv0 = bs.qv0.to_host();
  FArray1 tn("tn", {1}, {nk}), qn("qn", {1}, {nk});
  auto htn = tn.to_host(), hqn = qn.to_host();
  for (int k = 1; k <= nk; ++k) {
    htn(k) = -(a.thavg[k] - th0(1, 1, k)) * tem;
    hqn(k) = (o.imoist == 1) ? -(a.qavg[k + size_t(nkk) * (o.nqv - 1)] - qv0(1, 1, k)) * tem : 0.0;
  }
  tn.deep_copy_from(htn);
  qn.deep_copy_from(hqn);
  const bool moist = (o.imoist == 1);
  const int nqv = o.nqv;
  Kokkos::parallel_for("tqnudge", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    thten1(i, j, k) = thten1(i, j, k) + tn(k);
    if (moist) qten(i, j, k, nqv) = qten(i, j, k, nqv) + qn(k);
  });
}

//------------------------------------------------------------------------------
// testcase_simple_phys.F
//------------------------------------------------------------------------------

namespace {
// simplerad (Stevens et al 2005), one column per (i,j); ql, qt, dum1, dum2
// kept in registers-sized arrays on the stack (nk is small); the Fortran's
// scratch values at k = nk+1 (ql, qt: never set there) are taken as zero.
struct SimpleRad {
  int nk, nqv, nql1, nql2;
  Real cp;
  FArray3 rho0, zh, zf, frad, zir;
  FArray4 qa;
  FArray3 ql, qt, dum1, dum2;
  KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int& bad) const {
    dum2(i, j, 1) = 0.0;
    dum1(i, j, nk + 1) = 0.0;
    int kzi = 0;
    for (int k = 1; k <= nk; ++k) {
      ql(i, j, k) = 0.0;
      for (int n = nql1; n <= nql2; ++n) ql(i, j, k) = ql(i, j, k) + qa(i, j, k, n);
      qt(i, j, k) = qa(i, j, k, nqv) + ql(i, j, k);
      if (qt(i, j, k) > qcrit) kzi = k;
      dum2(i, j, k + 1) = dum2(i, j, k) + kappa * rho0(i, j, k) * ql(i, j, k) * (zf(i, j, k + 1) - zf(i, j, k));
    }
    if (kzi < 1 || kzi > nk - 1) bad += 1;   // the Fortran reads unset scratch values
    const int k = kzi < 1 ? 1 : (kzi > nk - 1 ? nk - 1 : kzi);
    zir(i, j, 1) = zh(i, j, k) + (zh(i, j, k + 1) - zh(i, j, k)) * (qcrit - qt(i, j, k)) / (qt(i, j, k + 1) - qt(i, j, k));
    for (int kk = nk + 1; kk >= 1; --kk) {
      const int kp = (nk + 1 < kk + 1) ? nk + 1 : kk + 1;
      dum1(i, j, kk) = dum1(i, j, kp) + kappa * rho0(i, j, kk) * ql(i, j, kk) * (zf(i, j, kk + 1) - zf(i, j, kk));
      const Real fr1 = f0 * Kokkos::exp(-dum1(i, j, kk));
      const Real fr2 = f1 * Kokkos::exp(-dum2(i, j, kk));
      Real fr3;
      if (zf(i, j, kk) < zir(i, j, 1)) {
        fr3 = 0.0;
      } else {
        fr3 = rhoi * cp * bigd * alphaz * (0.25 * (Kokkos::pow(zf(i, j, kk) - zir(i, j, 1), 1.333333))
                                           + zir(i, j, 1) * (Kokkos::pow(zf(i, j, kk) - zir(i, j, 1), 0.333333)));
      }
      frad(i, j, kk) = fr1 + fr2 + fr3;
    }
  }
};
} // namespace

void simple_phys(const Options& o, const Constants& c, const Grid& g, const BaseState& bs, double mtime,
                 const FArray3& ua, const FArray3& va, const FArray3& tha, const FArray4& qa, const Forcing& f,
                 const FArray3& uten1, const FArray3& vten1, const FArray3& thten1, const FArray4& qten) {
  if (o.testcase < 1) return;
  const auto& B = g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;

  // nonprecipitating / drizzling stratoCu: simple radiation
  if (o.simple_rad()) {
    SimpleRad sr{nk, o.nqv, o.nql1, o.nql2, c.cp, bs.rho0, g.zh, g.zf, f.frad, f.zir, qa, f.w_ql, f.w_qt, f.w_d1, f.w_d2};
    int bad = 0;
    Kokkos::parallel_reduce("simplerad", box2(1, ni, 1, nj), sr, bad);
    if (bad > 0)
      fail("simplerad: the qt = 0.008 crossing (cloud top) must lie between k = 1 and nk (CM1 reads unset values)");
    const auto frad = f.frad, thraten = f.thraten, mh = g.mh;
    const Real rdz = g.rdz, cp = c.cp;
    Kokkos::parallel_for("simplerad_th", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      thraten(i, j, k) = -(frad(i, j, k + 1) - frad(i, j, k)) * rdz * mh(i, j, k) / (cp * rhoi);
      thten1(i, j, k) = thten1(i, j, k) + thraten(i, j, k);
    });
  }

  // hurricane boundary layer: mesoscale tendencies (lspgrad = 3)
  if (o.lspgrad == 3) {
    Averages a;
    get_avg_uvt(o, g, bs.th0, ua, va, tha, nullptr, a);
    const Real tem = 1.0 / o.hurr_rad;
    auto vg = f.vg.to_host(), dvdr = f.dvdr.to_host(), ufrc = f.ufrc.to_host(), vfrc = f.vfrc.to_host();
    const Real ang = -o.hurr_angle;
    for (int k = 1; k <= nk; ++k) {
      const Real uu = a.vavg[k] * std::sin(ang) + a.uavg[k] * std::cos(ang);
      const Real vv = a.vavg[k] * std::cos(ang) - a.uavg[k] * std::sin(ang);
      const Real fr = +uu * uu * tem + vv * vg(k) * tem;
      const Real ft = -uu * dvdr(k) - uu * vg(k) * tem;
      ufrc(k) = fr * std::cos(ang) - ft * std::sin(ang);
      vfrc(k) = fr * std::sin(ang) + ft * std::cos(ang);
    }
    f.ufrc.deep_copy_from(ufrc);
    f.vfrc.deep_copy_from(vfrc);
  }

  // SGP shallow cumulus (Brown et al, 2002, QJRMS): prescribed tendencies
  if (o.testcase == 14) {
    Real tt1 = 0, tt2 = 1, at1 = 0, rt1 = 0, qt1 = 0, at2 = 0, rt2 = 0, qt2 = 0;
    if (mtime <= 3.0 * 3600.0) {
      tt1 = 0.0 * 3600.0; tt2 = 3.0 * 3600.0;
      at1 = 0.000; rt1 = -0.125; qt1 = 0.080; at2 = 0.000; rt2 = 0.000; qt2 = 0.020;
    } else if (mtime <= 6.0 * 3600.0) {
      tt1 = 3.0 * 3600.0; tt2 = 6.0 * 3600.0;
      at1 = 0.000; rt1 = 0.000; qt1 = 0.020; at2 = 0.000; rt2 = 0.000; qt2 = -0.040;
    } else if (mtime <= 9.0 * 3600.0) {
      tt1 = 6.0 * 3600.0; tt2 = 9.0 * 3600.0;
      at1 = 0.000; rt1 = 0.000; qt1 = -0.040; at2 = -0.080; rt2 = 0.000; qt2 = -0.100;
    } else if (mtime <= 12.0 * 3600.0) {
      tt1 = 9.0 * 3600.0; tt2 = 12.0 * 3600.0;
      at1 = -0.080; rt1 = 0.000; qt1 = -0.100; at2 = -0.160; rt2 = 0.000; qt2 = -0.160;
    } else if (mtime <= 14.5 * 3600.0) {
      tt1 = 12.0 * 3600.0; tt2 = 14.5 * 3600.0;
      at1 = -0.160; rt1 = 0.000; qt1 = -0.160; at2 = -0.160; rt2 = -0.100; qt2 = -0.300;
    } else if (mtime <= 99.0 * 3600.0) {
      tt1 = 14.5 * 3600.0; tt2 = 99.0 * 3600.0;
      at1 = -0.160; rt1 = -0.100; qt1 = -0.300; at2 = -0.160; rt2 = -0.100; qt2 = -0.300;
    } else {
      fail("testcase=14: the SGP forcing is only defined up to 99 h");
    }
    const Real at = at1 + (at2 - at1) * (mtime - tt1) / (tt2 - tt1);
    const Real rt = rt1 + (rt2 - rt1) * (mtime - tt1) / (tt2 - tt1);
    const Real qt = qt1 + (qt2 - qt1) * (mtime - tt1) / (tt2 - tt1);
    auto zh = g.zh.to_host();
    auto thfrc = f.thfrc.to_host(), qvfrc = f.qvfrc.to_host();
    for (int k = 1; k <= nk; ++k) {
      Real fac = 1.0 + (0.0 - 1.0) * (zh(1, 1, k) - 1000.0) / (2000.0 - 1000.0);
      fac = std::max(0.0, fac);
      fac = std::min(1.0, fac);
      thfrc(k) = fac * (at + rt) / 3600.0;
      qvfrc(k) = fac * qt * 0.001 / 3600.0;
    }
    f.thfrc.deep_copy_from(thfrc);
    f.qvfrc.deep_copy_from(qvfrc);
  }

  // add tendencies
  const auto ufrc = f.ufrc, vfrc = f.vfrc, thfrc = f.thfrc, qvfrc = f.qvfrc;
  Kokkos::parallel_for("tc_uten", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    uten1(i, j, k) = uten1(i, j, k) + ufrc(k);
  });
  Kokkos::parallel_for("tc_vten", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    vten1(i, j, k) = vten1(i, j, k) + vfrc(k);
  });
  const bool moist = (o.imoist == 1);
  const int nqv = o.nqv;
  Kokkos::parallel_for("tc_thten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    thten1(i, j, k) = thten1(i, j, k) + thfrc(k);
    if (moist) qten(i, j, k, nqv) = qten(i, j, k, nqv) + qvfrc(k);
  });
}

} // namespace cm1k::testcase
