// Port of CM1's simple PBL scheme (cm1setup=2, ipbl=2) and the horizontal
// Smagorinsky-like scheme (horizturb=1). See cm1k_pbl.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// x**2 is x*x, x**3 is (x*x)*x, x**4 is (x*x)*(x*x), x**(-2) is 1/(x*x),
// x**(-1) is 1/x, and Fortran's max of two non-NaN numbers is fmax2.
#include "physics/pbl/cm1k_pbl.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::pbl {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }

constexpr Real prandtl = 1.0;
constexpr Real prinv = 1.0 / prandtl;
constexpr Real dmin = 1.0e-10;

// turbparam_horiz's tlh: over water (sea level only; zs = 0 without
// terrain) a function of surface pressure, else l_h
KOKKOS_INLINE_FUNCTION Real tlh_of(Real xland, Real psfc, Real l_h, Real lhref1, Real lhref2) {
  const Real zs = 0.0;
  if ((xland > 1.5) && (zs < 1.0)) return lhref2 + (lhref1 - lhref2) * (psfc - 90000.0) / (101500.0 - 90000.0);
  return l_h;
}

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
  o.ipbl = nl.geti("ipbl", 0);
  o.horizturb = nl.geti("horizturb", 0);
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.idiss = nl.geti("idiss", 0);
  o.imoist = nl.geti("imoist", 0);
  o.ptype = nl.geti("ptype", 0);
  o.terrain_flag = nl.getb("terrain_flag", false);
  o.axisymm = nl.geti("axisymm", 0);
  o.l_h = nl.getr("l_h", 100.0);
  o.lhref1 = nl.getr("lhref1", 100.0);
  o.lhref2 = nl.getr("lhref2", 1000.0);
  o.l_inf = nl.getr("l_inf", 75.0);
  // param.F: ipbl = 1,3,4,5,6 need l_inf = 0
  const bool idopbl = (o.cm1setup == 2 || o.cm1setup == 4);
  if (idopbl && (o.ipbl == 1 || o.ipbl == 3 || o.ipbl == 4 || o.ipbl == 5 || o.ipbl == 6) &&
      Kokkos::fabs(o.l_inf) > 1.0e-6)
    o.l_inf = 0.0;
  o.use_pbl = idopbl && o.ipbl >= 1;
  // moisture layout (param.F)
  if (o.imoist == 1 && o.ptype == 0) { o.nqv = 1; }
  if (o.imoist == 1 && o.ptype == 1) { o.nqv = 1; o.nqc = 2; o.nqi = 0; }
  return o;
}

void Options::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k pbl: " + m); };
  if (cm1setup != 2) fail("cm1setup=" + std::to_string(cm1setup) + " is not ported (only 2)");
  if (ipbl != 2 && ipbl != 4 && ipbl != 5)
    fail("ipbl=" + std::to_string(ipbl) + " is not ported (only 2, the simple PBL scheme, and 4, 5, MYNN)");
  if (horizturb != 0 && horizturb != 1) fail("horizturb must be 0 or 1");
  if (ipbl == 2 && sfcmodel != 1) fail("ipbl=2 needs the surface layer sfcmodel=1 (the only one ported)");
  if ((ipbl == 4 || ipbl == 5) && sfcmodel != 6)
    fail("ipbl=" + std::to_string(ipbl) + " needs the MYNN surface layer sfcmodel=6 (the only one ported)");
  if (idiss != 0 && idiss != 1) fail("idiss must be 0 or 1");
  if (imoist == 1 && ptype != 1) fail("imoist=1 with ptype=" + std::to_string(ptype) + " is not ported (only 1)");
  if (terrain_flag) fail("terrain is not ported");
  if (axisymm != 0) fail("axisymm=1 is not ported");
}

//------------------------------------------------------------------------------
// Tendencies
//------------------------------------------------------------------------------

void Tendencies::allocate(const Bounds& b) {
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
  thpten = FArray3("thpten", lo, hi); qvpten = FArray3("qvpten", lo, hi);
  qcpten = FArray3("qcpten", lo, hi); qipten = FArray3("qipten", lo, hi);
  upten = FArray3("upten", lo, hi); vpten = FArray3("vpten", lo, hi);
}

std::vector<std::pair<std::string, FArray3>> Tendencies::dumped() const {
  return {{"thpten", thpten}, {"qvpten", qvpten}, {"qcpten", qcpten}, {"upten", upten}, {"vpten", vpten}};
}

//------------------------------------------------------------------------------
// turb.F
//------------------------------------------------------------------------------

void turbparam_horiz(const turb::Options& to, const Options& o, const Bounds& b, const FArray3& defh,
                     const FArray3& kmh, const FArray3& khh, const FArray3& dissten,
                     const sfc::SfcState& s) {
  if (!(o.l_h > 1.0e-12 || o.lhref1 > 1.0e-12 || o.lhref2 > 1.0e-12)) return;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real l_h = o.l_h, lhref1 = o.lhref1, lhref2 = o.lhref2;
  const auto xland = s.xland, psfc = s.psfc;
  if (nk >= 2)
    Kokkos::parallel_for("pbl_kmh", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real t = tlh_of(xland(i, j, 1), psfc(i, j, 1), l_h, lhref1, lhref2);
      kmh(i, j, k) = (t * t) * Kokkos::sqrt(fmax2(defh(i, j, k), dmin));
    });
  turb::bcw(to, b, kmh, 1);
  // extrapolate
  Kokkos::parallel_for("pbl_kmh1", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
    kmh(i, j, 1) = fmax2(0.0, 2.0 * kmh(i, j, 2) - kmh(i, j, 3));
  });
  Kokkos::parallel_for("pbl_khh", box3(0, ni + 1, 0, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    khh(i, j, k) = kmh(i, j, k) * prinv;
  });
  if (o.idiss == 1 && nk >= 2)
    Kokkos::parallel_for("pbl_dissh", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real t = tlh_of(xland(i, j, 1), psfc(i, j, 1), l_h, lhref1, lhref2);
      const Real km = kmh(i, j, k);
      dissten(i, j, k) = dissten(i, j, k) + ((km * km) * km) / ((t * t) * (t * t));
    });
}

void turbparam_vert(const turb::Options& to, const Options& o, const Grid& g, const FArray3& nm,
                    const FArray3& defv, const FArray3& kmv, const FArray3& khv, const FArray3& dissten,
                    const sfc::SfcState& s, const FArray3& lvz) {
  if (!(o.l_inf > 1.0e-12)) return;
  const auto& b = g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real a = 1.0e-6 + o.l_inf;
  const Real rlinf = 1.0 / (a * a);
  const auto zf = g.zf;
  const auto zntmp = s.zntmp, ust = s.ust;
  if (nk >= 2)
    Kokkos::parallel_for("pbl_kmv", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      constexpr Real karman = phys::karman;
      const Real x = karman * ((zf(i, j, k) - zf(i, j, 1)) + zntmp(i, j, 1));
      lvz(i, j, k) = Kokkos::sqrt(1.0 / (rlinf + 1.0 / (x * x)));
      kmv(i, j, k) = (lvz(i, j, k) * lvz(i, j, k)) * Kokkos::sqrt(fmax2(defv(i, j, k) - nm(i, j, k) * prinv, dmin));
    });
  turb::bcw(to, b, kmv, 1);
  Kokkos::parallel_for("pbl_khv", box3(0, ni + 1, 0, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    khv(i, j, k) = kmv(i, j, k) * prinv;
  });
  if (o.idiss == 1 && nk >= 2)
    Kokkos::parallel_for("pbl_dissv", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real km = kmv(i, j, k), l = lvz(i, j, k);
      dissten(i, j, k) = dissten(i, j, k) + ((km * km) * km) / ((l * l) * (l * l));
    });
  if (to.bbc == 3) {
    // cm1r18: surface
    Kokkos::parallel_for("pbl_kmv1", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      constexpr Real karman = phys::karman;
      kmv(i, j, 1) = karman * zntmp(i, j, 1) * ust(i, j, 1);
    });
    turb::bc2d_level(to, b, kmv, 1);
    Kokkos::parallel_for("pbl_khv1", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      khv(i, j, 1) = kmv(i, j, 1) * prinv;
    });
  }
}

void br09(const turb::Options& to, const Options& o, const Grid& g, const BaseState& base, Real dt,
          const FArray3& ua, const FArray3& va, const FArray3& wa, const FArray3& tha, const FArray4& qa,
          const FArray3& rho, const FArray3& rr, const FArray3& rf, const FArray3& kmv,
          const FArray3& khv, const FArray3& t13, const FArray3& t23, const sfc::SfcState& s,
          const Tendencies& pt, const turb::Work& w) {
  const auto& b = g.b;
  // thten, thten1 of turb.F: the vimpl coefficients for turbsz
  const FArray3 thten = w.d7, thten1 = w.d8;
  if (to.doimpl == 1) turb::vimpl_coefs(b, khv, g.mf, rf, g.mh, rr, thten, thten1);
  const auto th0 = base.th0, sadv = w.sadv;
  Kokkos::parallel_for("pbl_sadv", box3(0, b.ni + 1, 0, b.nj + 1, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    constexpr Real th0r = phys::th0r;
    sadv(i, j, k) = (th0(i, j, k) - th0r) + tha(i, j, k);
  });
  turb::turbsz(to, g, 1, dt, rho, rr, rf, sadv, khv, thten, thten1, pt.thpten, w.d1, w.d2, w.d3, s.thflux);
  if (o.imoist == 1)
    turb::turbsz(to, g, 1, dt, rho, rr, rf, slice(qa, o.nqv), khv, thten, thten1, pt.qvpten, w.d1, w.d2, w.d3,
                 s.qvflux);
  if (o.imoist == 1 && o.nqc >= 1)
    turb::turbsz(to, g, 0, dt, rho, rr, rf, slice(qa, o.nqc), khv, thten, thten1, pt.qcpten, w.d1, w.d2, w.d3);
  if (o.imoist == 1 && o.nqi >= 1)
    turb::turbsz(to, g, 0, dt, rho, rr, rf, slice(qa, o.nqi), khv, thten, thten1, pt.qipten, w.d1, w.d2, w.d3);
  turb::turbuz(to, g, dt, rho, rf, ua, wa, t13, kmv, pt.upten, w.d1, w.d2, w.d3, thten);
  turb::turbvz(to, g, dt, rho, rf, va, wa, t23, kmv, pt.vpten, w.d1, w.d2, w.d3, thten);
}

//------------------------------------------------------------------------------
// solve1.F
//------------------------------------------------------------------------------

void add_theta_tendency(const Bounds& b, const Tendencies& pt, const FArray3& thten1) {
  const auto thpten = pt.thpten;
  Kokkos::parallel_for("pbl_thten1", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    thten1(i, j, k) = thten1(i, j, k) + thpten(i, j, k);
  });
}

void add_tendencies(const Options& o, const Bounds& b, const Tendencies& pt, const FArray3& uten1,
                    const FArray3& vten1, const FArray4& qten) {
  const auto upten = pt.upten, vpten = pt.vpten;
  Kokkos::parallel_for("pbl_uten1", box3(1, b.ni + 1, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    uten1(i, j, k) = uten1(i, j, k) + 0.5 * (upten(i - 1, j, k) + upten(i, j, k));
  });
  Kokkos::parallel_for("pbl_vten1", box3(1, b.ni, 1, b.nj + 1, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    vten1(i, j, k) = vten1(i, j, k) + 0.5 * (vpten(i, j - 1, k) + vpten(i, j, k));
  });
  if (o.imoist != 1) return;
  std::vector<std::pair<int, FArray3>> adds;
  adds.emplace_back(o.nqv, pt.qvpten);
  if (o.nqc >= 1) adds.emplace_back(o.nqc, pt.qcpten);
  if (o.nqi >= 1) adds.emplace_back(o.nqi, pt.qipten);
  for (const auto& np : adds) {
    const auto q = slice(qten, np.first);
    const auto p = np.second;
    Kokkos::parallel_for("pbl_qten", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      q(i, j, k) = q(i, j, k) + p(i, j, k);
    });
  }
}

} // namespace cm1k::pbl
