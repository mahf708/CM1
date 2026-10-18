// Ports of src/sfclay.F (sfcmodel=2) and src/sf_sfclayrev.F (sfcmodel=3).
// See cm1k_sfclay.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// real**real is pow, x**2 (integer or 2.) is x*x, alog/log is log, and
// Fortran's max/min (amax1/amin1) of two non-NaN numbers is fmax2/fmin2.
// The Fortran loops over i in separate passes (5, 30, 50, 60, 260, 320, 330,
// 360, 370, 400, ...) only couple points through the local scalars Z0T/Z0Q
// of SFCLAY1D (see the header), so each column runs all passes in order.
#include "physics/sfclay/cm1k_sfclay.hpp"

#include <stdexcept>
#include <string>

#if defined(CM1K_HAVE_LIBMVEC) && defined(__x86_64__)
#include <emmintrin.h>
// glibc libmvec SSE2 variants (see sfclayinit below).
extern "C" {
__m128d _ZGVbN2vv_pow(__m128d, __m128d);
__m128d _ZGVbN2v_log(__m128d);
__m128d _ZGVbN2v_atan(__m128d);
}
#endif

namespace cm1k::sfclay {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// module parameters (both modules)
constexpr Real vconvc = 1.;
constexpr Real czo = 0.0185;
constexpr Real ozo = 1.59E-5;
// SFCLAY1D / sf_sfclayrev_run parameters
constexpr Real xka = 2.4E-5;
constexpr Real prt = 1.;

using TV = Kokkos::View<Real*, Device>;

//------------------------------------------------------------------------------
// sf_sfclayrev.F functions
//------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION Real psim_stable_full(Real zolf) {
  return -6.1 * Kokkos::log(zolf + Kokkos::pow(1 + Kokkos::pow(zolf, 2.5), 1. / 2.5));
}
KOKKOS_INLINE_FUNCTION Real psih_stable_full(Real zolf) {
  return -5.3 * Kokkos::log(zolf + Kokkos::pow(1 + Kokkos::pow(zolf, 1.1), 1. / 1.1));
}
KOKKOS_INLINE_FUNCTION Real psim_unstable_full(Real zolf) {
  const Real x = Kokkos::pow(1. - 16. * zolf, .25);
  const Real psimk = 2 * Kokkos::log(0.5 * (1 + x)) + Kokkos::log(0.5 * (1 + x * x)) - 2. * Kokkos::atan(x) +
                     2. * Kokkos::atan(1.);
  const Real ym = Kokkos::pow(1. - 10. * zolf, 0.33);
  const Real psimc = (3. / 2.) * Kokkos::log((ym * ym + ym + 1.) / 3.) -
                     Kokkos::sqrt(3.) * Kokkos::atan((2. * ym + 1) / Kokkos::sqrt(3.)) +
                     4. * Kokkos::atan(1.) / Kokkos::sqrt(3.);
  return (psimk + zolf * zolf * (psimc)) / (1 + zolf * zolf);
}
KOKKOS_INLINE_FUNCTION Real psih_unstable_full(Real zolf) {
  const Real y = Kokkos::pow(1. - 16. * zolf, .5);
  const Real psihk = 2. * Kokkos::log((1 + y) / 2.);
  const Real yh = Kokkos::pow(1. - 34. * zolf, 0.33);
  const Real psihc = (3. / 2.) * Kokkos::log((yh * yh + yh + 1.) / 3.) -
                     Kokkos::sqrt(3.) * Kokkos::atan((2. * yh + 1) / Kokkos::sqrt(3.)) +
                     4. * Kokkos::atan(1.) / Kokkos::sqrt(3.);
  return (psihk + zolf * zolf * (psihc)) / (1 + zolf * zolf);
}

// The look-up functions; tab is the 0..1000 table, full the closed form.
struct RevTab {
  TV psim_stab, psim_unstab, psih_stab, psih_unstab;

  KOKKOS_INLINE_FUNCTION Real psim_stable(Real zolf) const {
    const int nzol = int(zolf * 100.);
    const Real rzol = zolf * 100. - nzol;
    if (nzol + 1 < 1000) return psim_stab(nzol) + rzol * (psim_stab(nzol + 1) - psim_stab(nzol));
    return psim_stable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real psih_stable(Real zolf) const {
    const int nzol = int(zolf * 100.);
    const Real rzol = zolf * 100. - nzol;
    if (nzol + 1 < 1000) return psih_stab(nzol) + rzol * (psih_stab(nzol + 1) - psih_stab(nzol));
    return psih_stable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real psim_unstable(Real zolf) const {
    const int nzol = int(-zolf * 100.);
    const Real rzol = -zolf * 100. - nzol;
    if (nzol + 1 < 1000) return psim_unstab(nzol) + rzol * (psim_unstab(nzol + 1) - psim_unstab(nzol));
    return psim_unstable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real psih_unstable(Real zolf) const {
    const int nzol = int(-zolf * 100.);
    const Real rzol = -zolf * 100. - nzol;
    if (nzol + 1 < 1000) return psih_unstab(nzol) + rzol * (psih_unstab(nzol + 1) - psih_unstab(nzol));
    return psih_unstable_full(zolf);
  }

  // zolri2: zol2 is intent(inout) (reset to 0 when its sign differs from ri2)
  KOKKOS_INLINE_FUNCTION Real zolri2(Real& zol2, Real ri2, Real z, Real z0) const {
    if (zol2 * ri2 < 0.) zol2 = 0.;   // limit zol2 - must be same sign as ri2
    const Real zol20 = zol2 * z0 / z;  // z0/L
    const Real zol3 = zol2 + zol20;    // (z+z0)/L
    Real psih2, psix2;
    if (ri2 < 0) {
      psix2 = Kokkos::log((z + z0) / z0) - (psim_unstable(zol3) - psim_unstable(zol20));
      psih2 = Kokkos::log((z + z0) / z0) - (psih_unstable(zol3) - psih_unstable(zol20));
    } else {
      psix2 = Kokkos::log((z + z0) / z0) - (psim_stable(zol3) - psim_stable(zol20));
      psih2 = Kokkos::log((z + z0) / z0) - (psih_stable(zol3) - psih_stable(zol20));
    }
    return zol2 * psih2 / (psix2 * psix2) - ri2;
  }

  // zolri. The Fortran function result is undefined when it returns before
  // the first iteration assigns it (fx1 == fx2 on entry); here it is 0.
  KOKKOS_INLINE_FUNCTION Real zolri(Real ri, Real z, Real z0) const {
    Real x1, x2;
    if (ri < 0.) { x1 = -5.; x2 = 0.; }
    else         { x1 = 0.;  x2 = 5.; }
    Real result = 0.;
    Real fx1 = zolri2(x1, ri, z, z0);
    Real fx2 = zolri2(x2, ri, z, z0);
    int iter = 0;
    while (Kokkos::fabs(x1 - x2) > 0.01) {
      if (iter == 10) return result;
      // check added for potential divide by zero (2019/11)
      if (fx1 == fx2) return result;
      if (Kokkos::fabs(fx2) < Kokkos::fabs(fx1)) {
        x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1);
        fx1 = zolri2(x1, ri, z, z0);
        result = x1;
      } else {
        x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1);
        fx2 = zolri2(x2, ri, z, z0);
        result = x2;
      }
      iter = iter + 1;
    }
    return result;
  }

  // the psih/psih10/psih2 recomputation sf_sfclayrev_run repeats for z0t/z0q
  KOKKOS_INLINE_FUNCTION void psih_for(Real zol, Real za, Real z0x, Real& psih, Real& psih10, Real& psih2) const {
    const Real zolzz = zol * (za + z0x) / za;  // (z+z0x)/L
    const Real zol10 = zol * (10. + z0x) / za; // (10+z0x)/L
    const Real zol2 = zol * (2. + z0x) / za;   // (2+z0x)/L
    const Real zol0 = zol * z0x / za;          // z0x/L
    if (zol > 0.) {
      psih = psih_stable(zolzz) - psih_stable(zol0);
      psih10 = psih_stable(zol10) - psih_stable(zol0);
      psih2 = psih_stable(zol2) - psih_stable(zol0);
    } else {
      if (zol == 0) {
        psih = 0.;
        psih10 = 0.;
        psih2 = 0.;
      } else {
        psih = psih_unstable(zolzz) - psih_unstable(zol0);
        psih10 = psih_unstable(zol10) - psih_unstable(zol0);
        psih2 = psih_unstable(zol2) - psih_unstable(zol0);
      }
    }
  }
};

TV host_table_to_device(const char* name, const Kokkos::View<Real*, HostSpace>& h) {
  TV d(std::string(name), h.extent(0));
  Kokkos::deep_copy(d, h);
  return d;
}

} // namespace

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.isftcflx = nl.geti("isftcflx", 0);
  o.iz0tlnd = nl.geti("iz0tlnd", 0);
  o.isfflx = 1;
  return o;
}

void Options::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k sfclay: " + m); };
  if (sfcmodel != 2 && sfcmodel != 3)
    fail("sfcmodel=" + std::to_string(sfcmodel) + " is not a ported WRF surface layer (2, 3)");
  if (isftcflx < 0 || isftcflx > 2) fail("isftcflx=" + std::to_string(isftcflx) + " is not ported (0, 1, 2)");
  if (iz0tlnd < 0 || iz0tlnd > 2) fail("iz0tlnd=" + std::to_string(iz0tlnd) + " is not ported (0, 1, 2)");
  if (isfflx != 0 && isfflx != 1) fail("isfflx must be 0 or 1");
}

//------------------------------------------------------------------------------
// Tables
//------------------------------------------------------------------------------

// gfortran -O2 vectorizes the sfclayinit loop (two lanes, SSE2) through
// glibc's libmvec: X=(1-16z)**0.25 -> _ZGVbN2vv_pow, ALOG -> _ZGVbN2v_log,
// ATAN -> _ZGVbN2v_atan, while (1-16z)**0.5 becomes sqrtpd (the argument is
// provably >= 1) and 2.*ATAN(1.) is constant-folded. libmvec's log/pow/atan
// are not the scalar libm ones (they differ in the last bit for some
// arguments), so to reproduce the cm1f tables bit for bit we call the same
// vector entry points here. Lanes are independent, so pairing is irrelevant.
// Without libmvec (non-glibc or non-x86_64 hosts) the scalar libm is used.
#if defined(CM1K_HAVE_LIBMVEC) && defined(__x86_64__)
namespace {
inline Real mvec_pow(Real a, Real b) { return _mm_cvtsd_f64(_ZGVbN2vv_pow(_mm_set1_pd(a), _mm_set1_pd(b))); }
inline Real mvec_log(Real a) { return _mm_cvtsd_f64(_ZGVbN2v_log(_mm_set1_pd(a))); }
inline Real mvec_atan(Real a) { return _mm_cvtsd_f64(_ZGVbN2v_atan(_mm_set1_pd(a))); }
}  // namespace
#else
namespace {
inline Real mvec_pow(Real a, Real b) { return std::pow(a, b); }
inline Real mvec_log(Real a) { return std::log(a); }
inline Real mvec_atan(Real a) { return std::atan(a); }
}  // namespace
#endif

Tables sfclayinit() {
  Kokkos::View<Real*, HostSpace> m("psimtb", 1001), h("psihtb", 1001);
  const Real two_atan1 = 2. * std::atan(1.);  // folded at compile time by gfortran
  for (int n = 0; n <= 1000; ++n) {
    const Real zoln = -Real(n) * 0.01;
    const Real x = mvec_pow(1 - 16. * zoln, 0.25);
    m(n) = 2 * mvec_log(0.5 * (1 + x)) + mvec_log(0.5 * (1 + x * x)) - 2. * mvec_atan(x) + two_atan1;
    const Real y = std::sqrt(1 - 16 * zoln);
    h(n) = 2 * mvec_log(0.5 * (1 + y));
  }
  Tables t;
  t.psimtb = host_table_to_device("psimtb", m);
  t.psihtb = host_table_to_device("psihtb", h);
  return t;
}

Tables sf_sfclayrev_init() {
  Kokkos::View<Real*, HostSpace> ms("psim_stab", 1001), hs("psih_stab", 1001), mu("psim_unstab", 1001),
      hu("psih_unstab", 1001);
  for (int n = 0; n <= 1000; ++n) {
    // stable function tables
    Real zolf = Real(n) * 0.01;
    ms(n) = psim_stable_full(zolf);
    hs(n) = psih_stable_full(zolf);
    // unstable function tables
    zolf = -Real(n) * 0.01;
    mu(n) = psim_unstable_full(zolf);
    hu(n) = psih_unstable_full(zolf);
  }
  Tables t;
  t.psim_stab = host_table_to_device("psim_stab", ms);
  t.psih_stab = host_table_to_device("psih_stab", hs);
  t.psim_unstab = host_table_to_device("psim_unstab", mu);
  t.psih_unstab = host_table_to_device("psih_unstab", hu);
  return t;
}

//------------------------------------------------------------------------------
// State
//------------------------------------------------------------------------------

void SfcLayerState::allocate(const Bounds& b) {
  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  psfc = mk("psfc"); chs = mk("chs"); chs2 = mk("chs2"); cqs2 = mk("cqs2"); cpmm = mk("cpmm");
  znt = mk("znt"); ust = mk("ust"); hpbl = mk("hpbl"); mavail = mk("mavail"); zol = mk("zol");
  mol = mk("mol"); regime = mk("regime"); psim = mk("psim"); psih = mk("psih"); fm = mk("fm");
  fh = mk("fh"); xland = mk("xland"); hfx = mk("hfx"); qfx = mk("qfx"); lh = mk("lh"); tsk = mk("tsk");
  flhc = mk("flhc"); flqc = mk("flqc"); qgh = mk("qgh"); qsfc = mk("qsfc"); rmol = mk("rmol");
  u10 = mk("u10"); v10 = mk("v10"); th2 = mk("th2"); t2 = mk("t2"); q2 = mk("q2");
  gz1oz0 = mk("gz1oz0"); wspd = mk("wspd"); br = mk("br"); dsxy = mk("dsxy"); ustm = mk("ustm");
  ck = mk("ck"); cka = mk("cka"); cd = mk("cd"); cda = mk("cda"); z0t = mk("z0t"); z0q = mk("z0q");
  ch = mk("ch"); cq = mk("cq"); s10 = mk("s10"); thflux = mk("thflux"); qvflux = mk("qvflux");
}

//------------------------------------------------------------------------------
// sfcmodel = 2: SFCLAY / SFCLAY1D
//------------------------------------------------------------------------------

void sfclay(const Options& o, const WrfConst& c, const Tables& t, int ni, int nj, const FArray3& u3d,
            const FArray3& v3d, const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d,
            const FArray3& dz8w, const FArray3& rf, const SfcLayerState& s)
{
  o.check();
  if (o.sfcmodel != 2) throw std::runtime_error("cm1k sfclay::sfclay: called with sfcmodel != 2");
  if (!t.psimtb.is_allocated()) throw std::runtime_error("cm1k sfclay::sfclay: call sfclayinit() first");
  const int isftcflx = o.isftcflx, iz0tlnd = o.iz0tlnd, isfflx = o.isfflx;
  const Real CP = c.cp, G = c.g, ROVCP = c.rovcp, XLV = c.xlv, lv1 = c.lv1, lv2 = c.lv2;
  const Real SVP1 = c.svp1, SVP2 = c.svp2, SVP3 = c.svp3, SVPT0 = c.svpt0, EP1 = c.ep1, EP2 = c.ep2;
  const Real KARMAN = c.karman, P1000mb = c.p1000mb;
  (void)XLV;
  const TV PSIMTB = t.psimtb, PSIHTB = t.psihtb;
  const SfcLayerState S = s;

  Kokkos::parallel_for("cm1k::sfclay::sfclay", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    // SFCLAY: the k=1 values
    const Real dz8w1d = dz8w(i, j, 1);
    const Real UX = u3d(i, j, 1), VX = v3d(i, j, 1), QV1D = qv3d(i, j, 1), P1D = p3d(i, j, 1),
               T1D = t3d(i, j, 1);
    const Real PSFCPA = S.psfc(i, j, 1), TSK = S.tsk(i, j, 1), XLAND = S.xland(i, j, 1),
               MAVAIL = S.mavail(i, j, 1), PBLH = S.hpbl(i, j, 1), DX = S.dsxy(i, j, 1);
    Real& QGH = S.qgh(i, j, 1);
    Real& FLHC = S.flhc(i, j, 1);
    Real& FLQC = S.flqc(i, j, 1);
    Real& CPM = S.cpmm(i, j, 1);
    Real& QSFC = S.qsfc(i, j, 1);
    Real& ZNT = S.znt(i, j, 1);
    Real& UST = S.ust(i, j, 1);
    Real& GZ1OZ0 = S.gz1oz0(i, j, 1);
    Real& WSPD = S.wspd(i, j, 1);
    Real& BR = S.br(i, j, 1);
    Real& MOL = S.mol(i, j, 1);
    Real& RMOL = S.rmol(i, j, 1);
    Real& REGIME = S.regime(i, j, 1);
    Real& PSIM = S.psim(i, j, 1);
    Real& PSIH = S.psih(i, j, 1);
    Real& ZOL = S.zol(i, j, 1);
    Real& HFX = S.hfx(i, j, 1);
    Real& QFX = S.qfx(i, j, 1);

    // PSFC cb
    const Real PSFC = PSFCPA / 1000.;
    // ----CONVERT GROUND TEMPERATURE TO POTENTIAL TEMPERATURE:
    const Real TGDSA = TSK;
    const Real THGB = TSK * Kokkos::pow(P1000mb / PSFCPA, ROVCP);
    // loop 30
    Real PL = P1D / 1000.;
    const Real SCR3 = T1D;
    const Real THCON = Kokkos::pow(P1000mb * 0.001 / PL, ROVCP);
    const Real THX = SCR3 * THCON;
    Real THVX = THX;
    Real QX = 0.;
    QGH = 0.;
    FLHC = 0.;
    FLQC = 0.;
    CPM = CP;
    // loop 50
    QX = QV1D;
    const Real TVCON = (1. + EP1 * QX);
    THVX = THX * TVCON;
    // loop 60
    Real E1 = SVP1 * Kokkos::exp(SVP2 * (TGDSA - SVPT0) / (TGDSA - SVP3));
    // for land points QSFC can come from previous time step
    if (XLAND > 1.5 || QSFC <= 2.0e-10) QSFC = EP2 * E1 / (PSFC - E1);
    // QGH CHANGED TO USE LOWEST-LEVEL AIR TEMP CONSISTENT WITH MYJSFC CHANGE
    E1 = SVP1 * Kokkos::exp(SVP2 * (T1D - SVPT0) / (T1D - SVP3));
    PL = P1D / 1000.;
    QGH = EP2 * E1 / (PL - E1);
    CPM = CP * (1. + 0.8 * QX);
    // loops 90-160
    const Real ZQKLP1 = 0.;
    const Real RHOX = rf(i, j, 1);
    const Real ZQKL = dz8w1d + ZQKLP1;
    const Real ZA = 0.5 * (ZQKL + ZQKLP1);
    const Real GOVRTH = G / THX;

    // loop 260: bulk Richardson number
    GZ1OZ0 = Kokkos::log(ZA / ZNT);
    const Real GZ2OZ0 = Kokkos::log(2. / ZNT);
    const Real GZ10OZ0 = Kokkos::log(10. / ZNT);
    WSPD = Kokkos::sqrt(UX * UX + VX * VX);
    const Real TSKV = THGB * (1. + EP1 * QSFC);
    const Real DTHVDZ = (THVX - TSKV);
    Real VCONV;
    if (XLAND < 1.5) {
      const Real fluxc = fmax2(S.hfx(i, j, 1) / RHOX / CP + EP1 * TSKV * S.qfx(i, j, 1) / RHOX, 0.);
      VCONV = vconvc * Kokkos::pow(G / TGDSA * PBLH * fluxc, .33);
    } else {
      Real DTHVM;
      if (-DTHVDZ >= 0) DTHVM = -DTHVDZ;
      else DTHVM = 0.;
      VCONV = Kokkos::sqrt(DTHVM);
    }
    // Mahrt and Sun low-res correction
    const Real VSGD = 0.32 * Kokkos::pow(fmax2(DX / 5000. - 1., 0.), .33);
    WSPD = Kokkos::sqrt(WSPD * WSPD + VCONV * VCONV + VSGD * VSGD);
    WSPD = fmax2(WSPD, 0.1);
    BR = GOVRTH * ZA * DTHVDZ / (WSPD * WSPD);
    // IF PREVIOUSLY UNSTABLE, DO NOT LET INTO REGIMES 1 AND 2
    if (MOL < 0.) BR = fmin2(BR, 0.0);
    RMOL = -GOVRTH * DTHVDZ * ZA * KARMAN;

    // loop 320: stability classes
    Real PSIM10, PSIH10, PSIM2, PSIH2;
    if (BR < 0.) {
      // CLASS 4; FREE CONVECTION
      REGIME = 4.;
      if (UST < 0.01) ZOL = BR * GZ1OZ0;
      else ZOL = KARMAN * GOVRTH * ZA * MOL / (UST * UST);
      Real ZOL10 = 10. / ZA * ZOL;
      Real ZOL2 = 2. / ZA * ZOL;
      ZOL = fmin2(ZOL, 0.);
      ZOL = fmax2(ZOL, -9.9999);
      ZOL10 = fmin2(ZOL10, 0.);
      ZOL10 = fmax2(ZOL10, -9.9999);
      ZOL2 = fmin2(ZOL2, 0.);
      ZOL2 = fmax2(ZOL2, -9.9999);
      const int NZOL = int(-ZOL * 100.);
      const Real RZOL = -ZOL * 100. - NZOL;
      const int NZOL10 = int(-ZOL10 * 100.);
      const Real RZOL10 = -ZOL10 * 100. - NZOL10;
      const int NZOL2 = int(-ZOL2 * 100.);
      const Real RZOL2 = -ZOL2 * 100. - NZOL2;
      PSIM = PSIMTB(NZOL) + RZOL * (PSIMTB(NZOL + 1) - PSIMTB(NZOL));
      PSIH = PSIHTB(NZOL) + RZOL * (PSIHTB(NZOL + 1) - PSIHTB(NZOL));
      PSIM10 = PSIMTB(NZOL10) + RZOL10 * (PSIMTB(NZOL10 + 1) - PSIMTB(NZOL10));
      PSIH10 = PSIHTB(NZOL10) + RZOL10 * (PSIHTB(NZOL10 + 1) - PSIHTB(NZOL10));
      PSIM2 = PSIMTB(NZOL2) + RZOL2 * (PSIMTB(NZOL2 + 1) - PSIMTB(NZOL2));
      PSIH2 = PSIHTB(NZOL2) + RZOL2 * (PSIHTB(NZOL2 + 1) - PSIHTB(NZOL2));
      // LIMIT PSIH AND PSIM IN THE CASE OF THIN LAYERS AND HIGH ROUGHNESS
      PSIH = fmin2(PSIH, 0.9 * GZ1OZ0);
      PSIM = fmin2(PSIM, 0.9 * GZ1OZ0);
      PSIH2 = fmin2(PSIH2, 0.9 * GZ2OZ0);
      PSIM10 = fmin2(PSIM10, 0.9 * GZ10OZ0);
      PSIH10 = fmin2(PSIH10, 0.9 * GZ10OZ0);
      RMOL = ZOL / ZA;
    } else if (!(BR < 0.2)) {
      // CLASS 1; STABLE (NIGHTTIME) CONDITIONS
      REGIME = 1.;
      PSIM = -10. * GZ1OZ0;
      PSIM = fmax2(PSIM, -10.);
      PSIH = PSIM;
      PSIM10 = 10. / ZA * PSIM;
      PSIM10 = fmax2(PSIM10, -10.);
      PSIH10 = PSIM10;
      PSIM2 = 2. / ZA * PSIM;
      PSIM2 = fmax2(PSIM2, -10.);
      PSIH2 = PSIM2;
      if (UST < 0.01) RMOL = BR * GZ1OZ0;   // ZA/L
      else RMOL = KARMAN * GOVRTH * ZA * MOL / (UST * UST);
      RMOL = fmin2(RMOL, 9.999);
      RMOL = RMOL / ZA;
    } else if (!(BR == 0.0)) {
      // CLASS 2; DAMPED MECHANICAL TURBULENCE
      REGIME = 2.;
      PSIM = -5.0 * BR * GZ1OZ0 / (1.1 - 5.0 * BR);
      PSIM = fmax2(PSIM, -10.);
      PSIH = PSIM;
      PSIM10 = 10. / ZA * PSIM;
      PSIM10 = fmax2(PSIM10, -10.);
      PSIH10 = PSIM10;
      PSIM2 = 2. / ZA * PSIM;
      PSIM2 = fmax2(PSIM2, -10.);
      PSIH2 = PSIM2;
      ZOL = BR * GZ1OZ0 / (1.00001 - 5.0 * BR);
      if (ZOL > 0.5) {
        ZOL = (1.89 * GZ1OZ0 + 44.2) * BR * BR + (1.18 * GZ1OZ0 - 1.37) * BR;
        ZOL = fmin2(ZOL, 9.999);
      }
      RMOL = ZOL / ZA;
    } else {
      // CLASS 3; FORCED CONVECTION
      REGIME = 3.;
      PSIM = 0.0;
      PSIH = PSIM;
      PSIM10 = 0.;
      PSIH10 = PSIM10;
      PSIM2 = 0.;
      PSIH2 = PSIM2;
      if (UST < 0.01) ZOL = BR * GZ1OZ0;
      else ZOL = KARMAN * GOVRTH * ZA * MOL / (UST * UST);
      RMOL = ZOL / ZA;
    }

    // loop 330: frictional velocity
    const Real DTG = THX - THGB;
    const Real PSIX = GZ1OZ0 - PSIM;
    const Real PSIX10 = GZ10OZ0 - PSIM10;
    Real PSIT = fmax2(GZ1OZ0 - PSIH, 2.);
    Real ZL;
    if ((XLAND - 1.5) >= 0) ZL = ZNT;
    else ZL = 0.01;
    Real PSIQ = Kokkos::log(KARMAN * UST * ZA / xka + ZA / ZL) - PSIH;
    Real PSIT2 = GZ2OZ0 - PSIH2;
    Real PSIQ2 = Kokkos::log(KARMAN * UST * 2. / xka + 2. / ZL) - PSIH2;
    Real PSIQ10 = Kokkos::log(KARMAN * UST * 10. / xka + 10. / ZL) - PSIH10;
    bool z0t_set = false, z0q_set = false;
    Real Z0T = 0, Z0Q = 0;
    if ((XLAND - 1.5) >= 0.) {
      const Real VISC = (1.32 + 0.009 * (SCR3 - 273.15)) * 1.E-5;
      const Real RESTAR = UST * ZNT / VISC;
      Z0T = (5.5e-5) * (Kokkos::pow(RESTAR, -0.60));
      Z0T = fmin2(Z0T, 1.0e-4);
      Z0T = fmax2(Z0T, 2.0e-9);
      Z0Q = Z0T;
      z0t_set = z0q_set = true;
      PSIQ = fmax2(Kokkos::log((ZA + Z0Q) / Z0Q) - PSIH, 2.);
      PSIT = fmax2(Kokkos::log((ZA + Z0T) / Z0T) - PSIH, 2.);
      PSIQ2 = fmax2(Kokkos::log((2. + Z0Q) / Z0Q) - PSIH2, 2.);
      PSIT2 = fmax2(Kokkos::log((2. + Z0T) / Z0T) - PSIH2, 2.);
      PSIQ10 = fmax2(Kokkos::log((10. + Z0Q) / Z0Q) - PSIH10, 2.);
    }
    if (isftcflx == 1 && (XLAND - 1.5) >= 0.) {
      Z0Q = 1.e-4;
      z0q_set = true;
      PSIQ = Kokkos::log(ZA / Z0Q) - PSIH;
      PSIT = PSIQ;
      PSIQ2 = Kokkos::log(2. / Z0Q) - PSIH2;
      PSIQ10 = Kokkos::log(10. / Z0Q) - PSIH10;
      PSIT2 = PSIQ2;
    }
    if (isftcflx == 2 && (XLAND - 1.5) >= 0.) {
      const Real VISC = (1.32 + 0.009 * (SCR3 - 273.15)) * 1.E-5;
      const Real RESTAR = UST * ZNT / VISC;
      const Real GZ0OZT = 0.40 * (7.3 * Kokkos::sqrt(Kokkos::sqrt(RESTAR)) * Kokkos::sqrt(0.71) - 5.);
      const Real GZ0OZQ = 0.40 * (7.3 * Kokkos::sqrt(Kokkos::sqrt(RESTAR)) * Kokkos::sqrt(0.60) - 5.);
      PSIT = GZ1OZ0 - PSIH + GZ0OZT;
      PSIQ = GZ1OZ0 - PSIH + GZ0OZQ;
      PSIT2 = GZ2OZ0 - PSIH2 + GZ0OZT;
      PSIQ2 = GZ2OZ0 - PSIH2 + GZ0OZQ;
      PSIQ10 = GZ10OZ0 - PSIH + GZ0OZQ;
    }
    S.ck(i, j, 1) = (KARMAN / PSIX10) * (KARMAN / PSIQ10);
    S.cd(i, j, 1) = (KARMAN / PSIX10) * (KARMAN / PSIX10);
    S.cka(i, j, 1) = (KARMAN / PSIX) * (KARMAN / PSIQ);
    S.cda(i, j, 1) = (KARMAN / PSIX) * (KARMAN / PSIX);
    if (iz0tlnd >= 1 && (XLAND - 1.5) <= 0.) {
      ZL = ZNT;
      // CZIL RELATED CHANGES FOR LAND
      const Real VISC = (1.32 + 0.009 * (SCR3 - 273.15)) * 1.E-5;
      const Real RESTAR = UST * ZL / VISC;
      Real CZIL = 0;
      if (iz0tlnd == 1) CZIL = Kokkos::pow(10.0, (-0.40 * (ZL / 0.07)));
      else if (iz0tlnd == 2) CZIL = 0.1;
      PSIT = GZ1OZ0 - PSIH + CZIL * KARMAN * Kokkos::sqrt(RESTAR);
      PSIQ = GZ1OZ0 - PSIH + CZIL * KARMAN * Kokkos::sqrt(RESTAR);
      PSIT2 = GZ2OZ0 - PSIH2 + CZIL * KARMAN * Kokkos::sqrt(RESTAR);
      PSIQ2 = GZ2OZ0 - PSIH2 + CZIL * KARMAN * Kokkos::sqrt(RESTAR);
    }
    if (z0t_set) S.z0t(i, j, 1) = Z0T;
    if (z0q_set) S.z0q(i, j, 1) = Z0Q;
    // TO PREVENT OSCILLATIONS AVERAGE WITH OLD VALUE
    UST = 0.5 * UST + 0.5 * KARMAN * WSPD / PSIX;
    // TKE coupling: compute ust without vconv for use in tke scheme
    const Real WSPDI = Kokkos::sqrt(UX * UX + VX * VX);
    S.ustm(i, j, 1) = 0.5 * S.ustm(i, j, 1) + 0.5 * KARMAN * WSPDI / PSIX;
    S.u10(i, j, 1) = UX * PSIX10 / PSIX;
    S.v10(i, j, 1) = VX * PSIX10 / PSIX;
    const Real TH2 = THGB + DTG * PSIT2 / PSIT;
    S.th2(i, j, 1) = TH2;
    S.q2(i, j, 1) = QSFC + (QX - QSFC) * PSIQ2 / PSIQ;
    S.t2(i, j, 1) = TH2 * Kokkos::pow(PSFCPA / P1000mb, ROVCP);
    if ((XLAND - 1.5) < 0.) UST = fmax2(UST, 0.1);
    MOL = KARMAN * DTG / PSIT / prt;
    const Real DENOMQ = PSIQ, DENOMQ2 = PSIQ2, DENOMT2 = PSIT2;
    S.fm(i, j, 1) = PSIX;
    S.fh(i, j, 1) = PSIT;

    // COMPUTE THE SURFACE SENSIBLE AND LATENT HEAT FLUXES
    QFX = 0.;
    HFX = 0.;
    if (isfflx == 0) return;

    // loop 360: OVER WATER, ALTER ROUGHNESS LENGTH (ZNT) ACCORDING TO WIND (UST)
    if ((XLAND - 1.5) >= 0) {
      ZNT = czo * UST * UST / G + 0.11 * 1.5E-5 / UST;
      ZNT = fmin2(ZNT, 2.85e-3);
      if (isftcflx != 0) {
        const Real ZW = fmin2(Kokkos::pow(UST / 1.06, 0.3), 1.0);
        const Real ZN1 = 0.011 * UST * UST / G + ozo;
        const Real ZN2 = 10. * Kokkos::exp(-9.5 * Kokkos::pow(UST, -.3333)) + 0.11 * 1.5E-5 / fmax2(UST, 0.01);
        ZNT = (1.0 - ZW) * ZN1 + ZW * ZN2;
        ZNT = fmin2(ZNT, 2.85e-3);
        ZNT = fmax2(ZNT, 1.27e-7);
      }
    }
    FLQC = RHOX * MAVAIL * UST * KARMAN / DENOMQ;
    const Real DTTHX = Kokkos::fabs(THX - THGB);
    if (DTTHX > 1.E-5) FLHC = CPM * RHOX * UST * MOL / (THX - THGB);
    else FLHC = 0.;

    // loop 370: COMPUTE SURFACE MOIST FLUX
    QFX = FLQC * (QSFC - QX);
    QFX = fmax2(QFX, 0.);
    S.lh(i, j, 1) = (lv1 - lv2 * TSK) * QFX;

    // loop 400: COMPUTE SURFACE HEAT FLUX
    if (XLAND - 1.5 > 0.) {
      HFX = FLHC * (THGB - THX);
    } else if (XLAND - 1.5 < 0.) {
      HFX = FLHC * (THGB - THX);
      HFX = fmax2(HFX, -250.);
    }

    S.chs(i, j, 1) = UST * KARMAN / DENOMQ;
    S.cqs2(i, j, 1) = UST * KARMAN / DENOMQ2;
    S.chs2(i, j, 1) = UST * KARMAN / DENOMT2;
  });
}

//------------------------------------------------------------------------------
// sfcmodel = 3: sf_sfclayrev_run
//------------------------------------------------------------------------------

void sfclayrev(const Options& o, const WrfConst& c, const Tables& t, int ni, int nj, const FArray3& u3d,
               const FArray3& v3d, const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d,
               const FArray3& dz8w, const SfcLayerState& s)
{
  o.check();
  if (o.sfcmodel != 3) throw std::runtime_error("cm1k sfclay::sfclayrev: called with sfcmodel != 3");
  if (!t.psim_stab.is_allocated())
    throw std::runtime_error("cm1k sfclay::sfclayrev: call sf_sfclayrev_init() first");
  const int isftcflx = o.isftcflx, iz0tlnd = o.iz0tlnd;
  const bool isfflx = (o.isfflx != 0);
  const Real cp = c.cp, g = c.g, rovcp = c.rovcp, r = c.rd, xlv = c.xlv;
  const Real svp1 = c.svp1, svp2 = c.svp2, svp3 = c.svp3, svpt0 = c.svpt0, ep1 = c.ep1, ep2 = c.ep2;
  const Real karman = c.karman, p1000mb = c.p1000mb;
  const RevTab T{t.psim_stab, t.psim_unstab, t.psih_stab, t.psih_unstab};
  const SfcLayerState S = s;

  Kokkos::parallel_for("cm1k::sfclay::sfclayrev", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const Real dz8w1d = dz8w(i, j, 1);
    const Real ux = u3d(i, j, 1), vx = v3d(i, j, 1), qv1d = qv3d(i, j, 1), p1d = p3d(i, j, 1),
               t1d = t3d(i, j, 1);
    const Real psfcpa = S.psfc(i, j, 1), tsk = S.tsk(i, j, 1), xland = S.xland(i, j, 1),
               mavail = S.mavail(i, j, 1), pblh = S.hpbl(i, j, 1), dx = S.dsxy(i, j, 1);
    Real& qgh = S.qgh(i, j, 1);
    Real& flhc = S.flhc(i, j, 1);
    Real& flqc = S.flqc(i, j, 1);
    Real& cpm = S.cpmm(i, j, 1);
    Real& qsfc = S.qsfc(i, j, 1);
    Real& znt = S.znt(i, j, 1);
    Real& ust = S.ust(i, j, 1);
    Real& gz1oz0 = S.gz1oz0(i, j, 1);
    Real& wspd = S.wspd(i, j, 1);
    Real& br = S.br(i, j, 1);
    Real& mol = S.mol(i, j, 1);
    Real& rmol = S.rmol(i, j, 1);
    Real& regime = S.regime(i, j, 1);
    Real& psim = S.psim(i, j, 1);
    Real& psih = S.psih(i, j, 1);
    Real& zol = S.zol(i, j, 1);
    Real& hfx = S.hfx(i, j, 1);
    Real& qfx = S.qfx(i, j, 1);

    // PSFC cb
    const Real psfc = psfcpa / 1000.;
    // loop 5
    const Real tgdsa = tsk;
    const Real thgb = tsk * Kokkos::pow(p1000mb / psfcpa, rovcp);
    // loop 30
    Real pl = p1d / 1000.;
    const Real scr3 = t1d;
    const Real thcon = Kokkos::pow(p1000mb * 0.001 / pl, rovcp);
    const Real thx = scr3 * thcon;
    Real scr4 = scr3;
    Real thvx = thx;
    Real qx = 0.;
    qgh = 0.;
    flhc = 0.;
    flqc = 0.;
    cpm = cp;
    // loop 50
    qx = qv1d;
    const Real tvcon = (1. + ep1 * qx);
    thvx = thx * tvcon;
    scr4 = scr3 * tvcon;
    // loop 60
    Real e1 = svp1 * Kokkos::exp(svp2 * (tgdsa - svpt0) / (tgdsa - svp3));
    // for land points qsfc can come from previous time step
    if (xland > 1.5 || qsfc <= 0.0) qsfc = ep2 * e1 / (psfc - e1);
    e1 = svp1 * Kokkos::exp(svp2 * (t1d - svpt0) / (t1d - svp3));
    pl = p1d / 1000.;
    qgh = ep2 * e1 / (pl - e1);
    cpm = cp * (1. + 0.8 * qx);
    // loops 90-160
    const Real zqklp1 = 0.;
    const Real rhox = psfc * 1000. / (r * scr4);
    const Real zqkl = dz8w1d + zqklp1;
    const Real za = 0.5 * (zqkl + zqklp1);
    const Real govrth = g / thx;

    // loop 260
    gz1oz0 = Kokkos::log((za + znt) / znt);
    const Real gz2oz0 = Kokkos::log((2. + znt) / znt);
    const Real gz10oz0 = Kokkos::log((10. + znt) / znt);
    Real zl;
    wspd = Kokkos::sqrt(ux * ux + vx * vx);
    const Real tskv = thgb * (1. + ep1 * qsfc);
    const Real dthvdz = (thvx - tskv);
    Real vconv;
    if (xland < 1.5) {
      const Real fluxc = fmax2(S.hfx(i, j, 1) / rhox / cp + ep1 * tskv * S.qfx(i, j, 1) / rhox, 0.);
      vconv = vconvc * Kokkos::pow(g / tgdsa * pblh * fluxc, .33);
    } else {
      Real dthvm;
      if (-dthvdz >= 0) dthvm = -dthvdz;
      else dthvm = 0.;
      vconv = Kokkos::sqrt(dthvm);
    }
    const Real vsgd = 0.32 * Kokkos::pow(fmax2(dx / 5000. - 1., 0.), .33);
    wspd = Kokkos::sqrt(wspd * wspd + vconv * vconv + vsgd * vsgd);
    wspd = fmax2(wspd, 0.1);
    br = govrth * za * dthvdz / (wspd * wspd);
    if (mol < 0.) br = fmin2(br, 0.0);
    rmol = -govrth * dthvdz * za * karman;

    // loop 320
    if (br > 0) {
      if (br > 250.0) zol = T.zolri(250.0, za, znt);
      else zol = T.zolri(br, za, znt);
    }
    if (br < 0) {
      if (ust < 0.001) {
        zol = br * gz1oz0;
      } else {
        if (br < -250.0) zol = T.zolri(-250.0, za, znt);
        else zol = T.zolri(br, za, znt);
      }
    }
    Real zolzz = zol * (za + znt) / za;  // (z+z0/L
    Real zol10 = zol * (10. + znt) / za; // (10+z0)/L
    Real zol2 = zol * (2. + znt) / za;   // (2+z0)/L
    Real zol0 = zol * znt / za;          // z0/L
    const Real zl2 = (2.) / za * zol;    // 2/L
    const Real zl10 = (10.) / za * zol;  // 10/L
    if ((xland - 1.5) < 0.) zl = (0.01) / za * zol;   // (0.01)/L
    else zl = zol0;                                    // z0/L

    Real psim10, psih10, psim2, psih2, pq, pq2, pq10;
    if (br < 0.) {
      // CLASS 4; FREE CONVECTION
      regime = 4.;
      psim = T.psim_unstable(zolzz) - T.psim_unstable(zol0);
      psih = T.psih_unstable(zolzz) - T.psih_unstable(zol0);
      psim10 = T.psim_unstable(zol10) - T.psim_unstable(zol0);
      psih10 = T.psih_unstable(zol10) - T.psih_unstable(zol0);
      psim2 = T.psim_unstable(zol2) - T.psim_unstable(zol0);
      psih2 = T.psih_unstable(zol2) - T.psih_unstable(zol0);
      pq = T.psih_unstable(zol) - T.psih_unstable(zl);
      pq2 = T.psih_unstable(zl2) - T.psih_unstable(zl);
      pq10 = T.psih_unstable(zl10) - T.psih_unstable(zl);
      psih = fmin2(psih, 0.9 * gz1oz0);
      psim = fmin2(psim, 0.9 * gz1oz0);
      psih2 = fmin2(psih2, 0.9 * gz2oz0);
      psim10 = fmin2(psim10, 0.9 * gz10oz0);
      psih10 = fmin2(psih10, 0.9 * gz10oz0);
      rmol = zol / za;
    } else if (br == 0.) {
      // CLASS 3; FORCED CONVECTION
      regime = 3.;
      psim = 0.0;
      psih = psim;
      psim10 = 0.;
      psih10 = psim10;
      psim2 = 0.;
      psih2 = psim2;
      pq = psih;
      pq2 = psih2;
      pq10 = 0.;
      zol = 0.;
      rmol = zol / za;
    } else {
      // CLASS 1; STABLE (NIGHTTIME) CONDITIONS
      regime = 1.;
      psim = T.psim_stable(zolzz) - T.psim_stable(zol0);
      psih = T.psih_stable(zolzz) - T.psih_stable(zol0);
      psim10 = T.psim_stable(zol10) - T.psim_stable(zol0);
      psih10 = T.psih_stable(zol10) - T.psih_stable(zol0);
      psim2 = T.psim_stable(zol2) - T.psim_stable(zol0);
      psih2 = T.psih_stable(zol2) - T.psih_stable(zol0);
      pq = T.psih_stable(zol) - T.psih_stable(zl);
      pq2 = T.psih_stable(zl2) - T.psih_stable(zl);
      pq10 = T.psih_stable(zl10) - T.psih_stable(zl);
      rmol = zol / za;
    }

    // loop 330
    const Real dtg = thx - thgb;
    const Real psix = gz1oz0 - psim;
    const Real psix10 = gz10oz0 - psim10;
    Real psit = gz1oz0 - psih;
    Real psit2 = gz2oz0 - psih2;
    if ((xland - 1.5) >= 0) zl = znt;
    else zl = 0.01;
    Real psiq = Kokkos::log(karman * ust * za / xka + za / zl) - pq;
    Real psiq2 = Kokkos::log(karman * ust * 2. / xka + 2. / zl) - pq2;
    Real psiq10 = Kokkos::log(karman * ust * 10. / xka + 10. / zl) - pq10;
    Real z0t, z0q;
    if ((xland - 1.5) >= 0.) {
      const Real visc = (1.32 + 0.009 * (scr3 - 273.15)) * 1.e-5;
      const Real restar = ust * znt / visc;
      z0t = (5.5e-5) * (Kokkos::pow(restar, -0.60));
      z0t = fmin2(z0t, 1.0e-4);
      z0t = fmax2(z0t, 2.0e-9);
      z0q = z0t;
      // following paj:
      T.psih_for(zol, za, z0t, psih, psih10, psih2);
      psit = Kokkos::log((za + z0t) / z0t) - psih;
      psit2 = Kokkos::log((2. + z0t) / z0t) - psih2;
      T.psih_for(zol, za, z0q, psih, psih10, psih2);
      psiq = Kokkos::log((za + z0q) / z0q) - psih;
      psiq2 = Kokkos::log((2. + z0q) / z0q) - psih2;
      psiq10 = Kokkos::log((10. + z0q) / z0q) - psih10;
    }
    if (isftcflx == 1 && (xland - 1.5) >= 0.) {
      z0q = 1.e-4;
      // ... paj: recompute psih for z0q
      T.psih_for(zol, za, z0q, psih, psih10, psih2);
      psiq = Kokkos::log((za + z0q) / z0q) - psih;
      psit = psiq;
      psiq2 = Kokkos::log((2. + z0q) / z0q) - psih2;
      psiq10 = Kokkos::log((10. + z0q) / z0q) - psih10;
      psit2 = psiq2;
    }
    if (isftcflx == 2 && (xland - 1.5) >= 0.) {
      const Real visc = (1.32 + 0.009 * (scr3 - 273.15)) * 1.e-5;
      const Real restar = ust * znt / visc;
      const Real gz0ozt = 0.40 * (7.3 * Kokkos::sqrt(Kokkos::sqrt(restar)) * Kokkos::sqrt(0.71) - 5.);
      // ... paj: compute psih for z0t for temperature ...
      z0t = znt / Kokkos::exp(gz0ozt);
      T.psih_for(zol, za, z0t, psih, psih10, psih2);
      psit = Kokkos::log((za + z0t) / z0t) - psih;
      psit2 = Kokkos::log((2. + z0t) / z0t) - psih2;
      const Real gz0ozq = 0.40 * (7.3 * Kokkos::sqrt(Kokkos::sqrt(restar)) * Kokkos::sqrt(0.60) - 5.);
      z0q = znt / Kokkos::exp(gz0ozq);
      T.psih_for(zol, za, z0q, psih, psih10, psih2);
      psiq = Kokkos::log((za + z0q) / z0q) - psih;
      psiq2 = Kokkos::log((2. + z0q) / z0q) - psih2;
      psiq10 = Kokkos::log((10. + z0q) / z0q) - psih10;
    }
    S.ck(i, j, 1) = (karman / psix10) * (karman / psiq10);
    S.cd(i, j, 1) = (karman / psix10) * (karman / psix10);
    S.cka(i, j, 1) = (karman / psix) * (karman / psiq);
    S.cda(i, j, 1) = (karman / psix) * (karman / psix);
    if (iz0tlnd >= 1 && (xland - 1.5) <= 0.) {
      zl = znt;
      // CZIL RELATED CHANGES FOR LAND
      const Real visc = (1.32 + 0.009 * (scr3 - 273.15)) * 1.e-5;
      const Real restar = ust * zl / visc;
      Real czil = 0;
      if (iz0tlnd == 1) czil = Kokkos::pow(10.0, (-0.40 * (zl / 0.07)));
      else if (iz0tlnd == 2) czil = 0.1;
      // ... paj: compute phish for z0t over land
      z0t = znt / Kokkos::exp(czil * karman * Kokkos::sqrt(restar));
      T.psih_for(zol, za, z0t, psih, psih10, psih2);
      psiq = Kokkos::log((za + z0t) / z0t) - psih;
      psiq2 = Kokkos::log((2. + z0t) / z0t) - psih2;
      psit = psiq;
      psit2 = psiq2;
    }
    // TO PREVENT OSCILLATIONS AVERAGE WITH OLD VALUE
    ust = 0.5 * ust + 0.5 * karman * wspd / psix;
    // TKE coupling: compute ust without vconv for use in tke scheme
    const Real wspdi = Kokkos::sqrt(ux * ux + vx * vx);
    S.ustm(i, j, 1) = 0.5 * S.ustm(i, j, 1) + 0.5 * karman * wspdi / psix;
    S.u10(i, j, 1) = ux * psix10 / psix;
    S.v10(i, j, 1) = vx * psix10 / psix;
    const Real th2 = thgb + dtg * psit2 / psit;
    S.th2(i, j, 1) = th2;
    S.q2(i, j, 1) = qsfc + (qx - qsfc) * psiq2 / psiq;
    S.t2(i, j, 1) = th2 * Kokkos::pow(psfcpa / p1000mb, rovcp);
    if ((xland - 1.5) < 0.) ust = fmax2(ust, 0.001);
    mol = karman * dtg / psit / prt;
    const Real denomq = psiq, denomq2 = psiq2, denomt2 = psit2;
    S.fm(i, j, 1) = psix;
    S.fh(i, j, 1) = psit;

    // COMPUTE THE SURFACE SENSIBLE AND LATENT HEAT FLUXES (scm_force_flux = .false.)
    qfx = 0.;
    hfx = 0.;
    if (!isfflx) return;

    // loop 360 (shalwater_z0 = .false.)
    if ((xland - 1.5) >= 0) {
      znt = czo * ust * ust / g + 0.11 * 1.5e-5 / ust;
      znt = fmin2(znt, 2.85e-3);
      if (isftcflx != 0) {
        const Real zw = fmin2(Kokkos::pow(ust / 1.06, (0.3)), 1.0);
        const Real zn1 = 0.011 * ust * ust / g + ozo;
        const Real zn2 = 10. * Kokkos::exp(-9.5 * Kokkos::pow(ust, (-.3333))) + 0.11 * 1.5e-5 / fmax2(ust, 0.01);
        znt = (1.0 - zw) * zn1 + zw * zn2;
        znt = fmin2(znt, 2.85e-3);
        znt = fmax2(znt, 1.27e-7);
      }
    }
    flqc = rhox * mavail * ust * karman / denomq;
    const Real dtthx = Kokkos::fabs(thx - thgb);
    if (dtthx > 1.e-5) flhc = cpm * rhox * ust * mol / (thx - thgb);
    else flhc = 0.;

    // loop 370
    qfx = flqc * (qsfc - qx);
    qfx = fmax2(qfx, 0.);
    S.lh(i, j, 1) = xlv * qfx;

    // loop 400
    if (xland - 1.5 > 0.) {
      hfx = flhc * (thgb - thx);
    } else if (xland - 1.5 < 0.) {
      hfx = flhc * (thgb - thx);
      hfx = fmax2(hfx, -250.);
    }

    S.chs(i, j, 1) = ust * karman / denomq;
    S.cqs2(i, j, 1) = ust * karman / denomq2;
    S.chs2(i, j, 1) = ust * karman / denomt2;
  });
}

//------------------------------------------------------------------------------
// turb.F after the surface layer
//------------------------------------------------------------------------------

void sfclay_post(int ni, int nj, bool set_fluxes, const FArray3& qv3d, const FArray3& rf,
                 const SfcLayerState& s)
{
  const SfcLayerState S = s;
  Kokkos::parallel_for("cm1k::sfclay::sfclay_post", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    // dont update tsk, but diagnose qsfc:
    const Real qx = qv3d(i, j, 1);
    if (S.flqc(i, j, 1) != 0.) S.qsfc(i, j, 1) = qx + S.qfx(i, j, 1) / S.flqc(i, j, 1);
    else S.qsfc(i, j, 1) = qx;
    // put WRF parameters into CM1 arrays:
    S.ch(i, j, 1) = S.chs2(i, j, 1);
    S.cq(i, j, 1) = S.cqs2(i, j, 1);
    S.s10(i, j, 1) = Kokkos::sqrt(S.u10(i, j, 1) * S.u10(i, j, 1) + S.v10(i, j, 1) * S.v10(i, j, 1));
    if (set_fluxes) {
      S.thflux(i, j, 1) = S.hfx(i, j, 1) / (S.cpmm(i, j, 1) * rf(i, j, 1));
      S.qvflux(i, j, 1) = S.qfx(i, j, 1) / rf(i, j, 1);
    }
  });
}

} // namespace cm1k::sfclay
