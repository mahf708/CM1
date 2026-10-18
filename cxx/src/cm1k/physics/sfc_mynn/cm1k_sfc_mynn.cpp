// Port of src/module_sf_mynn.F (MYNN surface layer, sfcmodel=6). See the
// header for scope. Kernels keep the Fortran operation order; `x**2` (and
// `x**2.`, which GCC folds to x*x as well) is written x*x, other real
// exponents are pow.
#include "physics/sfc_mynn/cm1k_sfc_mynn.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::sfc_mynn {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// module_sf_mynn parameters
constexpr Real wmin = 0.1;
constexpr Real vconvc = 1.25;

// ---------------------------------------------------------------------------
// stability functions (psi_opt = 0)

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

// The lookup tables, captured by value in kernels.
struct Psi {
  Kokkos::View<const Real*, Device> msb, mun, hsb, hun;

  KOKKOS_INLINE_FUNCTION Real m_stable(Real zolf) const {
    const int nzol = int(zolf * 100.);
    const Real rzol = zolf * 100. - nzol;
    if (nzol + 1 <= 1000) return msb(nzol) + rzol * (msb(nzol + 1) - msb(nzol));
    return psim_stable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real h_stable(Real zolf) const {
    const int nzol = int(zolf * 100.);
    const Real rzol = zolf * 100. - nzol;
    if (nzol + 1 <= 1000) return hsb(nzol) + rzol * (hsb(nzol + 1) - hsb(nzol));
    return psih_stable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real m_unstable(Real zolf) const {
    const int nzol = int(-zolf * 100.);
    const Real rzol = -zolf * 100. - nzol;
    if (nzol + 1 <= 1000) return mun(nzol) + rzol * (mun(nzol + 1) - mun(nzol));
    return psim_unstable_full(zolf);
  }
  KOKKOS_INLINE_FUNCTION Real h_unstable(Real zolf) const {
    const int nzol = int(-zolf * 100.);
    const Real rzol = -zolf * 100. - nzol;
    if (nzol + 1 <= 1000) return hun(nzol) + rzol * (hun(nzol + 1) - hun(nzol));
    return psih_unstable_full(zolf);
  }
};

// ---------------------------------------------------------------------------
// roughness lengths

KOKKOS_INLINE_FUNCTION void charnock_1955(Real& z_0, Real ustar, Real wsp10, Real visc, Real zu) {
  constexpr Real G = 9.81, CZO2 = 0.011;
  const Real wsp10m = wsp10 * Kokkos::log(10. / 1e-4) / Kokkos::log(zu / 1e-4);
  const Real czc = CZO2 + 0.007 * fmin2(fmax2((wsp10m - 10.) / 8., 0.), 1.0);
  z_0 = czc * ustar * ustar / G + (0.11 * visc / fmax2(ustar, 0.05));
  z_0 = fmax2(z_0, 1.27e-7);
  z_0 = fmin2(z_0, 2.85e-3);
}

KOKKOS_INLINE_FUNCTION void davis_etal_2008(Real& z_0, Real ustar) {
  constexpr Real G = 9.81, OZO = 1.59E-5;
  const Real zw = fmin2(Kokkos::pow(ustar / 1.06, 0.3), 1.0);
  const Real zn1 = 0.011 * ustar * ustar / G + OZO;
  const Real zn2 = 10. * Kokkos::exp(-9.5 * Kokkos::pow(ustar, -.3333)) + 0.11 * 1.5E-5 / fmax2(ustar, 0.01);
  z_0 = (1.0 - zw) * zn1 + zw * zn2;
  z_0 = fmax2(z_0, 1.27e-7);
  z_0 = fmin2(z_0, 2.85e-3);
}

KOKKOS_INLINE_FUNCTION void taylor_yelland_2001(Real& z_0, Real /*ustar*/, Real wsp10) {
  constexpr Real g = 9.81, pi = 3.14159265;
  const Real hs = 0.0248 * (wsp10 * wsp10);
  const Real tp = 0.729 * fmax2(wsp10, 0.1);
  const Real lp = g * (tp * tp) / (2 * pi);
  z_0 = 1200. * hs * Kokkos::pow(hs / lp, 4.5);
  z_0 = fmax2(z_0, 1.27e-7);
  z_0 = fmin2(z_0, 2.85e-3);
}

KOKKOS_INLINE_FUNCTION void zilitinkevich_1995(Real z_0, Real& zt, Real& zq, Real restar, Real karman,
                                               Real landsea, int iz0tlnd2) {
  if (landsea - 1.5 > 0) {
    if (restar < 0.1) {
      zt = z_0 * Kokkos::exp(karman * 2.0);
      zt = fmin2(zt, 6.0e-5);
      zt = fmax2(zt, 2.0e-9);
      zq = z_0 * Kokkos::exp(karman * 3.0);
      zq = fmin2(zq, 6.0e-5);
      zq = fmax2(zq, 2.0e-9);
    } else {
      zt = z_0 * Kokkos::exp(-karman * (4.0 * Kokkos::sqrt(restar) - 3.2));
      zt = fmin2(zt, 6.0e-5);
      zt = fmax2(zt, 2.0e-9);
      zq = z_0 * Kokkos::exp(-karman * (4.0 * Kokkos::sqrt(restar) - 4.2));
      zq = fmin2(zt, 6.0e-5);   // sic: zt
      zq = fmax2(zt, 2.0e-9);
    }
  } else {
    Real czil;
    if (iz0tlnd2 == 1) {
      czil = Kokkos::pow(10.0, -0.40 * (z_0 / 0.07));
    } else {
      czil = 0.085;
    }
    zt = z_0 * Kokkos::exp(-karman * czil * Kokkos::sqrt(restar));
    zt = fmin2(zt, 0.75 * z_0);
    zq = z_0 * Kokkos::exp(-karman * czil * Kokkos::sqrt(restar));
    zq = fmin2(zq, 0.75 * z_0);
    // spp_pbl = 0
  }
}

KOKKOS_INLINE_FUNCTION void garratt_1992(Real& zt, Real& zq, Real z_0, Real ren, Real landsea) {
  constexpr Real e = 2.71828183;
  if (landsea - 1.5 > 0) {
    zt = z_0 * Kokkos::exp(2.0 - (2.48 * Kokkos::pow(ren, 0.25)));
    zq = z_0 * Kokkos::exp(2.0 - (2.28 * Kokkos::pow(ren, 0.25)));
    zq = fmin2(zq, 5.5e-5);
    zq = fmax2(zq, 2.0e-9);
    zt = fmin2(zt, 5.5e-5);
    zt = fmax2(zt, 2.0e-9);
  } else {
    zq = z_0 / (e * e);
    zt = zq;
  }
}

KOKKOS_INLINE_FUNCTION void fairall_etal_2003(Real& zt, Real& zq, Real ren) {
  // both branches of `Ren .le. 2.` are the same; spp_pbl = 0
  zt = (5.5e-5) * Kokkos::pow(ren, -0.60);
  zq = zt;
  zt = fmin2(zt, 1.0e-4);
  zt = fmax2(zt, 2.0e-9);
  zq = fmin2(zt, 1.0e-4);
  zq = fmax2(zt, 2.0e-9);
}

KOKKOS_INLINE_FUNCTION void andreas_2002(Real z_0, Real bvisc, Real ustar, Real& zt, Real& zq) {
  (void)z_0;
  constexpr Real bt0_s = 1.25, bt0_t = 0.149, bt0_r = 0.317, bt1_s = 0.0, bt1_t = -0.55, bt1_r = -0.565,
                 bt2_s = 0.0, bt2_t = 0.0, bt2_r = -0.183;
  constexpr Real bq0_s = 1.61, bq0_t = 0.351, bq0_r = 0.396, bq1_s = 0.0, bq1_t = -0.628, bq1_r = -0.512,
                 bq2_s = 0.0, bq2_t = 0.0, bq2_r = -0.180;
  const Real zntsno = 0.135 * bvisc / ustar +
                      (0.035 * (ustar * ustar) / 9.8) *
                          (5. * Kokkos::exp(-1. * (((ustar - 0.18) / 0.1) * ((ustar - 0.18) / 0.1))) + 1.);
  Real ren2 = ustar * zntsno / bvisc;
  if (ren2 > 1000.) ren2 = 1000.;
  const Real l = Kokkos::log(ren2);
  if (ren2 <= 0.135) {
    zt = zntsno * Kokkos::exp(bt0_s + bt1_s * l + bt2_s * (l * l));
    zq = zntsno * Kokkos::exp(bq0_s + bq1_s * l + bq2_s * (l * l));
  } else if (ren2 > 0.135 && ren2 < 2.5) {
    zt = zntsno * Kokkos::exp(bt0_t + bt1_t * l + bt2_t * (l * l));
    zq = zntsno * Kokkos::exp(bq0_t + bq1_t * l + bq2_t * (l * l));
  } else {
    zt = zntsno * Kokkos::exp(bt0_r + bt1_r * l + bt2_r * (l * l));
    zq = zntsno * Kokkos::exp(bq0_r + bq1_r * l + bq2_r * (l * l));
  }
}

// ---------------------------------------------------------------------------
// z/L from the bulk Richardson number

KOKKOS_INLINE_FUNCTION Real li_etal_2010(Real rib, Real zaz0, Real z0zt) {
  constexpr Real au11 = 0.045, bu11 = 0.003, bu12 = 0.0059, bu21 = -0.0828, bu22 = 0.8845, bu31 = 0.1739,
                 bu32 = -0.9213, bu33 = -0.1057;
  constexpr Real aw11 = 0.5738, aw12 = -0.4399, aw21 = -4.901, aw22 = 52.50, bw11 = -0.0539, bw12 = 1.540,
                 bw21 = -0.669, bw22 = -3.282;
  constexpr Real as11 = 0.7529, as21 = 14.94, bs11 = 0.1569, bs21 = -0.3091, bs22 = -1.303;
  Real zaz02 = zaz0;
  if (zaz0 < 100.0) zaz02 = 100.;
  if (zaz0 > 100000.0) zaz02 = 100000.;
  Real z0zt2 = z0zt;
  if (z0zt < 0.5) z0zt2 = 0.5;
  if (z0zt > 100.0) z0zt2 = 100.;
  const Real alfa = Kokkos::log(zaz02);
  const Real beta = Kokkos::log(z0zt2);
  Real zl;
  if (rib <= 0.0) {
    zl = au11 * alfa * (rib * rib) +
         ((bu11 * beta + bu12) * (alfa * alfa) + (bu21 * beta + bu22) * alfa + (bu31 * (beta * beta) + bu32 * beta + bu33)) *
             rib;
    zl = fmax2(zl, -15.);
    zl = fmin2(zl, 0.);
  } else if (rib > 0.0 && rib <= 0.2) {
    zl = ((aw11 * beta + aw12) * alfa + (aw21 * beta + aw22)) * (rib * rib) +
         ((bw11 * beta + bw12) * alfa + (bw21 * beta + bw22)) * rib;
    zl = fmin2(zl, 4.);
    zl = fmax2(zl, 0.);
  } else {
    zl = (as11 * alfa + as21) * rib + bs11 * alfa + bs21 * beta + bs22;
    zl = fmin2(zl, 20.);
    zl = fmax2(zl, 1.);
  }
  return zl;
}

KOKKOS_INLINE_FUNCTION Real zolrib(const Psi& psi, Real ri, Real za, Real z0, Real zt, Real logz0, Real logzt,
                                   Real zol1) {
  constexpr int nmax = 20;
  if (zol1 * ri < 0.) zol1 = 0.;
  Real zolold, res;
  if (ri < 0.) {
    zolold = -99999.;
    res = -66666.;
  } else {
    zolold = 99999.;
    res = 66666.;
  }
  int n = 1;
  while (Kokkos::fabs(zolold - res) > 0.01 && n < nmax) {
    if (n == 1) {
      zolold = zol1;
    } else {
      zolold = res;
    }
    const Real zol20 = zolold * z0 / za;
    const Real zol3 = zolold + zol20;
    const Real zolt = zolold * zt / za;
    Real psit2, psix2;
    if (ri < 0) {
      psit2 = fmax2(logzt - (psi.h_unstable(zol3) - psi.h_unstable(zolt)), 1.0);
      psix2 = fmax2(logz0 - (psi.m_unstable(zol3) - psi.m_unstable(zol20)), 1.0);
    } else {
      psit2 = fmax2(logzt - (psi.h_stable(zol3) - psi.h_stable(zolt)), 1.0);
      psix2 = fmax2(logz0 - (psi.m_stable(zol3) - psi.m_stable(zol20)), 1.0);
    }
    res = ri * (psix2 * psix2) / psit2;
    n = n + 1;
  }
  if (n == nmax && Kokkos::fabs(zolold - res) > 0.01) res = li_etal_2010(ri, za / z0, z0 / zt);
  return res;
}

} // namespace

// ---------------------------------------------------------------------------

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.sfcmodel = nl.geti("sfcmodel", o.sfcmodel);
  o.isftcflx = nl.geti("isftcflx", o.isftcflx);
  o.iz0tlnd = nl.geti("iz0tlnd", o.iz0tlnd);
  o.imoist = nl.geti("imoist", o.imoist);
  return o;
}

void Options::check() const {
  if (!active()) return;
  if (isftcflx < 0 || isftcflx > 3)
    throw std::runtime_error("sfc_mynn: isftcflx=" + std::to_string(isftcflx) +
                             " not supported (0..3; 4 leaves z_t undefined in module_sf_mynn)");
  if (!(iz0tlnd == 0 || iz0tlnd == 1 || iz0tlnd == 3))
    throw std::runtime_error("sfc_mynn: iz0tlnd=" + std::to_string(iz0tlnd) +
                             " not supported (0, 1, 3; 2 reads an uninitialized qstar in module_sf_mynn)");
}

PsiTables PsiTables::make() {
  PsiTables t;
  t.psim_stab = Kokkos::View<Real*, Device>("psim_stab", n);
  t.psim_unstab = Kokkos::View<Real*, Device>("psim_unstab", n);
  t.psih_stab = Kokkos::View<Real*, Device>("psih_stab", n);
  t.psih_unstab = Kokkos::View<Real*, Device>("psih_unstab", n);
  auto ms = Kokkos::create_mirror_view(t.psim_stab);
  auto mu = Kokkos::create_mirror_view(t.psim_unstab);
  auto hs = Kokkos::create_mirror_view(t.psih_stab);
  auto hu = Kokkos::create_mirror_view(t.psih_unstab);
  // psi_init(psi_opt=0), on the host (glibc, as the Fortran)
  for (int N = 0; N <= 1000; ++N) {
    Real zolf = Real(N) * 0.01;
    ms(N) = psim_stable_full(zolf);
    hs(N) = psih_stable_full(zolf);
    zolf = -Real(N) * 0.01;
    mu(N) = psim_unstable_full(zolf);
    hu(N) = psih_unstable_full(zolf);
  }
  Kokkos::deep_copy(t.psim_stab, ms);
  Kokkos::deep_copy(t.psim_unstab, mu);
  Kokkos::deep_copy(t.psih_stab, hs);
  Kokkos::deep_copy(t.psih_unstab, hu);
  return t;
}

void Fields::allocate(const Bounds& b) {
  auto a = [&](const char* n) { return FArray3(n, {b.ib, b.jb, 1}, {b.ie, b.je, 1}); };
  psfc = a("psfc"); hpbl = a("hpbl"); mavail = a("mavail"); xland = a("xland"); tsk = a("tsk");
  snowh = a("snowh");
  ust = a("ust"); mol = a("mol"); qsfc = a("qsfc"); znt = a("znt"); ustm = a("ustm"); hfx = a("hfx");
  qfx = a("qfx"); zol = a("zol"); rmol = a("rmol"); wspd = a("wspd");
  chs = a("chs"); chs2 = a("chs2"); cqs2 = a("cqs2"); cpmm = a("cpmm"); regime = a("regime");
  psim = a("psim"); psih = a("psih"); lh = a("lh"); flhc = a("flhc"); flqc = a("flqc"); qgh = a("qgh");
  u10 = a("u10"); v10 = a("v10"); th2 = a("th2"); t2 = a("t2"); q2 = a("q2"); gz1oz0 = a("gz1oz0");
  br = a("br"); ch = a("ch"); ck = a("ck"); cka = a("cka"); cd = a("cd"); cda = a("cda");
}

void prepare_inputs(const Options& o, const Grid& g, const FArray3& ugr, const FArray3& vgr,
                    const FArray3& th0, const FArray3& tha, const FArray3& pi0, const FArray3& ppi,
                    const FArray3& prs, const FArray3& rho, const FArray4& qa, int nqv, const Inputs& in) {
  const Bounds& b = g.b;
  const bool moist = (o.imoist == 1);
  const Real dz = g.dz;
  const auto rmh = g.rmh;
  const auto u3d = in.u3d, v3d = in.v3d, dz8w = in.dz8w, t3d = in.t3d, qv3d = in.qv3d, p3d = in.p3d,
             rho3d = in.rho3d;
  Kokkos::parallel_for("sfc_mynn_prep", box3(1, b.ni, 1, b.nj, 1, 2), KOKKOS_LAMBDA(int i, int j, int k) {
    u3d(i, j, k) = 0.5 * (ugr(i, j, k) + ugr(i + 1, j, k));
    v3d(i, j, k) = 0.5 * (vgr(i, j, k) + vgr(i, j + 1, k));
    const Real th = th0(i, j, k) + tha(i, j, k);
    const Real pi = pi0(i, j, k) + ppi(i, j, k);
    t3d(i, j, k) = th * pi;
    dz8w(i, j, k) = dz * rmh(i, j, k);
    qv3d(i, j, k) = moist ? qa(i, j, k, nqv) : 0.0;
    p3d(i, j, k) = prs(i, j, k);
    rho3d(i, j, k) = rho(i, j, k);
  });
}

void sfclay(const Options& o, const Constants& c, const PsiTables& t, int ni, int nj, Real dx,
            int itimestep, int isfflx, const Inputs& in, const Fields& f) {
  o.check();
  const Psi psi{t.psim_stab, t.psim_unstab, t.psih_stab, t.psih_unstab};
  const int isftcflx = o.isftcflx, iz0tlnd = o.iz0tlnd;
  const Real cp = c.cp, g = c.g, rovcp = c.rovcp, xlv = c.xlv;
  const Real svp1 = c.svp1, svp2 = c.svp2, svp3 = c.svp3, svpt0 = c.svpt0;
  const Real ep1 = c.rv / c.rd - 1.0, ep2 = c.rd / c.rv;
  const Real ep_3 = 1. - 287.04 / 461.5;   // module_sf_mynn: 1.-ep_2 (module_model_constants)
  constexpr Real karman = 0.4;              // CM1's karman
  constexpr Real prt = 1.;
  const auto u3d = in.u3d, v3d = in.v3d, dz8w = in.dz8w, t3d = in.t3d, qv3d = in.qv3d, p3d = in.p3d,
             rho3d = in.rho3d;
  const Fields F = f;

  Kokkos::parallel_for("sfc_mynn_sfclay", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    // SFCLAY_mynn: column copies
    const Real dz8w1d = dz8w(i, j, 1), dz2w1d = dz8w(i, j, 2);
    const Real u1d = u3d(i, j, 1), v1d = v3d(i, j, 1), u1d2 = u3d(i, j, 2), v1d2 = v3d(i, j, 2);
    const Real qv1d = qv3d(i, j, 1), p1d = p3d(i, j, 1), t1d = t3d(i, j, 1), rho1d = rho3d(i, j, 1);
    Real& ust = F.ust(i, j, 1);
    Real& mol = F.mol(i, j, 1);
    Real& qsfc = F.qsfc(i, j, 1);
    if (itimestep == 1) {
      ust = fmax2(0.04 * Kokkos::sqrt(u1d * u1d + v1d * v1d), 0.001);
      mol = 0.;
      qsfc = qv3d(i, j, 1) / (1. + qv3d(i, j, 1));
    }
    const Real tsk = F.tsk(i, j, 1), xland = F.xland(i, j, 1), pblh = F.hpbl(i, j, 1);
    Real& znt = F.znt(i, j, 1);
    Real& zol = F.zol(i, j, 1);
    Real& wspd = F.wspd(i, j, 1);
    Real& br = F.br(i, j, 1);
    Real& hfx = F.hfx(i, j, 1);
    Real& qfx = F.qfx(i, j, 1);
    Real& cpm = F.cpmm(i, j, 1);

    // SFCLAY1D_mynn
    const Real psfc = F.psfc(i, j, 1) / 1000.;
    const Real thgb = tsk * Kokkos::pow(100. / psfc, rovcp);
    Real pl = p1d / 1000.;
    const Real thcon = Kokkos::pow(100. / pl, rovcp);
    const Real th1d = t1d * thcon;
    const Real tc1d = t1d - 273.15;
    const Real qvsh = qv1d / (1. + qv1d);
    const Real tvcon = (1. + ep1 * qvsh);
    const Real thv1d = th1d * tvcon;
    const Real za = 0.5 * dz8w1d;
    const Real za2 = dz8w1d + 0.5 * dz2w1d;
    const Real govrth = g / th1d;

    Real e1;
    if (tsk < 273.15) {
      e1 = svp1 * Kokkos::exp(4648 * (1. / 273.15 - 1. / tsk) - 11.64 * Kokkos::log(273.15 / tsk) +
                              0.02265 * (273.15 - tsk));
    } else {
      e1 = svp1 * Kokkos::exp(svp2 * (tsk - svpt0) / (tsk - svp3));
    }
    Real qsfcmr;
    if (xland > 1.5 || qsfc <= 0.0) {
      qsfc = ep2 * e1 / (psfc - ep_3 * e1);
      qsfcmr = ep2 * e1 / (psfc - e1);
    } else {
      qsfcmr = qsfc / (1. - qsfc);
    }
    if (tsk < 273.15) {
      e1 = svp1 * Kokkos::exp(4648 * (1. / 273.15 - 1. / t1d) - 11.64 * Kokkos::log(273.15 / t1d) +
                              0.02265 * (273.15 - t1d));
    } else {
      e1 = svp1 * Kokkos::exp(svp2 * (t1d - svpt0) / (t1d - svp3));
    }
    pl = p1d / 1000.;
    F.qgh(i, j, 1) = ep2 * e1 / (pl - e1);
    cpm = cp * (1. + 0.84 * qv1d);

    wspd = Kokkos::sqrt(u1d * u1d + v1d * v1d);
    const Real thvgb = thgb * (1. + ep1 * qsfc);
    const Real dthvdz = (thv1d - thvgb);
    const Real fluxc = fmax2(hfx / rho1d / cp + ep1 * thvgb * qfx / rho1d, 0.);
    Real wstar;
    if (xland > 1.5 || qsfc <= 0.0) {
      wstar = vconvc * Kokkos::pow(g / tsk * pblh * fluxc, .33);
    } else {
      wstar = vconvc * Kokkos::pow(g / tsk * fmin2(1.5 * pblh, 4000.) * fluxc, .33);
    }
    const Real vsgd = 0.32 * Kokkos::pow(fmax2(dx / 5000. - 1., 0.), .33);
    wspd = Kokkos::sqrt(wspd * wspd + wstar * wstar + vsgd * vsgd);
    wspd = fmax2(wspd, wmin);
    br = govrth * za * dthvdz / (wspd * wspd);
    if (itimestep == 1) {
      br = fmax2(br, -2.0);
      br = fmin2(br, 2.0);
    } else {
      br = fmax2(br, -4.0);
      br = fmin2(br, 4.0);
    }

    const Real visc = 1.326e-5 * (1. + 6.542e-3 * tc1d + 8.301e-6 * tc1d * tc1d - 4.84e-9 * tc1d * tc1d * tc1d);
    Real zntstoch, z_t = 0, z_q = 0;
    if ((xland - 1.5) >= 0) {
      if (isftcflx == 0) {
        charnock_1955(znt, ust, wspd, visc, za);   // COARE_OPT = 3.0
      } else if (isftcflx == 1 || isftcflx == 2) {
        davis_etal_2008(znt, ust);
      } else if (isftcflx == 3) {
        taylor_yelland_2001(znt, ust, wspd);
      }
      zntstoch = znt;
      const Real restar = fmax2(ust * zntstoch / visc, 0.1);
      if (isftcflx == 0 || isftcflx == 1 || isftcflx == 3) {
        fairall_etal_2003(z_t, z_q, restar);
      } else if (isftcflx == 2) {
        garratt_1992(z_t, z_q, zntstoch, restar, xland);
      }
    } else {
      zntstoch = znt;
      const Real restar = fmax2(ust * zntstoch / visc, 0.1);
      if (F.snowh(i, j, 1) >= 0.1) {
        andreas_2002(zntstoch, visc, ust, z_t, z_q);
      } else {
        if (iz0tlnd <= 1) {
          zilitinkevich_1995(zntstoch, z_t, z_q, restar, karman, xland, iz0tlnd);
        } else if (iz0tlnd == 3) {
          garratt_1992(z_t, z_q, zntstoch, restar, xland);
        }
      }
    }
    const Real zratio = zntstoch / z_t;
    Real& gz1oz0 = F.gz1oz0(i, j, 1);
    gz1oz0 = Kokkos::log((za + zntstoch) / zntstoch);
    Real gz1ozt = Kokkos::log((za + zntstoch) / z_t);
    Real gz2oz0 = Kokkos::log((2.0 + zntstoch) / zntstoch);
    (void)gz2oz0;
    Real gz2ozt = Kokkos::log((2.0 + zntstoch) / z_t);
    const Real gz10oz0 = Kokkos::log((10. + zntstoch) / zntstoch);
    const Real gz10ozt = Kokkos::log((10. + zntstoch) / z_t);

    Real& regime = F.regime(i, j, 1);
    Real& psim = F.psim(i, j, 1);
    Real& psih = F.psih(i, j, 1);
    Real& rmol = F.rmol(i, j, 1);
    Real psim10, psih10, psih2;
    if (br > 0.0) {
      if (br > 0.2) {
        regime = 1.;
      } else {
        regime = 2.;
      }
      if (itimestep <= 1) {
        zol = li_etal_2010(br, za / zntstoch, zratio);
      } else {
        zol = za * karman * g * mol / (th1d * fmax2(ust * ust, 0.0001));
        zol = fmax2(zol, 0.0);
        zol = fmin2(zol, 20.);
      }
      zol = zolrib(psi, br, za, zntstoch, z_t, gz1oz0, gz1ozt, zol);
      zol = fmax2(zol, 0.0);
      zol = fmin2(zol, 20.);
      const Real zolzt = zol * z_t / za;
      const Real zolz0 = zol * zntstoch / za;
      const Real zolza = zol * (za + zntstoch) / za;
      const Real zol10 = zol * (10. + zntstoch) / za;
      const Real zol2 = zol * (2. + zntstoch) / za;
      // (the same over water and land)
      psim = psi.m_stable(zolza) - psi.m_stable(zolz0);
      psih = psi.h_stable(zolza) - psi.h_stable(zolzt);
      psim10 = psi.m_stable(zol10) - psi.m_stable(zolz0);
      psih10 = psi.h_stable(zol10) - psi.h_stable(zolz0);
      psih2 = psi.h_stable(zol2) - psi.h_stable(zolz0);
      rmol = zol / za;
    } else if (br == 0.) {
      regime = 3.;
      psim = 0.0;
      psih = psim;
      psim10 = 0.;
      psih10 = 0.;
      psih2 = 0.;
      zol = 0.;
      rmol = zol / za;
    } else if (br < 0.) {
      regime = 4.;
      if (itimestep <= 1) {
        zol = li_etal_2010(br, za / zntstoch, zratio);
      } else {
        zol = za * karman * g * mol / (th1d * fmax2(ust * ust, 0.001));
        zol = fmax2(zol, -20.0);
        zol = fmin2(zol, 0.0);
      }
      zol = zolrib(psi, br, za, zntstoch, z_t, gz1oz0, gz1ozt, zol);
      zol = fmax2(zol, -20.0);
      zol = fmin2(zol, 0.0);
      const Real zolzt = zol * z_t / za;
      const Real zolz0 = zol * zntstoch / za;
      const Real zolza = zol * (za + zntstoch) / za;
      const Real zol10 = zol * (10. + zntstoch) / za;
      const Real zol2 = zol * (2. + zntstoch) / za;
      psim = psi.m_unstable(zolza) - psi.m_unstable(zolz0);
      psih = psi.h_unstable(zolza) - psi.h_unstable(zolzt);
      psim10 = psi.m_unstable(zol10) - psi.m_unstable(zolz0);
      psih10 = psi.h_unstable(zol10) - psi.h_unstable(zolz0);
      psih2 = psi.h_unstable(zol2) - psi.h_unstable(zolz0);
      psih = fmin2(psih, 0.9 * gz1ozt);
      psim = fmin2(psim, 0.9 * gz1oz0);
      psih2 = fmin2(psih2, 0.9 * gz2ozt);
      psim10 = fmin2(psim10, 0.9 * gz10oz0);
      psih10 = fmin2(psih10, 0.9 * gz10ozt);
      rmol = zol / za;
    } else {
      // br is NaN: the Fortran leaves psim10/psih10/psih2 undefined
      psim10 = psih10 = psih2 = Kokkos::Experimental::quiet_NaN<Real>::value;
    }

    Real psix = gz1oz0 - psim;
    Real psix10 = gz10oz0 - psim10;
    ust = 0.5 * ust + 0.5 * karman * wspd / psix;
    const Real wspdi = fmax2(Kokkos::sqrt(u1d * u1d + v1d * v1d), wmin);
    Real& ustm = F.ustm(i, j, 1);
    ustm = 0.5 * ustm + 0.5 * karman * wspdi / psix;
    if ((xland - 1.5) < 0.) {
      ust = fmax2(ust, 0.005);
      ustm = ust;
    }
    gz1ozt = Kokkos::log((za + zntstoch) / z_t);
    gz2ozt = Kokkos::log((2.0 + zntstoch) / z_t);
    const Real dtg1 = thv1d - thvgb;
    mol = karman * dtg1 / fmax2(gz1ozt - psih, 1.) / prt;
    // (qstar is not kept: only Yang_2008, iz0tlnd=2, reads it)

    // second loop of SFCLAY1D_mynn
    psix = gz1oz0 - psim;
    psix10 = gz10oz0 - psim10;
    const Real psit = fmax2(gz1ozt - psih, 1.0);
    const Real psit2 = fmax2(gz2ozt - psih2, 1.0);
    const Real psiq = fmax2(Kokkos::log((za + z_q) / z_q) - psih, 1.0);
    const Real psiq2 = fmax2(Kokkos::log((2.0 + z_q) / z_q) - psih2, 1.0);
    const Real psiq10 = fmax2(Kokkos::log((10.0 + z_q) / z_q) - psih10, 1.0);
    Real& flhc = F.flhc(i, j, 1);
    Real& flqc = F.flqc(i, j, 1);
    if (isfflx < 1) {
      qfx = 0.;
      hfx = 0.;
      flhc = 0.;
      flqc = 0.;
      F.lh(i, j, 1) = 0.;
      F.chs(i, j, 1) = 0.;
      F.ch(i, j, 1) = 0.;
      F.chs2(i, j, 1) = 0.;
      F.cqs2(i, j, 1) = 0.;
      F.ck(i, j, 1) = 0.;
      F.cd(i, j, 1) = 0.;
      F.cka(i, j, 1) = 0.;
      F.cda(i, j, 1) = 0.;
    } else {
      flqc = rho1d * F.mavail(i, j, 1) * ust * karman / psiq;
      flhc = rho1d * cpm * ust * karman / psit;
      qfx = flqc * (qsfcmr - qv1d);
      qfx = fmax2(qfx, -0.02);
      F.lh(i, j, 1) = xlv * qfx;
      if (xland - 1.5 > 0.) {
        hfx = flhc * (thgb - th1d);
        if (isftcflx != 0) hfx = hfx + rho1d * ustm * ustm * wspdi;
      } else if (xland - 1.5 < 0.) {
        hfx = flhc * (thgb - th1d);
        hfx = fmax2(hfx, -250.);
      }
      F.chs(i, j, 1) = ust * karman / psit;
      F.ch(i, j, 1) = flhc / (cpm * rho1d);
      F.cqs2(i, j, 1) = ust * karman / psiq2;
      F.chs2(i, j, 1) = ust * karman / psit2;
      F.ck(i, j, 1) = (karman / psix10) * (karman / psiq10);
      F.cd(i, j, 1) = (karman / psix10) * (karman / psix10);
      F.cka(i, j, 1) = (karman / psix) * (karman / psiq);
      F.cda(i, j, 1) = (karman / psix) * (karman / psix);
    }
    Real& u10 = F.u10(i, j, 1);
    Real& v10 = F.v10(i, j, 1);
    if (za <= 7.0) {
      if (za2 > 7.0 && za2 < 13.0) {
        u10 = u1d2;
        v10 = v1d2;
      } else {
        u10 = u1d * Kokkos::log(10. / zntstoch) / Kokkos::log(za / zntstoch);
        v10 = v1d * Kokkos::log(10. / zntstoch) / Kokkos::log(za / zntstoch);
      }
    } else if (za > 7.0 && za < 13.0) {
      u10 = u1d * Kokkos::log(10. / zntstoch) / Kokkos::log(za / zntstoch);
      v10 = v1d * Kokkos::log(10. / zntstoch) / Kokkos::log(za / zntstoch);
    } else {
      u10 = u1d * psix10 / psix;
      v10 = v1d * psix10 / psix;
    }
    const Real dtg = th1d - thgb;
    Real& th2 = F.th2(i, j, 1);
    th2 = thgb + dtg * psit2 / psit;
    if ((th1d > thgb && (th2 < thgb || th2 > th1d)) || (th1d < thgb && (th2 > thgb || th2 < th1d))) {
      th2 = thgb + 2. * (th1d - thgb) / za;
    }
    F.t2(i, j, 1) = th2 * Kokkos::pow(psfc / 100., rovcp);
    Real& q2 = F.q2(i, j, 1);
    q2 = qsfcmr + (qv1d - qsfcmr) * psiq2 / psiq;
    q2 = fmax2(q2, fmin2(qsfcmr, qv1d));
    q2 = fmin2(q2, 1.05 * qv1d);
  });
}

void finish(int ni, int nj, bool dosfcflx, const Inputs& in, const FArray3& rf, const Fields& f,
            const FArray3& cq, const FArray3& s10, const FArray3& thflux, const FArray3& qvflux) {
  const Fields F = f;
  const auto qv3d = in.qv3d;
  Kokkos::parallel_for("sfc_mynn_finish", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const Real qx = qv3d(i, j, 1);
    if (F.flqc(i, j, 1) != 0.) {
      F.qsfc(i, j, 1) = qx + F.qfx(i, j, 1) / F.flqc(i, j, 1);
    } else {
      F.qsfc(i, j, 1) = qx;
    }
    F.ch(i, j, 1) = F.chs2(i, j, 1);
    cq(i, j, 1) = F.cqs2(i, j, 1);
    s10(i, j, 1) = Kokkos::sqrt(F.u10(i, j, 1) * F.u10(i, j, 1) + F.v10(i, j, 1) * F.v10(i, j, 1));
    F.ch(i, j, 1) = F.ck(i, j, 1);
    cq(i, j, 1) = F.ck(i, j, 1);
    if (dosfcflx) {
      thflux(i, j, 1) = F.hfx(i, j, 1) / (F.cpmm(i, j, 1) * rf(i, j, 1));
      qvflux(i, j, 1) = F.qfx(i, j, 1) / rf(i, j, 1);
    }
  });
}

} // namespace cm1k::sfc_mynn
