// Port of the YSU PBL (src/bl_ysu.F) as CM1 calls it. See cm1k_pbl_ysu.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// real**real (also **3., **4.) is pow, x**2 and x**2. are x*x, and
// Fortran's max/min of two non-NaN numbers is fmax2/fmin2. The Fortran's
// separate i-loops only couple points of the same column, so each column
// runs them in order. The always-zero BEP terms (a_*2d, b_*2d, bepswitch)
// and unit factors (sfk2d, vlk2d) are kept literally: x + 0*dt2 is not an
// identity for x = -0.
#include "physics/pbl_ysu/cm1k_pbl_ysu.hpp"

#include <stdexcept>

namespace cm1k::pbl_ysu {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// bl_ysu_run parameters
constexpr Real xkzminm = 0.1, xkzminh = 0.01;
constexpr Real xkzmax = 1000., rimin = -100.;
constexpr Real rlam = 30., prmin = 0.25, prmax = 4.;
constexpr Real brcr_ub = 0.0, brcr_sb = 0.25, cori = 1.e-4;
constexpr Real afac = 6.8, bfac = 6.8, pfac = 2.0, pfac_q = 2.0;
constexpr Real phifac = 8., sfcfrac = 0.1;
constexpr Real d1 = 0.02, d2 = 0.05, d3 = 0.001;
constexpr Real h1 = 0.33333335, h2 = 0.6666667;
constexpr Real zfmin = 1.e-8, aphi5 = 5., aphi16 = 16.;
constexpr Real tmin = 1.e-2;
constexpr Real gamcrt = 3., gamcrq = 2.e-3;
constexpr int imvdif = 1;
// rcl (= 1) only feeds the dusfc/dvsfc diagnostics, which CM1 does not request.

// the absent BEP/urban arrays (bepswitch = 0)
constexpr Real bepswitch = 0.0;
constexpr Real a_2d = 0.0, b_2d = 0.0, sfk2d = 1.0, vlk2d = 1.0;
constexpr Real uoxl = 0.0, voxl = 0.0;

// per-column work arrays: index n of Workspace::w
enum Work : int {
  W_ZQ, W_THX, W_THVX, W_THLIX, W_DEL, W_DZA, W_ZA, W_XKZH, W_XKZM, W_XKZQ, W_XKZML, W_XKZHL, W_F1, W_F2,
  W_AD, W_AU, W_AL, W_AD1, W_ADV, W_AUL, W_ENTFAC, W_ZFACENT, W_RHOX2, NWORK
};

using WView = Kokkos::View<Real****, Kokkos::LayoutLeft, Device>;

// A column of one work array, Fortran-indexed k = 1..nk+1.
struct Col {
  Real* p;
  size_t s;
  KOKKOS_INLINE_FUNCTION Real& operator()(int k) const { return p[size_t(k - 1) * s]; }
};

// tridin_ysu with nt = 1: cl(k) is al(k-1) (the Fortran passes al(its:ite,kts:kte)
// to cl(its:ite,kts+1:kte+1)); f holds r on entry (r2 is a copy of f2 in the
// Fortran, read before f2 is written at each k).
KOKKOS_INLINE_FUNCTION void tridin_ysu(const Col& al, const Col& cm, const Col& cu, const Col& aul, const Col& f,
                                       int n) {
  for (int k = 1; k <= n; ++k) aul(k) = 0.;
  Real fk = 1. / cm(1);
  aul(1) = fk * cu(1);
  f(1) = fk * f(1);
  for (int k = 2; k <= n - 1; ++k) {
    fk = 1. / (cm(k) - al(k - 1) * aul(k - 1));
    aul(k) = fk * cu(k);
    f(k) = fk * (f(k) - al(k - 1) * f(k - 1));
  }
  fk = 1. / (cm(n) - al(n - 1) * aul(n - 1));
  f(n) = fk * (f(n) - al(n - 1) * f(n - 1));
  for (int k = n - 1; k >= 1; --k) f(k) = f(k) - aul(k) * f(k + 1);
}

// tridi2n with nt = 1: au is overwritten (cu is a copy of au in the Fortran).
KOKKOS_INLINE_FUNCTION void tridi2n(const Col& al, const Col& cm, const Col& cm1, const Col& au, const Col& f1,
                                    const Col& f2, int n) {
  Real fk = 1. / cm(1);
  au(1) = fk * au(1);
  f1(1) = fk * f1(1);
  fk = 1. / cm1(1);
  f2(1) = fk * f2(1);
  for (int k = 2; k <= n - 1; ++k) {
    fk = 1. / (cm(k) - al(k - 1) * au(k - 1));
    au(k) = fk * au(k);
    f1(k) = fk * (f1(k) - al(k - 1) * f1(k - 1));
  }
  for (int k = 2; k <= n - 1; ++k) {
    fk = 1. / (cm1(k) - al(k - 1) * au(k - 1));
    f2(k) = fk * (f2(k) - al(k - 1) * f2(k - 1));
  }
  fk = 1. / (cm(n) - al(n - 1) * au(n - 1));
  f1(n) = fk * (f1(n) - al(n - 1) * f1(n - 1));
  fk = 1. / (cm1(n) - al(n - 1) * au(n - 1));
  f2(n) = fk * (f2(n) - al(n - 1) * f2(n - 1));
  for (int k = n - 1; k >= 1; --k) f1(k) = f1(k) - au(k) * f1(k + 1);
  for (int k = n - 1; k >= 1; --k) f2(k) = f2(k) - au(k) * f2(k + 1);
}

} // namespace

void Workspace::ensure(int ni, int nj, int nk) {
  if (w.extent(0) == size_t(ni) && w.extent(1) == size_t(nj) && w.extent(2) == size_t(nk + 2)) return;
  // nk+2 levels: the topdown branch may read k = nk+1 (zero here)
  w = WView("cm1k::pbl_ysu::work", ni, nj, nk + 2, NWORK);
}

void ysu(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
         const Outputs& out)
{
  Workspace ws;
  ysu(o, c, dt, ni, nj, nk, in, out, ws);
}

void ysu(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
         const Outputs& out, Workspace& ws)
{
  if (nk < 3) throw std::runtime_error("cm1k pbl_ysu: needs nk >= 3");
  // Every work array is written before it is read, except the k = nk+1
  // slots the topdown branch may read (zero since allocation).
  ws.ensure(ni, nj, nk);
  const WView W = ws.w;
  const bool ysu_topdown_pblmix = o.ysu_topdown_pblmix;
  const Real cp = c.cp, g = c.g, rovcp = c.rovcp, rd = c.rd, ep1 = c.ep1, ep2 = c.ep2, karman = c.karman,
             xlv = c.xlv, rv = c.rv;
  const Inputs I = in;
  const Outputs O = out;
  const int kte = nk, klpbl = nk;

  Kokkos::parallel_for("cm1k::pbl_ysu::ysu", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const size_t ks = W.stride(2);
    auto col = [&](int n) { return Col{&W(i - 1, j - 1, 0, n), ks}; };
    const Col zq = col(W_ZQ), thx = col(W_THX), thvx = col(W_THVX), thlix = col(W_THLIX), del = col(W_DEL),
              dza = col(W_DZA), za = col(W_ZA), xkzh = col(W_XKZH), xkzm = col(W_XKZM), xkzq = col(W_XKZQ),
              xkzml = col(W_XKZML), xkzhl = col(W_XKZHL), f1 = col(W_F1), f2 = col(W_F2), ad = col(W_AD),
              au = col(W_AU), al = col(W_AL), ad1 = col(W_AD1), adv = col(W_ADV), aul = col(W_AUL),
              entfac = col(W_ENTFAC), zfacent = col(W_ZFACENT), rhox2 = col(W_RHOX2);
    // column inputs, Fortran-indexed
    auto ux = [&](int k) { return I.ux(i, j, k); };
    auto vx = [&](int k) { return I.vx(i, j, k); };
    auto tx = [&](int k) { return I.tx(i, j, k); };
    auto qvx = [&](int k) { return k <= kte ? I.qvx(i, j, k) : 0.0; };
    auto qcxl = [&](int k) { return k <= kte ? I.qcx(i, j, k) : 0.0; };   // f_qc = .true.
    auto qixl = [&](int k) { return I.qix(i, j, k); };                    // f_qi = .true.
    auto p2d = [&](int k) { return I.p2d(i, j, k); };
    auto p2di = [&](int k) { return I.p2di(i, j, k); };
    auto pi2d = [&](int k) { return I.pi2d(i, j, k); };
    const Real psfcpa = I.psfcpa(i, j, 1), ust = I.ust(i, j, 1), znt = I.znt(i, j, 1), xland = I.xland(i, j, 1),
               hfx = I.hfx(i, j, 1), qfx = I.qfx(i, j, 1), wspd = I.wspd(i, j, 1), br = I.br(i, j, 1),
               psim = I.psim(i, j, 1), psih = I.psih(i, j, 1);
    const Real u10 = O.u10(i, j, 1), v10 = O.v10(i, j, 1);

    const Real cont = cp / g;
    const Real conq = xlv / g;
    (void)conq;
    const Real conpr = bfac * karman * sfcfrac;

    for (int k = 1; k <= kte; ++k) {
      thx(k) = tx(k) / pi2d(k);
      thlix(k) = (tx(k) - xlv * qcxl(k) / cp - 2.834E6 * qixl(k) / cp) / pi2d(k);
    }
    for (int k = 1; k <= kte; ++k) {
      const Real tvcon = (1. + ep1 * qvx(k));
      thvx(k) = thx(k) * tvcon;
    }
    Real rhox, govrth;
    {
      const Real tvcon = (1. + ep1 * qvx(1));
      rhox = psfcpa / (rd * tx(1) * tvcon);
      govrth = g / thx(1);
    }

    // height of full- and half-sigma levels above ground, layer thicknesses
    zq(1) = 0.;
    for (int k = 1; k <= kte; ++k) {
      zq(k + 1) = I.dz8w2d(i, j, k) + zq(k);
      const Real tvcon = (1. + ep1 * qvx(k));
      rhox2(k) = p2d(k) / (rd * tx(k) * tvcon);
    }
    for (int k = 1; k <= kte; ++k) {
      za(k) = 0.5 * (zq(k) + zq(k + 1));
      del(k) = p2di(k) - p2di(k + 1);
    }
    dza(1) = za(1);
    for (int k = 2; k <= kte; ++k) dza(k) = za(k) - za(k - 1);

    // initialize output and local exchange coefficents
    for (int k = 1; k <= kte; ++k) {
      O.exch_hx(i, j, k) = 0.;
      O.exch_mx(i, j, k) = 0.;
      xkzh(k) = 0.;
      xkzhl(k) = 0.;
      xkzm(k) = 0.;
      xkzml(k) = 0.;
      xkzq(k) = 0.;
    }
    const Real wspd1 =
        Kokkos::sqrt((ux(1) - uoxl) * (ux(1) - uoxl) + (vx(1) - voxl) * (vx(1) - voxl)) + 1.e-9;

    // compute preliminary variables
    const Real dtstep = dt;
    const Real dt2 = 2. * dtstep;
    const Real rdt = 1. / dt2;

    Real bfxpbl = 0.0, hfxpbl = 0.0, qfxpbl = 0.0, ufxpbl = 0.0, vfxpbl = 0.0, hgamu = 0.0, hgamv = 0.0;
    Real delta = 0.0, wstar3_2 = 0.0;
    Real hgamt = 0., hgamq = 0., wscale = 0.;
    int kpbl = 1;
    Real hpbl = zq(1);
    const Real zl1 = za(1);
    Real thermal = thvx(1);
    Real thermalli = thlix(1);
    bool pblflg = true, sfcflg = true;
    const Real sflux = hfx / rhox / cp + qfx / rhox * ep1 * thx(1);
    if (br > 0.0) sfcflg = false;

    // first guess of pbl height
    bool stable = false;
    Real brup = br, brdn = 0, brcr = brcr_ub;
    for (int k = 2; k <= klpbl; ++k) {
      if (!stable) {
        brdn = brup;
        const Real spdk2 = fmax2(ux(k) * ux(k) + vx(k) * vx(k), 1.);
        brup = (thvx(k) - thermal) * (g * za(k) / thvx(1)) / spdk2;
        kpbl = k;
        stable = brup > brcr;
      }
    }
    {
      const int k = kpbl;
      Real brint;
      if (brdn >= brcr) brint = 0.;
      else if (brup <= brcr) brint = 1.;
      else brint = (brcr - brdn) / (brup - brdn);
      hpbl = za(k - 1) + brint * (za(k) - za(k - 1));
      if (hpbl < zq(2)) kpbl = 1;
      if (kpbl <= 1) pblflg = false;
    }

    Real zol1, phim, phih, wstar, wstar3, ust3;
    {
      const Real fm = psim;
      const Real fh = psih;
      zol1 = fmax2(br * fm * fm / fh, rimin);
      if (sfcflg) zol1 = fmin2(zol1, -zfmin);
      else zol1 = fmax2(zol1, zfmin);
      const Real hol1 = zol1 * hpbl / zl1 * sfcfrac;
      if (sfcflg) {
        phim = Kokkos::pow((1. - aphi16 * hol1), (-1. / 4.));
        phih = Kokkos::pow((1. - aphi16 * hol1), (-1. / 2.));
        const Real bfx0 = fmax2(sflux, 0.);
        wstar3 = (govrth * bfx0 * hpbl);
        wstar = Kokkos::pow((wstar3), h1);
      } else {
        phim = (1. + aphi5 * hol1);
        phih = phim;
        wstar = 0.;
        wstar3 = 0.;
      }
      ust3 = Kokkos::pow(ust, 3.);
      wscale = Kokkos::pow((ust3 + phifac * karman * wstar3 * 0.5), h1);
      wscale = fmin2(wscale, ust * aphi16);
      wscale = fmax2(wscale, ust / aphi5);
    }

    // surface variables for pbl height estimation under unstable conditions
    if (sfcflg && sflux > 0.0) {
      const Real gamfac = bfac / rhox / wscale;
      hgamt = fmin2(gamfac * hfx / cp, gamcrt);
      hgamq = fmin2(gamfac * qfx, gamcrq);
      const Real vpert = (hgamt + ep1 * thx(1) * hgamq) / bfac * afac;
      thermal = thermal + fmax2(vpert, 0.) * fmin2(za(1) / (sfcfrac * hpbl), 1.0);
      thermalli = thermalli + fmax2(vpert, 0.) * fmin2(za(1) / (sfcfrac * hpbl), 1.0);
      hgamt = fmax2(hgamt, 0.0);
      hgamq = fmax2(hgamq, 0.0);
      const Real brint = -15.9 * ust * ust / wspd * wstar3 / (Kokkos::pow(wscale, 4.));
      hgamu = brint * ux(1);
      hgamv = brint * vx(1);
    } else {
      pblflg = false;
    }

    // enhance the pbl height by considering the thermal
    if (pblflg) {
      kpbl = 1;
      hpbl = zq(1);
    }
    if (pblflg) {
      stable = false;
      brup = br;
      brcr = brcr_ub;
    }
    for (int k = 2; k <= klpbl; ++k) {
      if (!stable && pblflg) {
        brdn = brup;
        const Real spdk2 = fmax2(ux(k) * ux(k) + vx(k) * vx(k), 1.);
        brup = (thvx(k) - thermal) * (g * za(k) / thvx(1)) / spdk2;
        kpbl = k;
        stable = brup > brcr;
      }
    }

    // enhance pbl by theta-li
    if (ysu_topdown_pblmix) {
      const int kpblold = kpbl;
      bool definebrup = false;
      for (int k = kpblold; k <= kte - 1; ++k) {
        const Real spdk2 = fmax2(ux(k) * ux(k) + vx(k) * vx(k), 1.);
        const Real bruptmp = (thlix(k) - thermalli) * (g * za(k) / thlix(1)) / spdk2;
        stable = bruptmp >= brcr;
        if (definebrup) {
          kpbl = k;
          brup = bruptmp;
          definebrup = false;
        }
        if (!stable) {   // overwrite brup brdn values
          brdn = bruptmp;
          definebrup = true;
          pblflg = true;
        }
      }
    }

    if (pblflg) {
      const int k = kpbl;
      Real brint;
      if (brdn >= brcr) brint = 0.;
      else if (brup <= brcr) brint = 1.;
      else brint = (brcr - brdn) / (brup - brdn);
      hpbl = za(k - 1) + brint * (za(k) - za(k - 1));
      if (hpbl < zq(2)) kpbl = 1;
      if (kpbl <= 1) pblflg = false;
    }

    // stable boundary layer
    if ((!sfcflg) && hpbl < zq(2)) {
      brup = br;
      stable = false;
    } else {
      stable = true;
    }
    Real brcr_sbro = 0;
    if ((!stable) && ((xland - 1.5) >= 0)) {
      Real wspd10 = u10 * u10 + v10 * v10;
      wspd10 = Kokkos::sqrt(wspd10);
      const Real ross = wspd10 / (cori * znt);
      brcr_sbro = fmin2(0.16 * Kokkos::pow((1.e-7 * ross), (-0.18)), .3);
    }
    if (!stable) {
      if ((xland - 1.5) >= 0) brcr = brcr_sbro;
      else brcr = brcr_sb;
    }
    for (int k = 2; k <= klpbl; ++k) {
      if (!stable) {
        brdn = brup;
        const Real spdk2 = fmax2(ux(k) * ux(k) + vx(k) * vx(k), 1.);
        brup = (thvx(k) - thermal) * (g * za(k) / thvx(1)) / spdk2;
        kpbl = k;
        stable = brup > brcr;
      }
    }
    if ((!sfcflg) && hpbl < zq(2)) {
      const int k = kpbl;
      Real brint;
      if (brdn >= brcr) brint = 0.;
      else if (brup <= brcr) brint = 1.;
      else brint = (brcr - brdn) / (brup - brdn);
      hpbl = za(k - 1) + brint * (za(k) - za(k - 1));
      if (hpbl < zq(2)) kpbl = 1;
      if (kpbl <= 1) pblflg = false;
    }

    // estimate the entrainment parameters
    bool cloudflg = false;
    Real we = 0, wm2 = 0, prpbl = 0;
    if (pblflg) {
      const int k = kpbl - 1;
      Real wm3 = wstar3 + 5. * ust3;
      wm2 = Kokkos::pow(wm3, h2);
      bfxpbl = -0.15 * thvx(1) / g * wm3 / hpbl;
      Real dthvx = fmax2(thvx(k + 1) - thvx(k), tmin);
      we = fmax2(bfxpbl / dthvx, -Kokkos::sqrt(wm2));
      if ((qcxl(k) + qixl(k)) > 0.01e-3 && ysu_topdown_pblmix) {
        if (kpbl >= 2) {
          cloudflg = true;
          const Real templ = thlix(k) * Kokkos::pow(p2di(k + 1) / 100000., rovcp);
          // rvls is ws at full level
          Real rvls = 100. * 6.112 * Kokkos::exp(17.67 * (templ - 273.16) / (templ - 29.65)) * (ep2 / p2di(k + 1));
          const Real temps = templ + ((qvx(k) + qcxl(k)) - rvls) / (cp / xlv + ep2 * xlv * rvls / (rd * (templ * templ)));
          rvls = 100. * 6.112 * Kokkos::exp(17.67 * (temps - 273.15) / (temps - 29.65)) * (ep2 / p2di(k + 1));
          const Real rcldb = fmax2((qvx(k) + qcxl(k)) - rvls, 0.);
          // entrainment efficiency
          dthvx = (thlix(k + 2) + thx(k + 2) * ep1 * (qvx(k + 2) + qcxl(k + 2))) -
                  (thlix(k) + thx(k) * ep1 * (qvx(k) + qcxl(k)));
          dthvx = fmax2(dthvx, 0.1);
          const Real tmp1 = xlv / cp * rcldb / (pi2d(k) * dthvx);
          const Real ent_eff = 0.2 * 8. * tmp1 + 0.2;

          Real radsum = 0.;
          for (int kk = 1; kk <= kpbl - 1; ++kk) {
            Real radflux = I.rthraten(i, j, kk) * pi2d(kk);                // converts theta/s to temp/s
            radflux = radflux * cp / g * (p2di(kk) - p2di(kk + 1));        // converts temp/s to W/m^2
            if (radflux < 0.0) radsum = Kokkos::fabs(radflux) + radsum;
          }
          radsum = fmax2(radsum, 0.0);

          // recompute entrainment from sfc thermals
          Real bfx0 = fmax2(sflux, 0.0);
          wm3 = (govrth * bfx0 * hpbl) + 5. * ust3;
          wm2 = Kokkos::pow(wm3, h2);
          bfxpbl = -0.15 * thvx(1) / g * wm3 / hpbl;
          dthvx = fmax2(thvx(k + 1) - thvx(k), tmin);
          we = fmax2(bfxpbl / dthvx, -Kokkos::sqrt(wm2));

          // entrainment from PBL top thermals
          bfx0 = fmax2(radsum / rhox2(k) / cp, 0.);
          wm3 = (g / thvx(k) * bfx0 * hpbl);   // this is wstar3(i)
          wm2 = wm2 + Kokkos::pow(wm3, h2);
          bfxpbl = -ent_eff * bfx0;
          dthvx = fmax2(thvx(k + 1) - thvx(k), 0.1);
          we = we + fmax2(bfxpbl / dthvx, -Kokkos::sqrt(Kokkos::pow(wm3, h2)));

          // wstar3_2
          bfx0 = fmax2(radsum / rhox2(k) / cp, 0.);
          wstar3_2 = (g / thvx(k) * bfx0 * hpbl);
          // recompute hgamt
          wscale = Kokkos::pow((ust3 + phifac * karman * (wstar3 + wstar3_2) * 0.5), h1);
          wscale = fmin2(wscale, ust * aphi16);
          wscale = fmax2(wscale, ust / aphi5);
          Real gamfac = bfac / rhox / wscale;
          hgamt = fmin2(gamfac * hfx / cp, gamcrt);
          hgamq = fmin2(gamfac * qfx, gamcrq);
          gamfac = bfac / rhox2(k) / wscale;
          const Real hgamt2 = fmin2(gamfac * radsum / cp, gamcrt);
          hgamt = fmax2(hgamt, 0.0) + fmax2(hgamt2, 0.0);
          const Real brint = -15.9 * ust * ust / wspd * (wstar3 + wstar3_2) / (Kokkos::pow(wscale, 4.));
          hgamu = brint * ux(1);
          hgamv = brint * vx(1);
        }
      }
      prpbl = 1.0;
      const Real dthx = fmax2(thx(k + 1) - thx(k), tmin);
      const Real dqx = fmin2(qvx(k + 1) - qvx(k), 0.0);
      hfxpbl = we * dthx;
      qfxpbl = we * dqx;

      const Real dux = ux(k + 1) - ux(k);
      const Real dvx = vx(k + 1) - vx(k);
      if (dux > tmin) ufxpbl = fmax2(prpbl * we * dux, -ust * ust);
      else if (dux < -tmin) ufxpbl = fmin2(prpbl * we * dux, ust * ust);
      else ufxpbl = 0.0;
      if (dvx > tmin) vfxpbl = fmax2(prpbl * we * dvx, -ust * ust);
      else if (dvx < -tmin) vfxpbl = fmin2(prpbl * we * dvx, ust * ust);
      else vfxpbl = 0.0;
      const Real delb = govrth * d3 * hpbl;
      delta = fmin2(d1 * hpbl + d2 * wm2 / delb, 100.);
    }

    for (int k = 1; k <= klpbl; ++k) {
      if (pblflg && k >= kpbl) {
        const Real x = (zq(k + 1) - hpbl) / delta;
        entfac(k) = x * x;
      } else {
        entfac(k) = 1.e30;
      }
    }

    // diffusion coefficients below pbl
    for (int k = 1; k <= klpbl; ++k) {
      if (k < kpbl) {
        const Real zfac = fmin2(fmax2((1. - (zq(k + 1) - zl1) / (hpbl - zl1)), zfmin), 1.);
        zfacent(k) = Kokkos::pow((1. - zfac), 3.);
        Real wscalek = Kokkos::pow((ust3 + phifac * karman * wstar3 * (1. - zfac)), h1);
        const Real wscalek2 = Kokkos::pow((phifac * karman * wstar3_2 * (zfac)), h1);
        Real prfac, prfac2, prnumfac;
        if (sfcflg) {
          prfac = conpr;
          prfac2 = 15.9 * (wstar3 + wstar3_2) / ust3 / (1. + 4. * karman * (wstar3 + wstar3_2) / ust3);
          const Real m = fmax2(zq(k + 1) - sfcfrac * hpbl, 0.);
          prnumfac = -3. * (m * m) / (hpbl * hpbl);
        } else {
          prfac = 0.;
          prfac2 = 0.;
          prnumfac = 0.;
          const Real phim8z = 1. + aphi5 * zol1 * zq(k + 1) / zl1;
          wscalek = ust / phim8z;
          wscalek = fmax2(wscalek, 0.001);
        }
        Real prnum0 = (phih / phim + prfac);
        prnum0 = fmax2(fmin2(prnum0, prmax), prmin);
        xkzm(k) = wscalek * karman * zq(k + 1) * (zfac * zfac) +
                  wscalek2 * karman * (hpbl - zq(k + 1)) * ((1 - zfac) * (1 - zfac));
        // Do not include xkzm at kpbl-1 since it changes entrainment
        if (k == kpbl - 1 && cloudflg && we < 0.0) xkzm(k) = 0.0;
        Real prnum = 1. + (prnum0 - 1.) * Kokkos::exp(prnumfac);
        xkzq(k) = xkzm(k) / prnum * Kokkos::pow(zfac, (pfac_q - pfac));
        prnum0 = prnum0 / (1. + prfac2 * karman * sfcfrac);
        prnum = 1. + (prnum0 - 1.) * Kokkos::exp(prnumfac);
        xkzh(k) = xkzm(k) / prnum;
        xkzm(k) = xkzm(k) + xkzminm;
        xkzh(k) = xkzh(k) + xkzminh;
        xkzq(k) = xkzq(k) + xkzminh;
        xkzm(k) = fmin2(xkzm(k), xkzmax);
        xkzh(k) = fmin2(xkzh(k), xkzmax);
        xkzq(k) = fmin2(xkzq(k), xkzmax);
      }
    }

    // diffusion coefficients over pbl (free atmosphere)
    for (int k = 1; k <= kte - 1; ++k) {
      if (k >= kpbl) {
        const Real ss = ((ux(k + 1) - ux(k)) * (ux(k + 1) - ux(k)) + (vx(k + 1) - vx(k)) * (vx(k + 1) - vx(k))) /
                            (dza(k + 1) * dza(k + 1)) +
                        1.e-9;
        const Real govrthv = g / (0.5 * (thvx(k + 1) + thvx(k)));
        Real ri = govrthv * (thvx(k + 1) - thvx(k)) / (ss * dza(k + 1));
        if (imvdif == 1) {
          if ((qcxl(k) + qixl(k)) > 0.01e-3 && (qcxl(k + 1) + qixl(k + 1)) > 0.01e-3) {
            // in cloud
            const Real qmean = 0.5 * (qvx(k) + qvx(k + 1));
            const Real tmean = 0.5 * (tx(k) + tx(k + 1));
            const Real alph = xlv * qmean / rd / tmean;
            const Real chi = xlv * xlv * qmean / cp / rv / tmean / tmean;
            ri = (1. + alph) * (ri - g * g / ss / tmean / cp * ((chi - alph) / (1. + chi)));
          }
        }
        const Real zk = karman * zq(k + 1);
        Real rlamdz = fmin2(fmax2(0.1 * dza(k + 1), rlam), 300.);
        rlamdz = fmin2(dza(k + 1), rlamdz);
        const Real rl = zk * rlamdz / (rlamdz + zk);
        const Real rl2 = rl * rl;
        const Real dk = rl2 * Kokkos::sqrt(ss);
        if (ri < 0.) {
          // unstable regime
          ri = fmax2(ri, rimin);
          const Real sri = Kokkos::sqrt(-ri);
          xkzm(k) = dk * (1 + 8. * (-ri) / (1 + 1.746 * sri));
          xkzh(k) = dk * (1 + 8. * (-ri) / (1 + 1.286 * sri));
        } else {
          // stable regime
          const Real r = (1 + 5. * ri);
          xkzh(k) = dk / (r * r);
          Real prnum = 1.0 + 2.1 * ri;
          prnum = fmin2(prnum, prmax);
          xkzm(k) = xkzh(k) * prnum;
        }
        xkzm(k) = xkzm(k) + xkzminm;
        xkzh(k) = xkzh(k) + xkzminh;
        xkzm(k) = fmin2(xkzm(k), xkzmax);
        xkzh(k) = fmin2(xkzh(k), xkzmax);
        xkzml(k) = xkzm(k);
        xkzhl(k) = xkzh(k);
      }
    }

    //------------------------------------------------------------------
    // tridiagonal matrix elements for heat
    for (int k = 1; k <= kte; ++k) {
      au(k) = 0.;
      al(k) = 0.;
      ad(k) = 0.;
      f1(k) = 0.;
    }
    ad(1) = 1.;
    f1(1) = thx(1) - 300. + (1.0 - bepswitch) * hfx / cont / del(1) * dt2;
    for (int k = 1; k <= kte - 1; ++k) {
      const Real dtodsd = sfk2d * dt2 / del(k);
      const Real dtodsu = sfk2d * dt2 / del(k + 1);
      const Real dsig = p2d(k) - p2d(k + 1);
      const Real rdz = 1. / dza(k + 1);
      Real tem1 = dsig * xkzh(k) * rdz;
      if (pblflg && k < kpbl) {
        const Real dsdzt = tem1 * (-hgamt / hpbl - hfxpbl * zfacent(k) / xkzh(k));
        f1(k) = f1(k) + dtodsd * dsdzt;
        f1(k + 1) = thx(k + 1) - 300. - dtodsu * dsdzt;
      } else if (pblflg && k >= kpbl && entfac(k) < 4.6) {
        xkzh(k) = -we * dza(kpbl) * Kokkos::exp(-entfac(k));
        xkzh(k) = Kokkos::sqrt(xkzh(k) * xkzhl(k));
        xkzh(k) = fmax2(xkzh(k), xkzminh);
        xkzh(k) = fmin2(xkzh(k), xkzmax);
        f1(k + 1) = thx(k + 1) - 300.;
      } else {
        f1(k + 1) = thx(k + 1) - 300.;
      }
      tem1 = dsig * xkzh(k) * rdz;
      const Real dsdz2 = tem1 * rdz;
      au(k) = -dtodsd * dsdz2 / vlk2d;
      al(k) = -dtodsu * dsdz2 / vlk2d;
      ad(k) = ad(k) - au(k);
      ad(k + 1) = 1. - al(k);
      O.exch_hx(i, j, k + 1) = xkzh(k);
    }
    // bep/bep+bem forcing for heat (zero)
    for (int k = 1; k <= kte; ++k) {
      ad(k) = ad(k) - a_2d * dt2;
      f1(k) = f1(k) + b_2d * dt2;
    }
    tridin_ysu(al, ad, au, aul, f1, kte);
    // recover tendencies of heat
    for (int k = kte; k >= 1; --k) O.ttnp(i, j, k) = (f1(k) - thx(k) + 300.) * rdt;

    //------------------------------------------------------------------
    // water vapor, cloud water, cloud ice
    for (int k = 1; k <= kte - 1; ++k)
      if (k >= kpbl) xkzq(k) = xkzh(k);
    for (int k = 1; k <= kte; ++k) {
      au(k) = 0.;
      al(k) = 0.;
      ad(k) = 0.;
      f1(k) = 0.;
    }
    ad(1) = 1.;
    f1(1) = qvx(1) + (1.0 - bepswitch) * qfx * g / del(1) * dt2;
    for (int k = 1; k <= kte - 1; ++k) {
      const Real dtodsd = sfk2d * dt2 / del(k);
      const Real dtodsu = sfk2d * dt2 / del(k + 1);
      const Real dsig = p2d(k) - p2d(k + 1);
      const Real rdz = 1. / dza(k + 1);
      Real tem1 = dsig * xkzq(k) * rdz;
      if (pblflg && k < kpbl) {
        const Real dsdzq = tem1 * (-qfxpbl * zfacent(k) / xkzq(k));
        f1(k) = f1(k) + dtodsd * dsdzq;
        f1(k + 1) = qvx(k + 1) - dtodsu * dsdzq;
      } else if (pblflg && k >= kpbl && entfac(k) < 4.6) {
        xkzq(k) = -we * dza(kpbl) * Kokkos::exp(-entfac(k));
        xkzq(k) = Kokkos::sqrt(xkzq(k) * xkzhl(k));
        xkzq(k) = fmax2(xkzq(k), xkzminh);
        xkzq(k) = fmin2(xkzq(k), xkzmax);
        f1(k + 1) = qvx(k + 1);
      } else {
        f1(k + 1) = qvx(k + 1);
      }
      tem1 = dsig * xkzq(k) * rdz;
      const Real dsdz2 = tem1 * rdz;
      au(k) = -dtodsd * dsdz2 / vlk2d;
      al(k) = -dtodsu * dsdz2 / vlk2d;
      ad(k) = ad(k) - au(k);
      ad(k + 1) = 1. - al(k);
    }
    for (int k = 1; k <= kte; ++k) {
      adv(k) = ad(k) - a_2d * dt2;
      f1(k) = f1(k) + b_2d * dt2;
    }
    tridin_ysu(al, adv, au, aul, f1, kte);
    for (int k = kte; k >= 1; --k) O.qvtnp(i, j, k) = (f1(k) - qvx(k)) * rdt;

    // cloud water (f_qc)
    for (int k = 1; k <= kte; ++k) f1(k) = qcxl(k);
    tridin_ysu(al, ad, au, aul, f1, kte);
    for (int k = kte; k >= 1; --k) O.qctnp(i, j, k) = (f1(k) - qcxl(k)) * rdt;

    // cloud ice (f_qi)
    for (int k = 1; k <= kte; ++k) f1(k) = qixl(k);
    tridin_ysu(al, ad, au, aul, f1, kte);
    for (int k = kte; k >= 1; --k) O.qitnp(i, j, k) = (f1(k) - qixl(k)) * rdt;

    //------------------------------------------------------------------
    // tridiagonal matrix elements for momentum
    for (int k = 1; k <= kte; ++k) {
      au(k) = 0.;
      al(k) = 0.;
      ad(k) = 0.;
      f1(k) = 0.;
      f2(k) = 0.;
    }
    {
      // (the paj tke / get_pblh / vconv block only matters with ctopo)
      const Real r = (wspd1 / wspd);
      const Real fric1 = ust * ust / wspd1 * rhox * g / del(1) * dt2 * (r * r);
      ad(1) = 1. + fric1;
      f1(1) = ux(1) + uoxl * (ust * ust) * rhox * g / del(1) * dt2 / wspd1 * (r * r);
      f2(1) = vx(1) + voxl * (ust * ust) * rhox * g / del(1) * dt2 / wspd1 * (r * r);
    }
    for (int k = 1; k <= kte - 1; ++k) {
      const Real dtodsd = sfk2d * dt2 / del(k);
      const Real dtodsu = sfk2d * dt2 / del(k + 1);
      const Real dsig = p2d(k) - p2d(k + 1);
      const Real rdz = 1. / dza(k + 1);
      Real tem1 = dsig * xkzm(k) * rdz;
      if (pblflg && k < kpbl) {
        const Real dsdzu = tem1 * (-hgamu / hpbl - ufxpbl * zfacent(k) / xkzm(k));
        const Real dsdzv = tem1 * (-hgamv / hpbl - vfxpbl * zfacent(k) / xkzm(k));
        f1(k) = f1(k) + dtodsd * dsdzu;
        f1(k + 1) = ux(k + 1) - dtodsu * dsdzu;
        f2(k) = f2(k) + dtodsd * dsdzv;
        f2(k + 1) = vx(k + 1) - dtodsu * dsdzv;
      } else if (pblflg && k >= kpbl && entfac(k) < 4.6) {
        xkzm(k) = prpbl * xkzh(k);
        xkzm(k) = Kokkos::sqrt(xkzm(k) * xkzml(k));
        xkzm(k) = fmax2(xkzm(k), xkzminm);
        xkzm(k) = fmin2(xkzm(k), xkzmax);
        f1(k + 1) = ux(k + 1);
        f2(k + 1) = vx(k + 1);
      } else {
        f1(k + 1) = ux(k + 1);
        f2(k + 1) = vx(k + 1);
      }
      tem1 = dsig * xkzm(k) * rdz;
      const Real dsdz2 = tem1 * rdz;
      au(k) = -dtodsd * dsdz2 / vlk2d;
      al(k) = -dtodsu * dsdz2 / vlk2d;
      ad(k) = ad(k) - au(k);
      ad(k + 1) = 1. - al(k);
      O.exch_mx(i, j, k + 1) = xkzm(k);
    }
    for (int k = 1; k <= kte; ++k) ad1(k) = ad(k);
    for (int k = 1; k <= kte; ++k) {
      ad(k) = ad(k) - a_2d * dt2;
      ad1(k) = ad1(k) - a_2d * dt2;
      f1(k) = f1(k) + b_2d * dt2;
      f2(k) = f2(k) + b_2d * dt2;
    }
    tridi2n(al, ad, ad1, au, f1, f2, kte);
    // recover tendencies of momentum
    for (int k = kte; k >= 1; --k) {
      O.utnp(i, j, k) = (f1(k) - ux(k)) * rdt;
      O.vtnp(i, j, k) = (f2(k) - vx(k)) * rdt;
    }

    O.hpbl(i, j, 1) = hpbl;
    O.kpbl2d(i, j, 1) = Real(kpbl);
    O.wstar(i, j, 1) = wstar;
    O.delta(i, j, 1) = delta;
  });
}

void ysu_dissten(int ni, int nj, int nk, Real dz, Real rdz, const FArray3& rmh, const FArray3& mf,
                 const FArray3& upten, const FArray3& vpten, const FArray3& dum1, const FArray3& dum2,
                 const FArray3& thten, const FArray3& thten1, const FArray3& dissten)
{
  Kokkos::parallel_for("cm1k::pbl_ysu::ysu_dissten", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    // assume t13,t23 are zero at top of domain
    thten(i, j, nk + 1) = 0.0;
    thten1(i, j, nk + 1) = 0.0;
    for (int k = nk; k >= 2; --k) {
      const Real tem1 = dz * rmh(i, j, k);
      const Real tem2 = rdz * mf(i, j, k);
      // get stresses from u,v tendencies
      thten(i, j, k) = thten(i, j, k + 1) - upten(i, j, k) * tem1;
      thten1(i, j, k) = thten1(i, j, k + 1) - vpten(i, j, k) * tem1;
      // NOTE: dissten is defined at w points
      dissten(i, j, k) = dissten(i, j, k) + (thten(i, j, k) * (dum1(i, j, k) - dum1(i, j, k - 1)) * tem2 +
                                             thten1(i, j, k) * (dum2(i, j, k) - dum2(i, j, k - 1)) * tem2);
    }
  });
}

} // namespace cm1k::pbl_ysu
