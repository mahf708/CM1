// Port of the GFS-EDMF PBL (src/module_bl_gfsedmf.F) as CM1 calls it. See
// cm1k_pbl_gfs.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// real**real is pow (also **3.), x**2 and x**pfac (pfac = 2.0, which gcc
// folds to a square) are x*x, Fortran's max/min of two numbers is
// fmax2/fmin2. The Fortran's i-loops only couple points of the same column,
// so each column runs them in order. The accumulators that start at zero
// (tau, rtg, du, dv) keep the 0 + x of the Fortran (x = -0 gives +0).
#include "physics/pbl_gfs/cm1k_pbl_gfs.hpp"

#include <stdexcept>

namespace cm1k::pbl_gfs {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// module_gfs_physcons
constexpr Real con_g = 9.80665e+0, con_rd = 2.8705e+2, con_rv = 4.6150e+2, con_cp = 1.0046e+3;
constexpr Real con_hvap = 2.5000e+6;
constexpr Real con_fvirt = con_rv / con_rd - 1.;

// moninedmf parameters
constexpr Real grav = con_g, cp = con_cp, hvap = con_hvap, fv = con_fvirt;
constexpr Real gravi = 1.0 / grav, g = grav, gocp = g / cp;
constexpr Real rlam = 30.0, vk = 0.4;
constexpr Real prmin = 0.25, prmax = 4., zolcr = 0.2, zolcru = -0.5;
constexpr Real dw2min = 0.0001, dkmin = 0.0, dkmax = 1000., rimin = -100.;
constexpr Real crbcon = 0.25, crbmin = 0.15, crbmax = 0.35;
constexpr Real wfac = 7.0, cfac = 6.5, sfcfrac = 0.1;   // pfac = 2.0: squares
constexpr Real qmin = 1.e-8, zfmin = 1.e-8, aphi5 = 5., aphi16 = 16.;
constexpr Real tdzmin = 1.e-3, qlmin = 1.e-12, f0 = 1.e-4;
constexpr Real h1 = 0.33333333;
constexpr Real cldtime = 500., xkzminv = 0.3;
constexpr Real gamcrt = 3., gamcrq = 0., rlamun = 150.0;
constexpr Real rentf1 = 0.2, rentf2 = 1.0, radfac = 0.85;
constexpr Real zstblmax = 2500., qlcr = 3.5e-5;
constexpr Real actei = 0.7;
// BL_GFSEDMF: background diffusivities
constexpr Real xkzm_m = 1.0, xkzm_h = 1.0, xkzm_s = 1.0;
// mfpbl parameters
constexpr Real ce0 = 0.38, alp = 1.0, pgcon = 0.55, a1c = 0.08;
constexpr Real bb1 = 1.8, bb2 = 3.5;

// per-column work arrays: index n of Workspace::w (levels 1..nk+1; DISS
// holds diss(0..nk) at slots 1..nk+1)
enum Work : int {
  W_Q1, W_Q2, W_Q3, W_PRSL, W_PRSLK, W_DEL, W_PRSI, W_ZI, W_ZL, W_RDZT, W_XKZO, W_XKZMO, W_THETA, W_QLX,
  W_QTX, W_THETAE, W_THVX, W_THLVX, W_DKU, W_DKT, W_DKTX, W_CKU, W_CKT, W_RADX, W_BF, W_TI, W_SHR2, W_TCKO,
  W_QCKO1, W_QCKO2, W_QCKO3, W_UCKO, W_VCKO, W_XMF, W_WU2, W_XLAMUE, W_THVU, W_BUO, W_AD, W_A1, W_AU, W_AL,
  W_A21, W_A22, W_A23, W_FKK, W_TAU, W_RTG1, W_RTG2, W_RTG3, W_DU, W_DV, W_DISHX, W_DISS, NWORK
};

using WView = Kokkos::View<Real****, Kokkos::LayoutLeft, Device>;

// A column of one work array, Fortran-indexed k = 1..nk+1.
struct Col {
  Real* p;
  size_t s;
  KOKKOS_INLINE_FUNCTION Real& operator()(int k) const { return p[size_t(k - 1) * s]; }
};

// tridi2 (in place: au = cu, a1 = r1, a2 = r2 in the call); cl(k) is al(k-1).
KOKKOS_INLINE_FUNCTION void tridi2(int n, const Col& al, const Col& cm, const Col& au, const Col& a1,
                                   const Col& a2) {
  Real fk = 1. / cm(1);
  au(1) = fk * au(1);
  a1(1) = fk * a1(1);
  a2(1) = fk * a2(1);
  for (int k = 2; k <= n - 1; ++k) {
    fk = 1. / (cm(k) - al(k - 1) * au(k - 1));
    au(k) = fk * au(k);
    a1(k) = fk * (a1(k) - al(k - 1) * a1(k - 1));
    a2(k) = fk * (a2(k) - al(k - 1) * a2(k - 1));
  }
  fk = 1. / (cm(n) - al(n - 1) * au(n - 1));
  a1(n) = fk * (a1(n) - al(n - 1) * a1(n - 1));
  a2(n) = fk * (a2(n) - al(n - 1) * a2(n - 1));
  for (int k = n - 1; k >= 1; --k) {
    a1(k) = a1(k) - au(k) * a1(k + 1);
    a2(k) = a2(k) - au(k) * a2(k + 1);
  }
}

// tridin with nt tracers in a2[0..nt-1] (in place, as called).
KOKKOS_INLINE_FUNCTION void tridin(int n, int nt, const Col& al, const Col& cm, const Col& au, const Col& a1,
                                   const Col* a2, const Col& fkk) {
  Real fk = 1. / cm(1);
  au(1) = fk * au(1);
  a1(1) = fk * a1(1);
  for (int t = 0; t < nt; ++t) a2[t](1) = fk * a2[t](1);
  for (int k = 2; k <= n - 1; ++k) {
    fkk(k) = 1. / (cm(k) - al(k - 1) * au(k - 1));
    au(k) = fkk(k) * au(k);
    a1(k) = fkk(k) * (a1(k) - al(k - 1) * a1(k - 1));
  }
  for (int t = 0; t < nt; ++t)
    for (int k = 2; k <= n - 1; ++k) a2[t](k) = fkk(k) * (a2[t](k) - al(k - 1) * a2[t](k - 1));
  fk = 1. / (cm(n) - al(n - 1) * au(n - 1));
  a1(n) = fk * (a1(n) - al(n - 1) * a1(n - 1));
  for (int t = 0; t < nt; ++t) a2[t](n) = fk * (a2[t](n) - al(n - 1) * a2[t](n - 1));
  for (int k = n - 1; k >= 1; --k) a1(k) = a1(k) - au(k) * a1(k + 1);
  for (int t = 0; t < nt; ++t)
    for (int k = n - 1; k >= 1; --k) a2[t](k) = a2[t](k) - au(k) * a2[t](k + 1);
}

} // namespace

void Workspace::ensure(int ni, int nj, int nk) {
  if (!err.data()) err = Kokkos::View<int, Device>("cm1k::pbl_gfs::err");
  if (w.extent(0) == size_t(ni) && w.extent(1) == size_t(nj) && w.extent(2) == size_t(nk + 2)) return;
  w = WView("cm1k::pbl_gfs::work", ni, nj, nk + 2, NWORK);
}

void gfsedmf(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
             const Outputs& out, Workspace& ws) {
  if (nk < 4) throw std::runtime_error("cm1k pbl_gfs: needs nk >= 4");
  ws.ensure(ni, nj, nk);
  Kokkos::deep_copy(ws.err, 0);
  const WView W = ws.w;
  const auto err = ws.err;
  const Inputs I = in;
  const Outputs O = out;
  const Real CP = c.cp, G = c.g, ROVCP = c.rovcp, R = c.rd, ROVG = c.rovg, EP1 = c.ep1, KARMAN = c.karman;
  const Real alpha = o.alpha, var_ric = o.var_ric, coef_ric_l = o.coef_ric_l, coef_ric_s = o.coef_ric_s;
  const bool dspheat = o.disheat;
  const int ntrac = o.qi ? 3 : 2;
  const int km = nk;

  Kokkos::parallel_for("cm1k::pbl_gfs::gfsedmf", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    const size_t ks = W.stride(2);
    auto col = [&](int n) { return Col{&W(i - 1, j - 1, 0, n), ks}; };
    const Col prsl = col(W_PRSL), prslk = col(W_PRSLK), del = col(W_DEL), prsi = col(W_PRSI), zi = col(W_ZI),
              zl = col(W_ZL), rdzt = col(W_RDZT), xkzo = col(W_XKZO), xkzmo = col(W_XKZMO), theta = col(W_THETA),
              qlx = col(W_QLX), qtx = col(W_QTX), thetae = col(W_THETAE), thvx = col(W_THVX),
              thlvx = col(W_THLVX), dku = col(W_DKU), dkt = col(W_DKT), dktx = col(W_DKTX), cku = col(W_CKU),
              ckt = col(W_CKT), radx = col(W_RADX), bf = col(W_BF), ti = col(W_TI), shr2 = col(W_SHR2),
              tcko = col(W_TCKO), ucko = col(W_UCKO), vcko = col(W_VCKO), xmf = col(W_XMF), wu2 = col(W_WU2),
              xlamue = col(W_XLAMUE), thvu = col(W_THVU), buo = col(W_BUO), ad = col(W_AD), a1 = col(W_A1),
              au = col(W_AU), al = col(W_AL), fkk = col(W_FKK), tau = col(W_TAU), du = col(W_DU), dv = col(W_DV),
              dishx = col(W_DISHX), dissc = col(W_DISS);
    const Col q1[3] = {col(W_Q1), col(W_Q2), col(W_Q3)};
    const Col qcko[3] = {col(W_QCKO1), col(W_QCKO2), col(W_QCKO3)};
    const Col a2[3] = {col(W_A21), col(W_A22), col(W_A23)};
    const Col rtg[3] = {col(W_RTG1), col(W_RTG2), col(W_RTG3)};
    auto diss = [&](int k) -> Real& { return dissc(k + 1); };   // diss(0:km)
    auto u1 = [&](int k) { return I.u3d(i, j, k); };
    auto v1 = [&](int k) { return I.v3d(i, j, k); };
    auto t1 = [&](int k) { return I.t3d(i, j, k); };

    //------------------------------------------------------------------
    // BL_GFSEDMF: column inputs
    const Real qv1 = I.qv3d(i, j, 1);
    const Real rrhox = (R * I.t3d(i, j, 1) * (1. + EP1 * qv1)) / I.psfc(i, j, 1);
    const Real cpm = CP * (1. + 0.8 * qv1);
    const Real fmtmp = I.gz1oz0(i, j, 1) - I.psim(i, j, 1);
    const Real psk = Kokkos::pow(I.psfc(i, j, 1) * .00001, ROVCP);
    const Real fm = fmtmp;
    const Real fh = I.gz1oz0(i, j, 1) - I.psih(i, j, 1);
    const Real tsea = I.tsk(i, j, 1);
    const Real heat = I.hfx(i, j, 1) / cpm * rrhox;
    const Real evap = I.qfx(i, j, 1) * rrhox;
    const Real wspd0 = O.wspd(i, j, 1);
    const Real stress = KARMAN * KARMAN * wspd0 * wspd0 / (fmtmp * fmtmp);
    const Real spd1 = wspd0;
    prsi(1) = I.psfc(i, j, 1) * .001;
    zi(1) = 0.;   // phii(1)
    const Real rbsoil = I.br(i, j, 1);
    const Real zorl = I.znt(i, j, 1) * 100.0;
    const Real u10m = I.u10(i, j, 1), v10m = I.v10(i, j, 1);
    const Real xland1 = I.xland(i, j, 1);
    const int kinver = km;

    for (int k = 1; k <= km; ++k) {
      du(k) = 0.;
      dv(k) = 0.;
      tau(k) = 0.;
      q1[0](k) = I.qv3d(i, j, k) / (1. + I.qv3d(i, j, k));
      q1[1](k) = I.qc3d(i, j, k) / (1. + I.qc3d(i, j, k));
      prsl(k) = I.p3d(i, j, k) * .001;
      dishx(k) = 0.0;
    }
    for (int k = 1; k <= km; ++k) prslk(k) = Kokkos::pow(prsl(k) * .01, ROVCP);
    // del, prsi; phii (in zi) and phil (in zl)
    for (int k = 2; k <= km; ++k) {
      const int kmm = k - 1;
      del(kmm) = prsl(kmm) / ROVG * I.dz8w(i, j, kmm) / I.t3d(i, j, kmm);
      prsi(k) = prsi(kmm) - del(kmm);
      zi(k) = (I.z(i, j, k) - I.z(i, j, 1)) * G;
      zl(kmm) = 0.5 * (I.z(i, j, k) + I.z(i, j, kmm) - 2. * I.z(i, j, 1)) * G;
    }
    del(km) = del(km - 1);
    prsi(km + 1) = prsi(km) - del(km - 1);
    zi(km + 1) = zi(km) + I.dz8w(i, j, km) * G;
    zl(km) = zi(km) - zl(km - 1) + zi(km);
    if (ntrac == 3)
      for (int k = 1; k <= km; ++k) q1[2](k) = I.qi3d(i, j, k) / (1. + I.qi3d(i, j, k));
    for (int l = 0; l < ntrac; ++l)
      for (int k = 1; k <= km; ++k) rtg[l](k) = 0.;

    //------------------------------------------------------------------
    // moninedmf
    const Real dt2 = dt;
    const Real rdt = 1. / dt2;
    const int km1 = km - 1;
    const int kmpbl = km / 2;
    for (int k = 1; k <= km; ++k) {
      zi(k) = zi(k) * gravi;
      zl(k) = zl(k) * gravi;
    }
    zi(km + 1) = zi(km + 1) * gravi;
    for (int k = 1; k <= km1; ++k) rdzt(k) = 1.0 / (zl(k + 1) - zl(k));
    // background diffusivities
    int kx1 = 1;
    const Real tx1 = 1.0 / prsi(1);
    Real tx2 = tx1;
    for (int k = 1; k <= km1; ++k) {
      xkzo(k) = 0.0;
      xkzmo(k) = 0.0;
      if (k < kinver) {
        const Real ptem = prsi(k + 1) * tx1;
        Real tem1 = 1.0 - ptem;
        tem1 = tem1 * tem1 * 10.0;
        xkzo(k) = xkzm_h * fmin2(1.0, Kokkos::exp(-tem1));
        if (ptem >= xkzm_s) {
          xkzmo(k) = xkzm_m;
          kx1 = k + 1;
        } else {
          if (k == kx1 && k > 1) tx2 = 1.0 / prsi(k);
          tem1 = 1.0 - prsi(k + 1) * tx2;
          tem1 = tem1 * tem1 * 5.0;
          xkzmo(k) = xkzm_m * fmin2(1.0, Kokkos::exp(-tem1));
        }
      }
    }
    // diffusivity in the inversion layer
    for (int k = 1; k <= kmpbl; ++k) {
      if (zi(k + 1) > 250.) {
        const Real tem1 = (t1(k + 1) - t1(k)) * rdzt(k);
        if (tem1 > 1.e-5) xkzo(k) = fmin2(xkzo(k), xkzminv);
      }
    }
    const Real z0 = 0.01 * zorl;
    int kpbl = 1;
    Real hpbl = zi(1);
    Real hpblx = zi(1);
    bool pblflg = true, sfcflg = true;
    if (rbsoil > 0.) sfcflg = false;
    bool ublflg = false, pcnvflg = false, scuflg = true;
    Real radmin = 0., rent = rentf1, hrad = zi(1), zd = 0.;
    int krad = 1, icld = 0, lcld = km1, kcld = km1;

    for (int k = 1; k <= km; ++k) {
      theta(k) = t1(k) * psk / prslk(k);
      qlx(k) = fmax2(q1[1](k), qlmin);
      qtx(k) = fmax2(q1[0](k), qmin) + qlx(k);
      const Real ptem = qlx(k);
      const Real ptem1 = hvap * fmax2(q1[0](k), qmin) / (cp * t1(k));
      thetae(k) = theta(k) * (1. + ptem1);
      thvx(k) = theta(k) * (1. + fv * fmax2(q1[0](k), qmin) - ptem);
      const Real ptem2 = theta(k) - (hvap / cp) * ptem;
      thlvx(k) = ptem2 * (1. + fv * qtx(k));
    }
    for (int k = 1; k <= km1; ++k) {
      dku(k) = 0.;
      dkt(k) = 0.;
      dktx(k) = 0.;
      cku(k) = 0.;
      ckt(k) = 0.;
      const Real tem = zi(k + 1) - zi(k);
      radx(k) = tem * I.rthraten(i, j, k);
    }
    {
      bool flg = scuflg;
      for (int k = 1; k <= km1; ++k)
        if (flg && zl(k) >= zstblmax) {
          lcld = k;
          flg = false;
        }
    }
    // virtual potential temperature gradient and wind shear squared
    for (int k = 1; k <= km1; ++k) {
      const Real rdz = rdzt(k);
      bf(k) = (thvx(k + 1) - thvx(k)) * rdz;
      ti(k) = 2. / (t1(k) + t1(k + 1));
      const Real dud = u1(k) - u1(k + 1), dvd = v1(k) - v1(k + 1);
      const Real dw2 = dud * dud + dvd * dvd;
      shr2(k) = fmax2(dw2, dw2min) * rdz * rdz;
    }
    const Real govrth = g / theta(1);
    const Real beta = dt2 / (zi(2) - zi(1));
    const Real ustar = Kokkos::sqrt(stress);
    const Real sflux = heat + evap * fv * theta(1);
    if (!sfcflg || sflux <= 0.) pblflg = false;

    // PBL height (variable critical Ri)
    bool flg = false;
    Real rbup = rbsoil, rbdn = 0;
    Real thermal;
    if (pblflg) thermal = thvx(1);
    else thermal = tsea * (1. + fv * fmax2(q1[0](1), qmin));
    Real crb;
    {
      Real tem = Kokkos::sqrt(u10m * u10m + v10m * v10m);
      tem = fmax2(tem, 1.);
      const Real robn = tem / (f0 * z0);
      const Real tem1 = 1.e-7 * robn;
      crb = crbcon;
      if (var_ric == 1.) {
        if (xland1 == 1) crb = coef_ric_l * Kokkos::pow(tem1, -0.18);
        if (xland1 == 2) crb = coef_ric_s * Kokkos::pow(tem1, -0.18);
      }
      crb = fmax2(fmin2(crb, crbmax), crbmin);
    }
    for (int k = 1; k <= kmpbl; ++k) {
      if (!flg) {
        rbdn = rbup;
        const Real spdk2 = fmax2((u1(k) * u1(k) + v1(k) * v1(k)), 1.);
        rbup = (thvx(k) - thermal) * (g * zl(k) / thvx(1)) / spdk2;
        kpbl = k;
        flg = rbup > crb;
      }
    }
    if (kpbl > 1) {
      const int k = kpbl;
      Real rbint;
      if (rbdn >= crb) rbint = 0.;
      else if (rbup <= crb) rbint = 1.;
      else rbint = (crb - rbdn) / (rbup - rbdn);
      hpbl = zl(k - 1) + rbint * (zl(k) - zl(k - 1));
      if (hpbl < zi(kpbl)) kpbl = kpbl - 1;
    } else {
      hpbl = zl(1);
      kpbl = 1;
    }
    const int kpblx = kpbl;
    hpblx = hpbl;

    // similarity parameters
    Real zol = fmax2(rbsoil * fm * fm / fh, rimin);
    if (sfcflg) zol = fmin2(zol, -zfmin);
    else zol = fmax2(zol, zfmin);
    Real phim, phih;
    {
      const Real zol1 = zol * sfcfrac * hpbl / zl(1);
      if (sfcflg) {
        const Real tem = 1.0 / (1. - aphi16 * zol1);
        phih = Kokkos::sqrt(tem);
        phim = Kokkos::sqrt(phih);
      } else {
        phim = 1. + aphi5 * zol1;
        phih = phim;
      }
    }
    Real wscale = ustar / phim;
    const Real ustmin = ustar / aphi5;
    wscale = fmax2(wscale, ustmin);
    Real wstar = 0, wscaleu = 0;
    if (pblflg) {
      if (zol < zolcru && kpbl > 1) pcnvflg = true;
      else ublflg = true;
      const Real wst3 = govrth * sflux * hpbl;
      wstar = Kokkos::pow(wst3, h1);
      const Real ust3 = Kokkos::pow(ustar, 3.);
      wscaleu = Kokkos::pow(ust3 + wfac * vk * wst3 * sfcfrac, h1);
      wscaleu = fmax2(wscaleu, ustmin);
    }
    // counter-gradient terms
    Real hgamt = 0, hgamq = 0;
    if (ublflg) {
      hgamt = fmin2(cfac * heat / wscaleu, gamcrt);
      hgamq = fmin2(cfac * evap / wscaleu, gamcrq);
      Real vpert = hgamt + hgamq * fv * theta(1);
      vpert = fmin2(vpert, gamcrt);
      thermal = thermal + fmax2(vpert, 0.);
      hgamt = fmax2(hgamt, 0.0);
      hgamq = fmax2(hgamq, 0.0);
    }
    // PBL height with the thermal excess
    flg = true;
    if (ublflg) {
      flg = false;
      rbup = rbsoil;
    }
    for (int k = 2; k <= kmpbl; ++k) {
      if (!flg) {
        rbdn = rbup;
        const Real spdk2 = fmax2((u1(k) * u1(k) + v1(k) * v1(k)), 1.);
        rbup = (thvx(k) - thermal) * (g * zl(k) / thvx(1)) / spdk2;
        kpbl = k;
        flg = rbup > crb;
      }
    }
    if (ublflg) {
      const int k = kpbl;
      Real rbint;
      if (rbdn >= crb) rbint = 0.;
      else if (rbup <= crb) rbint = 1.;
      else rbint = (crb - rbdn) / (rbup - rbdn);
      hpbl = zl(k - 1) + rbint * (zl(k) - zl(k - 1));
      if (hpbl < zi(kpbl)) kpbl = kpbl - 1;
      if (kpbl <= 1) {
        ublflg = false;
        pblflg = false;
      }
    }

    // stratocumulus
    flg = scuflg;
    for (int k = kmpbl; k >= 1; --k)
      if (flg && k <= lcld)
        if (qlx(k) >= qlcr) {
          kcld = k;
          flg = false;
        }
    if (scuflg && kcld == km1) scuflg = false;
    flg = scuflg;
    for (int k = kmpbl; k >= 1; --k)
      if (flg && k <= kcld) {
        if (qlx(k) >= qlcr) {
          if (radx(k) < radmin) {
            radmin = radx(k);
            krad = k;
          }
        } else {
          flg = false;
        }
      }
    if (scuflg && krad <= 1) scuflg = false;
    if (scuflg && radmin >= 0.) scuflg = false;
    flg = scuflg;
    for (int k = kmpbl; k >= 2; --k)
      if (flg && k <= krad) {
        if (qlx(k) >= qlcr) icld = icld + 1;
        else flg = false;
      }
    if (scuflg && icld < 1) scuflg = false;
    if (scuflg) hrad = zi(krad + 1);
    if (scuflg && hrad < zi(2)) scuflg = false;
    Real thlvx1 = 0;
    if (scuflg) {
      const int k = krad;
      const Real tem = zi(k + 1) - zi(k);
      const Real tem1 = cldtime * radmin / tem;
      thlvx1 = thlvx(k) + tem1;
    }
    flg = scuflg;
    for (int k = kmpbl; k >= 1; --k)
      if (flg && k <= krad) {
        if (thlvx1 <= thlvx(k)) {
          const Real tem = zi(k + 1) - zi(k);
          zd = zd + tem;
        } else {
          flg = false;
        }
      }
    Real zdd = 0, vrad = 0;
    if (scuflg) {
      const int kk = (krad + 1 - icld) > 1 ? (krad + 1 - icld) : 1;
      zdd = hrad - zi(kk);
    }
    if (scuflg) {
      zd = fmax2(zd, zdd);
      zd = fmin2(zd, hrad);
      const Real tem = govrth * zd * (-radmin);
      vrad = Kokkos::pow(tem, h1);
    }

    // inverse Prandtl number
    Real prinv;
    {
      Real tem;
      if (ublflg) tem = phih / phim + cfac * vk * sfcfrac;
      else tem = phih / phim;
      prinv = 1.0 / tem;
      prinv = fmin2(prinv, prmax);
      prinv = fmax2(prinv, prmin);
    }
    if (zol > zolcr) kpbl = 1;

    // RGF: wind speed near 500 m (alpha < 0)
    Real wspm1 = 0, wspm2 = 0;
    bool wspm2_set = false;
    if (alpha < 0) {
      wspm1 = 0.;
      for (int k = 1; k <= kmpbl; ++k) {
        if (zi(k) <= 500 && zi(k + 1) > 500.) {
          const Real spdk2 = Kokkos::sqrt(u1(k) * u1(k) + v1(k) * v1(k));
          wspm1 = spdk2 / 0.6;
          wspm2 = Real(k);
          wspm2_set = true;
        }
      }
    }

    // diffusion coefficients below the PBL (useshape = 2)
    auto below_pbl = [&](Real afac) {
      for (int k = 1; k <= kmpbl; ++k) {
        if (k < kpbl) {
          Real sz2h = (zi(k + 1) - zl(1)) / (hpbl - zl(1));
          sz2h = fmax2(sz2h, zfmin);
          sz2h = fmin2(sz2h, 1.0);
          const Real oms = 1.0 - sz2h;
          Real zfac = oms * oms;
          constexpr Real hmax = 0.333;
          constexpr Real omh = 1.0 - hmax;
          constexpr Real skmax = hmax * (omh * omh);
          Real sksfc = fmin2(zi(2) / hpbl, 0.05);
          const Real omk = 1 - sksfc;
          sksfc = sksfc * (omk * omk);
          zfac = fmax2(zfac, zfmin);
          const Real ashape = fmax2(afac, 0.2);
          Real ashape1 = 1.0;
          if (skmax > sksfc) ashape1 = (skmax * ashape - sksfc) / (skmax - sksfc);
          const Real skminusk0 = zi(k + 1) * zfac - hpbl * sksfc;
          Real tem = zi(k + 1) * (zfac);
          if (skminusk0 > 0) tem = skminusk0 * ashape1 + hpbl * sksfc;
          Real tem1;
          if (pblflg) {
            tem1 = vk * wscaleu * tem;
            dku(k) = tem1;
            dkt(k) = tem1 * prinv;
          } else {
            tem1 = vk * wscale * tem;
            dku(k) = tem1;
            dkt(k) = tem1 * prinv;
          }
          dku(k) = fmin2(dku(k), dkmax);
          dku(k) = fmax2(dku(k), xkzmo(k));
          dkt(k) = fmin2(dkt(k), dkmax);
          dkt(k) = fmax2(dkt(k), xkzo(k));
          dktx(k) = dkt(k);
        }
      }
    };
    below_pbl(Kokkos::fabs(alpha));
    if (xland1 == 2 && alpha < 0) {
      if (!wspm2_set) {
        Kokkos::atomic_store(&err(), 1);
      } else {
        const int kloc = int(wspm2);
        if (kpbl > kloc) {
          const Real xdku = dku(kloc);
          if (xdku >= wspm1) {
            const Real wspm3 = wspm1 / xdku;
            const Real wspm4 = fmin2(wspm3, 1.0);
            below_pbl(wspm4);
          }
        }
      }
    }

    // local diffusion above the PBL
    for (int k = 1; k <= km1; ++k) {
      if (k >= kpbl) {
        const Real bvf2 = g * bf(k) * ti(k);
        const Real ri = fmax2(bvf2 / shr2(k), rimin);
        const Real zk = vk * zi(k + 1);
        if (ri < 0.) {
          const Real rl2 = zk * rlamun / (rlamun + zk);
          const Real dk = rl2 * rl2 * Kokkos::sqrt(shr2(k));
          const Real sri = Kokkos::sqrt(-ri);
          dku(k) = dk * (1 + 8. * (-ri) / (1 + 1.746 * sri));
          dkt(k) = dk * (1 + 8. * (-ri) / (1 + 1.286 * sri));
        } else {
          const Real rl2 = zk * rlam / (rlam + zk);
          const Real dk = rl2 * rl2 * Kokkos::sqrt(shr2(k));
          const Real t5 = 1 + 5. * ri;
          const Real tem1 = dk / (t5 * t5);
          Real prnum;
          if (k >= kpblx) {
            prnum = 1.0 + 2.1 * ri;
            prnum = fmin2(prnum, prmax);
          } else {
            prnum = 1.0;
          }
          dku(k) = tem1 * prnum;
          dkt(k) = tem1;
        }
        dku(k) = fmin2(dku(k), dkmax);
        dku(k) = fmax2(dku(k), xkzmo(k));
        dkt(k) = fmin2(dkt(k), dkmax);
        dkt(k) = fmax2(dkt(k), xkzo(k));
      }
    }

    // mass flux of the large thermals
    if (pcnvflg) {
      for (int k = 1; k <= km; ++k) {
        tcko(k) = t1(k);
        ucko(k) = u1(k);
        vcko(k) = v1(k);
        xmf(k) = 0.;
      }
      for (int kk = 0; kk < ntrac; ++kk)
        for (int k = 1; k <= km; ++k) qcko[kk](k) = q1[kk](k);

      // mfpbl (zi there is zm(k+1), here zi(k+1))
      const int kmpblm = km / 2 + 1;
      Real delz, xlamax;
      {
        int k = kpbl / 2;
        k = k > 1 ? k : 1;
        delz = zl(k + 1) - zl(k);
        xlamax = ce0 / delz;
      }
      for (int k = 1; k <= kmpblm; ++k) {
        if (k < kpbl) {
          const Real ptem = 1. / (zi(k + 1) + delz);
          const Real tem = fmax2((hpbl - zi(k + 1) + delz), delz);
          const Real ptem1 = 1. / tem;
          xlamue(k) = ce0 * (ptem + ptem1);
        } else {
          xlamue(k) = xlamax;
        }
      }
      // thermal excess
      {
        const Real tem = zl(1) / hpbl;
        const Real usws3 = Kokkos::pow(ustar / wstar, 3.);
        const Real tem1 = usws3 + 0.6 * tem;
        const Real tem2 = fmax2((1. - tem), zfmin);
        const Real ptem = (Kokkos::pow(tem1, h1)) * Kokkos::sqrt(tem2);
        const Real sigw1 = 1.3 * ptem * wstar;
        const Real ptem1 = alp * sflux / sigw1;
        thvu(1) = thvx(1) + ptem1;
        buo(1) = g * (thvu(1) / thvx(1) - 1.);
      }
      for (int k = 2; k <= kmpblm; ++k) {
        const Real dz = zl(k) - zl(k - 1);
        const Real tem = xlamue(k - 1) * dz;
        const Real ptem = 2. + tem;
        const Real ptem1 = (2. - tem) / ptem;
        const Real tem1 = tem * (thvx(k) + thvx(k - 1)) / ptem;
        thvu(k) = ptem1 * thvu(k - 1) + tem1;
        buo(k) = g * (thvu(k) / thvx(k) - 1.);
      }
      // updraft velocity squared
      {
        const Real dz = zi(2);
        const Real tem = 0.5 * bb1 * xlamue(1) * dz;
        const Real tem1 = bb2 * buo(1) * dz;
        const Real ptem1 = 1. + tem;
        wu2(1) = tem1 / ptem1;
      }
      for (int k = 2; k <= kmpblm; ++k) {
        const Real dz = zi(k + 1) - zi(k);
        const Real tem = 0.25 * bb1 * (xlamue(k) + xlamue(k - 1)) * dz;
        const Real tem1 = bb2 * buo(k) * dz;
        const Real ptem = (1. - tem) * wu2(k - 1);
        const Real ptem1 = 1. + tem;
        wu2(k) = (ptem + tem1) / ptem1;
      }
      // PBL height where the updraft velocity vanishes
      {
        bool flgm = false;
        Real rbupm = wu2(1), rbdnm = 0;
        for (int k = 2; k <= kmpblm; ++k) {
          if (!flgm) {
            rbdnm = rbupm;
            rbupm = wu2(k);
            kpbl = k;
            flgm = rbupm <= 0.;
          }
        }
        const int k = kpbl;
        Real rbint;
        if (rbdnm <= 0.) rbint = 0.;
        else if (rbupm >= 0.) rbint = 1.;
        else rbint = rbdnm / (rbdnm - rbupm);
        hpbl = zi(k) + rbint * (zi(k + 1) - zi(k));
      }
      {
        int k = kpbl / 2;
        k = k > 1 ? k : 1;
        delz = zl(k + 1) - zl(k);
        xlamax = ce0 / delz;
      }
      for (int k = 1; k <= kmpblm; ++k) {
        if (k < kpbl) {
          const Real ptem = 1. / (zi(k + 1) + delz);
          const Real tem = fmax2((hpbl - zi(k + 1) + delz), delz);
          const Real ptem1 = 1. / tem;
          xlamue(k) = ce0 * (ptem + ptem1);
        } else {
          xlamue(k) = xlamax;
        }
      }
      // updraft mass flux from the updraft velocity
      for (int k = 1; k <= kmpblm; ++k) {
        if (k < kpbl) {
          xmf(k) = a1c * Kokkos::sqrt(wu2(k));
          const Real dz = zl(k + 1) - zl(k);
          const Real xmmx = dz / dt2;
          xmf(k) = fmin2(xmf(k), xmmx);
        }
      }
      // updraft properties
      for (int k = 2; k <= kmpblm; ++k) {
        if (k <= kpbl) {
          const Real dz = zl(k) - zl(k - 1);
          const Real tem = 0.5 * xlamue(k - 1) * dz;
          const Real factor = 1. + tem;
          const Real ptem = tem + pgcon;
          const Real ptem1 = tem - pgcon;
          tcko(k) = ((1. - tem) * tcko(k - 1) + tem * (t1(k) + t1(k - 1)) - gocp * dz) / factor;
          ucko(k) = ((1. - tem) * ucko(k - 1) + ptem * u1(k) + ptem1 * u1(k - 1)) / factor;
          vcko(k) = ((1. - tem) * vcko(k - 1) + ptem * v1(k) + ptem1 * v1(k - 1)) / factor;
        }
      }
      for (int n = 0; n < ntrac; ++n)
        for (int k = 2; k <= kmpblm; ++k) {
          if (k <= kpbl) {
            const Real dz = zl(k) - zl(k - 1);
            const Real tem = 0.5 * xlamue(k - 1) * dz;
            const Real factor = 1. + tem;
            qcko[n](k) = ((1. - tem) * qcko[n](k - 1) + tem * (q1[n](k) + q1[n](k - 1))) / factor;
          }
        }
    }

    // cloud-top driven diffusion
    if (scuflg) {
      const int k = krad;
      const Real tem = thetae(k) - thetae(k + 1);
      const Real tem1 = qtx(k) - qtx(k + 1);
      if (tem > 0. && tem1 > 0.) {
        const Real cteit = cp * tem / (hvap * tem1);
        if (cteit > actei) rent = rentf2;
      }
    }
    if (scuflg) {
      const int k = krad;
      const Real tem1 = fmax2(bf(k), tdzmin);
      ckt(k) = -rent * radmin / tem1;
      cku(k) = ckt(k);
    }
    for (int k = 1; k <= kmpbl; ++k) {
      if (scuflg && k < krad) {
        const Real tem1 = hrad - zd;
        const Real tem2 = zi(k + 1) - tem1;
        if (tem2 > 0.) {
          Real ptem = tem2 / zd;
          if (ptem >= 1.) ptem = 1.;
          ptem = tem2 * ptem * Kokkos::sqrt(1. - ptem);
          ckt(k) = radfac * vk * vrad * ptem;
          cku(k) = 0.75 * ckt(k);
          ckt(k) = fmax2(ckt(k), dkmin);
          ckt(k) = fmin2(ckt(k), dkmax);
          cku(k) = fmax2(cku(k), dkmin);
          cku(k) = fmin2(cku(k), dkmax);
        }
      }
    }
    for (int k = 1; k <= kmpbl; ++k) {
      if (scuflg) {
        if (alpha == 1.0) dkt(k) = dkt(k) + ckt(k);
        if (alpha == 1.0) dku(k) = dku(k) + cku(k);
        dkt(k) = fmin2(dkt(k), dkmax);
        dku(k) = fmin2(dku(k), dkmax);
      }
    }

    // tridiagonal matrix for heat and moisture
    ad(1) = 1.;
    a1(1) = t1(1) + beta * heat;
    a2[0](1) = q1[0](1) + beta * evap;
    for (int kk = 1; kk < ntrac; ++kk) a2[kk](1) = q1[kk](1);
    for (int k = 1; k <= km1; ++k) {
      const Real dtodsd = dt2 / del(k);
      const Real dtodsu = dt2 / del(k + 1);
      const Real dsig = prsl(k) - prsl(k + 1);
      const Real rdz = rdzt(k);
      const Real tem1 = dsig * dkt(k) * rdz;
      const Real dsdz2 = tem1 * rdz;
      au(k) = -dtodsd * dsdz2;
      al(k) = -dtodsu * dsdz2;
      if (pcnvflg && k < kpbl) {
        const Real tem2 = dsig * rdz;
        Real ptem = 0.5 * tem2 * xmf(k);
        const Real ptem1 = dtodsd * ptem;
        const Real ptem2 = dtodsu * ptem;
        ad(k) = ad(k) - au(k) - ptem1;
        ad(k + 1) = 1. - al(k) + ptem2;
        au(k) = au(k) - ptem1;
        al(k) = al(k) + ptem2;
        ptem = tcko(k) + tcko(k + 1);
        const Real dsdzt = tem1 * gocp;
        a1(k) = a1(k) + dtodsd * dsdzt - ptem1 * ptem;
        a1(k + 1) = t1(k + 1) - dtodsu * dsdzt + ptem2 * ptem;
        ptem = qcko[0](k) + qcko[0](k + 1);
        a2[0](k) = a2[0](k) - ptem1 * ptem;
        a2[0](k + 1) = q1[0](k + 1) + ptem2 * ptem;
      } else if (ublflg && k < kpbl) {
        const Real ptem1 = dsig * dktx(k) * rdz;
        const Real tem = 1.0 / hpbl;
        const Real dsdzt = tem1 * gocp - ptem1 * hgamt * tem;
        const Real dsdzq = -ptem1 * hgamq * tem;
        ad(k) = ad(k) - au(k);
        ad(k + 1) = 1. - al(k);
        a1(k) = a1(k) + dtodsd * dsdzt;
        a1(k + 1) = t1(k + 1) - dtodsu * dsdzt;
        a2[0](k) = a2[0](k) + dtodsd * dsdzq;
        a2[0](k + 1) = q1[0](k + 1) - dtodsu * dsdzq;
      } else {
        ad(k) = ad(k) - au(k);
        ad(k + 1) = 1. - al(k);
        const Real dsdzt = tem1 * gocp;
        a1(k) = a1(k) + dtodsd * dsdzt;
        a1(k + 1) = t1(k + 1) - dtodsu * dsdzt;
        a2[0](k + 1) = q1[0](k + 1);
      }
    }
    for (int kk = 1; kk < ntrac; ++kk)
      for (int k = 1; k <= km1; ++k) {
        if (pcnvflg && k < kpbl) {
          const Real dtodsd = dt2 / del(k);
          const Real dtodsu = dt2 / del(k + 1);
          const Real dsig = prsl(k) - prsl(k + 1);
          const Real tem = dsig * rdzt(k);
          const Real ptem = 0.5 * tem * xmf(k);
          const Real ptem1 = dtodsd * ptem;
          const Real ptem2 = dtodsu * ptem;
          const Real tem1 = qcko[kk](k) + qcko[kk](k + 1);
          a2[kk](k) = a2[kk](k) - ptem1 * tem1;
          a2[kk](k + 1) = q1[kk](k + 1) + ptem2 * tem1;
        } else {
          a2[kk](k + 1) = q1[kk](k + 1);
        }
      }
    tridin(km, ntrac, al, ad, au, a1, a2, fkk);
    // tendencies of heat and moisture
    for (int k = 1; k <= km; ++k) {
      const Real ttend = (a1(k) - t1(k)) * rdt;
      const Real qtend = (a2[0](k) - q1[0](k)) * rdt;
      tau(k) = tau(k) + ttend;
      rtg[0](k) = rtg[0](k) + qtend;
    }
    for (int kk = 1; kk < ntrac; ++kk)
      for (int k = 1; k <= km; ++k) {
        const Real qtend = (a2[kk](k) - q1[kk](k)) * rdt;
        rtg[kk](k) = rtg[kk](k) + qtend;
      }
    // TKE dissipation rate and dissipative heating
    if (dspheat) {
      for (int k = 1; k <= km1; ++k) diss(k) = dku(k) * shr2(k) - g * ti(k) * dkt(k) * bf(k);
      {
        const Real tem = govrth * sflux;
        const Real tem1 = tem + stress * spd1 / zl(1);
        Real tem2 = 0.5 * (tem1 + diss(1));
        tem2 = fmax2(tem2, 0.);
        const Real ttend = tem2 / cp;
        tau(1) = tau(1) + 0.7 * ttend;
        dishx(1) = 0.7 * ttend;
        diss(0) = tem1;
      }
      for (int k = 2; k <= km1; ++k) {
        Real tem = 0.5 * (diss(k - 1) + diss(k));
        tem = fmax2(tem, 0.);
        const Real ttend = tem / cp;
        tau(k) = tau(k) + 0.7 * ttend;
        dishx(k) = 0.7 * ttend;
      }
    }
    // tridiagonal matrix for momentum
    ad(1) = 1.0 + beta * stress / spd1;
    a1(1) = u1(1);
    a2[0](1) = v1(1);
    for (int k = 1; k <= km1; ++k) {
      const Real dtodsd = dt2 / del(k);
      const Real dtodsu = dt2 / del(k + 1);
      const Real dsig = prsl(k) - prsl(k + 1);
      const Real rdz = rdzt(k);
      const Real tem1 = dsig * dku(k) * rdz;
      const Real dsdz2 = tem1 * rdz;
      au(k) = -dtodsd * dsdz2;
      al(k) = -dtodsu * dsdz2;
      if (pcnvflg && k < kpbl) {
        const Real tem2 = dsig * rdz;
        Real ptem = 0.5 * tem2 * xmf(k);
        const Real ptem1 = dtodsd * ptem;
        const Real ptem2 = dtodsu * ptem;
        ad(k) = ad(k) - au(k) - ptem1;
        ad(k + 1) = 1. - al(k) + ptem2;
        au(k) = au(k) - ptem1;
        al(k) = al(k) + ptem2;
        ptem = ucko(k) + ucko(k + 1);
        a1(k) = a1(k) - ptem1 * ptem;
        a1(k + 1) = u1(k + 1) + ptem2 * ptem;
        ptem = vcko(k) + vcko(k + 1);
        a2[0](k) = a2[0](k) - ptem1 * ptem;
        a2[0](k + 1) = v1(k + 1) + ptem2 * ptem;
      } else {
        ad(k) = ad(k) - au(k);
        ad(k + 1) = 1. - al(k);
        a1(k + 1) = u1(k + 1);
        a2[0](k + 1) = v1(k + 1);
      }
    }
    tridi2(km, al, ad, au, a1, a2[0]);
    for (int k = 1; k <= km; ++k) {
      const Real utend = (a1(k) - u1(k)) * rdt;
      const Real vtend = (a2[0](k) - v1(k)) * rdt;
      du(k) = du(k) + utend;
      dv(k) = dv(k) + vtend;
    }

    //------------------------------------------------------------------
    // BL_GFSEDMF: outputs
    for (int k = 1; k <= km; ++k) {
      O.rvblten(i, j, k) = dv(k);
      O.rublten(i, j, k) = du(k);
      O.rthblten(i, j, k) = tau(k) / I.pi3d(i, j, k);
      const Real omq1 = 1. - q1[0](k);
      O.rqvblten(i, j, k) = rtg[0](k) / (omq1 * omq1);
      const Real omq2 = 1. - q1[1](k);
      O.rqcblten(i, j, k) = rtg[1](k) / (omq2 * omq2);
      O.dheat(i, j, k) = dishx(k);
      O.dissten(i, j, k) = diss(k - 1);
    }
    if (ntrac == 3)
      for (int k = 1; k <= km; ++k) {
        const Real omq3 = 1. - q1[2](k);
        O.rqiblten(i, j, k) = rtg[2](k) / (omq3 * omq3);
      }
    O.ust(i, j, 1) = Kokkos::sqrt(stress);
    O.wspd(i, j, 1) = Kokkos::sqrt(I.u3d(i, j, 1) * I.u3d(i, j, 1) + I.v3d(i, j, 1) * I.v3d(i, j, 1)) + 1.E-9;
    O.hpbl(i, j, 1) = hpblx;
    O.kpbl2d(i, j, 1) = Real(kpblx);
    for (int k = 1; k <= km - 1; ++k) {
      O.dku3d(i, j, k + 1) = dku(k);
      O.dkt3d(i, j, k + 1) = dkt(k);
    }
  });

  int e = 0;
  Kokkos::deep_copy(e, ws.err);
  if (e != 0)
    throw std::runtime_error("cm1k pbl_gfs: alpha < 0 over water with no level bracketing 500 m among k = 1..nk/2 "
                             "(WSPM(i,2) is undefined in module_bl_gfsedmf.F)");
}

} // namespace cm1k::pbl_gfs
