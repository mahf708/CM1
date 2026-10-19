#include "init/cm1k_init.hpp"
#include "share/cm1k_moist.hpp"

#include "init/cm1k_cm1libs.hpp"

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Ports of the initialization parts of param.F, base.F, init3d.F and cm1.F.
// Each block keeps the Fortran loop order and parenthesization; comments
// name the Fortran source the block comes from. Expressions such as
// `a*b/c` are written `a * b / c` (left to right, as in Fortran), and
// x**r with real r is std::pow(x, r) (gfortran also calls libm pow).

namespace cm1k {
namespace init {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
  throw std::runtime_error("cm1k init: unsupported option: " + what);
}

// Fortran nint (round half away from zero)
int nint(Real x) { return int(std::lround(x)); }

// smeps for the double-precision build (cm1.F, #ifdef DP)
constexpr Real smeps = 1.0e-60;

} // namespace

//---------------------------------------------------------------------------
// param.F: 2nd-order extrapolation coefficients (Fornberg 1988)
ExtrapCoeffs extrap_coeffs(const HArray1& sigma, const HArray1& sigmaf, int nk) {
  constexpr int bigm = 4, bign = 3;
  ExtrapCoeffs e;
  for (int nn = 1; nn <= 4; ++nn) {
    double x0 = 0, alpha[bign + 1];
    if (nn == 1) {
      const double d = sigmaf(2) - sigmaf(1);
      x0 = sigmaf(1) / d;
      for (int m = 0; m <= 3; ++m) alpha[m] = sigma(1 + m) / d;
    } else if (nn == 2) {
      const double d = sigmaf(nk + 1) - sigmaf(nk);
      x0 = sigmaf(nk + 1) / d;
      for (int m = 0; m <= 3; ++m) alpha[m] = sigma(nk - m) / d;
    } else if (nn == 3) {
      const double d = sigmaf(2) - sigmaf(1);
      x0 = sigmaf(1) / d;
      for (int m = 0; m <= 3; ++m) alpha[m] = sigmaf(2 + m) / d;
    } else {
      const double d = sigmaf(nk + 1) - sigmaf(nk);
      x0 = sigmaf(nk + 1) / d;
      for (int m = 0; m <= 3; ++m) alpha[m] = sigmaf(nk - m) / d;
    }
    // delta(0:bign, -1:bigm, 0:bign)
    double delta[bign + 1][bigm + 2][bign + 1] = {};
    auto D = [&](int n, int m, int nu) -> double& { return delta[n][m + 1][nu]; };
    D(0, 0, 0) = 1.0;
    double b1 = 1.0;
    for (int n = 1; n <= bign; ++n) {
      double b2 = 1.0;
      for (int nu = 0; nu <= n - 1; ++nu) {
        const double b3 = alpha[n] - alpha[nu];
        b2 = b2 * b3;
        if (n <= bigm) D(n - 1, n, nu) = 0.0;
        for (int m = 0; m <= std::min(n, bigm); ++m)
          D(n, m, nu) = ((alpha[n] - x0) * D(n - 1, m, nu) - double(m) * D(n - 1, m - 1, nu)) / b3;
      }
      for (int m = 0; m <= std::min(n, bigm); ++m)
        D(n, m, n) = (b1 / b2) * (double(m) * D(n - 1, m - 1, n - 1) - (alpha[n - 1] - x0) * D(n - 1, m, n - 1));
      b1 = b2;
    }
    if (nn == 1) {
      e.cgs1 = D(2, 0, 0); e.cgs2 = D(2, 0, 1); e.cgs3 = D(2, 0, 2);
      e.dgs1 = D(2, 1, 0); e.dgs2 = D(2, 1, 1); e.dgs3 = D(2, 1, 2);
    } else if (nn == 2) {
      e.cgt1 = D(2, 0, 0); e.cgt2 = D(2, 0, 1); e.cgt3 = D(2, 0, 2);
      e.dgt1 = D(2, 1, 0); e.dgt2 = D(2, 1, 1); e.dgt3 = D(2, 1, 2);
    } else if (nn == 3) {
      e.wbe1 = D(2, 0, 0); e.wbe2 = D(2, 0, 1); e.wbe3 = D(2, 0, 2);
    } else {
      e.wte1 = D(2, 0, 0); e.wte2 = D(2, 0, 1); e.wte3 = D(2, 0, 2);
    }
  }
  return e;
}

//---------------------------------------------------------------------------
// param.F: grid
InitInfo init_grid(const Namelist& nl, Grid& g) {
  const auto& B = g.b;
  const int nx = B.nx, ny = B.ny, nk = B.nk, ni = B.ni, nj = B.nj, ngxy = B.ngxy;
  const int ib = B.ib, ie = B.ie, jb = B.jb, je = B.je, kb = B.kb, ke = B.ke;

  if (nl.geti("stretch_x", 0) != 0) unsupported("stretch_x != 0");
  if (nl.geti("stretch_y", 0) != 0) unsupported("stretch_y != 0");
  const int stretch_z = nl.geti("stretch_z", 0);
  if (stretch_z != 0 && stretch_z != 1) unsupported("stretch_z = " + std::to_string(stretch_z));
  if (nl.getb("terrain_flag", false)) unsupported("terrain_flag = .true.");
  if (nl.geti("axisymm", 0) != 0) unsupported("axisymm = 1");
  if (nl.geti("irst", 0) != 0) unsupported("irst = 1 (restart)");
  if (nk < 4) unsupported("nz < 4");

  const Real dx = nl.getr("dx", 0.0), dy = nl.getr("dy", 0.0);
  Real dz = nl.getr("dz", 0.0);
  Real ztop = nl.getr("ztop", 0.0);
  if (stretch_z < 1) ztop = dz * double(nk);

  g.dx = dx; g.dy = dy; g.dz = dz;
  g.rdx = 1.0 / dx; g.rdy = 1.0 / dy; g.rdz = 1.0 / dz;

  InitInfo info;
  auto& bc = info.bc;
  bc.wbc = nl.geti("wbc", 1); bc.ebc = nl.geti("ebc", 1);
  bc.sbc = nl.geti("sbc", 1); bc.nbc = nl.geti("nbc", 1);
  for (int v : {bc.wbc, bc.ebc, bc.sbc, bc.nbc})
    if (v < 1 || v > 4) unsupported("lateral boundary condition " + std::to_string(v));
  bc.ni = ni; bc.nj = nj; bc.nk = nk; bc.ngxy = ngxy;
  bc.ib = ib; bc.ie = ie; bc.jb = jb; bc.je = je;
  // cm1r20 (serial: myi1 = myj1 = 1, myi2 = nx, myj2 = ny)
  bc.ibw = (bc.wbc != 1) ? 1 : 0;
  bc.ibe = (bc.ebc != 1) ? 1 : 0;
  bc.ibs = (bc.sbc != 1) ? 1 : 0;
  bc.ibn = (bc.nbc != 1) ? 1 : 0;

  // ---- horizontal grid (no stretching) ----
  const int iorigin = nl.geti("iorigin", 1);
  std::vector<double> xfdp_v(nx + 2 * ngxy + 1), yfdp_v(ny + 2 * ngxy + 1);
  auto xfdp = [&](int i) -> double& { return xfdp_v[i - (1 - ngxy)]; };
  auto yfdp = [&](int j) -> double& { return yfdp_v[j - (1 - ngxy)]; };
  if (iorigin == 1) {
    for (int i = 1 - ngxy; i <= nx + ngxy + 1; ++i) xfdp(i) = double(dx) * double(i - 1);
    for (int j = 1 - ngxy; j <= ny + ngxy + 1; ++j) yfdp(j) = double(dy) * double(j - 1);
  } else if (iorigin == 2) {
    for (int i = 1 - ngxy; i <= nx + ngxy + 1; ++i)
      xfdp(i) = double(dx) * double(i - 1) - 0.5 * double(dx) * double(nx);
    for (int j = 1 - ngxy; j <= ny + ngxy + 1; ++j)
      yfdp(j) = double(dy) * double(j - 1) - 0.5 * double(dy) * double(ny);
  } else {
    unsupported("iorigin = " + std::to_string(iorigin));
  }

  auto xh = g.xh.to_host(), rxh = g.rxh.to_host(), uh = g.uh.to_host(), ruh = g.ruh.to_host();
  auto xf = g.xf.to_host(), rxf = g.rxf.to_host(), uf = g.uf.to_host(), ruf = g.ruf.to_host();
  auto yh = g.yh.to_host(), vh = g.vh.to_host(), rvh = g.rvh.to_host();
  auto yf = g.yf.to_host(), vf = g.vf.to_host(), rvf = g.rvf.to_host();

  for (int i = ib; i <= ie + 1; ++i) uf(i) = 1.0;
  for (int i = ib; i <= ie + 1; ++i) xf(i) = xfdp(i);
  for (int i = ib; i <= ie; ++i) {
    xh(i) = 0.5 * (xfdp(i + 1) + xfdp(i));
    uh(i) = double(dx) / (xfdp(i + 1) - xfdp(i));
  }
  for (int i = ib + 1; i <= ie; ++i)
    uf(i) = double(dx) / (0.5 * (xfdp(i) + xfdp(i + 1)) - 0.5 * (xfdp(i) + xfdp(i - 1)));
  if (bc.ibw == 1) for (int i = 1 - ngxy; i <= 0; ++i) uf(i) = uf(1);
  if (bc.ibe == 1) for (int i = ni + 2; i <= ni + 1 + ngxy; ++i) uf(i) = uf(ni + 1);
  for (int i = ib; i <= ie; ++i) { rxh(i) = 1.0 / (smeps + xh(i)); ruh(i) = 1.0 / uh(i); }
  for (int i = ib; i <= ie + 1; ++i) { rxf(i) = 1.0 / (smeps + xf(i)); ruf(i) = 1.0 / uf(i); }
  info.minx = xfdp(1);
  info.maxx = xfdp(nx + 1);
  info.centerx = info.minx + 0.5 * (info.maxx - info.minx);

  for (int j = jb; j <= je + 1; ++j) vf(j) = 1.0;
  for (int j = jb; j <= je + 1; ++j) yf(j) = yfdp(j);
  for (int j = jb; j <= je; ++j) {
    yh(j) = 0.5 * (yfdp(j + 1) + yfdp(j));
    vh(j) = double(dy) / (yfdp(j + 1) - yfdp(j));
  }
  for (int j = jb + 1; j <= je; ++j)
    vf(j) = double(dy) / (0.5 * (yfdp(j) + yfdp(j + 1)) - 0.5 * (yfdp(j) + yfdp(j - 1)));
  if (bc.ibs == 1) for (int j = 1 - ngxy; j <= 0; ++j) vf(j) = vf(1);
  if (bc.ibn == 1) for (int j = nj + 2; j <= nj + 1 + ngxy; ++j) vf(j) = vf(nj + 1);
  for (int j = jb; j <= je; ++j) rvh(j) = 1.0 / vh(j);
  for (int j = jb; j <= je + 1; ++j) rvf(j) = 1.0 / vf(j);
  info.miny = yfdp(1);
  info.maxy = yfdp(ny + 1);
  info.centery = info.miny + 0.5 * (info.maxy - info.miny);

  // ---- vertical grid ----
  auto zh = g.zh.to_host(), mh = g.mh.to_host(), rmh = g.rmh.to_host();
  auto c1 = g.c1.to_host(), c2 = g.c2.to_host();
  auto zf = g.zf.to_host(), mf = g.mf.to_host(), rmf = g.rmf.to_host();
  auto sigma = g.sigma.to_host(), sigmaf = g.sigmaf.to_host();
  auto rds = g.rds.to_host(), rdsf = g.rdsf.to_host();

  for (int k = kb; k <= ke + 1; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) { zf(i, j, k) = dz * double(k - 1); mf(i, j, k) = 1.0; }
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) { zh(i, j, k) = 0.5 * (zf(i, j, k) + zf(i, j, k + 1)); mh(i, j, k) = 1.0; }
  for (int k = kb; k <= ke + 1; ++k) sigmaf(k) = zf(1, 1, k);
  for (int k = kb; k <= ke; ++k) sigma(k) = 0.5 * (sigmaf(k) + sigmaf(k + 1));

  if (stretch_z >= 1) {
    // stretch_z = 1: hard-wired analytic stretching function
    const Real str_bot = nl.getr("str_bot", 0.0), str_top = nl.getr("str_top", 0.0);
    const Real dz_bot = nl.getr("dz_bot", 0.0), dz_top = nl.getr("dz_top", 0.0);
    Real nominal_dz = 0.5 * (dz_bot + dz_top);
    const int nk1 = nint(str_bot / dz_bot);
    const int nk3 = nint((ztop - str_top) / dz_top);
    const int nk2 = nk - (nk1 + nk3);
    if (std::abs(nk1 * dz_bot - str_bot) > 0.01)
      throw std::runtime_error("cm1k init: stretch_z=1: depth of bottom layer does not divide by dz_bot");
    if (std::abs(nk2 * nominal_dz - (str_top - str_bot)) >= 0.01)
      throw std::runtime_error("cm1k init: stretch_z=1: depth of middle layer does not divide by nominal_dz");
    if (std::abs(nk3 * dz_top - (ztop - str_top)) > 0.01)
      throw std::runtime_error("cm1k init: stretch_z=1: depth of top layer does not divide by dz_top");
    nominal_dz = (str_top - str_bot) / nk2;
    const Real cc2 = (nominal_dz - dz_bot) / (nominal_dz * nominal_dz * double(nk2 - 1));
    const Real cc1 = (dz_bot / nominal_dz) - cc2 * nominal_dz;
    const Real test_len = dz_bot * nk1 + 0.5 * (dz_bot + dz_top) * nk2 + dz_top * nk3;
    if (B.nz != nk1 + nk2 + nk3 || nk1 < 0 || nk2 < 0 || nk3 < 0 || std::abs(ztop - test_len) > 1.0e-6 * ztop)
      throw std::runtime_error("cm1k init: stretch_z=1: inconsistent nz/ztop/str_bot/str_top/dz_bot/dz_top");

    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        for (int k = 1; k <= nk1 + 1; ++k) zf(i, j, k) = double(k - 1) * dz_bot;
        for (int k = nk1 + 1; k <= nk1 + nk2 + 1; ++k)
          zf(i, j, k) = zf(i, j, nk1 + 1) + (cc1 + cc2 * double(k - 1 - nk1) * nominal_dz) * double(k - 1 - nk1) * nominal_dz;
        for (int k = nk1 + nk2 + 2; k <= nk1 + nk2 + nk3 + 1; ++k) zf(i, j, k) = zf(i, j, k - 1) + dz_top;
      }
    for (int k = 1; k <= nk1 + 1; ++k) sigmaf(k) = double(k - 1) * dz_bot;
    for (int k = nk1 + 1; k <= nk1 + nk2 + 1; ++k)
      sigmaf(k) = sigmaf(nk1 + 1) + (cc1 + cc2 * double(k - 1 - nk1) * nominal_dz) * double(k - 1 - nk1) * nominal_dz;
    for (int k = nk1 + nk2 + 2; k <= nk1 + nk2 + nk3 + 1; ++k) sigmaf(k) = sigmaf(k - 1) + dz_top;
    sigmaf(0) = -sigmaf(2);
    sigmaf(nk + 2) = sigmaf(nk + 1) + (sigmaf(nk + 1) - sigmaf(nk));

    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        zf(i, j, 0) = -zf(i, j, 2);
        zf(i, j, nk + 2) = zf(i, j, nk + 1) + (zf(i, j, nk + 1) - zf(i, j, nk));
        for (int k = 0; k <= nk + 1; ++k) {
          zh(i, j, k) = 0.5 * (zf(i, j, k + 1) + zf(i, j, k));
          mh(i, j, k) = dz / (zf(i, j, k + 1) - zf(i, j, k));
        }
        zh(i, j, 0) = -zh(i, j, 1);
        zh(i, j, nk + 1) = zh(i, j, nk) + 2.0 * (zf(i, j, nk + 1) - zh(i, j, nk));
        for (int k = 1; k <= nk + 1; ++k) mf(i, j, k) = dz / (zh(i, j, k) - zh(i, j, k - 1));
        mf(i, j, 0) = mf(i, j, 1);
        mf(i, j, nk + 2) = mf(i, j, nk + 1);
      }
  }

  for (int k = kb; k <= ke; ++k) sigma(k) = 0.5 * (sigmaf(k) + sigmaf(k + 1));
  info.maxz = sigmaf(nk + 1);
  info.ztop = ztop;
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rmh(i, j, k) = 1.0 / mh(i, j, k);
  for (int k = kb; k <= ke + 1; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rmf(i, j, k) = 1.0 / mf(i, j, k);
  for (int k = 1; k <= nk; ++k)
    if ((zf(1, 1, k + 1) - zf(1, 1, k)) < smeps) throw std::runtime_error("cm1k init: dz <= 0");

  // terrain section (terrain_flag = .false.)
  for (int k = kb; k <= ke; ++k) rds(k) = 1.0;
  for (int k = kb; k <= ke + 1; ++k) rdsf(k) = 1.0;

  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        mh(i, j, k) = dz / (zf(i, j, k + 1) - zf(i, j, k));
        rmh(i, j, k) = 1.0 / mh(i, j, k);
      }
  for (int k = kb + 1; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) mf(i, j, k) = dz / (zh(i, j, k) - zh(i, j, k - 1));
  for (int j = jb; j <= je; ++j)
    for (int i = ib; i <= ie; ++i) { mf(i, j, 0) = mf(i, j, 1); mf(i, j, nk + 2) = mf(i, j, nk + 1); }
  for (int k = kb; k <= ke + 1; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rmf(i, j, k) = 1.0 / mf(i, j, k);

  // vertical interpolation coefficients (cm1r16): c2 then bcs(c2), c1 = 1-c2
  for (int k = 1; k <= nk + 1; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i)
        c2(i, j, k) = (zf(i, j, k) - zh(i, j, k - 1)) / (zh(i, j, k) - zh(i, j, k - 1));
  bcs(bc, c2);
  for (int k = 1; k <= nk + 1; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) c1(i, j, k) = 1.0 - c2(i, j, k);

  info.ex = extrap_coeffs(sigma, sigmaf, nk);
  bc.cgs1 = info.ex.cgs1; bc.cgs2 = info.ex.cgs2; bc.cgs3 = info.ex.cgs3;
  bc.cgt1 = info.ex.cgt1; bc.cgt2 = info.ex.cgt2; bc.cgt3 = info.ex.cgt3;

  // cm1r18: ghost points for zh (index 0 is the surface, nk+1 the top)
  for (int j = jb; j <= je; ++j)
    for (int i = ib; i <= ie; ++i) {
      zh(i, j, 0) = zf(i, j, 1);
      zh(i, j, nk + 1) = zf(i, j, nk + 1);
    }

  for (auto p : std::initializer_list<decltype(std::make_pair(&g.xh, &xh))>{{&g.xh, &xh}, {&g.rxh, &rxh}, {&g.uh, &uh}, {&g.ruh, &ruh}, {&g.xf, &xf},
                 {&g.rxf, &rxf}, {&g.uf, &uf}, {&g.ruf, &ruf}, {&g.yh, &yh}, {&g.vh, &vh}, {&g.rvh, &rvh},
                 {&g.yf, &yf}, {&g.vf, &vf}, {&g.rvf, &rvf}, {&g.sigma, &sigma}, {&g.sigmaf, &sigmaf},
                 {&g.rds, &rds}, {&g.rdsf, &rdsf}})
    p.first->deep_copy_from(*p.second);
  for (auto p : std::initializer_list<decltype(std::make_pair(&g.zh, &zh))>{{&g.zh, &zh}, {&g.mh, &mh}, {&g.rmh, &rmh}, {&g.c1, &c1}, {&g.c2, &c2},
                 {&g.zf, &zf}, {&g.mf, &mf}, {&g.rmf, &rmf}})
    p.first->deep_copy_from(*p.second);
  return info;
}

//---------------------------------------------------------------------------
// base.F
void init_base(const Namelist& nl, const Constants& c, const Grid& g, InitInfo& info, BaseState& bs) {
  using namespace phys;
  const auto& B = g.b;
  const auto& bc = info.bc;
  const auto& ex = info.ex;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const int ib = B.ib, ie = B.ie, jb = B.jb, je = B.je, kb = B.kb, ke = B.ke;
  const int isnd = nl.geti("isnd", 1), iwnd = nl.geti("iwnd", 0);
  const int imoist = nl.geti("imoist", 0), ptype = nl.geti("ptype", 0);
  const int psolver = nl.geti("psolver", 3), imove = nl.geti("imove", 0);
  if (nl.geti("testcase", 0) != 0) unsupported("testcase != 0");
  if (nl.getb("do_adapt_move", false)) unsupported("do_adapt_move = .true.");
  if (imoist == 1 && !MoistLayout::known(ptype)) unsupported("ptype = " + std::to_string(ptype));

  const Real g_ = c.g, cp = c.cp, rd = c.rd, cpdrd = c.cpdrd, reps = c.reps, eps = c.eps;
  auto rslf = [&](Real p, Real t) { return cm1libs::rslf(ptype, eps, p, t); };

  auto zh = g.zh.to_host(), c1 = g.c1.to_host(), c2 = g.c2.to_host();
  auto pi0 = bs.pi0.to_host(), th0 = bs.th0.to_host(), prs0 = bs.prs0.to_host(), rho0 = bs.rho0.to_host();
  auto thv0 = bs.thv0.to_host(), rth0 = bs.rth0.to_host(), qv0 = bs.qv0.to_host(), qc0 = bs.qc0.to_host();
  auto qi0 = bs.qi0.to_host(), rr0 = bs.rr0.to_host(), rf0 = bs.rf0.to_host(), rrf0 = bs.rrf0.to_host();
  auto u0 = bs.u0.to_host(), v0 = bs.v0.to_host();
  HArray3 rh0(Kokkos::View<Real***, Kokkos::LayoutLeft, HostSpace>("rh0", ie - ib + 1, je - jb + 1, ke - kb + 1), {ib, jb, kb});
  HArray3 t0(Kokkos::View<Real***, Kokkos::LayoutLeft, HostSpace>("t0", ie - ib + 1, je - jb + 1, ke - kb + 1), {ib, jb, kb});

  // ---- base-state sounding (isnd) ----
  if (isnd == 1) {
    // dry adiabatic
    const Real th_sfc = 300.0, pi_sfc = 1.0;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          th0(i, j, k) = th_sfc;
          pi0(i, j, k) = pi_sfc - g_ * zh(i, j, k) / (cp * th_sfc);
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
        }
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) { qv0(i, j, k) = 0.0; rh0(i, j, k) = 0.0; }
  } else if (isnd == 4) {
    // saturated, neutrally-stable sounding (Bryan and Fritsch 2002)
    const Real thec_mb = 320.0, qt_mb = 0.020, prs_sfc = 100000.0;
    const Real cpl = c.cpl, lv1 = c.lv1, lv2 = c.lv2;
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        prs0(i, j, 1) = prs_sfc;
        Real qv1 = qt_mb, tmp = thec_mb, tlast = tmp;
        for (int n = 1; n <= 20; ++n) {
          const Real thbar = (tmp * std::pow(p00 / prs0(i, j, 1), rd / cp)) * (1.0 + qv1 * reps) / (1.0 + qt_mb);
          pi0(i, j, 1) = 1.0 - g_ * zh(i, j, 1) / (cp * thbar);
          prs0(i, j, 1) = p00 * std::pow(pi0(i, j, 1), cpdrd);
          qv1 = rslf(prs0(i, j, 1), tmp);
          tmp = thec_mb * std::pow(prs0(i, j, 1) / (1.0 + qv1 / eps) / p00, rd / (cp + cpl * qt_mb))
                / std::exp((lv1 - lv2 * tmp) * qv1 / (tmp * (cp + cpl * qt_mb)));
          tmp = tlast + 0.3 * (tmp - tlast);
          tlast = tmp;
        }
        t0(i, j, 1) = tmp;
        th0(i, j, 1) = tmp / pi0(i, j, 1);
        rh0(i, j, 1) = 1.0;
        qv0(i, j, 1) = rslf(prs0(i, j, 1), tmp);
        Real t1 = tmp, th1 = th0(i, j, 1);
        qv1 = qv0(i, j, 1);
        for (int k = 2; k <= nk; ++k) {
          tlast = th1;
          Real t2 = t1, qv2 = qv1, th2 = th1;
          int n = 0;
          pi0(i, j, k) = pi0(i, j, k - 1) - g_ * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * th2);
          while (true) {
            n = n + 1;
            th2 = tlast;
            t2 = th2 * pi0(i, j, k);
            const Real thbar = 0.5 * (th1 * (1.0 + qv1 * reps) / (1.0 + qt_mb) + th2 * (1.0 + qv2 * reps) / (1.0 + qt_mb));
            pi0(i, j, k) = pi0(i, j, k - 1) - g_ * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * thbar);
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
            qv2 = rslf(prs0(i, j, k), t2);
            t2 = thec_mb * std::pow(prs0(i, j, k) / (1.0 + qv2 / eps) / p00, rd / (cp + cpl * qt_mb))
                 / std::exp((lv1 - lv2 * t2) * qv2 / (t2 * (cp + cpl * qt_mb)));
            th2 = t2 / pi0(i, j, k);
            if (std::abs(th2 - tlast) > 0.0001 && n < 100) {
              tlast = tlast + 0.3 * (th2 - tlast);
              continue;
            } else if (n >= 100) {
              throw std::runtime_error("cm1k init: isnd=4 iteration stuck in loop");
            }
            break;
          }
          t0(i, j, k) = t2;
          th0(i, j, k) = th2;
          qv0(i, j, k) = rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k));
          rh0(i, j, k) = 1.0;
          th1 = th2; qv1 = qv2; t1 = t2;
        }
        for (int k = 1; k <= nk; ++k) qc0(i, j, k) = qt_mb - qv0(i, j, k);
      }
  } else if (isnd == 5) {
    // Weisman-Klemp analytic sounding
    const Real z_trop = 12000.0, th_trop = 343.0, t_trop = 213.0, th_sfc = 300.0;
    const Real prs_sfc = 100000.0, qv_pbl = 0.014;
    const Real pi_sfc = std::pow(prs_sfc / p00, rd / cp);
    const Real qv_sfc = rslf(prs_sfc, th_sfc * pi_sfc);
    const Real thv_sfc = th_sfc * (1.0 + qv_sfc * reps) / (1.0 + qv_sfc);
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        for (int k = kb; k <= ke; ++k) rh0(i, j, k) = 0.0;
        for (int k = 1; k <= nk; ++k) {
          if (zh(i, j, k) < z_trop) {
            th0(i, j, k) = th_sfc + (th_trop - th_sfc) * std::pow(zh(i, j, k) / z_trop, 1.25);
            if (imoist == 1) rh0(i, j, k) = 1.0 - 0.75 * std::pow(zh(i, j, k) / z_trop, 1.25);
          } else {
            th0(i, j, k) = th_trop * std::exp((g_ / (t_trop * cp)) * (zh(i, j, k) - z_trop));
            if (imoist == 1) rh0(i, j, k) = 0.25;
          }
        }
        th0(i, j, 0) = th0(i, j, 1);
        th0(i, j, nk + 1) = th0(i, j, nk);
        for (int k = kb; k <= ke; ++k) qv0(i, j, k) = 0.0;
        for (int n = 1; n <= 20; ++n) {
          for (int k = kb; k <= ke; ++k)
            thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k));
          pi0(i, j, 1) = pi_sfc - g_ * zh(i, j, 1) / (cp * 0.5 * (thv_sfc + thv0(i, j, 1)));
          for (int k = 2; k <= nk; ++k)
            pi0(i, j, k) = pi0(i, j, k - 1) - g_ * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (thv0(i, j, k) + thv0(i, j, k - 1)));
          for (int k = 1; k <= nk; ++k) prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          for (int k = 1; k <= nk; ++k) {
            qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k));
            if (qv0(i, j, k) > qv_pbl) qv0(i, j, k) = qv_pbl;
          }
        }
        for (int k = 1; k <= nk; ++k)
          rh0(i, j, k) = qv0(i, j, k) / (rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k)));
      }
  } else {
    unsupported("isnd = " + std::to_string(isnd));
  }

  // ---- fill in ghost cells ----
  for (auto* a : {&pi0, &prs0, &th0, &qv0, &qc0, &qi0, &rh0}) bcs(bc, *a);
  for (auto* a : {&pi0, &prs0, &th0, &qv0, &qc0, &qi0, &rh0}) extrapbcs(bc, *a);

  for (int j = jb; j <= je; ++j)
    for (int i = ib; i <= ie; ++i) {
      if (imoist == 1) {
        for (int k = kb; k <= ke; ++k)
          thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k) + qc0(i, j, k) + qi0(i, j, k));
      } else {
        for (int k = kb; k <= ke; ++k) {
          qv0(i, j, k) = 0.0;
          rh0(i, j, k) = 0.0;
          thv0(i, j, k) = th0(i, j, k);
        }
      }
      for (int k = kb; k <= ke; ++k) {
        prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
        rho0(i, j, k) = prs0(i, j, k) / (rd * th0(i, j, k) * pi0(i, j, k) * (1.0 + qv0(i, j, k) * reps));
        t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
      }
      // must match how thv is computed in solve (bit-for-bit)
      for (int k = kb; k <= ke; ++k) {
        if (imoist == 1) {
          thv0(i, j, k) = (th0(i, j, k) + 0.0) * (1.0 + reps * std::max(0.0, qv0(i, j, k)))
                          / (1.0 + std::max(0.0, qv0(i, j, k)) + std::max(0.0, qc0(i, j, k) + qi0(i, j, k)));
        } else {
          thv0(i, j, k) = th0(i, j, k);
        }
      }
    }

  bcs(bc, rho0);
  extrapbcs(bc, rho0);
  if (psolver == 5 || psolver == 6)
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) rho0(i, j, k) = 1.0;
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rr0(i, j, k) = 1.0 / rho0(i, j, k);
  for (int k = 2; k <= nk; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rf0(i, j, k) = c1(i, j, k) * rho0(i, j, k - 1) + c2(i, j, k) * rho0(i, j, k);
  for (int j = jb; j <= je; ++j)
    for (int i = ib; i <= ie; ++i) {
      rf0(i, j, 1) = ex.cgs1 * rho0(i, j, 1) + ex.cgs2 * rho0(i, j, 2) + ex.cgs3 * rho0(i, j, 3);
      rf0(i, j, 0) = rf0(i, j, 1);
      rf0(i, j, nk + 1) = ex.cgt1 * rho0(i, j, nk) + ex.cgt2 * rho0(i, j, nk - 1) + ex.cgt3 * rho0(i, j, nk - 2);
    }
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rrf0(i, j, k) = 1.0 / rf0(i, j, k);

  // ---- base-state wind (iwnd) ----
  for (int k = 1; k <= nk; ++k)
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni + 1; ++i) u0(i, j, k) = 0.0;
  for (int k = 1; k <= nk; ++k)
    for (int j = 1; j <= nj + 1; ++j)
      for (int i = 1; i <= ni; ++i) v0(i, j, k) = 0.0;

  if (iwnd == 0) {
    // zero wind
  } else if (iwnd == 1) {
    // RKW-type wind profile
    const Real udep1 = 0.0, udep2 = 2500.0, uconst1 = 0.0, uconst2 = 10.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 0; j <= nj + 1; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (zu < udep1) u0(i, j, k) = uconst1;
          else if (zu > udep1 && zu < udep2) u0(i, j, k) = (uconst2 - uconst1) * (zu - udep1) / (udep2 - udep1) + uconst1;
          else u0(i, j, k) = uconst2;
        }
    for (int k = 1; k <= nk; ++k)
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 1; ++i) v0(i, j, k) = 0.0;
  } else if (iwnd == 2) {
    // Weisman-Klemp type supercell profile
    const Real udep1 = 2000.0, udep2 = 6000.0, umax1 = 7.0, umax2 = 31.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni + 1; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (zu <= udep1) {
            const Real angle = 90.0 * (zu / udep1) * (pi / 180.0);
            u0(i, j, k) = umax1 - umax1 * std::cos(angle);
          } else if (zu > udep1 && zu <= udep2) {
            u0(i, j, k) = umax1 + (zu - udep1) * (umax2 - umax1) / (udep2 - udep1);
          } else {
            u0(i, j, k) = umax2;
          }
        }
    const Real vmax1 = umax1;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni; ++i) {
          const Real zv = 0.5 * (zh(i, j - 1, k) + zh(i, j, k));
          if (zv <= udep1) {
            const Real angle = 90.0 * (zv / udep1) * (pi / 180.0);
            v0(i, j, k) = vmax1 * std::sin(angle);
          } else {
            v0(i, j, k) = vmax1;
          }
        }
  } else {
    unsupported("iwnd = " + std::to_string(iwnd));
  }

  // subtract off umove and vmove
  if (imove == 1) {
    info.umove = nl.getr("umove", 0.0);
    info.vmove = nl.getr("vmove", 0.0);
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni + 1; ++i) {
          u0(i, j, k) = u0(i, j, k) - info.umove;
          v0(i, j, k) = v0(i, j, k) - info.vmove;
        }
  } else {
    info.umove = 0.0;
    info.vmove = 0.0;
  }

  bcu(bc, u0);
  bcv(bc, v0);
  for (int j = jb; j <= je; ++j)
    for (int i = ib; i <= ie + 1; ++i) {
      u0(i, j, 0) = ex.cgs1 * u0(i, j, 1) + ex.cgs2 * u0(i, j, 2) + ex.cgs3 * u0(i, j, 3);
      u0(i, j, nk + 1) = ex.cgt1 * u0(i, j, nk) + ex.cgt2 * u0(i, j, nk - 1) + ex.cgt3 * u0(i, j, nk - 2);
    }
  for (int j = jb; j <= je + 1; ++j)
    for (int i = ib; i <= ie; ++i) {
      v0(i, j, 0) = ex.cgs1 * v0(i, j, 1) + ex.cgs2 * v0(i, j, 2) + ex.cgs3 * v0(i, j, 3);
      v0(i, j, nk + 1) = ex.cgt1 * v0(i, j, nk) + ex.cgt2 * v0(i, j, nk - 1) + ex.cgt3 * v0(i, j, nk - 2);
    }

  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rth0(i, j, k) = 1.0 / th0(i, j, k);

  for (auto p : std::initializer_list<decltype(std::make_pair(&bs.pi0, &pi0))>{{&bs.pi0, &pi0}, {&bs.th0, &th0}, {&bs.prs0, &prs0}, {&bs.rho0, &rho0},
                 {&bs.thv0, &thv0}, {&bs.rth0, &rth0}, {&bs.qv0, &qv0}, {&bs.qc0, &qc0}, {&bs.qi0, &qi0},
                 {&bs.rr0, &rr0}, {&bs.rf0, &rf0}, {&bs.rrf0, &rrf0}, {&bs.u0, &u0}, {&bs.v0, &v0}})
    p.first->deep_copy_from(*p.second);
}

//---------------------------------------------------------------------------
// gfortran random_number (libgfortran/intrinsics/random.c)
namespace {
inline std::uint64_t rotl64(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
std::array<std::int32_t, 8> cm1_seed() {
  constexpr int k = 8;
  std::array<std::int32_t, 8> sand{};
  for (int n = 1; n <= k; ++n)
    sand[n - 1] = std::int32_t(std::lround(2.0e9 * (2.0 * double(n - 1) / double(k - 1) - 1.0)));
  return sand;
}
} // namespace

GfortranRandom::GfortranRandom() : GfortranRandom(cm1_seed()) {}

GfortranRandom::GfortranRandom(const std::array<std::int32_t, 8>& put) {
  // random_seed(put=): the int32 array is copied in reverse order, then
  // xor'ed with the scramble keys
  std::int32_t rev[8];
  for (int i = 0; i < 8; ++i) rev[i] = put[7 - i];
  std::uint64_t seed[4];
  std::memcpy(seed, rev, sizeof(seed));
  constexpr std::uint64_t xor_keys[4] = {0xbd0c5b6e50c2df49ULL, 0xd46061cd46e1df38ULL,
                                         0xbb4f4d4ed6103544ULL, 0x114a583d0756ad39ULL};
  for (int i = 0; i < 4; ++i) m_s[i] = seed[i] ^ xor_keys[i];
}

double GfortranRandom::next() {
  // xoshiro256**
  const std::uint64_t result = rotl64(m_s[1] * 5, 7) * 9;
  const std::uint64_t t = m_s[1] << 17;
  m_s[2] ^= m_s[0];
  m_s[3] ^= m_s[1];
  m_s[1] ^= m_s[2];
  m_s[0] ^= m_s[3];
  m_s[2] ^= t;
  m_s[3] = rotl64(m_s[3], 45);
  // rnd_8: keep the top 53 bits, scale by 2^-64
  const std::uint64_t v = result & (~std::uint64_t(0) << 11);
  return double(v) * 0x1p-64;
}

//---------------------------------------------------------------------------
// init3d.F, getset (init3d.F) and the k/tke allocations in cm1.F
void init_state(const Namelist& nl, const Constants& c, const Grid& g, const InitInfo& info,
                BaseState& bs, State& s) {
  using namespace phys;
  const auto& B = g.b;
  const auto& bc = info.bc;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const int ib = B.ib, ie = B.ie, jb = B.jb, je = B.je, kb = B.kb, ke = B.ke;
  const int imoist = nl.geti("imoist", 0), ptype = nl.geti("ptype", 0), isnd = nl.geti("isnd", 1);
  const int iinit = nl.geti("iinit", 0), ibalance = nl.geti("ibalance", 0), psolver = nl.geti("psolver", 3);
  const int cm1setup = nl.geti("cm1setup", 1), sgsmodel = nl.geti("sgsmodel", 0), ipbl = nl.geti("ipbl", 0);
  const bool idoles = (cm1setup == 1 || cm1setup == 4);
  const bool idopbl = (cm1setup == 2 || cm1setup == 4);
  const int irandp = nl.geti("irandp", 0);
  if (irandp != 0 && irandp != 1) unsupported("irandp = " + std::to_string(irandp));
  if (irandp == 1 && nl.getb("use_truly_random_pert", false)) unsupported("use_truly_random_pert = .true.");
  if (nl.geti("iptra", 0) != 0) unsupported("iptra = 1");
  if (nl.getb("do_ib", false)) unsupported("do_ib = .true.");
  // param.F sets iusetke for MYNN (ipbl=4/5 or sfcmodel=6); init3d.F only
  // uses it together with idoles (cm1setup=1, 4), so cm1setup=2 is unaffected
  // (MYNN's arrays start at 0, cm1.F).
  if (idoles && ((idopbl && (ipbl == 4 || ipbl == 5)) || nl.geti("sfcmodel", 0) == 6))
    unsupported("MYNN tke with LES (cm1setup=1, 4)");
  const bool iusetke = idoles && (sgsmodel == 1 || sgsmodel == 3 || sgsmodel == 4 || sgsmodel == 5);
  if (imoist == 1 && !MoistLayout::known(ptype)) unsupported("ptype = " + std::to_string(ptype));
  const MoistLayout ml = MoistLayout::make(imoist == 1 ? 1 : 0, ptype);
  const int nqv = 1, nqc = ml.nqc, nqi = ml.nqi;

  auto xh = g.xh.to_host(), yh = g.yh.to_host();
  auto zh = g.zh.to_host();
  auto pi0 = bs.pi0.to_host(), th0 = bs.th0.to_host(), prs0 = bs.prs0.to_host(), rho0 = bs.rho0.to_host();
  auto rth0 = bs.rth0.to_host(), qv0 = bs.qv0.to_host(), qc0 = bs.qc0.to_host();
  auto u0 = bs.u0.to_host(), v0 = bs.v0.to_host();
  auto ua = s.ua.to_host(), va = s.va.to_host(), wa = s.wa.to_host();
  auto ppi = s.ppi.to_host(), tha = s.tha.to_host(), rho = s.rho.to_host(), prs = s.prs.to_host();
  auto qa = s.qa.to_host();
  auto tkea = s.tkea.to_host(), rain = s.rain.to_host();
  auto kmh = s.kmh.to_host(), kmv = s.kmv.to_host(), khh = s.khh.to_host(), khv = s.khv.to_host();
  auto q3 = [&](int n) {   // qa(:,:,:,n) as a host 3D array
    auto v = Kokkos::subview(qa.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1);
    return HArray3(v, {B.ibm, B.jbm, B.kbm});
  };

  // cm1.F: kmh/kmv = kminit, khh/khv = khinit (tkea = 0)
  {
    Real kminit = 0.0, khinit = 0.0;
    if (cm1setup == 3) {
      const Real viscosity = nl.getr("viscosity", 0.0), pr_num = nl.getr("pr_num", 1.0);
      kminit = viscosity;
      khinit = viscosity / pr_num;
    }
    Kokkos::deep_copy(kmh.view(), kminit); Kokkos::deep_copy(kmv.view(), kminit);
    Kokkos::deep_copy(khh.view(), khinit); Kokkos::deep_copy(khv.view(), khinit);
    Kokkos::deep_copy(tkea.view(), 0.0);
  }

  // ---- init3d ----
  Kokkos::deep_copy(rain.view(), 0.0);
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie + 1; ++i) ua(i, j, k) = u0(i, j, k);
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je + 1; ++j)
      for (int i = ib; i <= ie; ++i) va(i, j, k) = v0(i, j, k);

  if (imoist == 1) {
    auto qv = q3(nqv);
    for (int k = B.kbm; k <= B.kem; ++k)
      for (int j = B.jbm; j <= B.jem; ++j)
        for (int i = B.ibm; i <= B.iem; ++i) qv(i, j, k) = qv0(i, j, k);
    if (ptype >= 1 && (isnd == 4 || isnd == 9 || isnd == 10 || isnd == 11 || isnd == 15)) {
      auto qc = q3(nqc);
      for (int k = B.kbm; k <= B.kem; ++k)
        for (int j = B.jbm; j <= B.jem; ++j)
          for (int i = B.ibm; i <= B.iem; ++i) qc(i, j, k) = qc0(i, j, k);
    }
    if (ptype >= 1 && (isnd == 4 || isnd == 9 || isnd == 10) && ml.iice == 1) {
      auto qi0 = bs.qi0.to_host();
      auto qi = q3(nqi);
      for (int k = B.kbm; k <= B.kem; ++k)
        for (int j = B.jbm; j <= B.jem; ++j)
          for (int i = B.ibm; i <= B.iem; ++i) qi(i, j, k) = qi0(i, j, k);
    }
  }

  if (iinit == 0) {
    // no perturbation
  } else if (iinit == 1) {
    // warm bubble
    const Real ric = info.centerx, rjc = info.centery, zc = 1400.0;
    const Real bhrad = 10000.0, bvrad = 1400.0, bptpert = 1.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) {
          const Real a = (xh(i) - ric) / bhrad, b = (yh(j) - rjc) / bhrad, d = (zh(i, j, k) - zc) / bvrad;
          const Real beta = std::sqrt(a * a + b * b + d * d);
          if (beta < 1.0) {
            const Real cs = std::cos(0.5 * pi * beta);
            tha(i, j, k) = bptpert * (cs * cs);
          } else {
            tha(i, j, k) = 0.0;
          }
        }
  } else {
    unsupported("iinit = " + std::to_string(iinit));
  }
  // (maintain_rh is .false. for iinit = 0, 1)

  // random perturbations (testcase = 0): +-0.25 K on theta, two draws per
  // point of (0:nx+2, 0:ny+2, 1:nk), i fastest (serial: i = ii, j = jj).
  // init3d.F seeds the generator and draws one number ("rand-1") first.
  if (irandp == 1) {
    GfortranRandom rng;
    (void)rng.next();   // rand-1
    const Real amplitude = 0.25;
    for (int k = 1; k <= nk; ++k)
      for (int jj = 0; jj <= B.ny + 2; ++jj)
        for (int ii = 0; ii <= B.nx + 2; ++ii) {
          const Real rand = rng.next();
          (void)rng.next();   // rand2
          const int i = ii, j = jj;
          if (i >= ib && i <= ie && j >= jb && j <= je) tha(i, j, k) = tha(i, j, k) + amplitude * (2.0 * rand - 1.0);
        }
  }

  // ---- 3d pressure (setppi) ----
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) ppi(i, j, k) = 0.0;
  if (ibalance == 0) {
  } else if (ibalance == 1) {
    // hydrostatic balance, integrated top-down
    if (imoist == 1 && ptype == 0) unsupported("ibalance = 1 with imoist = 1 and ptype = 0");
    std::vector<Real> thvnew(nk + 1), pinew(nk + 1);
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        if (imoist == 1) {
          for (int k = 1; k <= nk; ++k) {
            Real qt = 0.0;
            for (int n = ml.nql1; n <= ml.nql2; ++n) qt = qt + qa(i, j, k, n);
            if (ml.iice == 1)
              for (int n = ml.nqs1; n <= ml.nqs2; ++n) qt = qt + qa(i, j, k, n);
            thvnew[k] = (th0(i, j, k) + tha(i, j, k)) * (1.0 + c.reps * qa(i, j, k, nqv)) / (1.0 + qa(i, j, k, nqv) + qt);
          }
        } else {
          for (int k = 1; k <= nk; ++k) thvnew[k] = th0(i, j, k) + tha(i, j, k);
        }
        pinew[nk] = pi0(i, j, nk);
        for (int k = nk - 1; k >= 1; --k)
          pinew[k] = pinew[k + 1] + c.g * (zh(i, j, k + 1) - zh(i, j, k)) / (c.cp * 0.5 * (thvnew[k + 1] + thvnew[k]));
        for (int k = 1; k <= nk; ++k) {
          ppi(i, j, k) = pinew[k] - pi0(i, j, k);
          if (std::abs(ppi(i, j, k)) < 1.0e-6) ppi(i, j, k) = 0.0;
        }
      }
  } else {
    unsupported("ibalance = " + std::to_string(ibalance));
  }

  // ---- getset: make sure boundary values are set properly ----
  bcu(bc, ua);
  bcv(bc, va);
  bcw(bc, wa, 1);
  bcs(bc, ppi);
  bcs(bc, tha);
  if (imoist == 1)
    for (int n = 1; n <= B.numq; ++n) bcs(bc, q3(n));
  if (idoles && iusetke) bcw(bc, tkea, 1);
  bcu(bc, u0);
  bcv(bc, v0);
  bcs(bc, th0);
  if (idoles && iusetke) {
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) tkea(i, j, 1) = tkea(i, j, 2);
    bc2d(bc, tkea, 1);
  }
  for (int k = kb; k <= ke; ++k)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) rth0(i, j, k) = 1.0 / th0(i, j, k);

  // prs, rho
  if (psolver == 4 || psolver == 5 || psolver == 6 || psolver == 7) {
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) { rho(i, j, k) = rho0(i, j, k); prs(i, j, k) = prs0(i, j, k); }
  } else if (imoist == 1) {
    auto qv = q3(nqv);
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) {
          prs(i, j, k) = p00 * std::pow(pi0(i, j, k) + ppi(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k) / ((th0(i, j, k) + tha(i, j, k)) * (pi0(i, j, k) + ppi(i, j, k))
                                         * (c.rd + std::max(0.0, qv(i, j, k)) * c.rv));
        }
  } else {
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) {
          prs(i, j, k) = p00 * std::pow(pi0(i, j, k) + ppi(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k) / (c.rd * (th0(i, j, k) + tha(i, j, k)) * (pi0(i, j, k) + ppi(i, j, k)));
        }
  }
  if (psolver <= 3) {
    bcs(bc, rho);
  } else {
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) rho(i, j, k) = rho0(i, j, k);
  }

  for (auto p : std::initializer_list<decltype(std::make_pair(&bs.th0, &th0))>{{&bs.th0, &th0}, {&bs.rth0, &rth0}, {&bs.u0, &u0}, {&bs.v0, &v0},
                 {&s.ua, &ua}, {&s.va, &va}, {&s.wa, &wa}, {&s.ppi, &ppi}, {&s.tha, &tha}, {&s.rho, &rho},
                 {&s.prs, &prs}, {&s.tkea, &tkea}, {&s.kmh, &kmh}, {&s.kmv, &kmv}, {&s.khh, &khh},
                 {&s.khv, &khv}, {&s.rain, &rain}})
    p.first->deep_copy_from(*p.second);
  s.qa.deep_copy_from(qa);
}

} // namespace init
} // namespace cm1k
