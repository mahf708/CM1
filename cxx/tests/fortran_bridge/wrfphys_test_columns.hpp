#ifndef CM1K_WRFPHYS_TEST_COLUMNS_HPP
#define CM1K_WRFPHYS_TEST_COLUMNS_HPP

// Synthetic but physically plausible model columns for the component tests
// of the WRF/MMM surface layers (sfclay_tests.cpp) and the YSU PBL
// (ysu_tests.cpp): CM1-shaped arrays (halos included) of a small grid whose
// interior columns sweep
//   i: stable / neutral / convective  x  land / water  x  cloud-topped or not
//   j: moist / dry  x  wind speed (calm .. 25 m/s)
// with per-column random roughness, u*, first-level height, surface
// temperature excess/deficit, grid spacing (Mahrt-Sun term), ...
// One column (i=2, j=1) is set up so that the surface layer sees exactly
// dthv/dz = 0 (bulk Richardson number 0: regime 3).

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cm1k::wrftest {

using H3 = FArray<3, HostSpace>;
using H4 = FArray<4, HostSpace>;

inline Bounds box(int ni, int nj, int nk, int ngxy) {
  Bounds b;
  b.nx = ni; b.ny = nj; b.nz = nk;
  b.ni = ni; b.nj = nj; b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = ngxy; b.ngz = 1;
  b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.numq = 3;
  return b;
}

inline H3 h3(const std::string& n, const std::array<int, 3>& lo, const std::array<int, 3>& hi) {
  return H3(Kokkos::View<Real***, Kokkos::LayoutLeft, HostSpace>(n, hi[0] - lo[0] + 1, hi[1] - lo[1] + 1,
                                                                   hi[2] - lo[2] + 1),
            lo);
}

inline H3 hcopy(const H3& h) {
  H3 c = h3(h.label() + "_copy", {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  std::memcpy(c.data(), h.data(), h.size() * sizeof(double));
  return c;
}

// Bit-for-bit comparison (NaN == NaN); optionally only where mask(i,j) holds.
template <typename Mask>
inline size_t count_diff(const H3& a, const H3& ref, double& maxd, Mask mask) {
  size_t nbad = 0;
  maxd = 0;
  for (int k = a.lb(2); k <= a.ub(2); ++k)
    for (int j = a.lb(1); j <= a.ub(1); ++j)
      for (int i = a.lb(0); i <= a.ub(0); ++i) {
        if (!mask(i, j)) continue;
        const double x = a(i, j, k), y = ref(i, j, k);
        if (!(x == y) && !(std::isnan(x) && std::isnan(y))) {
          ++nbad;
          maxd = std::max(maxd, std::fabs(x - y));
        }
      }
  return nbad;
}

struct Spec {
  int regime = 0;     // 0 stable, 1 neutral, 2 convective
  bool water = false, moist = true, cloud = false, exact_neutral = false;
  double S0 = 5, h = 800, ths = 300, dthsfc = 0, q0 = 0.01, zfac = 1, dir0 = 0;
};

// The gridded inputs (what sfc_and_turb has when it calls the surface layer
// and the PBL scheme) plus the 2D surface state.
struct Columns {
  Bounds b;
  Real dz = 100.0, cgt1 = 1.875, cgt2 = -1.25, cgt3 = 0.375;
  std::vector<Spec> spec;   // interior columns, (i-1) + ni*(j-1)
  // 3D (ib:ie,jb:je,kb:ke); ua/u0 (ib:ie+1,...), va/v0 (...,jb:je+1,...)
  H3 ua, va, u0, v0, th0, tha, pi0, ppi, prs, rmh, c1, c2, rf, rthraten, mf;
  H4 qa;   // (ib:ie,jb:je,kb:ke,1:3): qv, qc, qi
  // 2D (ib:ie,jb:je,1)
  H3 psfc, tsk, xland, znt, mavail, ust, mol, qsfc, hpbl, dsxy, hfx, qfx, zol, regime;

  const Spec& sp(int i, int j) const {
    const int ii = std::clamp(i, 1, b.ni), jj = std::clamp(j, 1, b.nj);
    return spec[(ii - 1) + b.ni * (jj - 1)];
  }
};

inline Columns make_columns(int ni, int nj, int nk, unsigned seed) {
  Columns C;
  C.b = box(ni, nj, nk, 2);
  const auto& b = C.b;
  std::mt19937_64 gen(seed);
  auto U01 = [&]() { return std::uniform_real_distribution<double>(0.0, 1.0)(gen); };
  auto U = [&]() { return 2.0 * U01() - 1.0; };
  const Constants kc = Constants::make(0);
  const Real g = kc.g, rd = kc.rd, rovcp = kc.rovcp;

  // column specs
  const double speeds[4] = {0.4, 3.0, 9.0, 22.0};
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      Spec s;
      s.regime = (i - 1) % 3;
      s.water = ((i - 1) / 3) % 2 == 1;
      s.moist = (j - 1) % 2 == 0;
      s.cloud = s.moist && ((i - 1) / 6) % 2 == 1 && s.regime > 0;
      s.S0 = speeds[((j - 1) / 2) % 4] * (0.7 + 0.6 * U01());
      s.ths = 300.0 + 2.0 * U();
      s.zfac = 0.5 + U01();
      s.dir0 = 3.14159 * U();
      s.q0 = 0.005 + 0.011 * U01();
      if (s.regime == 0) {
        s.h = 100 + 200 * U01();
        s.dthsfc = -(0.5 + 5.5 * U01());
      } else if (s.regime == 1) {
        s.h = 500 + 500 * U01();
        s.dthsfc = 0.1 * U();
      } else {
        s.h = 600 + 900 * U01();
        s.dthsfc = 1.0 + 7.0 * U01();
      }
      if (i == 2 && j == 1) {
        s.exact_neutral = true;
        s.water = false;
        s.moist = true;
      }
      C.spec.push_back(s);
    }

  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
  const std::array<int, 3> lo2{b.ib, b.jb, 1}, hi2{b.ie, b.je, 1};
  C.ua = h3("ua", lo, {b.ie + 1, b.je, b.ke});
  C.u0 = h3("u0", lo, {b.ie + 1, b.je, b.ke});
  C.va = h3("va", lo, {b.ie, b.je + 1, b.ke});
  C.v0 = h3("v0", lo, {b.ie, b.je + 1, b.ke});
  for (auto* p : {&C.th0, &C.tha, &C.pi0, &C.ppi, &C.prs, &C.rmh, &C.c1, &C.c2, &C.rf, &C.rthraten})
    *p = h3("a3", lo, hi);
  C.mf = h3("mf", lo, {b.ie, b.je, b.ke + 1});
  C.qa = H4(Kokkos::View<Real****, Kokkos::LayoutLeft, HostSpace>("qa", b.ie - b.ib + 1, b.je - b.jb + 1,
                                                                   b.ke - b.kb + 1, 3),
            {b.ib, b.jb, b.kb, 1});
  for (auto* p : {&C.psfc, &C.tsk, &C.xland, &C.znt, &C.mavail, &C.ust, &C.mol, &C.qsfc, &C.hpbl, &C.dsxy,
                  &C.hfx, &C.qfx, &C.zol, &C.regime})
    *p = h3("a2", lo2, hi2);

  std::vector<double> zh(nk + 2), dzk(nk + 2), th(nk + 2), q(nk + 2), p(nk + 2), spd(nk + 2), dir(nk + 2);
  for (int j = b.jb; j <= b.je + 1; ++j)
    for (int i = b.ib; i <= b.ie + 1; ++i) {
      const Spec& s = C.sp(i, j);
      // layer depths and heights
      double z = 0;
      for (int k = 1; k <= nk; ++k) {
        dzk[k] = s.zfac * std::min(20.0 + 12.0 * (k - 1), 300.0) * (1.0 + 0.02 * U());
        zh[k] = z + 0.5 * dzk[k];
        z += dzk[k];
      }
      // theta, qv, wind
      for (int k = 1; k <= nk; ++k) {
        const double zz = zh[k];
        if (s.regime == 0) {
          const double g1 = 0.01 + 0.05 * U01();
          th[k] = s.ths + g1 * std::min(zz, 250.0) + 0.004 * std::max(zz - 250.0, 0.0);
        } else if (s.regime == 1) {
          th[k] = s.ths + 0.003 * std::max(zz - s.h, 0.0);
        } else {
          th[k] = s.ths + 1.0 * std::exp(-zz / 40.0) + 3.0 * std::min(std::max(zz - s.h, 0.0) / 150.0, 1.0) +
                  0.004 * std::max(zz - s.h - 150.0, 0.0);
        }
        th[k] += 0.02 * U();
        q[k] = s.moist ? (zz < s.h ? s.q0 * (1.0 + 0.05 * std::exp(-zz / 50.0)) : 0.5 * s.q0 * std::exp(-(zz - s.h) / 2500.0))
                       : 0.0;
        spd[k] = s.S0 * (0.3 + 0.7 * std::min(zz / s.h, 1.0) + 0.2 * std::max(std::min((zz - s.h) / 1000.0, 1.0), 0.0));
        dir[k] = s.dir0 + 0.7 * std::min(zz / (2.0 * s.h), 1.0);
      }
      // hydrostatic pressure
      const double psfc = s.exact_neutral ? 1.0e5 : 1.0e5 + 2000.0 * U();
      const double thgb = th[1] + s.dthsfc;
      double pprev = psfc, zprev = 0;
      for (int k = 1; k <= nk; ++k) {
        const double tv = th[k] * std::pow(pprev / 1.0e5, rovcp) * (1.0 + 0.61 * q[k]);
        p[k] = pprev * std::exp(-g * (zh[k] - zprev) / (rd * tv));
        pprev = p[k];
        zprev = zh[k];
      }
      if (s.exact_neutral) p[1] = 1.0e5;
      const bool in_u = i <= b.ie + 1 && j <= b.je, in_v = i <= b.ie && j <= b.je + 1, in_s = i <= b.ie && j <= b.je;
      for (int k = b.kb; k <= b.ke; ++k) {
        const int kk = std::clamp(k, 1, nk);
        const double u = spd[kk] * std::cos(dir[kk]), v = spd[kk] * std::sin(dir[kk]);
        if (in_u) {
          C.ua(i, j, k) = u + 0.3 * U();
          C.u0(i, j, k) = 0.5 * u + 0.5 * U();
        }
        if (in_v) {
          C.va(i, j, k) = v + 0.3 * U();
          C.v0(i, j, k) = 0.5 * v + 0.5 * U();
        }
        if (!in_s) continue;
        double pk = p[kk];
        if (k == 0) pk = psfc + (psfc - p[1]);
        if (k == nk + 1) pk = p[nk] - (p[nk - 1] - p[nk]);
        const double pik = std::pow(pk / 1.0e5, rovcp);
        C.prs(i, j, k) = pk;
        C.tha(i, j, k) = 0.3 * U();
        C.th0(i, j, k) = th[kk] - C.tha(i, j, k);
        C.ppi(i, j, k) = 1.0e-4 * U();
        C.pi0(i, j, k) = pik - C.ppi(i, j, k);
        C.rmh(i, j, k) = dzk[kk] / C.dz;
        C.c1(i, j, k) = 0.5 + 0.05 * U();
        C.c2(i, j, k) = 1.0 - C.c1(i, j, k);
        C.rf(i, j, k) = pk / (rd * th[kk] * pik);
        C.mf(i, j, k) = 1.0 / (0.9 + 0.2 * U01());
        C.qa(i, j, k, 1) = q[kk];
        const double zz = zh[kk];
        const bool cl = s.cloud && zz > s.h - 150.0 && zz < s.h + 250.0;
        C.qa(i, j, k, 2) = cl ? 2.0e-4 + 4.0e-4 * U01() : 0.0;
        C.qa(i, j, k, 3) = (s.cloud && zz > 5000.0 && zz < 6500.0) ? 3.0e-5 * (1.0 + U()) : 0.0;
        C.rthraten(i, j, k) = cl ? -3.0e-4 * U01() : 1.0e-5 * U();
      }
      if (!in_s) continue;
      C.mf(i, j, b.ke + 1) = 1.0;
      if (s.exact_neutral) {
        C.tha(i, j, 1) = 0.0;
        C.ppi(i, j, 1) = 0.0;
        C.pi0(i, j, 1) = 1.0;
        C.th0(i, j, 1) = 295.0;
        C.qa(i, j, 1, 1) = 0.01;
      }
      // surface state
      C.psfc(i, j, 1) = psfc;
      C.tsk(i, j, 1) = s.exact_neutral ? 295.0 : thgb * std::pow(psfc / 1.0e5, rovcp);
      C.xland(i, j, 1) = s.water ? 2.0 : 1.0;
      C.znt(i, j, 1) = s.water ? 2.0e-4 * (0.5 + U01()) : std::pow(10.0, -3.0 + 2.7 * U01());
      C.mavail(i, j, 1) = s.water ? 1.0 : 0.2 + 0.8 * U01();
      double ust = 0.05 + 0.6 * U01();
      if ((i + j) % 5 == 0) ust = 0.005;
      if ((i + 2 * j) % 7 == 0) ust = 0.0005;
      C.ust(i, j, 1) = ust;
      C.mol(i, j, 1) = s.regime == 0 ? 0.05 + 0.45 * U01() : (s.regime == 2 ? -(0.05 + 0.45 * U01()) : 0.0);
      if ((i * 3 + j) % 11 == 0) C.mol(i, j, 1) = -C.mol(i, j, 1);
      C.qsfc(i, j, 1) = ((i + j) % 4 == 0) ? 0.0 : 0.01 * U01();
      if (s.exact_neutral) C.qsfc(i, j, 1) = 0.01;
      C.hpbl(i, j, 1) = s.h;
      C.dsxy(i, j, 1) = std::pow(10.0, 2.0 + 2.5 * U01());
      C.hfx(i, j, 1) = s.regime == 2 ? 50.0 + 350.0 * U01() : (s.regime == 0 ? -60.0 * U01() : 5.0 * U());
      C.qfx(i, j, 1) = s.moist ? 3.0e-4 * U01() : 0.0;
      C.zol(i, j, 1) = 2.0 * U();
      C.regime(i, j, 1) = double(1 + int(4 * U01()));
    }
  return C;
}

} // namespace cm1k::wrftest

#endif
