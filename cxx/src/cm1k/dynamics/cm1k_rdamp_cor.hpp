#ifndef CM1K_RDAMP_COR_HPP
#define CM1K_RDAMP_COR_HPP

// Rayleigh damping coefficients (param.F) and the Coriolis terms (solve2.F).
//
//   rdamp_coeffs: tauh, taus, tauf of param.F's "Rayleigh damper in
//     vertical" (irdamp >= 1, zd < maxz) and "Rayleigh damping near lateral
//     boundaries" (hrdamp >= 1, which acts on tauf, i.e. on w only). The
//     damping tendencies themselves are applied in solve1 (Dyncore::solve1).
//   coriolis: solve2.F's f-plane Coriolis terms (icor=1, betaplane=0,
//     axisymm=0), added to uten/vten in every RK stage.

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"

namespace cm1k {

struct RdampParams {
  int irdamp = 0, hrdamp = 0;
  Real zd = 0, xhd = 0;
  // param.F's minx, maxx, miny, maxy (xfref(1), xfref(nx+1), ...): set them
  // (have_extent) when the grid arrays are a subdomain's (MPI); otherwise
  // they are taken from xf(1), xf(nx+1), yf(1), yf(ny+1).
  bool have_extent = false;
  Real minx = 0, maxx = 0, miny = 0, maxy = 0;
};

// Fills tauh, taus (ib:ie,jb:je,kb:ke) and tauf (ib:ie,jb:je,kb:ke+1), on
// the host, from the grid (zh, zf, sigmaf, xh, yh, xf, yf): zero everywhere,
// then param.F's formulas. Throws (param.F stops) when the damping layer
// would cover more than half the domain.
void rdamp_coeffs(const RdampParams& p, const Grid& g, const FArray3& tauh, const FArray3& taus,
                  const FArray3& tauf);

// uten(1:ni+1,1:nj+1,1:nk) += fcor*(v at u points + vmove)
// vten(1:ni+1,1:nj+1,1:nk) -= fcor*(u at v points + umove)
void coriolis(int ni, int nj, int nk, Real fcor, Real umove, Real vmove, const FArray3& u3d,
              const FArray3& v3d, const FArray3& uten, const FArray3& vten);

} // namespace cm1k

#endif
