#ifndef CM1K_BC_HPP
#define CM1K_BC_HPP

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k {

// Lateral/vertical boundary conditions: port of bc.F (bcs, bcp, bcu, bcv,
// bcw) for serial runs (the `#ifndef MPI` branches): with no MPI, the
// periodic case (wbc=1, ...) copies across the domain.
//
// Each Fortran loop nest is one parallel_for, executed in the same order as
// the Fortran (west, east, south, north), so the corner points end up with
// exactly the same values.
// Periodic halo copies read points that the same loop writes when the
// domain is narrower than the halo (e.g. ny=1 with ngxy=3: s(i,-1) =
// s(i,0), which the loop sets from s(i,1) first). The serial Fortran loop
// order makes every such point a copy of the interior point src+m*n; pwrap
// maps the source index there directly (lo..lo+n-1), so each point of a
// parallel_for reads only interior values. Identity when src is already in
// range (n >= ngxy).
KOKKOS_INLINE_FUNCTION int pwrap(int src, int lo, int n) {
  int r = (src - lo) % n;
  if (r < 0) r += n;
  return lo + r;
}

struct BcParams {
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int axisymm = 0;
  int ni = 0, nj = 0, nk = 0, ngxy = 0;
  int ib = 0, ie = 0, jb = 0, je = 0;

  static BcParams make(const Bounds& b, const Namelist& nl);
};

// scalars (s at ib:ie,jb:je,kb:ke), all ngxy halo points
void bcs(const BcParams& p, const FArray3& s);
// pressure-like scalars: one halo point only
void bcp(const BcParams& p, const FArray3& s);
// u (ib:ie+1,jb:je,kb:ke)
void bcu(const BcParams& p, const FArray3& u);
// v (ib:ie,jb:je+1,kb:ke)
void bcv(const BcParams& p, const FArray3& v);
// w (ib:ie,jb:je,kb:ke+1); flag=1 also zeroes w at k=1 and k=nk+1
void bcw(const BcParams& p, const FArray3& w, int flag);
// bc2d on level k of a 3D array (bc.F bc2d(s(ib,jb,k))); all halo points
void bc2d(const BcParams& p, const FArray3& s, int k);
// bcwsfc (terrain): w at the surface (k=1) from the terrain slope,
// i=0..ni+1, j=0..nj+1; gz, dzdx, dzdy are the (i,j,1) terrain metrics
void bcwsfc(const BcParams& p, const FArray3& gz, const FArray3& dzdx, const FArray3& dzdy, const FArray3& u,
            const FArray3& v, const FArray3& w);

} // namespace cm1k

#endif
