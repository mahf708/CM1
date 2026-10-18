#ifndef CM1K_DIFF2_HPP
#define CM1K_DIFF2_HPP

// Port of src/diff2.F: 2nd-order explicit diffusion with a constant
// viscosity kdiff2 (idiff=1, difforder=2; param.F allows it only with
// cm1setup=0). Cartesian grid without terrain (axisymm=0,
// terrain_flag=.false.), as diff2.F requires.
//
//   diff2def: stress tensor t11..t33 and divx from u3d/v3d/w3d (cm1.F calls
//             it at the end of each pass of the time loop, after sfc_and_turb)
//   diff2u/v/w: momentum tendencies from the stresses (solve1)
//   diff2s:     scalar tendency (solve1, for theta and every q)
//
// Not ported (no effect on the prognostic variables here): the budget
// diagnostics (udiag, ..., dobud), and diff2def's dissipation rate
// (dissten, used only with idiss=1 or output_dissten=1; DynConfig::check
// throws for idiss=1 with difforder=2). bbc=3 (semi-slip, needs ust) is
// not ported either.

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"

namespace cm1k {

// The `input`-module variables diff2.F reads.
struct Diff2Params {
  int ni = 0, nj = 0, nk = 0;
  Real kdiff2 = 0, rdx = 0, rdy = 0, rdz = 0, smeps = 1.0e-60, umove = 0, vmove = 0;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;   // serial: ib? = 1 iff the boundary is not periodic
  int bbc = 1, tbc = 1;
};

// diff2def: divx (ib:ie,jb:je,kb:ke) and t11,t12,t13,t22,t23,t33 (same
// shape) from ua (u-staggered), va (v-staggered), wa (w-staggered).
void diff2def(const Diff2Params& p, const Grid& g, const FArray3& divx, const FArray3& rho,
              const FArray3& rf, const FArray3& t11, const FArray3& t12, const FArray3& t13,
              const FArray3& t22, const FArray3& t23, const FArray3& t33, const FArray3& ua,
              const FArray3& va, const FArray3& wa);

// dum, diffx, diffy, diffz: scratch (ib:ie,jb:je,kb:ke).
void diff2u(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& uten, const FArray3& rho,
            const FArray3& divx, const FArray3& t11, const FArray3& t12, const FArray3& t13);
void diff2v(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& vten, const FArray3& rho,
            const FArray3& divx, const FArray3& t22, const FArray3& t12, const FArray3& t23);
void diff2w(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& wten, const FArray3& rho,
            const FArray3& rf, const FArray3& divx, const FArray3& t33, const FArray3& t13,
            const FArray3& t23);
void diff2s(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& s, const FArray3& sten,
            const FArray3& rho, const FArray3& rr, const FArray3& rf);

// The arrays cm1.F carries for this scheme between diff2def and solve1
// (t11..t33) plus diff2's scratch arrays.
struct Diff2Work {
  FArray3 t11, t12, t13, t22, t23, t33;
  FArray3 dum, diffx, diffy, diffz;
  void allocate(const Bounds& b);
};

} // namespace cm1k

#endif
