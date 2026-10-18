#ifndef CM1K_PBL_YSU_HPP
#define CM1K_PBL_YSU_HPP

// Port of the YSU PBL scheme (ipbl=1): src/bl_ysu.F (bl_ysu_run,
// tridin_ysu, tridi2n), as CM1 calls it through mmm_physics_wrapper's
// ysu_wrapper from the ysu_j_loop of src/turb.F (sfc_and_turb, lines
// 1890-2063), plus the YSU dissipative heating of turb.F (ysudiss, lines
// 2717-2752).
//
// One parallel_for over the (i,j) columns, each column running the
// Fortran's passes in order with serial k loops; bit-for-bit with the
// double-precision Fortran (cxx/README.md).
//
// What CM1's call fixes, and so what is ported:
//   * ctopo, ctopo2, uox, vox, the BEP urban arrays, dusfc..dqsfc are not
//     passed (absent optionals): bepswitch = 0, uoxl = voxl = 0,
//     ad(1) = 1+fric(1), u10/v10 are not modified.
//   * f_qc = f_qi = .true.; qcx = qc (0 without qc), qix = qi (0 without qi).
//   * nmix = 1 with qmix = 0 and the result qmixtnp discarded: not ported.
//   * ysu_topdown_pblmix = .false. in CM1; the .true. branch (cloud-top
//     entrainment) is ported too (Options), but where kpbl = nk it reads
//     thlix/thx/qvx/qcxl at k = nk+1, beyond the Fortran arrays; the port
//     reads 0 there instead of whatever memory follows.
//   * The "paj tke" block and get_pblh (hybrid MYNN PBL height) only feed
//     vconv, which only matters when ctopo is present: dead code for CM1's
//     call, not ported.
//
// Inputs, as turb.F prepares them (see physics/sfclay/cm1k_wrf_common.hpp):
//   ux = dum1, vx = dum2 (u, v at scalar points), tx = dum4 (T),
//   qvx = divx, qcx = ppten, qix = dum8, p2d = prs, p2di = dum6 (p on w
//   levels, k=1..nk+1, dum6(1) = psfc), pi2d = dum7, dz8w2d = dum5,
//   rthraten = dum9 (radiative theta tendency; 0 without radiation),
//   psfcpa = psfc, znt = zntmp, ust, psim = fm, psih = fh (the integrated
//   ln(z/z0)-psi of the surface layer), xland, hfx, qfx, wspd, br, u10, v10.
// Outputs: hpbl, kpbl2d (as Real), wstar, delta, and the tendencies
// upten = utnp, vpten = vtnp, thpten = ttnp (theta tendency), qvpten,
// qcpten, qipten, and xkzh(i,j,k) = exch_hx(i,k), xkzm(i,j,k) = exch_mx(i,k)
// (k = 1..nk; exch(1) = 0, exch(k+1) = Kh/Km at the top of layer k).
// 3D arrays are (ib:ie,jb:je,kb:ke) (kb <= 1, ke >= nk+1); 2D arrays are
// FArray3 (ib:ie,jb:je,1:1) as in physics/sfc. Interior points 1..ni x 1..nj.
//
// Wiring into sfc_and_turb (Dyncore::sfc_and_turb / Dyncore::pbl, case 1):
//   turb.F 1764-1826  qi -> dum8, qc -> ppten, dum9 = thraten or 0
//                     -> wrfphys::pbl_moist (+ rthraten)
//   turb.F 1828-1849  dum1..dum7                     -> wrfphys::pbl_prep
//   turb.F 1890-2048  ysu_j_loop / ysu_wrapper        -> ysu()
//   turb.F 2050-2063  cm1setup=4 zeroing of xkzh/xkzm (not ported)
//   turb.F 2717-2752  ysudiss (idiss=1 or output_dissten=1) -> ysu_dissten()
//   turb.F 2758-2783  bcs(upten), bcs(vpten): the caller's halo fill
// The tendencies are then applied by the solver (solve1.F: thpten etc. added
// to the theta/qv/u/v tendencies), outside this component.

#include "physics/sfclay/cm1k_wrf_common.hpp"

namespace cm1k::pbl_ysu {

using wrfphys::WrfConst;

struct Options {
  bool ysu_topdown_pblmix = false;   // turb.F: ysu_topdown_pblmix = .false.
};

struct Inputs {
  FArray3 ux, vx, tx, qvx, qcx, qix, p2d, p2di, pi2d, dz8w2d, rthraten;   // 3D
  FArray3 psfcpa, znt, ust, psim, psih, xland, hfx, qfx, wspd, br;       // 2D
};

struct Outputs {
  FArray3 utnp, vtnp, ttnp, qvtnp, qctnp, qitnp, exch_hx, exch_mx;   // 3D (upten.. xkzh, xkzm)
  FArray3 hpbl, kpbl2d, wstar, delta, u10, v10;                       // 2D (u10, v10 inout)
};

// Scratch space for the per-column arrays (reused across calls).
struct Workspace {
  Kokkos::View<Real****, Kokkos::LayoutLeft, Device> w;
  void ensure(int ni, int nj, int nk);
};

// turb.F ysu_j_loop: for every column, ysu_wrapper -> bl_ysu_run.
void ysu(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
         const Outputs& out, Workspace& ws);
void ysu(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
         const Outputs& out);

// turb.F ysudiss (lines 2717-2752): dissipative heating from the YSU
// momentum tendencies, added to dissten (w levels 2..nk). thten/thten1 are
// work arrays (the column stresses t13/t23), dum1/dum2 = u/v at scalar
// points, rmh/mf grid metrics, dz/rdz grid spacing.
void ysu_dissten(int ni, int nj, int nk, Real dz, Real rdz, const FArray3& rmh, const FArray3& mf,
                 const FArray3& upten, const FArray3& vpten, const FArray3& dum1, const FArray3& dum2,
                 const FArray3& thten, const FArray3& thten1, const FArray3& dissten);

} // namespace cm1k::pbl_ysu

#endif
