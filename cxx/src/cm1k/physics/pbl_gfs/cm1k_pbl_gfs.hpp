#ifndef CM1K_PBL_GFS_HPP
#define CM1K_PBL_GFS_HPP

// Port of the GFS-EDMF PBL scheme (ipbl=3, "GFS-EDMF (as configured in
// HWRF)"): src/module_bl_gfsedmf.F
//
//   BL_GFSEDMF   the WRF-style driver (column packing, unit conversions,
//                tendencies, u*, wspd, PBL height, diffusivities)
//   moninedmf    the hybrid EDMF scheme: local K above the PBL, the K
//                profile with Weiguo Wang's height-dependent shape
//                (useshape = 2) and RGF's wind-dependent cap over water
//                (alpha < 0), stratocumulus cloud-top mixing, implicit
//                diffusion and the TKE dissipative heating (dspheat)
//   mfpbl        the mass flux of the strongly unstable (convective) PBL
//   tridin, tridi2  the tridiagonal solvers
//
// as src/turb.F (sfc_and_turb, lines 2123-2187) calls it. One parallel_for
// over the (i,j) columns, each running the Fortran's passes in order with
// serial k loops; bit-for-bit with the double-precision Fortran
// (tests/fortran_bridge/gfs_tests.cpp, against the unmodified module through
// cm1f_bridge_gfs.F90).
//
// What CM1's call fixes (turb.F 503-514, 2124-2160), and so what is ported:
//   disheat = .true., gfs_alpha = -1, var_ric = 1, coef_ric_l = 0.16,
//   coef_ric_s = 0.25, pert_pbl = .false. (the ran1 perturbation is
//   commented out), ntrac = 3 with cloud ice (p_qi = nqi >= 1), else 2;
//   ntcw = 2; xkzm_m = xkzm_h = xkzm_s = 1; kinver = nk. Options allows
//   other alpha, var_ric, coef_ric_* and disheat for the component tests.
// The diagnostics that never feed back are not computed: stau, hpbl2d,
// evap2d, heat2d, brcr, wscale, wscaleu, prkpp, and moninedmf's dusfc,
// dvsfc, dtsfc, dqsfc.
//
// alpha < 0 over water (xland = 2) reads WSPM(i,2), the level that brackets
// 500 m among k = 1..nk/2; the Fortran leaves it undefined when no level
// does (zi(nk/2+1) <= 500 m): the port throws then.
//
// Inputs, as turb.F prepares them (physics/sfclay/cm1k_wrf_common.hpp):
//   u3d = dum1, v3d = dum2, t3d = dum4, qv3d = divx, qc3d = ppten (qc),
//   qi3d = dum8 (qi), p3d = prs, pi3d = dum7, dz8w = dum5, z = zf (w
//   levels, k = 1..nk), rthraten = dum9 (0 without radiation), and the
//   2D psfc, psim, psih, hfx, qfx, tsk, gz1oz0, br, u10, v10, znt (mznt for
//   sfcmodel=4, else znt), xland.
// Inout: ust (= sqrt(stress)), wspd (= |(u,v)(k=1)| + 1e-9).
// Outputs: hpbl, kpbl2d (as Real), the tendencies upten = rublten,
// vpten = rvblten, thpten = rthblten (theta tendency, incl. the dissipative
// heating), qvpten, qcpten, qipten (only with ntrac = 3), xkzm = dku3d and
// xkzh = dkt3d at k = 2..nk (k = 1 untouched), dissten (k = 1..nk: moninedmf's
// diss(k-1)) and dheat (the dissipative heating rate, turb.F's thten).
// With disheat = .false. (not reachable from CM1) the Fortran never sets
// diss, so dissten is undefined there; the port leaves the work array's
// contents.
// 3D arrays are (ib:ie,jb:je,kb:ke) or wider (kb <= 1, ke >= nk); 2D arrays
// FArray3 (ib:ie,jb:je,1:1). Interior points 1..ni x 1..nj.

#include "physics/sfclay/cm1k_wrf_common.hpp"

namespace cm1k::pbl_gfs {

using wrfphys::WrfConst;

struct Options {
  bool disheat = true;         // turb.F
  Real alpha = -1.0;           // turb.F: gfs_alpha
  Real var_ric = 1.0;          // turb.F
  Real coef_ric_l = 0.16, coef_ric_s = 0.25;
  bool qi = false;             // P_QI >= P_FIRST_SCALAR (nqi >= 1): ntrac = 3
};

struct Inputs {
  FArray3 u3d, v3d, t3d, qv3d, qc3d, qi3d, p3d, pi3d, dz8w, z, rthraten;          // 3D
  FArray3 psfc, psim, psih, hfx, qfx, tsk, gz1oz0, br, u10, v10, znt, xland;      // 2D
};

struct Outputs {
  FArray3 rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, dku3d, dkt3d, dissten, dheat;   // 3D
  FArray3 ust, wspd, hpbl, kpbl2d;                                                                   // 2D
};

// Scratch space for the per-column arrays (reused across calls).
struct Workspace {
  Kokkos::View<Real****, Kokkos::LayoutLeft, Device> w;
  Kokkos::View<int, Device> err;
  void ensure(int ni, int nj, int nk);
};

// turb.F dogfsedmf: BL_GFSEDMF on all columns. Needs nk >= 4.
void gfsedmf(const Options& o, const WrfConst& c, Real dt, int ni, int nj, int nk, const Inputs& in,
             const Outputs& out, Workspace& ws);

} // namespace cm1k::pbl_gfs

#endif
