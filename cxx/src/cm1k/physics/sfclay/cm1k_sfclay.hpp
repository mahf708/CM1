#ifndef CM1K_SFCLAY_HPP
#define CM1K_SFCLAY_HPP

// Ports of the two MM5-type surface layers CM1 takes from WRF/MMM:
//
//   sfcmodel = 2   src/sfclay.F       module_sf_sfclay: SFCLAY/SFCLAY1D,
//                                     sfclayinit ("MM5 similarity theory")
//   sfcmodel = 3   src/sf_sfclayrev.F sf_sfclayrev: sf_sfclayrev_run (via
//                                     mmm_physics_wrapper's
//                                     sf_sfclayrev_wrapper), _init, zolri,
//                                     psi* table functions ("revised MM5")
//
// and the parts of src/turb.F sfc_and_turb around them (sfclay_post).
// Every (i,j) column is independent: one parallel_for over 1..ni x 1..nj,
// with the Fortran's per-point operation order kept exactly (bit-for-bit
// parity with the double-precision Fortran, see cxx/README.md).
//
// Wiring into sfc_and_turb (turb.F; Dyncore::sfc_and_turb through
// sfclay::wrf_surface_layer, cm1k_sfclay_driver.hpp):
//
//   turb.F ~292      psfc = cgs1*prs(1)+cgs2*prs(2)+cgs3*prs(3) (as for
//                    sfcmodel=1, see physics/sfc)
//   turb.F ~335-376  dum1/dum2 (u,v at scalar points), dum4 (T), dum5 (dz8w)
//                    -> wrfphys::pbl_prep (same formulas, all k)
//   turb.F ~388-440  divx = qv -> wrfphys::pbl_moist
//   turb.F 732-751   call SFCLAY(...)                     -> sfclay()
//   turb.F 753-839   sfclayrev_j_loop / sf_sfclayrev_wrapper -> sfclayrev()
//   turb.F 1007-1110 qsfc diagnosis (update_sfc = .false.), ch=chs2,
//                    cq=cqs2, s10, thflux/qvflux        -> sfclay_post()
//
// Initialization (init_physics.F, isfcflx=1): sfclayinit() for sfcmodel=2,
// sf_sfclayrev_init() for sfcmodel=3 build the stability-function tables
// (on the host, with the same libm calls as the Fortran, then copied to the
// device).
//
// The SLAB land model (turb.F 1019, called when update_sfc, i.e. every step
// after the first, unless testcase=11) is physics/slab; the whole sequence
// for the time step is sfclay::wrf_surface_layer (cm1k_sfclay_driver.hpp).
// Not ported: sfcmodel=4,6,7, shalwater_z0 (CM1 passes .false.), scm_force_flux
// (.false.), oceanmodel=2.
//
// Known Fortran quirk (sfcmodel=2): SFCLAY1D writes its local scalars Z0T
// and Z0Q to z0t(i)/z0q(i) at every point, but sets them only over water
// (Z0Q also when isftcflx=1). Over land the Fortran stores whatever the
// scalar held from an earlier point (or stack garbage); the port leaves
// z0t/z0q unchanged there.

#include "physics/sfclay/cm1k_wrf_common.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k::sfclay {

using wrfphys::WrfConst;

struct Options {
  int sfcmodel = 2;                 // 2 = SFCLAY (MM5), 3 = sf_sfclayrev
  int isftcflx = 0, iz0tlnd = 0;    // namelist param12 (defaults of run/namelist.input)
  int isfflx = 1;                   // turb.F: isfflx = 1
  static Options from_namelist(const Namelist& nl);
  void check() const;               // throws for anything not ported
};

// Stability-function tables (module SAVE arrays in the Fortran), index 0..1000.
struct Tables {
  // sfcmodel=2 (sfclayinit)
  Kokkos::View<Real*, Device> psimtb, psihtb;
  // sfcmodel=3 (sf_sfclayrev_init)
  Kokkos::View<Real*, Device> psim_stab, psim_unstab, psih_stab, psih_unstab;
};
Tables sfclayinit();          // fills psimtb, psihtb
Tables sf_sfclayrev_init();   // fills psim_stab, psim_unstab, psih_stab, psih_unstab

// The 2D surface arrays the WRF surface layers read and write (cm1.F names;
// FArray3 (ib:ie,jb:je,1:1) like physics/sfc's SfcState, indexed a(i,j,1)).
// In the calls: psfc, tsk, xland, mavail, hpbl (PBLH, from the previous
// step), dsxy (DX). ust, znt, mol, qsfc, hfx, qfx (and the rest) are inout
// as in the Fortran.
struct SfcLayerState {
  FArray3 psfc, chs, chs2, cqs2, cpmm, znt, ust, hpbl, mavail, zol, mol, regime, psim, psih, fm, fh,
          xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, rmol, u10, v10, th2, t2, q2, gz1oz0, wspd, br,
          dsxy, ustm, ck, cka, cd, cda, z0t, z0q,
          // turb.F's own arrays, set by sfclay_post
          ch, cq, s10, thflux, qvflux;
  void allocate(const Bounds& b);
};

// turb.F 732-751, sfcmodel=2:
//   call SFCLAY(dum1,dum2,dum4,divx,prs,dum5, CP,G,ROVCP,RD,XLV,lv1,lv2,PSFC,
//               ..., rf(ib,jb,1), ..., ustm,ck,cka,cd,cda,isftcflx,iz0tlnd,z0t,z0q)
// u3d=dum1, v3d=dum2, t3d=dum4, qv3d=divx, p3d=prs, dz8w=dum5 (only k=1 is
// read), rf = rho on w levels (rf(:,:,1) is SFCLAY's rhocm1). Interior
// points 1..ni x 1..nj.
void sfclay(const Options& o, const WrfConst& c, const Tables& t, int ni, int nj, const FArray3& u3d,
            const FArray3& v3d, const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d,
            const FArray3& dz8w, const FArray3& rf, const SfcLayerState& s);

// turb.F 753-839, sfcmodel=3: the sfclayrev_j_loop calling
// sf_sfclayrev_wrapper (isfflx=.true., shalwater_z0=.false., water_depth=0,
// scm_force_flux=.false., dx=dsxy). Same array arguments as sfclay (k=1
// only); rf is not used.
void sfclayrev(const Options& o, const WrfConst& c, const Tables& t, int ni, int nj, const FArray3& u3d,
               const FArray3& v3d, const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d,
               const FArray3& dz8w, const SfcLayerState& s);

// turb.F after the surface layer, sfcmodel=2/3:
//   1007-1018  update_sfc = .false.: qsfc = qv(k=1) + qfx/flqc (or qv where
//              flqc=0); skipped with diag_qsfc = false (update_sfc: SLAB,
//              physics/slab, has updated tsk, qsfc, hfx, qfx instead)
//   1039-1048  ch = chs2, cq = cqs2, s10 = sqrt(u10**2+v10**2)
//   1101-1110  if (dosfcflx .or. output_sfcflx.eq.1):
//              thflux = hfx/(cpmm*rf(1)), qvflux = qfx/rf(1)
// qv3d = divx.
void sfclay_post(int ni, int nj, bool set_fluxes, const FArray3& qv3d, const FArray3& rf,
                 const SfcLayerState& s, bool diag_qsfc = true);

} // namespace cm1k::sfclay

#endif
