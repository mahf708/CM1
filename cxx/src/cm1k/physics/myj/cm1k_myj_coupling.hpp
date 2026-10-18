#ifndef CM1K_MYJ_COUPLING_HPP
#define CM1K_MYJ_COUPLING_HPP

// The glue that runs the MYJ surface layer (sfcmodel=7) and the MYJ PBL
// (ipbl=6) inside the full model (cm1setup=2), as src/turb.F sfc_and_turb
// does (kernels: physics/myj/cm1k_myj.hpp):
//
//   surface_layer()   turb.F 932-1013: MYJSFC (ITIMESTEP = nstep, ZNT =
//                     zntmyj, z0out = znt, usemyjpbl, rhosfc = rf(k=1),
//                     isftcflx) on the turb.F tmp2d packing, which is the
//                     wrfphys::pbl_prep / pbl_moist arrays (same formulas:
//                     dz = dum5, pint = dum6, th = dum3, t = dum4, u = dum1,
//                     v = dum2, qv = divx, qc = ppten); br clipped to
//                     [-10, 10], gz1oz0 = log(zh(k=1)/znt); then SLAB
//                     (update_sfc .and. testcase /= 11: physics/slab, land
//                     and water; the startup pass, update_sfc = .false.,
//                     does nothing for sfcmodel=7); ch = chs2, cq = cqs2,
//                     s10; the sfcmodel=7 block (ustm, wspd, mol, cd);
//                     thflux/qvflux (dosfcflx is .true. for sfcmodel=7 with
//                     ipbl=6).
//   pbl()             turb.F 2567-2640 (domyj): MYJPBL on the get_vars
//                     arrays (the same pbl_prep / pbl_moist arrays; QCI =
//                     dum8), ELFLX = lh, CHKLOWQ (0 before the first SLAB
//                     call, then mavail), SICE = SNOW = 0; tke_myj, el_myj,
//                     xkzh(k+1) = EXCH_H(k), xkzm(k+1) = EXCH_M(k) (0 at
//                     k = nk+1), the u/v/theta/qv/qc/qi tendencies to
//                     pbl::Tendencies, hpbl, kpbl2d, mixht, ct.
//
// MYJ has no TKE advection in solve2 and no dissipative heating of its own
// (idiss=1 is turb::dissheat with dissten from turbparam_horiz, as for YSU
// without ysudiss).
//
// Initialization (cm1.F, init_physics.F MYJSFCINIT, init_surface.F; see
// init()): tke_myj = 0.01, el_myj = 0, zntmyj = z0base = 0, ... at
// allocation; ust = 0.1 on 1..ni x 1..nj (MYJSFCINIT); z0base = znt on
// the whole plane (init_surface.F); tslb from tsk and tmn.
//
// Over land, CM1 never sets zntmyj (it stays 0, the commented-out lines of
// MYJSFC would have set it), so MYJSFC's land roughness is zero: fm, and
// with it wspd, are infinite and the Fortran run is NaN within the first
// steps. The code path is ported as written (and tested with positive and
// zero roughness in tests/fortran_bridge/myj_tests.cpp), but the
// full-model cases run over water.
//
// Arrays (cm1.F names): the MYJ arrays CM1 shares with other schemes (tsk,
// znt, ust, hpbl, u10, ...) are the sfc::SfcState views; the PBL
// tendencies are pbl::Tendencies'.

#include "physics/myj/cm1k_myj.hpp"
#include "physics/pbl/cm1k_pbl.hpp"
#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/sfclay/cm1k_wrf_common.hpp"
#include "physics/slab/cm1k_slab.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::myj {

// The MYJ path's options (namelist, param.F's fixed values).
struct Coupling {
  int cm1setup = 1, ipbl = 0, sfcmodel = 0, isftcflx = 0, testcase = 0;
  int output_sfcflx = 0;
  bool dosfcflx = false;   // init_surface.F (.true. for sfcmodel=7 with ipbl=6)

  // param.F idopbl (cm1setup = 2, 4) with ipbl = 6
  bool pbl_active() const { return (cm1setup == 2 || cm1setup == 4) && ipbl == 6; }
  bool sfc_active() const { return sfcmodel == 7; }
  bool active() const { return pbl_active() || sfc_active(); }
  static Coupling from_namelist(const Namelist& nl);
  // Throws for what is not ported: ipbl=6 without sfcmodel=7 (param.F
  // stops), sfcmodel=7 without ipbl=6 (usemyjpbl = .false.: not ported),
  // cm1setup other than 2.
  void check() const;
};

// Carried arrays: persist across steps (the Model owns them; the dumped
// ones are registered fields). 2D arrays are (ib:ie,jb:je,1:1).
struct Carried {
  // MYJSFC state and outputs
  FArray3 mol, rmol, thz0, qz0, uz0, vz0, akhs, akms, zntmyj, z0base, th10, q10, u10e, v10e;
  FArray3 tshltr, qshltr, pshltr, chs, chs2, cqs2, cpmm, flhc, flqc, qgh, lh, psim, psih, fm, fh, ct, ustm;
  // SLAB (as sfclay::WrfSurface): tslb (ib:ie,jb:je,1:num_soil_layers)
  slab::SoilLayers soil;
  FArray3 tslb, tmn, emiss, thc, capg, chklowq, gsw, glw, snowc;
  // MYJPBL: tke_myj, el_myj (ib:ie,jb:je,kb:ke+1), xkzh, xkzm
  // (ib:ie,jb:je,kb:ke), mixht, kpbl2d (2D)
  FArray3 tke_myj, el_myj, xkzh, xkzm, mixht, kpbl2d;

  // Allocates the arrays with cm1.F's initial values (tke_myj = 0.01, the
  // rest 0) and the constant SLAB land properties (init_surface.F,
  // initsfc=1: tmn = tmn0, emiss and thc from LANDUSE.TBL for lu0).
  void allocate(const Bounds& b, const sfc::Options& so, int num_soil_layers = 5);
  // (name, array) as src/cm1dump_state.inc writes them: the sfcmodel=7
  // block, and the ipbl=6 block without hpbl and the pbl::Tendencies
  std::vector<std::pair<std::string, FArray3>> dumped_sfc() const;
  std::vector<std::pair<std::string, FArray3>> dumped_pbl() const;
};

// MYJSFCINIT's ust = 0.1 (1..ni x 1..nj), init_surface.F's z0base = znt and
// tslb (call after sfc::init_surface).
void init(const Bounds& b, const sfc::SfcState& s, const Carried& cr);

// Work arrays and tables (the Dyncore owns them; not carried).
struct Work {
  Tables tab;        // init_physics.F: MYJSFCINIT
  Workspace ws;
  FArray3 ht;        // zs = 0 (no terrain)
  FArray3 exch_h, exch_m;
  void allocate(const Bounds& b);
};

// sfc_and_turb's surface part for sfcmodel=7 (see above), after
// sfc::sfc_prelude (psfc, u1, v1, s1, t1) and the pbl_prep / pbl_moist
// arrays. update_sfc: cm1.F's (nstep > 0). dt: SLAB's deltsm.
struct WrfArrays {
  FArray3 d1, d2, d3, d4, d5, d6, d7, qv, qc, qi;
};
void surface_layer(const Coupling& o, const wrfphys::WrfConst& c, const Grid& g, Real dt, int nstep,
                   bool update_sfc, const WrfArrays& wa, const FArray3& prs, const FArray3& rf,
                   const sfc::SfcState& s, const Carried& cr, const Work& w);

// sfc_and_turb's domyj block (after turb::sfc_bcs ... gettau).
void pbl(const Coupling& o, const Bounds& b, Real dt, const WrfArrays& wa, const FArray3& prs,
         const sfc::SfcState& s, const pbl::Tendencies& pt, const Carried& cr, const Work& w);

} // namespace cm1k::myj

#endif
