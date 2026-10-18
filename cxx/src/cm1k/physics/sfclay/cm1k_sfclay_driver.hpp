#ifndef CM1K_SFCLAY_DRIVER_HPP
#define CM1K_SFCLAY_DRIVER_HPP

// The WRF/MM5 surface layers (sfcmodel = 2, 3) as one step of the time
// loop: the sfcmodel=2/3 part of src/turb.F sfc_and_turb's bbc=3 block,
//
//   turb.F 732-839   SFCLAY (sfcmodel=2) or sfclayrev_j_loop (sfcmodel=3)
//   turb.F 1016-1048 update_sfc .and. testcase/=11: SLAB (physics/slab:
//                    tsk, tslb, land hfx/qfx/lh, qsfc); else the qsfc
//                    diagnosis; then ch = chs2, cq = cqs2, s10
//   turb.F 1101-1110 thflux/qvflux from hfx/qfx (dosfcflx or output_sfcflx)
//
// with the state it carries: the surface-layer arrays (SfcLayerState, whose
// arrays that physics/sfc also has, tsk, znt, ust, hfx, ..., are the same
// views as sfc::SfcState's, as in cm1.F where they are one array), the
// stability tables (init_physics.F), and SLAB's soil temperatures and land
// properties (init_surface.F: tmn = tmn0, emiss, thc from LANDUSE.TBL,
// tslb interpolated between tsk and tmn).
//
// The caller (Dyncore::sfc_and_turb) runs sfc::sfc_prelude (psfc, u1, v1,
// s1, t1; gethpbl2 for ipbl=2) and wrfphys::pbl_prep / pbl_moist before
// wrf_surface_layer, and turb::sfc_bcs after it.
//
// update_sfc: cm1.F passes .false. in the startup pass of the time loop
// (nstep = 0, sfc_and_turb before the first solve) and .true. afterwards.

#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/sfclay/cm1k_sfclay.hpp"
#include "physics/slab/cm1k_slab.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::sfclay {

struct WrfSurface {
  SfcLayerState s;
  Tables tab;
  slab::SoilLayers soil;
  // SLAB's arrays (cm1.F names): tslb (ib:ie,jb:je,1:num_soil_layers), the
  // rest 2D
  FArray3 tslb, tmn, emiss, thc, capg, chklowq, gsw, glw, snowc;

  // Allocates the arrays (sharing those of `common`), builds the stability
  // tables for o.sfcmodel and sets the constant land properties tmn, emiss,
  // thc (init_surface.F, initsfc=1: tmn0 and lu0 everywhere). gsw = glw =
  // snowc = 0 (no radiation, no snow). num_soil_layers = 5 (cm1.F).
  void setup(const sfc::Options& so, const Options& o, const Bounds& b, const sfc::SfcState& common,
             int num_soil_layers = 5);
  // The arrays src/cm1dump_state.inc writes for sfcmodel=2,3 besides the
  // sfc::SfcState ones: mol (carried by the surface layer) and tslb.
  std::vector<std::pair<std::string, FArray3>> dumped() const;
};

// init_surface.F: the initial soil temperatures, from tsk and tmn (call after
// sfc::init_surface).
void init_soil(const Bounds& b, const WrfSurface& w);

// One pass of the surface layer (see above). dum1, dum2, dum4, dum5, divx:
// wrfphys::pbl_prep / pbl_moist arrays (k = 1 is read); prs, rf: pressure
// and density on w levels (rf(:,:,1) is SFCLAY's rhosfc). dt: the time step
// (SLAB's deltsm).
void wrf_surface_layer(const sfc::Options& so, const Options& o, const WrfConst& c, int ni, int nj, Real dt,
                       bool update_sfc, const FArray3& dum1, const FArray3& dum2, const FArray3& dum4,
                       const FArray3& dum5, const FArray3& divx, const FArray3& prs, const FArray3& rf,
                       const WrfSurface& w);

} // namespace cm1k::sfclay

#endif
