#ifndef CM1K_GFDL_DRIVER_HPP
#define CM1K_GFDL_DRIVER_HPP

// The GFDL surface layer (sfcmodel = 4) as one step of the time loop: the
// sfcmodel=4 part of src/turb.F sfc_and_turb's bbc=3 block,
//
//   turb.F 841-876   cd_out = ch_out = mol = zol = 0, zkmax = zh(:,:,1),
//                    SF_GFDL (physics/sfc_gfdl), then u* = max(u*, 1e-8),
//                    wspd = swspd
//   turb.F 1016-1048 update_sfc .and. testcase/=11: SLAB (physics/slab,
//                    slab::Land); else the qsfc diagnosis
//   turb.F 1062-1077 cd = cd_out, ch = cq = ch_out, s10 = |(u10,v10)|,
//                    rmol = 1/mol (+-1e20 where |mol| <= smeps)
//   turb.F 1101-1110 thflux/qvflux from hfx/qfx (dosfcflx or output_sfcflx)
//
// and, after the surface boundary conditions (turb::sfc_bcs), turb.F
// 1166-1230: bc2d(mznt), zntmp = mznt (gfdl_bcs).
//
// The state: SF_GFDL's 2D arrays (sfc_gfdl::Fields; tsk, znt, ust, u10,
// v10, hfx, qfx, qsfc, gz1oz0, br are sfc::SfcState's views, one array in
// cm1.F), the CM1 arrays only it uses (mznt, swspd, zkmax, cd_out, ch_out,
// taux, tauy, rmol, smois = 0), its tables, and SLAB's (slab::Land).
//
// Over land SF_GFDL reads smois out of bounds (cm1k_sfc_gfdl.hpp), so
// sfc::Options::check allows sfcmodel=4 over water (xland0 = 2) only.

#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/sfc_gfdl/cm1k_sfc_gfdl.hpp"
#include "physics/slab/cm1k_slab_land.hpp"
#include "physics/turb/cm1k_turb.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::sfc_gfdl {

struct GfdlSurface {
  Options o;           // turb.F's settings (ntsflg = 1, icoef_sf = 6, ...)
  Tables tab;
  slab::Land land;
  Fields f;            // f.wspd is CM1's swspd
  FArray3 psfc, xland, mavail, hfx, qfx, cd, ch, cq, s10, thflux, qvflux, wspd;   // sfc::SfcState's
  FArray3 rmol, zkmax, smois;

  // Allocates the arrays (sharing those of `common`; zero, as cm1.F
  // allocates them), builds the tables and SLAB's land properties.
  void setup(const sfc::Options& so, const Bounds& b, const sfc::SfcState& common);
  // init_surface.F: SLAB's initial soil temperatures (after sfc::init_surface)
  void init_soil(const Bounds& b) const;
  // The arrays src/cm1dump_state.inc writes for sfcmodel=4 (in its order):
  // mznt, mol, rmol, zol, br, gz1oz0, psim, psih.
  std::vector<std::pair<std::string, FArray3>> dumped() const;
};

// One pass of the surface layer (see above). dum1, dum2, dum4, divx:
// wrfphys::pbl_prep / pbl_moist arrays (k = 1 is read); prs; rf (rf(:,:,1)
// for the fluxes); zh (zkmax). dt: the time step. smeps: cm1.F's smeps.
void gfdl_surface_layer(const sfc::Options& so, const WrfConst& c, const FArray3& zh, int ni, int nj, Real dt,
                        Real smeps, bool update_sfc, const FArray3& dum1, const FArray3& dum2, const FArray3& dum4,
                        const FArray3& divx, const FArray3& prs, const FArray3& rf, const GfdlSurface& w);

// turb.F's sfcmodel=4 boundary conditions, after turb::sfc_bcs: bc2d(mznt),
// then zntmp = mznt on 0..ni+1, 0..nj+1 (in place of zntmp = znt).
void gfdl_bcs(const turb::Options& to, const Bounds& b, const GfdlSurface& w, const FArray3& zntmp);

} // namespace cm1k::sfc_gfdl

#endif
