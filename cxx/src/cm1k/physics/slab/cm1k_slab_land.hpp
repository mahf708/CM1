#ifndef CM1K_SLAB_LAND_HPP
#define CM1K_SLAB_LAND_HPP

// SLAB's own state (cm1.F arrays) and its call from src/turb.F, for a
// surface layer that keeps its 2D arrays elsewhere (the MYNN surface layer,
// sfcmodel=6: physics/pbl_mynn/cm1k_mynn_coupling). The WRF/MM5 layers
// (sfcmodel=2,3) carry the same arrays in sfclay::WrfSurface.
//
//   setup()       cm1.F allocation (num_soil_layers = 5, everything 0) and
//                 init_surface.F (initsfc=1): tmn = tmn0, and getlanduse's
//                 emiss, thc from LANDUSE.TBL for lu_index = lu0
//   init_soil()   init_surface.F (sfcmodel = 2,3,4,6,7): the soil layers'
//                 initial temperatures tslb between tsk and tmn
//   run()         turb.F 1019-1031 (update_sfc .and. testcase /= 11):
//                 call SLAB(dum4, divx, prs, FLHC, FLQC, PSFC, XLAND, TMN,
//                 HFX, QFX, LH, TSK, QSFC, CHKLOWQ, GSW, GLW, CAPG, THC,
//                 SNOWC, EMISS, MAVAIL, DT, ..., IFSNOW = 0, ..., TSLB,
//                 slab_ZS, slab_DZS, num_soil_layers, .true., ...)
//   diag_qsfc()   turb.F 1032-1047 (the other branch, sfcmodel /= 7):
//                 qsfc = qv + qfx/flqc (qv where flqc = 0), chklowq = mavail
//
// gsw = glw = snowc = 0 (no radiation, no snow: cm1.F's zero allocation).

#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/slab/cm1k_slab.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::slab {

struct Land {
  SoilLayers soil;
  // tslb (ib:ie,jb:je,1:num_soil_layers); the rest (ib:ie,jb:je,1:1)
  FArray3 tslb, tmn, emiss, thc, capg, chklowq, gsw, glw, snowc;

  // so: tmn0, lu0 (1..33), season
  void setup(const sfc::Options& so, const Bounds& b, int num_soil_layers = 5);
  void init_soil(const Bounds& b, const FArray3& tsk) const;
  // (name, array) as src/cm1dump_state.inc writes them: tslb
  std::vector<std::pair<std::string, FArray3>> dumped() const;

  // The surface layer's 2D arrays SLAB reads and writes (cm1.F names).
  struct Surface {
    FArray3 flhc, flqc, psfc, xland, hfx, qfx, lh, tsk, qsfc, mavail;
  };
  // t3d = dum4 (T), qv3d = divx (qv), p3d = prs: k = 1 is read. dt = SLAB's deltsm.
  void run(const WrfConst& c, Real dt, int ni, int nj, const FArray3& t3d, const FArray3& qv3d,
           const FArray3& p3d, const Surface& s) const;
  void diag_qsfc(int ni, int nj, const FArray3& qv3d, const Surface& s) const;
};

} // namespace cm1k::slab

#endif
