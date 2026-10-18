#ifndef CM1K_SLAB_HPP
#define CM1K_SLAB_HPP

// Port of the MM5/WRF SLAB land-surface model, src/slab.F (SLAB, SLAB1D),
// as CM1 calls it from src/turb.F (sfc_and_turb, lines 1019-1031) with the
// WRF surface layers (sfcmodel = 2, 3) when update_sfc is true (every pass
// of the time loop except the startup pass, cm1.F:3744), and the soil-layer
// setup of src/init_surface.F (slab_zs, slab_dzs, initial tslb).
//
// SLAB predicts the skin temperature tsk over land from the residual of the
// surface energy budget, with heat diffusion through num_soil_layers soil
// layers (CM1: 5, so the multi-layer branch with the deepest layer fixed),
// and recomputes the land hfx, qfx, lh from the surface layer's exchange
// coefficients flhc, flqc and the old tsk. Over water (xland > 1.5) only
// capg and the qsfc/chklowq diagnosis at the end apply.
//
// Every (i,j) point is independent: SLAB1D's row test (XLDCOL, "all points
// of the row are water") only skips points that are skipped anyway, so the
// port is one parallel_for over 1..ni x 1..nj running the Fortran's loops
// for one point in order (bit-for-bit with the double-precision Fortran).
//
// CM1's call fixes: ifsnow = 0, radiation = .true. (RADSWTCH = 1: the
// upward long-wave flux stbolt*tg0**4 is always on), gsw = glw = 0 without
// radiation (turb.F ~416), snowc = 0, dtmin = dt/60 (unused by SLAB1D).
// The snow branches (ifsnow = 1) are ported too.

#include "physics/sfclay/cm1k_wrf_common.hpp"
#include "share/cm1k_grid.hpp"

namespace cm1k::slab {

using wrfphys::WrfConst;

constexpr int max_soil_layers = 10;

// Soil layer depths (init_surface.F): slab_dzs(1) = 0.01 m, doubling with
// depth; slab_zs = layer centres. num_soil_layers = 5 in cm1.F.
struct SoilLayers {
  int n = 0;
  Real zs[max_soil_layers] = {}, dzs[max_soil_layers] = {};   // zs[l-1] = slab_zs(l)
  static SoilLayers make(int num_soil_layers);
};

// init_surface.F (sfcmodel = 2,3,4,6,7 with num_soil_layers /= 1):
// tslb(i,j,l) = (tsk*(zs(n)-zs(l)) + tmn*(zs(l)-zs(1))) / (zs(n)-zs(1)) on
// the whole (ib:ie,jb:je) plane. tslb is (ib:ie,jb:je,1:n); tsk, tmn are
// FArray3 (ib:ie,jb:je,1:1).
void init_tslb(const SoilLayers& sl, const Bounds& b, const FArray3& tsk, const FArray3& tmn,
               const FArray3& tslb);

// The 2D arrays SLAB reads and writes (FArray3 (ib:ie,jb:je,1:1), indexed
// a(i,j,1)); tslb is (ib:ie,jb:je,1:num_soil_layers).
struct Arrays {
  FArray3 flhc, flqc, psfc, xland, tmn, hfx, qfx, lh, tsk, qsfc, chklowq, gsw, glw, capg, thc, snowc,
          emiss, mavail, tslb;
};

// turb.F 1021: call SLAB(dum4,divx,prs,FLHC,FLQC,PSFC,XLAND,TMN,HFX,QFX,LH,
// TSK,QSFC,CHKLOWQ,GSW,GLW,CAPG,THC,SNOWC,EMISS,MAVAIL,DT,ROVCP,XLV,lv1,
// lv2,DTMIN,IFSNOW,SVP1,SVP2,SVP3,SVPT0,EP2,KARMAN,EOMEG,STBOLT,TSLB,
// slab_ZS,slab_DZS,num_soil_layers,.true.,P1000mb, ...)
// t3d = T (dum4), qv3d = qv (divx), p3d = prs: only k = 1 is read.
void slab(const WrfConst& c, Real deltsm, int ifsnow, bool radiation, const SoilLayers& sl, int ni, int nj,
          const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d, const Arrays& a);

} // namespace cm1k::slab

#endif
