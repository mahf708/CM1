#ifndef CM1K_RRTMG_SW_HPP
#define CM1K_RRTMG_SW_HPP

// RRTMG_SWRAD (src/module_ra_rrtmg_sw.F) for radopt = 2, with the arguments
// src/radiation_driver.F passes. Array conventions as cm1k_rrtmg_lw.hpp:
// 3-D arrays (i,j,k), 2-D arrays FArray3 (ib:ie,jb:je,1:1) indexed (i,j,1).
//
// Supported as CM1 calls it: has_reqc = has_reqi = has_reqs = 0 (inflgsw = 2,
// iceflgsw = 3, liqflgsw = 1), is_CAMMGMP_used = .false., no F_ICE_PHY,
// no progn / QNDROP, aer_opt = 0 (no aerod, tauaer3d_sw not associated),
// sf_surface_physics /= 8, calc_clean_atm_diag = 0, no FER_MP_HIRES, mcica
// with the KISS generator (irng = 0, permuteseed = 1), cldovrlp = 0..4.
// Anything else throws.
//
// As the Fortran, a column with xcoszen <= 0 zeroes rthratensw and the 2-D
// diagnostics but leaves gsw and swtenc untouched.

#include "physics/rrtmg/cm1k_rrtmg_sw_tables.hpp"
#include "share/cm1k_farray.hpp"

namespace cm1k::rrtmg {

struct SwRadArgs {
  // out
  FArray3 rthratensw, swtenc;                                  // K/s (temperature)
  FArray3 swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc, swcf, gsw, coszr;  // 2-D
  FArray3 swddir, swddni, swddif, swdownc, swddnic, swddirc;   // 2-D
  // in
  FArray3 xcoszen, albedo, tsk, xland, xice, snow;             // 2-D
  FArray3 t3d, t8w, p3d, p8w, pi3d, rho3d, dz8w, cldfra3d;     // 3-D (p8w, t8w to k = kte+1)
  FArray3 qv3d, qc3d, qr3d, qi3d, qs3d, qg3d;
  FArray3 o33d;
  Real r = 0, g = 0, solcon = 0;
  int icloud = 1, cldovrlp = 2, o3input = 0;
  bool warm_rain = false, f_qv = true, f_qc = false, f_qr = false, f_qi = false, f_qs = false, f_qg = false;
  int has_reqc = 0, has_reqi = 0, has_reqs = 0;
  bool is_cammgmp_used = false;
  int calc_clean_atm_diag = 0;
  int aer_opt = 0, sf_surface_physics = 1;
  int yr = 0;
  int mp_physics = 1;
  int its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = 1;
  int chunk = 0;     // columns per kernel launch (0: automatic)
};

// Runs RRTMG_SWRAD; returns co2out.
Real rrtmg_swrad(const SwTableData& t, const SwRadArgs& a);

} // namespace cm1k::rrtmg

#endif
