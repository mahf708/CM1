#ifndef CM1K_RRTMG_LW_HPP
#define CM1K_RRTMG_LW_HPP

// RRTMG_LWRAD (src/module_ra_rrtmg_lw.F) for radopt = 2, with the arguments
// src/radiation_driver.F passes. 3-D arrays are (ib:ie,jb:je,kb:ke)
// FArray3's indexed (i,j,k) like the Fortran's (ims:ime,jms:jme,kms:kme);
// 2-D arrays are FArray3 (ib:ie,jb:je,1:1) indexed (i,j,1).
//
// Supported as CM1 calls it: has_reqc = has_reqi = has_reqs = 0 (inflglw = 2,
// iceflglw = 3, liqflglw = 1), is_CAMMGMP_used = .false., no F_ICE_PHY, no
// aerosols (WRF_CHEM), calc_clean_atm_diag = 0, no FER_MP_HIRES, mcica with
// the KISS generator (irng = 0, permuteseed = 150), cldovrlp = 0..4.
// Anything else throws.

#include "physics/rrtmg/cm1k_rrtmg_lw_tables.hpp"
#include "share/cm1k_farray.hpp"

namespace cm1k::rrtmg {

struct LwRadArgs {
  // out
  FArray3 rthratenlw, lwtenc;                                  // K/s (temperature)
  FArray3 lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc, glw, olr, lwcf;  // 2-D
  // in
  FArray3 emiss, tsk, xland, xice, snow;                       // 2-D
  FArray3 p8w, p3d, pi3d, dz8w, t3d, t8w, rho3d, cldfra3d;     // 3-D (p8w, t8w to k = kte+1)
  FArray3 qv3d, qc3d, qr3d, qi3d, qs3d, qg3d;
  FArray3 o33d;                                                // inout (written when o3input = 0)
  Real r = 0, g = 0;
  int icloud = 1, cldovrlp = 2, o3input = 0;
  bool warm_rain = false, f_qv = true, f_qc = false, f_qr = false, f_qi = false, f_qs = false, f_qg = false;
  int has_reqc = 0, has_reqi = 0, has_reqs = 0;
  bool is_cammgmp_used = false;
  int calc_clean_atm_diag = 0;
  int yr = 0;
  Real julian = 0;
  int mp_physics = 1;
  int its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = 1;
  int nlayers = 0;   // module_ra_rrtmg_lw's nlayers (lw_nlayers)
  int chunk = 0;     // columns per kernel launch (0: automatic)
};

// rrtmg_lwinit's nlayers = kme + nint(p_top*0.01/deltap) - 1.
int lw_nlayers(int kme, Real p_top);

// Runs RRTMG_LWRAD; returns co2out.
Real rrtmg_lwrad(const LwTableData& t, const LwRadArgs& a);

} // namespace cm1k::rrtmg

#endif
