#ifndef CM1K_WRF_COMMON_HPP
#define CM1K_WRF_COMMON_HPP

// What src/turb.F (sfc_and_turb) prepares before it calls the WRF/MMM
// physics (surface layers sfcmodel=2..7, PBL schemes ipbl=1,3..6): the
// constants it passes, and the "dum" arrays it fills with u, v at scalar
// points, T, Exner, layer depths and pressure on w levels.
//
// Used by physics/sfclay (sfcmodel=2,3) and physics/pbl_ysu (ipbl=1).

#include "share/cm1k_constants.hpp"
#include "share/cm1k_farray.hpp"

namespace cm1k::wrfphys {

// Constants as turb.F passes them to the WRF physics (the "variables for wrf
// physics" block, ~line 380, plus constants.F).
struct WrfConst {
  Real cp, g, rovcp, rd, rv, xlv, lv1, lv2;
  Real svp1, svp2, svp3, svpt0;
  Real ep1, ep2, rovg, karman, eomeg, stbolt, p1000mb;

  static WrfConst make(const Constants& c) {
    WrfConst w{};
    w.cp = c.cp; w.g = c.g; w.rovcp = c.rovcp; w.rd = c.rd; w.rv = c.rv; w.xlv = c.xlv;
    w.lv1 = c.lv1; w.lv2 = c.lv2;
    w.svp1 = c.svp1; w.svp2 = c.svp2; w.svp3 = c.svp3; w.svpt0 = c.svpt0;
    w.p1000mb = 100000.;
    w.eomeg = 7.2921E-5;
    w.stbolt = 5.67051E-8;
    w.ep1 = c.rv / c.rd - 1.0;
    w.ep2 = c.rd / c.rv;
    w.rovg = c.rd / c.g;
    w.karman = phys::karman;
    return w;
  }
};

// The arrays turb.F fills for the PBL scheme (sfc_and_turb, "Variables for
// PBL and/or CuParam", lines ~1828-1849), on 1..ni x 1..nj x 1..nk (dum6 on
// 1..nk+1). The surface layers read the k=1 values (and dum5 at k=1,2),
// which the earlier block at line ~343 computes with the same formulas.
//   dum1 = u (ground-relative, pertflx) at scalar points
//   dum2 = v (ground-relative, pertflx) at scalar points
//   dum3 = th = th0+tha
//   dum4 = T = th*pi
//   dum5 = dz8w = dz*rmh (layer depth at scalar levels)
//   dum6 = p8w: pressure at w levels; dum6(:,:,1) = psfc, dum6(:,:,nk+1) by
//          cgt1..3 extrapolation, c1/c2 interpolation in between
//   dum7 = pi = pi0+ppi
// imove = 0 (ugr = ua, or ua-u0 with pertflx=1). All arrays (ib:ie,jb:je,kb:ke)
// except psfc (ib:ie,jb:je,1). Needs nk >= 3.
struct PrepIn {
  int ni = 0, nj = 0, nk = 0;
  int pertflx = 0;
  Real dz = 0, cgt1 = 0, cgt2 = 0, cgt3 = 0;
  FArray3 ua, va, u0, v0, th0, tha, pi0, ppi, prs, rmh, c1, c2, psfc;
};
void pbl_prep(const PrepIn& in, const FArray3& dum1, const FArray3& dum2, const FArray3& dum3,
              const FArray3& dum4, const FArray3& dum5, const FArray3& dum6, const FArray3& dum7);

// Moisture fields turb.F hands to the WRF physics, on 1..ni x 1..nj x 1..nk:
//   divx  = qv  (qa(nqv) if imoist=1, else 0)         (line ~430)
//   ppten = qc  (qa(nqc) if nqc>=1, else 0)           (get_vars, line ~1785)
//   dum8  = qi  (qa(nqi) if nqi>=1, else 0)           (get_vars, line ~1768)
// nqv/nqc/nqi are CM1's 1-based indices into qa(:,:,:,n) (0 = absent).
void pbl_moist(int ni, int nj, int nk, int imoist, int nqv, int nqc, int nqi, const FArray4& qa,
               const FArray3& divx, const FArray3& ppten, const FArray3& dum8);

} // namespace cm1k::wrfphys

#endif
