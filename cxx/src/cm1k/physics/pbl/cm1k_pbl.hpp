#ifndef CM1K_PBL_HPP
#define CM1K_PBL_HPP

// Port of CM1's boundary-layer path for the "mesoscale" setup, cm1setup=2:
//
//   src/turb.F      turbparam_vert (ipbl=2: Louis-type vertical eddy
//                   viscosity, length scale l_inf), turbparam_horiz
//                   (horizturb=1: horizontal Smagorinsky-like scheme, length
//                   scale l_h, or lhref1/lhref2 over water), and the dobr09
//                   block of sfc_and_turb (ipbl=2: the PBL tendencies from
//                   the implicit vertical diffusion of turbsz/turbuz/turbvz)
//   src/solve1.F    the use_pbl blocks (thten1 += thpten before the doturb
//                   block; uten1/vten1/qten += upten/vpten/qvpten/qcpten
//                   after it)
//   src/sfcphys.F   gethpbl2 is in physics/sfc (sfc::gethpbl2)
//
// The ipbl=2 scheme is Bryan and Rotunno (2009)'s simple PBL: K_v =
// l_v^2 sqrt(max(S_v^2 - N^2, 1e-10)) with 1/l_v^2 = 1/l_inf^2 +
// 1/(karman (z + z0))^2, and K_v(k=1) = karman z0 u* (bbc=3).
//
// Supported (Options::check throws for anything else):
//   * cm1setup = 2 with ipbl = 2 (param.F: sgsmodel = 0, tconfig = 2,
//     dovturb = .false., dohturb = (horizturb = 1), bbc = 3)
//   * horizturb = 0, 1; idiss = 0, 1; doimpl = 0, 1; bcturbs = 1, 2
//   * sfcmodel = 1 (physics/sfc, with isfcflx = 0/1)
//   * imoist = 0, or 1 with ptype = 1 (Kessler: qv, qc; no qi)
//   * no terrain (zs = 0), axisymm = 0
//
// Where each piece runs (Dyncore, cm1.F:3748 sfc_and_turb at the start of
// every Model::step, then solve1):
//   Dyncore::sfc_and_turb (cm1setup=2): dissten = 0; sfc::sfc_layer (with
//     gethpbl2); turb::sfc_bcs; turb::calcnm; turb::calcdef;
//     turbparam_horiz (horizturb=1); turbparam_vert; turb::gettau; then
//     Dyncore::pbl, the dispatch hook below: ipbl=2 -> br09, followed by
//     bcs(upten), bcs(vpten) (use_pbl).
//   Dyncore::solve1: turb::dissheat (idiss=1); add_theta_tendency;
//     turb::turb_tendencies (only with horizturb=1: dohturb); add_tendencies.
//
// Not ported: budgets (tdiag/udiag/vdiag/qdiag td_pbl...), the nstep=1 print
// of turbparam_vert, output_dissten, the other PBL schemes (YSU, GFS-EDMF,
// MYNN, MYJ: ipbl = 1, 3..6) and cm1setup = 4.

#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/turb/cm1k_turb.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::pbl {

// Options from the `input` module (namelist param1/param2/param7) with
// param.F's adjustments.
struct Options {
  int cm1setup = 1, ipbl = 0, horizturb = 0, sfcmodel = 0, idiss = 0;
  int imoist = 0, ptype = 0, nqv = 0, nqc = 0, nqi = 0;
  bool terrain_flag = false;
  int axisymm = 0;
  Real l_h = 100.0, lhref1 = 100.0, lhref2 = 1000.0, l_inf = 75.0;
  // param.F: (cm1setup = 2 or 4) and ipbl >= 1 -> boundary-layer tendency
  // arrays (thpten, ...) on the full grid
  bool use_pbl = false;

  static Options from_namelist(const Namelist& nl);
  void check() const;   // throws std::runtime_error for anything not ported
};

// The PBL tendencies sfc_and_turb hands to solve1 (cm1.F thpten, qvpten,
// qcpten, qipten, upten, vpten; (ibb:ieb,jbb:jeb,kbb:keb) = (ib:ie,jb:je,
// kb:ke) with use_pbl). They persist across time steps. Allocated by the
// Model when DynConfig::pbl_opt.use_pbl. A new scheme adds its outputs here.
struct Tendencies {
  FArray3 thpten, qvpten, qcpten, qipten, upten, vpten;
  void allocate(const Bounds& b);
  // (name, array) of the arrays src/cm1dump_state.inc writes for ipbl=2
  std::vector<std::pair<std::string, FArray3>> dumped() const;
};

// turb.F turbparam_horiz (cm1setup=2, horizturb=1): kmh (k=2..nk, bcw,
// k=1 extrapolated), khh = kmh/Pr, dissten += kmh^3/tlh^4 (idiss=1).
// Needs xland and psfc from the surface layer (tlh over water).
void turbparam_horiz(const turb::Options& to, const Options& o, const Bounds& b, const FArray3& defh,
                     const FArray3& kmh, const FArray3& khh, const FArray3& dissten,
                     const sfc::SfcState& s);

// turb.F turbparam_vert (ipbl=2): kmv (k=2..nk, bcw), khv = kmv/Pr,
// dissten += kmv^3/lvz^4 (idiss=1), and kmv = khv = karman*z0*u* at k=1
// (bbc=3; bc2d). lvz: scratch (ib:ie,jb:je,kb:ke), CM1's dum4.
void turbparam_vert(const turb::Options& to, const Options& o, const Grid& g, const FArray3& nm,
                    const FArray3& defv, const FArray3& kmv, const FArray3& khv, const FArray3& dissten,
                    const sfc::SfcState& s, const FArray3& lvz);

// turb.F sfc_and_turb, dobr09 block (ipbl=2): vimpl coefficients (doimpl=1)
// in w.d7/w.d8 (CM1's thten/thten1), sadv = th0-th0r+tha, then
// thpten/qvpten/qcpten from turbsz (surface fluxes thflux/qvflux with
// dosfcflx), upten/vpten from turbuz/turbvz with the t13/t23 of gettau.
// The tendencies are rho-weighted, as CM1 has them (turbsz/turbuz output).
// Scratch: w.d1..d3, d7, d8, sadv. The caller applies bcs(upten/vpten).
void br09(const turb::Options& to, const Options& o, const Grid& g, const BaseState& base, Real dt,
          const FArray3& ua, const FArray3& va, const FArray3& wa, const FArray3& tha, const FArray4& qa,
          const FArray3& rho, const FArray3& rr, const FArray3& rf, const FArray3& kmv,
          const FArray3& khv, const FArray3& t13, const FArray3& t23, const sfc::SfcState& s,
          const Tendencies& pt, const turb::Work& w);

// solve1.F, use_pbl: thten1(1:ni,1:nj,1:nk) += thpten (before doturb).
void add_theta_tendency(const Bounds& b, const Tendencies& pt, const FArray3& thten1);

// solve1.F, use_pbl, after doturb: uten1(1:ni+1) += 0.5*(upten(i-1)+upten(i)),
// vten1(1:nj+1) += 0.5*(vpten(j-1)+vpten(j)), and (imoist=1)
// qten(nqv) += qvpten, qten(nqc) += qcpten, qten(nqi) += qipten.
void add_tendencies(const Options& o, const Bounds& b, const Tendencies& pt, const FArray3& uten1,
                    const FArray3& vten1, const FArray4& qten);

} // namespace cm1k::pbl

#endif
