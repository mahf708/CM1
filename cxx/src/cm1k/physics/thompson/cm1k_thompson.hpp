#ifndef CM1K_THOMPSON_HPP
#define CM1K_THOMPSON_HPP

#include "cm1k_thompson_tables.hpp"

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

namespace cm1k::thompson {

// Port of the Thompson microphysics (src/thompson.F: mp_gt_driver,
// mp_thompson; thompson_init in cm1k_thompson_tables) and of the ptype=3
// branch of src/mp_driver.F.
//
// Moisture layout (param.F, ptype=3): q3d(:,:,:,n) = qv, qc, qr, qi, qs, qg,
// nci, ncr (numq = 8). CM1 calls mp_gt_driver without the aerosol arrays, so
// the scheme is not aerosol aware: cloud droplet number is the constant
// Nt_c = 1e6*nt_c, and the aerosol bookkeeping (nwfa/nifa, Eff_aero,
// iceDeMott, iceKoop, activ_ncloud), which then never feeds back into the
// state, is not ported. Nothing in the scheme is random.
//
// Not ported (diagnostics only, never feeding back): budgets (qbudget, asq),
// tdiag/qdiag, the radar reflectivity (calc_refl10cm, getdbz) and the
// effective radii for radiation (calc_effectRad, radopt=2 throws).
constexpr int nqv = 1, nqc = 2, nqr = 3, nqi = 4, nqs = 5, nqg = 6, nnci = 7, nncr = 8;

struct Options {
  int eqtset = 2;          // 1: traditional, 2: mass/energy conserving (param.F clamps to [1,2])
  double nt_c = 100.0;     // namelist nt_c (1/cm^3)
  Real qsmall = 1.0e-24;   // cm1.F, double precision build
  Real tsmall = 1.0e-12;

  // Throws unless imoist=1 and ptype=3; radopt=2 (effective radii) is not ported.
  static Options from_namelist(const Namelist& nl);
};

// Scratch arrays (cm1.F names in comments); allocate() also builds (or loads)
// the lookup tables.
struct Work {
  FArray3 tmp;      // dum1: temperature in, out
  FArray3 thten;    // thten: copy of the temperature before the scheme
  FArray3 qvold;    // qten(:,:,:,nqv): copy of qv before the scheme
  FArray3 dzq;      // dum4: dz*rmh
  FArray3 vtc, vtr, vti, vts, vtg;   // fall speeds, mass weighted (qten(...,nqc..nqg) when getvt)
  FArray2 prate;    // surface precipitation rate (kg/m2/s)
  Kokkos::View<double*, Device> scr;   // column scratch: ncol x NVAR x (nk+2)
  Kokkos::View<int*, Device> iscr;     // column scratch, logicals
  DevTables tab;
  void allocate(const Bounds& b, const Options& opt);
};

//---------------------------------------------------------------------------
// mp_driver for ptype=3 (cm1.F between solve2 and solve3):
//   T = (th0+th3d)*(pi0+pp3d); thten = T; qvold = qv
//   pdefq(0, qv); pdefq(qsmall, qc, qr, qi, qs, qg)
//   mp_gt_driver(T, prs, w3d, dz*rmh)  -> q3d(1..8), T, rain, prate, vt*
//   eqtset=2: prs, pp3d, th3d from T where T or qv changed
//   eqtset=1: th3d, rho
// Inputs/outputs as in kessler::mp_driver; w3d = vertical velocity (ke+1 levels).
//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real dz,
               const FArray3& rmh, const FArray3& pi0, const FArray3& th0, const FArray3& w3d,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w);

void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& w3d, const FArray3& rho, const FArray3& prs, const FArray3& pp3d,
               const FArray3& th3d, const FArray4& q3d, const FArray3& rain, const Work& w);

// thompson.F mp_gt_driver (interior columns 1..ni x 1..nj, levels 1..nk):
// q3d(:,:,:,1..8) and t3d updated in place; rain(:,:,1..nrain) += 0.1*ppt,
// prate = ppt/dt; vt* filled on 1..nk.
void mp_gt_driver(const Options& opt, const Bounds& b, Real dt, const FArray3& t3d, const FArray3& p,
                  const FArray3& w, const FArray3& dz, const FArray4& q3d, const FArray3& rain,
                  const Work& wk);

} // namespace cm1k::thompson

#endif
