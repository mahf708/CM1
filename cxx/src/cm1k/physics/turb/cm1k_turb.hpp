#ifndef CM1K_TURB_HPP
#define CM1K_TURB_HPP

// Port of CM1's LES subgrid turbulence for cm1setup=1, sgsmodel=1
// (prognostic-TKE Deardorff scheme):
//
//   src/turb.F      sfc_and_turb (the LES/sgsmodel=1 path), calcnm, calcdef,
//                   tkekm, tkebc, gettau, sfcstress (bbc=1)
//   src/turbtend.F  turbs/turbsz, turbt, turbu/turbuz, turbv/turbvz, turbw
//   src/solve1.F    the TKE source terms (shear/buoyancy production,
//                   dissipation), the vimpl coefficient arrays for turbs,
//                   the theta-th0r trick for turbs(theta), and the
//                   dissipative heating (idiss=1)
//   src/solve2.F/solve3.F  the TKE update tke3d = tkea + dt*tten and the
//                   copy tkea = tke3d (helpers below; TKE advection itself is
//                   Advection::advw with the scalar orders, see cm1k_adv.hpp)
//
// Supported (Options::check throws for anything else):
//   * cm1setup=1, sgsmodel=1, ipbl=0, horizturb=0, cuparam=0
//   * tconfig=1,2; bcturbs=1,2; doimpl=0,1; idiss=0,1; ramp_sgs=0,1
//   * no surface fluxes (isfcflx=0, sfcmodel=0 -> dosfcflx=.false.)
//   * bbc=1, tbc=1 (free-slip lower/upper boundaries)
//   * lateral boundaries wbc..nbc = 1 (periodic), 2 (open), 3, 4 (walls)
//   * no terrain, axisymm=0, do_ib=.false.
//   * imoist=0, or imoist=1 with the moisture layout in Options (ptype=1
//     Kessler from the namelist; the ice branch of calcnm is ported and
//     tested through Options set by hand)
//
// Not ported (diagnostics only; never feed back into the state): budgets
// (tdiag/udiag/.../kdiag, dotbud), stau and psfc, iamsat/cloud/rh of calcnm,
// out3d, the timers, and the nstep=0 print in tkekm.
//
//------------------------------------------------------------------------------
// Where the time step calls each piece (cm1.F / solve1.F order, kessler_tke):
//
// (A) cm1.F:3748, at the end of every pass of the time loop, after solve,
//     mtime += dt, the reference dump, writeout and radiation. The first pass
//     of the loop has dosolve=.false. and is the startup call (nstep=0,
//     rtime=0, state = initial state). So a step N does: solve1/2/3 (state
//     N-1 -> N), then sfc_and_turb(state N, rtime = N*dt); the K's that step
//     N+1 uses (and that dump N+1 shows) are sfc_and_turb(state N).
//       sfc_and_turb(rtime = mtime, ...) on the NEW state (u3d, v3d, w3d,
//       pp3d, th3d, q3d == ua, va, wa, ppi, tha, qa after solve3), with prs,
//       rho, rf (rf from solve3) and tkea.
//     It computes ts.cme..csm (ramp_sgs), ts.dissten (zeroed, then
//     dissipation for k=2..nk), ts.nm, ts.defv, ts.defh, ts.t11..t33 (the
//     stresses rho*2K*S_ij), kmh/kmv/khh/khv (all halos, bcw + tkebc), sets
//     tkea(:,:,1)=0 and copies tkea's k=1 and k=nk+1 into tke3d.
//     All of these persist to the next step's solve1.
//
// (B) solve1.F (once per step, before the RK loop; tke source ~line 315,
//     turbt ~362, cvm ~744, dissipative heating ~919, doturb block ~1119),
//     on the "a" time level
//     (ua, va, wa, tha, qa, ppi, rho, rr, rf of the start of the step):
//       1. tke_source(...)    tketen = -khv*nm - dissten + kmv*max(0,defv+defh)
//       2. turbt(...)         tketen += diffusion of tkea (kmh, kmv)
//       ... (rdamp, diff2, cvm)
//       3. dissheat(...)      thten1 += dissipative heating   (idiss=1)
//       ... (rterm, radiation, pbl: off)
//       4. turb_tendencies(...) = turbu(ua -> uten1), turbv(va -> vten1),
//          turbw(wa -> wten1), vimpl arrays, turbs(tha-th0r -> thten1),
//          turbs(qa(n) -> qten(n)) for n=1..numq.
//     (Order matters only for thten1: dissheat before turbs, as in solve1.)
//
// (C) solve2.F (~line 1711), every RK stage, after the scalar advection:
//     wten(1:ni,1:nj,2:nk) := tketen; then
//     Advection::advw(nrk, dttmp, fluxes, tke3d, wten, hadvordrs, vadvordrs,
//     advwenos) (the explicit-order overload; Fortran passes iamw=.false.,
//     which only matters with immersed boundaries); then
//     tke_rk_update(dttmp, tkea, wten, tke3d):
//     tke3d = tkea + dttmp*wten, zeroed below 1e-10 (k=2..nk), bcw(tke3d,1).
//     Note advw reads tke3d (the latest RK stage), the update starts from
//     tkea, and wten must be reset from tketen at every stage.
//
// (D) solve3.F (~line 573), after the RK loop: tke_copy(tke3d, tkea)
//     (k=1..nk, i,j=0..n+1).
//------------------------------------------------------------------------------

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

namespace cm1k::turb {

// Options read by turb.F/turbtend.F/solve1.F from the `input` module.
struct Options {
  int cm1setup = 1, sgsmodel = 1, ipbl = 0, horizturb = 0, cuparam = 0;
  int tconfig = 1, bcturbs = 1, doimpl = 1, idiss = 0;
  int ramp_sgs = 0;
  Real ramp_time = 0.0;
  bool dosfcflx = false;
  int bbc = 1, tbc = 1;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;   // serial: 1 iff the boundary is not periodic
  bool terrain_flag = false;
  int axisymm = 0;
  bool do_ib = false;
  // dohturb/dovturb (param.F): true for LES
  bool dohturb = true, dovturb = true;
  // moisture (param.F): imoist, eqtset (solve1 dissipative heating),
  // ptype (rslf/rsif and calcnm's qvci), species indices (1-based).
  int imoist = 0, eqtset = 2, ptype = 0, iice = 0;
  int numq = 1, nqv = 1, nqc = 0, nqi = 0, nql1 = 0, nql2 = 0, nqs1 = 0, nqs2 = 0;

  static Options from_namelist(const Namelist& nl);
  void set_boundary_flags();
  void check() const;   // throws std::runtime_error for anything not ported
};

// Arrays computed by sfc_and_turb and read by the next solve1 (cm1.F names
// and shapes). They persist across time steps.
struct TurbState {
  FArray2 cm0;                               // (ib:ie,jb:je)
  FArray3 cme, csm, ce1, ce2;                // (ibc:iec,jbc:jec,kbc:kec)
  FArray3 nm, defv, defh, lenscl, dissten;   // (ib:ie,jb:je,kb:ke+1)
  FArray3 t11, t12, t13, t22, t23, t33;      // (ib:ie,jb:je,kb:ke)
  void allocate(const Bounds& b);
  // init3d.F (idoles, cm1setup=1): cm0 = c_m, and cme/ce1/ce2/csm from cm0
  // at every point (kb..ke, as in init3d). Call once at initialization.
  void init_constants(const Bounds& b);
};

// Scratch arrays (cm1.F dum1..dum9 roles). All (ib:ie,jb:je,kb:ke).
struct Work {
  FArray3 d1, d2, d3, d4, d5, d6, d7, d8, d9, sadv;
  void allocate(const Bounds& b);
};

//------------------------------------------------------------------------------
// (A) sfc_and_turb, LES path. Inputs at the new time level (see above).
//   rtime: model time (s) at the end of the step (ramp_sgs)
//   ua,va,wa,ppi,tha,qa,prs,rho,rf : cm1.F u3d,v3d,w3d,pp3d,th3d,q3d,prs,rho,rf
//   tkea (inout: k=1 set to 0, halos), tke3d (out: k=1 and nk+1 copied)
//   kmh,kmv,khh,khv (out, incl. halos)
//------------------------------------------------------------------------------
void sfc_and_turb(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
                  Real rtime, const FArray3& ua, const FArray3& va, const FArray3& wa,
                  const FArray3& ppi, const FArray3& tha, const FArray4& qa,
                  const FArray3& prs, const FArray3& rho, const FArray3& rf,
                  const FArray3& tkea, const FArray3& tke3d,
                  const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
                  const TurbState& ts, const Work& w);

// sfc_and_turb: ramp of the turbulence "constants" (ramp_sgs>=1).
void update_constants(const Options& o, const Bounds& b, Real rtime, const TurbState& ts);

// turb.F calcnm: squared Brunt-Vaisala frequency nm (k=2..nk; 0 at k=1,nk+1).
//   t, qt, thv, qvci: scratch (ib:ie,jb:je,kb:ke)
void calcnm(const Options& o, const Constants& c, const Bounds& b, Real rdz,
            const FArray3& c1, const FArray3& c2, const FArray3& mf,
            const FArray3& pi0, const FArray3& th0, const FArray3& nm,
            const FArray3& t, const FArray3& qt, const FArray3& thv, const FArray3& qvci,
            const FArray3& prs, const FArray3& pp, const FArray3& th, const FArray4& qa);

// turb.F calcdef: s_ij (times density) into s11..s33, deformation defv/defh.
void calcdef(const Options& o, const Grid& g, const FArray3& rho, const FArray3& rf,
             const FArray3& ua, const FArray3& va, const FArray3& wa,
             const FArray3& s11, const FArray3& s12, const FArray3& s13,
             const FArray3& s22, const FArray3& s23, const FArray3& s33,
             const FArray3& defv, const FArray3& defh);

// turb.F tkekm: length scales, dissipation (dissten +=), kmh/kmv/khh/khv
// (k=2..nk), then bcw(K,1) on all four.
//   tk, lenh, grdscl, rgrdscl: scratch (ib:ie,jb:je,kb:ke)
void tkekm(const Options& o, const Grid& g, const FArray3& nm,
           const FArray3& tk, const FArray3& lenh, const FArray3& grdscl, const FArray3& rgrdscl,
           const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
           const FArray3& cme, const FArray3& ce1, const FArray3& ce2,
           const FArray3& tkea, const FArray3& lenscl, const FArray3& dissten);

// turb.F tkebc (bbc=1, tbc=1): tkea, K at the lower boundary.
void tkebc(const Options& o, const Bounds& b, const FArray3& tkea,
           const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv);

// turb.F gettau (+ sfcstress for bbc=1): t_ij = 2 K s_ij, boundary values.
void gettau(const Options& o, const Bounds& b, const FArray3& kmh, const FArray3& kmv,
            const FArray3& t11, const FArray3& t12, const FArray3& t13,
            const FArray3& t22, const FArray3& t23, const FArray3& t33);

//------------------------------------------------------------------------------
// (B) solve1
//------------------------------------------------------------------------------

// solve1.F: TKE sources, tketen(1:ni,1:nj,2:nk) = (tconfig=1)
//   -khv*nm - dissten + kmv*max(0,defv+defh)
void tke_source(const Options& o, const Bounds& b, const FArray3& nm, const FArray3& defv,
                const FArray3& defh, const FArray3& dissten, const FArray3& kmh,
                const FArray3& kmv, const FArray3& khv, const FArray3& tketen);

// turbtend.F turbt: tten(1:ni,1:nj,2:nk) += subgrid diffusion of t (TKE).
void turbt(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& t, const FArray3& tten, const FArray3& kmh, const FArray3& kmv,
           const Work& w);

// solve1.F (idiss=1): thten1(1:ni,1:nj,1:nk) += dissipative heating, with
// cvm from qa (eqtset=2, imoist=1) or cv/cp.
void dissheat(const Options& o, const Constants& c, const Bounds& b, const FArray3& pi0,
              const FArray3& ppi, const FArray4& qa, const FArray3& dissten,
              const FArray3& thten1, const FArray3& cvm);

// turbtend.F turbu/turbv/turbw (+ turbuz/turbvz).
void turbu(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& u, const FArray3& uten, const FArray3& w,
           const FArray3& t11, const FArray3& t12, const FArray3& t13, const FArray3& kmv,
           const Work& wk);
void turbv(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& v, const FArray3& vten, const FArray3& w,
           const FArray3& t12, const FArray3& t22, const FArray3& t23, const FArray3& kmv,
           const Work& wk);
void turbw(const Options& o, const Grid& g, Real dt, const FArray3& rho, const FArray3& rf,
           const FArray3& w, const FArray3& wten, const FArray3& t13, const FArray3& t23,
           const FArray3& t33, const FArray3& kmh, const Work& wk);

// solve1.F: vimpl coefficient arrays for turbs (doimpl=1):
//   dum7 = khv(k)*mf(k)*rf(k)*mh*rr, dum8 = khv(k+1)*mf(k+1)*rf(k+1)*mh*rr
void vimpl_coefs(const Bounds& b, const FArray3& khv, const FArray3& mf, const FArray3& rf,
                 const FArray3& mh, const FArray3& rr, const FArray3& dum7, const FArray3& dum8);

// turbtend.F turbs: sten(1:ni,1:nj,1:nk) += subgrid diffusion of s.
//   dum7/dum8 from vimpl_coefs (doimpl=1). iflux only matters with surface
//   fluxes (not ported).
void turbs(const Options& o, const Grid& g, int iflux, Real dt, const FArray3& rho,
           const FArray3& rr, const FArray3& rf, const FArray3& s, const FArray3& sten,
           const FArray3& khh, const FArray3& khv, const FArray3& dum7, const FArray3& dum8,
           const Work& wk);

// solve1.F doturb block: turbu/turbv/turbw into uten1/vten1/wten1, vimpl
// arrays, turbs on theta (tha + th0(1,1,k) - th0r) into thten1 and on each
// moisture species qa(:,:,:,n) into qten(:,:,:,n).
void turb_tendencies(const Options& o, const Grid& g, const BaseState& base, Real dt,
                     const FArray3& rho, const FArray3& rr, const FArray3& rf,
                     const FArray3& ua, const FArray3& va, const FArray3& wa,
                     const FArray3& tha, const FArray4& qa,
                     const FArray3& uten1, const FArray3& vten1, const FArray3& wten1,
                     const FArray3& thten1, const FArray4& qten,
                     const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& khv,
                     const TurbState& ts, const Work& wk);

//------------------------------------------------------------------------------
// (C)/(D) solve2/solve3 helpers
//------------------------------------------------------------------------------

// solve2.F: tke3d(1:ni,1:nj,2:nk) = tkea + dttmp*tten (zeroed below 1e-10),
// then bcw(tke3d,1).
void tke_rk_update(const Options& o, const Bounds& b, Real dttmp, const FArray3& tkea,
                   const FArray3& tten, const FArray3& tke3d);
// solve3.F: tkea(0:ni+1,0:nj+1,1:nk) = tke3d.
void tke_copy(const Bounds& b, const FArray3& tke3d, const FArray3& tkea);

// bc.F bcw (non-MPI branches) on a w-level array (ib:ie,jb:je,kb:ke+1);
// flag=1 also zeroes k=1 and k=nk+1 on 0..ni+1 x 0..nj+1.
void bcw(const Options& o, const Bounds& b, const FArray3& w, int flag);
// bc.F bc2d applied to level k of a 3D array (bc2d(a(ib,jb,k))).
void bc2d_level(const Options& o, const Bounds& b, const FArray3& a, int k);

} // namespace cm1k::turb

#endif
