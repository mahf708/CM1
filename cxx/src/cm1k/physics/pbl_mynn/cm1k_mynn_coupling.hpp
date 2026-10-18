#ifndef CM1K_MYNN_COUPLING_HPP
#define CM1K_MYNN_COUPLING_HPP

// The glue that runs the MYNN surface layer (sfcmodel=6, physics/sfc_mynn)
// and the MYNN-EDMF PBL (ipbl=4/5, physics/pbl_mynn) inside the full model
// (cm1setup=2), as src/turb.F sfc_and_turb and src/solve2.F do:
//
//   surface_layer()   turb.F: psfc (cgs1..3), the "variables for wrf
//                     physics" block (k2=2), divx = qv, the bbc=3 block
//                     (u1, v1, s1, t1 at k=1), the mynn_sf_j_loop
//                     (SFCLAY_mynn, itimestep = nstep), then turb.F
//                     1019-1047: with update_sfc (and testcase /= 11) the
//                     SLAB land model (src/slab.F, slab::Land: over land
//                     tsk, tslb and the land hfx, qfx, lh from SFCLAY_mynn's
//                     flhc, flqc; qsfc everywhere), else the qsfc diagnosis
//                     qsfc = qv + qfx/flqc (what SLAB reduces to over
//                     water); then sfc_mynn::finish (ch = ck, cq = ck, s10;
//                     thflux/qvflux with dosfcflx or output_sfcflx = 1).
//   pbl()             turb.F get_vars + the domynn block (initflag =
//                     (nstep == 0)); the tendencies go to pbl::Tendencies
//                     (thpten, qvpten, qcpten, qipten, upten, vpten), which
//                     solve1 applies as for every PBL scheme. MYNN's
//                     dissipative heating is part of thpten (param.F sets
//                     idiss = 0 for ipbl = 3, 4, 5).
//   qke advection     solve2.F (bl_mynn_tkeadvect = .true.): per RK stage,
//                     qke3d = qke at nrk = 1, bcs(qke3d), advs with pdef = 1,
//                     pdefweno = 1, weps = 10*epsilon into qke_adv, then
//                     qke3d = qke + dttmp*qke_adv (0 below 1e-10). Done by
//                     Dyncore::solve2 with the arrays here (Carried::st.qke_adv
//                     is CM1's qke3d; Work::qketen is CM1's qke_adv there).
//
// CM1 never calls MYNN on a restart (turb.F: .not. restarted); the Model has
// no restart path (init_from_dump continues a run like the Fortran does), so
// it is always called.
//
// Arrays (cm1.F names). The sfc_mynn::Fields, pbl_mynn::Surface and the
// u/v/th/q tendencies alias the surface-layer (sfc::SfcState) and PBL
// (pbl::Tendencies) arrays where CM1 passes the same array, so a value
// written by one routine is seen by the next as in the Fortran.

#include "physics/pbl/cm1k_pbl.hpp"
#include "physics/pbl_mynn/cm1k_pbl_mynn.hpp"
#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/sfc_mynn/cm1k_sfc_mynn.hpp"
#include "physics/slab/cm1k_slab_land.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_state.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::pbl_mynn {

// The MYNN path's options (from the namelist, param.F's fixed values).
struct Coupling {
  Options pbl;                    // ipbl, closure, bl_mynn_* (param.F)
  sfc_mynn::Options sfc;          // sfcmodel=6, isftcflx, iz0tlnd
  bool dosfcflx = false;          // init_surface.F (.false. for ipbl=4,5)
  int output_sfcflx = 0;          // namelist param9: thflux/qvflux are diagnosed then
  int lu0 = 16, initsfc = 1;      // init_surface.F (water: lu_index 16)
  Real tmn0 = 297.28;             // init_surface.F: SLAB's deep soil temperature
  int testcase = 0;               // turb.F: no SLAB for testcase = 11

  bool active() const { return pbl.active(); }
  static Coupling from_namelist(const Namelist& nl);
  // Throws for what is not ported: ipbl=4/5 without sfcmodel=6 (and
  // sfcmodel=6 without ipbl=4/5), the options Options::check and
  // sfc_mynn::Options::check reject, and what init_surface.F stops for
  // (lu0 outside 1..33, tmn0 < 100 K).
  void check() const;
};

// Carried arrays: persist across steps (the Model owns them; dumped).
struct Carried {
  sfc_mynn::Fields sf;    // MYNN surface layer (mol, rmol, zol, ustm, ... own; the rest alias SfcState)
  Surface sfc2d;          // MYNN PBL 2D in/out (kpbl, nupdraft, maxmf, ktop_plume own; the rest alias)
  State st;               // qke, qke_adv (= CM1's qke3d), tsq, qsq, cov, ...
  Tendencies td;          // upten..qipten alias pbl::Tendencies; xkzh, xkzm, dheat own
  slab::Land land;        // SLAB: tslb, tmn, emiss, thc, ... (cm1.F, init_surface.F)
  // Allocates the arrays of their own and sets the aliases; so: SLAB's
  // tmn0, lu0, season (slab::Land::setup).
  void allocate(const Bounds& b, const sfc::SfcState& s, const pbl::Tendencies& pt, const sfc::Options& so);
  // init_surface.F: the initial soil temperatures (after sfc::init_surface)
  void init_soil(const Bounds& b, const sfc::SfcState& s) const;
  // (name, array) as src/cm1dump_state.inc writes them (sfcmodel=6 block
  // incl. tslb, then the ipbl=4/5 block without hpbl and the pbl::Tendencies
  // arrays)
  std::vector<std::pair<std::string, FArray3>> dumped_sfc() const;
  std::vector<std::pair<std::string, FArray3>> dumped_pbl() const;
};

// Work arrays and tables (the Dyncore owns them; not carried).
struct Work {
  sfc_mynn::PsiTables psi;   // init_physics.F: mynn_sf_init_driver
  sfc_mynn::Inputs sin;
  Inputs pin;
  FArray3 qketen;            // CM1's qke_adv in solve2 (advective tendency)
  void allocate(const Bounds& b);
};

// sfc_and_turb's surface part for sfcmodel=6 (see above), on the state
// (ua, va, tha, ppi, qa, prs, rho, rf) at the start of a step.
//   cgs1..3: param.F extrapolation coefficients to the surface
//   dt: the time step (SLAB's deltsm); update_sfc: cm1.F's (.false. only in
//   the startup pass)
void surface_layer(const Coupling& o, const Constants& c, const Grid& g, const BaseState& base, Real cgs1,
                   Real cgs2, Real cgs3, int nstep, Real dt, bool update_sfc, const FArray3& ua, const FArray3& va, const FArray3& tha,
                   const FArray3& ppi, const FArray4& qa, int nqv, const FArray3& prs, const FArray3& rho,
                   const FArray3& rf, const sfc::SfcState& s, const Carried& cr, const Work& w);

// sfc_and_turb's domynn block (after the surface layer and turb::sfc_bcs).
// nqv/nqc/nqi: index of qv/qc/qi in qa (0 if absent).
void pbl(const Coupling& o, const Grid& g, const BaseState& base, Real dt, int nstep, const FArray3& ua,
         const FArray3& va, const FArray3& wa, const FArray3& tha, const FArray3& ppi, const FArray4& qa,
         int nqv, int nqc, int nqi, const FArray3& prs, const FArray3& rho, const Carried& cr, const Work& w);

} // namespace cm1k::pbl_mynn

#endif
