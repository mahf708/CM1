#ifndef CM1K_DYNCORE_HPP
#define CM1K_DYNCORE_HPP

#include "dynamics/cm1k_bc.hpp"
#include "dynamics/cm1k_diff2.hpp"
#include "dynamics/cm1k_openbc.hpp"
#include "dynamics/cm1k_sound.hpp"
#include "physics/myj/cm1k_myj_coupling.hpp"
#include "physics/pbl/cm1k_pbl.hpp"
#include "physics/pbl_gfs/cm1k_pbl_gfs.hpp"
#include "physics/pbl_mynn/cm1k_mynn_coupling.hpp"
#include "physics/sfc_gfdl/cm1k_gfdl_driver.hpp"
#include "physics/pbl_ysu/cm1k_pbl_ysu.hpp"
#include "physics/sfclay/cm1k_sfclay_driver.hpp"
#include "physics/turb/cm1k_turb.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <memory>

namespace cm1k {

// Options of the dynamical core (the `input` module variables that solve1,
// solve2, solve3 and sound read), with param.F's defaults and adjustments.
// check() throws for every setting whose code path is not ported.
struct DynConfig {
  int cm1setup = 1, sgsmodel = 0, ipbl = 0, idiff = 0, difforder = 6;
  int imoist = 0, ptype = 0, eqtset = 2, psolver = 3, apmasscon = 1;
  int irdamp = 0, hrdamp = 0, icor = 0, lspgrad = 0, testcase = 0;
  int axisymm = 0, cuparam = 0, radopt = 0, iptra = 0, iprcl = 0, iinit = 1;
  int bbc = 1, idiss = 0, rterm = 0, adapt_dt = 0, isfcflx = 0, sfcmodel = 0, irbc = 4;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1, roflux = 0, nudgeobc = 0;
  bool terrain_flag = false, do_ib = false, do_lsnudge = false, do_recycle = false, do_adapt_move = false;
  Real dtl = 0, alph = 0.60, kdiv = 0.10, rdalpha = 0, zd = 0, xhd = 0, csound = 300.0;
  Real smeps = 1.0e-60, tsmall = 0.000000000001, qsmall = 1.0e-24;   // cm1.F, DP build
  int hadvordrs = 5, vadvordrs = 5;
  // Coriolis (icor=1, solve2), explicit diffusion (idiff/difforder; 6: in
  // advection, 2: diff2.F), lateral Rayleigh damping (hrdamp, xhd)
  int imove = 0, betaplane = 0, tbc = 1;
  Real fcor = 0, umove = 0, vmove = 0, kdiff2 = 75.0;
  // derived (param.F)
  bool idoles = false, dohturb = false, dovturb = false;
  int numq = 0, nqv = 0, nqc = 0, nql1 = -2, nql2 = -2, iice = 0;
  // subgrid turbulence (sgsmodel=1 with cm1setup=1): physics/turb options
  bool turb = false;
  turb::Options turb_opt;
  // surface layer (sfcmodel>=1, bbc=3): physics/sfc options; for the
  // WRF/MM5 surface layers (sfcmodel=2,3) also physics/sfclay's
  sfc::Options sfc_opt;
  sfclay::Options sfclay_opt;
  // boundary-layer schemes (param.F use_pbl: cm1setup=2,4 with ipbl>=1):
  // physics/pbl options; turb_opt is then filled too (boundary flags,
  // turbsz/turbuz/turbvz, horizturb=1). Ported: cm1setup=2 with ipbl=1
  // (YSU, physics/pbl_ysu; sfcmodel=2,3), ipbl=2 (sfcmodel=1,2,3) or ipbl=3
  // (GFS-EDMF, physics/pbl_gfs; sfcmodel=2,3, or 4: physics/sfc_gfdl).
  bool pbl = false;
  pbl::Options pbl_opt;
  // MYNN (cm1setup=2, ipbl=4/5 with the MYNN surface layer sfcmodel=6):
  // physics/pbl_mynn/cm1k_mynn_coupling.hpp
  bool mynn = false;
  pbl_mynn::Coupling mynn_opt;
  // MYJ (cm1setup=2, ipbl=6 with the MYJ surface layer sfcmodel=7):
  // physics/myj/cm1k_myj_coupling.hpp
  bool myj = false;
  myj::Coupling myj_opt;

  // explicit diffusion: 6th order inside advection, or diff2.F
  bool diff6() const { return idiff >= 1 && difforder == 6; }
  bool diff2() const { return idiff >= 1 && difforder == 2; }

  static DynConfig from_namelist(const Namelist& nl);
  void check() const;
};

// The dynamical core: cm1.F's time-step body solve1 (pre-RK tendencies),
// solve2 (RK3 loop with advection, buoyancy and the acoustic solver), and
// solve3 (finish: boundary conditions, update the "a" arrays, rr/rf).
//
// It owns the arrays CM1 carries between time steps that are not in the
// reference dumps: the "3d" copies of the prognostic variables (u3d, ...),
// ppx (pressure at the last acoustic step plus divergence damping), the
// diabatic tendencies of the previous step (qpten, ...), mass1 (initial dry
// air mass, for apmasscon=1), rr/rf, and the Rayleigh damping coefficients.
class Advection;

class Dyncore {
public:
  Dyncore();
  ~Dyncore();

  // MPI domain decomposition (share/cm1k_comm.hpp): the halo exchanges and
  // reductions of comm.F at the Fortran call sites, bc.F's `#ifdef MPI`
  // branches. Call before setup(); the Comm must outlive the Dyncore. A
  // serial Comm (or none) keeps the serial code paths.
  void set_comm(Comm* c) { m_comm = c; }
  bool mpi() const { return m_comm && m_comm->mpi(); }

  // Sets up the work arrays and the derived quantities (tauh/taus/tauf,
  // dtu0/dtv0, nsound, rdx...) from the grid. Call after the grid and base
  // state are filled.
  void setup(const Namelist& nl, const Constants& c, Grid& grid, const BaseState& base, State& state);
  bool is_setup() const { return m_setup; }

  // Initialize the carried arrays from the "a" arrays, as at the start of a
  // run (init3d.F: u3d=ua..., ppx=ppi, rr/rf from rho, mass1 from rho).
  // For a dump from a later step ppx, qpten/... and the open-boundary phase
  // speeds (radbcw/...) are not recoverable; they are then approximated by
  // the step-0 rules (radbc* = 0).
  void start_from_state();

  void solve1();
  void solve2();
  // Open-radiative boundaries (wbc..nbc = 2): solve1.F's irbc=1 phase
  // speeds, solve2.F's nudgeobc and irbc=4 phase speeds (public: they launch
  // device lambdas, see compute_rf).
  void openbc_solve1();
  void openbc_nudge();
  void openbc_radbc4(Real dttmp);
  // Called between solve2 and solve3 for imoist=1 (see Model::step).
  void solve3();

  // Subgrid turbulence (sgsmodel=1), cm1.F:3748 sfc_and_turb: eddy
  // viscosities, deformation, N^2, dissipation and stresses from the
  // current "a" state, at model time rtime. CM1 calls it at the end of
  // every pass of the time loop (and in the startup pass, dosolve=.false.);
  // Model::step calls it at the start of each step, which is the same point
  // in the sequence (between mtime=mtime+dt and the next solve1).
  //   update_sfc: cm1.F's update_sfc, .false. only in the startup pass
  //   (nstep = 0): with sfcmodel=2,3,6 the SLAB land model then does not run.
  bool turb_enabled() const { return m_cfg.turb; }
  // nstep: the model step (CM1's nstep at that call: 0 in the startup pass,
  // where update_sfc = .false.); update_sfc = (nstep > 0).
  void sfc_and_turb(double rtime, int nstep);
  // cm1setup=2 (ipbl>=1) runs the mesoscale sequence of sfc_and_turb
  // instead: surface layer (sfc::sfc_prelude with gethpbl2 for ipbl=2, then
  // sfcmodel=1's getcecd/sfcflux/sfcdiags or sfclay::wrf_surface_layer; or
  // the MYNN surface layer sfcmodel=6, pbl_mynn::surface_layer; or the MYJ
  // surface layer sfcmodel=7, myj::surface_layer), calcnm,
  // calcdef, turbparam_horiz (horizturb=1), turbparam_vert (ipbl=2),
  // gettau (ipbl=2 or horizturb=1), then the PBL scheme through pbl() below.
  bool pbl_enabled() const { return m_cfg.pbl; }
  // The surface-layer arrays (sfcmodel>=1), owned by the Model; sfc_and_turb
  // updates them and solve1 reads them. Call after setup().
  void set_surface(const sfc::SfcState& s);
  // sfcmodel=2,3: the WRF surface-layer state (its common arrays are the
  // set_surface ones), owned by the Model. Call after setup().
  void set_wrf_surface(const sfclay::WrfSurface& w);
  // The PBL tendencies (DynConfig::pbl), owned by the Model (they are in the
  // dumps); pbl() fills them and solve1 applies them. Call after setup().
  void set_pbl(const pbl::Tendencies& t);
  // MYNN's carried arrays (DynConfig::mynn), owned by the Model (they are in
  // the dumps). Call after setup().
  void set_mynn(const pbl_mynn::Carried& c);
  // sfcmodel=4: the GFDL surface-layer state (its common arrays are the
  // set_surface ones), owned by the Model. Call after setup().
  void set_gfdl_surface(const sfc_gfdl::GfdlSurface& w);
  // ipbl=3: GFS-EDMF's diffusivities xkzh, xkzm (in the dumps), owned by
  // the Model. Call after setup().
  void set_gfs(const pbl_gfs::Carried& c);
  // MYJ's carried arrays (DynConfig::myj), owned by the Model (most are in
  // the dumps). Call after setup().
  void set_myj(const myj::Carried& c);
  // solve2.F: qke advection for the MYNN PBL, one RK stage (public: it
  // launches device lambdas)
  void advect_qke(int nrk, Real dttmp);
  // PBL dispatch hook, the "doingpbl" block of turb.F sfc_and_turb: called
  // by sfc_and_turb (cm1setup=2) after the eddy viscosities and stresses of
  // the current state are set (s.kmh..khv, ts.t11..t33, ts.nm/defv/defh,
  // surface layer incl. hpbl). Each scheme writes its tendencies into the
  // pbl::Tendencies arrays (thpten, qvpten, qcpten, qipten, upten, vpten:
  // the u/v ones at scalar points, solve1 averages them to u/v points),
  // which solve1 adds to thten1/qten before and uten1/vten1 after the doturb
  // block; then pbl() applies bcs to upten and vpten (use_pbl). Dispatch on
  // m_cfg.pbl_opt.ipbl:
  //   1: pbl_ysu::ysu (YSU, CM1's doysu: wrfphys::pbl_moist/pbl_prep
  //      inputs, hpbl, u10, v10 from the scheme) and, for idiss=1,
  //      pbl_ysu::ysu_dissten into ts.dissten
  //   2: pbl::br09 (simple Bryan-Rotunno/Louis scheme, CM1's dobr09)
  //   4/5: pbl_mynn::pbl (MYNN-EDMF; its surface layer sfcmodel=6 runs in
  //     sfc_and_turb, qke is advected in solve2; carried arrays in
  //     pbl_mynn::Carried, set with set_mynn())
  //   3: pbl_gfs::gfsedmf (GFS-EDMF, CM1's dogfsedmf: the wrfphys inputs,
  //     z = zf, znt = mznt with sfcmodel=4 (else znt); u*, wspd, hpbl and
  //     xkzh/xkzm (set_gfs) from the scheme; its dissipative heating is
  //     part of thpten, dissten = its TKE dissipation)
  //   6: myj::pbl (MYJ; its surface layer sfcmodel=7 runs in sfc_and_turb;
  //     carried arrays in myj::Carried, set with set_myj())
  //   7, ...: not ported yet. A new scheme adds its case
  //     here (inputs: u3d/v3d/w3d/th3d/pp3d/q3d, s.prs, s.rho, rr, rf, the
  //     surface state m_sfc), any extra outputs or carried arrays to
  //     pbl::Tendencies (allocated from DynConfig::pbl_opt), its options to
  //     pbl::Options/check() and DynConfig::check, and its dissipative
  //     heating / TKE terms to solve1.
  // Public: it launches device lambdas.
  void pbl(double rtime);
  // DynConfig::check for MPI runs: throws for every option whose MPI code
  // path (exchanges, reductions) is not ported or not validated
  void check_mpi(const Bounds& b) const;
  // solve3.F/init3d.F: rf from rho on 0..ni+1, 0..nj+1 (public: it launches
  // device lambdas, which nvcc does not allow in private member functions)
  void compute_rf();
  // turb.F: the WRF physics inputs (m_ww) from the current state (public:
  // it launches device lambdas)
  void wrf_inputs();
  // idiff=1, difforder=2: cm1.F's diff2def (stresses for diff2u/v/w), called
  // right after sfc_and_turb, at the same point of the time loop.
  void diff2_stress();

  const DynConfig& config() const { return m_cfg; }
  const OpenBcParams& openbc() const { return m_obc; }
  // radiative phase speeds carried between steps (zero at start, as in cm1.F)
  const RadBc& radbc() const { return m_rad; }
  int nsound() const { return m_nsound; }
  double mass1() const { return m_mass1; }

  // Arrays microphysics works on (mp_driver's th3d, q3d, pp3d, ...).
  FArray3 u3d, v3d, w3d, pp3d, th3d, ppx;
  FArray4 q3d;
  FArray3 rr, rf, divx;
  FArray3 qpten, qtten, qvten, qcten;   // ibm:iem,...
  FArray3 tauh, taus, tauf;
  // subgrid turbulence: tke3d (RK-stage TKE), tketen (TKE tendency of
  // solve1, reused in every RK stage), the arrays sfc_and_turb hands to
  // solve1 (TurbState), and scratch
  FArray3 tke3d, tketen, cvm;
  turb::TurbState ts;

private:
  void bc_scalars_after_init();

  turb::Work m_tw;
  Diff2Work m_d2;   // idiff=1, difforder=2
  sfc::SfcState m_sfc;
  bool m_has_sfc = false;
  sfclay::WrfSurface m_wsfc;
  bool m_has_wsfc = false;
  // the WRF physics inputs turb.F builds (wrfphys::pbl_prep: dum1..dum7;
  // pbl_moist: divx = qv, ppten = qc, dum8 = qi; dum9 = rthraten = 0) and
  // YSU's other outputs (xkzh, xkzm, kpbl2d, wstar, delta)
  struct WrfWork {
    FArray3 d1, d2, d3, d4, d5, d6, d7, qv, qc, qi, rth, xkzh, xkzm, kpbl2d, wstar, delta;
    pbl_ysu::Workspace ysu;
    pbl_gfs::Workspace gfs;
    wrfphys::WrfConst c{};
  } m_ww;
  sfc_gfdl::GfdlSurface m_gsfc;   // sfcmodel=4
  bool m_has_gsfc = false;
  pbl_gfs::Carried m_gc;          // ipbl=3
  bool m_has_gfs = false;
  pbl::Tendencies m_pt;
  bool m_has_pbl = false;
  pbl_mynn::Carried m_mc;   // MYNN (DynConfig::mynn)
  pbl_mynn::Work m_mw;
  bool m_has_mynn = false;
  myj::Carried m_myc;       // MYJ (DynConfig::myj)
  myj::Work m_myw;
  bool m_has_myj = false;
  int m_nstep = 0;          // model step at sfc_and_turb (MYNN initflag, SFCLAY_mynn itimestep)
  Real m_cgs1 = 0, m_cgs2 = 0, m_cgs3 = 0, m_cgt1 = 0, m_cgt2 = 0, m_cgt3 = 0;

  DynConfig m_cfg;
  Constants m_c{};
  BcParams m_bc;
  OpenBcParams m_obc;
  RadBc m_rad;
  SoundOpenBc m_sobc;
  Grid m_g;
  BaseState m_b;
  State m_s;
  std::unique_ptr<Advection> m_adv;

  Comm* m_comm = nullptr;
  bool m_setup = false;
  int m_nsound = 0;
  Real m_dt = 0;
  double m_mass1 = 0, m_mass2 = 0;

  // tendencies / work arrays (solve1/2/3 names)
  FArray3 rru, rrv, rrw, uten, vten, wten, uten1, vten1, wten1;
  FArray3 thten, thten1, ppten, sten, sadv, thterm, t11, t22;
  FArray3 dum1, dum2, dum3, dum8, mm, ppd, fpk, qk, pk1, pk2;
  FArray4 qten;
  FArray2 dtu, dtv, dtu0, dtv0;
  FArray1 dumk3, dumk4;
};

} // namespace cm1k

#endif
