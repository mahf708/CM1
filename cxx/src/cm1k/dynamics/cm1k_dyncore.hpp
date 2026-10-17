#ifndef CM1K_DYNCORE_HPP
#define CM1K_DYNCORE_HPP

#include "dynamics/cm1k_bc.hpp"
#include "dynamics/cm1k_diff2.hpp"
#include "dynamics/cm1k_openbc.hpp"
#include "dynamics/cm1k_sound.hpp"
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
  // surface layer (sfcmodel=1, bbc=3): physics/sfc options
  sfc::Options sfc_opt;

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
  bool turb_enabled() const { return m_cfg.turb; }
  void sfc_and_turb(double rtime);
  // The surface-layer arrays (sfcmodel=1), owned by the Model; sfc_and_turb
  // updates them and solve1 reads them. Call after setup().
  void set_surface(const sfc::SfcState& s);
  // solve3.F/init3d.F: rf from rho on 0..ni+1, 0..nj+1 (public: it launches
  // device lambdas, which nvcc does not allow in private member functions)
  void compute_rf();
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
