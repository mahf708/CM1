#ifndef CM1K_ADV_HPP
#define CM1K_ADV_HPP

// Port of CM1's advection (src/adv.F: advs, advu, advv, advw; the flux
// routines of src/adv_routines.F; and the positive-definite correction of
// src/pdef.F, which advs calls with pdscheme=1, hard-wired in param.F).
//
// Supported (everything else throws in AdvOptions::check):
//   * Cartesian grid (axisymm=0), with or without terrain (terrain_flag:
//     the terrain branches of adv.F and pdef.F, metrics from the Grid),
//     no immersed boundaries (do_ib=.false.), no large-scale w (dolsw=.false.)
//   * flux-form advection of orders 2..10 (hadvord*, vadvord*)
//   * WENO for scalars/velocities with weno_order = 5 (advwenos/advwenov = 1,2),
//     including the pdef / pdefweno variants used for moisture
//   * 6th-order horizontal diffusion inside advection (idiff=1, difforder=6,
//     mdiff = 0/1)
//   * lateral boundaries wbc..nbc = 1 (periodic), 2 (open-radiative: inflow
//     fluxes set to zero), 3/4 (rigid walls)
//
// Not ported (do not change prognostic results):
//   * budgets: advs's bflag/bsq/dumk1/dumk2 (stat_qsrc), the *diag arrays
//     (dotbud/doqbud/doubud/...), timers.
//
// Index conventions follow the Fortran exactly (see cxx/README.md): ua(i,j,k)
// is ua(i,j,k). The routines are called once per RK stage with the mass
// fluxes rru/rrv/rrw and divx that solve2 computes.

#include "share/cm1k_grid.hpp"
#include "share/cm1k_state.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k {

// Options read by adv.F / adv_routines.F / pdef.F from the `input` module.
struct AdvOptions {
  int hadvordrs = 5, vadvordrs = 5, hadvordrv = 5, vadvordrv = 5;
  int advwenos = 0, advwenov = 0, weno_order = 5;
  int idiff = 0, mdiff = 0, difforder = 6;
  Real kdiff6 = 0.040;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  // ibw..ibn: "this process owns the west/... boundary and it is not
  // periodic" (param.F, serial: ibw = 1 iff wbc != 1).
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;
  bool terrain_flag = false;
  int axisymm = 0;
  bool do_ib = false;
  bool dolsw = false;
  Real smeps = 1.0e-60;   // cm1.F, double precision build

  static AdvOptions from_namelist(const Namelist& nl);
  // Recompute ibw..ibn from wbc..nbc (serial run).
  void set_boundary_flags();
  // Throws std::runtime_error for anything not ported.
  void check() const;
};

// Inputs of one RK stage shared by advs/advu/advv/advw (solve2 names).
struct AdvFluxes {
  FArray3 rru;    // (ib:ie+1,jb:je,kb:ke)   rho0*u
  FArray3 rrv;    // (ib:ie,jb:je+1,kb:ke)   rho0*v
  FArray3 rrw;    // (ib:ie,jb:je,kb:ke+1)   rf0*w
  FArray3 divx;   // (ib:ie,jb:je,kb:ke)     divergence of the mass flux
  FArray3 rho;    // (ib:ie,jb:je,kb:ke)     full density (6th-order diffusion only)
  FArray3 rr;     // (ib:ie,jb:je,kb:ke)     1/rho        (advs diffusion only, "ri")
};

// Per-call arguments of advs that solve2 varies between variables.
struct AdvsArgs {
  int wflag = 1;        // 0: never WENO (e.g. pressure); 1: WENO per advwenos
  int pdef = 0;         // 1: positive-definite correction (pdefx/y/z)
  int pdefweno = 0;     // 1: positive-definite variant of vertical WENO
  double weps = 0.0;    // WENO epsilon (solve2: 10*epsilon for theta, qmag(n)*epsilon for q)
  int diffit = 0;       // 1: 6th-order horizontal diffusion (idiff=1, difforder=6)
  int hadvorder = 5;    // hadvordrs
  int vadvorder = 5;    // vadvordrs
};

// Advection operator. Holds the grid/base state it needs and the scratch
// arrays (solve2's dum1..dum8), so repeated calls do not allocate.
class Advection {
public:
  Advection() = default;
  Advection(const Grid& grid, const BaseState& base, const AdvOptions& opt);

  // sten += advective tendency of s (rr0 * flux divergence + s*divx term).
  //   s0: value at the start of the RK step (used by the pdef correction)
  //   s : value advected (halos filled)
  void advs(int nrk, const AdvsArgs& a, Real dt, const AdvFluxes& f,
            const FArray3& s0, const FArray3& s, const FArray3& sten) const;

  // uten(i1:i2,1:nj,1:nk) += advection of u3d (u-staggered).
  void advu(int nrk, Real dt, const AdvFluxes& f, const FArray3& u3d, const FArray3& uten) const;
  // vten(1:ni,j1:j2,1:nk) += advection of v3d (v-staggered).
  void advv(int nrk, Real dt, const AdvFluxes& f, const FArray3& v3d, const FArray3& vten) const;
  // wten(1:ni,1:nj,2:nk) += advection of w3d (w-staggered). solve2 also uses
  // this for TKE (hadvordrs, vadvordrs, advwenos, iamw=.false.).
  void advw(int nrk, Real dt, const AdvFluxes& f, const FArray3& w3d, const FArray3& wten,
            int hadvorder, int vadvorder, int advweno) const;
  void advw(int nrk, Real dt, const AdvFluxes& f, const FArray3& w3d, const FArray3& wten) const {
    advw(nrk, dt, f, w3d, wten, m_opt.hadvordrv, m_opt.vadvordrv, m_opt.advwenov);
  }

  const AdvOptions& options() const { return m_opt; }

  // Scratch arrays (dumx..advz as in adv.F), exposed for tests.
  FArray3 dumx, dumy, dumz, advx, advy, advz;

private:
  Grid m_g;
  BaseState m_b;
  AdvOptions m_opt;
};

// Low-level flux routines (adv_routines.F), exposed for tests.
//   stag: 1 scalar, 2 u, 3 v, 4 w. scheme: order 2..10, or -5 for WENO5.
void hadv_flux(const Grid& g, const AdvOptions& o, int stag, int scheme, int pdef, double weps,
               const FArray3& rru, const FArray3& rrv, const FArray3& a,
               const FArray3& dumx, const FArray3& dumy);
void vadv_flux(const Grid& g, const AdvOptions& o, int stag, int scheme, int pdefweno, double weps,
               const FArray3& rrw, const FArray3& a, const FArray3& dumz);

} // namespace cm1k

#endif
