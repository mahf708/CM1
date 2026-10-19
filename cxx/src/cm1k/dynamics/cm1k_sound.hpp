#ifndef CM1K_SOUND_HPP
#define CM1K_SOUND_HPP

#include "dynamics/cm1k_bc.hpp"
#include "dynamics/cm1k_openbc.hpp"
#include "share/cm1k_farray.hpp"

namespace cm1k {

// Port of sound.F (psolver=3): acoustic sub-steps, horizontally explicit,
// vertically implicit, with divergence damping (kdiv). Cartesian grid,
// with or without terrain (terrain=true: the terrain-following branches),
// no updraft nudging / convergence forcing / immersed boundaries
// (callers check the options; see DynConfig). Open-radiative lateral
// boundaries (wbc..nbc = 2) through `obc` (cm1k_openbc.hpp).
//
// Argument names follow the Fortran (sound.F); the work arrays are the
// ones solve2 passes in (dum1, dum2, mm, ppd, fpk, qk, pk1, pk2).
struct SoundArgs {
  // scalars
  Real dt = 0, dttmp = 0, alph = 0, kdiv = 0, cp = 0, rdx = 0, rdy = 0, rdz = 0, smeps = 0;
  int nsound = 0, nrk = 0, nrkmax = 3;
  // grid / base state
  FArray1 uh, vh, uf, vf;
  FArray3 mh, mf, c1, c2, rr0, rf0;
  FArray2 dtu, dtv, dtu0, dtv0;
  // work
  FArray3 dum1, dum2, mm, ppd, fpk, qk, pk1, pk2;
  // fields
  FArray3 rru, ua, u3d, uten;
  FArray3 rrv, va, v3d, vten;
  FArray3 rrw, wa, w3d, wten;
  FArray3 ppi, pp3d, piadv, ppten, ppx;
  FArray3 thv, ppterm;
  // terrain (terrain_flag): metrics (2D ones as (i,j,1)), rho0/rrf0, the
  // scratch dum3 and the open-boundary flags (ibw..ibn)
  bool terrain = false;
  Real zt = 0, rzt = 0;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;
  FArray1 rds, rdsf, sigma, sigmaf;
  FArray3 gz, rgzu, rgzv, dzdx, dzdy, gxu, gyv, rho0, rrf0, dum3;
  // open boundaries: phase speeds, options and arrays of the open-bc block
  // at the start of each acoustic step (nullptr: no open boundary)
  const SoundOpenBc* obc = nullptr;
};

void sound(const BcParams& bc, const SoundArgs& a);

// misclibs.F dtsmall: nsound for the large step dbldt. With adapt_dt, an
// nsound above 192 is capped and dbldt shortened to 192 small steps (the
// caller then sets dt = dbldt); otherwise that throws.
int dtsmall(int psolver, int nx, int ny, Real min_dx, Real min_dy, Real min_dz, Real csound, double& dbldt,
            bool adapt_dt);
// Number of acoustic steps (dtsmall in misclibs.F) for psolver=2,3,6,7.
// min_dx/min_dy/min_dz as in param.F.
int sound_nsound(int psolver, int nx, int ny, Real min_dx, Real min_dy, Real min_dz,
                 Real csound, double dbldt);

} // namespace cm1k

#endif
