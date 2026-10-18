#ifndef CM1K_OPENBC_HPP
#define CM1K_OPENBC_HPP

// Open-radiative lateral boundary conditions (wbc..nbc = 2): port of the
// radiative phase-speed and open-boundary routines of bc.F (radbcew,
// radbcns, radbcew4, radbcns4, ssopenbcw/e/s/n, restrict_openbc_we/sn), and
// of the open-boundary pieces of solve1.F (irbc=1), solve2.F (nudgeobc,
// irbc=4) and sound.F (irbc=2, ssopenbc*, roflux).
//
// Serial runs only (the `#ifndef MPI` view of the Fortran): ibw = 1 iff
// wbc /= 1, etc. Cartesian grid, with or without terrain (ssopenbc*: the
// terrain branches through OpenBcTerrain).
//
// Options (namelist, with param.F/cm1.F defaults):
//   irbc     = 1 (Klemp-Wilhelmson, large steps), 2 (Klemp-Wilhelmson,
//              small steps), 4 (Durran-Klemp; default). Other values throw
//              when a boundary is open.
//   roflux   = 0 (default) or 1: restrict the outward mass flux.
//   nudgeobc = 0 (default) or 1: nudge inflow u/v at open boundaries to the
//              base state with time scale alphobc (default 60 s).
//   cstar    = phase speed for irbc=1,2 (param.F: values < 0.1 become 30).

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k {

struct OpenBcParams {
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;   // param.F (serial): 1 iff the boundary is not periodic
  int irbc = 4, roflux = 0, nudgeobc = 0;
  Real cstar = 30.0, alphobc = 60.0;
  Real dx = 0, dy = 0, rdx = 0, rdy = 0;
  int ni = 0, nj = 0, nk = 0;

  static OpenBcParams make(const Grid& g, const Namelist& nl);
  // Throws for option values that are not ported.
  void check() const;
  bool any_open() const { return wbc == 2 || ebc == 2 || sbc == 2 || nbc == 2; }
};

// Radiative phase speeds cm1.F carries between time steps:
// radbcw/radbce (jb:je,kb:ke), radbcs/radbcn (ib:ie,kb:ke); zero at start.
struct RadBc {
  FArray2 radbcw, radbce, radbcs, radbcn;
  void allocate(const Bounds& b);
};

// bc.F radbcew / radbcns (irbc = 1, 2): phase speed from ua -/+ cstar.
void radbcew(const OpenBcParams& p, const RadBc& r, const FArray3& ua);
void radbcns(const OpenBcParams& p, const RadBc& r, const FArray3& va);
// bc.F radbcew4 / radbcns4 (irbc = 4): Durran-Klemp phase speed from the
// change u1 -> u2 over dt, vertically averaged.
void radbcew4(const OpenBcParams& p, const FArray1& ruf, const RadBc& r, const FArray3& u1,
              const FArray3& u2, Real dt);
void radbcns4(const OpenBcParams& p, const FArray1& rvf, const RadBc& r, const FArray3& v1,
              const FArray3& v2, Real dt);

// bc.F ssopenbcw/e/s/n (no terrain): small-step update of the normal
// velocity on an open boundary.
void ssopenbcw(const OpenBcParams& p, const FArray1& uh, const RadBc& r, const FArray3& u3d,
               const FArray3& uten, Real dts);
void ssopenbce(const OpenBcParams& p, const FArray1& uh, const RadBc& r, const FArray3& u3d,
               const FArray3& uten, Real dts);
void ssopenbcs(const OpenBcParams& p, const FArray1& vh, const RadBc& r, const FArray3& v3d,
               const FArray3& vten, Real dts);
void ssopenbcn(const OpenBcParams& p, const FArray1& vh, const RadBc& r, const FArray3& v3d,
               const FArray3& vten, Real dts);

// Terrain metrics for the terrain branches of ssopenbcw/e/s/n (the 2D ones
// as (i,j,1)); dum1 is scratch (u or v at w points), cgs*/cgt* the
// extrapolation coefficients of param.F.
struct OpenBcTerrain {
  FArray1 rds, sigma, rdsf, sigmaf;
  FArray3 gz, rgzu, rgzv, gx, gy, dum1;
  Real cgs1 = 0, cgs2 = 0, cgs3 = 0, cgt1 = 0, cgt2 = 0, cgt3 = 0;
};
void ssopenbcw(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& uh, const RadBc& r,
               const FArray3& u3d, const FArray3& uten, Real dts);
void ssopenbce(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& uh, const RadBc& r,
               const FArray3& u3d, const FArray3& uten, Real dts);
void ssopenbcs(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& vh, const RadBc& r,
               const FArray3& v3d, const FArray3& vten, Real dts);
void ssopenbcn(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& vh, const RadBc& r,
               const FArray3& v3d, const FArray3& vten, Real dts);

// bc.F restrict_openbc_we / restrict_openbc_sn (roflux = 1). The per-level
// sums are double precision (as in the Fortran), summed over k on the host.
// `work` is a scratch array with at least nk elements starting at index 1
// (two values per level are stored in it and in `work2`).
void restrict_openbc_we(const OpenBcParams& p, const FArray1& rvh, const FArray3& rmh,
                        const FArray3& rho0, const FArray3& u3d, const FArray1& work,
                        const FArray1& work2);
void restrict_openbc_sn(const OpenBcParams& p, const FArray1& ruh, const FArray3& rmh,
                        const FArray3& rho0, const FArray3& v3d, const FArray1& work,
                        const FArray1& work2);

// The open-boundary block at the start of each acoustic step of sound.F
// (irbc=2 phase speeds, ssopenbcw/e, roflux, ssopenbcs/n, roflux).
struct SoundOpenBc {
  OpenBcParams p;
  RadBc rad;
  FArray1 uh, vh, ruh, rvh, work, work2;
  FArray3 rmh, rho0;
  bool terrain = false;   // terrain_flag: ssopenbc* take the terrain branches
  OpenBcTerrain t;
};
void sound_openbc(const SoundOpenBc& o, const FArray3& u3d, const FArray3& uten, const FArray3& v3d,
                  const FArray3& vten, Real dts);

// solve2.F (nudgeobc = 1): nudge the inflow normal velocity at open
// boundaries back towards the base state, uten/vten -= (u3d-u0)/alphobc.
void nudgeobc_uv(const OpenBcParams& p, const FArray3& u0, const FArray3& v0, const FArray3& u3d,
                 const FArray3& v3d, const FArray3& uten, const FArray3& vten);

} // namespace cm1k

#endif
