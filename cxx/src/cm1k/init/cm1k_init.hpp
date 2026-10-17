#ifndef CM1K_INIT_HPP
#define CM1K_INIT_HPP

#include "init/cm1k_init_bc.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <array>

namespace cm1k {
namespace init {

// Native initialization (what CM1 does from the start of cm1.F up to the
// time loop, for the options listed below). Everything runs on the host:
// the Fortran is one-time, serial and full of column recurrences, and bit
// parity with it is the goal. Device arrays are filled by deep copies.
//
// Supported (anything else throws std::runtime_error):
//   param.F : stretch_x = stretch_y = 0, stretch_z = 0 or 1, terrain_flag
//             = .false., iorigin = 1 or 2, axisymm = 0, wbc..nbc = 1..4
//   base.F  : isnd = 1, 4, 5;  iwnd = 0, 1, 2;  imove = 0 or 1;
//             testcase = 0;  psolver any
//   init3d.F: iinit = 0 or 1, irandp = 0, ibalance = 0 or 1, ptype 0/1
//   cm1.F / getset: kmh/kmv/khh/khv initial values, tkea, rho/prs

// 2nd-order extrapolation coefficients (Fornberg 1988), param.F.
//   cgs*: to the surface from scalar levels 1..3   (nn=1)
//   cgt*: to the model top from levels nk..nk-2    (nn=2)
//   wbe*/wte*: same from w levels (nn=3,4); dgs*/dgt*: 1st derivatives
struct ExtrapCoeffs {
  Real cgs1 = 0, cgs2 = 0, cgs3 = 0, dgs1 = 0, dgs2 = 0, dgs3 = 0;
  Real cgt1 = 0, cgt2 = 0, cgt3 = 0, dgt1 = 0, dgt2 = 0, dgt3 = 0;
  Real wbe1 = 0, wbe2 = 0, wbe3 = 0, wte1 = 0, wte2 = 0, wte3 = 0;
};
ExtrapCoeffs extrap_coeffs(const HArray1& sigma, const HArray1& sigmaf, int nk);

// Scalars that param.F/base.F derive and later stages use.
struct InitInfo {
  BcInfo bc;
  ExtrapCoeffs ex;
  Real minx = 0, maxx = 0, centerx = 0, miny = 0, maxy = 0, centery = 0;
  Real ztop = 0, maxz = 0;
  Real umove = 0, vmove = 0;
};

// param.F grid section: xh..rvf, sigma/sigmaf/rds/rdsf, zh/zf/mh/mf/rmh/rmf,
// c1/c2, and dx/dy/dz/rdx/rdy/rdz.
InitInfo init_grid(const Namelist& nl, Grid& g);

// base.F: pi0, th0, prs0, rho0, thv0, rth0, qv0, qc0, qi0, rr0, rf0, rrf0, u0, v0.
void init_base(const Namelist& nl, const Constants& c, const Grid& g, InitInfo& info, BaseState& bs);

// init3d.F + getset (init3d.F) + the initial values cm1.F gives kmh..khv.
void init_state(const Namelist& nl, const Constants& c, const Grid& g, const InitInfo& info,
                BaseState& bs, State& s);

} // namespace init
} // namespace cm1k

#endif
