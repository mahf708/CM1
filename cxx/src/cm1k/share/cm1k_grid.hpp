#ifndef CM1K_GRID_HPP
#define CM1K_GRID_HPP

#include "cm1k_farray.hpp"
#include "cm1k_namelist.hpp"

namespace cm1k {

// Index bounds, named exactly as in CM1's `input` module (serial run, so
// ni=nx, nj=ny, nk=nz). Arrays are allocated as in cm1.F, e.g.
//   ua(ib:ie+1,jb:je,kb:ke), wa(ib:ie,jb:je,kb:ke+1), qa(ibm:iem,...,numq),
//   tkea(ibt:iet,jbt:jet,kbt:ket), kmh(ibc:iec,jbc:jec,kbc:kec).
struct Bounds {
  int nx = 0, ny = 0, nz = 0;
  int ni = 0, nj = 0, nk = 0, nkp1 = 0;
  int ngxy = 0, ngz = 0;
  int ib = 0, ie = 0, jb = 0, je = 0, kb = 0, ke = 0;
  int ibm = 1, iem = 1, jbm = 1, jem = 1, kbm = 1, kem = 1;   // moisture
  int ibt = 1, iet = 1, jbt = 1, jet = 1, kbt = 1, ket = 1;   // tke
  int ibc = 1, iec = 1, jbc = 1, jec = 1, kbc = 1, kec = 1;   // eddy viscosities
  int numq = 1, nrain = 1;
  // terrain (cm1.F): itb:ite, jtb:jte, ktb:kte = ib:ie, jb:je, kb:ke with
  // terrain_flag, else 1:1 (the terrain metrics are then never read)
  bool terrain_flag = false;
  int itb = 1, ite = 1, jtb = 1, jte = 1, ktb = 1, kte = 1;

  // Same logic as cm1.F (ngxy/ngz) and param.F (ibm/ibt/ibc, numq).
  static Bounds from_namelist(const Namelist& nl);
};

// Grid metrics (port of the grid parts of param.F). For now only the arrays
// the dry dynamics needs; all names/shapes match cm1.F.
struct Grid {
  Bounds b;
  Real dx = 0, dy = 0, dz = 0, rdx = 0, rdy = 0, rdz = 0;

  FArray1 xh, rxh, uh, ruh, xf, rxf, uf, ruf;   // (ib:ie), (ib:ie+1)
  FArray1 yh, vh, rvh, yf, vf, rvf;             // (jb:je), (jb:je+1)
  FArray1 sigma, rds, sigmaf, rdsf;             // (kb:ke), (kb:ke+1)
  FArray3 zh, mh, rmh, c1, c2;                  // (ib:ie,jb:je,kb:ke)
  FArray3 zf, mf, rmf;                          // (ib:ie,jb:je,kb:ke+1)

  // terrain (param.F / init_terrain.F). The Fortran 2D arrays are stored
  // with a third dimension 1:1 (as in the dumps): zs(i,j) is zs(i,j,1).
  //   zs (ib:ie,jb:je), gz rgz gzu rgzu gzv rgzv dzdx dzdy (itb:ite,jtb:jte)
  //   gx gxu gy gyv (itb:ite,jtb:jte,ktb:kte)
  // zt = rzt = 0 without terrain (as in the Fortran: never used then).
  Real zt = 0, rzt = 0;
  // param.F 2nd-order extrapolation coefficients to the surface (cgs*) and
  // the top (cgt*) (init::extrap_coeffs; set by Model::init and
  // Model::init_from_dump; the terrain branches of turb read them here)
  Real cgs1 = 0, cgs2 = 0, cgs3 = 0, cgt1 = 0, cgt2 = 0, cgt3 = 0;
  FArray3 zs, gz, rgz, gzu, rgzu, gzv, rgzv, dzdx, dzdy;
  FArray3 gx, gxu, gy, gyv;

  void allocate(const Bounds& bounds);
};

} // namespace cm1k

#endif
