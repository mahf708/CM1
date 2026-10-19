#ifndef CM1K_GRID_HPP
#define CM1K_GRID_HPP

#include "cm1k_comm.hpp"
#include "cm1k_farray.hpp"
#include "cm1k_namelist.hpp"

namespace cm1k {

// Index bounds, named exactly as in CM1's `input` module (serial run:
// ni=nx, nj=ny; MPI: this rank's ni, nj from its Decomp; nk=nz). Arrays
// are allocated as in cm1.F, e.g.
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

  // Same logic as cm1.F (ngxy/ngz) and param.F (ibm/ibt/ibc, numq).
  static Bounds from_namelist(const Namelist& nl);
  // MPI: the local bounds of one rank (ni = d.ni, nj = d.nj).
  static Bounds from_namelist(const Namelist& nl, const Decomp& d);
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

  void allocate(const Bounds& bounds);
};

} // namespace cm1k

#endif
