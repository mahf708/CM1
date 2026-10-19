#ifndef CM1K_BC_HPP
#define CM1K_BC_HPP

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"

namespace cm1k {

// Lateral/vertical boundary conditions: port of bc.F (bcs, bcp, bcu, bcv,
// bcw). Serial runs take the `#ifndef MPI` branches: the periodic case
// (wbc=1, ...) copies across the domain. MPI runs (mpi = true) take the
// `#ifdef MPI` ones: no periodic copy (comm.F's exchanges fill those
// halos), and the open/wall branches only on ranks at that physical
// boundary (ibw=1, ...).
//
// Each Fortran loop nest is one parallel_for, executed in the same order as
// the Fortran (west, east, south, north), so the corner points end up with
// exactly the same values.
struct BcParams {
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int axisymm = 0;
  int ni = 0, nj = 0, nk = 0, ngxy = 0;
  int ib = 0, ie = 0, jb = 0, je = 0;
  // MPI (param.F): ib<side> = 1 on a rank at a physical boundary
  bool mpi = false;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;
  // the non-periodic branch of a side applies (serial: always)
  bool on_w() const { return !mpi || ibw == 1; }
  bool on_e() const { return !mpi || ibe == 1; }
  bool on_s() const { return !mpi || ibs == 1; }
  bool on_n() const { return !mpi || ibn == 1; }

  static BcParams make(const Bounds& b, const Namelist& nl);
  // MPI: the rank's boundary flags (Comm::edges())
  void set_mpi(const EdgeFlags& e);
};

// scalars (s at ib:ie,jb:je,kb:ke), all ngxy halo points
void bcs(const BcParams& p, const FArray3& s);
// pressure-like scalars: one halo point only
void bcp(const BcParams& p, const FArray3& s);
// u (ib:ie+1,jb:je,kb:ke)
void bcu(const BcParams& p, const FArray3& u);
// v (ib:ie,jb:je+1,kb:ke)
void bcv(const BcParams& p, const FArray3& v);
// w (ib:ie,jb:je,kb:ke+1); flag=1 also zeroes w at k=1 and k=nk+1
void bcw(const BcParams& p, const FArray3& w, int flag);

// MPI only (bc.F bcs2, bct2, bcu2, bcv2, bcw2, bcs2_2d): halo corner points
// next to a physical boundary (param.F's patch* flags). No-ops in serial
// runs and on ranks away from open boundaries.
void bcs2(const BcParams& p, const FArray3& s);
void bct2(const BcParams& p, const FArray3& t);
void bcu2(const BcParams& p, const FArray3& u);
void bcv2(const BcParams& p, const FArray3& v);
void bcw2(const BcParams& p, const FArray3& w);
// 2D array stored as level k of an FArray3
void bcs2_2d(const BcParams& p, const FArray3& s, int k);

} // namespace cm1k

#endif
