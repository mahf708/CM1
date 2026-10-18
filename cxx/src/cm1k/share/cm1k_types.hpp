#ifndef CM1K_TYPES_HPP
#define CM1K_TYPES_HPP

#include <ekat_kokkos_types.hpp>
#include <Kokkos_Core.hpp>

namespace cm1k {

// CM1 is validated in double precision (Fortran built with -fdefault-real-8).
using Real = double;

using KT        = ekat::KokkosTypes<ekat::DefaultDevice>;
using ExeSpace  = KT::ExeSpace;
using MemSpace  = KT::MemSpace;
using Device    = KT::Device;
using HostSpace = Kokkos::HostSpace;

// Fortran-ordered (i fastest) device views. CM1 stores every array as
// a(i,j,k) with i varying fastest; LayoutLeft keeps the exact same memory
// order, so raw Fortran dumps load without transposition and numpy sees
// Fortran-ordered arrays.
template <typename DT>
using LView = Kokkos::View<DT, Kokkos::LayoutLeft, Device>;

// 3D loop over a Fortran index box [i1,i2] x [j1,j2] x [k1,k2] (inclusive),
// with i fastest. Use this for the ~2,100 `!$omp parallel do` nests in CM1.
using Policy3 = Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, Kokkos::IndexType<int>>;
using Policy2 = Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, Kokkos::IndexType<int>>;

inline Policy3 box3(int i1, int i2, int j1, int j2, int k1, int k2) {
  return Policy3({i1, j1, k1}, {i2 + 1, j2 + 1, k2 + 1});
}
inline Policy2 box2(int i1, int i2, int j1, int j2) {
  return Policy2({i1, j1}, {i2 + 1, j2 + 1});
}

} // namespace cm1k

#endif
