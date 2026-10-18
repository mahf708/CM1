#ifndef CM1K_FARRAY_HPP
#define CM1K_FARRAY_HPP

#include "cm1k_types.hpp"

#include <array>
#include <string>

namespace cm1k {

namespace impl {
template <typename T, int N> struct AddPtr { using type = typename AddPtr<T, N - 1>::type*; };
template <typename T> struct AddPtr<T, 0> { using type = T; };
} // namespace impl

// A device array with Fortran index bounds, e.g. ua(ib:ie+1, jb:je, kb:ke).
//
//   FArray<3> ua("ua", {ib, jb, kb}, {ie + 1, je, ke});
//   Kokkos::parallel_for(box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
//     ua(i, j, k) = ...;   // same indices as the Fortran
//   });
//
// FArray is a lightweight handle (copies share data, like Kokkos::View), so
// capture it by value in lambdas.
template <int Rank, typename Space = Device>
class FArray {
public:
  static_assert(Rank >= 1 && Rank <= 4, "FArray supports rank 1..4");
  using data_type = typename impl::AddPtr<Real, Rank>::type;
  using view_type = Kokkos::View<data_type, Kokkos::LayoutLeft, Space>;

  FArray() = default;

  FArray(const std::string& name, const std::array<int, Rank>& lb, const std::array<int, Rank>& ub)
  {
    for (int d = 0; d < Rank; ++d) { m_lb[d] = lb[d]; m_ub[d] = ub[d]; }
    alloc(name);
  }

  // Wrap an existing view (e.g. a host mirror) with the given lower bounds.
  FArray(const view_type& v, const std::array<int, Rank>& lb) : m_view(v)
  {
    for (int d = 0; d < Rank; ++d) { m_lb[d] = lb[d]; m_ub[d] = lb[d] + int(v.extent(d)) - 1; }
  }

  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int i) const
    requires(Rank == 1) { return m_view(i - m_lb[0]); }
  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int i, int j) const
    requires(Rank == 2) { return m_view(i - m_lb[0], j - m_lb[1]); }
  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int i, int j, int k) const
    requires(Rank == 3) { return m_view(i - m_lb[0], j - m_lb[1], k - m_lb[2]); }
  KOKKOS_FORCEINLINE_FUNCTION Real& operator()(int i, int j, int k, int n) const
    requires(Rank == 4) { return m_view(i - m_lb[0], j - m_lb[1], k - m_lb[2], n - m_lb[3]); }

  KOKKOS_FORCEINLINE_FUNCTION int lb(int d) const { return m_lb[d]; }
  KOKKOS_FORCEINLINE_FUNCTION int ub(int d) const { return m_ub[d]; }
  KOKKOS_FORCEINLINE_FUNCTION int extent(int d) const { return m_ub[d] - m_lb[d] + 1; }
  size_t size() const { return m_view.size(); }
  bool is_allocated() const { return m_view.is_allocated(); }
  std::string label() const { return m_view.label(); }
  Real* data() const { return m_view.data(); }

  const view_type& view() const { return m_view; }

  // Host copy with the same bounds (allocates; use for I/O and tests).
  FArray<Rank, HostSpace> to_host() const {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), m_view);
    std::array<int, Rank> lb;
    for (int d = 0; d < Rank; ++d) lb[d] = m_lb[d];
    return FArray<Rank, HostSpace>(h, lb);
  }

  template <typename OtherSpace>
  void deep_copy_from(const FArray<Rank, OtherSpace>& src) const { Kokkos::deep_copy(m_view, src.view()); }

  void fill(Real v) const { Kokkos::deep_copy(m_view, v); }

private:
  void alloc(const std::string& name) {
    if constexpr (Rank == 1) m_view = view_type(name, extent(0));
    if constexpr (Rank == 2) m_view = view_type(name, extent(0), extent(1));
    if constexpr (Rank == 3) m_view = view_type(name, extent(0), extent(1), extent(2));
    if constexpr (Rank == 4) m_view = view_type(name, extent(0), extent(1), extent(2), extent(3));
  }

  view_type m_view;
  Kokkos::Array<int, Rank> m_lb{}, m_ub{};
};

using FArray1 = FArray<1>;
using FArray2 = FArray<2>;
using FArray3 = FArray<3>;
using FArray4 = FArray<4>;

} // namespace cm1k

#endif
