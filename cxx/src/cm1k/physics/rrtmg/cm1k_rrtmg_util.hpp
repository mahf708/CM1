#ifndef CM1K_RRTMG_UTIL_HPP
#define CM1K_RRTMG_UTIL_HPP

// Building blocks of the RRTMG port (physics/rrtmg):
//
//  * FA<N>/IA<N>: a device-safe Fortran-indexed accessor (pointer, lower
//    bounds, strides) to Real/int data. The k-distribution tables and the
//    per-column work arrays are both FA's, so the ported code indexes them
//    exactly as the Fortran does (absa(ind0,ig), taug(lay,ig), pz(0), ...).
//  * Binders that lay arrays out in a pool: tables are bound once on the
//    host (filled), copied to a device View and bound again to it; column
//    work arrays are re-bound inside every kernel for column c of a chunk of
//    nc columns (element stride nc, so neighbouring columns are adjacent in
//    memory). A layout function `template <class B> void layout(S&, B&)`
//    calls b.arr(member, lb, ub) for every member, in a fixed order.
//  * HView/HArr/fset/TabBinder: host Fortran-indexed arrays and the table
//    layout used by the table initialization.

#include "share/cm1k_types.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

namespace cm1k::rrtmg {

template <typename T, int N>
struct FAcc {
  T* p = nullptr;
  int lb[N] = {};
  long st[N] = {};
  KOKKOS_FORCEINLINE_FUNCTION T& operator()(int i) const requires(N == 1) { return p[(i - lb[0]) * st[0]]; }
  KOKKOS_FORCEINLINE_FUNCTION T& operator()(int i, int j) const requires(N == 2) {
    return p[(i - lb[0]) * st[0] + (j - lb[1]) * st[1]];
  }
  KOKKOS_FORCEINLINE_FUNCTION T& operator()(int i, int j, int k) const requires(N == 3) {
    return p[(i - lb[0]) * st[0] + (j - lb[1]) * st[1] + (k - lb[2]) * st[2]];
  }
  KOKKOS_FORCEINLINE_FUNCTION T& operator()(int i, int j, int k, int l) const requires(N == 4) {
    return p[(i - lb[0]) * st[0] + (j - lb[1]) * st[1] + (k - lb[2]) * st[2] + (l - lb[3]) * st[3]];
  }
};
template <int N> using FA = FAcc<Real, N>;
template <int N> using IA = FAcc<int, N>;

// Sets a's bounds for an array of `count` elements starting at element
// offset `off` of a pool whose element e of column c is at base[c + nc*e].
template <typename T, int N>
KOKKOS_INLINE_FUNCTION void bind_at(FAcc<T, N>& a, T* base, long off, int nc, int c, const int (&lb)[N],
                                    const int (&ub)[N]) {
  a.p = base + c + long(nc) * off;
  long s = nc;
  for (int d = 0; d < N; ++d) {
    a.lb[d] = lb[d];
    a.st[d] = s;
    s *= (ub[d] - lb[d] + 1);
  }
}
template <int N>
KOKKOS_INLINE_FUNCTION long count_of(const int (&lb)[N], const int (&ub)[N]) {
  long s = 1;
  for (int d = 0; d < N; ++d) s *= (ub[d] - lb[d] + 1);
  return s;
}

// Lays arrays out consecutively in a Real pool and an int pool. With
// null bases it only counts (nr, ni = pool sizes per column).
struct Binder {
  Real* rbase = nullptr;
  int* ibase = nullptr;
  int nc = 1, c = 0;
  long nr = 0, ni = 0;
  KOKKOS_INLINE_FUNCTION Binder() = default;
  KOKKOS_INLINE_FUNCTION Binder(Real* r, int* i, int nc_, int c_) : rbase(r), ibase(i), nc(nc_), c(c_) {}
  template <int N>
  KOKKOS_INLINE_FUNCTION void arr(FA<N>& a, const int (&lb)[N], const int (&ub)[N]) {
    bind_at(a, rbase, nr, nc, c, lb, ub);
    nr += count_of(lb, ub);
  }
  template <int N>
  KOKKOS_INLINE_FUNCTION void arr(IA<N>& a, const int (&lb)[N], const int (&ub)[N]) {
    bind_at(a, ibase, ni, nc, c, lb, ub);
    ni += count_of(lb, ub);
  }
};


// Fortran NINT (round half away from zero) and integer powers as gfortran
// expands them at -O2 (x**3 = x*x*x, x**4 = (x*x)*(x*x)).
KOKKOS_INLINE_FUNCTION int NINT(Real x) { return int(Kokkos::round(x)); }
template <int n>
KOKKOS_INLINE_FUNCTION Real POWI(Real x) {
  static_assert(n >= 2 && n <= 4, "POWI: add the gfortran expansion for this power");
  if constexpr (n == 2) return x * x;
  else if constexpr (n == 3) return x * x * x;
  else { const Real x2 = x * x; return x2 * x2; }
}

// Host view of an array with Fortran bounds and order (table setup).
template <typename T = Real>
struct HView {
  T* p = nullptr;
  std::vector<int> lb, ub;
  HView() = default;
  HView(T* p_, std::vector<int> lb_, std::vector<int> ub_) : p(p_), lb(std::move(lb_)), ub(std::move(ub_)) {}
  template <typename... I>
  T& operator()(I... idx) const {
    const int ix[] = {int(idx)...};
    size_t off = 0, s = 1;
    for (size_t d = 0; d < sizeof...(I); ++d) {
      off += size_t(ix[d] - lb[d]) * s;
      s *= size_t(ub[d] - lb[d] + 1);
    }
    return p[off];
  }
  size_t size() const {
    size_t n = 1;
    for (size_t d = 0; d < lb.size(); ++d) n *= size_t(ub[d] - lb[d] + 1);
    return n;
  }
  int extent(int d) const { return ub[d] - lb[d] + 1; }
};
// Owning host array with Fortran bounds (not copyable: the view points into it).
template <typename T = Real>
struct HArr : HView<T> {
  std::vector<T> v;
  HArr() = default;
  HArr(std::vector<int> lb_, std::vector<int> ub_) : HView<T>(nullptr, std::move(lb_), std::move(ub_)) {
    v.assign(this->size(), T(0));
    this->p = v.data();
  }
  HArr(const HArr&) = delete;
  HArr& operator=(const HArr&) = delete;
  HArr(HArr&& o) noexcept : HView<T>(o), v(std::move(o.v)) { this->p = v.data(); }
  HArr& operator=(HArr&& o) noexcept {
    HView<T>::operator=(o); v = std::move(o.v); this->p = v.data(); return *this;
  }
};

// A Fortran array section subscript: S() = ':', S(a, b) = 'a:b', S(i) = 'i'.
struct S {
  int a = 0, b = 0;
  int kind = 0; // 0 full, 1 range, 2 index
  S() = default;
  S(int a_, int b_) : a(a_), b(b_), kind(1) {}
  explicit S(int i) : a(i), b(i), kind(2) {}
};
// Fortran `h(sections) = (/ vals /)`: the values fill the section in array
// element order. Throws if the counts differ.
void fset_impl(void* p, bool is_int, const std::vector<int>& lb, const std::vector<int>& ub,
               std::initializer_list<S> sec, std::initializer_list<double> vals);
template <typename T>
void fset(const HView<T>& h, std::initializer_list<S> sec, std::initializer_list<double> vals) {
  static_assert(std::is_same_v<T, Real> || std::is_same_v<T, int>);
  fset_impl(h.p, std::is_same_v<T, int>, h.lb, h.ub, sec, vals);
}

// Lays out the tables (trivially copyable structs of FA/IA members) in a
// Real and an int pool and records every array by name. Pass 1 (null
// bases) counts; pass 2 binds to host storage; pass 3 binds the device copy.
struct TabBinder {
  struct Reg {
    std::string name;
    bool is_int;
    std::vector<int> lb, ub;
    long off, n;
  };
  Real* rbase = nullptr;
  int* ibase = nullptr;
  long nr = 0, ni = 0;
  std::vector<Reg> regs;
  template <int N>
  void arr(const std::string& name, FA<N>& a, const int (&lb)[N], const int (&ub)[N]) {
    if (rbase) bind_at(a, rbase, nr, 1, 0, lb, ub);
    const long n = count_of(lb, ub);
    regs.push_back({name, false, std::vector<int>(lb, lb + N), std::vector<int>(ub, ub + N), nr, n});
    nr += n;
  }
  template <int N>
  void arr(const std::string& name, IA<N>& a, const int (&lb)[N], const int (&ub)[N]) {
    if (ibase) bind_at(a, ibase, ni, 1, 0, lb, ub);
    const long n = count_of(lb, ub);
    regs.push_back({name, true, std::vector<int>(lb, lb + N), std::vector<int>(ub, ub + N), ni, n});
    ni += n;
  }
  const Reg& find(const std::string& name) const;
  HView<Real> rview(const std::string& name) const;
  HView<int> iview(const std::string& name) const;
};

// Reads a Fortran unformatted sequential big-endian file (gfortran record
// markers: 4-byte length before and after each record).
class FortranRecordReader {
public:
  explicit FortranRecordReader(const std::string& path);
  // Next record; throws if the file is exhausted.
  void next();
  size_t record_bytes() const { return m_rec.size(); }
  // Read consecutive 4-byte big-endian reals (widened to double) / ints.
  void reals(Real* dst, size_t n);
  void ints(int* dst, size_t n);
  int int1() { int v; ints(&v, 1); return v; }
  Real real1() { Real v; reals(&v, 1); return v; }
  template <typename T> void into(const HView<T>& a) {
    if constexpr (std::is_same_v<T, int>) ints(a.p, a.size());
    else reals(a.p, a.size());
  }
  // Checks the whole record was consumed.
  void end_record();

private:
  std::string m_path;
  std::vector<unsigned char> m_data;
  size_t m_pos = 0;
  std::vector<unsigned char> m_rec;
  size_t m_rpos = 0;
};

// Locates an RRTMG data file (RRTMG_LW_DATA, RRTMG_SW_DATA): `dir` if not
// empty, else $CM1K_RRTMG_DATA_DIR, else the working directory (as CM1),
// else the source tree's run/ directory. Throws if not found.
std::string find_data_file(const std::string& name, const std::string& dir = "");

} // namespace cm1k::rrtmg

#endif
