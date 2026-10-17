#ifndef CM1K_ADV_ROUTINES_HPP
#define CM1K_ADV_ROUTINES_HPP

// Device-side flux formulas of src/adv_routines.F (flx3..flx10, weno3,
// weno5, upstrpd) and a generic stencil helper that replaces the Fortran's
// 9 horizontal + 9 vertical near-copies (hadv_flxN, vadv_flxN, vadv_lwr_*).
//
// Round-off parity: every formula keeps the Fortran operation order and
// parenthesization; real literals are doubles (the reference is built with
// -fdefault-real-8), e.g. (2.0/60.0) is folded exactly as gfortran does.

#include "share/cm1k_farray.hpp"

namespace cm1k {
namespace adv {

// scheme ids: 2..10 = flux-form order (odd: upwind-biased, even: centered),
// WENO5 = 5th-order WENO (Borges et al. smoothness indicators, siform=2).
constexpr int WENO5 = -5;

KOKKOS_FORCEINLINE_FUNCTION Real fmax2(Real a, Real b) { return (a > b) ? a : b; }
KOKKOS_FORCEINLINE_FUNCTION Real fmin2(Real a, Real b) { return (a < b) ? a : b; }

KOKKOS_FORCEINLINE_FUNCTION Real flx3(Real s1, Real s2, Real s3) {
  return ((-1.0/6.0)*s1
         +( 5.0/6.0)*s2
         +( 2.0/6.0)*s3);
}
KOKKOS_FORCEINLINE_FUNCTION Real flx4(Real s1, Real s2, Real s3, Real s4) {
  return ((7.0/12.0)*(s3+s2)
         -(1.0/12.0)*(s4+s1));
}
KOKKOS_FORCEINLINE_FUNCTION Real flx5(Real s1, Real s2, Real s3, Real s4, Real s5) {
  return ((  2.0/60.0)*s1
         +(-13.0/60.0)*s2
         +( 47.0/60.0)*s3
         +( 27.0/60.0)*s4
         +( -3.0/60.0)*s5);
}
KOKKOS_FORCEINLINE_FUNCTION Real flx6(Real s1, Real s2, Real s3, Real s4, Real s5, Real s6) {
  return ((37.0/60.0)*(s4+s3)
         +(-8.0/60.0)*(s5+s2)
         +( 1.0/60.0)*(s6+s1));
}
KOKKOS_FORCEINLINE_FUNCTION Real flx7(Real s1, Real s2, Real s3, Real s4, Real s5, Real s6, Real s7) {
  return ((  -3.0/420.0)*s1
         +(  25.0/420.0)*s2
         +(-101.0/420.0)*s3
         +( 319.0/420.0)*s4
         +( 214.0/420.0)*s5
         +( -38.0/420.0)*s6
         +(   4.0/420.0)*s7);
}
KOKKOS_FORCEINLINE_FUNCTION Real flx8(Real s1, Real s2, Real s3, Real s4, Real s5, Real s6, Real s7, Real s8) {
  return (( 533.0/840.0)*(s5+s4)
         +(-139.0/840.0)*(s6+s3)
         +(  29.0/840.0)*(s7+s2)
         +(  -3.0/840.0)*(s8+s1));
}
KOKKOS_FORCEINLINE_FUNCTION Real flx9(Real s1, Real s2, Real s3, Real s4, Real s5, Real s6, Real s7, Real s8, Real s9) {
  return ((   12.0/7560.0)*s1
         +( -123.0/7560.0)*s2
         +(  597.0/7560.0)*s3
         +(-1923.0/7560.0)*s4
         +( 5637.0/7560.0)*s5
         +( 4125.0/7560.0)*s6
         +( -915.0/7560.0)*s7
         +(  165.0/7560.0)*s8
         +(  -15.0/7560.0)*s9);
}
KOKKOS_FORCEINLINE_FUNCTION Real flx10(Real s1, Real s2, Real s3, Real s4, Real s5, Real s6, Real s7, Real s8, Real s9, Real s10) {
  return (( 4881.0/7560.0)*( s6+s5)
         +(-1419.0/7560.0)*( s7+s4)
         +(  381.0/7560.0)*( s8+s3)
         +(  -69.0/7560.0)*( s9+s2)
         +(   +6.0/7560.0)*(s10+s1));
}

// weno5 (siform = 2)
KOKKOS_FORCEINLINE_FUNCTION Real weno5(Real s1, Real s2, Real s3, Real s4, Real s5, double weps) {
  const double t1 = s1 -2.0*s2 +s3, u1 = s1 -4.0*s2 +3.0*s3;
  const double t2 = s2 -2.0*s3 +s4, u2 = s2 -s4;
  const double t3 = s3 -2.0*s4 +s5, u3 = 3.0*s3 -4.0*s4 +s5;
  const double b1 = (13.0/12.0)*(t1*t1) + 0.25*(u1*u1);
  const double b2 = (13.0/12.0)*(t2*t2) + 0.25*(u2*u2);
  const double b3 = (13.0/12.0)*(t3*t3) + 0.25*(u3*u3);
  const double d13 = Kokkos::fabs(b1-b3);
  const double q1 = fmin2(1.0e30, d13/(b1+weps));
  const double q2 = fmin2(1.0e30, d13/(b2+weps));
  const double q3 = fmin2(1.0e30, d13/(b3+weps));
  const double w1 = 0.1*(1.0+q1*q1);
  const double w2 = 0.6*(1.0+q2*q2);
  const double w3 = 0.3*(1.0+q3*q3);
  return ( w1*( ( 2.0/6.0)*s1 + (-7.0/6.0)*s2 + (11.0/6.0)*s3 )
          +w2*( (-1.0/6.0)*s2 + ( 5.0/6.0)*s3 + ( 2.0/6.0)*s4 )
          +w3*( ( 2.0/6.0)*s3 + ( 5.0/6.0)*s4 + (-1.0/6.0)*s5 )
         )/( w1+w2+w3 );
}

// Positive-definite upstream scheme (Beets & Koren 1996).
KOKKOS_FORCEINLINE_FUNCTION Real upstrpd(Real s1, Real s2, Real s3, double weps) {
  const Real dd = s2-s1;
  const Real rr = (s3-s2)/(Kokkos::copysign(Real(weps), dd)+dd);
  const Real phi = fmax2(0.0, fmin2(rr, fmin2((1.0/6.0)+(2.0/6.0)*rr, 1.0)));
  return s2 + phi*(s2-s1);
}

// Flux function of scheme S on N=|order| stencil values s[0..N-1].
template <int S>
KOKKOS_FORCEINLINE_FUNCTION Real flxS(const Real* s, double weps) {
  if constexpr (S == 3) return flx3(s[0], s[1], s[2]);
  else if constexpr (S == 4) return flx4(s[0], s[1], s[2], s[3]);
  else if constexpr (S == 5) return flx5(s[0], s[1], s[2], s[3], s[4]);
  else if constexpr (S == 6) return flx6(s[0], s[1], s[2], s[3], s[4], s[5]);
  else if constexpr (S == 7) return flx7(s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
  else if constexpr (S == 8) return flx8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
  else if constexpr (S == 9) return flx9(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
  else if constexpr (S == 10) return flx10(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]);
  else if constexpr (S == WENO5) return weno5(s[0], s[1], s[2], s[3], s[4], weps);
  else { static_assert(S == 3, "unsupported scheme"); return 0; }
}

template <int S> KOKKOS_INLINE_FUNCTION constexpr int stencil_width() { return S == WENO5 ? 5 : S; }

// Stencil values along a line: a(L + n*d), d = unit step in i, j or k.
struct Line {
  FArray3 a;
  int i, j, k, di, dj, dk;
  KOKKOS_FORCEINLINE_FUNCTION Real operator()(int n) const { return a(i + n*di, j + n*dj, k + n*dk); }
};

// Flux through the face between cells L (n=0) and L+1 (n=1) of the line, for
// face velocity vel: upwind-biased for odd orders/WENO5, centered for even.
//   odd 2m+1, vel>=0: f(a(L-m), ..., a(L+m))
//             vel< 0: f(a(L+1+m), ..., a(L+1-m))
//   even 2m         : f(a(L-m+1), ..., a(L+m))
// For S=2: vel*0.5*(a(L)+a(L+1)) (hadv_flx2).
// pdefchk (hadv_weno5 with pdef=1): zero flux if max|s| <= weps.
template <int S>
KOKKOS_FORCEINLINE_FUNCTION Real face_flux(Real vel, const Line& l, double weps, bool pdefchk = false) {
  if constexpr (S == 2) {
    return vel*0.5*(l(0)+l(1));
  } else {
    constexpr int N = stencil_width<S>();
    Real s[N];
    if constexpr (N % 2 == 1) {
      constexpr int m = (N - 1) / 2;
      if (vel >= 0.0) { for (int n = 0; n < N; ++n) s[n] = l(n - m); }
      else            { for (int n = 0; n < N; ++n) s[n] = l(1 + m - n); }
    } else {
      constexpr int m = N / 2;
      for (int n = 0; n < N; ++n) s[n] = l(n - m + 1);
    }
    if constexpr (S == WENO5) {
      if (pdefchk) {
        Real bmax = Kokkos::fabs(s[0]);
        for (int n = 1; n < N; ++n) bmax = fmax2(bmax, Kokkos::fabs(s[n]));
        if (!(bmax > weps)) return 0.0;
      }
    }
    return vel*flxS<S>(s, weps);
  }
}

KOKKOS_FORCEINLINE_FUNCTION Real flx_order(int p, const Real* s) {
  switch (p) {
    case 3:  return flx3(s[0], s[1], s[2]);
    case 4:  return flx4(s[0], s[1], s[2], s[3]);
    case 5:  return flx5(s[0], s[1], s[2], s[3], s[4]);
    case 6:  return flx6(s[0], s[1], s[2], s[3], s[4], s[5]);
    case 7:  return flx7(s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    case 8:  return flx8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
    case 9:  return flx9(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
    default: return flx10(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]);
  }
}

// Vertical flux of order S (2..10) through the face between levels L and
// L+1, with the order reduced near the bottom/top exactly as vadv_flxN +
// vadv_lwr_flx_* do: the highest order (same parity) whose stencil lies in
// [lo,hi] is used, down to 3 (odd) or 4 (even); below that the 2nd-order
// interpolation   vel*(w1*a(L)+w2*a(L+1))        (stag 1,2,3: c1/c2 weights)
//              or vel*0.5*(a(L)+a(L+1))          (stag 4, half=true).
template <int S>
KOKKOS_FORCEINLINE_FUNCTION Real vface_flux(Real vel, const Line& l, int L, int lo, int hi,
                                            Real w1, Real w2, bool half) {
  Real s[10];
  if constexpr (S % 2 == 1) {
    const bool pos = (vel >= 0.0);
    for (int m = (S - 1) / 2; m >= 1; --m) {
      if (pos) {
        if (L - m >= lo && L + m <= hi) {
          for (int n = 0; n <= 2*m; ++n) s[n] = l(n - m);
          return vel*flx_order(2*m + 1, s);
        }
      } else {
        if (L + 1 + m <= hi && L + 1 - m >= lo) {
          for (int n = 0; n <= 2*m; ++n) s[n] = l(1 + m - n);
          return vel*flx_order(2*m + 1, s);
        }
      }
    }
  } else {
    for (int m = S / 2; m >= 2; --m) {
      if (L - m + 1 >= lo && L + m <= hi) {
        for (int n = 0; n < 2*m; ++n) s[n] = l(n - m + 1);
        return vel*flx_order(2*m, s);
      }
    }
  }
  if (half) return vel*0.5*(l(0)+l(1));
  return vel*(w1*l(0)+w2*l(1));
}

} // namespace adv
} // namespace cm1k

#endif
