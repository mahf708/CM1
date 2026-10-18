#ifndef CM1K_CM1LIBS_HPP
#define CM1K_CM1LIBS_HPP

#include "share/cm1k_types.hpp"

namespace cm1k {
namespace cm1libs {

// Port of RSLF / RSIF from src/cm1libs.F (module cm1libs): saturation
// mixing ratio over liquid / ice, in kg/kg, for pressure p (Pa) and
// temperature t (K). `eps` is Constants::eps (rd/rv). Same operation order
// as the Fortran, so results are bit-for-bit with gfortran on the same libm.
KOKKOS_INLINE_FUNCTION Real rslf(int ptype, Real eps, Real p, Real t) {
  if (ptype == 4) {
    return 380.00 * Kokkos::exp(17.2693882 - 4097.8531 / (t - 35.86)) / p;
  }
  // ptype = 1,2,3,5,6 and the default branch are identical: Bolton (1980)
  Real esl = 611.2 * Kokkos::exp(17.67 * (t - 273.15) / (t - 29.65));
  // 171023 (fix for very cold temps):
  const Real half_p = p * 0.5;
  esl = (esl < half_p) ? esl : half_p;   // Fortran min(esl, p*0.5)
  return eps * esl / (p - esl);
}

KOKKOS_INLINE_FUNCTION Real rsif(int ptype, Real eps, Real p, Real t) {
  if (ptype == 4) {
    return 380.00 * Kokkos::exp(21.87455 - 5807.4743 / (t - 7.66)) / p;
  }
  // Tao et al (1989)
  Real esi = 611.2 * Kokkos::exp(21.8745584 * (t - 273.15) / (t - 7.66));
  const Real half_p = p * 0.5;
  esi = (esi < half_p) ? esi : half_p;
  return eps * esi / (p - esi);
}

} // namespace cm1libs
} // namespace cm1k

#endif
