#ifndef CM1K_RRTMG_COMMON_HPP
#define CM1K_RRTMG_COMMON_HPP

// Routines RRTMG_LWRAD and RRTMG_SWRAD share (module_ra_rrtmg_lw.F; the SW
// driver uses them from that module): the KISS generator of
// mcica_subcol_gen_lw/sw (kissvec, irng = 0), INIRAD/O3DATA (usecode = 2),
// relcalc and reicalc. One column (ncol = pcols = 1).

#include "physics/rrtmg/cm1k_rrtmg_util.hpp"

namespace cm1k::rrtmg {

// kissvec for one column: 32-bit integer arithmetic wrapping as the
// Fortran's does.
KOKKOS_INLINE_FUNCTION int kiss_m(int k, int n) {
  const uint32_t u = uint32_t(k);
  return int(u ^ (n >= 0 ? (u << n) : (u >> (-n))));
}
KOKKOS_INLINE_FUNCTION void kissvec(int& seed1, int& seed2, int& seed3, int& seed4, Real& ran_arr) {
  seed1 = int(69069u * uint32_t(seed1) + 1327217885u);
  seed2 = kiss_m(kiss_m(kiss_m(seed2, 13), -17), 5);
  seed3 = int(18000u * (uint32_t(seed3) & 65535u) + (uint32_t(seed3) >> 16));
  seed4 = int(30903u * (uint32_t(seed4) & 65535u) + (uint32_t(seed4) >> 16));
  const int kiss = int(uint32_t(seed1) + uint32_t(seed2) + (uint32_t(seed3) << 16) + uint32_t(seed4));
  ran_arr = kiss * 2.328306e-10 + 0.5;
}

// INIRAD + O3DATA (usecode = 2): O3PROF(kts:kte+1) from Plev(1,kts:kte+2)
// (ncol = 1, so the 2-D actual argument is the Fortran's 1-D dummy).
// Returns false where O3DATA stops (frac out of range, stop 1231).
KOKKOS_INLINE_FUNCTION bool inirad(const FA<1>& O3PROF, const FA<2>& Plev2, int kts, int kte, const FA<1>& O3SUM,
                                   const FA<1>& PPSUM, const FA<1>& O3WIN, const FA<1>& PPWIN) {
  for (int K = kts; K <= kte + 1; ++K) O3PROF(K) = 0.;
  Real PPWRKH[33], O3WRK[32], PPWRK[32], O3ANN[32], PPANN[32];
  Real frac;
  int K, kk;
  for (K = 1; K <= 31; ++K) PPANN[K] = PPSUM(K);
  O3ANN[1] = 0.5 * (O3SUM(1) + O3WIN(1));
  for (K = 2; K <= 31; ++K)
    O3ANN[K] = O3WIN(K - 1) + (O3WIN(K) - O3WIN(K - 1)) / (PPWIN(K) - PPWIN(K - 1)) * (PPSUM(K) - PPWIN(K - 1));
  for (K = 2; K <= 31; ++K) O3ANN[K] = 0.5 * (O3ANN[K] + O3SUM(K));
  for (K = 1; K <= 31; ++K) {
    O3WRK[K] = O3ANN[K];
    PPWRK[K] = PPANN[K];
  }
  PPWRKH[1] = 1100.;
  for (K = 2; K <= 31; ++K) PPWRKH[K] = (PPWRK[K] + PPWRK[K - 1]) / 2.;
  PPWRKH[32] = 0.;
  for (int k = kts; k <= kte + 1; ++k) {
    const Real plevk = Plev2(1, k);
    if (plevk > PPWRKH[1]) {
      kk = 1;
      frac = (plevk - PPWRKH[kk]) / (PPWRKH[kk + 1] - PPWRKH[kk]);
      O3PROF(k) = O3WRK[kk] + (O3WRK[kk + 1] - O3WRK[kk]) * frac;
    } else if (plevk < PPWRKH[31]) {
      kk = 31;
      frac = (plevk - PPWRKH[kk]) / (PPWRKH[kk] - PPWRKH[kk - 1]);
      O3PROF(k) = O3WRK[kk] + (O3WRK[kk] - O3WRK[kk - 1]) * frac;
    } else {
      kk = 1;
      while (PPWRKH[kk] > plevk && kk < 31) kk = kk + 1;
      frac = (plevk - PPWRKH[kk - 1]) / (PPWRKH[kk] - PPWRKH[kk - 1]);
      O3PROF(k) = O3WRK[kk - 1] + (O3WRK[kk] - O3WRK[kk - 1]) * frac;
      if (frac < -0.0001 || frac > 1.0001) return false;
    }
  }
  return true;
}

// relcalc (ncol = pcols = 1): liquid effective radius from t (= tlay)
KOKKOS_INLINE_FUNCTION void relcalc(int pver, const FA<2>& t, Real landfrac, Real landm, Real icefrac,
                                    const FA<2>& rel, Real snowh) {
  const Real tmelt = 273.16, rliqocean = 14.0, rliqice = 14.0, rliqland = 8.0;
  for (int k = 1; k <= pver; ++k) {
    const int i = 1;
    rel(i, k) = rliqland + (rliqocean - rliqland) * Kokkos::min(1.0, Kokkos::max(0.0, (tmelt - t(i, k)) * 0.05));
    rel(i, k) = rel(i, k) + (rliqocean - rel(i, k)) * Kokkos::min(1.0, Kokkos::max(0.0, snowh * 10.));
    rel(i, k) = rel(i, k) + (rliqocean - rel(i, k)) * Kokkos::min(1.0, Kokkos::max(0.0, 1.0 - landm));
    rel(i, k) = rel(i, k) + (rliqice - rel(i, k)) * Kokkos::min(1.0, Kokkos::max(0.0, icefrac));
  }
}

// reicalc (ncol = pcols = 1): ice effective radius from t (= tlay)
KOKKOS_INLINE_FUNCTION void reicalc(int pver, const FA<2>& t, const FA<2>& re, const FA<1>& retab) {
  for (int k = 1; k <= pver; ++k) {
    const int i = 1;
    int index = int(t(i, k) - 179.);
    index = Kokkos::min(Kokkos::max(index, 1), 94);
    const Real corr = t(i, k) - Real(int(t(i, k)));
    re(i, k) = retab(index) * (1. - corr) + retab(index + 1) * corr;
  }
}

} // namespace cm1k::rrtmg

#endif
