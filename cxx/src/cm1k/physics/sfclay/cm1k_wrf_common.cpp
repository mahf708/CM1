// turb.F preparation of the WRF-physics inputs. See cm1k_wrf_common.hpp.
#include "physics/sfclay/cm1k_wrf_common.hpp"

#include <stdexcept>

namespace cm1k::wrfphys {

void pbl_prep(const PrepIn& in, const FArray3& dum1, const FArray3& dum2, const FArray3& dum3,
              const FArray3& dum4, const FArray3& dum5, const FArray3& dum6, const FArray3& dum7)
{
  if (in.nk < 3) throw std::runtime_error("cm1k wrfphys::pbl_prep: needs nk >= 3");
  const int ni = in.ni, nj = in.nj, nk = in.nk, pertflx = in.pertflx;
  const Real dz = in.dz, cgt1 = in.cgt1, cgt2 = in.cgt2, cgt3 = in.cgt3;
  const auto ua = in.ua, va = in.va, u0 = in.u0, v0 = in.v0, th0 = in.th0, tha = in.tha, pi0 = in.pi0,
             ppi = in.ppi, prs = in.prs, rmh = in.rmh, c1 = in.c1, c2 = in.c2, psfc = in.psfc;
  Kokkos::parallel_for("cm1k::wrfphys::pbl_prep", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    for (int k = 1; k <= nk; ++k) {
      // ground-relative winds (imove=0): ugr = ua (- u0 with pertflx=1)
      Real ugr0, ugr1, vgr0, vgr1;
      if (pertflx == 1) {
        ugr0 = ua(i, j, k) - u0(i, j, k);
        ugr1 = ua(i + 1, j, k) - u0(i + 1, j, k);
        vgr0 = va(i, j, k) - v0(i, j, k);
        vgr1 = va(i, j + 1, k) - v0(i, j + 1, k);
      } else {
        ugr0 = ua(i, j, k);
        ugr1 = ua(i + 1, j, k);
        vgr0 = va(i, j, k);
        vgr1 = va(i, j + 1, k);
      }
      dum1(i, j, k) = 0.5 * (ugr0 + ugr1);
      dum2(i, j, k) = 0.5 * (vgr0 + vgr1);
      dum3(i, j, k) = th0(i, j, k) + tha(i, j, k);
      dum7(i, j, k) = pi0(i, j, k) + ppi(i, j, k);
      dum4(i, j, k) = dum3(i, j, k) * dum7(i, j, k);
      dum5(i, j, k) = dz * rmh(i, j, k);
      dum6(i, j, k) = c1(i, j, k) * prs(i, j, k - 1) + c2(i, j, k) * prs(i, j, k);
    }
    // surface:
    dum6(i, j, 1) = psfc(i, j, 1);
    // top of model:
    dum6(i, j, nk + 1) = cgt1 * prs(i, j, nk) + cgt2 * prs(i, j, nk - 1) + cgt3 * prs(i, j, nk - 2);
  });
}

void pbl_moist(int ni, int nj, int nk, int imoist, int nqv, int nqc, int nqi, const FArray4& qa,
               const FArray3& divx, const FArray3& ppten, const FArray3& dum8)
{
  const bool hasv = (imoist == 1 && nqv >= 1);
  Kokkos::parallel_for("cm1k::wrfphys::pbl_moist", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    divx(i, j, k) = hasv ? qa(i, j, k, nqv) : 0.0;
    ppten(i, j, k) = (nqc >= 1) ? qa(i, j, k, nqc) : 0.0;
    dum8(i, j, k) = (nqi >= 1) ? qa(i, j, k, nqi) : 0.0;
  });
}

} // namespace cm1k::wrfphys
