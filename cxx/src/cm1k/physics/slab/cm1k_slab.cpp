// Port of src/slab.F (SLAB/SLAB1D) and the soil-layer setup of
// src/init_surface.F. See cm1k_slab.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// x**4 (integer power) is (x*x)*(x*x), real**real is pow, AMIN1 of two
// non-NaN numbers is a plain comparison.
#include "physics/slab/cm1k_slab.hpp"

#include "share/cm1k_grid.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::slab {

namespace {
// module_sf_slab parameters
constexpr Real DIFSL = 5.e-7;    // soil diffusion constant (m^2/s)
constexpr Real SOILFAC = 1.25;   // factor to make the soil step more conservative
} // namespace

SoilLayers SoilLayers::make(int num_soil_layers) {
  if (num_soil_layers < 1 || num_soil_layers > max_soil_layers)
    throw std::runtime_error("cm1k slab: num_soil_layers must be 1.." + std::to_string(max_soil_layers));
  SoilLayers s;
  s.n = num_soil_layers;
  // init_surface.F (num_soil_layers /= 1); cm1.F zeroes them otherwise
  if (num_soil_layers != 1) {
    s.dzs[0] = .01;
    s.zs[0] = .5 * s.dzs[0];
    for (int l = 2; l <= num_soil_layers; ++l) {
      s.dzs[l - 1] = 2 * s.dzs[l - 2];
      s.zs[l - 1] = s.zs[l - 2] + .5 * s.dzs[l - 2] + .5 * s.dzs[l - 1];
    }
  }
  return s;
}

void init_tslb(const SoilLayers& sl, const Bounds& b, const FArray3& tsk, const FArray3& tmn,
               const FArray3& tslb) {
  if (sl.n == 1) return;
  const int n = sl.n;
  const SoilLayers S = sl;
  Kokkos::parallel_for("cm1k::slab::init_tslb", box3(b.ib, b.ie, b.jb, b.je, 1, n), KOKKOS_LAMBDA(int i, int j, int l) {
    tslb(i, j, l) = (tsk(i, j, 1) * (S.zs[n - 1] - S.zs[l - 1]) + tmn(i, j, 1) * (S.zs[l - 1] - S.zs[0])) /
                    (S.zs[n - 1] - S.zs[0]);
  });
}

void slab(const WrfConst& c, Real deltsm, int ifsnow, bool radiation, const SoilLayers& sl, int ni, int nj,
          const FArray3& t3d, const FArray3& qv3d, const FArray3& p3d, const Arrays& a) {
  const int nsl = sl.n;
  if (nsl < 1) throw std::runtime_error("cm1k slab::slab: soil layers not set");
  const SoilLayers S = sl;
  const Real ROVCP = c.rovcp, lv1 = c.lv1, lv2 = c.lv2, SVP1 = c.svp1, SVP2 = c.svp2, SVP3 = c.svp3,
             SVPT0 = c.svpt0, EP2 = c.ep2, EOMEG = c.eomeg, STBOLT = c.stbolt, P1000mb = c.p1000mb;
  const int RADSWTCH = radiation ? 1 : 0;
  // soil sub-timesteps (Fortran IFIX truncates toward zero)
  int NSOIL = 1;
  Real RNSOIL = 1.;
  if (nsl > 1) {
    NSOIL = 1 + int(SOILFAC * 4 * DIFSL / S.dzs[0] * deltsm / S.dzs[0]);
    RNSOIL = 1. / Real(NSOIL);
  }
  const Arrays A = a;

  Kokkos::parallel_for("cm1k::slab::slab", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    // SLAB: the k=1 values
    const Real T1D = t3d(i, j, 1), QV1D = qv3d(i, j, 1), P1D = p3d(i, j, 1);
    const Real PSFCPA = A.psfc(i, j, 1);
    Real& HFX = A.hfx(i, j, 1);
    Real& QFX = A.qfx(i, j, 1);
    Real& TSK = A.tsk(i, j, 1);
    const Real FLHC = A.flhc(i, j, 1), FLQC = A.flqc(i, j, 1);
    // in cmb
    const Real PSFC = PSFCPA / 1000.;
    // PL cmb
    const Real PL = P1D / 1000.;
    const Real SCR3 = T1D;
    const Real THCON = Kokkos::pow(P1000mb * 0.001 / PL, ROVCP);
    const Real THX = SCR3 * THCON;
    const Real QX = QV1D;
    // the slab thermal capacity
    Real& CAPG = A.capg(i, j, 1);
    CAPG = 3.298E6 * A.thc(i, j, 1);
    if (nsl > 1) CAPG = 5.9114E7 * A.thc(i, j, 1);

    // land points only (XLD1 = 1); ocean surface temperatures do not change
    if (!((A.xland(i, j, 1) - 1.5) >= 0)) {
      const Real TG0 = TSK;
      // surface energy budget (loop 70)
      const Real UPFLUX = Real(RADSWTCH) * STBOLT * ((TG0 * TG0) * (TG0 * TG0));
      const Real XINET = A.emiss(i, j, 1) * (A.glw(i, j, 1) - UPFLUX);
      const Real RNET = A.gsw(i, j, 1) + XINET;
      const Real HM = 1.18 * EOMEG * (TG0 - A.tmn(i, j, 1));
      // moisture flux calculated here (overwrites sfc layer value for land)
      Real ESG = SVP1 * Kokkos::exp(SVP2 * (TG0 - SVPT0) / (TG0 - SVP3));
      Real QSG = EP2 * ESG / (PSFC - ESG);
      Real THG = TSK * Kokkos::pow(100. / PSFC, ROVCP);
      HFX = FLHC * (THG - THX);
      QFX = FLQC * (QSG - QX);
      A.lh(i, j, 1) = QFX * (lv1 - lv2 * TSK);
      Real QS = HFX + QFX * (lv1 - lv2 * TSK);
      Real DTHGDT;
      if (nsl == 1) DTHGDT = (RNET - QS) / CAPG - HM;
      else          DTHGDT = 0.;
      if (nsl > 1) {
        // soil sub-timestep
        Real FLUX[max_soil_layers + 1];
        for (int ITSOIL = 1; ITSOIL <= NSOIL; ++ITSOIL) {
          for (int L = 1; L <= nsl - 1; ++L) {
            Real& T1 = A.tslb(i, j, 1);
            if (L == 1 && ITSOIL > 1) {
              const Real PS1 = Kokkos::pow(PSFCPA / P1000mb, ROVCP);
              // for rk scheme A and B are the same
              const Real PS = PSFC;
              THG = T1 / PS1;
              ESG = SVP1 * Kokkos::exp(SVP2 * (T1 - SVPT0) / (T1 - SVP3));
              QSG = EP2 * ESG / (PS - ESG);
              // update fluxes for new ground temperature
              const Real HFXT = FLHC * (THG - THX);
              const Real QFXT = FLQC * (QSG - QX);
              QS = HFXT + QFXT * (lv1 - lv2 * TSK);
              // sum hfx and qfx over soil timesteps
              HFX = HFX + HFXT;
              QFX = QFX + QFXT;
            }
            FLUX[1] = RNET - QS;
            FLUX[L + 1] = -DIFSL * A.capg(i, j, 1) * (A.tslb(i, j, L + 1) - A.tslb(i, j, L)) /
                          (S.zs[L] - S.zs[L - 1]);
            const Real DTSDT = -(FLUX[L + 1] - FLUX[L]) / (S.dzs[L - 1] * A.capg(i, j, 1));
            A.tslb(i, j, L) = A.tslb(i, j, L) + DTSDT * deltsm * RNSOIL;
            if (ifsnow == 1 && L == 1) {
              if (A.snowc(i, j, 1) > 0. && A.tslb(i, j, 1) > 273.16) A.tslb(i, j, 1) = 273.16;
            }
            if (L == 1) DTHGDT = DTHGDT + RNSOIL * DTSDT;
            if (ITSOIL == NSOIL && L == 1) {
              // average hfx and qfx over soil timesteps for output to pbl
              HFX = HFX * RNSOIL;
              QFX = QFX * RNSOIL;
              A.lh(i, j, 1) = QFX * (lv1 - lv2 * TSK);
            }
          }
        }
      }
      // loop 80
      const Real TSKX = TG0 + deltsm * DTHGDT;
      TSK = TSKX;
      // snow cover effects: limit the ground temperature under 0 C
      if (ifsnow != 0) {
        Real TSCVN = TSK;
        if (A.snowc(i, j, 1) > 0. && TSCVN > 273.16) TSCVN = 273.16;
        TSK = TSCVN;
      }
    }
    // 90: qsfc and chklowq needed by Eta PBL
    if (FLQC != 0.) A.qsfc(i, j, 1) = QX + QFX / FLQC;
    else            A.qsfc(i, j, 1) = QX;
    A.chklowq(i, j, 1) = A.mavail(i, j, 1);
  });
}

} // namespace cm1k::slab
