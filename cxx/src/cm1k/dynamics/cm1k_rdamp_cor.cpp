// Rayleigh damping coefficients (param.F) and Coriolis terms (solve2.F).
#include "dynamics/cm1k_rdamp_cor.hpp"
#include "share/cm1k_constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cm1k {

void rdamp_coeffs(const RdampParams& p, const Grid& g, const FArray3& tauh, const FArray3& taus,
                  const FArray3& tauf)
{
  const auto& B = g.b;
  const int nk = B.nk;
  auto th = tauh.to_host(), ts = taus.to_host(), tf = tauf.to_host();
  for (size_t n = 0; n < th.size(); ++n) th.data()[n] = 0.0;
  for (size_t n = 0; n < ts.size(); ++n) ts.data()[n] = 0.0;
  for (size_t n = 0; n < tf.size(); ++n) tf.data()[n] = 0.0;
  auto zh = g.zh.to_host(), zf = g.zf.to_host();
  auto sigmaf = g.sigmaf.to_host();

  //  Specify coefficient for Rayleigh damper in vertical
  const Real maxz = sigmaf(nk + 1);
  const Real zd = p.zd;
  if (p.irdamp >= 1 && zd < maxz) {
    if (zd < 0.5 * maxz)
      throw std::runtime_error("cm1k: Rayleigh damping would be applied over more than half the domain (zd < maxz/2)");
    for (int j = B.jb; j <= B.je; ++j) for (int i = B.ib; i <= B.ie; ++i)
      for (int k = 1; k <= nk; ++k)
        if (zh(i, j, k) > zd) {
          th(i, j, k) = 0.5 * (1.0 - std::cos(phys::pi * (zh(i, j, k) - zd) / (zf(i, j, nk + 1) - zd)));
          ts(i, j, k) = th(i, j, k);
        }
    for (int j = B.jb; j <= B.je; ++j) for (int i = B.ib; i <= B.ie; ++i)
      for (int k = 1; k <= nk + 1; ++k)
        if (zf(i, j, k) > zd)
          tf(i, j, k) = 0.5 * (1.0 - std::cos(phys::pi * (zf(i, j, k) - zd) / (zf(i, j, nk + 1) - zd)));
  }

  //  Rayleigh damping near lateral boundaries (cm1r19.6: acts only on w)
  if (p.hrdamp >= 1) {
    auto xh = g.xh.to_host(), yh = g.yh.to_host(), xf = g.xf.to_host(), yf = g.yf.to_host();
    // serial run: xfref = xf, yfref = yf (param.F)
    const Real minx = p.have_extent ? p.minx : xf(1), maxx = p.have_extent ? p.maxx : xf(B.nx + 1);
    const Real miny = p.have_extent ? p.miny : yf(1), maxy = p.have_extent ? p.maxy : yf(B.ny + 1);
    const Real xhd = p.xhd;
    if (B.nx > 1 && xhd > (0.5 * (maxx - minx)))
      throw std::runtime_error("cm1k: horizontal Rayleigh damping would be applied over more than half the domain (xhd)");
    if (B.ny > 1 && xhd > (0.5 * (maxy - miny)))
      throw std::runtime_error("cm1k: horizontal Rayleigh damping would be applied over more than half the domain (xhd)");
    // hrdamp_west/east/south/north = .true. (param.F)
    for (int k = 1; k <= nk; ++k)
      for (int j = B.jb; j <= B.je; ++j)
        for (int i = B.ib; i <= B.ie; ++i) {
          if (B.nx > 1) {
            const Real x1 = (xhd + minx) - xh(i);   // west (axisymm /= 1)
            if (x1 > 0.0) tf(i, j, k) = std::max(tf(i, j, k), 0.5 * (1.0 - std::cos(phys::pi * x1 / xhd)));
            const Real x2 = xh(i) - (maxx - xhd);   // east
            if (x2 > 0.0) tf(i, j, k) = std::max(tf(i, j, k), 0.5 * (1.0 - std::cos(phys::pi * x2 / xhd)));
          }
          if (B.ny > 1) {
            const Real y1 = (xhd + miny) - yh(j);   // south
            if (y1 > 0.0) tf(i, j, k) = std::max(tf(i, j, k), 0.5 * (1.0 - std::cos(phys::pi * y1 / xhd)));
            const Real y2 = yh(j) - (maxy - xhd);   // north
            if (y2 > 0.0) tf(i, j, k) = std::max(tf(i, j, k), 0.5 * (1.0 - std::cos(phys::pi * y2 / xhd)));
          }
        }
  }
  tauh.deep_copy_from(th); taus.deep_copy_from(ts); tauf.deep_copy_from(tf);
}

void coriolis(int ni, int nj, int nk, Real fcor, Real umove, Real vmove, const FArray3& u3d,
              const FArray3& v3d, const FArray3& uten, const FArray3& vten)
{
  Kokkos::parallel_for("coriolis", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    uten(i, j, k) = uten(i, j, k)+fcor*( 0.25*( (v3d(i  , j, k)+v3d(i  , j + 1, k))
                                               +(v3d(i - 1, j, k)+v3d(i - 1, j + 1, k)) ) + vmove );
    vten(i, j, k) = vten(i, j, k)-fcor*( 0.25*( (u3d(i, j  , k)+u3d(i + 1, j  , k))
                                               +(u3d(i, j - 1, k)+u3d(i + 1, j - 1, k)) ) + umove );
  });
}

} // namespace cm1k
