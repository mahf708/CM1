// Port of the open-radiative boundary routines of bc.F and of the
// open-boundary pieces of solve1/solve2/sound. See cm1k_openbc.hpp.
#include "cm1k_openbc.hpp"

#include <stdexcept>
#include <string>

namespace cm1k {

namespace {
// constants.F
constexpr Real csmax = 350.0;
using Range = Kokkos::RangePolicy<ExeSpace, Kokkos::IndexType<int>>;
inline Range range(int i1, int i2) { return Range(i1, i2 + 1); }
} // namespace

OpenBcParams OpenBcParams::make(const Grid& g, const Namelist& nl) {
  OpenBcParams p;
  p.wbc = nl.geti("wbc", 1);
  p.ebc = nl.geti("ebc", 1);
  p.sbc = nl.geti("sbc", 1);
  p.nbc = nl.geti("nbc", 1);
  // param.F (serial build): ibw = 1 iff wbc /= 1, ...
  p.ibw = (p.wbc != 1) ? 1 : 0;
  p.ibe = (p.ebc != 1) ? 1 : 0;
  p.ibs = (p.sbc != 1) ? 1 : 0;
  p.ibn = (p.nbc != 1) ? 1 : 0;
  p.irbc = nl.geti("irbc", 4);
  p.roflux = nl.geti("roflux", 0);
  p.nudgeobc = nl.geti("nudgeobc", 0);   // cm1.F default
  p.alphobc = nl.getr("alphobc", 60.0);  // cm1.F default
  p.cstar = nl.getr("cstar", 0.0);       // cm1.F default, then param.F:
  if (p.cstar < 0.1) p.cstar = 30.0;
  const int psolver = nl.geti("psolver", 3);
  if (psolver == 4 || psolver == 5) p.roflux = 1;   // param.F
  p.dx = g.dx; p.dy = g.dy; p.rdx = g.rdx; p.rdy = g.rdy;
  p.ni = g.b.ni; p.nj = g.b.nj; p.nk = g.b.nk;
  return p;
}

void OpenBcParams::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k open bc: unsupported option: " + m); };
  for (int v : {wbc, ebc, sbc, nbc})
    if (v < 1 || v > 4) fail("lateral boundary condition " + std::to_string(v) + " (must be 1..4)");
  if (any_open() && irbc != 1 && irbc != 2 && irbc != 4)
    fail("irbc=" + std::to_string(irbc) + " (only 1, 2, 4)");
  if (roflux != 0 && roflux != 1) fail("roflux=" + std::to_string(roflux) + " (only 0, 1)");
  if (nudgeobc != 0 && nudgeobc != 1) fail("nudgeobc=" + std::to_string(nudgeobc) + " (only 0, 1)");
  if (nudgeobc == 1 && any_open() && !(alphobc != 0.0)) fail("alphobc = 0 with nudgeobc=1");
}

void RadBc::allocate(const Bounds& b) {
  radbcw = FArray2("radbcw", {b.jb, b.kb}, {b.je, b.ke});
  radbce = FArray2("radbce", {b.jb, b.kb}, {b.je, b.ke});
  radbcs = FArray2("radbcs", {b.ib, b.kb}, {b.ie, b.ke});
  radbcn = FArray2("radbcn", {b.ib, b.kb}, {b.ie, b.ke});
  radbcw.fill(0.0); radbce.fill(0.0); radbcs.fill(0.0); radbcn.fill(0.0);
}

//---------------------------------------------------------------------------
void radbcew(const OpenBcParams& p, const RadBc& r, const FArray3& ua) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real cstar = p.cstar;
  if (p.ibw == 1 && p.wbc == 2) {
    auto radbcw = r.radbcw;
    Kokkos::parallel_for("radbcew_w", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const Real cbcw = ua(1, j, k) - cstar;
      if (cbcw < 0.0) radbcw(j, k) = cbcw;
      else            radbcw(j, k) = 0.;
    });
  }
  if (p.ibe == 1 && p.ebc == 2) {
    auto radbce = r.radbce;
    Kokkos::parallel_for("radbcew_e", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const Real cbce = ua(ni + 1, j, k) + cstar;
      if (cbce > 0.0) radbce(j, k) = cbce;
      else            radbce(j, k) = 0.;
    });
  }
}

void radbcns(const OpenBcParams& p, const RadBc& r, const FArray3& va) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real cstar = p.cstar;
  if (p.ibs == 1 && p.sbc == 2) {
    auto radbcs = r.radbcs;
    Kokkos::parallel_for("radbcns_s", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const Real cbcs = va(i, 1, k) - cstar;
      if (cbcs < 0.0) radbcs(i, k) = cbcs;
      else            radbcs(i, k) = 0.;
    });
  }
  if (p.ibn == 1 && p.nbc == 2) {
    auto radbcn = r.radbcn;
    Kokkos::parallel_for("radbcns_n", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const Real cbcn = va(i, nj + 1, k) + cstar;
      if (cbcn > 0.0) radbcn(i, k) = cbcn;
      else            radbcn(i, k) = 0.;
    });
  }
}

//---------------------------------------------------------------------------
// sign(1.e-10,foo) is copysign (gfortran honours the sign of -0.0).
void radbcew4(const OpenBcParams& p, const FArray1& ruf, const RadBc& r, const FArray3& u1,
              const FArray3& u2, Real dt) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real umax = csmax, dx = p.dx;
  if (p.ibw == 1 && p.wbc == 2) {
    auto radbcw = r.radbcw;
    Kokkos::parallel_for("radbcew4_w", range(1, nj), KOKKOS_LAMBDA(int j) {
      Real avgw = 0.;
      for (int k = 1; k <= nk; ++k) {
        const Real foo = (u1(3, j, k) - u1(2, j, k));
        Real cbcw = dx * ruf(2) * (u1(2, j, k) - u2(2, j, k))
                    / (dt * (Kokkos::copysign(1.e-10, foo) + foo));
        cbcw = Kokkos::fmax(Kokkos::fmin(cbcw, 0.0), -umax);
        avgw = avgw + cbcw;
      }
      avgw = avgw / Real(nk);
      for (int k = 1; k <= nk; ++k) radbcw(j, k) = avgw;
    });
  }
  if (p.ibe == 1 && p.ebc == 2) {
    auto radbce = r.radbce;
    Kokkos::parallel_for("radbcew4_e", range(1, nj), KOKKOS_LAMBDA(int j) {
      Real avge = 0.;
      for (int k = 1; k <= nk; ++k) {
        const Real foo = (u1(ni + 1 - 1, j, k) - u1(ni + 1 - 2, j, k));
        Real cbce = dx * ruf(ni + 1 - 1) * (u1(ni + 1 - 1, j, k) - u2(ni + 1 - 1, j, k))
                    / (dt * (Kokkos::copysign(1.e-10, foo) + foo));
        cbce = Kokkos::fmin(Kokkos::fmax(cbce, 0.0), umax);
        avge = avge + cbce;
      }
      avge = avge / Real(nk);
      for (int k = 1; k <= nk; ++k) radbce(j, k) = avge;
    });
  }
}

void radbcns4(const OpenBcParams& p, const FArray1& rvf, const RadBc& r, const FArray3& v1,
              const FArray3& v2, Real dt) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real umax = csmax, dy = p.dy;
  if (p.ibs == 1 && p.sbc == 2) {
    auto radbcs = r.radbcs;
    Kokkos::parallel_for("radbcns4_s", range(1, ni), KOKKOS_LAMBDA(int i) {
      Real avgs = 0.;
      for (int k = 1; k <= nk; ++k) {
        const Real foo = (v1(i, 3, k) - v1(i, 2, k));
        Real cbcs = dy * rvf(2) * (v1(i, 2, k) - v2(i, 2, k))
                    / (dt * (Kokkos::copysign(1.e-10, foo) + foo));
        cbcs = Kokkos::fmax(Kokkos::fmin(cbcs, 0.0), -umax);
        avgs = avgs + cbcs;
      }
      avgs = avgs / Real(nk);
      for (int k = 1; k <= nk; ++k) radbcs(i, k) = avgs;
    });
  }
  if (p.ibn == 1 && p.nbc == 2) {
    auto radbcn = r.radbcn;
    Kokkos::parallel_for("radbcns4_n", range(1, ni), KOKKOS_LAMBDA(int i) {
      Real avgn = 0.;
      for (int k = 1; k <= nk; ++k) {
        const Real foo = (v1(i, nj + 1 - 1, k) - v1(i, nj + 1 - 2, k));
        Real cbcn = dy * rvf(nj + 1 - 1) * (v1(i, nj + 1 - 1, k) - v2(i, nj + 1 - 1, k))
                    / (dt * (Kokkos::copysign(1.e-10, foo) + foo));
        cbcn = Kokkos::fmin(Kokkos::fmax(cbcn, 0.0), umax);
        avgn = avgn + cbcn;
      }
      avgn = avgn / Real(nk);
      for (int k = 1; k <= nk; ++k) radbcn(i, k) = avgn;
    });
  }
}

//---------------------------------------------------------------------------
void ssopenbcw(const OpenBcParams& p, const FArray1& uh, const RadBc& r, const FArray3& u3d,
               const FArray3& uten, Real dts) {
  const Real rdx = p.rdx;
  auto radbcw = r.radbcw;
  Kokkos::parallel_for("ssopenbcw", box2(1, p.nj, 1, p.nk), KOKKOS_LAMBDA(int j, int k) {
    u3d(1, j, k) = u3d(1, j, k) + dts * (-radbcw(j, k)
                   * (u3d(2, j, k) - u3d(1, j, k)) * rdx * uh(1)
                   + uten(1, j, k));
  });
}

void ssopenbce(const OpenBcParams& p, const FArray1& uh, const RadBc& r, const FArray3& u3d,
               const FArray3& uten, Real dts) {
  const Real rdx = p.rdx;
  const int ni = p.ni;
  auto radbce = r.radbce;
  Kokkos::parallel_for("ssopenbce", box2(1, p.nj, 1, p.nk), KOKKOS_LAMBDA(int j, int k) {
    u3d(ni + 1, j, k) = u3d(ni + 1, j, k) + dts * (-radbce(j, k)
                        * (u3d(ni + 1, j, k) - u3d(ni, j, k)) * rdx * uh(ni)
                        + uten(ni + 1, j, k));
  });
}

void ssopenbcs(const OpenBcParams& p, const FArray1& vh, const RadBc& r, const FArray3& v3d,
               const FArray3& vten, Real dts) {
  const Real rdy = p.rdy;
  auto radbcs = r.radbcs;
  Kokkos::parallel_for("ssopenbcs", box2(1, p.ni, 1, p.nk), KOKKOS_LAMBDA(int i, int k) {
    v3d(i, 1, k) = v3d(i, 1, k) + dts * (-radbcs(i, k)
                   * (v3d(i, 2, k) - v3d(i, 1, k)) * rdy * vh(1)
                   + vten(i, 1, k));
  });
}

void ssopenbcn(const OpenBcParams& p, const FArray1& vh, const RadBc& r, const FArray3& v3d,
               const FArray3& vten, Real dts) {
  const Real rdy = p.rdy;
  const int nj = p.nj;
  auto radbcn = r.radbcn;
  Kokkos::parallel_for("ssopenbcn", box2(1, p.ni, 1, p.nk), KOKKOS_LAMBDA(int i, int k) {
    v3d(i, nj + 1, k) = v3d(i, nj + 1, k) + dts * (-radbcn(i, k)
                        * (v3d(i, nj + 1, k) - v3d(i, nj, k)) * rdy * vh(nj)
                        + vten(i, nj + 1, k));
  });
}

//---------------------------------------------------------------------------
// ssopenbc* with terrain: dum1 = u (v) at w points on the two columns next
// to the boundary (i1, i1+1), then the update with the metric terms
namespace {
void uv_at_w(const OpenBcParams& p, const OpenBcTerrain& t, const FArray3& u, int i1, int i2, int j1, int j2) {
  const int nk = p.nk;
  auto dum1 = t.dum1; auto rds = t.rds; auto sigma = t.sigma; auto sigmaf = t.sigmaf;
  const Real cgs1 = t.cgs1, cgs2 = t.cgs2, cgs3 = t.cgs3, cgt1 = t.cgt1, cgt2 = t.cgt2, cgt3 = t.cgt3;
  Kokkos::parallel_for("ssopenbc_uw", box3(i1, i2, j1, j2, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    if (k == 1) {
      dum1(i, j, 1) = cgs1 * u(i, j, 1) + cgs2 * u(i, j, 2) + cgs3 * u(i, j, 3);
    } else if (k == nk + 1) {
      dum1(i, j, nk + 1) = cgt1 * u(i, j, nk) + cgt2 * u(i, j, nk - 1) + cgt3 * u(i, j, nk - 2);
    } else {
      const Real r2 = (sigmaf(k) - sigma(k - 1)) * rds(k);
      const Real r1 = 1.0 - r2;
      dum1(i, j, k) = r1 * u(i, j, k - 1) + r2 * u(i, j, k);
    }
  });
}
} // namespace

void ssopenbcw(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& uh, const RadBc& r,
               const FArray3& u3d, const FArray3& uten, Real dts) {
  uv_at_w(p, t, u3d, 1, 2, 1, p.nj);
  const Real rdx = p.rdx;
  auto radbcw = r.radbcw; auto gz = t.gz; auto rgzu = t.rgzu; auto gx = t.gx; auto dum1 = t.dum1; auto rdsf = t.rdsf;
  Kokkos::parallel_for("ssopenbcw_t", box2(1, p.nj, 1, p.nk), KOKKOS_LAMBDA(int j, int k) {
    u3d(1, j, k) = u3d(1, j, k) + dts * (-radbcw(j, k) * (
                     gz(1, j, 1) * (u3d(2, j, k) * rgzu(2, j, 1) - u3d(1, j, k) * rgzu(1, j, 1)) * rdx * uh(1)
                     + 0.5 * (gx(1, j, k + 1) * (dum1(2, j, k + 1) + dum1(1, j, k + 1))
                              - gx(1, j, k) * (dum1(2, j, k) + dum1(1, j, k))) * rdsf(k))
                   + uten(1, j, k));
  });
}

void ssopenbce(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& uh, const RadBc& r,
               const FArray3& u3d, const FArray3& uten, Real dts) {
  const int ni = p.ni;
  uv_at_w(p, t, u3d, ni, ni + 1, 1, p.nj);
  const Real rdx = p.rdx;
  auto radbce = r.radbce; auto gz = t.gz; auto rgzu = t.rgzu; auto gx = t.gx; auto dum1 = t.dum1; auto rdsf = t.rdsf;
  Kokkos::parallel_for("ssopenbce_t", box2(1, p.nj, 1, p.nk), KOKKOS_LAMBDA(int j, int k) {
    u3d(ni + 1, j, k) = u3d(ni + 1, j, k) + dts * (-radbce(j, k) * (
                          gz(ni, j, 1) * (u3d(ni + 1, j, k) * rgzu(ni + 1, j, 1) - u3d(ni, j, k) * rgzu(ni, j, 1)) * rdx * uh(ni)
                          + 0.5 * (gx(ni, j, k + 1) * (dum1(ni + 1, j, k + 1) + dum1(ni, j, k + 1))
                                   - gx(ni, j, k) * (dum1(ni + 1, j, k) + dum1(ni, j, k))) * rdsf(k))
                        + uten(ni + 1, j, k));
  });
}

void ssopenbcs(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& vh, const RadBc& r,
               const FArray3& v3d, const FArray3& vten, Real dts) {
  uv_at_w(p, t, v3d, 1, p.ni, 1, 2);
  const Real rdy = p.rdy;
  auto radbcs = r.radbcs; auto gz = t.gz; auto rgzv = t.rgzv; auto gy = t.gy; auto dum1 = t.dum1; auto rdsf = t.rdsf;
  Kokkos::parallel_for("ssopenbcs_t", box2(1, p.ni, 1, p.nk), KOKKOS_LAMBDA(int i, int k) {
    v3d(i, 1, k) = v3d(i, 1, k) + dts * (-radbcs(i, k) * (
                     gz(i, 1, 1) * (v3d(i, 2, k) * rgzv(i, 2, 1) - v3d(i, 1, k) * rgzv(i, 1, 1)) * rdy * vh(1)
                     + 0.5 * (gy(i, 1, k + 1) * (dum1(i, 2, k + 1) + dum1(i, 1, k + 1))
                              - gy(i, 1, k) * (dum1(i, 2, k) + dum1(i, 1, k))) * rdsf(k))
                   + vten(i, 1, k));
  });
}

void ssopenbcn(const OpenBcParams& p, const OpenBcTerrain& t, const FArray1& vh, const RadBc& r,
               const FArray3& v3d, const FArray3& vten, Real dts) {
  const int nj = p.nj;
  uv_at_w(p, t, v3d, 1, p.ni, nj, nj + 1);
  const Real rdy = p.rdy;
  auto radbcn = r.radbcn; auto gz = t.gz; auto rgzv = t.rgzv; auto gy = t.gy; auto dum1 = t.dum1; auto rdsf = t.rdsf;
  Kokkos::parallel_for("ssopenbcn_t", box2(1, p.ni, 1, p.nk), KOKKOS_LAMBDA(int i, int k) {
    v3d(i, nj + 1, k) = v3d(i, nj + 1, k) + dts * (-radbcn(i, k) * (
                          gz(i, nj, 1) * (v3d(i, nj + 1, k) * rgzv(i, nj + 1, 1) - v3d(i, nj, k) * rgzv(i, nj, 1)) * rdy * vh(nj)
                          + 0.5 * (gy(i, nj, k + 1) * (dum1(i, nj + 1, k + 1) + dum1(i, nj, k + 1))
                                   - gy(i, nj, k) * (dum1(i, nj + 1, k) + dum1(i, nj, k))) * rdsf(k))
                        + vten(i, nj + 1, k));
  });
}

//---------------------------------------------------------------------------
namespace {
// fluxout/fluxin: serial sum over k (double), as in the Fortran.
void sum_levels(int nk, const FArray1& temout, const FArray1& temin, double& fluxout, double& fluxin) {
  auto ho = temout.to_host(), hi = temin.to_host();
  fluxout = 0.0;
  fluxin = 0.0;
  for (int k = 1; k <= nk; ++k) {
    fluxout = fluxout + ho(k);
    fluxin = fluxin + hi(k);
  }
}
} // namespace

void restrict_openbc_we(const OpenBcParams& p, const FArray1& rvh, const FArray3& rmh,
                        const FArray3& rho0, const FArray3& u3d, const FArray1& temout,
                        const FArray1& temin) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const bool w = (p.wbc == 2 && p.ibw == 1), e = (p.ebc == 2 && p.ibe == 1);
  Kokkos::parallel_for("restrict_we_sum", range(1, nk), KOKKOS_LAMBDA(int k) {
    double to = 0.0, ti = 0.0;
    if (w) {
      const int i = 1;
      for (int j = 1; j <= nj; ++j) {
        to = to - Kokkos::fmin(0.0, rho0(1, j, k) * u3d(i, j, k) * rvh(j) * rmh(1, j, k));
        ti = ti + Kokkos::fmax(0.0, rho0(1, j, k) * u3d(i, j, k) * rvh(j) * rmh(1, j, k));
      }
    }
    if (e) {
      const int i = ni + 1;
      for (int j = 1; j <= nj; ++j) {
        to = to + Kokkos::fmax(0.0, rho0(ni, j, k) * u3d(i, j, k) * rvh(j) * rmh(ni, j, k));
        ti = ti - Kokkos::fmin(0.0, rho0(ni, j, k) * u3d(i, j, k) * rvh(j) * rmh(ni, j, k));
      }
    }
    temout(k) = to;
    temin(k) = ti;
  });
  double fluxout, fluxin;
  sum_levels(nk, temout, temin, fluxout, fluxin);
  const double t3 = (fluxin + 1.0e-20) / (fluxout + 1.0e-20);
  if (w)
    Kokkos::parallel_for("restrict_we_w", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const int i = 1;
      const double u1 = rho0(1, j, k) * u3d(i, j, k);
      if (u1 < 0.0) u3d(i, j, k) = u1 * t3 / rho0(1, j, k);
    });
  if (e)
    Kokkos::parallel_for("restrict_we_e", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const int i = ni + 1;
      const double u1 = rho0(ni, j, k) * u3d(i, j, k);
      if (u1 > 0.0) u3d(i, j, k) = u1 * t3 / rho0(ni, j, k);
    });
}

void restrict_openbc_sn(const OpenBcParams& p, const FArray1& ruh, const FArray3& rmh,
                        const FArray3& rho0, const FArray3& v3d, const FArray1& temout,
                        const FArray1& temin) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const bool s = (p.sbc == 2 && p.ibs == 1), n = (p.nbc == 2 && p.ibn == 1);
  Kokkos::parallel_for("restrict_sn_sum", range(1, nk), KOKKOS_LAMBDA(int k) {
    double to = 0.0, ti = 0.0;
    if (s) {
      const int j = 1;
      for (int i = 1; i <= ni; ++i) {
        to = to - Kokkos::fmin(0.0, rho0(i, 1, k) * v3d(i, j, k) * ruh(i) * rmh(i, 1, k));
        ti = ti + Kokkos::fmax(0.0, rho0(i, 1, k) * v3d(i, j, k) * ruh(i) * rmh(i, 1, k));
      }
    }
    if (n) {
      const int j = nj + 1;
      for (int i = 1; i <= ni; ++i) {
        to = to + Kokkos::fmax(0.0, rho0(i, nj, k) * v3d(i, j, k) * ruh(i) * rmh(i, nj, k));
        ti = ti - Kokkos::fmin(0.0, rho0(i, nj, k) * v3d(i, j, k) * ruh(i) * rmh(i, nj, k));
      }
    }
    temout(k) = to;
    temin(k) = ti;
  });
  double fluxout, fluxin;
  sum_levels(nk, temout, temin, fluxout, fluxin);
  const double t3 = (fluxin + 1.0e-20) / (fluxout + 1.0e-20);
  if (s)
    Kokkos::parallel_for("restrict_sn_s", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const int j = 1;
      const double u1 = rho0(i, 1, k) * v3d(i, j, k);
      if (u1 < 0.0) v3d(i, j, k) = u1 * t3 / rho0(i, 1, k);
    });
  if (n)
    Kokkos::parallel_for("restrict_sn_n", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const int j = nj + 1;
      const double u1 = rho0(i, nj, k) * v3d(i, j, k);
      if (u1 > 0.0) v3d(i, j, k) = u1 * t3 / rho0(i, nj, k);
    });
}

//---------------------------------------------------------------------------
// sound.F, start of each acoustic step (axisymm = 0)
void sound_openbc(const SoundOpenBc& o, const FArray3& u3d, const FArray3& uten, const FArray3& v3d,
                  const FArray3& vten, Real dts) {
  const auto& p = o.p;
  if (p.irbc == 2) {
    if (p.ibw == 1 || p.ibe == 1) radbcew(p, o.rad, u3d);
    if (p.ibs == 1 || p.ibn == 1) radbcns(p, o.rad, v3d);
  }
  if (o.terrain) {
    if (p.wbc == 2 && p.ibw == 1) ssopenbcw(p, o.t, o.uh, o.rad, u3d, uten, dts);
    if (p.ebc == 2 && p.ibe == 1) ssopenbce(p, o.t, o.uh, o.rad, u3d, uten, dts);
  } else {
    if (p.wbc == 2 && p.ibw == 1) ssopenbcw(p, o.uh, o.rad, u3d, uten, dts);
    if (p.ebc == 2 && p.ibe == 1) ssopenbce(p, o.uh, o.rad, u3d, uten, dts);
  }
  if (p.roflux == 1) restrict_openbc_we(p, o.rvh, o.rmh, o.rho0, u3d, o.work, o.work2);
  if (o.terrain) {
    if (p.sbc == 2 && p.ibs == 1) ssopenbcs(p, o.t, o.vh, o.rad, v3d, vten, dts);
    if (p.nbc == 2 && p.ibn == 1) ssopenbcn(p, o.t, o.vh, o.rad, v3d, vten, dts);
  } else {
    if (p.sbc == 2 && p.ibs == 1) ssopenbcs(p, o.vh, o.rad, v3d, vten, dts);
    if (p.nbc == 2 && p.ibn == 1) ssopenbcn(p, o.vh, o.rad, v3d, vten, dts);
  }
  if (p.roflux == 1) restrict_openbc_sn(p, o.ruh, o.rmh, o.rho0, v3d, o.work, o.work2);
}

//---------------------------------------------------------------------------
// solve2.F (190315): nudge inflow point back towards base state
void nudgeobc_uv(const OpenBcParams& p, const FArray3& u0, const FArray3& v0, const FArray3& u3d,
                 const FArray3& v3d, const FArray3& uten, const FArray3& vten) {
  if (p.nudgeobc != 1) return;
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real tem = 1.0 / p.alphobc;
  if (p.wbc == 2 && p.ibw == 1)
    Kokkos::parallel_for("nudgeobc_w", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      if (u3d(1, j, k) > 0.0) uten(1, j, k) = uten(1, j, k) - (u3d(1, j, k) - u0(1, j, k)) * tem;
    });
  if (p.ebc == 2 && p.ibe == 1)
    Kokkos::parallel_for("nudgeobc_e", box2(1, nj, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      if (u3d(ni + 1, j, k) < 0.0) uten(ni + 1, j, k) = uten(ni + 1, j, k) - (u3d(ni + 1, j, k) - u0(ni + 1, j, k)) * tem;
    });
  if (p.sbc == 2 && p.ibs == 1)
    Kokkos::parallel_for("nudgeobc_s", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      if (v3d(i, 1, k) > 0.0) vten(i, 1, k) = vten(i, 1, k) - (v3d(i, 1, k) - v0(i, 1, k)) * tem;
    });
  if (p.nbc == 2 && p.ibn == 1)
    Kokkos::parallel_for("nudgeobc_n", box2(1, ni, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      if (v3d(i, nj + 1, k) < 0.0) vten(i, nj + 1, k) = vten(i, nj + 1, k) - (v3d(i, nj + 1, k) - v0(i, nj + 1, k)) * tem;
    });
}

} // namespace cm1k
