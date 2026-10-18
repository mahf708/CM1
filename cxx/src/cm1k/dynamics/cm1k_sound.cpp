#include "cm1k_sound.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cm1k {

int sound_nsound(int psolver, int nx, int ny, Real min_dx, Real min_dy, Real min_dz,
                 Real csound, double dbldt) {
  constexpr int max_nsound = 192;
  if (psolver == 1 || psolver == 4 || psolver == 5) return 1;
  Real dtsm = 0;
  if (psolver == 2) {
    if (ny == 1)      dtsm = 0.60 * std::min(min_dx, min_dz) / 360.0;
    else if (nx == 1) dtsm = 0.60 * std::min(min_dy, min_dz) / 360.0;
    else              dtsm = 0.50 * std::min(std::min(min_dx, min_dy), min_dz) / 360.0;
  } else if (psolver == 3) {
    if (ny == 1)      dtsm = 0.60 * min_dx / 360.0;
    else if (nx == 1) dtsm = 0.60 * min_dy / 360.0;
    else              dtsm = 0.60 * std::min(min_dx, min_dy) / 360.0;
  } else if (psolver == 6 || psolver == 7) {
    if (ny == 1)      dtsm = 0.60 * std::min(min_dx, min_dz) / csound;
    else if (nx == 1) dtsm = 0.60 * std::min(min_dy, min_dz) / csound;
    else              dtsm = 0.50 * std::min(std::min(min_dx, min_dy), min_dz) / csound;
  } else {
    throw std::runtime_error("cm1k: sound_nsound: unsupported psolver " + std::to_string(psolver));
  }
  int nsound = std::max(int(std::lround(dbldt / dtsm)), 4);   // nint
  if (nsound % 2 != 0) nsound = nsound + 1;
  if (dbldt / Real(nsound) > dtsm) nsound = nsound + 2;
  if (nsound > max_nsound)
    throw std::runtime_error("cm1k: limit for number of small steps exceeded (nsound=" +
                             std::to_string(nsound) + "); adapt_dt is not supported");
  return nsound;
}

void sound(const BcParams& bc, const SoundArgs& a) {
  const int ni = bc.ni, nj = bc.nj, nk = bc.nk;
  if (nk < 3) throw std::runtime_error("cm1k sound: nk must be >= 3");
  const Real dt = a.dt, alph = a.alph, kdiv = a.kdiv, cp = a.cp, smeps = a.smeps;
  const Real rdx = a.rdx, rdy = a.rdy, rdz = a.rdz;
  const int nrk = a.nrk;

  // beta = backward weight, alph = forward weight
  const Real beta = 1.0 - alph;
  const Real asq = alph * alph;

  int nloop = 0;
  Real dts = 0;
  if (a.nrkmax != 3) throw std::runtime_error("cm1k sound: nrkmax must be 3");
  if (nrk == 1) {
    nloop = int(std::lround(Real(a.nsound) / 3.0));
    dts = dt / (nloop * 3.0);
    if (dts > (dt / a.nsound)) {
      nloop = nloop + 1;
      dts = dt / (nloop * 3.0);
    }
  } else if (nrk == 2) {
    nloop = int(0.5 * a.nsound);
    dts = dt / a.nsound;
  } else {
    nloop = a.nsound;
    dts = dt / a.nsound;
  }

  auto uh = a.uh; auto vh = a.vh; auto uf = a.uf; auto vf = a.vf;
  auto mh = a.mh; auto mf = a.mf; auto c1 = a.c1; auto c2 = a.c2; auto rr0 = a.rr0; auto rf0 = a.rf0;
  auto dtu = a.dtu; auto dtv = a.dtv; auto dtu0 = a.dtu0; auto dtv0 = a.dtv0;
  auto dum1 = a.dum1; auto dum2 = a.dum2; auto mm = a.mm; auto ppd = a.ppd; auto fpk = a.fpk;
  auto qk = a.qk; auto pk1 = a.pk1; auto pk2 = a.pk2;
  auto rru = a.rru; auto ua = a.ua; auto u3d = a.u3d; auto uten = a.uten;
  auto rrv = a.rrv; auto va = a.va; auto v3d = a.v3d; auto vten = a.vten;
  auto rrw = a.rrw; auto wa = a.wa; auto w3d = a.w3d; auto wten = a.wten;
  auto ppi = a.ppi; auto pp3d = a.pp3d; auto piadv = a.piadv; auto ppten = a.ppten; auto ppx = a.ppx;
  auto thv = a.thv; auto ppterm = a.ppterm;

  Kokkos::parallel_for("sound_dtu", box2(1, ni + 1, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
    dtu(i, j) = dts * dtu0(i, j);
    dtv(i, j) = dts * dtv0(i, j);
  });

  //  Arrays for vertically implicit solver (without terrain)
  Kokkos::parallel_for("sound_coef", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real tem = dts * rdz * mh(1, 1, k);
    const Real r2 = dts * rdz * mh(1, 1, k) * rr0(1, 1, k) * rf0(1, 1, k + 1);
    const Real r1 = dts * rdz * mh(1, 1, k) * rr0(1, 1, k) * rf0(1, 1, k);
    const Real tem1 = dts * rdz * cp * mf(1, 1, k);
    const Real c1a = c1(1, 1, k), c1b = c1(1, 1, k + 1);
    const Real c2a = c2(1, 1, k), c2b = c2(1, 1, k + 1);
    qk(i, j, k) = tem * ppterm(i, j, k);
    pk2(i, j, k) = r2 * (-c2b * piadv(i, j, k + 1) + (1.0 - c1b) * piadv(i, j, k));
    pk1(i, j, k) = r1 * (+c1a * piadv(i, j, k - 1) + (c2a - 1.0) * piadv(i, j, k));
    mm(i, j, k) = tem1 * (c2a * thv(i, j, k) + c1a * thv(i, j, k - 1));
  });
  Kokkos::parallel_for("sound_pk0", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    pk1(i, j, 1) = 0.0;
    pk2(i, j, nk) = 0.0;
  });

  //  Prepare for acoustic steps
  if (nrk == 1) {
    Kokkos::parallel_for("sound_prep1", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      ppd(i, j, k) = ppx(i, j, k);
      rru(i, j, k) = 0.0;
      rrv(i, j, k) = 0.0;
      rrw(i, j, k) = wa(i, j, k) * beta;
    });
  } else {
    Kokkos::parallel_for("sound_prep", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u3d(i, j, k) = ua(i, j, k);
      rru(i, j, k) = 0.0;
      v3d(i, j, k) = va(i, j, k);
      rrv(i, j, k) = 0.0;
      w3d(i, j, k) = wa(i, j, k);
      rrw(i, j, k) = wa(i, j, k) * beta;
      pp3d(i, j, k) = ppi(i, j, k);
      ppd(i, j, k) = ppx(i, j, k);
    });
  }

  const Real tem1 = rdx * cp * 0.5;
  const Real tem2 = rdy * cp * 0.5;
  const Real temx = dts * 0.5 * rdx;
  const Real temy = dts * 0.5 * rdy;

  for (int n = 1; n <= nloop; ++n) {
    //  open boundary conditions (radbc for irbc=2, ssopenbc*, roflux)
    if (a.obc) sound_openbc(*a.obc, u3d, uten, v3d, vten, dts);

    //  integrate u,v forward in time (Cartesian grid without terrain)
    Kokkos::parallel_for("sound_uv", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u3d(i, j, k) = u3d(i, j, k) + dtu(i, j) * (uten(i, j, k)
                     - tem1 * (ppd(i, j, k) - ppd(i - 1, j, k)) * uf(i)
                           * (thv(i, j, k) + thv(i - 1, j, k)));
      v3d(i, j, k) = v3d(i, j, k) + dtv(i, j) * (vten(i, j, k)
                     - tem2 * (ppd(i, j, k) - ppd(i, j - 1, k)) * vf(j)
                           * (thv(i, j, k) + thv(i, j - 1, k)));
    });

    //  update arrays for vertically implicit solver
    Kokkos::parallel_for("sound_fpk", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real div = (u3d(i + 1, j, k) - u3d(i, j, k)) * rdx * uh(i)
               + (v3d(i, j + 1, k) - v3d(i, j, k)) * rdy * vh(j);
      if (Kokkos::fabs(div) < smeps) div = 0.0;
      fpk(i, j, k) = dts * (ppten(i, j, k) - ppterm(i, j, k) * div)
                   + (-(u3d(i, j, k) * (piadv(i, j, k) - piadv(i - 1, j, k))
                        + u3d(i + 1, j, k) * (piadv(i + 1, j, k) - piadv(i, j, k))) * temx * uh(i)
                      - (v3d(i, j, k) * (piadv(i, j, k) - piadv(i, j - 1, k))
                        + v3d(i, j + 1, k) * (piadv(i, j + 1, k) - piadv(i, j, k))) * temy * vh(j))
                   + beta * (w3d(i, j, k) * (pk1(i, j, k) + qk(i, j, k))
                           + w3d(i, j, k + 1) * (pk2(i, j, k) - qk(i, j, k)));
    });

    //  the vertically implicit solver .... get new w
    Kokkos::parallel_for("sound_tridiag", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      {
        const int k = 2;
        const Real cc = -asq * (-mm(i, j, k) * (pk2(i, j, k) - qk(i, j, k)));
        const Real bb = 1.0 - asq * (+mm(i, j, k) * (-(pk1(i, j, k) + qk(i, j, k))
                                                    + (pk2(i, j, k - 1) - qk(i, j, k - 1))));
        const Real fwk = dts * wten(i, j, k) + beta * (-mm(i, j, k) * (pp3d(i, j, k) - pp3d(i, j, k - 1)));
        const Real dd = w3d(i, j, k) + fwk + alph * (
                        -mm(i, j, k) * ((pp3d(i, j, k) + fpk(i, j, k))
                                      - (pp3d(i, j, k - 1) + fpk(i, j, k - 1))));
        const Real r1 = 1.0 / bb;
        dum1(i, j, k) = -cc * r1;
        dum2(i, j, k) = dd * r1;
      }
      for (int k = 3; k <= nk - 1; ++k) {
        const Real aa = -asq * (+mm(i, j, k) * (pk1(i, j, k - 1) + qk(i, j, k - 1)));
        const Real cc = -asq * (-mm(i, j, k) * (pk2(i, j, k) - qk(i, j, k)));
        const Real bb = 1.0 - asq * (+mm(i, j, k) * (-(pk1(i, j, k) + qk(i, j, k))
                                                    + (pk2(i, j, k - 1) - qk(i, j, k - 1))));
        const Real fwk = dts * wten(i, j, k) + beta * (-mm(i, j, k) * (pp3d(i, j, k) - pp3d(i, j, k - 1)));
        const Real dd = w3d(i, j, k) + fwk + alph * (
                        -mm(i, j, k) * ((pp3d(i, j, k) + fpk(i, j, k))
                                      - (pp3d(i, j, k - 1) + fpk(i, j, k - 1))));
        const Real r1 = 1.0 / (aa * dum1(i, j, k - 1) + bb);
        dum1(i, j, k) = -cc * r1;
        dum2(i, j, k) = (dd - aa * dum2(i, j, k - 1)) * r1;
      }
      {
        const int k = nk;
        const Real aa = -asq * (+mm(i, j, k) * (pk1(i, j, k - 1) + qk(i, j, k - 1)));
        const Real bb = 1.0 - asq * (+mm(i, j, k) * (-(pk1(i, j, k) + qk(i, j, k))
                                                    + (pk2(i, j, k - 1) - qk(i, j, k - 1))));
        const Real fwk = dts * wten(i, j, k) + beta * (-mm(i, j, k) * (pp3d(i, j, k) - pp3d(i, j, k - 1)));
        const Real dd = w3d(i, j, k) + fwk + alph * (
                        -mm(i, j, k) * ((pp3d(i, j, k) + fpk(i, j, k))
                                      - (pp3d(i, j, k - 1) + fpk(i, j, k - 1))));
        const Real r1 = 1.0 / (aa * dum1(i, j, k - 1) + bb);
        w3d(i, j, k) = (dd - aa * dum2(i, j, k - 1)) * r1;
      }
      for (int k = nk - 1; k >= 2; --k)
        w3d(i, j, k) = dum1(i, j, k) * w3d(i, j, k + 1) + dum2(i, j, k);
    });

    //  get new pp (without terrain), divergence damping
    Kokkos::parallel_for("sound_pp", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      ppd(i, j, k) = pp3d(i, j, k);
      pp3d(i, j, k) = pp3d(i, j, k) + fpk(i, j, k)
                    + alph * (w3d(i, j, k) * (pk1(i, j, k) + qk(i, j, k))
                            + w3d(i, j, k + 1) * (pk2(i, j, k) - qk(i, j, k)));
      if (Kokkos::fabs(pp3d(i, j, k)) < smeps) pp3d(i, j, k) = 0.0;
      dum1(i, j, k) = kdiv * (pp3d(i, j, k) - ppd(i, j, k));
      ppd(i, j, k) = pp3d(i, j, k) + dum1(i, j, k);
    });

    if (n < nloop) bcp(bc, ppd);

    //  time-averaged velocities
    if (n < nloop) {
      Kokkos::parallel_for("sound_avg", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        rru(i, j, k) = rru(i, j, k) + u3d(i, j, k);
        rrv(i, j, k) = rrv(i, j, k) + v3d(i, j, k);
        rrw(i, j, k) = rrw(i, j, k) + w3d(i, j, k);
      });
    } else {
      const Real tavg = 1.0 / Real(nloop);
      Kokkos::parallel_for("sound_avg", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        rru(i, j, k) = (rru(i, j, k) + u3d(i, j, k)) * tavg;
        rrv(i, j, k) = (rrv(i, j, k) + v3d(i, j, k)) * tavg;
        rrw(i, j, k) = (rrw(i, j, k) + w3d(i, j, k) * alph) * tavg;
      });
    }
  }

  if (nrk == a.nrkmax) {
    // pressure tendency term: save for next timestep
    Kokkos::parallel_for("sound_ppx", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      ppx(i, j, k) = dum1(i, j, k);
    });
  }
}

} // namespace cm1k
