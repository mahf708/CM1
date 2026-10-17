// Flux routines of src/adv_routines.F: hadv_flx2..10, hadv_weno5,
// vadv_flx2..10 (+ vadv_lwr_flx_*), vadv_weno5 (+ vadv_lwr_weno_*).
#include "dynamics/cm1k_adv.hpp"
#include "dynamics/cm1k_adv_routines.hpp"

#include <stdexcept>
#include <string>

namespace cm1k {

using namespace adv;

namespace {

// Loop ranges of hadv_*: orders 5 and WENO5 compute dumx and dumy over the
// same ("wide") box, the others over separate boxes. Only the values that
// advs/advu/advv/advw read afterwards matter, but we keep the same ranges.
template <int S>
void hadv_impl(const Grid& g, const AdvOptions& o, int stag, int pdef, double weps,
               const FArray3& rru, const FArray3& rrv, const FArray3& a,
               const FArray3& dumx, const FArray3& dumy)
{
  const int ni = g.b.ni, nj = g.b.nj, nk = g.b.nk;
  constexpr bool wide = (S == 5 || S == WENO5);
  const FArray3 c2 = g.c2;
  const bool pdefchk = (S == WENO5 && pdef == 1);

  if (stag == 1) {
    auto fx = KOKKOS_LAMBDA(int i, int j, int k) {
      dumx(i, j, k) = face_flux<S>(rru(i, j, k), Line{a, i - 1, j, k, 1, 0, 0}, weps, pdefchk);
    };
    auto fy = KOKKOS_LAMBDA(int i, int j, int k) {
      dumy(i, j, k) = face_flux<S>(rrv(i, j, k), Line{a, i, j - 1, k, 0, 1, 0}, weps, pdefchk);
    };
    if (wide) {
      Kokkos::parallel_for("hadv_s", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        fx(i, j, k); fy(i, j, k);
      });
    } else {
      Kokkos::parallel_for("hadv_sx", box3(1, ni + 1, 1, nj, 1, nk), fx);
      Kokkos::parallel_for("hadv_sy", box3(1, ni, 1, nj + 1, 1, nk), fy);
    }
  } else if (stag == 2) {
    const int i1 = (o.ibw == 1) ? 2 : 1;
    const int i2 = (o.ibe == 1) ? ni + 1 - 1 : ni + 1;
    auto fx = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real ubar = 0.5*(rru(i, j, k)+rru(i + 1, j, k));
      dumx(i, j, k) = face_flux<S>(ubar, Line{a, i, j, k, 1, 0, 0}, weps);
    };
    auto fy = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real vbar = 0.5*(rrv(i, j, k)+rrv(i - 1, j, k));
      dumy(i, j, k) = face_flux<S>(vbar, Line{a, i, j - 1, k, 0, 1, 0}, weps);
    };
    if (wide) {
      Kokkos::parallel_for("hadv_u", box3(i1 - 1, i2, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        fx(i, j, k); fy(i, j, k);
      });
    } else {
      Kokkos::parallel_for("hadv_ux", box3(i1 - 1, i2, 1, nj, 1, nk), fx);
      Kokkos::parallel_for("hadv_uy", box3(i1, i2, 1, nj + 1, 1, nk), fy);
    }
  } else if (stag == 3) {
    const int j1 = (o.ibs == 1) ? 2 : 1;
    const int j2 = (o.ibn == 1) ? nj + 1 - 1 : nj + 1;
    auto fx = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real ubar = 0.5*(rru(i, j, k)+rru(i, j - 1, k));
      dumx(i, j, k) = face_flux<S>(ubar, Line{a, i - 1, j, k, 1, 0, 0}, weps);
    };
    auto fy = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real vbar = 0.5*(rrv(i, j, k)+rrv(i, j + 1, k));
      dumy(i, j, k) = face_flux<S>(vbar, Line{a, i, j, k, 0, 1, 0}, weps);
    };
    if (wide) {
      Kokkos::parallel_for("hadv_v", box3(1, ni + 1, j1 - 1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        fx(i, j, k); fy(i, j, k);
      });
    } else {
      Kokkos::parallel_for("hadv_vx", box3(1, ni + 1, j1, j2, 1, nk), fx);
      Kokkos::parallel_for("hadv_vy", box3(1, ni, j1 - 1, j2, 1, nk), fy);
    }
  } else if (stag == 4) {
    auto fx = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real cc2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      const Real ubar = cc2*rru(i, j, k)+cc1*rru(i, j, k - 1);
      dumx(i, j, k) = face_flux<S>(ubar, Line{a, i - 1, j, k, 1, 0, 0}, weps);
    };
    auto fy = KOKKOS_LAMBDA(int i, int j, int k) {
      const Real cc2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      const Real vbar = cc2*rrv(i, j, k)+cc1*rrv(i, j, k - 1);
      dumy(i, j, k) = face_flux<S>(vbar, Line{a, i, j - 1, k, 0, 1, 0}, weps);
    };
    if (wide) {
      Kokkos::parallel_for("hadv_w", box3(1, ni + 1, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        fx(i, j, k); fy(i, j, k);
      });
    } else {
      Kokkos::parallel_for("hadv_wx", box3(1, ni + 1, 1, nj, 2, nk), fx);
      Kokkos::parallel_for("hadv_wy", box3(1, ni, 1, nj + 1, 2, nk), fy);
    }
  } else {
    throw std::runtime_error("hadv_flux: bad stag " + std::to_string(stag));
  }
}

// vadv_flxN: flux at w-level faces k=2..nk (stag 1,2,3) or at scalar levels
// k=1..nk between w levels (stag 4).
template <int S>
void vadv_impl(const Grid& g, const AdvOptions& o, int stag,
               const FArray3& rrw, const FArray3& a, const FArray3& dumz)
{
  const int ni = g.b.ni, nj = g.b.nj, nk = g.b.nk;
  const FArray3 c1 = g.c1, c2 = g.c2;
  if (stag == 1) {
    Kokkos::parallel_for("vadv_s", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumz(i, j, k) = vface_flux<S>(rrw(i, j, k), Line{a, i, j, k - 1, 0, 0, 1}, k - 1, 1, nk,
                                    c1(i, j, k), c2(i, j, k), false);
    });
  } else if (stag == 2) {
    const int i1 = (o.ibw == 1) ? 2 : 1;
    const int i2 = (o.ibe == 1) ? ni + 1 - 1 : ni + 1;
    Kokkos::parallel_for("vadv_u", box3(i1, i2, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real wbar = 0.5*(rrw(i, j, k)+rrw(i - 1, j, k));
      const Real cc2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      dumz(i, j, k) = vface_flux<S>(wbar, Line{a, i, j, k - 1, 0, 0, 1}, k - 1, 1, nk, cc1, cc2, false);
    });
  } else if (stag == 3) {
    const int j1 = (o.ibs == 1) ? 2 : 1;
    const int j2 = (o.ibn == 1) ? nj + 1 - 1 : nj + 1;
    Kokkos::parallel_for("vadv_v", box3(1, ni, j1, j2, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real wbar = 0.5*(rrw(i, j, k)+rrw(i, j - 1, k));
      const Real cc2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      dumz(i, j, k) = vface_flux<S>(wbar, Line{a, i, j, k - 1, 0, 0, 1}, k - 1, 1, nk, cc1, cc2, false);
    });
  } else if (stag == 4) {
    Kokkos::parallel_for("vadv_w", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real wbar = 0.5*(rrw(i, j, k)+rrw(i, j, k + 1));
      dumz(i, j, k) = vface_flux<S>(wbar, Line{a, i, j, k, 0, 0, 1}, k, 1, nk + 1, 0.0, 0.0, true);
    });
  } else {
    throw std::runtime_error("vadv_flux: bad stag " + std::to_string(stag));
  }
}

// vadv_weno5 + vadv_lwr_weno_23 + vadv_lwr_weno_35 (or _35_b with pdefweno).
// Near the boundaries the Fortran uses upstrpd (3rd order, positive
// definite) and 2nd-order interpolation; the comparisons are `>=` except at
// the top faces noted below, which use `>`.
void vadv_weno5_impl(const Grid& g, const AdvOptions& o, int stag, int pdefweno, double weps,
                     const FArray3& rrw, const FArray3& a, const FArray3& dumz)
{
  const int ni = g.b.ni, nj = g.b.nj, nk = g.b.nk;
  const FArray3 c1 = g.c1, c2 = g.c2;
  const Real bmin = weps;

  if (stag == 4) {
    Kokkos::parallel_for("vadv_weno5_w", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real wbar = 0.5*(rrw(i, j, k)+rrw(i, j, k + 1));
      Real r;
      if (k == 1) {
        if (wbar >= 0.0) r = wbar*0.5*(a(i, j, k)+a(i, j, k + 1));
        else             r = wbar*upstrpd(a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), weps);
      } else if (k == nk) {
        if (wbar > 0.0)  r = wbar*upstrpd(a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), weps);
        else             r = wbar*0.5*(a(i, j, k)+a(i, j, k + 1));
      } else if (k == 2) {
        if (wbar >= 0.0) r = wbar*upstrpd(a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), weps);
        else r = wbar*weno5(a(i, j, k + 3), a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
      } else if (k == nk - 1) {
        if (wbar > 0.0) r = wbar*weno5(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), a(i, j, k + 2), weps);
        else            r = wbar*upstrpd(a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), weps);
      } else {
        if (wbar >= 0.0) r = wbar*weno5(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), a(i, j, k + 2), weps);
        else             r = wbar*weno5(a(i, j, k + 3), a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
      }
      dumz(i, j, k) = r;
    });
    return;
  }

  int i1 = 1, i2 = ni, j1 = 1, j2 = nj;
  if (stag == 2) { i1 = (o.ibw == 1) ? 2 : 1; i2 = (o.ibe == 1) ? ni + 1 - 1 : ni + 1; }
  if (stag == 3) { j1 = (o.ibs == 1) ? 2 : 1; j2 = (o.ibn == 1) ? nj + 1 - 1 : nj + 1; }
  if (stag < 1 || stag > 3) throw std::runtime_error("vadv_flux: bad stag " + std::to_string(stag));
  const bool pdw = (stag == 1 && pdefweno == 1);

  Kokkos::parallel_for("vadv_weno5", box3(i1, i2, j1, j2, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    Real vel, w1, w2;
    if (stag == 1) {
      vel = rrw(i, j, k); w1 = c1(i, j, k); w2 = c2(i, j, k);
    } else if (stag == 2) {
      vel = 0.5*(rrw(i, j, k)+rrw(i - 1, j, k));
      w2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k)); w1 = 1.0-w2;
    } else {
      vel = 0.5*(rrw(i, j, k)+rrw(i, j - 1, k));
      w2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k)); w1 = 1.0-w2;
    }
    Real r;
    if (k == 2) {
      if (vel >= 0.0) r = vel*(w1*a(i, j, k - 1)+w2*a(i, j, k));
      else            r = vel*upstrpd(a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
    } else if (k == nk) {
      if (vel > 0.0)  r = vel*upstrpd(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), weps);
      else            r = vel*(w1*a(i, j, k - 1)+w2*a(i, j, k));
    } else if (pdw) {
      // vadv_weno5 (pdefweno) interior k=3..nk-1, then vadv_lwr_weno_35_b
      if (k == 3 && vel > 0.0) {
        r = vel*upstrpd(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), weps);
      } else if (k == nk - 1 && vel < 0.0) {
        r = vel*upstrpd(a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
      } else {
        Real s1, s2, s3, s4, s5;
        if (vel >= 0.0) {
          s1 = a(i, j, k - 3); s2 = a(i, j, k - 2); s3 = a(i, j, k - 1); s4 = a(i, j, k); s5 = a(i, j, k + 1);
        } else {
          s1 = a(i, j, k + 2); s2 = a(i, j, k + 1); s3 = a(i, j, k); s4 = a(i, j, k - 1); s5 = a(i, j, k - 2);
        }
        const Real bmax = fmax2(fmax2(fmax2(fmax2(Kokkos::fabs(s1), Kokkos::fabs(s2)), Kokkos::fabs(s3)),
                                      Kokkos::fabs(s4)), Kokkos::fabs(s5));
        if (bmax > bmin) r = vel*weno5(s1, s2, s3, s4, s5, weps);
        else             r = 0.0;
      }
    } else if (k == 3) {
      if (vel >= 0.0) r = vel*upstrpd(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), weps);
      else r = vel*weno5(a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), a(i, j, k - 2), weps);
    } else if (k == nk - 1) {
      const bool pos = (stag == 1) ? (vel >= 0.0) : (vel > 0.0);
      if (pos) r = vel*weno5(a(i, j, k - 3), a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), weps);
      else     r = vel*upstrpd(a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
    } else {
      if (vel >= 0.0) r = vel*weno5(a(i, j, k - 3), a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), a(i, j, k + 1), weps);
      else            r = vel*weno5(a(i, j, k + 2), a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), a(i, j, k - 2), weps);
    }
    dumz(i, j, k) = r;
  });
}

} // namespace

void hadv_flux(const Grid& g, const AdvOptions& o, int stag, int scheme, int pdef, double weps,
               const FArray3& rru, const FArray3& rrv, const FArray3& a,
               const FArray3& dumx, const FArray3& dumy)
{
  switch (scheme) {
    case 2:  hadv_impl<2>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 3:  hadv_impl<3>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 4:  hadv_impl<4>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 5:  hadv_impl<5>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 6:  hadv_impl<6>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 7:  hadv_impl<7>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 8:  hadv_impl<8>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 9:  hadv_impl<9>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case 10: hadv_impl<10>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    case WENO5: hadv_impl<WENO5>(g, o, stag, pdef, weps, rru, rrv, a, dumx, dumy); break;
    default: throw std::runtime_error("hadv_flux: unsupported scheme " + std::to_string(scheme));
  }
}

void vadv_flux(const Grid& g, const AdvOptions& o, int stag, int scheme, int pdefweno, double weps,
               const FArray3& rrw, const FArray3& a, const FArray3& dumz)
{
  // For small nk the Fortran's interior and lower-order loops overlap (and
  // some have zero stride); we only reproduce the regular case.
  if (g.b.nk < 2*((scheme == WENO5) ? 5 : scheme))
    throw std::runtime_error("vadv_flux: nk = " + std::to_string(g.b.nk) + " too small for vertical order "
                             + std::to_string(scheme == WENO5 ? 5 : scheme));
  switch (scheme) {
    case 2:  vadv_impl<2>(g, o, stag, rrw, a, dumz); break;
    case 3:  vadv_impl<3>(g, o, stag, rrw, a, dumz); break;
    case 4:  vadv_impl<4>(g, o, stag, rrw, a, dumz); break;
    case 5:  vadv_impl<5>(g, o, stag, rrw, a, dumz); break;
    case 6:  vadv_impl<6>(g, o, stag, rrw, a, dumz); break;
    case 7:  vadv_impl<7>(g, o, stag, rrw, a, dumz); break;
    case 8:  vadv_impl<8>(g, o, stag, rrw, a, dumz); break;
    case 9:  vadv_impl<9>(g, o, stag, rrw, a, dumz); break;
    case 10: vadv_impl<10>(g, o, stag, rrw, a, dumz); break;
    case WENO5: vadv_weno5_impl(g, o, stag, pdefweno, weps, rrw, a, dumz); break;
    default: throw std::runtime_error("vadv_flux: unsupported scheme " + std::to_string(scheme));
  }
}

} // namespace cm1k
