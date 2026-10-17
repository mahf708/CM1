// Port of src/diff2.F (Cartesian grid, no terrain). See cm1k_diff2.hpp.
#include "dynamics/cm1k_diff2.hpp"

namespace cm1k {

void Diff2Work::allocate(const Bounds& b) {
  auto s3 = [&](const char* n) { return FArray3(n, {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke}); };
  t11 = s3("t11"); t12 = s3("t12"); t13 = s3("t13"); t22 = s3("t22"); t23 = s3("t23"); t33 = s3("t33");
  dum = s3("d2_dum"); diffx = s3("d2_diffx"); diffy = s3("d2_diffy"); diffz = s3("d2_diffz");
}

//------------------------------------------------------------------------------
// diff2def
//------------------------------------------------------------------------------

void diff2def(const Diff2Params& p, const Grid& g, const FArray3& divx, const FArray3& rho,
              const FArray3& rf, const FArray3& t11, const FArray3& t12, const FArray3& t13,
              const FArray3& t22, const FArray3& t23, const FArray3& t33, const FArray3& ua,
              const FArray3& va, const FArray3& wa)
{
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real coef = p.kdiff2;
  const Real rdx = p.rdx, rdy = p.rdy, rdz = p.rdz, smeps = p.smeps;
  const Real umove = p.umove, vmove = p.vmove;
  const FArray1 uh = g.uh, vh = g.vh, uf = g.uf, vf = g.vf;
  const FArray3 mh = g.mh, mf = g.mf;

  Kokkos::parallel_for("diff2def_divx", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    divx(i, j, k) = (ua(i + 1, j, k)-ua(i, j, k))*rdx*uh(i)
                   +(va(i, j + 1, k)-va(i, j, k))*rdy*vh(j)
                   +(wa(i, j, k + 1)-wa(i, j, k))*rdz*mh(1, 1, k);
    if (Kokkos::fabs(divx(i, j, k)) < smeps) divx(i, j, k) = 0.0;
  });
  Kokkos::parallel_for("diff2def_t", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    t11(i, j, k) = coef*rho(i, j, k)*(ua(i + 1, j, k)-ua(i, j, k))*rdx*uh(i);
    t22(i, j, k) = coef*rho(i, j, k)*(va(i, j + 1, k)-va(i, j, k))*rdy*vh(j);
    t33(i, j, k) = coef*rho(i, j, k)*(wa(i, j, k + 1)-wa(i, j, k))*rdz*mh(1, 1, k);
    t12(i, j, k) = coef*( (ua(i, j, k)-ua(i, j - 1, k))*rdy*vf(j)
                         +(va(i, j, k)-va(i - 1, j, k))*rdx*uf(i) )
             *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))+(rho(i - 1, j, k)+rho(i, j - 1, k)) );
  });

  // lateral boundary conditions for t12 (west, east, south, north, as in the Fortran)
  if (p.ibw == 1 && (p.wbc == 3 || p.wbc == 4)) {
    const int wbc = p.wbc;
    Kokkos::parallel_for("diff2def_t12w", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const int i = 1;
      if (wbc == 3) t12(1, j, k) = t12(2, j, k);
      else          t12(1, j, k) = coef*2.0*va(1, j, k)*rdx*uf(1)
                        *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))+(rho(i - 1, j, k)+rho(i, j - 1, k)) );
    });
  }
  if (p.ibe == 1 && (p.ebc == 3 || p.ebc == 4)) {
    const int ebc = p.ebc;
    Kokkos::parallel_for("diff2def_t12e", box2(1, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      const int i = ni + 1;
      if (ebc == 3) t12(ni + 1, j, k) = t12(ni, j, k);
      else          t12(ni + 1, j, k) = -coef*2.0*va(ni, j, k)*rdx*uf(ni + 1)
                        *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))+(rho(i - 1, j, k)+rho(i, j - 1, k)) );
    });
  }
  if (p.ibs == 1 && (p.sbc == 3 || p.sbc == 4)) {
    const int sbc = p.sbc;
    Kokkos::parallel_for("diff2def_t12s", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const int j = 1;
      if (sbc == 3) t12(i, 1, k) = t12(i, 2, k);
      else          t12(i, 1, k) = coef*2.0*ua(i, 1, k)*rdy*vf(1)
                        *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))+(rho(i - 1, j, k)+rho(i, j - 1, k)) );
    });
  }
  if (p.ibn == 1 && (p.nbc == 3 || p.nbc == 4)) {
    const int nbc = p.nbc;
    Kokkos::parallel_for("diff2def_t12n", box2(1, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      const int j = nj + 1;
      if (nbc == 3) t12(i, nj + 1, k) = t12(i, nj, k);
      else          t12(i, nj + 1, k) = -coef*2.0*ua(i, nj, k)*rdy*vf(nj + 1)
                        *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))+(rho(i - 1, j, k)+rho(i, j - 1, k)) );
    });
  }

  if (nk >= 2) {
    Kokkos::parallel_for("diff2def_t13", box3(1, ni + 1, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      t13(i, j, k) = coef*( (ua(i, j, k)-ua(i, j, k - 1))*rdz*mf(1, 1, k)
                           +(wa(i, j, k)-wa(i - 1, j, k))*rdx*uf(i) );
      t23(i, j, k) = coef*( (va(i, j, k)-va(i, j, k - 1))*rdz*mf(1, 1, k)
                           +(wa(i, j, k)-wa(i, j - 1, k))*rdy*vf(j) );
    });
    if (p.ibw == 1 && (p.wbc == 3 || p.wbc == 4)) {
      const int wbc = p.wbc;
      Kokkos::parallel_for("diff2def_t13w", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) {
        if (wbc == 3) t13(1, j, k) = t13(2, j, k);
        else          t13(1, j, k) = coef*2.0*wa(1, j, k)*rdx*uf(1);
      });
    }
    if (p.ibe == 1 && (p.ebc == 3 || p.ebc == 4)) {
      const int ebc = p.ebc;
      Kokkos::parallel_for("diff2def_t13e", box2(1, nj, 2, nk), KOKKOS_LAMBDA(int j, int k) {
        if (ebc == 3) t13(ni + 1, j, k) = t13(ni, j, k);
        else          t13(ni + 1, j, k) = -coef*2.0*wa(ni, j, k)*rdx*uf(ni + 1);
      });
    }
    if (p.ibs == 1 && (p.sbc == 3 || p.sbc == 4)) {
      const int sbc = p.sbc;
      Kokkos::parallel_for("diff2def_t23s", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) {
        if (sbc == 3) t23(i, 1, k) = t23(i, 2, k);
        else          t23(i, 1, k) = coef*2.0*wa(i, 1, k)*rdy*vf(1);
      });
    }
    if (p.ibn == 1 && (p.nbc == 3 || p.nbc == 4)) {
      const int nbc = p.nbc;
      Kokkos::parallel_for("diff2def_t23n", box2(1, ni, 2, nk), KOKKOS_LAMBDA(int i, int k) {
        if (nbc == 3) t23(i, nj + 1, k) = t23(i, nj, k);
        else          t23(i, nj + 1, k) = -coef*2.0*wa(i, nj, k)*rdy*vf(nj + 1);
      });
    }
  }

  // bottom/top boundary conditions
  const int bbc = p.bbc, tbc = p.tbc;
  if (bbc == 1 || bbc == 2) {
    Kokkos::parallel_for("diff2def_t13b", box2(1, ni + 1, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      if (bbc == 1) t13(i, j, 1) = t13(i, j, 2);
      else          t13(i, j, 1) = coef*2.0*(ua(i, j, 1)+umove)*rdz*mf(1, 1, 1);
    });
    Kokkos::parallel_for("diff2def_t23b", box2(1, ni, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      if (bbc == 1) t23(i, j, 1) = t23(i, j, 2);
      else          t23(i, j, 1) = coef*2.0*(va(i, j, 1)+vmove)*rdz*mf(1, 1, 1);
    });
  }
  if (tbc == 1 || tbc == 2) {
    Kokkos::parallel_for("diff2def_t13t", box2(1, ni + 1, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      if (tbc == 1) t13(i, j, nk + 1) = t13(i, j, nk);
      else          t13(i, j, nk + 1) = -coef*2.0*ua(i, j, nk)*rdz*mf(1, 1, nk + 1);
    });
    Kokkos::parallel_for("diff2def_t23t", box2(1, ni, 1, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      if (tbc == 1) t23(i, j, nk + 1) = t23(i, j, nk);
      else          t23(i, j, nk + 1) = -coef*2.0*va(i, j, nk)*rdz*mf(1, 1, nk + 1);
    });
  }

  // mult t13,t23 by density
  Kokkos::parallel_for("diff2def_rf", box3(1, ni + 1, 1, nj + 1, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    t13(i, j, k) = t13(i, j, k)*0.5*(rf(i - 1, j, k)+rf(i, j, k));
    t23(i, j, k) = t23(i, j, k)*0.5*(rf(i, j - 1, k)+rf(i, j, k));
  });
}

//------------------------------------------------------------------------------
// diff2u / diff2v / diff2w
//------------------------------------------------------------------------------

namespace {
constexpr Real twdth = 2.0/3.0;
}

void diff2u(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& uten, const FArray3& rho,
            const FArray3& divx, const FArray3& t11, const FArray3& t12, const FArray3& t13)
{
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real rdx = p.rdx, rdy = p.rdy, rdz = p.rdz;
  const Real tw = twdth;
  const FArray1 uf = g.uf, vh = g.vh;
  const FArray3 mh = g.mh;
  Kokkos::parallel_for("diff2u_dum", box3(0, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum(i, j, k) = 2.0*t11(i, j, k)-rho(i, j, k)*tw*divx(i, j, k);
  });
  Kokkos::parallel_for("diff2u_d", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffx(i, j, k) = (dum(i, j, k)-dum(i - 1, j, k))*rdx*uf(i);
    diffy(i, j, k) = (t12(i, j + 1, k)-t12(i, j, k))*rdy*vh(j);
    diffz(i, j, k) = (t13(i, j, k + 1)-t13(i, j, k))*rdz*mh(1, 1, k);
  });
  Kokkos::parallel_for("diff2u_ten", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    uten(i, j, k) = uten(i, j, k)+((diffx(i, j, k)+diffy(i, j, k))+diffz(i, j, k))
                                 /(0.5*(rho(i - 1, j, k)+rho(i, j, k)));
  });
}

void diff2v(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& vten, const FArray3& rho,
            const FArray3& divx, const FArray3& t22, const FArray3& t12, const FArray3& t23)
{
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real rdx = p.rdx, rdy = p.rdy, rdz = p.rdz;
  const Real tw = twdth;
  const FArray1 uh = g.uh, vf = g.vf;
  const FArray3 mh = g.mh;
  Kokkos::parallel_for("diff2v_dum", box3(1, ni, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum(i, j, k) = 2.0*t22(i, j, k)-rho(i, j, k)*tw*divx(i, j, k);
  });
  Kokkos::parallel_for("diff2v_d", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffx(i, j, k) = (t12(i + 1, j, k)-t12(i, j, k))*rdx*uh(i);
    diffy(i, j, k) = (dum(i, j, k)-dum(i, j - 1, k))*rdy*vf(j);
    diffz(i, j, k) = (t23(i, j, k + 1)-t23(i, j, k))*rdz*mh(1, 1, k);
  });
  Kokkos::parallel_for("diff2v_ten", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    vten(i, j, k) = vten(i, j, k)+((diffx(i, j, k)+diffy(i, j, k))+diffz(i, j, k))
                                 /(0.5*(rho(i, j - 1, k)+rho(i, j, k)));
  });
}

void diff2w(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& wten, const FArray3& rho,
            const FArray3& rf, const FArray3& divx, const FArray3& t33, const FArray3& t13,
            const FArray3& t23)
{
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real rdx = p.rdx, rdy = p.rdy, rdz = p.rdz;
  const Real tw = twdth;
  const FArray1 uh = g.uh, vh = g.vh;
  const FArray3 mf = g.mf;
  Kokkos::parallel_for("diff2w_dum", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum(i, j, k) = 2.0*t33(i, j, k)-rho(i, j, k)*tw*divx(i, j, k);
  });
  if (nk < 2) return;
  Kokkos::parallel_for("diff2w_d", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffx(i, j, k) = (t13(i + 1, j, k)-t13(i, j, k))*rdx*uh(i);
    diffy(i, j, k) = (t23(i, j + 1, k)-t23(i, j, k))*rdy*vh(j);
    diffz(i, j, k) = (dum(i, j, k)-dum(i, j, k - 1))*rdz*mf(1, 1, k);
  });
  Kokkos::parallel_for("diff2w_ten", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    wten(i, j, k) = wten(i, j, k)+((diffx(i, j, k)+diffy(i, j, k))+diffz(i, j, k))/rf(i, j, k);
  });
}

//------------------------------------------------------------------------------
// diff2s
//------------------------------------------------------------------------------

void diff2s(const Diff2Params& p, const Grid& g, const FArray3& dum, const FArray3& diffx,
            const FArray3& diffy, const FArray3& diffz, const FArray3& s, const FArray3& sten,
            const FArray3& rho, const FArray3& rr, const FArray3& rf)
{
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  const Real coef = p.kdiff2;
  const Real rdx = p.rdx, rdy = p.rdy, rdz = p.rdz;
  const FArray1 uh = g.uh, vh = g.vh, uf = g.uf, vf = g.vf;
  const FArray3 mh = g.mh, mf = g.mf;

  // x-direction
  Kokkos::parallel_for("diff2s_dx", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum(i, j, k) = 0.5*(rho(i, j, k)+rho(i - 1, j, k))*(s(i, j, k)-s(i - 1, j, k))*rdx*uf(i);
  });
  Kokkos::parallel_for("diff2s_x", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffx(i, j, k) = coef*(dum(i + 1, j, k)-dum(i, j, k))*rdx*uh(i);
  });
  // y-direction
  Kokkos::parallel_for("diff2s_dy", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dum(i, j, k) = 0.5*(rho(i, j, k)+rho(i, j - 1, k))*(s(i, j, k)-s(i, j - 1, k))*rdy*vf(j);
  });
  Kokkos::parallel_for("diff2s_y", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffy(i, j, k) = coef*(dum(i, j + 1, k)-dum(i, j, k))*rdy*vh(j);
  });
  // z-direction (zero-flux boundary conditions)
  Kokkos::parallel_for("diff2s_dz", box3(1, ni, 1, nj, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    if (k == 1 || k == nk + 1) dum(i, j, k) = 0.0;
    else                       dum(i, j, k) = (s(i, j, k)-s(i, j, k - 1))*rdz*mf(1, 1, k);
  });
  Kokkos::parallel_for("diff2s_z", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    diffz(i, j, k) = coef*(rf(i, j, k + 1)*dum(i, j, k + 1)-rf(i, j, k)*dum(i, j, k))*rdz*mh(1, 1, k);
  });
  Kokkos::parallel_for("diff2s_ten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    sten(i, j, k) = sten(i, j, k)+((diffx(i, j, k)+diffy(i, j, k))+diffz(i, j, k))*rr(i, j, k);
  });
}

} // namespace cm1k
