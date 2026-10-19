#include "dynamics/cm1k_adapt_dt.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cm1k {

Real calccflquick(const Bounds& b, int axisymm, Real dt, Real rdx, Real rdy, Real rdz, const FArray1& uh,
                  const FArray1& vh, const FArray3& mh, const FArray3& ua, const FArray3& va, const FArray3& wa) {
  const Real dtdx = 0.5 * dt * rdx;
  const Real dtdy = 0.5 * dt * rdy;
  const Real dtdz = 0.5 * dt * rdz;
  Real fmax = 0.0;   // cfl(k) = 0.0 to start; max is exact in any order
  if (b.nx > 1 && b.ny > 1) {
    Kokkos::parallel_reduce("calccflquick", box3(1, b.ni, 1, b.nj, 1, b.nk),
      KOKKOS_LAMBDA(int i, int j, int k, Real& m) {
        const Real a = ((ua(i, j, k) + ua(i + 1, j, k)) * dtdx * uh(i));
        const Real bb = ((va(i, j, k) + va(i, j + 1, k)) * dtdy * vh(j));
        const Real c = ((wa(i, j, k) + wa(i, j, k + 1)) * dtdz * mh(i, j, k));
        const Real wsp = Kokkos::sqrt((a * a + bb * bb) + c * c);
        if (wsp > m) m = wsp;
      }, Kokkos::Max<Real>(fmax));
  } else if (b.nx > 1) {
    Kokkos::parallel_reduce("calccflquick2d", box3(1, b.ni, 1, b.nj, 1, b.nk),
      KOKKOS_LAMBDA(int i, int j, int k, Real& m) {
        const Real a = ((ua(i, j, k) + ua(i + 1, j, k)) * dtdx * uh(i));
        const Real c = ((wa(i, j, k) + wa(i, j, k + 1)) * dtdz * mh(i, j, k));
        const Real wsp = Kokkos::sqrt(a * a + c * c);
        if (wsp > m) m = wsp;
      }, Kokkos::Max<Real>(fmax));
  } else if (axisymm == 0 && b.ny > 1) {
    Kokkos::parallel_reduce("calccflquick2dy", box3(1, b.ni, 1, b.nj, 1, b.nk),
      KOKKOS_LAMBDA(int i, int j, int k, Real& m) {
        const Real bb = ((va(i, j, k) + va(i, j + 1, k)) * dtdy * vh(j));
        const Real c = ((wa(i, j, k) + wa(i, j, k + 1)) * dtdz * mh(i, j, k));
        const Real wsp = Kokkos::sqrt(bb * bb + c * c);
        if (wsp > m) m = wsp;
      }, Kokkos::Max<Real>(fmax));
  }
  return std::max(Real(0.0), fmax);
}

Real calcksquick(const Bounds& b, int axisymm, Real dt, Real rdx, Real rdy, const FArray1& uh, const FArray1& vh,
                 const FArray3& kmh, const FArray3& kmv, const FArray3& khh, const FArray3& /*khv*/) {
  const Real dtdx = dt * rdx * rdx;
  const Real dtdy = dt * rdy * rdy;
  Real fmax = 0.0;   // ks(k) = 0.0 to start
  if (b.nk < 2) return -99999999.0;
  if (b.nx > 1 && b.ny > 1) {
    Kokkos::parallel_reduce("calcksquick", box3(1, b.ni, 1, b.nj, 2, b.nk),
      KOKKOS_LAMBDA(int i, int j, int k, Real& m) {
        const Real a1 = kmh(i, j, k) * dtdx * uh(i) * uh(i), b1 = kmh(i, j, k) * dtdy * vh(j) * vh(j);
        const Real a2 = khh(i, j, k) * dtdx * uh(i) * uh(i), b2 = khh(i, j, k) * dtdy * vh(j) * vh(j);
        const Real a3 = kmv(i, j, k) * dtdx * uh(i) * uh(i), b3 = kmv(i, j, k) * dtdy * vh(j) * vh(j);
        const Real tem1 = Kokkos::sqrt(a1 * a1 + b1 * b1);
        const Real tem2 = Kokkos::sqrt(a2 * a2 + b2 * b2);
        const Real tem3 = Kokkos::sqrt(a3 * a3 + b3 * b3);
        if (tem1 > m) m = tem1;
        if (tem2 > m) m = tem2;
        if (tem3 > m) m = tem3;
      }, Kokkos::Max<Real>(fmax));
  } else if (b.nx > 1) {
    Kokkos::parallel_reduce("calcksquick2d", box3(1, b.ni, 1, b.nj, 2, b.nk),
      KOKKOS_LAMBDA(int i, int j, int k, Real& m) {
        const Real tem1 = kmh(i, j, k) * dtdx * uh(i) * uh(i);
        const Real tem2 = khh(i, j, k) * dtdx * uh(i) * uh(i);
        const Real tem3 = kmv(i, j, k) * dtdx * uh(i) * uh(i);
        if (tem1 > m) m = tem1;
        if (tem2 > m) m = tem2;
        if (tem3 > m) m = tem3;
      }, Kokkos::Max<Real>(fmax));
  } else if (axisymm == 0 && b.ny > 1) {
    throw std::runtime_error("cm1k: calcksquick with nx = 1 (misclibs.F: stop 1112)");
  }
  return std::max(Real(0.0), fmax);
}

AdaptDt AdaptDt::from_namelist(const Namelist& nl, const Bounds& b) {
  AdaptDt a;
  a.active = nl.geti("adapt_dt", 0) == 1;
  if (!a.active) return a;
  auto unsupported = [](const std::string& w) {
    throw std::runtime_error("cm1k: unsupported option with adapt_dt = 1: " + w);
  };
  a.cm1setup = nl.geti("cm1setup", 1);
  a.nx = b.nx; a.ny = b.ny;
  a.dtl = nl.getr("dtl", 0.0);
  a.tapfrq = nl.getr("tapfrq", 0.0);
  a.statfrq = nl.getr("statfrq", 0.0);
  a.rstfrq = nl.getr("rstfrq", 0.0);
  a.dbldt = double(a.dtl);
  a.taptim = 0.0; a.stattim = 0.0; a.rsttim = a.rstfrq;   // param.F
  // startup pass flags (cm1.F: startup = .true.)
  a.dowriteout = true; a.dostat = true; a.dorestart = false;
  const int psolver = nl.geti("psolver", 3);
  if (psolver == 1) unsupported("psolver = 1 (param.F stops)");
  if (std::abs(a.rstfrq) < 1.0e-10) unsupported("rstfrq = 0 (write a restart and stop)");
  if (nl.geti("radopt", 0) != 0) unsupported("radiation (radopt >= 1: radtim/dtrad timing)");
  if (nl.getb("dodomaindiag", false)) unsupported("dodomaindiag (diagtim ramp-down)");
  if (nl.getb("doazimavg", false)) unsupported("doazimavg (azimavgtim ramp-down)");
  if (nl.getb("dohifrq", false)) unsupported("dohifrq (hifrqtim ramp-down)");
  if (nl.geti("iprcl", 0) != 0) unsupported("iprcl = 1");
  if (nl.geti("ptype", 0) == 2 || nl.geti("ptype", 0) == 4) unsupported("ptype = 2, 4 (consat2/lfoice_init(dt))");
  // calcksquick reads kmh..khv on the interior: CM1 dimensions them 1:1
  // unless an eddy-viscosity scheme is on (param.F iusekm), so cm1setup >= 1
  // without one reads unallocated memory in the Fortran.
  if (a.cm1setup >= 1 && !(b.iec > 1)) unsupported("cm1setup >= 1 without subgrid turbulence (sgsmodel = 0, ipbl = 0)");
  return a;
}

void AdaptDt::getnewdt(double mtime, Real dt, bool doinit) {
  constexpr Real cfl_limit = 1.00;    // maximum CFL allowed (actually a "target" value)
  constexpr Real max_change = 0.10;   // maximum (percentage) change in timestep
  Real ks_limit = 0;
  if (cm1setup >= 1) {
    if (nx > 3 && ny > 3) ks_limit = 0.18 / 2.0;   // 3d
    else                  ks_limit = 0.18;         // 2d
  }
  cflmax = std::max(cflmax, Real(1.0e-10));
  if (cflmax > cfl_limit) dbldt = dbldt * (cfl_limit / cflmax);
  else                    dbldt = dbldt * std::min(Real(1.0 + max_change), Real(cfl_limit / cflmax));
  if (cm1setup >= 1 && (ksmax * dbldt / dt) > ks_limit)
    dbldt = std::min(dbldt, double(dt * ks_limit / std::max(Real(1.0e-10), ksmax)));
  // don't allow dt to exceed twice initial timestep
  dbldt = std::min(dbldt, double(2.0 * dtl));
  if (!doinit) {
    // ramp-down timestep when approaching output / restart / stat time
    auto ramp = [&](double tim, Real frq, bool flag) {
      if (!(tim > 0.0)) return;
      const double tout = flag ? ((tim - mtime) + frq) : (tim - mtime);
      if (tout > (2.00001 * dbldt) && tout <= (3.0 * dbldt))      dbldt = tout / 3.0;
      else if (tout > (1.00001 * dbldt) && tout <= (2.0 * dbldt)) dbldt = tout / 2.0;
      else if (tout <= dbldt)                                     dbldt = tout;
    };
    ramp(taptim, tapfrq, dowriteout);
    ramp(rsttim, rstfrq, dorestart);
    ramp(stattim, statfrq, dostat);
  }
}

void AdaptDt::advance_timers(double mtime) {
  // statpack (dostat), writeout (dowriteout), restart_write (dorestart)
  if (dostat && statfrq > 0.0) {
    do { stattim = stattim + statfrq; } while (!(stattim > mtime));
  }
  if (dowriteout && tapfrq > 0.0) {
    do { taptim = taptim + tapfrq; } while (!(taptim > mtime));
  }
  if (dorestart && rstfrq > 0.0) {
    do { rsttim = rsttim + rstfrq; } while (!(rsttim > mtime));
  }
}

void AdaptDt::set_flags(double mtime) {
  dowriteout = (mtime + dbldt) >= (taptim - 0.1 * dbldt) || tapfrq < 0.0;
  dostat = (mtime + dbldt) >= (stattim - 0.1 * dbldt) || statfrq < 0.0;
  dorestart = (mtime + dbldt) >= (rsttim - 0.1 * dbldt) && rstfrq > 0.0;
}

} // namespace cm1k
