// Port of src/adv.F (advs, advu, advv, advw) and src/pdef.F (pdefx1/x2,
// pdefy1/y2, pdefz). See cm1k_adv.hpp for what is supported.
#include "dynamics/cm1k_adv.hpp"
#include "dynamics/cm1k_adv_routines.hpp"
#include "dynamics/cm1k_bc.hpp"
#include "share/cm1k_constants.hpp"

#include <stdexcept>
#include <string>

namespace cm1k {

using adv::WENO5;
using adv::fmax2;

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

AdvOptions AdvOptions::from_namelist(const Namelist& nl) {
  AdvOptions o;
  o.hadvordrs = nl.geti("hadvordrs", 5);
  o.vadvordrs = nl.geti("vadvordrs", 5);
  o.hadvordrv = nl.geti("hadvordrv", 5);
  o.vadvordrv = nl.geti("vadvordrv", 5);
  o.advwenos = nl.geti("advwenos", 0);
  o.advwenov = nl.geti("advwenov", 0);
  o.weno_order = nl.geti("weno_order", 5);
  o.idiff = nl.geti("idiff", 0);
  o.mdiff = nl.geti("mdiff", 0);
  o.difforder = nl.geti("difforder", 6);
  o.kdiff6 = nl.getr("kdiff6", 0.040);
  o.wbc = nl.geti("wbc", 1);
  o.ebc = nl.geti("ebc", 1);
  o.sbc = nl.geti("sbc", 1);
  o.nbc = nl.geti("nbc", 1);
  o.terrain_flag = nl.getb("terrain_flag", false);
  o.axisymm = nl.geti("axisymm", 0);
  o.do_ib = nl.getb("do_ib", false);
  const int testcase = nl.geti("testcase", 0);
  o.dolsw = (testcase >= 3 && testcase <= 7);   // param.F
  // param.F: wenocheck
  if (o.advwenos >= 1 && (o.weno_order == 3 || o.weno_order == 5 || o.weno_order == 7 || o.weno_order == 9)) {
    o.hadvordrs = o.weno_order; o.vadvordrs = o.weno_order;
  }
  if (o.advwenov >= 1 && (o.weno_order == 3 || o.weno_order == 5 || o.weno_order == 7 || o.weno_order == 9)) {
    o.hadvordrv = o.weno_order; o.vadvordrv = o.weno_order;
  }
  if (o.axisymm == 1 && o.idiff == 1 && o.difforder == 6) { o.idiff = 0; o.difforder = 0; }
  o.set_boundary_flags();
  return o;
}

void AdvOptions::set_boundary_flags() {
  // param.F (serial): ibw=1 iff wbc /= 1 (this process has a non-periodic west boundary)
  ibw = (wbc != 1) ? 1 : 0;
  ibe = (ebc != 1) ? 1 : 0;
  ibs = (sbc != 1) ? 1 : 0;
  ibn = (nbc != 1) ? 1 : 0;
}

void AdvOptions::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k advection: " + m); };
  if (axisymm != 0) fail("axisymm = 1 is not ported");
  if (do_ib) fail("do_ib = .true. (immersed boundaries) is not ported");
  if (dolsw && terrain_flag) fail("large-scale subsidence (dolsw) with terrain (param.F stops)");
  for (int v : {hadvordrs, vadvordrs, hadvordrv, vadvordrv})
    if (v < 2 || v > 10) fail("advection order " + std::to_string(v) + " (must be 2..10)");
  if (advwenos < 0 || advwenos > 2) fail("advwenos must be 0, 1 or 2");
  if (advwenov < 0 || advwenov > 2) fail("advwenov must be 0, 1 or 2");
  if ((advwenos >= 1 || advwenov >= 1) && weno_order != 5)
    fail("weno_order = " + std::to_string(weno_order) + " is not ported (only 5)");
  if (mdiff != 0 && mdiff != 1) fail("mdiff must be 0 or 1");
  for (int v : {wbc, ebc, sbc, nbc})
    if (v < 1 || v > 4) fail("lateral boundary condition " + std::to_string(v) + " is not ported");
}

//------------------------------------------------------------------------------

Advection::Advection(const Grid& grid, const BaseState& base, const AdvOptions& opt)
  : m_g(grid), m_b(base), m_opt(opt)
{
  m_opt.check();
  const auto& b = grid.b;
  dumx = FArray3("adv_dumx", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  dumy = FArray3("adv_dumy", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  dumz = FArray3("adv_dumz", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  advx = FArray3("adv_advx", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  advy = FArray3("adv_advy", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  advz = FArray3("adv_advz", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
}

namespace {

int hscheme(bool doweno, int weno_order, int order) { return doweno ? (weno_order == 5 ? WENO5 : -weno_order) : order; }

// Halo fill of bc.F `bcs` restricted to what pdefx2/pdefy2 read:
// x halos for j=1..nj, k=1..nk (dir 0), y halos for i=1..ni (dir 1).
void bcs_halo(const Bounds& b, const AdvOptions& o, const FArray3& s, int dir) {
  const int ni = b.ni, nj = b.nj, nk = b.nk, ng = b.ngxy;
  if (dir == 0) {
    const int wbc = o.wbc, ebc = o.ebc;
    Kokkos::parallel_for("bcs_x", box3(1, ng, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (wbc == 1)                  s(1 - i, j, k) = s(pwrap(ni + 1 - i, 1, ni), j, k);
      else if (wbc == 2)             s(1 - i, j, k) = s(1, j, k);
      else                           s(1 - i, j, k) = s(i, j, k);
    });
    Kokkos::parallel_for("bcs_x", box3(1, ng, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (ebc == 1)                  s(ni + i, j, k) = s(pwrap(i, 1, ni), j, k);
      else if (ebc == 2)             s(ni + i, j, k) = s(ni, j, k);
      else                           s(ni + i, j, k) = s(ni + 1 - i, j, k);
    });
  } else {
    const int sbc = o.sbc, nbc = o.nbc;
    Kokkos::parallel_for("bcs_y", box3(1, ni, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (sbc == 1)                  s(i, 1 - j, k) = s(i, pwrap(nj + 1 - j, 1, nj), k);
      else if (sbc == 2)             s(i, 1 - j, k) = s(i, 1, k);
      else                           s(i, 1 - j, k) = s(i, j, k);
    });
    Kokkos::parallel_for("bcs_y", box3(1, ni, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (nbc == 1)                  s(i, nj + j, k) = s(i, pwrap(j, 1, nj), k);
      else if (nbc == 2)             s(i, nj + j, k) = s(i, nj, k);
      else                           s(i, nj + j, k) = s(i, nj + 1 - j, k);
    });
  }
}

// Mass redistribution of pdefx2/pdefy2/pdefz, written as a gather: the
// Fortran scatters from each negative point s to s-1, s and s+1 in
// increasing s order, so the target t receives (in this order) the
// contribution of source t-1, then t, then t+1. Starting from 0.0 and
// adding in the same order gives the same bits. Returns flag(t).
struct PdefLine {
  FArray3 dum;
  int i, j, k, di, dj, dk;
  KOKKOS_FORCEINLINE_FUNCTION Real d(int n) const { return dum(i + n*di, j + n*dj, k + n*dk); }
};
KOKKOS_FORCEINLINE_FUNCTION bool pdef_gather(const PdefLine& l, Real smeps, Real& mass) {
  // source at offset n (n = -1, 0, 1 relative to the target)
  bool flag = false;
  mass = 0.0;
  for (int n = -1; n <= 1; ++n) {
    const Real dn = l.d(n);
    if (dn < 0.0) {
      const Real foo1 = fmax2(0.0, l.d(n - 1));
      const Real foo2 = fmax2(0.0, l.d(n + 1));
      if (foo1 + foo2 > smeps) {
        const Real foo3 = fmax2(dn, -(foo1+foo2))/(foo1+foo2);
        if (n == -1) {        // target is the source's i+1 neighbour
          mass = mass+foo2*foo3;
          if (l.d(0) > smeps) flag = true;
        } else if (n == 0) {
          mass = mass-(foo1+foo2)*foo3;
          flag = true;
        } else {              // target is the source's i-1 neighbour
          mass = mass+foo1*foo3;
          if (l.d(0) > smeps) flag = true;
        }
      }
    }
  }
  return flag;
}

} // namespace

//------------------------------------------------------------------------------
// wsub (adv_routines.F): advection by a specified large-scale w profile
// (dolsw). dumz gets the fluxes, subs the tendency on 1..ix x 1..jy x 1..nk.
//------------------------------------------------------------------------------

void Advection::set_wprof(const FArray1& wprof) {
  m_wprof = wprof;
  const auto& b = m_g.b;
  subs = FArray3("adv_subs", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  dumzs = FArray3("adv_dumzs", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
}

void Advection::wsub(int ix, int jy, const FArray3& a, double weps) const {
  if (!m_wprof.is_allocated()) throw std::runtime_error("cm1k advection: dolsw needs set_wprof()");
  const int nk = m_g.b.nk;
  const Real rdz = m_g.rdz;
  const FArray3 c1 = m_g.c1, c2 = m_g.c2, mh = m_g.mh;
  const FArray3 rr0 = m_b.rr0, rf0 = m_b.rf0;
  const FArray1 wprof = m_wprof;
  const FArray3 dumz = this->dumzs, subs = this->subs;
  Kokkos::parallel_for("wsub_flux", box3(1, ix, 1, jy, 1, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
    Real d;
    if (k == 1 || k == nk + 1) {
      d = 0.0;
    } else if (k == nk) {
      d = rf0(1, 1, nk)*wprof(nk)*(c1(i, j, nk)*a(i, j, nk - 1)+c2(i, j, nk)*a(i, j, nk));
    } else if (wprof(k) <= 0.0) {
      d = rf0(1, 1, k)*wprof(k)*adv::upstrpd(a(i, j, k + 1), a(i, j, k), a(i, j, k - 1), weps);
    } else if (k == 2) {
      d = rf0(1, 1, k)*wprof(k)*(c1(i, j, k)*a(i, j, k - 1)+c2(i, j, k)*a(i, j, k));
    } else {
      d = rf0(1, 1, k)*wprof(k)*adv::upstrpd(a(i, j, k - 2), a(i, j, k - 1), a(i, j, k), weps);
    }
    dumz(i, j, k) = d;
  });
  Kokkos::parallel_for("wsub_ten", box3(1, ix, 1, jy, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real div = (rf0(1, 1, k + 1)*wprof(k + 1)-rf0(1, 1, k)*wprof(k))*rdz*mh(1, 1, k);
    subs(i, j, k) = ( -(dumz(i, j, k + 1)-dumz(i, j, k))*rdz*mh(1, 1, k)
                      +a(i, j, k)*div )*rr0(1, 1, k);
  });
}

//------------------------------------------------------------------------------
// advs
//------------------------------------------------------------------------------

void Advection::advs(int nrk, const AdvsArgs& args, Real dt, const AdvFluxes& f,
                     const FArray3& s0, const FArray3& s, const FArray3& sten) const
{
  const auto& o = m_opt;
  const auto& b = m_g.b;
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  const Real rdx = m_g.rdx, rdy = m_g.rdy, rdz = m_g.rdz;
  const FArray1 uh = m_g.uh, vh = m_g.vh;
  const FArray3 mh = m_g.mh;
  const FArray3 rho0 = m_b.rho0, rr0 = m_b.rr0;
  const FArray3 rru = f.rru, rrv = f.rrv, rrw = f.rrw, divx = f.divx;
  const FArray3 dumx = this->dumx, dumy = this->dumy, dumz = this->dumz;
  const FArray3 advx = this->advx, advy = this->advy, advz = this->advz;
  const bool terrain = o.terrain_flag;
  const FArray3 gz = m_g.gz, rgz = m_g.rgz;
  const FArray1 rdsf = m_g.rdsf;
  const double weps = args.weps;
  const int pdef = args.pdef;
  const Real smeps = o.smeps;

  if (args.hadvorder < 2 || args.hadvorder > 10 || args.vadvorder < 2 || args.vadvorder > 10)
    throw std::runtime_error("advs: advection order must be 2..10");

  bool doweno = false;
  if (args.wflag == 1)
    if (o.advwenos == 1 || (o.advwenos == 2 && nrk == phys::nrkmax)) doweno = true;
  if (doweno && o.weno_order != 5) throw std::runtime_error("advs: only weno_order = 5 is ported");

  Real coef = 0.0;
  if (args.diffit == 1) coef = o.kdiff6/64.0/dt;

  const bool doitw = (o.wbc == 2 && o.ibw == 1);
  const bool doite = (o.ebc == 2 && o.ibe == 1);
  const bool doits = (o.sbc == 2 && o.ibs == 1);
  const bool doitn = (o.nbc == 2 && o.ibn == 1);

  // horizontal fluxes
  hadv_flux(m_g, o, 1, hscheme(doweno, o.weno_order, args.hadvorder), pdef, weps, rru, rrv, s, dumx, dumy);

  // open bc: set hadv to zero at inflow
  if (doitw) Kokkos::parallel_for("advs_w", box3(1, 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rru(i, j, k) >= 0.0) dumx(i, j, k) = dumx(i + 1, j, k);
  });
  if (doite) Kokkos::parallel_for("advs_e", box3(ni + 1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rru(i, j, k) <= 0.0) dumx(i, j, k) = dumx(i - 1, j, k);
  });
  if (doits) Kokkos::parallel_for("advs_s", box3(1, ni, 1, 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rrv(i, j, k) >= 0.0) dumy(i, j, k) = dumy(i, j + 1, k);
  });
  if (doitn) Kokkos::parallel_for("advs_n", box3(1, ni, nj + 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rrv(i, j, k) <= 0.0) dumy(i, j, k) = dumy(i, j - 1, k);
  });

  Kokkos::parallel_for("advs_hadv", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    advx(i, j, k) = -(dumx(i + 1, j, k)-dumx(i, j, k))*rdx*uh(i);
    advy(i, j, k) = -(dumy(i, j + 1, k)-dumy(i, j, k))*rdy*vh(j);
  });

  // open bc: subtract-off the divx piece at inflow
  if (doitw) Kokkos::parallel_for("advs_w2", box3(1, 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rru(1, j, k) >= 0.0) advx(i, j, k) = advx(i, j, k)-s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i);
  });
  if (doite) Kokkos::parallel_for("advs_e2", box3(ni, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rru(ni + 1, j, k) <= 0.0) advx(i, j, k) = advx(i, j, k)-s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i);
  });
  if (doits) Kokkos::parallel_for("advs_s2", box3(1, ni, 1, 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rrv(i, 1, k) >= 0.0) advy(i, j, k) = advy(i, j, k)-s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j);
  });
  if (doitn) Kokkos::parallel_for("advs_n2", box3(1, ni, nj, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (rrv(i, nj + 1, k) <= 0.0) advy(i, j, k) = advy(i, j, k)-s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j);
  });

  // 6th-order diffusion
  if (args.diffit == 1) {
    const FArray3 rho = f.rho, ri = f.rr;
    const int mdiff = o.mdiff;
    Kokkos::parallel_for("advs_d6x", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(s(i  , j, k)-s(i - 1, j, k))
                 -5.0*(s(i + 1, j, k)-s(i - 2, j, k))
                     +(s(i + 2, j, k)-s(i - 3, j, k)) )
              *0.5*(rho(i - 1, j, k)+rho(i, j, k));
      if (mdiff == 1 && d*(s(i, j, k)-s(i - 1, j, k)) <= 0.0) d = 0.0;
      dumx(i, j, k) = d;
    });
    Kokkos::parallel_for("advs_d6y", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(s(i, j  , k)-s(i, j - 1, k))
                 -5.0*(s(i, j + 1, k)-s(i, j - 2, k))
                     +(s(i, j + 2, k)-s(i, j - 3, k)) )
              *0.5*(rho(i, j - 1, k)+rho(i, j, k));
      if (mdiff == 1 && d*(s(i, j, k)-s(i, j - 1, k)) <= 0.0) d = 0.0;
      dumy(i, j, k) = d;
    });
    Kokkos::parallel_for("advs_d6", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      advx(i, j, k) = advx(i, j, k)+coef*(dumx(i + 1, j, k)-dumx(i, j, k))*ri(i, j, k)*rho0(i, j, k);
      advy(i, j, k) = advy(i, j, k)+coef*(dumy(i, j + 1, k)-dumy(i, j, k))*ri(i, j, k)*rho0(i, j, k);
    });
  }

  const bool dopdef = (pdef == 1);   // pdscheme = 1 (hard-wired in param.F)
  // pdefx1 / pdefy1: dum = rho0*s0 + dt*(adv + s*div)   (dumx, dumy reused)
  if (dopdef) {
    if (terrain)
      Kokkos::parallel_for("pdefx1_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dumx(i, j, k) = rho0(i, j, k)*s0(i, j, k)+dt*( advx(i, j, k)
                       +s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i) )*gz(i, j, 1);
      });
    else
    Kokkos::parallel_for("pdefx1", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumx(i, j, k) = rho0(1, 1, k)*s0(i, j, k)+dt*( advx(i, j, k)
                     +s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i) );
    });
    if (doitw) Kokkos::parallel_for("pdefx1_w", box3(1, 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (rru(1, j, k) >= 0.0) dumx(i, j, k) = rho0(i, j, k)*s0(i, j, k);
    });
    if (doite) Kokkos::parallel_for("pdefx1_e", box3(ni, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (rru(ni + 1, j, k) <= 0.0) dumx(i, j, k) = rho0(i, j, k)*s0(i, j, k);
    });
    bcs_halo(b, o, dumx, 0);

    if (terrain)
      Kokkos::parallel_for("pdefy1_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dumy(i, j, k) = rho0(i, j, k)*s0(i, j, k)+dt*( advy(i, j, k)
                       +s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j) )*gz(i, j, 1);
      });
    else
    Kokkos::parallel_for("pdefy1", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      dumy(i, j, k) = rho0(1, 1, k)*s0(i, j, k)+dt*( advy(i, j, k)
                     +s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j) );
    });
    if (doits) Kokkos::parallel_for("pdefy1_s", box3(1, ni, 1, 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (rrv(i, 1, k) >= 0.0) dumy(i, j, k) = rho0(i, j, k)*s0(i, j, k);
    });
    if (doitn) Kokkos::parallel_for("pdefy1_n", box3(1, ni, nj, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (rrv(i, nj + 1, k) <= 0.0) dumy(i, j, k) = rho0(i, j, k)*s0(i, j, k);
    });
    bcs_halo(b, o, dumy, 1);
  }

  // vertical fluxes
  vadv_flux(m_g, o, 1, hscheme(doweno, o.weno_order, args.vadvorder), args.pdefweno, weps, rrw, s, dumz);

  if (terrain)
    Kokkos::parallel_for("advs_vadv_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (k == 1)       advz(i, j, k) = -dumz(i, j, 2)*rdsf(1);
      else if (k == nk) advz(i, j, k) = +dumz(i, j, nk)*rdsf(nk);
      else              advz(i, j, k) = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdsf(k);
    });
  else
  Kokkos::parallel_for("advs_vadv", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (k == 1)       advz(i, j, k) = -dumz(i, j, 2)*rdz*mh(1, 1, 1);
    else if (k == nk) advz(i, j, k) = +dumz(i, j, nk)*rdz*mh(1, 1, nk);
    else              advz(i, j, k) = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdz*mh(1, 1, k);
  });

  // large-scale subsidence: added to advz so that pdefz includes it
  if (o.dolsw) {
    wsub(ni, nj, s, weps);
    const FArray3 subs = this->subs;
    Kokkos::parallel_for("advs_subs", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      advz(i, j, k) = advz(i, j, k)+subs(i, j, k)*rho0(1, 1, k);
    });
  }

  if (dopdef) {
    const Real rdt = 1.0/dt;
    // pdefz
    if (terrain) {
      Kokkos::parallel_for("pdefz1_t", box3(1, ni, 1, nj, 0, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
        if (k == 0 || k == nk + 1) dumz(i, j, k) = 0.0;
        else dumz(i, j, k) = rho0(i, j, k)*s0(i, j, k)+dt*( advz(i, j, k)
                            +s(i, j, k)*(rrw(i, j, k + 1)-rrw(i, j, k))*rdsf(k) )*gz(i, j, 1);
      });
      Kokkos::parallel_for("pdefz2_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        Real mass;
        if (pdef_gather(PdefLine{dumz, i, j, k, 0, 0, 1}, smeps, mass))
          advz(i, j, k) = (dumz(i, j, k)+mass-rho0(i, j, k)*s0(i, j, k))*rdt*rgz(i, j, 1)
                         -s(i, j, k)*(rrw(i, j, k + 1)-rrw(i, j, k))*rdsf(k);
      });
      Kokkos::parallel_for("pdefx2_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        Real mass;
        if (pdef_gather(PdefLine{dumx, i, j, k, 1, 0, 0}, smeps, mass))
          advx(i, j, k) = (dumx(i, j, k)+mass-rho0(i, j, k)*s0(i, j, k))*rdt*rgz(i, j, 1)
                         -s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i);
      });
      Kokkos::parallel_for("pdefy2_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        Real mass;
        if (pdef_gather(PdefLine{dumy, i, j, k, 0, 1, 0}, smeps, mass))
          advy(i, j, k) = (dumy(i, j, k)+mass-rho0(i, j, k)*s0(i, j, k))*rdt*rgz(i, j, 1)
                         -s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j);
      });
    } else {
    Kokkos::parallel_for("pdefz1", box3(1, ni, 1, nj, 0, nk + 1), KOKKOS_LAMBDA(int i, int j, int k) {
      if (k == 0 || k == nk + 1) dumz(i, j, k) = 0.0;
      else dumz(i, j, k) = rho0(1, 1, k)*s0(i, j, k)+dt*( advz(i, j, k)
                          +s(i, j, k)*(rrw(i, j, k + 1)-rrw(i, j, k))*rdz*mh(1, 1, k) );
    });
    Kokkos::parallel_for("pdefz2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      // sources are k=1..nk only (dum(0)=dum(nk+1)=0 are never negative)
      Real mass;
      if (pdef_gather(PdefLine{dumz, i, j, k, 0, 0, 1}, smeps, mass))
        advz(i, j, k) = (dumz(i, j, k)+mass-rho0(1, 1, k)*s0(i, j, k))*rdt
                       -s(i, j, k)*(rrw(i, j, k + 1)-rrw(i, j, k))*rdz*mh(1, 1, k);
    });
    // pdefx2, pdefy2
    Kokkos::parallel_for("pdefx2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real mass;
      if (pdef_gather(PdefLine{dumx, i, j, k, 1, 0, 0}, smeps, mass))
        advx(i, j, k) = (dumx(i, j, k)+mass-rho0(1, 1, k)*s0(i, j, k))*rdt
                       -s(i, j, k)*(rru(i + 1, j, k)-rru(i, j, k))*rdx*uh(i);
    });
    Kokkos::parallel_for("pdefy2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real mass;
      if (pdef_gather(PdefLine{dumy, i, j, k, 0, 1, 0}, smeps, mass))
        advy(i, j, k) = (dumy(i, j, k)+mass-rho0(1, 1, k)*s0(i, j, k))*rdt
                       -s(i, j, k)*(rrv(i, j + 1, k)-rrv(i, j, k))*rdy*vh(j);
    });
    }
  }

  if (terrain)
    Kokkos::parallel_for("advs_sten_t", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      sten(i, j, k) = sten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+advz(i, j, k)
                                     +s(i, j, k)*divx(i, j, k) )*rr0(i, j, k)*gz(i, j, 1);
    });
  else
  Kokkos::parallel_for("advs_sten", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    sten(i, j, k) = sten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+advz(i, j, k)
                                   +s(i, j, k)*divx(i, j, k) )*rr0(1, 1, k);
  });
}

//------------------------------------------------------------------------------
// advu
//------------------------------------------------------------------------------

void Advection::advu(int nrk, Real dt, const AdvFluxes& f, const FArray3& u3d, const FArray3& uten) const
{
  const auto& o = m_opt;
  const int ni = m_g.b.ni, nj = m_g.b.nj, nk = m_g.b.nk;
  const Real rdx = m_g.rdx, rdy = m_g.rdy, rdz = m_g.rdz;
  const FArray1 uf = m_g.uf, vh = m_g.vh;
  const FArray3 mh = m_g.mh;
  const FArray3 rho0 = m_b.rho0, rr0 = m_b.rr0;
  const FArray3 rru = f.rru, rrv = f.rrv, rrw = f.rrw, divx = f.divx, rho = f.rho;
  const FArray3 dumx = this->dumx, dumy = this->dumy, dumz = this->dumz;
  const FArray3 advx = this->advx, advy = this->advy, advz = this->advz;

  const bool doweno = (o.advwenov == 1) || (o.advwenov == 2 && nrk == phys::nrkmax);
  if (doweno && o.weno_order != 5) throw std::runtime_error("advu: only weno_order = 5 is ported");
  const double weps = 100.0*phys::epsilon;
  const bool dodiff = (o.idiff >= 1 && o.difforder == 6);
  const Real coef = dodiff ? o.kdiff6/64.0/dt : 0.0;

  const int i1 = (o.ibw == 1) ? 2 : 1;
  const int i2 = (o.ibe == 1) ? ni + 1 - 1 : ni + 1;
  const bool doits = (o.sbc == 2 && o.ibs == 1);
  const bool doitn = (o.nbc == 2 && o.ibn == 1);

  hadv_flux(m_g, o, 2, hscheme(doweno, o.weno_order, o.hadvordrv), 0, weps, rru, rrv, u3d, dumx, dumy);

  if (doits) Kokkos::parallel_for("advu_s", box3(i1, i2, 1, 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rrv(i, j, k)+rrv(i - 1, j, k)) >= 0.0) dumy(i, j, k) = dumy(i, j + 1, k);
  });
  if (doitn) Kokkos::parallel_for("advu_n", box3(i1, i2, nj + 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rrv(i, j, k)+rrv(i - 1, j, k)) <= 0.0) dumy(i, j, k) = dumy(i, j - 1, k);
  });

  Kokkos::parallel_for("advu_hadv", box3(i1, i2, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    advx(i, j, k) = -(dumx(i, j, k)-dumx(i - 1, j, k))*rdx*uf(i);
    advy(i, j, k) = -(dumy(i, j + 1, k)-dumy(i, j, k))*rdy*vh(j);
  });

  if (doits) Kokkos::parallel_for("advu_s2", box3(i1, i2, 1, 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rrv(i, 1, k)+rrv(i - 1, 1, k)) >= 0.0)
      advy(i, j, k) = advy(i, j, k)-u3d(i, j, k)*0.5*(
                          (rrv(i - 1, j + 1, k)-rrv(i - 1, j, k))
                         +(rrv(i  , j + 1, k)-rrv(i  , j, k)) )*rdy*vh(j);
  });
  if (doitn) Kokkos::parallel_for("advu_n2", box3(i1, i2, nj, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rrv(i, nj + 1, k)+rrv(i - 1, nj + 1, k)) <= 0.0)
      advy(i, j, k) = advy(i, j, k)-u3d(i, j, k)*0.5*(
                          (rrv(i - 1, j + 1, k)-rrv(i - 1, j, k))
                         +(rrv(i  , j + 1, k)-rrv(i  , j, k)) )*rdy*vh(j);
  });

  if (dodiff) {
    const int mdiff = o.mdiff;
    Kokkos::parallel_for("advu_d6x", box3(1, ni + 2, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(u3d(i  , j, k)-u3d(i - 1, j, k))
                 -5.0*(u3d(i + 1, j, k)-u3d(i - 2, j, k))
                     +(u3d(i + 2, j, k)-u3d(i - 3, j, k)) )*rho(i - 1, j, k);
      if (mdiff == 1 && d*(u3d(i, j, k)-u3d(i - 1, j, k)) <= 0.0) d = 0.0;
      dumx(i, j, k) = d;
    });
    Kokkos::parallel_for("advu_d6y", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(u3d(i, j  , k)-u3d(i, j - 1, k))
                 -5.0*(u3d(i, j + 1, k)-u3d(i, j - 2, k))
                     +(u3d(i, j + 2, k)-u3d(i, j - 3, k)) )
              *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))
                     +(rho(i - 1, j, k)+rho(i, j - 1, k)) );
      if (mdiff == 1 && d*(u3d(i, j, k)-u3d(i, j - 1, k)) <= 0.0) d = 0.0;
      dumy(i, j, k) = d;
    });
    Kokkos::parallel_for("advu_d6", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      advx(i, j, k) = advx(i, j, k)+coef*(dumx(i + 1, j, k)-dumx(i, j, k))*(rho0(i - 1, j, k)+rho0(i, j, k))/(rho(i - 1, j, k)+rho(i, j, k));
      advy(i, j, k) = advy(i, j, k)+coef*(dumy(i, j + 1, k)-dumy(i, j, k))*(rho0(i - 1, j, k)+rho0(i, j, k))/(rho(i - 1, j, k)+rho(i, j, k));
    });
  }

  vadv_flux(m_g, o, 2, hscheme(doweno, o.weno_order, o.vadvordrv), 0, weps, rrw, u3d, dumz);

  if (o.terrain_flag) {
    const FArray1 rdsf = m_g.rdsf;
    const FArray3 gzu = m_g.gzu;
    Kokkos::parallel_for("advu_ten_t", box3(i1, i2, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real az;
      if (k == 1)       az = -dumz(i, j, k + 1)*rdsf(k);
      else if (k == nk) az = +dumz(i, j, k)*rdsf(k);
      else              az = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdsf(k);
      advz(i, j, k) = az;
      uten(i, j, k) = uten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                     +u3d(i, j, k)*0.5*(divx(i, j, k)+divx(i - 1, j, k)) )
                    *gzu(i, j, 1)/(0.5*(rho0(i - 1, j, k)+rho0(i, j, k)));
    });
    return;
  }
  Kokkos::parallel_for("advu_ten", box3(i1, i2, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    Real az;
    if (k == 1)       az = -dumz(i, j, k + 1)*rdz*mh(1, 1, k);
    else if (k == nk) az = +dumz(i, j, k)*rdz*mh(1, 1, k);
    else              az = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdz*mh(1, 1, k);
    advz(i, j, k) = az;
    uten(i, j, k) = uten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                   +u3d(i, j, k)*0.5*(divx(i, j, k)+divx(i - 1, j, k)) )*rr0(1, 1, k);
  });
  if (o.dolsw) {
    wsub(ni + 1, nj, u3d, weps);
    const FArray3 subs = this->subs;
    Kokkos::parallel_for("advu_subs", box3(1, ni + 1, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      uten(i, j, k) = uten(i, j, k)+subs(i, j, k);
    });
  }
}

//------------------------------------------------------------------------------
// advv
//------------------------------------------------------------------------------

void Advection::advv(int nrk, Real dt, const AdvFluxes& f, const FArray3& v3d, const FArray3& vten) const
{
  const auto& o = m_opt;
  const int ni = m_g.b.ni, nj = m_g.b.nj, nk = m_g.b.nk;
  const Real rdx = m_g.rdx, rdy = m_g.rdy, rdz = m_g.rdz;
  const FArray1 uh = m_g.uh, vf = m_g.vf;
  const FArray3 mh = m_g.mh;
  const FArray3 rho0 = m_b.rho0, rr0 = m_b.rr0;
  const FArray3 rru = f.rru, rrv = f.rrv, rrw = f.rrw, divx = f.divx, rho = f.rho;
  const FArray3 dumx = this->dumx, dumy = this->dumy, dumz = this->dumz;
  const FArray3 advx = this->advx, advy = this->advy, advz = this->advz;

  const bool doweno = (o.advwenov == 1) || (o.advwenov == 2 && nrk == phys::nrkmax);
  if (doweno && o.weno_order != 5) throw std::runtime_error("advv: only weno_order = 5 is ported");
  const double weps = 100.0*phys::epsilon;
  const bool dodiff = (o.idiff >= 1 && o.difforder == 6);
  const Real coef = dodiff ? o.kdiff6/64.0/dt : 0.0;

  const bool doitw = (o.wbc == 2 && o.ibw == 1);
  const bool doite = (o.ebc == 2 && o.ibe == 1);
  const int j1 = (o.ibs == 1) ? 2 : 1;
  const int j2 = (o.ibn == 1) ? nj + 1 - 1 : nj + 1;

  hadv_flux(m_g, o, 3, hscheme(doweno, o.weno_order, o.hadvordrv), 0, weps, rru, rrv, v3d, dumx, dumy);

  if (doitw) Kokkos::parallel_for("advv_w", box3(1, 1, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rru(i, j, k)+rru(i, j - 1, k)) >= 0.0) dumx(i, j, k) = dumx(i + 1, j, k);
  });
  if (doite) Kokkos::parallel_for("advv_e", box3(ni + 1, ni + 1, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rru(i, j, k)+rru(i, j - 1, k)) <= 0.0) dumx(i, j, k) = dumx(i - 1, j, k);
  });

  Kokkos::parallel_for("advv_hadv", box3(1, ni, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    advx(i, j, k) = -(dumx(i + 1, j, k)-dumx(i, j, k))*rdx*uh(i);
    advy(i, j, k) = -(dumy(i, j, k)-dumy(i, j - 1, k))*rdy*vf(j);
  });

  if (doitw) Kokkos::parallel_for("advv_w2", box3(1, 1, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rru(1, j, k)+rru(1, j - 1, k)) >= 0.0)
      advx(i, j, k) = advx(i, j, k)-v3d(i, j, k)*0.5*(
                  (rru(i + 1, j - 1, k)-rru(i, j - 1, k))
                 +(rru(i + 1, j  , k)-rru(i, j  , k)) )*rdx*uh(i);
  });
  if (doite) Kokkos::parallel_for("advv_e2", box3(ni, ni, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if ((rru(ni + 1, j, k)+rru(ni + 1, j - 1, k)) <= 0.0)
      advx(i, j, k) = advx(i, j, k)-v3d(i, j, k)*0.5*(
                  (rru(i + 1, j - 1, k)-rru(i, j - 1, k))
                 +(rru(i + 1, j  , k)-rru(i, j  , k)) )*rdx*uh(i);
  });

  if (dodiff) {
    const int mdiff = o.mdiff;
    Kokkos::parallel_for("advv_d6x", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(v3d(i  , j, k)-v3d(i - 1, j, k))
                 -5.0*(v3d(i + 1, j, k)-v3d(i - 2, j, k))
                     +(v3d(i + 2, j, k)-v3d(i - 3, j, k)) )
              *0.25*( (rho(i - 1, j - 1, k)+rho(i, j, k))
                     +(rho(i - 1, j, k)+rho(i, j - 1, k)) );
      if (mdiff == 1 && d*(v3d(i, j, k)-v3d(i - 1, j, k)) <= 0.0) d = 0.0;
      dumx(i, j, k) = d;
    });
    Kokkos::parallel_for("advv_d6y", box3(1, ni, 1, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real d = ( 10.0*(v3d(i, j  , k)-v3d(i, j - 1, k))
                 -5.0*(v3d(i, j + 1, k)-v3d(i, j - 2, k))
                     +(v3d(i, j + 2, k)-v3d(i, j - 3, k)) )*rho(i, j - 1, k);
      if (mdiff == 1 && d*(v3d(i, j, k)-v3d(i, j - 1, k)) <= 0.0) d = 0.0;
      dumy(i, j, k) = d;
    });
    Kokkos::parallel_for("advv_d6", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      advx(i, j, k) = advx(i, j, k)+coef*(dumx(i + 1, j, k)-dumx(i, j, k))*(rho0(i, j - 1, k)+rho0(i, j, k))/(rho(i, j - 1, k)+rho(i, j, k));
      advy(i, j, k) = advy(i, j, k)+coef*(dumy(i, j + 1, k)-dumy(i, j, k))*(rho0(i, j - 1, k)+rho0(i, j, k))/(rho(i, j - 1, k)+rho(i, j, k));
    });
  }

  vadv_flux(m_g, o, 3, hscheme(doweno, o.weno_order, o.vadvordrv), 0, weps, rrw, v3d, dumz);

  if (o.terrain_flag) {
    const FArray1 rdsf = m_g.rdsf;
    const FArray3 gzv = m_g.gzv;
    Kokkos::parallel_for("advv_ten_t", box3(1, ni, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      Real az;
      if (k == 1)       az = -dumz(i, j, k + 1)*rdsf(k);
      else if (k == nk) az = +dumz(i, j, k)*rdsf(k);
      else              az = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdsf(k);
      advz(i, j, k) = az;
      vten(i, j, k) = vten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                     +v3d(i, j, k)*0.5*(divx(i, j, k)+divx(i, j - 1, k)) )
                    *gzv(i, j, 1)/(0.5*(rho0(i, j - 1, k)+rho0(i, j, k)));
    });
    return;
  }
  Kokkos::parallel_for("advv_ten", box3(1, ni, j1, j2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    Real az;
    if (k == 1)       az = -dumz(i, j, k + 1)*rdz*mh(1, 1, k);
    else if (k == nk) az = +dumz(i, j, k)*rdz*mh(1, 1, k);
    else              az = -(dumz(i, j, k + 1)-dumz(i, j, k))*rdz*mh(1, 1, k);
    advz(i, j, k) = az;
    vten(i, j, k) = vten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                   +v3d(i, j, k)*0.5*(divx(i, j, k)+divx(i, j - 1, k)) )*rr0(1, 1, k);
  });
  if (o.dolsw) {
    wsub(ni, nj + 1, v3d, weps);
    const FArray3 subs = this->subs;
    Kokkos::parallel_for("advv_subs", box3(1, ni, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      vten(i, j, k) = vten(i, j, k)+subs(i, j, k);
    });
  }
}

//------------------------------------------------------------------------------
// advw
//------------------------------------------------------------------------------

void Advection::advw(int nrk, Real dt, const AdvFluxes& f, const FArray3& w3d, const FArray3& wten,
                     int hadvorder, int vadvorder, int advweno) const
{
  const auto& o = m_opt;
  const int ni = m_g.b.ni, nj = m_g.b.nj, nk = m_g.b.nk;
  const Real rdx = m_g.rdx, rdy = m_g.rdy, rdz = m_g.rdz;
  const FArray1 uh = m_g.uh, vh = m_g.vh;
  const FArray3 mf = m_g.mf, c1 = m_g.c1, c2 = m_g.c2;
  const FArray3 rf0 = m_b.rf0, rrf0 = m_b.rrf0;
  const FArray3 rru = f.rru, rrv = f.rrv, rrw = f.rrw, divx = f.divx, rho = f.rho;
  const FArray3 dumx = this->dumx, dumy = this->dumy, dumz = this->dumz;
  const FArray3 advx = this->advx, advy = this->advy, advz = this->advz;

  if (hadvorder < 2 || hadvorder > 10 || vadvorder < 2 || vadvorder > 10)
    throw std::runtime_error("advw: advection order must be 2..10");
  const bool doweno = (advweno == 1) || (advweno == 2 && nrk == phys::nrkmax);
  if (doweno && o.weno_order != 5) throw std::runtime_error("advw: only weno_order = 5 is ported");
  const double weps = 100.0*phys::epsilon;
  const bool dodiff = (o.idiff >= 1 && o.difforder == 6);
  const Real coef = dodiff ? o.kdiff6/64.0/dt : 0.0;

  const bool doitw = (o.wbc == 2 && o.ibw == 1);
  const bool doite = (o.ebc == 2 && o.ibe == 1);
  const bool doits = (o.sbc == 2 && o.ibs == 1);
  const bool doitn = (o.nbc == 2 && o.ibn == 1);

  hadv_flux(m_g, o, 4, hscheme(doweno, o.weno_order, hadvorder), 0, weps, rru, rrv, w3d, dumx, dumy);

  if (doitw) Kokkos::parallel_for("advw_w", box3(1, 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k));
    const Real cc1 = 1.0-cc2;
    const Real ubar = cc2*rru(i, j, k)+cc1*rru(i, j, k - 1);
    if (ubar >= 0.0) dumx(i, j, k) = dumx(i + 1, j, k);
  });
  if (doite) Kokkos::parallel_for("advw_e", box3(ni + 1, ni + 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k));
    const Real cc1 = 1.0-cc2;
    const Real ubar = cc2*rru(i, j, k)+cc1*rru(i, j, k - 1);
    if (ubar <= 0.0) dumx(i, j, k) = dumx(i - 1, j, k);
  });
  if (doits) Kokkos::parallel_for("advw_s", box3(1, ni, 1, 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k));
    const Real cc1 = 1.0-cc2;
    const Real vbar = cc2*rrv(i, j, k)+cc1*rrv(i, j, k - 1);
    if (vbar >= 0.0) dumy(i, j, k) = dumy(i, j + 1, k);
  });
  if (doitn) Kokkos::parallel_for("advw_n", box3(1, ni, nj + 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k));
    const Real cc1 = 1.0-cc2;
    const Real vbar = cc2*rrv(i, j, k)+cc1*rrv(i, j, k - 1);
    if (vbar <= 0.0) dumy(i, j, k) = dumy(i, j - 1, k);
  });

  Kokkos::parallel_for("advw_hadv", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    advx(i, j, k) = -(dumx(i + 1, j, k)-dumx(i, j, k))*rdx*uh(i);
    advy(i, j, k) = -(dumy(i, j + 1, k)-dumy(i, j, k))*rdy*vh(j);
  });

  if (doitw) Kokkos::parallel_for("advw_w2", box3(1, 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(0, j, k)+c2(1, j, k));
    const Real cc1 = 1.0-cc2;
    const Real ubar = cc2*rru(1, j, k)+cc1*rru(1, j, k - 1);
    if (ubar >= 0.0)
      advx(i, j, k) = advx(i, j, k)-w3d(i, j, k)*(
                  c1(i, j, k)*(rru(i + 1, j, k - 1)-rru(i, j, k - 1))
                 +c2(i, j, k)*(rru(i + 1, j, k  )-rru(i, j, k  )) )*rdx*uh(i);
  });
  if (doite) Kokkos::parallel_for("advw_e2", box3(ni, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(ni, j, k)+c2(ni + 1, j, k));
    const Real cc1 = 1.0-cc2;
    const Real ubar = cc2*rru(ni + 1, j, k)+cc1*rru(ni + 1, j, k - 1);
    if (ubar <= 0.0)
      advx(i, j, k) = advx(i, j, k)-w3d(i, j, k)*(
                  c1(i, j, k)*(rru(i + 1, j, k - 1)-rru(i, j, k - 1))
                 +c2(i, j, k)*(rru(i + 1, j, k  )-rru(i, j, k  )) )*rdx*uh(i);
  });
  if (doits) Kokkos::parallel_for("advw_s2", box3(1, ni, 1, 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i, 0, k)+c2(i, 1, k));
    const Real cc1 = 1.0-cc2;
    const Real vbar = cc2*rrv(i, 1, k)+cc1*rrv(i, 1, k - 1);
    if (vbar >= 0.0)
      advy(i, j, k) = advy(i, j, k)-w3d(i, j, k)*(
                         c1(i, j, k)*(rrv(i, j + 1, k - 1)-rrv(i, j, k - 1))
                        +c2(i, j, k)*(rrv(i, j + 1, k  )-rrv(i, j, k  )) )*rdy*vh(j);
  });
  if (doitn) Kokkos::parallel_for("advw_n2", box3(1, ni, nj, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real cc2 = 0.5*(c2(i, nj, k)+c2(i, nj + 1, k));
    const Real cc1 = 1.0-cc2;
    const Real vbar = cc2*rrv(i, nj + 1, k)+cc1*rrv(i, nj + 1, k - 1);
    if (vbar <= 0.0)
      advy(i, j, k) = advy(i, j, k)-w3d(i, j, k)*(
                         c1(i, j, k)*(rrv(i, j + 1, k - 1)-rrv(i, j, k - 1))
                        +c2(i, j, k)*(rrv(i, j + 1, k  )-rrv(i, j, k  )) )*rdy*vh(j);
  });

  if (dodiff) {
    const int mdiff = o.mdiff;
    Kokkos::parallel_for("advw_d6x", box3(1, ni + 1, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real cc2 = 0.5*(c2(i - 1, j, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      Real d = ( 10.0*(w3d(i  , j, k)-w3d(i - 1, j, k))
                 -5.0*(w3d(i + 1, j, k)-w3d(i - 2, j, k))
                     +(w3d(i + 2, j, k)-w3d(i - 3, j, k)) )
            *0.5*( cc2*(rho(i - 1, j, k  )+rho(i, j, k  ))
                  +cc1*(rho(i - 1, j, k - 1)+rho(i, j, k - 1)) );
      if (mdiff == 1 && d*(w3d(i, j, k)-w3d(i - 1, j, k)) <= 0.0) d = 0.0;
      dumx(i, j, k) = d;
    });
    Kokkos::parallel_for("advw_d6y", box3(1, ni, 1, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real cc2 = 0.5*(c2(i, j - 1, k)+c2(i, j, k));
      const Real cc1 = 1.0-cc2;
      Real d = ( 10.0*(w3d(i, j  , k)-w3d(i, j - 1, k))
                 -5.0*(w3d(i, j + 1, k)-w3d(i, j - 2, k))
                     +(w3d(i, j + 2, k)-w3d(i, j - 3, k)) )
            *0.5*( cc2*(rho(i, j - 1, k  )+rho(i, j, k  ))
                  +cc1*(rho(i, j - 1, k - 1)+rho(i, j, k - 1)) );
      if (mdiff == 1 && d*(w3d(i, j, k)-w3d(i, j - 1, k)) <= 0.0) d = 0.0;
      dumy(i, j, k) = d;
    });
    Kokkos::parallel_for("advw_d6", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      advx(i, j, k) = advx(i, j, k)+coef*(dumx(i + 1, j, k)-dumx(i, j, k))*rf0(i, j, k)/(0.5*(rho(i, j, k - 1)+rho(i, j, k)));
      advy(i, j, k) = advy(i, j, k)+coef*(dumy(i, j + 1, k)-dumy(i, j, k))*rf0(i, j, k)/(0.5*(rho(i, j, k - 1)+rho(i, j, k)));
    });
  }

  vadv_flux(m_g, o, 4, hscheme(doweno, o.weno_order, vadvorder), 0, weps, rrw, w3d, dumz);

  if (o.terrain_flag) {
    const FArray1 rds = m_g.rds;
    const FArray3 gz = m_g.gz;
    Kokkos::parallel_for("advw_ten_t", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real az = -(dumz(i, j, k)-dumz(i, j, k - 1))*rds(k);
      advz(i, j, k) = az;
      wten(i, j, k) = wten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                     +w3d(i, j, k)*(c2(i, j, k)*divx(i, j, k)+c1(i, j, k)*divx(i, j, k - 1)) )*rrf0(i, j, k)*gz(i, j, 1);
    });
    return;
  }
  Kokkos::parallel_for("advw_ten", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    const Real az = -(dumz(i, j, k)-dumz(i, j, k - 1))*rdz*mf(1, 1, k);
    advz(i, j, k) = az;
    wten(i, j, k) = wten(i, j, k)+( (advx(i, j, k)+advy(i, j, k))+az
                   +w3d(i, j, k)*(c2(1, 1, k)*divx(i, j, k)+c1(1, 1, k)*divx(i, j, k - 1)) )*rrf0(1, 1, k);
  });
}

} // namespace cm1k
