// morrison.F GRAUPEL_INIT, GAMMA, DERF1 (host side), and the Options.
// Operation order follows the Fortran (left-to-right a*b*c, a*b/c, ...);
// x**n for a constant integer n is gfortran's multiplication chain
// (x**3 = (x*x)*x), a real exponent is pow().
#include "physics/morrison/cm1k_morrison.hpp"
#include "physics/morrison/cm1k_morrison_impl.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cm1k::morrison {

Options Options::from_namelist(const Namelist& nl) {
  if (nl.geti("imoist", 0) != 1 || nl.geti("ptype", 0) != 5)
    throw std::runtime_error("morrison::Options: requires imoist=1 and ptype=5");
  Options o;
  o.ihail = nl.geti("ihail", 0);
  o.ndcnst = nl.getr("ndcnst", 250.0);
  o.inum = 1;   // GRAUPEL_INIT
  o.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  const int psolver = nl.geti("psolver", 3);
  if (psolver >= 4 && psolver <= 7 && o.eqtset == 2) o.eqtset = 1;
  o.efall = nl.geti("efall", 0);
  o.radopt = nl.geti("radopt", 0);
  o.nrain = (nl.geti("imove", 0) == 1) ? 2 : 1;
  return o;
}

Real gamma_host(Real x) { return impl::gamma(x); }
Real derf1_host(Real x) { return impl::derf1(x); }

int satadj2_host(const Params& p, Real& tnew, Real& prs, Real& qvnew, Real& qcnew, Real qrnew,
                 Real qinew, Real qsnew, Real qgnew, Real rho, int eqtset, Real tsmall) {
  return impl::satadj2(p, tnew, prs, qvnew, qcnew, qrnew, qinew, qsnew, qgnew, rho, eqtset, tsmall);
}

Params Params::init(int cm1hail, Real cm1ndcnst, int inum) {
  using impl::gamma;
  using impl::PI;
  Params P;
  P.inum = inum;
  P.ndcnst = cm1ndcnst;
  P.ihail = cm1hail;

  // FALLSPEED PARAMETERS (V=AD^B)
  P.ai = 700.0;
  P.ac = 3.0e7;
  P.as = 11.72;
  P.ar = 841.99667;
  P.bi = 1.0;
  P.bc = 2.0;
  P.bs = 0.41;
  P.br = 0.8;
  if (P.ihail == 0) {
    P.ag = 19.3;
    P.bg = 0.37;
  } else {
    P.ag = 114.5;
    P.bg = 0.5;
  }

  // CONSTANTS AND PARAMETERS
  P.r = 287.04;
  P.rv = 461.5;
  P.cp = 1005.7;
  P.ep_2 = P.r / P.rv;
  P.cv = P.cp - P.r;
  P.cvv = 1408.5;
  P.cpv = 1870.0;
  P.cpl = 4190.0;
  P.cpi = 2106.0;
  P.xlv0 = 2501000.0;
  P.xls0 = 2834000.0;
  P.lv1 = P.xlv0 + (P.cpl - P.cpv) * 273.15;
  P.lv2 = P.cpl - P.cpv;
  P.ls1 = P.xls0 + (P.cpi - P.cpv) * 273.15;
  P.ls2 = P.cpi - P.cpv;
  P.tmelt = 273.15;
  P.rhosu = 85000.0 / (P.r * P.tmelt);
  P.rhow = 997.0;
  P.rhoi = 500.0;
  P.rhosn = 100.0;
  P.rhog = (P.ihail == 0) ? 400.0 : 900.0;
  P.aimm = 0.66;
  P.bimm = 100.0;
  P.ecr = 1.0;
  P.dcs = 125.0e-6;
  // (10.E-6)**3 and (5.E-6)**3 are folded by gfortran (correctly rounded);
  // for these two the multiplication chain gives the same double.
  P.mi0 = 4.0 / 3.0 * PI * P.rhoi * ((10.0e-6 * 10.0e-6) * 10.0e-6);
  P.mg0 = 1.6e-10;
  P.f1s = 0.86;
  P.f2s = 0.28;
  P.f1r = 0.78;
  P.f2r = 0.308;
  P.g = 9.81;
  P.qsmall = 1.0e-14;
  P.qthr = 1.0e-10;
  P.eii = 0.1;
  P.eci = 0.7;
  P.cpw = P.cpl;

  // SIZE DISTRIBUTION PARAMETERS
  P.ci = P.rhoi * PI / 6.0;
  P.di = 3.0;
  P.cs = P.rhosn * PI / 6.0;
  P.ds = 3.0;
  P.cg = P.rhog * PI / 6.0;
  P.dg = 3.0;

  P.rin = 0.1e-6;
  P.mmult = 4.0 / 3.0 * PI * P.rhoi * ((5.0e-6 * 5.0e-6) * 5.0e-6);

  // SIZE LIMITS FOR LAMBDA
  P.lammaxi = 1.0 / 1.0e-6;
  P.lammini = 1.0 / (2.0 * P.dcs + 100.0e-6);
  P.lammaxr = 1.0 / 20.0e-6;
  P.lamminr = 1.0 / 2800.0e-6;
  P.d_b = 300.0e-6;
  P.lammaxs = 1.0 / 10.0e-6;
  P.lammins = 1.0 / 2000.0e-6;
  P.lammaxg = 1.0 / 20.0e-6;
  P.lamming = 1.0 / 2000.0e-6;

  // CCN SPECTRA FOR IACT = 1
  P.k1 = 0.4;
  P.c1 = 120.0;

  // AEROSOL ACTIVATION PARAMETERS FOR IACT = 2
  P.mw = 0.018;
  P.osm = 1.0;
  P.vi = 3.0;
  P.epsm = 0.7;
  P.rhoa = 1777.0;
  P.map = 0.132;
  P.ma = 0.0284;
  P.rr = 8.3145;
  P.bact = P.vi * P.osm * P.epsm * P.mw * P.rhoa / (P.map * P.rhow);

  // MODE 1
  P.rm1 = 0.052e-6;
  P.sig1 = 2.04;
  P.nanew1 = 72.2e6;
  {
    const Real l = std::log(P.sig1);
    P.f11 = 0.5 * std::exp(2.5 * (l * l));
    P.f21 = 1.0 + 0.25 * std::log(P.sig1);
  }
  // MODE 2
  P.rm2 = 1.3e-6;
  P.sig2 = 2.5;
  P.nanew2 = 1.8e6;
  {
    const Real l = std::log(P.sig2);
    P.f12 = 0.5 * std::exp(2.5 * (l * l));
    P.f22 = 1.0 + 0.25 * std::log(P.sig2);
  }

  // CONSTANTS FOR EFFICIENCY
  Real* C = P.cons;
  C[0] = 0.0;
  C[1] = gamma(1.0 + P.ds) * P.cs;
  C[2] = gamma(1.0 + P.dg) * P.cg;
  C[3] = gamma(4.0 + P.bs) / 6.0;
  C[4] = gamma(4.0 + P.br) / 6.0;
  C[5] = gamma(1.0 + P.bs);
  C[6] = gamma(1.0 + P.br);
  C[7] = gamma(4.0 + P.bg) / 6.0;
  C[8] = gamma(1.0 + P.bg);
  C[9] = gamma(5.0 / 2.0 + P.br / 2.0);
  C[10] = gamma(5.0 / 2.0 + P.bs / 2.0);
  C[11] = gamma(5.0 / 2.0 + P.bg / 2.0);
  C[12] = gamma(1.0 + P.di) * P.ci;
  C[13] = gamma(P.bs + 3.0) * PI / 4.0 * P.eci;
  C[14] = gamma(P.bg + 3.0) * PI / 4.0 * P.eci;
  C[15] = -1108.0 * P.eii * std::pow(PI, (1.0 - P.bs) / 3.0) * std::pow(P.rhosn, (-2.0 - P.bs) / 3.0) /
          (4.0 * 720.0);
  C[16] = gamma(P.bi + 3.0) * PI / 4.0 * P.eci;
  C[17] = 4.0 * 2.0 * 3.0 * P.rhosu * PI * P.eci * P.eci * gamma(2.0 * P.bs + 2.0) / (8.0 * (P.rhog - P.rhosn));
  C[18] = P.rhosn * P.rhosn;
  C[19] = P.rhow * P.rhow;
  C[20] = 20.0 * PI * PI * P.rhow * P.bimm;
  C[21] = 4.0 / (P.dcs * P.rhoi);
  C[22] = PI * P.rhoi * ((P.dcs * P.dcs) * P.dcs) / 6.0;
  C[23] = PI / 4.0 * P.eii * gamma(P.bs + 3.0);
  C[24] = PI / 4.0 * P.ecr * gamma(P.br + 3.0);
  C[25] = PI * PI / 24.0 * P.rhow * P.ecr * gamma(P.br + 6.0);
  C[26] = PI / 6.0 * P.rhow;
  C[27] = gamma(1.0 + P.bi);
  C[28] = gamma(4.0 + P.bi) / 6.0;
  // (25.E-6)**3 is folded by gfortran to the correctly rounded cube, which
  // is one ulp away from (x*x)*x: 0x1.19799812dea12p-46.
  C[29] = 4.0 / 3.0 * PI * P.rhow * 0x1.19799812dea12p-46;
  C[30] = 4.0 / 3.0 * PI * P.rhow;
  C[31] = PI * PI * P.ecr * P.rhosn;
  C[32] = PI / 2.0 * P.ecr;
  C[33] = PI * PI * P.ecr * P.rhog;
  C[34] = 5.0 / 2.0 + P.br / 2.0;
  C[35] = 5.0 / 2.0 + P.bs / 2.0;
  C[36] = 5.0 / 2.0 + P.bg / 2.0;
  C[37] = 4.0 * PI * 1.38e-23 / (6.0 * PI * P.rin);
  C[38] = PI * PI / 3.0 * P.rhow;
  C[39] = PI * PI / 36.0 * P.rhow * P.bimm;
  C[40] = PI / 6.0 * P.bimm;
  C[41] = PI * PI * P.ecr * P.rhow;
  return P;
}

} // namespace cm1k::morrison
