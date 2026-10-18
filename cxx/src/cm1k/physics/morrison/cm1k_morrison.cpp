// Port of morrison.F MP_GRAUPEL / M2005MICRO_GRAUPEL and of the ptype=5
// branch of mp_driver.F. One thread per column (i,j); the column work
// arrays of M2005MICRO live in Work::wk. Operation order and
// parenthesization follow the Fortran exactly (left-to-right a*b*c, a*b/c;
// x**n for constant integer n as gfortran's multiplication chain:
// x**2 = x*x, x**3 = (x*x)*x, x**4 = (x*x)*(x*x); real exponents as pow),
// so results match the double-precision Fortran bit for bit.
#include "physics/morrison/cm1k_morrison.hpp"
#include "physics/morrison/cm1k_morrison_impl.hpp"

#include "physics/kessler/cm1k_kessler.hpp"
#include "share/cm1k_moist.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::morrison {

namespace {

using impl::fmax2;
using impl::fmin2;
using impl::PI;

// Work::wk slots: M2005MICRO's column arrays that live across its k loops.
enum WorkVar {
  W_QCT = 0, W_QIT, W_QNIT, W_QRT, W_NCT, W_NIT, W_NST, W_NRT, W_TT, W_QVT, W_QGT, W_NGT,   // *3DTEN
  W_QRS, W_QIS, W_QNIS, W_QCS, W_QGS,                                                     // *STEN
  W_ECND, W_EDEP, W_EFRZ,
  W_AIN, W_ARN, W_ASN, W_ACN, W_AGN,
  W_LAMI, W_NC, W_P,
  W_FR, W_FI, W_FNI, W_FS, W_FNS, W_FNR, W_FC, W_FNC, W_FG, W_FNG,
  W_DUMI, W_DUMQS, W_DUMR, W_DUMFNI, W_DUMFNS, W_DUMFNR, W_DUMC, W_DUMFNC, W_DUMG, W_DUMFNG,
  NWORK
};

KOKKOS_INLINE_FUNCTION Real sq(Real x) { return x * x; }
KOKKOS_INLINE_FUNCTION Real cube(Real x) { return (x * x) * x; }
KOKKOS_INLINE_FUNCTION Real pow4(Real x) { const Real x2 = x * x; return x2 * x2; }
KOKKOS_INLINE_FUNCTION Real pw(Real x, Real y) { return Kokkos::pow(x, y); }

// One column of MP_GRAUPEL (incl. M2005MICRO_GRAUPEL).
struct Column {
  Params p;
  int nk, eqtset, radopt, nrain;
  bool dorad;
  Real dt, rdt, tsmall;
  FArray3 t, pres, rho, dzq, w;
  FArray4 q, wk;
  FArray3 rain;
  FArray2 prate;
  FArray3 vtc, vtr, vti, vts, vtg;
  FArray3 effc, effi, effs, effr, effg, effis;

  KOKKOS_INLINE_FUNCTION void operator()(int i, int j, int& err) const;
  KOKKOS_INLINE_FUNCTION int micro(int i, int j, Real& precrt, Real* eff1) const;
};

// Column arrays of M2005MICRO_GRAUPEL as macros on the (i,j) column, so the
// code below reads like the Fortran.
#define QC3D(k) q(i, j, k, nqc)
#define QI3D(k) q(i, j, k, nqi)
#define QNI3D(k) q(i, j, k, nqs)
#define QR3D(k) q(i, j, k, nqr)
#define QG3D(k) q(i, j, k, nqg)
#define QV3D(k) q(i, j, k, nqv)
#define NI3D(k) q(i, j, k, nnci)
#define NS3D(k) q(i, j, k, nncs)
#define NR3D(k) q(i, j, k, nncr)
#define NG3D(k) q(i, j, k, nncg)
#define NC3D(k) wk(i, j, k, W_NC)
#define T3D(k) t(i, j, k)
#define PRES(k) wk(i, j, k, W_P)
#define RHO(k) rho(i, j, k)
#define DZQ(k) dzq(i, j, k)
#define QC3DTEN(k) wk(i, j, k, W_QCT)
#define QI3DTEN(k) wk(i, j, k, W_QIT)
#define QNI3DTEN(k) wk(i, j, k, W_QNIT)
#define QR3DTEN(k) wk(i, j, k, W_QRT)
#define NC3DTEN(k) wk(i, j, k, W_NCT)
#define NI3DTEN(k) wk(i, j, k, W_NIT)
#define NS3DTEN(k) wk(i, j, k, W_NST)
#define NR3DTEN(k) wk(i, j, k, W_NRT)
#define T3DTEN(k) wk(i, j, k, W_TT)
#define QV3DTEN(k) wk(i, j, k, W_QVT)
#define QG3DTEN(k) wk(i, j, k, W_QGT)
#define NG3DTEN(k) wk(i, j, k, W_NGT)
#define QRSTEN(k) wk(i, j, k, W_QRS)
#define QISTEN(k) wk(i, j, k, W_QIS)
#define QNISTEN(k) wk(i, j, k, W_QNIS)
#define QCSTEN(k) wk(i, j, k, W_QCS)
#define QGSTEN(k) wk(i, j, k, W_QGS)
#define ECND(k) wk(i, j, k, W_ECND)
#define EDEP(k) wk(i, j, k, W_EDEP)
#define EFRZ(k) wk(i, j, k, W_EFRZ)
#define AIN(k) wk(i, j, k, W_AIN)
#define ARN(k) wk(i, j, k, W_ARN)
#define ASN(k) wk(i, j, k, W_ASN)
#define ACN(k) wk(i, j, k, W_ACN)
#define AGN(k) wk(i, j, k, W_AGN)
#define LAMI(k) wk(i, j, k, W_LAMI)
#define FR(k) wk(i, j, k, W_FR)
#define FI(k) wk(i, j, k, W_FI)
#define FNI(k) wk(i, j, k, W_FNI)
#define FS(k) wk(i, j, k, W_FS)
#define FNS(k) wk(i, j, k, W_FNS)
#define FNR(k) wk(i, j, k, W_FNR)
#define FC(k) wk(i, j, k, W_FC)
#define FNC(k) wk(i, j, k, W_FNC)
#define FG(k) wk(i, j, k, W_FG)
#define FNG(k) wk(i, j, k, W_FNG)
#define DUMI(k) wk(i, j, k, W_DUMI)
#define DUMQS(k) wk(i, j, k, W_DUMQS)
#define DUMR(k) wk(i, j, k, W_DUMR)
#define DUMFNI(k) wk(i, j, k, W_DUMFNI)
#define DUMFNS(k) wk(i, j, k, W_DUMFNS)
#define DUMFNR(k) wk(i, j, k, W_DUMFNR)
#define DUMC(k) wk(i, j, k, W_DUMC)
#define DUMFNC(k) wk(i, j, k, W_DUMFNC)
#define DUMG(k) wk(i, j, k, W_DUMG)
#define DUMFNG(k) wk(i, j, k, W_DUMFNG)

// Droplet activation (INUM=0, IACT=2): Abdul-Razzak and Ghan (2000),
// number activated at the maximum supersaturation for updraft dum.
// Returns the activated number (kg-1) before the "MIN((NANEW1+NANEW2)/RHO, .)".
KOKKOS_INLINE_FUNCTION Real arg_activated(const Params& p, Real t3d, Real xxlv, Real cpm, Real evs, Real pres,
                                          Real dv, Real kap, Real dum, Real rho) {
  const Real sigvl = 0.0761 - 1.55e-4 * (t3d - p.tmelt);
  const Real aact = 2.0 * p.mw / (p.rhow * p.rr) * sigvl / t3d;
  const Real alpha = p.g * p.mw * xxlv / (cpm * p.rr * sq(t3d)) - p.g * p.ma / (p.rr * t3d);
  const Real gamm = p.rr * t3d / (evs * p.mw) + p.mw * sq(xxlv) / (cpm * pres * p.ma * t3d);
  const Real gg = 1.0 / (p.rhow * p.rr * t3d / (evs * dv * p.mw) +
                         xxlv * p.rhow / (kap * t3d) * (xxlv * p.mw / (t3d * p.rr) - 1.0));
  const Real psi = 2.0 / 3.0 * pw(alpha * dum / gg, 0.5) * aact;
  const Real eta1 = pw(alpha * dum / gg, 1.5) / (2.0 * PI * p.rhow * gamm * p.nanew1);
  const Real eta2 = pw(alpha * dum / gg, 1.5) / (2.0 * PI * p.rhow * gamm * p.nanew2);
  const Real sm1 = 2.0 / pw(p.bact, 0.5) * pw(aact / (3.0 * p.rm1), 1.5);
  const Real sm2 = 2.0 / pw(p.bact, 0.5) * pw(aact / (3.0 * p.rm2), 1.5);
  Real dum1 = 1.0 / sq(sm1) * (p.f11 * pw(psi / eta1, 1.5) + p.f21 * pw(sq(sm1) / (eta1 + 3.0 * psi), 0.75));
  Real dum2 = 1.0 / sq(sm2) * (p.f12 * pw(psi / eta2, 1.5) + p.f22 * pw(sq(sm2) / (eta2 + 3.0 * psi), 0.75));
  const Real smax = 1.0 / pw(dum1 + dum2, 0.5);
  const Real uu1 = 2.0 * Kokkos::log(sm1 / smax) / (4.242 * Kokkos::log(p.sig1));
  const Real uu2 = 2.0 * Kokkos::log(sm2 / smax) / (4.242 * Kokkos::log(p.sig2));
  dum1 = p.nanew1 / 2.0 * (1.0 - impl::derf1(uu1));
  dum2 = p.nanew2 / 2.0 * (1.0 - impl::derf1(uu2));
  return (dum1 + dum2) / rho;
}

// Number activated at the equilibrium supersaturation smax (IDROP=0 branch).
KOKKOS_INLINE_FUNCTION Real arg_activated_ss(const Params& p, Real t3d, Real smax, Real rho) {
  const Real sigvl = 0.0761 - 1.55e-4 * (t3d - p.tmelt);
  const Real aact = 2.0 * p.mw / (p.rhow * p.rr) * sigvl / t3d;
  const Real sm1 = 2.0 / pw(p.bact, 0.5) * pw(aact / (3.0 * p.rm1), 1.5);
  const Real sm2 = 2.0 / pw(p.bact, 0.5) * pw(aact / (3.0 * p.rm2), 1.5);
  const Real uu1 = 2.0 * Kokkos::log(sm1 / smax) / (4.242 * Kokkos::log(p.sig1));
  const Real uu2 = 2.0 * Kokkos::log(sm2 / smax) / (4.242 * Kokkos::log(p.sig2));
  const Real dum1 = p.nanew1 / 2.0 * (1.0 - impl::derf1(uu1));
  const Real dum2 = p.nanew2 / 2.0 * (1.0 - impl::derf1(uu2));
  return (dum1 + dum2) / rho;
}

//---------------------------------------------------------------------------
// M2005MICRO_GRAUPEL on column (i,j). precrt: precipitation (mm... as the
// Fortran: kg/m2 per time step). eff1: unused. Returns satadj2's iflag.
KOKKOS_INLINE_FUNCTION int Column::micro(int i, int j, Real& precrt, Real* /*eff1*/) const {
  const Params& P = p;
  const Real* CONS = P.cons;
  const Real DT = dt, RDT = rdt;
  const int KTS = 1, KTE = nk;
  int iflag = 0;
  int LTRUE = 0;

  if (dorad) {
    for (int k = KTS; k <= KTE; ++k) {
      effc(i, j, k) = 25.0;
      effi(i, j, k) = 25.0;
      effs(i, j, k) = 25.0;
      effr(i, j, k) = 25.0;
      effg(i, j, k) = 25.0;
      effis(i, j, k) = 25.0;
    }
  }

  // scalars that persist across k in the Fortran
  Real EPSR = 0.0, EPSI = 0.0, EPSS = 0.0, EPSG = 0.0;
  Real FMULT = 0.0;

  //-------------------------------------------------------------- kloop1
  for (int k = KTS; k <= KTE; ++k) {
    const Real XXLV = P.lv1 - P.lv2 * T3D(k);
    const Real XXLS = P.ls1 - P.ls2 * T3D(k);
    const Real CPM = P.cp * (1.0 + 0.887 * QV3D(k));
    if (eqtset == 2) {
      const Real cvm = 1.0 / (P.cv + P.cvv * QV3D(k) + P.cpl * (QC3D(k) + QR3D(k)) +
                              P.cpi * (QI3D(k) + QNI3D(k) + QG3D(k)));
      ECND(k) = (XXLV - P.rv * T3D(k)) * cvm;
      EDEP(k) = (XXLS - P.rv * T3D(k)) * cvm;
      EFRZ(k) = (XXLS - XXLV) * cvm;
    } else {
      ECND(k) = XXLV / CPM;
      EDEP(k) = XXLS / CPM;
      EFRZ(k) = (XXLS - XXLV) / CPM;
    }

    Real EVS = fmin2(PRES(k) * 0.5, 611.2 * Kokkos::exp(17.67 * (T3D(k) - 273.15) / (T3D(k) - 29.65)));
    Real EIS = fmin2(PRES(k) * 0.5, 611.2 * Kokkos::exp(21.8745584 * (T3D(k) - 273.15) / (T3D(k) - 7.66)));
    if (EIS > EVS) EIS = EVS;
    const Real QVS = P.ep_2 * EVS / (PRES(k) - EVS);
    const Real QVI = P.ep_2 * EIS / (PRES(k) - EIS);
    const Real QVQVS = QV3D(k) / QVS;
    const Real QVQVSI = QV3D(k) / QVI;
    const Real rho_k = RHO(k);

    if (QVQVS < 0.9) {
      if (QR3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QR3D(k);
        T3D(k) = T3D(k) - QR3D(k) * ECND(k);
        QR3D(k) = 0.0;
      }
      if (QC3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QC3D(k);
        T3D(k) = T3D(k) - QC3D(k) * ECND(k);
        QC3D(k) = 0.0;
      }
    }
    if (QVQVSI < 0.9) {
      if (QI3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QI3D(k);
        T3D(k) = T3D(k) - QI3D(k) * EDEP(k);
        QI3D(k) = 0.0;
      }
      if (QNI3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QNI3D(k);
        T3D(k) = T3D(k) - QNI3D(k) * EDEP(k);
        QNI3D(k) = 0.0;
      }
      if (QG3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QG3D(k);
        T3D(k) = T3D(k) - QG3D(k) * EDEP(k);
        QG3D(k) = 0.0;
      }
    }

    const Real XLF = XXLS - XXLV;

    if (QC3D(k) < P.qsmall) { QC3D(k) = 0.0; NC3D(k) = 0.0; }
    if (QR3D(k) < P.qsmall) { QR3D(k) = 0.0; NR3D(k) = 0.0; }
    if (QI3D(k) < P.qsmall) { QI3D(k) = 0.0; NI3D(k) = 0.0; }
    if (QNI3D(k) < P.qsmall) { QNI3D(k) = 0.0; NS3D(k) = 0.0; }
    if (QG3D(k) < P.qsmall) { QG3D(k) = 0.0; NG3D(k) = 0.0; }

    QRSTEN(k) = 0.0;
    QISTEN(k) = 0.0;
    QNISTEN(k) = 0.0;
    QCSTEN(k) = 0.0;
    QGSTEN(k) = 0.0;

    const Real MU = 1.496e-6 * pw(T3D(k), 1.5) / (T3D(k) + 120.0);
    Real DUM = pw(P.rhosu / rho_k, 0.54);
    AIN(k) = pw(P.rhosu / rho_k, 0.35) * P.ai;
    ARN(k) = DUM * P.ar;
    ASN(k) = DUM * P.as;
    ACN(k) = P.g * P.rhow / (18.0 * MU);
    AGN(k) = DUM * P.ag;
    LAMI(k) = 0.0;

    if (QC3D(k) < P.qsmall && QI3D(k) < P.qsmall && QNI3D(k) < P.qsmall && QR3D(k) < P.qsmall &&
        QG3D(k) < P.qsmall) {
      if (T3D(k) < P.tmelt && QVQVSI < 0.999) continue;   // GOTO 200
      if (T3D(k) >= P.tmelt && QVQVS < 0.999) continue;   // GOTO 200
    }

    const Real KAP = 1.414e3 * MU;
    const Real DV = 8.794e-5 * pw(T3D(k), 1.81) / PRES(k);
    const Real SC = MU / (rho_k * DV);
    DUM = (P.rv * sq(T3D(k)));
    const Real DQSDT = XXLV * QVS / DUM;
    const Real DQSIDT = XXLS * QVI / DUM;
    const Real ABI = 1.0 + DQSIDT * XXLS / CPM;
    const Real AB = 1.0 + DQSDT * XXLV / CPM;

    // per-level values (arrays in the Fortran, only used at this k)
    Real LAMC = 0.0, LAMR = 0.0, LAMS = 0.0, LAMG = 0.0, N0RR = 0.0, N0S = 0.0, N0G = 0.0, N0I = 0.0;
    Real PGAM = 0.0, CDIST1 = 0.0;
    Real DUM1, DUM2, DUM3, DUMACT, RATIO;
    Real UMS, UMR, UNS, UNR, UMG, UNG;

    if (T3D(k) >= P.tmelt) {
      //=========================================================== T >= 273.15
      if (P.inum == 1) NC3D(k) = P.ndcnst * 1.0e6 / rho_k;

      if (QNI3D(k) < 1.0e-6) {
        QR3D(k) = QR3D(k) + QNI3D(k);
        NR3D(k) = NR3D(k) + NS3D(k);
        T3D(k) = T3D(k) - QNI3D(k) * EFRZ(k);
        QNI3D(k) = 0.0;
        NS3D(k) = 0.0;
      }
      if (QG3D(k) < 1.0e-6) {
        QR3D(k) = QR3D(k) + QG3D(k);
        NR3D(k) = NR3D(k) + NG3D(k);
        T3D(k) = T3D(k) - QG3D(k) * EFRZ(k);
        QG3D(k) = 0.0;
        NG3D(k) = 0.0;
      }

      if (!(QC3D(k) < P.qsmall && QNI3D(k) < 1.0e-8 && QR3D(k) < P.qsmall && QG3D(k) < 1.0e-8)) {
        NS3D(k) = fmax2(0.0, NS3D(k));
        if (P.inum == 0) NC3D(k) = fmax2(0.0, NC3D(k));
        NR3D(k) = fmax2(0.0, NR3D(k));
        NG3D(k) = fmax2(0.0, NG3D(k));

        // RAIN
        if (QR3D(k) >= P.qsmall) {
          LAMR = pw(PI * P.rhow * NR3D(k) / QR3D(k), 1.0 / 3.0);
          N0RR = NR3D(k) * LAMR;
          if (LAMR < P.lamminr) {
            LAMR = P.lamminr;
            N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
            NR3D(k) = N0RR / LAMR;
          } else if (LAMR > P.lammaxr) {
            LAMR = P.lammaxr;
            N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
            NR3D(k) = N0RR / LAMR;
          }
        }
        // CLOUD DROPLETS
        if (QC3D(k) >= P.qsmall) {
          PGAM = 0.0005714 * (NC3D(k) / 1.0e6 * rho_k) + 0.2714;
          PGAM = 1.0 / sq(PGAM) - 1.0;
          PGAM = fmax2(PGAM, 2.0);
          PGAM = fmin2(PGAM, 10.0);
          LAMC = pw(CONS[26] * NC3D(k) * impl::gamma(PGAM + 4.0) / (QC3D(k) * impl::gamma(PGAM + 1.0)),
                    1.0 / 3.0);
          const Real LAMMIN = (PGAM + 1.0) / 60.0e-6;
          const Real LAMMAX = (PGAM + 1.0) / 1.0e-6;
          if (LAMC < LAMMIN) {
            LAMC = LAMMIN;
            NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                  Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                      CONS[26];
          } else if (LAMC > LAMMAX) {
            LAMC = LAMMAX;
            NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                  Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                      CONS[26];
          }
        }
        // SNOW
        if (QNI3D(k) >= P.qsmall) {
          LAMS = pw(CONS[1] * NS3D(k) / QNI3D(k), 1.0 / P.ds);
          N0S = NS3D(k) * LAMS;
          if (LAMS < P.lammins) {
            LAMS = P.lammins;
            N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
            NS3D(k) = N0S / LAMS;
          } else if (LAMS > P.lammaxs) {
            LAMS = P.lammaxs;
            N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
            NS3D(k) = N0S / LAMS;
          }
        }
        // GRAUPEL
        if (QG3D(k) >= P.qsmall) {
          LAMG = pw(CONS[2] * NG3D(k) / QG3D(k), 1.0 / P.dg);
          N0G = NG3D(k) * LAMG;
          if (LAMG < P.lamming) {
            LAMG = P.lamming;
            N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
            NG3D(k) = N0G / LAMG;
          } else if (LAMG > P.lammaxg) {
            LAMG = P.lammaxg;
            N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
            NG3D(k) = N0G / LAMG;
          }
        }

        // ZERO OUT PROCESS RATES
        Real PRC = 0.0, NPRC = 0.0, NPRC1 = 0.0, PRA = 0.0, NPRA = 0.0, NRAGG = 0.0, NSMLTS = 0.0,
             NSMLTR = 0.0, EVPMS = 0.0, PRE = 0.0, NSUBR = 0.0, PRACG = 0.0, NPRACG = 0.0, PSMLT = 0.0,
             PGMLT = 0.0, EVPMG = 0.0, PRACS = 0.0, NGMLTG = 0.0, NGMLTR = 0.0;

        // AUTOCONVERSION (Khairoutdinov and Kogan 2000)
        if (QC3D(k) >= 1.0e-6) {
          PRC = 1350.0 * pw(QC3D(k), 2.47) * pw(NC3D(k) / 1.0e6 * rho_k, -1.79);
          NPRC1 = PRC / CONS[29];
          NPRC = PRC / (QC3D(k) / NC3D(k));
          NPRC = fmin2(NPRC, NC3D(k) / DT);
          NPRC1 = fmin2(NPRC1, NPRC);
        }

        // COLLECTION OF SNOW BY RAIN ABOVE FREEZING
        if (QR3D(k) >= 1.0e-8 && QNI3D(k) >= 1.0e-8) {
          UMS = ASN(k) * CONS[3] / pw(LAMS, P.bs);
          UMR = ARN(k) * CONS[4] / pw(LAMR, P.br);
          UNS = ASN(k) * CONS[5] / pw(LAMS, P.bs);
          UNR = ARN(k) * CONS[6] / pw(LAMR, P.br);
          DUM = pw(P.rhosu / rho_k, 0.54);
          UMS = fmin2(UMS, 1.2 * DUM);
          UNS = fmin2(UNS, 1.2 * DUM);
          UMR = fmin2(UMR, 9.1 * DUM);
          UNR = fmin2(UNR, 9.1 * DUM);
          PRACS = CONS[41] * (pw(sq(1.2 * UMR - 0.95 * UMS) + 0.08 * UMS * UMR, 0.5) * rho_k * N0RR * N0S /
                              cube(LAMR) *
                              (5.0 / (cube(LAMR) * LAMS) + 2.0 / (sq(LAMR) * sq(LAMS)) + 0.5 / (LAMR * cube(LAMS))));
        }

        // COLLECTION OF GRAUPEL BY RAIN ABOVE FREEZING
        if (QR3D(k) >= 1.0e-8 && QG3D(k) >= 1.0e-8) {
          UMG = AGN(k) * CONS[7] / pw(LAMG, P.bg);
          UMR = ARN(k) * CONS[4] / pw(LAMR, P.br);
          UNG = AGN(k) * CONS[8] / pw(LAMG, P.bg);
          UNR = ARN(k) * CONS[6] / pw(LAMR, P.br);
          DUM = pw(P.rhosu / rho_k, 0.54);
          UMG = fmin2(UMG, 20.0 * DUM);
          UNG = fmin2(UNG, 20.0 * DUM);
          UMR = fmin2(UMR, 9.1 * DUM);
          UNR = fmin2(UNR, 9.1 * DUM);
          PRACG = CONS[41] * (pw(sq(1.2 * UMR - 0.95 * UMG) + 0.08 * UMG * UMR, 0.5) * rho_k * N0RR * N0G /
                              cube(LAMR) *
                              (5.0 / (cube(LAMR) * LAMG) + 2.0 / (sq(LAMR) * sq(LAMG)) + 0.5 / (LAMR * cube(LAMG))));
          DUM = PRACG / 5.2e-7;
          NPRACG = CONS[32] * rho_k * pw(1.7 * sq(UNR - UNG) + 0.3 * UNR * UNG, 0.5) * N0RR * N0G *
                   (1.0 / (cube(LAMR) * LAMG) + 1.0 / (sq(LAMR) * sq(LAMG)) + 1.0 / (LAMR * cube(LAMG)));
          NPRACG = NPRACG - DUM;
        }

        // ACCRETION OF CLOUD LIQUID WATER BY RAIN
        if (QR3D(k) >= 1.0e-8 && QC3D(k) >= 1.0e-8) {
          DUM = (QC3D(k) * QR3D(k));
          PRA = 67.0 * pw(DUM, 1.15);
          NPRA = PRA / (QC3D(k) / NC3D(k));
        }

        // SELF-COLLECTION OF RAIN DROPS (with breakup)
        if (QR3D(k) >= 1.0e-8) {
          DUM1 = P.d_b;
          if (1.0 / LAMR < DUM1) {
            DUM = 1.0;
          } else if (1.0 / LAMR >= DUM1) {
            DUM = 2.0 - Kokkos::exp(2300.0 * (1.0 / LAMR - DUM1));
          }
          NRAGG = -5.78 * DUM * QR3D(k) * NR3D(k) * rho_k;
        }

        // EVAP OF RAIN
        if (QR3D(k) >= P.qsmall) {
          EPSR = 2.0 * PI * N0RR * rho_k * DV *
                 (P.f1r / (LAMR * LAMR) + P.f2r * pw(ARN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[9] /
                                              (pw(LAMR, CONS[34])));
        } else {
          EPSR = 0.0;
        }
        if (QV3D(k) < QVS) {
          PRE = EPSR * (QV3D(k) - QVS) / AB;
          PRE = fmin2(PRE, 0.0);
          DUM = QV3D(k) - PRE * DT;
          if (DUM > QVS) PRE = (QV3D(k) - QVS) / DT;
        } else {
          PRE = 0.0;
        }

        // MELTING OF SNOW
        if (QNI3D(k) >= 1.0e-8) {
          DUM = -P.cpw / XLF * (T3D(k) - 273.15) * PRACS;
          PSMLT = 2.0 * PI * N0S * KAP * (P.tmelt - T3D(k)) / XLF *
                      (P.f1s / (LAMS * LAMS) + P.f2s * pw(ASN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) *
                                                   CONS[10] / (pw(LAMS, CONS[35]))) +
                  DUM;
          if (QVQVS < 1.0) {
            EPSS = 2.0 * PI * N0S * rho_k * DV *
                   (P.f1s / (LAMS * LAMS) + P.f2s * pw(ASN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[10] /
                                                (pw(LAMS, CONS[35])));
            EVPMS = (QV3D(k) - QVS) * EPSS / AB;
            EVPMS = fmax2(EVPMS, PSMLT);
            PSMLT = PSMLT - EVPMS;
          }
        }

        // MELTING OF GRAUPEL
        if (QG3D(k) >= 1.0e-8) {
          DUM = -P.cpw / XLF * (T3D(k) - 273.15) * PRACG;
          PGMLT = 2.0 * PI * N0G * KAP * (P.tmelt - T3D(k)) / XLF *
                      (P.f1s / (LAMG * LAMG) + P.f2s * pw(AGN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) *
                                                   CONS[11] / (pw(LAMG, CONS[36]))) +
                  DUM;
          if (QVQVS < 1.0) {
            EPSG = 2.0 * PI * N0G * rho_k * DV *
                   (P.f1s / (LAMG * LAMG) + P.f2s * pw(AGN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[11] /
                                                (pw(LAMG, CONS[36])));
            EVPMG = (QV3D(k) - QVS) * EPSG / AB;
            EVPMG = fmax2(EVPMG, PGMLT);
            PGMLT = PGMLT - EVPMG;
          }
        }

        PRACG = 0.0;
        PRACS = 0.0;

        // CONSERVATION OF QC
        DUM = (PRC + PRA) * DT;
        if (DUM > QC3D(k) && QC3D(k) >= P.qsmall) {
          RATIO = QC3D(k) / DUM;
          PRC = PRC * RATIO;
          PRA = PRA * RATIO;
        }
        // CONSERVATION OF SNOW
        DUM = (-PSMLT - EVPMS + PRACS) * DT;
        if (DUM > QNI3D(k) && QNI3D(k) >= P.qsmall) {
          RATIO = QNI3D(k) / DUM;
          PSMLT = PSMLT * RATIO;
          EVPMS = EVPMS * RATIO;
          PRACS = PRACS * RATIO;
        }
        // CONSERVATION OF GRAUPEL
        DUM = (-PGMLT - EVPMG + PRACG) * DT;
        if (DUM > QG3D(k) && QG3D(k) >= P.qsmall) {
          RATIO = QG3D(k) / DUM;
          PGMLT = PGMLT * RATIO;
          EVPMG = EVPMG * RATIO;
          PRACG = PRACG * RATIO;
        }
        // CONSERVATION OF QR
        DUM = (-PRACS - PRACG - PRE - PRA - PRC + PSMLT + PGMLT) * DT;
        if (DUM > QR3D(k) && QR3D(k) >= P.qsmall) {
          RATIO = (QR3D(k) / DT + PRACS + PRACG + PRA + PRC - PSMLT - PGMLT) / (-PRE);
          PRE = PRE * RATIO;
        }

        QV3DTEN(k) = QV3DTEN(k) + (-PRE - EVPMS - EVPMG);
        T3DTEN(k) = T3DTEN(k) + (PRE * ECND(k) + (EVPMS + EVPMG) * EDEP(k) + (PSMLT + PGMLT - PRACS - PRACG) * EFRZ(k));
        QC3DTEN(k) = QC3DTEN(k) + (-PRA - PRC);
        QR3DTEN(k) = QR3DTEN(k) + (PRE + PRA + PRC - PSMLT - PGMLT + PRACS + PRACG);
        QNI3DTEN(k) = QNI3DTEN(k) + (PSMLT + EVPMS - PRACS);
        QG3DTEN(k) = QG3DTEN(k) + (PGMLT + EVPMG - PRACG);
        NC3DTEN(k) = NC3DTEN(k) + (-NPRA - NPRC);
        NR3DTEN(k) = NR3DTEN(k) + (NPRC1 + NRAGG - NPRACG);

        if (PRE < 0.0) {
          DUM = PRE * DT / QR3D(k);
          DUM = fmax2(-1.0, DUM);
          NSUBR = DUM * NR3D(k) / DT;
        }
        if (EVPMS + PSMLT < 0.0) {
          DUM = (EVPMS + PSMLT) * DT / QNI3D(k);
          DUM = fmax2(-1.0, DUM);
          NSMLTS = DUM * NS3D(k) / DT;
        }
        if (PSMLT < 0.0) {
          DUM = PSMLT * DT / QNI3D(k);
          DUM = fmax2(-1.0, DUM);
          NSMLTR = DUM * NS3D(k) / DT;
        }
        if (EVPMG + PGMLT < 0.0) {
          DUM = (EVPMG + PGMLT) * DT / QG3D(k);
          DUM = fmax2(-1.0, DUM);
          NGMLTG = DUM * NG3D(k) / DT;
        }
        if (PGMLT < 0.0) {
          DUM = PGMLT * DT / QG3D(k);
          DUM = fmax2(-1.0, DUM);
          NGMLTR = DUM * NG3D(k) / DT;
        }
        NS3DTEN(k) = NS3DTEN(k) + (NSMLTS);
        NG3DTEN(k) = NG3DTEN(k) + (NGMLTG);
        NR3DTEN(k) = NR3DTEN(k) + (NSUBR - NSMLTR - NGMLTR);
      }
      // 300 CONTINUE

      // SATURATION ADJUSTMENT
      {
        Real DUMT = T3D(k) + DT * T3DTEN(k);
        Real DUMQV = QV3D(k) + DT * QV3DTEN(k);
        Real DUMQC = QC3D(k) + DT * QC3DTEN(k);
        const Real DUMQR = QR3D(k) + DT * QR3DTEN(k);
        const Real DUMMQS = QNI3D(k) + DT * QNI3DTEN(k);
        const Real DUMQG = QG3D(k) + DT * QG3DTEN(k);
        const Real DUMQI = 0.0;
        if (eqtset == 2) PRES(k) = rho_k * DUMT * (P.r + P.rv * DUMQV);
        const Real SAVEQC = DUMQC;
        const Real SAVET = DUMT;
        Real prs_k = PRES(k);
        iflag |= impl::satadj2(P, DUMT, prs_k, DUMQV, DUMQC, DUMQR, DUMQI, DUMMQS, DUMQG, rho_k, eqtset, tsmall);
        PRES(k) = prs_k;
        if (Kokkos::fabs(DUMT - SAVET) >= 0.0001) {
          const Real PCC = (DUMQC - SAVEQC) * RDT;
          QV3DTEN(k) = QV3DTEN(k) - PCC;
          QC3DTEN(k) = QC3DTEN(k) + PCC;
          T3DTEN(k) = T3DTEN(k) + (DUMT - SAVET) * RDT;
        }
      }

      // ACTIVATION OF CLOUD DROPLETS (INUM=0; ISUB=1, IBASE=1, IACT=2)
      if (P.inum == 0) {
        if (QC3D(k) + QC3DTEN(k) * DT >= P.qsmall) {
          DUM = 0.5 * (w(i, j, k) + w(i, j, k + 1));
          if (DUM >= 0.001) {
            int IDROP = 0;
            if (QC3D(k) <= 0.05e-3 / rho_k) IDROP = 1;
            if (k == 1) {
              IDROP = 1;
            } else if (k >= 2) {
              if (QC3D(k) > 0.05e-3 / rho_k && QC3D(k - 1) <= 0.05e-3 / RHO(k - 1)) IDROP = 1;
            }
            if (IDROP == 1) {
              DUM2 = arg_activated(P, T3D(k), XXLV, CPM, EVS, PRES(k), DV, KAP, DUM, rho_k);
              DUM2 = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
              DUM2 = (DUM2 - NC3D(k)) / DT;
              DUM2 = fmax2(0.0, DUM2);
              NC3DTEN(k) = NC3DTEN(k) + DUM2;
            } else {
              const Real TAUC = 1.0 / (2.0 * PI * rho_k * DV * NC3D(k) * (PGAM + 1.0) / LAMC);
              const Real TAUR = (EPSR > 1.0e-8) ? 1.0 / EPSR : 1.0e8;
              DUM3 = (-QVS * rho_k / (PRES(k) - EVS) + DQSDT / P.cp) * P.g * DUM;
              DUM3 = DUM3 * TAUC * TAUR / (TAUC + TAUR);
              if (DUM3 / QVS >= 1.0e-6) {
                DUM2 = arg_activated(P, T3D(k), XXLV, CPM, EVS, PRES(k), DV, KAP, DUM, rho_k);
                DUMACT = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
                DUM2 = arg_activated_ss(P, T3D(k), DUM3 / QVS, rho_k);
                DUM2 = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
                DUM2 = fmin2(DUM2, DUMACT);
                DUM2 = (DUM2 - NC3D(k)) / DT;
                DUM2 = fmax2(0.0, DUM2);
                NC3DTEN(k) = NC3DTEN(k) + DUM2;
              }
            }
          }
        }
      }
    } else {
      //=========================================================== T < 273.15
      if (P.inum == 1) NC3D(k) = P.ndcnst * 1.0e6 / rho_k;

      NI3D(k) = fmax2(0.0, NI3D(k));
      NS3D(k) = fmax2(0.0, NS3D(k));
      if (P.inum == 0) NC3D(k) = fmax2(0.0, NC3D(k));
      NR3D(k) = fmax2(0.0, NR3D(k));
      NG3D(k) = fmax2(0.0, NG3D(k));

      // CLOUD ICE
      if (QI3D(k) >= P.qsmall) {
        LAMI(k) = pw(CONS[12] * NI3D(k) / QI3D(k), 1.0 / P.di);
        N0I = NI3D(k) * LAMI(k);
        if (LAMI(k) < P.lammini) {
          LAMI(k) = P.lammini;
          N0I = pw(LAMI(k), P.di + 1.0) * QI3D(k) / CONS[12];
          NI3D(k) = N0I / LAMI(k);
        } else if (LAMI(k) > P.lammaxi) {
          LAMI(k) = P.lammaxi;
          N0I = pw(LAMI(k), P.di + 1.0) * QI3D(k) / CONS[12];
          NI3D(k) = N0I / LAMI(k);
        }
      }
      // RAIN
      if (QR3D(k) >= P.qsmall) {
        LAMR = pw(PI * P.rhow * NR3D(k) / QR3D(k), 1.0 / 3.0);
        N0RR = NR3D(k) * LAMR;
        if (LAMR < P.lamminr) {
          LAMR = P.lamminr;
          N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
          NR3D(k) = N0RR / LAMR;
        } else if (LAMR > P.lammaxr) {
          LAMR = P.lammaxr;
          N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
          NR3D(k) = N0RR / LAMR;
        }
      }
      // CLOUD DROPLETS
      if (QC3D(k) >= P.qsmall) {
        PGAM = 0.0005714 * (NC3D(k) / 1.0e6 * rho_k) + 0.2714;
        PGAM = 1.0 / sq(PGAM) - 1.0;
        PGAM = fmax2(PGAM, 2.0);
        PGAM = fmin2(PGAM, 10.0);
        LAMC = pw(CONS[26] * NC3D(k) * impl::gamma(PGAM + 4.0) / (QC3D(k) * impl::gamma(PGAM + 1.0)), 1.0 / 3.0);
        const Real LAMMIN = (PGAM + 1.0) / 60.0e-6;
        const Real LAMMAX = (PGAM + 1.0) / 1.0e-6;
        if (LAMC < LAMMIN) {
          LAMC = LAMMIN;
          NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                    CONS[26];
        } else if (LAMC > LAMMAX) {
          LAMC = LAMMAX;
          NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                    CONS[26];
        }
        CDIST1 = NC3D(k) / impl::gamma(PGAM + 1.0);
      }
      // SNOW
      if (QNI3D(k) >= P.qsmall) {
        LAMS = pw(CONS[1] * NS3D(k) / QNI3D(k), 1.0 / P.ds);
        N0S = NS3D(k) * LAMS;
        if (LAMS < P.lammins) {
          LAMS = P.lammins;
          N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
          NS3D(k) = N0S / LAMS;
        } else if (LAMS > P.lammaxs) {
          LAMS = P.lammaxs;
          N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
          NS3D(k) = N0S / LAMS;
        }
      }
      // GRAUPEL
      if (QG3D(k) >= P.qsmall) {
        LAMG = pw(CONS[2] * NG3D(k) / QG3D(k), 1.0 / P.dg);
        N0G = NG3D(k) * LAMG;
        if (LAMG < P.lamming) {
          LAMG = P.lamming;
          N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
          NG3D(k) = N0G / LAMG;
        } else if (LAMG > P.lammaxg) {
          LAMG = P.lammaxg;
          N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
          NG3D(k) = N0G / LAMG;
        }
      }

      // ZERO OUT PROCESS RATES
      Real MNUCCC = 0.0, NNUCCC = 0.0, PRC = 0.0, NPRC = 0.0, NPRC1 = 0.0, NSAGG = 0.0, PSACWS = 0.0,
           NPSACWS = 0.0, PSACWI = 0.0, NPSACWI = 0.0, PRACS = 0.0, NPRACS = 0.0, NMULTS = 0.0, QMULTS = 0.0,
           NMULTR = 0.0, QMULTR = 0.0, NMULTG = 0.0, QMULTG = 0.0, NMULTRG = 0.0, QMULTRG = 0.0, MNUCCR = 0.0,
           NNUCCR = 0.0, PRA = 0.0, NPRA = 0.0, NRAGG = 0.0, PRCI = 0.0, NPRCI = 0.0, PRAI = 0.0, NPRAI = 0.0,
           NNUCCD = 0.0, MNUCCD = 0.0, PRE = 0.0, PRD = 0.0, PRDS = 0.0, EPRD = 0.0, EPRDS = 0.0,
           NSUBI = 0.0, NSUBS = 0.0, NSUBR = 0.0, PIACR = 0.0, NIACR = 0.0, PRACI = 0.0, PIACRS = 0.0,
           NIACRS = 0.0, PRACIS = 0.0, PRACG = 0.0, PSACR = 0.0, PSACWG = 0.0, PGSACW = 0.0, PGRACS = 0.0,
           PRDG = 0.0, EPRDG = 0.0, NPRACG = 0.0, NPSACWG = 0.0, NSCNG = 0.0, NGRACS = 0.0, NSUBG = 0.0;

      // FREEZING OF CLOUD DROPLETS (below -4 C)
      if (QC3D(k) >= P.qsmall && T3D(k) < 269.15) {
        const Real NACNT = Kokkos::exp(-2.80 + 0.262 * (P.tmelt - T3D(k))) * 1000.0;
        DUM = 7.37 * T3D(k) / (288.0 * 10.0 * PRES(k)) / 100.0;
        const Real DAP = CONS[37] * T3D(k) * (1.0 + DUM / P.rin) / MU;
        MNUCCC = CONS[38] * DAP * NACNT *
                 Kokkos::exp(Kokkos::log(CDIST1) + Kokkos::log(impl::gamma(PGAM + 5.0)) - 4.0 * Kokkos::log(LAMC));
        NNUCCC = 2.0 * PI * DAP * NACNT * CDIST1 * impl::gamma(PGAM + 2.0) / LAMC;
        MNUCCC = MNUCCC + CONS[39] *
                              Kokkos::exp(Kokkos::log(CDIST1) + Kokkos::log(impl::gamma(7.0 + PGAM)) -
                                          6.0 * Kokkos::log(LAMC)) *
                              (Kokkos::exp(P.aimm * (273.15 - T3D(k))) - 1.0);
        NNUCCC = NNUCCC + CONS[40] *
                              Kokkos::exp(Kokkos::log(CDIST1) + Kokkos::log(impl::gamma(PGAM + 4.0)) -
                                          3.0 * Kokkos::log(LAMC)) *
                              (Kokkos::exp(P.aimm * (273.15 - T3D(k))) - 1.0);
        NNUCCC = fmin2(NNUCCC, NC3D(k) / DT);
      }

      // AUTOCONVERSION OF CLOUD LIQUID WATER TO RAIN
      if (QC3D(k) >= 1.0e-6) {
        PRC = 1350.0 * pw(QC3D(k), 2.47) * pw(NC3D(k) / 1.0e6 * rho_k, -1.79);
        NPRC1 = PRC / CONS[29];
        NPRC = PRC / (QC3D(k) / NC3D(k));
        NPRC = fmin2(NPRC, NC3D(k) / DT);
        NPRC1 = fmin2(NPRC1, NPRC);
      }

      // SNOW AGGREGATION
      if (QNI3D(k) >= 1.0e-8) {
        NSAGG = CONS[15] * ASN(k) * pw(rho_k, (2.0 + P.bs) / 3.0) * pw(QNI3D(k), (2.0 + P.bs) / 3.0) *
                pw(NS3D(k) * rho_k, (4.0 - P.bs) / 3.0) / (rho_k);
      }

      // ACCRETION OF CLOUD DROPLETS ONTO SNOW
      if (QNI3D(k) >= 1.0e-8 && QC3D(k) >= P.qsmall) {
        PSACWS = CONS[13] * ASN(k) * QC3D(k) * rho_k * N0S / pw(LAMS, P.bs + 3.0);
        NPSACWS = CONS[13] * ASN(k) * NC3D(k) * rho_k * N0S / pw(LAMS, P.bs + 3.0);
      }
      // COLLECTION OF CLOUD WATER BY GRAUPEL
      if (QG3D(k) >= 1.0e-8 && QC3D(k) >= P.qsmall) {
        PSACWG = CONS[14] * AGN(k) * QC3D(k) * rho_k * N0G / pw(LAMG, P.bg + 3.0);
        NPSACWG = CONS[14] * AGN(k) * NC3D(k) * rho_k * N0G / pw(LAMG, P.bg + 3.0);
      }
      // CLOUD ICE COLLECTING DROPLETS
      if (QI3D(k) >= 1.0e-8 && QC3D(k) >= P.qsmall) {
        if (1.0 / LAMI(k) >= 100.0e-6) {
          PSACWI = CONS[16] * AIN(k) * QC3D(k) * rho_k * N0I / pw(LAMI(k), P.bi + 3.0);
          NPSACWI = CONS[16] * AIN(k) * NC3D(k) * rho_k * N0I / pw(LAMI(k), P.bi + 3.0);
        }
      }

      // ACCRETION OF RAIN WATER BY SNOW
      if (QR3D(k) >= 1.0e-8 && QNI3D(k) >= 1.0e-8) {
        UMS = ASN(k) * CONS[3] / pw(LAMS, P.bs);
        UMR = ARN(k) * CONS[4] / pw(LAMR, P.br);
        UNS = ASN(k) * CONS[5] / pw(LAMS, P.bs);
        UNR = ARN(k) * CONS[6] / pw(LAMR, P.br);
        DUM = pw(P.rhosu / rho_k, 0.54);
        UMS = fmin2(UMS, 1.2 * DUM);
        UNS = fmin2(UNS, 1.2 * DUM);
        UMR = fmin2(UMR, 9.1 * DUM);
        UNR = fmin2(UNR, 9.1 * DUM);
        PRACS = CONS[41] * (pw(sq(1.2 * UMR - 0.95 * UMS) + 0.08 * UMS * UMR, 0.5) * rho_k * N0RR * N0S /
                            cube(LAMR) *
                            (5.0 / (cube(LAMR) * LAMS) + 2.0 / (sq(LAMR) * sq(LAMS)) + 0.5 / (LAMR * cube(LAMS))));
        NPRACS = CONS[32] * rho_k * pw(1.7 * sq(UNR - UNS) + 0.3 * UNR * UNS, 0.5) * N0RR * N0S *
                 (1.0 / (cube(LAMR) * LAMS) + 1.0 / (sq(LAMR) * sq(LAMS)) + 1.0 / (LAMR * cube(LAMS)));
        PRACS = fmin2(PRACS, QR3D(k) / DT);
        if (QNI3D(k) >= 0.1e-3 && QR3D(k) >= 0.1e-3) {
          PSACR = CONS[31] * (pw(sq(1.2 * UMR - 0.95 * UMS) + 0.08 * UMS * UMR, 0.5) * rho_k * N0RR * N0S /
                              cube(LAMS) *
                              (5.0 / (cube(LAMS) * LAMR) + 2.0 / (sq(LAMS) * sq(LAMR)) + 0.5 / (LAMS * cube(LAMR))));
        }
      }

      // COLLECTION OF RAINWATER BY GRAUPEL
      if (QR3D(k) >= 1.0e-8 && QG3D(k) >= 1.0e-8) {
        UMG = AGN(k) * CONS[7] / pw(LAMG, P.bg);
        UMR = ARN(k) * CONS[4] / pw(LAMR, P.br);
        UNG = AGN(k) * CONS[8] / pw(LAMG, P.bg);
        UNR = ARN(k) * CONS[6] / pw(LAMR, P.br);
        DUM = pw(P.rhosu / rho_k, 0.54);
        UMG = fmin2(UMG, 20.0 * DUM);
        UNG = fmin2(UNG, 20.0 * DUM);
        UMR = fmin2(UMR, 9.1 * DUM);
        UNR = fmin2(UNR, 9.1 * DUM);
        PRACG = CONS[41] * (pw(sq(1.2 * UMR - 0.95 * UMG) + 0.08 * UMG * UMR, 0.5) * rho_k * N0RR * N0G /
                            cube(LAMR) *
                            (5.0 / (cube(LAMR) * LAMG) + 2.0 / (sq(LAMR) * sq(LAMG)) + 0.5 / (LAMR * cube(LAMG))));
        NPRACG = CONS[32] * rho_k * pw(1.7 * sq(UNR - UNG) + 0.3 * UNR * UNG, 0.5) * N0RR * N0G *
                 (1.0 / (cube(LAMR) * LAMG) + 1.0 / (sq(LAMR) * sq(LAMG)) + 1.0 / (LAMR * cube(LAMG)));
        PRACG = fmin2(PRACG, QR3D(k) / DT);
      }

      // RIME-SPLINTERING - SNOW
      if (QNI3D(k) >= 0.1e-3) {
        if (QC3D(k) >= 0.5e-3 || QR3D(k) >= 0.1e-3) {
          if (PSACWS > 0.0 || PRACS > 0.0) {
            if (T3D(k) < 270.16 && T3D(k) > 265.16) {
              if (T3D(k) > 270.16) {
                FMULT = 0.0;
              } else if (T3D(k) <= 270.16 && T3D(k) > 268.16) {
                FMULT = (270.16 - T3D(k)) / 2.0;
              } else if (T3D(k) >= 265.16 && T3D(k) <= 268.16) {
                FMULT = (T3D(k) - 265.16) / 3.0;
              } else if (T3D(k) < 265.16) {
                FMULT = 0.0;
              }
              if (PSACWS > 0.0) {
                NMULTS = 35.0e4 * PSACWS * FMULT * 1000.0;
                QMULTS = NMULTS * P.mmult;
                QMULTS = fmin2(QMULTS, PSACWS);
                PSACWS = PSACWS - QMULTS;
              }
              if (PRACS > 0.0) {
                NMULTR = 35.0e4 * PRACS * FMULT * 1000.0;
                QMULTR = NMULTR * P.mmult;
                QMULTR = fmin2(QMULTR, PRACS);
                PRACS = PRACS - QMULTR;
              }
            }
          }
        }
      }

      // RIME-SPLINTERING - GRAUPEL
      if (QG3D(k) >= 0.1e-3) {
        if (QC3D(k) >= 0.5e-3 || QR3D(k) >= 0.1e-3) {
          if (PSACWG > 0.0 || PRACG > 0.0) {
            if (T3D(k) < 270.16 && T3D(k) > 265.16) {
              if (T3D(k) > 270.16) {
                FMULT = 0.0;
              } else if (T3D(k) <= 270.16 && T3D(k) > 268.16) {
                FMULT = (270.16 - T3D(k)) / 2.0;
              } else if (T3D(k) >= 265.16 && T3D(k) <= 268.16) {
                FMULT = (T3D(k) - 265.16) / 3.0;
              } else if (T3D(k) < 265.16) {
                FMULT = 0.0;
              }
              if (PSACWG > 0.0) {
                NMULTG = 35.0e4 * PSACWG * FMULT * 1000.0;
                QMULTG = NMULTG * P.mmult;
                QMULTG = fmin2(QMULTG, PSACWG);
                PSACWG = PSACWG - QMULTG;
              }
              if (PRACG > 0.0) {
                NMULTRG = 35.0e4 * PRACG * FMULT * 1000.0;
                QMULTRG = NMULTRG * P.mmult;
                QMULTRG = fmin2(QMULTRG, PRACG);
                PRACG = PRACG - QMULTRG;
              }
            }
          }
        }
      }

      // CONVERSION OF RIMED CLOUD WATER ONTO SNOW TO GRAUPEL/HAIL
      if (PSACWS > 0.0) {
        if (QNI3D(k) >= 0.1e-3 && QC3D(k) >= 0.5e-3) {
          PGSACW = fmin2(PSACWS, CONS[17] * DT * N0S * QC3D(k) * QC3D(k) * ASN(k) * ASN(k) /
                                     (rho_k * pw(LAMS, 2.0 * P.bs + 2.0)));
          DUM = fmax2(P.rhosn / (P.rhog - P.rhosn) * PGSACW, 0.0);
          NSCNG = DUM / P.mg0 * rho_k;
          NSCNG = fmin2(NSCNG, NS3D(k) / DT);
          PSACWS = PSACWS - PGSACW;
        }
      }

      // CONVERSION OF RIMED RAINWATER ONTO SNOW CONVERTED TO GRAUPEL
      if (PRACS > 0.0) {
        if (QNI3D(k) >= 0.1e-3 && QR3D(k) >= 0.1e-3) {
          const Real a = cube(4.0 / LAMS);
          const Real b = cube(4.0 / LAMR);
          DUM = CONS[18] * a * a / (CONS[18] * a * a + CONS[19] * b * b);
          DUM = fmin2(DUM, 1.0);
          DUM = fmax2(DUM, 0.0);
          PGRACS = (1.0 - DUM) * PRACS;
          NGRACS = (1.0 - DUM) * NPRACS;
          NGRACS = fmin2(NGRACS, NR3D(k) / DT);
          NGRACS = fmin2(NGRACS, NS3D(k) / DT);
          PRACS = PRACS - PGRACS;
          NPRACS = NPRACS - NGRACS;
          PSACR = PSACR * (1.0 - DUM);
        }
      }

      // FREEZING OF RAIN DROPS (below -4 C)
      if (T3D(k) < 269.15 && QR3D(k) >= P.qsmall) {
        MNUCCR = CONS[20] * NR3D(k) * ((Kokkos::exp(P.aimm * (273.15 - T3D(k))) - 1.0) / cube(LAMR) / cube(LAMR));
        NNUCCR = PI * NR3D(k) * P.bimm * (Kokkos::exp(P.aimm * (273.15 - T3D(k))) - 1.0) / cube(LAMR);
        NNUCCR = fmin2(NNUCCR, NR3D(k) / DT);
      }

      // ACCRETION OF CLOUD LIQUID WATER BY RAIN
      if (QR3D(k) >= 1.0e-8 && QC3D(k) >= 1.0e-8) {
        DUM = (QC3D(k) * QR3D(k));
        PRA = 67.0 * pw(DUM, 1.15);
        NPRA = PRA / (QC3D(k) / NC3D(k));
      }

      // SELF-COLLECTION OF RAIN DROPS
      if (QR3D(k) >= 1.0e-8) {
        DUM1 = P.d_b;
        if (1.0 / LAMR < DUM1) {
          DUM = 1.0;
        } else if (1.0 / LAMR >= DUM1) {
          DUM = 2.0 - Kokkos::exp(2300.0 * (1.0 / LAMR - DUM1));
        }
        NRAGG = -5.78 * DUM * QR3D(k) * NR3D(k) * rho_k;
      }

      // AUTOCONVERSION OF CLOUD ICE TO SNOW
      if (QI3D(k) >= 1.0e-8 && QVQVSI >= 1.0) {
        NPRCI = CONS[21] * (QV3D(k) - QVI) * rho_k * N0I * Kokkos::exp(-LAMI(k) * P.dcs) * DV / ABI;
        PRCI = CONS[22] * NPRCI;
        NPRCI = fmin2(NPRCI, NI3D(k) / DT);
      }

      // ACCRETION OF CLOUD ICE BY SNOW
      if (QNI3D(k) >= 1.0e-8 && QI3D(k) >= P.qsmall) {
        PRAI = CONS[23] * ASN(k) * QI3D(k) * rho_k * N0S / pw(LAMS, P.bs + 3.0);
        NPRAI = CONS[23] * ASN(k) * NI3D(k) * rho_k * N0S / pw(LAMS, P.bs + 3.0);
        NPRAI = fmin2(NPRAI, NI3D(k) / DT);
      }

      // COLLISION OF RAIN AND ICE TO PRODUCE SNOW OR GRAUPEL
      if (QR3D(k) >= 1.0e-8 && QI3D(k) >= 1.0e-8 && T3D(k) <= P.tmelt) {
        if (QR3D(k) >= 0.1e-3) {
          NIACR = CONS[24] * NI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) * rho_k;
          PIACR = CONS[25] * NI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) / cube(LAMR) * rho_k;
          PRACI = CONS[24] * QI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) * rho_k;
          NIACR = fmin2(NIACR, NR3D(k) / DT);
          NIACR = fmin2(NIACR, NI3D(k) / DT);
        } else {
          NIACRS = CONS[24] * NI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) * rho_k;
          PIACRS = CONS[25] * NI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) / cube(LAMR) * rho_k;
          PRACIS = CONS[24] * QI3D(k) * N0RR * ARN(k) / pw(LAMR, P.br + 3.0) * rho_k;
          NIACRS = fmin2(NIACRS, NR3D(k) / DT);
          NIACRS = fmin2(NIACRS, NI3D(k) / DT);
        }
      }

      // NUCLEATION OF CLOUD ICE (INUC=0)
      if ((QVQVS >= 0.999 && T3D(k) <= 265.15) || QVQVSI >= 1.25) {
        Real kc2 = 0.005 * Kokkos::exp(0.304 * (P.tmelt - T3D(k))) * 1000.0;
        kc2 = fmin2(kc2, 500.0e3);
        kc2 = fmax2(kc2 / rho_k, 0.0);
        if (kc2 > NI3D(k) + NS3D(k) + NG3D(k)) {
          NNUCCD = (kc2 - NI3D(k) - NS3D(k) - NG3D(k)) / DT;
          MNUCCD = NNUCCD * P.mi0;
        }
      }

      // EVAP/SUB/DEP TERMS FOR QI, QNI, QR
      if (QI3D(k) >= P.qsmall) {
        EPSI = 2.0 * PI * N0I * rho_k * DV / (LAMI(k) * LAMI(k));
      } else {
        EPSI = 0.0;
      }
      if (QNI3D(k) >= P.qsmall) {
        EPSS = 2.0 * PI * N0S * rho_k * DV *
               (P.f1s / (LAMS * LAMS) + P.f2s * pw(ASN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[10] /
                                            (pw(LAMS, CONS[35])));
      } else {
        EPSS = 0.0;
      }
      if (QG3D(k) >= P.qsmall) {
        EPSG = 2.0 * PI * N0G * rho_k * DV *
               (P.f1s / (LAMG * LAMG) + P.f2s * pw(AGN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[11] /
                                            (pw(LAMG, CONS[36])));
      } else {
        EPSG = 0.0;
      }
      if (QR3D(k) >= P.qsmall) {
        EPSR = 2.0 * PI * N0RR * rho_k * DV *
               (P.f1r / (LAMR * LAMR) + P.f2r * pw(ARN(k) * rho_k / MU, 0.5) * pw(SC, 1.0 / 3.0) * CONS[9] /
                                            (pw(LAMR, CONS[34])));
      } else {
        EPSR = 0.0;
      }

      if (QI3D(k) >= P.qsmall) {
        DUM = (1.0 - Kokkos::exp(-LAMI(k) * P.dcs) * (1.0 + LAMI(k) * P.dcs));
        PRD = EPSI * (QV3D(k) - QVI) / ABI * DUM;
      } else {
        DUM = 0.0;
      }
      if (QNI3D(k) >= P.qsmall) {
        PRDS = EPSS * (QV3D(k) - QVI) / ABI + EPSI * (QV3D(k) - QVI) / ABI * (1.0 - DUM);
      } else {
        PRD = PRD + EPSI * (QV3D(k) - QVI) / ABI * (1.0 - DUM);
      }
      PRDG = EPSG * (QV3D(k) - QVI) / ABI;

      if (QV3D(k) < QVS) {
        PRE = EPSR * (QV3D(k) - QVS) / AB;
        PRE = fmin2(PRE, 0.0);
        DUM = QV3D(k) - PRE * DT;
        if (DUM > QVS) PRE = (QV3D(k) - QVS) / DT;
      } else {
        PRE = 0.0;
      }

      DUM = (QV3D(k) - QVI) / DT;
      const Real FUDGEF = 0.9999;
      const Real SUM_DEP = PRD + PRDS + MNUCCD + PRDG;
      if ((DUM > 0.0 && SUM_DEP > DUM * FUDGEF) || (DUM < 0.0 && SUM_DEP < DUM * FUDGEF)) {
        MNUCCD = FUDGEF * MNUCCD * DUM / SUM_DEP;
        PRD = FUDGEF * PRD * DUM / SUM_DEP;
        PRDS = FUDGEF * PRDS * DUM / SUM_DEP;
        PRDG = FUDGEF * PRDG * DUM / SUM_DEP;
      }
      if (PRD < 0.0) { EPRD = PRD; PRD = 0.0; }
      if (PRDS < 0.0) { EPRDS = PRDS; PRDS = 0.0; }
      if (PRDG < 0.0) { EPRDG = PRDG; PRDG = 0.0; }

      // CONSERVATION OF QC
      DUM = (PRC + PRA + MNUCCC + PSACWS + PSACWI + QMULTS + PSACWG + PGSACW + QMULTG) * DT;
      if (DUM > QC3D(k) && QC3D(k) >= P.qsmall) {
        RATIO = QC3D(k) / DUM;
        PRC = PRC * RATIO;
        PRA = PRA * RATIO;
        MNUCCC = MNUCCC * RATIO;
        PSACWS = PSACWS * RATIO;
        PSACWI = PSACWI * RATIO;
        QMULTS = QMULTS * RATIO;
        QMULTG = QMULTG * RATIO;
        PSACWG = PSACWG * RATIO;
        PGSACW = PGSACW * RATIO;
      }
      // CONSERVATION OF QI
      DUM = (-PRD - MNUCCC + PRCI + PRAI - QMULTS - QMULTG - QMULTR - QMULTRG - MNUCCD + PRACI + PRACIS - EPRD -
             PSACWI) * DT;
      if (DUM > QI3D(k) && QI3D(k) >= P.qsmall) {
        RATIO = (QI3D(k) / DT + PRD + MNUCCC + QMULTS + QMULTG + QMULTR + QMULTRG + MNUCCD + PSACWI) /
                (PRCI + PRAI + PRACI + PRACIS - EPRD);
        PRCI = PRCI * RATIO;
        PRAI = PRAI * RATIO;
        PRACI = PRACI * RATIO;
        PRACIS = PRACIS * RATIO;
        EPRD = EPRD * RATIO;
      }
      // CONSERVATION OF QR
      DUM = ((PRACS - PRE) + (QMULTR + QMULTRG - PRC) + (MNUCCR - PRA) + PIACR + PIACRS + PGRACS + PRACG) * DT;
      if (DUM > QR3D(k) && QR3D(k) >= P.qsmall) {
        RATIO = (QR3D(k) / DT + PRC + PRA) /
                (-PRE + QMULTR + QMULTRG + PRACS + MNUCCR + PIACR + PIACRS + PGRACS + PRACG);
        PRE = PRE * RATIO;
        PRACS = PRACS * RATIO;
        QMULTR = QMULTR * RATIO;
        QMULTRG = QMULTRG * RATIO;
        MNUCCR = MNUCCR * RATIO;
        PIACR = PIACR * RATIO;
        PIACRS = PIACRS * RATIO;
        PGRACS = PGRACS * RATIO;
        PRACG = PRACG * RATIO;
      }
      // CONSERVATION OF QNI (IGRAUP=0)
      DUM = (-PRDS - PSACWS - PRAI - PRCI - PRACS - EPRDS + PSACR - PIACRS - PRACIS) * DT;
      if (DUM > QNI3D(k) && QNI3D(k) >= P.qsmall) {
        RATIO = (QNI3D(k) / DT + PRDS + PSACWS + PRAI + PRCI + PRACS + PIACRS + PRACIS) / (-EPRDS + PSACR);
        EPRDS = EPRDS * RATIO;
        PSACR = PSACR * RATIO;
      }
      // CONSERVATION OF QG
      DUM = (-PSACWG - PRACG - PGSACW - PGRACS - PRDG - MNUCCR - EPRDG - PIACR - PRACI - PSACR) * DT;
      if (DUM > QG3D(k) && QG3D(k) >= P.qsmall) {
        RATIO = (QG3D(k) / DT + PSACWG + PRACG + PGSACW + PGRACS + PRDG + MNUCCR + PSACR + PIACR + PRACI) /
                (-EPRDG);
        EPRDG = EPRDG * RATIO;
      }

      // TENDENCIES
      QV3DTEN(k) = QV3DTEN(k) + (-PRE - PRD - PRDS - MNUCCD - EPRD - EPRDS - PRDG - EPRDG);
      T3DTEN(k) = T3DTEN(k) + (PRE * ECND(k) + (PRD + PRDS + MNUCCD + EPRD + EPRDS + PRDG + EPRDG) * EDEP(k) +
                               (PSACWS + PSACWI + MNUCCC + MNUCCR + QMULTS + QMULTG + QMULTR + QMULTRG + PRACS +
                                PSACWG + PRACG + PGSACW + PGRACS + PIACR + PIACRS) *
                                   EFRZ(k));
      QC3DTEN(k) = QC3DTEN(k) + (-PRA - PRC - MNUCCC + 0.0 /*PCC*/ - PSACWS - PSACWI - QMULTS - QMULTG - PSACWG -
                                 PGSACW);
      QI3DTEN(k) = QI3DTEN(k) + (PRD + EPRD + PSACWI + MNUCCC - PRCI - PRAI + QMULTS + QMULTG + QMULTR + QMULTRG +
                                 MNUCCD - PRACI - PRACIS);
      QR3DTEN(k) = QR3DTEN(k) + (PRE + PRA + PRC - PRACS - MNUCCR - QMULTR - QMULTRG - PIACR - PIACRS - PRACG -
                                 PGRACS);
      QNI3DTEN(k) = QNI3DTEN(k) + (PRAI + PSACWS + PRDS + PRACS + PRCI + EPRDS - PSACR + PIACRS + PRACIS);
      NS3DTEN(k) = NS3DTEN(k) + (NSAGG + NPRCI - NSCNG - NGRACS + NIACRS);
      QG3DTEN(k) = QG3DTEN(k) + (PRACG + PSACWG + PGSACW + PGRACS + PRDG + EPRDG + MNUCCR + PIACR + PRACI + PSACR);
      NG3DTEN(k) = NG3DTEN(k) + (NSCNG + NGRACS + NNUCCR + NIACR);
      NC3DTEN(k) = NC3DTEN(k) + (-NNUCCC - NPSACWS - NPRA - NPRC - NPSACWI - NPSACWG);
      NI3DTEN(k) = NI3DTEN(k) + (NNUCCC - NPRCI - NPRAI + NMULTS + NMULTG + NMULTR + NMULTRG + NNUCCD - NIACR -
                                 NIACRS);
      NR3DTEN(k) = NR3DTEN(k) + (NPRC1 - NPRACS - NNUCCR + NRAGG - NIACR - NIACRS - NPRACG - NGRACS);

      if (EPRD < 0.0) {
        DUM = EPRD * DT / QI3D(k);
        DUM = fmax2(-1.0, DUM);
        NSUBI = DUM * NI3D(k) / DT;
      }
      if (EPRDS < 0.0) {
        DUM = EPRDS * DT / QNI3D(k);
        DUM = fmax2(-1.0, DUM);
        NSUBS = DUM * NS3D(k) / DT;
      }
      if (PRE < 0.0) {
        DUM = PRE * DT / QR3D(k);
        DUM = fmax2(-1.0, DUM);
        NSUBR = DUM * NR3D(k) / DT;
      }
      if (EPRDG < 0.0) {
        DUM = EPRDG * DT / QG3D(k);
        DUM = fmax2(-1.0, DUM);
        NSUBG = DUM * NG3D(k) / DT;
      }
      NI3DTEN(k) = NI3DTEN(k) + NSUBI;
      NS3DTEN(k) = NS3DTEN(k) + NSUBS;
      NG3DTEN(k) = NG3DTEN(k) + NSUBG;
      NR3DTEN(k) = NR3DTEN(k) + NSUBR;

      // SATURATION ADJUSTMENT
      {
        Real DUMT = T3D(k) + DT * T3DTEN(k);
        Real DUMQV = QV3D(k) + DT * QV3DTEN(k);
        Real DUMQC = QC3D(k) + DT * QC3DTEN(k);
        const Real DUMQR = QR3D(k) + DT * QR3DTEN(k);
        const Real DUMMQS = QNI3D(k) + DT * QNI3DTEN(k);
        const Real DUMQG = QG3D(k) + DT * QG3DTEN(k);
        const Real DUMQI = QI3D(k) + DT * QI3DTEN(k);
        if (eqtset == 2) PRES(k) = rho_k * DUMT * (P.r + P.rv * DUMQV);
        const Real SAVEQC = DUMQC;
        const Real SAVET = DUMT;
        Real prs_k = PRES(k);
        iflag |= impl::satadj2(P, DUMT, prs_k, DUMQV, DUMQC, DUMQR, DUMQI, DUMMQS, DUMQG, rho_k, eqtset, tsmall);
        PRES(k) = prs_k;
        if (Kokkos::fabs(DUMT - SAVET) >= 0.0001) {
          const Real PCC = (DUMQC - SAVEQC) * RDT;
          QV3DTEN(k) = QV3DTEN(k) - PCC;
          QC3DTEN(k) = QC3DTEN(k) + PCC;
          T3DTEN(k) = T3DTEN(k) + (DUMT - SAVET) * RDT;
        }
      }

      // ACTIVATION OF CLOUD DROPLETS (INUM=0; ISUB=1, IBASE=1, IACT=2)
      if (P.inum == 0) {
        if (QC3D(k) + QC3DTEN(k) * DT >= P.qsmall) {
          DUM = 0.5 * (w(i, j, k) + w(i, j, k + 1));
          if (DUM >= 0.001) {
            int IDROP = 0;
            if (QC3D(k) <= 0.05e-3 / rho_k) IDROP = 1;
            if (k == 1) {
              IDROP = 1;
            } else if (k >= 2) {
              if (QC3D(k) > 0.05e-3 / rho_k && QC3D(k - 1) <= 0.05e-3 / RHO(k - 1)) IDROP = 1;
            }
            if (IDROP == 1) {
              DUM2 = arg_activated(P, T3D(k), XXLV, CPM, EVS, PRES(k), DV, KAP, DUM, rho_k);
              DUM2 = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
              DUM2 = (DUM2 - NC3D(k)) / DT;
              DUM2 = fmax2(0.0, DUM2);
              NC3DTEN(k) = NC3DTEN(k) + DUM2;
            } else {
              const Real TAUC = 1.0 / (2.0 * PI * rho_k * DV * NC3D(k) * (PGAM + 1.0) / LAMC);
              const Real TAUR = (EPSR > 1.0e-8) ? 1.0 / EPSR : 1.0e8;
              const Real TAUI = (EPSI > 1.0e-8) ? 1.0 / EPSI : 1.0e8;
              const Real TAUS = (EPSS > 1.0e-8) ? 1.0 / EPSS : 1.0e8;
              const Real TAUG = (EPSG > 1.0e-8) ? 1.0 / EPSG : 1.0e8;
              DUM3 = (-QVS * rho_k / (PRES(k) - EVS) + DQSDT / P.cp) * P.g * DUM;
              DUM3 = (DUM3 * TAUC * TAUR * TAUI * TAUS * TAUG -
                      (QVS - QVI) * (TAUC * TAUR * TAUI * TAUG + TAUC * TAUR * TAUS * TAUG + TAUC * TAUR * TAUI * TAUS)) /
                     (TAUC * TAUR * TAUI * TAUG + TAUC * TAUR * TAUS * TAUG + TAUC * TAUR * TAUI * TAUS +
                      TAUR * TAUI * TAUS * TAUG + TAUC * TAUI * TAUS * TAUG);
              if (DUM3 / QVS >= 1.0e-6) {
                DUM2 = arg_activated(P, T3D(k), XXLV, CPM, EVS, PRES(k), DV, KAP, DUM, rho_k);
                DUMACT = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
                DUM2 = arg_activated_ss(P, T3D(k), DUM3 / QVS, rho_k);
                DUM2 = fmin2((P.nanew1 + P.nanew2) / rho_k, DUM2);
                DUM2 = fmin2(DUM2, DUMACT);
                DUM2 = (DUM2 - NC3D(k)) / DT;
                DUM2 = fmax2(0.0, DUM2);
                NC3DTEN(k) = NC3DTEN(k) + DUM2;
              }
            }
          }
        }
      }
    }
    LTRUE = 1;
  }   // kloop1

  precrt = 0.0;
  if (LTRUE == 0) return iflag;   // GOTO 400

  //---------------------------------------------------------- sedimentation
  int NSTEPR = 1, NSTEPI = 1, NSTEPS = 1, NSTEPC = 1, NSTEPG = 1;

  Real DLAMI = 0.0, DLAMR = 0.0, DLAMC = 0.0, DLAMS = 0.0, DLAMG = 0.0;
  for (int k = KTE; k >= KTS; --k) {
    const Real rho_k = RHO(k);
    const Real temz = 1.0 / DZQ(k);
    DUMI(k) = QI3D(k) + QI3DTEN(k) * DT;
    DUMQS(k) = QNI3D(k) + QNI3DTEN(k) * DT;
    DUMR(k) = QR3D(k) + QR3DTEN(k) * DT;
    DUMFNI(k) = NI3D(k) + NI3DTEN(k) * DT;
    DUMFNS(k) = NS3D(k) + NS3DTEN(k) * DT;
    DUMFNR(k) = NR3D(k) + NR3DTEN(k) * DT;
    DUMC(k) = QC3D(k) + QC3DTEN(k) * DT;
    DUMFNC(k) = NC3D(k) + NC3DTEN(k) * DT;
    DUMG(k) = QG3D(k) + QG3DTEN(k) * DT;
    DUMFNG(k) = NG3D(k) + NG3DTEN(k) * DT;
    if (P.inum == 1) DUMFNC(k) = NC3D(k);

    DUMFNI(k) = fmax2(0.0, DUMFNI(k));
    DUMFNS(k) = fmax2(0.0, DUMFNS(k));
    DUMFNC(k) = fmax2(0.0, DUMFNC(k));
    DUMFNR(k) = fmax2(0.0, DUMFNR(k));
    DUMFNG(k) = fmax2(0.0, DUMFNG(k));

    if (DUMI(k) >= P.qsmall) {
      DLAMI = pw(CONS[12] * DUMFNI(k) / DUMI(k), 1.0 / P.di);
      DLAMI = fmax2(DLAMI, P.lammini);
      DLAMI = fmin2(DLAMI, P.lammaxi);
    }
    if (DUMR(k) >= P.qsmall) {
      DLAMR = pw(PI * P.rhow * DUMFNR(k) / DUMR(k), 1.0 / 3.0);
      DLAMR = fmax2(DLAMR, P.lamminr);
      DLAMR = fmin2(DLAMR, P.lammaxr);
    }
    Real PGAM = 0.0;
    if (DUMC(k) >= P.qsmall) {
      PGAM = 0.0005714 * (NC3D(k) / 1.0e6 * rho_k) + 0.2714;
      PGAM = 1.0 / sq(PGAM) - 1.0;
      PGAM = fmax2(PGAM, 2.0);
      PGAM = fmin2(PGAM, 10.0);
      DLAMC = pw(CONS[26] * DUMFNC(k) * impl::gamma(PGAM + 4.0) / (DUMC(k) * impl::gamma(PGAM + 1.0)), 1.0 / 3.0);
      const Real LAMMIN = (PGAM + 1.0) / 60.0e-6;
      const Real LAMMAX = (PGAM + 1.0) / 1.0e-6;
      DLAMC = fmax2(DLAMC, LAMMIN);
      DLAMC = fmin2(DLAMC, LAMMAX);
    }
    if (DUMQS(k) >= P.qsmall) {
      DLAMS = pw(CONS[1] * DUMFNS(k) / DUMQS(k), 1.0 / P.ds);
      DLAMS = fmax2(DLAMS, P.lammins);
      DLAMS = fmin2(DLAMS, P.lammaxs);
    }
    if (DUMG(k) >= P.qsmall) {
      DLAMG = pw(CONS[2] * DUMFNG(k) / DUMG(k), 1.0 / P.dg);
      DLAMG = fmax2(DLAMG, P.lamming);
      DLAMG = fmin2(DLAMG, P.lammaxg);
    }

    Real UNC, UMC, UNI, UMI, UNR, UMR, UMS, UNS, UMG, UNG;
    if (DUMC(k) >= P.qsmall) {
      UNC = ACN(k) * impl::gamma(1.0 + P.bc + PGAM) / (pw(DLAMC, P.bc) * impl::gamma(PGAM + 1.0));
      UMC = ACN(k) * impl::gamma(4.0 + P.bc + PGAM) / (pw(DLAMC, P.bc) * impl::gamma(PGAM + 4.0));
    } else {
      UMC = 0.0;
      UNC = 0.0;
    }
    if (DUMI(k) >= P.qsmall) {
      UNI = AIN(k) * CONS[27] / pw(DLAMI, P.bi);
      UMI = AIN(k) * CONS[28] / (pw(DLAMI, P.bi));
    } else {
      UMI = 0.0;
      UNI = 0.0;
    }
    if (DUMR(k) >= P.qsmall) {
      UNR = ARN(k) * CONS[6] / pw(DLAMR, P.br);
      UMR = ARN(k) * CONS[4] / (pw(DLAMR, P.br));
    } else {
      UMR = 0.0;
      UNR = 0.0;
    }
    if (DUMQS(k) >= P.qsmall) {
      UMS = ASN(k) * CONS[3] / (pw(DLAMS, P.bs));
      UNS = ASN(k) * CONS[5] / pw(DLAMS, P.bs);
    } else {
      UMS = 0.0;
      UNS = 0.0;
    }
    if (DUMG(k) >= P.qsmall) {
      UMG = AGN(k) * CONS[7] / (pw(DLAMG, P.bg));
      UNG = AGN(k) * CONS[8] / pw(DLAMG, P.bg);
    } else {
      UMG = 0.0;
      UNG = 0.0;
    }

    const Real dum = pw(P.rhosu / rho_k, 0.54);
    UMS = fmin2(UMS, 1.2 * dum);
    UNS = fmin2(UNS, 1.2 * dum);
    UMI = fmin2(UMI, 1.2 * pw(P.rhosu / rho_k, 0.35));
    UNI = fmin2(UNI, 1.2 * pw(P.rhosu / rho_k, 0.35));
    UMR = fmin2(UMR, 9.1 * dum);
    UNR = fmin2(UNR, 9.1 * dum);
    UMG = fmin2(UMG, 20.0 * dum);
    UNG = fmin2(UNG, 20.0 * dum);

    FR(k) = UMR;
    FI(k) = UMI;
    FNI(k) = UNI;
    FS(k) = UMS;
    FNS(k) = UNS;
    FNR(k) = UNR;
    FC(k) = UMC;
    FNC(k) = UNC;
    FG(k) = UMG;
    FNG(k) = UNG;

    if (k <= KTE - 1) {
      if (FR(k) < 1.0e-10) FR(k) = FR(k + 1);
      if (FI(k) < 1.0e-10) FI(k) = FI(k + 1);
      if (FNI(k) < 1.0e-10) FNI(k) = FNI(k + 1);
      if (FS(k) < 1.0e-10) FS(k) = FS(k + 1);
      if (FNS(k) < 1.0e-10) FNS(k) = FNS(k + 1);
      if (FNR(k) < 1.0e-10) FNR(k) = FNR(k + 1);
      if (FC(k) < 1.0e-10) FC(k) = FC(k + 1);
      if (FNC(k) < 1.0e-10) FNC(k) = FNC(k + 1);
      if (FG(k) < 1.0e-10) FG(k) = FG(k + 1);
      if (FNG(k) < 1.0e-10) FNG(k) = FNG(k + 1);
    }

    Real RGVM = fmax2(FR(k), FNR(k));
    NSTEPR = Kokkos::max(int(RGVM * DT * temz + 1.0), NSTEPR);
    RGVM = fmax2(FI(k), FNI(k));
    NSTEPI = Kokkos::max(int(RGVM * DT * temz + 1.0), NSTEPI);
    RGVM = fmax2(FS(k), FNS(k));
    NSTEPS = Kokkos::max(int(RGVM * DT * temz + 1.0), NSTEPS);
    RGVM = fmax2(FC(k), FNC(k));
    NSTEPC = Kokkos::max(int(RGVM * DT * temz + 1.0), NSTEPC);
    RGVM = fmax2(FG(k), FNG(k));
    NSTEPG = Kokkos::max(int(RGVM * DT * temz + 1.0), NSTEPG);

    DUMR(k) = DUMR(k) * rho_k;
    DUMI(k) = DUMI(k) * rho_k;
    DUMFNI(k) = DUMFNI(k) * rho_k;
    DUMQS(k) = DUMQS(k) * rho_k;
    DUMFNS(k) = DUMFNS(k) * rho_k;
    DUMFNR(k) = DUMFNR(k) * rho_k;
    DUMC(k) = DUMC(k) * rho_k;
    DUMFNC(k) = DUMFNC(k) * rho_k;
    DUMG(k) = DUMG(k) * rho_k;
    DUMFNG(k) = DUMFNG(k) * rho_k;
  }

  // Substepped fallout of one species (mass q, number n): the Fortran's
  // FALOUT arrays are formed from the values before this substep's
  // updates, so carry FALOUT(k+1) down the column in a scalar.
  Real snowrt = 0.0;
  auto fall = [&](int nstep, int iF, int iFN, int iD, int iDN, int iSTEN, int iNTEN, bool snow) {
    const Real dttmp = DT / Real(nstep);
    for (int n = 1; n <= nstep; ++n) {
      int k = KTE;
      Real fq_kp1 = wk(i, j, k, iF) * wk(i, j, k, iD);
      Real fn_kp1 = wk(i, j, k, iFN) * wk(i, j, k, iDN);
      Real temz = 1.0 / DZQ(k);
      Real temr = 1.0 / (Real(nstep) * RHO(k));
      Real faltnd = fq_kp1 * temz;
      Real faltndn = fn_kp1 * temz;
      wk(i, j, k, iSTEN) = wk(i, j, k, iSTEN) - faltnd * temr;
      wk(i, j, k, iNTEN) = wk(i, j, k, iNTEN) - faltndn * temr;
      wk(i, j, k, iD) = wk(i, j, k, iD) - faltnd * dttmp;
      wk(i, j, k, iDN) = wk(i, j, k, iDN) - faltndn * dttmp;
      Real fq_k = fq_kp1;
      for (k = KTE - 1; k >= KTS; --k) {
        fq_k = wk(i, j, k, iF) * wk(i, j, k, iD);
        const Real fn_k = wk(i, j, k, iFN) * wk(i, j, k, iDN);
        temz = 1.0 / DZQ(k);
        temr = 1.0 / (Real(nstep) * RHO(k));
        faltnd = (fq_kp1 - fq_k) * temz;
        faltndn = (fn_kp1 - fn_k) * temz;
        wk(i, j, k, iSTEN) = wk(i, j, k, iSTEN) + faltnd * temr;
        wk(i, j, k, iNTEN) = wk(i, j, k, iNTEN) + faltndn * temr;
        wk(i, j, k, iD) = wk(i, j, k, iD) + faltnd * dttmp;
        wk(i, j, k, iDN) = wk(i, j, k, iDN) + faltndn * dttmp;
        fq_kp1 = fq_k;
        fn_kp1 = fn_k;
      }
      precrt = precrt + (fq_k) * dttmp;
      if (snow) snowrt = snowrt + (fq_k) * dttmp;
    }
  };
  fall(NSTEPR, W_FR, W_FNR, W_DUMR, W_DUMFNR, W_QRS, W_NRT, false);    // RAIN
  fall(NSTEPI, W_FI, W_FNI, W_DUMI, W_DUMFNI, W_QIS, W_NIT, true);     // CLOUD ICE
  fall(NSTEPC, W_FC, W_FNC, W_DUMC, W_DUMFNC, W_QCS, W_NCT, false);    // CLOUD DROPLETS
  fall(NSTEPS, W_FS, W_FNS, W_DUMQS, W_DUMFNS, W_QNIS, W_NST, true);   // SNOW
  fall(NSTEPG, W_FG, W_FNG, W_DUMG, W_DUMFNG, W_QGS, W_NGT, true);     // GRAUPEL

  //---------------------------------------------------------- final kloop
  for (int k = KTS; k <= KTE; ++k) {
    const Real rho_k = RHO(k);
    QR3DTEN(k) = QR3DTEN(k) + QRSTEN(k);
    QI3DTEN(k) = QI3DTEN(k) + QISTEN(k);
    QC3DTEN(k) = QC3DTEN(k) + QCSTEN(k);
    QG3DTEN(k) = QG3DTEN(k) + QGSTEN(k);
    QNI3DTEN(k) = QNI3DTEN(k) + QNISTEN(k);

    if (QI3D(k) >= P.qsmall && T3D(k) < P.tmelt && LAMI(k) >= 1.0e-10) {
      if (1.0 / LAMI(k) >= 2.0 * P.dcs) {
        QNI3DTEN(k) = QNI3DTEN(k) + QI3D(k) / DT + QI3DTEN(k);
        NS3DTEN(k) = NS3DTEN(k) + NI3D(k) / DT + NI3DTEN(k);
        QI3DTEN(k) = -QI3D(k) / DT;
        NI3DTEN(k) = -NI3D(k) / DT;
      }
    }

    QC3D(k) = QC3D(k) + QC3DTEN(k) * DT;
    QI3D(k) = QI3D(k) + QI3DTEN(k) * DT;
    QNI3D(k) = QNI3D(k) + QNI3DTEN(k) * DT;
    QR3D(k) = QR3D(k) + QR3DTEN(k) * DT;
    NC3D(k) = NC3D(k) + NC3DTEN(k) * DT;
    NI3D(k) = NI3D(k) + NI3DTEN(k) * DT;
    NS3D(k) = NS3D(k) + NS3DTEN(k) * DT;
    NR3D(k) = NR3D(k) + NR3DTEN(k) * DT;
    QG3D(k) = QG3D(k) + QG3DTEN(k) * DT;
    NG3D(k) = NG3D(k) + NG3DTEN(k) * DT;

    T3D(k) = T3D(k) + T3DTEN(k) * DT;
    QV3D(k) = QV3D(k) + QV3DTEN(k) * DT;
    if (eqtset == 2) PRES(k) = rho_k * T3D(k) * (P.r + P.rv * QV3D(k));

    Real EVS = fmin2(PRES(k) * 0.5, 611.2 * Kokkos::exp(17.67 * (T3D(k) - 273.15) / (T3D(k) - 29.65)));
    Real EIS = fmin2(PRES(k) * 0.5, 611.2 * Kokkos::exp(21.8745584 * (T3D(k) - 273.15) / (T3D(k) - 7.66)));
    if (EIS > EVS) EIS = EVS;
    const Real QVS = P.ep_2 * EVS / (PRES(k) - EVS);
    const Real QVI = P.ep_2 * EIS / (PRES(k) - EIS);
    const Real QVQVS = QV3D(k) / QVS;
    const Real QVQVSI = QV3D(k) / QVI;

    if (QVQVS < 0.9) {
      if (QR3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QR3D(k);
        T3D(k) = T3D(k) - QR3D(k) * ECND(k);
        QR3D(k) = 0.0;
      }
      if (QC3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QC3D(k);
        T3D(k) = T3D(k) - QC3D(k) * ECND(k);
        QC3D(k) = 0.0;
      }
    }
    if (QVQVSI < 0.9) {
      if (QI3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QI3D(k);
        T3D(k) = T3D(k) - QI3D(k) * EDEP(k);
        QI3D(k) = 0.0;
      }
      if (QNI3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QNI3D(k);
        T3D(k) = T3D(k) - QNI3D(k) * EDEP(k);
        QNI3D(k) = 0.0;
      }
      if (QG3D(k) < P.qthr) {
        QV3D(k) = QV3D(k) + QG3D(k);
        T3D(k) = T3D(k) - QG3D(k) * EDEP(k);
        QG3D(k) = 0.0;
      }
    }

    if (QC3D(k) < P.qsmall) { QC3D(k) = 0.0; NC3D(k) = 0.0; }
    if (QR3D(k) < P.qsmall) { QR3D(k) = 0.0; NR3D(k) = 0.0; }
    if (QI3D(k) < P.qsmall) { QI3D(k) = 0.0; NI3D(k) = 0.0; }
    if (QNI3D(k) < P.qsmall) { QNI3D(k) = 0.0; NS3D(k) = 0.0; }
    if (QG3D(k) < P.qsmall) { QG3D(k) = 0.0; NG3D(k) = 0.0; }

    Real LAMS = 0.0, LAMR = 0.0, LAMG = 0.0, LAMC = 0.0, PGAM = 0.0;
    if (!(QC3D(k) < P.qsmall && QI3D(k) < P.qsmall && QNI3D(k) < P.qsmall && QR3D(k) < P.qsmall &&
          QG3D(k) < P.qsmall)) {
      // INSTANTANEOUS PROCESSES
      if (QI3D(k) >= P.qsmall && T3D(k) >= P.tmelt) {
        QR3D(k) = QR3D(k) + QI3D(k);
        T3D(k) = T3D(k) - QI3D(k) * EFRZ(k);
        QI3D(k) = 0.0;
        NR3D(k) = NR3D(k) + NI3D(k);
        NI3D(k) = 0.0;
      }
      // HOMOGENEOUS FREEZING OF CLOUD WATER
      if (T3D(k) <= 233.15 && QC3D(k) >= P.qsmall) {
        QI3D(k) = QI3D(k) + QC3D(k);
        T3D(k) = T3D(k) + QC3D(k) * EFRZ(k);
        QC3D(k) = 0.0;
        NI3D(k) = NI3D(k) + NC3D(k);
        NC3D(k) = 0.0;
      }
      // HOMOGENEOUS FREEZING OF RAIN (IGRAUP=0)
      if (T3D(k) <= 233.15 && QR3D(k) >= P.qsmall) {
        QG3D(k) = QG3D(k) + QR3D(k);
        T3D(k) = T3D(k) + QR3D(k) * EFRZ(k);
        QR3D(k) = 0.0;
        NG3D(k) = NG3D(k) + NR3D(k);
        NR3D(k) = 0.0;
      }

      NI3D(k) = fmax2(0.0, NI3D(k));
      NS3D(k) = fmax2(0.0, NS3D(k));
      NC3D(k) = fmax2(0.0, NC3D(k));
      NR3D(k) = fmax2(0.0, NR3D(k));
      NG3D(k) = fmax2(0.0, NG3D(k));

      // CLOUD ICE
      if (QI3D(k) >= P.qsmall) {
        LAMI(k) = pw(CONS[12] * NI3D(k) / QI3D(k), 1.0 / P.di);
        if (LAMI(k) < P.lammini) {
          LAMI(k) = P.lammini;
          const Real N0I = pw(LAMI(k), P.di + 1.0) * QI3D(k) / CONS[12];
          NI3D(k) = N0I / LAMI(k);
        } else if (LAMI(k) > P.lammaxi) {
          LAMI(k) = P.lammaxi;
          const Real N0I = pw(LAMI(k), P.di + 1.0) * QI3D(k) / CONS[12];
          NI3D(k) = N0I / LAMI(k);
        }
      }
      // RAIN
      if (QR3D(k) >= P.qsmall) {
        LAMR = pw(PI * P.rhow * NR3D(k) / QR3D(k), 1.0 / 3.0);
        if (LAMR < P.lamminr) {
          LAMR = P.lamminr;
          const Real N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
          NR3D(k) = N0RR / LAMR;
        } else if (LAMR > P.lammaxr) {
          LAMR = P.lammaxr;
          const Real N0RR = pow4(LAMR) * QR3D(k) / (PI * P.rhow);
          NR3D(k) = N0RR / LAMR;
        }
      }
      // CLOUD DROPLETS
      if (QC3D(k) >= P.qsmall) {
        PGAM = 0.0005714 * (NC3D(k) / 1.0e6 * rho_k) + 0.2714;
        PGAM = 1.0 / sq(PGAM) - 1.0;
        PGAM = fmax2(PGAM, 2.0);
        PGAM = fmin2(PGAM, 10.0);
        LAMC = pw(CONS[26] * NC3D(k) * impl::gamma(PGAM + 4.0) / (QC3D(k) * impl::gamma(PGAM + 1.0)), 1.0 / 3.0);
        const Real LAMMIN = (PGAM + 1.0) / 60.0e-6;
        const Real LAMMAX = (PGAM + 1.0) / 1.0e-6;
        if (LAMC < LAMMIN) {
          LAMC = LAMMIN;
          NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                    CONS[26];
        } else if (LAMC > LAMMAX) {
          LAMC = LAMMAX;
          NC3D(k) = Kokkos::exp(3.0 * Kokkos::log(LAMC) + Kokkos::log(QC3D(k)) +
                                Kokkos::log(impl::gamma(PGAM + 1.0)) - Kokkos::log(impl::gamma(PGAM + 4.0))) /
                    CONS[26];
        }
      }
      // SNOW
      if (QNI3D(k) >= P.qsmall) {
        LAMS = pw(CONS[1] * NS3D(k) / QNI3D(k), 1.0 / P.ds);
        if (LAMS < P.lammins) {
          LAMS = P.lammins;
          const Real N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
          NS3D(k) = N0S / LAMS;
        } else if (LAMS > P.lammaxs) {
          LAMS = P.lammaxs;
          const Real N0S = pw(LAMS, P.ds + 1.0) * QNI3D(k) / CONS[1];
          NS3D(k) = N0S / LAMS;
        }
      }
      // GRAUPEL
      if (QG3D(k) >= P.qsmall) {
        LAMG = pw(CONS[2] * NG3D(k) / QG3D(k), 1.0 / P.dg);
        if (LAMG < P.lamming) {
          LAMG = P.lamming;
          const Real N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
          NG3D(k) = N0G / LAMG;
        } else if (LAMG > P.lammaxg) {
          LAMG = P.lammaxg;
          const Real N0G = pw(LAMG, P.dg + 1.0) * QG3D(k) / CONS[2];
          NG3D(k) = N0G / LAMG;
        }
      }
    }
    // 500 CONTINUE

    if (dorad) {
      effi(i, j, k) = (QI3D(k) >= P.qsmall) ? 3.0 / LAMI(k) / 2.0 * 1.0e6 : 25.0;
      effs(i, j, k) = (QNI3D(k) >= P.qsmall) ? 3.0 / LAMS / 2.0 * 1.0e6 : 25.0;
      effr(i, j, k) = (QR3D(k) >= P.qsmall) ? 3.0 / LAMR / 2.0 * 1.0e6 : 25.0;
      effc(i, j, k) = (QC3D(k) >= P.qsmall)
                          ? impl::gamma(PGAM + 4.0) / impl::gamma(PGAM + 3.0) / LAMC / 2.0 * 1.0e6
                          : 25.0;
      effg(i, j, k) = (QG3D(k) >= P.qsmall) ? 3.0 / LAMG / 2.0 * 1.0e6 : 25.0;
      if (QI3D(k) >= P.qsmall && QNI3D(k) >= P.qsmall)
        effis(i, j, k) = 3.0 / 2.0 * 1.0e6 * (1.0 / pow4(LAMI(k)) + 1.0 / pow4(LAMS)) /
                         (1.0 / cube(LAMI(k)) + 1.0 / cube(LAMS));
      if (QI3D(k) >= P.qsmall && QNI3D(k) < P.qsmall) effis(i, j, k) = effi(i, j, k);
      if (QI3D(k) < P.qsmall && QNI3D(k) >= P.qsmall) effis(i, j, k) = effs(i, j, k);
      if (QI3D(k) < P.qsmall && QNI3D(k) < P.qsmall) effis(i, j, k) = 25.0;
    }

    NI3D(k) = fmin2(NI3D(k), 0.3e6 / rho_k);
    if (P.inum == 0) NC3D(k) = fmin2(NC3D(k), (P.nanew1 + P.nanew2) / rho_k);   // IACT=2
    if (P.inum == 1) NC3D(k) = P.ndcnst * 1.0e6 / rho_k;
  }
  (void)snowrt;
  return iflag;
}

//---------------------------------------------------------------------------
// MP_GRAUPEL's per-column part.
KOKKOS_INLINE_FUNCTION void Column::operator()(int i, int j, int& err) const {
  for (int k = 1; k <= nk; ++k) {
    for (int v = W_QCT; v <= W_NGT; ++v) wk(i, j, k, v) = 0.0;
    QRSTEN(k) = 0.0;
    QISTEN(k) = 0.0;
    QNISTEN(k) = 0.0;
    QCSTEN(k) = 0.0;
    QGSTEN(k) = 0.0;
    FR(k) = 0.0;
    FI(k) = 0.0;
    FC(k) = 0.0;
    FS(k) = 0.0;
    FG(k) = 0.0;
    NC3D(k) = (p.inum == 0) ? q(i, j, k, nncc) : 0.0;
    PRES(k) = pres(i, j, k);
  }
  if (dorad) {
    for (int k = 1; k <= nk; ++k) {
      effc(i, j, k) = 0.0;
      effi(i, j, k) = 0.0;
      effs(i, j, k) = 0.0;
      effr(i, j, k) = 0.0;
      effg(i, j, k) = 0.0;
      effis(i, j, k) = 0.0;
    }
  }
  Real precprt = 0.0;
  const int iflag = micro(i, j, precprt, nullptr);
  if (iflag != 0) err = err + 1;

  if (dorad) {
    if (radopt == 1) {
      for (int k = 1; k <= nk; ++k) {
        effi(i, j, k) = fmin2(effi(i, j, k), 400.0);
        effs(i, j, k) = fmin2(effs(i, j, k), 400.0);
        effis(i, j, k) = fmin2(effis(i, j, k), 400.0);
      }
    } else if (radopt == 2) {
      for (int k = 1; k <= nk; ++k) {
        effc(i, j, k) = effc(i, j, k) * 1.0e-6;
        effi(i, j, k) = fmin2(140.0, effi(i, j, k)) * 1.0e-6;
        effs(i, j, k) = effs(i, j, k) * 1.0e-6;
      }
    }
  }
  if (p.inum == 0)
    for (int k = 1; k <= nk; ++k) q(i, j, k, nncc) = NC3D(k);
  for (int k = 1; k <= nk; ++k) {
    vtc(i, j, k) = FC(k);
    vtr(i, j, k) = FR(k);
    vti(i, j, k) = FI(k);
    vts(i, j, k) = FS(k);
    vtg(i, j, k) = FG(k);
  }
  for (int n = 1; n <= nrain; ++n) rain(i, j, n) = rain(i, j, n) + 0.1 * precprt;
  prate(i, j) = precprt * rdt;
}

} // namespace

//---------------------------------------------------------------------------
void Work::allocate(const Bounds& B, const Params& pp) {
  p = pp;
  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  wk = FArray4("morrison_wk", {1, 1, 1, 0}, {B.ni, B.nj, B.nk, NWORK - 1});
  tmp = s3("morrison_tmp");
  told = s3("morrison_told");
  qvold = s3("morrison_qvold");
  dzq = s3("morrison_dzq");
  cvm = s3("morrison_cvm");
  tten = s3("morrison_tten");
  vtc = s3("morrison_vtc");
  vtr = s3("morrison_vtr");
  vti = s3("morrison_vti");
  vts = s3("morrison_vts");
  vtg = s3("morrison_vtg");
  effc = s3("morrison_effc");
  effi = s3("morrison_effi");
  effs = s3("morrison_effs");
  effr = s3("morrison_effr");
  effg = s3("morrison_effg");
  effis = s3("morrison_effis");
  prate = FArray2("prate", {B.ib, B.jb}, {B.ie, B.je});
}

//---------------------------------------------------------------------------
void mp_graupel(const Params& p, const Bounds& b, int eqtset, int radopt, Real dt, Real tsmall, bool dorad,
                const FArray3& t, const FArray4& q3d, const FArray3& pres, const FArray3& rho, const FArray3& dzq,
                const FArray3& w, const FArray3& rain, const Work& wk) {
  if (p.inum == 0 && q3d.view().extent(3) < 11)
    throw std::runtime_error("morrison::mp_graupel: inum=0 needs numq=11 (ncc)");
  Column col;
  col.p = p;
  col.nk = b.nk;
  col.eqtset = eqtset;
  col.radopt = radopt;
  col.nrain = b.nrain;
  col.dorad = dorad;
  col.dt = dt;
  col.rdt = 1.0 / dt;
  col.tsmall = tsmall;
  col.t = t;
  col.pres = pres;
  col.rho = rho;
  col.dzq = dzq;
  col.w = w;
  col.q = q3d;
  col.wk = wk.wk;
  col.rain = rain;
  col.prate = wk.prate;
  col.vtc = wk.vtc;
  col.vtr = wk.vtr;
  col.vti = wk.vti;
  col.vts = wk.vts;
  col.vtg = wk.vtg;
  col.effc = wk.effc;
  col.effi = wk.effi;
  col.effs = wk.effs;
  col.effr = wk.effr;
  col.effg = wk.effg;
  col.effis = wk.effis;
  int nerr = 0;
  Kokkos::parallel_reduce("morrison_mp_graupel", box2(1, b.ni, 1, b.nj), col, nerr);
  if (nerr != 0)
    throw std::runtime_error("morrison::satadj2: convergence cannot be reached (stopcm1), " +
                             std::to_string(nerr) + " column(s)");
}

//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real dz, Real rdz,
               const FArray3& rmh, const FArray3& mf, const FArray3& pi0, const FArray3& th0, const FArray3& w3d,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w, bool dorad) {
  const int numq_needed = (w.p.inum == 0) ? 11 : 10;
  if (b.numq < numq_needed)
    throw std::runtime_error("morrison::mp_driver: numq must be " + std::to_string(numq_needed) + " for ptype=5");
  const auto tmp = w.tmp, told = w.told, qvold = w.qvold, dzq = w.dzq;

  // dum1 = T, thten = T, qten(nqv) = qv
  Kokkos::parallel_for("morrison_T", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    tmp(i, j, k) = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));
    told(i, j, k) = tmp(i, j, k);
    qvold(i, j, k) = q3d(i, j, k, nqv);
  });

  kessler::pdefq(b, 0.0, rho, q3d, nqv);
  for (int n = nqc; n <= nqg; ++n) kessler::pdefq(b, opt.qsmall, rho, q3d, n);

  Kokkos::parallel_for("morrison_dz", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    dzq(i, j, k) = dz * rmh(i, j, k);
  });

  mp_graupel(w.p, b, opt.eqtset, opt.radopt, dt, opt.tsmall, dorad, tmp, q3d, prs, rho, dzq, w3d, rain, w);

  if (opt.efall == 1) {
    // dum2 = cvm, dum4 = T tendency
    moist::getcvm(MoistLayout::make(1, 5, w.p.inum), b, c, opt.eqtset, w.cvm, q3d);
    const auto tten = w.tten;
    kessler::getefall(b, c, rdz, 1, c.cpl, mf, tmp, w.cvm, tten, q3d, nqc, w.vtc);
    kessler::getefall(b, c, rdz, 0, c.cpl, mf, tmp, w.cvm, tten, q3d, nqr, w.vtr);
    kessler::getefall(b, c, rdz, 0, c.cpi, mf, tmp, w.cvm, tten, q3d, nqi, w.vti);
    kessler::getefall(b, c, rdz, 0, c.cpi, mf, tmp, w.cvm, tten, q3d, nqs, w.vts);
    kessler::getefall(b, c, rdz, 0, c.cpi, mf, tmp, w.cvm, tten, q3d, nqg, w.vtg);
    Kokkos::parallel_for("morrison_efall", box3(1, b.ni, 1, b.nj, 1, b.nk - 1), KOKKOS_LAMBDA(int i, int j, int k) {
      tmp(i, j, k) = tmp(i, j, k) + dt * tten(i, j, k);
    });
  }

  // final th3d, pp3d, prs (eqtset=2) or th3d, rho (eqtset=1)
  const Constants C = c;
  const Real tsmall = opt.tsmall, qsmall = opt.qsmall;
  if (opt.eqtset == 2) {
    Kokkos::parallel_for("morrison_final2", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (Kokkos::fabs(tmp(i, j, k) - told(i, j, k)) >= tsmall ||
          Kokkos::fabs(q3d(i, j, k, nqv) - qvold(i, j, k)) >= qsmall) {
        prs(i, j, k) = rho(i, j, k) * (C.rd + C.rv * q3d(i, j, k, nqv)) * tmp(i, j, k);
        pp3d(i, j, k) = Kokkos::pow(prs(i, j, k) * phys::rp00, C.rovcp) - pi0(i, j, k);
        th3d(i, j, k) = tmp(i, j, k) / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
      }
    });
  } else {
    Kokkos::parallel_for("morrison_final1", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (Kokkos::fabs(tmp(i, j, k) - told(i, j, k)) >= tsmall ||
          Kokkos::fabs(q3d(i, j, k, nqv) - qvold(i, j, k)) >= qsmall) {
        th3d(i, j, k) = tmp(i, j, k) / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
        rho(i, j, k) = prs(i, j, k) / (C.rd * tmp(i, j, k) * (1.0 + q3d(i, j, k, nqv) * C.reps));
      }
    });
  }
}

void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& w3d, const FArray3& rho, const FArray3& prs, const FArray3& pp3d,
               const FArray3& th3d, const FArray4& q3d, const FArray3& rain, const Work& w, bool dorad) {
  mp_driver(opt, c, g.b, dt, g.dz, g.rdz, g.rmh, g.mf, base.pi0, base.th0, w3d, rho, prs, pp3d, th3d, q3d, rain, w,
            dorad);
}

} // namespace cm1k::morrison
