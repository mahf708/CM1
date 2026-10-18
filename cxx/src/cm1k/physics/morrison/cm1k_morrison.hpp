#ifndef CM1K_MORRISON_HPP
#define CM1K_MORRISON_HPP

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

namespace cm1k::morrison {

// Port of the Morrison two-moment microphysics (src/morrison.F:
// GRAUPEL_INIT, MP_GRAUPEL, M2005MICRO_GRAUPEL, GAMMA, DERF1, satadj2) and
// of the ptype=5 branch of src/mp_driver.F.
//
// Moisture layout (param.F, ptype=5; share/cm1k_moist):
//   q3d(:,:,:,1..10) = qv qc qr qi qs qg nci ncs ncr ncg     (inum=1, numq=10)
//   q3d(:,:,:,11)    = ncc                                    (inum=0, numq=11)
//
// GRAUPEL_INIT hard-wires the scheme's switches; only those paths are
// ported: IACT=2 (lognormal aerosol activation), IBASE=1, ISUB=1, ILIQ=0
// (ice on), INUC=0 (Cooper/Rasmussen nucleation), IGRAUP=0 (graupel on),
// IRAIN=0 (Khairoutdinov-Kogan warm rain). IHAIL and NDCNST come from the
// namelist. GRAUPEL_INIT also hard-wires INUM=1 (constant droplet number)
// and hands it back to param.F, so CM1 always runs with numq=10; the
// predicted-droplet-number path (INUM=0, numq=11) is ported and tested
// against the Fortran module with INUM forced to 0, but no namelist
// reaches it.
//
// Not ported: the radar reflectivity (calc_refl10cm, getdbz), the budget
// and tdiag/qdiag diagnostics (they never feed back into the state), and
// testcase=5's cloud-droplet fall speed (cm1k rejects testcase != 0).
constexpr int nqv = 1, nqc = 2, nqr = 3, nqi = 4, nqs = 5, nqg = 6;
constexpr int nnci = 7, nncs = 8, nncr = 9, nncg = 10, nncc = 11;

// GRAUPEL_INIT's module variables (lower-case Fortran names; cons[n] =
// CONSn, n = 1..41). Plain data, copied into the kernels.
struct Params {
  int inum = 1, ihail = 0;
  Real ndcnst = 0;
  Real ai, ac, as, ar, ag, bi, bc, bs, br, bg;
  Real r, rv, cp, ep_2, cv, cvv, cpv, cpl, cpi, xlv0, xls0, lv1, lv2, ls1, ls2;
  Real tmelt, rhosu, rhow, rhoi, rhosn, rhog, aimm, bimm, ecr, dcs, mi0, mg0;
  Real f1s, f2s, f1r, f2r, g, qsmall, qthr, eii, eci, cpw;
  Real ci, di, cs, ds, cg, dg, rin, mmult;
  Real lammaxi, lammini, lammaxr, lamminr, d_b, lammaxs, lammins, lammaxg, lamming;
  Real k1, c1, mw, osm, vi, epsm, rhoa, map, ma, rr, bact;
  Real rm1, sig1, nanew1, f11, f21, rm2, sig2, nanew2, f12, f22;
  Real cons[42];

  // GRAUPEL_INIT(cm1hail, cm1inum, cm1ndcnst), except that inum is taken
  // as given (GRAUPEL_INIT always sets 1): pass 1 for CM1's behaviour.
  static Params init(int ihail, Real ndcnst, int inum = 1);
};

struct Options {
  int ihail = 0;
  Real ndcnst = 250.0;
  int inum = 1;            // GRAUPEL_INIT: always 1
  int eqtset = 2;          // param.F clamps to [1,2]; psolver=4..7 -> 1
  int efall = 0;           // 1: heating from falling hydrometeors (getefall)
  int radopt = 0;          // effective-radius units when dorad (1: microns, 2: m)
  int nrain = 1;
  Real qsmall = 1.0e-24;   // cm1.F, double precision build (pdefq)
  Real tsmall = 1.0e-12;   // cm1.F, double precision build
  // Throws unless imoist=1 and ptype=5.
  static Options from_namelist(const Namelist& nl);
};

// Column work space of MP_GRAUPEL/M2005MICRO (one slot per (i,j,k)), the
// scratch arrays mp_driver uses, and the scheme's outputs.
struct Work {
  Params p;
  FArray4 wk;                       // (1:ni,1:nj,1:nk,nwork) column arrays
  FArray3 tmp, told, qvold;         // mp_driver: dum1 = T, thten = old T, qten(nqv) = old qv
  FArray3 dzq, cvm, tten;           // dum4 = dz*rmh (then the efall T tendency), dum2 = cvm
  FArray3 vtc, vtr, vti, vts, vtg;  // fall speeds (m/s), MP_GRAUPEL's getvt output
  FArray3 effc, effi, effs, effr, effg, effis;   // effective radii (dorad)
  FArray2 prate;                    // surface precipitation rate (kg/m2/s)
  void allocate(const Bounds& b, const Params& p);
};

//---------------------------------------------------------------------------
// mp_driver for ptype=5 (between solve2 and solve3):
//
//   T = (th0+th3d)*(pi0+pp3d); told = T; qvold = qv
//   pdefq(0, qv); pdefq(qsmall, qc qr qi qs qg)              (pdscheme=1)
//   dzq = dz*rmh
//   MP_GRAUPEL -> T, q3d(1..10 [,11]), rain, prate, vt*, [eff*]
//   [efall=1: getcvm, getefall(qc,qr: cpl; qi,qs,qg: cpi), T += dt*tten (k<nk)]
//   where |T-told| >= tsmall or |qv-qvold| >= qsmall:
//     eqtset=2: prs = rho*(rd+rv*qv)*T; pp3d = (prs*rp00)**rovcp - pi0; th3d = T/(pi0+pp3d) - th0
//     eqtset=1: th3d = T/(pi0+pp3d) - th0; rho = prs/(rd*T*(1+qv*reps))
//
// w3d is the vertical velocity (ib:ie,jb:je,kb:ke+1), used for droplet
// activation (inum=0 only). dorad: compute the effective radii.
//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real dz, Real rdz,
               const FArray3& rmh, const FArray3& mf, const FArray3& pi0, const FArray3& th0,
               const FArray3& w3d, const FArray3& rho, const FArray3& prs, const FArray3& pp3d,
               const FArray3& th3d, const FArray4& q3d, const FArray3& rain, const Work& w,
               bool dorad = false);

void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& w3d, const FArray3& rho, const FArray3& prs, const FArray3& pp3d,
               const FArray3& th3d, const FArray4& q3d, const FArray3& rain, const Work& w,
               bool dorad = false);

//---------------------------------------------------------------------------
// MP_GRAUPEL: every column (1..ni,1..nj) through M2005MICRO_GRAUPEL.
//   t      : temperature (inout)
//   q3d    : species 1..10 (inout); 11 = ncc when p.inum = 0 (inout)
//   p      : pressure (in; M2005MICRO's copy is modified, this one is not)
//   rho    : density; dzq: layer depth (m); w: (kb:ke+1) vertical velocity
//   rain   : (ib:ie,jb:je,nrain) += 0.1*precip; prate = precip/dt
// Fall speeds go to w.vt*, effective radii (dorad) to w.eff*. Throws if
// satadj2 does not converge (stopcm1).
//---------------------------------------------------------------------------
void mp_graupel(const Params& p, const Bounds& b, int eqtset, int radopt, Real dt, Real tsmall,
                bool dorad, const FArray3& t, const FArray4& q3d, const FArray3& pres,
                const FArray3& rho, const FArray3& dzq, const FArray3& w, const FArray3& rain,
                const Work& wk);

// morrison.F GAMMA (Cody & Stoltz) and DERF1, for testing.
Real gamma_host(Real x);
Real derf1_host(Real x);

// morrison.F satadj2 on one point; returns iflag (1: no convergence).
int satadj2_host(const Params& p, Real& tnew, Real& prs, Real& qvnew, Real& qcnew, Real qrnew,
                 Real qinew, Real qsnew, Real qgnew, Real rho, int eqtset, Real tsmall);

} // namespace cm1k::morrison

#endif
