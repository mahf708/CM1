#ifndef CM1K_KESSLER_HPP
#define CM1K_KESSLER_HPP

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

namespace cm1k::kessler {

// Port of the Kessler warm-rain scheme (src/kessler.F) and of the ptype=1
// branch of src/mp_driver.F.
//
// Moisture layout (param.F, ptype=1): q3d(:,:,:,1)=qv, 2=qc, 3=qr
// (nqv=1, nql1=2, nql2=3, iice=0, numq=3).
//
// Budgets (qbudget, asq) and tdiag/qdiag diagnostics are not computed: they
// never feed back into the model state.
constexpr int nqv = 1, nqc = 2, nqr = 3;

struct Options {
  int eqtset = 2;          // 1: traditional, 2: mass/energy conserving (param.F clamps to [1,2])
  int efall = 0;           // >=1: heating from falling rain (getefall)
  Real qsmall = 1.0e-24;   // cm1.F, double precision build
  Real tsmall = 1.0e-12;   // cm1.F, double precision build

  // eqtset/efall as param.F sets them (incl. the psolver=4..7 -> eqtset=1 override).
  // Throws unless imoist=1 and ptype=1.
  static Options from_namelist(const Namelist& nl);
};

// Scratch arrays mp_driver uses for ptype=1 (cm1.F names in comments).
struct Work {
  FArray3 tmp;    // dum1: temperature
  FArray3 cvm;    // dum2: cv of moist air (efall only)
  FArray3 tten;   // dum4: efall temperature tendency
  FArray3 vr;     // qten(:,:,:,nqr): rain fall speed (m/s); output, like getvt
  FArray2 prate;  // prate: surface precipitation rate (kg/m2/s), last fallout substep
  void allocate(const Bounds& b);   // all (ib:ie,jb:je,kb:ke) / (ib:ie,jb:je)
};

//---------------------------------------------------------------------------
// mp_driver for ptype=1, i.e. what cm1.F does between solve2 and solve3:
//
//   T = (th0+th3d)*(pi0+pp3d)
//   pdefq(0, qv); pdefq(qsmall, qc); pdefq(qsmall, qr)      (pdscheme=1)
//   k_fallout -> vr
//   [efall>=1: getcvm, getefall, update T/prs/pp3d/th3d]
//   fallout(qr, vr) -> qr, rain, prate
//   kessler        -> th3d, pp3d, prs (eqtset=2) or th3d, rho (eqtset=1), qv, qc, qr
//   satadj(nrk=4)  -> th3d, pp3d, prs (eqtset=2) or th3d, rho (eqtset=1), qv, qc
//
// Inputs/outputs (all inout, interior points 1..ni,1..nj,1..nk):
//   th3d, pp3d  : theta' and pi' at the new time level (cm1.F th3d/pp3d after solve2)
//   q3d         : (ibm:iem,...,numq) moisture at the new time level (cm1.F q3d)
//   rho, prs    : full density and pressure (cm1.F rho/prs)
//   rain        : (ib:ie,jb:je,nrain) accumulated surface rain (cm), all nrain slices
// Read only: pi0, th0 (base state), mh, mf (grid), rdz, dt.
//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real rdz,
               const FArray3& mh, const FArray3& mf, const FArray3& pi0, const FArray3& th0,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w);

// Same, taking grid metrics and base state from the cm1k structs.
void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w);

//---------------------------------------------------------------------------
// The individual routines (same names/arguments as the Fortran; budgets dropped).
//---------------------------------------------------------------------------

// misclibs.F pdefq, pdscheme=1 (hard-wired in param.F): per column, zero
// q(:,:,:,n) < rmax and rescale the column so that sum(rho*q) is conserved.
void pdefq(const Bounds& b, Real rmax, const FArray3& rho, const FArray4& q3d, int n);

// kessler.F k_fallout: rain fall speed.
void k_fallout(const Bounds& b, const FArray3& rho, const FArray4& q3d, int n, const FArray3& vr);

// misclibs.F getcvm (iice=0, nql1..nql2 = 2..3).
void getcvm(const Bounds& b, const Constants& c, int eqtset, const FArray3& cvm, const FArray4& q);

// kessler.F getefall.
void getefall(const Bounds& b, const Constants& c, Real rdz, int setzero, Real cpx, const FArray3& mf,
              const FArray3& t, const FArray3& cvm, const FArray3& tten, const FArray4& q3d, int n,
              const FArray3& vr);

// kessler.F fallout: column sedimentation of q3d(:,:,:,n) with fall speed vq
// (vq is modified below the precipitation edge, as in the Fortran).
void fallout(const Bounds& b, Real dt, Real rdz, const FArray3& mh, const FArray3& rho,
             const FArray4& q3d, int n, const FArray3& vq, const FArray3& rain, const FArray2& prate);

// kessler.F kessler: autoconversion, accretion, rain evaporation.
void kessler(const Bounds& b, const Constants& c, int eqtset, Real dt, const FArray3& pi0,
             const FArray3& th0, const FArray3& tmp, const FArray3& rho, const FArray3& pp3d,
             const FArray3& th3d, const FArray3& prs, const FArray4& q3d);

// kessler.F satadj: saturation adjustment. Throws (like stopcm1) if the
// eqtset=2 iteration does not converge in 100 iterations.
void satadj(const Bounds& b, const Constants& c, int eqtset, Real tsmall, int nrk, Real dt,
            const FArray3& pi0, const FArray3& th0, const FArray3& rho, const FArray3& pp3d,
            const FArray3& prs, const FArray3& th3d, const FArray4& q3d);

} // namespace cm1k::kessler

#endif
