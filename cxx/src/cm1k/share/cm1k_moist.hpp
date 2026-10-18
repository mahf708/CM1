#ifndef CM1K_MOIST_HPP
#define CM1K_MOIST_HPP

#include "cm1k_farray.hpp"
#include "cm1k_namelist.hpp"

#include <string>
#include <vector>

namespace cm1k {

struct Bounds;
struct Constants;

// The moisture layout of a microphysics scheme: param.F's "Configuration for
// simulations with moisture" (numq, qname, qmag, iice, idm, nql1..nnc2) and
// the indices it derives from qname (nqc, nqr, nqi, nqs, nqg, nnci, nncc).
// Every part of cm1k that loops over species (dyncore, turbulence, PBL,
// init, dumps) takes its indices from here, so a new scheme only adds a row
// to make().
//
// Indices are 1-based (qa(:,:,:,n)), and -2 / 0 mean "absent" exactly as in
// param.F (nql1 = -2 when there is no liquid, nqc = 0 when there is no qc).
//
// Rows (imoist = 1):
//   ptype 0: qv
//   ptype 1: Kessler            qv qc qr
//   ptype 3: Thompson           qv qc qr qi qs qg nci ncr                 (iice, idm)
//   ptype 5: Morrison, inum=1   qv qc qr qi qs qg nci ncs ncr ncg         (iice, idm)
//            Morrison, inum=0   ... + ncc                                 (numq = 11)
// GRAUPEL_INIT (morrison.F) hard-wires INUM = 1 and hands it back to
// param.F, so CM1 always runs ptype=5 with numq = 10; the inum=0 row is
// there for the port's own tests (and is what param.F would set up).
struct MoistLayout {
  int imoist = 0, ptype = 0;
  int numq = 0;                       // 0 when imoist = 0
  int iice = 0, idm = 0;
  int nqv = 0, nqc = 0, nqr = 0, nqi = 0, nqs = 0, nqg = 0, nnci = 0, nncc = 0;
  int nql1 = -2, nql2 = -2, nqs1 = -2, nqs2 = -2, nnc1 = -2, nnc2 = -2;
  std::vector<std::string> qname;     // qname[n-1]
  std::vector<Real> qmag;             // qmag[n-1]: WENO pdef epsilon scale (solve2.F: weps = qmag(n)*epsilon)

  // Throws for (imoist=1) ptypes cm1k has no layout for.
  static MoistLayout make(int imoist, int ptype, int inum = 1);
  // ptypes make() has a row for (0, 1, 3, 5).
  static bool known(int ptype) { return ptype == 0 || ptype == 1 || ptype == 3 || ptype == 5; }
  // imoist, ptype from the namelist (param.F: imoist clamped to 0..1).
  static MoistLayout from_namelist(const Namelist& nl);

  bool has_liquid() const { return nql1 >= 1; }
  bool is_liquid(int n) const { return n >= nql1 && n <= nql2; }
  bool is_solid(int n) const { return iice == 1 && n >= nqs1 && n <= nqs2; }
  bool is_number(int n) const { return n >= nnc1 && n <= nnc2; }
};

namespace moist {

// misclibs.F getcvm, for any layout: cvm = cv (eqtset<=1 or imoist=0), else
// cv + cvv*qv + cpl*q(nql1..nql2) + cpi*q(nqs1..nqs2) (iice=1), accumulated
// in that order. Interior points 1..ni,1..nj,1..nk.
void getcvm(const MoistLayout& ml, const Bounds& b, const Constants& c, int eqtset, const FArray3& cvm,
            const FArray4& q);

} // namespace moist

} // namespace cm1k

#endif
