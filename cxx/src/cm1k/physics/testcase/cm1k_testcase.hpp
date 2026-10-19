#ifndef CM1K_TESTCASE_HPP
#define CM1K_TESTCASE_HPP

// Port of CM1's pre-configured test cases (namelist `testcase` = 1..15) and
// the large-scale pressure-gradient options lspgrad = 2, 3, 4:
//
//   src/param.F                  imoist forced by the test case (1, 2: dry;
//                                3, 4, 5, 7, 8, 11: moist), dolsw (3..7),
//                                use_avg_sfc (1, 2, 9, 11, 14 with sfcmodel=1),
//                                hurr_angle; the stops for testcase 4/5 with
//                                radiation and dolsw with terrain
//   src/base.F                   (after the base-state wind) wprof, thfrc,
//                                qvfrc, ug, vg, dvdr, ufrc, vfrc, ulspg,
//                                vlspg for testcase 3, 4, 5, 6, 7, 10, 11, 12,
//                                13, 15 (init_profiles); ug/vg of iwnd = 8;
//                                the testcase-dependent constant wind of
//                                iwnd = 6 and u0 = 18 m/s of testcase 13
//                                (init/cm1k_init.cpp)
//   src/solve1.F                 lspgrad = 2, 3, 4 (lspgrad), the testcase 10
//                                theta/qv nudging (tqnudge, with
//                                get_avg_uvt(q)), and the call of
//   src/testcase_simple_phys.F   testcase_simple_phys: simple radiation
//                                (testcase 4, 5: simplerad, thraten), the
//                                lspgrad = 3 mesoscale tendencies (ufrc,
//                                vfrc), the time-dependent SGP forcing
//                                (testcase 14), and the prescribed
//                                tendencies added to uten1, vten1, thten1,
//                                qten(nqv)
//   src/adv.F, adv_routines.F    large-scale w (dolsw: wsub, in
//                                dynamics/cm1k_adv.cpp)
//   src/init3d.F                 initial TKE (testcase 3, 4, 5, 9, 11) and
//                                the testcase random perturbations
//                                (init/cm1k_init.cpp)
//   src/sfcphys.F, turb.F        the testcase branches of sfcflux (2, 6, 7,
//                                11), use_avg_sfc (getavgsfc, getcecd,
//                                sfcstress), the testcase 7 surface stress,
//                                the max-gradient hpbl of testcase 1..7
//                                (physics/sfc)
//   src/radiation_driver.F       testcase 8: fixed solcon and coszen
//                                (physics/rrtmg)
//   src/constants.F              testcase 4, 5 (share/cm1k_constants.hpp)
//
// Not ported (Options::check throws): any testcase with sfcmodel other than
// 0 or 1 (the MOST layer sfcmodel = 5 and its testcase branches in cm1most:
// testcase 1, 2, 3, 4, 7, 9, 11, 14), testcase 4, 5 with a PBL scheme (the
// PBL schemes would read the simple-radiation thraten) or without liquid
// water (ptype = 0), testcase 11's two-part subgrid model (sgsmodel = 3, 4),
// the adaptive large-scale pressure gradient (do_adapt_lspg, hard-wired off
// in solve1.F). Diagnostics (domaindiag, writeout, budgets) are not ported.

#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <string>
#include <utility>
#include <vector>

namespace cm1k::testcase {

// param.F: hurr_angle = dble(hurr_rotate)*(pi_dp/180.0d0). With
// -fdefault-real-8, input.F's kind dp is selected_real_kind(30), i.e. quad
// precision: the product is formed in binary128 and rounded to double on
// the assignment (for hurr_rotate = 30 that gives sin(-hurr_angle) = -0.5
// exactly, not the -0.49999999999999994 of a double pi/6). param.F only sets
// it for lspgrad = 3 (it stays 0 otherwise).
inline double param_hurr_angle(int lspgrad, double hurr_rotate) {
  if (lspgrad != 3) return 0.0;
  // 4*atan(1) in binary128, as the exact sum of three doubles (the Q literal
  // suffix is a GNU extension that -std=c++20 under nvcc_wrapper rejects)
  const __float128 pi_qp = __float128(0x1.921fb54442d18p+1) + __float128(0x1.1a62633145c07p-53) +
                           __float128(-0x1.f1976b7ed8fbcp-109);
  return double(__float128(hurr_rotate) * (pi_qp / __float128(180.0)));
}

struct Options {
  int testcase = 0, lspgrad = 0;
  int imoist = 0, ptype = 0, numq = 0, nqv = 1, nql1 = 0, nql2 = 0;
  int isnd = 1, iwnd = 0, cm1setup = 1, ipbl = 0, sgsmodel = 1, sfcmodel = 0, radopt = 0;
  int imove = 0, icor = 0, axisymm = 0;
  bool terrain_flag = false;
  Real fcor = 0.0, umove = 0.0, vmove = 0.0;
  Real hurr_vg = 0.0, hurr_rad = 0.0, hurr_vgpl = 0.0, hurr_rotate = 0.0;
  double hurr_angle = 0.0;   // param.F: dble(hurr_rotate)*(pi_dp/180.0d0)
  bool has_hurr_rad = false;

  bool active() const { return testcase >= 1 || lspgrad >= 2; }
  bool dolsw() const { return testcase >= 3 && testcase <= 7; }   // param.F
  bool tqnudge() const { return testcase == 10; }                  // solve1.F
  bool simple_rad() const { return testcase == 4 || testcase == 5; }
  static Options from_namelist(const Namelist& nl);
  void check() const;   // throws std::runtime_error for anything not ported
};

// param.F: the namelist values the test case overrides (imoist).
Namelist param_overrides(const Namelist& nl);

// The 1D forcing profiles of base.F (kb:ke), as cm1.F allocates them (all
// zero at start), and the simple-radiation arrays of testcase 4, 5 (CM1's
// o30 = frad, thraten (ibr:ier,...) = (ib:ie,jb:je,kb:ke), zir).
struct Forcing {
  FArray1 wprof, ufrc, vfrc, thfrc, qvfrc, ug, vg, dvdr, ulspg, vlspg;
  FArray3 frad, thraten, zir;
  FArray3 w_ql, w_qt, w_d1, w_d2;   // simplerad scratch
  void allocate(const Bounds& b, const Options& o);
  bool allocated() const { return wprof.is_allocated(); }
  // (name, array) of the arrays src/cm1dump_state.inc writes
  std::vector<std::pair<std::string, FArray1>> dumped1() const;
  std::vector<std::pair<std::string, FArray3>> dumped3() const;
};

// base.F: the forcing profiles, from the grid and the base-state wind
// (u0, v0 at i = j = 1, after umove/vmove were subtracted). umove, vmove:
// the moving-domain speeds (0 unless imove = 1).
void init_profiles(const Options& o, const Grid& g, const BaseState& bs, const Forcing& f);

// solve1.F, lspgrad = 2, 3, 4: ulspg/vlspg (2, 3: from ug, vg) added to
// uten1, vten1 on 1..ni+1 x 1..nj+1.
void lspgrad(const Options& o, const Bounds& b, const Forcing& f, const FArray3& uten1, const FArray3& vten1);

// Domain averages of testcase_simple_phys.F get_avg_uvt / get_avg_uvtq
// (area-weighted, summed in double precision in the Fortran loop order).
struct Averages {
  std::vector<double> uavg, vavg, thavg;   // index k = 0..nk+1 (kb:ke)
  std::vector<double> qavg;                // (k, n): k + (nk+2)*(n-1)
};
void get_avg_uvt(const Options& o, const Grid& g, const FArray3& th0, const FArray3& ua, const FArray3& va,
                 const FArray3& tha, const FArray4* qa, Averages& a);

// solve1.F (testcase 10): nudge the domain-mean theta and qv towards the
// base state (ramped up over 900 s, tau = 300 s).
void tqnudge(const Options& o, const Grid& g, const BaseState& bs, double mtime, const FArray3& ua,
             const FArray3& va, const FArray3& tha, const FArray4& qa, const FArray3& thten1,
             const FArray4& qten);

// testcase_simple_phys.F testcase_simple_phys (testcase >= 1).
void simple_phys(const Options& o, const Constants& c, const Grid& g, const BaseState& bs, double mtime,
                 const FArray3& ua, const FArray3& va, const FArray3& tha, const FArray4& qa, const Forcing& f,
                 const FArray3& uten1, const FArray3& vten1, const FArray3& thten1, const FArray4& qten);

} // namespace cm1k::testcase

#endif
