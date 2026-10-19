#ifndef CM1K_BASE_SND_HPP
#define CM1K_BASE_SND_HPP

#include "init/cm1k_init.hpp"

#include <string>
#include <vector>

namespace cm1k {
namespace init {

// The two option blocks of base.F that define the base-state sounding
// (isnd) and the base-state wind (iwnd), on host arrays with Fortran bounds.
//
// isnd: 1-5, 7-15, 17-23 (isnd = 6 stops in the Fortran and throws here).
// iwnd: 0-11 (base.F has no iwnd = 7 branch, so 7 is zero wind, as in CM1);
//       ignored for isnd = 7 (the winds come from input_sounding; param.F
//       resets iwnd to 0).
// testcase only selects the analytic profiles here (isnd = 8, 15, 19;
// iwnd = 6, 8); the large-scale forcing profiles that base.F sets for a
// testcase (wprof, ufrc, ug/vg, ulspg, dvdr, ...) are not part of this.
struct BaseArrays {
  HArray3 zh;                                    // grid (in)
  HArray3 pi0, prs0, th0, thv0, qv0, qc0, qi0;   // (out)
  HArray3 rh0, t0, rho0;                         // (out; scratch for rho0)
  HArray3 u0, v0;                                // (out: isnd = 7 and iwnd)
};

// eqtset after param.F's adjustments (clamped to 1..2; 1 for psolver 4-7
// and for ptype 4, 7, 50-53, 55).
int param_eqtset(const Namelist& nl);
// iice as param.F sets it (imoist = 1 and an ice scheme)
int param_iice(const Namelist& nl);

// The input_sounding file base.F reads for isnd = 7, 17: $CM1K_INPUT_SOUNDING
// if set, else ./input_sounding.
std::string input_sounding_path();

// Contents of an input_sounding file as base.F reads it (list-directed):
// one header record (p_sfc [mb], th_sfc [K], qv_sfc [g/kg]) then records of
// z, th, qv (and u, v when with_wind).
struct InputSounding {
  double p_sfc = 0, th_sfc = 0, qv_sfc = 0;
  std::vector<double> z, th, qv, u, v;   // levels 2..nsnd (1 is the surface)
};
InputSounding read_input_sounding(const std::string& path, bool with_wind);

// base.F, "Start definition of base state sounding (isnd option)" up to
// "End definition ...": pi0, prs0, th0, qv0, qc0, qi0, rh0, t0, thv0 (and
// u0, v0 for isnd = 7) on the points the Fortran sets.
void base_sounding(const Namelist& nl, const Constants& c, const Grid& g, const InitInfo& info, BaseArrays& a);

// base.F, "Start definition of base state wind (iwnd option)" (u0, v0 on the
// points the Fortran sets; skipped for isnd = 7). umove/vmove, bcu/bcv and the
// k = 0, nk+1 extrapolation follow in init_base.
void base_wind(const Namelist& nl, const Constants& c, const Grid& g, const InitInfo& info, BaseArrays& a);

} // namespace init
} // namespace cm1k

#endif
