#ifndef CM1K_CONSTANTS_HPP
#define CM1K_CONSTANTS_HPP

#include "cm1k_types.hpp"

#include <cmath>

namespace cm1k {

// Port of src/constants.F. Parameters are compile-time; the rest depend on
// `testcase` (set_constants) and live in a small trivially-copyable struct
// that kernels capture by value.
namespace phys {
constexpr Real pi      = 3.1415926535897932384626433;
constexpr Real th0r    = 300.0;
constexpr Real to      = 273.15;
constexpr Real p00     = 1.0e5;
constexpr Real rp00    = 1.0 / p00;
constexpr Real piddeg  = pi / 180.0;
constexpr Real degdpi  = 180.0 / pi;
constexpr Real clwsat  = 1.e-6;
constexpr Real cmemin  = 1.e-4;
constexpr Real karman  = 0.40;
constexpr Real rkarman = 1.0 / karman;
constexpr Real omega   = 2.0 * pi / 86400.0;
constexpr Real vialpha = 1.0;
constexpr Real vibeta  = 1.0 - vialpha;
constexpr Real csmax   = 350.0;
constexpr Real c_m     = 0.10;
constexpr Real c_l     = 0.8165;
constexpr Real ri_c    = 0.25;
constexpr Real epsilon = 1.0e-18;
constexpr int  nrkmax  = 3;
constexpr Real grads_undef = -999999.9;
} // namespace phys

struct Constants {
  Real g, rd, rv, cp, cpinv, cv, cpv, cvv, rcp,
       cpdcv, cvdcp, rovcp, rddcp, rddcv, rddrv, cvdrd, cpdrd, eps, reps, repsm1,
       cpt, cvt, pnum, xlv, lathv, xls, xlf, lvdcp, condc, cpl, cpi, lv1, lv2, ls1, ls2,
       rhow, svp1, svp2, svp3, svpt0, c_e1, c_e2, c_s, rcs, earth_radius, govtwo;

  // Same operation order as set_constants in constants.F.
  static Constants make(int testcase) {
    using namespace phys;
    Constants c{};
    if (testcase == 4) {
      c.g = 9.81; c.rd = 287.0; c.cp = 1015.0; c.cpv = 1870.0; c.xlv = 2.47e6; c.xls = 2834000.0;
    } else if (testcase == 5) {
      c.g = 9.81; c.rd = 287.0; c.cp = 1004.0; c.cpv = 1870.0; c.xlv = 2.50e6; c.xls = 2834000.0;
    } else {
      c.g = 9.81; c.rd = 287.04; c.cp = 1005.7; c.cpv = 1870.0; c.xlv = 2501000.0; c.xls = 2834000.0;
    }
    c.earth_radius = 6370000.0;
    c.govtwo = 0.5 * c.g;
    c.rv     = 461.5;
    c.cpinv  = 1.0 / c.cp;
    c.cv     = c.cp - c.rd;
    c.cvv    = c.cpv - c.rv;
    c.rcp    = 1.0 / c.cp;
    c.cpdcv  = c.cp / c.cv;
    c.cvdcp  = c.cv / c.cp;
    c.rovcp  = c.rd / c.cp;
    c.rddcp  = c.rd / c.cp;
    c.rddcv  = c.rd / c.cv;
    c.rddrv  = c.rd / c.rv;
    c.cvdrd  = c.cv / c.rd;
    c.cpdrd  = c.cp / c.rd;
    c.eps    = c.rd / c.rv;
    c.reps   = c.rv / c.rd;
    c.repsm1 = c.rv / c.rd - 1.0;
    c.cpt    = (c.cpv / c.cp) - 1.0;
    c.cvt    = (c.cvv / c.cv) - 1.0;
    c.pnum   = (c.cp * c.rv) - (c.cpv * c.rd);
    c.lathv  = c.xlv;
    c.lvdcp  = c.xlv / c.cp;
    c.condc  = c.xlv * c.xlv / (c.rv * c.cp);
    c.cpl    = 4190.0;
    c.cpi    = 2106.0;
    c.lv1    = c.xlv + (c.cpl - c.cpv) * to;
    c.lv2    = c.cpl - c.cpv;
    c.ls1    = c.xls + (c.cpi - c.cpv) * to;
    c.ls2    = c.cpi - c.cpv;
    c.xlf    = c.xls - c.xlv;
    c.rhow   = 1.0e3;
    c.svp1   = 0.6112;
    c.svp2   = 17.67;
    c.svp3   = 29.65;
    c.svpt0  = 273.15;
    if (testcase == 4 || testcase == 5) { c.lv1 = c.xlv; c.lv2 = 0.0; }
    c.c_e1 = c_m * c_l * c_l * (1.0 / ri_c - 1.0);
    c.c_e2 = std::max(0.0, c_m * pi * pi - c.c_e1);
    c.c_s  = std::pow(c_m * c_m * c_m / (c.c_e1 + c.c_e2), 0.25);
    c.rcs  = 1.0 / c.c_s;
    return c;
  }
};

} // namespace cm1k

#endif
