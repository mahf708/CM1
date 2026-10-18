#ifndef CM1K_RRTMG_SW_TABLES_HPP
#define CM1K_RRTMG_SW_TABLES_HPP

// RRTMG_SW's data (module_ra_rrtmg_sw.F modules parrrsw, rrsw_aer, rrsw_cld,
// rrsw_con, rrsw_kg16..29, rrsw_ref, rrsw_tbl, rrsw_wvn) after rrtmg_swinit:
// rrtmg_swlookuptable reads RRTMG_SW_DATA, then rrtmg_sw_ini runs swdatinit,
// swcmbdat, swaerpr, swcldpr, swatmref, builds exp_tbl and rwgt, and
// cmbgb16s..29 reduce the 224 g-points of the data file to the 112 used.
// Names and bounds are the Fortran's (see cm1k_rrtmg_lw_tables.hpp).

#include "physics/rrtmg/cm1k_rrtmg_util.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cm1k::rrtmg {

// parrrsw
namespace sw {
constexpr int mxlay = 203, mg = 16, nbndsw = 14, naerec = 6, mxmol = 38, nstr = 2, nmol = 7, ngptsw = 112,
              jpband = 29, jpb1 = 16, jpb2 = 29;
constexpr int ng16 = 6, ng17 = 12, ng18 = 8, ng19 = 8, ng20 = 10, ng21 = 10, ng22 = 2, ng23 = 10, ng24 = 8,
              ng25 = 6, ng26 = 6, ng27 = 8, ng28 = 6, ng29 = 12;
constexpr int ngs16 = 6, ngs17 = 18, ngs18 = 26, ngs19 = 34, ngs20 = 44, ngs21 = 54, ngs22 = 56, ngs23 = 66,
              ngs24 = 74, ngs25 = 80, ngs26 = 86, ngs27 = 94, ngs28 = 100, ngs29 = 112;
constexpr Real rrsw_scon = 1.36822e+03;
// rrsw_tbl
constexpr int ntbl = 10000;
constexpr Real tblint = 10000.0, od_lo = 0.06, pade = 0.278;
} // namespace sw

struct SwKg16 { Real rayl, strrat1; int layreffr; FA<2> absa, absb, selfref, forref; FA<1> sfluxref; };
struct SwKg17 { Real rayl, strrat; int layreffr; FA<2> absa, absb, selfref, forref, sfluxref; };
struct SwKg18 { Real rayl, strrat; int layreffr; FA<2> absa, absb, selfref, forref, sfluxref; };
struct SwKg19 { Real rayl, strrat; int layreffr; FA<2> absa, absb, selfref, forref, sfluxref; };
struct SwKg20 { Real rayl; int layreffr; FA<1> absch4, sfluxref; FA<2> absa, absb, selfref, forref; };
struct SwKg21 { Real rayl, strrat; int layreffr; FA<2> absa, absb, selfref, forref, sfluxref; };
struct SwKg22 { Real rayl, strrat; int layreffr; FA<2> absa, absb, selfref, forref, sfluxref; };
struct SwKg23 { FA<1> rayl, sfluxref; Real givfac; int layreffr; FA<2> absa, selfref, forref; };
struct SwKg24 { FA<2> rayla, absa, absb, selfref, forref, sfluxref; FA<1> raylb, abso3a, abso3b; Real strrat; int layreffr; };
struct SwKg25 { FA<1> rayl, abso3a, abso3b, sfluxref; int layreffr; FA<2> absa; };
struct SwKg26 { FA<1> rayl, sfluxref; };
struct SwKg27 { FA<1> rayl, sfluxref; Real scalekur; int layreffr; FA<2> absa, absb; };
struct SwKg28 { Real rayl, strrat; int layreffr; FA<2> absa, absb, sfluxref; };
struct SwKg29 { Real rayl; int layreffr; FA<1> absh2o, absco2, sfluxref; FA<2> absa, absb, selfref, forref; };

struct SwTables {
  // rrsw_con
  Real fluxfac, heatfac, oneminus, pi, grav, planck, boltz, clight, avogad, alosmt, gascon, radcn1, radcn2,
      sbcnst, secdy;
  // rrsw_aer
  FA<2> rsrtaua, rsrpiza, rsrasya;
  // rrsw_cld
  FA<2> extliq1, ssaliq1, asyliq1, extice2, ssaice2, asyice2, extice3, ssaice3, asyice3, fdlice3;
  FA<1> abari, bbari, cbari, dbari, ebari, fbari;
  // rrsw_ref
  FA<1> pref, preflog, tref;
  // rrsw_tbl
  Real tau_tbl, bpade;
  FA<1> exp_tbl;
  // rrsw_wvn
  IA<1> ng, nspa, nspb, ngc, ngs, ngn, ngb, ngm;
  FA<1> wavenum1, wavenum2, delwave, wt, rwgt;
  // module_ra_rrtmg_lw's retab and O3DATA's data (used by RRTMG_SWRAD), rad.<name>
  FA<1> retab, O3SUM, PPSUM, O3WIN, PPWIN;
  SwKg16 kg16; SwKg17 kg17; SwKg18 kg18; SwKg19 kg19; SwKg20 kg20; SwKg21 kg21; SwKg22 kg22;
  SwKg23 kg23; SwKg24 kg24; SwKg25 kg25; SwKg26 kg26; SwKg27 kg27; SwKg28 kg28; SwKg29 kg29;
};

struct SwTableData {
  SwTables dev, host;
  std::vector<Real> hr;
  std::vector<int> hi;
  Kokkos::View<Real*, Device> dr;
  Kokkos::View<int*, Device> di;
  std::vector<std::pair<std::string, std::pair<const Real*, size_t>>> real_arrays;
  std::vector<std::pair<std::string, std::pair<const int*, size_t>>> int_arrays;
  // band scalars read from the file: (name "kgNN.x", value)
  std::vector<std::pair<std::string, double>> scalars;
};

// rrtmg_swinit's table part: rrtmg_swlookuptable (data_file = RRTMG_SW_DATA)
// + rrtmg_sw_ini(cpdair, grv). Shared per (cpdair, grv, file).
std::shared_ptr<const SwTableData> sw_tables(Real cpdair, Real grv, const std::string& data_file);

} // namespace cm1k::rrtmg

#endif
