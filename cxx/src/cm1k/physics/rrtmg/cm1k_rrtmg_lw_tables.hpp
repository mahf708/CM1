#ifndef CM1K_RRTMG_LW_TABLES_HPP
#define CM1K_RRTMG_LW_TABLES_HPP

// RRTMG_LW's data (module_ra_rrtmg_lw.F modules parrrtm, rrlw_cld, rrlw_con,
// rrlw_kg01..16, rrlw_ref, rrlw_tbl, rrlw_wvn) after rrtmg_lwinit:
// rrtmg_lwlookuptable reads RRTMG_LW_DATA, then rrtmg_lw_ini runs lwdatinit,
// lwcmbdat, lwcldpr, lwatmref, lwavplank, builds the transmittance tables
// and the g-point weights rwgt, and cmbgb1..16 reduce the 256 g-points of
// the data file to the 140 used.
//
// Every array keeps its Fortran name and bounds; of the EQUIVALENCE'd
// ka/absa, kb/absb pairs only absa/absb (what taumol reads) are kept.
// LwTables is trivially copyable: kernels capture it by value; the data
// live in device Views owned by LwTableData.

#include "physics/rrtmg/cm1k_rrtmg_util.hpp"

#include <memory>
#include <string>

namespace cm1k::rrtmg {

// parrrtm
namespace lw {
constexpr int mg = 16, nbndlw = 16, maxxsec = 4, mxmol = 38, maxinpx = 38, nmol = 7, ngptlw = 140;
constexpr int ng1 = 10, ng2 = 12, ng3 = 16, ng4 = 14, ng5 = 16, ng6 = 8, ng7 = 12, ng8 = 8, ng9 = 12,
              ng10 = 6, ng11 = 8, ng12 = 8, ng13 = 4, ng14 = 2, ng15 = 2, ng16 = 2;
constexpr int ngs1 = 10, ngs2 = 22, ngs3 = 38, ngs4 = 52, ngs5 = 68, ngs6 = 76, ngs7 = 88, ngs8 = 96,
              ngs9 = 108, ngs10 = 114, ngs11 = 122, ngs12 = 130, ngs13 = 134, ngs14 = 136, ngs15 = 138;
// rrlw_tbl
constexpr int ntbl = 10000;
constexpr Real tblint = 10000.0;
constexpr Real pade = 0.278;
} // namespace lw

struct LwKg01 { FA<1> fracrefa, fracrefb; FA<2> absa, absb, ka_mn2, kb_mn2, selfref, forref; };
struct LwKg02 { FA<1> fracrefa, fracrefb; FA<2> absa, absb, selfref, forref; };
struct LwKg03 { FA<2> fracrefa, fracrefb, absa, absb; FA<3> ka_mn2o, kb_mn2o; FA<2> selfref, forref; };
struct LwKg04 { FA<2> fracrefa, fracrefb, absa, absb, selfref, forref; };
struct LwKg05 { FA<2> fracrefa, fracrefb, absa, absb; FA<3> ka_mo3; FA<2> selfref, forref; FA<1> ccl4; };
struct LwKg06 { FA<1> fracrefa; FA<2> absa, ka_mco2, selfref, forref; FA<1> cfc11adj, cfc12; };
struct LwKg07 { FA<1> fracrefb; FA<2> fracrefa, absa, absb; FA<3> ka_mco2; FA<2> kb_mco2, selfref, forref; };
struct LwKg08 {
  FA<1> fracrefa, fracrefb, cfc12, cfc22adj;
  FA<2> absa, absb, ka_mco2, ka_mn2o, ka_mo3, kb_mco2, kb_mn2o, selfref, forref;
};
struct LwKg09 { FA<1> fracrefb; FA<2> fracrefa, absa, absb; FA<3> ka_mn2o; FA<2> kb_mn2o, selfref, forref; };
struct LwKg10 { FA<1> fracrefa, fracrefb; FA<2> absa, absb, selfref, forref; };
struct LwKg11 { FA<1> fracrefa, fracrefb; FA<2> absa, absb, ka_mo2, kb_mo2, selfref, forref; };
struct LwKg12 { FA<2> fracrefa, absa, selfref, forref; };
struct LwKg13 { FA<1> fracrefb; FA<2> fracrefa, absa; FA<3> ka_mco2, ka_mco; FA<2> kb_mo3, selfref, forref; };
struct LwKg14 { FA<1> fracrefa, fracrefb; FA<2> absa, absb, selfref, forref; };
struct LwKg15 { FA<2> fracrefa, absa; FA<3> ka_mn2; FA<2> selfref, forref; };
struct LwKg16 { FA<1> fracrefb; FA<2> fracrefa, absa, absb, selfref, forref; };

struct LwTables {
  // rrlw_con
  Real fluxfac, heatfac, oneminus, pi, grav, planck, boltz, clight, avogad, alosmt, gascon, radcn1, radcn2,
      sbcnst, secdy;
  // rrlw_cld
  Real abscld1, absliq0;
  FA<1> absice0;
  FA<2> absice1, absice2, absice3, absliq1;
  // rrlw_ref
  FA<1> pref, preflog, tref;
  FA<2> chi_mls;
  // rrlw_tbl
  Real bpade;
  FA<1> tau_tbl, exp_tbl, tfn_tbl;
  // rrlw_wvn
  IA<1> ng, nspa, nspb, ngc, ngs, ngn, ngb, ngm, ixindx;
  int nxmol;
  FA<1> wavenum1, wavenum2, delwave, totplk16, wt, rwgt;
  FA<2> totplnk;
  // rtrnmc module data
  Real wtdiff, rec_6;
  FA<1> a0, a1, a2;
  // the combined k-distributions
  LwKg01 kg01; LwKg02 kg02; LwKg03 kg03; LwKg04 kg04; LwKg05 kg05; LwKg06 kg06; LwKg07 kg07; LwKg08 kg08;
  LwKg09 kg09; LwKg10 kg10; LwKg11 kg11; LwKg12 kg12; LwKg13 kg13; LwKg14 kg14; LwKg15 kg15; LwKg16 kg16;
};

// Owner of the table storage; `dev` points into device memory, `host` into
// a host copy (for tests).
struct LwTableData {
  LwTables dev, host;
  std::vector<Real> hr;
  std::vector<int> hi;
  Kokkos::View<Real*, Device> dr;
  Kokkos::View<int*, Device> di;
  // name ("kg01.absa", "tau_tbl", ...) -> host pointer and element count,
  // for the comparison against the Fortran's arrays
  std::vector<std::pair<std::string, std::pair<const Real*, size_t>>> real_arrays;
  std::vector<std::pair<std::string, std::pair<const int*, size_t>>> int_arrays;
};

// rrtmg_lwinit's table part: rrtmg_lwlookuptable (data_file, the path of
// RRTMG_LW_DATA) + rrtmg_lw_ini(cpdair, grv). The tables are the same for
// every caller with the same constants; the result is shared.
std::shared_ptr<const LwTableData> lw_tables(Real cpdair, Real grv, const std::string& data_file);

} // namespace cm1k::rrtmg

#endif
