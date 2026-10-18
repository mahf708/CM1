// RRTMG_LW tables: rrtmg_lwlookuptable + rrtmg_lw_ini (module_ra_rrtmg_lw.F).
#include "physics/rrtmg/cm1k_rrtmg_lw_tables.hpp"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace cm1k::rrtmg {

namespace {

#include "physics/rrtmg/cm1k_rrtmg_lw_data.inc"

// Every LwTables array, with its Fortran name and bounds.
template <class B>
void layout(LwTables& t, B& b) {
  using namespace lw;
  // rrlw_cld
  b.arr("absice0", t.absice0, {1}, {2});
  b.arr("absice1", t.absice1, {1, 1}, {2, 5});
  b.arr("absice2", t.absice2, {1, 1}, {43, 16});
  b.arr("absice3", t.absice3, {1, 1}, {46, 16});
  b.arr("absliq1", t.absliq1, {1, 1}, {58, 16});
  // rrlw_ref
  b.arr("pref", t.pref, {1}, {59});
  b.arr("preflog", t.preflog, {1}, {59});
  b.arr("tref", t.tref, {1}, {59});
  b.arr("chi_mls", t.chi_mls, {1, 1}, {7, 59});
  // rrlw_tbl
  b.arr("tau_tbl", t.tau_tbl, {0}, {ntbl});
  b.arr("exp_tbl", t.exp_tbl, {0}, {ntbl});
  b.arr("tfn_tbl", t.tfn_tbl, {0}, {ntbl});
  // rrlw_wvn
  b.arr("ng", t.ng, {1}, {nbndlw});
  b.arr("nspa", t.nspa, {1}, {nbndlw});
  b.arr("nspb", t.nspb, {1}, {nbndlw});
  b.arr("wavenum1", t.wavenum1, {1}, {nbndlw});
  b.arr("wavenum2", t.wavenum2, {1}, {nbndlw});
  b.arr("delwave", t.delwave, {1}, {nbndlw});
  b.arr("totplnk", t.totplnk, {1, 1}, {181, nbndlw});
  b.arr("totplk16", t.totplk16, {1}, {181});
  b.arr("ngc", t.ngc, {1}, {nbndlw});
  b.arr("ngs", t.ngs, {1}, {nbndlw});
  b.arr("ngn", t.ngn, {1}, {ngptlw});
  b.arr("ngb", t.ngb, {1}, {ngptlw});
  b.arr("ngm", t.ngm, {1}, {nbndlw * mg});
  b.arr("wt", t.wt, {1}, {mg});
  b.arr("rwgt", t.rwgt, {1}, {nbndlw * mg});
  b.arr("ixindx", t.ixindx, {1}, {maxinpx});
  // rrtmg_lw_rtrnmc
  b.arr("a0", t.a0, {1}, {nbndlw});
  b.arr("a1", t.a1, {1}, {nbndlw});
  b.arr("a2", t.a2, {1}, {nbndlw});
  // rrlw_kg01..16 (the reduced g-point arrays)
  b.arr("kg01.fracrefa", t.kg01.fracrefa, {1}, {lw::ng1});
  b.arr("kg01.fracrefb", t.kg01.fracrefb, {1}, {lw::ng1});
  b.arr("kg01.absa", t.kg01.absa, {1, 1}, {65, lw::ng1});
  b.arr("kg01.absb", t.kg01.absb, {1, 1}, {235, lw::ng1});
  b.arr("kg01.ka_mn2", t.kg01.ka_mn2, {1, 1}, {19, lw::ng1});
  b.arr("kg01.kb_mn2", t.kg01.kb_mn2, {1, 1}, {19, lw::ng1});
  b.arr("kg01.selfref", t.kg01.selfref, {1, 1}, {10, lw::ng1});
  b.arr("kg01.forref", t.kg01.forref, {1, 1}, {4, lw::ng1});
  b.arr("kg02.fracrefa", t.kg02.fracrefa, {1}, {lw::ng2});
  b.arr("kg02.fracrefb", t.kg02.fracrefb, {1}, {lw::ng2});
  b.arr("kg02.absa", t.kg02.absa, {1, 1}, {65, lw::ng2});
  b.arr("kg02.absb", t.kg02.absb, {1, 1}, {235, lw::ng2});
  b.arr("kg02.selfref", t.kg02.selfref, {1, 1}, {10, lw::ng2});
  b.arr("kg02.forref", t.kg02.forref, {1, 1}, {4, lw::ng2});
  b.arr("kg03.fracrefa", t.kg03.fracrefa, {1, 1}, {lw::ng3, 9});
  b.arr("kg03.fracrefb", t.kg03.fracrefb, {1, 1}, {lw::ng3, 5});
  b.arr("kg03.absa", t.kg03.absa, {1, 1}, {585, lw::ng3});
  b.arr("kg03.absb", t.kg03.absb, {1, 1}, {1175, lw::ng3});
  b.arr("kg03.ka_mn2o", t.kg03.ka_mn2o, {1, 1, 1}, {9, 19, lw::ng3});
  b.arr("kg03.kb_mn2o", t.kg03.kb_mn2o, {1, 1, 1}, {5, 19, lw::ng3});
  b.arr("kg03.selfref", t.kg03.selfref, {1, 1}, {10, lw::ng3});
  b.arr("kg03.forref", t.kg03.forref, {1, 1}, {4, lw::ng3});
  b.arr("kg04.fracrefa", t.kg04.fracrefa, {1, 1}, {lw::ng4, 9});
  b.arr("kg04.fracrefb", t.kg04.fracrefb, {1, 1}, {lw::ng4, 5});
  b.arr("kg04.absa", t.kg04.absa, {1, 1}, {585, lw::ng4});
  b.arr("kg04.absb", t.kg04.absb, {1, 1}, {1175, lw::ng4});
  b.arr("kg04.selfref", t.kg04.selfref, {1, 1}, {10, lw::ng4});
  b.arr("kg04.forref", t.kg04.forref, {1, 1}, {4, lw::ng4});
  b.arr("kg05.fracrefa", t.kg05.fracrefa, {1, 1}, {lw::ng5, 9});
  b.arr("kg05.fracrefb", t.kg05.fracrefb, {1, 1}, {lw::ng5, 5});
  b.arr("kg05.absa", t.kg05.absa, {1, 1}, {585, lw::ng5});
  b.arr("kg05.absb", t.kg05.absb, {1, 1}, {1175, lw::ng5});
  b.arr("kg05.ka_mo3", t.kg05.ka_mo3, {1, 1, 1}, {9, 19, lw::ng5});
  b.arr("kg05.ccl4", t.kg05.ccl4, {1}, {lw::ng5});
  b.arr("kg05.selfref", t.kg05.selfref, {1, 1}, {10, lw::ng5});
  b.arr("kg05.forref", t.kg05.forref, {1, 1}, {4, lw::ng5});
  b.arr("kg06.fracrefa", t.kg06.fracrefa, {1}, {lw::ng6});
  b.arr("kg06.absa", t.kg06.absa, {1, 1}, {65, lw::ng6});
  b.arr("kg06.ka_mco2", t.kg06.ka_mco2, {1, 1}, {19, lw::ng6});
  b.arr("kg06.cfc11adj", t.kg06.cfc11adj, {1}, {lw::ng6});
  b.arr("kg06.cfc12", t.kg06.cfc12, {1}, {lw::ng6});
  b.arr("kg06.selfref", t.kg06.selfref, {1, 1}, {10, lw::ng6});
  b.arr("kg06.forref", t.kg06.forref, {1, 1}, {4, lw::ng6});
  b.arr("kg07.fracrefa", t.kg07.fracrefa, {1, 1}, {lw::ng7, 9});
  b.arr("kg07.fracrefb", t.kg07.fracrefb, {1}, {lw::ng7});
  b.arr("kg07.absa", t.kg07.absa, {1, 1}, {585, lw::ng7});
  b.arr("kg07.absb", t.kg07.absb, {1, 1}, {235, lw::ng7});
  b.arr("kg07.ka_mco2", t.kg07.ka_mco2, {1, 1, 1}, {9, 19, lw::ng7});
  b.arr("kg07.kb_mco2", t.kg07.kb_mco2, {1, 1}, {19, lw::ng7});
  b.arr("kg07.selfref", t.kg07.selfref, {1, 1}, {10, lw::ng7});
  b.arr("kg07.forref", t.kg07.forref, {1, 1}, {4, lw::ng7});
  b.arr("kg08.fracrefa", t.kg08.fracrefa, {1}, {lw::ng8});
  b.arr("kg08.fracrefb", t.kg08.fracrefb, {1}, {lw::ng8});
  b.arr("kg08.absa", t.kg08.absa, {1, 1}, {65, lw::ng8});
  b.arr("kg08.absb", t.kg08.absb, {1, 1}, {235, lw::ng8});
  b.arr("kg08.ka_mco2", t.kg08.ka_mco2, {1, 1}, {19, lw::ng8});
  b.arr("kg08.kb_mco2", t.kg08.kb_mco2, {1, 1}, {19, lw::ng8});
  b.arr("kg08.ka_mn2o", t.kg08.ka_mn2o, {1, 1}, {19, lw::ng8});
  b.arr("kg08.kb_mn2o", t.kg08.kb_mn2o, {1, 1}, {19, lw::ng8});
  b.arr("kg08.ka_mo3", t.kg08.ka_mo3, {1, 1}, {19, lw::ng8});
  b.arr("kg08.cfc12", t.kg08.cfc12, {1}, {lw::ng8});
  b.arr("kg08.cfc22adj", t.kg08.cfc22adj, {1}, {lw::ng8});
  b.arr("kg08.selfref", t.kg08.selfref, {1, 1}, {10, lw::ng8});
  b.arr("kg08.forref", t.kg08.forref, {1, 1}, {4, lw::ng8});
  b.arr("kg09.fracrefa", t.kg09.fracrefa, {1, 1}, {lw::ng9, 9});
  b.arr("kg09.fracrefb", t.kg09.fracrefb, {1}, {lw::ng9});
  b.arr("kg09.absa", t.kg09.absa, {1, 1}, {585, lw::ng9});
  b.arr("kg09.absb", t.kg09.absb, {1, 1}, {235, lw::ng9});
  b.arr("kg09.ka_mn2o", t.kg09.ka_mn2o, {1, 1, 1}, {9, 19, lw::ng9});
  b.arr("kg09.kb_mn2o", t.kg09.kb_mn2o, {1, 1}, {19, lw::ng9});
  b.arr("kg09.selfref", t.kg09.selfref, {1, 1}, {10, lw::ng9});
  b.arr("kg09.forref", t.kg09.forref, {1, 1}, {4, lw::ng9});
  b.arr("kg10.fracrefa", t.kg10.fracrefa, {1}, {lw::ng10});
  b.arr("kg10.fracrefb", t.kg10.fracrefb, {1}, {lw::ng10});
  b.arr("kg10.absa", t.kg10.absa, {1, 1}, {65, lw::ng10});
  b.arr("kg10.absb", t.kg10.absb, {1, 1}, {235, lw::ng10});
  b.arr("kg10.selfref", t.kg10.selfref, {1, 1}, {10, lw::ng10});
  b.arr("kg10.forref", t.kg10.forref, {1, 1}, {4, lw::ng10});
  b.arr("kg11.fracrefa", t.kg11.fracrefa, {1}, {lw::ng11});
  b.arr("kg11.fracrefb", t.kg11.fracrefb, {1}, {lw::ng11});
  b.arr("kg11.absa", t.kg11.absa, {1, 1}, {65, lw::ng11});
  b.arr("kg11.absb", t.kg11.absb, {1, 1}, {235, lw::ng11});
  b.arr("kg11.ka_mo2", t.kg11.ka_mo2, {1, 1}, {19, lw::ng11});
  b.arr("kg11.kb_mo2", t.kg11.kb_mo2, {1, 1}, {19, lw::ng11});
  b.arr("kg11.selfref", t.kg11.selfref, {1, 1}, {10, lw::ng11});
  b.arr("kg11.forref", t.kg11.forref, {1, 1}, {4, lw::ng11});
  b.arr("kg12.fracrefa", t.kg12.fracrefa, {1, 1}, {lw::ng12, 9});
  b.arr("kg12.absa", t.kg12.absa, {1, 1}, {585, lw::ng12});
  b.arr("kg12.selfref", t.kg12.selfref, {1, 1}, {10, lw::ng12});
  b.arr("kg12.forref", t.kg12.forref, {1, 1}, {4, lw::ng12});
  b.arr("kg13.fracrefa", t.kg13.fracrefa, {1, 1}, {lw::ng13, 9});
  b.arr("kg13.fracrefb", t.kg13.fracrefb, {1}, {lw::ng13});
  b.arr("kg13.absa", t.kg13.absa, {1, 1}, {585, lw::ng13});
  b.arr("kg13.ka_mco2", t.kg13.ka_mco2, {1, 1, 1}, {9, 19, lw::ng13});
  b.arr("kg13.ka_mco", t.kg13.ka_mco, {1, 1, 1}, {9, 19, lw::ng13});
  b.arr("kg13.kb_mo3", t.kg13.kb_mo3, {1, 1}, {19, lw::ng13});
  b.arr("kg13.selfref", t.kg13.selfref, {1, 1}, {10, lw::ng13});
  b.arr("kg13.forref", t.kg13.forref, {1, 1}, {4, lw::ng13});
  b.arr("kg14.fracrefa", t.kg14.fracrefa, {1}, {lw::ng14});
  b.arr("kg14.fracrefb", t.kg14.fracrefb, {1}, {lw::ng14});
  b.arr("kg14.absa", t.kg14.absa, {1, 1}, {65, lw::ng14});
  b.arr("kg14.absb", t.kg14.absb, {1, 1}, {235, lw::ng14});
  b.arr("kg14.selfref", t.kg14.selfref, {1, 1}, {10, lw::ng14});
  b.arr("kg14.forref", t.kg14.forref, {1, 1}, {4, lw::ng14});
  b.arr("kg15.fracrefa", t.kg15.fracrefa, {1, 1}, {lw::ng15, 9});
  b.arr("kg15.absa", t.kg15.absa, {1, 1}, {585, lw::ng15});
  b.arr("kg15.ka_mn2", t.kg15.ka_mn2, {1, 1, 1}, {9, 19, lw::ng15});
  b.arr("kg15.selfref", t.kg15.selfref, {1, 1}, {10, lw::ng15});
  b.arr("kg15.forref", t.kg15.forref, {1, 1}, {4, lw::ng15});
  b.arr("kg16.fracrefa", t.kg16.fracrefa, {1, 1}, {lw::ng16, 9});
  b.arr("kg16.fracrefb", t.kg16.fracrefb, {1}, {lw::ng16});
  b.arr("kg16.absa", t.kg16.absa, {1, 1}, {585, lw::ng16});
  b.arr("kg16.absb", t.kg16.absb, {1, 1}, {235, lw::ng16});
  b.arr("kg16.selfref", t.kg16.selfref, {1, 1}, {10, lw::ng16});
  b.arr("kg16.forref", t.kg16.forref, {1, 1}, {4, lw::ng16});
}

// The arrays each lw_kgbNN reads from RRTMG_LW_DATA, in record order.
struct ODecl {
  const char* name;
  bool is_int;
  std::vector<int> lb, ub;
};
const std::vector<std::pair<int, std::vector<ODecl>>>& lw_records() {
  static const std::vector<std::pair<int, std::vector<ODecl>>> r = {
  {1, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"kao_mn2", false, {1,1}, {19,16}}, {"kbo_mn2", false, {1,1}, {19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {2, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {3, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1,1}, {16,5}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"kao_mn2o", false, {1,1,1}, {9,19,16}}, {"kbo_mn2o", false, {1,1,1}, {5,19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {4, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1,1}, {16,5}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {5, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1,1}, {16,5}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"kao_mo3", false, {1,1,1}, {9,19,16}}, {"ccl4o", false, {1}, {16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {6, {{"fracrefao", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kao_mco2", false, {1,1}, {19,16}}, {"cfc11adjo", false, {1}, {16}}, {"cfc12o", false, {1}, {16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {7, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"kao_mco2", false, {1,1,1}, {9,19,16}}, {"kbo_mco2", false, {1,1}, {19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {8, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"kao_mco2", false, {1,1}, {19,16}}, {"kbo_mco2", false, {1,1}, {19,16}}, {"kao_mn2o", false, {1,1}, {19,16}}, {"kbo_mn2o", false, {1,1}, {19,16}}, {"kao_mo3", false, {1,1}, {19,16}}, {"cfc12o", false, {1}, {16}}, {"cfc22adjo", false, {1}, {16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {9, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"kao_mn2o", false, {1,1,1}, {9,19,16}}, {"kbo_mn2o", false, {1,1}, {19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {10, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {11, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"kao_mo2", false, {1,1}, {19,16}}, {"kbo_mo2", false, {1,1}, {19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {12, {{"fracrefao", false, {1,1}, {16,9}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {13, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kao_mco2", false, {1,1,1}, {9,19,16}}, {"kao_mco", false, {1,1,1}, {9,19,16}}, {"kbo_mo3", false, {1,1}, {19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {14, {{"fracrefao", false, {1}, {16}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {15, {{"fracrefao", false, {1,1}, {16,9}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kao_mn2", false, {1,1,1}, {9,19,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  {16, {{"fracrefao", false, {1,1}, {16,9}}, {"fracrefbo", false, {1}, {16}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}}},
  };
  return r;
}

// Host views of the arrays the generated data routines assign.
struct LwDataH {
  HView<Real> pref, preflog, tref, chi_mls, totplk16, totplnk, absice0, absice1, absice2, absice3, absliq1, wt;
  HView<int> ngc, ngs, ngm, ngn, ngb;
  Real& abscld1;
  Real& absliq0;
};

// cmbgbNN for one array: sums the 16 g-points of the data file into the
// ngc(band) reduced ones (weighted by rwgt, except the fracref* arrays).
void combine(const HArr<Real>& src, const HView<Real>& dst, bool g_first, bool weighted, int band,
             const HView<int>& ngc, const HView<int>& ngn, const HView<int>& ngs, const HView<Real>& rwgt) {
  const size_t nsrc = src.size();
  const int ngo = g_first ? src.extent(0) : src.extent(int(src.lb.size()) - 1);
  const size_t nlead = nsrc / size_t(ngo);
  const int nr = ngc(band);
  if (dst.size() != nlead * size_t(nr)) throw std::runtime_error("cm1k rrtmg cmbgb: size mismatch");
  const int ngs0 = band == 1 ? 0 : ngs(band - 1);
  for (size_t lead = 0; lead < nlead; ++lead) {
    int iprsm = 0;
    for (int igc = 1; igc <= nr; ++igc) {
      Real sumk = 0.;
      for (int ipr = 1; ipr <= ngn(ngs0 + igc); ++ipr) {
        iprsm = iprsm + 1;
        const Real v = g_first ? src.p[size_t(iprsm - 1) + size_t(ngo) * lead] : src.p[lead + nlead * size_t(iprsm - 1)];
        if (weighted) sumk = sumk + v * rwgt(iprsm + 16 * (band - 1));
        else sumk = sumk + v;
      }
      if (g_first) dst.p[size_t(igc - 1) + size_t(nr) * lead] = sumk;
      else dst.p[lead + nlead * size_t(igc - 1)] = sumk;
    }
  }
}

std::string combined_name(const std::string& nm) {
  if (nm == "kao") return "absa";
  if (nm == "kbo") return "absb";
  if (nm.rfind("kao_", 0) == 0) return "ka_" + nm.substr(4);
  if (nm.rfind("kbo_", 0) == 0) return "kb_" + nm.substr(4);
  if (!nm.empty() && nm.back() == 'o') return nm.substr(0, nm.size() - 1);
  throw std::runtime_error("cm1k rrtmg: unexpected table " + nm);
}

std::shared_ptr<LwTableData> build(Real cpdair, Real grv, const std::string& data_file) {
  using namespace lw;
  auto d = std::make_shared<LwTableData>();
  LwTables& t = d->host;
  TabBinder cnt;
  layout(t, cnt);
  d->hr.assign(size_t(cnt.nr), Real(0));
  d->hi.assign(size_t(cnt.ni), 0);
  TabBinder b;
  b.rbase = d->hr.data();
  b.ibase = d->hi.data();
  layout(t, b);

  // rrtmg_lwlookuptable
  FortranRecordReader rd(data_file);
  std::map<std::string, HArr<Real>> orig;
  for (auto& [band, decl] : lw_records()) {
    rd.next();
    for (auto& o : decl) {
      if (o.is_int) throw std::runtime_error("cm1k rrtmg: unexpected int in RRTMG_LW_DATA");
      char key[64];
      std::snprintf(key, sizeof key, "kg%02d.%s", band, o.name);
      HArr<Real> a(o.lb, o.ub);
      rd.into(a);
      orig.emplace(key, std::move(a));
    }
    rd.end_record();
  }

  // rrtmg_lw_ini: lwdatinit
  fset(b.rview("wavenum1"), {S()}, {10., 350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800.,
                                    2080., 2250., 2380., 2600.});
  fset(b.rview("wavenum2"), {S()}, {350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800., 2080.,
                                    2250., 2380., 2600., 3250.});
  fset(b.rview("delwave"), {S()}, {340., 150., 130., 70., 120., 160., 100., 100., 210., 90., 320., 280., 170., 130.,
                                   220., 650.});
  fset(b.iview("ng"), {S()}, {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16});
  fset(b.iview("nspa"), {S()}, {1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9});
  fset(b.iview("nspb"), {S()}, {1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0});
  t.nxmol = 4;
  {
    auto ixindx = b.iview("ixindx");
    ixindx(1) = 1; ixindx(2) = 2; ixindx(3) = 3; ixindx(4) = 4;
    for (int i = 5; i <= maxinpx; ++i) ixindx(i) = 0;
  }
  t.grav = grv;
  t.planck = 6.62606876e-27;
  t.boltz = 1.3806503e-16;
  t.clight = 2.99792458e+10;
  t.avogad = 6.02214199e+23;
  t.alosmt = 2.6867775e+19;
  t.gascon = 8.31447200e+07;
  t.radcn1 = 1.191042722e-12;
  t.radcn2 = 1.4387752;
  t.sbcnst = 5.670400e-04;
  t.secdy = 8.6400e4;
  t.oneminus = 1. - 1.e-6;
  t.pi = 2. * std::asin(1.);
  t.fluxfac = t.pi * 2.e4;
  t.heatfac = t.grav * t.secdy / (cpdair * 1.e2);

  // lwcmbdat, lwcldpr, lwatmref, lwavplank
  LwDataH h{b.rview("pref"),    b.rview("preflog"), b.rview("tref"),    b.rview("chi_mls"), b.rview("totplk16"),
            b.rview("totplnk"), b.rview("absice0"), b.rview("absice1"), b.rview("absice2"), b.rview("absice3"),
            b.rview("absliq1"), b.rview("wt"),      b.iview("ngc"),     b.iview("ngs"),     b.iview("ngm"),
            b.iview("ngn"),     b.iview("ngb"),     t.abscld1,          t.absliq0};
  lwcmbdat_data(h);
  lwcldpr_data(h);
  lwatmref_data(h);
  lwavplank_data(h);

  // rrtmg_lw_rtrnmc module data
  t.wtdiff = 0.5;
  t.rec_6 = 0.166667;
  fset(b.rview("a0"), {S()}, {1.66, 1.55, 1.58, 1.66, 1.54, 1.454, 1.89, 1.33, 1.668, 1.66, 1.66, 1.66, 1.66, 1.66,
                              1.66, 1.66});
  fset(b.rview("a1"), {S()}, {0.00, 0.25, 0.22, 0.00, 0.13, 0.446, -0.10, 0.40, -0.006, 0.00, 0.00, 0.00, 0.00,
                              0.00, 0.00, 0.00});
  fset(b.rview("a2"), {S()}, {0.00, -12.0, -11.7, 0.00, -0.72, -0.243, 0.19, -0.062, 0.414, 0.00, 0.00, 0.00, 0.00,
                              0.00, 0.00, 0.00});

  // transmittance lookup tables
  {
    const Real expeps = 1.e-20;
    auto tau_tbl = b.rview("tau_tbl"), exp_tbl = b.rview("exp_tbl"), tfn_tbl = b.rview("tfn_tbl");
    tau_tbl(0) = 0.0;
    tau_tbl(ntbl) = 1.e10;
    exp_tbl(0) = 1.0;
    exp_tbl(ntbl) = expeps;
    tfn_tbl(0) = 0.0;
    tfn_tbl(ntbl) = 1.0;
    t.bpade = 1.0 / pade;
    for (int itr = 1; itr <= ntbl - 1; ++itr) {
      const Real tfn = Real(itr) / Real(ntbl);
      tau_tbl(itr) = t.bpade * tfn / (1. - tfn);
      exp_tbl(itr) = std::exp(-tau_tbl(itr));
      if (exp_tbl(itr) <= expeps) exp_tbl(itr) = expeps;
      if (tau_tbl(itr) < 0.06) tfn_tbl(itr) = tau_tbl(itr) / 6.;
      else tfn_tbl(itr) = 1. - 2. * ((1. / tau_tbl(itr)) - (exp_tbl(itr) / (1. - exp_tbl(itr))));
    }
  }

  // g-point weights
  auto ngc = b.iview("ngc"), ngn = b.iview("ngn"), ngs = b.iview("ngs"), ngm = b.iview("ngm"), ng = b.iview("ng");
  auto wt = b.rview("wt"), rwgt = b.rview("rwgt");
  {
    Real wtsm[mg + 1];
    int igcsm = 0;
    for (int ibnd = 1; ibnd <= nbndlw; ++ibnd) {
      int iprsm = 0;
      if (ngc(ibnd) < mg) {
        for (int igc = 1; igc <= ngc(ibnd); ++igc) {
          igcsm = igcsm + 1;
          Real wtsum = 0.;
          for (int ipr = 1; ipr <= ngn(igcsm); ++ipr) {
            iprsm = iprsm + 1;
            wtsum = wtsum + wt(iprsm);
          }
          wtsm[igc] = wtsum;
        }
        for (int ig = 1; ig <= ng(ibnd); ++ig) {
          const int ind = (ibnd - 1) * mg + ig;
          rwgt(ind) = wt(ig) / wtsm[ngm(ind)];
        }
      } else {
        for (int ig = 1; ig <= ng(ibnd); ++ig) {
          igcsm = igcsm + 1;
          const int ind = (ibnd - 1) * mg + ig;
          rwgt(ind) = 1.0;
        }
      }
    }
  }

  // cmbgb1..16
  for (auto& [band, decl] : lw_records())
    for (auto& o : decl) {
      char key[64], ckey[64];
      std::snprintf(key, sizeof key, "kg%02d.%s", band, o.name);
      const std::string nm = o.name;
      std::snprintf(ckey, sizeof ckey, "kg%02d.%s", band, combined_name(nm).c_str());
      const bool frac = nm.rfind("fracref", 0) == 0;
      combine(orig.at(key), b.rview(ckey), frac && o.lb.size() == 2, !frac, band, ngc, ngn, ngs, rwgt);
    }

  // registry (host) and the device copy
  for (auto& r : b.regs) {
    if (r.is_int) d->int_arrays.push_back({r.name, {d->hi.data() + r.off, size_t(r.n)}});
    else d->real_arrays.push_back({r.name, {d->hr.data() + r.off, size_t(r.n)}});
  }
  d->dr = Kokkos::View<Real*, Device>("rrtmg_lw_tables_r", size_t(cnt.nr));
  d->di = Kokkos::View<int*, Device>("rrtmg_lw_tables_i", size_t(cnt.ni));
  Kokkos::deep_copy(d->dr, Kokkos::View<const Real*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                               d->hr.data(), d->hr.size()));
  Kokkos::deep_copy(d->di, Kokkos::View<const int*, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                               d->hi.data(), d->hi.size()));
  d->dev = t;
  TabBinder bd;
  bd.rbase = d->dr.data();
  bd.ibase = d->di.data();
  layout(d->dev, bd);
  return d;
}

} // namespace

std::shared_ptr<const LwTableData> lw_tables(Real cpdair, Real grv, const std::string& data_file) {
  static std::mutex mtx;
  static std::map<std::tuple<Real, Real, std::string>, std::shared_ptr<const LwTableData>> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto key = std::make_tuple(cpdair, grv, data_file);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  // the device Views must go before Kokkos::finalize
  if (cache.empty()) Kokkos::push_finalize_hook([] { std::lock_guard<std::mutex> l(mtx); cache.clear(); });
  auto d = build(cpdair, grv, data_file);
  cache.emplace(key, d);
  return d;
}

} // namespace cm1k::rrtmg
