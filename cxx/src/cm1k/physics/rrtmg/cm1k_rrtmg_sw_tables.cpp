// RRTMG_SW tables: rrtmg_swlookuptable + rrtmg_sw_ini (module_ra_rrtmg_sw.F).
#include "physics/rrtmg/cm1k_rrtmg_sw_tables.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace cm1k::rrtmg {

namespace {

#include "physics/rrtmg/cm1k_rrtmg_sw_data.inc"

template <class B>
void layout(SwTables& t, B& b) {
  using namespace sw;
  b.arr("rsrtaua", t.rsrtaua, {1, 1}, {nbndsw, naerec});
  b.arr("rsrpiza", t.rsrpiza, {1, 1}, {nbndsw, naerec});
  b.arr("rsrasya", t.rsrasya, {1, 1}, {nbndsw, naerec});
  b.arr("extliq1", t.extliq1, {1, 16}, {58, 29});
  b.arr("ssaliq1", t.ssaliq1, {1, 16}, {58, 29});
  b.arr("asyliq1", t.asyliq1, {1, 16}, {58, 29});
  b.arr("extice2", t.extice2, {1, 16}, {43, 29});
  b.arr("ssaice2", t.ssaice2, {1, 16}, {43, 29});
  b.arr("asyice2", t.asyice2, {1, 16}, {43, 29});
  b.arr("extice3", t.extice3, {1, 16}, {46, 29});
  b.arr("ssaice3", t.ssaice3, {1, 16}, {46, 29});
  b.arr("asyice3", t.asyice3, {1, 16}, {46, 29});
  b.arr("fdlice3", t.fdlice3, {1, 16}, {46, 29});
  b.arr("abari", t.abari, {1}, {5});
  b.arr("bbari", t.bbari, {1}, {5});
  b.arr("cbari", t.cbari, {1}, {5});
  b.arr("dbari", t.dbari, {1}, {5});
  b.arr("ebari", t.ebari, {1}, {5});
  b.arr("fbari", t.fbari, {1}, {5});
  b.arr("pref", t.pref, {1}, {59});
  b.arr("preflog", t.preflog, {1}, {59});
  b.arr("tref", t.tref, {1}, {59});
  b.arr("exp_tbl", t.exp_tbl, {0}, {ntbl});
  b.arr("ng", t.ng, {jpb1}, {jpb2});
  b.arr("nspa", t.nspa, {jpb1}, {jpb2});
  b.arr("nspb", t.nspb, {jpb1}, {jpb2});
  b.arr("wavenum1", t.wavenum1, {jpb1}, {jpb2});
  b.arr("wavenum2", t.wavenum2, {jpb1}, {jpb2});
  b.arr("delwave", t.delwave, {jpb1}, {jpb2});
  b.arr("ngc", t.ngc, {1}, {nbndsw});
  b.arr("ngs", t.ngs, {1}, {nbndsw});
  b.arr("ngn", t.ngn, {1}, {ngptsw});
  b.arr("ngb", t.ngb, {1}, {ngptsw});
  b.arr("ngm", t.ngm, {1}, {nbndsw * mg});
  b.arr("wt", t.wt, {1}, {mg});
  b.arr("rwgt", t.rwgt, {1}, {nbndsw * mg});
  b.arr("rad.retab", t.retab, {1}, {95});
  b.arr("rad.O3SUM", t.O3SUM, {1}, {31});
  b.arr("rad.PPSUM", t.PPSUM, {1}, {31});
  b.arr("rad.O3WIN", t.O3WIN, {1}, {31});
  b.arr("rad.PPWIN", t.PPWIN, {1}, {31});
  b.arr("kg16.absa", t.kg16.absa, {1, 1}, {585, sw::ng16});
  b.arr("kg16.absb", t.kg16.absb, {1, 1}, {235, sw::ng16});
  b.arr("kg16.selfref", t.kg16.selfref, {1, 1}, {10, sw::ng16});
  b.arr("kg16.forref", t.kg16.forref, {1, 1}, {3, sw::ng16});
  b.arr("kg16.sfluxref", t.kg16.sfluxref, {1}, {sw::ng16});
  b.arr("kg17.absa", t.kg17.absa, {1, 1}, {585, sw::ng17});
  b.arr("kg17.absb", t.kg17.absb, {1, 1}, {1175, sw::ng17});
  b.arr("kg17.selfref", t.kg17.selfref, {1, 1}, {10, sw::ng17});
  b.arr("kg17.forref", t.kg17.forref, {1, 1}, {4, sw::ng17});
  b.arr("kg17.sfluxref", t.kg17.sfluxref, {1, 1}, {sw::ng17, 5});
  b.arr("kg18.absa", t.kg18.absa, {1, 1}, {585, sw::ng18});
  b.arr("kg18.absb", t.kg18.absb, {1, 1}, {235, sw::ng18});
  b.arr("kg18.selfref", t.kg18.selfref, {1, 1}, {10, sw::ng18});
  b.arr("kg18.forref", t.kg18.forref, {1, 1}, {3, sw::ng18});
  b.arr("kg18.sfluxref", t.kg18.sfluxref, {1, 1}, {sw::ng18, 9});
  b.arr("kg19.absa", t.kg19.absa, {1, 1}, {585, sw::ng19});
  b.arr("kg19.absb", t.kg19.absb, {1, 1}, {235, sw::ng19});
  b.arr("kg19.selfref", t.kg19.selfref, {1, 1}, {10, sw::ng19});
  b.arr("kg19.forref", t.kg19.forref, {1, 1}, {3, sw::ng19});
  b.arr("kg19.sfluxref", t.kg19.sfluxref, {1, 1}, {sw::ng19, 9});
  b.arr("kg20.absch4", t.kg20.absch4, {1}, {sw::ng20});
  b.arr("kg20.absa", t.kg20.absa, {1, 1}, {65, sw::ng20});
  b.arr("kg20.absb", t.kg20.absb, {1, 1}, {235, sw::ng20});
  b.arr("kg20.selfref", t.kg20.selfref, {1, 1}, {10, sw::ng20});
  b.arr("kg20.forref", t.kg20.forref, {1, 1}, {4, sw::ng20});
  b.arr("kg20.sfluxref", t.kg20.sfluxref, {1}, {sw::ng20});
  b.arr("kg21.absa", t.kg21.absa, {1, 1}, {585, sw::ng21});
  b.arr("kg21.absb", t.kg21.absb, {1, 1}, {1175, sw::ng21});
  b.arr("kg21.selfref", t.kg21.selfref, {1, 1}, {10, sw::ng21});
  b.arr("kg21.forref", t.kg21.forref, {1, 1}, {4, sw::ng21});
  b.arr("kg21.sfluxref", t.kg21.sfluxref, {1, 1}, {sw::ng21, 9});
  b.arr("kg22.absa", t.kg22.absa, {1, 1}, {585, sw::ng22});
  b.arr("kg22.absb", t.kg22.absb, {1, 1}, {235, sw::ng22});
  b.arr("kg22.selfref", t.kg22.selfref, {1, 1}, {10, sw::ng22});
  b.arr("kg22.forref", t.kg22.forref, {1, 1}, {3, sw::ng22});
  b.arr("kg22.sfluxref", t.kg22.sfluxref, {1, 1}, {sw::ng22, 9});
  b.arr("kg23.rayl", t.kg23.rayl, {1}, {sw::ng23});
  b.arr("kg23.absa", t.kg23.absa, {1, 1}, {65, sw::ng23});
  b.arr("kg23.selfref", t.kg23.selfref, {1, 1}, {10, sw::ng23});
  b.arr("kg23.forref", t.kg23.forref, {1, 1}, {3, sw::ng23});
  b.arr("kg23.sfluxref", t.kg23.sfluxref, {1}, {sw::ng23});
  b.arr("kg24.rayla", t.kg24.rayla, {1, 1}, {sw::ng24, 9});
  b.arr("kg24.raylb", t.kg24.raylb, {1}, {sw::ng24});
  b.arr("kg24.abso3a", t.kg24.abso3a, {1}, {sw::ng24});
  b.arr("kg24.abso3b", t.kg24.abso3b, {1}, {sw::ng24});
  b.arr("kg24.absa", t.kg24.absa, {1, 1}, {585, sw::ng24});
  b.arr("kg24.absb", t.kg24.absb, {1, 1}, {235, sw::ng24});
  b.arr("kg24.selfref", t.kg24.selfref, {1, 1}, {10, sw::ng24});
  b.arr("kg24.forref", t.kg24.forref, {1, 1}, {3, sw::ng24});
  b.arr("kg24.sfluxref", t.kg24.sfluxref, {1, 1}, {sw::ng24, 9});
  b.arr("kg25.rayl", t.kg25.rayl, {1}, {sw::ng25});
  b.arr("kg25.abso3a", t.kg25.abso3a, {1}, {sw::ng25});
  b.arr("kg25.abso3b", t.kg25.abso3b, {1}, {sw::ng25});
  b.arr("kg25.absa", t.kg25.absa, {1, 1}, {65, sw::ng25});
  b.arr("kg25.sfluxref", t.kg25.sfluxref, {1}, {sw::ng25});
  b.arr("kg26.rayl", t.kg26.rayl, {1}, {sw::ng26});
  b.arr("kg26.sfluxref", t.kg26.sfluxref, {1}, {sw::ng26});
  b.arr("kg27.rayl", t.kg27.rayl, {1}, {sw::ng27});
  b.arr("kg27.absa", t.kg27.absa, {1, 1}, {65, sw::ng27});
  b.arr("kg27.absb", t.kg27.absb, {1, 1}, {235, sw::ng27});
  b.arr("kg27.sfluxref", t.kg27.sfluxref, {1}, {sw::ng27});
  b.arr("kg28.absa", t.kg28.absa, {1, 1}, {585, sw::ng28});
  b.arr("kg28.absb", t.kg28.absb, {1, 1}, {1175, sw::ng28});
  b.arr("kg28.sfluxref", t.kg28.sfluxref, {1, 1}, {sw::ng28, 5});
  b.arr("kg29.absh2o", t.kg29.absh2o, {1}, {sw::ng29});
  b.arr("kg29.absco2", t.kg29.absco2, {1}, {sw::ng29});
  b.arr("kg29.absa", t.kg29.absa, {1, 1}, {65, sw::ng29});
  b.arr("kg29.absb", t.kg29.absb, {1, 1}, {235, sw::ng29});
  b.arr("kg29.selfref", t.kg29.selfref, {1, 1}, {10, sw::ng29});
  b.arr("kg29.forref", t.kg29.forref, {1, 1}, {4, sw::ng29});
  b.arr("kg29.sfluxref", t.kg29.sfluxref, {1}, {sw::ng29});
}

struct ODecl {
  const char* name;
  bool is_int;
  std::vector<int> lb, ub;
};
const std::vector<std::pair<int, std::vector<ODecl>>>& sw_records() {
  static const std::vector<std::pair<int, std::vector<ODecl>>> r = {
  {16, {{"rayl", false, {}, {}}, {"strrat1", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1}, {16}}}},
  {17, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}, {"sfluxrefo", false, {1,1}, {16,5}}}},
  {18, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1,1}, {16,9}}}},
  {19, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1,1}, {16,9}}}},
  {20, {{"rayl", false, {}, {}}, {"layreffr", true, {}, {}}, {"absch4o", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}, {"sfluxrefo", false, {1}, {16}}}},
  {21, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}, {"sfluxrefo", false, {1,1}, {16,9}}}},
  {22, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1,1}, {16,9}}}},
  {23, {{"raylo", false, {1}, {16}}, {"givfac", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1}, {5,13,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1}, {16}}}},
  {24, {{"raylao", false, {1,1}, {16,9}}, {"raylbo", false, {1}, {16}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"abso3ao", false, {1}, {16}}, {"abso3bo", false, {1}, {16}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {3,16}}, {"sfluxrefo", false, {1,1}, {16,9}}}},
  {25, {{"raylo", false, {1}, {16}}, {"layreffr", true, {}, {}}, {"abso3ao", false, {1}, {16}}, {"abso3bo", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"sfluxrefo", false, {1}, {16}}}},
  {26, {{"raylo", false, {1}, {16}}, {"sfluxrefo", false, {1}, {16}}}},
  {27, {{"raylo", false, {1}, {16}}, {"scalekur", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"sfluxrefo", false, {1}, {16}}}},
  {28, {{"rayl", false, {}, {}}, {"strrat", false, {}, {}}, {"layreffr", true, {}, {}}, {"kao", false, {1,1,1,1}, {9,5,13,16}}, {"kbo", false, {1,1,13,1}, {5,5,59,16}}, {"sfluxrefo", false, {1,1}, {16,5}}}},
  {29, {{"rayl", false, {}, {}}, {"layreffr", true, {}, {}}, {"absh2oo", false, {1}, {16}}, {"absco2o", false, {1}, {16}}, {"kao", false, {1,1,1}, {5,13,16}}, {"kbo", false, {1,13,1}, {5,59,16}}, {"selfrefo", false, {1,1}, {10,16}}, {"forrefo", false, {1,1}, {4,16}}, {"sfluxrefo", false, {1}, {16}}}},
  };
  return r;
}

struct SwDataH {
  HView<Real> pref, preflog, tref, rsrtaua, rsrpiza, rsrasya, extliq1, ssaliq1, asyliq1, extice2, ssaice2, asyice2,
      extice3, ssaice3, asyice3, fdlice3, abari, bbari, cbari, dbari, ebari, fbari, wt;
  HView<int> ngc, ngs, ngm, ngn, ngb;
};

// cmbgbNN for one array (ib = band - 15): sums the 16 g-points into ngc(ib)
// (weighted by rwgt except sfluxref).
void combine(const HArr<Real>& src, const HView<Real>& dst, bool g_first, bool weighted, int ib,
             const HView<int>& ngc, const HView<int>& ngn, const HView<int>& ngs, const HView<Real>& rwgt) {
  const size_t nsrc = src.size();
  const int ngo = g_first ? src.extent(0) : src.extent(int(src.lb.size()) - 1);
  const size_t nlead = nsrc / size_t(ngo);
  const int nr = ngc(ib);
  if (dst.size() != nlead * size_t(nr)) throw std::runtime_error("cm1k rrtmg sw cmbgb: size mismatch");
  const int ngs0 = ib == 1 ? 0 : ngs(ib - 1);
  for (size_t lead = 0; lead < nlead; ++lead) {
    int iprsm = 0;
    for (int igc = 1; igc <= nr; ++igc) {
      Real sumk = 0.;
      for (int ipr = 1; ipr <= ngn(ngs0 + igc); ++ipr) {
        iprsm = iprsm + 1;
        const Real v = g_first ? src.p[size_t(iprsm - 1) + size_t(ngo) * lead] : src.p[lead + nlead * size_t(iprsm - 1)];
        if (weighted) sumk = sumk + v * rwgt(iprsm + 16 * (ib - 1));
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
  if (!nm.empty() && nm.back() == 'o') return nm.substr(0, nm.size() - 1);
  throw std::runtime_error("cm1k rrtmg: unexpected sw table " + nm);
}

// The band scalars read from the file, by name.
Real* sw_real_scalar(SwTables& t, int band, const std::string& n) {
  switch (band) {
  case 16: if (n == "rayl") return &t.kg16.rayl; if (n == "strrat1") return &t.kg16.strrat1; break;
  case 17: if (n == "rayl") return &t.kg17.rayl; if (n == "strrat") return &t.kg17.strrat; break;
  case 18: if (n == "rayl") return &t.kg18.rayl; if (n == "strrat") return &t.kg18.strrat; break;
  case 19: if (n == "rayl") return &t.kg19.rayl; if (n == "strrat") return &t.kg19.strrat; break;
  case 20: if (n == "rayl") return &t.kg20.rayl; break;
  case 21: if (n == "rayl") return &t.kg21.rayl; if (n == "strrat") return &t.kg21.strrat; break;
  case 22: if (n == "rayl") return &t.kg22.rayl; if (n == "strrat") return &t.kg22.strrat; break;
  case 23: if (n == "givfac") return &t.kg23.givfac; break;
  case 24: if (n == "strrat") return &t.kg24.strrat; break;
  case 27: if (n == "scalekur") return &t.kg27.scalekur; break;
  case 28: if (n == "rayl") return &t.kg28.rayl; if (n == "strrat") return &t.kg28.strrat; break;
  case 29: if (n == "rayl") return &t.kg29.rayl; break;
  }
  throw std::runtime_error("cm1k rrtmg: unexpected sw scalar " + n);
}
int* sw_layreffr(SwTables& t, int band) {
  switch (band) {
  case 16: return &t.kg16.layreffr; case 17: return &t.kg17.layreffr; case 18: return &t.kg18.layreffr;
  case 19: return &t.kg19.layreffr; case 20: return &t.kg20.layreffr; case 21: return &t.kg21.layreffr;
  case 22: return &t.kg22.layreffr; case 23: return &t.kg23.layreffr; case 24: return &t.kg24.layreffr;
  case 25: return &t.kg25.layreffr; case 27: return &t.kg27.layreffr; case 28: return &t.kg28.layreffr;
  case 29: return &t.kg29.layreffr;
  }
  throw std::runtime_error("cm1k rrtmg: no layreffr in sw band " + std::to_string(band));
}

std::shared_ptr<SwTableData> build(Real cpdair, Real grv, const std::string& data_file) {
  using namespace sw;
  auto d = std::make_shared<SwTableData>();
  SwTables& t = d->host;
  TabBinder cnt;
  layout(t, cnt);
  d->hr.assign(size_t(cnt.nr), Real(0));
  d->hi.assign(size_t(cnt.ni), 0);
  TabBinder b;
  b.rbase = d->hr.data();
  b.ibase = d->hi.data();
  layout(t, b);

  // rrtmg_swlookuptable
  FortranRecordReader rd(data_file);
  std::map<std::string, HArr<Real>> orig;
  for (auto& [band, decl] : sw_records()) {
    rd.next();
    for (auto& o : decl) {
      char key[64];
      std::snprintf(key, sizeof key, "kg%02d.%s", band, o.name);
      if (o.lb.empty()) {
        if (o.is_int) {
          *sw_layreffr(t, band) = rd.int1();
          d->scalars.push_back({key, double(*sw_layreffr(t, band))});
        } else {
          Real* p = sw_real_scalar(t, band, o.name);
          *p = rd.real1();
          d->scalars.push_back({key, *p});
        }
        continue;
      }
      HArr<Real> a(o.lb, o.ub);
      rd.into(a);
      orig.emplace(key, std::move(a));
    }
    rd.end_record();
  }

  // rrtmg_sw_ini: swdatinit
  fset(b.rview("wavenum1"), {S()}, {2600., 3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000., 22650.,
                                    29000., 38000., 820.});
  fset(b.rview("wavenum2"), {S()}, {3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000., 22650., 29000.,
                                    38000., 50000., 2600.});
  fset(b.rview("delwave"), {S()}, {650., 750., 650., 500., 1000., 1550., 350., 4800., 3150., 6650., 6350., 9000.,
                                   12000., 1780.});
  fset(b.iview("ng"), {S()}, {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16});
  fset(b.iview("nspa"), {S()}, {9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1});
  fset(b.iview("nspb"), {S()}, {1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1});
  t.grav = grv;
  t.planck = 6.62606876e-27;
  t.boltz = 1.3806503e-16;
  t.clight = 2.99792458e+10;
  t.avogad = 6.02214199e+23;
  t.alosmt = 2.6867775e+19;
  t.gascon = 8.31447200e+07;
  t.radcn1 = 1.191042772e-12;
  t.radcn2 = 1.4387752;
  t.sbcnst = 5.670400e-04;
  t.secdy = 8.6400e4;
  t.oneminus = 1.0 - 1.e-06;
  t.pi = 2. * std::asin(1.);
  t.heatfac = t.grav * t.secdy / (cpdair * 1.e2);
  t.fluxfac = 0;   // not set by swdatinit

  // swcmbdat, swaerpr, swcldpr, swatmref
  SwDataH h{b.rview("pref"),    b.rview("preflog"), b.rview("tref"),    b.rview("rsrtaua"), b.rview("rsrpiza"),
            b.rview("rsrasya"), b.rview("extliq1"), b.rview("ssaliq1"), b.rview("asyliq1"), b.rview("extice2"),
            b.rview("ssaice2"), b.rview("asyice2"), b.rview("extice3"), b.rview("ssaice3"), b.rview("asyice3"),
            b.rview("fdlice3"), b.rview("abari"),   b.rview("bbari"),   b.rview("cbari"),   b.rview("dbari"),
            b.rview("ebari"),   b.rview("fbari"),   b.rview("wt"),      b.iview("ngc"),     b.iview("ngs"),
            b.iview("ngm"),     b.iview("ngn"),     b.iview("ngb")};
  swcmbdat_data(h);
  swaerpr_data(h);
  swcldpr_data(h);
  swatmref_data(h);

  // retab and O3DATA's data (module_ra_rrtmg_lw)
  fset(b.rview("rad.retab"), {S()}, {5.92779, 6.26422, 6.61973, 6.99539, 7.39234, 7.81177, 8.25496, 8.72323, 9.21800, 9.74075, 10.2930, 10.8765, 11.4929, 12.1440, 12.8317, 13.5581, 14.2319, 15.0351, 15.8799, 16.7674, 17.6986, 18.6744, 19.6955, 20.7623, 21.8757, 23.0364, 24.2452, 25.5034, 26.8125, 27.7895, 28.6450, 29.4167, 30.1088, 30.7306, 31.2943, 31.8151, 32.3077, 32.7870, 33.2657, 33.7540, 34.2601, 34.7892, 35.3442, 35.9255, 36.5316, 37.1602, 37.8078, 38.4720, 39.1508, 39.8442, 40.5552, 41.2912, 42.0635, 42.8876, 43.7863, 44.7853, 45.9170, 47.2165, 48.7221, 50.4710, 52.4980, 54.8315, 57.4898, 60.4785, 63.7898, 65.5604, 71.2885, 75.4113, 79.7368, 84.2351, 88.8833, 93.6658, 98.5739, 103.603, 108.752, 114.025, 119.424, 124.954, 130.630, 136.457, 142.446, 148.608, 154.956, 161.503, 168.262, 175.248, 182.473, 189.952, 197.699, 205.728, 214.055, 222.694, 231.661, 240.971, 250.639});
  fset(b.rview("rad.O3SUM"), {S()}, {5.297e-8, 5.852e-8, 6.579e-8, 7.505e-8, 8.577e-8, 9.895e-8, 1.175e-7, 1.399e-7, 1.677e-7, 2.003e-7, 2.571e-7, 3.325e-7, 4.438e-7, 6.255e-7, 8.168e-7, 1.036e-6, 1.366e-6, 1.855e-6, 2.514e-6, 3.240e-6, 4.033e-6, 4.854e-6, 5.517e-6, 6.089e-6, 6.689e-6, 1.106e-5, 1.462e-5, 1.321e-5, 9.856e-6, 5.960e-6, 5.960e-6});
  fset(b.rview("rad.PPSUM"), {S()}, {955.890, 850.532, 754.599, 667.742, 589.841, 519.421, 455.480, 398.085, 347.171, 301.735, 261.310, 225.360, 193.419, 165.490, 141.032, 120.125, 102.689, 87.829, 75.123, 64.306, 55.086, 47.209, 40.535, 34.795, 29.865, 19.122, 9.277, 4.660, 2.421, 1.294, 0.647});
  fset(b.rview("rad.O3WIN"), {S()}, {4.629e-8, 4.686e-8, 5.017e-8, 5.613e-8, 6.871e-8, 8.751e-8, 1.138e-7, 1.516e-7, 2.161e-7, 3.264e-7, 4.968e-7, 7.338e-7, 1.017e-6, 1.308e-6, 1.625e-6, 2.011e-6, 2.516e-6, 3.130e-6, 3.840e-6, 4.703e-6, 5.486e-6, 6.289e-6, 6.993e-6, 7.494e-6, 8.197e-6, 9.632e-6, 1.113e-5, 1.146e-5, 9.389e-6, 6.135e-6, 6.135e-6});
  fset(b.rview("rad.PPWIN"), {S()}, {955.747, 841.783, 740.199, 649.538, 568.404, 495.815, 431.069, 373.464, 322.354, 277.190, 237.635, 203.433, 174.070, 148.949, 127.408, 108.915, 93.114, 79.551, 67.940, 58.072, 49.593, 42.318, 36.138, 30.907, 26.362, 16.423, 7.583, 3.620, 1.807, 0.938, 0.469});

  // exponential lookup table
  {
    const Real expeps = 1.e-20;
    auto exp_tbl = b.rview("exp_tbl");
    exp_tbl(0) = 1.0;
    exp_tbl(ntbl) = expeps;
    t.bpade = 1.0 / pade;
    for (int itr = 1; itr <= ntbl - 1; ++itr) {
      const Real tfn = Real(itr) / Real(ntbl);
      t.tau_tbl = t.bpade * tfn / (1. - tfn);
      exp_tbl(itr) = std::exp(-t.tau_tbl);
      if (exp_tbl(itr) <= expeps) exp_tbl(itr) = expeps;
    }
  }

  auto ngc = b.iview("ngc"), ngn = b.iview("ngn"), ngs = b.iview("ngs"), ngm = b.iview("ngm"), ng = b.iview("ng");
  auto wt = b.rview("wt"), rwgt = b.rview("rwgt");
  {
    Real wtsm[mg + 1];
    int igcsm = 0;
    for (int ibnd = 1; ibnd <= nbndsw; ++ibnd) {
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
        for (int ig = 1; ig <= ng(ibnd + 15); ++ig) {
          const int ind = (ibnd - 1) * mg + ig;
          rwgt(ind) = wt(ig) / wtsm[ngm(ind)];
        }
      } else {
        for (int ig = 1; ig <= ng(ibnd + 15); ++ig) {
          igcsm = igcsm + 1;
          const int ind = (ibnd - 1) * mg + ig;
          rwgt(ind) = 1.0;
        }
      }
    }
  }

  // cmbgb16s..29
  for (auto& [band, decl] : sw_records())
    for (auto& o : decl) {
      if (o.lb.empty()) continue;
      char key[64], ckey[64];
      std::snprintf(key, sizeof key, "kg%02d.%s", band, o.name);
      const std::string nm = o.name;
      std::snprintf(ckey, sizeof ckey, "kg%02d.%s", band, combined_name(nm).c_str());
      const bool g_first = o.lb.size() == 2 && (nm == "sfluxrefo" || nm == "raylao");
      combine(orig.at(key), b.rview(ckey), g_first, nm != "sfluxrefo", band - 15, ngc, ngn, ngs, rwgt);
    }

  for (auto& r : b.regs) {
    if (r.is_int) d->int_arrays.push_back({r.name, {d->hi.data() + r.off, size_t(r.n)}});
    else d->real_arrays.push_back({r.name, {d->hr.data() + r.off, size_t(r.n)}});
  }
  d->dr = Kokkos::View<Real*, Device>("rrtmg_sw_tables_r", size_t(cnt.nr));
  d->di = Kokkos::View<int*, Device>("rrtmg_sw_tables_i", size_t(cnt.ni));
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

std::shared_ptr<const SwTableData> sw_tables(Real cpdair, Real grv, const std::string& data_file) {
  static std::mutex mtx;
  static std::map<std::tuple<Real, Real, std::string>, std::shared_ptr<const SwTableData>> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto key = std::make_tuple(cpdair, grv, data_file);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.empty()) Kokkos::push_finalize_hook([] { std::lock_guard<std::mutex> l(mtx); cache.clear(); });
  auto d = build(cpdair, grv, data_file);
  cache.emplace(key, d);
  return d;
}

} // namespace cm1k::rrtmg
