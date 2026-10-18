// Component tests of the surface-layer port (physics/sfc/cm1k_sfc.*) against
// CM1's sfcphys.F (getcecd, sfcflux, sfcdiags), init_surface.F, the surface
// blocks of turb.F's sfc_and_turb (verbatim copies) and gfortran's
// random_number as init3d.F seeds it, through cm1f_bridge_sfc.F90.
//
// Both sides get identical random inputs (halos included) and every output
// array is compared bit for bit over its whole extent. The bbc=3 pieces of
// the turbulence (tkebc, gettau/sfcstress, turbs with surface fluxes, the
// surface dissipative heating) are in turb_tests.cpp.
#include "init/cm1k_init.hpp"
#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/turb/cm1k_turb.hpp"
#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <vector>

extern "C" {
void cm1f_turb_setup(int ni, int nj, int nk, int ngxy, int ngz, const int* iopt, const double* ropt);
void cm1f_sfc_options(const int* iopt, const double* ropt);
void cm1f_sfc_getcecd(const double* za, double* u1, double* v1, double* s1, double* u10, double* v10,
                      double* s10, const double* xland, double* znt, double* ust, double* cd, double* ch,
                      double* cq);
void cm1f_sfc_sfcflux(double dt, const double* ruh, const double* rvh, const double* rf, const double* psfc,
                      const double* ch, const double* cq, const double* th0, const double* tsk, double* thflux,
                      double* qvflux, const double* mavail, const double* s1, double* s10, const double* tha,
                      const double* qva, double* qsfc, double* znt);
void cm1f_sfc_sfcdiags(const double* tsk, const double* thflux, const double* qvflux, const double* cd,
                       const double* ch, const double* cq, const double* u1, const double* v1, const double* s1,
                       double* wspd, const double* xland, const double* psfc, double* qsfc, double* u10,
                       double* v10, double* hfx, double* qfx, double* cda, double* znt, double* gz1oz0,
                       double* br, const double* hpbl, const double* dsxy, double* th2, double* t2, double* q2,
                       const double* za, const double* th0, const double* tha, const double* rf,
                       const double* qa);
void cm1f_sfc_init_surface(const double* ruh, const double* rvh, double* tsk, double* xland, double* znt,
                           double* mavail, double* dsxy, int* dosfcflx);
void cm1f_sfc_layer_k1(double cgs1, double cgs2, double cgs3, const double* prs, const double* ua,
                       const double* va, const double* u0, const double* v0, const double* th0,
                       const double* tha, double* psfc, double* u1, double* v1, double* s1, double* t1,
                       double* za);
void cm1f_sfc_bcs(double* ust, double* u1, double* v1, double* s1, double* znt, double* wspd, double* zntmp);
void cm1f_random_cm1(int n, double* out, int* seed_size);
}

using namespace cm1k;
namespace S = cm1k::sfc;

namespace {

using H3 = FArray<3, HostSpace>;

double U(std::mt19937_64& gen) { return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); }

H3 hcopy(const H3& h) {
  H3 c("copy", {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  std::memcpy(c.data(), h.data(), h.size() * sizeof(double));
  return c;
}

// Bit-for-bit comparison (NaN == NaN).
void require_same(const H3& a, const H3& ref, const std::string& what) {
  REQUIRE(a.size() == ref.size());
  size_t nbad = 0;
  double maxd = 0;
  for (size_t n = 0; n < a.size(); ++n) {
    const double x = a.data()[n], y = ref.data()[n];
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
  }
  INFO(what << ": " << nbad << " differing points, max |diff| = " << maxd);
  REQUIRE(nbad == 0);
}

Bounds box(int ni, int nj, int nk, int ngxy, int numq) {
  Bounds b;
  b.nx = ni; b.ny = nj; b.nz = nk;
  b.ni = ni; b.nj = nj; b.nk = nk; b.nkp1 = nk + 1;
  b.ngxy = ngxy; b.ngz = 1;
  b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
  b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  b.ibt = b.ib; b.iet = b.ie; b.jbt = b.jb; b.jet = b.je; b.kbt = b.kb; b.ket = b.ke + 1;
  b.ibc = b.ib; b.iec = b.ie; b.jbc = b.jb; b.jec = b.je; b.kbc = b.kb; b.kec = b.ke + 1;
  b.numq = numq;
  return b;
}

struct Case {
  int isfcflx = 1, cecd = 3, set_znt = 0, set_ust = 0, set_flx = 0, pertflx = 0, lu0 = 16, season = 1;
  int imoist = 1;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  double cnstce = 0.0011, cnstcd = 0.0011, cnst_znt = 0.16, cnst_ust = 0.25;
  double cnst_shflx = 0.24, cnst_lhflx = 5.2e-5, tsk0 = 299.28, xland0 = 2.0;
  std::string str() const {
    std::ostringstream s;
    s << "isfcflx=" << isfcflx << " cecd=" << cecd << " set_znt=" << set_znt << " set_ust=" << set_ust
      << " set_flx=" << set_flx << " pertflx=" << pertflx << " lu0=" << lu0 << " season=" << season
      << " imoist=" << imoist << " bc=" << wbc << ebc << sbc << nbc;
    return s.str();
  }
};

S::Options options_for(const Case& c) {
  S::Options o;
  o.sfcmodel = 1; o.bbc = 3; o.oceanmodel = 1; o.cm1setup = 1;
  o.isfcflx = c.isfcflx; o.cecd = c.cecd; o.set_znt = c.set_znt; o.set_ust = c.set_ust;
  o.set_flx = c.set_flx; o.pertflx = c.pertflx; o.lu0 = c.lu0; o.season = c.season;
  o.imoist = c.imoist; o.ptype = c.imoist ? 1 : 0;
  o.cnstce = c.cnstce; o.cnstcd = c.cnstcd; o.cnst_znt = c.cnst_znt; o.cnst_ust = c.cnst_ust;
  o.cnst_shflx = c.cnst_shflx; o.cnst_lhflx = c.cnst_lhflx; o.tsk0 = c.tsk0; o.xland0 = c.xland0;
  o.dosfcflx = (c.isfcflx == 1);
  return o;
}

// Fortran `input` state: grid bounds and options (cm1f_turb_setup), then the
// surface-layer options.
void fortran_setup(const Bounds& b, const Case& c, double dx, double dy, double dz) {
  const int moist = c.imoist;
  const int iopt[21] = {1, 1, 1, c.wbc, c.ebc, c.sbc, c.nbc, moist, moist ? 1 : 0, 0, moist ? 3 : 1, 1,
                        moist ? 2 : 0, 0, moist ? 2 : 0, moist ? 3 : 0, moist ? -2 : 0, moist ? -2 : 0,
                        2, 1, 0};
  const double ropt[3] = {dx, dy, dz};
  cm1f_turb_setup(b.ni, b.nj, b.nk, b.ngxy, b.ngz, iopt, ropt);
  const int so[8] = {c.isfcflx, c.cecd, c.set_znt, c.set_ust, c.set_flx, c.pertflx, c.lu0, c.season};
  const double sr[8] = {c.cnstce, c.cnstcd, c.cnst_znt, c.cnst_ust, c.cnst_shflx, c.cnst_lhflx, c.tsk0, c.xland0};
  cm1f_sfc_options(so, sr);
}

// A device array and its host copy.
struct P {
  FArray3 d;
  H3 h;
  void to_dev() const { d.deep_copy_from(h); }
  void to_host() { h.deep_copy_from(d); }
  double* p() { return h.data(); }
};
P wrap(const FArray3& d) { return P{d, d.to_host()}; }
P mk(const std::string& n, const std::array<int, 3>& lb, const std::array<int, 3>& ub) {
  FArray3 d(n, lb, ub);
  return wrap(d);
}

void fill(H3& a, std::mt19937_64& gen, double mean, double amp) {
  for (size_t n = 0; n < a.size(); ++n) a.data()[n] = mean + amp * U(gen);
}

// Surface arrays on both sides: name -> device/host pair.
struct Sfc {
  S::SfcState s;
  std::vector<std::pair<std::string, P>> a;
  explicit Sfc(const Bounds& b) {
    s.allocate(b);
    for (auto* x : {&s.tsk, &s.znt, &s.xland, &s.mavail, &s.ust, &s.u1, &s.v1, &s.s1, &s.cd, &s.ch, &s.cq,
                    &s.thflux, &s.qvflux, &s.psfc, &s.u10, &s.v10, &s.s10, &s.hfx, &s.qfx, &s.qsfc, &s.wspd,
                    &s.zntmp, &s.t1, &s.hpbl, &s.dsxy, &s.br, &s.cda, &s.gz1oz0, &s.th2, &s.t2, &s.q2})
      a.emplace_back(x->label(), wrap(*x));
  }
  P& operator[](const std::string& n) {
    for (auto& [k, v] : a) if (k == n) return v;
    throw std::runtime_error("no array " + n);
  }
  // random, plausible values everywhere (halos included)
  void randomize(std::mt19937_64& gen) {
    std::uniform_int_distribution<int> I(0, 9);
    for (auto& [n, p] : a) fill(p.h, gen, 1.0, 0.5);
    fill((*this)["tsk"].h, gen, 300.0, 6.0);
    fill((*this)["psfc"].h, gen, 1.0e5, 3.0e3);
    fill((*this)["u1"].h, gen, 2.0, 12.0);
    fill((*this)["v1"].h, gen, -1.0, 12.0);
    auto& s1 = (*this)["s1"].h;
    auto& u1 = (*this)["u1"].h;
    auto& v1 = (*this)["v1"].h;
    for (size_t n = 0; n < s1.size(); ++n) {
      s1.data()[n] = std::sqrt(u1.data()[n] * u1.data()[n] + v1.data()[n] * v1.data()[n]);
      if (I(gen) == 0) { u1.data()[n] = 0; v1.data()[n] = 0; s1.data()[n] = 0; }        // calm
      if (I(gen) == 0) { s1.data()[n] = 0.005; }                                        // s1 < 0.01
    }
    for (auto* n : {"znt", "zntmp"}) {
      auto& z = (*this)[n].h;
      for (size_t k = 0; k < z.size(); ++k) z.data()[k] = std::pow(10.0, -4.0 + 3.0 * (0.5 + 0.5 * U(gen)));
    }
    auto& xl = (*this)["xland"].h;
    for (size_t k = 0; k < xl.size(); ++k) xl.data()[k] = (I(gen) < 5) ? 1.0 : 2.0;
    auto& mav = (*this)["mavail"].h;
    for (size_t k = 0; k < mav.size(); ++k) mav.data()[k] = 0.5 + 0.5 * U(gen);
    fill((*this)["ust"].h, gen, 0.3, 0.29);
    fill((*this)["ch"].h, gen, 1.2e-3, 0.5e-3);
    fill((*this)["cq"].h, gen, 1.2e-3, 0.5e-3);
    fill((*this)["cd"].h, gen, 1.2e-3, 0.5e-3);
    fill((*this)["s10"].h, gen, 8.0, 8.0);
    fill((*this)["thflux"].h, gen, 0.05, 0.1);   // some negative
    fill((*this)["qvflux"].h, gen, 2.0e-5, 3.0e-5);
    fill((*this)["hpbl"].h, gen, 500.0, 499.0);
    auto& ds = (*this)["dsxy"].h;
    for (size_t k = 0; k < ds.size(); ++k) ds.data()[k] = std::pow(10.0, 1.5 + 3.0 * (0.5 + 0.5 * U(gen)));
    fill((*this)["qsfc"].h, gen, 0.02, 0.005);
    for (auto& [n, p] : a) p.to_dev();
  }
  void to_host() { for (auto& [n, p] : a) p.to_host(); }
  // host copies (the Fortran side)
  std::vector<std::pair<std::string, H3>> copies() const {
    std::vector<std::pair<std::string, H3>> c;
    for (auto& [n, p] : a) c.emplace_back(n, hcopy(p.h));
    return c;
  }
};

struct Fields {
  Bounds b;
  P za, th0, tha, rf, prs, ua, va, u0, v0;
  FArray4 qa;
  FArray<4, HostSpace> qah;
  FArray1 ruh, rvh;
  FArray<1, HostSpace> ruhh, rvhh;
  P zeros3;
  explicit Fields(const Bounds& bb, int numq) : b(bb) {
    const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
    za = mk("za", {b.ib, b.jb, 1}, {b.ie, b.je, 1});
    th0 = mk("th0", lo, hi); tha = mk("tha", lo, hi); rf = mk("rf", lo, hi); prs = mk("prs", lo, hi);
    ua = mk("ua", lo, {b.ie + 1, b.je, b.ke}); u0 = mk("u0", lo, {b.ie + 1, b.je, b.ke});
    va = mk("va", lo, {b.ie, b.je + 1, b.ke}); v0 = mk("v0", lo, {b.ie, b.je + 1, b.ke});
    zeros3 = mk("zeros3", lo, hi);
    qa = FArray4("qa", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, numq});
    qah = qa.to_host();
    ruh = FArray1("ruh", {b.ib}, {b.ie}); rvh = FArray1("rvh", {b.jb}, {b.je});
    ruhh = ruh.to_host(); rvhh = rvh.to_host();
  }
  void randomize(std::mt19937_64& gen) {
    fill(za.h, gen, 30.0, 20.0);
    for (int k = b.kb; k <= b.ke; ++k)
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          th0.h(i, j, k) = 298.0 + 3.0 * k;
          prs.h(i, j, k) = 1.0e5 * std::exp(-0.012 * k) * (1.0 + 0.002 * U(gen));
          rf.h(i, j, k) = 1.2 * std::exp(-0.01 * k) * (1.0 + 0.01 * U(gen));
        }
    fill(tha.h, gen, 0.0, 1.5);
    fill(ua.h, gen, 4.0, 8.0); fill(va.h, gen, -2.0, 8.0);
    fill(u0.h, gen, 5.0, 3.0); fill(v0.h, gen, -1.0, 3.0);
    for (size_t n = 0; n < qah.size(); ++n) qah.data()[n] = std::max(0.0, 0.012 + 0.008 * U(gen));
    for (int i = b.ib; i <= b.ie; ++i) ruhh(i) = 1.0 / (1.0 + 0.5 * U(gen));
    for (int j = b.jb; j <= b.je; ++j) rvhh(j) = 1.0 / (1.0 + 0.5 * U(gen));
    for (auto* p : {&za, &th0, &tha, &rf, &prs, &ua, &va, &u0, &v0, &zeros3}) p->to_dev();
    qa.deep_copy_from(qah); ruh.deep_copy_from(ruhh); rvh.deep_copy_from(rvhh);
  }
};

void compare(Sfc& sf, const std::vector<std::pair<std::string, H3>>& f, const std::string& what) {
  sf.to_host();
  for (size_t n = 0; n < sf.a.size(); ++n) require_same(sf.a[n].second.h, f[n].second, what + " " + f[n].first);
}

H3& fa(std::vector<std::pair<std::string, H3>>& f, const std::string& n) {
  for (auto& [k, v] : f) if (k == n) return v;
  throw std::runtime_error("no array " + n);
}

const Constants kC = Constants::make(0);

void check_getcecd(const Case& c, unsigned seed) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  const auto b = box(9, 8, 6, 3, c.imoist ? 3 : 1);
  fortran_setup(b, c, 100.0, 100.0, 50.0);
  const auto o = options_for(c);
  Sfc sf(b);
  sf.randomize(gen);
  Fields F(b, b.numq);
  F.randomize(gen);
  auto f = sf.copies();
  S::getcecd(o, b, F.za.d, sf.s);
  cm1f_sfc_getcecd(F.za.p(), fa(f, "u1").data(), fa(f, "v1").data(), fa(f, "s1").data(), fa(f, "u10").data(),
                   fa(f, "v10").data(), fa(f, "s10").data(), fa(f, "xland").data(), fa(f, "znt").data(),
                   fa(f, "ust").data(), fa(f, "cd").data(), fa(f, "ch").data(), fa(f, "cq").data());
  compare(sf, f, "getcecd");
}

void check_sfcflux(const Case& c, unsigned seed) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  const auto b = box(9, 8, 6, 3, c.imoist ? 3 : 1);
  fortran_setup(b, c, 100.0, 120.0, 50.0);
  const auto o = options_for(c);
  Sfc sf(b);
  sf.randomize(gen);
  Fields F(b, b.numq);
  F.randomize(gen);
  auto f = sf.copies();
  S::sfcflux(o, kC, b, F.th0.d, F.tha.d, F.qa, sf.s);
  cm1f_sfc_sfcflux(2.0, F.ruhh.data(), F.rvhh.data(), F.rf.p(), fa(f, "psfc").data(), fa(f, "ch").data(),
                   fa(f, "cq").data(), F.th0.p(), fa(f, "tsk").data(), fa(f, "thflux").data(),
                   fa(f, "qvflux").data(), fa(f, "mavail").data(), fa(f, "s1").data(), fa(f, "s10").data(),
                   F.tha.p(), F.qah.data(), fa(f, "qsfc").data(), fa(f, "znt").data());
  compare(sf, f, "sfcflux");
}

void check_sfcdiags(const Case& c, unsigned seed) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  const auto b = box(9, 8, 6, 3, c.imoist ? 3 : 1);
  fortran_setup(b, c, 100.0, 120.0, 50.0);
  const auto o = options_for(c);
  Sfc sf(b);
  sf.randomize(gen);
  Fields F(b, b.numq);
  F.randomize(gen);
  auto f = sf.copies();
  S::sfcdiags(o, kC, b, F.za.d, F.th0.d, F.tha.d, F.qa, F.rf.d, sf.s);
  // solve1/turb.F pass divx: qv for imoist=1, zeros otherwise
  const double* qv = c.imoist ? F.qah.data() : F.zeros3.p();
  cm1f_sfc_sfcdiags(fa(f, "tsk").data(), fa(f, "thflux").data(), fa(f, "qvflux").data(), fa(f, "cd").data(),
                    fa(f, "ch").data(), fa(f, "cq").data(), fa(f, "u1").data(), fa(f, "v1").data(),
                    fa(f, "s1").data(), fa(f, "wspd").data(), fa(f, "xland").data(), fa(f, "psfc").data(),
                    fa(f, "qsfc").data(), fa(f, "u10").data(), fa(f, "v10").data(), fa(f, "hfx").data(),
                    fa(f, "qfx").data(), fa(f, "cda").data(), fa(f, "znt").data(), fa(f, "gz1oz0").data(),
                    fa(f, "br").data(), fa(f, "hpbl").data(), fa(f, "dsxy").data(), fa(f, "th2").data(),
                    fa(f, "t2").data(), fa(f, "q2").data(), F.za.p(), F.th0.p(), F.tha.p(), F.rf.p(), qv);
  compare(sf, f, "sfcdiags");
}

// sfc::sfc_layer (the whole surface block of sfc_and_turb) against the
// verbatim k=1 block followed by getcecd, sfcflux (isfcflx=1) and sfcdiags.
void check_sfc_layer(const Case& c, unsigned seed) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  const auto b = box(9, 8, 6, 3, c.imoist ? 3 : 1);
  fortran_setup(b, c, 100.0, 120.0, 50.0);
  const auto o = options_for(c);
  Sfc sf(b);
  sf.randomize(gen);
  Fields F(b, b.numq);
  F.randomize(gen);
  Grid g;
  g.allocate(b);
  g.b = b;
  BaseState base;
  base.allocate(b);
  Kokkos::deep_copy(base.u0.view(), F.u0.d.view());
  Kokkos::deep_copy(base.v0.view(), F.v0.d.view());
  Kokkos::deep_copy(base.th0.view(), F.th0.d.view());
  // zh(:,:,1) = za; the rest of zh unused
  auto zh = g.zh.to_host();
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) zh(i, j, 1) = F.za.h(i, j, 1);
  g.zh.deep_copy_from(zh);
  const double cgs1 = 1.6, cgs2 = -0.7, cgs3 = 0.1;
  auto f = sf.copies();
  auto zaf = hcopy(F.za.h);
  S::sfc_layer(o, kC, g, base, cgs1, cgs2, cgs3, F.ua.d, F.va.d, F.tha.d, F.qa, F.prs.d, F.rf.d, sf.s);
  cm1f_sfc_layer_k1(cgs1, cgs2, cgs3, F.prs.p(), F.ua.p(), F.va.p(), F.u0.p(), F.v0.p(), F.th0.p(), F.tha.p(),
                    fa(f, "psfc").data(), fa(f, "u1").data(), fa(f, "v1").data(), fa(f, "s1").data(),
                    fa(f, "t1").data(), zaf.data());
  cm1f_sfc_getcecd(F.za.p(), fa(f, "u1").data(), fa(f, "v1").data(), fa(f, "s1").data(), fa(f, "u10").data(),
                   fa(f, "v10").data(), fa(f, "s10").data(), fa(f, "xland").data(), fa(f, "znt").data(),
                   fa(f, "ust").data(), fa(f, "cd").data(), fa(f, "ch").data(), fa(f, "cq").data());
  if (c.isfcflx == 1)
    cm1f_sfc_sfcflux(2.0, F.ruhh.data(), F.rvhh.data(), F.rf.p(), fa(f, "psfc").data(), fa(f, "ch").data(),
                     fa(f, "cq").data(), F.th0.p(), fa(f, "tsk").data(), fa(f, "thflux").data(),
                     fa(f, "qvflux").data(), fa(f, "mavail").data(), fa(f, "s1").data(), fa(f, "s10").data(),
                     F.tha.p(), F.qah.data(), fa(f, "qsfc").data(), fa(f, "znt").data());
  const double* qv = c.imoist ? F.qah.data() : F.zeros3.p();
  cm1f_sfc_sfcdiags(fa(f, "tsk").data(), fa(f, "thflux").data(), fa(f, "qvflux").data(), fa(f, "cd").data(),
                    fa(f, "ch").data(), fa(f, "cq").data(), fa(f, "u1").data(), fa(f, "v1").data(),
                    fa(f, "s1").data(), fa(f, "wspd").data(), fa(f, "xland").data(), fa(f, "psfc").data(),
                    fa(f, "qsfc").data(), fa(f, "u10").data(), fa(f, "v10").data(), fa(f, "hfx").data(),
                    fa(f, "qfx").data(), fa(f, "cda").data(), fa(f, "znt").data(), fa(f, "gz1oz0").data(),
                    fa(f, "br").data(), fa(f, "hpbl").data(), fa(f, "dsxy").data(), fa(f, "th2").data(),
                    fa(f, "t2").data(), fa(f, "q2").data(), F.za.p(), F.th0.p(), F.tha.p(), F.rf.p(), qv);
  compare(sf, f, "sfc_layer");
}

void check_bcs(const Case& c, unsigned seed) {
  INFO(c.str());
  std::mt19937_64 gen(seed);
  const auto b = box(9, 8, 6, 3, 1);
  fortran_setup(b, c, 100.0, 120.0, 50.0);
  turb::Options to;
  to.wbc = c.wbc; to.ebc = c.ebc; to.sbc = c.sbc; to.nbc = c.nbc;
  to.set_boundary_flags();
  Sfc sf(b);
  sf.randomize(gen);
  auto f = sf.copies();
  turb::sfc_bcs(to, b, sf.s);
  cm1f_sfc_bcs(fa(f, "ust").data(), fa(f, "u1").data(), fa(f, "v1").data(), fa(f, "s1").data(),
               fa(f, "znt").data(), fa(f, "wspd").data(), fa(f, "zntmp").data());
  compare(sf, f, "sfc_bcs");
}

void check_init_surface(const Case& c, int ni, int nj, unsigned seed) {
  INFO(c.str() << " ni=" << ni << " nj=" << nj);
  std::mt19937_64 gen(seed);
  const auto b = box(ni, nj, 6, 3, 1);
  const double dx = 100.0, dy = 140.0;
  fortran_setup(b, c, dx, dy, 50.0);
  const auto o = options_for(c);
  Sfc sf(b);
  sf.randomize(gen);
  Fields F(b, 1);
  F.randomize(gen);
  Grid g;
  g.allocate(b);
  g.b = b;
  g.dx = dx; g.dy = dy;
  Kokkos::deep_copy(g.ruh.view(), F.ruh.view());
  Kokkos::deep_copy(g.rvh.view(), F.rvh.view());
  auto f = sf.copies();
  S::init_surface(o, g, sf.s);
  int dosfcflx = -1;
  cm1f_sfc_init_surface(F.ruhh.data(), F.rvhh.data(), fa(f, "tsk").data(), fa(f, "xland").data(),
                        fa(f, "znt").data(), fa(f, "mavail").data(), fa(f, "dsxy").data(), &dosfcflx);
  compare(sf, f, "init_surface");
  REQUIRE(dosfcflx == (o.dosfcflx ? 1 : 0));
}

// getlanduse reads ./LANDUSE.TBL: run the Fortran in a case directory that has it.
struct InCaseDir {
  std::filesystem::path old;
  explicit InCaseDir(const std::string& case_name) : old(std::filesystem::current_path()) {
    const auto dir = std::filesystem::path(testing::case_namelist(case_name)).parent_path();
    REQUIRE(std::filesystem::exists(dir / "LANDUSE.TBL"));
    std::filesystem::current_path(dir);
  }
  ~InCaseDir() { std::filesystem::current_path(old); }
};

} // namespace

TEST_CASE("sfc getcecd matches sfcphys.F", "[bridge][sfc]") {
  unsigned seed = 101;
  for (int isfcflx : {0, 1})
    for (int cecd : {1, 2, 3}) {
      Case c; c.isfcflx = isfcflx; c.cecd = cecd;
      check_getcecd(c, seed++);
    }
  Case z; z.set_znt = 1; z.cnst_znt = 0.1; check_getcecd(z, seed++);
  Case u; u.set_ust = 1; check_getcecd(u, seed++);
  Case zu; zu.set_znt = 1; zu.set_ust = 1; check_getcecd(zu, seed++);
  Case c1; c1.cecd = 1; c1.cnstcd = 2.0e-3; c1.cnstce = 1.5e-3; check_getcecd(c1, seed++);
}

TEST_CASE("sfc sfcflux matches sfcphys.F", "[bridge][sfc]") {
  unsigned seed = 201;
  for (int imoist : {0, 1})
    for (int set_flx : {0, 1}) {
      Case c; c.imoist = imoist; c.set_flx = set_flx;
      check_sfcflux(c, seed++);
    }
}

TEST_CASE("sfc sfcdiags matches sfcphys.F", "[bridge][sfc]") {
  unsigned seed = 301;
  for (int imoist : {0, 1}) {
    Case c; c.imoist = imoist;
    check_sfcdiags(c, seed++);
  }
}

TEST_CASE("sfc sfc_layer matches the surface block of sfc_and_turb", "[bridge][sfc]") {
  unsigned seed = 401;
  for (int imoist : {0, 1})
    for (int pertflx : {0, 1})
      for (int isfcflx : {0, 1}) {
        Case c; c.imoist = imoist; c.pertflx = pertflx; c.isfcflx = isfcflx;
        check_sfc_layer(c, seed++);
      }
  Case z; z.set_znt = 1; z.set_flx = 1; check_sfc_layer(z, seed++);
  Case d; d.cecd = 2; check_sfc_layer(d, seed++);
}

TEST_CASE("sfc bc/comms block matches sfc_and_turb", "[bridge][sfc]") {
  unsigned seed = 501;
  for (auto bc : std::vector<std::array<int, 4>>{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {2, 3, 4, 2}}) {
    Case c; c.wbc = bc[0]; c.ebc = bc[1]; c.sbc = bc[2]; c.nbc = bc[3];
    check_bcs(c, seed++);
  }
}

TEST_CASE("sfc init_surface matches init_surface.F", "[bridge][sfc]") {
  InCaseDir in("cbl_sfc");
  unsigned seed = 601;
  for (int season : {1, 2})
    for (int lu0 : {0, 1, 2, 3, 7, 16, 19, 24, 28, 33}) {
      Case c; c.lu0 = lu0; c.season = season; c.tsk0 = 290.0 + lu0;
      check_init_surface(c, 9, 8, seed++);
    }
  Case z; z.set_znt = 1; z.cnst_znt = 0.05; z.lu0 = 3; check_init_surface(z, 9, 8, seed++);
  Case z0; z0.set_znt = 1; z0.cnst_znt = 0.0; check_init_surface(z0, 9, 8, seed++);   // max(1e-12, .)
  Case nf; nf.isfcflx = 0; check_init_surface(nf, 9, 8, seed++);
  Case y1; check_init_surface(y1, 9, 1, seed++);   // ny=1: dsxy = dx*ruh
  Case x1; check_init_surface(x1, 1, 8, seed++);   // nx=1: dsxy = dy*rvh
}

TEST_CASE("sfc options: unsupported values throw", "[sfc]") {
  S::Options o0 = options_for(Case{});
  REQUIRE_NOTHROW(o0.check());
  auto bad = [&](auto f) { S::Options x = o0; f(x); REQUIRE_THROWS(x.check()); };
  bad([](S::Options& x) { x.sfcmodel = 2; });
  bad([](S::Options& x) { x.bbc = 1; });
  bad([](S::Options& x) { x.oceanmodel = 2; });
  bad([](S::Options& x) { x.cecd = 0; });
  bad([](S::Options& x) { x.initsfc = 2; });
  bad([](S::Options& x) { x.tbc = 3; });
  bad([](S::Options& x) { x.testcase = 1; });
  bad([](S::Options& x) { x.imove = 1; });
  bad([](S::Options& x) { x.axisymm = 1; });
  bad([](S::Options& x) { x.terrain_flag = true; });
  bad([](S::Options& x) { x.ipbl = 2; });
  bad([](S::Options& x) { x.season = 3; });
  bad([](S::Options& x) { x.lu0 = 34; });
  bad([](S::Options& x) { x.tsk0 = 50.0; });
  bad([](S::Options& x) { x.imoist = 1; x.ptype = 2; });
  S::Options none;   // sfcmodel = 0
  REQUIRE_NOTHROW(none.check());
  REQUIRE_FALSE(none.active());
  none.isfcflx = 1;
  REQUIRE_THROWS(none.check());
  // param.F: sfcmodel >= 1 sets bbc = 3
  auto nl = Namelist::from_string("&param2\n sfcmodel = 1, isfcflx = 1, bbc = 1 /\n");
  const auto o = S::Options::from_namelist(nl);
  REQUIRE(o.bbc == 3);
  REQUIRE(o.dosfcflx);
  REQUIRE(turb::Options::from_namelist(nl).bbc == 3);
  REQUIRE_THROWS(S::landuse_sfz0(34, 1));
  REQUIRE(S::landuse_sfz0(16, 1) == 0.01);
}

TEST_CASE("gfortran random_number as seeded by init3d.F", "[bridge][init][sfc]") {
  const int n = 20000;
  std::vector<double> f(n);
  int k = 0;
  cm1f_random_cm1(n, f.data(), &k);
  REQUIRE(k == 8);   // xoshiro256** (gfortran >= 7): the port assumes this seed size
  init::GfortranRandom r;
  int nbad = 0;
  for (int i = 0; i < n; ++i)
    if (r.next() != f[i]) ++nbad;
  REQUIRE(nbad == 0);
}
