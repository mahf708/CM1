// Component tests of the advection port (cm1k_adv.*) against CM1's
// adv.F/adv_routines.F/pdef.F, called through cm1f_bridge_adv.F90.
//
// Both sides get identical inputs (random but plausible fields, halos
// included) and the tendencies are compared bit-for-bit over the whole
// array (points the routines do not touch must stay untouched too).
#include "dynamics/cm1k_adv.hpp"
#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>

extern "C" {
void cm1f_adv_setup(int ni, int nj, int nk, int ngxy, int ngz, const int* iopt, const double* ropt);
void cm1f_advs(int nrk, int wflag, int pdef, int pdefweno, double dt, double weps, int diffit,
               int hadvorder, int vadvorder, const double* uh, const double* vh, const double* mh,
               const double* c1, const double* c2, const double* rho0, const double* rr0,
               const double* rf0, const double* rrf0, const double* rru, const double* rrv,
               const double* rrw, const double* divx, const double* rho, const double* rr,
               const double* s0, const double* s, double* sten);
void cm1f_advu(int nrk, double dt, const double* uh, const double* uf, const double* vh, const double* vf,
               const double* mh, const double* mf, const double* c1, const double* c2,
               const double* rho0, const double* rr0, const double* rf0, const double* rrf0,
               const double* rru, const double* rrv, const double* rrw, const double* divx,
               const double* rho, const double* u3d, const double* v3d, const double* w3d, double* uten);
void cm1f_advv(int nrk, double dt, const double* uh, const double* uf, const double* vh, const double* vf,
               const double* mh, const double* mf, const double* c1, const double* c2,
               const double* rho0, const double* rr0, const double* rf0, const double* rrf0,
               const double* rru, const double* rrv, const double* rrw, const double* divx,
               const double* rho, const double* u3d, const double* v3d, const double* w3d, double* vten);
void cm1f_advw(int nrk, double dt, int hadvorder, int vadvorder, int advweno, int iamw,
               const double* uh, const double* uf, const double* vh, const double* vf,
               const double* mh, const double* mf, const double* c1, const double* c2,
               const double* rho0, const double* rr0, const double* rf0, const double* rrf0,
               const double* rru, const double* rrv, const double* rrw, const double* divx,
               const double* rho, const double* u3d, const double* v3d, const double* w3d, double* wten);
}

using namespace cm1k;

namespace {

using H1 = FArray<1, HostSpace>;
using H3 = FArray<3, HostSpace>;

// Host and device copies of one array.
template <int R>
struct Pair {
  FArray<R> d;
  FArray<R, HostSpace> h;
  void to_dev() const { d.deep_copy_from(h); }
  void to_host() { h.deep_copy_from(d); }
  const double* p() const { return h.data(); }
  double* p() { return h.data(); }
};
template <int R>
Pair<R> make_pair_arr(const std::string& n, const std::array<int, R>& lb, const std::array<int, R>& ub) {
  Pair<R> p;
  p.d = FArray<R>(n, lb, ub);
  p.h = p.d.to_host();
  return p;
}

// Everything advection needs, on both sides.
struct Setup {
  Bounds b;
  Grid g;
  BaseState base;
  AdvOptions o;
  Real dt = 2.0;
  Pair<1> uh, uf, vh, vf;
  Pair<3> mh, mf, c1, c2, rho0, rr0, rf0, rrf0;
  Pair<3> rru, rrv, rrw, divx, rho, rr;

  Setup(int ni, int nj, int nk, int ngxy) {
    b.nx = ni; b.ny = nj; b.nz = nk;
    b.ni = ni; b.nj = nj; b.nk = nk; b.nkp1 = nk + 1;
    b.ngxy = ngxy; b.ngz = 1;
    b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
    g.allocate(b);
    base.allocate(b);
    auto a1 = [](const FArray1& d) { Pair<1> p; p.d = d; p.h = d.to_host(); return p; };
    auto a3 = [](const FArray3& d) { Pair<3> p; p.d = d; p.h = d.to_host(); return p; };
    uh = a1(g.uh); uf = a1(g.uf); vh = a1(g.vh); vf = a1(g.vf);
    mh = a3(g.mh); mf = a3(g.mf); c1 = a3(g.c1); c2 = a3(g.c2);
    rho0 = a3(base.rho0); rr0 = a3(base.rr0); rf0 = a3(base.rf0); rrf0 = a3(base.rrf0);
    const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hi{b.ie, b.je, b.ke};
    rru = make_pair_arr<3>("rru", lo, {b.ie + 1, b.je, b.ke});
    rrv = make_pair_arr<3>("rrv", lo, {b.ie, b.je + 1, b.ke});
    rrw = make_pair_arr<3>("rrw", lo, {b.ie, b.je, b.ke + 1});
    divx = make_pair_arr<3>("divx", lo, hi);
    rho = make_pair_arr<3>("rho", lo, hi);
    rr = make_pair_arr<3>("rr", lo, hi);
  }

  // Random, physically plausible grid/base state/mass fluxes.
  void randomize(std::mt19937_64& gen) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    g.dx = 1000.0; g.dy = 900.0; g.dz = 500.0;
    g.rdx = 1.0/g.dx; g.rdy = 1.0/g.dy; g.rdz = 1.0/g.dz;
    for (int i = b.ib; i <= b.ie; ++i) uh.h(i) = 1.0 + 0.2*U(gen);
    for (int i = b.ib; i <= b.ie + 1; ++i) uf.h(i) = 1.0 + 0.2*U(gen);
    for (int j = b.jb; j <= b.je; ++j) vh.h(j) = 1.0 + 0.2*U(gen);
    for (int j = b.jb; j <= b.je + 1; ++j) vf.h(j) = 1.0 + 0.2*U(gen);
    for (int k = b.kb; k <= b.ke + 1; ++k)
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          const double z = 0.1*k;
          mf.h(i, j, k) = 1.0 + 0.3*U(gen);
          if (k <= b.ke) {
            mh.h(i, j, k) = 1.0 + 0.3*U(gen);
            c2.h(i, j, k) = 0.5 + 0.1*U(gen);
            c1.h(i, j, k) = 1.0 - c2.h(i, j, k);
            rho0.h(i, j, k) = 1.2*std::exp(-z) * (1.0 + 0.01*U(gen));
            rr0.h(i, j, k) = 1.0/rho0.h(i, j, k);
            rf0.h(i, j, k) = 1.2*std::exp(-z + 0.05) * (1.0 + 0.01*U(gen));
            rrf0.h(i, j, k) = 1.0/rf0.h(i, j, k);
            rho.h(i, j, k) = rho0.h(i, j, k)*(1.0 + 0.01*U(gen));
            rr.h(i, j, k) = 1.0/rho.h(i, j, k);
            divx.h(i, j, k) = 1.0e-3*U(gen);
          }
        }
    fill(rru.h, gen, 10.0, 3.0);
    fill(rrv.h, gen, -5.0, 3.0);
    fill(rrw.h, gen, 0.0, 2.0);
    // a few exact zeros (the >= / > branches)
    std::uniform_int_distribution<int> I(0, 9);
    for (auto* a : {&rru.h, &rrv.h, &rrw.h})
      for (size_t n = 0; n < a->size(); ++n) if (I(gen) == 0) a->data()[n] = 0.0;
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) { rrw.h(i, j, 1) = 0.0; rrw.h(i, j, b.nk + 1) = 0.0; }
    sync();
  }
  static void fill(H3& a, std::mt19937_64& gen, double mean, double amp) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (size_t n = 0; n < a.size(); ++n) a.data()[n] = mean + amp*U(gen);
  }
  void sync() {
    for (auto* p : {&uh, &uf, &vh, &vf}) p->to_dev();
    for (auto* p : {&mh, &mf, &c1, &c2, &rho0, &rr0, &rf0, &rrf0, &rru, &rrv, &rrw, &divx, &rho, &rr}) p->to_dev();
  }

  void fortran_setup() const {
    const int iopt[14] = {o.hadvordrs, o.vadvordrs, o.hadvordrv, o.vadvordrv, o.advwenos, o.advwenov,
                          o.weno_order, o.idiff, o.mdiff, o.difforder, o.wbc, o.ebc, o.sbc, o.nbc};
    const double ropt[4] = {g.rdx, g.rdy, g.rdz, o.kdiff6};
    cm1f_adv_setup(b.ni, b.nj, b.nk, b.ngxy, b.ngz, iopt, ropt);
  }
  AdvFluxes fluxes() const { return AdvFluxes{rru.d, rrv.d, rrw.d, divx.d, rho.d, rr.d}; }
};

// Deep copy of a host array (to_host() of a host array aliases it).
H3 hcopy(const H3& h) {
  H3 c("copy", {h.lb(0), h.lb(1), h.lb(2)}, {h.ub(0), h.ub(1), h.ub(2)});
  std::memcpy(c.data(), h.data(), h.size()*sizeof(double));
  return c;
}

// Compare two host arrays bit-for-bit; report the max abs difference.
void require_same(const H3& a, const H3& ref, const std::string& what) {
  size_t nbad = 0;
  double maxd = 0, maxref = 0;
  for (size_t n = 0; n < a.size(); ++n) {
    const double x = a.data()[n], y = ref.data()[n];
    maxref = std::max(maxref, std::fabs(y));
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) { ++nbad; maxd = std::max(maxd, std::fabs(x - y)); }
  }
  INFO(what << ": " << nbad << " differing points, max |diff| = " << maxd << " (max |ref| = " << maxref << ")");
  REQUIRE(nbad == 0);
}

Pair<3> rand_field(const std::string& n, const std::array<int, 3>& lb, const std::array<int, 3>& ub,
                   std::mt19937_64& gen, double mean, double amp, bool positive = false) {
  auto p = make_pair_arr<3>(n, lb, ub);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  for (size_t m = 0; m < p.h.size(); ++m) {
    double v = mean + amp*U(gen);
    if (positive) v = std::max(0.0, v);
    p.h.data()[m] = v;
  }
  p.to_dev();
  return p;
}

struct ScalarCase {
  int order_h, order_v, wbc, nrk, wflag, advwenos, pdef, pdefweno, diffit, mdiff;
};

void run_advs(const ScalarCase& c, int ngxy, unsigned seed) {
  std::mt19937_64 gen(seed);
  Setup S(12, 10, 22, ngxy);
  S.o.hadvordrs = c.order_h; S.o.vadvordrs = c.order_v;
  S.o.advwenos = c.advwenos; S.o.weno_order = 5;
  S.o.idiff = c.diffit; S.o.mdiff = c.mdiff; S.o.difforder = 6;
  S.o.wbc = S.o.ebc = S.o.sbc = S.o.nbc = c.wbc;
  S.o.set_boundary_flags();
  S.randomize(gen);
  S.fortran_setup();

  const std::array<int, 3> lo{S.b.ib, S.b.jb, S.b.kb}, hi{S.b.ie, S.b.je, S.b.ke};
  // moisture-like (many zeros, pdef active) or theta-like fields
  const bool q = (c.pdef == 1 || c.pdefweno == 1);
  auto s0 = rand_field("s0", lo, hi, gen, q ? 0.0 : 300.0, q ? 1.0e-3 : 5.0, q);
  auto s = rand_field("s", lo, hi, gen, q ? 0.0 : 300.0, q ? 1.0e-3 : 5.0, q);
  auto sten = rand_field("sten", lo, hi, gen, 0.0, 1.0e-3);
  auto sten_f = hcopy(sten.h);   // input/output of the Fortran call

  AdvsArgs a;
  a.wflag = c.wflag; a.pdef = c.pdef; a.pdefweno = c.pdefweno;
  a.weps = q ? 1.0e-3*phys::epsilon : 10.0*phys::epsilon;
  a.diffit = c.diffit; a.hadvorder = c.order_h; a.vadvorder = c.order_v;

  Advection adv(S.g, S.base, S.o);
  adv.advs(c.nrk, a, S.dt, S.fluxes(), s0.d, s.d, sten.d);
  sten.to_host();

  cm1f_advs(c.nrk, a.wflag, a.pdef, a.pdefweno, S.dt, a.weps, a.diffit, a.hadvorder, a.vadvorder,
            S.uh.p(), S.vh.p(), S.mh.p(), S.c1.p(), S.c2.p(), S.rho0.p(), S.rr0.p(), S.rf0.p(), S.rrf0.p(),
            S.rru.p(), S.rrv.p(), S.rrw.p(), S.divx.p(), S.rho.p(), S.rr.p(), s0.p(), s.p(), sten_f.data());

  std::ostringstream os;
  os << "advs order " << c.order_h << "/" << c.order_v << " wbc " << c.wbc << " nrk " << c.nrk
     << " wflag " << c.wflag << " advwenos " << c.advwenos << " pdef " << c.pdef
     << " pdefweno " << c.pdefweno << " diffit " << c.diffit << " mdiff " << c.mdiff << " ngxy " << ngxy;
  require_same(sten.h, sten_f, os.str());
}

struct VelCase {
  int order_h, order_v, bc, nrk, advwenov, idiff, mdiff;
};

void run_advuvw(const VelCase& c, int ngxy, unsigned seed) {
  std::mt19937_64 gen(seed);
  Setup S(12, 10, 22, ngxy);
  S.o.hadvordrv = c.order_h; S.o.vadvordrv = c.order_v;
  S.o.hadvordrs = c.order_h; S.o.vadvordrs = c.order_v;
  S.o.advwenov = c.advwenov; S.o.advwenos = c.advwenov; S.o.weno_order = 5;
  S.o.idiff = c.idiff; S.o.mdiff = c.mdiff; S.o.difforder = 6;
  S.o.wbc = S.o.ebc = S.o.sbc = S.o.nbc = c.bc;
  S.o.set_boundary_flags();
  S.randomize(gen);
  S.fortran_setup();

  const auto& b = S.b;
  auto u3d = rand_field("u3d", {b.ib, b.jb, b.kb}, {b.ie + 1, b.je, b.ke}, gen, 10.0, 5.0);
  auto v3d = rand_field("v3d", {b.ib, b.jb, b.kb}, {b.ie, b.je + 1, b.ke}, gen, -3.0, 5.0);
  auto w3d = rand_field("w3d", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1}, gen, 0.0, 2.0);
  auto uten = rand_field("uten", {b.ib, b.jb, b.kb}, {b.ie + 1, b.je, b.ke}, gen, 0.0, 1.0e-2);
  auto vten = rand_field("vten", {b.ib, b.jb, b.kb}, {b.ie, b.je + 1, b.ke}, gen, 0.0, 1.0e-2);
  auto wten = rand_field("wten", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1}, gen, 0.0, 1.0e-2);
  auto copy = [](const H3& h) { return hcopy(h); };
  auto uten_f = copy(uten.h), vten_f = copy(vten.h), wten_f = copy(wten.h), tke_f = copy(wten.h);
  auto tketen = rand_field("tketen", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1}, gen, 0.0, 1.0e-2);
  tke_f = copy(tketen.h);

  Advection adv(S.g, S.base, S.o);
  const auto f = S.fluxes();
  adv.advu(c.nrk, S.dt, f, u3d.d, uten.d);
  adv.advv(c.nrk, S.dt, f, v3d.d, vten.d);
  adv.advw(c.nrk, S.dt, f, w3d.d, wten.d);
  // TKE-like use of advw (scalar orders, advwenos, iamw = .false.)
  adv.advw(c.nrk, S.dt, f, w3d.d, tketen.d, S.o.hadvordrs, S.o.vadvordrs, S.o.advwenos);
  uten.to_host(); vten.to_host(); wten.to_host(); tketen.to_host();

#define CM1F_ARGS S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.c1.p(), S.c2.p(), \
    S.rho0.p(), S.rr0.p(), S.rf0.p(), S.rrf0.p(), S.rru.p(), S.rrv.p(), S.rrw.p(), S.divx.p(), S.rho.p(), \
    u3d.p(), v3d.p(), w3d.p()
  cm1f_advu(c.nrk, S.dt, CM1F_ARGS, uten_f.data());
  cm1f_advv(c.nrk, S.dt, CM1F_ARGS, vten_f.data());
  cm1f_advw(c.nrk, S.dt, S.o.hadvordrv, S.o.vadvordrv, S.o.advwenov, 1, CM1F_ARGS, wten_f.data());
  cm1f_advw(c.nrk, S.dt, S.o.hadvordrs, S.o.vadvordrs, S.o.advwenos, 0, CM1F_ARGS, tke_f.data());
#undef CM1F_ARGS

  std::ostringstream os;
  os << "order " << c.order_h << "/" << c.order_v << " bc " << c.bc << " nrk " << c.nrk
     << " advwenov " << c.advwenov << " idiff " << c.idiff << " mdiff " << c.mdiff << " ngxy " << ngxy;
  require_same(uten.h, uten_f, "advu " + os.str());
  require_same(vten.h, vten_f, "advv " + os.str());
  require_same(wten.h, wten_f, "advw " + os.str());
  require_same(tketen.h, tke_f, "advw(tke) " + os.str());
}

int ngxy_for(int order) { return order <= 4 ? 2 : (order <= 6 ? 3 : (order <= 8 ? 4 : 5)); }

} // namespace

TEST_CASE("advs matches adv.F: default 5th order", "[bridge][adv]") {
  unsigned seed = 1;
  for (int bc : {1, 2, 3})
    for (int nrk : {1, 2, 3})
      for (int pdef : {0, 1})
        run_advs({5, 5, bc, nrk, 1, 0, pdef, 0, 0, 0}, 3, seed++);
}

TEST_CASE("advs matches adv.F: orders 2..10", "[bridge][adv]") {
  unsigned seed = 100;
  for (int order = 2; order <= 10; ++order)
    for (int bc : {1, 2})
      for (int pdef : {0, 1}) {
        run_advs({order, order, bc, 3, 1, 0, pdef, 0, 0, 0}, ngxy_for(order), seed++);
        run_advs({order, order, bc, 3, 1, 0, pdef, 0, 0, 0}, 5, seed++);
      }
  // mixed horizontal/vertical orders, as for pressure (wflag=0, orders 2/2)
  run_advs({5, 3, 1, 2, 0, 0, 0, 0, 0, 0}, 3, seed++);
  run_advs({6, 2, 2, 1, 0, 0, 0, 0, 0, 0}, 3, seed++);
  run_advs({2, 2, 1, 1, 0, 0, 0, 0, 0, 0}, 3, seed++);
}

TEST_CASE("advs matches adv.F: WENO5 and pdefweno", "[bridge][adv]") {
  unsigned seed = 200;
  for (int bc : {1, 2})
    for (int advwenos : {1, 2})
      for (int nrk : {1, 3})
        for (int pdef : {0, 1})
          for (int pdefweno : {0, 1})
            run_advs({5, 5, bc, nrk, 1, advwenos, pdef, pdefweno, 0, 0}, 3, seed++);
  // wflag = 0 disables WENO
  run_advs({5, 5, 1, 3, 0, 1, 0, 0, 0, 0}, 3, seed++);
}

TEST_CASE("advs matches adv.F: 6th-order diffusion", "[bridge][adv]") {
  unsigned seed = 300;
  for (int bc : {1, 2})
    for (int mdiff : {0, 1})
      for (int pdef : {0, 1})
        run_advs({5, 5, bc, 3, 1, 0, pdef, 0, 1, mdiff}, 3, seed++);
}

TEST_CASE("advu/advv/advw match adv.F", "[bridge][adv]") {
  unsigned seed = 400;
  for (int bc : {1, 2, 3})
    for (int nrk : {1, 2, 3})
      run_advuvw({5, 5, bc, nrk, 0, 0, 0}, 3, seed++);
  for (int order = 2; order <= 10; ++order)
    for (int bc : {1, 2})
      run_advuvw({order, order, bc, 3, 0, 0, 0}, ngxy_for(order), seed++);
  for (int bc : {1, 2})
    for (int advwenov : {1, 2})
      for (int nrk : {1, 3})
        run_advuvw({5, 5, bc, nrk, advwenov, 0, 0}, 3, seed++);
  for (int bc : {1, 2})
    for (int mdiff : {0, 1})
      run_advuvw({5, 5, bc, 3, 0, 1, mdiff}, 3, seed++);
}

TEST_CASE("advection options: unsupported values throw", "[adv]") {
  AdvOptions o;
  o.check();
  { auto x = o; x.terrain_flag = true; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.axisymm = 1; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.do_ib = true; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.dolsw = true; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.advwenos = 1; x.weno_order = 7; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.hadvordrs = 11; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.wbc = 5; REQUIRE_THROWS(x.check()); }
  Namelist nl = Namelist::from_string("&param1\n advwenos = 2, weno_order = 5, hadvordrs = 3, wbc = 2 /\n");
  auto y = AdvOptions::from_namelist(nl);
  REQUIRE(y.hadvordrs == 5);   // wenocheck
  REQUIRE(y.ibw == 1);
  REQUIRE(y.ibe == 0);
}
