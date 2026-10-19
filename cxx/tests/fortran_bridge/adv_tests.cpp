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
void cm1f_adv_terrain(const double* gz, const double* rgz, const double* gzu, const double* gzv,
                      const double* rds, const double* rdsf);
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

  static Bounds box(int ni, int nj, int nk, int ngxy) {
    Bounds b;
    b.nx = ni; b.ny = nj; b.nz = nk;
    b.ni = ni; b.nj = nj; b.nk = nk; b.nkp1 = nk + 1;
    b.ngxy = ngxy; b.ngz = 1;
    b.ib = 1 - ngxy; b.ie = ni + ngxy; b.jb = 1 - ngxy; b.je = nj + ngxy; b.kb = 0; b.ke = nk + 1;
    return b;
  }
  static Bounds with_terrain(Bounds b, bool terrain) {
    if (terrain) {
      b.terrain_flag = true;
      b.itb = b.ib; b.ite = b.ie; b.jtb = b.jb; b.jte = b.je; b.ktb = b.kb; b.kte = b.ke;
    }
    return b;
  }
  Setup(int ni, int nj, int nk, int ngxy, bool terrain = false)
      : Setup(with_terrain(box(ni, nj, nk, ngxy), terrain)) {}
  explicit Setup(const Bounds& bb) : b(bb) {
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

  // terrain_flag: random (not mutually consistent) metrics on both sides;
  // call after fortran_setup
  void terrain(std::mt19937_64& gen) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    o.terrain_flag = true;
    auto h2 = [](const FArray3& a) { return a.to_host(); };
    auto gz = h2(g.gz), rgz = h2(g.rgz), gzu = h2(g.gzu), gzv = h2(g.gzv);
    for (int j = b.jb; j <= b.je; ++j)
      for (int i = b.ib; i <= b.ie; ++i) {
        gz(i, j, 1) = 1.0 + 0.1*U(gen); rgz(i, j, 1) = 1.0/gz(i, j, 1);
        gzu(i, j, 1) = 1.0 + 0.1*U(gen); gzv(i, j, 1) = 1.0 + 0.1*U(gen);
      }
    auto rds = g.rds.to_host(), rdsf = g.rdsf.to_host();
    for (int k = b.kb; k <= b.ke; ++k) rds(k) = g.rdz*(1.0 + 0.2*U(gen));
    for (int k = b.kb; k <= b.ke + 1; ++k) rdsf(k) = g.rdz*(1.0 + 0.2*U(gen));
    g.gz.deep_copy_from(gz); g.rgz.deep_copy_from(rgz); g.gzu.deep_copy_from(gzu); g.gzv.deep_copy_from(gzv);
    g.rds.deep_copy_from(rds); g.rdsf.deep_copy_from(rdsf);
    cm1f_adv_terrain(gz.data(), rgz.data(), gzu.data(), gzv.data(), rds.data(), rdsf.data());
  }
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
  bool terrain = false;
};

void run_advs(const ScalarCase& c, int ngxy, unsigned seed) {
  std::mt19937_64 gen(seed);
  Setup S(12, 10, 22, ngxy, c.terrain);
  S.o.hadvordrs = c.order_h; S.o.vadvordrs = c.order_v;
  S.o.advwenos = c.advwenos; S.o.weno_order = 5;
  S.o.idiff = c.diffit; S.o.mdiff = c.mdiff; S.o.difforder = 6;
  S.o.wbc = S.o.ebc = S.o.sbc = S.o.nbc = c.wbc;
  S.o.set_boundary_flags();
  S.randomize(gen);
  S.fortran_setup();
  if (c.terrain) S.terrain(gen);

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
     << " pdefweno " << c.pdefweno << " diffit " << c.diffit << " mdiff " << c.mdiff << " ngxy " << ngxy
     << " terrain " << c.terrain;
  require_same(sten.h, sten_f, os.str());
}

struct VelCase {
  int order_h, order_v, bc, nrk, advwenov, idiff, mdiff;
  bool terrain = false;
};

void run_advuvw(const VelCase& c, int ngxy, unsigned seed) {
  std::mt19937_64 gen(seed);
  Setup S(12, 10, 22, ngxy, c.terrain);
  S.o.hadvordrv = c.order_h; S.o.vadvordrv = c.order_v;
  S.o.hadvordrs = c.order_h; S.o.vadvordrs = c.order_v;
  S.o.advwenov = c.advwenov; S.o.advwenos = c.advwenov; S.o.weno_order = 5;
  S.o.idiff = c.idiff; S.o.mdiff = c.mdiff; S.o.difforder = 6;
  S.o.wbc = S.o.ebc = S.o.sbc = S.o.nbc = c.bc;
  S.o.set_boundary_flags();
  S.randomize(gen);
  S.fortran_setup();
  if (c.terrain) S.terrain(gen);

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
     << " advwenov " << c.advwenov << " idiff " << c.idiff << " mdiff " << c.mdiff << " ngxy " << ngxy
     << " terrain " << c.terrain;
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

TEST_CASE("advs/advu/advv/advw match adv.F with terrain", "[bridge][adv][terrain]") {
  unsigned seed = 700;
  for (int bc : {1, 2})
    for (int nrk : {1, 3}) {
      for (int pdef : {0, 1}) run_advs({5, 5, bc, nrk, 1, 0, pdef, 0, 0, 0, true}, 3, seed++);
      for (int advwenos : {1, 2})
        for (int pdefweno : {0, 1}) run_advs({5, 5, bc, nrk, 1, advwenos, 1, pdefweno, 0, 0, true}, 3, seed++);
      run_advs({5, 5, bc, nrk, 1, 0, 0, 0, 1, 1, true}, 3, seed++);   // diff6 inside advs
      run_advuvw({5, 5, bc, nrk, 0, 0, 0, true}, 3, seed++);
      run_advuvw({5, 5, bc, nrk, 1, 1, 1, true}, 3, seed++);
    }
  for (int order : {2, 3, 4, 6})
    run_advs({order, order, 1, 3, 1, 0, 1, 0, 0, 0, true}, ngxy_for(order), seed++);
  // reset the Fortran state (terrain off) for the other tests
  Setup S(12, 10, 22, 3);
  S.fortran_setup();
}

TEST_CASE("advection options: unsupported values throw", "[adv]") {
  AdvOptions o;
  o.check();
  { auto x = o; x.terrain_flag = true; REQUIRE_NOTHROW(x.check()); }   // terrain branches are ported
  { auto x = o; x.axisymm = 1; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.do_ib = true; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.dolsw = true; REQUIRE_NOTHROW(x.check()); }   // wsub is ported
  { auto x = o; x.dolsw = true; x.terrain_flag = true; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.advwenos = 1; x.weno_order = 7; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.hadvordrs = 11; REQUIRE_THROWS(x.check()); }
  { auto x = o; x.wbc = 5; REQUIRE_THROWS(x.check()); }
  Namelist nl = Namelist::from_string("&param1\n advwenos = 2, weno_order = 5, hadvordrs = 3, wbc = 2 /\n");
  auto y = AdvOptions::from_namelist(nl);
  REQUIRE(y.hadvordrs == 5);   // wenocheck
  REQUIRE(y.ibw == 1);
  REQUIRE(y.ibe == 0);
}

//------------------------------------------------------------------------------
// Fields from the Fortran reference dumps (cases in cxx/cases)
//------------------------------------------------------------------------------

namespace {

Pair<3> zeros_like(const Pair<3>& like) {
  auto p = make_pair_arr<3>("ten", {like.h.lb(0), like.h.lb(1), like.h.lb(2)},
                            {like.h.ub(0), like.h.ub(1), like.h.ub(2)});
  Kokkos::deep_copy(p.h.view(), 0.0);
  p.to_dev();
  return p;
}

void run_dump_case(const std::string& case_name, int nstep) {
  const auto root = testing::case_ref(case_name);
  if (root.empty()) return;
  const auto dir = DumpReader::step_dir(root, nstep);
  if (!std::filesystem::exists(dir)) { WARN("missing " << dir); return; }
  DumpReader rd(dir);
  const auto nl = Namelist::from_file(testing::case_namelist(case_name));

  Setup S(Bounds::from_namelist(nl));
  S.o = AdvOptions::from_namelist(nl);
  const auto& b = S.b;
  S.g.dx = nl.getr("dx", 0); S.g.dy = nl.getr("dy", 0); S.g.dz = nl.getr("dz", 0);
  S.g.rdx = 1.0/S.g.dx; S.g.rdy = 1.0/S.g.dy; S.g.rdz = 1.0/S.g.dz;
  rd.read_into<1>("uh", S.g.uh); rd.read_into<1>("uf", S.g.uf);
  rd.read_into<1>("vh", S.g.vh); rd.read_into<1>("vf", S.g.vf);
  const std::vector<std::pair<std::string, FArray3>> f3 = {
    {"mh", S.g.mh}, {"mf", S.g.mf}, {"c1", S.g.c1}, {"c2", S.g.c2}, {"rho0", S.base.rho0},
    {"rr0", S.base.rr0}, {"rf0", S.base.rf0}, {"rrf0", S.base.rrf0}, {"rho", S.rho.d}};
  for (const auto& [n, a] : f3) rd.read_into<3>(n, a);
  for (auto* p : {&S.uh, &S.uf, &S.vh, &S.vf}) p->to_host();
  for (auto* p : {&S.mh, &S.mf, &S.c1, &S.c2, &S.rho0, &S.rr0, &S.rf0, &S.rrf0, &S.rho}) p->to_host();

  auto u3d = make_pair_arr<3>("u3d", {b.ib, b.jb, b.kb}, {b.ie + 1, b.je, b.ke});
  auto v3d = make_pair_arr<3>("v3d", {b.ib, b.jb, b.kb}, {b.ie, b.je + 1, b.ke});
  auto w3d = make_pair_arr<3>("w3d", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
  auto tha = make_pair_arr<3>("tha", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
  rd.read_into<3>("ua", u3d.d); rd.read_into<3>("va", v3d.d); rd.read_into<3>("wa", w3d.d);
  rd.read_into<3>("tha", tha.d);
  u3d.to_host(); v3d.to_host(); w3d.to_host(); tha.to_host();

  // solve2: rru, rrv, rrw (no terrain) and divx, same ranges
  const int ni = b.ni, nj = b.nj, nk = b.nk;
  for (auto* p : {&S.rru, &S.rrv, &S.rrw, &S.divx}) Kokkos::deep_copy(p->h.view(), 0.0);
  for (int k = 1; k <= nk; ++k) {
    for (int j = 0; j <= nj + 1; ++j) for (int i = 0; i <= ni + 2; ++i) S.rru.h(i, j, k) = S.rho0.h(1, 1, k)*u3d.h(i, j, k);
    for (int j = 0; j <= nj + 2; ++j) for (int i = 0; i <= ni + 1; ++i) S.rrv.h(i, j, k) = S.rho0.h(1, 1, k)*v3d.h(i, j, k);
    for (int j = 0; j <= nj + 1; ++j) for (int i = 0; i <= ni + 1; ++i)
      S.rrw.h(i, j, k) = (k == 1) ? 0.0 : S.rf0.h(1, 1, k)*w3d.h(i, j, k);
  }
  for (int j = 0; j <= nj + 1; ++j) for (int i = 0; i <= ni + 1; ++i) S.rrw.h(i, j, nk + 1) = 0.0;
  for (int k = 1; k <= nk; ++k) for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i)
    S.divx.h(i, j, k) = ( (S.rru.h(i + 1, j, k)-S.rru.h(i, j, k))*S.g.rdx*S.uh.h(i)
                         +(S.rrv.h(i, j + 1, k)-S.rrv.h(i, j, k))*S.g.rdy*S.vh.h(j) )
                         +(S.rrw.h(i, j, k + 1)-S.rrw.h(i, j, k))*S.g.rdz*S.mh.h(1, 1, k);
  for (size_t n = 0; n < S.rr.h.size(); ++n) S.rr.h.data()[n] = 1.0/S.rho.h.data()[n];
  S.sync();
  S.fortran_setup();

  Advection adv(S.g, S.base, S.o);
  const auto f = S.fluxes();
  const Real dt = nl.getr("dtl", 1.0);
  for (int nrk = 1; nrk <= 3; ++nrk) {
    INFO(case_name << " step " << nstep << " nrk " << nrk);
    auto uten = zeros_like(u3d), vten = zeros_like(v3d), wten = zeros_like(w3d), thten = zeros_like(tha);
    auto uf_ = hcopy(uten.h), vf_ = hcopy(vten.h), wf_ = hcopy(wten.h), tf_ = hcopy(thten.h);
    adv.advu(nrk, dt, f, u3d.d, uten.d);
    adv.advv(nrk, dt, f, v3d.d, vten.d);
    adv.advw(nrk, dt, f, w3d.d, wten.d);
    AdvsArgs a;
    a.weps = 10.0*phys::epsilon; a.hadvorder = S.o.hadvordrs; a.vadvorder = S.o.vadvordrs;
    a.diffit = (S.o.idiff == 1 && S.o.difforder == 6) ? 1 : 0;
    adv.advs(nrk, a, dt, f, tha.d, tha.d, thten.d);
    uten.to_host(); vten.to_host(); wten.to_host(); thten.to_host();

#define CM1F_ARGS S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.c1.p(), S.c2.p(), \
    S.rho0.p(), S.rr0.p(), S.rf0.p(), S.rrf0.p(), S.rru.p(), S.rrv.p(), S.rrw.p(), S.divx.p(), S.rho.p(), \
    u3d.p(), v3d.p(), w3d.p()
    cm1f_advu(nrk, dt, CM1F_ARGS, uf_.data());
    cm1f_advv(nrk, dt, CM1F_ARGS, vf_.data());
    cm1f_advw(nrk, dt, S.o.hadvordrv, S.o.vadvordrv, S.o.advwenov, 1, CM1F_ARGS, wf_.data());
#undef CM1F_ARGS
    cm1f_advs(nrk, 1, 0, 0, dt, a.weps, a.diffit, a.hadvorder, a.vadvorder,
              S.uh.p(), S.vh.p(), S.mh.p(), S.c1.p(), S.c2.p(), S.rho0.p(), S.rr0.p(), S.rf0.p(), S.rrf0.p(),
              S.rru.p(), S.rrv.p(), S.rrw.p(), S.divx.p(), S.rho.p(), S.rr.p(), tha.p(), tha.p(), tf_.data());
    require_same(uten.h, uf_, "advu");
    require_same(vten.h, vf_, "advv");
    require_same(wten.h, wf_, "advw");
    require_same(thten.h, tf_, "advs(tha)");

    // moisture: pdef at the last RK stage, pdefweno for q /= qv (solve2)
    for (const auto& name : rd.names()) {
      if (name.rfind("qa_", 0) != 0) continue;
      auto q = zeros_like(tha);
      rd.read_into<3>(name, q.d); q.to_host();
      auto qten = zeros_like(tha);
      auto qf = hcopy(qten.h);
      AdvsArgs aq = a;
      aq.pdef = (nrk == phys::nrkmax) ? 1 : 0;
      aq.pdefweno = (name == "qa_qv") ? 0 : 1;
      aq.weps = 1.0e-3*phys::epsilon;
      adv.advs(nrk, aq, dt, f, q.d, q.d, qten.d);
      qten.to_host();
      cm1f_advs(nrk, 1, aq.pdef, aq.pdefweno, dt, aq.weps, aq.diffit, aq.hadvorder, aq.vadvorder,
                S.uh.p(), S.vh.p(), S.mh.p(), S.c1.p(), S.c2.p(), S.rho0.p(), S.rr0.p(), S.rf0.p(), S.rrf0.p(),
                S.rru.p(), S.rrv.p(), S.rrw.p(), S.divx.p(), S.rho.p(), S.rr.p(), q.p(), q.p(), qf.data());
      require_same(qten.h, qf, "advs(" + name + ")");
    }
  }
}

} // namespace

TEST_CASE("advection on reference-dump fields matches adv.F", "[bridge][adv][refs]") {
  for (int nstep : {0, 10}) {
    run_dump_case("dry_shear", nstep);
    run_dump_case("kessler_bubble", nstep);
  }
  // Late enough that cloud and rain water are nonzero (exercises pdef on qc/qr).
  run_dump_case("kessler_bubble", 150);
}
