// Component tests of the LES turbulence port (physics/turb/cm1k_turb.*)
// against CM1's turb.F / turbtend.F / bc.F (and verbatim copies of the
// inline solve1.F code), called through cm1f_bridge_turb.F90.
//
// Both sides get identical inputs (random but plausible fields with halos,
// or fields from the kessler_tke reference dumps) and every output array is
// compared bit-for-bit over its whole extent (points a routine does not
// touch must stay untouched too).
#include "physics/turb/cm1k_turb.hpp"
#include "init/cm1k_init.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_namelist.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>

extern "C" {
void cm1f_turb_setup(int ni, int nj, int nk, int ngxy, int ngz, const int* iopt, const double* ropt);
void cm1f_turb_bcw(double* w, int flag);
void cm1f_turb_bc2d_level(double* w, int k);
void cm1f_turb_calcnm(const double* c1, const double* c2, const double* mf, const double* pi0,
                      const double* th0, const double* prs, const double* pp, const double* th,
                      const double* qa, double* nm);
void cm1f_turb_calcdef(const double* uh, const double* uf, const double* vh, const double* vf,
                       const double* mh, const double* c1, const double* c2, const double* mf,
                       const double* rho, const double* rr, const double* rf, const double* ua,
                       const double* va, const double* wa, double* s11, double* s12, double* s13,
                       double* s22, double* s23, double* s33, double* defv, double* defh);
void cm1f_turb_tkekm(const double* ruh, const double* rvh, const double* rmh, const double* zf,
                     const double* mf, const double* rmf, const double* rf, const double* nm,
                     const double* defv, const double* defh, double* kmh, double* kmv, double* khh,
                     double* khv, const double* cme, const double* csm, const double* ce1,
                     const double* ce2, double* tkea, double* lenscl, double* dissten);
void cm1f_turb_tkebc(double* tkea, double* kmh, double* kmv, double* khh, double* khv);
void cm1f_turb_gettau(const double* rf, const double* mf, const double* kmh, const double* kmv,
                      const double* ua, const double* va, const double* wa, double* t11, double* t12,
                      double* t13, double* t22, double* t23, double* t33);
void cm1f_turb_turbt(double dt, const double* uh, const double* uf, const double* vh, const double* vf,
                     const double* mh, const double* mf, const double* rho, const double* rr,
                     const double* rf, const double* t, double* tten, const double* kmh, const double* kmv);
void cm1f_turb_turbu(double dt, const double* ruh, const double* uf, const double* vh, const double* mh,
                     const double* mf, const double* rmf, const double* rho, const double* rf,
                     const double* u, double* uten, const double* w, const double* t11, const double* t12,
                     const double* t13, const double* t22, const double* kmv);
void cm1f_turb_turbv(double dt, const double* uh, const double* rvh, const double* vf, const double* mh,
                     const double* mf, const double* rho, const double* rr, const double* rf,
                     const double* v, double* vten, const double* w, const double* t12, const double* t22,
                     const double* t23, const double* kmv);
void cm1f_turb_turbw(double dt, const double* uh, const double* vh, const double* mh, const double* mf,
                     const double* rho, const double* rf, const double* w, double* wten, const double* t13,
                     const double* t23, const double* t33, const double* t22, const double* kmh);
void cm1f_turb_turbs(int iflux, double dt, const double* uh, const double* uf, const double* vh,
                     const double* vf, const double* mh, const double* mf, const double* rho,
                     const double* rr, const double* rf, const double* s, double* sten,
                     const double* khh, const double* khv, const double* dum7, const double* dum8);
void cm1f_turb_tke_source(const double* nm, const double* defv, const double* defh, const double* dissten,
                          const double* kmh, const double* kmv, const double* khv, double* tketen);
void cm1f_turb_dissheat(const double* pi0, const double* ppi, const double* qa, const double* dissten,
                        double* thten1);
void cm1f_turb_vimpl(const double* khv, const double* mf, const double* rf, const double* mh,
                     const double* rr, double* dum7, double* dum8);
void cm1f_turb_ramp(double rtime, double ramp_time, const double* cm0, double* cme, double* csm,
                    double* ce1, double* ce2);
void cm1f_turb_tke_update(double dttmp, const double* tkea, const double* tten, double* tke3d);
// bbc=3 / surface fluxes (cm1f_bridge_sfc.F90, cm1f_bridge_turbpriv.F90)
void cm1f_sfc_options(const int* iopt, const double* ropt);
void cm1f_sfc_tkebc(double* tkea, double* kmh, double* kmv, double* khh, double* khv, const double* zntmp,
                    const double* ust);
void cm1f_sfc_gettau(const double* rf, const double* mf, const double* kmh, const double* kmv, const double* ua,
                     const double* va, const double* wa, const double* ust, const double* u1, const double* v1,
                     const double* s1, double* t11, double* t12, double* t13, double* t22, double* t23,
                     double* t33);
void cm1f_sfc_turbs(int iflux, double dt, const double* uh, const double* uf, const double* vh,
                    const double* vf, const double* mh, const double* mf, const double* rho,
                    const double* rr, const double* rf, const double* s, double* sten,
                    const double* khh, const double* khv, const double* dum7, const double* dum8,
                    const double* sflux);
void cm1f_sfc_dissheat(const double* pi0, const double* ppi, const double* qa, const double* dissten,
                       const double* zf, const double* ust, const double* zntmp, double* thten1);
}

using namespace cm1k;
namespace T = cm1k::turb;

namespace {

using H3 = FArray<3, HostSpace>;
using H4 = FArray<4, HostSpace>;

// Host and device copies of one array (they alias on host-only builds).
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
Pair<R> wrap(const FArray<R>& d) { Pair<R> p; p.d = d; p.h = d.to_host(); return p; }
template <int R>
Pair<R> mk(const std::string& n, const std::array<int, R>& lb, const std::array<int, R>& ub) {
  Pair<R> p;
  p.d = FArray<R>(n, lb, ub);
  p.h = p.d.to_host();
  return p;
}

// Deep host copy (for the Fortran side of inout arrays).
template <int R>
FArray<R, HostSpace> hcopy(const FArray<R, HostSpace>& h) {
  std::array<int, R> lb, ub;
  for (int d = 0; d < R; ++d) { lb[d] = h.lb(d); ub[d] = h.ub(d); }
  FArray<R, HostSpace> c("copy", lb, ub);
  std::memcpy(c.data(), h.data(), h.size() * sizeof(double));
  return c;
}

// Bit-for-bit comparison (NaN == NaN); reports the max abs difference.
template <int R>
double require_same(const FArray<R, HostSpace>& a, const FArray<R, HostSpace>& ref, const std::string& what) {
  REQUIRE(a.size() == ref.size());
  size_t nbad = 0, first = 0;
  double maxd = 0, maxref = 0;
  for (size_t n = 0; n < a.size(); ++n) {
    const double x = a.data()[n], y = ref.data()[n];
    if (std::isfinite(y)) maxref = std::max(maxref, std::fabs(y));
    if (!(x == y) && !(std::isnan(x) && std::isnan(y))) {
      if (nbad == 0) first = n;
      ++nbad;
      maxd = std::max(maxd, std::fabs(x - y));
    }
  }
  INFO(what << ": " << nbad << " differing points (first at flat index " << first << "), max |diff| = " << maxd
            << " (max |ref| = " << maxref << ")");
  REQUIRE(nbad == 0);
  return maxref;
}

double U(std::mt19937_64& gen) { return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); }

void fill(H3& a, std::mt19937_64& gen, double mean, double amp) {
  for (size_t n = 0; n < a.size(); ++n) a.data()[n] = mean + amp * U(gen);
}

// Moisture layouts (param.F)
enum class Moist { dry, kessler, ice };

struct Case {
  int tconfig = 1, bcturbs = 1, doimpl = 1;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  Moist moist = Moist::kessler;
  int ptype = 1;        // for Moist::ice: 2 (Goddard layout) or 4
  int eqtset = 2, idiss = 1;
  std::string str() const {
    std::ostringstream s;
    s << "tconfig=" << tconfig << " bcturbs=" << bcturbs << " doimpl=" << doimpl << " bc=" << wbc << ebc << sbc << nbc
      << " moist=" << int(moist) << " ptype=" << ptype << " eqtset=" << eqtset << " idiss=" << idiss;
    return s.str();
  }
};

T::Options options_for(const Case& c) {
  T::Options o;
  o.tconfig = c.tconfig; o.bcturbs = c.bcturbs; o.doimpl = c.doimpl;
  o.wbc = c.wbc; o.ebc = c.ebc; o.sbc = c.sbc; o.nbc = c.nbc;
  o.set_boundary_flags();
  o.eqtset = c.eqtset; o.idiss = c.idiss;
  switch (c.moist) {
    case Moist::dry:
      o.imoist = 0; o.ptype = 0; o.iice = 0; o.numq = 1; o.nqv = 1; o.nqc = 0; o.nqi = 0;
      o.nql1 = 0; o.nql2 = 0; o.nqs1 = 0; o.nqs2 = 0;
      break;
    case Moist::kessler:
      o.imoist = 1; o.ptype = 1; o.iice = 0; o.numq = 3; o.nqv = 1; o.nqc = 2; o.nqi = 0;
      o.nql1 = 2; o.nql2 = 3; o.nqs1 = -2; o.nqs2 = -2;
      break;
    case Moist::ice:   // qv qc qr qi qs qg
      o.imoist = 1; o.ptype = c.ptype; o.iice = 1; o.numq = 6; o.nqv = 1; o.nqc = 2; o.nqi = 4;
      o.nql1 = 2; o.nql2 = 3; o.nqs1 = 4; o.nqs2 = 6;
      break;
  }
  return o;
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

// Everything the turbulence routines read, on both sides.
struct Setup {
  Bounds b;
  Grid g;
  BaseState base;
  T::Options o;
  Constants c;
  T::TurbState ts;
  T::Work w;
  int testcase = 0;
  Pair<1> uh, ruh, uf, vh, rvh, vf;
  Pair<3> mh, rmh, mf, rmf, zf, c1, c2, pi0, th0;
  Pair<3> rho, rr, rf, ua, va, wa, ppi, tha, prs, tkea, kmh, kmv, khh, khv;
  Pair<4> qa;
  Pair<2> cm0;
  Pair<3> cme, csm, ce1, ce2, nm, defv, defh, lenscl, dissten, t11, t12, t13, t22, t23, t33;

  Setup(const Bounds& bb, const T::Options& oo) : b(bb), o(oo), c(Constants::make(0)) {
    g.allocate(b);
    g.b = b;
    base.allocate(b);
    ts.allocate(b);
    ts.init_constants(b);
    w.allocate(b);
    uh = wrap(g.uh); ruh = wrap(g.ruh); uf = wrap(g.uf); vh = wrap(g.vh); rvh = wrap(g.rvh); vf = wrap(g.vf);
    mh = wrap(g.mh); rmh = wrap(g.rmh); mf = wrap(g.mf); rmf = wrap(g.rmf); zf = wrap(g.zf);
    c1 = wrap(g.c1); c2 = wrap(g.c2); pi0 = wrap(base.pi0); th0 = wrap(base.th0);
    const std::array<int, 3> s{b.ib, b.jb, b.kb}, se{b.ie, b.je, b.ke};
    rho = mk<3>("rho", s, se); rr = mk<3>("rr", s, se); rf = mk<3>("rf", s, se);
    ua = mk<3>("ua", s, {b.ie + 1, b.je, b.ke});
    va = mk<3>("va", s, {b.ie, b.je + 1, b.ke});
    wa = mk<3>("wa", s, {b.ie, b.je, b.ke + 1});
    ppi = mk<3>("ppi", s, se); tha = mk<3>("tha", s, se); prs = mk<3>("prs", s, se);
    tkea = mk<3>("tkea", {b.ibt, b.jbt, b.kbt}, {b.iet, b.jet, b.ket});
    const std::array<int, 3> lc{b.ibc, b.jbc, b.kbc}, uc{b.iec, b.jec, b.kec};
    kmh = mk<3>("kmh", lc, uc); kmv = mk<3>("kmv", lc, uc); khh = mk<3>("khh", lc, uc); khv = mk<3>("khv", lc, uc);
    qa = mk<4>("qa", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, o.numq});
    cm0 = wrap(ts.cm0);
    cme = wrap(ts.cme); csm = wrap(ts.csm); ce1 = wrap(ts.ce1); ce2 = wrap(ts.ce2);
    nm = wrap(ts.nm); defv = wrap(ts.defv); defh = wrap(ts.defh); lenscl = wrap(ts.lenscl); dissten = wrap(ts.dissten);
    t11 = wrap(ts.t11); t12 = wrap(ts.t12); t13 = wrap(ts.t13); t22 = wrap(ts.t22); t23 = wrap(ts.t23); t33 = wrap(ts.t33);
  }

  std::vector<Pair<3>*> all3() {
    return {&mh, &rmh, &mf, &rmf, &zf, &c1, &c2, &pi0, &th0, &rho, &rr, &rf, &ua, &va, &wa, &ppi, &tha, &prs,
            &tkea, &kmh, &kmv, &khh, &khv, &cme, &csm, &ce1, &ce2, &nm, &defv, &defh, &lenscl, &dissten,
            &t11, &t12, &t13, &t22, &t23, &t33};
  }
  void sync_to_dev() {
    for (auto* p : {&uh, &ruh, &uf, &vh, &rvh, &vf}) p->to_dev();
    for (auto* p : all3()) p->to_dev();
    qa.to_dev(); cm0.to_dev();
  }
  void sync_to_host() {
    for (auto* p : {&uh, &ruh, &uf, &vh, &rvh, &vf}) p->to_host();
    for (auto* p : all3()) p->to_host();
    qa.to_host(); cm0.to_host();
  }

  // Random, physically plausible fields (halos included).
  void randomize(std::mt19937_64& gen) {
    g.dx = 1000.0; g.dy = 900.0; g.dz = 500.0;
    g.rdx = 1.0 / g.dx; g.rdy = 1.0 / g.dy; g.rdz = 1.0 / g.dz;
    for (int i = b.ib; i <= b.ie; ++i) { uh.h(i) = 1.0 + 0.2 * U(gen); ruh.h(i) = 1.0 / uh.h(i); }
    for (int i = b.ib; i <= b.ie + 1; ++i) uf.h(i) = 1.0 + 0.2 * U(gen);
    for (int j = b.jb; j <= b.je; ++j) { vh.h(j) = 1.0 + 0.2 * U(gen); rvh.h(j) = 1.0 / vh.h(j); }
    for (int j = b.jb; j <= b.je + 1; ++j) vf.h(j) = 1.0 + 0.2 * U(gen);
    const int nk = b.nk;
    for (int k = b.kb; k <= b.ke + 1; ++k)
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          mf.h(i, j, k) = 1.0 + 0.3 * U(gen);
          rmf.h(i, j, k) = 1.0 / mf.h(i, j, k);
          zf.h(i, j, k) = 500.0 * (k - 1);
          if (k > b.ke) continue;
          const double z = 0.08 * k;
          mh.h(i, j, k) = 1.0 + 0.3 * U(gen);
          rmh.h(i, j, k) = 1.0 / mh.h(i, j, k);
          c2.h(i, j, k) = 0.5 + 0.1 * U(gen);
          c1.h(i, j, k) = 1.0 - c2.h(i, j, k);
          // pi0 from 1 down to ~0.62: T spans ~300 K to ~205 K, so every
          // branch of calcnm (warm, cold, mixed-phase) is reached
          pi0.h(i, j, k) = 1.0 - 0.38 * (k - 0.5) / nk;
          th0.h(i, j, k) = 300.0 + 2.0 * k;
          rho.h(i, j, k) = 1.2 * std::exp(-z) * (1.0 + 0.01 * U(gen));
          rr.h(i, j, k) = 1.0 / rho.h(i, j, k);
          rf.h(i, j, k) = 1.2 * std::exp(-z + 0.04) * (1.0 + 0.01 * U(gen));
          tha.h(i, j, k) = 0.8 * U(gen);
          ppi.h(i, j, k) = 1.0e-3 * U(gen);
          prs.h(i, j, k) = 1.0e5 * std::pow(pi0.h(i, j, k) + ppi.h(i, j, k), 1005.7 / 287.04);
        }
    fill(ua.h, gen, 5.0, 3.0);
    fill(va.h, gen, -2.0, 3.0);
    fill(wa.h, gen, 0.0, 1.0);
    std::uniform_int_distribution<int> I(0, 9);
    for (size_t n = 0; n < tkea.h.size(); ++n) {
      double v = 0.5 * (1.0 + U(gen));
      const int r = I(gen);
      if (r == 0) v = 0.0;
      if (r == 1) v = 1.0e-11;
      tkea.h.data()[n] = v;
    }
    for (auto* p : {&kmh, &kmv, &khh, &khv}) fill(p->h, gen, 60.0, 30.0);
    fill(nm.h, gen, 1.0e-4, 2.0e-4);
    for (size_t n = 0; n < nm.h.size(); ++n) if (I(gen) == 0) nm.h.data()[n] = 1.0e-6;   // nm == nm_min
    fill(defv.h, gen, 1.0e-4, 2.0e-4);
    fill(defh.h, gen, 1.0e-4, 2.0e-4);
    fill(dissten.h, gen, 1.0e-3, 1.0e-3);
    fill(lenscl.h, gen, 100.0, 50.0);
    for (auto* p : {&t11, &t12, &t13, &t22, &t23, &t33}) fill(p->h, gen, 0.0, 0.1);
    // moisture: near saturation, some cloud (and ice), some exact zeros
    for (int n = 1; n <= o.numq; ++n)
      for (int k = b.kbm; k <= b.kem; ++k)
        for (int j = b.jbm; j <= b.jem; ++j)
          for (int i = b.ibm; i <= b.iem; ++i) {
            double v;
            if (n == o.nqv) v = 0.02 * std::exp(-0.28 * k) * (1.0 + 0.4 * U(gen));
            else v = std::max(0.0, 1.0e-3 * U(gen));
            qa.h(i, j, k, n) = v;
          }
    // cme..ce2: init3d values on most points, a few ramped ones
    for (int k = b.kbc; k <= b.kec; ++k)
      for (int j = b.jbc; j <= b.jec; ++j)
        for (int i = b.ibc; i <= b.iec; ++i)
          if (I(gen) == 0) {
            cme.h(i, j, k) = 0.05 * (1.0 + 0.5 * U(gen));
            ce1.h(i, j, k) = cme.h(i, j, k) * 0.8165 * 0.8165 * 3.0;
            ce2.h(i, j, k) = std::max(0.0, cme.h(i, j, k) * phys::pi * phys::pi - ce1.h(i, j, k));
          }
    sync_to_dev();
  }

  void fortran_setup() const {
    const int iopt[21] = {o.tconfig, o.bcturbs, o.doimpl, o.wbc, o.ebc, o.sbc, o.nbc, o.imoist, o.ptype,
                          o.iice, o.numq, o.nqv, o.nqc, o.nqi, o.nql1, o.nql2, o.nqs1, o.nqs2,
                          o.eqtset, o.idiss, testcase};
    const double ropt[3] = {g.dx, g.dy, g.dz};
    cm1f_turb_setup(b.ni, b.nj, b.nk, b.ngxy, b.ngz, iopt, ropt);
  }
};

template <int R>
Pair<R> zeros_like(const Pair<R>& like) {
  std::array<int, R> lb, ub;
  for (int d = 0; d < R; ++d) { lb[d] = like.h.lb(d); ub[d] = like.h.ub(d); }
  auto p = mk<R>("zeros", lb, ub);
  Kokkos::deep_copy(p.h.view(), 0.0);
  p.to_dev();
  return p;
}

template <int R>
Pair<R> random_like(const Pair<R>& like, std::mt19937_64& gen, double mean, double amp) {
  auto p = zeros_like(like);
  for (size_t n = 0; n < p.h.size(); ++n) p.h.data()[n] = mean + amp * U(gen);
  p.to_dev();
  return p;
}

const std::vector<std::array<int, 4>> kLateral = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {2, 3, 4, 2}};

Case lateral_case(const std::array<int, 4>& bc) {
  Case c; c.wbc = bc[0]; c.ebc = bc[1]; c.sbc = bc[2]; c.nbc = bc[3];
  return c;
}

//------------------------------------------------------------------------------
// one test per routine, random fields

void check_bcw(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 12, 3, 1), options_for(cs));
  S.randomize(gen);
  S.fortran_setup();
  for (int flag : {0, 1}) {
    auto w = random_like(S.kmh, gen, 1.0, 1.0);
    auto wf = hcopy(w.h);
    T::bcw(S.o, S.b, w.d, flag);
    w.to_host();
    cm1f_turb_bcw(wf.data(), flag);
    require_same(w.h, wf, "bcw flag=" + std::to_string(flag));
  }
  for (int k : {1, 5}) {
    auto w = random_like(S.kmh, gen, 1.0, 1.0);
    auto wf = hcopy(w.h);
    T::bc2d_level(S.o, S.b, w.d, k);
    w.to_host();
    cm1f_turb_bc2d_level(wf.data(), k);
    require_same(w.h, wf, "bc2d level " + std::to_string(k));
  }
}

void check_calcnm(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 14, 3, options_for(cs).numq), options_for(cs));
  S.randomize(gen);
  S.fortran_setup();
  auto nmf = hcopy(S.nm.h);
  T::calcnm(S.o, S.c, S.b, S.g.rdz, S.g.c1, S.g.c2, S.g.mf, S.base.pi0, S.base.th0, S.ts.nm,
            S.w.d1, S.w.d2, S.w.d3, S.w.d6, S.prs.d, S.ppi.d, S.tha.d, S.qa.d);
  S.nm.to_host();
  cm1f_turb_calcnm(S.c1.p(), S.c2.p(), S.mf.p(), S.pi0.p(), S.th0.p(), S.prs.p(), S.ppi.p(), S.tha.p(),
                   S.qa.p(), nmf.data());
  require_same(S.nm.h, nmf, "calcnm");
  if (S.o.imoist == 1) {
    // the saturated formulas must actually have been exercised: count points
    // where nm is not the subsaturated value, by temperature range
    const auto t = S.w.d1.to_host(), thv = S.w.d3.to_host();
    int nsat[3] = {0, 0, 0}, npts[3] = {0, 0, 0};
    for (int k = 2; k <= S.b.nk; ++k)
      for (int j = 1; j <= S.b.nj; ++j)
        for (int i = 1; i <= S.b.ni; ++i) {
          const double tavg = S.c1.h(i, j, k) * t(i, j, k - 1) + S.c2.h(i, j, k) * t(i, j, k);
          const int r = tavg >= 273.15 ? 0 : (tavg <= 233.15 ? 1 : 2);
          ++npts[r];
          const double sub = S.c.g * std::log(thv(i, j, k) / thv(i, j, k - 1)) * S.g.rdz * S.mf.h(i, j, k);
          if (S.nm.h(i, j, k) != sub) ++nsat[r];
        }
    INFO("saturated points warm/cold/mixed: " << nsat[0] << "/" << npts[0] << " " << nsat[1] << "/" << npts[1]
         << " " << nsat[2] << "/" << npts[2]);
    if (S.o.iice == 1) { REQUIRE(nsat[0] > 0); REQUIRE(nsat[1] > 0); REQUIRE(nsat[2] > 0); }
    else REQUIRE(nsat[0] + nsat[1] + nsat[2] > 0);   // liquid-only formula at all temperatures
  }
}

void check_calcdef(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 12, 3, 1), options_for(cs));
  S.randomize(gen);
  S.fortran_setup();
  std::vector<FArray<3, HostSpace>> f;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33, &S.defv, &S.defh}) f.push_back(hcopy(p->h));
  T::calcdef(S.o, S.g, S.rho.d, S.rf.d, S.ua.d, S.va.d, S.wa.d, S.ts.t11, S.ts.t12, S.ts.t13, S.ts.t22,
             S.ts.t23, S.ts.t33, S.ts.defv, S.ts.defh);
  S.sync_to_host();
  cm1f_turb_calcdef(S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.c1.p(), S.c2.p(), S.mf.p(),
                    S.rho.p(), S.rr.p(), S.rf.p(), S.ua.p(), S.va.p(), S.wa.p(), f[0].data(), f[1].data(),
                    f[2].data(), f[3].data(), f[4].data(), f[5].data(), f[6].data(), f[7].data());
  const char* names[] = {"s11", "s12", "s13", "s22", "s23", "s33", "defv", "defh"};
  int n = 0;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33, &S.defv, &S.defh}) {
    require_same(p->h, f[n], std::string("calcdef ") + names[n]);
    ++n;
  }
}

void check_tkekm_tkebc(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 12, 3, 1), options_for(cs));
  S.randomize(gen);
  S.fortran_setup();
  std::vector<FArray<3, HostSpace>> f;
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea, &S.lenscl, &S.dissten}) f.push_back(hcopy(p->h));
  T::tkekm(S.o, S.g, S.ts.nm, S.w.d1, S.w.d3, S.w.d4, S.w.d5, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d,
           S.ts.cme, S.ts.ce1, S.ts.ce2, S.tkea.d, S.ts.lenscl, S.ts.dissten);
  S.sync_to_host();
  cm1f_turb_tkekm(S.ruh.p(), S.rvh.p(), S.rmh.p(), S.zf.p(), S.mf.p(), S.rmf.p(), S.rf.p(), S.nm.p(),
                  S.defv.p(), S.defh.p(), f[0].data(), f[1].data(), f[2].data(), f[3].data(), S.cme.p(),
                  S.csm.p(), S.ce1.p(), S.ce2.p(), f[4].data(), f[5].data(), f[6].data());
  const char* names[] = {"kmh", "kmv", "khh", "khv", "tkea", "lenscl", "dissten"};
  int n = 0;
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea, &S.lenscl, &S.dissten}) {
    require_same(p->h, f[n], std::string("tkekm ") + names[n]);
    ++n;
  }
  // tkebc on the result
  T::tkebc(S.o, S.b, S.tkea.d, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d);
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea}) p->to_host();
  cm1f_turb_tkebc(f[4].data(), f[0].data(), f[1].data(), f[2].data(), f[3].data());
  n = 0;
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea}) {
    require_same(p->h, f[n], std::string("tkebc ") + names[n]);
    ++n;
  }
}

void check_gettau(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 12, 3, 1), options_for(cs));
  S.randomize(gen);
  S.fortran_setup();
  std::vector<FArray<3, HostSpace>> f;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33}) f.push_back(hcopy(p->h));
  T::gettau(S.o, S.b, S.kmh.d, S.kmv.d, S.ts.t11, S.ts.t12, S.ts.t13, S.ts.t22, S.ts.t23, S.ts.t33);
  S.sync_to_host();
  cm1f_turb_gettau(S.rf.p(), S.mf.p(), S.kmh.p(), S.kmv.p(), S.ua.p(), S.va.p(), S.wa.p(), f[0].data(),
                   f[1].data(), f[2].data(), f[3].data(), f[4].data(), f[5].data());
  const char* names[] = {"t11", "t12", "t13", "t22", "t23", "t33"};
  int n = 0;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33}) {
    require_same(p->h, f[n], std::string("gettau ") + names[n]);
    ++n;
  }
}

void check_tendencies(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  const auto o = options_for(cs);
  Setup S(box(10, 9, 12, 3, o.numq), o);
  S.randomize(gen);
  S.fortran_setup();
  const double dt = 6.0;

  // tke_source + turbt
  auto tketen = random_like(S.tkea, gen, 0.0, 1.0e-3);
  auto tkf = hcopy(tketen.h);
  T::tke_source(S.o, S.b, S.ts.nm, S.ts.defv, S.ts.defh, S.ts.dissten, S.kmh.d, S.kmv.d, S.khv.d, tketen.d);
  tketen.to_host();
  cm1f_turb_tke_source(S.nm.p(), S.defv.p(), S.defh.p(), S.dissten.p(), S.kmh.p(), S.kmv.p(), S.khv.p(), tkf.data());
  require_same(tketen.h, tkf, "tke_source");
  T::turbt(S.o, S.g, dt, S.rho.d, S.rf.d, S.tkea.d, tketen.d, S.kmh.d, S.kmv.d, S.w);
  tketen.to_host();
  cm1f_turb_turbt(dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(),
                  S.tkea.p(), tkf.data(), S.kmh.p(), S.kmv.p());
  require_same(tketen.h, tkf, "turbt");

  // dissipative heating
  auto thten = random_like(S.tha, gen, 0.0, 1.0e-3);
  auto thf = hcopy(thten.h);
  T::dissheat(S.o, S.c, S.b, S.base.pi0, S.ppi.d, S.qa.d, S.ts.dissten, thten.d, S.w.d9);
  thten.to_host();
  cm1f_turb_dissheat(S.pi0.p(), S.ppi.p(), S.qa.p(), S.dissten.p(), thf.data());
  require_same(thten.h, thf, "dissheat");

  // momentum
  auto uten = random_like(S.ua, gen, 0.0, 1.0e-3), vten = random_like(S.va, gen, 0.0, 1.0e-3),
       wten = random_like(S.wa, gen, 0.0, 1.0e-3);
  auto uf_ = hcopy(uten.h), vf_ = hcopy(vten.h), wf_ = hcopy(wten.h);
  T::turbu(S.o, S.g, dt, S.rho.d, S.rf.d, S.ua.d, uten.d, S.wa.d, S.ts.t11, S.ts.t12, S.ts.t13, S.kmv.d, S.w);
  T::turbv(S.o, S.g, dt, S.rho.d, S.rf.d, S.va.d, vten.d, S.wa.d, S.ts.t12, S.ts.t22, S.ts.t23, S.kmv.d, S.w);
  T::turbw(S.o, S.g, dt, S.rho.d, S.rf.d, S.wa.d, wten.d, S.ts.t13, S.ts.t23, S.ts.t33, S.kmh.d, S.w);
  uten.to_host(); vten.to_host(); wten.to_host();
  cm1f_turb_turbu(dt, S.ruh.p(), S.uf.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rmf.p(), S.rho.p(), S.rf.p(),
                  S.ua.p(), uf_.data(), S.wa.p(), S.t11.p(), S.t12.p(), S.t13.p(), S.t22.p(), S.kmv.p());
  cm1f_turb_turbv(dt, S.uh.p(), S.rvh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(),
                  S.va.p(), vf_.data(), S.wa.p(), S.t12.p(), S.t22.p(), S.t23.p(), S.kmv.p());
  cm1f_turb_turbw(dt, S.uh.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rf.p(), S.wa.p(), wf_.data(),
                  S.t13.p(), S.t23.p(), S.t33.p(), S.t22.p(), S.kmh.p());
  require_same(uten.h, uf_, "turbu");
  require_same(vten.h, vf_, "turbv");
  require_same(wten.h, wf_, "turbw");

  // scalars: vimpl arrays, then turbs on theta (iflux=1) and each q
  auto d7 = zeros_like(S.tha), d8 = zeros_like(S.tha);
  auto d7f = hcopy(d7.h), d8f = hcopy(d8.h);
  T::vimpl_coefs(S.b, S.khv.d, S.g.mf, S.rf.d, S.g.mh, S.rr.d, d7.d, d8.d);
  d7.to_host(); d8.to_host();
  cm1f_turb_vimpl(S.khv.p(), S.mf.p(), S.rf.p(), S.mh.p(), S.rr.p(), d7f.data(), d8f.data());
  require_same(d7.h, d7f, "vimpl dum7");
  require_same(d8.h, d8f, "vimpl dum8");
  for (int iflux : {1, 0}) {
    auto sten = random_like(S.tha, gen, 0.0, 1.0e-3);
    auto sf = hcopy(sten.h);
    T::turbs(S.o, S.g, iflux, dt, S.rho.d, S.rr.d, S.rf.d, S.tha.d, sten.d, S.khh.d, S.khv.d, d7.d, d8.d, S.w);
    sten.to_host();
    cm1f_turb_turbs(iflux, dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(),
                    S.rf.p(), S.tha.p(), sf.data(), S.khh.p(), S.khv.p(), d7f.data(), d8f.data());
    require_same(sten.h, sf, "turbs iflux=" + std::to_string(iflux));
  }

  // solve2 TKE update
  auto tke3d = random_like(S.tkea, gen, 0.1, 0.1);
  auto t3f = hcopy(tke3d.h);
  T::tke_rk_update(S.o, S.b, 2.0, S.tkea.d, tketen.d, tke3d.d);
  tke3d.to_host();
  cm1f_turb_tke_update(2.0, S.tkea.p(), tkf.data(), t3f.data());
  require_same(tke3d.h, t3f, "tke_rk_update");
}

TEST_CASE("turb bcw matches bc.F", "[bridge][turb]") {
  unsigned seed = 11;
  for (const auto& bc : kLateral) check_bcw(lateral_case(bc), seed++);
}

TEST_CASE("turb calcnm matches turb.F", "[bridge][turb]") {
  unsigned seed = 21;
  Case dry; dry.moist = Moist::dry;
  check_calcnm(dry, seed++);
  Case kes; kes.moist = Moist::kessler;
  for (int r = 0; r < 3; ++r) check_calcnm(kes, seed++);
  Case ice; ice.moist = Moist::ice; ice.ptype = 2;
  for (int r = 0; r < 3; ++r) check_calcnm(ice, seed++);
  Case ice4; ice4.moist = Moist::ice; ice4.ptype = 4;
  check_calcnm(ice4, seed++);
}

TEST_CASE("turb calcdef matches turb.F", "[bridge][turb]") {
  unsigned seed = 31;
  for (const auto& bc : kLateral) check_calcdef(lateral_case(bc), seed++);
}

TEST_CASE("turb tkekm and tkebc match turb.F", "[bridge][turb]") {
  unsigned seed = 41;
  for (int tconfig : {1, 2})
    for (const auto& bc : kLateral) {
      Case c = lateral_case(bc); c.tconfig = tconfig;
      check_tkekm_tkebc(c, seed++);
    }
}

TEST_CASE("turb gettau matches turb.F", "[bridge][turb]") {
  unsigned seed = 51;
  for (const auto& bc : kLateral) check_gettau(lateral_case(bc), seed++);
}

TEST_CASE("turb tendencies match turbtend.F and solve1.F", "[bridge][turb]") {
  unsigned seed = 61;
  for (const auto& bc : kLateral)
    for (int doimpl : {0, 1})
      for (int bcturbs : {1, 2}) {
        Case c = lateral_case(bc); c.doimpl = doimpl; c.bcturbs = bcturbs;
        check_tendencies(c, seed++);
      }
  Case t2; t2.tconfig = 2; check_tendencies(t2, seed++);
  Case e1; e1.eqtset = 1; check_tendencies(e1, seed++);
  Case dry; dry.moist = Moist::dry; check_tendencies(dry, seed++);
  Case ice; ice.moist = Moist::ice; ice.ptype = 2; check_tendencies(ice, seed++);
  Case nod; nod.idiss = 0; check_tendencies(nod, seed++);
}

TEST_CASE("turb ramp_sgs matches sfc_and_turb", "[bridge][turb]") {
  std::mt19937_64 gen(71);
  T::Options o = options_for(Case{});
  o.ramp_sgs = 1; o.ramp_time = 1800.0;
  Setup S(box(8, 7, 10, 3, o.numq), o);
  S.randomize(gen);
  S.fortran_setup();
  for (double rtime : {0.0, 6.0, 900.0, 1794.0, 1800.0, 3600.0}) {
    INFO("rtime " << rtime);
    std::vector<FArray<3, HostSpace>> f;
    for (auto* p : {&S.cme, &S.csm, &S.ce1, &S.ce2}) f.push_back(hcopy(p->h));
    T::update_constants(S.o, S.b, rtime, S.ts);
    for (auto* p : {&S.cme, &S.csm, &S.ce1, &S.ce2}) p->to_host();
    cm1f_turb_ramp(rtime, 1800.0, S.cm0.p(), f[0].data(), f[1].data(), f[2].data(), f[3].data());
    require_same(S.cme.h, f[0], "cme");
    require_same(S.csm.h, f[1], "csm");
    require_same(S.ce1.h, f[2], "ce1");
    require_same(S.ce2.h, f[3], "ce2");
  }
}

TEST_CASE("turb options: unsupported values throw", "[turb]") {
  T::Options o;
  REQUIRE_NOTHROW(o.check());
  auto bad = [](auto f) { T::Options x; f(x); REQUIRE_THROWS(x.check()); };
  bad([](T::Options& x) { x.sgsmodel = 2; });
  bad([](T::Options& x) { x.cm1setup = 2; });
  bad([](T::Options& x) { x.ipbl = 1; });
  bad([](T::Options& x) { x.horizturb = 1; });
  bad([](T::Options& x) { x.tconfig = 3; });
  bad([](T::Options& x) { x.bcturbs = 3; });
  bad([](T::Options& x) { x.doimpl = 2; });
  bad([](T::Options& x) { x.bbc = 3; });                   // without sfcmodel=1
  bad([](T::Options& x) { x.bbc = 2; });
  bad([](T::Options& x) { x.sfcmodel = 1; });              // without bbc=3
  bad([](T::Options& x) { x.sfcmodel = 2; x.bbc = 3; });
  bad([](T::Options& x) { x.tbc = 2; });
  bad([](T::Options& x) { x.dosfcflx = true; });           // without sfcmodel=1
  bad([](T::Options& x) { x.terrain_flag = true; });
  bad([](T::Options& x) { x.axisymm = 1; });
  bad([](T::Options& x) { x.wbc = 5; });
  REQUIRE_THROWS(T::Options::from_namelist(Namelist::from_string("&param2\n sgsmodel = 2 /\n")).check());
  // the surface layer (physics/sfc): sfcmodel=1 sets bbc=3; isfcflx=1 needs it
  REQUIRE_NOTHROW(T::Options::from_namelist(Namelist::from_string("&param2\n isfcflx = 1, sfcmodel = 1 /\n")).check());
  REQUIRE(T::Options::from_namelist(Namelist::from_string("&param2\n isfcflx = 1, sfcmodel = 1 /\n")).dosfcflx);
  REQUIRE_THROWS(T::Options::from_namelist(Namelist::from_string("&param2\n isfcflx = 1 /\n")).check());
  REQUIRE_THROWS(T::Options::from_namelist(Namelist::from_string("&param2\n sfcmodel = 5 /\n")).check());
  REQUIRE_THROWS(T::Options::from_namelist(Namelist::from_string("&param2\n imoist = 1, ptype = 2 /\n")));
}

//------------------------------------------------------------------------------
// kessler_tke reference dumps

// Load a dump step into a Setup: grid, base state, state, and rr/rf as
// solve3.F computes them from rho.
std::unique_ptr<Setup> load_dump(const std::string& case_name, int nstep, double& mtime) {
  const auto root = testing::case_ref(case_name);
  if (root.empty()) return nullptr;
  const auto dir = DumpReader::step_dir(root, nstep);
  if (!std::filesystem::exists(dir)) { WARN("missing " << dir); return nullptr; }
  DumpReader rd(dir);
  mtime = rd.mtime();
  const auto nl = Namelist::from_file(testing::case_namelist(case_name));
  auto o = T::Options::from_namelist(nl);
  o.check();
  auto S = std::make_unique<Setup>(Bounds::from_namelist(nl), o);
  auto& b = S->b;
  REQUIRE(b.numq == o.numq);
  S->g.dx = nl.getr("dx", 0); S->g.dy = nl.getr("dy", 0); S->g.dz = nl.getr("dz", 0);
  S->g.rdx = 1.0 / S->g.dx; S->g.rdy = 1.0 / S->g.dy; S->g.rdz = 1.0 / S->g.dz;
  for (const auto& [n, a] : std::vector<std::pair<std::string, FArray1>>{
         {"uh", S->g.uh}, {"ruh", S->g.ruh}, {"uf", S->g.uf}, {"vh", S->g.vh}, {"rvh", S->g.rvh}, {"vf", S->g.vf},
         {"sigma", S->g.sigma}, {"sigmaf", S->g.sigmaf}})
    rd.read_into<1>(n, a);
  for (const auto& [n, a] : std::vector<std::pair<std::string, FArray3>>{
         {"mh", S->g.mh}, {"rmh", S->g.rmh}, {"mf", S->g.mf}, {"rmf", S->g.rmf}, {"zf", S->g.zf},
         {"c1", S->g.c1}, {"c2", S->g.c2}, {"pi0", S->base.pi0}, {"th0", S->base.th0},
         {"ua", S->ua.d}, {"va", S->va.d}, {"wa", S->wa.d}, {"ppi", S->ppi.d}, {"tha", S->tha.d},
         {"rho", S->rho.d}, {"prs", S->prs.d}, {"tkea", S->tkea.d},
         {"kmh", S->kmh.d}, {"kmv", S->kmv.d}, {"khh", S->khh.d}, {"khv", S->khv.d}})
    rd.read_into<3>(n, a);
  S->sync_to_host();
  for (int n = 1; n <= o.numq; ++n) {
    const std::string name = "qa_" + std::string(n == 1 ? "qv" : n == 2 ? "qc" : "qr");
    const auto q = rd.read_host<3>(name);
    for (int k = b.kbm; k <= b.kem; ++k)
      for (int j = b.jbm; j <= b.jem; ++j)
        for (int i = b.ibm; i <= b.iem; ++i) S->qa.h(i, j, k, n) = q(i, j, k);
  }
  // rr/rf (solve3.F) on 0..ni+1; the rest of the halo from the same formula
  const auto ex = init::extrap_coeffs(rd.read_host<1>("sigma"), rd.read_host<1>("sigmaf"), b.nk);
  for (int j = b.jb; j <= b.je; ++j)
    for (int i = b.ib; i <= b.ie; ++i) {
      for (int k = b.kb; k <= b.ke; ++k) { S->rr.h(i, j, k) = 1.0 / S->rho.h(i, j, k); S->rf.h(i, j, k) = S->rho.h(i, j, k); }
      for (int k = 2; k <= b.nk; ++k)
        S->rf.h(i, j, k) = (S->c1.h(i, j, k) * S->rho.h(i, j, k - 1) + S->c2.h(i, j, k) * S->rho.h(i, j, k));
      S->rf.h(i, j, 1) = ex.cgs1 * S->rho.h(i, j, 1) + ex.cgs2 * S->rho.h(i, j, 2) + ex.cgs3 * S->rho.h(i, j, 3);
      S->rf.h(i, j, b.nk + 1) = ex.cgt1 * S->rho.h(i, j, b.nk) + ex.cgt2 * S->rho.h(i, j, b.nk - 1)
                                + ex.cgt3 * S->rho.h(i, j, b.nk - 2);
    }
  S->sync_to_dev();
  return S;
}

// Full sfc_and_turb + solve1 turbulence chain on a dump state, C++ vs the
// Fortran routines called in cm1.F/solve1.F order.
void run_dump_chain(const std::string& case_name, int nstep) {
  double mtime = 0;
  auto Sp = load_dump(case_name, nstep, mtime);
  if (!Sp) return;
  auto& S = *Sp;
  const auto& b = S.b;
  INFO(case_name << " step " << nstep << " rtime " << mtime);
  S.fortran_setup();
  const auto nl = Namelist::from_file(testing::case_namelist(case_name));
  const double dt = nl.getr("dtl", 1.0);

  // ---- (A) sfc_and_turb
  std::vector<std::pair<const char*, Pair<3>*>> outA = {
    {"kmh", &S.kmh}, {"kmv", &S.kmv}, {"khh", &S.khh}, {"khv", &S.khv}, {"tkea", &S.tkea},
    {"cme", &S.cme}, {"csm", &S.csm}, {"ce1", &S.ce1}, {"ce2", &S.ce2},
    {"nm", &S.nm}, {"defv", &S.defv}, {"defh", &S.defh}, {"lenscl", &S.lenscl}, {"dissten", &S.dissten},
    {"t11", &S.t11}, {"t12", &S.t12}, {"t13", &S.t13}, {"t22", &S.t22}, {"t23", &S.t23}, {"t33", &S.t33}};
  std::map<std::string, FArray<3, HostSpace>> F;
  for (auto& [n, p] : outA) F[n] = hcopy(p->h);
  auto tke3d = zeros_like(S.tkea);
  auto tke3df = hcopy(tke3d.h);

  T::sfc_and_turb(S.o, S.c, S.g, S.base, mtime, S.ua.d, S.va.d, S.wa.d, S.ppi.d, S.tha.d, S.qa.d, S.prs.d,
                  S.rho.d, S.rf.d, S.tkea.d, tke3d.d, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d, S.ts, S.w);
  S.sync_to_host();
  tke3d.to_host();

  if (S.o.ramp_sgs >= 1)
    cm1f_turb_ramp(mtime, S.o.ramp_time, S.cm0.p(), F["cme"].data(), F["csm"].data(), F["ce1"].data(), F["ce2"].data());
  for (int k = 1; k <= b.nk + 1; ++k)
    for (int j = 1; j <= b.nj; ++j)
      for (int i = 1; i <= b.ni; ++i) F["dissten"](i, j, k) = 0.0;
  cm1f_turb_calcnm(S.c1.p(), S.c2.p(), S.mf.p(), S.pi0.p(), S.th0.p(), S.prs.p(), S.ppi.p(), S.tha.p(), S.qa.p(),
                   F["nm"].data());
  cm1f_turb_calcdef(S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.c1.p(), S.c2.p(), S.mf.p(), S.rho.p(),
                    S.rr.p(), S.rf.p(), S.ua.p(), S.va.p(), S.wa.p(), F["t11"].data(), F["t12"].data(),
                    F["t13"].data(), F["t22"].data(), F["t23"].data(), F["t33"].data(), F["defv"].data(),
                    F["defh"].data());
  cm1f_turb_tkekm(S.ruh.p(), S.rvh.p(), S.rmh.p(), S.zf.p(), S.mf.p(), S.rmf.p(), S.rf.p(), F["nm"].data(),
                  F["defv"].data(), F["defh"].data(), F["kmh"].data(), F["kmv"].data(), F["khh"].data(),
                  F["khv"].data(), F["cme"].data(), F["csm"].data(), F["ce1"].data(), F["ce2"].data(),
                  F["tkea"].data(), F["lenscl"].data(), F["dissten"].data());
  cm1f_turb_tkebc(F["tkea"].data(), F["kmh"].data(), F["kmv"].data(), F["khh"].data(), F["khv"].data());
  for (int j = 0; j <= b.nj + 1; ++j)
    for (int i = 0; i <= b.ni + 1; ++i) {
      tke3df(i, j, 1) = F["tkea"](i, j, 1);
      tke3df(i, j, b.nk + 1) = F["tkea"](i, j, b.nk + 1);
    }
  cm1f_turb_gettau(S.rf.p(), S.mf.p(), F["kmh"].data(), F["kmv"].data(), S.ua.p(), S.va.p(), S.wa.p(),
                   F["t11"].data(), F["t12"].data(), F["t13"].data(), F["t22"].data(), F["t23"].data(),
                   F["t33"].data());
  for (auto& [n, p] : outA) require_same(p->h, F[n], std::string("sfc_and_turb ") + n);
  require_same(tke3d.h, tke3df, "sfc_and_turb tke3d");

  // ---- (B) solve1 on the same state (the next step's "a" level)
  auto tketen = zeros_like(S.tkea);
  auto tkf = hcopy(tketen.h);
  T::tke_source(S.o, b, S.ts.nm, S.ts.defv, S.ts.defh, S.ts.dissten, S.kmh.d, S.kmv.d, S.khv.d, tketen.d);
  T::turbt(S.o, S.g, dt, S.rho.d, S.rf.d, S.tkea.d, tketen.d, S.kmh.d, S.kmv.d, S.w);
  tketen.to_host();
  cm1f_turb_tke_source(S.nm.p(), S.defv.p(), S.defh.p(), S.dissten.p(), S.kmh.p(), S.kmv.p(), S.khv.p(), tkf.data());
  cm1f_turb_turbt(dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(),
                  S.tkea.p(), tkf.data(), S.kmh.p(), S.kmv.p());
  require_same(tketen.h, tkf, "tketen (tke_source + turbt)");

  auto uten = zeros_like(S.ua), vten = zeros_like(S.va), wten = zeros_like(S.wa), thten = zeros_like(S.tha);
  auto qten = zeros_like(S.qa);
  auto uf_ = hcopy(uten.h), vf_ = hcopy(vten.h), wf_ = hcopy(wten.h), thf = hcopy(thten.h);
  auto qf = hcopy(qten.h);
  T::dissheat(S.o, S.c, b, S.base.pi0, S.ppi.d, S.qa.d, S.ts.dissten, thten.d, S.w.d9);
  T::turb_tendencies(S.o, S.g, S.base, dt, S.rho.d, S.rr.d, S.rf.d, S.ua.d, S.va.d, S.wa.d, S.tha.d, S.qa.d,
                     uten.d, vten.d, wten.d, thten.d, qten.d, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d, S.ts, S.w);
  uten.to_host(); vten.to_host(); wten.to_host(); thten.to_host(); qten.to_host();

  cm1f_turb_dissheat(S.pi0.p(), S.ppi.p(), S.qa.p(), S.dissten.p(), thf.data());
  cm1f_turb_turbu(dt, S.ruh.p(), S.uf.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rmf.p(), S.rho.p(), S.rf.p(),
                  S.ua.p(), uf_.data(), S.wa.p(), S.t11.p(), S.t12.p(), S.t13.p(), S.t22.p(), S.kmv.p());
  cm1f_turb_turbv(dt, S.uh.p(), S.rvh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(),
                  S.va.p(), vf_.data(), S.wa.p(), S.t12.p(), S.t22.p(), S.t23.p(), S.kmv.p());
  cm1f_turb_turbw(dt, S.uh.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rf.p(), S.wa.p(), wf_.data(),
                  S.t13.p(), S.t23.p(), S.t33.p(), S.t22.p(), S.kmh.p());
  auto d7 = zeros_like(S.tha), d8 = zeros_like(S.tha);
  cm1f_turb_vimpl(S.khv.p(), S.mf.p(), S.rf.p(), S.mh.p(), S.rr.p(), d7.h.data(), d8.h.data());
  auto sadv = hcopy(S.tha.h);
  for (int k = 1; k <= b.nk; ++k) {
    const double tem = S.th0.h(1, 1, k) - phys::th0r;
    for (int j = 0; j <= b.nj + 1; ++j)
      for (int i = 0; i <= b.ni + 1; ++i) sadv(i, j, k) = tem + S.tha.h(i, j, k);
  }
  cm1f_turb_turbs(1, dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(),
                  S.rf.p(), sadv.data(), thf.data(), S.khh.p(), S.khv.p(), d7.h.data(), d8.h.data());
  const size_t n3 = size_t(b.ie - b.ib + 1) * (b.je - b.jb + 1) * (b.ke - b.kb + 1);
  for (int n = 1; n <= S.o.numq; ++n)
    cm1f_turb_turbs(n == S.o.nqv ? 1 : 0, dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(),
                    S.rho.p(), S.rr.p(), S.rf.p(), S.qa.p() + (n - 1) * n3, qf.data() + (n - 1) * n3,
                    S.khh.p(), S.khv.p(), d7.h.data(), d8.h.data());
  require_same(uten.h, uf_, "uten1 (turbu)");
  require_same(vten.h, vf_, "vten1 (turbv)");
  require_same(wten.h, wf_, "wten1 (turbw)");
  require_same(thten.h, thf, "thten1 (dissheat + turbs)");
  require_same(qten.h, qf, "qten (turbs)");
}

TEST_CASE("turb chain matches the Fortran on kessler_tke dumps", "[bridge][turb][dump]") {
  for (int n : {0, 10, 50, 150}) run_dump_chain("kessler_tke", n);
}

// End to end: sfc_and_turb on dump N reproduces the eddy viscosities that
// the Fortran model wrote to dump N+1 (needs every-step dumps; the regular
// references are every 10 steps, so this runs only where step N+1 exists).
TEST_CASE("turb sfc_and_turb reproduces the model's K in the next dump", "[turb][dump]") {
  const auto root = testing::case_ref("kessler_tke");
  if (root.empty()) return;
  int nchecked = 0;
  for (int n = 0; n <= 200; ++n) {
    if (!std::filesystem::exists(DumpReader::step_dir(root, n + 1))) continue;
    if (!std::filesystem::exists(DumpReader::step_dir(root, n))) continue;
    double mtime = 0;
    auto Sp = load_dump("kessler_tke", n, mtime);
    if (!Sp) continue;
    auto& S = *Sp;
    INFO("step " << n << " -> " << n + 1);
    auto tke3d = zeros_like(S.tkea);
    T::sfc_and_turb(S.o, S.c, S.g, S.base, mtime, S.ua.d, S.va.d, S.wa.d, S.ppi.d, S.tha.d, S.qa.d, S.prs.d,
                    S.rho.d, S.rf.d, S.tkea.d, tke3d.d, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d, S.ts, S.w);
    S.sync_to_host();
    DumpReader rd(DumpReader::step_dir(root, n + 1));
    for (auto [name, p] : std::vector<std::pair<const char*, Pair<3>*>>{
           {"kmh", &S.kmh}, {"kmv", &S.kmv}, {"khh", &S.khh}, {"khv", &S.khv}})
      require_same(p->h, rd.read_host<3>(name), std::string("K at step ") + std::to_string(n + 1) + " " + name);
    ++nchecked;
  }
  if (nchecked == 0) WARN("no consecutive kessler_tke dumps; end-to-end K check skipped");
}

} // namespace

//------------------------------------------------------------------------------
// bbc=3 (semi-slip lower boundary, sfcmodel=1) and surface fluxes: the
// pieces of the turbulence that read the surface layer (physics/sfc)

namespace {

// Fortran side: cm1f_turb_setup, then bbc=3/sfcmodel=1 (defaults otherwise).
void fortran_setup_sfc(const Setup& S) {
  S.fortran_setup();
  const int so[8] = {1, 3, 0, 0, 0, 0, 16, 1};
  const double sr[8] = {0.0011, 0.0011, 0.16, 0.25, 0.24, 5.2e-5, 299.28, 2.0};
  cm1f_sfc_options(so, sr);
}

T::Options options_sfc(const Case& cs) {
  auto o = options_for(cs);
  o.bbc = 3; o.sfcmodel = 1; o.isfcflx = 1; o.dosfcflx = true;
  o.check();
  return o;
}

// The surface arrays the turbulence reads, random (halos included).
struct SfcArrays {
  sfc::SfcState s;
  FArray<3, HostSpace> ust, u1, v1, s1, zntmp, thflux, qvflux;
  SfcArrays(const Bounds& b, std::mt19937_64& gen) {
    s.allocate(b);
    std::uniform_int_distribution<int> I(0, 9);
    ust = s.ust.to_host(); u1 = s.u1.to_host(); v1 = s.v1.to_host(); s1 = s.s1.to_host();
    zntmp = s.zntmp.to_host(); thflux = s.thflux.to_host(); qvflux = s.qvflux.to_host();
    fill(ust, gen, 0.3, 0.29);
    fill(u1, gen, 2.0, 8.0);
    fill(v1, gen, -1.0, 8.0);
    for (size_t n = 0; n < s1.size(); ++n) {
      s1.data()[n] = std::sqrt(u1.data()[n] * u1.data()[n] + v1.data()[n] * v1.data()[n]);
      if (I(gen) == 0) s1.data()[n] = 0.004;   // below the 0.01 floor of sfcstress
    }
    for (size_t n = 0; n < zntmp.size(); ++n) zntmp.data()[n] = std::pow(10.0, -4.0 + 3.0 * (0.5 + 0.5 * U(gen)));
    fill(thflux, gen, 0.05, 0.1);
    fill(qvflux, gen, 2.0e-5, 3.0e-5);
    s.ust.deep_copy_from(ust); s.u1.deep_copy_from(u1); s.v1.deep_copy_from(v1); s.s1.deep_copy_from(s1);
    s.zntmp.deep_copy_from(zntmp); s.thflux.deep_copy_from(thflux); s.qvflux.deep_copy_from(qvflux);
  }
};

void check_sfc_tkebc_gettau(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(box(10, 9, 12, 3, 1), options_sfc(cs));
  S.randomize(gen);
  fortran_setup_sfc(S);
  SfcArrays sf(S.b, gen);
  // tkebc: kmh(k=1) = karman*zntmp*ust
  std::vector<FArray<3, HostSpace>> f;
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea}) f.push_back(hcopy(p->h));
  T::tkebc(S.o, S.b, S.tkea.d, S.kmh.d, S.kmv.d, S.khh.d, S.khv.d, &sf.s);
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea}) p->to_host();
  cm1f_sfc_tkebc(f[4].data(), f[0].data(), f[1].data(), f[2].data(), f[3].data(), sf.zntmp.data(), sf.ust.data());
  const char* kn[] = {"kmh", "kmv", "khh", "khv", "tkea"};
  int n = 0;
  for (auto* p : {&S.kmh, &S.kmv, &S.khh, &S.khv, &S.tkea}) {
    require_same(p->h, f[n], std::string("tkebc bbc=3 ") + kn[n]);
    ++n;
  }
  // gettau + sfcstress
  std::vector<FArray<3, HostSpace>> t;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33}) t.push_back(hcopy(p->h));
  T::gettau(S.o, S.b, S.kmh.d, S.kmv.d, S.ts.t11, S.ts.t12, S.ts.t13, S.ts.t22, S.ts.t23, S.ts.t33, S.rf.d, &sf.s);
  S.sync_to_host();
  cm1f_sfc_gettau(S.rf.p(), S.mf.p(), S.kmh.p(), S.kmv.p(), S.ua.p(), S.va.p(), S.wa.p(), sf.ust.data(),
                  sf.u1.data(), sf.v1.data(), sf.s1.data(), t[0].data(), t[1].data(), t[2].data(), t[3].data(),
                  t[4].data(), t[5].data());
  const char* tn[] = {"t11", "t12", "t13", "t22", "t23", "t33"};
  n = 0;
  for (auto* p : {&S.t11, &S.t12, &S.t13, &S.t22, &S.t23, &S.t33}) {
    require_same(p->h, t[n], std::string("gettau bbc=3 ") + tn[n]);
    ++n;
  }
}

void check_sfc_turbs_dissheat(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  const auto o = options_sfc(cs);
  Setup S(box(10, 9, 12, 3, o.numq), o);
  S.randomize(gen);
  fortran_setup_sfc(S);
  SfcArrays sf(S.b, gen);
  const double dt = 2.0;
  // dissipative heating with the surface term
  auto thten = random_like(S.tha, gen, 0.0, 1.0e-3);
  auto thf = hcopy(thten.h);
  T::dissheat(S.o, S.c, S.b, S.base.pi0, S.ppi.d, S.qa.d, S.ts.dissten, thten.d, S.w.d9, S.g.zf, &sf.s);
  thten.to_host();
  cm1f_sfc_dissheat(S.pi0.p(), S.ppi.p(), S.qa.p(), S.dissten.p(), S.zf.p(), sf.ust.data(), sf.zntmp.data(),
                    thf.data());
  require_same(thten.h, thf, "dissheat bbc=3");
  // turbs with the surface flux (theta: iflux=1; other scalars: iflux=0)
  auto d7 = zeros_like(S.tha), d8 = zeros_like(S.tha);
  T::vimpl_coefs(S.b, S.khv.d, S.g.mf, S.rf.d, S.g.mh, S.rr.d, d7.d, d8.d);
  d7.to_host(); d8.to_host();
  for (int iflux : {1, 0}) {
    auto sten = random_like(S.tha, gen, 0.0, 1.0e-3);
    auto stf = hcopy(sten.h);
    T::turbs(S.o, S.g, iflux, dt, S.rho.d, S.rr.d, S.rf.d, S.tha.d, sten.d, S.khh.d, S.khv.d, d7.d, d8.d, S.w,
             sf.s.thflux);
    sten.to_host();
    cm1f_sfc_turbs(iflux, dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(),
                   S.rf.p(), S.tha.p(), stf.data(), S.khh.p(), S.khv.p(), d7.h.data(), d8.h.data(),
                   sf.thflux.data());
    require_same(sten.h, stf, "turbs dosfcflx iflux=" + std::to_string(iflux));
  }
}

} // namespace

TEST_CASE("turb bbc=3: tkebc and gettau/sfcstress match turb.F", "[bridge][turb][sfc]") {
  unsigned seed = 901;
  for (const auto& bc : kLateral) check_sfc_tkebc_gettau(lateral_case(bc), seed++);
}

TEST_CASE("turb bbc=3: dissheat and turbs with surface fluxes match solve1.F/turbtend.F", "[bridge][turb][sfc]") {
  unsigned seed = 951;
  for (int doimpl : {0, 1})
    for (int bcturbs : {1, 2}) {
      Case c; c.doimpl = doimpl; c.bcturbs = bcturbs;
      check_sfc_turbs_dissheat(c, seed++);
    }
  Case dry; dry.moist = Moist::dry; check_sfc_turbs_dissheat(dry, seed++);
  Case e1; e1.eqtset = 1; check_sfc_turbs_dissheat(e1, seed++);
  Case nod; nod.idiss = 0; check_sfc_turbs_dissheat(nod, seed++);
  Case opn = lateral_case({2, 3, 4, 2}); check_sfc_turbs_dissheat(opn, seed++);
}
