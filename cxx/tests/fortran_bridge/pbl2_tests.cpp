// Component tests of the simple PBL scheme port (cm1setup=2, ipbl=2:
// physics/pbl/cm1k_pbl.*, sfc::gethpbl2, and the cm1setup=2 gating of the
// turbtend.F routines in physics/turb) against CM1's sfcphys.F gethpbl2,
// turb.F turbparam_vert/turbparam_horiz, the dobr09 block of turb.F and the
// use_pbl blocks of solve1.F (verbatim copies), turbtend.F turbu/turbv/
// turbw/turbs with dohturb=(horizturb=1), dovturb=.false., through
// cm1f_bridge_pbl2.F90 (and cm1f_bridge_turb.F90 for turbu..turbs).
//
// Both sides get identical random inputs (halos included) and every output
// array is compared bit for bit over its whole extent.
#include "physics/pbl/cm1k_pbl.hpp"
#include "physics/sfc/cm1k_sfc.hpp"
#include "physics/turb/cm1k_turb.hpp"
#include "share/cm1k_constants.hpp"

#include "cm1k_test_utils.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>

extern "C" {
void cm1f_turb_setup(int ni, int nj, int nk, int ngxy, int ngz, const int* iopt, const double* ropt);
void cm1f_sfc_options(const int* iopt, const double* ropt);
void cm1f_pbl2_options(const int* iopt, const double* ropt);
void cm1f_pbl2_gethpbl2(const double* psfc, double* qsfc, const double* thflux, const double* qvflux,
                        const double* ust, const double* tsk, const double* zh, const double* th0,
                        const double* tha, const double* divx, const double* ugr, const double* vgr, double* hpbl);
void cm1f_pbl2_turbparam_vert(const double* zf, const double* mf, const double* nm, const double* defv,
                              double* kmv, double* khv, double* dissten, const double* zntmp, const double* ust);
void cm1f_pbl2_turbparam_horiz(const double* zf, const double* mf, const double* nm, const double* defh,
                               double* kmh, double* khh, double* dissten, const double* xland, const double* psfc);
void cm1f_pbl2_br09(int dosfc, double dt, const double* ruh, const double* uf, const double* uh, const double* rvh,
                    const double* vf, const double* mh, const double* mf, const double* rmf, const double* rho,
                    const double* rr, const double* rf, const double* th0, const double* tha, const double* qa,
                    const double* ua, const double* va, const double* wa, const double* t13, const double* t23,
                    const double* kmv, const double* khv, const double* thflux, const double* qvflux,
                    double* thpten, double* qvpten, double* qcpten, double* upten, double* vpten);
void cm1f_pbl2_solve1(const double* thpten, const double* qvpten, const double* qcpten, const double* upten,
                      const double* vpten, double* thten1, double* uten1, double* vten1, double* qten);
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
}

using namespace cm1k;

namespace {

// Host and device copies of one array.
template <int R>
struct Pair {
  FArray<R> d;
  FArray<R, HostSpace> h;
  void to_dev() const { d.deep_copy_from(h); }
  void to_host() { h.deep_copy_from(d); }
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

template <int R>
FArray<R, HostSpace> hcopy(const FArray<R, HostSpace>& h) {
  std::array<int, R> lb, ub;
  for (int d = 0; d < R; ++d) { lb[d] = h.lb(d); ub[d] = h.ub(d); }
  FArray<R, HostSpace> c("copy", lb, ub);
  std::memcpy(c.data(), h.data(), h.size() * sizeof(double));
  return c;
}

// Bit-for-bit comparison (NaN == NaN); returns the max |ref|.
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

template <int R>
void fill(Pair<R>& a, std::mt19937_64& gen, double mean, double amp) {
  for (size_t n = 0; n < a.h.size(); ++n) a.h.data()[n] = mean + amp * U(gen);
  a.to_dev();
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
  bool moist = true;
  int doimpl = 1, bcturbs = 1, idiss = 1, horizturb = 1, pertflx = 0, dosfc = 1;
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  double l_h = 100.0, lhref1 = 100.0, lhref2 = 1000.0, l_inf = 75.0;
  std::string str() const {
    std::ostringstream s;
    s << "moist=" << moist << " doimpl=" << doimpl << " bcturbs=" << bcturbs << " idiss=" << idiss
      << " horizturb=" << horizturb << " pertflx=" << pertflx << " dosfcflx=" << dosfc << " bc=" << wbc << ebc
      << sbc << nbc << " l_h=" << l_h << " lhref1=" << lhref1 << " lhref2=" << lhref2 << " l_inf=" << l_inf;
    return s.str();
  }
};

// Everything the routines read, on both sides (cm1setup=2, ipbl=2).
struct Setup {
  Case cs;
  Bounds b;
  Grid g;
  BaseState base;
  Constants c;
  turb::Options to;
  pbl::Options po;
  sfc::Options so;
  sfc::SfcState sfc;
  pbl::Tendencies pt;
  turb::Work w;
  Pair<1> uh, ruh, uf, vh, rvh, vf;
  Pair<3> mh, mf, rmf, zf, zh, th0, u0, v0;
  Pair<3> rho, rr, rf, ua, va, wa, tha, kmh, kmv, khh, khv, nm, defv, defh, dissten;
  Pair<3> t11, t12, t13, t22, t23, t33;
  Pair<3> psfc, qsfc, thflux, qvflux, ust, tsk, zntmp, xland, hpbl;
  Pair<3> thpten, qvpten, qcpten, upten, vpten;
  Pair<4> qa;

  explicit Setup(const Case& cc) : cs(cc), c(Constants::make(0)) {
    const int numq = cs.moist ? 3 : 1;
    b = box(10, 9, 12, 3, numq);
    g.allocate(b);
    g.b = b;
    base.allocate(b);
    sfc.allocate(b);
    pt.allocate(b);
    w.allocate(b);
    // turb options as from_namelist gives them for cm1setup=2, ipbl=2
    to.cm1setup = 2; to.sgsmodel = 0; to.ipbl = 2; to.horizturb = cs.horizturb; to.tconfig = 2;
    to.bcturbs = cs.bcturbs; to.doimpl = cs.doimpl; to.idiss = cs.idiss;
    to.sfcmodel = 1; to.isfcflx = cs.dosfc; to.bbc = 3; to.dosfcflx = (cs.dosfc == 1);
    to.wbc = cs.wbc; to.ebc = cs.ebc; to.sbc = cs.sbc; to.nbc = cs.nbc;
    to.set_boundary_flags();
    to.dohturb = (cs.horizturb == 1); to.dovturb = false;
    if (cs.moist) {
      to.imoist = 1; to.ptype = 1; to.numq = 3; to.nqv = 1; to.nqc = 2; to.nqi = 0; to.nql1 = 2; to.nql2 = 3;
      to.nqs1 = -2; to.nqs2 = -2;
    } else {
      to.imoist = 0; to.ptype = 0; to.numq = 1; to.nqv = 1; to.nqc = 0; to.nqi = 0; to.nql1 = 0; to.nql2 = 0;
      to.nqs1 = 0; to.nqs2 = 0;
    }
    to.check();
    po.cm1setup = 2; po.ipbl = 2; po.horizturb = cs.horizturb; po.sfcmodel = 1; po.idiss = cs.idiss;
    po.imoist = to.imoist; po.ptype = to.ptype;
    po.nqv = cs.moist ? 1 : 0; po.nqc = cs.moist ? 2 : 0; po.nqi = 0;
    po.l_h = cs.l_h; po.lhref1 = cs.lhref1; po.lhref2 = cs.lhref2; po.l_inf = cs.l_inf;
    po.use_pbl = true;
    po.check();
    so.sfcmodel = 1; so.bbc = 3; so.cm1setup = 2; so.ipbl = 2; so.isfcflx = cs.dosfc; so.pertflx = cs.pertflx;
    so.imoist = to.imoist; so.ptype = to.ptype; so.dosfcflx = to.dosfcflx;
    so.check();

    uh = wrap(g.uh); ruh = wrap(g.ruh); uf = wrap(g.uf); vh = wrap(g.vh); rvh = wrap(g.rvh); vf = wrap(g.vf);
    mh = wrap(g.mh); mf = wrap(g.mf); rmf = wrap(g.rmf); zf = wrap(g.zf); zh = wrap(g.zh);
    th0 = wrap(base.th0); u0 = wrap(base.u0); v0 = wrap(base.v0);
    const std::array<int, 3> s{b.ib, b.jb, b.kb}, se{b.ie, b.je, b.ke};
    rho = mk<3>("rho", s, se); rr = mk<3>("rr", s, se); rf = mk<3>("rf", s, se); tha = mk<3>("tha", s, se);
    ua = mk<3>("ua", s, {b.ie + 1, b.je, b.ke});
    va = mk<3>("va", s, {b.ie, b.je + 1, b.ke});
    wa = mk<3>("wa", s, {b.ie, b.je, b.ke + 1});
    const std::array<int, 3> lc{b.ibc, b.jbc, b.kbc}, uc{b.iec, b.jec, b.kec};
    kmh = mk<3>("kmh", lc, uc); kmv = mk<3>("kmv", lc, uc); khh = mk<3>("khh", lc, uc); khv = mk<3>("khv", lc, uc);
    const std::array<int, 3> uw{b.ie, b.je, b.ke + 1};
    nm = mk<3>("nm", s, uw); defv = mk<3>("defv", s, uw); defh = mk<3>("defh", s, uw);
    dissten = mk<3>("dissten", s, uw);
    t11 = mk<3>("t11", s, se); t12 = mk<3>("t12", s, se); t13 = mk<3>("t13", s, se);
    t22 = mk<3>("t22", s, se); t23 = mk<3>("t23", s, se); t33 = mk<3>("t33", s, se);
    psfc = wrap(sfc.psfc); qsfc = wrap(sfc.qsfc); thflux = wrap(sfc.thflux); qvflux = wrap(sfc.qvflux);
    ust = wrap(sfc.ust); tsk = wrap(sfc.tsk); zntmp = wrap(sfc.zntmp); xland = wrap(sfc.xland); hpbl = wrap(sfc.hpbl);
    thpten = wrap(pt.thpten); qvpten = wrap(pt.qvpten); qcpten = wrap(pt.qcpten);
    upten = wrap(pt.upten); vpten = wrap(pt.vpten);
    qa = mk<4>("qa", {b.ibm, b.jbm, b.kbm, 1}, {b.iem, b.jem, b.kem, numq});
  }

  // Random, physically plausible fields (halos included): a stably
  // stratified layer (the bulk Richardson number crosses 0.125 within the
  // column), surface layer over land and water.
  void randomize(std::mt19937_64& gen) {
    g.dx = 1000.0; g.dy = 900.0; g.dz = 200.0;
    g.rdx = 1.0 / g.dx; g.rdy = 1.0 / g.dy; g.rdz = 1.0 / g.dz;
    for (int i = b.ib; i <= b.ie; ++i) { uh.h(i) = 1.0 + 0.2 * U(gen); ruh.h(i) = 1.0 / uh.h(i); }
    for (int i = b.ib; i <= b.ie + 1; ++i) uf.h(i) = 1.0 + 0.2 * U(gen);
    for (int j = b.jb; j <= b.je; ++j) { vh.h(j) = 1.0 + 0.2 * U(gen); rvh.h(j) = 1.0 / vh.h(j); }
    for (int j = b.jb; j <= b.je + 1; ++j) vf.h(j) = 1.0 + 0.2 * U(gen);
    for (int k = b.kb; k <= b.ke + 1; ++k)
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) {
          mf.h(i, j, k) = 1.0 + 0.3 * U(gen);
          rmf.h(i, j, k) = 1.0 / mf.h(i, j, k);
          zf.h(i, j, k) = 200.0 * (k - 1);
          if (k > b.ke) continue;
          const double z = 200.0 * (k - 0.5) + 20.0 * U(gen);
          zh.h(i, j, k) = z;
          mh.h(i, j, k) = 1.0 + 0.3 * U(gen);
          th0.h(i, j, k) = 300.0 + 0.004 * z;
          rho.h(i, j, k) = 1.2 * std::exp(-z / 8000.0) * (1.0 + 0.01 * U(gen));
          rr.h(i, j, k) = 1.0 / rho.h(i, j, k);
          rf.h(i, j, k) = 1.2 * std::exp(-(z - 100.0) / 8000.0) * (1.0 + 0.01 * U(gen));
          tha.h(i, j, k) = 0.5 * U(gen);
        }
    for (auto* p : {&uh, &ruh, &uf, &vh, &rvh, &vf}) p->to_dev();
    for (auto* p : {&mf, &rmf, &zf, &zh, &mh, &th0, &rho, &rr, &rf, &tha}) p->to_dev();
    fill(ua, gen, 5.0, 3.0);
    fill(va, gen, -2.0, 3.0);
    fill(wa, gen, 0.0, 1.0);
    fill(u0, gen, 3.0, 1.0);
    fill(v0, gen, -1.0, 1.0);
    for (auto* p : {&kmh, &kmv, &khh, &khv}) fill(*p, gen, 60.0, 30.0);
    fill(nm, gen, 1.0e-4, 2.0e-4);
    fill(defv, gen, 1.0e-4, 2.0e-4);
    fill(defh, gen, 1.0e-4, 2.0e-4);
    fill(dissten, gen, 1.0e-3, 1.0e-3);
    for (auto* p : {&t11, &t12, &t13, &t22, &t23, &t33}) fill(*p, gen, 0.0, 0.1);
    fill(psfc, gen, 96000.0, 5000.0);
    fill(qsfc, gen, 0.01, 0.005);
    fill(thflux, gen, 0.05, 0.1);
    fill(qvflux, gen, 2.0e-5, 3.0e-5);
    fill(ust, gen, 0.3, 0.29);
    fill(tsk, gen, 300.0, 3.0);
    for (size_t n = 0; n < zntmp.h.size(); ++n) zntmp.h.data()[n] = std::pow(10.0, -4.0 + 3.0 * (0.5 + 0.5 * U(gen)));
    zntmp.to_dev();
    for (size_t n = 0; n < xland.h.size(); ++n) xland.h.data()[n] = U(gen) > 0.0 ? 2.0 : 1.0;
    xland.to_dev();
    fill(hpbl, gen, 500.0, 400.0);
    for (auto* p : {&thpten, &qvpten, &qcpten, &upten, &vpten}) fill(*p, gen, 0.0, 1.0e-3);
    for (int n = 1; n <= b.numq; ++n)
      for (int k = b.kbm; k <= b.kem; ++k)
        for (int j = b.jbm; j <= b.jem; ++j)
          for (int i = b.ibm; i <= b.iem; ++i)
            qa.h(i, j, k, n) = (n == 1) ? 0.015 * std::exp(-0.15 * k) * (1.0 + 0.3 * U(gen))
                                        : std::max(0.0, 1.0e-3 * U(gen));
    qa.to_dev();
  }

  void fortran_setup() const {
    const int iopt[21] = {to.tconfig, to.bcturbs, to.doimpl, to.wbc, to.ebc, to.sbc, to.nbc, to.imoist, to.ptype,
                          to.iice, to.numq, to.nqv, to.nqc, to.nqi, to.nql1, to.nql2, to.nqs1, to.nqs2,
                          to.eqtset, to.idiss, 0};
    const double ropt[3] = {g.dx, g.dy, g.dz};
    cm1f_turb_setup(b.ni, b.nj, b.nk, b.ngxy, b.ngz, iopt, ropt);
    const int so8[8] = {cs.dosfc, 3, 0, 0, 0, cs.pertflx, 16, 1};
    const double sr8[8] = {0.0011, 0.0011, 0.16, 0.25, 0.24, 5.2e-5, 299.28, 2.0};
    cm1f_sfc_options(so8, sr8);
    const int pi[1] = {cs.horizturb};
    const double pr[4] = {cs.l_h, cs.lhref1, cs.lhref2, cs.l_inf};
    cm1f_pbl2_options(pi, pr);
  }

  // qa(:,:,:,n) on the host as a 3D (ib:ie,jb:je,kb:ke) array
  FArray<3, HostSpace> qslice(int n) const {
    FArray<3, HostSpace> a("q", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    for (int k = b.kb; k <= b.ke; ++k)
      for (int j = b.jb; j <= b.je; ++j)
        for (int i = b.ib; i <= b.ie; ++i) a(i, j, k) = qa.h(i, j, k, n);
    return a;
  }
};

//------------------------------------------------------------------------------

void check_gethpbl2(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(cs);
  S.randomize(gen);
  S.fortran_setup();
  // Fortran inputs: divx = qv (or 0), ugr/vgr = ground-relative winds
  FArray<3, HostSpace> divx("divx", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke});
  if (cs.moist) divx = S.qslice(1);
  else Kokkos::deep_copy(divx.view(), 0.0);
  auto ugr = hcopy(S.ua.h), vgr = hcopy(S.va.h);
  if (cs.pertflx == 1) {
    for (size_t n = 0; n < ugr.size(); ++n) ugr.data()[n] = S.ua.h.data()[n] - S.u0.h.data()[n];
    for (size_t n = 0; n < vgr.size(); ++n) vgr.data()[n] = S.va.h.data()[n] - S.v0.h.data()[n];
  }
  auto qsfcf = hcopy(S.qsfc.h), hpblf = hcopy(S.hpbl.h);
  sfc::gethpbl2(S.so, S.c, S.b, S.g.zh, S.base.th0, S.tha.d, S.qa.d, S.ua.d, S.va.d, S.base.u0, S.base.v0, S.sfc);
  S.qsfc.to_host(); S.hpbl.to_host();
  cm1f_pbl2_gethpbl2(S.psfc.p(), qsfcf.data(), S.thflux.p(), S.qvflux.p(), S.ust.p(), S.tsk.p(), S.zh.p(),
                     S.th0.p(), S.tha.p(), divx.data(), ugr.data(), vgr.data(), hpblf.data());
  require_same(S.qsfc.h, qsfcf, "gethpbl2 qsfc");
  require_same(S.hpbl.h, hpblf, "gethpbl2 hpbl");
  // the search must have found the layer top in most columns, between the
  // first two levels and the top
  int nfound = 0;
  for (int j = 1; j <= S.b.nj; ++j)
    for (int i = 1; i <= S.b.ni; ++i) nfound += (S.hpbl.h(i, j, 1) > 0.0);
  CHECK(nfound > S.b.ni * S.b.nj / 2);
}

void check_turbparam(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(cs);
  S.randomize(gen);
  S.fortran_setup();
  // turb.F order: turbparam_horiz (horizturb=1), then turbparam_vert
  auto kmhf = hcopy(S.kmh.h), khhf = hcopy(S.khh.h), kmvf = hcopy(S.kmv.h), khvf = hcopy(S.khv.h);
  auto dissf = hcopy(S.dissten.h);
  if (cs.horizturb == 1) {
    pbl::turbparam_horiz(S.to, S.po, S.b, S.defh.d, S.kmh.d, S.khh.d, S.dissten.d, S.sfc);
    cm1f_pbl2_turbparam_horiz(S.zf.p(), S.mf.p(), S.nm.p(), S.defh.p(), kmhf.data(), khhf.data(), dissf.data(),
                              S.xland.p(), S.psfc.p());
  }
  pbl::turbparam_vert(S.to, S.po, S.g, S.nm.d, S.defv.d, S.kmv.d, S.khv.d, S.dissten.d, S.sfc, S.w.d4);
  cm1f_pbl2_turbparam_vert(S.zf.p(), S.mf.p(), S.nm.p(), S.defv.p(), kmvf.data(), khvf.data(), dissf.data(),
                           S.zntmp.p(), S.ust.p());
  for (auto* p : {&S.kmh, &S.khh, &S.kmv, &S.khv, &S.dissten}) p->to_host();
  require_same(S.kmh.h, kmhf, "turbparam kmh");
  require_same(S.khh.h, khhf, "turbparam khh");
  require_same(S.kmv.h, kmvf, "turbparam kmv");
  require_same(S.khv.h, khvf, "turbparam khv");
  require_same(S.dissten.h, dissf, "turbparam dissten");
}

void check_br09(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(cs);
  S.randomize(gen);
  S.fortran_setup();
  const double dt = 6.0;
  std::vector<FArray<3, HostSpace>> f;
  for (auto* p : {&S.thpten, &S.qvpten, &S.qcpten, &S.upten, &S.vpten}) f.push_back(hcopy(p->h));
  pbl::br09(S.to, S.po, S.g, S.base, dt, S.ua.d, S.va.d, S.wa.d, S.tha.d, S.qa.d, S.rho.d, S.rr.d, S.rf.d, S.kmv.d,
            S.khv.d, S.t13.d, S.t23.d, S.sfc, S.pt, S.w);
  cm1f_pbl2_br09(cs.dosfc, dt, S.ruh.p(), S.uf.p(), S.uh.p(), S.rvh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rmf.p(),
                 S.rho.p(), S.rr.p(), S.rf.p(), S.th0.p(), S.tha.p(), S.qa.p(), S.ua.p(), S.va.p(), S.wa.p(),
                 S.t13.p(), S.t23.p(), S.kmv.p(), S.khv.p(), S.thflux.p(), S.qvflux.p(), f[0].data(), f[1].data(),
                 f[2].data(), f[3].data(), f[4].data());
  const char* names[] = {"thpten", "qvpten", "qcpten", "upten", "vpten"};
  int n = 0;
  for (auto* p : {&S.thpten, &S.qvpten, &S.qcpten, &S.upten, &S.vpten}) {
    p->to_host();
    require_same(p->h, f[n], std::string("br09 ") + names[n]);
    ++n;
  }
}

void check_solve1(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(cs);
  S.randomize(gen);
  S.fortran_setup();
  auto thten1 = mk<3>("thten1", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke});
  auto uten1 = mk<3>("uten1", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie + 1, S.b.je, S.b.ke});
  auto vten1 = mk<3>("vten1", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je + 1, S.b.ke});
  auto qten = mk<4>("qten", {S.b.ibm, S.b.jbm, S.b.kbm, 1}, {S.b.iem, S.b.jem, S.b.kem, S.b.numq});
  fill(thten1, gen, 0.0, 1.0e-3);
  fill(uten1, gen, 0.0, 1.0e-3);
  fill(vten1, gen, 0.0, 1.0e-3);
  fill(qten, gen, 0.0, 1.0e-6);
  auto thf = hcopy(thten1.h), uf_ = hcopy(uten1.h), vf_ = hcopy(vten1.h);
  auto qf = hcopy(qten.h);
  pbl::add_theta_tendency(S.b, S.pt, thten1.d);
  pbl::add_tendencies(S.po, S.b, S.pt, uten1.d, vten1.d, qten.d);
  cm1f_pbl2_solve1(S.thpten.p(), S.qvpten.p(), S.qcpten.p(), S.upten.p(), S.vpten.p(), thf.data(), uf_.data(),
                   vf_.data(), qf.data());
  thten1.to_host(); uten1.to_host(); vten1.to_host(); qten.to_host();
  require_same(thten1.h, thf, "solve1 thten1");
  require_same(uten1.h, uf_, "solve1 uten1");
  require_same(vten1.h, vf_, "solve1 vten1");
  require_same(qten.h, qf, "solve1 qten");
}

// turbtend.F with cm1setup=2: horizontal parts only with dohturb
// (horizturb=1), no turbuz/turbvz/turbsz (dovturb=.false.), but turbw's
// vertical part (ipbl=2).
void check_turbtend(const Case& cs, unsigned seed) {
  INFO(cs.str());
  std::mt19937_64 gen(seed);
  Setup S(cs);
  S.randomize(gen);
  S.fortran_setup();
  const double dt = 6.0;
  auto uten = mk<3>("uten", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie + 1, S.b.je, S.b.ke});
  auto vten = mk<3>("vten", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je + 1, S.b.ke});
  auto wten = mk<3>("wten", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke + 1});
  auto sten = mk<3>("sten", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke});
  for (auto* p : {&uten, &vten, &wten, &sten}) fill(*p, gen, 0.0, 1.0e-3);
  auto utf = hcopy(uten.h), vtf = hcopy(vten.h), wtf = hcopy(wten.h), stf = hcopy(sten.h);
  turb::turbu(S.to, S.g, dt, S.rho.d, S.rf.d, S.ua.d, uten.d, S.wa.d, S.t11.d, S.t12.d, S.t13.d, S.kmv.d, S.w);
  turb::turbv(S.to, S.g, dt, S.rho.d, S.rf.d, S.va.d, vten.d, S.wa.d, S.t12.d, S.t22.d, S.t23.d, S.kmv.d, S.w);
  turb::turbw(S.to, S.g, dt, S.rho.d, S.rf.d, S.wa.d, wten.d, S.t13.d, S.t23.d, S.t33.d, S.kmh.d, S.w);
  auto d7 = mk<3>("d7", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke}), d8 = d7;
  d8 = mk<3>("d8", {S.b.ib, S.b.jb, S.b.kb}, {S.b.ie, S.b.je, S.b.ke});
  turb::vimpl_coefs(S.b, S.khv.d, S.g.mf, S.rf.d, S.g.mh, S.rr.d, d7.d, d8.d);
  d7.to_host(); d8.to_host();
  turb::turbs(S.to, S.g, 1, dt, S.rho.d, S.rr.d, S.rf.d, S.tha.d, sten.d, S.khh.d, S.khv.d, d7.d, d8.d, S.w,
              S.sfc.thflux);
  for (auto* p : {&uten, &vten, &wten, &sten}) p->to_host();
  cm1f_turb_turbu(dt, S.ruh.p(), S.uf.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rmf.p(), S.rho.p(), S.rf.p(), S.ua.p(),
                  utf.data(), S.wa.p(), S.t11.p(), S.t12.p(), S.t13.p(), S.t22.p(), S.kmv.p());
  cm1f_turb_turbv(dt, S.uh.p(), S.rvh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(), S.va.p(),
                  vtf.data(), S.wa.p(), S.t12.p(), S.t22.p(), S.t23.p(), S.kmv.p());
  cm1f_turb_turbw(dt, S.uh.p(), S.vh.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rf.p(), S.wa.p(), wtf.data(), S.t13.p(),
                  S.t23.p(), S.t33.p(), S.t22.p(), S.kmh.p());
  cm1f_turb_turbs(1, dt, S.uh.p(), S.uf.p(), S.vh.p(), S.vf.p(), S.mh.p(), S.mf.p(), S.rho.p(), S.rr.p(), S.rf.p(),
                  S.tha.p(), stf.data(), S.khh.p(), S.khv.p(), d7.h.data(), d8.h.data());
  require_same(uten.h, utf, "turbu cm1setup=2");
  require_same(vten.h, vtf, "turbv cm1setup=2");
  require_same(wten.h, wtf, "turbw cm1setup=2");
  require_same(sten.h, stf, "turbs cm1setup=2");
}

const std::vector<std::array<int, 4>> kLateral = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {2, 3, 4, 2}};

} // namespace

TEST_CASE("pbl2 gethpbl2 matches sfcphys.F", "[bridge][pbl][sfc]") {
  unsigned seed = 1201;
  for (bool moist : {false, true})
    for (int pertflx : {0, 1}) {
      Case c; c.moist = moist; c.pertflx = pertflx;
      check_gethpbl2(c, seed++);
    }
}

TEST_CASE("pbl2 turbparam_vert/turbparam_horiz match turb.F", "[bridge][pbl]") {
  unsigned seed = 1301;
  for (const auto& bc : kLateral)
    for (int idiss : {0, 1})
      for (int horizturb : {0, 1}) {
        Case c; c.wbc = bc[0]; c.ebc = bc[1]; c.sbc = bc[2]; c.nbc = bc[3];
        c.idiss = idiss; c.horizturb = horizturb;
        check_turbparam(c, seed++);
      }
  // other length scales (l_inf = 0 switches turbparam_vert off)
  for (auto [lh, l1, l2, li] : std::vector<std::array<double, 4>>{
         {250.0, 50.0, 1500.0, 30.0}, {0.0, 0.0, 0.0, 0.0}}) {
    Case c; c.l_h = lh; c.lhref1 = l1; c.lhref2 = l2; c.l_inf = li;
    check_turbparam(c, seed++);
  }
}

TEST_CASE("pbl2 dobr09 block matches turb.F (turbsz, turbuz, turbvz)", "[bridge][pbl]") {
  unsigned seed = 1401;
  for (bool moist : {false, true})
    for (int doimpl : {0, 1})
      for (int bcturbs : {1, 2})
        for (int dosfc : {0, 1}) {
          Case c; c.moist = moist; c.doimpl = doimpl; c.bcturbs = bcturbs; c.dosfc = dosfc;
          check_br09(c, seed++);
        }
}

TEST_CASE("pbl2 solve1 use_pbl tendencies match solve1.F", "[bridge][pbl]") {
  unsigned seed = 1501;
  for (bool moist : {false, true}) {
    Case c; c.moist = moist;
    check_solve1(c, seed++);
  }
}

TEST_CASE("pbl2 turbtend.F with cm1setup=2 (dohturb, no dovturb, turbw vertical)", "[bridge][pbl][turb]") {
  unsigned seed = 1601;
  for (const auto& bc : kLateral)
    for (int doimpl : {0, 1}) {
      Case c; c.wbc = bc[0]; c.ebc = bc[1]; c.sbc = bc[2]; c.nbc = bc[3]; c.doimpl = doimpl;
      check_turbtend(c, seed++);
    }
}

TEST_CASE("pbl options: unsupported values throw", "[pbl]") {
  pbl::Options o;
  o.cm1setup = 2; o.ipbl = 2; o.sfcmodel = 1; o.use_pbl = true;
  REQUIRE_NOTHROW(o.check());
  auto bad = [&](auto f) { pbl::Options x = o; f(x); REQUIRE_THROWS(x.check()); };
  bad([](pbl::Options& x) { x.ipbl = 1; });   // YSU needs sfcmodel=2,3
  bad([](pbl::Options& x) { x.ipbl = 3; });
  bad([](pbl::Options& x) { x.ipbl = 4; });
  bad([](pbl::Options& x) { x.ipbl = 5; });
  bad([](pbl::Options& x) { x.cm1setup = 4; });
  bad([](pbl::Options& x) { x.sfcmodel = 0; });
  bad([](pbl::Options& x) { x.sfcmodel = 4; });
  bad([](pbl::Options& x) { x.horizturb = 2; });
  // ipbl=2 also runs with the WRF/MM5 surface layers; YSU only with those
  for (int sfcmodel : {2, 3}) {
    pbl::Options y = o;
    y.sfcmodel = sfcmodel;
    REQUIRE_NOTHROW(y.check());
    y.ipbl = 1;
    REQUIRE_NOTHROW(y.check());
  }
  bad([](pbl::Options& x) { x.ipbl = 1; x.sfcmodel = 0; });
  bad([](pbl::Options& x) { x.terrain_flag = true; });
  bad([](pbl::Options& x) { x.imoist = 1; x.ptype = 2; });
  // param.F: use_pbl for cm1setup=2,4 with ipbl>=1; l_inf=0 for ipbl/=2
  const auto n = pbl::Options::from_namelist(
      Namelist::from_string("&param1\n /\n&param2\n cm1setup = 2, ipbl = 2, sfcmodel = 1 /\n"));
  REQUIRE(n.use_pbl);
  REQUIRE(n.l_inf == 75.0);
  const auto y = pbl::Options::from_namelist(Namelist::from_string("&param2\n cm1setup = 2, ipbl = 1 /\n"));
  REQUIRE(y.use_pbl);
  REQUIRE(y.l_inf == 0.0);
  REQUIRE_FALSE(pbl::Options::from_namelist(Namelist::from_string("&param2\n cm1setup = 1 /\n")).use_pbl);
  // turb accepts cm1setup=2 with ipbl=1 (YSU) or 2; sfc accepts sfcmodel=1
  // with ipbl=2 and sfcmodel=2,3 with ipbl=1,2
  turb::Options t;
  t.cm1setup = 2; t.sgsmodel = 0; t.ipbl = 2; t.dohturb = false; t.dovturb = false;
  REQUIRE_NOTHROW(t.check());
  t.horizturb = 1; t.dohturb = true;
  REQUIRE_NOTHROW(t.check());
  t.ipbl = 1;
  REQUIRE_NOTHROW(t.check());
  t.sfcmodel = 3; t.bbc = 3;
  REQUIRE_NOTHROW(t.check());
  t.ipbl = 3;   // GFS-EDMF
  REQUIRE_NOTHROW(t.check());
  t.sfcmodel = 4;   // with the GFDL surface layer
  REQUIRE_NOTHROW(t.check());
  t.sfcmodel = 3;
  t.ipbl = 7;
  REQUIRE_THROWS(t.check());
  // GFS-EDMF with sfcmodel=2,3,4 only
  for (int sfcmodel : {2, 3, 4}) {
    pbl::Options g = o;
    g.ipbl = 3; g.sfcmodel = sfcmodel;
    REQUIRE_NOTHROW(g.check());
  }
  for (int sfcmodel : {0, 1, 6, 7}) {
    pbl::Options g = o;
    g.ipbl = 3; g.sfcmodel = sfcmodel;
    REQUIRE_THROWS(g.check());
  }
  sfc::Options s;
  s.sfcmodel = 1; s.bbc = 3; s.cm1setup = 2; s.ipbl = 2;
  REQUIRE_NOTHROW(s.check());
  s.ipbl = 1;
  REQUIRE_THROWS(s.check());
  s.sfcmodel = 2;
  REQUIRE_NOTHROW(s.check());
  s.lu0 = 0;   // init_surface.F: lu_index must be 1..33 for sfcmodel=2,3
  REQUIRE_THROWS(s.check());
}
