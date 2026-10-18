// Open-radiative boundaries: bc.F (radbcew, radbcns, radbcew4, radbcns4,
// ssopenbcw/e/s/n, restrict_openbc_we/sn) and sound.F with wbc..nbc = 2
// vs cm1k_openbc / cm1k::sound, bit for bit (halos included).
#include "dynamics/cm1k_openbc.hpp"
#include "dynamics/cm1k_sound.hpp"
#include "share/cm1k_constants.hpp"

#include "dyn_bridge_utils.hpp"

extern "C" {
void cm1f_obc_set_input(int irbc, int roflux, double cstar);
void cm1f_radbcew(double* radbcw, double* radbce, double* ua);
void cm1f_radbcns(double* radbcs, double* radbcn, double* va);
void cm1f_radbcew4(double* ruf, double* radbcw, double* radbce, double* u1, double* u2, double dt);
void cm1f_radbcns4(double* rvf, double* radbcs, double* radbcn, double* v1, double* v2, double dt);
void cm1f_ssopenbc(int side, double* uvh, double* rad, double* uv3d, double* uvten, double dts);
void cm1f_restrict_openbc_we(double* rvh, double* rmh, double* rho0, double* u3d);
void cm1f_restrict_openbc_sn(double* ruh, double* rmh, double* rho0, double* v3d);
void cm1f_sound_obc(double dt, int nrk, double dttmp, double* uh, double* vh, double* uf, double* vf,
                    double* mh, double* mf, double* c1, double* c2, double* rr0, double* rf0,
                    double* dtu0, double* dtv0,
                    double* rru, double* ua, double* u3d, double* uten,
                    double* rrv, double* va, double* v3d, double* vten,
                    double* rrw, double* wa, double* w3d, double* wten,
                    double* ppi, double* pp3d, double* piadv, double* ppten, double* ppx,
                    double* thv, double* ppterm,
                    double* radbcw, double* radbce, double* radbcs, double* radbcn,
                    double* ruh, double* rvh, double* rmh, double* rho0);
}

using namespace cm1k;
using cm1k::testing::Pair;

namespace {

struct Setup {
  int nx = 12, ny = 9, nz = 7, ng = 3;
  double dx = 2000, dy = 1500, dz = 400;
  Bounds b;
  OpenBcParams p;
  Setup(int w, int e, int s, int n, int irbc, int roflux, double cstar) {
    cm1f_dyn_set_input(nx, ny, nz, ng, w, e, s, n, dx, dy, dz, 0.6, 0.1, 4);
    cm1f_obc_set_input(irbc, roflux, cstar);
    Namelist nl;
    nl.set("nx", std::to_string(nx)); nl.set("ny", std::to_string(ny)); nl.set("nz", std::to_string(nz));
    b = Bounds::from_namelist(nl);
    REQUIRE(b.ngxy == ng);
    p.wbc = w; p.ebc = e; p.sbc = s; p.nbc = n;
    p.ibw = (w != 1); p.ibe = (e != 1); p.ibs = (s != 1); p.ibn = (n != 1);
    p.irbc = irbc; p.roflux = roflux; p.cstar = cstar;
    p.dx = dx; p.dy = dy; p.rdx = 1.0 / dx; p.rdy = 1.0 / dy;
    p.ni = b.ni; p.nj = b.nj; p.nk = b.nk;
    p.check();
  }
  std::array<int, 3> lo() const { return {b.ib, b.jb, b.kb}; }
  std::array<int, 3> hu() const { return {b.ie + 1, b.je, b.ke}; }
  std::array<int, 3> hv() const { return {b.ie, b.je + 1, b.ke}; }
  std::array<int, 3> hs() const { return {b.ie, b.je, b.ke}; }
};

struct RadPair {
  Pair<2> w, e, s, n;
  explicit RadPair(const Bounds& b)
    : w("radbcw", {b.jb, b.kb}, {b.je, b.ke}), e("radbce", {b.jb, b.kb}, {b.je, b.ke}),
      s("radbcs", {b.ib, b.kb}, {b.ie, b.ke}), n("radbcn", {b.ib, b.kb}, {b.ie, b.ke}) {}
  void random(std::mt19937_64& g, double lo, double hi) {
    w.random(g, lo, hi); e.random(g, lo, hi); s.random(g, lo, hi); n.random(g, lo, hi);
  }
  RadBc dev() const { RadBc r; r.radbcw = w.d; r.radbce = e.d; r.radbcs = s.d; r.radbcn = n.d; return r; }
  void compare() const { w.compare("radbcw"); e.compare("radbce"); s.compare("radbcs"); n.compare("radbcn"); }
};

// Edit both sides of a Pair on the host (then re-sync the device copy).
template <int R, typename Fn> void edit(Pair<R>& a, Fn fn) {
  fn(a.f);
  a.d.deep_copy_from(a.f);
}

struct Combo { int w, e, s, n; };
const Combo combos[] = {{2, 2, 2, 2}, {2, 3, 4, 2}, {1, 1, 2, 2}, {2, 2, 1, 1}};

} // namespace

TEST_CASE("bc.F radbcew/radbcns (irbc=1,2)", "[bridge][dyn][openbc]") {
  std::mt19937_64 gen(7);
  for (const auto& c : combos) {
    INFO("bc = " << c.w << c.e << c.s << c.n);
    Setup S(c.w, c.e, c.s, c.n, 1, 0, 25.0);
    const auto& b = S.b;
    RadPair r(b);
    r.random(gen, -5, 5);
    Pair<3> ua("ua", S.lo(), S.hu()), va("va", S.lo(), S.hv());
    ua.random(gen, -60, 60); va.random(gen, -60, 60);
    radbcew(S.p, r.dev(), ua.d);
    radbcns(S.p, r.dev(), va.d);
    Kokkos::fence();
    cm1f_radbcew(r.w.fp(), r.e.fp(), ua.fp());
    cm1f_radbcns(r.s.fp(), r.n.fp(), va.fp());
    r.compare();
  }
}

TEST_CASE("bc.F radbcew4/radbcns4 (irbc=4)", "[bridge][dyn][openbc]") {
  std::mt19937_64 gen(11);
  for (const auto& c : combos) {
    INFO("bc = " << c.w << c.e << c.s << c.n);
    Setup S(c.w, c.e, c.s, c.n, 4, 0, 30.0);
    const auto& b = S.b;
    const int ni = b.ni, nj = b.nj, nk = b.nk;
    RadPair r(b);
    r.random(gen, -5, 5);
    Pair<1> ruf("ruf", {b.ib}, {b.ie + 1}), rvf("rvf", {b.jb}, {b.je + 1});
    ruf.random(gen, 0.9, 1.1); rvf.random(gen, 0.9, 1.1);
    Pair<3> u1("u1", S.lo(), S.hu()), u2("u2", S.lo(), S.hu()), v1("v1", S.lo(), S.hv()), v2("v2", S.lo(), S.hv());
    u1.random(gen, -20, 20); u2.random(gen, -20, 20); v1.random(gen, -20, 20); v2.random(gen, -20, 20);
    // edge cases: zero gradient (sign(1e-10, 0)), -0.0 gradient, no change
    // (zero phase speed), and tiny gradients that hit the +-csmax clipping
    edit(u1, [&](auto& a) {
      a(3, 1, 1) = a(2, 1, 1);
      a(3, 2, 1) = -0.0; a(2, 2, 1) = 0.0;
      a(ni - 1, 1, 2) = a(ni, 1, 2) + 1e-9;
      a(ni, 2, 2) = a(ni - 1, 2, 2);
    });
    edit(u2, [&](auto& a) { a(2, 3, 3) = 0.0; });
    edit(u1, [&](auto& a) { a(2, 3, 3) = 0.0; });
    edit(v1, [&](auto& a) {
      a(1, 3, 1) = a(1, 2, 1);
      a(2, nj - 1, 1) = a(2, nj, 1) - 1e-9;
      a(3, nj, nk) = a(3, nj - 1, nk);
    });
    for (double dt : {2.0, 3.0, 6.0}) {
      radbcew4(S.p, ruf.d, r.dev(), u1.d, u2.d, dt);
      radbcns4(S.p, rvf.d, r.dev(), v1.d, v2.d, dt);
      Kokkos::fence();
      cm1f_radbcew4(ruf.fp(), r.w.fp(), r.e.fp(), u1.fp(), u2.fp(), dt);
      cm1f_radbcns4(rvf.fp(), r.s.fp(), r.n.fp(), v1.fp(), v2.fp(), dt);
      r.compare();
    }
  }
}

TEST_CASE("bc.F ssopenbcw/e/s/n (no terrain)", "[bridge][dyn][openbc]") {
  std::mt19937_64 gen(13);
  Setup S(2, 2, 2, 2, 4, 0, 30.0);
  const auto& b = S.b;
  RadPair r(b);
  r.random(gen, -40, 40);
  Pair<1> uh("uh", {b.ib}, {b.ie}), vh("vh", {b.jb}, {b.je});
  uh.random(gen, 0.9, 1.1); vh.random(gen, 0.9, 1.1);
  Pair<3> u3d("u3d", S.lo(), S.hu()), uten("uten", S.lo(), S.hu()), v3d("v3d", S.lo(), S.hv()),
      vten("vten", S.lo(), S.hv());
  u3d.random(gen, -20, 20); uten.random(gen, -1e-2, 1e-2); v3d.random(gen, -20, 20); vten.random(gen, -1e-2, 1e-2);
  const double dts = 1.5;
  for (int side = 1; side <= 4; ++side) {
    INFO("side " << side);
    const auto rad = r.dev();
    if (side == 1) { ssopenbcw(S.p, uh.d, rad, u3d.d, uten.d, dts); cm1f_ssopenbc(1, uh.fp(), r.w.fp(), u3d.fp(), uten.fp(), dts); }
    if (side == 2) { ssopenbce(S.p, uh.d, rad, u3d.d, uten.d, dts); cm1f_ssopenbc(2, uh.fp(), r.e.fp(), u3d.fp(), uten.fp(), dts); }
    if (side == 3) { ssopenbcs(S.p, vh.d, rad, v3d.d, vten.d, dts); cm1f_ssopenbc(3, vh.fp(), r.s.fp(), v3d.fp(), vten.fp(), dts); }
    if (side == 4) { ssopenbcn(S.p, vh.d, rad, v3d.d, vten.d, dts); cm1f_ssopenbc(4, vh.fp(), r.n.fp(), v3d.fp(), vten.fp(), dts); }
    Kokkos::fence();
    u3d.compare("u3d"); v3d.compare("v3d");
  }
}

TEST_CASE("bc.F restrict_openbc_we/sn (roflux=1)", "[bridge][dyn][openbc]") {
  std::mt19937_64 gen(17);
  for (const auto& c : combos) {
    INFO("bc = " << c.w << c.e << c.s << c.n);
    Setup S(c.w, c.e, c.s, c.n, 4, 1, 30.0);
    const auto& b = S.b;
    Pair<1> ruh("ruh", {b.ib}, {b.ie}), rvh("rvh", {b.jb}, {b.je}), t1("t1", {1}, {b.nk}), t2("t2", {1}, {b.nk});
    ruh.random(gen, 0.9, 1.1); rvh.random(gen, 0.9, 1.1);
    Pair<3> rmh("rmh", S.lo(), S.hs()), rho0("rho0", S.lo(), S.hs()), u3d("u3d", S.lo(), S.hu()), v3d("v3d", S.lo(), S.hv());
    rmh.random(gen, 0.8, 1.2); rho0.random(gen, 0.3, 1.2);
    // biased so that outflow exceeds inflow on one pair and not the other
    u3d.random(gen, -10, 14); v3d.random(gen, -14, 10);
    restrict_openbc_we(S.p, rvh.d, rmh.d, rho0.d, u3d.d, t1.d, t2.d);
    restrict_openbc_sn(S.p, ruh.d, rmh.d, rho0.d, v3d.d, t1.d, t2.d);
    Kokkos::fence();
    cm1f_restrict_openbc_we(rvh.fp(), rmh.fp(), rho0.fp(), u3d.fp());
    cm1f_restrict_openbc_sn(ruh.fp(), rmh.fp(), rho0.fp(), v3d.fp());
    u3d.compare("u3d"); v3d.compare("v3d");
  }
}

TEST_CASE("sound.F acoustic solver with open boundaries", "[bridge][dyn][sound][openbc]") {
  struct SCase { Combo c; int irbc, roflux; };
  const SCase cases[] = {{{2, 2, 2, 2}, 4, 0}, {{2, 2, 2, 2}, 2, 1}, {{2, 2, 3, 4}, 1, 1}, {{1, 1, 2, 2}, 2, 0}};
  std::mt19937_64 gen(42);
  for (const auto& sc : cases) {
    const auto& cb = sc.c;
    INFO("bc = " << cb.w << cb.e << cb.s << cb.n << " irbc " << sc.irbc << " roflux " << sc.roflux);
    Setup S(cb.w, cb.e, cb.s, cb.n, sc.irbc, sc.roflux, 30.0);
    const auto& b = S.b;
    const int ni = b.ni, nj = b.nj;
    const double alph = 0.6, kdiv = 0.1, dt = 6.0;
    const int nsound = sound_nsound(3, S.nx, S.ny, S.dx, S.dy, S.dz, 300.0, dt);
    Namelist nl;
    nl.set("nx", std::to_string(S.nx)); nl.set("ny", std::to_string(S.ny)); nl.set("nz", std::to_string(S.nz));
    nl.set("wbc", std::to_string(cb.w)); nl.set("ebc", std::to_string(cb.e));
    nl.set("sbc", std::to_string(cb.s)); nl.set("nbc", std::to_string(cb.n));
    const auto bc = BcParams::make(b, nl);
    const auto c = Constants::make(0);

    const auto lo = S.lo(), hs = S.hs(), hu = S.hu(), hv = S.hv();
    const std::array<int, 3> hw{b.ie, b.je, b.ke + 1};
    Pair<1> uh("uh", {b.ib}, {b.ie}), vh("vh", {b.jb}, {b.je}), uf("uf", {b.ib}, {b.ie + 1}), vf("vf", {b.jb}, {b.je + 1});
    Pair<1> ruh("ruh", {b.ib}, {b.ie}), rvh("rvh", {b.jb}, {b.je});
    uh.random(gen, 0.9, 1.1); vh.random(gen, 0.9, 1.1); uf.random(gen, 0.9, 1.1); vf.random(gen, 0.9, 1.1);
    ruh.random(gen, 0.9, 1.1); rvh.random(gen, 0.9, 1.1);
    Pair<3> mh("mh", lo, hs), mf("mf", lo, hw), c1("c1", lo, hs), c2("c2", lo, hs), rr0("rr0", lo, hs), rf0("rf0", lo, hs);
    Pair<3> rmh("rmh", lo, hs), rho0("rho0", lo, hs);
    mh.random(gen, 0.8, 1.2); mf.random(gen, 0.8, 1.2); rmh.random(gen, 0.8, 1.2); rho0.random(gen, 0.5, 1.2);
    c1.random(gen, 0.4, 0.6);
    edit(c2, [&](auto& a) { auto p1 = c1.f.data(); for (size_t n = 0; n < a.size(); ++n) a.data()[n] = 1.0 - p1[n]; });
    rr0.random(gen, 0.8, 2.0); rf0.random(gen, 0.5, 1.2);
    // param.F dtu0/dtv0: zero on non-periodic boundaries
    Pair<2> dtu0("dtu0", {b.ib, b.jb}, {b.ie, b.je}), dtv0("dtv0", {b.ib, b.jb}, {b.ie, b.je});
    edit(dtu0, [&](auto& a) {
      for (int j = b.jb; j <= b.je; ++j) for (int i = b.ib; i <= b.ie; ++i) a(i, j) = 0.0;
      for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni + 1; ++i) a(i, j) = 1.0;
      if (S.p.ibw) for (int j = 1; j <= nj; ++j) a(1, j) = 0.0;
      if (S.p.ibe) for (int j = 1; j <= nj; ++j) a(ni + 1, j) = 0.0;
    });
    edit(dtv0, [&](auto& a) {
      for (int j = b.jb; j <= b.je; ++j) for (int i = b.ib; i <= b.ie; ++i) a(i, j) = 0.0;
      for (int j = 1; j <= nj + 1; ++j) for (int i = 1; i <= ni; ++i) a(i, j) = 1.0;
      if (S.p.ibs) for (int i = 1; i <= ni; ++i) a(i, 1) = 0.0;
      if (S.p.ibn) for (int i = 1; i <= ni; ++i) a(i, nj + 1) = 0.0;
    });

    Pair<3> rru("rru", lo, hu), ua("ua", lo, hu), u3d("u3d", lo, hu), uten("uten", lo, hu);
    Pair<3> rrv("rrv", lo, hv), va("va", lo, hv), v3d("v3d", lo, hv), vten("vten", lo, hv);
    Pair<3> rrw("rrw", lo, hw), wa("wa", lo, hw), w3d("w3d", lo, hw), wten("wten", lo, hw);
    Pair<3> ppi("ppi", lo, hs), pp3d("pp3d", lo, hs), piadv("piadv", lo, hs), ppten("ppten", lo, hs),
        ppx("ppx", lo, hs), thv("thv", lo, hs), ppterm("ppterm", lo, hs);
    rru.constant(0); rrv.constant(0); rrw.constant(0);
    ua.random(gen, -10, 12); u3d.random(gen, -10, 12); uten.random(gen, -1e-2, 1e-2);
    va.random(gen, -12, 10); v3d.random(gen, -12, 10); vten.random(gen, -1e-2, 1e-2);
    wa.random(gen, -3, 3); w3d.random(gen, -3, 3); wten.random(gen, -1e-2, 1e-2);
    ppi.random(gen, -1e-3, 1e-3); pp3d.random(gen, -1e-3, 1e-3); piadv.random(gen, 0.8, 1.0);
    ppten.random(gen, -1e-6, 1e-6); ppx.random(gen, -1e-3, 1e-3);
    thv.random(gen, 290, 330); ppterm.random(gen, 0.3, 0.4);
    RadPair r(b);
    r.random(gen, -30, 30);

    Pair<3> mm("mm", lo, hs), ppd("ppd", lo, hs), fpk("fpk", lo, hs), qk("qk", lo, hs), pk1("pk1", lo, hs),
        pk2("pk2", lo, hs), dum1("dum1", lo, hs), dum2("dum2", lo, hs);
    Pair<2> dtu("dtu", {b.ib, b.jb}, {b.ie, b.je}), dtv("dtv", {b.ib, b.jb}, {b.ie, b.je});
    SoundOpenBc obc;
    obc.p = S.p; obc.rad = r.dev();
    obc.uh = uh.d; obc.vh = vh.d; obc.ruh = ruh.d; obc.rvh = rvh.d; obc.rmh = rmh.d; obc.rho0 = rho0.d;
    obc.work = FArray1("t1", {1}, {b.nk}); obc.work2 = FArray1("t2", {1}, {b.nk});

    for (int nrk = 1; nrk <= 3; ++nrk) {
      INFO("nrk = " << nrk);
      const double dttmp = dt / double(3 + 1 - nrk);
      SoundArgs a;
      a.dt = dt; a.dttmp = dttmp; a.alph = alph; a.kdiv = kdiv; a.cp = c.cp;
      a.rdx = 1.0 / S.dx; a.rdy = 1.0 / S.dy; a.rdz = 1.0 / S.dz; a.smeps = 1.0e-60;
      a.nsound = nsound; a.nrk = nrk;
      a.uh = uh.d; a.vh = vh.d; a.uf = uf.d; a.vf = vf.d; a.mh = mh.d; a.mf = mf.d; a.c1 = c1.d; a.c2 = c2.d;
      a.rr0 = rr0.d; a.rf0 = rf0.d; a.dtu = dtu.d; a.dtv = dtv.d; a.dtu0 = dtu0.d; a.dtv0 = dtv0.d;
      a.dum1 = dum1.d; a.dum2 = dum2.d; a.mm = mm.d; a.ppd = ppd.d; a.fpk = fpk.d; a.qk = qk.d; a.pk1 = pk1.d; a.pk2 = pk2.d;
      a.rru = rru.d; a.ua = ua.d; a.u3d = u3d.d; a.uten = uten.d;
      a.rrv = rrv.d; a.va = va.d; a.v3d = v3d.d; a.vten = vten.d;
      a.rrw = rrw.d; a.wa = wa.d; a.w3d = w3d.d; a.wten = wten.d;
      a.ppi = ppi.d; a.pp3d = pp3d.d; a.piadv = piadv.d; a.ppten = ppten.d; a.ppx = ppx.d;
      a.thv = thv.d; a.ppterm = ppterm.d;
      a.obc = &obc;
      sound(bc, a);
      Kokkos::fence();
      cm1f_sound_obc(dt, nrk, dttmp, uh.fp(), vh.fp(), uf.fp(), vf.fp(), mh.fp(), mf.fp(), c1.fp(), c2.fp(),
                     rr0.fp(), rf0.fp(), dtu0.fp(), dtv0.fp(), rru.fp(), ua.fp(), u3d.fp(), uten.fp(),
                     rrv.fp(), va.fp(), v3d.fp(), vten.fp(), rrw.fp(), wa.fp(), w3d.fp(), wten.fp(),
                     ppi.fp(), pp3d.fp(), piadv.fp(), ppten.fp(), ppx.fp(), thv.fp(), ppterm.fp(),
                     r.w.fp(), r.e.fp(), r.s.fp(), r.n.fp(), ruh.fp(), rvh.fp(), rmh.fp(), rho0.fp());
      u3d.compare("u3d"); v3d.compare("v3d"); w3d.compare("w3d"); pp3d.compare("pp3d");
      rru.compare("rru"); rrv.compare("rrv"); rrw.compare("rrw"); ppx.compare("ppx");
      r.compare();
      // solve2.F after sound: irbc=4 phase speeds from ua and the new u3d
      if (sc.irbc == 4) {
        RadBc rd = r.dev();
        Pair<1> ruf("ruf", {b.ib}, {b.ie + 1}), rvf("rvf", {b.jb}, {b.je + 1});
        ruf.random(gen, 0.9, 1.1); rvf.random(gen, 0.9, 1.1);
        radbcew4(S.p, ruf.d, rd, ua.d, u3d.d, dttmp);
        radbcns4(S.p, rvf.d, rd, va.d, v3d.d, dttmp);
        Kokkos::fence();
        cm1f_radbcew4(ruf.fp(), r.w.fp(), r.e.fp(), ua.fp(), u3d.fp(), dttmp);
        cm1f_radbcns4(rvf.fp(), r.s.fp(), r.n.fp(), va.fp(), v3d.fp(), dttmp);
        r.compare();
      }
    }
  }
}
