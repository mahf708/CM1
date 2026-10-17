// sound.F (psolver=3) vs cm1k::sound: three RK stages in sequence on
// synthetic fields, compared bit for bit (halos included).
#include "dynamics/cm1k_sound.hpp"
#include "share/cm1k_constants.hpp"

#include "dyn_bridge_utils.hpp"

#include <cmath>

extern "C" {
void cm1f_sound(double dt, int nrk, double dttmp, double* uh, double* vh, double* uf, double* vf,
                double* mh, double* mf, double* c1, double* c2, double* rr0, double* rf0,
                double* dtu0, double* dtv0,
                double* rru, double* ua, double* u3d, double* uten,
                double* rrv, double* va, double* v3d, double* vten,
                double* rrw, double* wa, double* w3d, double* wten,
                double* ppi, double* pp3d, double* piadv, double* ppten, double* ppx,
                double* thv, double* ppterm);
}

using namespace cm1k;
using cm1k::testing::Pair;

TEST_CASE("sound.F acoustic solver", "[bridge][dyn][sound]") {
  const int nx = 10, ny = 8, nz = 9, ng = 3;
  const double dx = 2000, dy = 1500, dz = 400, alph = 0.6, kdiv = 0.1, dt = 6.0;
  const int nsound = sound_nsound(3, nx, ny, dx, dy, dz, 300.0, dt);
  REQUIRE(nsound == 4);
  cm1f_dyn_set_input(nx, ny, nz, ng, 1, 1, 1, 1, dx, dy, dz, alph, kdiv, nsound);

  Namelist nl;
  nl.set("nx", std::to_string(nx)); nl.set("ny", std::to_string(ny)); nl.set("nz", std::to_string(nz));
  const auto b = Bounds::from_namelist(nl);
  const auto bc = BcParams::make(b, nl);
  const auto c = Constants::make(0);
  std::mt19937_64 gen(42);

  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hs{b.ie, b.je, b.ke}, hu{b.ie + 1, b.je, b.ke},
      hv{b.ie, b.je + 1, b.ke}, hw{b.ie, b.je, b.ke + 1};
  Pair<1> uh("uh", {b.ib}, {b.ie}), vh("vh", {b.jb}, {b.je}), uf("uf", {b.ib}, {b.ie + 1}), vf("vf", {b.jb}, {b.je + 1});
  uh.random(gen, 0.9, 1.1); vh.random(gen, 0.9, 1.1); uf.random(gen, 0.9, 1.1); vf.random(gen, 0.9, 1.1);
  Pair<3> mh("mh", lo, hs), mf("mf", lo, hw), c1("c1", lo, hs), c2("c2", lo, hs), rr0("rr0", lo, hs), rf0("rf0", lo, hs);
  mh.random(gen, 0.8, 1.2); mf.random(gen, 0.8, 1.2);
  c1.random(gen, 0.4, 0.6);
  {
    auto p1 = c1.f.data(); auto p2 = c2.f.data();
    for (size_t n = 0; n < c2.f.size(); ++n) p2[n] = 1.0 - p1[n];
    c2.d.deep_copy_from(c2.f);
  }
  rr0.random(gen, 0.8, 2.0); rf0.random(gen, 0.5, 1.2);
  Pair<2> dtu0("dtu0", {b.ib, b.jb}, {b.ie, b.je}), dtv0("dtv0", {b.ib, b.jb}, {b.ie, b.je});
  dtu0.constant(1.0); dtv0.constant(1.0);

  Pair<3> rru("rru", lo, hu), ua("ua", lo, hu), u3d("u3d", lo, hu), uten("uten", lo, hu);
  Pair<3> rrv("rrv", lo, hv), va("va", lo, hv), v3d("v3d", lo, hv), vten("vten", lo, hv);
  Pair<3> rrw("rrw", lo, hw), wa("wa", lo, hw), w3d("w3d", lo, hw), wten("wten", lo, hw);
  Pair<3> ppi("ppi", lo, hs), pp3d("pp3d", lo, hs), piadv("piadv", lo, hs), ppten("ppten", lo, hs),
      ppx("ppx", lo, hs), thv("thv", lo, hs), ppterm("ppterm", lo, hs);
  rru.constant(0); rrv.constant(0); rrw.constant(0);
  ua.random(gen, -10, 10); u3d.random(gen, -10, 10); uten.random(gen, -1e-2, 1e-2);
  va.random(gen, -10, 10); v3d.random(gen, -10, 10); vten.random(gen, -1e-2, 1e-2);
  wa.random(gen, -3, 3); w3d.random(gen, -3, 3); wten.random(gen, -1e-2, 1e-2);
  ppi.random(gen, -1e-3, 1e-3); pp3d.random(gen, -1e-3, 1e-3); piadv.random(gen, 0.8, 1.0);
  ppten.random(gen, -1e-6, 1e-6); ppx.random(gen, -1e-3, 1e-3);
  thv.random(gen, 290, 330); ppterm.random(gen, 0.3, 0.4);

  Pair<3> mm("mm", lo, hs), ppd("ppd", lo, hs), fpk("fpk", lo, hs), qk("qk", lo, hs), pk1("pk1", lo, hs),
      pk2("pk2", lo, hs), dum1("dum1", lo, hs), dum2("dum2", lo, hs);
  Pair<2> dtu("dtu", {b.ib, b.jb}, {b.ie, b.je}), dtv("dtv", {b.ib, b.jb}, {b.ie, b.je});

  for (int nrk = 1; nrk <= 3; ++nrk) {
    INFO("nrk = " << nrk);
    const double dttmp = dt / double(3 + 1 - nrk);
    SoundArgs a;
    a.dt = dt; a.dttmp = dttmp; a.alph = alph; a.kdiv = kdiv; a.cp = c.cp;
    a.rdx = 1.0 / dx; a.rdy = 1.0 / dy; a.rdz = 1.0 / dz; a.smeps = 1.0e-60;
    a.nsound = nsound; a.nrk = nrk;
    a.uh = uh.d; a.vh = vh.d; a.uf = uf.d; a.vf = vf.d; a.mh = mh.d; a.mf = mf.d; a.c1 = c1.d; a.c2 = c2.d;
    a.rr0 = rr0.d; a.rf0 = rf0.d; a.dtu = dtu.d; a.dtv = dtv.d; a.dtu0 = dtu0.d; a.dtv0 = dtv0.d;
    a.dum1 = dum1.d; a.dum2 = dum2.d; a.mm = mm.d; a.ppd = ppd.d; a.fpk = fpk.d; a.qk = qk.d; a.pk1 = pk1.d; a.pk2 = pk2.d;
    a.rru = rru.d; a.ua = ua.d; a.u3d = u3d.d; a.uten = uten.d;
    a.rrv = rrv.d; a.va = va.d; a.v3d = v3d.d; a.vten = vten.d;
    a.rrw = rrw.d; a.wa = wa.d; a.w3d = w3d.d; a.wten = wten.d;
    a.ppi = ppi.d; a.pp3d = pp3d.d; a.piadv = piadv.d; a.ppten = ppten.d; a.ppx = ppx.d;
    a.thv = thv.d; a.ppterm = ppterm.d;
    sound(bc, a);
    Kokkos::fence();
    cm1f_sound(dt, nrk, dttmp, uh.fp(), vh.fp(), uf.fp(), vf.fp(), mh.fp(), mf.fp(), c1.fp(), c2.fp(),
               rr0.fp(), rf0.fp(), dtu0.fp(), dtv0.fp(), rru.fp(), ua.fp(), u3d.fp(), uten.fp(),
               rrv.fp(), va.fp(), v3d.fp(), vten.fp(), rrw.fp(), wa.fp(), w3d.fp(), wten.fp(),
               ppi.fp(), pp3d.fp(), piadv.fp(), ppten.fp(), ppx.fp(), thv.fp(), ppterm.fp());
    u3d.compare("u3d"); v3d.compare("v3d"); w3d.compare("w3d"); pp3d.compare("pp3d");
    rru.compare("rru"); rrv.compare("rrv"); rrw.compare("rrw"); ppx.compare("ppx");
  }
}
