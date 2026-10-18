// diff2.F (diff2def, diff2u, diff2v, diff2w, diff2s) vs cm1k's port, on
// synthetic fields, compared bit for bit (halos included).
#include "dynamics/cm1k_diff2.hpp"

#include "dyn_bridge_utils.hpp"

#include <array>

extern "C" {
void cm1f_diff2_set_input(double kdiff2, int bbc, int tbc, double umove, double vmove);
void cm1f_diff2def(double* uh, double* uf, double* vh, double* vf, double* mh, double* c1, double* c2,
                   double* mf, double* divx, double* rho, double* rr, double* rf, double* t11, double* t12,
                   double* t13, double* t22, double* t23, double* t33, double* ua, double* va, double* wa);
void cm1f_diff2uvw(int which, double* uh, double* uf, double* vh, double* vf, double* mh, double* mf,
                   double* rho, double* rr, double* rf, double* divx, double* ta, double* tb, double* tc,
                   double* ten);
void cm1f_diff2s(double* uh, double* uf, double* vh, double* vf, double* mh, double* mf, double* s,
                 double* sten, double* rho, double* rr, double* rf);
}

using namespace cm1k;
using cm1k::testing::Pair;

namespace {

struct BcCase { int wbc, ebc, sbc, nbc, bbc, tbc; };

void run_diff2(const BcCase& bcc, unsigned seed) {
  const int nx = 10, ny = 8, nz = 9, ng = 3;
  const double dx = 2000, dy = 1500, dz = 400, kdiff2 = 75.0, umove = 3.5, vmove = -2.25;
  cm1f_dyn_set_input(nx, ny, nz, ng, bcc.wbc, bcc.ebc, bcc.sbc, bcc.nbc, dx, dy, dz, 0.6, 0.1, 4);
  cm1f_diff2_set_input(kdiff2, bcc.bbc, bcc.tbc, umove, vmove);

  Namelist nl;
  nl.set("nx", std::to_string(nx)); nl.set("ny", std::to_string(ny)); nl.set("nz", std::to_string(nz));
  const auto b = Bounds::from_namelist(nl);
  REQUIRE(b.ngxy == ng);
  Diff2Params p;
  p.ni = b.ni; p.nj = b.nj; p.nk = b.nk;
  p.kdiff2 = kdiff2; p.rdx = 1.0 / dx; p.rdy = 1.0 / dy; p.rdz = 1.0 / dz; p.smeps = 1.0e-60;
  p.umove = umove; p.vmove = vmove;
  p.wbc = bcc.wbc; p.ebc = bcc.ebc; p.sbc = bcc.sbc; p.nbc = bcc.nbc;
  p.ibw = bcc.wbc != 1; p.ibe = bcc.ebc != 1; p.ibs = bcc.sbc != 1; p.ibn = bcc.nbc != 1;
  p.bbc = bcc.bbc; p.tbc = bcc.tbc;

  std::mt19937_64 gen(seed);
  const std::array<int, 3> lo{b.ib, b.jb, b.kb}, hs{b.ie, b.je, b.ke}, hu{b.ie + 1, b.je, b.ke},
      hv{b.ie, b.je + 1, b.ke}, hw{b.ie, b.je, b.ke + 1};
  Pair<1> uh("uh", {b.ib}, {b.ie}), vh("vh", {b.jb}, {b.je}), uf("uf", {b.ib}, {b.ie + 1}), vf("vf", {b.jb}, {b.je + 1});
  uh.random(gen, 0.9, 1.1); vh.random(gen, 0.9, 1.1); uf.random(gen, 0.9, 1.1); vf.random(gen, 0.9, 1.1);
  Pair<3> mh("mh", lo, hs), mf("mf", lo, hw), c1("c1", lo, hs), c2("c2", lo, hs);
  mh.random(gen, 0.8, 1.2); mf.random(gen, 0.8, 1.2); c1.random(gen, 0.4, 0.6); c2.random(gen, 0.4, 0.6);
  Pair<3> rho("rho", lo, hs), rr("rr", lo, hs), rf("rf", lo, hs), divx("divx", lo, hs);
  rho.random(gen, 0.5, 1.2); rf.random(gen, 0.5, 1.2);
  rr.init([&](size_t n) { return 1.0 / rho.f.data()[n]; });
  divx.random(gen, -1e-3, 1e-3);
  Pair<3> ua("ua", lo, hu), va("va", lo, hv), wa("wa", lo, hw);
  ua.random(gen, -20, 20); va.random(gen, -20, 20); wa.random(gen, -5, 5);
  Pair<3> t11("t11", lo, hs), t12("t12", lo, hs), t13("t13", lo, hs), t22("t22", lo, hs), t23("t23", lo, hs),
      t33("t33", lo, hs);
  for (auto* t : {&t11, &t12, &t13, &t22, &t23, &t33}) t->random(gen, -1, 1);   // stale values

  Grid g;
  g.b = b; g.rdx = p.rdx; g.rdy = p.rdy; g.rdz = p.rdz;
  g.uh = uh.d; g.vh = vh.d; g.uf = uf.d; g.vf = vf.d; g.mh = mh.d; g.mf = mf.d; g.c1 = c1.d; g.c2 = c2.d;
  Diff2Work w;
  w.allocate(b);

  // diff2def
  diff2def(p, g, divx.d, rho.d, rf.d, t11.d, t12.d, t13.d, t22.d, t23.d, t33.d, ua.d, va.d, wa.d);
  Kokkos::fence();
  cm1f_diff2def(uh.fp(), uf.fp(), vh.fp(), vf.fp(), mh.fp(), c1.fp(), c2.fp(), mf.fp(), divx.fp(), rho.fp(),
                rr.fp(), rf.fp(), t11.fp(), t12.fp(), t13.fp(), t22.fp(), t23.fp(), t33.fp(), ua.fp(),
                va.fp(), wa.fp());
  divx.compare("divx");
  t11.compare("t11"); t12.compare("t12"); t13.compare("t13");
  t22.compare("t22"); t23.compare("t23"); t33.compare("t33");

  // diff2u / diff2v / diff2w on the stresses diff2def produced
  Pair<3> uten("uten", lo, hu), vten("vten", lo, hv), wten("wten", lo, hw);
  uten.random(gen, -1e-2, 1e-2); vten.random(gen, -1e-2, 1e-2); wten.random(gen, -1e-2, 1e-2);
  diff2u(p, g, w.dum, w.diffx, w.diffy, w.diffz, uten.d, rho.d, divx.d, t11.d, t12.d, t13.d);
  diff2v(p, g, w.dum, w.diffx, w.diffy, w.diffz, vten.d, rho.d, divx.d, t22.d, t12.d, t23.d);
  diff2w(p, g, w.dum, w.diffx, w.diffy, w.diffz, wten.d, rho.d, rf.d, divx.d, t33.d, t13.d, t23.d);
  Kokkos::fence();
  cm1f_diff2uvw(1, uh.fp(), uf.fp(), vh.fp(), vf.fp(), mh.fp(), mf.fp(), rho.fp(), rr.fp(), rf.fp(), divx.fp(),
                t11.fp(), t12.fp(), t13.fp(), uten.fp());
  cm1f_diff2uvw(2, uh.fp(), uf.fp(), vh.fp(), vf.fp(), mh.fp(), mf.fp(), rho.fp(), rr.fp(), rf.fp(), divx.fp(),
                t22.fp(), t12.fp(), t23.fp(), vten.fp());
  cm1f_diff2uvw(3, uh.fp(), uf.fp(), vh.fp(), vf.fp(), mh.fp(), mf.fp(), rho.fp(), rr.fp(), rf.fp(), divx.fp(),
                t33.fp(), t13.fp(), t23.fp(), wten.fp());
  uten.compare("uten"); vten.compare("vten"); wten.compare("wten");

  // diff2s
  Pair<3> s("s", lo, hs), sten("sten", lo, hs);
  s.random(gen, -2, 2); sten.random(gen, -1e-3, 1e-3);
  diff2s(p, g, w.dum, w.diffx, w.diffy, w.diffz, s.d, sten.d, rho.d, rr.d, rf.d);
  Kokkos::fence();
  cm1f_diff2s(uh.fp(), uf.fp(), vh.fp(), vf.fp(), mh.fp(), mf.fp(), s.fp(), sten.fp(), rho.fp(), rr.fp(), rf.fp());
  sten.compare("sten");
}

} // namespace

TEST_CASE("diff2.F 2nd-order diffusion (diff2def, diff2u/v/w/s)", "[bridge][dyn][diff2]") {
  unsigned seed = 7;
  for (const BcCase c : {BcCase{1, 1, 1, 1, 1, 1}, BcCase{1, 1, 1, 1, 2, 2}, BcCase{3, 4, 4, 3, 1, 2},
                         BcCase{4, 3, 3, 4, 2, 1}, BcCase{2, 2, 2, 2, 1, 1}}) {
    INFO("wbc " << c.wbc << " ebc " << c.ebc << " sbc " << c.sbc << " nbc " << c.nbc << " bbc " << c.bbc
                << " tbc " << c.tbc);
    run_diff2(c, seed++);
  }
}
