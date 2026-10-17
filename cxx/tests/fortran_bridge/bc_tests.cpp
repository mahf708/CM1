// bc.F (bcs, bcp, bcu, bcv, bcw) vs cm1k_bc: bit-for-bit on every element,
// halos included, for each lateral boundary type.
#include "dynamics/cm1k_bc.hpp"

#include "dyn_bridge_utils.hpp"

extern "C" {
void cm1f_bcs(double*);
void cm1f_bcp(double*);
void cm1f_bcu(double*);
void cm1f_bcv(double*);
void cm1f_bcw(double*, int);
}

using namespace cm1k;
using cm1k::testing::Pair;

TEST_CASE("bc.F boundary conditions", "[bridge][dyn][bc]") {
  const int nx = 12, ny = 9, nz = 6, ng = 3;
  struct Combo { int w, e, s, n; };
  const Combo combos[] = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4},
                          {1, 1, 3, 3}, {2, 3, 1, 1}, {4, 2, 2, 3}};
  std::mt19937_64 gen(1234);
  for (const auto& cb : combos) {
    INFO("wbc,ebc,sbc,nbc = " << cb.w << "," << cb.e << "," << cb.s << "," << cb.n);
    cm1f_dyn_set_input(nx, ny, nz, ng, cb.w, cb.e, cb.s, cb.n, 2000., 2000., 500., 0.6, 0.1, 4);
    Namelist nl;
    nl.set("nx", std::to_string(nx)); nl.set("ny", std::to_string(ny)); nl.set("nz", std::to_string(nz));
    nl.set("wbc", std::to_string(cb.w)); nl.set("ebc", std::to_string(cb.e));
    nl.set("sbc", std::to_string(cb.s)); nl.set("nbc", std::to_string(cb.n));
    const auto b = Bounds::from_namelist(nl);
    REQUIRE(b.ngxy == ng);
    BcParams p;
    // BcParams::make rejects one-sided periodic combos; set fields directly.
    p.wbc = cb.w; p.ebc = cb.e; p.sbc = cb.s; p.nbc = cb.n;
    p.ni = b.ni; p.nj = b.nj; p.nk = b.nk; p.ngxy = b.ngxy;
    p.ib = b.ib; p.ie = b.ie; p.jb = b.jb; p.je = b.je;

    Pair<3> s("s", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke});
    s.random(gen, -1, 1); bcs(p, s.d); cm1f_bcs(s.fp()); s.compare("bcs");
    s.random(gen, -1, 1); bcp(p, s.d); cm1f_bcp(s.fp()); s.compare("bcp");
    Pair<3> u("u", {b.ib, b.jb, b.kb}, {b.ie + 1, b.je, b.ke});
    u.random(gen, -1, 1); bcu(p, u.d); cm1f_bcu(u.fp()); u.compare("bcu");
    Pair<3> v("v", {b.ib, b.jb, b.kb}, {b.ie, b.je + 1, b.ke});
    v.random(gen, -1, 1); bcv(p, v.d); cm1f_bcv(v.fp()); v.compare("bcv");
    Pair<3> w("w", {b.ib, b.jb, b.kb}, {b.ie, b.je, b.ke + 1});
    for (int flag : {0, 1}) {
      w.random(gen, -1, 1); bcw(p, w.d, flag); cm1f_bcw(w.fp(), flag); w.compare("bcw");
    }
  }
}
