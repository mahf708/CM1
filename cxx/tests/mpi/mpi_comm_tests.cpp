// MPI decomposition (cm1.F/param.F) and halo exchanges (comm.F) on
// synthetic fields: no reference dumps needed. Run at 2 and 4 ranks.
#include "share/cm1k_comm.hpp"
#include "share/cm1k_types.hpp"

#include <catch2/catch.hpp>

#include <mpi.h>

#include <functional>
#include <string>

using namespace cm1k;

namespace {

int nranks() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

// a field value that identifies the global point (periodic)
KOKKOS_INLINE_FUNCTION double gval(int gi, int gj, int k, int nx, int ny) {
  const int i = ((gi - 1) % nx + nx) % nx + 1;
  const int j = ((gj - 1) % ny + ny) % ny + 1;
  return double(i) + 1000.0 * double(j) + 1.0e6 * double(k);
}

// Fill a with gval on [i1,i2] x [j1,j2] x [k1,k2] (local indices), -1 elsewhere.
void fill(const FArray3& a, const Decomp& d, int i1, int i2, int j1, int j2, int k1, int k2) {
  a.fill(-1.0);
  const int oi = d.myi1 - 1, oj = d.myj1 - 1, nx = d.nx, ny = d.ny;
  Kokkos::parallel_for("fill", box3(i1, i2, j1, j2, k1, k2), KOKKOS_LAMBDA(int i, int j, int k) {
    a(i, j, k) = gval(oi + i, oj + j, k, nx, ny);
  });
}

// Every point of a: gval where `expect(i,j,k)` says the exchange (or the
// fill) set it, -1 elsewhere. Returns the number of wrong points.
int check(const FArray3& a, const Decomp& d, const std::function<bool(int, int, int)>& expect) {
  auto h = a.to_host();
  int bad = 0;
  for (int k = h.lb(2); k <= h.ub(2); ++k)
    for (int j = h.lb(1); j <= h.ub(1); ++j)
      for (int i = h.lb(0); i <= h.ub(0); ++i) {
        const double want = expect(i, j, k) ? gval(d.myi1 - 1 + i, d.myj1 - 1 + j, k, d.nx, d.ny) : -1.0;
        if (h(i, j, k) != want) ++bad;
      }
  return bad;
}

bool in(int x, int a, int b) { return x >= a && x <= b; }

} // namespace

TEST_CASE("mpi: decomposition matches cm1.F", "[mpi]") {
  const int np = nranks();
  struct Exp { int nx, ny, np, nodex, nodey; };
  // cm1.F: nodey = the n in numprocs..1 (first minimum) of |(nx/(np/n))/(ny/n) - 2|
  const Exp exps[] = {{32, 32, 2, 1, 2}, {32, 32, 4, 2, 2}, {40, 40, 2, 1, 2}, {40, 40, 4, 2, 2},
                      {48, 24, 2, 2, 1}, {48, 24, 4, 2, 2}, {24, 24, 2, 1, 2}, {24, 24, 4, 2, 2}};
  for (const auto& e : exps) {
    if (e.np != np) continue;
    const auto d = Decomp::make(MPI_COMM_WORLD, e.nx, e.ny);
    INFO("nx=" << e.nx << " ny=" << e.ny << " np=" << np);
    CHECK(d.nodex == e.nodex);
    CHECK(d.nodey == e.nodey);
    CHECK(d.myi2 - d.myi1 + 1 == d.ni);
    CHECK(d.myj2 - d.myj1 + 1 == d.nj);
    // the subdomains tile the grid
    int cells = d.ni * d.nj;
    MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    CHECK(cells == e.nx * e.ny);
  }
}

TEST_CASE("mpi: comm.F halo exchanges, periodic", "[mpi]") {
  const int nx = 24, ny = 20, nk = 4, ng = 3;
  const auto d = Decomp::make(MPI_COMM_WORLD, nx, ny);
  const EdgeFlags e = EdgeFlags::make(d, 1, 1, 1, 1);
  Comm c(d, e, ng, nk);
  const int ni = d.ni, nj = d.nj;
  const std::array<int, 3> lo{1 - ng, 1 - ng, 0};
  FArray3 s("s", lo, {ni + ng, nj + ng, nk + 1});
  FArray3 u("u", lo, {ni + ng + 1, nj + ng, nk + 1});
  FArray3 v("v", lo, {ni + ng, nj + ng + 1, nk + 1});

  SECTION("comm_3s: ngxy-wide halos, no corners") {
    fill(s, d, 1, ni, 1, nj, 1, nk);
    c.comm_3s_start(s, "s");
    c.comm_3s_end(s, "s");
    CHECK(check(s, d, [&](int i, int j, int k) {
      if (!in(k, 1, nk)) return false;
      return (in(i, 1 - ng, ni + ng) && in(j, 1, nj)) || (in(i, 1, ni) && in(j, 1 - ng, nj + ng));
    }) == 0);
  }
  SECTION("comm_3t: k = 1..nk+1") {
    fill(s, d, 1, ni, 1, nj, 1, nk + 1);
    c.comm_3t_start(s, "t");
    c.comm_3t_end(s, "t");
    CHECK(check(s, d, [&](int i, int j, int k) {
      if (!in(k, 1, nk + 1)) return false;
      return (in(i, 1 - ng, ni + ng) && in(j, 1, nj)) || (in(i, 1, ni) && in(j, 1 - ng, nj + ng));
    }) == 0);
  }
  SECTION("comm_3w: k = 2..nk") {
    fill(s, d, 1, ni, 1, nj, 1, nk + 1);
    c.comm_3w_start(s, "w");
    c.comm_3w_end(s, "w");
    CHECK(check(s, d, [&](int i, int j, int k) {
      if (in(k, 1, nk + 1) && in(i, 1, ni) && in(j, 1, nj)) return true;
      if (!in(k, 2, nk)) return false;
      return (in(i, 1 - ng, ni + ng) && in(j, 1, nj)) || (in(i, 1, ni) && in(j, 1 - ng, nj + ng));
    }) == 0);
  }
  SECTION("comm_3u: staggered in x") {
    fill(u, d, 1, ni + 1, 1, nj, 1, nk);
    c.comm_3u_start(u, "u");
    c.comm_3u_end(u, "u");
    CHECK(check(u, d, [&](int i, int j, int k) {
      if (!in(k, 1, nk)) return false;
      return (in(i, 1 - ng, ni + 1 + ng) && in(j, 1, nj)) || (in(i, 1, ni + 1) && in(j, 1 - ng, nj + ng));
    }) == 0);
  }
  SECTION("comm_3v: staggered in y") {
    fill(v, d, 1, ni, 1, nj + 1, 1, nk);
    c.comm_3v_start(v, "v");
    c.comm_3v_end(v, "v");
    CHECK(check(v, d, [&](int i, int j, int k) {
      if (!in(k, 1, nk)) return false;
      return (in(i, 1 - ng, ni + ng) && in(j, 1, nj + 1)) || (in(i, 1, ni) && in(j, 1 - ng, nj + 1 + ng));
    }) == 0);
  }
  SECTION("comm_1s, then getcorner") {
    fill(s, d, 1, ni, 1, nj, 1, nk);
    c.comm_1s_start(s, "s");
    c.comm_1s_end(s, "s");
    auto one = [&](int i, int j, int k) {
      if (!in(k, 1, nk)) return false;
      return (in(i, 0, ni + 1) && in(j, 1, nj)) || (in(i, 1, ni) && in(j, 0, nj + 1));
    };
    CHECK(check(s, d, one) == 0);
    c.getcorner(s);
    CHECK(check(s, d, [&](int i, int j, int k) {
      return one(i, j, k) || (in(k, 1, nk) && (i == 0 || i == ni + 1) && (j == 0 || j == nj + 1));
    }) == 0);
  }
  SECTION("comm_2we and comm_2sn: two points, one direction") {
    fill(s, d, 1, ni, 1, nj, 1, nk);
    c.comm_2we_start(s, "x");
    c.comm_2we_end(s, "x");
    CHECK(check(s, d, [&](int i, int j, int k) {
      return in(k, 1, nk) && in(i, -1, ni + 2) && in(j, 1, nj);
    }) == 0);
    fill(s, d, 1, ni, 1, nj, 1, nk);
    c.comm_2sn_start(s, "y");
    c.comm_2sn_end(s, "y");
    CHECK(check(s, d, [&](int i, int j, int k) {
      return in(k, 1, nk) && in(i, 1, ni) && in(j, -1, nj + 2);
    }) == 0);
  }
  SECTION("comm_1s2d and comm_2d_corner on one level") {
    fill(s, d, 1, ni, 1, nj, 1, nk);
    c.comm_1s2d_start(s, 2, "a");
    c.comm_1s2d_end(s, 2, "a");
    c.comm_2d_corner(s, 2);
    CHECK(check(s, d, [&](int i, int j, int k) {
      if (!in(k, 1, nk)) return false;
      if (in(i, 1, ni) && in(j, 1, nj)) return true;
      return k == 2 && in(i, 0, ni + 1) && in(j, 0, nj + 1);
    }) == 0);
  }
  SECTION("reductions") {
    double v[2] = {1.0 + d.myid, -double(d.myid)};
    c.allreduce_sum(v, 2);
    CHECK(v[0] == double(d.numprocs * (d.numprocs + 1) / 2));
    double m = d.myid;
    c.allreduce_max(&m, 1);
    CHECK(m == double(d.numprocs - 1));
    double w[3] = {1.0, 2.0, 3.0};
    const int h = c.iallreduce_sum(w, 3);
    c.iwait(h);
    CHECK(w[2] == 3.0 * d.numprocs);
  }
}
