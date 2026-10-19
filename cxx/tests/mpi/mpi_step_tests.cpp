// MPI step tests: each rank starts from its own Fortran dump (MPI run with
// the same number of ranks, <ref>/<case>/np<N>/rank_RRRR/step_000000),
// advances with Model::step and is compared, halos included, with every
// dumped step of that rank at tolerance 0. The interior is also compared
// with the serial reference of the same case (<ref>/<case>/step_*); that
// difference is printed, and checked to be zero for cases that the Fortran
// itself runs independently of the rank count (see cases/README.md).
#include "driver/cm1k_model.hpp"
#include "share/cm1k_comm.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace cm1k;

namespace {

int nranks() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

// The rank counts a case has MPI references for (its `ranks` file).
bool case_has_ranks(const std::string& case_name, int np) {
  const auto nl = std::filesystem::path(cm1k::testing::case_namelist(case_name));
  std::ifstream f(nl.parent_path() / "ranks");
  int n;
  while (f >> n)
    if (n == np) return true;
  return false;
}

// Collective "any rank": logical or over the ranks.
bool any_rank(bool b) {
  int v = b ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return v != 0;
}

struct Result {
  double rank_max_abs = 0;    // vs the MPI Fortran at this rank count (halos included)
  double serial_max_abs = 0;  // interior vs the serial Fortran
};

// rank_independent: also require the interior to match the serial run bitwise.
Result run_mpi_case(const std::string& case_name, bool turb, bool rank_independent) {
  Result res;
  const int np = nranks();
  if (!case_has_ranks(case_name, np)) {
    WARN(case_name << ": no MPI references at " << np << " ranks (cases/" << case_name << "/ranks)");
    return res;
  }
  const auto ref = cm1k::testing::case_ref(case_name);
  if (ref.empty()) return res;
  const auto nl = Namelist::from_file(cm1k::testing::case_namelist(case_name));
  const auto d = Decomp::make(MPI_COMM_WORLD, nl.geti("nx", 0), nl.geti("ny", 0));
  char rk[32];
  std::snprintf(rk, sizeof rk, "np%d/rank_%04d", np, d.myid);
  const std::string rref = ref + "/" + rk;
  const bool missing = !std::filesystem::exists(DumpReader::step_dir(rref, 0));
  if (any_rank(missing)) {
    if (cm1k::testing::refs_required()) FAIL("missing MPI reference dumps " << rref);
    WARN("missing MPI reference dumps " << rref);
    return res;
  }

  // the Fortran's decomposition (dump_decomp) is ours
  {
    DumpReader r0(DumpReader::step_dir(rref, 0));
    const std::vector<int> want{d.nodex, d.nodey, d.myi, d.myj, d.myi1, d.myi2, d.myj1, d.myj2};
    REQUIRE(r0.decomp() == want);
  }

  Model m(nl, d);
  REQUIRE(m.mpi());
  m.init_from_dump(DumpReader::step_dir(rref, 0));

  std::vector<std::string> fields = {"ua", "va", "wa", "ppi", "tha", "rho", "prs"};
  if (turb) for (const char* n : {"tkea", "kmh", "kmv", "khh", "khv"}) fields.push_back(n);
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(rref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > 0) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  cm1k::testing::limit_steps(steps);
  REQUIRE(!steps.empty());

  const int ni = d.ni, nj = d.nj, oi = d.myi1 - 1, oj = d.myj1 - 1;
  for (int step : steps) {
    while (m.nstep() < step) m.step();
    DumpReader r(DumpReader::step_dir(rref, step));
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    const auto sdir = DumpReader::step_dir(ref, step);
    const bool have_serial = std::filesystem::exists(sdir);
    std::unique_ptr<DumpReader> rs;
    if (have_serial) rs = std::make_unique<DumpReader>(sdir);
    double step_rank = 0, step_serial = 0;
    std::string worst_field;
    for (auto& n : fields) {
      auto a = m.field3(n).to_host();
      auto b = r.read_host<3>(n);
      std::array<int, 3> lo, hi;
      for (int k = 0; k < 3; ++k) { lo[k] = b.lb(k); hi[k] = b.ub(k); }
      const auto st = diff<3>(a, b, lo, hi);
      INFO(case_name << " np" << np << " rank " << d.myid << " step " << step << " field " << n << ": max abs "
                     << st.max_abs << " at (" << (st.where.size() == 3 ? st.where[0] : 0) << ","
                     << (st.where.size() == 3 ? st.where[1] : 0) << ","
                     << (st.where.size() == 3 ? st.where[2] : 0) << ")");
      CHECK(st.max_abs == 0.0);
      if (st.max_abs > step_rank) { step_rank = st.max_abs; worst_field = n; }
      if (have_serial && rs->has(n)) {
        // interior (and all levels of the local array) vs the global array
        auto g = rs->read_host<3>(n);
        double mx = 0;
        for (int k = a.lb(2); k <= a.ub(2); ++k)
          for (int j = 1; j <= nj; ++j)
            for (int i = 1; i <= ni; ++i)
              mx = std::max(mx, std::abs(a(i, j, k) - g(oi + i, oj + j, k)));
        step_serial = std::max(step_serial, mx);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &step_rank, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &step_serial, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    res.rank_max_abs = std::max(res.rank_max_abs, step_rank);
    res.serial_max_abs = std::max(res.serial_max_abs, step_serial);
    if (d.myid == 0)
      std::printf("%s np%d step %d: max |cm1k - Fortran np%d| = %.3e, interior max |cm1k np%d - Fortran serial| = %.3e%s\n",
                  case_name.c_str(), np, step, np, step_rank, np, step_serial, have_serial ? "" : " (no serial ref)");
  }
  if (rank_independent) CHECK(res.serial_max_abs == 0.0);
  return res;
}

} // namespace

TEST_CASE("mpi: mpi_kessler_periodic vs MPI Fortran", "[mpi][step]") {
  run_mpi_case("mpi_kessler_periodic", false, false);
}

TEST_CASE("mpi: mpi_kessler_2x1 vs MPI Fortran", "[mpi][step]") {
  run_mpi_case("mpi_kessler_2x1", false, false);
}

TEST_CASE("mpi: mpi_kessler_tke vs MPI Fortran", "[mpi][step][turb]") {
  run_mpi_case("mpi_kessler_tke", true, false);
}

TEST_CASE("mpi: mpi_kessler_tke_nomc vs MPI Fortran (rank-count independent)", "[mpi][step][turb]") {
  // apmasscon=0: no global sum in the time step, the Fortran is the same at
  // any rank count, and so is cm1k
  run_mpi_case("mpi_kessler_tke_nomc", true, true);
}
