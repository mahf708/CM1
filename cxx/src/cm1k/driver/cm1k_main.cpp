// cm1k: run a CM1 namelist with the C++/Kokkos port.
//
//   cm1k [namelist.input] [--from-dump DIR/step_000000] [--steps N]
//        [--dump-dir OUT] [--dump-every N]
#include "cm1k_model.hpp"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::string nml = "namelist.input", from_dump, dump_dir;
    int steps = -1, dump_every = 1;
    for (int a = 1; a < argc; ++a) {
      std::string s = argv[a];
      auto next = [&]() { return std::string(argv[++a]); };
      if (s == "--from-dump") from_dump = next();
      else if (s == "--steps") steps = std::stoi(next());
      else if (s == "--dump-dir") dump_dir = next();
      else if (s == "--dump-every") dump_every = std::stoi(next());
      else if (s.rfind("--kokkos", 0) == 0) continue;
      else nml = s;
    }
    cm1k::Model m(cm1k::Namelist::from_file(nml));
    if (!from_dump.empty()) m.init_from_dump(from_dump);
    else m.init();
    if (!dump_dir.empty()) m.write_dump(dump_dir);
    // cm1.F: iconly=1 stops after the initial conditions; run_time > 0
    // runs for run_time seconds from the start time instead of to timax
    const auto& nl = m.namelist();
    if (nl.geti("iconly", 0) == 1 && steps < 0) steps = 0;
    double timax = nl.getr("timax", 0.0);
    const double run_time = nl.getr("run_time", -999.0);
    if (run_time > 0.0) timax = m.mtime() + run_time;
    while (steps < 0 ? m.mtime() < timax : m.nstep() < steps) {
      m.step();
      std::printf("cm1k: nstep %d  mtime %.3f\n", m.nstep(), m.mtime());
      if (!dump_dir.empty() && m.nstep() % dump_every == 0) m.write_dump(dump_dir);
    }
  } catch (std::exception& e) {
    std::fprintf(stderr, "cm1k error: %s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
