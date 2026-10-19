#include "cm1k_dump_io.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cm1k {

namespace fs = std::filesystem;

std::string DumpReader::step_dir(const std::string& root, int nstep) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "step_%06d", nstep);
  return (fs::path(root) / buf).string();
}

DumpReader::DumpReader(const std::string& step_dir) : m_dir(step_dir) {
  std::ifstream f(fs::path(step_dir) / "index.txt");
  if (!f) throw std::runtime_error("cm1k: cannot open dump index in '" + step_dir + "'");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::string name;
    if (!(is >> name)) continue;
    if (name == "#step_time") { is >> m_nstep >> m_mtime; continue; }
    if (name == "#adapt_dt") {   // adapt_dt = 1 (src/cm1dump.F dump_adapt)
      double v;
      while (is >> v) m_adapt.push_back(v);
      continue;
    }
    if (name[0] == '#') continue;   // other comment lines
    DumpEntry e;
    e.name = name;
    int nd = 0;
    is >> e.kind >> nd;
    e.lb.resize(nd); e.ub.resize(nd);
    for (int d = 0; d < nd; ++d) is >> e.lb[d] >> e.ub[d];
    if (!is) throw std::runtime_error("cm1k: bad dump index line: " + line);
    m_entries[name] = e;
  }
}

const DumpEntry& DumpReader::entry(const std::string& name) const {
  auto it = m_entries.find(name);
  if (it == m_entries.end()) throw std::runtime_error("cm1k: '" + name + "' not in dump " + m_dir);
  return it->second;
}

std::vector<std::string> DumpReader::names() const {
  std::vector<std::string> n;
  for (auto& [k, v] : m_entries) n.push_back(k);
  return n;
}

std::vector<double> DumpReader::read_raw(const std::string& name) const {
  const auto& e = entry(name);
  if (e.kind != 8) throw std::runtime_error("cm1k: dump '" + name + "' is not float64 (build CM1 with USE_DOUBLE=true)");
  std::vector<double> buf(e.size());
  std::ifstream f(fs::path(m_dir) / (name + ".bin"), std::ios::binary);
  if (!f) throw std::runtime_error("cm1k: cannot open dump file for '" + name + "'");
  f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size() * sizeof(double)));
  if (!f) throw std::runtime_error("cm1k: short read for '" + name + "'");
  return buf;
}

template <int Rank>
FArray<Rank, HostSpace> DumpReader::read_host(const std::string& name) const {
  const auto& e = entry(name);
  if (int(e.lb.size()) != Rank)
    throw std::runtime_error("cm1k: dump '" + name + "' has rank " + std::to_string(e.lb.size()));
  std::array<int, Rank> lb, ub;
  for (int d = 0; d < Rank; ++d) { lb[d] = e.lb[d]; ub[d] = e.ub[d]; }
  FArray<Rank, HostSpace> h(name, lb, ub);
  auto raw = read_raw(name);
  std::copy(raw.begin(), raw.end(), h.data());
  return h;
}

template <int Rank>
void DumpReader::read_into(const std::string& name, const FArray<Rank>& dst) const {
  auto h = read_host<Rank>(name);
  for (int d = 0; d < Rank; ++d)
    if (h.lb(d) != dst.lb(d) || h.ub(d) != dst.ub(d))
      throw std::runtime_error("cm1k: bounds mismatch reading '" + name + "' (dim " + std::to_string(d + 1) +
                               ": dump " + std::to_string(h.lb(d)) + ":" + std::to_string(h.ub(d)) +
                               ", cm1k " + std::to_string(dst.lb(d)) + ":" + std::to_string(dst.ub(d)) + ")");
  Kokkos::deep_copy(dst.view(), h.view());
}

DumpWriter::DumpWriter(const std::string& step_dir, int nstep, double mtime) : m_dir(step_dir) {
  fs::create_directories(step_dir);
  auto* f = new std::ofstream(fs::path(step_dir) / "index.txt");
  char buf[128];
  std::snprintf(buf, sizeof(buf), "#step_time %10d %24.16e\n", nstep, mtime);
  *f << buf;
  m_index = f;
}

void DumpWriter::comment(const std::string& line) {
  *static_cast<std::ofstream*>(m_index) << line << "\n";
}

DumpWriter::~DumpWriter() { delete static_cast<std::ofstream*>(m_index); }

template <int Rank>
void DumpWriter::write(const std::string& name, const FArray<Rank>& a) {
  auto h = a.to_host();
  std::ofstream f(fs::path(m_dir) / (name + ".bin"), std::ios::binary);
  f.write(reinterpret_cast<const char*>(h.data()), std::streamsize(h.size() * sizeof(double)));
  auto& idx = *static_cast<std::ofstream*>(m_index);
  idx << name << " 8 " << Rank;
  for (int d = 0; d < Rank; ++d) idx << " " << a.lb(d) << " " << a.ub(d);
  idx << "\n";
}

template <int Rank>
DiffStats diff(const FArray<Rank, HostSpace>& a, const FArray<Rank, HostSpace>& ref,
               const std::array<int, Rank>& lo, const std::array<int, Rank>& hi) {
  DiffStats s;
  std::array<int, 4> l{0, 0, 0, 0}, h{0, 0, 0, 0};
  for (int d = 0; d < Rank; ++d) { l[d] = lo[d]; h[d] = hi[d]; }
  for (int n = l[3]; n <= h[3]; ++n)
  for (int k = l[2]; k <= h[2]; ++k)
  for (int j = l[1]; j <= h[1]; ++j)
  for (int i = l[0]; i <= h[0]; ++i) {
    double x, r;
    if constexpr (Rank == 1) { x = a(i); r = ref(i); }
    if constexpr (Rank == 2) { x = a(i, j); r = ref(i, j); }
    if constexpr (Rank == 3) { x = a(i, j, k); r = ref(i, j, k); }
    if constexpr (Rank == 4) { x = a(i, j, k, n); r = ref(i, j, k, n); }
    const double d = std::abs(x - r);
    s.ref_max = std::max(s.ref_max, std::abs(r));
    if (d > s.max_abs || std::isnan(d)) {
      s.max_abs = std::isnan(d) ? INFINITY : d;
      s.where = {i, j, k, n};
      s.where.resize(Rank);
    }
  }
  s.rel = s.ref_max > 0 ? s.max_abs / s.ref_max : s.max_abs;
  return s;
}

#define CM1K_INST(R)                                                                              \
  template FArray<R, HostSpace> DumpReader::read_host<R>(const std::string&) const;               \
  template void DumpReader::read_into<R>(const std::string&, const FArray<R>&) const;             \
  template void DumpWriter::write<R>(const std::string&, const FArray<R>&);                       \
  template DiffStats diff<R>(const FArray<R, HostSpace>&, const FArray<R, HostSpace>&,             \
                             const std::array<int, R>&, const std::array<int, R>&);
CM1K_INST(1)
CM1K_INST(2)
CM1K_INST(3)
CM1K_INST(4)
#undef CM1K_INST

} // namespace cm1k
