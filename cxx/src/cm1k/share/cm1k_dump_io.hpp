#ifndef CM1K_DUMP_IO_HPP
#define CM1K_DUMP_IO_HPP

#include "cm1k_farray.hpp"

#include <map>
#include <string>
#include <vector>

namespace cm1k {

// Reader/writer for the raw reference dumps written by src/cm1dump.F
// (CM1_DUMP_DIR=... ./cm1.exe) and by cm1k itself.
//
//   <dir>/step_NNNNNN/index.txt   "#step_time <nstep> <mtime>" then one line
//                                 per array: name kind ndims lb1 ub1 ...
//   <dir>/step_NNNNNN/<name>.bin  raw float64, Fortran order, halos included
struct DumpEntry {
  std::string name;
  int kind = 8;
  std::vector<int> lb, ub;
  size_t size() const {
    size_t n = 1;
    for (size_t d = 0; d < lb.size(); ++d) n *= size_t(ub[d] - lb[d] + 1);
    return n;
  }
};

class DumpReader {
public:
  explicit DumpReader(const std::string& step_dir);
  int nstep() const { return m_nstep; }
  double mtime() const { return m_mtime; }
  bool has(const std::string& name) const { return m_entries.count(name) > 0; }
  const DumpEntry& entry(const std::string& name) const;
  std::vector<std::string> names() const;

  // Read into a host array with the dumped bounds.
  std::vector<double> read_raw(const std::string& name) const;
  template <int Rank> FArray<Rank, HostSpace> read_host(const std::string& name) const;
  // Read and copy to device; `dst` must have the same bounds as the dump.
  template <int Rank> void read_into(const std::string& name, const FArray<Rank>& dst) const;

  static std::string step_dir(const std::string& root, int nstep);

private:
  std::string m_dir;
  int m_nstep = 0;
  double m_mtime = 0;
  std::map<std::string, DumpEntry> m_entries;
};

class DumpWriter {
public:
  DumpWriter(const std::string& step_dir, int nstep, double mtime);
  ~DumpWriter();
  template <int Rank> void write(const std::string& name, const FArray<Rank>& a);
private:
  std::string m_dir;
  void* m_index = nullptr;
};

// Max abs and max relative (to max |ref|) differences between two host arrays.
struct DiffStats {
  double max_abs = 0, ref_max = 0, rel = 0;
  std::vector<int> where; // Fortran indices of max abs difference
};
template <int Rank>
DiffStats diff(const FArray<Rank, HostSpace>& a, const FArray<Rank, HostSpace>& ref,
               const std::array<int, Rank>& lo, const std::array<int, Rank>& hi);

} // namespace cm1k

#endif
