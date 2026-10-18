#include "physics/rrtmg/cm1k_rrtmg_util.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace cm1k::rrtmg {

namespace {
uint32_t be32(const unsigned char* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}
} // namespace

FortranRecordReader::FortranRecordReader(const std::string& path) : m_path(path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cm1k rrtmg: cannot open " + path);
  m_data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void FortranRecordReader::next() {
  if (m_pos + 4 > m_data.size()) throw std::runtime_error("cm1k rrtmg: unexpected end of " + m_path);
  const uint32_t n = be32(&m_data[m_pos]);
  if (m_pos + 8 + n > m_data.size() || be32(&m_data[m_pos + 4 + n]) != n)
    throw std::runtime_error("cm1k rrtmg: bad record marker in " + m_path);
  m_rec.assign(m_data.begin() + long(m_pos + 4), m_data.begin() + long(m_pos + 4 + n));
  m_pos += 8 + n;
  m_rpos = 0;
}

void FortranRecordReader::reals(Real* dst, size_t n) {
  if (m_rpos + 4 * n > m_rec.size()) throw std::runtime_error("cm1k rrtmg: short record in " + m_path);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t u = be32(&m_rec[m_rpos + 4 * i]);
    float f;
    std::memcpy(&f, &u, 4);
    dst[i] = Real(f);
  }
  m_rpos += 4 * n;
}

void FortranRecordReader::ints(int* dst, size_t n) {
  if (m_rpos + 4 * n > m_rec.size()) throw std::runtime_error("cm1k rrtmg: short record in " + m_path);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t u = be32(&m_rec[m_rpos + 4 * i]);
    int32_t v;
    std::memcpy(&v, &u, 4);
    dst[i] = int(v);
  }
  m_rpos += 4 * n;
}

void FortranRecordReader::end_record() {
  if (m_rpos != m_rec.size()) throw std::runtime_error("cm1k rrtmg: record length mismatch in " + m_path);
}

void fset_impl(void* p, bool is_int, const std::vector<int>& lb, const std::vector<int>& ub,
               std::initializer_list<S> sec, std::initializer_list<double> vals) {
  const size_t nd = lb.size();
  if (sec.size() != nd) throw std::runtime_error("cm1k rrtmg fset: rank mismatch");
  std::vector<int> lo(nd), hi(nd), str(nd);
  size_t cnt = 1, s = 1;
  auto it = sec.begin();
  for (size_t d = 0; d < nd; ++d, ++it) {
    lo[d] = it->kind == 0 ? lb[d] : it->a;
    hi[d] = it->kind == 0 ? ub[d] : it->b;
    if (lo[d] < lb[d] || hi[d] > ub[d] || hi[d] < lo[d]) throw std::runtime_error("cm1k rrtmg fset: bad section");
    str[d] = int(s);
    s *= size_t(ub[d] - lb[d] + 1);
    cnt *= size_t(hi[d] - lo[d] + 1);
  }
  if (cnt != vals.size()) throw std::runtime_error("cm1k rrtmg fset: section/value count mismatch");
  std::vector<int> ix(lo);
  for (double v : vals) {
    size_t off = 0;
    for (size_t d = 0; d < nd; ++d) off += size_t(ix[d] - lb[d]) * size_t(str[d]);
    if (is_int) static_cast<int*>(p)[off] = int(v);
    else static_cast<Real*>(p)[off] = Real(v);
    for (size_t d = 0; d < nd; ++d) {
      if (++ix[d] <= hi[d]) break;
      ix[d] = lo[d];
    }
  }
}

const TabBinder::Reg& TabBinder::find(const std::string& name) const {
  for (auto& r : regs)
    if (r.name == name) return r;
  throw std::runtime_error("cm1k rrtmg: no table array " + name);
}
HView<Real> TabBinder::rview(const std::string& name) const {
  auto& r = find(name);
  if (r.is_int || !rbase) throw std::runtime_error("cm1k rrtmg: " + name + " is not a bound real table");
  return HView<Real>(rbase + r.off, r.lb, r.ub);
}
HView<int> TabBinder::iview(const std::string& name) const {
  auto& r = find(name);
  if (!r.is_int || !ibase) throw std::runtime_error("cm1k rrtmg: " + name + " is not a bound int table");
  return HView<int>(ibase + r.off, r.lb, r.ub);
}

std::string find_data_file(const std::string& name, const std::string& dir) {
  namespace fs = std::filesystem;
  std::vector<fs::path> cand;
  if (!dir.empty()) cand.push_back(fs::path(dir) / name);
  else {
    if (const char* e = std::getenv("CM1K_RRTMG_DATA_DIR")) cand.push_back(fs::path(e) / name);
    cand.push_back(fs::path(name));
#ifdef CM1K_SOURCE_RUN_DIR
    cand.push_back(fs::path(CM1K_SOURCE_RUN_DIR) / name);
#endif
  }
  for (auto& p : cand)
    if (fs::exists(p)) return p.string();
  throw std::runtime_error("cm1k rrtmg: " + name + " not found (looked in " + cand.front().string() +
                           (cand.size() > 1 ? " and others" : "") + "); set CM1K_RRTMG_DATA_DIR");
}

} // namespace cm1k::rrtmg
