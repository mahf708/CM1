#include "cm1k_model.hpp"

#include "init/cm1k_init.hpp"
#include "share/cm1k_dump_io.hpp"

#include <stdexcept>

namespace cm1k {

Model::Model(const Namelist& nl) : m_nl(nl) {
  m_c = Constants::make(nl.geti("testcase", 0));
  const auto b = Bounds::from_namelist(nl);
  if (b.nx <= 0 || b.ny <= 0 || b.nz <= 0)
    throw std::runtime_error("cm1k: nx, ny and nz must be set in the namelist");
  m_grid.allocate(b);
  // grid spacings (param.F)
  m_grid.dx = nl.getr("dx", 0.0);
  m_grid.dy = nl.getr("dy", 0.0);
  m_grid.dz = nl.getr("dz", 0.0);
  m_grid.rdx = 1.0 / m_grid.dx;
  m_grid.rdy = 1.0 / m_grid.dy;
  m_grid.rdz = 1.0 / m_grid.dz;
  m_base.allocate(b);
  m_state.allocate(b);
  if (nl.geti("imoist", 0) == 1) {
    const int ptype = nl.geti("ptype", 0);
    if (ptype == 0) m_state.qname = {"qv"};
    else if (ptype == 1) m_state.qname = {"qv", "qc", "qr"};
  }
  m_dt = nl.getr("dtl", 0.0);
  build_registry();
}

void Model::build_registry() {
  auto& g = m_grid;
  for (auto* a : {&g.xh, &g.rxh, &g.uh, &g.ruh, &g.xf, &g.rxf, &g.uf, &g.ruf, &g.yh, &g.vh, &g.rvh,
                  &g.yf, &g.vf, &g.rvf, &g.sigma, &g.rds, &g.sigmaf, &g.rdsf})
    m_reg.add(a->label(), *a);
  for (auto* a : {&g.zh, &g.mh, &g.rmh, &g.c1, &g.c2, &g.zf, &g.mf, &g.rmf})
    m_reg.add(a->label(), *a);
  auto& bs = m_base;
  for (auto* a : {&bs.pi0, &bs.th0, &bs.rho0, &bs.prs0, &bs.thv0, &bs.rth0, &bs.qv0, &bs.qc0, &bs.qi0,
                  &bs.rr0, &bs.rf0, &bs.rrf0, &bs.u0, &bs.v0})
    m_reg.add(a->label(), *a);
  auto& s = m_state;
  for (auto* a : {&s.ua, &s.va, &s.wa, &s.ppi, &s.tha, &s.rho, &s.prs, &s.tkea, &s.kmh, &s.kmv,
                  &s.khh, &s.khv, &s.rain})
    m_reg.add(a->label(), *a);
}

bool Model::has_field(const std::string& name) const {
  if (m_reg.f1.count(name) || m_reg.f3.count(name)) return true;
  for (auto& q : m_state.qname) if (name == "qa_" + q) return true;
  return false;
}

int Model::field_rank(const std::string& name) const {
  if (m_reg.f1.count(name)) return 1;
  if (has_field(name)) return 3;
  throw std::runtime_error("cm1k: unknown field '" + name + "'");
}

FArray1 Model::field1(const std::string& name) const {
  auto it = m_reg.f1.find(name);
  if (it == m_reg.f1.end()) throw std::runtime_error("cm1k: unknown 1D field '" + name + "'");
  return it->second;
}

FArray3 Model::field3(const std::string& name) const {
  auto it = m_reg.f3.find(name);
  if (it != m_reg.f3.end()) return it->second;
  const auto& b = m_grid.b;
  for (size_t n = 0; n < m_state.qname.size(); ++n) {
    if (name == "qa_" + m_state.qname[n]) {
      auto v = Kokkos::subview(m_state.qa.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, int(n));
      return FArray3(v, {b.ibm, b.jbm, b.kbm});
    }
  }
  throw std::runtime_error("cm1k: unknown 3D field '" + name + "'");
}

std::vector<std::string> Model::field_names() const {
  std::vector<std::string> n;
  for (auto& [k, v] : m_reg.f1) n.push_back(k);
  for (auto& [k, v] : m_reg.f3) n.push_back(k);
  for (auto& q : m_state.qname) n.push_back("qa_" + q);
  return n;
}

void Model::init_from_dump(const std::string& step_dir) {
  DumpReader r(step_dir);
  for (auto& name : field_names()) {
    if (!r.has(name)) continue;   // e.g. tkea when sgsmodel has no TKE
    if (field_rank(name) == 1) r.read_into<1>(name, field1(name));
    else                       r.read_into<3>(name, field3(name));
  }
  m_nstep = r.nstep();
  m_mtime = r.mtime();
}

void Model::write_dump(const std::string& root) const {
  DumpWriter w(DumpReader::step_dir(root, m_nstep), m_nstep, m_mtime);
  for (auto& name : field_names()) {
    if (field_rank(name) == 1) w.write<1>(name, field1(name));
    else                       w.write<3>(name, field3(name));
  }
}

void Model::init() {
  // param.F (grid), base.F (base state), init3d.F + getset (initial state);
  // see init/cm1k_init.hpp for the supported options.
  auto info = init::init_grid(m_nl, m_grid);
  init::init_base(m_nl, m_c, m_grid, info, m_base);
  init::init_state(m_nl, m_c, m_grid, info, m_base, m_state);
  m_nstep = 0;
  m_mtime = 0;
}

void Model::step() {
  throw std::runtime_error("cm1k: Model::step is not implemented yet");
}

} // namespace cm1k
