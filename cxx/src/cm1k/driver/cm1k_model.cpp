#include "cm1k_model.hpp"

#include "dynamics/cm1k_dyncore.hpp"
#include "init/cm1k_init.hpp"
#include "share/cm1k_dump_io.hpp"

#include "physics/kessler/cm1k_kessler.hpp"

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
  m_sfc_opt = sfc::Options::from_namelist(nl);
  if (m_sfc_opt.active()) {
    m_sfc.allocate(b);
    sfc::set_defaults(m_sfc);   // cm1.F
  }
  // PBL tendency arrays: allocated from the dynamical core's configuration
  // (DynConfig::pbl, param.F use_pbl); zero at start as in cm1.F
  {
    const auto cfg = DynConfig::from_namelist(nl);
    m_use_pbl = cfg.pbl;
    m_ipbl = cfg.pbl_opt.ipbl;
    if (m_use_pbl) m_pbl.allocate(b);
  }
  build_registry();
}

Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

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
  if (m_sfc_opt.active())
    for (auto& [name, a] : m_sfc.dumped()) m_reg.add(name, a);
  if (m_use_pbl && m_ipbl == 2) {
    if (m_sfc_opt.active()) m_reg.add("hpbl", m_sfc.hpbl);
    for (auto& [name, a] : m_pbl.dumped()) m_reg.add(name, a);
  }
}

std::vector<std::string> Model::pbl_field_names() const {
  std::vector<std::string> n;
  if (!(m_use_pbl && m_ipbl == 2)) return n;
  if (m_sfc_opt.active()) n.push_back("hpbl");
  for (auto& [name, a] : m_pbl.dumped()) n.push_back(name);
  return n;
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
  if (m_sfc_opt.active()) sfc::set_dsxy(m_sfc_opt, m_grid, m_sfc);   // not in the dumps
  m_nstep = r.nstep();
  m_mtime = r.mtime();
  m_dyn.reset();           // the dynamical core is set up again from these arrays
  m_dyn_started = false;
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
  if (m_sfc_opt.active()) sfc::init_surface(m_sfc_opt, m_grid, m_sfc);   // init_surface.F
  m_nstep = 0;
  m_mtime = 0;
  m_dyn.reset();            // the dynamical core is set up again from the new grid
  m_dyn_started = false;
}

Dyncore& Model::dyncore() {
  if (!m_dyn) {
    m_dyn = std::make_unique<Dyncore>();
    m_dyn->setup(m_nl, m_c, m_grid, m_base, m_state);
    if (m_sfc_opt.active()) m_dyn->set_surface(m_sfc);
    if (m_use_pbl) m_dyn->set_pbl(m_pbl);
  }
  return *m_dyn;
}

// One time step: the body of cm1.F's time loop (solve1, solve2, microphysics,
// solve3; then mtime = mtime + dt).
//
// Subgrid turbulence (sgsmodel=1): CM1 calls sfc_and_turb at the end of each
// pass of the time loop (cm1.F:3748), after mtime = mtime + dt and after the
// reference dump; the very first pass (dosolve=.false.) only does that on
// the initial state. Here the call is made at the start of every step, on
// the current state at the current mtime: the same sequence (nothing happens
// between the end of one pass and the next solve1), and the model state
// between steps then matches the dumps exactly (dump N holds the eddy
// viscosities computed from state N-1). It also makes a restart from any
// dump (init_from_dump) reproduce the Fortran's next step, and covers the
// startup pass after init() or init_from_dump(step_000000).
void Model::step() {
  auto& dyn = dyncore();
  if (!m_dyn_started) {
    dyn.start_from_state();
    m_dyn_started = true;
  }
  dyn.sfc_and_turb(m_mtime);                     // no-op unless sgsmodel=1 or a PBL scheme
  dyn.diff2_stress();                            // no-op unless idiff=1, difforder=2
  m_nstep = m_nstep + 1;
  dyn.solve1();                                  // pre-RK tendencies
  dyn.solve2();                                  // RK3 loop, acoustic steps
  if (dyn.config().imoist == 1) microphysics();  // mp_driver
  dyn.solve3();                                  // finish: bcs, update "a" arrays
  m_mtime = m_mtime + double(m_dt);
  Kokkos::fence();
}

// cm1.F: `IF(imoist.eq.1) ... call mp_driver(...)` between solve2 and solve3.
// Microphysics acts on the new-time-level arrays (th3d, pp3d, q3d) and on
// rho/prs; th3d/pp3d/q3d live in the dynamical core.
void Model::microphysics() {
  auto& dyn = *m_dyn;
  const int ptype = dyn.config().ptype;
  if (ptype == 0) {
    // cm1r20.3: qv only -> pdefq(0.0, ..., rho, q3d(:,:,:,1))
    kessler::pdefq(m_grid.b, 0.0, m_state.rho, dyn.q3d, 1);
  } else if (ptype == 1) {
    if (!m_mp) {
      auto w = std::make_shared<kessler::Work>();
      w->allocate(m_grid.b);
      m_mp = w;
    }
    static_assert(kessler::nqv == 1 && kessler::nqc == 2, "moisture layout must match the dyncore");
    const auto opt = kessler::Options::from_namelist(m_nl);
    kessler::mp_driver(opt, m_c, m_grid, m_base, m_dt, m_state.rho, m_state.prs, dyn.pp3d, dyn.th3d,
                       dyn.q3d, m_state.rain, *std::static_pointer_cast<kessler::Work>(m_mp));
  } else {
    throw std::runtime_error("cm1k: microphysics ptype=" + std::to_string(ptype) + " is not ported");
  }
}

} // namespace cm1k
