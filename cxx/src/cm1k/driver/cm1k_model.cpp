#include "cm1k_model.hpp"

#include "dynamics/cm1k_dyncore.hpp"
#include "init/cm1k_init.hpp"
#include "share/cm1k_dump_io.hpp"

#include "physics/kessler/cm1k_kessler.hpp"
#include "physics/rrtmg/cm1k_radiation.hpp"

#include <stdexcept>

namespace cm1k {

namespace {
// param.F extrapolation coefficients, kept on the Grid for the terrain
// branches of the subgrid turbulence
void set_extrap_coeffs(Grid& g) {
  const auto ex = init::extrap_coeffs(g.sigma.to_host(), g.sigmaf.to_host(), g.b.nk);
  g.cgs1 = ex.cgs1; g.cgs2 = ex.cgs2; g.cgs3 = ex.cgs3;
  g.cgt1 = ex.cgt1; g.cgt2 = ex.cgt2; g.cgt3 = ex.cgt3;
}
} // namespace

Model::Model(const Namelist& nl) : m_nl(nl) {
  m_c = Constants::make(nl.geti("testcase", 0));
  const auto b = Bounds::from_namelist(nl);
  if (b.nx <= 0 || b.ny <= 0 || b.nz <= 0)
    throw std::runtime_error("cm1k: nx, ny and nz must be set in the namelist");
  // restarts (irst=1 reads a CM1 restart file) are not ported; Model's
  // init_from_dump starts from a cm1k/cm1dump dump instead
  if (nl.geti("irst", 0) != 0)
    throw std::runtime_error("cm1k: irst=" + std::to_string(nl.geti("irst", 0)) +
                             " (CM1 restart files) is not ported; use init_from_dump");
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
  if (m_sfc_opt.has_layer()) {
    m_sfc.allocate(b);
    sfc::set_defaults(m_sfc);   // cm1.F
  }
  // sfcmodel=2,3: the WRF surface-layer state (tables, tmn, emiss, thc)
  if (m_sfc_opt.wrf()) m_wsfc.setup(m_sfc_opt, sfclay::Options::from_namelist(nl), b, m_sfc);
  // sfcmodel=4: the GFDL surface-layer state (tables, SLAB's tmn, emiss, thc)
  if (m_sfc_opt.gfdl()) m_gsfc.setup(m_sfc_opt, b, m_sfc);
  // PBL tendency arrays: allocated from the dynamical core's configuration
  // (DynConfig::pbl, param.F use_pbl); zero at start as in cm1.F
  {
    const auto cfg = DynConfig::from_namelist(nl);
    m_use_pbl = cfg.pbl;
    m_ipbl = cfg.pbl_opt.ipbl;
    if (m_use_pbl) m_pbl.allocate(b);
    if (m_use_pbl && m_ipbl == 3) m_gfs.allocate(b);   // GFS-EDMF: xkzh, xkzm
    // MYNN: carried arrays (zero at start, cm1.F), aliasing the surface and
    // PBL arrays CM1 shares with it
    m_use_mynn = cfg.mynn && m_sfc_opt.mynn();
    if (m_use_mynn) m_mynn.allocate(b, m_sfc, m_pbl, m_sfc_opt);
    // MYJ: carried arrays (cm1.F initial values: tke_myj = 0.01, the rest 0)
    m_use_myj = cfg.myj && m_sfc_opt.myj();
    if (m_use_myj) m_myj.allocate(b, m_sfc_opt);
  }
  // radiation (radopt=2): gsw, glw, emiss are SLAB's (sfcmodel=2,3)
  {
    const auto ro = rrtmg::RadOptions::from_namelist(nl);
    if (ro.active()) {
      ro.check();
      m_rad = std::make_unique<rrtmg::Radiation>();
      m_rad->setup(ro, m_c, b, m_sfc.tsk, m_sfc.xland, m_wsfc.gsw, m_wsfc.glw, m_wsfc.emiss);
    }
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
  if (g.b.terrain_flag)   // src/cm1dump_state.inc: dumped only with terrain
    for (auto* a : {&g.zs, &g.gz, &g.rgz, &g.gzu, &g.rgzu, &g.gzv, &g.rgzv, &g.dzdx, &g.dzdy,
                    &g.gx, &g.gxu, &g.gy, &g.gyv})
      m_reg.add(a->label(), *a);
  auto& bs = m_base;
  for (auto* a : {&bs.pi0, &bs.th0, &bs.rho0, &bs.prs0, &bs.thv0, &bs.rth0, &bs.qv0, &bs.qc0, &bs.qi0,
                  &bs.rr0, &bs.rf0, &bs.rrf0, &bs.u0, &bs.v0})
    m_reg.add(a->label(), *a);
  auto& s = m_state;
  for (auto* a : {&s.ua, &s.va, &s.wa, &s.ppi, &s.tha, &s.rho, &s.prs, &s.tkea, &s.kmh, &s.kmv,
                  &s.khh, &s.khv, &s.rain})
    m_reg.add(a->label(), *a);
  if (m_sfc_opt.has_layer())
    for (auto& [name, a] : m_sfc.dumped()) m_reg.add(name, a);
  if (m_sfc_opt.wrf())
    for (auto& [name, a] : m_wsfc.dumped()) m_reg.add(name, a);
  if (m_use_pbl && (m_ipbl == 1 || m_ipbl == 2)) {
    if (m_sfc_opt.has_layer()) m_reg.add("hpbl", m_sfc.hpbl);
    for (auto& [name, a] : m_pbl.dumped()) m_reg.add(name, a);
  }
  if (m_sfc_opt.gfdl())
    for (auto& [name, a] : m_gsfc.dumped()) m_reg.add(name, a);
  if (m_use_pbl && m_ipbl == 3) {
    m_reg.add("hpbl", m_sfc.hpbl);
    for (auto& [name, a] : m_pbl.dumped()) m_reg.add(name, a);
    for (auto& [name, a] : m_gfs.dumped()) m_reg.add(name, a);
  }
  if (m_use_mynn) {
    for (auto& [name, a] : m_mynn.dumped_sfc()) m_reg.add(name, a);
    m_reg.add("hpbl", m_sfc.hpbl);
    for (auto& [name, a] : m_pbl.dumped()) m_reg.add(name, a);
    m_reg.add("qipten", m_pbl.qipten);
    for (auto& [name, a] : m_mynn.dumped_pbl()) m_reg.add(name, a);
  }
  if (m_rad)
    for (auto& [name, a] : m_rad->dumped()) m_reg.add(name, a);
  if (m_use_myj) {
    for (auto& [name, a] : m_myj.dumped_sfc()) m_reg.add(name, a);
    m_reg.add("hpbl", m_sfc.hpbl);
    for (auto& [name, a] : m_pbl.dumped()) m_reg.add(name, a);
    m_reg.add("qipten", m_pbl.qipten);
    for (auto& [name, a] : m_myj.dumped_pbl()) m_reg.add(name, a);
  }
}

std::vector<std::string> Model::pbl_field_names() const {
  std::vector<std::string> n;
  if (m_use_pbl && (m_ipbl == 1 || m_ipbl == 2)) {
    if (m_sfc_opt.has_layer()) n.push_back("hpbl");
    for (auto& [name, a] : m_pbl.dumped()) n.push_back(name);
  }
  if (m_use_mynn) {
    // src/cm1dump_state.inc: sfcmodel=6, then ipbl=4/5 blocks
    for (auto& [name, a] : m_mynn.dumped_sfc()) n.push_back(name);
    n.push_back("hpbl");
    for (auto& [name, a] : m_pbl.dumped()) n.push_back(name);
    n.push_back("qipten");
    for (auto& [name, a] : m_mynn.dumped_pbl()) n.push_back(name);
  }
  if (m_use_myj) {
    // src/cm1dump_state.inc: sfcmodel=7, then ipbl=6 blocks
    for (auto& [name, a] : m_myj.dumped_sfc()) n.push_back(name);
    n.push_back("hpbl");
    for (auto& [name, a] : m_pbl.dumped()) n.push_back(name);
    n.push_back("qipten");
    for (auto& [name, a] : m_myj.dumped_pbl()) n.push_back(name);
  }
  // src/cm1dump_state.inc: sfcmodel=4, then ipbl=3 blocks
  if (m_sfc_opt.gfdl())
    for (auto& [name, a] : m_gsfc.dumped()) n.push_back(name);
  if (m_use_pbl && m_ipbl == 3) {
    n.push_back("hpbl");
    for (auto& [name, a] : m_pbl.dumped()) n.push_back(name);
    for (auto& [name, a] : m_gfs.dumped()) n.push_back(name);
  }
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
  if (m_sfc_opt.has_layer()) sfc::set_dsxy(m_sfc_opt, m_grid, m_sfc);   // not in the dumps
  if (m_grid.b.terrain_flag) {   // init_terrain.F: zt = maxz = sigmaf(nk+1) (not in the dumps)
    auto sigmaf = m_grid.sigmaf.to_host();
    m_grid.zt = sigmaf(m_grid.b.nk + 1);
    m_grid.rzt = 1.0 / sigmaf(m_grid.b.nk + 1);
  }
  set_extrap_coeffs(m_grid);
  m_nstep = r.nstep();
  m_mtime = r.mtime();
  if (m_rad) m_rad->set_clock(m_nstep, m_dt);   // radtim of the Fortran at this dump
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
  set_extrap_coeffs(m_grid);
  init::init_base(m_nl, m_c, m_grid, info, m_base);
  init::init_state(m_nl, m_c, m_grid, info, m_base, m_state);
  if (m_sfc_opt.has_layer()) sfc::init_surface(m_sfc_opt, m_grid, m_sfc);   // init_surface.F
  if (m_sfc_opt.wrf()) sfclay::init_soil(m_grid.b, m_wsfc);                 // tslb
  if (m_rad) m_rad->init_state();                                           // o30, radtim
  if (m_use_mynn) m_mynn.init_soil(m_grid.b, m_sfc);                       // tslb (sfcmodel=6)
  if (m_use_myj) myj::init(m_grid.b, m_sfc, m_myj);   // MYJSFCINIT ust, z0base, tslb
  if (m_sfc_opt.gfdl()) m_gsfc.init_soil(m_grid.b);                          // tslb (sfcmodel=4)
  m_nstep = 0;
  m_mtime = 0;
  m_dyn.reset();            // the dynamical core is set up again from the new grid
  m_dyn_started = false;
}

Dyncore& Model::dyncore() {
  if (!m_dyn) {
    m_dyn = std::make_unique<Dyncore>();
    m_dyn->setup(m_nl, m_c, m_grid, m_base, m_state);
    if (m_sfc_opt.has_layer()) m_dyn->set_surface(m_sfc);
    if (m_sfc_opt.wrf()) m_dyn->set_wrf_surface(m_wsfc);
    if (m_use_pbl) m_dyn->set_pbl(m_pbl);
    if (m_use_mynn) m_dyn->set_mynn(m_mynn);
    if (m_rad) m_dyn->set_radiation(m_rad->thraten);
    if (m_use_myj) m_dyn->set_myj(m_myj);
    if (m_sfc_opt.gfdl()) m_dyn->set_gfdl_surface(m_gsfc);
    if (m_use_pbl && m_ipbl == 3) m_dyn->set_gfs(m_gfs);
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
  // radiation (radopt=2): cm1.F calls radiation_driver at the end of a pass
  // when due (always in the startup pass), after the dump and before
  // sfc_and_turb; with the same argument as for sfc_and_turb this is here
  if (m_rad) m_rad->pass(m_mtime, m_nstep, m_dt, m_grid, m_base, m_state);
  // nstep: cm1.F's update_sfc is .false. in the startup pass (nstep = 0)
  dyn.sfc_and_turb(m_mtime, m_nstep);            // no-op unless sgsmodel=1 or a PBL scheme
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
