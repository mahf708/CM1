#ifndef CM1K_MODEL_HPP
#define CM1K_MODEL_HPP

#include "physics/pbl/cm1k_pbl.hpp"
#include "physics/sfc/cm1k_sfc.hpp"
#include "share/cm1k_constants.hpp"
#include "share/cm1k_grid.hpp"
#include "share/cm1k_namelist.hpp"
#include "share/cm1k_state.hpp"

#include <memory>

#include <string>
#include <vector>

namespace cm1k {

// The cm1k model: one CM1 domain on one device (no MPI yet).
//
// Typical use (C++ or Python):
//   Model m(Namelist::from_file("namelist.input"));
//   m.init_from_dump(DumpReader::step_dir(ref, 0));   // or m.init() (native)
//   for (int n = 0; n < 10; ++n) m.step();
//   m.write_dump(out_dir);
class Dyncore;

class Model {
public:
  explicit Model(const Namelist& nl);
  ~Model();
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;

  // Native initialization: grid (param.F), base state (base.F), initial
  // perturbations (init3d.F).
  void init();

  // Load grid, base state and prognostic state from a Fortran reference dump
  // directory (step_NNNNNN). Lets the dynamics be tested independently of
  // the initialization code.
  void init_from_dump(const std::string& step_dir);

  // Advance one large time step (solve1, solve2, microphysics, solve3).
  void step();

  // Write every registered array (Fortran dump format) to root/step_NNNNNN.
  void write_dump(const std::string& root) const;

  // Named field access, using the reference-dump names.
  bool has_field(const std::string& name) const;
  int field_rank(const std::string& name) const;
  FArray1 field1(const std::string& name) const;
  FArray3 field3(const std::string& name) const;   // includes "qa_<qname>" (copy-free subview)
  std::vector<std::string> field_names() const;

  const Namelist& namelist() const { return m_nl; }
  const Constants& constants() const { return m_c; }
  const Bounds& bounds() const { return m_grid.b; }
  Grid& grid() { return m_grid; }
  BaseState& base() { return m_base; }
  State& state() { return m_state; }
  // Surface layer (allocated only for sfcmodel=1; its dumped arrays are
  // registered fields then).
  bool has_surface() const { return m_sfc_opt.active(); }
  sfc::SfcState& surface() { return m_sfc; }
  // PBL tendencies (allocated only for use_pbl, i.e. cm1setup=2 with
  // ipbl>=1; for ipbl=2 they and the surface layer's hpbl are registered
  // fields, as src/cm1dump_state.inc dumps them).
  bool has_pbl() const { return m_use_pbl; }
  pbl::Tendencies& pbl_tendencies() { return m_pbl; }
  // Names of the PBL fields in the dumps (empty without ipbl=2).
  std::vector<std::string> pbl_field_names() const;
  // The dynamical core (set up on the first step()).
  Dyncore& dyncore();

  int nstep() const { return m_nstep; }
  double mtime() const { return m_mtime; }
  Real dt() const { return m_dt; }

private:
  void build_registry();
  void microphysics();   // cm1.F: mp_driver call between solve2 and solve3

  Namelist m_nl;
  Constants m_c;
  Grid m_grid;
  BaseState m_base;
  State m_state;
  sfc::Options m_sfc_opt;
  sfc::SfcState m_sfc;
  bool m_use_pbl = false;
  int m_ipbl = 0;
  pbl::Tendencies m_pbl;
  FieldRegistry m_reg;

  int m_nstep = 0;
  double m_mtime = 0;
  Real m_dt = 0;

  std::unique_ptr<Dyncore> m_dyn;   // created/started lazily in step()
  bool m_dyn_started = false;
  std::shared_ptr<void> m_mp;       // microphysics work arrays
};

} // namespace cm1k

#endif
