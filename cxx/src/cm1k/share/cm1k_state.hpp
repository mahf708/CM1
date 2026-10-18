#ifndef CM1K_STATE_HPP
#define CM1K_STATE_HPP

#include "cm1k_grid.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cm1k {

// Base (reference) state, as in base.F / cm1.F.
struct BaseState {
  FArray3 pi0, th0, rho0, prs0, thv0, rth0, qv0, qc0, qi0, rr0, rf0, rrf0;  // (ib:ie,jb:je,kb:ke)
  FArray3 u0;   // (ib:ie+1,jb:je,kb:ke)
  FArray3 v0;   // (ib:ie,jb:je+1,kb:ke)
  void allocate(const Bounds& b);
};

// Prognostic and diagnostic model state (cm1.F names).
struct State {
  FArray3 ua, va, wa, ppi, tha;   // u (ib:ie+1,..), v (..,jb:je+1,..), w (..,kb:ke+1), pi', th'
  FArray3 rho, prs;               // full density and pressure
  FArray4 qa;                     // (ibm:iem,jbm:jem,kbm:kem,numq)
  FArray3 tkea;                   // (ibt:iet,jbt:jet,kbt:ket)
  FArray3 kmh, kmv, khh, khv;     // (ibc:iec,jbc:jec,kbc:kec)
  FArray3 rain;                   // (ib:ie,jb:je,nrain)
  std::vector<std::string> qname; // species names (qa index n -> qname[n-1])
  void allocate(const Bounds& b);
};

// Named access to the 1D/3D arrays of a model (grid, base state, state),
// using the same names as the Fortran reference dumps. The 4D moisture array
// qa is handled by the Model, which maps "qa_<qname>" to qa(:,:,:,n).
struct FieldRegistry {
  std::map<std::string, FArray1> f1;
  std::map<std::string, FArray3> f3;
  void add(const std::string& n, const FArray1& a) { f1[n] = a; }
  void add(const std::string& n, const FArray3& a) { f3[n] = a; }
};

} // namespace cm1k

#endif
