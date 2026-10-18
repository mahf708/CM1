#include "cm1k_state.hpp"

namespace cm1k {

void BaseState::allocate(const Bounds& B) {
  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  pi0 = s3("pi0"); th0 = s3("th0"); rho0 = s3("rho0"); prs0 = s3("prs0");
  thv0 = s3("thv0"); rth0 = s3("rth0"); qv0 = s3("qv0"); qc0 = s3("qc0");
  qi0 = s3("qi0"); rr0 = s3("rr0"); rf0 = s3("rf0"); rrf0 = s3("rrf0");
  u0 = FArray3("u0", {B.ib, B.jb, B.kb}, {B.ie + 1, B.je, B.ke});
  v0 = FArray3("v0", {B.ib, B.jb, B.kb}, {B.ie, B.je + 1, B.ke});
}

void State::allocate(const Bounds& B) {
  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  ua  = FArray3("ua", {B.ib, B.jb, B.kb}, {B.ie + 1, B.je, B.ke});
  va  = FArray3("va", {B.ib, B.jb, B.kb}, {B.ie, B.je + 1, B.ke});
  wa  = FArray3("wa", {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke + 1});
  ppi = s3("ppi"); tha = s3("tha"); rho = s3("rho"); prs = s3("prs");
  qa  = FArray4("qa", {B.ibm, B.jbm, B.kbm, 1}, {B.iem, B.jem, B.kem, B.numq});
  tkea = FArray3("tkea", {B.ibt, B.jbt, B.kbt}, {B.iet, B.jet, B.ket});
  kmh = FArray3("kmh", {B.ibc, B.jbc, B.kbc}, {B.iec, B.jec, B.kec});
  kmv = FArray3("kmv", {B.ibc, B.jbc, B.kbc}, {B.iec, B.jec, B.kec});
  khh = FArray3("khh", {B.ibc, B.jbc, B.kbc}, {B.iec, B.jec, B.kec});
  khv = FArray3("khv", {B.ibc, B.jbc, B.kbc}, {B.iec, B.jec, B.kec});
  rain = FArray3("rain", {B.ib, B.jb, 1}, {B.ie, B.je, B.nrain});
}

} // namespace cm1k
