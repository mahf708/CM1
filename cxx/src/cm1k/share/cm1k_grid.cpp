#include "cm1k_grid.hpp"

#include <algorithm>

namespace cm1k {

Bounds Bounds::from_namelist(const Namelist& nl) {
  Bounds b;
  b.nx = nl.geti("nx", 0);
  b.ny = nl.geti("ny", 0);
  b.nz = nl.geti("nz", 0);
  b.ni = b.nx; b.nj = b.ny; b.nk = b.nz; b.nkp1 = b.nk + 1;

  const int hs = nl.geti("hadvordrs", 5), hv = nl.geti("hadvordrv", 5);
  auto any = [&](int a, int c) { return hs == a || hv == a || hs == c || hv == c; };
  b.ngxy = 2;
  if (any(3, 4)) b.ngxy = 2;
  if (any(5, 6)) b.ngxy = 3;
  if (any(7, 8)) b.ngxy = 4;
  if (any(9, 10)) b.ngxy = 5;
  if (nl.geti("advwenos", 0) >= 1 || nl.geti("advwenov", 0) >= 1) {
    const int wo = nl.geti("weno_order", 5);
    if (wo == 3) b.ngxy = std::max(b.ngxy, 2);
    if (wo == 5) b.ngxy = std::max(b.ngxy, 3);
    if (wo == 7) b.ngxy = std::max(b.ngxy, 4);
    if (wo == 9) b.ngxy = std::max(b.ngxy, 5);
  }
  b.ngz = 1;
  b.ib = 1 - b.ngxy; b.ie = b.ni + b.ngxy;
  b.jb = 1 - b.ngxy; b.je = b.nj + b.ngxy;
  b.kb = 1 - b.ngz;  b.ke = b.nk + b.ngz;

  const int imoist = nl.geti("imoist", 0);
  if (imoist == 1) {
    b.ibm = b.ib; b.iem = b.ie; b.jbm = b.jb; b.jem = b.je; b.kbm = b.kb; b.kem = b.ke;
  }

  const int cm1setup = nl.geti("cm1setup", 1);
  const int sgsmodel = nl.geti("sgsmodel", 0);
  const int ipbl = nl.geti("ipbl", 0);
  const bool idoles = (cm1setup == 1 || cm1setup == 4);
  const bool idopbl = (cm1setup == 2 || cm1setup == 4);
  if (idoles || idopbl || ipbl >= 1 || cm1setup == 3) {
    b.ibc = b.ib; b.iec = b.ie; b.jbc = b.jb; b.jec = b.je; b.kbc = b.kb; b.kec = b.ke + 1;
  }
  if (idoles && (sgsmodel == 1 || sgsmodel == 3 || sgsmodel == 4 || sgsmodel == 5)) {
    b.ibt = b.ib; b.iet = b.ie; b.jbt = b.jb; b.jet = b.je; b.kbt = b.kb; b.ket = b.ke + 1;
  }

  // numq for the microphysics options cm1k supports (see param.F).
  const int ptype = nl.geti("ptype", 0);
  b.numq = 1;
  if (imoist == 1 && ptype == 1) b.numq = 3;   // Kessler: qv, qc, qr
  b.nrain = 1;
  return b;
}

void Grid::allocate(const Bounds& bb) {
  b = bb;
  const auto& B = b;
  xh  = FArray1("xh",  {B.ib}, {B.ie});
  rxh = FArray1("rxh", {B.ib}, {B.ie});
  uh  = FArray1("uh",  {B.ib}, {B.ie});
  ruh = FArray1("ruh", {B.ib}, {B.ie});
  xf  = FArray1("xf",  {B.ib}, {B.ie + 1});
  rxf = FArray1("rxf", {B.ib}, {B.ie + 1});
  uf  = FArray1("uf",  {B.ib}, {B.ie + 1});
  ruf = FArray1("ruf", {B.ib}, {B.ie + 1});
  yh  = FArray1("yh",  {B.jb}, {B.je});
  vh  = FArray1("vh",  {B.jb}, {B.je});
  rvh = FArray1("rvh", {B.jb}, {B.je});
  yf  = FArray1("yf",  {B.jb}, {B.je + 1});
  vf  = FArray1("vf",  {B.jb}, {B.je + 1});
  rvf = FArray1("rvf", {B.jb}, {B.je + 1});
  sigma  = FArray1("sigma",  {B.kb}, {B.ke});
  rds    = FArray1("rds",    {B.kb}, {B.ke});
  sigmaf = FArray1("sigmaf", {B.kb}, {B.ke + 1});
  rdsf   = FArray1("rdsf",   {B.kb}, {B.ke + 1});
  zh  = FArray3("zh",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  mh  = FArray3("mh",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  rmh = FArray3("rmh", {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  c1  = FArray3("c1",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  c2  = FArray3("c2",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke});
  zf  = FArray3("zf",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke + 1});
  mf  = FArray3("mf",  {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke + 1});
  rmf = FArray3("rmf", {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke + 1});
}

} // namespace cm1k
