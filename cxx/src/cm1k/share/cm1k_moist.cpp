#include "cm1k_moist.hpp"

#include "cm1k_constants.hpp"
#include "cm1k_grid.hpp"

#include <algorithm>
#include <stdexcept>

namespace cm1k {

MoistLayout MoistLayout::make(int imoist, int ptype, int inum) {
  MoistLayout m;
  m.imoist = imoist;
  m.ptype = ptype;
  if (imoist != 1) return m;
  // param.F defaults
  m.numq = 1; m.nqv = 1;
  auto set = [&](std::vector<std::string> names, std::vector<Real> mags) {
    m.qname = std::move(names);
    m.qmag = std::move(mags);
    m.numq = int(m.qname.size());
  };
  if (ptype == 0) {
    set({"qv"}, {0.01});   // cm1.F: qmag = 0.01
  } else if (ptype == 1) {
    m.nql1 = 2; m.nql2 = 3;
    set({"qv", "qc", "qr"}, {1.0e-2, 1.0e-2, 1.0e-2});
  } else if (ptype == 3) {
    m.iice = 1; m.idm = 1;
    m.nql1 = 2; m.nql2 = 3; m.nqs1 = 4; m.nqs2 = 6; m.nnc1 = 7; m.nnc2 = 8;
    set({"qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr"},
        {1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e6, 1.0e6});
  } else if (ptype == 5) {
    m.iice = 1; m.idm = 1;
    m.nql1 = 2; m.nql2 = 3; m.nqs1 = 4; m.nqs2 = 6; m.nnc1 = 7;
    if (inum == 1) {
      m.nnc2 = 10;
      set({"qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"},
          {1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e6, 1.0e6, 1.0e6, 1.0e6});
    } else if (inum == 0) {
      m.nnc2 = 11;
      set({"qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg", "ncc"},
          {1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5});
    } else {
      throw std::runtime_error("cm1k: unrecognized value for inum (param.F)");
    }
  } else {
    throw std::runtime_error("cm1k: no moisture layout for ptype=" + std::to_string(ptype) +
                             " (ported: 0, 1 Kessler, 3 Thompson, 5 Morrison)");
  }
  // param.F: indices from the names
  for (int n = 1; n <= m.numq; ++n) {
    const auto& q = m.qname[n - 1];
    if (q == "qc" || q == "ql") m.nqc = n;
    if (q == "qr") m.nqr = n;
    if (q == "qi" || q == "qi1") m.nqi = n;
    if (q == "qs") m.nqs = n;
    if (q == "qg") m.nqg = n;
    if (q == "nci") m.nnci = n;
    if (q == "ncc") m.nncc = n;
  }
  return m;
}

MoistLayout MoistLayout::from_namelist(const Namelist& nl) {
  const int imoist = std::min(std::max(nl.geti("imoist", 0), 0), 1);
  return make(imoist, nl.geti("ptype", 0));
}

namespace moist {

void getcvm(const MoistLayout& ml, const Bounds& b, const Constants& c, int eqtset, const FArray3& cvm,
            const FArray4& q) {
  const Real cv = c.cv, cvv = c.cvv, cpl = c.cpl, cpi = c.cpi;
  const bool dry = (eqtset <= 1 || ml.imoist == 0);
  const int nqv = ml.nqv, nql1 = ml.nql1, nql2 = ml.nql2, nqs1 = ml.nqs1, nqs2 = ml.nqs2, iice = ml.iice;
  Kokkos::parallel_for("getcvm", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (dry) {
      cvm(i, j, k) = cv;
    } else {
      Real x = cv + cvv * q(i, j, k, nqv);
      if (nql1 > 0)
        for (int n = nql1; n <= nql2; ++n) x = x + cpl * q(i, j, k, n);
      if (iice == 1)
        for (int n = nqs1; n <= nqs2; ++n) x = x + cpi * q(i, j, k, n);
      cvm(i, j, k) = x;
    }
  });
}

} // namespace moist

} // namespace cm1k
