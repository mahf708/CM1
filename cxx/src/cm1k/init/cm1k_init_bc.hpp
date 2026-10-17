#ifndef CM1K_INIT_BC_HPP
#define CM1K_INIT_BC_HPP

#include "share/cm1k_farray.hpp"
#include "share/cm1k_grid.hpp"

#include <stdexcept>

namespace cm1k {
namespace init {

using HArray1 = FArray<1, HostSpace>;
using HArray3 = FArray<3, HostSpace>;

// Lateral boundary settings and extrapolation coefficients needed by the
// host-side boundary routines below (serial run: myi1 = myj1 = 1).
struct BcInfo {
  int wbc = 1, ebc = 1, sbc = 1, nbc = 1;
  int ibw = 0, ibe = 0, ibs = 0, ibn = 0;
  int ni = 0, nj = 0, nk = 0, ngxy = 0;
  int ib = 0, ie = 0, jb = 0, je = 0;
  Real cgs1 = 0, cgs2 = 0, cgs3 = 0, cgt1 = 0, cgt2 = 0, cgt3 = 0;
};

// Host ports of the non-MPI branches of bc.F (bcs, bcu, bcv, bcw, bc2d,
// extrapbcs). Plain serial loops in the Fortran order: they only copy values,
// so order matters only where a later loop reads halo points set earlier
// (e.g. south/north loops over i=ib,ie read the west/east halos).

// bcs: scalar (ib:ie,jb:je,kb:ke)
inline void bcs(const BcInfo& c, const HArray3& s) {
  const int ni = c.ni, nj = c.nj, nk = c.nk, ng = c.ngxy;
  // west
  if (c.wbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      s(1 - i, j, k) = s(ni + 1 - i, j, k);
  } else if (c.wbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1 - ng; i <= 0; ++i)
      s(i, j, k) = s(1, j, k);
  } else if (c.wbc == 3 || c.wbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1; i <= ng; ++i)
      s(1 - i, j, k) = s(i, j, k);
  }
  // east
  if (c.ebc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      s(ni + i, j, k) = s(i, j, k);
  } else if (c.ebc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = ni + 1; i <= ni + ng; ++i)
      s(i, j, k) = s(ni, j, k);
  } else if (c.ebc == 3 || c.ebc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1; i <= ng; ++i)
      s(ni + i, j, k) = s(ni + 1 - i, j, k);
  }
  // south
  if (c.sbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      s(i, 1 - j, k) = s(i, nj + 1 - j, k);
  } else if (c.sbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 1 - ng; j <= 0; ++j) for (int i = 0; i <= ni + 1; ++i)
      s(i, j, k) = s(i, 1, k);
  } else if (c.sbc == 3 || c.sbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      s(i, 1 - j, k) = s(i, j, k);
  }
  // north
  if (c.nbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      s(i, nj + j, k) = s(i, j, k);
  } else if (c.nbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = nj + 1; j <= nj + ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      s(i, j, k) = s(i, nj, k);
  } else if (c.nbc == 3 || c.nbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      s(i, nj + j, k) = s(i, nj + 1 - j, k);
  }
}

// bcu: u-points (ib:ie+1,jb:je,kb:ke)
inline void bcu(const BcInfo& c, const HArray3& u) {
  const int ni = c.ni, nj = c.nj, nk = c.nk, ng = c.ngxy;
  if (c.wbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      u(1 - i, j, k) = u(ni + 1 - i, j, k);
  } else if (c.wbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1 - ng; i <= 0; ++i)
      u(i, j, k) = u(1, j, k);
  } else if (c.wbc == 3 || c.wbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) {
      u(1, j, k) = 0.;
      for (int i = 1; i <= ng; ++i) u(1 - i, j, k) = -u(1 + i, j, k);
    }
  }
  if (c.ebc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      u(ni + 1 + i, j, k) = u(1 + i, j, k);
  } else if (c.ebc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = ni + 2; i <= ni + 1 + ng; ++i)
      u(i, j, k) = u(ni + 1, j, k);
  } else if (c.ebc == 3 || c.ebc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) {
      u(ni + 1, j, k) = 0.0;
      for (int i = 1; i <= ng; ++i) u(ni + 1 + i, j, k) = -u(ni + 1 - i, j, k);
    }
  }
  if (c.sbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie + 1; ++i)
      u(i, 1 - j, k) = u(i, nj + 1 - j, k);
  } else if (c.sbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 1 - ng; j <= 0; ++j) for (int i = 0; i <= ni + 2; ++i)
      u(i, j, k) = u(i, 1, k);
  } else if (c.sbc == 3 || c.sbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 2; ++i)
      u(i, 1 - j, k) = u(i, j, k);
  }
  if (c.nbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie + 1; ++i)
      u(i, nj + j, k) = u(i, j, k);
  } else if (c.nbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = nj + 1; j <= nj + ng; ++j) for (int i = 0; i <= ni + 2; ++i)
      u(i, j, k) = u(i, nj, k);
  } else if (c.nbc == 3 || c.nbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 2; ++i)
      u(i, nj + j, k) = u(i, nj + 1 - j, k);
  }
}

// bcv: v-points (ib:ie,jb:je+1,kb:ke); south/north first, as in bc.F
inline void bcv(const BcInfo& c, const HArray3& v) {
  const int ni = c.ni, nj = c.nj, nk = c.nk, ng = c.ngxy;
  if (c.sbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      v(i, 1 - j, k) = v(i, nj + 1 - j, k);
  } else if (c.sbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 1 - ng; j <= 0; ++j) for (int i = 0; i <= ni + 1; ++i)
      v(i, j, k) = v(i, 1, k);
  } else if (c.sbc == 3 || c.sbc == 4) {
    for (int k = 1; k <= nk; ++k) {
      for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i) v(i, 1 - j, k) = -v(i, j + 1, k);
      for (int i = 0; i <= ni + 1; ++i) v(i, 1, k) = 0.;
    }
  }
  if (c.nbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      v(i, nj + 1 + j, k) = v(i, 1 + j, k);
  } else if (c.nbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = nj + 2; j <= nj + 1 + ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      v(i, j, k) = v(i, nj + 1, k);
  } else if (c.nbc == 3 || c.nbc == 4) {
    for (int k = 1; k <= nk; ++k) {
      for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i) v(i, nj + 1 + j, k) = -v(i, nj + 1 - j, k);
      for (int i = 0; i <= ni + 1; ++i) v(i, nj + 1, k) = 0.;
    }
  }
  if (c.wbc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je + 1; ++j) for (int i = 1; i <= ng; ++i)
      v(1 - i, j, k) = v(ni + 1 - i, j, k);
  } else if (c.wbc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 2; ++j) for (int i = 1 - ng; i <= 0; ++i)
      v(i, j, k) = v(1, j, k);
  } else if (c.wbc == 3 || c.wbc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 2; ++j) for (int i = 1; i <= ng; ++i)
      v(1 - i, j, k) = v(i, j, k);
  }
  if (c.ebc == 1) {
    for (int k = 1; k <= nk; ++k) for (int j = c.jb; j <= c.je + 1; ++j) for (int i = 1; i <= ng; ++i)
      v(ni + i, j, k) = v(i, j, k);
  } else if (c.ebc == 2) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 2; ++j) for (int i = ni + 1; i <= ni + ng; ++i)
      v(i, j, k) = v(ni, j, k);
  } else if (c.ebc == 3 || c.ebc == 4) {
    for (int k = 1; k <= nk; ++k) for (int j = 0; j <= nj + 2; ++j) for (int i = 1; i <= ng; ++i)
      v(ni + i, j, k) = v(ni + 1 - i, j, k);
  }
}

// bcw: w-points (ib:ie,jb:je,kb:ke+1); flag=1 also zeroes w at k=1 and nk+1
inline void bcw(const BcInfo& c, const HArray3& w, int flag) {
  const int ni = c.ni, nj = c.nj, nk = c.nk, ng = c.ngxy;
  if (c.wbc == 1) {
    for (int k = 2; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      w(1 - i, j, k) = w(ni + 1 - i, j, k);
  } else if (c.wbc == 2) {
    for (int k = 2; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1 - ng; i <= 0; ++i)
      w(i, j, k) = w(1, j, k);
  } else if (c.wbc == 3 || c.wbc == 4) {
    for (int k = 2; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1; i <= ng; ++i)
      w(1 - i, j, k) = w(i, j, k);
  }
  if (c.ebc == 1) {
    for (int k = 2; k <= nk; ++k) for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i)
      w(ni + i, j, k) = w(i, j, k);
  } else if (c.ebc == 2) {
    for (int k = 2; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = ni + 1; i <= ni + ng; ++i)
      w(i, j, k) = w(ni, j, k);
  } else if (c.ebc == 3 || c.ebc == 4) {
    for (int k = 2; k <= nk; ++k) for (int j = 0; j <= nj + 1; ++j) for (int i = 1; i <= ng; ++i)
      w(ni + i, j, k) = w(ni + 1 - i, j, k);
  }
  if (c.sbc == 1) {
    for (int k = 2; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      w(i, 1 - j, k) = w(i, nj + 1 - j, k);
  } else if (c.sbc == 2) {
    for (int k = 2; k <= nk; ++k) for (int j = 1 - ng; j <= 0; ++j) for (int i = 0; i <= ni + 1; ++i)
      w(i, j, k) = w(i, 1, k);
  } else if (c.sbc == 3 || c.sbc == 4) {
    for (int k = 2; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      w(i, 1 - j, k) = w(i, j, k);
  }
  if (c.nbc == 1) {
    for (int k = 2; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i)
      w(i, nj + j, k) = w(i, j, k);
  } else if (c.nbc == 2) {
    for (int k = 2; k <= nk; ++k) for (int j = nj + 1; j <= nj + ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      w(i, j, k) = w(i, nj, k);
  } else if (c.nbc == 3 || c.nbc == 4) {
    for (int k = 2; k <= nk; ++k) for (int j = 1; j <= ng; ++j) for (int i = 0; i <= ni + 1; ++i)
      w(i, nj + j, k) = w(i, nj + 1 - j, k);
  }
  if (flag == 1) {
    for (int j = 0; j <= nj + 1; ++j) for (int i = 0; i <= ni + 1; ++i) {
      w(i, j, 1) = 0.0;
      w(i, j, nk + 1) = 0.0;
    }
  }
}

// bc2d applied to level k of a 3D array (Fortran passes s(ib,jb,k) as 2D)
inline void bc2d(const BcInfo& c, const HArray3& s, int k) {
  const int ni = c.ni, nj = c.nj, ng = c.ngxy;
  if (c.wbc == 1) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i) s(1 - i, j, k) = s(ni + 1 - i, j, k);
  } else if (c.wbc == 2) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = 1 - ng; i <= 0; ++i) s(i, j, k) = s(1, j, k);
  } else if (c.wbc == 3 || c.wbc == 4) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i) s(1 - i, j, k) = s(i, j, k);
  }
  if (c.ebc == 1) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i) s(ni + i, j, k) = s(i, j, k);
  } else if (c.ebc == 2) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = ni + 1; i <= ni + ng; ++i) s(i, j, k) = s(ni, j, k);
  } else if (c.ebc == 3 || c.ebc == 4) {
    for (int j = c.jb; j <= c.je; ++j) for (int i = 1; i <= ng; ++i) s(ni + i, j, k) = s(ni + 1 - i, j, k);
  }
  if (c.sbc == 1) {
    for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, 1 - j, k) = s(i, nj + 1 - j, k);
  } else if (c.sbc == 2) {
    for (int j = 1 - ng; j <= 0; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, j, k) = s(i, 1, k);
  } else if (c.sbc == 3 || c.sbc == 4) {
    for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, 1 - j, k) = s(i, j, k);
  }
  if (c.nbc == 1) {
    for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, nj + j, k) = s(i, j, k);
  } else if (c.nbc == 2) {
    for (int j = nj + 1; j <= nj + ng; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, j, k) = s(i, nj, k);
  } else if (c.nbc == 3 || c.nbc == 4) {
    for (int j = 1; j <= ng; ++j) for (int i = c.ib; i <= c.ie; ++i) s(i, nj + j, k) = s(i, nj + 1 - j, k);
  }
}

// extrapbcs: 2nd-order extrapolation to k=0 and k=nk+1 (cm1r18)
inline void extrapbcs(const BcInfo& c, const HArray3& s) {
  const int nk = c.nk;
  for (int j = c.jb; j <= c.je; ++j)
    for (int i = c.ib; i <= c.ie; ++i) {
      s(i, j, 0) = c.cgs1 * s(i, j, 1) + c.cgs2 * s(i, j, 2) + c.cgs3 * s(i, j, 3);
      s(i, j, nk + 1) = c.cgt1 * s(i, j, nk) + c.cgt2 * s(i, j, nk - 1) + c.cgt3 * s(i, j, nk - 2);
    }
}

} // namespace init
} // namespace cm1k

#endif
