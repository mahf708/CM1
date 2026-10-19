#include "cm1k_bc.hpp"

#include <stdexcept>
#include <string>

namespace cm1k {

BcParams BcParams::make(const Bounds& b, const Namelist& nl) {
  BcParams p;
  p.wbc = nl.geti("wbc", 1);
  p.ebc = nl.geti("ebc", 1);
  p.sbc = nl.geti("sbc", 1);
  p.nbc = nl.geti("nbc", 1);
  p.axisymm = nl.geti("axisymm", 0);
  for (int v : {p.wbc, p.ebc, p.sbc, p.nbc})
    if (v < 1 || v > 4) throw std::runtime_error("cm1k bc: lateral bc must be 1..4, got " + std::to_string(v));
  if ((p.wbc == 1) != (p.ebc == 1) || (p.sbc == 1) != (p.nbc == 1))
    throw std::runtime_error("cm1k bc: periodic boundaries must be set on both sides (wbc/ebc, sbc/nbc)");
  p.ni = b.ni; p.nj = b.nj; p.nk = b.nk; p.ngxy = b.ngxy;
  p.ib = b.ib; p.ie = b.ie; p.jb = b.jb; p.je = b.je;
  return p;
}

void BcParams::set_mpi(const EdgeFlags& e) {
  mpi = true;
  ibw = e.ibw; ibe = e.ibe; ibs = e.ibs; ibn = e.ibn;
}

namespace {
using P2 = Policy2;
inline P2 jk(int j1, int j2, int k1, int k2) { return box2(j1, j2, k1, k2); }
} // namespace

//---------------------------------------------------------------------------
void bcs(const BcParams& p, const FArray3& s) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // west
  if (p.wbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcs_w", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(1 - i, j, k) = s(ni + 1 - i, j, k); });
  else if (p.wbc == 2 && p.on_w())
    Kokkos::parallel_for("bcs_w", box3(1 - ng, 0, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(1, j, k); });
  else if (p.wbc != 1 && p.on_w())
    Kokkos::parallel_for("bcs_w", box3(1, ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(1 - i, j, k) = s(i, j, k); });
  // east
  if (p.ebc == 1 && !p.mpi)
    Kokkos::parallel_for("bcs_e", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(ni + i, j, k) = s(i, j, k); });
  else if (p.ebc == 2 && p.on_e())
    Kokkos::parallel_for("bcs_e", box3(ni + 1, ni + ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(ni, j, k); });
  else if (p.ebc != 1 && p.on_e())
    Kokkos::parallel_for("bcs_e", box3(1, ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(ni + i, j, k) = s(ni + 1 - i, j, k); });
  // south
  if (p.sbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcs_s", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, 1 - j, k) = s(i, nj + 1 - j, k); });
  else if (p.sbc == 2 && p.on_s())
    Kokkos::parallel_for("bcs_s", box3(0, ni + 1, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(i, 1, k); });
  else if (p.sbc != 1 && p.on_s())
    Kokkos::parallel_for("bcs_s", box3(0, ni + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, 1 - j, k) = s(i, j, k); });
  // north
  if (p.nbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcs_n", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, nj + j, k) = s(i, j, k); });
  else if (p.nbc == 2 && p.on_n())
    Kokkos::parallel_for("bcs_n", box3(0, ni + 1, nj + 1, nj + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(i, nj, k); });
  else if (p.nbc != 1 && p.on_n())
    Kokkos::parallel_for("bcs_n", box3(0, ni + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, nj + j, k) = s(i, nj + 1 - j, k); });
}

//---------------------------------------------------------------------------
// bc.F bcs2/bct2/bcu2/bcv2/bcw2: corner halo rows next to a physical
// boundary (MPI only). su/sv: u/v staggering (one more point east/north),
// k2: last level (nk or nk+1).
namespace {
// s(i1:i2, j, k1:k2) = s(isrc, j, k1:k2)
void bc2_row(const FArray3& s, int j, int i1, int i2, int isrc, int k1, int k2) {
  Kokkos::parallel_for("bc2_row", jk(i1, i2, k1, k2), KOKKOS_LAMBDA(int i, int k) { s(i, j, k) = s(isrc, j, k); });
}
// s(i, j1:j2, k1:k2) = s(i, jsrc, k1:k2)
void bc2_col(const FArray3& s, int i, int j1, int j2, int jsrc, int k1, int k2) {
  Kokkos::parallel_for("bc2_col", jk(j1, j2, k1, k2), KOKKOS_LAMBDA(int j, int k) { s(i, j, k) = s(i, jsrc, k); });
}
// level k: s(i1:i2, j1:j2) = s(isrc, j) (ew) or s(i, jsrc) (!ew)
void bc2_2d(const FArray3& s, int k, int i1, int i2, int j1, int j2, bool ew, int src) {
  if (ew)
    Kokkos::parallel_for("bcs2_2d", jk(i1, i2, j1, j2), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(src, j, k); });
  else
    Kokkos::parallel_for("bcs2_2d", jk(i1, i2, j1, j2), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(i, src, k); });
}

// patch* flags of param.F
void corner2(const BcParams& p, const FArray3& s, int k2, int su, int sv) {
  if (!p.mpi) return;
  const int ng = p.ngxy;
  const int ie = p.ni + su, je = p.nj + sv;  // last interior point of this staggering
  if (p.ibw == 1) {
    if (p.ibs == 0) bc2_row(s, 0, 1 - ng, 0, 1, 1, k2);           // patchsww
    if (p.ibn == 0) bc2_row(s, je + 1, 1 - ng, 0, 1, 1, k2);      // patchnww
  }
  if (p.ibe == 1) {
    if (p.ibs == 0) bc2_row(s, 0, ie + 1, ie + ng, ie, 1, k2);      // patchsee
    if (p.ibn == 0) bc2_row(s, je + 1, ie + 1, ie + ng, ie, 1, k2); // patchnee
  }
  if (p.ibs == 1) {
    if (p.ibw == 0) bc2_col(s, 0, 1 - ng, 0, 1, 1, k2);           // patchsws
    if (p.ibe == 0) bc2_col(s, ie + 1, 1 - ng, 0, 1, 1, k2);      // patchses
  }
  if (p.ibn == 1) {
    if (p.ibw == 0) bc2_col(s, 0, je + 1, je + ng, je, 1, k2);      // patchnwn
    if (p.ibe == 0) bc2_col(s, ie + 1, je + 1, je + ng, je, 1, k2); // patchnen
  }
}
} // namespace

void bcs2(const BcParams& p, const FArray3& s) { corner2(p, s, p.nk, 0, 0); }
void bct2(const BcParams& p, const FArray3& t) { corner2(p, t, p.nk + 1, 0, 0); }
void bcu2(const BcParams& p, const FArray3& u) { corner2(p, u, p.nk, 1, 0); }
void bcv2(const BcParams& p, const FArray3& v) { corner2(p, v, p.nk, 0, 1); }
void bcw2(const BcParams& p, const FArray3& w) { corner2(p, w, p.nk + 1, 0, 0); }

// bc.F bcs2_2d: a 2D array (level k of s); the rows/columns are 3 deep
void bcs2_2d(const BcParams& p, const FArray3& s, int k) {
  if (!p.mpi) return;
  const int ni = p.ni, nj = p.nj, ng = p.ngxy;
  if (ng < 3) throw std::runtime_error("cm1k bcs2_2d: needs ngxy >= 3");
  if (p.ibw == 1) {
    if (p.ibs == 0) bc2_2d(s, k, 1 - ng, 0, -2, 0, true, 1);
    if (p.ibn == 0) bc2_2d(s, k, 1 - ng, 0, nj + 1, nj + 3, true, 1);
  }
  if (p.ibe == 1) {
    if (p.ibs == 0) bc2_2d(s, k, ni + 1, ni + ng, -2, 0, true, ni);
    if (p.ibn == 0) bc2_2d(s, k, ni + 1, ni + ng, nj + 1, nj + 3, true, ni);
  }
  if (p.ibs == 1) {
    if (p.ibw == 0) bc2_2d(s, k, -2, 0, 1 - ng, 0, false, 1);
    if (p.ibe == 0) bc2_2d(s, k, ni + 1, ni + 3, 1 - ng, 0, false, 1);
  }
  if (p.ibn == 1) {
    if (p.ibw == 0) bc2_2d(s, k, -2, 0, nj + 1, nj + ng, false, nj);
    if (p.ibe == 0) bc2_2d(s, k, ni + 1, ni + 3, nj + 1, nj + ng, false, nj);
  }
}

//---------------------------------------------------------------------------
void bcp(const BcParams& p, const FArray3& s) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  if ((p.wbc == 1 && !p.mpi) || (p.wbc != 1 && p.on_w())) {
    const int src = (p.wbc == 1) ? ni : 1;
    Kokkos::parallel_for("bcp_w", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s(0, j, k) = s(src, j, k); });
  }
  if ((p.ebc == 1 && !p.mpi) || (p.ebc != 1 && p.on_e())) {
    const int src = (p.ebc == 1) ? 1 : ni;
    Kokkos::parallel_for("bcp_e", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s(ni + 1, j, k) = s(src, j, k); });
  }
  if ((p.sbc == 1 && !p.mpi) || (p.sbc != 1 && p.on_s())) {
    const int src = (p.sbc == 1) ? nj : 1;
    Kokkos::parallel_for("bcp_s", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s(i, 0, k) = s(i, src, k); });
  }
  if ((p.nbc == 1 && !p.mpi) || (p.nbc != 1 && p.on_n())) {
    const int src = (p.nbc == 1) ? 1 : nj;
    Kokkos::parallel_for("bcp_n", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s(i, nj + 1, k) = s(i, src, k); });
  }
}

//---------------------------------------------------------------------------
void bcu(const BcParams& p, const FArray3& u) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // west
  if (p.wbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcu_w", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(1 - i, j, k) = u(ni + 1 - i, j, k); });
  else if (p.wbc == 2 && p.on_w())
    Kokkos::parallel_for("bcu_w", box3(1 - ng, 0, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(1, j, k); });
  else if (p.wbc != 1 && p.on_w())
    Kokkos::parallel_for("bcu_w", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      u(1, j, k) = 0.0;
      for (int i = 1; i <= ng; ++i) u(1 - i, j, k) = -u(1 + i, j, k);
    });
  // east
  if (p.ebc == 1 && !p.mpi)
    Kokkos::parallel_for("bcu_e", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(ni + 1 + i, j, k) = u(1 + i, j, k); });
  else if (p.ebc == 2 && p.on_e())
    Kokkos::parallel_for("bcu_e", box3(ni + 2, ni + 1 + ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(ni + 1, j, k); });
  else if (p.ebc != 1 && p.on_e())
    Kokkos::parallel_for("bcu_e", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      u(ni + 1, j, k) = 0.0;
      for (int i = 1; i <= ng; ++i) u(ni + 1 + i, j, k) = -u(ni + 1 - i, j, k);
    });
  // south
  if (p.sbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcu_s", box3(p.ib, p.ie + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, 1 - j, k) = u(i, nj + 1 - j, k); });
  else if (p.sbc == 2 && p.on_s())
    Kokkos::parallel_for("bcu_s", box3(0, ni + 2, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(i, 1, k); });
  else if (p.sbc != 1 && p.on_s())
    Kokkos::parallel_for("bcu_s", box3(0, ni + 2, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, 1 - j, k) = u(i, j, k); });
  // north
  if (p.nbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcu_n", box3(p.ib, p.ie + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, nj + j, k) = u(i, j, k); });
  else if (p.nbc == 2 && p.on_n())
    Kokkos::parallel_for("bcu_n", box3(0, ni + 2, nj + 1, nj + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(i, nj, k); });
  else if (p.nbc != 1 && p.on_n())
    Kokkos::parallel_for("bcu_n", box3(0, ni + 2, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, nj + j, k) = u(i, nj + 1 - j, k); });
}

//---------------------------------------------------------------------------
void bcv(const BcParams& p, const FArray3& v) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // south
  if (p.sbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcv_s", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, 1 - j, k) = v(i, nj + 1 - j, k); });
  else if (p.sbc == 2 && p.on_s())
    Kokkos::parallel_for("bcv_s", box3(0, ni + 1, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(i, 1, k); });
  else if (p.sbc != 1 && p.on_s())
    Kokkos::parallel_for("bcv_s", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      for (int j = 1; j <= ng; ++j) v(i, 1 - j, k) = -v(i, j + 1, k);
      v(i, 1, k) = 0.0;
    });
  // north
  if (p.axisymm == 1)
    Kokkos::parallel_for("bcv_axi", jk(p.ib, p.ie, 1, nk), KOKKOS_LAMBDA(int i, int k) { v(i, 2, k) = v(i, 1, k); });
  if (p.nbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcv_n", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, nj + 1 + j, k) = v(i, 1 + j, k); });
  else if (p.nbc == 2 && p.on_n())
    Kokkos::parallel_for("bcv_n", box3(0, ni + 1, nj + 2, nj + 1 + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(i, nj + 1, k); });
  else if (p.nbc != 1 && p.on_n())
    Kokkos::parallel_for("bcv_n", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      for (int j = 1; j <= ng; ++j) v(i, nj + 1 + j, k) = -v(i, nj + 1 - j, k);
      v(i, nj + 1, k) = 0.0;
    });
  // west
  if (p.wbc == 1 && !p.mpi)
    Kokkos::parallel_for("bcv_w", box3(1, ng, p.jb, p.je + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = v(ni + 1 - i, j, k); });
  else if (p.wbc == 2 && p.on_w())
    Kokkos::parallel_for("bcv_w", box3(1 - ng, 0, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(1, j, k); });
  else if (p.wbc != 1 && p.on_w())
    Kokkos::parallel_for("bcv_w", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = v(i, j, k); });
  if (p.axisymm == 1)
    Kokkos::parallel_for("bcv_axi_w", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = -v(i, j, k); });
  // east
  if (p.ebc == 1 && !p.mpi)
    Kokkos::parallel_for("bcv_e", box3(1, ng, p.jb, p.je + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(ni + i, j, k) = v(i, j, k); });
  else if (p.ebc == 2 && p.on_e())
    Kokkos::parallel_for("bcv_e", box3(ni + 1, ni + ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(ni, j, k); });
  else if (p.ebc != 1 && p.on_e())
    Kokkos::parallel_for("bcv_e", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(ni + i, j, k) = v(ni + 1 - i, j, k); });
}

//---------------------------------------------------------------------------
void bcw(const BcParams& p, const FArray3& w, int flag) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  if (nk >= 2) {
    // west
    if (p.wbc == 1 && !p.mpi)
      Kokkos::parallel_for("bcw_w", box3(1, ng, p.jb, p.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(ni + 1 - i, j, k); });
    else if (p.wbc == 2 && p.on_w())
      Kokkos::parallel_for("bcw_w", box3(1 - ng, 0, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(1, j, k); });
    else if (p.wbc != 1 && p.on_w())
      Kokkos::parallel_for("bcw_w", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(i, j, k); });
    // east
    if (p.ebc == 1 && !p.mpi)
      Kokkos::parallel_for("bcw_e", box3(1, ng, p.jb, p.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(i, j, k); });
    else if (p.ebc == 2 && p.on_e())
      Kokkos::parallel_for("bcw_e", box3(ni + 1, ni + ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(ni, j, k); });
    else if (p.ebc != 1 && p.on_e())
      Kokkos::parallel_for("bcw_e", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(ni + 1 - i, j, k); });
    // south
    if (p.sbc == 1 && !p.mpi)
      Kokkos::parallel_for("bcw_s", box3(p.ib, p.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, nj + 1 - j, k); });
    else if (p.sbc == 2 && p.on_s())
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1 - ng, 0, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, 1, k); });
    else if (p.sbc != 1 && p.on_s())
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, j, k); });
    // north
    if (p.nbc == 1 && !p.mpi)
      Kokkos::parallel_for("bcw_n", box3(p.ib, p.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, j, k); });
    else if (p.nbc == 2 && p.on_n())
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, nj + 1, nj + ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, nj, k); });
    else if (p.nbc != 1 && p.on_n())
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, nj + 1 - j, k); });
  }
  if (flag == 1)
    Kokkos::parallel_for("bcw_tb", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      w(i, j, 1) = 0.0;
      w(i, j, nk + 1) = 0.0;
    });
}

} // namespace cm1k
