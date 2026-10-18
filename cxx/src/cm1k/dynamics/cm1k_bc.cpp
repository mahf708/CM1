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

namespace {
using P2 = Policy2;
inline P2 jk(int j1, int j2, int k1, int k2) { return box2(j1, j2, k1, k2); }
} // namespace

//---------------------------------------------------------------------------
void bcs(const BcParams& p, const FArray3& s) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // west
  if (p.wbc == 1)
    Kokkos::parallel_for("bcs_w", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(1 - i, j, k) = s(pwrap(ni + 1 - i, 1, ni), j, k); });
  else if (p.wbc == 2)
    Kokkos::parallel_for("bcs_w", box3(1 - ng, 0, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(1, j, k); });
  else
    Kokkos::parallel_for("bcs_w", box3(1, ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(1 - i, j, k) = s(i, j, k); });
  // east
  if (p.ebc == 1)
    Kokkos::parallel_for("bcs_e", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(ni + i, j, k) = s(pwrap(i, 1, ni), j, k); });
  else if (p.ebc == 2)
    Kokkos::parallel_for("bcs_e", box3(ni + 1, ni + ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(ni, j, k); });
  else
    Kokkos::parallel_for("bcs_e", box3(1, ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(ni + i, j, k) = s(ni + 1 - i, j, k); });
  // south
  if (p.sbc == 1)
    Kokkos::parallel_for("bcs_s", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, 1 - j, k) = s(i, pwrap(nj + 1 - j, 1, nj), k); });
  else if (p.sbc == 2)
    Kokkos::parallel_for("bcs_s", box3(0, ni + 1, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(i, 1, k); });
  else
    Kokkos::parallel_for("bcs_s", box3(0, ni + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, 1 - j, k) = s(i, j, k); });
  // north
  if (p.nbc == 1)
    Kokkos::parallel_for("bcs_n", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, nj + j, k) = s(i, pwrap(j, 1, nj), k); });
  else if (p.nbc == 2)
    Kokkos::parallel_for("bcs_n", box3(0, ni + 1, nj + 1, nj + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, j, k) = s(i, nj, k); });
  else
    Kokkos::parallel_for("bcs_n", box3(0, ni + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      s(i, nj + j, k) = s(i, nj + 1 - j, k); });
}

//---------------------------------------------------------------------------
void bcp(const BcParams& p, const FArray3& s) {
  const int ni = p.ni, nj = p.nj, nk = p.nk;
  {
    const int src = (p.wbc == 1) ? ni : 1;
    Kokkos::parallel_for("bcp_w", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s(0, j, k) = s(src, j, k); });
  }
  {
    const int src = (p.ebc == 1) ? 1 : ni;
    Kokkos::parallel_for("bcp_e", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) { s(ni + 1, j, k) = s(src, j, k); });
  }
  {
    const int src = (p.sbc == 1) ? nj : 1;
    Kokkos::parallel_for("bcp_s", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s(i, 0, k) = s(i, src, k); });
  }
  {
    const int src = (p.nbc == 1) ? 1 : nj;
    Kokkos::parallel_for("bcp_n", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) { s(i, nj + 1, k) = s(i, src, k); });
  }
}

//---------------------------------------------------------------------------
void bcu(const BcParams& p, const FArray3& u) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // west
  if (p.wbc == 1)
    Kokkos::parallel_for("bcu_w", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(1 - i, j, k) = u(pwrap(ni + 1 - i, 1, ni), j, k); });
  else if (p.wbc == 2)
    Kokkos::parallel_for("bcu_w", box3(1 - ng, 0, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(1, j, k); });
  else
    Kokkos::parallel_for("bcu_w", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      u(1, j, k) = 0.0;
      for (int i = 1; i <= ng; ++i) u(1 - i, j, k) = -u(1 + i, j, k);
    });
  // east
  if (p.ebc == 1)
    Kokkos::parallel_for("bcu_e", box3(1, ng, p.jb, p.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(ni + 1 + i, j, k) = u(pwrap(1 + i, 2, ni), j, k); });
  else if (p.ebc == 2)
    Kokkos::parallel_for("bcu_e", box3(ni + 2, ni + 1 + ng, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(ni + 1, j, k); });
  else
    Kokkos::parallel_for("bcu_e", jk(0, nj + 1, 1, nk), KOKKOS_LAMBDA(int j, int k) {
      u(ni + 1, j, k) = 0.0;
      for (int i = 1; i <= ng; ++i) u(ni + 1 + i, j, k) = -u(ni + 1 - i, j, k);
    });
  // south
  if (p.sbc == 1)
    Kokkos::parallel_for("bcu_s", box3(p.ib, p.ie + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, 1 - j, k) = u(i, pwrap(nj + 1 - j, 1, nj), k); });
  else if (p.sbc == 2)
    Kokkos::parallel_for("bcu_s", box3(0, ni + 2, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(i, 1, k); });
  else
    Kokkos::parallel_for("bcu_s", box3(0, ni + 2, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, 1 - j, k) = u(i, j, k); });
  // north
  if (p.nbc == 1)
    Kokkos::parallel_for("bcu_n", box3(p.ib, p.ie + 1, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, nj + j, k) = u(i, pwrap(j, 1, nj), k); });
  else if (p.nbc == 2)
    Kokkos::parallel_for("bcu_n", box3(0, ni + 2, nj + 1, nj + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, j, k) = u(i, nj, k); });
  else
    Kokkos::parallel_for("bcu_n", box3(0, ni + 2, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      u(i, nj + j, k) = u(i, nj + 1 - j, k); });
}

//---------------------------------------------------------------------------
void bcv(const BcParams& p, const FArray3& v) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  // south
  if (p.sbc == 1)
    Kokkos::parallel_for("bcv_s", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, 1 - j, k) = v(i, pwrap(nj + 1 - j, 1, nj), k); });
  else if (p.sbc == 2)
    Kokkos::parallel_for("bcv_s", box3(0, ni + 1, 1 - ng, 0, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(i, 1, k); });
  else
    Kokkos::parallel_for("bcv_s", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      for (int j = 1; j <= ng; ++j) v(i, 1 - j, k) = -v(i, j + 1, k);
      v(i, 1, k) = 0.0;
    });
  // north
  if (p.axisymm == 1)
    Kokkos::parallel_for("bcv_axi", jk(p.ib, p.ie, 1, nk), KOKKOS_LAMBDA(int i, int k) { v(i, 2, k) = v(i, 1, k); });
  if (p.nbc == 1)
    Kokkos::parallel_for("bcv_n", box3(p.ib, p.ie, 1, ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, nj + 1 + j, k) = v(i, pwrap(1 + j, 2, nj), k); });
  else if (p.nbc == 2)
    Kokkos::parallel_for("bcv_n", box3(0, ni + 1, nj + 2, nj + 1 + ng, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(i, nj + 1, k); });
  else
    Kokkos::parallel_for("bcv_n", jk(0, ni + 1, 1, nk), KOKKOS_LAMBDA(int i, int k) {
      for (int j = 1; j <= ng; ++j) v(i, nj + 1 + j, k) = -v(i, nj + 1 - j, k);
      v(i, nj + 1, k) = 0.0;
    });
  // west
  if (p.wbc == 1)
    Kokkos::parallel_for("bcv_w", box3(1, ng, p.jb, p.je + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = v(pwrap(ni + 1 - i, 1, ni), j, k); });
  else if (p.wbc == 2)
    Kokkos::parallel_for("bcv_w", box3(1 - ng, 0, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(1, j, k); });
  else
    Kokkos::parallel_for("bcv_w", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = v(i, j, k); });
  if (p.axisymm == 1)
    Kokkos::parallel_for("bcv_axi_w", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(1 - i, j, k) = -v(i, j, k); });
  // east
  if (p.ebc == 1)
    Kokkos::parallel_for("bcv_e", box3(1, ng, p.jb, p.je + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(ni + i, j, k) = v(pwrap(i, 1, ni), j, k); });
  else if (p.ebc == 2)
    Kokkos::parallel_for("bcv_e", box3(ni + 1, ni + ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(i, j, k) = v(ni, j, k); });
  else
    Kokkos::parallel_for("bcv_e", box3(1, ng, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      v(ni + i, j, k) = v(ni + 1 - i, j, k); });
}

//---------------------------------------------------------------------------
void bcw(const BcParams& p, const FArray3& w, int flag) {
  const int ni = p.ni, nj = p.nj, nk = p.nk, ng = p.ngxy;
  if (nk >= 2) {
    // west
    if (p.wbc == 1)
      Kokkos::parallel_for("bcw_w", box3(1, ng, p.jb, p.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(pwrap(ni + 1 - i, 1, ni), j, k); });
    else if (p.wbc == 2)
      Kokkos::parallel_for("bcw_w", box3(1 - ng, 0, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(1, j, k); });
    else
      Kokkos::parallel_for("bcw_w", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(1 - i, j, k) = w(i, j, k); });
    // east
    if (p.ebc == 1)
      Kokkos::parallel_for("bcw_e", box3(1, ng, p.jb, p.je, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(pwrap(i, 1, ni), j, k); });
    else if (p.ebc == 2)
      Kokkos::parallel_for("bcw_e", box3(ni + 1, ni + ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(ni, j, k); });
    else
      Kokkos::parallel_for("bcw_e", box3(1, ng, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(ni + i, j, k) = w(ni + 1 - i, j, k); });
    // south
    if (p.sbc == 1)
      Kokkos::parallel_for("bcw_s", box3(p.ib, p.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, pwrap(nj + 1 - j, 1, nj), k); });
    else if (p.sbc == 2)
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1 - ng, 0, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, 1, k); });
    else
      Kokkos::parallel_for("bcw_s", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, 1 - j, k) = w(i, j, k); });
    // north
    if (p.nbc == 1)
      Kokkos::parallel_for("bcw_n", box3(p.ib, p.ie, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, pwrap(j, 1, nj), k); });
    else if (p.nbc == 2)
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, nj + 1, nj + ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, j, k) = w(i, nj, k); });
    else
      Kokkos::parallel_for("bcw_n", box3(0, ni + 1, 1, ng, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        w(i, nj + j, k) = w(i, nj + 1 - j, k); });
  }
  if (flag == 1)
    Kokkos::parallel_for("bcw_tb", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
      w(i, j, 1) = 0.0;
      w(i, j, nk + 1) = 0.0;
    });
}

//---------------------------------------------------------------------------
void bc2d(const BcParams& p, const FArray3& s, int k) {
  const int ni = p.ni, nj = p.nj, ng = p.ngxy;
  const int ib = p.ib, ie = p.ie, jb = p.jb, je = p.je;
  // west
  if (p.wbc == 1)
    Kokkos::parallel_for("bc2d_w", box2(1, ng, jb, je), KOKKOS_LAMBDA(int i, int j) {
      s(1 - i, j, k) = s(pwrap(ni + 1 - i, 1, ni), j, k); });
  else if (p.wbc == 2)
    Kokkos::parallel_for("bc2d_w", box2(1 - ng, 0, jb, je), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(1, j, k); });
  else
    Kokkos::parallel_for("bc2d_w", box2(1, ng, jb, je), KOKKOS_LAMBDA(int i, int j) { s(1 - i, j, k) = s(i, j, k); });
  // east
  if (p.ebc == 1)
    Kokkos::parallel_for("bc2d_e", box2(1, ng, jb, je), KOKKOS_LAMBDA(int i, int j) {
      s(ni + i, j, k) = s(pwrap(i, 1, ni), j, k); });
  else if (p.ebc == 2)
    Kokkos::parallel_for("bc2d_e", box2(ni + 1, ni + ng, jb, je), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(ni, j, k); });
  else
    Kokkos::parallel_for("bc2d_e", box2(1, ng, jb, je), KOKKOS_LAMBDA(int i, int j) {
      s(ni + i, j, k) = s(ni + 1 - i, j, k); });
  // south
  if (p.sbc == 1)
    Kokkos::parallel_for("bc2d_s", box2(ib, ie, 1, ng), KOKKOS_LAMBDA(int i, int j) {
      s(i, 1 - j, k) = s(i, pwrap(nj + 1 - j, 1, nj), k); });
  else if (p.sbc == 2)
    Kokkos::parallel_for("bc2d_s", box2(ib, ie, 1 - ng, 0), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(i, 1, k); });
  else
    Kokkos::parallel_for("bc2d_s", box2(ib, ie, 1, ng), KOKKOS_LAMBDA(int i, int j) { s(i, 1 - j, k) = s(i, j, k); });
  // north
  if (p.nbc == 1)
    Kokkos::parallel_for("bc2d_n", box2(ib, ie, 1, ng), KOKKOS_LAMBDA(int i, int j) {
      s(i, nj + j, k) = s(i, pwrap(j, 1, nj), k); });
  else if (p.nbc == 2)
    Kokkos::parallel_for("bc2d_n", box2(ib, ie, nj + 1, nj + ng), KOKKOS_LAMBDA(int i, int j) { s(i, j, k) = s(i, nj, k); });
  else
    Kokkos::parallel_for("bc2d_n", box2(ib, ie, 1, ng), KOKKOS_LAMBDA(int i, int j) {
      s(i, nj + j, k) = s(i, nj + 1 - j, k); });
}

//---------------------------------------------------------------------------
void bcwsfc(const BcParams& p, const FArray3& gz, const FArray3& dzdx, const FArray3& dzdy, const FArray3& u,
            const FArray3& v, const FArray3& w) {
  Kokkos::parallel_for("bcwsfc", box2(0, p.ni + 1, 0, p.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    w(i, j, 1) = 0.5 * ((u(i, j, 1) + u(i + 1, j, 1)) * dzdx(i, j, 1)
                        + (v(i, j, 1) + v(i, j + 1, 1)) * dzdy(i, j, 1)) * gz(i, j, 1);
  });
}

} // namespace cm1k
