// Port of src/kessler.F, misclibs.F pdefq/getcvm, and the ptype=1 branch of
// src/mp_driver.F. Operation order and parenthesization follow the Fortran
// exactly (left-to-right evaluation of a*b*c, a*b/c, ...), so results match
// the double-precision Fortran bit for bit.
#include "cm1k_kessler.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cm1k::kessler {

namespace {
KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }
} // namespace

Options Options::from_namelist(const Namelist& nl) {
  if (nl.geti("imoist", 0) != 1 || nl.geti("ptype", 0) != 1)
    throw std::runtime_error("kessler::Options: requires imoist=1 and ptype=1");
  Options o;
  o.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  const int psolver = nl.geti("psolver", 3);
  if (psolver >= 4 && psolver <= 7 && o.eqtset == 2) o.eqtset = 1;
  o.efall = nl.geti("efall", 0);
  return o;
}

void Work::allocate(const Bounds& B) {
  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  tmp = s3("kessler_tmp");
  cvm = s3("kessler_cvm");
  tten = s3("kessler_tten");
  vr = s3("kessler_vr");
  prate = FArray2("prate", {B.ib, B.jb}, {B.ie, B.je});
}

//---------------------------------------------------------------------------
void pdefq(const Bounds& b, Real rmax, const FArray3& rho, const FArray4& q3d, int n) {
  const int nk = b.nk;
  Kokkos::parallel_for("pdefq", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    double t1 = 0.0, t2 = 0.0;
    for (int k = 1; k <= nk; ++k) {
      t1 = t1 + rho(i, j, k) * q3d(i, j, k, n);
      if (q3d(i, j, k, n) < rmax) q3d(i, j, k, n) = 0.0;
      t2 = t2 + rho(i, j, k) * q3d(i, j, k, n);
    }
    double t3 = (t1 + 1.0e-20) / (t2 + 1.0e-20);
    if (t3 < 0.0) t3 = 1.0;
    for (int k = 1; k <= nk; ++k) q3d(i, j, k, n) = t3 * q3d(i, j, k, n);
  });
}

//---------------------------------------------------------------------------
void k_fallout(const Bounds& b, const FArray3& rho, const FArray4& q3d, int n, const FArray3& vr) {
  Kokkos::parallel_for("k_fallout", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    vr(i, j, k) = 0.0;
    if (q3d(i, j, k, n) > 1.0e-12) {
      vr(i, j, k) = 14.34 * Kokkos::pow(rho(i, j, k) * q3d(i, j, k, n), 0.1346)
                  * Kokkos::sqrt(1.15 / rho(i, j, k));
    }
  });
}

//---------------------------------------------------------------------------
void getcvm(const Bounds& b, const Constants& c, int eqtset, const FArray3& cvm, const FArray4& q) {
  const Real cv = c.cv, cvv = c.cvv, cpl = c.cpl;
  const bool dry = (eqtset <= 1);   // imoist=1 here
  Kokkos::parallel_for("getcvm", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    if (dry) {
      cvm(i, j, k) = cv;
    } else {
      Real x = cv + cvv * q(i, j, k, nqv);
      for (int n = 2; n <= 3; ++n) x = x + cpl * q(i, j, k, n);   // nql1..nql2; iice=0
      cvm(i, j, k) = x;
    }
  });
}

//---------------------------------------------------------------------------
void getefall(const Bounds& b, const Constants& c, Real rdz, int setzero, Real cpx, const FArray3& mf,
              const FArray3& t, const FArray3& cvm, const FArray3& tten, const FArray4& q3d, int n,
              const FArray3& vr) {
  const Real g = c.g;
  Kokkos::parallel_for("getefall", box3(1, b.ni, 1, b.nj, 1, b.nk - 1), KOKKOS_LAMBDA(int i, int j, int k) {
    if (setzero == 1) tten(i, j, k) = 0.0;
    tten(i, j, k) = tten(i, j, k) + q3d(i, j, k, n) * vr(i, j, k) * (
                      cpx * (t(i, j, k + 1) - t(i, j, k)) * rdz * mf(i, j, k + 1) + g
                    ) / cvm(i, j, k);
  });
}

//---------------------------------------------------------------------------
// Column recurrence: serial in k (and in the nfall substeps), parallel over (i,j).
// rq(k) and rq(k+1) are computed from q before q(k) is updated, which is
// what the Fortran's rq(-1:nk+3) work array holds.
void fallout(const Bounds& b, Real dt, Real rdz, const FArray3& mh, const FArray3& rho,
             const FArray4& q3d, int n, const FArray3& vq, const FArray3& rain, const FArray2& prate) {
  const int nk = b.nk, nrain = b.nrain;
  Kokkos::parallel_for("fallout", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    Real crmax = 0.0;
    for (int k = 1; k <= nk; ++k) crmax = fmax2(crmax, vq(i, j, k) * dt * rdz * mh(i, j, k));
    int nfall = int(crmax + 1.0);
    if (nfall < 1) nfall = 1;
    for (int k = nk - 1; k >= 1; --k)
      if (q3d(i, j, k, n) < 1.0e-8) vq(i, j, k) = vq(i, j, k + 1);
    const Real dtfall = dt / nfall;
    for (int nf = 1; nf <= nfall; ++nf) {
      const Real rq1 = rho(i, j, 1) * vq(i, j, 1) * fmax2(0.0, q3d(i, j, 1, n));
      Real rqk = rq1;
      for (int k = 1; k <= nk; ++k) {
        const Real rqkp1 = (k < nk) ? rho(i, j, k + 1) * vq(i, j, k + 1) * fmax2(0.0, q3d(i, j, k + 1, n)) : 0.0;
        q3d(i, j, k, n) = q3d(i, j, k, n) + dtfall * (rqkp1 - rqk) * rdz * mh(i, j, k) / rho(i, j, k);
        rqk = rqkp1;
      }
      for (int nr = 1; nr <= nrain; ++nr)
        rain(i, j, nr) = rain(i, j, nr) + 0.1 * dtfall * rq1;
      prate(i, j) = rq1;
    }
  });
}

//---------------------------------------------------------------------------
void kessler(const Bounds& b, const Constants& c, int eqtset, Real dt, const FArray3& pi0,
             const FArray3& th0, const FArray3& tmp, const FArray3& rho, const FArray3& pp3d,
             const FArray3& th3d, const FArray3& prs, const FArray4& q3d) {
  const Constants C = c;
  Kokkos::parallel_for("kessler", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    tmp(i, j, k) = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));

    Real qvnew = q3d(i, j, k, nqv);
    Real qcnew = q3d(i, j, k, nqc);
    Real qrnew = q3d(i, j, k, nqr);

    Real ar = 0.0, cr = 0.0, er = 0.0;

    // autoconversion of cloud to rain
    if (qcnew > 0.0) {
      ar = fmax2(0.001 * (qcnew - 0.001), 0.0);
      ar = ar * dt;
    }
    // accretion of cloud by rain
    if (qcnew > 0.0 && qrnew > 0.0) {
      cr = 2.2 * qcnew * Kokkos::pow(qrnew, 0.875);
      cr = cr * dt;
    }
    // evap of rain to vapor
    if (qrnew > 0.0) {
      Real esl = 611.2 * Kokkos::exp(17.67 * (tmp(i, j, k) - 273.15) / (tmp(i, j, k) - 29.65));
      esl = fmin2(esl, prs(i, j, k) * 0.5);
      const Real qvs = C.eps * esl / (prs(i, j, k) - esl);
      if (qvnew < qvs) {
        er = (1.6 + 30.3922 * Kokkos::pow(rho(i, j, k) * qrnew, 0.2046)) *
             (1.0 - (qvnew / qvs)) *
             Kokkos::pow(rho(i, j, k) * qrnew, 0.525) /
             ((2.03e4 + 9.584e6 / (qvs * prs(i, j, k))) * rho(i, j, k));
        er = fmin2(er * dt, qrnew);
        if ((qvnew + er) > qvs) er = qvs - qvnew;
      }
    }

    if ((ar + cr) > qcnew) {
      const Real term1 = ar + cr;
      ar = qcnew * ar / term1;
      cr = qcnew * cr / term1;
    }

    qvnew = qvnew + er;
    qcnew = qcnew - (ar + cr);
    qrnew = qrnew + (ar + cr - er);

    if (er > 1.0e-7) {
      if (eqtset == 2) {
        const Real cpml = C.cp + C.cpv * qvnew + C.cpl * (qcnew + qrnew);
        const Real cvml = C.cv + C.cvv * qvnew + C.cpl * (qcnew + qrnew);
        const Real rm = C.rd + C.rv * qvnew;
        th3d(i, j, k) = th3d(i, j, k) - er * (
                (C.lv1 - C.lv2 * tmp(i, j, k)) / (C.cpdcv * cvml * (pi0(i, j, k) + pp3d(i, j, k)))
              - (th0(i, j, k) + th3d(i, j, k)) * (C.rv / cvml) * (1.0 - C.rovcp * cpml / rm));
        pp3d(i, j, k) = Kokkos::pow(rho(i, j, k) * rm * (th0(i, j, k) + th3d(i, j, k)) * phys::rp00, C.rddcv) - pi0(i, j, k);
        prs(i, j, k) = phys::p00 * Kokkos::pow(pi0(i, j, k) + pp3d(i, j, k), C.cpdrd);
      } else {
        th3d(i, j, k) = th3d(i, j, k) - er * ((C.lv1 - C.lv2 * tmp(i, j, k))
                                              / (C.cp * (pi0(i, j, k) + pp3d(i, j, k))));
        rho(i, j, k) = prs(i, j, k)
             / (C.rd * (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k)) * (1.0 + qvnew * C.reps));
      }
    }

    q3d(i, j, k, nqv) = qvnew;
    q3d(i, j, k, nqc) = qcnew;
    q3d(i, j, k, nqr) = qrnew;
  });
}

//---------------------------------------------------------------------------
void satadj(const Bounds& b, const Constants& c, int eqtset, Real tsmall, int nrk, Real dt,
            const FArray3& pi0, const FArray3& th0, const FArray3& rho, const FArray3& pp3d,
            const FArray3& prs, const FArray3& th3d, const FArray4& q3d) {
  (void)dt;   // only used for rdt in the Fortran, which is unused
  const Constants C = c;
  const auto policy = box3(1, b.ni, 1, b.nj, 1, b.nk);

  if (eqtset == 2) {
    const Real converge = (nrk == 4) ? 2.0 * tsmall : 20.0 * tsmall;
    int iflag = 0;
    Kokkos::parallel_reduce("satadj", policy, KOKKOS_LAMBDA(int i, int j, int k, int& flag) {
      Real tnew = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));
      Real esl = 611.2 * Kokkos::exp(17.67 * (tnew - 273.15) / (tnew - 29.65));
      esl = fmin2(esl, prs(i, j, k) * 0.5);
      Real qvs = C.eps * esl / (prs(i, j, k) - esl);

      if (q3d(i, j, k, nqc) > 1.0e-12 || q3d(i, j, k, nqv) > qvs) {
        Real qvnew = q3d(i, j, k, nqv);
        Real qcnew = q3d(i, j, k, nqc);
        Real ql = 0.0, qi = 0.0;
        for (int n = 2; n <= 3; ++n) ql = ql + q3d(i, j, k, n);   // nql1..nql2
        ql = fmax2(0.0, ql);
        qi = fmax2(0.0, qi);                                      // iice=0
        const Real cvml = C.cv + C.cvv * qvnew + C.cpl * ql + C.cpi * qi;
        const Real lhv = C.lv1 - C.lv2 * tnew;

        const Real t1 = (lhv - C.rv * tnew) / cvml;
        const Real d1 = t1 * 17.67 * 243.5;

        int n = 0;
        Real tlast = tnew, pnew = 0.0;
        bool doit = true;
        while (doit) {
          n = n + 1;
          Real dqv = (qvs - qvnew) / (1.0 + d1 * qvs / ((tnew - 29.65) * (tnew - 29.65)));
          dqv = fmin2(dqv, qcnew);
          if ((qvnew + dqv) < 1.0e-20) dqv = 1.0e-20 - qvnew;

          qvnew = qvnew + dqv;
          qcnew = qcnew - dqv;
          tnew = tnew - dqv * t1;
          pnew = rho(i, j, k) * (C.rd + C.rv * qvnew) * tnew;

          doit = false;
          if (Kokkos::fabs(tnew - tlast) > converge) {
            tlast = tnew;
            esl = 611.2 * Kokkos::exp(17.67 * (tnew - 273.15) / (tnew - 29.65));
            esl = fmin2(esl, prs(i, j, k) * 0.5);
            qvs = C.eps * esl / (pnew - esl);
            doit = true;
          }
          if (n == 100) { flag = 1; doit = false; }
        }

        prs(i, j, k) = pnew;
        pp3d(i, j, k) = Kokkos::pow(pnew * phys::rp00, C.rovcp) - pi0(i, j, k);
        th3d(i, j, k) = tnew / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
        q3d(i, j, k, nqc) = qcnew;
        q3d(i, j, k, nqv) = qvnew;
      }
    }, Kokkos::Max<int>(iflag));
    if (iflag > 0)   // (Max reducer identity is INT_MIN)
      throw std::runtime_error("kessler::satadj: convergence cannot be reached (stopcm1)");
  } else {
    Kokkos::parallel_for("satadj", policy, KOKKOS_LAMBDA(int i, int j, int k) {
      Real qvnew = q3d(i, j, k, nqv);
      Real qcnew = q3d(i, j, k, nqc);
      const Real tnew = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));
      Real esl = 611.2 * Kokkos::exp(17.67 * (tnew - 273.15) / (tnew - 29.65));
      esl = fmin2(esl, prs(i, j, k) * 0.5);
      const Real qvs = C.eps * esl / (prs(i, j, k) - esl);
      const Real lhv = C.lv1 - C.lv2 * tnew;
      Real dqv = (qvs - qvnew) / (1.0 + lhv * qvs * 17.67 * 243.5 / (C.cp * ((tnew - 29.65) * (tnew - 29.65))));
      if ((qvnew + dqv) < 1.0e-20) dqv = 1.0e-20 - qvnew;
      dqv = fmin2(dqv, fmax2(0.0, qcnew));

      qvnew = qvnew + dqv;
      qcnew = qcnew - dqv;

      th3d(i, j, k) = th3d(i, j, k) - dqv * (lhv / (C.cp * (pi0(i, j, k) + pp3d(i, j, k))));
      q3d(i, j, k, nqc) = qcnew;
      q3d(i, j, k, nqv) = qvnew;
      rho(i, j, k) = prs(i, j, k)
           / (C.rd * (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k)) * (1.0 + qvnew * C.reps));
    });
  }
}

//---------------------------------------------------------------------------
void mp_driver(const Options& opt, const Constants& c, const Bounds& b, Real dt, Real rdz,
               const FArray3& mh, const FArray3& mf, const FArray3& pi0, const FArray3& th0,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w) {
  if (b.numq < 3) throw std::runtime_error("kessler::mp_driver: numq must be 3 for ptype=1");
  const auto tmp = w.tmp;

  // dum1 = T
  Kokkos::parallel_for("mp_driver_T", box3(1, b.ni, 1, b.nj, 1, b.nk), KOKKOS_LAMBDA(int i, int j, int k) {
    tmp(i, j, k) = (th0(i, j, k) + th3d(i, j, k)) * (pi0(i, j, k) + pp3d(i, j, k));
  });

  pdefq(b, 0.0, rho, q3d, nqv);
  pdefq(b, opt.qsmall, rho, q3d, nqc);
  pdefq(b, opt.qsmall, rho, q3d, nqr);
  k_fallout(b, rho, q3d, nqr, w.vr);

  if (opt.efall >= 1) {
    getcvm(b, c, opt.eqtset, w.cvm, q3d);
    getefall(b, c, rdz, 1, c.cpl, mf, tmp, w.cvm, w.tten, q3d, nqr, w.vr);
    const Constants C = c;
    const Real tsmall = opt.tsmall;
    const auto tten = w.tten;
    Kokkos::parallel_for("mp_driver_efall", box3(1, b.ni, 1, b.nj, 1, b.nk - 1), KOKKOS_LAMBDA(int i, int j, int k) {
      if (Kokkos::fabs(dt * tten(i, j, k)) >= tsmall) {
        tmp(i, j, k) = tmp(i, j, k) + dt * tten(i, j, k);
        prs(i, j, k) = rho(i, j, k) * C.rd * tmp(i, j, k) * (1.0 + q3d(i, j, k, nqv) * C.reps);
        pp3d(i, j, k) = Kokkos::pow(prs(i, j, k) * phys::rp00, C.rovcp) - pi0(i, j, k);
        th3d(i, j, k) = tmp(i, j, k) / (pi0(i, j, k) + pp3d(i, j, k)) - th0(i, j, k);
      }
    });
  }

  fallout(b, dt, rdz, mh, rho, q3d, nqr, w.vr, rain, w.prate);
  kessler(b, c, opt.eqtset, dt, pi0, th0, tmp, rho, pp3d, th3d, prs, q3d);
  satadj(b, c, opt.eqtset, opt.tsmall, 4, dt, pi0, th0, rho, pp3d, prs, th3d, q3d);
}

void mp_driver(const Options& opt, const Constants& c, const Grid& g, const BaseState& base, Real dt,
               const FArray3& rho, const FArray3& prs, const FArray3& pp3d, const FArray3& th3d,
               const FArray4& q3d, const FArray3& rain, const Work& w) {
  mp_driver(opt, c, g.b, dt, g.rdz, g.mh, g.mf, base.pi0, base.th0, rho, prs, pp3d, th3d, q3d, rain, w);
}

} // namespace cm1k::kessler
