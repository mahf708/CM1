#ifndef CM1K_MORRISON_IMPL_HPP
#define CM1K_MORRISON_IMPL_HPP

// Device-callable pieces of the Morrison port: morrison.F's GAMMA, DERF1
// and satadj2. Internal to physics/morrison.
#include "physics/morrison/cm1k_morrison.hpp"

namespace cm1k::morrison::impl {

constexpr Real PI = 3.1415926535897932384626434;
constexpr Real XXX = 0.9189385332046727417803297;

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// morrison.F GAMMA: W. J. Cody and L. Stoltz's rational approximations.
KOKKOS_INLINE_FUNCTION Real gamma(Real x) {
  constexpr Real ONE = 1.0, HALF = 0.5, TWELVE = 12.0, TWO = 2.0, ZERO = 0.0;
  constexpr Real XBIG = 35.040, XMININ = 1.18e-38, EPS = 1.19e-7, XINF = 3.4e38;
  const Real P[8] = {-1.71618513886549492533811e+0, 2.47656508055759199108314e+1,
                     -3.79804256470945635097577e+2, 6.29331155312818442661052e+2,
                     8.66966202790413211295064e+2, -3.14512729688483675254357e+4,
                     -3.61444134186911729807069e+4, 6.64561438202405440627855e+4};
  const Real Q[8] = {-3.08402300119738975254353e+1, 3.15350626979604161529144e+2,
                     -1.01515636749021914166146e+3, -3.10777167157231109440444e+3,
                     2.25381184209801510330112e+4, 4.75584627752788110767815e+3,
                     -1.34659959864969306392456e+5, -1.15132259675553483497211e+5};
  const Real C[7] = {-1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04,
                     7.93650793500350248e-04, -2.777777777777681622553e-03,
                     8.333333333333333331554247e-02, 5.7083835261e-03};
  bool parity = false;
  Real fact = ONE;
  int n = 0;
  Real y = x, y1, res, z;
  if (y <= ZERO) {
    // argument is negative
    y = -x;
    y1 = Kokkos::trunc(y);
    res = y - y1;
    if (res != ZERO) {
      if (y1 != Kokkos::trunc(y1 * HALF) * TWO) parity = true;
      fact = -PI / Kokkos::sin(PI * res);
      y = y + ONE;
    } else {
      return XINF;
    }
  }
  // argument is positive
  if (y < EPS) {
    if (y >= XMININ) {
      res = ONE / y;
    } else {
      return XINF;
    }
  } else if (y < TWELVE) {
    y1 = y;
    if (y < ONE) {
      z = y;
      y = y + ONE;
    } else {
      n = int(y) - 1;
      y = y - Real(n);
      z = y - ONE;
    }
    Real xnum = ZERO, xden = ONE;
    for (int i = 0; i < 8; ++i) {
      xnum = (xnum + P[i]) * z;
      xden = xden * z + Q[i];
    }
    res = xnum / xden + ONE;
    if (y1 < y) {
      res = res / y1;
    } else if (y1 > y) {
      for (int i = 1; i <= n; ++i) {
        res = res * y;
        y = y + ONE;
      }
    }
  } else {
    if (y <= XBIG) {
      const Real ysq = y * y;
      Real sum = C[6];
      for (int i = 0; i < 6; ++i) sum = sum / ysq + C[i];
      sum = sum / y - y + XXX;
      sum = sum + (y - HALF) * Kokkos::log(y);
      res = Kokkos::exp(sum);
    } else {
      return XINF;
    }
  }
  if (parity) res = -res;
  if (fact != ONE) res = fact / res;
  return res;
}

// morrison.F DERF1: error function by piecewise polynomials.
KOKKOS_INLINE_FUNCTION Real derf1(Real x) {
  const Real A[65] = {
      0.00000000005958930743e0, -0.00000000113739022964e0, 0.00000001466005199839e0, -0.00000016350354461960e0,
      0.00000164610044809620e0, -0.00001492559551950604e0, 0.00012055331122299265e0, -0.00085483269811296660e0,
      0.00522397762482322257e0, -0.02686617064507733420e0, 0.11283791670954881569e0, -0.37612638903183748117e0,
      1.12837916709551257377e0,
      0.00000000002372510631e0, -0.00000000045493253732e0, 0.00000000590362766598e0, -0.00000006642090827576e0,
      0.00000067595634268133e0, -0.00000621188515924000e0, 0.00005103883009709690e0, -0.00037015410692956173e0,
      0.00233307631218880978e0, -0.01254988477182192210e0, 0.05657061146827041994e0, -0.21379664776456006580e0,
      0.84270079294971486929e0,
      0.00000000000949905026e0, -0.00000000018310229805e0, 0.00000000239463074000e0, -0.00000002721444369609e0,
      0.00000028045522331686e0, -0.00000261830022482897e0, 0.00002195455056768781e0, -0.00016358986921372656e0,
      0.00107052153564110318e0, -0.00608284718113590151e0, 0.02986978465246258244e0, -0.13055593046562267625e0,
      0.67493323603965504676e0,
      0.00000000000382722073e0, -0.00000000007421598602e0, 0.00000000097930574080e0, -0.00000001126008898854e0,
      0.00000011775134830784e0, -0.00000111992758382650e0, 0.00000962023443095201e0, -0.00007404402135070773e0,
      0.00050689993654144881e0, -0.00307553051439272889e0, 0.01668977892553165586e0, -0.08548534594781312114e0,
      0.56909076642393639985e0,
      0.00000000000155296588e0, -0.00000000003032205868e0, 0.00000000040424830707e0, -0.00000000471135111493e0,
      0.00000005011915876293e0, -0.00000048722516178974e0, 0.00000430683284629395e0, -0.00003445026145385764e0,
      0.00024879276133931664e0, -0.00162940941748079288e0, 0.00988786373932350462e0, -0.05962426839442303805e0,
      0.49766113250947636708e0};
  const Real B[65] = {
      -0.00000000029734388465e0, 0.00000000269776334046e0, -0.00000000640788827665e0, -0.00000001667820132100e0,
      -0.00000021854388148686e0, 0.00000266246030457984e0, 0.00001612722157047886e0, -0.00025616361025506629e0,
      0.00015380842432375365e0, 0.00815533022524927908e0, -0.01402283663896319337e0, -0.19746892495383021487e0,
      0.71511720328842845913e0,
      -0.00000000001951073787e0, -0.00000000032302692214e0, 0.00000000522461866919e0, 0.00000000342940918551e0,
      -0.00000035772874310272e0, 0.00000019999935792654e0, 0.00002687044575042908e0, -0.00011843240273775776e0,
      -0.00080991728956032271e0, 0.00661062970502241174e0, 0.00909530922354827295e0, -0.20160072778491013140e0,
      0.51169696718727644908e0,
      0.00000000003147682272e0, -0.00000000048465972408e0, 0.00000000063675740242e0, 0.00000003377623323271e0,
      -0.00000015451139637086e0, -0.00000203340624738438e0, 0.00001947204525295057e0, 0.00002854147231653228e0,
      -0.00101565063152200272e0, 0.00271187003520095655e0, 0.02328095035422810727e0, -0.16725021123116877197e0,
      0.32490054966649436974e0,
      0.00000000002319363370e0, -0.00000000006303206648e0, -0.00000000264888267434e0, 0.00000002050708040581e0,
      0.00000011371857327578e0, -0.00000211211337219663e0, 0.00000368797328322935e0, 0.00009823686253424796e0,
      -0.00065860243990455368e0, -0.00075285814895230877e0, 0.02585434424202960464e0, -0.11637092784486193258e0,
      0.18267336775296612024e0,
      -0.00000000000367789363e0, 0.00000000020876046746e0, -0.00000000193319027226e0, -0.00000000435953392472e0,
      0.00000018006992266137e0, -0.00000078441223763969e0, -0.00000675407647949153e0, 0.00008428418334440096e0,
      -0.00017604388937031815e0, -0.00239729611435071610e0, 0.02064129023876022970e0, -0.06905562880005864105e0,
      0.09084526782065478489e0};
  const Real w = Kokkos::fabs(x);
  Real y;
  if (w < 2.2) {
    Real t = w * w;
    int k = int(t);
    t = t - Real(k);
    k = k * 13;
    y = ((((((((((((A[k] * t + A[k + 1]) * t + A[k + 2]) * t + A[k + 3]) * t + A[k + 4]) * t + A[k + 5]) * t +
                A[k + 6]) * t + A[k + 7]) * t + A[k + 8]) * t + A[k + 9]) * t + A[k + 10]) * t + A[k + 11]) * t +
         A[k + 12]) * w;
  } else if (w < 6.9) {
    int k = int(w);
    const Real t = w - Real(k);
    k = 13 * (k - 2);
    y = (((((((((((B[k] * t + B[k + 1]) * t + B[k + 2]) * t + B[k + 3]) * t + B[k + 4]) * t + B[k + 5]) * t +
               B[k + 6]) * t + B[k + 7]) * t + B[k + 8]) * t + B[k + 9]) * t + B[k + 10]) * t + B[k + 11]) * t +
        B[k + 12];
    y = y * y;
    y = y * y;
    y = y * y;
    y = 1.0 - y * y;
  } else {
    y = 1.0;
  }
  if (x < 0.0) y = -y;
  return y;
}

// es over liquid (Bolton-type formula used throughout morrison.F), capped at prs/2
KOKKOS_INLINE_FUNCTION Real esl_fn(Real prs, Real t) {
  return fmin2(prs * 0.5, 611.2 * Kokkos::exp(17.67 * (t - 273.15) / (t - 29.65)));
}

// morrison.F satadj2; returns iflag (1: no convergence in 100 iterations,
// where the Fortran calls stopcm1). Uses the module's constants and QSMALL.
KOKKOS_INLINE_FUNCTION int satadj2(const Params& p, Real& tnew, Real& prs, Real& qvnew, Real& qcnew,
                                   Real qrnew, Real qinew, Real qsnew, Real qgnew, Real rho, int eqtset,
                                   Real tsmall) {
  int iflag = 0;
  if (eqtset == 2) {
    const Real converge = 2.0 * tsmall;
    Real esl = esl_fn(prs, tnew);
    Real qvs = p.ep_2 * esl / (prs - esl);
    if (qcnew > p.qsmall || qvnew > qvs) {
      const Real ql = fmax2(0.0, qcnew + qrnew);
      const Real qi = fmax2(0.0, qinew + qsnew + qgnew);
      const Real cvml = p.cv + p.cvv * qvnew + p.cpl * ql + p.cpi * qi;
      const Real lhv = p.lv1 - p.lv2 * tnew;
      const Real t1 = (lhv - p.rv * tnew) / cvml;
      const Real d1 = t1 * 17.67 * 243.5;
      int n = 0;
      Real tlast = tnew;
      bool doit = true;
      while (doit) {
        n = n + 1;
        const Real tm = tnew - 29.65;
        Real dqv = (qvs - qvnew) / (1.0 + d1 * qvs / (tm * tm));
        dqv = fmin2(dqv, qcnew);
        if ((qvnew + dqv) < 1.0e-20) dqv = 1.0e-20 - qvnew;
        qvnew = qvnew + dqv;
        qcnew = qcnew - dqv;
        tnew = tnew - dqv * t1;
        prs = rho * (p.r + p.rv * qvnew) * tnew;
        doit = false;
        if (Kokkos::fabs(tnew - tlast) > converge) {
          tlast = tnew;
          esl = esl_fn(prs, tnew);
          qvs = p.ep_2 * esl / (prs - esl);
          doit = true;
        }
        if (n == 100) {
          iflag = 1;
          doit = false;
        }
      }
    }
  } else {
    const Real esl = esl_fn(prs, tnew);
    const Real qvs = p.ep_2 * esl / (prs - esl);
    const Real lhv = p.lv1 - p.lv2 * tnew;
    const Real tm = tnew - 29.65;
    Real dqv = (qvs - qvnew) / (1.0 + lhv * qvs * 17.67 * 243.5 / (p.cp * (tm * tm)));
    if ((qvnew + dqv) < 1.0e-20) dqv = 1.0e-20 - qvnew;
    dqv = fmin2(dqv, fmax2(0.0, qcnew));
    qvnew = qvnew + dqv;
    qcnew = qcnew - dqv;
    tnew = tnew - dqv * lhv / (p.cp * (1.0 + 0.887 * qvnew));
  }
  return iflag;
}

} // namespace cm1k::morrison::impl

#endif
