#include "init/cm1k_base_snd.hpp"

#include "init/cm1k_cm1libs.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Ports of the isnd and iwnd blocks of base.F (and getthe/revthe), on the
// host, in the Fortran loop order and with its parenthesization. Comments
// give the Fortran names of the cases; numbers in "stuck in loop (nnn)" are
// the Fortran's.

namespace cm1k {
namespace init {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
  throw std::runtime_error("cm1k init: unsupported option: " + what);
}
[[noreturn]] void stop(const std::string& what) {
  throw std::runtime_error("cm1k init (base.F): " + what);
}

// Goddard constants base.F uses (isnd = 10 with ice). They are set only by
// consat (goddard.F), which param.F calls for ptype = 2; for every other
// ptype the module variables stay 0.
struct GoddardT {
  Real t0k = 0, t00k = 0, rt0 = 0;
};
GoddardT goddard_t(int ptype) {
  GoddardT gt;
  if (ptype == 2) {
    gt.t0k = 273.15;
    gt.t00k = 233.15;
    gt.rt0 = 1. / (gt.t0k - gt.t00k);
  }
  return gt;
}

} // namespace

int param_eqtset(const Namelist& nl) {
  int eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  const int psolver = nl.geti("psolver", 3), ptype = nl.geti("ptype", 0);
  if ((psolver == 4 || psolver == 5 || psolver == 6 || psolver == 7) && eqtset == 2) eqtset = 1;
  if ((ptype == 4 || ptype == 7 || ptype == 50 || ptype == 51 || ptype == 52 || ptype == 53 || ptype == 55) && eqtset == 2)
    eqtset = 1;
  return eqtset;
}

int param_iice(const Namelist& nl) {
  if (nl.geti("imoist", 0) != 1) return 0;
  const int ptype = nl.geti("ptype", 0);
  return (ptype == 0 || ptype == 1 || ptype == 6) ? 0 : 1;
}

std::string input_sounding_path() {
  const char* p = std::getenv("CM1K_INPUT_SOUNDING");
  return (p && *p) ? std::string(p) : std::string("input_sounding");
}

//---------------------------------------------------------------------------
// list-directed reads of input_sounding
InputSounding read_input_sounding(const std::string& path, bool with_wind) {
  std::ifstream f(path);
  if (!f) stop("isnd = 7/17: cannot open input_sounding file '" + path + "'");
  // A list-directed read takes values from the current record onward and
  // discards the rest of its last record. Returns false at end of file
  // before the first value (end=), throws on a malformed value (err=).
  auto read_values = [&](int n, double* out) -> bool {
    int got = 0;
    std::string line;
    while (got < n) {
      if (!std::getline(f, line)) {
        if (got == 0) return false;
        stop("input_sounding: end of file inside a record");
      }
      for (char& ch : line)
        if (ch == ',' || ch == '\t' || ch == '\r') ch = ' ';
      std::istringstream ss(line);
      std::string tok;
      while (got < n && (ss >> tok)) {
        if (tok == "/") { got = n; break; }   // slash ends the read
        // Fortran accepts d/D exponents
        for (char& ch : tok)
          if (ch == 'd' || ch == 'D') ch = 'e';
        char* end = nullptr;
        const double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str() || *end != '\0') stop("input_sounding: cannot read value '" + tok + "'");
        out[got++] = v;
      }
    }
    return true;
  };
  InputSounding s;
  double hdr[3];
  if (!read_values(3, hdr)) stop("input_sounding: empty file");
  s.p_sfc = hdr[0]; s.th_sfc = hdr[1]; s.qv_sfc = hdr[2];
  const int nv = with_wind ? 5 : 3;
  double rec[5];
  while (read_values(nv, rec)) {
    s.z.push_back(rec[0]);
    s.th.push_back(rec[1]);
    s.qv.push_back(rec[2] / 1000.0);   // put qv in g/g
    s.u.push_back(with_wind ? rec[3] : 0.0);
    s.v.push_back(with_wind ? rec[4] : 0.0);
  }
  return s;
}

//---------------------------------------------------------------------------
// base.F: isnd
void base_sounding(const Namelist& nl, const Constants& c, const Grid& gr, const InitInfo& info, BaseArrays& a) {
  using namespace phys;
  const auto& B = gr.b;
  const auto& ex = info.ex;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const int ib = B.ib, ie = B.ie, jb = B.jb, je = B.je, kb = B.kb, ke = B.ke;
  const int isnd = nl.geti("isnd", 1), imoist = nl.geti("imoist", 0), ptype = nl.geti("ptype", 0);
  const int testcase = nl.geti("testcase", 0);
  const Real g = c.g, cp = c.cp, rd = c.rd, rv = c.rv, cpdrd = c.cpdrd, reps = c.reps, eps = c.eps;
  const Real rovcp = c.rovcp, cpl = c.cpl, cpv = c.cpv, lv1 = c.lv1, lv2 = c.lv2, ls1 = c.ls1, ls2 = c.ls2;
  const Real cgs1 = ex.cgs1, cgs2 = ex.cgs2, cgs3 = ex.cgs3;
  auto rslf = [&](Real p, Real t) { return cm1libs::rslf(ptype, eps, p, t); };
  auto rsif = [&](Real p, Real t) { return cm1libs::rsif(ptype, eps, p, t); };
  auto& zh = a.zh;
  auto& pi0 = a.pi0; auto& prs0 = a.prs0; auto& th0 = a.th0; auto& thv0 = a.thv0;
  auto& qv0 = a.qv0; auto& qc0 = a.qc0; auto& qi0 = a.qi0; auto& rh0 = a.rh0; auto& t0 = a.t0;
  auto& rho0 = a.rho0;
  (void)cgs1; (void)cgs2; (void)cgs3;

  // param.F: imoist = 1 needs a moist sounding
  if (imoist == 1 && (isnd == 2 || isnd == 3)) stop("isnd = " + std::to_string(isnd) + " requires imoist = 0");

  if (isnd == 1) {
    // dry adiabatic
    const Real th_sfc = 300.0, pi_sfc = 1.0;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          th0(i, j, k) = th_sfc;
          pi0(i, j, k) = pi_sfc - g * zh(i, j, k) / (cp * th_sfc);
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
        }
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) { qv0(i, j, k) = 0.0; rh0(i, j, k) = 0.0; }

  } else if (isnd == 2) {
    // dry isothermal
    const Real t_sfc = 250.0, prs_sfc = p00;
    const Real hs = rd * t_sfc / g;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          prs0(i, j, k) = prs_sfc * std::exp(-zh(i, j, k) / hs);
          pi0(i, j, k) = std::pow(prs0(i, j, k) / p00, rd / cp);
          th0(i, j, k) = t_sfc / pi0(i, j, k);
        }
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) { qv0(i, j, k) = 0.0; rh0(i, j, k) = 0.0; }

  } else if (isnd == 3) {
    // dry, constant dT/dz
    const Real th_sfc = 300.0, lapse = 0.0065;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          t0(i, j, k) = th_sfc - lapse * zh(i, j, k);
          prs0(i, j, k) = p00 * std::pow(t0(i, j, k) / th_sfc, g / (lapse * rd));
          pi0(i, j, k) = std::pow(prs0(i, j, k) / p00, rd / cp);
          th0(i, j, k) = th_sfc * std::pow(t0(i, j, k) / th_sfc, 1 - g / (lapse * cp));
        }
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) { qv0(i, j, k) = 0.0; rh0(i, j, k) = 0.0; }

  } else if (isnd == 4) {
    // saturated, neutrally-stable sounding (Bryan and Fritsch 2002)
    const Real thec_mb = 320.0, qt_mb = 0.020, prs_sfc = 100000.0;
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        prs0(i, j, 1) = prs_sfc;
        Real qv1 = qt_mb, tmp = thec_mb, tlast = tmp;
        for (int n = 1; n <= 20; ++n) {
          const Real thbar = (tmp * std::pow(p00 / prs0(i, j, 1), rd / cp)) * (1.0 + qv1 * reps) / (1.0 + qt_mb);
          pi0(i, j, 1) = 1.0 - g * zh(i, j, 1) / (cp * thbar);
          prs0(i, j, 1) = p00 * std::pow(pi0(i, j, 1), cpdrd);
          qv1 = rslf(prs0(i, j, 1), tmp);
          tmp = thec_mb * std::pow(prs0(i, j, 1) / (1.0 + qv1 / eps) / p00, rd / (cp + cpl * qt_mb))
                / std::exp((lv1 - lv2 * tmp) * qv1 / (tmp * (cp + cpl * qt_mb)));
          tmp = tlast + 0.3 * (tmp - tlast);
          tlast = tmp;
        }
        t0(i, j, 1) = tmp;
        th0(i, j, 1) = tmp / pi0(i, j, 1);
        rh0(i, j, 1) = 1.0;
        qv0(i, j, 1) = rslf(prs0(i, j, 1), tmp);
        Real t1 = tmp, th1 = th0(i, j, 1);
        qv1 = qv0(i, j, 1);
        for (int k = 2; k <= nk; ++k) {
          tlast = th1;
          Real t2 = t1, qv2 = qv1, th2 = th1;
          int n = 0;
          pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * th2);
          while (true) {
            n = n + 1;
            th2 = tlast;
            t2 = th2 * pi0(i, j, k);
            const Real thbar = 0.5 * (th1 * (1.0 + qv1 * reps) / (1.0 + qt_mb) + th2 * (1.0 + qv2 * reps) / (1.0 + qt_mb));
            pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * thbar);
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
            qv2 = rslf(prs0(i, j, k), t2);
            t2 = thec_mb * std::pow(prs0(i, j, k) / (1.0 + qv2 / eps) / p00, rd / (cp + cpl * qt_mb))
                 / std::exp((lv1 - lv2 * t2) * qv2 / (t2 * (cp + cpl * qt_mb)));
            th2 = t2 / pi0(i, j, k);
            if (std::abs(th2 - tlast) > 0.0001 && n < 100) {
              tlast = tlast + 0.3 * (th2 - tlast);
              continue;
            } else if (n >= 100) {
              stop("isnd = 4: stuck in loop!");
            }
            break;
          }
          t0(i, j, k) = t2;
          th0(i, j, k) = th2;
          qv0(i, j, k) = rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k));
          rh0(i, j, k) = 1.0;
          th1 = th2; qv1 = qv2; t1 = t2;
        }
        for (int k = 1; k <= nk; ++k) qc0(i, j, k) = qt_mb - qv0(i, j, k);
      }

  } else if (isnd == 5) {
    // Weisman-Klemp analytic sounding
    const Real z_trop = 12000.0, th_trop = 343.0, t_trop = 213.0, th_sfc = 300.0;
    const Real prs_sfc = 100000.0, qv_pbl = 0.014;
    const Real pi_sfc = std::pow(prs_sfc / p00, rd / cp);
    const Real qv_sfc = rslf(prs_sfc, th_sfc * pi_sfc);
    const Real thv_sfc = th_sfc * (1.0 + qv_sfc * reps) / (1.0 + qv_sfc);
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        for (int k = kb; k <= ke; ++k) rh0(i, j, k) = 0.0;
        for (int k = 1; k <= nk; ++k) {
          if (zh(i, j, k) < z_trop) {
            th0(i, j, k) = th_sfc + (th_trop - th_sfc) * std::pow(zh(i, j, k) / z_trop, 1.25);
            if (imoist == 1) rh0(i, j, k) = 1.0 - 0.75 * std::pow(zh(i, j, k) / z_trop, 1.25);
          } else {
            th0(i, j, k) = th_trop * std::exp((g / (t_trop * cp)) * (zh(i, j, k) - z_trop));
            if (imoist == 1) rh0(i, j, k) = 0.25;
          }
        }
        th0(i, j, 0) = th0(i, j, 1);
        th0(i, j, nk + 1) = th0(i, j, nk);
        for (int k = kb; k <= ke; ++k) qv0(i, j, k) = 0.0;
        for (int n = 1; n <= 20; ++n) {
          for (int k = kb; k <= ke; ++k)
            thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k));
          pi0(i, j, 1) = pi_sfc - g * zh(i, j, 1) / (cp * 0.5 * (thv_sfc + thv0(i, j, 1)));
          for (int k = 2; k <= nk; ++k)
            pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (thv0(i, j, k) + thv0(i, j, k - 1)));
          for (int k = 1; k <= nk; ++k) prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          for (int k = 1; k <= nk; ++k) {
            qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k));
            if (qv0(i, j, k) > qv_pbl) qv0(i, j, k) = qv_pbl;
          }
        }
        for (int k = 1; k <= nk; ++k)
          rh0(i, j, k) = qv0(i, j, k) / (rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k)));
      }

  } else if (isnd == 6) {
    stop("isnd = 6 code has been removed (CM1 stops)");

  } else if (isnd == 7 || isnd == 17) {
    // input_sounding (isnd = 17: thermodynamics only, winds from iwnd)
    const auto s = read_input_sounding(input_sounding_path(), isnd == 7);
    const int nsnd = int(s.z.size()) + 1;
    if (nsnd <= 2) stop("isnd = 7/17: there is only one u,v level; at least two u,v levels are needed");
    std::vector<double> zsnd(nsnd + 1, 0.0), thsnd(nsnd + 1, 0.0), qvsnd(nsnd + 1, 0.0), usnd(nsnd + 1, 0.0),
        vsnd(nsnd + 1, 0.0), thvsnd(nsnd + 1, 0.0), pisnd(nsnd + 1, 0.0), psnd(nsnd + 1, 0.0), tsnd(nsnd + 1, 0.0),
        rhsnd(nsnd + 1, 0.0);
    for (int k = 2; k <= nsnd; ++k) {
      zsnd[k] = s.z[k - 2]; thsnd[k] = s.th[k - 2]; qvsnd[k] = s.qv[k - 2];
      usnd[k] = s.u[k - 2]; vsnd[k] = s.v[k - 2];
    }
    Real p_sfc = s.p_sfc, th_sfc = s.th_sfc, qv_sfc = s.qv_sfc;
    if (imoist == 0) qv_sfc = 0.0;
    p_sfc = p_sfc * 100.0;
    qv_sfc = qv_sfc / 1000.0;
    const Real pi_sfc = std::pow(p_sfc / p00, rd / cp);
    const Real thv_sfc = th_sfc * (1.0 + qv_sfc * reps) / (1.0 + qv_sfc);
    zsnd[1] = 0.0;
    thsnd[1] = th_sfc;
    qvsnd[1] = qv_sfc;
    if (imoist == 0)
      for (auto& q : qvsnd) q = 0.0;
    if (isnd == 7) {
      // 190324: simple extrapolation
      usnd[1] = usnd[2] - (zsnd[2] - zsnd[1]) * (usnd[3] - usnd[2]) / (zsnd[3] - zsnd[2]);
      vsnd[1] = vsnd[2] - (zsnd[2] - zsnd[1]) * (vsnd[3] - vsnd[2]) / (zsnd[3] - zsnd[2]);
    }
    for (int k = 1; k <= nsnd; ++k) thvsnd[k] = thsnd[k] * (1.0 + reps * qvsnd[k]) / (1.0 + qvsnd[k]);
    pisnd[1] = pi_sfc;
    for (int k = 2; k <= nsnd; ++k)
      pisnd[k] = pisnd[k - 1] - g * (zsnd[k] - zsnd[k - 1]) / (cp * 0.5 * (thvsnd[k] + thvsnd[k - 1]));
    for (int k = 1; k <= nsnd; ++k) {
      psnd[k] = p00 * std::pow(pisnd[k], cp / rd);
      tsnd[k] = thsnd[k] * pisnd[k];
    }
    if (imoist == 1)
      for (int k = 1; k <= nsnd; ++k) {
        if (qvsnd[k] < 1.0e-12) qvsnd[k] = 0.0001 * rslf(psnd[k], thsnd[k] * pisnd[k]);
        rhsnd[k] = qvsnd[k] / rslf(psnd[k], tsnd[k]);
      }

    std::vector<double> thinterp(nk + 1), qvinterp(nk + 1), uinterp(nk + 1), vinterp(nk + 1), pinterp(nk + 1),
        tinterp(nk + 1), rhinterp(nk + 1);
    auto& u0 = a.u0; auto& v0 = a.v0;
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        if (zsnd[nsnd] < zh(i, j, nk)) stop("isnd = 7/17: zmax of sounding < zmax of grid");
        for (int k = 1; k <= nk; ++k) {
          int kk = 1;
          while (zsnd[kk] < zh(i, j, k)) kk = kk + 1;
          const int kdn = kk - 1, kup = kk;
          if (kdn < 1) stop("isnd = 7/17: grid level at or below the sounding's surface (z = 0)");
          const Real zu = 0.5 * (zh(std::max(ib, i - 1), j, k) + zh(i, j, k));
          Real interp_frac = (zu - zsnd[kdn]) / (zsnd[kup] - zsnd[kdn]);
          uinterp[k] = usnd[kdn] + (usnd[kup] - usnd[kdn]) * interp_frac;
          const Real zv = 0.5 * (zh(i, std::max(jb, j - 1), k) + zh(i, j, k));
          interp_frac = (zv - zsnd[kdn]) / (zsnd[kup] - zsnd[kdn]);
          vinterp[k] = vsnd[kdn] + (vsnd[kup] - vsnd[kdn]) * interp_frac;
          interp_frac = (zh(i, j, k) - zsnd[kdn]) / (zsnd[kup] - zsnd[kdn]);
          thinterp[k] = thsnd[kdn] + (thsnd[kup] - thsnd[kdn]) * interp_frac;
          qvinterp[k] = qvsnd[kdn] + (qvsnd[kup] - qvsnd[kdn]) * interp_frac;
          pinterp[k] = psnd[kdn] + (psnd[kup] - psnd[kdn]) * interp_frac;
          tinterp[k] = tsnd[kdn] + (tsnd[kup] - tsnd[kdn]) * interp_frac;
          rhinterp[k] = rhsnd[kdn] + (rhsnd[kup] - rhsnd[kdn]) * interp_frac;
        }
        for (int k = 1; k <= nk; ++k) {
          u0(i, j, k) = uinterp[k];
          v0(i, j, k) = vinterp[k];
          qv0(i, j, k) = qvinterp[k];
          th0(i, j, k) = thinterp[k];
          prs0(i, j, k) = pinterp[k];
          t0(i, j, k) = tinterp[k];
          rh0(i, j, k) = rhinterp[k];
        }
        for (int k = 1; k <= nk; ++k) qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
        for (int k = 1; k <= nk; ++k)
          thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k));
        pi0(i, j, 1) = pi_sfc - g * zh(i, j, 1) / (cp * 0.5 * (thv_sfc + thv0(i, j, 1)));
        for (int k = 2; k <= nk; ++k)
          pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (thv0(i, j, k) + thv0(i, j, k - 1)));
        for (int k = 1; k <= nk; ++k) prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
        if (imoist == 1)
          for (int k = 1; k <= nk; ++k)
            rh0(i, j, k) = qv0(i, j, k) / (rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k)));
      }

  } else if (isnd == 8) {
    // dry (or constant-RH), constant d(theta)/dz
    const Real th_sfc = 300.0, pi_sfc = 1.0;
    Real lapse = -0.0030581, rhval = 0.0;
    if (testcase == 6 || testcase == 10) lapse = 0.005;   // hurricane boundary layer
    if (testcase == 10) rhval = 0.90;
    if (testcase == 15) { lapse = 0.005; rhval = 0.80; }
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          th0(i, j, k) = th_sfc + lapse * zh(i, j, k);
          pi0(i, j, k) = pi_sfc - (g / (cp * lapse)) * std::log(th0(i, j, k) / th_sfc);
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          rh0(i, j, k) = rhval;
          qv0(i, j, k) = rhval * rslf(prs0(i, j, k), t0(i, j, k));
        }

  } else if (isnd == 9 || isnd == 13) {
    // 9: constant Brunt-Vaisala frequency; 13: Klemp (2011), three layers
    const bool klemp = (isnd == 13);
    const Real ns1 = 0.0001, ns2 = klemp ? 0.0004 : 0.0000, ns3 = 0.0001;
    const Real zl1 = klemp ? 2000.0 : 40000.0, zl2 = 3000.0;
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        Real th_sfc = 288.00, pi_sfc = 1.0, zsfc, ns;
        if (zh(i, j, 1) < zl1) {
          zsfc = 0.0;
          ns = ns1;
        } else {
          if (std::abs(ns1) < 1.0e-6) {
            pi_sfc = pi_sfc - g * zl1 / (cp * th_sfc);
          } else {
            pi_sfc = pi_sfc + g * g / (cp * ns1 * th_sfc) * (std::exp(-ns1 * zl1 / g) - 1.0);
            th_sfc = th_sfc * std::exp(ns1 * zl1 / g);
          }
          zsfc = zl1;
          ns = ns2;
        }
        for (int k = 1; k <= nk; ++k) {
          if (!klemp) {
            qv0(i, j, k) = 0.0;
            qc0(i, j, k) = 0.0;
            rh0(i, j, k) = 0.0;
            if (std::abs(ns) < 1.0e-6) {
              thv0(i, j, k) = th_sfc;
              pi0(i, j, k) = pi_sfc - g * (zh(i, j, k) - zsfc) / (cp * th_sfc);
            } else {
              thv0(i, j, k) = th_sfc * std::exp(ns * (zh(i, j, k) - zsfc) / g);
              pi0(i, j, k) = pi_sfc + g * g / (cp * ns * th_sfc) * (std::exp(-ns * (zh(i, j, k) - zsfc) / g) - 1.0);
            }
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
            th0(i, j, k) = thv0(i, j, k);
            for (int n = 1; n <= 20; ++n) {
              t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
              qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
              th0(i, j, k) = thv0(i, j, k) * (1.0 + qv0(i, j, k) + qc0(i, j, k)) / (1.0 + reps * qv0(i, j, k));
            }
          } else {
            if (std::abs(ns) < 1.0e-6) {
              th0(i, j, k) = th_sfc;
              pi0(i, j, k) = pi_sfc - g * (zh(i, j, k) - zsfc) / (cp * th_sfc);
            } else {
              th0(i, j, k) = th_sfc * std::exp(ns * (zh(i, j, k) - zsfc) / g);
              pi0(i, j, k) = pi_sfc + g * g / (cp * ns * th_sfc) * (std::exp(-ns * (zh(i, j, k) - zsfc) / g) - 1.0);
            }
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cpdrd);
            thv0(i, j, k) = th0(i, j, k);
            qv0(i, j, k) = 0.0;
            rh0(i, j, k) = 0.0;
            t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          }
          if (zh(i, j, k + 1) > zl1 && zsfc < 1.0) {
            if (std::abs(ns1) < 1.0e-6) {
              pi_sfc = pi_sfc - g * zl1 / (cp * th_sfc);
            } else {
              pi_sfc = pi_sfc + g * g / (cp * ns1 * th_sfc) * (std::exp(-ns1 * zl1 / g) - 1.0);
              th_sfc = th_sfc * std::exp(ns1 * zl1 / g);
            }
            zsfc = zl1;
            ns = ns2;
          } else if (klemp && zh(i, j, k + 1) > zl2 && zsfc < zl2) {
            // (the Fortran tests ns1 here, not ns2)
            if (std::abs(ns1) < 1.0e-6) {
              pi_sfc = pi_sfc - g * (zl2 - zl1) / (cp * th_sfc);
            } else {
              pi_sfc = pi_sfc + g * g / (cp * ns2 * th_sfc) * (std::exp(-ns2 * (zl2 - zl1) / g) - 1.0);
              th_sfc = th_sfc * std::exp(ns2 * (zl2 - zl1) / g);
            }
            zsfc = zl2;
            ns = ns3;
          }
        }
      }

  } else if (isnd == 10) {
    // moist, constant Brunt-Vaisala frequency (reversible moist microphysics)
    const int eqtset = param_eqtset(nl), iice = param_iice(nl);
    const GoddardT gt = goddard_t(ptype);
    const Real t00k = gt.t00k, t0k = gt.t0k, rt0 = gt.rt0, cpi = c.cpi;
    const Real qcval = 0.005;
    const Real ns1 = 0.0, ns2 = 4.0e-4, zl1 = 111700.0;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          rh0(i, j, k) = 1.0;   // saturated
          qc0(i, j, k) = qcval;
        }
    Real desdt = 0.0;   // carried from k = 1 (eqtset = 1 uses it at every level)
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        const Real th_sfc = 288.0, pi_sfc = 1.0;
        Real ns = ns1;
        const Real p_sfc = p00 * std::pow(pi_sfc, cp / rd);
        const Real t_sfc = th_sfc * pi_sfc;
        const Real qv_sfc = rh0(i, j, 1) * rslf(p_sfc, t_sfc);
        const Real ql_sfc = qcval;
        const Real thv_sfc = th_sfc * (1.0 + reps * qv_sfc) / (1.0 + qv_sfc + ql_sfc);
        Real thn = th_sfc, thlast = thn;
        // ---- k = 1 ----
        {
          const int k = 1;
          int n = 1;
          qv0(i, j, k) = qv_sfc;
          qc0(i, j, k) = ql_sfc;
          while (true) {
            th0(i, j, k) = thn;
            thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k) + qc0(i, j, k));
            pi0(i, j, k) = pi_sfc - g * zh(i, j, k) / (cp * 0.5 * (thv0(i, j, k) + thv_sfc));
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
            t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
            const Real qvs = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
            qv0(i, j, k) = qvs;
            qc0(i, j, k) = qcval;
            const Real qtm = qv_sfc + ql_sfc;
            const Real qtp = qv0(i, j, k) + qc0(i, j, k);
            const Real tavg = 0.5 * (t0(i, j, k) + t_sfc);
            const Real lhv = lv1 - lv2 * tavg;
            const Real qvavg = 0.5 * (qv_sfc + qv0(i, j, k));
            const Real qlavg = 0.5 * (ql_sfc + qc0(i, j, k));
            const Real qtavg = qvavg + qlavg;
            desdt = 17.67 * (273.15 - 29.65) / ((tavg - 29.65) * (tavg - 29.65));
            const Real drdt = 17.67 * (273.15 - 29.65) * qvavg / ((tavg - 29.65) * (tavg - 29.65));
            Real gamma;
            if (eqtset == 2) {
              gamma = g * (1.0 + qtavg) * (1.0 + lhv * qvavg / rd / tavg) / (cp + cpv * qvavg + cpl * qlavg + lhv * drdt);
            } else {
              gamma = g * (1.0 + qtavg) * (rd / (rd + rv * qvavg) + lhv * qvavg / rd / tavg)
                      / (cp + lhv * (1.0 + qvavg * reps) * qvavg * desdt);
            }
            const Real tp = t_sfc * std::exp(zh(i, j, k) * (((ns / g) + std::log((1.0 + qtp) / (1.0 + qtm)) / zh(i, j, k))
                                                                / (1.0 + tavg * drdt / (eps + qvavg))
                                                            - gamma / tavg));
            thn = tp / pi0(i, j, k);
            n = n + 1;
            if (std::abs(thn - thlast) > 0.0001 && n < 200) {
              thn = thlast + 0.3 * (thn - thlast);
              thlast = thn;
              continue;
            } else if (n >= 200) {
              stop("isnd = 10: stuck in loop (111)!");
            }
            break;
          }
          th0(i, j, k) = thn;
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
          qc0(i, j, k) = qcval;
          if (qc0(i, j, 1) < 0.0) stop("isnd = 10: calling stopcm1 (222)");
        }
        // ---- all other levels ----
        ns = ns1;
        for (int k = 2; k <= nk; ++k) {
          int tflag = 0;
          for (int niter = 1; niter <= 2; ++niter) {
            if (tflag >= 2) continue;
            Real delz, pim, thm, thvm, qtm, qvm, qlm, qim, tm;
            auto take = [&](int kk) {
              pim = pi0(i, j, kk);
              thm = th0(i, j, kk);
              thvm = thv0(i, j, kk);
              qtm = qv0(i, j, kk) + qc0(i, j, kk) + qi0(i, j, kk);
              qvm = qv0(i, j, kk);
              qlm = qc0(i, j, kk);
              qim = qi0(i, j, kk);
              tm = t0(i, j, kk);
            };
            if (zh(i, j, k) > zl1 && ns == ns1 && tflag == 0 && zh(i, j, k - 1) < zl1) {
              tflag = 1;
              delz = zl1 - zh(i, j, k - 1);
              take(k - 1);
            } else if (tflag == 1) {
              tflag = 0;
              ns = ns2;
              delz = zh(i, j, k) - zl1;
              take(k);
            } else {
              tflag = 2;
              delz = zh(i, j, k) - zh(i, j, k - 1);
              take(k - 1);
            }
            th0(i, j, k) = thm;
            qv0(i, j, k) = qvm;
            qc0(i, j, k) = qlm;
            qi0(i, j, k) = qim;
            thv0(i, j, k) = thvm;
            thlast = thm;
            int n = 0;
            thn = thlast;
            thlast = thn;
            Real tp;
            while (true) {
              th0(i, j, k) = thn;
              thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k) + qc0(i, j, k) + qi0(i, j, k));
              pi0(i, j, k) = pim - g * delz / (cp * 0.5 * (thv0(i, j, k) + thvm));
              prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
              t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
              Real qtp, tavg, qvavg, qlavg, qtavg, drdt, lhv, cpml;
              if (iice == 0) {
                qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
                qc0(i, j, k) = qcval;
                qtp = qv0(i, j, k) + qc0(i, j, k);
                const Real prsavg = 0.5 * (prs0(i, j, k - 1) + prs0(i, j, k));
                tavg = 0.5 * (t0(i, j, k) + tm);
                qvavg = rh0(i, j, k) * rslf(prsavg, tavg);
                qlavg = 0.5 * (qlm + qc0(i, j, k));
                qtavg = 0.5 * (qtm + qtp);
                drdt = 17.67 * (273.15 - 29.65) * qvavg / ((tavg - 29.65) * (tavg - 29.65));
                lhv = lv1 - lv2 * tavg;
                cpml = cp + cpv * qvavg + cpl * qlavg;
              } else {
                Real qvl = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
                Real qvi = rh0(i, j, k) * rsif(prs0(i, j, k), t0(i, j, k));
                Real fliq = std::max(std::min((t0(i, j, k) - t00k) * rt0, 1.0), 0.0);
                Real fice = 1.0 - fliq;
                qv0(i, j, k) = fliq * qvl + fice * qvi;
                qc0(i, j, k) = fliq * qcval;
                qi0(i, j, k) = fice * qcval;
                qtp = qv0(i, j, k) + qc0(i, j, k) + qi0(i, j, k);
                const Real prsavg = 0.5 * (prs0(i, j, k - 1) + prs0(i, j, k));
                tavg = 0.5 * (t0(i, j, k) + tm);
                qvl = rh0(i, j, k) * rslf(prsavg, tavg);
                qvi = rh0(i, j, k) * rsif(prsavg, tavg);
                fliq = std::max(std::min((tavg - t00k) * rt0, 1.0), 0.0);
                fice = 1.0 - fliq;
                qvavg = fliq * qvl + fice * qvi;
                qlavg = 0.5 * (qlm + qc0(i, j, k));
                qtavg = 0.5 * (qtm + qtp);
                drdt = fliq * 17.67 * (273.15 - 29.65) * qvl / ((tavg - 29.65) * (tavg - 29.65))
                       + fice * 21.8745584 * (273.15 - 7.66) * qvi / ((tavg - 7.66) * (tavg - 7.66));
                if (tavg > t00k && tavg < t0k) drdt = drdt + (qvl - qvi) * rt0;
                lhv = fliq * (lv1 - lv2 * tavg) + fice * (ls1 - ls2 * tavg);
                cpml = cp + cpv * qvavg + cpl * qlavg + cpi * (qtavg - qlavg - qvavg);
              }
              Real gamma;
              if (eqtset == 2) {
                gamma = g * (1.0 + qtavg) * (1.0 + lhv * qvavg / rd / tavg) / (cpml + lhv * drdt);
              } else {
                gamma = g * (1.0 + qtavg) * (rd / (rd + rv * qvavg) + lhv * qvavg / rd / tavg)
                        / (cp + lhv * (1.0 + qvavg * reps) * qvavg * desdt);
              }
              tp = tm * std::exp(delz * (((ns / g) + std::log((1.0 + qtp) / (1.0 + qtm)) / delz)
                                             / (1.0 + tavg * drdt / (eps + qvavg))
                                         - gamma / tavg));
              thn = tp / pi0(i, j, k);
              n = n + 1;
              if (std::abs(thn - thlast) > 0.0001 && n < 200) {
                thn = thlast + 0.3 * (thn - thlast);
                thlast = thn;
                continue;
              } else if (n >= 200) {
                stop("isnd = 10: stuck in loop (333)!");
              }
              break;
            }
            t0(i, j, k) = tp;
            th0(i, j, k) = thn;
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
            t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
            if (iice == 0) {
              qv0(i, j, k) = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
            } else {
              const Real qvl = rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k));
              const Real qvi = rh0(i, j, k) * rsif(prs0(i, j, k), t0(i, j, k));
              const Real fliq = std::max(std::min((t0(i, j, k) - t00k) * rt0, 1.0), 0.0);
              const Real fice = 1.0 - fliq;
              qv0(i, j, k) = fliq * qvl + fice * qvi;
              qc0(i, j, k) = fliq * qcval;
              qi0(i, j, k) = fice * qcval;
            }
            if (qc0(i, j, 1) < 0.0) stop("isnd = 10: calling stopcm1 (444)");
          }
        }
      }

  } else if (isnd == 11) {
    // constant theta-e, saturated (Bryan and Rotunno 2009)
    if (imoist == 0) stop("isnd = 11 requires imoist = 1");
    const Real zl1 = 15000.0;   // tropopause height
    const int ttype = 2;        // pseudoadiabatic
    const Real psurf = 101510.0;
    const Real tsk0 = nl.getr("tsk0", 299.28);
    auto getthe = [&](Real p, Real t, Real qv, Real qt) -> Real {
      Real the = 0.0;
      if (ttype == 1) {
        const Real pd = p / (1.0 + reps * qv), cpm = cp + cpl * qt, lhv = lv1 - lv2 * t;
        the = t * (std::pow(p00 / pd, rd / cpm)) * std::exp(lhv * qv / (cpm * t));
      } else if (ttype == 2) {
        the = t * (std::pow(p00 / p, 0.2854 * (1.0 - 0.28 * qv))) * std::exp(((3376.0 / t) - 2.54) * qv * (1.0 + 0.81 * qv));
      }
      return the;
    };
    // revthe: t is the first guess on input
    auto revthe = [&](Real the, Real p, Real qt, Real& t, Real& qv) {
      int n = 1;
      Real tlast = t, tinc = 0.0;
      while (true) {
        t = tlast + tinc;
        qv = rslf(p, t);
        Real thx = 0.0;
        if (ttype == 1) {
          const Real pd = p / (1.0 + reps * qv), cpm = cp + cpl * qt, lhv = lv1 - lv2 * t;
          thx = t * (std::pow(p00 / pd, rd / cpm)) * std::exp(lhv * qv / (cpm * t));
        } else if (ttype == 2) {
          thx = t * (std::pow(p00 / p, 0.2854 * (1.0 - 0.28 * qv))) * std::exp(((3376.0 / t) - 2.54) * qv * (1.0 + 0.81 * qv));
        }
        const Real diff = the - thx;
        if (std::abs(diff) > 0.0001 && n < 50) {
          n = n + 1;
          tinc = 0.30 * diff;
          tlast = t;
          if (std::abs(tinc) >= 0.0001) continue;
        } else if (n >= 50) {
          stop("isnd = 11: revthe: n exceeded 50!");
        }
        return;
      }
    };
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i) {
        Real pi_sfc = std::pow(psurf * rp00, rovcp);
        Real th_sfc = (tsk0 - 3.0) / pi_sfc;
        Real t_sfc = th_sfc * pi_sfc;
        const Real p_sfc = p00 * std::pow(pi_sfc, cp / rd);
        Real qv_sfc = rslf(p_sfc, t_sfc);
        const Real qt_sfc = qv_sfc;
        const Real thv_sfc = th_sfc * (1.0 + reps * qv_sfc) / (1.0 + qv_sfc);
        const Real the_sfc = getthe(p_sfc, t_sfc, qv_sfc, qt_sfc);
        th_sfc = 0.0;
        qv_sfc = 0.0;
        revthe(the_sfc, p_sfc, qt_sfc, t_sfc, qv_sfc);
        Real pi1 = pi_sfc, z1 = 0.0, t1 = t_sfc, thv1 = thv_sfc, thv2 = thv1;
        int flag = 0;
        Real zsfc = 0.0, pi2 = 0.0, p2 = 0.0, t2 = 0.0, qv2 = 0.0, th2 = 0.0, ql2 = 0.0;
        auto troposphere = [&](Real z2) {
          Real tlast = 0.0;
          const Real theq = the_sfc;
          while (std::abs(t2 - tlast) > 0.0001) {
            tlast = t2;
            pi2 = pi1 - g * (z2 - z1) / (cp * 0.5 * (thv1 + thv2));
            p2 = p00 * std::pow(pi2, cp / rd);
            revthe(theq, p2, qt_sfc, t2, qv2);
            th2 = t2 / pi2;
            if (ttype == 1) ql2 = qt_sfc - qv2;
            else if (ttype == 2) ql2 = 0.0;
            thv2 = th2 * (1.0 + reps * qv2) / (1.0 + qv2 + ql2);
          }
        };
        for (int k = 1; k <= nk; ++k) {
          Real z2 = zh(i, j, k);
          t2 = t1;
          if (z2 <= zl1) {
            troposphere(z2);
          } else {
            const Real ns = 4.0e-4;
            if (flag == 0) {
              // first time in this section: get the tropopause values
              z2 = zl1;
              t2 = t1;
              troposphere(z2);
              flag = 1;
              th_sfc = thv2;
              pi_sfc = pi2;
              zsfc = z2;
              z2 = zh(i, j, k);
              t2 = t1;
            }
            if (std::abs(ns) < 1.0e-6) {
              thv2 = th_sfc;
              pi2 = pi_sfc - g * (z2 - zsfc) / (cp * th_sfc);
            } else {
              thv2 = th_sfc * std::exp(ns * (z2 - zsfc) / g);
              pi2 = pi_sfc + g * g / (cp * ns * th_sfc) * (std::exp(-ns * (z2 - zsfc) / g) - 1.0);
            }
            p2 = p00 * std::pow(pi2, cpdrd);
            th2 = thv2;
            for (int n = 1; n <= 20; ++n) {
              t2 = th2 * pi2;
              qv2 = rslf(p2, t2);
              if (ttype == 1) ql2 = qt_sfc - qv2;
              else if (ttype == 2) ql2 = 0.0;
              th2 = thv2 * (1.0 + qv2 + ql2) / (1.0 + reps * qv2);
            }
          }
          t1 = t2;
          thv1 = thv2;
          z1 = z2;
          pi1 = pi2;
          pi0(i, j, k) = pi2;
          prs0(i, j, k) = p2;
          t0(i, j, k) = t2;
          th0(i, j, k) = th2;
          thv0(i, j, k) = thv2;
          rh0(i, j, k) = 1.0;
          qv0(i, j, k) = qv2;
          qc0(i, j, k) = ql2;
        }
      }

  } else if (isnd == 12) {
    // PBL simulation, dry
    const Real pi_sfc = 1.0, th_sfc = 300.0, zl1 = 960.0, lapse = 0.010;
    const Real pisfc = pi_sfc - g * zl1 / (cp * th_sfc);
    for (int k = 1; k <= nk; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          if (zh(i, j, k) <= zl1) {
            th0(i, j, k) = th_sfc;
            pi0(i, j, k) = pi_sfc - g * zh(i, j, k) / (cp * th_sfc);
          } else {
            th0(i, j, k) = th_sfc + lapse * (zh(i, j, k) - zl1);
            pi0(i, j, k) = pisfc - (g / (cp * lapse)) * std::log(th0(i, j, k) / th_sfc);
          }
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          rh0(i, j, k) = 0.0;
          qv0(i, j, k) = 0.0;
        }

  } else if (isnd == 14 || isnd == 18 || isnd == 22) {
    // 14: convective boundary layer (Sullivan and Patton 2011)
    // 18: Moeng and Sullivan (1994) PBL
    // 22: stable boundary layer (Beare et al 2006), one layer above zl1
    Real pi_sfc = 1.0, th_sfc = 300.0, lr1, lr2 = 0.0, zl1, zl2 = 0.0;
    if (isnd == 14) {
      lr1 = 0.08; lr2 = 0.003; zl1 = 974.0; zl2 = 1074.0;
    } else if (isnd == 18) {
      const Real delz = 60.0;
      lr1 = 8.0 / delz; lr2 = 0.003;
      zl1 = 465.0 - delz / 2.0;
      zl2 = 465.0 + delz / 2.0;
    } else {
      th_sfc = 265.0; zl1 = 100.0; lr1 = 0.01;
    }
    for (int k = 1; k <= nk; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie; ++i) {
          if (zh(i, j, k) <= zl1) {
            th0(i, j, k) = th_sfc;
            pi0(i, j, k) = pi_sfc - g * zh(i, j, k) / (cp * th_sfc);
          } else if (isnd == 22) {
            th0(i, j, k) = th_sfc + (zh(i, j, k) - zl1) * lr1;
            pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (th0(i, j, k) + th0(i, j, k - 1)));
          } else if (zh(i, j, k) <= zl2) {
            th0(i, j, k) = th_sfc + (zh(i, j, k) - zl1) * lr1;
            pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (th0(i, j, k) + th0(i, j, k - 1)));
          } else {
            th0(i, j, k) = th_sfc + (zl2 - zl1) * lr1 + (zh(i, j, k) - zl2) * lr2;
            pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (th0(i, j, k) + th0(i, j, k - 1)));
          }
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          thv0(i, j, k) = th0(i, j, k);
          rh0(i, j, k) = 0.0;
          qv0(i, j, k) = (isnd == 22) ? rh0(i, j, k) * rslf(prs0(i, j, k), t0(i, j, k)) : 0.0;
        }

  } else if (isnd == 15) {
    // DYCOMS-II RF01 (testcase 4) / RF02 (testcase 5) stratocumulus
    if (testcase < 4 || testcase > 5) stop("isnd = 15: testcase must be 4 or 5");
    const Real xlv = c.xlv;
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        Real th_sfc, qv_sfc, prs_sfc, z_inv;
        if (testcase == 4) {
          th_sfc = 289.0; qv_sfc = 0.009; prs_sfc = 101780.0; z_inv = 840.0;
        } else {
          th_sfc = 288.3; qv_sfc = 0.00945; prs_sfc = 101780.0; z_inv = 795.0;
        }
        const Real pi_sfc = std::pow(prs_sfc / p00, rd / cp);
        const Real thv_sfc = th_sfc * (1.0 + qv_sfc * reps) / (1.0 + qv_sfc);
        Real thvlast = thv_sfc, zlast = 0.0, pilast = pi_sfc;
        for (int k = 1; k <= nk; ++k) {
          Real thl, qt;
          if (zh(i, j, k) <= z_inv) {
            thl = th_sfc;
            qt = qv_sfc;
          } else if (testcase == 4) {
            thl = 297.5 + std::pow(zh(i, j, k) - z_inv, 1.0 / 3.0);
            qt = 0.0015;
            qt = qt / (1.0 - qt);   // specific humidity to mixing ratio
          } else {
            thl = 295.0 + std::pow(zh(i, j, k) - z_inv, 1.0 / 3.0);
            qt = 0.001 * (5.0 - 3.0 * (1.0 - std::exp((z_inv - zh(i, j, k)) / 500.0)));
          }
          Real thlast = 0.0;
          th0(i, j, k) = thl;
          qv0(i, j, k) = qt;
          qc0(i, j, k) = 0.0;
          int n = 0;
          while (std::abs(th0(i, j, k) - thlast) >= 0.0001) {
            thlast = th0(i, j, k);
            thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k) + qc0(i, j, k));
            pi0(i, j, k) = pilast - g * (zh(i, j, k) - zlast) / (cp * 0.5 * (thvlast + thv0(i, j, k)));
            prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
            t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
            const Real qvs = rslf(prs0(i, j, k), t0(i, j, k));
            if (qt < qvs) {
              qv0(i, j, k) = qt;
              qc0(i, j, k) = 0.0;
              th0(i, j, k) = thl;
            } else {
              qv0(i, j, k) = qvs;
              qc0(i, j, k) = qt - qvs;
              th0(i, j, k) = thl / (1.0 - xlv * qc0(i, j, k) / (cp * t0(i, j, k)));
            }
            th0(i, j, k) = thlast + 0.3 * (th0(i, j, k) - thlast);
            n = n + 1;
            if (n > 100.0) stop("isnd = 15: no convergence (stop 1111)");
          }
          t0(i, j, k) = th0(i, j, k) * pi0(i, j, k);
          rho0(i, j, k) = prs0(i, j, k) / (rd * t0(i, j, k) * (1.0 + qv0(i, j, k) * reps));
          thvlast = thv0(i, j, k);
          pilast = pi0(i, j, k);
          zlast = zh(i, j, k);
        }
      }

  } else if (isnd == 19 || isnd == 20 || isnd == 23) {
    // piecewise-linear th and specific humidity, then hydrostatic pi
    //   19: Siebesma et al (2003) shallow Cu (testcase = 11: SAS convective PBL)
    //   20: RICO (VanZanten et al 2011)
    //   23: diurnal cycle of shallow Cu over land (Brown et al 2002)
    std::vector<Real> z, th, qv;
    Real prs_sfc;
    bool clip = false;   // isnd = 19: max(0, qv/(1-qv))
    if (isnd == 19) {
      clip = true;
      z = {0.0, 520.0, 1480.0, 2000.0, 3000.0};
      th = {298.7, 298.7, 302.4, 308.2, 311.85};
      qv = {0.0170, 0.0163, 0.0107, 0.0042, 0.0030};
      prs_sfc = 101500.0;
      if (testcase == 11) {
        z = {0.0, 352.5, 442.5, 3000.0, 20000.0};
        th = {296.6, 296.6, 298.1, 0.0, 0.0};
        th[3] = th[2] + 0.003 * (z[3] - z[2]);
        th[4] = th[2] + 0.003 * (z[4] - z[2]);
        qv = {11.8e-3, 11.8e-3, 7.8e-3, 0.0, 0.0};
        qv[3] = qv[2] - 0.004e-3 * (z[3] - z[2]);
        qv[4] = qv[2] - 0.004e-3 * (z[4] - z[2]);
        prs_sfc = 100000.0;
      }
    } else if (isnd == 20) {
      z = {0.0, 740.0, 3260.0, 4000.0};
      th = {297.9, 297.9, 312.6644, 317.0};
      qv = {0.0160, 0.0138, 0.0024, 0.0018};
      prs_sfc = 101540.0;
    } else {
      z = {0.0, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0, 10000.0};
      th = {299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2, 0.0};
      qv = {0.0152, 0.01517, 0.01498, 0.0148, 0.0147, 0.0135, 0.003, 0.003, 0.0};
      prs_sfc = 97000.0;
      th[8] = th[7] + (z[8] - z[7]) * (th[7] - th[6]) / (z[7] - z[6]);
      qv[8] = qv[7] + (z[8] - z[7]) * (qv[7] - qv[6]) / (z[7] - z[6]);
    }
    const int nz = int(z.size());
    const Real pi_sfc = std::pow(prs_sfc * rp00, rovcp);
    const Real thv_sfc = th[0] * (1.0 + reps * qv[0]) / (1.0 + qv[0]);
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        for (int k = kb; k <= ke; ++k) {
          for (int m = 1; m < nz; ++m) {
            if (zh(i, j, k) <= z[m]) {
              th0(i, j, k) = th[m - 1] + (th[m] - th[m - 1]) * (zh(i, j, k) - z[m - 1]) / (z[m] - z[m - 1]);
              qv0(i, j, k) = qv[m - 1] + (qv[m] - qv[m - 1]) * (zh(i, j, k) - z[m - 1]) / (z[m] - z[m - 1]);
              break;
            }
          }
          // convert specific humidity to mixing ratio
          if (clip)
            qv0(i, j, k) = std::max(0.0, qv0(i, j, k) / (1.0 - qv0(i, j, k)));
          else
            qv0(i, j, k) = qv0(i, j, k) / (1.0 - qv0(i, j, k));
          thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k));
        }
        pi0(i, j, 1) = pi_sfc - g * zh(i, j, 1) / (cp * 0.5 * (thv_sfc + thv0(i, j, 1)));
        for (int k = 2; k <= nk; ++k)
          pi0(i, j, k) = pi0(i, j, k - 1) - g * (zh(i, j, k) - zh(i, j, k - 1)) / (cp * 0.5 * (thv0(i, j, k) + thv0(i, j, k - 1)));
        for (int k = 1; k <= nk; ++k) {
          prs0(i, j, k) = p00 * std::pow(pi0(i, j, k), cp / rd);
          rh0(i, j, k) = qv0(i, j, k) / (rslf(prs0(i, j, k), th0(i, j, k) * pi0(i, j, k)));
        }
      }

  } else if (isnd == 21) {
    // RCEMIP (Wing et al 2018)
    const Real ztrop = 15000.0, qsfc = 0.01865, qt = 1.0e-11;   // 10.0**(-11)
    const Real zq1 = 4000.0, zq2 = 7500.0;
    const Real tsfc = nl.getr("tsk0", 299.28);   // air temperature = SST
    const Real tvsfc = tsfc * (1.0 + 0.608 * qsfc);
    const Real psfc = 101480.0;
    const Real lr = 0.0067;
    const Real tvtrop = tvsfc - lr * ztrop;
    const Real ptrop = psfc * std::pow(tvtrop / tvsfc, g / (rd * lr));
    for (int j = jb; j <= je; ++j)
      for (int i = ib; i <= ie; ++i)
        for (int k = 1; k <= nk; ++k) {
          Real tv0;
          if (zh(i, j, k) < ztrop) {
            const Real r = zh(i, j, k) / zq2;
            qv0(i, j, k) = qsfc * std::exp(-zh(i, j, k) / zq1) * std::exp(-(r * r));
            tv0 = tvsfc - lr * zh(i, j, k);
            prs0(i, j, k) = psfc * (std::pow((tvsfc - lr * zh(i, j, k)) / tvsfc, g / (rd * lr)));
          } else {
            qv0(i, j, k) = qt;
            tv0 = tvtrop;
            prs0(i, j, k) = ptrop * std::exp(-(g * (zh(i, j, k) - ztrop) / (rd * tvtrop)));
          }
          t0(i, j, k) = tv0 / (1.0 + 0.608 * qv0(i, j, k));
          pi0(i, j, k) = std::pow(prs0(i, j, k) * rp00, rovcp);
          th0(i, j, k) = t0(i, j, k) / pi0(i, j, k);
          qv0(i, j, k) = qv0(i, j, k) / (1.0 - qv0(i, j, k));
          thv0(i, j, k) = th0(i, j, k) * (1.0 + reps * qv0(i, j, k)) / (1.0 + qv0(i, j, k));
        }

  } else {
    unsupported("isnd = " + std::to_string(isnd));
  }
}

//---------------------------------------------------------------------------
// base.F: iwnd
void base_wind(const Namelist& nl, const Constants& c, const Grid& gr, const InitInfo& info, BaseArrays& a) {
  using phys::pi;
  (void)c;
  (void)info;
  const auto& B = gr.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const int ib = B.ib, ie = B.ie, jb = B.jb, je = B.je, kb = B.kb, ke = B.ke;
  const int isnd = nl.geti("isnd", 1), iwnd = nl.geti("iwnd", 0), testcase = nl.geti("testcase", 0);
  auto& zh = a.zh;
  auto& u0 = a.u0;
  auto& v0 = a.v0;
  // isnd = 7: winds come from input_sounding (param.F sets iwnd = 0)
  if (isnd == 7) return;
  if (iwnd < 0 || iwnd > 11) unsupported("iwnd = " + std::to_string(iwnd));

  for (int k = 1; k <= nk; ++k)
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni + 1; ++i) u0(i, j, k) = 0.0;
  for (int k = 1; k <= nk; ++k)
    for (int j = 1; j <= nj + 1; ++j)
      for (int i = 1; i <= ni; ++i) v0(i, j, k) = 0.0;

  // shear layer: uconst1 below udep1, linear to uconst2 at udep2, uconst2
  // above (iwnd = 1 uses zu < udep1, iwnd = 9, 11 zu <= udep1)
  auto shear = [&](Real udep1, Real udep2, Real uconst1, Real uconst2, bool le) {
    for (int k = 1; k <= nk; ++k)
      for (int j = 0; j <= nj + 1; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (le ? (zu <= udep1) : (zu < udep1)) u0(i, j, k) = uconst1;
          else if (zu > udep1 && zu < udep2) u0(i, j, k) = (uconst2 - uconst1) * (zu - udep1) / (udep2 - udep1) + uconst1;
          else u0(i, j, k) = uconst2;
        }
  };

  if (iwnd == 0 || iwnd == 7) {
    // zero wind (base.F has no iwnd = 7 branch)
  } else if (iwnd == 1) {
    // RKW-type wind profile
    shear(0.0, 2500.0, 0.0, 10.0, false);
    for (int k = 1; k <= nk; ++k)
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 1; ++i) v0(i, j, k) = 0.0;
  } else if (iwnd == 2) {
    // Weisman-Klemp type supercell profile
    const Real udep1 = 2000.0, udep2 = 6000.0, umax1 = 7.0, umax2 = 31.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni + 1; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (zu <= udep1) {
            const Real angle = 90.0 * (zu / udep1) * (pi / 180.0);
            u0(i, j, k) = umax1 - umax1 * std::cos(angle);
          } else if (zu > udep1 && zu <= udep2) {
            u0(i, j, k) = umax1 + (zu - udep1) * (umax2 - umax1) / (udep2 - udep1);
          } else {
            u0(i, j, k) = umax2;
          }
        }
    const Real vmax1 = umax1;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni; ++i) {
          const Real zv = 0.5 * (zh(i, j - 1, k) + zh(i, j, k));
          if (zv <= udep1) {
            const Real angle = 90.0 * (zv / udep1) * (pi / 180.0);
            v0(i, j, k) = vmax1 * std::sin(angle);
          } else {
            v0(i, j, k) = vmax1;
          }
        }
  } else if (iwnd == 3) {
    // multi-cell type profile
    const Real udep1 = 0.0, udep2 = 7500.0, umax1 = -40.0 / pi, umax2 = 40.0 / pi + 40.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni + 1; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (zu <= udep2) u0(i, j, k) = umax1 + (zu - udep1) * (umax2 - umax1) / (udep2 - udep1);
          else u0(i, j, k) = umax2;
        }
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni; ++i) v0(i, j, k) = 40.0 / pi;
  } else if (iwnd == 4) {
    // multi-cell (Weisman and Klemp 1982)
    const Real umax1 = 35.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni + 1; ++i) u0(i, j, k) = umax1 * std::tanh(0.5 * (zh(i - 1, j, k) + zh(i, j, k)) / 3000.0);
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni; ++i) v0(i, j, k) = 0.;
  } else if (iwnd == 5) {
    // Doernbrack et al (2005)
    const Real umax = 15.0, udep1 = 4000.0, udep2 = 6000.0;
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni + 1; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          if (zu < udep1) {
            u0(i, j, k) = umax;
          } else if (zu < udep2) {
            const Real alpha = 0.25 * pi * (1.0 + std::cos(pi * (zu - udep1) / (udep2 - udep1)));
            u0(i, j, k) = umax * std::sin(alpha);
          } else {
            u0(i, j, k) = 0.0;
          }
        }
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj + 1; ++j)
        for (int i = 1; i <= ni; ++i) {
          const Real zv = 0.5 * (zh(i, j - 1, k) + zh(i, j, k));
          if (zv < udep1) {
            v0(i, j, k) = 0.0;
          } else if (zv < udep2) {
            const Real alpha = 0.25 * pi * (1.0 + std::cos(pi * (zv - udep1) / (udep2 - udep1)));
            v0(i, j, k) = umax * std::cos(alpha);
          } else {
            v0(i, j, k) = umax;
          }
        }
  } else if (iwnd == 6) {
    // constant wind
    Real uval = 10.0, vval = 0.0;   // default
    if (testcase == 1) { uval = 1.0; vval = 0.0; }          // Sullivan and Patton (2011)
    else if (testcase == 2) { uval = 15.0; vval = 0.0; }    // Moeng and Sullivan (1994)
    else if (testcase == 4) { uval = 7.0; vval = -5.5; }    // DYCOMS-II RF01
    else if (testcase == 9) { uval = 8.0; vval = 0.0; }     // Beare et al (2006)
    else if (testcase == 11) { uval = 2.0; vval = 0.0; }    // SAS
    else if (testcase == 12) { uval = 20.0; vval = 0.0; }   // Martinuzzi and Tropea (1993)
    else if (testcase == 14) { uval = 10.0; vval = 0.0; }   // Brown et al (2002)
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je; ++j)
        for (int i = ib; i <= ie + 1; ++i) u0(i, j, k) = uval;
    for (int k = kb; k <= ke; ++k)
      for (int j = jb; j <= je + 1; ++j)
        for (int i = ib; i <= ie; ++i) v0(i, j, k) = vval;
  } else if (iwnd == 8) {
    // hurricane boundary layer (Bryan et al 2017): gradient wind hurr_vg,
    // rotated by hurr_rotate (param.F: hurr_angle in radians)
    const Real hurr_vg = nl.getr("hurr_vg", 40.0);
    const double pi_dp = 4.0 * std::atan(1.0);
    const Real hurr_angle = double(nl.getr("hurr_rotate", 0.0)) * (pi_dp / 180.0);
    for (int k = 1; k <= nk; ++k) {
      const Real vg = hurr_vg;   // vg(k): constant with height (ug(k) = 0)
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          u0(i, j, k) = -vg * std::sin(-hurr_angle);
          v0(i, j, k) = vg * std::cos(-hurr_angle);
        }
    }
  } else if (iwnd == 9) {
    // Siebesma et al (2003) shallow Cu
    for (int k = 1; k <= nk; ++k) {
      for (int j = 0; j <= nj + 1; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          const Real udep1 = 700.0, udep2 = 3000.0, uconst1 = -8.75, uconst2 = -4.61;
          if (zu <= udep1) u0(i, j, k) = uconst1;
          else if (zu > udep1 && zu < udep2) u0(i, j, k) = (uconst2 - uconst1) * (zu - udep1) / (udep2 - udep1) + uconst1;
          else u0(i, j, k) = uconst2;
        }
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 1; ++i) v0(i, j, k) = 0.0;
    }
  } else if (iwnd == 10) {
    // Ackerman et al (2009) drizzling Sc
    for (int k = 1; k <= nk; ++k) {
      for (int j = 0; j <= nj + 1; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          u0(i, j, k) = 3.0 + 4.3 * zu / 1000.0;
        }
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 1; ++i) {
          const Real zv = 0.5 * (zh(i, j - 1, k) + zh(i, j, k));
          v0(i, j, k) = -9.0 + 5.6 * zv / 1000.0;
        }
    }
  } else if (iwnd == 11) {
    // RICO (VanZanten et al 2011)
    for (int k = 1; k <= nk; ++k) {
      for (int j = 0; j <= nj + 1; ++j)
        for (int i = 0; i <= ni + 2; ++i) {
          const Real zu = 0.5 * (zh(i - 1, j, k) + zh(i, j, k));
          const Real udep1 = 0.0, udep2 = 4000.0, uconst1 = -9.9, uconst2 = -1.9;
          if (zu <= udep1) u0(i, j, k) = uconst1;
          else if (zu > udep1 && zu < udep2) u0(i, j, k) = (uconst2 - uconst1) * (zu - udep1) / (udep2 - udep1) + uconst1;
          else u0(i, j, k) = uconst2;
        }
      for (int j = 0; j <= nj + 2; ++j)
        for (int i = 0; i <= ni + 1; ++i) v0(i, j, k) = -3.8;
    }
  }
  (void)shear;
}

} // namespace init
} // namespace cm1k
