// Component tests of the MYNN-EDMF boundary-layer port (physics/pbl_mynn)
// against the unmodified src/module_bl_mynn.F / module_bl_mynn_wrapper.F,
// through cm1f_bridge_mynn_pbl.F90 (turb.F's domynn block).
//
// Synthetic but physical columns on a stretched grid up to ~12 km: water and
// land; convective (superadiabatic surface layer, strong upward fluxes, so
// the mass-flux plumes fire), near-neutral and stable (inversion, downward
// heat flux); moist (near saturation at the boundary-layer top, so the
// subgrid clouds and the plume condensation are exercised) and dry; calm to
// strong winds. A run is the initialization call (nstep=0) and then several
// steps, with the MYNN state carried over and th/qv/qc/qi/u/v updated with
// the tendencies, as CM1 does; after every call every output is compared
// bit for bit.
#include "physics/pbl_mynn/cm1k_pbl_mynn.hpp"
#include "share/cm1k_constants.hpp"

#include <catch2/catch.hpp>

#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void cm1f_mynn_pbl(int ni, int nj, int nk, int nstep, int ipbl, int ptype, int edmf, int edmf_mom, double dt,
                   double dx, double rthr, const double* in3, double* st3, double* te3, double* s2);
}

using namespace cm1k;
namespace M = cm1k::pbl_mynn;

namespace {

using H3 = FArray<3, HostSpace>;

H3 hcopy(const FArray3& a) {
  H3 h(a.label() + "_h", {a.lb(0), a.lb(1), a.lb(2)}, {a.ub(0), a.ub(1), a.ub(2)});
  Kokkos::deep_copy(h.view(), a.view());
  return h;
}

struct Case {
  int ipbl = 4, edmf = 1, edmf_mom = 1, ptype = 5;
  bool moist = true;
  double dx = 1000., dt = 10.;
  std::size_t work = 256u << 20;
  std::string str() const {
    std::ostringstream s;
    s << "ipbl=" << ipbl << " edmf=" << edmf << " edmf_mom=" << edmf_mom << " ptype=" << ptype
      << " moist=" << moist << " dx=" << dx << " dt=" << dt << " work=" << work;
    return s.str();
  }
};

double qsat(double t, double p) {
  const double es = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
  return 0.622 * es / (p - es);
}

// paths reached, summed over cases
struct Coverage {
  int n_neg_maxmf = 0;
  double max_edmf_qc = 0.;
};

void run_case(const Case& c, unsigned seed, Coverage& cov) {
  INFO(c.str());
  const int ni = 6, nj = 4, nk = 40, nsteps = 4;
  Bounds b;
  b.ni = ni; b.nj = nj; b.nk = nk;
  b.ib = -1; b.ie = ni + 2; b.jb = -1; b.je = nj + 2; b.kb = 0; b.ke = nk + 1;   // with halos, like CM1
  const auto C = Constants::make(0);

  M::Options o;
  o.ipbl = c.ipbl; o.imoist = c.moist ? 1 : 0; o.ptype = c.ptype;
  o.bl_mynn_edmf = c.edmf; o.bl_mynn_edmf_mom = c.edmf_mom;
  REQUIRE_NOTHROW(o.check());

  M::Inputs in;
  auto a3 = [&](const char* n) { return FArray3(n, {1, 1, 1}, {ni, nj, nk}); };
  in.dz = a3("dz"); in.u = a3("u"); in.v = a3("v"); in.w = a3("w"); in.th = a3("th"); in.qv = a3("qv");
  in.qc = a3("qc"); in.qi = a3("qi"); in.p = a3("p"); in.exner = a3("exner"); in.rho = a3("rho");
  in.t3d = a3("t3d");
  const FArray3 in_list[12] = {in.dz, in.u, in.v, in.w, in.th, in.qv, in.qc, in.qi, in.p, in.exner, in.rho, in.t3d};
  M::Surface sfc;
  sfc.allocate(b);
  const FArray3 sfc_list[16] = {sfc.znt, sfc.xland, sfc.tsk, sfc.qsfc, sfc.psfc, sfc.ust, sfc.ch, sfc.hfx,
                                sfc.qfx, sfc.wspd, sfc.rmol, sfc.hpbl, sfc.kpbl, sfc.nupdraft, sfc.maxmf,
                                sfc.ktop_plume};
  M::State st;
  st.allocate(b);
  const FArray3 st_list[26] = {st.qke, st.qke_adv, st.sh3d, st.sm3d, st.tsq, st.qsq, st.cov, st.el_pbl,
                               st.qc_bl, st.qi_bl, st.cldfra_bl, st.edmf_a, st.edmf_w, st.edmf_qt, st.edmf_thl,
                               st.edmf_ent, st.edmf_qc, st.sub_thl3d, st.sub_sqv3d, st.det_thl3d, st.det_sqv3d,
                               st.dqke, st.qwt, st.qshear, st.qbuoy, st.qdiss};
  M::Tendencies td;
  td.allocate(b);
  const FArray3 td_list[9] = {td.upten, td.vpten, td.thpten, td.qvpten, td.qcpten, td.qipten, td.xkzh,
                              td.xkzm, td.dheat};

  // Fortran-side packed arrays (ni,nj,nk,n) / (ni,nj,n)
  const std::size_t n3 = std::size_t(ni) * nj * nk, n2 = std::size_t(ni) * nj;
  std::vector<double> fin(12 * n3), fst(26 * n3, 0.0), fte(9 * n3, 0.0), fs2(16 * n2, 0.0);
  auto I3 = [&](int i, int j, int k, int n) { return (i - 1) + ni * ((j - 1) + nj * ((k - 1) + nk * n)); };
  auto I2 = [&](int i, int j, int n) { return (i - 1) + ni * ((j - 1) + nj * n); };

  std::mt19937_64 gen(seed);
  auto U = [&](double a, double bb) { return std::uniform_real_distribution<double>(a, bb)(gen); };

  // ---- columns ----
  std::vector<double> dzv(nk + 1), zh(nk + 1);
  {
    double zf = 0;
    for (int k = 1; k <= nk; ++k) {
      dzv[k] = 20.0 * std::pow(1.11, k - 1);
      zh[k] = zf + 0.5 * dzv[k];
      zf += dzv[k];
    }
  }
  // host input values, (i,j,k) -> fin
  auto IN = [&](int n, int i, int j, int k) -> double& { return fin[I3(i, j, k, n)]; };
  auto S2 = [&](int n, int i, int j) -> double& { return fs2[I2(i, j, n)]; };
  for (int j = 1; j <= nj; ++j)
    for (int i = 1; i <= ni; ++i) {
      const int regime = (i + j) % 3;   // 0 convective, 1 near-neutral, 2 stable
      const bool water = (i % 2 == 0);
      const double hbl = regime == 0 ? U(800., 1800.) : regime == 1 ? U(300., 800.) : U(80., 250.);
      const double ths = 290.0 + U(-5., 8.) + (j == nj ? -20.0 : 0.0);
      const double psfc = 1.0e5 + U(-1500., 1500.);
      const double wind = (j == 1) ? U(0.2, 1.0) : (j == 2 ? U(3., 7.) : U(10., 20.));
      const double dir = U(0., 6.283185307179586);
      const double rhs = c.moist ? (regime == 0 ? U(0.75, 0.9) : U(0.5, 0.85)) : 0.0;
      double pif = std::pow(psfc / phys::p00, C.rovcp);   // exner at the face below
      for (int k = 1; k <= nk; ++k) {
        const double z = zh[k];
        double th;
        if (regime == 0) {
          // superadiabatic surface layer, mixed layer, inversion, stable aloft
          th = ths + (z < 100. ? 1.5 * (1.0 - z / 100.) : 0.0) + (z > hbl ? 3.0 + 0.004 * (z - hbl) : 0.0);
        } else if (regime == 1) {
          th = ths + (z > hbl ? 1.0 + 0.004 * (z - hbl) : 0.0005 * z);
        } else {
          th = ths + 6.0 * (1.0 - std::exp(-z / hbl)) + 0.004 * z;
        }
        th += U(-0.05, 0.05);
        // moisture: rh rising towards the boundary-layer top, dry aloft
        double rh = 0.0;
        if (c.moist) {
          rh = z < hbl ? rhs + (0.995 - rhs) * (z / hbl) : std::max(0.05, 0.6 * std::exp(-(z - hbl) / 2500.));
          if (z > 9000.) rh = 0.02;
        }
        // hydrostatic exner at the scalar level
        const double pih = pif - 0.5 * C.g * dzv[k] / (C.cp * th);
        const double t = th * pih;
        const double p = phys::p00 * std::pow(pih, C.cpdrd);
        const double qv = rh * qsat(t, p);
        pif = pih - 0.5 * C.g * dzv[k] / (C.cp * th);
        IN(0, i, j, k) = dzv[k];
        const double prof = std::log(1.0 + z / 0.1) / std::log(1.0 + 1000. / 0.1);
        IN(1, i, j, k) = wind * std::cos(dir) * std::min(1.2, prof) + U(-0.2, 0.2);
        IN(2, i, j, k) = wind * std::sin(dir) * std::min(1.2, prof) + U(-0.2, 0.2);
        IN(3, i, j, k) = U(-0.05, 0.05);
        IN(4, i, j, k) = th;
        IN(5, i, j, k) = qv;
        // a little condensate near the boundary-layer top and ice aloft
        IN(6, i, j, k) = (c.moist && std::fabs(z - hbl) < 150. && regime == 0) ? U(0., 2.e-4) : 0.0;
        IN(7, i, j, k) = (c.moist && z > 7000. && z < 9000.) ? U(0., 2.e-5) : 0.0;
        IN(8, i, j, k) = p;
        IN(9, i, j, k) = pih;
        IN(10, i, j, k) = p / (C.rd * t * (1.0 + 0.61 * qv));
        IN(11, i, j, k) = t;
      }
      const double t1 = IN(11, i, j, 1);
      const double dts = regime == 0 ? U(2., 8.) : regime == 1 ? U(-0.3, 0.3) : U(-6., -1.);
      const double tsk = t1 + dts + 0.3;
      const double ust = regime == 2 ? U(0.05, 0.25) : std::max(0.1, 0.05 * wind + U(0., 0.2));
      const double hfx = regime == 0 ? U(150., 450.) : regime == 1 ? U(-5., 20.) : U(-60., -5.);
      const double qfx = c.moist ? (water ? U(5.e-5, 2.e-4) : U(0., 8.e-5)) : 0.0;
      const double rho1 = IN(10, i, j, 1);
      S2(0, i, j) = water ? 1.0e-4 : U(0.01, 0.5);                // znt
      S2(1, i, j) = water ? 2.0 : 1.0;                             // xland
      S2(2, i, j) = tsk;                                           // tsk
      S2(3, i, j) = c.moist ? qsat(tsk, psfc) : 0.0;               // qsfc
      S2(4, i, j) = psfc;                                          // psfc
      S2(5, i, j) = ust;                                           // ust
      S2(6, i, j) = U(0.002, 0.02);                                // ch
      S2(7, i, j) = hfx;                                           // hfx
      S2(8, i, j) = qfx;                                           // qfx
      S2(9, i, j) = std::max(1.0, std::hypot(IN(1, i, j, 1), IN(2, i, j, 1)));   // wspd
      S2(10, i, j) = -0.4 * C.g * hfx / (rho1 * C.cp * IN(4, i, j, 1) * ust * ust * ust);   // rmol
      S2(11, i, j) = hbl;                                          // hpbl
      // kpbl, nupdraft, maxmf, ktop_plume: 0 (CM1's initial values)
    }

  // upload the inputs/state (Fortran side keeps its own copies)
  auto upload_inputs = [&]() {
    for (int n = 0; n < 12; ++n) {
      H3 h = hcopy(in_list[n]);
      for (int k = 1; k <= nk; ++k)
        for (int j = 1; j <= nj; ++j)
          for (int i = 1; i <= ni; ++i) h(i, j, k) = fin[I3(i, j, k, n)];
      in_list[n].deep_copy_from(h);
    }
  };
  for (int n = 0; n < 16; ++n) {
    H3 h = hcopy(sfc_list[n]);
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) h(i, j, 1) = fs2[I2(i, j, n)];
    sfc_list[n].deep_copy_from(h);
  }
  // sentinels in the halos / unused levels: the port must not touch them
  const double sentinel = -12345.0;
  for (int n = 0; n < 26; ++n) st_list[n].fill(sentinel);
  for (int n = 0; n < 9; ++n) td_list[n].fill(sentinel);
  for (int n = 0; n < 26; ++n) {
    H3 h = hcopy(st_list[n]);
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) h(i, j, k) = 0.0;
    st_list[n].deep_copy_from(h);
  }

  const double rthr = -2.0e-5;
  int max_nup = 0;
  double max_cld = 0., max_edmf_a = 0.;
  for (int nstep = 0; nstep <= nsteps; ++nstep) {
    INFO("nstep " << nstep);
    upload_inputs();
    M::run(o, ni, nj, nk, c.dt, c.dx, nstep, in, sfc, st, td, c.work);
    cm1f_mynn_pbl(ni, nj, nk, nstep, c.ipbl, c.ptype, c.edmf, c.edmf_mom, c.dt, c.dx, rthr, fin.data(),
                  fst.data(), fte.data(), fs2.data());

    auto compare3 = [&](const FArray3& a, const std::vector<double>& ref, int n) {
      const H3 h = hcopy(a);
      std::size_t nbad = 0, nhalo = 0;
      double maxd = 0;
      int bi = 0, bj = 0, bk = 0;
      for (int k = a.lb(2); k <= a.ub(2); ++k)
        for (int j = a.lb(1); j <= a.ub(1); ++j)
          for (int i = a.lb(0); i <= a.ub(0); ++i) {
            const double x = h(i, j, k);
            if (i < 1 || i > ni || j < 1 || j > nj || k < 1 || k > nk) {
              if (!(x == sentinel)) ++nhalo;
              continue;
            }
            const double y = ref[I3(i, j, k, n)];
            if (!(x == y) && !(std::isnan(x) && std::isnan(y))) {
              if (nbad == 0) { bi = i; bj = j; bk = k; }
              ++nbad;
              maxd = std::max(maxd, std::fabs(x - y));
            }
          }
      INFO(a.label() << ": " << nbad << " differing points (first at " << bi << "," << bj << "," << bk
                     << "), max |diff| = " << maxd << "; halo points touched: " << nhalo);
      REQUIRE(nbad == 0);
      REQUIRE(nhalo == 0);
    };
    for (int n = 0; n < 26; ++n) compare3(st_list[n], fst, n);
    for (int n = 0; n < 9; ++n) compare3(td_list[n], fte, n);
    for (int n = 0; n < 16; ++n) {
      const H3 h = hcopy(sfc_list[n]);
      std::size_t nbad = 0;
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) {
          const double x = h(i, j, 1), y = fs2[I2(i, j, n)];
          if (!(x == y)) ++nbad;
        }
      INFO(sfc_list[n].label() << ": " << nbad << " differing points");
      REQUIRE(nbad == 0);
    }

    // sanity: finite, and the paths of interest are reached
    for (std::size_t m = 0; m < fst.size(); ++m) REQUIRE(std::isfinite(fst[m]));
    for (std::size_t m = 0; m < fte.size(); ++m) REQUIRE(std::isfinite(fte[m]));
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        max_nup = std::max(max_nup, int(fs2[I2(i, j, 13)]));
        if (fs2[I2(i, j, 14)] < 0.0) ++cov.n_neg_maxmf;   // plumes without condensate
      }
    for (std::size_t m = 0; m < n3; ++m) {
      max_cld = std::max(max_cld, fst[10 * n3 + m]);
      max_edmf_a = std::max(max_edmf_a, fst[11 * n3 + m]);
      cov.max_edmf_qc = std::max(cov.max_edmf_qc, fst[16 * n3 + m]);
    }

    // advance as CM1 does: apply the tendencies, carry the MYNN state
    // (qke_adv stays as the scheme left it, plus a little "advection")
    for (int k = 1; k <= nk; ++k)
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) {
          IN(1, i, j, k) += c.dt * fte[I3(i, j, k, 0)];
          IN(2, i, j, k) += c.dt * fte[I3(i, j, k, 1)];
          IN(4, i, j, k) += c.dt * fte[I3(i, j, k, 2)];
          IN(5, i, j, k) = std::max(0.0, IN(5, i, j, k) + c.dt * fte[I3(i, j, k, 3)]);
          IN(6, i, j, k) = std::max(0.0, IN(6, i, j, k) + c.dt * fte[I3(i, j, k, 4)]);
          IN(7, i, j, k) = std::max(0.0, IN(7, i, j, k) + c.dt * fte[I3(i, j, k, 5)]);
          IN(11, i, j, k) = IN(4, i, j, k) * IN(9, i, j, k);
          double& qa = fst[I3(i, j, k, 1)];
          qa = std::max(0.0, qa * (1.0 + 0.01 * std::sin(0.7 * i + 0.3 * j + 0.1 * k + nstep)));
        }
    {
      H3 h = hcopy(st.qke_adv);
      for (int k = 1; k <= nk; ++k)
        for (int j = 1; j <= nj; ++j)
          for (int i = 1; i <= ni; ++i) h(i, j, k) = fst[I3(i, j, k, 1)];
      st.qke_adv.deep_copy_from(h);
    }
    // the surface fluxes drift
    for (int j = 1; j <= nj; ++j)
      for (int i = 1; i <= ni; ++i) {
        S2(7, i, j) *= U(0.9, 1.1);
        S2(8, i, j) *= U(0.9, 1.1);
        S2(5, i, j) *= U(0.95, 1.05);
      }
    for (int n : {5, 7, 8}) {
      H3 h = hcopy(sfc_list[n]);
      for (int j = 1; j <= nj; ++j)
        for (int i = 1; i <= ni; ++i) h(i, j, 1) = fs2[I2(i, j, n)];
      sfc_list[n].deep_copy_from(h);
    }
  }
  if (c.edmf == 1) CHECK(max_nup > 0);
  if (c.edmf == 1) CHECK(max_edmf_a > 0.0);
  if (c.moist) CHECK(max_cld > 0.0);
}

} // namespace

TEST_CASE("mynn pbl matches module_bl_mynn.F", "[bridge][pbl_mynn]") {
  unsigned seed = 11;
  Coverage cov;
  for (int ipbl : {4, 5}) {
    for (auto [edmf, mom] : {std::pair<int, int>{1, 1}, std::pair<int, int>{1, 0}, std::pair<int, int>{0, 0}}) {
      for (bool moist : {true, false}) {
        Case c;
        c.ipbl = ipbl; c.edmf = edmf; c.edmf_mom = mom; c.moist = moist;
        run_case(c, seed++, cov);
      }
    }
  }
  // scale awareness, other dt, ptype, and chunked columns
  {
    Case c;
    c.dx = 3000.; c.dt = 30.; c.ptype = 1;
    run_case(c, 101, cov);
  }
  {
    Case c;
    c.ipbl = 5; c.dx = 250.; c.dt = 2.;
    c.work = 3 * 160 * 42 * sizeof(double);   // 3 columns per chunk
    run_case(c, 102, cov);
  }
  CHECK(cov.max_edmf_qc > 0.0);   // condensation in the plumes
  CHECK(cov.n_neg_maxmf > 0);     // dry plumes (maxmf = -maxmf)
}

TEST_CASE("mynn pbl options: unsupported values throw", "[pbl_mynn]") {
  M::Options o;
  o.ipbl = 4;
  REQUIRE_NOTHROW(o.check());
  o.ipbl = 5;
  REQUIRE_NOTHROW(o.check());
  auto bad = [](auto set) {
    M::Options o;
    o.ipbl = 4;
    set(o);
    REQUIRE_THROWS(o.check());
  };
  bad([](M::Options& o) { o.bl_mynn_cloudpdf = 1; });
  bad([](M::Options& o) { o.icloud_bl = 0; });
  bad([](M::Options& o) { o.spp_pbl = 1; });
  bad([](M::Options& o) { o.bl_mynn_tkebudget = 0; });
  bad([](M::Options& o) { o.bl_mynn_mixlength = 2; });
  bad([](M::Options& o) { o.bl_mynn_edmf = 2; });
  bad([](M::Options& o) { o.bl_mynn_edmf_tke = 1; });
  bad([](M::Options& o) { o.bl_mynn_cloudmix = 0; });
  bad([](M::Options& o) { o.bl_mynn_mixscalars = 1; });
  bad([](M::Options& o) { o.bl_mynn_mixqt = 1; });
  bad([](M::Options& o) { o.bl_mynn_tkeadvect = false; });
  bad([](M::Options& o) { o.cm1setup = 4; });
  M::Options off;   // inactive: nothing to check
  off.ipbl = 1;
  off.bl_mynn_cloudpdf = 0;
  REQUIRE_NOTHROW(off.check());
}
