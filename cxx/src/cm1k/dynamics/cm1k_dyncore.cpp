#include "cm1k_dyncore.hpp"

#include "dynamics/cm1k_adv.hpp"
#include "init/cm1k_init.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cm1k {

namespace {
[[noreturn]] void unsupported(const std::string& what) {
  throw std::runtime_error("cm1k dyncore: unsupported option: " + what);
}
using Range = Kokkos::RangePolicy<ExeSpace, Kokkos::IndexType<int>>;
inline Range krange(int k1, int k2) { return Range(k1, k2 + 1); }

// Sum of doubles in quadruple precision, rounded to double: CM1 sums the
// per-level masses in real(kind=qp) (getset in init3d.F, apmasscon in solve2.F).
double qp_sum(const std::vector<double>& v) {
#if defined(__SIZEOF_FLOAT128__) && !defined(__CUDA_ARCH__)
  __float128 s = 0;
  for (double x : v) s = s + (__float128)x;
  return (double)s;
#else
  long double s = 0;
  for (double x : v) s = s + (long double)x;
  return (double)s;
#endif
}
} // namespace

//===========================================================================
DynConfig DynConfig::from_namelist(const Namelist& nl) {
  DynConfig c;
  c.cm1setup = nl.geti("cm1setup", 1);
  c.sgsmodel = nl.geti("sgsmodel", 1);
  c.ipbl = nl.geti("ipbl", 0);
  c.idiff = nl.geti("idiff", 0);
  c.difforder = nl.geti("difforder", 6);
  c.imoist = std::min(std::max(nl.geti("imoist", 1), 0), 1);
  c.ptype = nl.geti("ptype", 0);
  c.eqtset = std::min(std::max(nl.geti("eqtset", 2), 1), 2);
  c.psolver = nl.geti("psolver", 3);
  c.apmasscon = nl.geti("apmasscon", 1);
  c.irdamp = nl.geti("irdamp", 0);
  c.hrdamp = nl.geti("hrdamp", 0);
  c.icor = std::min(std::max(nl.geti("icor", 0), 0), 1);
  c.lspgrad = nl.geti("lspgrad", 0);
  c.testcase = nl.geti("testcase", 0);
  c.axisymm = nl.geti("axisymm", 0);
  c.cuparam = nl.geti("cuparam", 0);
  c.radopt = nl.geti("radopt", 0);
  c.iptra = nl.geti("iptra", 0);
  c.iprcl = nl.geti("iprcl", 0);
  c.iinit = nl.geti("iinit", 1);
  c.idiss = nl.geti("idiss", 0);
  c.bbc = nl.geti("bbc", 1);
  c.rterm = nl.geti("rterm", 0);
  c.adapt_dt = nl.geti("adapt_dt", 0);
  c.isfcflx = nl.geti("isfcflx", 0);
  c.sfcmodel = nl.geti("sfcmodel", 0);
  c.irbc = nl.geti("irbc", 4);
  c.wbc = nl.geti("wbc", 1); c.ebc = nl.geti("ebc", 1);
  c.sbc = nl.geti("sbc", 1); c.nbc = nl.geti("nbc", 1);
  c.roflux = nl.geti("roflux", 0); c.nudgeobc = nl.geti("nudgeobc", 0);
  c.terrain_flag = nl.getb("terrain_flag", false);
  c.do_ib = nl.getb("do_ib", false);
  c.do_lsnudge = nl.getb("do_lsnudge", false);
  c.do_recycle = nl.getb("do_recycle_w", false) || nl.getb("do_recycle_e", false) ||
                 nl.getb("do_recycle_s", false) || nl.getb("do_recycle_n", false);
  c.dtl = nl.getr("dtl", 0.0);
  c.alph = nl.getr("alph", 0.60);
  c.kdiv = nl.getr("kdiv", 0.10);
  c.rdalpha = nl.getr("rdalpha", 0.0);
  c.zd = nl.getr("zd", 0.0);
  c.xhd = nl.getr("xhd", 0.0);
  c.csound = nl.getr("csound", 300.0);
  c.hadvordrs = nl.geti("hadvordrs", 5);
  c.vadvordrs = nl.geti("vadvordrs", 5);

  // param.F adjustments
  c.idoles = (c.cm1setup == 1 || c.cm1setup == 4);
  if (!c.idoles && c.sgsmodel >= 1) c.sgsmodel = 0;
  if ((c.psolver >= 4 && c.psolver <= 7) && c.apmasscon == 1) c.apmasscon = 0;
  if (c.psolver >= 4 && c.psolver <= 7) c.eqtset = 1;
  c.dohturb = c.dovturb = (c.idoles || c.cm1setup == 3);
  if (c.cm1setup == 2 && nl.geti("horizturb", 0) == 1) c.dohturb = true;

  // moisture indices (param.F)
  c.numq = 0;
  if (c.imoist == 1) {
    if (c.ptype == 0) { c.numq = 1; c.nqv = 1; }   // water vapor only
    if (c.ptype == 1) { c.numq = 3; c.nqv = 1; c.nqc = 2; c.nql1 = 2; c.nql2 = 3; c.iice = 0; }
  }
  // subgrid turbulence (LES): only sgsmodel=1 is ported (physics/turb)
  c.turb = (c.idoles && c.sgsmodel >= 1);
  if (c.turb && c.sgsmodel == 1) c.turb_opt = turb::Options::from_namelist(nl);
  return c;
}

void DynConfig::check() const {
  if (psolver != 3) unsupported("psolver=" + std::to_string(psolver) + " (only 3)");
  if (cm1setup != 0 && cm1setup != 1) unsupported("cm1setup=" + std::to_string(cm1setup) + " (only 0,1)");
  if (sgsmodel != 0 && sgsmodel != 1) unsupported("sgsmodel=" + std::to_string(sgsmodel) + " (only 0, 1)");
  if (turb) turb_opt.check();   // throws for every turbulence option that is not ported
  if (ipbl != 0) unsupported("ipbl");
  if (idiff != 0) unsupported("idiff");
  if (imoist == 1 && ptype != 0 && ptype != 1)
    unsupported("ptype=" + std::to_string(ptype) + " with imoist=1 (only 0 = qv only, 1 = Kessler)");
  if (irdamp != 0 && irdamp != 1) unsupported("irdamp=" + std::to_string(irdamp));
  if (hrdamp != 0) unsupported("hrdamp");
  if (icor != 0) unsupported("icor");
  if (lspgrad != 0) unsupported("lspgrad");
  if (testcase != 0) unsupported("testcase");
  if (axisymm != 0) unsupported("axisymm");
  if (cuparam != 0) unsupported("cuparam");
  if (radopt != 0) unsupported("radopt");
  if (iptra != 0) unsupported("iptra");
  if (iprcl != 0) unsupported("iprcl");
  if (iinit == 10 || iinit == 11 || iinit == 12) unsupported("iinit=" + std::to_string(iinit) + " (forcing in sound/solve1)");
  // idiss=1 adds epsd/(cp*pi) with epsd from dissten, which only the subgrid
  // turbulence / PBL schemes set: with sgsmodel=0 and ipbl=0 it is exactly
  // zero and the term is a no-op; with sgsmodel=1 it is turb::dissheat
  // (bbc=3 adds a surface term: not ported).
  if (idiss != 0 && (ipbl != 0 || bbc == 3)) unsupported("idiss=1 with PBL or bbc=3 dissipation");
  if (rterm != 0) unsupported("rterm");
  if (adapt_dt != 0) unsupported("adapt_dt");
  if (isfcflx != 0 || sfcmodel != 0) unsupported("surface fluxes (isfcflx/sfcmodel)");
  if (terrain_flag) unsupported("terrain_flag");
  if (do_ib) unsupported("do_ib");
  if (do_lsnudge) unsupported("do_lsnudge");
  if (do_recycle) unsupported("eddy recycling");
  // lateral boundaries 1 (periodic), 2 (open-radiative, see cm1k_openbc),
  // 3, 4 (rigid walls), with the irbc/roflux/nudgeobc options of cm1k_openbc
  for (int v : {wbc, ebc, sbc, nbc})
    if (v < 1 || v > 4) unsupported("lateral boundary condition " + std::to_string(v) + " (only 1..4)");
  if ((wbc == 2 || ebc == 2 || sbc == 2 || nbc == 2) && irbc != 1 && irbc != 2 && irbc != 4)
    unsupported("irbc=" + std::to_string(irbc) + " with open boundaries (only 1, 2, 4)");
  if (roflux != 0 && roflux != 1) unsupported("roflux=" + std::to_string(roflux));
  if (nudgeobc != 0 && nudgeobc != 1) unsupported("nudgeobc=" + std::to_string(nudgeobc));
  if (dtl <= 0) throw std::runtime_error("cm1k dyncore: dtl must be > 0");
}

//===========================================================================
Dyncore::Dyncore() = default;
Dyncore::~Dyncore() = default;

void Dyncore::setup(const Namelist& nl, const Constants& c, Grid& grid, const BaseState& base, State& state) {
  m_cfg = DynConfig::from_namelist(nl);
  m_cfg.check();
  m_c = c;
  m_bc = BcParams::make(grid.b, nl);
  m_obc = OpenBcParams::make(grid, nl);
  m_obc.check();
  const auto& B = grid.b;
  if (B.nk < 3) throw std::runtime_error("cm1k dyncore: nz must be >= 3");

  // rdx/rdy/rdz are set by the Model constructor (param.F: rdx=1.0/dx)
  if (!(grid.rdx > 0 && grid.rdy > 0 && grid.rdz > 0))
    throw std::runtime_error("cm1k dyncore: grid spacings (rdx, rdy, rdz) are not set");
  m_g = grid; m_b = base; m_s = state;
  m_dt = m_cfg.dtl;

  auto s3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke}); };
  auto u3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie + 1, B.je, B.ke}); };
  auto v3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je + 1, B.ke}); };
  auto w3 = [&](const char* n) { return FArray3(n, {B.ib, B.jb, B.kb}, {B.ie, B.je, B.ke + 1}); };
  auto m3 = [&](const char* n) { return FArray3(n, {B.ibm, B.jbm, B.kbm}, {B.iem, B.jem, B.kem}); };
  u3d = u3("u3d"); v3d = v3("v3d"); w3d = w3("w3d");
  pp3d = s3("pp3d"); th3d = s3("th3d"); ppx = s3("ppx");
  q3d = FArray4("q3d", {B.ibm, B.jbm, B.kbm, 1}, {B.iem, B.jem, B.kem, B.numq});
  qten = FArray4("qten", {B.ibm, B.jbm, B.kbm, 1}, {B.iem, B.jem, B.kem, B.numq});
  rr = s3("rr"); rf = s3("rf"); divx = s3("divx");
  qpten = m3("qpten"); qtten = m3("qtten"); qvten = m3("qvten"); qcten = m3("qcten");
  tauh = s3("tauh"); taus = s3("taus"); tauf = w3("tauf");
  rru = u3("rru"); rrv = v3("rrv"); rrw = w3("rrw");
  uten = u3("uten"); vten = v3("vten"); wten = w3("wten");
  uten1 = u3("uten1"); vten1 = v3("vten1"); wten1 = w3("wten1");
  thten = s3("thten"); thten1 = s3("thten1"); ppten = s3("ppten"); sten = s3("sten");
  sadv = s3("sadv"); thterm = s3("thterm"); t11 = s3("t11"); t22 = s3("t22");
  dum1 = s3("dum1"); dum2 = s3("dum2"); dum3 = s3("dum3"); dum8 = s3("dum8");
  mm = s3("mm"); ppd = s3("ppd"); fpk = s3("fpk"); qk = s3("qk"); pk1 = s3("pk1"); pk2 = s3("pk2");
  dtu = FArray2("dtu", {B.ib, B.jb}, {B.ie, B.je}); dtv = FArray2("dtv", {B.ib, B.jb}, {B.ie, B.je});
  dtu0 = FArray2("dtu0", {B.ib, B.jb}, {B.ie, B.je}); dtv0 = FArray2("dtv0", {B.ib, B.jb}, {B.ie, B.je});
  dumk3 = FArray1("dumk3", {1}, {B.nk}); dumk4 = FArray1("dumk4", {1}, {B.nk});

  const int ni = B.ni, nj = B.nj, nk = B.nk;

  // param.F: dtu0/dtv0 (zero on non-periodic boundaries: ibw..ibn = 1)
  {
    auto hu = dtu0.to_host(), hv = dtv0.to_host();
    for (int j = B.jb; j <= B.je; ++j) for (int i = B.ib; i <= B.ie; ++i) { hu(i, j) = 0.0; hv(i, j) = 0.0; }
    for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni + 1; ++i) hu(i, j) = 1.0;
    if (m_obc.ibw == 1) for (int j = 1; j <= nj; ++j) hu(1, j) = 0.0;
    if (m_obc.ibe == 1) for (int j = 1; j <= nj; ++j) hu(ni + 1, j) = 0.0;
    for (int j = 1; j <= nj + 1; ++j) for (int i = 1; i <= ni; ++i) hv(i, j) = 1.0;
    if (m_obc.ibs == 1) for (int i = 1; i <= ni; ++i) hv(i, 1) = 0.0;
    if (m_obc.ibn == 1) for (int i = 1; i <= ni; ++i) hv(i, nj + 1) = 0.0;
    dtu0.deep_copy_from(hu); dtv0.deep_copy_from(hv);
  }

  // open boundaries: radiative phase speeds (cm1.F: radbcw = 0.0, ...)
  m_rad.allocate(B);
  if (m_obc.any_open() || m_obc.roflux == 1) {
    m_sobc.p = m_obc;
    m_sobc.rad = m_rad;
    m_sobc.uh = grid.uh; m_sobc.vh = grid.vh; m_sobc.ruh = grid.ruh; m_sobc.rvh = grid.rvh;
    m_sobc.rmh = grid.rmh; m_sobc.rho0 = base.rho0;
    m_sobc.work = FArray1("obc_temout", {1}, {B.nk});
    m_sobc.work2 = FArray1("obc_temin", {1}, {B.nk});
  }

  // param.F: min_dx, min_dy, min_dz, Rayleigh damping coefficients
  auto xf = grid.xf.to_host(), yf = grid.yf.to_host();
  auto zh = grid.zh.to_host(), zf = grid.zf.to_host();
  auto sigmaf = grid.sigmaf.to_host();
  Real min_dx = 1.0e20, min_dy = 1.0e20, min_dz = 1.0e20;
  for (int i = 1; i <= ni; ++i) min_dx = std::min(min_dx, xf(i + 1) - xf(i));
  for (int j = 1; j <= nj; ++j) min_dy = std::min(min_dy, yf(j + 1) - yf(j));
  for (int k = 1; k <= nk; ++k) for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i)
    min_dz = std::min(min_dz, zf(i, j, k + 1) - zf(i, j, k));
  m_nsound = sound_nsound(m_cfg.psolver, B.nx, B.ny, min_dx, min_dy, min_dz, m_cfg.csound, double(m_dt));

  {
    auto th = tauh.to_host(), ts = taus.to_host(), tf = tauf.to_host();
    for (size_t n = 0; n < th.size(); ++n) { th.data()[n] = 0.0; ts.data()[n] = 0.0; }
    for (size_t n = 0; n < tf.size(); ++n) tf.data()[n] = 0.0;
    const Real maxz = sigmaf(nk + 1);
    const Real zd = m_cfg.zd;
    if (m_cfg.irdamp >= 1 && zd < maxz) {
      if (zd < 0.5 * maxz)
        throw std::runtime_error("cm1k dyncore: Rayleigh damping would be applied over more than half the domain (zd < maxz/2)");
      for (int j = B.jb; j <= B.je; ++j) for (int i = B.ib; i <= B.ie; ++i) {
        for (int k = 1; k <= nk; ++k)
          if (zh(i, j, k) > zd) {
            th(i, j, k) = 0.5 * (1.0 - std::cos(phys::pi * (zh(i, j, k) - zd) / (zf(i, j, nk + 1) - zd)));
            ts(i, j, k) = th(i, j, k);
          }
        for (int k = 1; k <= nk + 1; ++k)
          if (zf(i, j, k) > zd)
            tf(i, j, k) = 0.5 * (1.0 - std::cos(phys::pi * (zf(i, j, k) - zd) / (zf(i, j, nk + 1) - zd)));
      }
    }
    tauh.deep_copy_from(th); taus.deep_copy_from(ts); tauf.deep_copy_from(tf);
  }

  // 2nd-order extrapolation coefficients for rf at the surface and top (param.F)
  {
    const auto ex = init::extrap_coeffs(grid.sigma.to_host(), grid.sigmaf.to_host(), nk);
    m_cgs1 = ex.cgs1; m_cgs2 = ex.cgs2; m_cgs3 = ex.cgs3;
    m_cgt1 = ex.cgt1; m_cgt2 = ex.cgt2; m_cgt3 = ex.cgt3;
  }

  if (m_cfg.turb) {
    tke3d = FArray3("tke3d", {B.ibt, B.jbt, B.kbt}, {B.iet, B.jet, B.ket});
    tketen = FArray3("tketen", {B.ibt, B.jbt, B.kbt}, {B.iet, B.jet, B.ket});
    cvm = s3("cvm");
    ts.allocate(B);
    m_tw.allocate(B);
  }

  AdvOptions ao = AdvOptions::from_namelist(nl);
  ao.check();
  m_adv = std::make_unique<Advection>(m_g, m_b, ao);
  m_setup = true;
}

//===========================================================================
void Dyncore::start_from_state() {
  const auto& B = m_g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  auto& s = m_s;
  // CM1 with sgsmodel=0 still calls the subgrid turbulence routines (idoles),
  // but with km = kh = 0 every turbulent tendency is exactly zero (the
  // implicit vertical solve returns s unchanged, all fluxes are 0*x), so
  // they are skipped. Make sure that premise holds.
  if ((m_cfg.dohturb || m_cfg.dovturb) && !m_cfg.turb) {
    for (auto* a : {&s.kmh, &s.kmv, &s.khh, &s.khv}) {
      auto h = a->to_host();
      for (size_t n = 0; n < h.size(); ++n)
        if (h.data()[n] != 0.0)
          unsupported("nonzero eddy viscosity " + a->label() + " (subgrid turbulence is not ported)");
    }
  }

  u3d.deep_copy_from(s.ua); bcu(m_bc, u3d);
  v3d.deep_copy_from(s.va); bcv(m_bc, v3d);
  w3d.deep_copy_from(s.wa); bcw(m_bc, w3d, 1);
  pp3d.deep_copy_from(s.ppi);
  th3d.deep_copy_from(s.tha); bcs(m_bc, th3d);
  if (m_cfg.imoist == 1) {
    q3d.deep_copy_from(s.qa);
    for (int n = 1; n <= B.numq; ++n) {
      auto q = Kokkos::subview(q3d.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1);
      bcs(m_bc, FArray3(q, {B.ibm, B.jbm, B.kbm}));
    }
  }
  // init3d.F (getset): ppx = ppi in the interior, then bcs
  ppx.fill(0.0);
  {
    auto ppx_ = ppx; auto ppi = s.ppi;
    Kokkos::parallel_for("ppx0", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { ppx_(i, j, k) = ppi(i, j, k); });
    bcs(m_bc, ppx);
  }
  // rr from rho (solve3)
  {
    auto rr_ = rr; auto rho = s.rho;
    Kokkos::parallel_for("rr0", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { rr_(i, j, k) = 1.0 / rho(i, j, k); });
  }
  compute_rf();
  qpten.fill(0.0); qtten.fill(0.0); qvten.fill(0.0); qcten.fill(0.0);

  if (m_cfg.turb) {
    // init3d.F: cm0, cme/ce1/ce2/csm; tke3d = tkea (the previous step's last
    // RK stage, whose halos bcw filled); tketen is set by solve1
    ts.init_constants(B);
    tke3d.deep_copy_from(s.tkea);
    turb::bcw(m_cfg.turb_opt, B, tke3d, 1);
    tketen.fill(0.0);
  }

  // init3d.F (getset): mass1 = total dry-air mass
  {
    auto rho = s.rho; auto ruh = m_g.ruh; auto rvh = m_g.rvh; auto rmh = m_g.rmh; auto d = dumk3;
    Kokkos::parallel_for("mass1", krange(1, nk), KOKKOS_LAMBDA(int k) {
      double sum = 0.0;
      for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i)
        sum = sum + rho(i, j, k) * ruh(i) * rvh(j) * rmh(i, j, k);
      d(k) = sum;
    });
    auto h = dumk3.to_host();
    std::vector<double> v(nk);
    for (int k = 1; k <= nk; ++k) v[k - 1] = h(k);
    m_mass1 = qp_sum(v);
  }
}

//===========================================================================
// solve3.F (and init3d.F getset): rf from rho on 0..ni+1, 0..nj+1
void Dyncore::compute_rf() {
  const auto& B = m_g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  auto rf_ = rf; auto rho = m_s.rho; auto c1 = m_g.c1; auto c2 = m_g.c2;
  const Real cgs1 = m_cgs1, cgs2 = m_cgs2, cgs3 = m_cgs3, cgt1 = m_cgt1, cgt2 = m_cgt2, cgt3 = m_cgt3;
  Kokkos::parallel_for("rf", box3(0, ni + 1, 0, nj + 1, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    rf_(i, j, k) = (c1(i, j, k) * rho(i, j, k - 1) + c2(i, j, k) * rho(i, j, k));
  });
  Kokkos::parallel_for("rf_bt", box2(0, ni + 1, 0, nj + 1), KOKKOS_LAMBDA(int i, int j) {
    rf_(i, j, 1) = cgs1 * rho(i, j, 1) + cgs2 * rho(i, j, 2) + cgs3 * rho(i, j, 3);
    rf_(i, j, nk + 1) = cgt1 * rho(i, j, nk) + cgt2 * rho(i, j, nk - 1) + cgt3 * rho(i, j, nk - 2);
  });
}

//===========================================================================
// cm1.F:3748 sfc_and_turb (LES path), on the 3d arrays as CM1 passes them
void Dyncore::sfc_and_turb(double rtime) {
  if (!m_cfg.turb) return;
  auto& s = m_s;
  turb::sfc_and_turb(m_cfg.turb_opt, m_c, m_g, m_b, Real(rtime), u3d, v3d, w3d, pp3d, th3d, q3d, s.prs,
                     s.rho, rf, s.tkea, tke3d, s.kmh, s.kmv, s.khh, s.khv, ts, m_tw);
}

//===========================================================================
void Dyncore::solve1() {
  openbc_solve1();   // radbc (irbc=1)

  const auto& B = m_g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const auto& cfg = m_cfg;
  const Constants c = m_c;
  auto ua = m_s.ua; auto va = m_s.va; auto wa = m_s.wa; auto tha = m_s.tha; auto ppi = m_s.ppi;
  auto rho = m_s.rho; auto qa = m_s.qa;
  auto u0 = m_b.u0; auto v0 = m_b.v0; auto th0 = m_b.th0; auto pi0 = m_b.pi0;
  auto uten1_ = uten1; auto vten1_ = vten1; auto wten1_ = wten1; auto thten1_ = thten1;
  auto tauh_ = tauh; auto taus_ = taus; auto tauf_ = tauf;

  //  Rayleigh damping and set tendency arrays
  if (cfg.irdamp == 1) {
    const Real rdalpha = cfg.rdalpha;
    Kokkos::parallel_for("rdamp", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      uten1_(i, j, k) = -rdalpha * 0.5 * (tauh_(i - 1, j, k) + tauh_(i, j, k)) * (ua(i, j, k) - u0(i, j, k));
      vten1_(i, j, k) = -rdalpha * 0.5 * (tauh_(i, j - 1, k) + tauh_(i, j, k)) * (va(i, j, k) - v0(i, j, k));
      wten1_(i, j, k) = -rdalpha * tauf_(i, j, k) * wa(i, j, k);
      thten1_(i, j, k) = -rdalpha * taus_(i, j, k) * tha(i, j, k);
    });
  } else {
    Kokkos::parallel_for("rdamp0", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      uten1_(i, j, k) = 0.0; vten1_(i, j, k) = 0.0; wten1_(i, j, k) = 0.0; thten1_(i, j, k) = 0.0;
    });
  }
  auto qten_ = qten;
  if (cfg.imoist == 1) {
    for (int n = 1; n <= cfg.numq; ++n)
      Kokkos::parallel_for("qten0", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { qten_(i, j, k, n) = 0.0; });
  }

  // subgrid turbulence (sgsmodel=0: exact no-op with km=kh=0; see
  // start_from_state)
  if (cfg.turb) {
    const auto& to = cfg.turb_opt;
    // RHS of the TKE equation (solve1.F ~315): sources, then diffusion
    turb::tke_source(to, B, ts.nm, ts.defv, ts.defh, ts.dissten, m_s.kmh, m_s.kmv, m_s.khv, tketen);
    turb::turbt(to, m_g, m_dt, rho, rf, m_s.tkea, tketen, m_s.kmh, m_s.kmv, m_tw);
    // dissipative heating (~919), before the turbulent fluxes of theta
    turb::dissheat(to, c, B, pi0, ppi, qa, ts.dissten, thten1, cvm);
    // doturb block (~1119)
    turb::turb_tendencies(to, m_g, m_b, m_dt, rho, rr, rf, ua, va, wa, tha, qa, uten1, vten1, wten1, thten1,
                          qten, m_s.kmh, m_s.kmv, m_s.khh, m_s.khv, ts, m_tw);
  }

  //  contribution to pressure tendency from potential temperature
  auto ppten_ = ppten;
  if (cfg.eqtset == 1) {
    Kokkos::parallel_for("ppten0", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { ppten_(i, j, k) = 0.0; });
  } else {
    const Real dt = m_dt;
    const Real rdt = 1.0 / dt;
    const Real tem = 0.0001 * cfg.tsmall;
    const Real rp00 = phys::rp00;
    if (cfg.imoist == 1) {
      const int nqv = cfg.nqv, nqc = cfg.nqc;
      const Real qsmall = cfg.qsmall;
      Kokkos::parallel_for("ppten_m", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        if (Kokkos::fabs(dt * thten1_(i, j, k)) > tem || Kokkos::fabs(dt * qten_(i, j, k, nqv)) > qsmall) {
          const Real thnew = tha(i, j, k) + dt * thten1_(i, j, k);
          const Real qvnew = qa(i, j, k, nqv) + dt * qten_(i, j, k, nqv);
          const Real pinew = Kokkos::pow(rho(i, j, k) * (th0(i, j, k) + thnew) * (c.rd + c.rv * qvnew) * rp00, c.rddcv) - pi0(i, j, k);
          ppten_(i, j, k) = (pinew - ppi(i, j, k)) * rdt;
        } else {
          ppten_(i, j, k) = 0.0;
        }
      });
      if (cfg.ptype >= 1) {
        auto qpten_ = qpten; auto qtten_ = qtten; auto qvten_ = qvten; auto qcten_ = qcten;
        Kokkos::parallel_for("ppten_diab", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          ppten_(i, j, k) = ppten_(i, j, k) + qpten_(i, j, k);
          thten1_(i, j, k) = thten1_(i, j, k) + qtten_(i, j, k);
          qten_(i, j, k, nqv) = qten_(i, j, k, nqv) + qvten_(i, j, k);
          qten_(i, j, k, nqc) = qten_(i, j, k, nqc) + qcten_(i, j, k);
        });
      }
    } else {
      Kokkos::parallel_for("ppten_d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        if (Kokkos::fabs(dt * thten1_(i, j, k)) > tem) {
          const Real thnew = tha(i, j, k) + dt * thten1_(i, j, k);
          const Real pinew = Kokkos::pow(rho(i, j, k) * (th0(i, j, k) + thnew) * c.rd * rp00, c.rddcv) - pi0(i, j, k);
          ppten_(i, j, k) = (pinew - ppi(i, j, k)) * rdt;
        } else {
          ppten_(i, j, k) = 0.0;
        }
      });
    }
  }
}

//===========================================================================
void Dyncore::solve2() {
  const auto& B = m_g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const auto& cfg = m_cfg;
  const Constants c = m_c;
  const Real dt = m_dt;
  const Real smeps = cfg.smeps;
  const Real rdx = m_g.rdx, rdy = m_g.rdy, rdz = m_g.rdz;
  constexpr int nrkmax = phys::nrkmax;

  auto& s = m_s;
  auto ua = s.ua; auto va = s.va; auto wa = s.wa; auto tha = s.tha; auto ppi = s.ppi; auto qa = s.qa;
  auto rho = s.rho; auto prs = s.prs;
  auto th0 = m_b.th0; auto pi0 = m_b.pi0; auto rth0 = m_b.rth0; auto qv0 = m_b.qv0; auto qc0 = m_b.qc0;
  auto qi0 = m_b.qi0; auto rho0 = m_b.rho0; auto rf0 = m_b.rf0;
  auto uh = m_g.uh; auto vh = m_g.vh; auto ruh = m_g.ruh; auto rvh = m_g.rvh;
  auto mh = m_g.mh; auto rmh = m_g.rmh; auto c1 = m_g.c1; auto c2 = m_g.c2;
  auto u3d_ = u3d; auto v3d_ = v3d; auto w3d_ = w3d; auto pp3d_ = pp3d; auto th3d_ = th3d; auto q3d_ = q3d;
  auto rru_ = rru; auto rrv_ = rrv; auto rrw_ = rrw; auto divx_ = divx;
  auto uten_ = uten; auto vten_ = vten; auto wten_ = wten;
  auto uten1_ = uten1; auto vten1_ = vten1; auto wten1_ = wten1;
  auto thten_ = thten; auto thten1_ = thten1; auto sadv_ = sadv; auto sten_ = sten; auto qten_ = qten;
  auto t11_ = t11; auto t22_ = t22; auto dum2_ = dum2; auto dum3_ = dum3; auto dum8_ = dum8; auto thterm_ = thterm;
  auto qpten_ = qpten; auto qtten_ = qtten; auto qvten_ = qvten; auto qcten_ = qcten;
  auto dumk3_ = dumk3; auto dumk4_ = dumk4;

  const AdvFluxes fl{rru, rrv, rrw, divx, rho, rr};

  for (int nrk = 1; nrk <= nrkmax; ++nrk) {
    const Real dttmp = dt / Real(nrkmax + 1 - nrk);

    //  misc. variables: dum8 = buoyancy, t11 = theta_rho, t22 = ppterm
    if (cfg.imoist == 1) {
      const int nql1 = cfg.nql1, nql2 = cfg.nql2, nqv = cfg.nqv;
      if (nql1 <= 0) {
        Kokkos::parallel_for("dum2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { dum2_(i, j, k) = 0.0; });
      } else {
        Kokkos::parallel_for("dum2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          dum2_(i, j, k) = q3d_(i, j, k, nql1);
          for (int n = nql1 + 1; n <= nql2; ++n) dum2_(i, j, k) = dum2_(i, j, k) + q3d_(i, j, k, n);
        });
      }
      if (cfg.iice == 1) unsupported("ice microphysics");
      Kokkos::parallel_for("dum3", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) { dum3_(i, j, k) = 0.0; });
      if (cfg.eqtset == 2) {
        Kokkos::parallel_for("buoy_m2", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          const Real qv = Kokkos::fmax(q3d_(i, j, k, nqv), 0.0);
          const Real qli = Kokkos::fmax(0.0, dum2_(i, j, k) + dum3_(i, j, k));
          const Real cpli = c.cpl * Kokkos::fmax(0.0, dum2_(i, j, k)) + c.cpi * Kokkos::fmax(0.0, dum3_(i, j, k));
          dum8_(i, j, k) = c.g * (th3d_(i, j, k) * rth0(i, j, k) + c.repsm1 * (qv - qv0(i, j, k)) - (qli - qc0(i, j, k) - qi0(i, j, k)));
          t11_(i, j, k) = (th0(i, j, k) + th3d_(i, j, k)) * (1.0 + c.reps * qv) / (1.0 + qv + qli);
          const Real cpm = c.cp + c.cpv * qv + cpli;
          const Real cvm = 1.0 / (c.cv + c.cvv * qv + cpli);
          thterm_(i, j, k) = (th0(i, j, k) + th3d_(i, j, k)) * (c.rd + c.rv * qv - c.rovcp * cpm) * cvm;
          t22_(i, j, k) = (pi0(i, j, k) + pp3d_(i, j, k)) * c.rovcp * cpm * cvm;
        });
      } else {
        Kokkos::parallel_for("buoy_m1", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          const Real qv = Kokkos::fmax(q3d_(i, j, k, nqv), 0.0);
          const Real qli = Kokkos::fmax(0.0, dum2_(i, j, k) + dum3_(i, j, k));
          dum8_(i, j, k) = c.g * (th3d_(i, j, k) * rth0(i, j, k) + c.repsm1 * (qv - qv0(i, j, k)) - (qli - qc0(i, j, k) - qi0(i, j, k)));
          t11_(i, j, k) = (th0(i, j, k) + th3d_(i, j, k)) * (1.0 + c.reps * qv) / (1.0 + qv + qli);
          t22_(i, j, k) = (pi0(i, j, k) + pp3d_(i, j, k)) * c.rddcv;
        });
      }
    } else {
      Kokkos::parallel_for("buoy_d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        dum8_(i, j, k) = c.g * th3d_(i, j, k) * rth0(i, j, k);
        t11_(i, j, k) = th0(i, j, k) + th3d_(i, j, k);
        t22_(i, j, k) = (pi0(i, j, k) + pp3d_(i, j, k)) * c.rddcv;
      });
    }
    bcp(m_bc, t11);

    //  Set RK tendency arrays
    Kokkos::parallel_for("ten_rk", box3(1, ni + 1, 1, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      uten_(i, j, k) = uten1_(i, j, k);
      vten_(i, j, k) = vten1_(i, j, k);
      wten_(i, j, k) = wten1_(i, j, k);
    });

    //  Get rru,rrv,rrw,divx (without terrain)
    Kokkos::parallel_for("rru", box3(0, ni + 2, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      rru_(i, j, k) = rho0(1, 1, k) * u3d_(i, j, k);
    });
    Kokkos::parallel_for("rrv", box3(0, ni + 1, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      rrv_(i, j, k) = rho0(1, 1, k) * v3d_(i, j, k);
    });
    Kokkos::parallel_for("rrw", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      if (k == 1) {
        rrw_(i, j, 1) = 0.0;
        rrw_(i, j, nk + 1) = 0.0;
      } else {
        rrw_(i, j, k) = rf0(1, 1, k) * w3d_(i, j, k);
      }
    });
    Kokkos::parallel_for("divx", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      divx_(i, j, k) = ((rru_(i + 1, j, k) - rru_(i, j, k)) * rdx * uh(i)
                       + (rrv_(i, j + 1, k) - rrv_(i, j, k)) * rdy * vh(j))
                       + (rrw_(i, j, k + 1) - rrw_(i, j, k)) * rdz * mh(1, 1, k);
      if (Kokkos::fabs(divx_(i, j, k)) < smeps) divx_(i, j, k) = 0.0;
    });

    //  open boundaries: nudge inflow u/v (solve2.F: after Coriolis, before advection)
    openbc_nudge();

    //  advection of u, v, w
    m_adv->advu(nrk, dttmp, fl, u3d, uten);
    m_adv->advv(nrk, dttmp, fl, v3d, vten);
    m_adv->advw(nrk, dttmp, fl, w3d, wten);

    //  Buoyancy
    Kokkos::parallel_for("buoy", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      wten_(i, j, k) = wten_(i, j, k) + (c1(i, j, k) * dum8_(i, j, k - 1) + c2(i, j, k) * dum8_(i, j, k));
    });

    //  Pressure equation
    Kokkos::parallel_for("sadv_pi", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real tem = pi0(1, 1, k);
      sadv_(i, j, k) = tem + pp3d_(i, j, k);
    });

    //  call sound
    {
      SoundArgs a;
      a.dt = dt; a.dttmp = dttmp; a.alph = cfg.alph; a.kdiv = cfg.kdiv; a.cp = c.cp;
      a.rdx = rdx; a.rdy = rdy; a.rdz = rdz; a.smeps = smeps;
      a.nsound = m_nsound; a.nrk = nrk; a.nrkmax = nrkmax;
      a.uh = m_g.uh; a.vh = m_g.vh; a.uf = m_g.uf; a.vf = m_g.vf;
      a.mh = m_g.mh; a.mf = m_g.mf; a.c1 = m_g.c1; a.c2 = m_g.c2; a.rr0 = m_b.rr0; a.rf0 = m_b.rf0;
      a.dtu = dtu; a.dtv = dtv; a.dtu0 = dtu0; a.dtv0 = dtv0;
      a.dum1 = dum1; a.dum2 = dum2; a.mm = mm; a.ppd = ppd; a.fpk = fpk; a.qk = qk; a.pk1 = pk1; a.pk2 = pk2;
      a.rru = rru; a.ua = ua; a.u3d = u3d; a.uten = uten;
      a.rrv = rrv; a.va = va; a.v3d = v3d; a.vten = vten;
      a.rrw = rrw; a.wa = wa; a.w3d = w3d; a.wten = wten;
      a.ppi = ppi; a.pp3d = pp3d; a.piadv = sadv; a.ppten = ppten; a.ppx = ppx;
      a.thv = t11; a.ppterm = t22;
      if (m_sobc.rad.radbcw.is_allocated()) a.obc = &m_sobc;
      sound(m_bc, a);
    }

    //  radbc (irbc=4)
    openbc_radbc4(dttmp);

    //  For Bryan-Fritsch equation set, compute 3d divergence (into t11),
    //  from the small-step-averaged velocities
    if (cfg.imoist == 1 && cfg.eqtset == 2) {
      Kokkos::parallel_for("getdiv", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        t11_(i, j, k) = ((rru_(i + 1, j, k) - rru_(i, j, k)) * rdx * uh(i)
                        + (rrv_(i, j + 1, k) - rrv_(i, j, k)) * rdy * vh(j))
                        + (rrw_(i, j, k + 1) - rrw_(i, j, k)) * rdz * mh(1, 1, k);
        if (Kokkos::fabs(t11_(i, j, k)) < smeps) t11_(i, j, k) = 0.0;
      });
    }

    //  getdivx: small-step-averaged mass fluxes and their divergence
    Kokkos::parallel_for("getdivx_uv", krange(1, nk), KOKKOS_LAMBDA(int k) {
      const Real tem = rho0(1, 1, k);
      for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni + 1; ++i) rru_(i, j, k) = rru_(i, j, k) * tem;
      for (int j = 1; j <= nj + 1; ++j) for (int i = 1; i <= ni; ++i) rrv_(i, j, k) = rrv_(i, j, k) * tem;
      if (k == 1) {
        for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i) { rrw_(i, j, 1) = 0.0; rrw_(i, j, nk + 1) = 0.0; }
      } else {
        const Real temf = rf0(1, 1, k);
        for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i) rrw_(i, j, k) = rrw_(i, j, k) * temf;
      }
    });
    Kokkos::parallel_for("getdivx", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      divx_(i, j, k) = ((rru_(i + 1, j, k) - rru_(i, j, k)) * rdx * uh(i)
                       + (rrv_(i, j + 1, k) - rrv_(i, j, k)) * rdy * vh(j))
                       + (rrw_(i, j, k + 1) - rrw_(i, j, k)) * rdz * mh(1, 1, k);
      if (Kokkos::fabs(divx_(i, j, k)) < smeps) divx_(i, j, k) = 0.0;
    });

    //  THETA-equation
    if (cfg.imoist == 1 && cfg.eqtset == 2) {
      Kokkos::parallel_for("thten_m", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        thten_(i, j, k) = thten1_(i, j, k) - t11_(i, j, k) * thterm_(i, j, k);
      });
    } else {
      Kokkos::parallel_for("thten_d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        thten_(i, j, k) = thten1_(i, j, k);
      });
    }
    Kokkos::parallel_for("sadv_th", box3(B.ib, B.ie, B.jb, B.je, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      const Real tem = th0(1, 1, k) - phys::th0r;
      sadv_(i, j, k) = tem + th3d_(i, j, k);
    });
    {
      AdvsArgs aa;
      aa.wflag = 1; aa.pdef = 0; aa.pdefweno = 0; aa.weps = 10.0 * phys::epsilon; aa.diffit = 0;
      aa.hadvorder = cfg.hadvordrs; aa.vadvorder = cfg.vadvordrs;
      m_adv->advs(nrk, aa, dttmp, fl, tha, sadv, thten);
    }
    Kokkos::parallel_for("th3d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      th3d_(i, j, k) = tha(i, j, k) + dttmp * thten_(i, j, k);
      if (Kokkos::fabs(th3d_(i, j, k)) < smeps) th3d_(i, j, k) = 0.0;
    });

    //  Moisture
    if (cfg.imoist == 1) {
      for (int n = 1; n <= cfg.numq; ++n) {
        Kokkos::parallel_for("sten_q", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          sten_(i, j, k) = qten_(i, j, k, n);
        });
        AdvsArgs aa;
        aa.wflag = 1; aa.pdef = (nrk == nrkmax) ? 1 : 0; aa.pdefweno = (n == cfg.nqv) ? 0 : 1;
        aa.weps = 1.0e-2 * phys::epsilon;   // qmag(n)*epsilon, qmag = 1.0e-2 for Kessler
        aa.diffit = 0; aa.hadvorder = cfg.hadvordrs; aa.vadvorder = cfg.vadvordrs;
        auto qan = FArray3(Kokkos::subview(qa.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1), {B.ibm, B.jbm, B.kbm});
        auto q3n = FArray3(Kokkos::subview(q3d.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1), {B.ibm, B.jbm, B.kbm});
        m_adv->advs(nrk, aa, dttmp, fl, qan, q3n, sten);
        Kokkos::parallel_for("q3d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          q3n(i, j, k) = qan(i, j, k) + dttmp * sten_(i, j, k);
          if (Kokkos::fabs(q3n(i, j, k)) < smeps) q3n(i, j, k) = 0.0;
        });
      }
    }

    //  Get pressure, density
    bool dopf = false;
    const Real p00 = phys::p00;
    if (cfg.imoist == 1) {
      const int nqv = cfg.nqv, nqc = cfg.nqc;
      if (nrk == nrkmax && cfg.eqtset == 2 && cfg.ptype >= 1) {
        Kokkos::parallel_for("diab_sub", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          pp3d_(i, j, k) = pp3d_(i, j, k) - dt * qpten_(i, j, k);
          qpten_(i, j, k) = pp3d_(i, j, k);
          th3d_(i, j, k) = th3d_(i, j, k) - dt * qtten_(i, j, k);
          qtten_(i, j, k) = th3d_(i, j, k);
          q3d_(i, j, k, nqv) = q3d_(i, j, k, nqv) - dt * qvten_(i, j, k);
          qvten_(i, j, k) = q3d_(i, j, k, nqv);
          q3d_(i, j, k, nqc) = q3d_(i, j, k, nqc) - dt * qcten_(i, j, k);
          qcten_(i, j, k) = q3d_(i, j, k, nqc);
        });
      }
      if (nrk == nrkmax) {
        Kokkos::parallel_for("prsrho_m", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          prs(i, j, k) = p00 * Kokkos::pow(pi0(i, j, k) + pp3d_(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k)
                       / ((th0(i, j, k) + th3d_(i, j, k)) * (pi0(i, j, k) + pp3d_(i, j, k))
                          * (c.rd + Kokkos::fmax(0.0, q3d_(i, j, k, nqv)) * c.rv));
        });
      }
    } else {
      if (nrk == nrkmax) {
        Kokkos::parallel_for("prsrho_d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          prs(i, j, k) = p00 * Kokkos::pow(pi0(i, j, k) + pp3d_(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k) / (c.rd * (th0(i, j, k) + th3d_(i, j, k)) * (pi0(i, j, k) + pp3d_(i, j, k)));
        });
      }
    }
    if (cfg.apmasscon == 1 && nrk == nrkmax) {
      // per-level sums in the Fortran loop order (k parallel, j/i serial)
      Kokkos::parallel_for("apmass", krange(1, nk), KOKKOS_LAMBDA(int k) {
        double d3 = 0.0, d4 = 0.0;
        for (int j = 1; j <= nj; ++j) for (int i = 1; i <= ni; ++i) {
          d3 = d3 + rho(i, j, k) * ruh(i) * rvh(j) * rmh(i, j, k);
          d4 = d4 + (pi0(i, j, k) + pp3d_(i, j, k));
        }
        dumk3_(k) = d3; dumk4_(k) = d4;
      });
      dopf = true;
    }

    //  bcs
    bcu(m_bc, u3d);
    bcv(m_bc, v3d);
    bcw(m_bc, w3d, 1);
    if (nrk < nrkmax) {
      bcp(m_bc, rho);
      bcp(m_bc, pp3d);
      bcs(m_bc, th3d);
      if (cfg.imoist == 1)
        for (int n = 1; n <= cfg.numq; ++n)
          bcs(m_bc, FArray3(Kokkos::subview(q3d.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1), {B.ibm, B.jbm, B.kbm}));
    }

    //  TKE (solve2.F ~1709): wten = tketen, advection of tke3d, step forward
    if (cfg.turb) {
      auto tketen_ = tketen;
      Kokkos::parallel_for("wten_tke", box3(1, ni, 1, nj, 2, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        wten_(i, j, k) = tketen_(i, j, k);
      });
      m_adv->advw(nrk, dttmp, fl, tke3d, wten, cfg.hadvordrs, cfg.vadvordrs, m_adv->options().advwenos);
      turb::tke_rk_update(cfg.turb_opt, B, dttmp, s.tkea, wten, tke3d);
    }

    //  Finish pressure, density (apmasscon)
    if (dopf) {
      auto h3 = dumk3.to_host(), h4 = dumk4.to_host();
      std::vector<double> v3(nk), v4(nk);
      for (int k = 1; k <= nk; ++k) { v3[k - 1] = h3(k); v4[k - 1] = h4(k); }
      m_mass2 = qp_sum(v3);
      const double p2 = qp_sum(v4);
      const Real tem = (std::pow(m_mass1 / m_mass2, c.rddcv) - 1.0) * p2 / double(long(B.nx) * long(B.ny) * long(B.nz));
      if (cfg.imoist == 1) {
        const int nqv = cfg.nqv;
        Kokkos::parallel_for("pf_m", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          pp3d_(i, j, k) = pp3d_(i, j, k) + tem;
          prs(i, j, k) = p00 * Kokkos::pow(pi0(i, j, k) + pp3d_(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k)
                       / ((th0(i, j, k) + th3d_(i, j, k)) * (pi0(i, j, k) + pp3d_(i, j, k))
                          * (c.rd + Kokkos::fmax(0.0, q3d_(i, j, k, nqv)) * c.rv));
        });
      } else {
        Kokkos::parallel_for("pf_d", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
          pp3d_(i, j, k) = pp3d_(i, j, k) + tem;
          prs(i, j, k) = p00 * Kokkos::pow(pi0(i, j, k) + pp3d_(i, j, k), c.cpdrd);
          rho(i, j, k) = prs(i, j, k) / (c.rd * (th0(i, j, k) + th3d_(i, j, k)) * (pi0(i, j, k) + pp3d_(i, j, k)));
        });
      }
    }
  } // rk loop
}

//===========================================================================
// Open-radiative lateral boundaries (cm1k_openbc)

// solve1.F: radbc (irbc=1), from the "a" velocities
void Dyncore::openbc_solve1() {
  if (m_obc.irbc != 1) return;
  if (m_obc.ibw == 1 || m_obc.ibe == 1) radbcew(m_obc, m_rad, m_s.ua);
  if (m_obc.ibs == 1 || m_obc.ibn == 1) radbcns(m_obc, m_rad, m_s.va);
}

// solve2.F (nudgeobc=1): nudge inflow u/v at open boundaries to u0/v0
void Dyncore::openbc_nudge() {
  nudgeobc_uv(m_obc, m_b.u0, m_b.v0, u3d, v3d, uten, vten);
}

// solve2.F: radbc (irbc=4), after the acoustic steps of each RK stage
void Dyncore::openbc_radbc4(Real dttmp) {
  if (m_obc.irbc != 4) return;
  if (m_obc.ibw == 1 || m_obc.ibe == 1) radbcew4(m_obc, m_g.ruf, m_rad, m_s.ua, u3d, dttmp);
  if (m_obc.ibs == 1 || m_obc.ibn == 1) radbcns4(m_obc, m_g.rvf, m_rad, m_s.va, v3d, dttmp);
}

//===========================================================================
void Dyncore::solve3() {
  const auto& B = m_g.b;
  const int ni = B.ni, nj = B.nj, nk = B.nk;
  const auto& cfg = m_cfg;
  auto& s = m_s;

  bcp(m_bc, pp3d);
  bcs(m_bc, th3d);
  if (cfg.imoist == 1)
    for (int n = 1; n <= cfg.numq; ++n)
      bcs(m_bc, FArray3(Kokkos::subview(q3d.view(), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, n - 1), {B.ibm, B.jbm, B.kbm}));

  auto pp3d_ = pp3d; auto ppx_ = ppx; auto th3d_ = th3d; auto q3d_ = q3d;
  auto u3d_ = u3d; auto v3d_ = v3d; auto w3d_ = w3d;
  Kokkos::parallel_for("ppx", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    ppx_(i, j, k) = pp3d_(i, j, k) + ppx_(i, j, k);
  });
  bcp(m_bc, ppx);
  bcp(m_bc, s.rho);

  //  diabatic tendencies for next timestep
  if (cfg.imoist == 1 && cfg.eqtset == 2 && cfg.ptype >= 1) {
    const Real rdt = 1.0 / m_dt;
    const int nqv = cfg.nqv, nqc = cfg.nqc;
    auto qpten_ = qpten; auto qtten_ = qtten; auto qvten_ = qvten; auto qcten_ = qcten;
    Kokkos::parallel_for("diab", box3(1, ni, 1, nj, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
      qpten_(i, j, k) = (pp3d_(i, j, k) - qpten_(i, j, k)) * rdt;
      qtten_(i, j, k) = (th3d_(i, j, k) - qtten_(i, j, k)) * rdt;
      qvten_(i, j, k) = (q3d_(i, j, k, nqv) - qvten_(i, j, k)) * rdt;
      qcten_(i, j, k) = (q3d_(i, j, k, nqc) - qcten_(i, j, k)) * rdt;
    });
  }

  //  Equate the two arrays
  auto ua = s.ua; auto va = s.va; auto wa = s.wa; auto ppi = s.ppi; auto tha = s.tha; auto qa = s.qa;
  Kokkos::parallel_for("equate_uvw", box3(0, ni + 2, 0, nj + 2, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    ua(i, j, k) = u3d_(i, j, k);
    va(i, j, k) = v3d_(i, j, k);
    wa(i, j, k) = w3d_(i, j, k);
  });
  Kokkos::parallel_for("equate_s", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    ppi(i, j, k) = pp3d_(i, j, k);
    tha(i, j, k) = th3d_(i, j, k);
  });
  if (cfg.imoist == 1) {
    for (int n = 1; n <= cfg.numq; ++n)
      Kokkos::parallel_for("equate_q", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
        qa(i, j, k, n) = q3d_(i, j, k, n);
      });
  }

  //  TKE (solve3.F ~563)
  if (cfg.turb) turb::tke_copy(B, tke3d, s.tkea);

  //  rr, rf
  auto rr_ = rr; auto rho = s.rho;
  Kokkos::parallel_for("rr", box3(0, ni + 1, 0, nj + 1, 1, nk), KOKKOS_LAMBDA(int i, int j, int k) {
    rr_(i, j, k) = 1.0 / rho(i, j, k);
  });
  compute_rf();
}

} // namespace cm1k
