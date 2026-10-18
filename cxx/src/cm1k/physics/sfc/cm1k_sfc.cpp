// Port of CM1's original surface layer (sfcmodel=1) and the surface
// arrays/initialization shared with sfcmodel=2,3. See cm1k_sfc.hpp.
//
// Every expression keeps the Fortran's operation order (cxx/README.md):
// x**2 is x*x, real**real is pow, alog is log, and Fortran's max/min of two
// non-NaN numbers is fmax2/fmin2.
#include "physics/sfc/cm1k_sfc.hpp"

#include "init/cm1k_cm1libs.hpp"

#include <stdexcept>
#include <string>

namespace cm1k::sfc {

namespace {

KOKKOS_INLINE_FUNCTION Real fmax2(Real a, Real b) { return a > b ? a : b; }
KOKKOS_INLINE_FUNCTION Real fmin2(Real a, Real b) { return a < b ? a : b; }

// sfcphys.F module parameters (cecd=3)
constexpr Real dcd1  = 1.0e-3;
constexpr Real dcd2  = 2.4e-3;
constexpr Real dwsp1 = 5.0;
constexpr Real dwsp2 = 25.0;
constexpr Real dfac  = (dcd2 - dcd1) / (dwsp2 - dwsp1);

constexpr int iswater = 16;   // getlanduse: water must be category 16

// run/LANDUSE.TBL (USGS): slmo and sfz0 for categories 1..33, summer then
// winter (getlanduse reads the table in default real, i.e. double here).
constexpr Real tbl_slmo[2][33] = {
  {.10, .30, .50, .25, .25, .35, .15, .10, .15, .15, .30, .30, .50, .30, .30, 1.0, .60,
   .35, .02, .50, .50, .50, .02, .95, .40, .50, .40, .02, .02, .10, .10, .10, .10},
  {.10, .60, .50, .50, .40, .60, .30, .20, .25, .15, .60, .60, .50, .60, .60, 1.0, .75,
   .70, .05, .60, .60, .60, .05, .95, .40, .40, .40, .02, .02, .10, .10, .10, .10}};
constexpr Real tbl_sfz0[2][33] = {
  {80., 15., 10., 15., 14., 20., 12., 5., 6., 15., 50., 50., 50., 50., 50., 0.01, 20.,
   40., 1., 10., 30., 15., 10., 0.1, 1., 15., 1., 80., 80., 80., 80., 80., 80.},
  {80., 5., 2., 5., 5., 20., 10., 1., 1., 15., 50., 50., 50., 50., 20., 0.01, 20.,
   40., 1., 10., 30., 15., 5., 0.1, 1., 15., 1., 80., 80., 80., 80., 80., 80.}};
// sfem (emissivity) and therin (thermal inertia x 100)
constexpr Real tbl_sfem[2][33] = {
  {.88, .985, .985, .985, .98, .985, .96, .93, .95, .92, .93, .94, .95, .95, .97, .98, .95,
   .95, .90, .92, .93, .92, .90, .95, .90, .95, .90, .88, .88, .88, .97, .97, .97},
  {.88, .92, .93, .92, .92, .93, .92, .93, .93, .92, .93, .93, .95, .95, .93, .98, .95,
   .95, .90, .92, .93, .92, .90, .95, .90, .95, .90, .88, .88, .88, .97, .97, .97}};
constexpr Real tbl_therin[2][33] = {
  {3., 4., 4., 4., 4., 4., 3., 3., 3., 3., 4., 4., 5., 4., 4., 6., 6.,
   5., 2., 5., 5., 5., 2., 5., 5., 6., 5., 3., 3., 3., 3., 3., 3.},
  {3., 4., 4., 4., 4., 4., 4., 4., 4., 3., 5., 5., 5., 5., 6., 6., 6.,
   6., 2., 5., 5., 5., 5., 5., 5., 5., 5., 3., 3., 3., 3., 3., 3.}};

void check_cat(int cat, int season) {
  if (cat < 1 || cat > 33 || season < 1 || season > 2)
    throw std::runtime_error("cm1k sfc: land-use category " + std::to_string(cat) + " / season " +
                             std::to_string(season) + " is not in LANDUSE.TBL");
}

} // namespace

Real landuse_slmo(int cat, int season) { check_cat(cat, season); return tbl_slmo[season - 1][cat - 1]; }
Real landuse_sfz0(int cat, int season) { check_cat(cat, season); return tbl_sfz0[season - 1][cat - 1]; }
Real landuse_sfem(int cat, int season) { check_cat(cat, season); return tbl_sfem[season - 1][cat - 1]; }
Real landuse_therin(int cat, int season) { check_cat(cat, season); return tbl_therin[season - 1][cat - 1]; }

//------------------------------------------------------------------------------
// Options
//------------------------------------------------------------------------------

Options Options::from_namelist(const Namelist& nl) {
  Options o;
  o.sfcmodel = nl.geti("sfcmodel", 0);
  o.isfcflx = nl.geti("isfcflx", 0);
  o.oceanmodel = nl.geti("oceanmodel", 1);
  o.bbc = nl.geti("bbc", 1);
  o.tbc = nl.geti("tbc", 1);
  o.cecd = nl.geti("cecd", 3);
  o.pertflx = nl.geti("pertflx", 0);
  o.set_flx = nl.geti("set_flx", 0);
  o.set_znt = nl.geti("set_znt", 0);
  o.set_ust = nl.geti("set_ust", 0);
  o.initsfc = nl.geti("initsfc", 1);
  o.lu0 = nl.geti("lu0", 16);
  o.season = nl.geti("season", 1);
  o.testcase = nl.geti("testcase", 0);
  o.imove = nl.geti("imove", 0);
  o.axisymm = nl.geti("axisymm", 0);
  o.terrain_flag = nl.getb("terrain_flag", false);
  o.imoist = nl.geti("imoist", 0);
  o.ptype = nl.geti("ptype", 0);
  o.cm1setup = nl.geti("cm1setup", 1);
  o.ipbl = nl.geti("ipbl", 0);
  o.tsk0 = nl.getr("tsk0", 299.28);
  o.tmn0 = nl.getr("tmn0", 297.28);
  o.output_sfcflx = nl.geti("output_sfcflx", 0);
  o.output_sfcflx = o.output_sfcflx < 0 ? 0 : (o.output_sfcflx > 1 ? 1 : o.output_sfcflx);
  o.xland0 = nl.getr("xland0", 2.0);
  o.cnstce = nl.getr("cnstce", 0.0011);
  o.cnstcd = nl.getr("cnstcd", 0.0011);
  o.cnst_shflx = nl.getr("cnst_shflx", 0.24);
  o.cnst_lhflx = nl.getr("cnst_lhflx", 5.2e-5);
  o.cnst_znt = nl.getr("cnst_znt", 0.16);
  o.cnst_ust = nl.getr("cnst_ust", 0.25);
  // param.F: sfcmodel = 2..7 requires isfcflx = 1
  if (o.sfcmodel >= 2 && o.sfcmodel <= 7) o.isfcflx = 1;
  // param.F: sfcmodel >= 1 (or a PBL scheme, or the ocean model) needs bbc=3
  if (o.sfcmodel >= 1 || o.oceanmodel == 2 || o.ipbl >= 1) o.bbc = 3;
  // init_surface.F: dosfcflx
  o.dosfcflx = false;
  if (o.cm1setup >= 1 || o.ipbl == 2) {
    if (o.isfcflx == 1) o.dosfcflx = true;
    if (o.sfcmodel == 2 || o.sfcmodel == 3 || o.sfcmodel == 4 || o.sfcmodel == 6 || o.sfcmodel == 7)
      o.dosfcflx = true;
  }
  if (o.ipbl == 1 || o.ipbl == 3 || o.ipbl == 4 || o.ipbl == 5) o.dosfcflx = false;
  if (o.cm1setup == 4 && o.isfcflx == 1) o.dosfcflx = true;
  return o;
}

void Options::check() const {
  auto fail = [](const std::string& m) { throw std::runtime_error("cm1k sfc: " + m); };
  auto is01 = [](int v) { return v == 0 || v == 1; };
  if (!((sfcmodel >= 0 && sfcmodel <= 3) || sfcmodel == 6))
    fail("sfcmodel=" + std::to_string(sfcmodel) + " is not ported (only 0, 1, 2, 3, 6)");
  if (sfcmodel == 0) {
    // param.F stops for these
    if (isfcflx != 0) fail("isfcflx=1 requires sfcmodel=1..7");
    if (bbc == 3) fail("bbc=3 requires sfcmodel>=1");
    return;
  }
  if (bbc != 3) fail("sfcmodel>=1 requires bbc=3");
  if (sfcmodel == 6) {
    // MYNN surface layer (physics/sfc_mynn, pbl_mynn::surface_layer): the
    // arrays and init_surface here; its own options in pbl_mynn::Coupling
    if (oceanmodel != 1) fail("oceanmodel=" + std::to_string(oceanmodel) + " is not ported (only 1)");
    if (pertflx != 0) fail("pertflx=1 requires sfcmodel=1 (param.F stops)");
    if (initsfc != 1) fail("initsfc=" + std::to_string(initsfc) + " is not ported (only 1)");
    if (season != 1 && season != 2) fail("season must be 1 or 2");
    if (lu0 < 0 || lu0 > 33) fail("lu0 must be 0..33 (LANDUSE.TBL)");
    if (tbc != 1) fail("tbc=" + std::to_string(tbc) + " is not ported with the surface layer (only 1)");
    if (testcase != 0) fail("testcase=" + std::to_string(testcase) + " is not ported (only 0)");
    if (imove != 0) fail("imove=1 (ground-relative winds with a moving domain) is not ported");
    if (axisymm != 0) fail("axisymm=1 is not ported");
    if (terrain_flag) fail("terrain is not ported");
    if (!(cm1setup == 2 && (ipbl == 4 || ipbl == 5)))
      fail("sfcmodel=6 is only ported with cm1setup=2 and the MYNN PBL (ipbl=4, 5)");
    if (imoist == 1 && ptype != 0 && ptype != 1)
      fail("imoist=1 with ptype=" + std::to_string(ptype) + " is not ported");
    if (tsk0 < 100.0) fail("tsk is less than 100 K");   // init_surface.F stops
    return;
  }
  if (oceanmodel != 1) fail("oceanmodel=" + std::to_string(oceanmodel) + " is not ported (only 1)");
  if (!is01(isfcflx)) fail("isfcflx must be 0 or 1");
  if (cecd < 1 || cecd > 3) fail("cecd=" + std::to_string(cecd) + " is not ported (1, 2, 3)");
  if (!is01(pertflx) || !is01(set_flx) || !is01(set_znt) || !is01(set_ust))
    fail("pertflx, set_flx, set_znt, set_ust must be 0 or 1");
  if (initsfc != 1) fail("initsfc=" + std::to_string(initsfc) + " is not ported (only 1)");
  if (season != 1 && season != 2) fail("season must be 1 or 2");
  if (lu0 < 0 || lu0 > 33) fail("lu0 must be 0..33 (LANDUSE.TBL)");
  if (tbc != 1) fail("tbc=" + std::to_string(tbc) + " is not ported with the surface layer (only 1)");
  if (testcase != 0) fail("testcase=" + std::to_string(testcase) + " is not ported (only 0)");
  if (imove != 0) fail("imove=1 (ground-relative winds with a moving domain) is not ported");
  if (axisymm != 0) fail("axisymm=1 is not ported");
  if (terrain_flag) fail("terrain is not ported");
  if (sfcmodel == 1) {
    // with the LES (cm1setup=1, ipbl=0) or with the simple PBL scheme
    // (cm1setup=2, ipbl=2: gethpbl2)
    if (!(cm1setup == 1 && ipbl == 0) && !(cm1setup == 2 && ipbl == 2))
      fail("sfcmodel=1 with cm1setup=" + std::to_string(cm1setup) + " and ipbl=" + std::to_string(ipbl) +
           " is not ported (only cm1setup=1 with ipbl=0, or cm1setup=2 with ipbl=2)");
  } else {
    // the WRF/MM5 surface layers with YSU (ipbl=1) or the simple PBL scheme
    if (!(cm1setup == 2 && (ipbl == 1 || ipbl == 2)))
      fail("sfcmodel=" + std::to_string(sfcmodel) + " with cm1setup=" + std::to_string(cm1setup) +
           " and ipbl=" + std::to_string(ipbl) + " is not ported (only cm1setup=2 with ipbl=1 or 2)");
    if (pertflx != 0) fail("pertflx=1 can only be used with sfcmodel=1");   // param.F stops
    if (set_flx != 0 || set_ust != 0) fail("set_flx/set_ust are not ported with sfcmodel=2,3");
    if (lu0 < 1 || lu0 > 33) fail("lu_index must be between 1 and 33 with sfcmodel=2,3");   // init_surface.F stops
    if (tmn0 < 100.0) fail("tmn is less than 100 K");   // init_surface.F stops
  }
  if (imoist == 1 && ptype != 0 && ptype != 1)
    fail("imoist=1 with ptype=" + std::to_string(ptype) + " is not ported");
  if (tsk0 < 100.0) fail("tsk is less than 100 K");   // init_surface.F stops
}

//------------------------------------------------------------------------------
// State
//------------------------------------------------------------------------------

void SfcState::allocate(const Bounds& b) {
  const std::array<int, 3> lo{b.ib, b.jb, 1}, hi{b.ie, b.je, 1};
  auto mk = [&](const char* n) { return FArray3(n, lo, hi); };
  tsk = mk("tsk"); znt = mk("znt"); xland = mk("xland"); mavail = mk("mavail");
  ust = mk("ust"); u1 = mk("u1"); v1 = mk("v1"); s1 = mk("s1");
  cd = mk("cd"); ch = mk("ch"); cq = mk("cq"); thflux = mk("thflux"); qvflux = mk("qvflux");
  psfc = mk("psfc"); u10 = mk("u10"); v10 = mk("v10"); s10 = mk("s10");
  hfx = mk("hfx"); qfx = mk("qfx"); qsfc = mk("qsfc"); wspd = mk("wspd");
  zntmp = mk("zntmp"); t1 = mk("t1"); hpbl = mk("hpbl"); dsxy = mk("dsxy");
  br = mk("br"); cda = mk("cda"); gz1oz0 = mk("gz1oz0"); th2 = mk("th2"); t2 = mk("t2"); q2 = mk("q2");
}

std::vector<std::pair<std::string, FArray3>> SfcState::dumped() const {
  return {{"tsk", tsk}, {"znt", znt}, {"xland", xland}, {"mavail", mavail}, {"ust", ust},
          {"u1", u1}, {"v1", v1}, {"s1", s1}, {"cd", cd}, {"ch", ch}, {"cq", cq},
          {"thflux", thflux}, {"qvflux", qvflux}, {"psfc", psfc}, {"u10", u10}, {"v10", v10},
          {"s10", s10}, {"hfx", hfx}, {"qfx", qfx}, {"qsfc", qsfc}, {"wspd", wspd}};
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

void set_defaults(const SfcState& s) {
  // cm1.F: every array = 0.0 at allocation (Kokkos zero-fills), then
  for (auto* a : {&s.znt, &s.zntmp, &s.ust}) a->fill(1.0e-6);
  s.tsk.fill(300.0);
  s.psfc.fill(100000.0);
  s.qsfc.fill(0.00001);
  s.hpbl.fill(100.0);
}

void set_dsxy(const Options& o, const Grid& g, const SfcState& s) {
  const auto& b = g.b;
  const Real dx = g.dx, dy = g.dy;
  const auto ruh = g.ruh;
  const auto rvh = g.rvh;
  const auto dsxy = s.dsxy;
  const int mode = (o.axisymm == 1 || b.ny == 1) ? 1 : (b.nx == 1 ? 2 : 0);
  Kokkos::parallel_for("sfc_dsxy", box2(b.ib, b.ie, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) {
    if (mode == 1)      dsxy(i, j, 1) = dx * ruh(i);
    else if (mode == 2) dsxy(i, j, 1) = dy * rvh(j);
    else                dsxy(i, j, 1) = fmax2(dx * ruh(i), dy * rvh(j));
  });
}

void init_surface(const Options& o, const Grid& g, const SfcState& s) {
  o.check();
  if (!o.has_layer()) return;
  const auto& b = g.b;
  // initsfc=1: tsk = tsk0, xland = xland0, lu_index = lu0; then getlanduse
  // (category 0 -> water) sets znt, mavail and xland from LANDUSE.TBL
  const int cat = (o.lu0 == 0) ? iswater : o.lu0;
  const Real tsk0 = o.tsk0;
  const Real z0 = landuse_sfz0(cat, o.season) / 100.;
  const Real slmo = landuse_slmo(cat, o.season);
  const Real xl = (cat != iswater) ? 1.0 : 2.0;
  const bool set_znt = (o.set_znt == 1);
  const Real cnst_znt = o.cnst_znt;
  const auto tsk = s.tsk, znt = s.znt, xland = s.xland, mavail = s.mavail;
  Kokkos::parallel_for("sfc_init", box2(b.ib, b.ie, b.jb, b.je), KOKKOS_LAMBDA(int i, int j) {
    tsk(i, j, 1) = tsk0;
    znt(i, j, 1) = z0;
    mavail(i, j, 1) = slmo;
    xland(i, j, 1) = xl;
    if (set_znt) znt(i, j, 1) = fmax2(1.0e-12, cnst_znt);
    // 171020:
    if (xland(i, j, 1) > 1.5) mavail(i, j, 1) = 1.0;
  });
  set_dsxy(o, g, s);
}

//------------------------------------------------------------------------------
// sfcphys.F
//------------------------------------------------------------------------------

void getcecd(const Options& o, const Bounds& b, const FArray3& zh, const SfcState& s) {
  using phys::karman;
  const int set_znt = o.set_znt, set_ust = o.set_ust, cecd = o.cecd, isfcflx = o.isfcflx;
  const Real cnst_ust = o.cnst_ust, cnstcd = o.cnstcd, cnstce = o.cnstce;
  const auto u1 = s.u1, v1 = s.v1, s1 = s.s1, u10 = s.u10, v10 = s.v10, s10 = s.s10;
  const auto xland = s.xland, znt = s.znt, ust = s.ust, cd = s.cd, ch = s.ch, cq = s.cq;
  int nfail = 0;
  Kokkos::parallel_reduce("sfc_getcecd", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j, int& bad) {
    const Real za = zh(i, j, 1);
    Real rznt, var;
    if (set_znt == 1) {
      // specify z0 (roughness length)
      rznt = 1.0 / znt(i, j, 1);
      const Real t = karman / Kokkos::log(10.0 * rznt);
      cd(i, j, 1) = t * t;
      var = Kokkos::log(10.0 * rznt) / Kokkos::log(za * rznt);
      u10(i, j, 1) = u1(i, j, 1) * var;
      v10(i, j, 1) = v1(i, j, 1) * var;
      s10(i, j, 1) = s1(i, j, 1) * var;
      ust(i, j, 1) = fmax2(s1(i, j, 1) * karman / Kokkos::log(za * rznt), 1.0e-6);
    } else if (set_ust == 1) {
      // specify ustar (friction velocity)
      ust(i, j, 1) = cnst_ust;
      znt(i, j, 1) = za / (Kokkos::exp(fmax2(1.0e-10, s1(i, j, 1)) * karman / fmax2(1.0e-10, ust(i, j, 1))) - 1.0);
      rznt = 1.0 / znt(i, j, 1);
      const Real t = karman / Kokkos::log((10.0 + znt(i, j, 1)) * rznt);
      cd(i, j, 1) = t * t;
      var = Kokkos::log((10.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
      u10(i, j, 1) = u1(i, j, 1) * var;
      v10(i, j, 1) = v1(i, j, 1) * var;
      s10(i, j, 1) = s1(i, j, 1) * var;
    } else if (cecd == 1) {
      // specify Cd (drag coefficient)
      cd(i, j, 1) = fmax2(1.0e-10, cnstcd);
      znt(i, j, 1) = 10.0 / (Kokkos::exp(karman / Kokkos::sqrt(cd(i, j, 1))) - 1.0);
      rznt = 1.0 / znt(i, j, 1);
      var = Kokkos::log((10.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
      u10(i, j, 1) = u1(i, j, 1) * var;
      v10(i, j, 1) = v1(i, j, 1) * var;
      s10(i, j, 1) = s1(i, j, 1) * var;
      ust(i, j, 1) = fmax2(s1(i, j, 1) * karman / Kokkos::log((za + znt(i, j, 1)) * rznt), 1.0e-6);
    } else {
      // Cd, z0, ust are functions of the 10-m wind speed over water: iterate
      if (xland(i, j, 1) > 1.5) {
        // water: first guess from the last known z0
        rznt = 1.0 / znt(i, j, 1);
        var = Kokkos::log((10.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
        s10(i, j, 1) = s1(i, j, 1) * var;
        Real wlast = -1000.0;
        int n = 0;
        while (Kokkos::fabs(s10(i, j, 1) - wlast) > 0.001) {
          n = n + 1;
          wlast = s10(i, j, 1);
          if (cecd == 2) {
            // Deacon's formula
            cd(i, j, 1) = 1.1e-3 + (4.0e-5 * s10(i, j, 1));
          } else if (cecd == 3) {
            // Fairall et al (2003) / Donelan et al (2004)
            cd(i, j, 1) = dcd1 + (s10(i, j, 1) - dwsp1) * dfac;
            cd(i, j, 1) = fmin2(cd(i, j, 1), dcd2);
            cd(i, j, 1) = fmax2(cd(i, j, 1), dcd1);
          }
          znt(i, j, 1) = fmax2(1.0e-20, 10.0 / (Kokkos::exp(karman / Kokkos::sqrt(cd(i, j, 1))) - 1.0));
          rznt = 1.0 / znt(i, j, 1);
          var = Kokkos::log((10.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
          s10(i, j, 1) = s1(i, j, 1) * var;
          if (n > 20) { bad += 1; break; }   // CM1: call stopcm1
        }
      } else {
        // land: z0 is specified
        rznt = 1.0 / znt(i, j, 1);
        const Real t = karman / Kokkos::log((10.0 + znt(i, j, 1)) * rznt);
        cd(i, j, 1) = t * t;
        var = Kokkos::log((10.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
        s10(i, j, 1) = s1(i, j, 1) * var;
      }
      u10(i, j, 1) = u1(i, j, 1) * var;
      v10(i, j, 1) = v1(i, j, 1) * var;
      ust(i, j, 1) = fmax2(s1(i, j, 1) * karman / Kokkos::log((za + znt(i, j, 1)) * rznt), 1.0e-6);
    }
    // ch, cq
    if (isfcflx == 1) {
      if (xland(i, j, 1) > 1.5) {
        if (cecd == 1) {
          ch(i, j, 1) = cnstce;
          cq(i, j, 1) = cnstce;
        } else if (cecd == 2) {
          ch(i, j, 1) = 1.1e-3 + (4.0e-5 * s10(i, j, 1));
          cq(i, j, 1) = 1.1e-3 + (4.0e-5 * s10(i, j, 1));
        } else if (cecd == 3) {
          ch(i, j, 1) = 1.20e-3;
          cq(i, j, 1) = 1.20e-3;
        }
      } else {
        ch(i, j, 1) = cd(i, j, 1);
        cq(i, j, 1) = cd(i, j, 1);
      }
    }
  }, nfail);
  if (nfail > 0)
    throw std::runtime_error("cm1k sfc::getcecd: the z0 iteration did not converge in 20 steps (CM1 stops)");
}

void gethpbl2(const Options& o, const Constants& c, const Bounds& b, const FArray3& zh,
              const FArray3& th0, const FArray3& tha, const FArray4& qa, const FArray3& ua,
              const FArray3& va, const FArray3& u0, const FArray3& v0, const SfcState& s) {
  const int nk = b.nk, ni = b.ni, nj = b.nj, ptype = o.ptype;
  const bool moist = (o.imoist == 1);
  const bool pert = (o.pertflx == 1);
  const Real repsm1 = c.repsm1, g = c.g, rovcp = c.rovcp, eps = c.eps;
  constexpr Real p00 = phys::p00, karman = phys::karman;
  constexpr Real bric = 0.125;   // critical bulk Richardson number
  const auto psfc = s.psfc, qsfc = s.qsfc, thflux = s.thflux, qvflux = s.qvflux, ust = s.ust;
  const auto tsk = s.tsk, hpbl = s.hpbl;
  // per-column work (Fortran thv1, thvsfc, govthv, brilast): w(i,j,1..4)
  FArray3 w("gethpbl2_work", {b.ib, b.jb, 1}, {b.ie, b.je, 4});
  // divx: qv (imoist=1) or 0; ugr/vgr: ground-relative winds (pertflx)
#define CM1K_QV(k) (moist ? qa(i, j, k, 1) : 0.0)
#define CM1K_UGR(ii, k) (pert ? ua(ii, j, k) - u0(ii, j, k) : ua(ii, j, k))
#define CM1K_VGR(jj, k) (pert ? va(i, jj, k) - v0(i, jj, k) : va(i, jj, k))
  Kokkos::parallel_for("sfc_gethpbl2_init", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
    qsfc(i, j, 1) = moist ? cm1libs::rslf(ptype, eps, psfc(i, j, 1), tsk(i, j, 1)) : 0.0;
    const Real thv = (th0(i, j, 1) + tha(i, j, 1)) * (1.0 + repsm1 * CM1K_QV(1));
    w(i, j, 3) = g / thv;   // govthv
  });
  for (int nloop = 1; nloop <= 2; ++nloop) {
    Kokkos::parallel_for("sfc_gethpbl2_k1", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j) {
      const int k = 1;
      const Real govthv = w(i, j, 3);
      const Real thv = (th0(i, j, 1) + tha(i, j, 1)) * (1.0 + repsm1 * CM1K_QV(1));
      const Real uavg = 0.5 * (CM1K_UGR(i, k) + CM1K_UGR(i + 1, k));
      const Real vavg = 0.5 * (CM1K_VGR(j, k) + CM1K_VGR(j + 1, k));
      if (nloop == 1) {
        w(i, j, 2) = tsk(i, j, 1) * Kokkos::pow(p00 / psfc(i, j, 1), rovcp) * (1.0 + repsm1 * qsfc(i, j, 1));
        w(i, j, 4) = govthv * zh(i, j, 1) * (thv - w(i, j, 2)) / fmax2(1.0, uavg * uavg + vavg * vavg);
        w(i, j, 1) = thv;
      } else {
        const Real thf1 = (1.0 + repsm1 * CM1K_QV(1));
        const Real thf2 = repsm1 * (th0(i, j, 1) + tha(i, j, 1));
        const Real thvflux = fmax2(0.0, thf1 * thflux(i, j, 1) + thf2 * qvflux(i, j, 1));
        const Real u = ust(i, j, 1);
        // hpbl: the first pass's value (-1 where it found none)
        Real ws = Kokkos::pow((u * u) * u + 8.0 * karman * 0.5 * govthv * thvflux * hpbl(i, j, 1), 0.3333333);
        ws = fmin2(ws, u * 16.0);
        ws = fmax2(ws, u / 5.0);
        ws = fmax2(ws, 0.0001);
        const Real gamfac = 6.8 / ws;
        const Real tmpprt = fmax2(0.0, fmin2(gamfac * thflux(i, j, 1), 3.0) * thf1
                                         + fmin2(gamfac * qvflux(i, j, 1), 2.0e-3) * thf2);
        w(i, j, 1) = thv + tmpprt;
        w(i, j, 4) = govthv * zh(i, j, 1) * (thv - w(i, j, 2)) / fmax2(1.0, uavg * uavg + vavg * vavg);
      }
    });
    // hpbl = -1.0 on the whole array (halos included). The search goes up
    // level by level over all columns and, as in sfcphys.F, stops once
    // ntot (the number of times any column found its top, on this rank)
    // reaches ni*nj. A column whose interpolated hpbl comes out negative
    // keeps searching and counts again, so the stop can come before every
    // column has a top: ntot is a global count, not a per-column test.
    hpbl.fill(-1.0);
    int ntot = 0;
    for (int k = 2; k <= nk; ++k) {
      int nnew = 0;
      Kokkos::parallel_reduce("sfc_gethpbl2_k", box2(1, ni, 1, nj), KOKKOS_LAMBDA(int i, int j, int& cnt) {
        if (hpbl(i, j, 1) < 0.0) {
          const Real thv = (th0(i, j, k) + tha(i, j, k)) * (1.0 + repsm1 * CM1K_QV(k));
          const Real uavg = 0.5 * (CM1K_UGR(i, k) + CM1K_UGR(i + 1, k));
          const Real vavg = 0.5 * (CM1K_VGR(j, k) + CM1K_VGR(j + 1, k));
          const Real brilast = w(i, j, 4);
          const Real bri = w(i, j, 3) * zh(i, j, k) * (thv - w(i, j, 1)) / fmax2(1.0, uavg * uavg + vavg * vavg);
          if (bri > bric) {
            if (Kokkos::fabs(bri - brilast) < 0.000001)
              hpbl(i, j, 1) = zh(i, j, k);
            else
              hpbl(i, j, 1) = zh(i, j, k - 1) + (zh(i, j, k) - zh(i, j, k - 1)) * (bric - brilast) / (bri - brilast);
            cnt += 1;
          }
          w(i, j, 4) = bri;
        }
      }, nnew);
      ntot += nnew;
      if (ntot == ni * nj) break;
    }
  }
#undef CM1K_QV
#undef CM1K_UGR
#undef CM1K_VGR
}

void sfcflux(const Options& o, const Constants& c, const Bounds& b, const FArray3& th0,
             const FArray3& tha, const FArray4& qa, const SfcState& s) {
  const auto thflux = s.thflux, qvflux = s.qvflux, qsfc = s.qsfc;
  const bool moist = (o.imoist == 1);
  if (o.set_flx == 1) {
    // specified fluxes
    const Real shf = o.cnst_shflx, lhf = o.cnst_lhflx;
    Kokkos::parallel_for("sfc_flux_set", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
      thflux(i, j, 1) = shf;
      if (moist) qvflux(i, j, 1) = lhf;
    });
    return;
  }
  const Real rovcp = c.rovcp, eps = c.eps;
  constexpr Real rp00 = phys::rp00;
  const int ptype = o.ptype;
  const auto psfc = s.psfc, ch = s.ch, cq = s.cq, s10 = s.s10, tsk = s.tsk, mavail = s.mavail;
  Kokkos::parallel_for("sfc_flux", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    // sensible heat flux
    const Real pisfc = Kokkos::pow(psfc(i, j, 1) * rp00, rovcp);
    thflux(i, j, 1) = ch(i, j, 1) * s10(i, j, 1) * (tsk(i, j, 1) / pisfc - th0(i, j, 1) - tha(i, j, 1));
    // latent heat flux
    if (moist) {
      const Real qvsat = cm1libs::rslf(ptype, eps, psfc(i, j, 1), tsk(i, j, 1));
      qsfc(i, j, 1) = qvsat;
      qvflux(i, j, 1) = cq(i, j, 1) * s10(i, j, 1) * (qvsat - qa(i, j, 1, 1)) * mavail(i, j, 1);
    }
  });
}

void sfcdiags(const Options& o, const Constants& c, const Bounds& b, const FArray3& zh,
              const FArray3& th0, const FArray3& tha, const FArray4& qa, const FArray3& rf,
              const SfcState& s) {
  const Real ep1 = c.rv / c.rd - 1.0;
  const Real rovcp = c.rovcp, g = c.g, cp = c.cp, eps = c.eps;
  constexpr Real rp00 = phys::rp00;
  constexpr Real vconvc = 1.;
  const int ptype = o.ptype;
  const bool moist = (o.imoist == 1);
  const auto psfc = s.psfc, tsk = s.tsk, qsfc = s.qsfc, znt = s.znt, gz1oz0 = s.gz1oz0;
  const auto thflux = s.thflux, qvflux = s.qvflux, hpbl = s.hpbl, dsxy = s.dsxy, s1 = s.s1;
  const auto wspd = s.wspd, br = s.br, hfx = s.hfx, qfx = s.qfx, cda = s.cda, cd = s.cd;
  const auto th2 = s.th2, q2 = s.q2, t2 = s.t2;
  Kokkos::parallel_for("sfc_diags", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    const Real qa1 = moist ? qa(i, j, 1, 1) : 0.0;   // solve1's divx: qv, or 0
    const Real za = zh(i, j, 1);
    const Real pisfc = Kokkos::pow(psfc(i, j, 1) * rp00, rovcp);
    const Real thgb = tsk(i, j, 1) / pisfc;
    const Real thx = th0(i, j, 1) + tha(i, j, 1);
    const Real thvx = thx * (1. + ep1 * qa1);
    qsfc(i, j, 1) = cm1libs::rslf(ptype, eps, psfc(i, j, 1), tsk(i, j, 1));
    const Real tskv = thgb * (1.0 + ep1 * qsfc(i, j, 1));
    const Real govrth = g / thx;
    const Real rznt = 1.0 / znt(i, j, 1);
    gz1oz0(i, j, 1) = Kokkos::log((za + znt(i, j, 1)) * rznt);
    const Real dthvdz = thvx - tskv;
    const Real fluxc = fmax2(thflux(i, j, 1) + ep1 * tskv * qvflux(i, j, 1), 0.);
    const Real vconv = vconvc * Kokkos::pow(g / tsk(i, j, 1) * hpbl(i, j, 1) * fluxc, .33);
    // Mahrt and Sun low-res correction
    const Real vsgd = 0.32 * Kokkos::pow(fmax2(dsxy(i, j, 1) / 5000. - 1., 0.), .33);
    wspd(i, j, 1) = Kokkos::sqrt(s1(i, j, 1) * s1(i, j, 1) + vconv * vconv + vsgd * vsgd);
    wspd(i, j, 1) = fmax2(0.1, wspd(i, j, 1));
    br(i, j, 1) = govrth * za * dthvdz / (wspd(i, j, 1) * wspd(i, j, 1));
    hfx(i, j, 1) = thflux(i, j, 1) * cp * rf(i, j, 1);
    qfx(i, j, 1) = qvflux(i, j, 1) * rf(i, j, 1);
    cda(i, j, 1) = cd(i, j, 1);
    // 2-m th/q/t
    const Real val = Kokkos::log((2.0 + znt(i, j, 1)) * rznt) / Kokkos::log((za + znt(i, j, 1)) * rznt);
    th2(i, j, 1) = thgb + (thx - thgb) * val;
    q2(i, j, 1) = qsfc(i, j, 1) + (qa1 - qsfc(i, j, 1)) * val;
    t2(i, j, 1) = th2(i, j, 1) * pisfc;
  });
}

//------------------------------------------------------------------------------
// turb.F
//------------------------------------------------------------------------------

void sfc_prelude(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
                 Real cgs1, Real cgs2, Real cgs3, const FArray3& ua, const FArray3& va,
                 const FArray3& tha, const FArray4& qa, const FArray3& prs, const SfcState& s) {
  const auto& b = g.b;
  const auto psfc = s.psfc, u1 = s.u1, v1 = s.v1, s1 = s.s1, t1 = s.t1;
  const auto u0 = base.u0, v0 = base.v0, th0 = base.th0;
  const bool pert = (o.pertflx == 1);
  Kokkos::parallel_for("sfc_layer_k1", box2(1, b.ni, 1, b.nj), KOKKOS_LAMBDA(int i, int j) {
    psfc(i, j, 1) = cgs1 * prs(i, j, 1) + cgs2 * prs(i, j, 2) + cgs3 * prs(i, j, 3);
    // ground-relative winds (pertflx=1: relative to the base-state wind)
    const Real ugr1 = pert ? ua(i, j, 1) - u0(i, j, 1) : ua(i, j, 1);
    const Real ugr2 = pert ? ua(i + 1, j, 1) - u0(i + 1, j, 1) : ua(i + 1, j, 1);
    const Real vgr1 = pert ? va(i, j, 1) - v0(i, j, 1) : va(i, j, 1);
    const Real vgr2 = pert ? va(i, j + 1, 1) - v0(i, j + 1, 1) : va(i, j + 1, 1);
    u1(i, j, 1) = 0.5 * (ugr1 + ugr2);
    v1(i, j, 1) = 0.5 * (vgr1 + vgr2);
    s1(i, j, 1) = Kokkos::sqrt(u1(i, j, 1) * u1(i, j, 1) + v1(i, j, 1) * v1(i, j, 1));
    t1(i, j, 1) = th0(i, j, 1) + tha(i, j, 1);
  });
  // bulk-Ri boundary-layer depth (testcase=0), before getcecd
  if (o.ipbl == 2) gethpbl2(o, c, b, g.zh, th0, tha, qa, ua, va, u0, v0, s);
}

void sfc_layer(const Options& o, const Constants& c, const Grid& g, const BaseState& base,
               Real cgs1, Real cgs2, Real cgs3, const FArray3& ua, const FArray3& va,
               const FArray3& tha, const FArray4& qa, const FArray3& prs, const FArray3& rf,
               const SfcState& s) {
  o.check();
  if (!o.active()) return;
  const auto& b = g.b;
  const auto th0 = base.th0;
  sfc_prelude(o, c, g, base, cgs1, cgs2, cgs3, ua, va, tha, qa, prs, s);
  getcecd(o, b, g.zh, s);
  if (o.isfcflx == 1) sfcflux(o, c, b, th0, tha, qa, s);
  sfcdiags(o, c, b, g.zh, th0, tha, qa, rf, s);
}

void sfcstress(const Bounds& b, const FArray3& rf, const SfcState& s,
               const FArray3& t13, const FArray3& t23) {
  const auto ust = s.ust, u1 = s.u1, v1 = s.v1, s1 = s.s1;
  Kokkos::parallel_for("sfc_stress", box2(1, b.ni + 1, 1, b.nj + 1), KOKKOS_LAMBDA(int i, int j) {
    t13(i, j, 1) = 0.25 * ((ust(i - 1, j, 1) * ust(i - 1, j, 1)) * (u1(i - 1, j, 1) / fmax2(s1(i - 1, j, 1), 0.01))
                          + (ust(i, j, 1) * ust(i, j, 1)) * (u1(i, j, 1) / fmax2(s1(i, j, 1), 0.01)))
                   * (rf(i - 1, j, 1) + rf(i, j, 1));
    t23(i, j, 1) = 0.25 * ((ust(i, j - 1, 1) * ust(i, j - 1, 1)) * (v1(i, j - 1, 1) / fmax2(s1(i, j - 1, 1), 0.01))
                          + (ust(i, j, 1) * ust(i, j, 1)) * (v1(i, j, 1) / fmax2(s1(i, j, 1), 0.01)))
                   * (rf(i, j - 1, 1) + rf(i, j, 1));
  });
}

} // namespace cm1k::sfc
