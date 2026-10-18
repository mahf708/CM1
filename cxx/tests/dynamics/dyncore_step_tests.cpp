// Step-level tests of the dynamical core: start from the Fortran reference
// dump step_000000, advance with Model::step, and compare the prognostic
// arrays (halos included) against every dumped step.
#include "driver/cm1k_model.hpp"
#include "dynamics/cm1k_dyncore.hpp"
#include "physics/rrtmg/cm1k_radiation.hpp"
#include "share/cm1k_dump_io.hpp"

#include "cm1k_test_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace cm1k;

namespace {

// Largest relative difference (|a-b| / max|ref|) over all compared fields.
// start_step > 0 restarts from that (mid-run) dump instead of step_000000.
double run_case(const std::string& case_name, double tol, bool native_init = false, bool turb = false,
                int start_step = 0, const std::string& ref_root = "") {
  const auto ref = ref_root.empty() ? cm1k::testing::case_ref(case_name) : ref_root + "/" + case_name;
  if (ref.empty()) return 0;
  if (!std::filesystem::exists(DumpReader::step_dir(ref, start_step))) {
    WARN("missing " << DumpReader::step_dir(ref, start_step));
    return 0;
  }
  Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
  if (native_init) m.init();   // init/ port: param.F, base.F, init3d.F
  else             m.init_from_dump(DumpReader::step_dir(ref, start_step));

  std::vector<std::string> fields = {"ua", "va", "wa", "ppi", "tha", "rho", "prs"};
  if (turb) for (const char* n : {"tkea", "kmh", "kmv", "khh", "khv"}) fields.push_back(n);
  if (m.has_surface())   // sfcmodel>=1: every dumped surface array (tsk, znt, ust, thflux, ...)
    for (auto& [n, a] : m.surface().dumped()) fields.push_back(n);
  if (m.has_wrf_surface())   // sfcmodel=2,3: mol, and the SLAB soil temperatures tslb
    for (auto& [n, a] : m.wrf_surface().dumped()) fields.push_back(n);
  for (auto& n : m.pbl_field_names()) fields.push_back(n);   // ipbl=1,2: hpbl, thpten, ..., vpten
  if (m.radiation())   // radopt=2: tendencies, fluxes, o30, cldfra, coszen, gsw, glw, ...
    for (auto& [n, a] : m.radiation()->dumped()) fields.push_back(n);
  for (auto& n : m.field_names())
    if (n.rfind("qa_", 0) == 0) fields.push_back(n);

  // dumped steps (every step, or every N steps for some cases)
  std::vector<int> steps;
  for (auto& e : std::filesystem::directory_iterator(ref)) {
    const auto n = e.path().filename().string();
    if (n.rfind("step_", 0) == 0 && std::stoi(n.substr(5)) > start_step) steps.push_back(std::stoi(n.substr(5)));
  }
  std::sort(steps.begin(), steps.end());
  cm1k::testing::limit_steps(steps);
  REQUIRE(!steps.empty());
  const int nsteps = int(steps.size());
  double worst = 0;
  for (int step : steps) {
    const auto dir = DumpReader::step_dir(ref, step);
    while (m.nstep() < step) m.step();
    DumpReader r(dir);
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    std::string line = case_name + (start_step > 0 ? " (from " + std::to_string(start_step) + ")" : std::string()) +
                       " step " + std::to_string(step) + ":";
    for (auto& n : fields) {
      auto a = m.field3(n).to_host();
      auto b = r.read_host<3>(n);
      std::array<int, 3> lo, hi;
      for (int d = 0; d < 3; ++d) { lo[d] = b.lb(d); hi[d] = b.ub(d); }
      const auto st = diff<3>(a, b, lo, hi);
      char buf[128];
      std::snprintf(buf, sizeof buf, " %s %.2e/%.2e", n.c_str(), st.max_abs, st.rel);
      line += buf;
      worst = std::max(worst, st.rel);
      INFO(case_name << " step " << step << " field " << n << ": max abs " << st.max_abs << " rel " << st.rel);
      CHECK(st.rel <= tol);
    }
    std::printf("%s\n", line.c_str());
  }
  std::printf("%s: worst relative difference over %d dumps (to step %d): %.3e\n", case_name.c_str(), nsteps,
              steps.back(), worst);
  return worst;
}

} // namespace

TEST_CASE("dyncore: dry_bubble vs Fortran, every step", "[dyn][step]") {
  run_case("dry_bubble", 1e-12);
}

TEST_CASE("dyncore: dry_shear vs Fortran, every step", "[dyn][step]") {
  run_case("dry_shear", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_saturated vs Fortran, every step", "[dyn][step][kessler]") {
  run_case("kessler_saturated", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_bubble vs Fortran", "[dyn][step][kessler]") {
  run_case("kessler_bubble", 1e-12);
}

TEST_CASE("dyncore + Kessler: kessler_periodic vs Fortran", "[dyn][step][kessler]") {
  run_case("kessler_periodic", 1e-12);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke vs Fortran", "[dyn][step][kessler][turb]") {
  // sgsmodel=1: also compares tkea and the eddy viscosities (dump N holds
  // the K computed from state N-1, as the model state between steps does)
  run_case("kessler_tke", 0.0, false, true);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke from Model::init (no dump)", "[dyn][step][init][kessler][turb]") {
  run_case("kessler_tke", 0.0, true, true);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke restarted from a mid-run dump", "[dyn][step][kessler][turb]") {
  // Needs consecutive dumps (e.g. local every-step references of steps
  // 140..151 in CM1K_EXTRA_REF_DIR/kessler_tke). Restarting from dump N, the
  // first step must recompute the eddy viscosities from state N exactly as
  // the Fortran did (dump N+1 holds them). The other fields of the restarted
  // run are only approximate: ppx and the diabatic tendencies (qpten, ...)
  // of the previous step are not in the dumps (see Dyncore::start_from_state).
  const char* e = std::getenv("CM1K_EXTRA_REF_DIR");
  if (!e) return;
  const std::string ref = std::string(e) + "/kessler_tke";
  if (!std::filesystem::exists(ref)) return;
  int nchecked = 0;
  for (auto& d : std::filesystem::directory_iterator(ref)) {
    const auto n = d.path().filename().string();
    if (n.rfind("step_", 0) != 0) continue;
    const int step = std::stoi(n.substr(5));
    if (!std::filesystem::exists(DumpReader::step_dir(ref, step + 1))) continue;
    Model m(Namelist::from_file(cm1k::testing::case_namelist("kessler_tke")));
    m.init_from_dump(DumpReader::step_dir(ref, step));
    m.step();
    DumpReader r(DumpReader::step_dir(ref, step + 1));
    REQUIRE(r.nstep() == m.nstep());
    REQUIRE(r.mtime() == m.mtime());
    for (const char* f : {"kmh", "kmv", "khh", "khv"}) {
      auto a = m.field3(f).to_host();
      auto b = r.read_host<3>(f);
      std::array<int, 3> lo, hi;
      for (int k = 0; k < 3; ++k) { lo[k] = b.lb(k); hi[k] = b.ub(k); }
      const auto st = diff<3>(a, b, lo, hi);
      INFO("restart from step " << step << ": " << f << " max abs " << st.max_abs);
      CHECK(st.max_abs == 0.0);
    }
    ++nchecked;
  }
  std::printf("kessler_tke: restart K checked from %d dumps\n", nchecked);
}

TEST_CASE("dyncore + Kessler: kessler_openbc (open-radiative bc, irbc=4) vs Fortran", "[dyn][step][kessler][openbc]") {
  // wbc=ebc=sbc=nbc=2: radbcew4/radbcns4 phase speeds carried between RK
  // stages and steps, ssopenbc* in the acoustic steps, inflow-zeroed fluxes
  // in advection and pdef
  run_case("kessler_openbc", 0.0);
}

TEST_CASE("dyncore: dry_openbc (open bc, irbc=2, roflux=1, nudgeobc=1) vs Fortran", "[dyn][step][openbc]") {
  run_case("dry_openbc", 0.0);
}

TEST_CASE("dyncore: dry_openbc_walls (open E/W, walls S/N, irbc=1) vs Fortran", "[dyn][step][openbc]") {
  run_case("dry_openbc_walls", 0.0);
}

TEST_CASE("dyncore + LES TKE + Kessler: kessler_tke_openbc (open E/W, walls S/N) vs Fortran",
          "[dyn][step][kessler][turb][openbc]") {
  run_case("kessler_tke_openbc", 0.0, false, true);
}

TEST_CASE("dyncore + Kessler: kessler_openbc from Model::init (no dump)", "[dyn][step][init][kessler][openbc]") {
  run_case("kessler_openbc", 0.0, true);
}

TEST_CASE("dyncore + LES TKE + surface layer: cbl_sfc vs Fortran", "[dyn][step][turb][sfc]") {
  // sfcmodel=1, bbc=3 (semi-slip), isfcflx=1 with set_flx=1 (constant
  // heat flux), set_znt=1, pertflx=1, idiss=1, dry; also compares every
  // dumped surface array (dump N holds the surface layer of state N-1)
  run_case("cbl_sfc", 0.0, false, true);
}

TEST_CASE("dyncore + LES TKE + surface layer: cbl_sfc from Model::init (no dump)", "[dyn][step][init][turb][sfc]") {
  // init_surface (LANDUSE.TBL) and irandp=1 (gfortran random_number)
  run_case("cbl_sfc", 0.0, true, true);
}

TEST_CASE("dyncore + LES TKE + surface layer + Kessler: cbl_sfc_moist vs Fortran",
          "[dyn][step][turb][sfc][kessler]") {
  // water surface (lu0=16): bulk fluxes of theta and qv from Ch, Cq (cecd=2,
  // z0 iterated from the 10-m wind), eqtset=2 dissipative heating
  run_case("cbl_sfc_moist", 0.0, false, true);
}

TEST_CASE("dyncore + LES TKE + surface layer + Kessler: cbl_sfc_moist from Model::init (no dump)",
          "[dyn][step][init][turb][sfc][kessler]") {
  run_case("cbl_sfc_moist", 0.0, true, true);
}

TEST_CASE("dyncore + simple PBL + surface layer: pbl2_dry vs Fortran", "[dyn][step][pbl][sfc]") {
  // cm1setup=2, ipbl=2 (Bryan-Rotunno/Louis vertical eddy viscosity,
  // gethpbl2), horizturb=0, sfcmodel=1 with surface fluxes (isfcflx=1),
  // dry. Compares the eddy viscosities, every dumped surface array, hpbl
  // and the PBL tendencies (dump N holds those of state N-1).
  run_case("pbl2_dry", 0.0, false, true);
}

TEST_CASE("dyncore + simple PBL + horizontal Smagorinsky + Kessler: pbl2_kessler vs Fortran",
          "[dyn][step][pbl][sfc][kessler]") {
  // cm1setup=2, ipbl=2, horizturb=1 (turbparam_horiz, over water: tlh from
  // psfc), idiss=1, sfcmodel=1 over water, Kessler
  run_case("pbl2_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + simple PBL + Kessler: pbl2_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][kessler]") {
  run_case("pbl2_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + MYNN 2.5 PBL + MYNN surface layer + Kessler: mynn25_kessler vs Fortran",
          "[dyn][step][pbl][mynn][sfc][kessler]") {
  // cm1setup=2, ipbl=4 (MYNN-EDMF level 2.5, qke advected in solve2,
  // dissipative heating inside the scheme: param.F sets idiss=0),
  // sfcmodel=6 over water (SLAB only diagnoses qsfc there), horizturb=1,
  // Kessler. Compares every dumped surface array, the MYNN carried state
  // (qke, qke3d, tsq, ..., mol, rmol, ...), hpbl and the PBL tendencies.
  run_case("mynn25_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + MYNN 2.5 PBL + Kessler: mynn25_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][mynn][sfc][kessler]") {
  run_case("mynn25_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + MYNN 3 PBL + MYNN surface layer: mynn3_dry vs Fortran", "[dyn][step][pbl][mynn][sfc]") {
  // ipbl=5 (level 3: tsq, qsq, cov prognostic), sfcmodel=6 over water with
  // isftcflx=1, horizturb=0 (no gettau), dry
  run_case("mynn3_dry", 0.0, false, true);
}

TEST_CASE("dyncore + MYNN 3 PBL: mynn3_dry from Model::init (no dump)", "[dyn][step][init][pbl][mynn][sfc]") {
  run_case("mynn3_dry", 0.0, true, true);
}

TEST_CASE("dyncore + MYNN 2.5 PBL + MYNN surface layer + SLAB: mynn25_land_dry vs Fortran",
          "[dyn][step][pbl][mynn][sfc][slab]") {
  // ipbl=4, sfcmodel=6 over land (lu0=3, xland0=1, iz0tlnd=1: Chen-Zhang
  // thermal roughness) with the SLAB land model after SFCLAY_mynn (tsk,
  // tslb, the land hfx/qfx/lh; qsfc), horizturb=0, dry. Compares every
  // dumped surface array, tslb, the MYNN carried state, hpbl and the PBL
  // tendencies.
  run_case("mynn25_land_dry", 0.0, false, true);
}

TEST_CASE("dyncore + MYNN 2.5 PBL + SLAB: mynn25_land_dry from Model::init (no dump)",
          "[dyn][step][init][pbl][mynn][sfc][slab]") {
  // init_surface for sfcmodel=6 over land: LANDUSE.TBL (znt, mavail, emiss,
  // thc), tmn0, the soil layers and tslb
  run_case("mynn25_land_dry", 0.0, true, true);
}

TEST_CASE("dyncore + MYNN 3 PBL + MYNN surface layer + SLAB + Kessler: mynn3_land_kessler vs Fortran",
          "[dyn][step][pbl][mynn][sfc][slab][kessler]") {
  // ipbl=5 (level 3), sfcmodel=6 over land (lu0=7, iz0tlnd=0) with SLAB
  // (moist: SLAB's qfx/lh from qsg over land), horizturb=1, Kessler
  run_case("mynn3_land_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + MYNN 3 PBL + SLAB + Kessler: mynn3_land_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][mynn][sfc][slab][kessler]") {
  run_case("mynn3_land_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + MYNN PBL: restarted from a mid-run dump", "[dyn][step][pbl][mynn][sfc]") {
  // Needs consecutive dumps (local every-step references, e.g. of steps
  // 40..51, in CM1K_EXTRA_REF_DIR/<case>). Restarting from dump N, the first
  // step's sfc_and_turb must reproduce dump N+1's surface layer and MYNN
  // output from state N and the dumped carried arrays alone, i.e. the dumps
  // hold all of the scheme's state. (Other fields are approximate, see the
  // kessler_tke restart test.)
  const char* e = std::getenv("CM1K_EXTRA_REF_DIR");
  if (!e) return;
  for (const char* case_name : {"mynn25_kessler", "mynn3_dry", "mynn25_land_dry", "mynn3_land_kessler"}) {
    const std::string ref = std::string(e) + "/" + case_name;
    if (!std::filesystem::exists(ref)) continue;
    int nchecked = 0;
    for (auto& d : std::filesystem::directory_iterator(ref)) {
      const auto n = d.path().filename().string();
      if (n.rfind("step_", 0) != 0) continue;
      const int step = std::stoi(n.substr(5));
      if (step == 0 || !std::filesystem::exists(DumpReader::step_dir(ref, step + 1))) continue;
      Model m(Namelist::from_file(cm1k::testing::case_namelist(case_name)));
      m.init_from_dump(DumpReader::step_dir(ref, step));
      m.step();
      DumpReader r(DumpReader::step_dir(ref, step + 1));
      REQUIRE(r.nstep() == m.nstep());
      std::vector<std::string> fields = m.pbl_field_names();
      for (auto& [f, a] : m.surface().dumped()) fields.push_back(f);
      for (auto& f : fields) {
        if (f == "qke3d") continue;   // advected by the (approximately restarted) step
        auto a = m.field3(f).to_host();
        auto b = r.read_host<3>(f);
        std::array<int, 3> lo, hi;
        for (int k = 0; k < 3; ++k) { lo[k] = b.lb(k); hi[k] = b.ub(k); }
        const auto st = diff<3>(a, b, lo, hi);
        INFO(case_name << " restart from step " << step << ": " << f << " max abs " << st.max_abs);
        CHECK(st.max_abs == 0.0);
      }
      ++nchecked;
    }
    std::printf("%s: restart PBL/surface checked from %d dumps\n", case_name, nchecked);
  }
}

TEST_CASE("dyncore + YSU + revised MM5 surface layer + SLAB: ysu_dry vs Fortran", "[dyn][step][pbl][sfc][ysu]") {
  // cm1setup=2, ipbl=1 (YSU), sfcmodel=3 (sf_sfclayrev) over land (lu0=3)
  // with the SLAB soil model (tsk cools by its upward long-wave flux),
  // horizturb=0 (no gettau), idiss=1 (ysudiss + the surface term), dry.
  // Compares every dumped surface array, mol, tslb, hpbl and the YSU
  // tendencies (dump N holds those of state N-1).
  run_case("ysu_dry", 0.0, false, true);
}

TEST_CASE("dyncore + YSU + revised MM5 surface layer + SLAB: ysu_dry from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][ysu]") {
  // init_surface for sfcmodel=3: LANDUSE.TBL (znt, mavail, emiss, thc),
  // tmn0, the soil layers and tslb; init_physics (sf_sfclayrev_init)
  run_case("ysu_dry", 0.0, true, true);
}

TEST_CASE("dyncore + YSU + MM5 surface layer + Kessler: ysu_kessler vs Fortran",
          "[dyn][step][pbl][sfc][ysu][kessler]") {
  // ipbl=1, sfcmodel=2 (SFCLAY) over water (SLAB keeps tsk), horizturb=1,
  // idiss=1, Kessler (YSU's qc tendency)
  run_case("ysu_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + YSU + MM5 surface layer + Kessler: ysu_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][ysu][kessler]") {
  run_case("ysu_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + simple PBL + MM5 surface layer + SLAB + Kessler: pbl2_sfclay_land vs Fortran",
          "[dyn][step][pbl][sfc][kessler]") {
  // ipbl=2 (gethpbl2 before SFCLAY) with sfcmodel=2 over land: SLAB, and the
  // surface fluxes thflux/qvflux from SFCLAY/SLAB's hfx, qfx (dosfcflx)
  run_case("pbl2_sfclay_land", 0.0, false, true);
}

TEST_CASE("dyncore + RRTMG + YSU + MM5 surface layer + SLAB + Kessler: rrtmg_ysu_kessler vs Fortran",
          "[dyn][step][pbl][sfc][ysu][kessler][rrtmg]") {
  // radopt=2 (RRTMG LW + SW every dtrad = 36 s = 6 steps, daytime: 21 UTC
  // at -98.35 E in May) over land (lu0=3, SLAB driven by gsw/glw), Kessler
  // clouds (cal_cldfra1), YSU with rthraten = thraten. Compares the
  // radiation arrays too (dump N holds the last call before pass N's).
  run_case("rrtmg_ysu_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + RRTMG + YSU + Kessler: rrtmg_ysu_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][ysu][kessler][rrtmg]") {
  run_case("rrtmg_ysu_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + RRTMG + GFS-EDMF + MM5 surface layer + SLAB + Kessler: rrtmg_gfs_kessler vs Fortran",
          "[dyn][step][pbl][sfc][gfs][kessler][rrtmg]") {
  // as gfs_sfclay_land with radopt=2 (dtrad = 36 s, daytime): GFS-EDMF
  // reads rthraten = thraten, SLAB gsw/glw
  run_case("rrtmg_gfs_kessler", 0.0, false, true);
}

TEST_CASE("radiation options: unsupported ones throw", "[rrtmg]") {
  rrtmg::RadOptions o;
  o.radopt = 2; o.imoist = 1; o.ptype = 1; o.sfcmodel = 2;
  REQUIRE_NOTHROW(o.check());
  auto with = [&](auto f) { auto p = o; f(p); return p; };
  CHECK_THROWS(with([](auto& p) { p.radopt = 1; }).check());                 // NASA-GSFC
  CHECK_THROWS(with([](auto& p) { p.radopt = 3; }).check());
  CHECK_THROWS(with([](auto& p) { p.ptype = 0; }).check());                  // cal_cldfra1: no branch
  CHECK_THROWS(with([](auto& p) { p.ptype = 5; }).check());                  // has_req* = 1 schemes
  CHECK_THROWS(with([](auto& p) { p.imoist = 0; }).check());
  CHECK_THROWS(with([](auto& p) { p.sfcmodel = 0; }).check());
  CHECK_THROWS(with([](auto& p) { p.sfcmodel = 1; }).check());
  CHECK_THROWS(with([](auto& p) { p.ipbl = 4; }).check());                   // MYNN icloud_bl
  CHECK_THROWS(with([](auto& p) { p.ipbl = 6; p.sfcmodel = 7; }).check());   // MYJ
  CHECK_THROWS(with([](auto& p) { p.ipbl = 3; p.sfcmodel = 4; }).check());   // GFDL surface layer
  CHECK_THROWS(with([](auto& p) { p.sfcmodel = 6; }).check());               // MYNN surface layer
  REQUIRE_NOTHROW(with([](auto& p) { p.ipbl = 3; p.sfcmodel = 3; }).check()); // GFS-EDMF
  CHECK_THROWS(with([](auto& p) { p.testcase = 8; }).check());
  CHECK_THROWS(with([](auto& p) { p.rterm = 1; }).check());
  REQUIRE_NOTHROW(with([](auto& p) { p.radopt = 0; p.ptype = 0; }).check());
}

TEST_CASE("radiation clock: radtim after a restart", "[rrtmg]") {
  // cm1.F: radiation in the startup pass and in every pass whose end time
  // reaches radtim - 0.1*dt; Radiation::clock_at must give radtim before
  // the radiation of pass n, for a restart from dump n
  const Real dt = 6.0;
  for (Real dtrad : {36.0, 30.0, 6.0, 1.0, 100.0}) {
    std::vector<double> radtim{0.0};   // radtim[n]: before the radiation of pass n
    std::vector<int> calls;
    double mtime = 0.0, rt = 0.0;
    const Real dtr = std::max(dtrad, dt);
    for (int n = 0; n <= 60; ++n) {
      if (n > 0) mtime += double(dt);
      if (n == 0 || mtime >= rt - 0.1 * double(dt)) {
        calls.push_back(n);
        do { rt += dtr; } while (!(rt > mtime));
      }
      radtim.push_back(rt);
    }
    for (int n = 0; n <= 60; ++n) {
      INFO("dtrad " << dtrad << " nstep " << n);
      CHECK(rrtmg::Radiation::clock_at(n, dt, dtrad) == radtim[n]);
    }
    if (dtrad == 36.0) CHECK(calls[1] == 6);
    if (dtrad <= dt) CHECK(calls[1] == 1);
  }
}

TEST_CASE("dyncore + GFS-EDMF PBL + GFDL surface layer + Kessler: gfs_kessler vs Fortran",
          "[dyn][step][pbl][sfc][gfs][kessler]") {
  // cm1setup=2, ipbl=3 (GFS-EDMF: K profile, mass flux, stratocumulus
  // mixing, dissipative heating in thpten) with sfcmodel=4 (SF_GFDL over
  // water: z0 from the 10-m wind, mznt carried across steps, zntmp = mznt),
  // horizturb=1, Kessler. Compares every dumped surface array, the GFDL
  // ones (mznt, mol, rmol, zol, br, gz1oz0, psim, psih), hpbl, the PBL
  // tendencies and xkzh, xkzm.
  run_case("gfs_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + GFS-EDMF + GFDL surface layer + Kessler: gfs_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][gfs][kessler]") {
  run_case("gfs_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + GFS-EDMF PBL + revised MM5 surface layer + SLAB: gfs_dry vs Fortran",
          "[dyn][step][pbl][sfc][gfs][slab]") {
  // ipbl=3 with sfcmodel=3 (sf_sfclayrev) over land (lu0=3: SLAB, mznt =
  // znt), horizturb=0, dry
  run_case("gfs_dry", 0.0, false, true);
}

TEST_CASE("dyncore + GFS-EDMF PBL + revised MM5 surface layer + SLAB: gfs_dry from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][gfs][slab]") {
  run_case("gfs_dry", 0.0, true, true);
}

TEST_CASE("dyncore + GFS-EDMF PBL + MM5 surface layer + SLAB + Kessler: gfs_sfclay_land vs Fortran",
          "[dyn][step][pbl][sfc][gfs][slab][kessler]") {
  // ipbl=3 with sfcmodel=2 (SFCLAY) over land, Kessler
  run_case("gfs_sfclay_land", 0.0, false, true);
}

TEST_CASE("dyncore + GFS-EDMF PBL + MM5 surface layer + SLAB + Kessler: gfs_sfclay_land from Model::init (no dump)",
          "[dyn][step][init][pbl][sfc][gfs][slab][kessler]") {
  run_case("gfs_sfclay_land", 0.0, true, true);
}

TEST_CASE("dyncore + MYJ PBL + MYJ surface layer + Kessler: myj_kessler vs Fortran",
          "[dyn][step][pbl][myj][sfc][kessler]") {
  // cm1setup=2, ipbl=6 (MYJ: MYJPBL with the TKE it carries, no advection),
  // sfcmodel=7 over water (MYJSFC with isftcflx=1, then SLAB's qsfc),
  // horizturb=1, idiss=1 (turbparam_horiz's dissten + the surface term),
  // Kessler. Compares every dumped surface array, the MYJ carried state
  // (tke_myj, el_myj, thz0, ..., akhs, akms, zntmyj), hpbl, xkzh/xkzm and
  // the PBL tendencies (dump N holds those of state N-1).
  run_case("myj_kessler", 0.0, false, true);
}

TEST_CASE("dyncore + MYJ PBL + Kessler: myj_kessler from Model::init (no dump)",
          "[dyn][step][init][pbl][myj][sfc][kessler]") {
  // init_physics (MYJSFCINIT: tables, ust = 0.1), init_surface (z0base,
  // tslb), tke_myj = 0.01
  run_case("myj_kessler", 0.0, true, true);
}

TEST_CASE("dyncore + MYJ PBL + MYJ surface layer: myj_dry vs Fortran", "[dyn][step][pbl][myj][sfc]") {
  // ipbl=6, sfcmodel=7 over warm water (tsk0=305) with isftcflx=0,
  // horizturb=0 (no gettau), idiss=1, dry
  run_case("myj_dry", 0.0, false, true);
}

TEST_CASE("dyncore + MYJ PBL: myj_dry from Model::init (no dump)", "[dyn][step][init][pbl][myj][sfc]") {
  run_case("myj_dry", 0.0, true, true);
}

TEST_CASE("dyncore: dry_bubble from Model::init (no dump), every step", "[dyn][step][init]") {
  run_case("dry_bubble", 1e-12, true);
}

TEST_CASE("dyncore + Kessler: kessler_saturated from Model::init (no dump)", "[dyn][step][init][kessler]") {
  run_case("kessler_saturated", 1e-12, true);
}

TEST_CASE("dyncore: option checks", "[dyn]") {
  auto nl = Namelist::from_file(cm1k::testing::case_namelist("dry_bubble"));
  nl.set("psolver", "2");
  REQUIRE_THROWS(DynConfig::from_namelist(nl).check());
  nl.set("psolver", "3");
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());
  nl.set("terrain_flag", ".true.");
  REQUIRE_THROWS(DynConfig::from_namelist(nl).check());
  nl.set("terrain_flag", ".false.");
  // lateral boundaries: 1..4 are ported (2 with irbc = 1, 2, 4)
  nl.set("wbc", "2"); nl.set("ebc", "2");
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"irbc", "3"}, {"irbc", "0"}, {"roflux", "2"}, {"nudgeobc", "2"}, {"wbc", "5"}}) {
    auto x = nl;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  nl.set("wbc", "1"); nl.set("ebc", "1");
  nl.set("irbc", "3");   // irbc only matters for open boundaries
  REQUIRE_NOTHROW(DynConfig::from_namelist(nl).check());

  // subgrid turbulence: sgsmodel=1 (LES TKE) is ported, other options throw
  auto tk = Namelist::from_file(cm1k::testing::case_namelist("kessler_tke"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(tk).check());
  REQUIRE(DynConfig::from_namelist(tk).turb);
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"sgsmodel", "2"}, {"sgsmodel", "3"}, {"horizturb", "1"}, {"bbc", "2"}, {"tbc", "2"},
         {"isfcflx", "1"}, {"ipbl", "1"}, {"tconfig", "3"}}) {
    auto x = tk;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }

  // surface layer: sfcmodel=1 (bbc=3) is ported, other options throw
  auto sf = Namelist::from_file(cm1k::testing::case_namelist("cbl_sfc"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(sf).check());
  REQUIRE(DynConfig::from_namelist(sf).bbc == 3);
  REQUIRE(DynConfig::from_namelist(sf).turb_opt.dosfcflx);
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"sfcmodel", "2"}, {"sfcmodel", "5"}, {"oceanmodel", "2"}, {"cecd", "4"}, {"initsfc", "2"},
         {"tbc", "3"}, {"imove", "1"}, {"ipbl", "2"}, {"sgsmodel", "0"}, {"sgsmodel", "2"},
         {"set_flx", "2"}, {"season", "3"}, {"lu0", "34"}}) {
    auto x = sf;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(x).check());
  }
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"isfcflx", "0"}, {"cecd", "1"}, {"cecd", "2"}, {"set_znt", "0"}, {"set_ust", "1"},
         {"pertflx", "0"}, {"idiss", "0"}, {"doimpl", "0"}}) {
    auto x = sf;
    x.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_NOTHROW(DynConfig::from_namelist(x).check());
  }
  // bbc=3 needs a surface model; isfcflx=1 needs sfcmodel=1..7
  auto x = sf;
  x.set("sfcmodel", "0");
  REQUIRE_THROWS(DynConfig::from_namelist(x).check());

  // PBL: cm1setup=2 with ipbl=2 (simple Bryan-Rotunno/Louis scheme) and
  // horizturb=0/1 is ported; other PBL schemes and combinations throw
  auto pb = Namelist::from_file(cm1k::testing::case_namelist("pbl2_kessler"));
  REQUIRE_NOTHROW(DynConfig::from_namelist(pb).check());
  REQUIRE(DynConfig::from_namelist(pb).pbl);
  REQUIRE_FALSE(DynConfig::from_namelist(pb).turb);
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"ipbl", "0"}, {"ipbl", "1"}, {"ipbl", "3"}, {"ipbl", "4"}, {"ipbl", "5"}, {"ipbl", "6"},
         {"cm1setup", "4"}, {"sfcmodel", "0"}, {"sfcmodel", "4"}, {"horizturb", "2"},
         {"terrain_flag", ".true."}, {"imove", "1"}}) {
    auto y = pb;
    y.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_THROWS(DynConfig::from_namelist(y).check());
  }
  for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
         {"horizturb", "0"}, {"idiss", "0"}, {"doimpl", "0"}, {"isfcflx", "0"}, {"bcturbs", "2"},
         {"pertflx", "1"}, {"sfcmodel", "2"}, {"sfcmodel", "3"}}) {
    auto y = pb;
    y.set(k, v);
    INFO(k << " = " << v);
    REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
  }

  // MYNN: cm1setup=2 with ipbl=4/5 and the MYNN surface layer (sfcmodel=6)
  // over water or land (SLAB); other surface layers and options throw
  for (const char* c : {"mynn25_kessler", "mynn3_dry", "mynn25_land_dry", "mynn3_land_kessler"}) {
    auto my = Namelist::from_file(cm1k::testing::case_namelist(c));
    REQUIRE_NOTHROW(DynConfig::from_namelist(my).check());
    const auto cfg = DynConfig::from_namelist(my);
    REQUIRE(cfg.pbl);
    REQUIRE(cfg.mynn);
    REQUIRE(cfg.idiss == 0);   // param.F: MYNN does its own dissipative heating
    REQUIRE_FALSE(cfg.mynn_opt.dosfcflx);
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "1"}, {"sfcmodel", "2"}, {"sfcmodel", "0"}, {"ipbl", "2"}, {"ipbl", "1"},
           {"ipbl", "0"}, {"cm1setup", "1"}, {"cm1setup", "4"}, {"initsfc", "2"}, {"isftcflx", "4"},
           {"isftcflx", "5"}, {"iz0tlnd", "2"}, {"iz0tlnd", "4"}, {"oceanmodel", "2"}, {"pertflx", "1"},
           {"imove", "1"}, {"terrain_flag", ".true."}, {"horizturb", "2"}, {"tbc", "3"},
           // init_surface.F stops: lu_index must be 1..33 (0 too), tmn >= 100 K
           {"lu0", "0"}, {"lu0", "34"}, {"tmn0", "50.0"}}) {
      auto y = my;
      y.set(k, v);
      INFO(c << ": " << k << " = " << v);
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
    if (my.geti("imoist", 0) == 1) {   // (ptype is ignored when dry)
      auto y = my;
      y.set("ptype", "2");
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"horizturb", "0"}, {"horizturb", "1"}, {"idiss", "0"}, {"idiss", "1"}, {"isftcflx", "2"},
           {"isftcflx", "3"}, {"isftcflx", "0"}, {"iz0tlnd", "0"}, {"iz0tlnd", "1"}, {"iz0tlnd", "3"},
           {"lu0", "16"}, {"lu0", "3"}, {"lu0", "24"}, {"xland0", "1.0"}, {"season", "2"},
           {"doimpl", "0"}}) {
      auto y = my;
      y.set(k, v);
      INFO(c << ": " << k << " = " << v);
      REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
    }
  }

  // YSU: cm1setup=2, ipbl=1 with the WRF/MM5 surface layers (sfcmodel=2,3)
  for (const char* c : {"ysu_dry", "ysu_kessler"}) {
    auto ys = Namelist::from_file(cm1k::testing::case_namelist(c));
    INFO(c);
    REQUIRE_NOTHROW(DynConfig::from_namelist(ys).check());
    const auto cfg = DynConfig::from_namelist(ys);
    REQUIRE(cfg.pbl);
    REQUIRE(cfg.bbc == 3);
    REQUIRE(cfg.sfc_opt.isfcflx == 1);      // param.F: sfcmodel=2..7
    REQUIRE_FALSE(cfg.sfc_opt.dosfcflx);    // init_surface.F: ipbl=1
    REQUIRE(cfg.pbl_opt.l_inf == 0.0);      // param.F: ipbl=1
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "0"}, {"sfcmodel", "1"}, {"sfcmodel", "4"}, {"sfcmodel", "6"}, {"sfcmodel", "7"},
           {"cm1setup", "1"}, {"cm1setup", "4"}, {"ipbl", "4"}, {"ipbl", "5"},
           {"pertflx", "1"}, {"lu0", "0"}, {"lu0", "34"}, {"tmn0", "50.0"}, {"oceanmodel", "2"},
           {"isftcflx", "3"}, {"iz0tlnd", "3"}, {"set_flx", "1"}, {"imove", "1"}, {"terrain_flag", ".true."}}) {
      auto y = ys;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "2"}, {"sfcmodel", "3"}, {"ipbl", "2"}, {"ipbl", "3"}, {"horizturb", "0"}, {"horizturb", "1"},
           {"idiss", "0"}, {"isftcflx", "1"}, {"iz0tlnd", "1"}, {"lu0", "16"}, {"lu0", "18"},
           {"season", "2"}, {"set_znt", "1"}, {"isfcflx", "0"}}) {
      auto y = ys;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
    }
  }

  // MYJ: cm1setup=2, ipbl=6 with the MYJ surface layer (sfcmodel=7)
  for (const char* c : {"myj_kessler", "myj_dry"}) {
    auto mj = Namelist::from_file(cm1k::testing::case_namelist(c));
    INFO(c);
    REQUIRE_NOTHROW(DynConfig::from_namelist(mj).check());
    const auto cfg = DynConfig::from_namelist(mj);
    REQUIRE(cfg.pbl);
    REQUIRE(cfg.myj);
    REQUIRE_FALSE(cfg.mynn);
    REQUIRE(cfg.bbc == 3);
    REQUIRE(cfg.sfc_opt.isfcflx == 1);   // param.F: sfcmodel=2..7
    REQUIRE(cfg.myj_opt.dosfcflx);       // init_surface.F: sfcmodel=7, ipbl=6
    REQUIRE(cfg.sfc_opt.dosfcflx);
    REQUIRE(cfg.pbl_opt.l_inf == 0.0);   // param.F: ipbl=6
    REQUIRE(cfg.idiss == mj.geti("idiss", 0));   // not reset for ipbl=6
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "0"}, {"sfcmodel", "1"}, {"sfcmodel", "2"}, {"sfcmodel", "3"}, {"sfcmodel", "6"},
           {"ipbl", "0"}, {"ipbl", "1"}, {"ipbl", "2"}, {"ipbl", "3"}, {"ipbl", "4"}, {"cm1setup", "1"},
           {"cm1setup", "4"}, {"pertflx", "1"}, {"lu0", "34"}, {"tmn0", "50.0"}, {"oceanmodel", "2"},
           {"set_flx", "1"}, {"initsfc", "2"}, {"imove", "1"}, {"terrain_flag", ".true."}, {"horizturb", "2"},
           {"tbc", "3"}}) {
      auto y = mj;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"horizturb", "0"}, {"horizturb", "1"}, {"idiss", "0"}, {"idiss", "1"}, {"isftcflx", "0"},
           {"isftcflx", "1"}, {"isftcflx", "2"}, {"lu0", "3"}, {"lu0", "16"}, {"season", "2"}, {"set_znt", "1"},
           {"isfcflx", "0"}, {"doimpl", "0"}}) {
      auto y = mj;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
    }
  }

  // GFS-EDMF: cm1setup=2, ipbl=3 with the GFDL surface layer (sfcmodel=4,
  // water only) or the WRF/MM5 ones (sfcmodel=2,3)
  for (const char* c : {"gfs_kessler", "gfs_dry", "gfs_sfclay_land"}) {
    auto gf = Namelist::from_file(cm1k::testing::case_namelist(c));
    INFO(c);
    REQUIRE_NOTHROW(DynConfig::from_namelist(gf).check());
    const auto cfg = DynConfig::from_namelist(gf);
    REQUIRE(cfg.pbl);
    REQUIRE(cfg.idiss == 0);                // param.F: ipbl=3
    REQUIRE_FALSE(cfg.sfc_opt.dosfcflx);    // init_surface.F: ipbl=3
    REQUIRE(cfg.pbl_opt.l_inf == 0.0);      // param.F: ipbl=3
    const bool gfdl = gf.geti("sfcmodel", 0) == 4;
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "0"}, {"sfcmodel", "1"}, {"sfcmodel", "5"}, {"sfcmodel", "6"}, {"sfcmodel", "7"},
           {"cm1setup", "1"}, {"cm1setup", "4"}, {"ipbl", "6"}, {"pertflx", "1"}, {"lu0", "34"},
           {"tmn0", "50.0"}, {"oceanmodel", "2"}, {"set_flx", "1"}, {"imove", "1"}, {"terrain_flag", ".true."}}) {
      auto y = gf;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
    for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"sfcmodel", "2"}, {"sfcmodel", "3"}, {"horizturb", "0"}, {"horizturb", "1"}, {"idiss", "1"},
           {"season", "2"}}) {
      auto y = gf;
      y.set(k, v);
      INFO(k << " = " << v);
      REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
    }
    if (gfdl) {
      // GFDL over land (SF_GFDL reads smois out of bounds), with other PBL
      // schemes
      for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
             {"lu0", "3"}, {"lu0", "18"}, {"lu0", "0"}, {"ipbl", "1"}, {"ipbl", "2"}, {"ipbl", "4"},
             {"ipbl", "5"}, {"set_ust", "1"}}) {
        auto y = gf;
        y.set(k, v);
        INFO(k << " = " << v);
        REQUIRE_THROWS(DynConfig::from_namelist(y).check());
      }
    } else {
      for (auto [k, v] : std::vector<std::pair<std::string, std::string>>{
             {"ipbl", "1"}, {"ipbl", "2"}, {"lu0", "16"}, {"lu0", "18"}}) {
        auto y = gf;
        y.set(k, v);
        INFO(k << " = " << v);
        REQUIRE_NOTHROW(DynConfig::from_namelist(y).check());
      }
      auto y = gf;
      y.set("sfcmodel", "4");   // GFDL over land
      REQUIRE_THROWS(DynConfig::from_namelist(y).check());
    }
  }
}
