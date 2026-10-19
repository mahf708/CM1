// Stretched vertical grids (param.F stretch_z = 1, 2, 3, 4): step tests
// from the Fortran reference dumps and from Model::init (which also checks
// the grid metrics, base state and initial state at step 0), comparing
// every registered array bit for bit at every dump.
#include "grid_test_utils.hpp"

using cm1k::testing::run_all_fields;

TEST_CASE("stretch_z=1: dry_stretch (dry, sheared flow, Rayleigh damping) vs Fortran", "[grid][stretch][step]") {
  run_all_fields("dry_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1: dry_stretch from Model::init", "[grid][stretch][step][init]") {
  run_all_fields("dry_stretch", 0.0, true);
}

TEST_CASE("stretch_z=2: kessler_tke_stretch (ZGRID, Kessler, LES TKE) vs Fortran",
          "[grid][stretch][step][kessler][turb]") {
  run_all_fields("kessler_tke_stretch", 0.0, false);
}

TEST_CASE("stretch_z=2: kessler_tke_stretch from Model::init", "[grid][stretch][step][init][kessler][turb]") {
  run_all_fields("kessler_tke_stretch", 0.0, true);
}

TEST_CASE("stretch_z=3: cbl_sfc_stretch (input_grid_z, LES TKE + surface layer) vs Fortran",
          "[grid][stretch][step][turb][sfc]") {
  run_all_fields("cbl_sfc_stretch", 0.0, false);
}

TEST_CASE("stretch_z=3: cbl_sfc_stretch from Model::init", "[grid][stretch][step][init][turb][sfc]") {
  run_all_fields("cbl_sfc_stretch", 0.0, true);
}

TEST_CASE("stretch_z=4: pbl2_kessler_stretch (input_grid_z w levels, simple PBL, Kessler) vs Fortran",
          "[grid][stretch][step][pbl][kessler]") {
  run_all_fields("pbl2_kessler_stretch", 0.0, false);
}

TEST_CASE("stretch_z=4: pbl2_kessler_stretch from Model::init", "[grid][stretch][step][init][pbl][kessler]") {
  run_all_fields("pbl2_kessler_stretch", 0.0, true);
}

TEST_CASE("stretch_z=1: ysu_kessler_stretch (YSU + MM5 surface layer, Kessler) vs Fortran",
          "[grid][stretch][step][pbl][kessler]") {
  run_all_fields("ysu_kessler_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1: ysu_kessler_stretch from Model::init", "[grid][stretch][step][init][pbl][kessler]") {
  run_all_fields("ysu_kessler_stretch", 0.0, true);
}

TEST_CASE("stretch_z=1 + terrain: kessler_terrain_stretch vs Fortran", "[grid][stretch][step][terrain][turb]") {
  run_all_fields("kessler_terrain_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1 + terrain: kessler_terrain_stretch from Model::init",
          "[grid][stretch][step][init][terrain][turb]") {
  run_all_fields("kessler_terrain_stretch", 0.0, true);
}

TEST_CASE("stretch_z=1: mynn25_kessler_stretch (MYNN-EDMF 2.5 + MYNN surface layer) vs Fortran", "[grid][stretch][step][pbl][mynn]") {
  run_all_fields("mynn25_kessler_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1: mynn25_kessler_stretch (MYNN-EDMF 2.5 + MYNN surface layer) from Model::init", "[grid][stretch][step][pbl][mynn][init]") {
  run_all_fields("mynn25_kessler_stretch", 0.0, true);
}

TEST_CASE("stretch_z=1: gfs_kessler_stretch (GFS-EDMF + GFDL surface layer) vs Fortran", "[grid][stretch][step][pbl][gfs]") {
  run_all_fields("gfs_kessler_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1: gfs_kessler_stretch (GFS-EDMF + GFDL surface layer) from Model::init", "[grid][stretch][step][pbl][gfs][init]") {
  run_all_fields("gfs_kessler_stretch", 0.0, true);
}

TEST_CASE("stretch_z=1: myj_kessler_stretch (MYJ PBL + MYJ surface layer) vs Fortran", "[grid][stretch][step][pbl][myj]") {
  run_all_fields("myj_kessler_stretch", 0.0, false);
}

TEST_CASE("stretch_z=1: myj_kessler_stretch (MYJ PBL + MYJ surface layer) from Model::init", "[grid][stretch][step][pbl][myj][init]") {
  run_all_fields("myj_kessler_stretch", 0.0, true);
}

TEST_CASE("stretch_z=4: kessler_tke_openbc_stretch (open E/W, walls S/N, TKE) vs Fortran", "[grid][stretch][step][openbc][turb]") {
  run_all_fields("kessler_tke_openbc_stretch", 0.0, false);
}

TEST_CASE("stretch_z=4: kessler_tke_openbc_stretch (open E/W, walls S/N, TKE) from Model::init", "[grid][stretch][step][openbc][turb][init]") {
  run_all_fields("kessler_tke_openbc_stretch", 0.0, true);
}

TEST_CASE("stretch_z: unsupported values throw", "[grid][stretch]") {
  auto nl = cm1k::Namelist::from_file(cm1k::testing::case_namelist("dry_stretch"));
  nl.set("stretch_z", "5");
  cm1k::Model m(nl);
  CHECK_THROWS(m.init());
  nl.set("stretch_z", "2");
  nl.set("terrain_flag", ".true.");
  nl.set("itern", "1");
  cm1k::Model m2(nl);
  CHECK_THROWS(m2.init());
}
