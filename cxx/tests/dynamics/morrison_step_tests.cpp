// Step tests of the Morrison microphysics (ptype=5) in the full model, bit
// for bit against the Fortran reference runs. A separate executable from
// dyncore_step_tests so each gets its own ctest time budget.
#include "step_case.hpp"

TEST_CASE("dyncore + Morrison: morrison_saturated vs Fortran, every step", "[dyn][step][morrison]") {
  // ptype=5, ihail=0, efall=1, eqtset=1: ice, snow and graupel from step 1
  run_case("morrison_saturated", 0.0);
}

TEST_CASE("dyncore + Morrison: morrison_bubble vs Fortran", "[dyn][step][morrison]") {
  // ptype=5, ihail=1 (hail), eqtset=2: warm rain, then ice by 1800 s
  run_case("morrison_bubble", 0.0);
}

TEST_CASE("dyncore + LES TKE + Morrison: morrison_supercell vs Fortran", "[dyn][step][morrison][turb]") {
  // reduced run/config_files/supercell: ptype=5, ihail=1, sgsmodel=1, idiss=1,
  // open lateral boundaries, imove=1, advwenos=2
  run_case("morrison_supercell", 0.0, false, true);
}
