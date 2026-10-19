! A second copy of src/thompson.F (module_mp_thompson) and of
! src/mp_driver.F, for the Thompson bridge tests:
!
! * thompson.F is compiled under another module name with its default
!   accessibility switched to public, so the bridge can read the module's
!   private constants and lookup tables and replace the tables.
! * It is compiled with DM_PARALLEL, against the stubs below, so that the
!   expensive tables (qr_acr_qg, qr_acr_qs, freezeH2O: about 6 minutes in
!   full) are computed only over a test-chosen subset of their (zero-based)
!   decomposition index km, which is how WRF splits them over MPI ranks.
!   Nothing else in thompson.F depends on DM_PARALLEL except the aerosol-aware
!   branch of thompson_init (wrf_dm_max_real), which CM1 never takes.
! * mp_driver.F is compiled again, under another module name, against this
!   copy of the scheme, so its ptype=3 branch can be called with the copy's
!   tables.
!
! The includes go through macros so CMake's Fortran dependency scanner does
! not mistake this file for a provider of module_mp_thompson or
! mp_driver_module (see cm1f_bridge_turbpriv.F90).

module cm1f_thomcopy_dm
  implicit none
  ! km ranges (zero-based, inclusive) handed out by the decomposition stub,
  ! one per call: qr_acr_qg, qr_acr_qs, freezeH2O (thompson_init's order).
  integer :: decomp_s(3) = 0, decomp_e(3) = -1
  integer :: decomp_calls = 0
contains
  real function wrf_dm_max_real(x)
    real, intent(in) :: x
    wrf_dm_max_real = x
  end function wrf_dm_max_real
end module cm1f_thomcopy_dm

subroutine cm1f_thomcopy_decomp1d(n, km_s, km_e)
  use cm1f_thomcopy_dm
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: km_s, km_e
  integer :: c
  decomp_calls = decomp_calls + 1
  c = min(decomp_calls, 3)
  km_s = max(decomp_s(c), 0)
  km_e = min(decomp_e(c), n - 1)
end subroutine cm1f_thomcopy_decomp1d

subroutine cm1f_thomcopy_gatherv(buf, n, km_s, km_e, wordsz)
  implicit none
  double precision :: buf(*)
  integer :: n, km_s, km_e, wordsz
end subroutine cm1f_thomcopy_gatherv

subroutine cm1f_thomcopy_bcast_double(buf, n)
  implicit none
  double precision :: buf(*)
  integer :: n
end subroutine cm1f_thomcopy_bcast_double

#define DM_PARALLEL
#define IWORDSIZE 4
#define module_dm cm1f_thomcopy_dm
#define wrf_dm_decomp1d cm1f_thomcopy_decomp1d
#define wrf_dm_gatherv cm1f_thomcopy_gatherv
#define wrf_dm_bcast_double cm1f_thomcopy_bcast_double
#define module_mp_thompson cm1f_thomcopy
#define private public
#define PRIVATE PUBLIC
#define CM1F_THOMPSON_SOURCE "../../../src/thompson.F"
#include CM1F_THOMPSON_SOURCE
#undef PRIVATE
#undef private
#undef wrf_dm_bcast_double
#undef wrf_dm_gatherv
#undef wrf_dm_decomp1d
#undef module_dm
#undef DM_PARALLEL
#undef IWORDSIZE

#define mp_driver_module cm1f_mpdrvcopy
#define CM1F_MP_DRIVER_SOURCE "../../../src/mp_driver.F"
#include CM1F_MP_DRIVER_SOURCE
#undef mp_driver_module
#undef module_mp_thompson
