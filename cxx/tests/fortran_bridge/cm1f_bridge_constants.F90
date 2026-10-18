! bind(C) shims around CM1 Fortran routines, for component tests.
! Pattern: one shim per routine, taking C pointers/scalars, setting any
! `input` module state the routine needs, then calling the original code.
subroutine cm1f_set_constants(testcase, out) bind(C, name="cm1f_set_constants")
  use iso_c_binding
  use constants
  implicit none
  integer(c_int), value :: testcase
  real(c_double), intent(out) :: out(8)
  call set_constants(testcase)
  out = (/ g, rd, cp, cv, cpdrd, lv1, c_s, condc /)
end subroutine cm1f_set_constants
