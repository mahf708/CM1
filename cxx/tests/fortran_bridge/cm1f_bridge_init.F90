! bind(C) shims for the helpers used by the native initialization
! (src/cm1k/init): saturation mixing ratios from cm1libs.F.
subroutine cm1f_rslf_rsif(ptype_in, testcase, n, p, t, rsl, rsi) bind(C, name="cm1f_rslf_rsif")
  use iso_c_binding
  use input, only : ptype
  use constants, only : set_constants
  use cm1libs, only : rslf, rsif
  implicit none
  integer(c_int), value :: ptype_in, testcase, n
  real(c_double), intent(in) :: p(n), t(n)
  real(c_double), intent(out) :: rsl(n), rsi(n)
  integer :: i
  call set_constants(testcase)
  ptype = ptype_in
  do i = 1, n
    rsl(i) = rslf(p(i), t(i))
    rsi(i) = rsif(p(i), t(i))
  enddo
end subroutine cm1f_rslf_rsif
