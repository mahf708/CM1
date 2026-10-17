! bind(C) shims for the dynamical core component tests: set the `input`
! module variables the routines read, then call the original bc.F routines.
! (sound.F's shim is in cm1f_bridge_sound.F90.)

subroutine cm1f_dyn_set_input(nx_, ny_, nz_, ngxy_, wbc_, ebc_, sbc_, nbc_, dx_, dy_, dz_, &
                              alph_, kdiv_, nsound_) bind(C, name="cm1f_dyn_set_input")
  use iso_c_binding
  use input
  use constants
  use ib_module, only : ibib,ieib,jbib,jeib,kbib,keib
  implicit none
  integer(c_int), value :: nx_, ny_, nz_, ngxy_, wbc_, ebc_, sbc_, nbc_, nsound_
  real(c_double), value :: dx_, dy_, dz_, alph_, kdiv_
  call set_constants(0)
  nx = nx_; ny = ny_; nz = nz_
  ni = nx; nj = ny; nk = nz; nkp1 = nk+1
  ngxy = ngxy_; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy
  jb = 1-ngxy; je = nj+ngxy
  kb = 1-ngz;  ke = nk+ngz
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
  ibib = 1; ieib = 1; jbib = 1; jeib = 1; kbib = 1; keib = 1
  imp = 1; jmp = 1; kmp = 1; rmp = 1; cmp = 1
  wbc = wbc_; ebc = ebc_; sbc = sbc_; nbc = nbc_
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if( wbc.ne.1 ) ibw = 1
  if( ebc.ne.1 ) ibe = 1
  if( sbc.ne.1 ) ibs = 1
  if( nbc.ne.1 ) ibn = 1
  axisymm = 0
  terrain_flag = .false.
  irbc = 4
  roflux = 0
  do_ib = .false.
  convinit = 0
  wnudge = 0
  timestats = 0
  smeps = 1.0e-60
  dx = dx_; dy = dy_; dz = dz_
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  alph = alph_
  kdiv = kdiv_
  nsound = nsound_
end subroutine cm1f_dyn_set_input

subroutine cm1f_bcs(s) bind(C, name="cm1f_bcs")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double) :: s(*)
  call bcs(s)
end subroutine cm1f_bcs

subroutine cm1f_bcp(s) bind(C, name="cm1f_bcp")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double) :: s(*)
  call bcp(s)
end subroutine cm1f_bcp

subroutine cm1f_bcu(s) bind(C, name="cm1f_bcu")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double) :: s(*)
  call bcu(s)
end subroutine cm1f_bcu

subroutine cm1f_bcv(s) bind(C, name="cm1f_bcv")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double) :: s(*)
  call bcv(s)
end subroutine cm1f_bcv

subroutine cm1f_bcw(s, flag) bind(C, name="cm1f_bcw")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double) :: s(*)
  integer(c_int), value :: flag
  call bcw(s, flag)
end subroutine cm1f_bcw
