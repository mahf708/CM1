! turb_module keeps tkekm and gettau (and sfcstress) private. This file
! compiles the unmodified src/turb.F a second time, under another module
! name and with its default accessibility switched to public, so the bridge
! can call those routines directly. (The include goes through a macro so
! CMake's Fortran dependency scanner does not mistake this file for a
! provider of turb_module.)
#define turb_module cm1f_turbcopy_module
#define private public
#define CM1F_TURB_SOURCE "../../../src/turb.F"
#include CM1F_TURB_SOURCE
#undef private
#undef turb_module

subroutine cm1f_turb_tkekm(ruh, rvh, rmh, zf, mf, rmf, rf, nm, defv, defh, kmh, kmv, khh, khv, &
                           cme, csm, ce1, ce2, tkea, lenscl, dissten) bind(C, name="cm1f_turb_tkekm")
  use iso_c_binding
  use input
  use cm1f_turbcopy_module, only: tkekm
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in) :: ruh(ib:ie), rvh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: rmh, rf
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: zf, mf, rmf, nm, defv, defh
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv, khh, khv
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: cme, csm, ce1, ce2
  real(c_double), intent(inout) :: tkea(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke+1) :: lenscl, dissten
  call tkekm(1, 0.0, 1.0, ruh, rvh, rmh, zf, mf, rmf, zero2, zero2, zero2, rf, &
             nm, defv, defh, s1, s2, s3, s4, s5, &
             kmh, kmv, khh, khv, cme, csm, ce1, ce2, tkea, lenscl, dissten, out3d, &
             c1d, c1d, c1d, c1d, c1d, c1d, c1d, c1d, &
             cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci, &
             cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
end subroutine cm1f_turb_tkekm

subroutine cm1f_turb_gettau(rf, mf, kmh, kmv, ua, va, wa, t11, t12, t13, t22, t23, t33) &
    bind(C, name="cm1f_turb_gettau")
  use iso_c_binding
  use input
  use cm1f_turbcopy_module, only: gettau
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in) :: rf(ib:ie,jb:je,kb:ke), mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv
  real(c_double), intent(in) :: ua(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke), wa(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: t11, t12, t13, t22, t23, t33
  real, allocatable :: stau(:,:)
  double precision :: avg
  allocate(stau(ib:ie,jb:je)); stau = 0; avg = 0
  call gettau(one_f, one_f, one_f, one_f, zero2, stau, zero2, zero2, zero2, zero2, zero2, zero2, zero2, &
              rf, mf, s1, s2, s3, s4, s5, s6, kmh, kmv, t11, t12, t13, t22, t23, t33, &
              ua, ua, va, va, wa, avg, avg, avg, avg, avg, bndy, kbdy)
end subroutine cm1f_turb_gettau

