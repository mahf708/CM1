! bind(C) shims around the terrain branches (terrain_flag=.true.) of CM1's
! LES subgrid turbulence: turb.F calcdef and turbtend.F turbt, turbu (with
! turbuz), turbv (with turbvz), turbw, turbs.
!
! cm1f_turbterrain_setup is called after cm1f_turb_setup (cm1f_bridge_turb.F90,
! which sets the rest of the `input` state): it switches terrain on, sets the
! terrain bounds itb:ite, jtb:jte, ktb:kte = ib:ie, jb:je, kb:ke (cm1.F), zt
! and the param.F extrapolation coefficients cgs1..cgt3. The metrics come in
! as arguments (2D ones as (itb:ite,jtb:jte), 3D ones as (itb:ite,jtb:jte,ktb:kte)).

subroutine cm1f_turbterrain_setup(zt_, cg) bind(C, name="cm1f_turbterrain_setup")
  use iso_c_binding
  use input
  implicit none
  real(c_double), value :: zt_
  real(c_double), intent(in) :: cg(6)
  terrain_flag = .true.
  itb = ib; ite = ie; jtb = jb; jte = je; ktb = kb; kte = ke
  zt = zt_; rzt = 1.0/zt_
  cgs1 = cg(1); cgs2 = cg(2); cgs3 = cg(3); cgt1 = cg(4); cgt2 = cg(5); cgt3 = cg(6)
end subroutine cm1f_turbterrain_setup

subroutine cm1f_turbterrain_off() bind(C, name="cm1f_turbterrain_off")
  use iso_c_binding
  use input
  implicit none
  terrain_flag = .false.
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
end subroutine cm1f_turbterrain_off

subroutine cm1f_turbterrain_calcdef(rds, sigma, rdsf, sigmaf, zs, gz, rgz, gzu, rgzu, gzv, rgzv, gx, gy, &
                                    uh, uf, vh, vf, mh, c1, c2, mf, rho, rr, rf, ua, va, wa, &
                                    s11, s12, s13, s22, s23, s33, defv, defh) bind(C, name="cm1f_turbterrain_calcdef")
  use iso_c_binding
  use input
  use turb_module, only: calcdef
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1)
  real(c_double), intent(in) :: zs(ib:ie,jb:je)
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, rgzu, gzv, rgzv
  real(c_double), intent(in), dimension(itb:ite,jtb:jte,ktb:kte) :: gx, gy
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho, rr, rf
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: ua(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke), wa(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: s11, s12, s13, s22, s23, s33
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke+1) :: defv, defh
  call calcdef(rds, sigma, rdsf, sigmaf, zs, gz, rgz, gzu, rgzu, gzv, rgzv, &
               one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, one_f, uf, vh, vf, mh, c1, c2, mf, defv, defh, &
               s1, s2, ua, va, wa, s11, s12, s13, s22, s23, s33, gx, gy, rho, rr, rf)
end subroutine cm1f_turbterrain_calcdef

subroutine cm1f_turbterrain_turbt(dt, rds, sigma, gz, rgz, gzu, rgzu, gzv, rgzv, &
                                  uh, uf, vh, vf, mh, mf, rho, rr, rf, t, tten, kmh, kmv) &
    bind(C, name="cm1f_turbterrain_turbt")
  use iso_c_binding
  use input
  use turbtend_module, only: turbt
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke)
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, rgzu, gzv, rgzv
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: t(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(inout) :: tten(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv
  call turbt(dt, one_h, one_h, uh, one_f, uf, vh, vf, mh, mf, rho, rr, rf, &
             rds, sigma, gz, rgz, gzu, rgzu, gzv, rgzv, &
             s1, s2, s3, s4, s5, s6, t, tten, kmh, kmv)
end subroutine cm1f_turbterrain_turbt

subroutine cm1f_turbterrain_turbu(dt, zs, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gxu, &
                                  ruh, uf, vh, mh, mf, rmf, rho, rf, u, uten, w, t11, t12, t13, t22, kmv) &
    bind(C, name="cm1f_turbterrain_turbu")
  use iso_c_binding
  use input
  use turbtend_module, only: turbu
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: zs(ib:ie,jb:je)
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, gzv
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1)
  real(c_double), intent(in) :: gxu(itb:ite,jtb:jte,ktb:kte)
  real(c_double), intent(in) :: ruh(ib:ie), uf(ib:ie+1), vh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rf, t11, t12, t13, t22
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, rmf, w
  real(c_double), intent(in) :: u(ib:ie+1,jb:je,kb:ke)
  real(c_double), intent(inout) :: uten(ib:ie+1,jb:je,kb:ke)
  real(c_double), intent(in) :: kmv(ibc:iec,jbc:jec,kbc:kec)
  call turbu(dt, one_h, ruh, one_f, one_f, one_f, one_f, uf, vh, mh, mf, rmf, rho, rf, &
             zs, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gxu, &
             s1, s2, s3, s4, s5, s6, s7, s8, s9, u, uten, w, t11, t12, t13, t22, kmv, cm0_, &
             kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
end subroutine cm1f_turbterrain_turbu

subroutine cm1f_turbterrain_turbv(dt, zs, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gyv, &
                                  uh, rvh, vf, mh, mf, rho, rr, rf, v, vten, w, t12, t22, t23, kmv) &
    bind(C, name="cm1f_turbterrain_turbv")
  use iso_c_binding
  use input
  use turbtend_module, only: turbv
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: zs(ib:ie,jb:je)
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, gzv
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1)
  real(c_double), intent(in) :: gyv(itb:ite,jtb:jte,ktb:kte)
  real(c_double), intent(in) :: uh(ib:ie), rvh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, t12, t22, t23
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w
  real(c_double), intent(in) :: v(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(inout) :: vten(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(in) :: kmv(ibc:iec,jbc:jec,kbc:kec)
  call turbv(dt, one_h, one_h, one_h, one_h, uh, one_f, rvh, vf, mh, mf, rho, rr, rf, &
             zs, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gyv, &
             s1, s2, s3, s4, s5, s6, s7, s8, s9, v, vten, w, t12, t22, t23, kmv, cm0_, &
             kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
end subroutine cm1f_turbterrain_turbv

subroutine cm1f_turbterrain_turbw(dt, gz, rgzu, rgzv, rds, sigma, &
                                  uh, vh, mh, mf, rho, rf, w, wten, t13, t23, t33, t22, kmh) &
    bind(C, name="cm1f_turbterrain_turbw")
  use iso_c_binding
  use input
  use turbtend_module, only: turbw
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgzu, rgzv
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke)
  real(c_double), intent(in) :: uh(ib:ie), vh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rf, t13, t23, t33, t22
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w
  real(c_double), intent(inout) :: wten(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: kmh(ibc:iec,jbc:jec,kbc:kec)
  call turbw(dt, one_h, one_h, one_h, one_h, uh, one_f, vh, mh, mf, rho, rf, gz, rgzu, rgzv, rds, sigma, &
             s1, s2, s3, s4, s5, s6, w, wten, t13, t23, t33, t22, kmh, cm0_, .false., diag)
end subroutine cm1f_turbterrain_turbw

subroutine cm1f_turbterrain_turbs(iflux, dt, rds, sigma, rdsf, sigmaf, gz, rgz, gzu, rgzu, gzv, rgzv, &
                                  gx, gxu, gy, gyv, uh, uf, vh, vf, mh, mf, rho, rr, rf, s, sten, khh, khv, &
                                  dum7, dum8) bind(C, name="cm1f_turbterrain_turbs")
  use iso_c_binding
  use input
  use turbtend_module, only: turbs
  use cm1f_bridge_turb_mod
  implicit none
  integer(c_int), value :: iflux
  real(c_double), value :: dt
  real(c_double), intent(in) :: rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1)
  real(c_double), intent(in), dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, rgzu, gzv, rgzv
  real(c_double), intent(in), dimension(itb:ite,jtb:jte,ktb:kte) :: gx, gxu, gy, gyv
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, s, dum7, dum8
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: sten(ib:ie,jb:je,kb:ke)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: khh, khv
  call turbs(iflux, dt, .false., one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, uf, vh, vf, zero2, &
             rds, sigma, rdsf, sigmaf, mh, mf, gz, rgz, gzu, rgzu, gzv, rgzv, &
             gx, gxu, gy, gyv, &
             s1, s2, s3, s4, s5, s6, rho, rr, rf, s, sten, khh, khv, cm0_, dum7, dum8, &
             .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1)
end subroutine cm1f_turbterrain_turbs
