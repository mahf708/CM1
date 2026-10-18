! bind(C) shims around diff2.F (2nd-order explicit diffusion, idiff=1 with
! difforder=2). Call cm1f_dyn_set_input first (grid sizes, lateral boundary
! conditions, dx/dy/dz), then cm1f_diff2_set_input for the options diff2.F
! reads. Budget diagnostics are off (dobud=.false., dummy diag arrays), and
! the dissipation rate is not computed (idiss=0, output_dissten=0).

subroutine cm1f_diff2_set_input(kdiff2_, bbc_, tbc_, umove_, vmove_) bind(C, name="cm1f_diff2_set_input")
  use iso_c_binding
  use input
  implicit none
  real(c_double), value :: kdiff2_, umove_, vmove_
  integer(c_int), value :: bbc_, tbc_
  kdiff2 = kdiff2_
  bbc = bbc_; tbc = tbc_
  umove = umove_; vmove = vmove_
  idiss = 0; output_dissten = 0
  ibdv = 1; iedv = 1; jbdv = 1; jedv = 1; kbdv = 1; kedv = 1
  nudiag = 1; nvdiag = 1; nwdiag = 1
  ud_hediff = 0; ud_vediff = 0; vd_hediff = 0; vd_vediff = 0; wd_hediff = 0; wd_vediff = 0
end subroutine cm1f_diff2_set_input

subroutine cm1f_diff2def(uh, uf, vh, vf, mh, c1, c2, mf, divx, rho, rr, rf, &
                         t11, t12, t13, t22, t23, t33, ua, va, wa) bind(C, name="cm1f_diff2def")
  use iso_c_binding
  use input
  use diff2_module
  implicit none
  real(c_double), dimension(ib:ie) :: uh
  real(c_double), dimension(ib:ie+1) :: uf
  real(c_double), dimension(jb:je) :: vh
  real(c_double), dimension(jb:je+1) :: vf
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, divx, rho, rr, rf
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: t11, t12, t13, t22, t23, t33
  real(c_double), dimension(ib:ie+1,jb:je,kb:ke) :: ua
  real(c_double), dimension(ib:ie,jb:je+1,kb:ke) :: va
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: wa

  real, dimension(ib:ie) :: arh1, arh2
  real, dimension(ib:ie+1) :: arf1, arf2
  real, dimension(ib:ie,jb:je) :: ust, znt, u1, v1, s1
  real, dimension(ib:ie,jb:je,kb:ke+1) :: dissten

  arh1 = 1.0; arh2 = 1.0; arf1 = 1.0; arf2 = 1.0
  ust = 0.0; znt = 0.0; u1 = 0.0; v1 = 0.0; s1 = 1.0
  dissten = 0.0
  call diff2def(uh,arh1,arh2,uf,arf1,arf2,vh,vf,mh,c1,c2,mf,ust,znt,u1,v1,s1,  &
                divx,rho,rr,rf,t11,t12,t13,t22,t23,t33,ua,va,wa,dissten)
end subroutine cm1f_diff2def

! which: 1 = diff2u (ten u-staggered, ta/tb/tc = t11,t12,t13),
!        2 = diff2v (ten v-staggered, ta/tb/tc = t22,t12,t23),
!        3 = diff2w (ten w-staggered, ta/tb/tc = t33,t13,t23)
subroutine cm1f_diff2uvw(which, uh, uf, vh, vf, mh, mf, rho, rr, rf, divx, ta, tb, tc, ten) &
    bind(C, name="cm1f_diff2uvw")
  use iso_c_binding
  use input
  use diff2_module
  implicit none
  integer(c_int), value :: which
  real(c_double), dimension(ib:ie) :: uh
  real(c_double), dimension(ib:ie+1) :: uf
  real(c_double), dimension(jb:je) :: vh
  real(c_double), dimension(jb:je+1) :: vf
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, divx, ta, tb, tc
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double) :: ten(*)

  real, dimension(ib:ie) :: xh, rxh, arh1, arh2
  real, dimension(ib:ie+1) :: xf, rxf, arf1, arf2
  real, dimension(ib:ie,jb:je,kb:ke) :: dum, diffx, diffy, diffz
  real, dimension(ib:ie,jb:je) :: ust
  real, dimension(1,1,1,1) :: diag

  xh = 0.0; rxh = 0.0; xf = 0.0; rxf = 0.0
  arh1 = 1.0; arh2 = 1.0; arf1 = 1.0; arf2 = 1.0
  dum = 0.0; diffx = 0.0; diffy = 0.0; diffz = 0.0; ust = 0.0; diag = 0.0
  if( which.eq.1 )then
    call diff2u(rxh,arh1,arh2,uh,xf,arf1,arf2,uf,vh,vf,mh,mf,  &
                dum,diffx,diffy,diffz,ten,ust,rho,rr,rf,divx,ta,tb,tc,.false.,diag)
  elseif( which.eq.2 )then
    call diff2v(xh,arh1,arh2,uh,rxf,arf1,arf2,uf,vh,vf,mh,mf,  &
                dum,diffx,diffy,diffz,ten,ust,rho,rr,rf,divx,ta,tb,tc,.false.,diag)
  else
    call diff2w(rxh,arh1,arh2,uh,xf,arf1,arf2,uf,vh,vf,mh,mf,  &
                dum,diffx,diffy,diffz,ten,rho,rr,rf,divx,ta,tb,tc,.false.,diag)
  endif
end subroutine cm1f_diff2uvw

subroutine cm1f_diff2s(uh, uf, vh, vf, mh, mf, s, sten, rho, rr, rf) bind(C, name="cm1f_diff2s")
  use iso_c_binding
  use input
  use diff2_module
  implicit none
  real(c_double), dimension(ib:ie) :: uh
  real(c_double), dimension(ib:ie+1) :: uf
  real(c_double), dimension(jb:je) :: vh
  real(c_double), dimension(jb:je+1) :: vf
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, s, sten, rho, rr, rf
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf

  real, dimension(ib:ie) :: rxh, arh1, arh2
  real, dimension(ib:ie+1) :: xf, arf1, arf2
  real, dimension(ib:ie,jb:je,kb:ke) :: dum, diffx, diffy, diffz
  real, dimension(1,1,1,1) :: diag

  rxh = 0.0; xf = 0.0; arh1 = 1.0; arh2 = 1.0; arf1 = 1.0; arf2 = 1.0
  dum = 0.0; diffx = 0.0; diffy = 0.0; diffz = 0.0; diag = 0.0
  call diff2s(rxh,arh1,arh2,uh,xf,arf1,arf2,uf,vh,vf,mh,mf,  &
              dum,diffx,diffy,diffz,s,sten,rho,rr,rf,  &
              .false.,1,1,1,1,1,1,1,diag,0,0)
end subroutine cm1f_diff2s
