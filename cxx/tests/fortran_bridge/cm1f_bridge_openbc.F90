! bind(C) shims around the open-radiative boundary routines of bc.F
! (radbcew, radbcns, radbcew4, radbcns4, ssopenbcw/e/s/n,
! restrict_openbc_we/sn). Call cm1f_dyn_set_input first, then
! cm1f_obc_set_input for the open-bc options.

subroutine cm1f_obc_set_input(irbc_, roflux_, cstar_) bind(C, name="cm1f_obc_set_input")
  use iso_c_binding
  use input
  implicit none
  integer(c_int), value :: irbc_, roflux_
  real(c_double), value :: cstar_
  irbc = irbc_
  roflux = roflux_
  cstar = cstar_
end subroutine cm1f_obc_set_input

subroutine cm1f_radbcew(radbcw, radbce, ua) bind(C, name="cm1f_radbcew")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(jb:je,kb:ke) :: radbcw, radbce
  real(c_double), dimension(ib:ie+1,jb:je,kb:ke) :: ua
  call radbcew(radbcw, radbce, ua)
end subroutine cm1f_radbcew

subroutine cm1f_radbcns(radbcs, radbcn, va) bind(C, name="cm1f_radbcns")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(ib:ie,kb:ke) :: radbcs, radbcn
  real(c_double), dimension(ib:ie,jb:je+1,kb:ke) :: va
  call radbcns(radbcs, radbcn, va)
end subroutine cm1f_radbcns

subroutine cm1f_radbcew4(ruf, radbcw, radbce, u1, u2, dt) bind(C, name="cm1f_radbcew4")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(ib:ie+1) :: ruf
  real(c_double), dimension(jb:je,kb:ke) :: radbcw, radbce
  real(c_double), dimension(ib:ie+1,jb:je,kb:ke) :: u1, u2
  real(c_double), value :: dt
  call radbcew4(ruf, radbcw, radbce, u1, u2, dt)
end subroutine cm1f_radbcew4

subroutine cm1f_radbcns4(rvf, radbcs, radbcn, v1, v2, dt) bind(C, name="cm1f_radbcns4")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(jb:je+1) :: rvf
  real(c_double), dimension(ib:ie,kb:ke) :: radbcs, radbcn
  real(c_double), dimension(ib:ie,jb:je+1,kb:ke) :: v1, v2
  real(c_double), value :: dt
  call radbcns4(rvf, radbcs, radbcn, v1, v2, dt)
end subroutine cm1f_radbcns4

! side: 1 = west, 2 = east, 3 = south, 4 = north (no terrain)
subroutine cm1f_ssopenbc(side, uvh, rad, uv3d, uvten, dts) bind(C, name="cm1f_ssopenbc")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  integer(c_int), value :: side
  real(c_double) :: uvh(*), rad(*), uv3d(*), uvten(*)
  real(c_double), value :: dts
  real, dimension(kb:ke) :: rds, sigma
  real, dimension(kb:ke+1) :: rdsf, sigmaf
  real, dimension(itb:ite,jtb:jte) :: gz, rgzu
  real, dimension(itb:ite,jtb:jte,ktb:kte) :: gx
  real, dimension(ib:ie,jb:je,kb:ke) :: dum1
  rds = 0.0; sigma = 0.0; rdsf = 0.0; sigmaf = 0.0
  gz = 1.0; rgzu = 1.0; gx = 0.0; dum1 = 0.0
  select case (side)
  case (1)
    call ssopenbcw(uvh, rds, sigma, rdsf, sigmaf, gz, rgzu, gx, rad, dum1, uv3d, uvten, dts)
  case (2)
    call ssopenbce(uvh, rds, sigma, rdsf, sigmaf, gz, rgzu, gx, rad, dum1, uv3d, uvten, dts)
  case (3)
    call ssopenbcs(uvh, rds, sigma, rdsf, sigmaf, gz, rgzu, gx, rad, dum1, uv3d, uvten, dts)
  case (4)
    call ssopenbcn(uvh, rds, sigma, rdsf, sigmaf, gz, rgzu, gx, rad, dum1, uv3d, uvten, dts)
  end select
end subroutine cm1f_ssopenbc

subroutine cm1f_restrict_openbc_we(rvh, rmh, rho0, u3d) bind(C, name="cm1f_restrict_openbc_we")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(jb:je) :: rvh
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: rmh, rho0
  real(c_double), dimension(ib:ie+1,jb:je,kb:ke) :: u3d
  call restrict_openbc_we(rvh, rmh, rho0, u3d)
end subroutine cm1f_restrict_openbc_we

subroutine cm1f_restrict_openbc_sn(ruh, rmh, rho0, v3d) bind(C, name="cm1f_restrict_openbc_sn")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), dimension(ib:ie) :: ruh
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: rmh, rho0
  real(c_double), dimension(ib:ie,jb:je+1,kb:ke) :: v3d
  call restrict_openbc_sn(ruh, rmh, rho0, v3d)
end subroutine cm1f_restrict_openbc_sn

! sound.F with open boundaries: as cm1f_sound, plus the radiative phase
! speeds (in/out) and the arrays restrict_openbc_we/sn read (roflux=1).
subroutine cm1f_sound_obc(dt, nrk, dttmp, uh, vh, uf, vf, mh, mf, c1, c2, rr0, rf0, dtu0, dtv0, &
                          rru, ua, u3d, uten, rrv, va, v3d, vten, rrw, wa, w3d, wten,        &
                          ppi, pp3d, piadv, ppten, ppx, thv, ppterm,                         &
                          radbcw, radbce, radbcs, radbcn, ruh, rvh, rmh, rho0) bind(C, name="cm1f_sound_obc")
  use iso_c_binding
  use input
  use sound_module
  use ib_module, only : ibib,ieib,jbib,jeib,kbib,keib
  implicit none
  real(c_double), value :: dt, dttmp
  integer(c_int), value :: nrk
  real(c_double), dimension(ib:ie) :: uh
  real(c_double), dimension(jb:je) :: vh
  real(c_double), dimension(ib:ie+1) :: uf
  real(c_double), dimension(jb:je+1) :: vf
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rr0, rf0
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double), dimension(ib:ie,jb:je) :: dtu0, dtv0
  real(c_double), dimension(ib:ie+1,jb:je,kb:ke) :: rru, ua, u3d, uten
  real(c_double), dimension(ib:ie,jb:je+1,kb:ke) :: rrv, va, v3d, vten
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: rrw, wa, w3d, wten
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: ppi, pp3d, piadv, ppten, ppx, thv, ppterm
  real(c_double), dimension(jb:je,kb:ke) :: radbcw, radbce
  real(c_double), dimension(ib:ie,kb:ke) :: radbcs, radbcn
  real(c_double), dimension(ib:ie) :: ruh
  real(c_double), dimension(jb:je) :: rvh
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: rmh, rho0

  real, dimension(ib:ie) :: xh, arh1, arh2
  real, dimension(ib:ie+1) :: xf
  real, dimension(jb:je) :: yh
  real, dimension(jb:je+1) :: yf
  real, dimension(kb:ke) :: rds, sigma
  real, dimension(kb:ke+1) :: rdsf, sigmaf
  real, dimension(ib:ie,jb:je,kb:ke) :: zh, rrf0, th0, rth0, pi0
  real, dimension(ib:ie,jb:je,kb:ke+1) :: zf
  real, dimension(ib:ie,jb:je) :: zs, dtu, dtv
  real, dimension(itb:ite,jtb:jte) :: gz, rgz, gzu, rgzu, gzv, rgzv, dzdx, dzdy
  real, dimension(itb:ite,jtb:jte,ktb:kte) :: gx, gxu, gy, gyv
  real, dimension(ib:ie,jb:je,kb:ke) :: dum1, dum2, dum3, dum4, mm, bk, ppd, fpk, qk, pk1, pk2, wtens
  real, dimension(ib:ie+1,jb:je,kb:ke) :: u0
  real, dimension(ib:ie,jb:je+1,kb:ke) :: v0
  logical, dimension(ibib:ieib,jbib:jeib,kbib:keib) :: bndy
  integer, dimension(ibib:ieib,jbib:jeib) :: kbdy
  real, dimension(jmp,kmp) :: pw1, pw2, pe1, pe2
  real, dimension(imp,kmp) :: ps1, ps2, pn1, pn2
  integer, dimension(rmp) :: reqs_p
  real :: rtime
  double precision :: mtime

  xh = 0.0; arh1 = 1.0; arh2 = 1.0; xf = 0.0; yh = 0.0; yf = 0.0
  rds = 0.0; sigma = 0.0; rdsf = 0.0; sigmaf = 0.0
  zh = 0.0; rrf0 = 1.0; th0 = 300.0; rth0 = 1.0/300.0; pi0 = 1.0
  zf = 0.0; zs = 0.0; dtu = 0.0; dtv = 0.0
  gz = 1.0; rgz = 1.0; gzu = 1.0; rgzu = 1.0; gzv = 1.0; rgzv = 1.0; dzdx = 0.0; dzdy = 0.0
  gx = 0.0; gxu = 0.0; gy = 0.0; gyv = 0.0
  dum1 = 0.0; dum2 = 0.0; dum3 = 0.0; dum4 = 0.0; mm = 0.0; bk = 0.0
  ppd = 0.0; fpk = 0.0; qk = 0.0; pk1 = 0.0; pk2 = 0.0; wtens = 0.0
  u0 = 0.0; v0 = 0.0
  bndy = .false.; kbdy = 1
  rtime = 0.0; mtime = 0.0d0

  call sound( dt,xh,arh1,arh2,uh,ruh,xf,uf,yh,vh,rvh,yf,vf,     &
              rds,sigma,rdsf,sigmaf,zh,mh,rmh,c1,c2,mf,zf,      &
              pi0,rho0,rr0,rf0,rrf0,th0,rth0,zs,                &
              gz,rgz,gzu,rgzu,gzv,rgzv,                         &
              dzdx,dzdy,gx,gxu,gy,gyv,                          &
              radbcw,radbce,radbcs,radbcn,dtu,dtu0,dtv,dtv0,    &
              dum1,dum2,dum3,dum4,mm,bk,                        &
              ppd,fpk,qk,pk1,pk2,wtens,                         &
              u0,rru,ua,u3d,uten,                               &
              v0,rrv,va,v3d,vten,                               &
              rrw,wa,w3d,wten,                                  &
              ppi,pp3d,piadv,ppten,ppx,                         &
              thv,ppterm,nrk,dttmp,rtime,mtime,.true.,          &
              bndy,kbdy,                                        &
              pw1,pw2,pe1,pe2,ps1,ps2,pn1,pn2,reqs_p)
end subroutine cm1f_sound_obc
