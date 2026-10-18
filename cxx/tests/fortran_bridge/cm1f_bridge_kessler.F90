! bind(C) shims around the Kessler microphysics (ptype=1): the full
! mp_driver path (mp_driver_module) and the individual routines of
! kessler_module / misclibs. Arrays are passed as raw float64 pointers in
! Fortran order, with the bounds set by cm1f_kessler_setup.

subroutine cm1f_kessler_setup(ni_, nj_, nk_, ngxy_, ngz_, nrain_, testcase_, eqtset_, efall_, &
                              dx_, dy_, dz_) bind(C, name="cm1f_kessler_setup")
  use iso_c_binding
  use input
  use constants
  implicit none
  integer(c_int), value :: ni_, nj_, nk_, ngxy_, ngz_, nrain_, testcase_, eqtset_, efall_
  real(c_double), value :: dx_, dy_, dz_

  nx = ni_ ; ny = nj_ ; nz = nk_
  ni = ni_ ; nj = nj_ ; nk = nk_ ; nkp1 = nk + 1
  ib = 1 - ngxy_ ; ie = ni + ngxy_
  jb = 1 - ngxy_ ; je = nj + ngxy_
  kb = 1 - ngz_  ; ke = nk + ngz_
  ibm = ib ; iem = ie ; jbm = jb ; jem = je ; kbm = kb ; kem = ke
  nrain = nrain_

  ! the rest of mp_driver's argument shapes: unused for ptype=1
  ibp3 = 1 ; iep3 = 1 ; jbp3 = 1 ; jep3 = 1 ; kbp3 = 1 ; kep3 = 1 ; np3a = 1 ; np3o = 1
  ibr = 1 ; ier = 1 ; jbr = 1 ; jer = 1 ; kbr = 1 ; ker = 1
  ibdt = 1 ; iedt = 1 ; jbdt = 1 ; jedt = 1 ; kbdt = 1 ; kedt = 1 ; ntdiag = 1
  ibdq = 1 ; iedq = 1 ; jbdq = 1 ; jedq = 1 ; kbdq = 1 ; kedq = 1 ; nqdiag = 1
  ib2d = 1 ; ie2d = 1 ; jb2d = 1 ; je2d = 1 ; nout2d = 1
  ib3d = 1 ; ie3d = 1 ; jb3d = 1 ; je3d = 1 ; kb3d = 1 ; ke3d = 1 ; nout3d = 1
  ibtmp = 1 ; ietmp = 1 ; jbtmp = 1 ; jetmp = 1 ; kbtmp = 1 ; ketmp = 1 ; ntmp2d = 1 ; ntmp1d = 1

  ! param.F for imoist=1, ptype=1
  imoist = 1 ; ptype = 1 ; iice = 0
  numq = 3 ; nqv = 1 ; nql1 = 2 ; nql2 = 3 ; nqs1 = -2 ; nqs2 = -2
  nqc = 2 ; nqr = 3 ; nqi = 0 ; nqs = 0 ; nqg = 0 ; nnci = 0 ; nncc = 0 ; nqi2 = 0 ; nqi3 = 0
  nbudget = 10
  eqtset = eqtset_ ; efall = efall_
  pdscheme = 1
  axisymm = 0
  timestats = 0
  myid = 0
  stopit = .false.
  qd_dbz = 0 ; td_mp = 0 ; qd_mp = 0 ; td_efall = 0
  output_fallvel = 0

  ! cm1.F (double precision)
  smeps = 1.0e-60
  tsmall = 0.000000000001
  qsmall = 1.0e-24

  dx = dx_ ; dy = dy_ ; dz = dz_
  rdx = 1.0/dx ; rdy = 1.0/dy ; rdz = 1.0/dz

  call set_constants(testcase_)
end subroutine cm1f_kessler_setup

! Full mp_driver call (ptype=1). Budgets use ruh=rvh=rmh=1 and are discarded.
subroutine cm1f_kessler_mp_driver(dt_, mh, mf, pi0, th0, rho, prs, pp3d, th3d, q3d, rain, prate, vr) &
    bind(C, name="cm1f_kessler_mp_driver")
  use iso_c_binding
  use input
  use mp_driver_module
  implicit none
  real(c_double), value :: dt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, pi0, th0, rho, prs, pp3d, th3d, vr
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate

  real :: dt
  double precision, allocatable :: qbudget(:), asq(:), bsq(:), d2(:,:,:)
  real, allocatable, dimension(:) :: xh, ruh, xf, ruf, yh, rvh, yf, rvf
  real, allocatable, dimension(:,:,:) :: rmh, c1, c2, zh, rmf, zf, rho0, prs0, qv0
  real, allocatable, dimension(:,:,:,:) :: dum, qa, qten
  real, allocatable :: w3d(:,:,:), p3a(:,:,:), p3o(:,:,:,:), eff(:,:,:,:), tdiag(:,:,:,:), qdiag(:,:,:,:)
  real, allocatable :: out2d(:,:,:), out3d(:,:,:,:), tmp2d(:,:,:), tmp1d(:,:)
  logical :: getdbz, getvt
  integer :: nstep

  dt = dt_
  nstep = 1
  allocate(qbudget(nbudget), asq(numq), bsq(numq), d2(ib:ie,jb:je,5))
  qbudget = 0.0d0 ; asq = 0.0d0 ; bsq = 0.0d0 ; d2 = 0.0d0
  allocate(xh(ib:ie), ruh(ib:ie), xf(ib:ie+1), ruf(ib:ie+1), yh(jb:je), rvh(jb:je), yf(jb:je+1), rvf(jb:je+1))
  xh = 0.0 ; ruh = 1.0 ; xf = 0.0 ; ruf = 1.0 ; yh = 0.0 ; rvh = 1.0 ; yf = 0.0 ; rvf = 1.0
  allocate(rmh(ib:ie,jb:je,kb:ke), c1(ib:ie,jb:je,kb:ke), c2(ib:ie,jb:je,kb:ke), zh(ib:ie,jb:je,kb:ke))
  allocate(rmf(ib:ie,jb:je,kb:ke+1), zf(ib:ie,jb:je,kb:ke+1))
  allocate(rho0(ib:ie,jb:je,kb:ke), prs0(ib:ie,jb:je,kb:ke), qv0(ib:ie,jb:je,kb:ke))
  rmh = 1.0 ; c1 = 0.0 ; c2 = 1.0 ; zh = 0.0 ; rmf = 1.0 ; zf = 0.0 ; rho0 = 1.0 ; prs0 = 1.0e5 ; qv0 = 0.0
  allocate(dum(ib:ie,jb:je,kb:ke,13), w3d(ib:ie,jb:je,kb:ke+1))
  dum = 0.0 ; w3d = 0.0
  allocate(qa(ibm:iem,jbm:jem,kbm:kem,numq), qten(ibm:iem,jbm:jem,kbm:kem,numq))
  qa = q3d ; qten = 0.0
  allocate(p3a(ibp3:iep3,kbp3:kep3,np3a), p3o(ibp3:iep3,jbp3:jep3,kbp3:kep3,np3o), eff(ibr:ier,jbr:jer,kbr:ker,6))
  allocate(tdiag(ibdt:iedt,jbdt:jedt,kbdt:kedt,ntdiag), qdiag(ibdq:iedq,jbdq:jedq,kbdq:kedq,nqdiag))
  allocate(out2d(ib2d:ie2d,jb2d:je2d,nout2d), out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  allocate(tmp2d(ibtmp:ietmp,kbtmp:ketmp,ntmp2d), tmp1d(ibtmp:ietmp,ntmp1d))
  p3a = 0.0 ; p3o = 0.0 ; eff = 0.0 ; tdiag = 0.0 ; qdiag = 0.0 ; out2d = 0.0 ; out3d = 0.0 ; tmp2d = 0.0 ; tmp1d = 0.0
  getdbz = .false. ; getvt = .false.

  ! dum(:,:,:,9..13): ppi, ppten, sten, tha, thten
  call mp_driver(nstep,dt,qbudget,asq,bsq,xh,ruh,xf,ruf,yh,rvh,yf,rvf,  &
                 mh,rmh,c1,c2,zh,mf,rmf,zf,rain,prate,pi0,th0,rho0,prs0,qv0,   &
                 rho,prs,dum(ib,jb,kb,1),dum(ib,jb,kb,2),dum(ib,jb,kb,3),dum(ib,jb,kb,4),  &
                 dum(ib,jb,kb,5),dum(ib,jb,kb,6),dum(ib,jb,kb,7),dum(ib,jb,kb,8),  &
                 w3d,dum(ib,jb,kb,9),pp3d,dum(ib,jb,kb,10),dum(ib,jb,kb,11),  &
                 dum(ib,jb,kb,12),th3d,dum(ib,jb,kb,13),qa,q3d,qten,   &
                 p3a,p3o,d2(ib,jb,1),d2(ib,jb,2),d2(ib,jb,3),d2(ib,jb,4),d2(ib,jb,5),  &
                 eff(ibr,jbr,kbr,1),eff(ibr,jbr,kbr,2),eff(ibr,jbr,kbr,3),eff(ibr,jbr,kbr,4),  &
                 eff(ibr,jbr,kbr,5),eff(ibr,jbr,kbr,6),    &
                 tdiag,qdiag,out2d,out3d,tmp2d,tmp1d,  &
                 .false.,.false.,.false.,.false.,.false.,  &
                 getdbz,getvt,.false.,.false.)

  vr = qten(:,:,:,3)
end subroutine cm1f_kessler_mp_driver

subroutine cm1f_kessler_pdefq(rmax, rho, q3d) bind(C, name="cm1f_kessler_pdefq")
  use iso_c_binding
  use input
  use misclibs, only : pdefq
  implicit none
  real(c_double), value :: rmax
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: rho, q3d
  double precision :: asq
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:)
  allocate(ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke))
  ruh = 1.0 ; rvh = 1.0 ; rmh = 1.0 ; asq = 0.0d0
  call pdefq(rmax, asq, ruh, rvh, rmh, rho, q3d)
end subroutine cm1f_kessler_pdefq

subroutine cm1f_kessler_k_fallout(rho, qr, vr) bind(C, name="cm1f_kessler_k_fallout")
  use iso_c_binding
  use input
  use kessler_module, only : k_fallout
  implicit none
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: rho, qr, vr
  call k_fallout(rho, qr, vr)
end subroutine cm1f_kessler_k_fallout

subroutine cm1f_kessler_fallout(dt_, mh, mf, rho, qr, vr, rain, prate) bind(C, name="cm1f_kessler_fallout")
  use iso_c_binding
  use input
  use kessler_module, only : fallout
  implicit none
  real(c_double), value :: dt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, qr, vr
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate
  real :: dt
  double precision :: train
  real, allocatable :: ruh(:), rvh(:), zh(:,:,:), rr(:,:,:)
  allocate(ruh(ib:ie), rvh(jb:je), zh(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke))
  ruh = 1.0 ; rvh = 1.0 ; zh = 0.0 ; rr = rho ; train = 0.0d0
  dt = dt_
  call fallout(dt, train, ruh, rvh, zh, mh, mf, rain, prate, rr, rho, qr, vr)
end subroutine cm1f_kessler_fallout

subroutine cm1f_kessler_kessler(dt_, pi0, th0, tmp, rho, pp3d, th3d, prs, qv, qc, qr) &
    bind(C, name="cm1f_kessler_kessler")
  use iso_c_binding
  use input
  use kessler_module, only : kessler
  implicit none
  real(c_double), value :: dt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: pi0, th0, tmp, rho, pp3d, th3d, prs, qv, qc, qr
  real :: dt
  double precision :: b1, b2, b3
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:), rr(:,:,:)
  allocate(ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke))
  ruh = 1.0 ; rvh = 1.0 ; rmh = 1.0 ; rr = rho
  b1 = 0.0d0 ; b2 = 0.0d0 ; b3 = 0.0d0
  dt = dt_
  call kessler(dt, b1, b2, b3, ruh, rvh, rmh, pi0, th0, tmp, rho, rr, pp3d, th3d, prs, qv, qc, qr)
end subroutine cm1f_kessler_kessler

subroutine cm1f_kessler_satadj(nrk, dt_, pi0, th0, rho, pp3d, prs, th3d, q3d) bind(C, name="cm1f_kessler_satadj")
  use iso_c_binding
  use input
  use kessler_module, only : satadj
  implicit none
  integer(c_int), value :: nrk
  real(c_double), value :: dt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: pi0, th0, rho, pp3d, prs, th3d
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real :: dt
  double precision :: b1, b2
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:), rr(:,:,:)
  allocate(ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke))
  ruh = 1.0 ; rvh = 1.0 ; rmh = 1.0 ; rr = rho
  b1 = 0.0d0 ; b2 = 0.0d0
  dt = dt_
  call satadj(nrk, dt, b1, b2, ruh, rvh, rmh, pi0, th0, rho, rr, pp3d, prs, th3d, q3d)
end subroutine cm1f_kessler_satadj
