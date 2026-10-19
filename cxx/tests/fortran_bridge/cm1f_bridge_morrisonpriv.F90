! module_mp_GRAUPEL keeps GAMMA, DERF1, satadj2, M2005MICRO_GRAUPEL and all
! of its constants private, and GRAUPEL_INIT hard-wires INUM=1. This file
! compiles the unmodified src/morrison.F a second time, under another
! module name and with its default accessibility switched to public, so the
! bridge can read the constants, call the private routines, and run the
! INUM=0 (predicted droplet number) path. The include goes through a macro
! so CMake's Fortran dependency scanner does not mistake this file for a
! provider of module_mp_GRAUPEL. The shims for the real module
! (mp_driver_module) are in this file too, so nothing outside it needs the
! copy's .mod file.
#define module_mp_GRAUPEL cm1f_graupelcopy_module
#define private public
#define PRIVATE PUBLIC
#define CM1F_MORRISON_SOURCE "../../../src/morrison.F"
#include CM1F_MORRISON_SOURCE
#undef PRIVATE
#undef private
#undef module_mp_GRAUPEL

subroutine cm1f_morrison_setup(ni_, nj_, nk_, ngxy_, ngz_, nrain_, eqtset_, efall_, radopt_, ihail_, &
                               ndcnst_, inum_, dx_, dy_, dz_) bind(C, name="cm1f_morrison_setup")
  use iso_c_binding
  use input
  use constants
  use module_mp_graupel, only : graupel_init
  use cm1f_graupelcopy_module, only : copy_graupel_init => graupel_init, copy_inum => inum
  implicit none
  integer(c_int), value :: ni_, nj_, nk_, ngxy_, ngz_, nrain_, eqtset_, efall_, radopt_, ihail_, inum_
  real(c_double), value :: ndcnst_, dx_, dy_, dz_
  integer :: inum_out

  nx = ni_ ; ny = nj_ ; nz = nk_
  ni = ni_ ; nj = nj_ ; nk = nk_ ; nkp1 = nk + 1
  ib = 1 - ngxy_ ; ie = ni + ngxy_
  jb = 1 - ngxy_ ; je = nj + ngxy_
  kb = 1 - ngz_  ; ke = nk + ngz_
  ibm = ib ; iem = ie ; jbm = jb ; jem = je ; kbm = kb ; kem = ke
  ! effective radii (dorad)
  ibr = ib ; ier = ie ; jbr = jb ; jer = je ; kbr = kb ; ker = ke
  nrain = nrain_

  ! the rest of mp_driver's argument shapes: unused for ptype=5
  ibp3 = 1 ; iep3 = 1 ; jbp3 = 1 ; jep3 = 1 ; kbp3 = 1 ; kep3 = 1 ; np3a = 1 ; np3o = 1
  ibdt = 1 ; iedt = 1 ; jbdt = 1 ; jedt = 1 ; kbdt = 1 ; kedt = 1 ; ntdiag = 1
  ibdq = 1 ; iedq = 1 ; jbdq = 1 ; jedq = 1 ; kbdq = 1 ; kedq = 1 ; nqdiag = 1
  ib2d = 1 ; ie2d = 1 ; jb2d = 1 ; je2d = 1 ; nout2d = 1
  ib3d = 1 ; ie3d = 1 ; jb3d = 1 ; je3d = 1 ; kb3d = 1 ; ke3d = 1 ; nout3d = 1
  ibtmp = 1 ; ietmp = 1 ; jbtmp = 1 ; jetmp = 1 ; kbtmp = 1 ; ketmp = 1 ; ntmp2d = 1 ; ntmp1d = 1

  ! param.F for imoist=1, ptype=5
  imoist = 1 ; ptype = 5 ; iice = 1 ; idm = 1
  ihail = ihail_ ; ndcnst = ndcnst_
  nqv = 1 ; nql1 = 2 ; nql2 = 3 ; nqs1 = 4 ; nqs2 = 6
  nqc = 2 ; nqr = 3 ; nqi = 4 ; nqs = 5 ; nqg = 6
  if (inum_ == 0) then
    numq = 11 ; nnc1 = 7 ; nnc2 = 11
  else
    numq = 10 ; nnc1 = 7 ; nnc2 = 10
  endif
  nbudget = 10
  eqtset = eqtset_ ; efall = efall_ ; radopt = radopt_
  testcase = 0
  pdscheme = 1
  axisymm = 0
  timestats = 0
  myid = 0
  stopit = .false.
  qd_dbz = 0 ; td_mp = 0 ; qd_mp = 0 ; td_efall = 0
  qd_vtc = 0 ; qd_vtr = 0 ; qd_vts = 0 ; qd_vtg = 0 ; qd_vti = 0
  output_fallvel = 0

  ! cm1.F (double precision)
  smeps = 1.0e-60
  tsmall = 0.000000000001
  qsmall = 1.0e-24

  dx = dx_ ; dy = dy_ ; dz = dz_
  rdx = 1.0/dx ; rdy = 1.0/dy ; rdz = 1.0/dz

  call set_constants(0)

  call graupel_init(ihail, inum_out, ndcnst)
  call copy_graupel_init(ihail, inum_out, ndcnst)
  copy_inum = inum_
end subroutine cm1f_morrison_setup

! Full mp_driver call (ptype=5, real module, INUM=1). Budgets use
! ruh=rvh=rmh... as passed and are discarded. vt* are mp_driver's qten fall
! speeds (getvt is forced on), eff* the effective radii (dorad).
subroutine cm1f_morrison_mp_driver(dt_, dorad_, mh, rmh, mf, pi0, th0, rho, prs, pp3d, th3d, w3d, q3d, &
                                   rain, prate, vt, eff) bind(C, name="cm1f_morrison_mp_driver")
  use iso_c_binding
  use input
  use mp_driver_module
  implicit none
  real(c_double), value :: dt_
  integer(c_int), value :: dorad_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, rmh, pi0, th0, rho, prs, pp3d, th3d
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w3d
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate
  real(c_double), dimension(ib:ie,jb:je,kb:ke,5) :: vt
  real(c_double), dimension(ibr:ier,jbr:jer,kbr:ker,6) :: eff

  real :: dt
  double precision, allocatable :: qbudget(:), asq(:), bsq(:), d2(:,:,:)
  real, allocatable, dimension(:) :: xh, ruh, xf, ruf, yh, rvh, yf, rvf
  real, allocatable, dimension(:,:,:) :: c1, c2, zh, rmf, zf, rho0, prs0, qv0
  real, allocatable, dimension(:,:,:,:) :: dum, qa, qten
  real, allocatable :: p3a(:,:,:), p3o(:,:,:,:), tdiag(:,:,:,:), qdiag(:,:,:,:)
  real, allocatable :: out2d(:,:,:), out3d(:,:,:,:), tmp2d(:,:,:), tmp1d(:,:)
  logical :: getdbz, getvt, dorad
  integer :: nstep

  dt = dt_
  nstep = 1
  dorad = (dorad_ /= 0)
  allocate(qbudget(nbudget), asq(numq), bsq(numq), d2(ib:ie,jb:je,5))
  qbudget = 0.0d0 ; asq = 0.0d0 ; bsq = 0.0d0 ; d2 = 0.0d0
  allocate(xh(ib:ie), ruh(ib:ie), xf(ib:ie+1), ruf(ib:ie+1), yh(jb:je), rvh(jb:je), yf(jb:je+1), rvf(jb:je+1))
  xh = 0.0 ; ruh = 1.0 ; xf = 0.0 ; ruf = 1.0 ; yh = 0.0 ; rvh = 1.0 ; yf = 0.0 ; rvf = 1.0
  allocate(c1(ib:ie,jb:je,kb:ke), c2(ib:ie,jb:je,kb:ke), zh(ib:ie,jb:je,kb:ke))
  allocate(rmf(ib:ie,jb:je,kb:ke+1), zf(ib:ie,jb:je,kb:ke+1))
  allocate(rho0(ib:ie,jb:je,kb:ke), prs0(ib:ie,jb:je,kb:ke), qv0(ib:ie,jb:je,kb:ke))
  c1 = 0.0 ; c2 = 1.0 ; zh = 0.0 ; rmf = 1.0 ; zf = 0.0 ; rho0 = 1.0 ; prs0 = 1.0e5 ; qv0 = 0.0
  allocate(dum(ib:ie,jb:je,kb:ke,13))
  dum = 0.0
  allocate(qa(ibm:iem,jbm:jem,kbm:kem,numq), qten(ibm:iem,jbm:jem,kbm:kem,numq))
  qa = q3d ; qten = 0.0
  allocate(p3a(ibp3:iep3,kbp3:kep3,np3a), p3o(ibp3:iep3,jbp3:jep3,kbp3:kep3,np3o))
  allocate(tdiag(ibdt:iedt,jbdt:jedt,kbdt:kedt,ntdiag), qdiag(ibdq:iedq,jbdq:jedq,kbdq:kedq,nqdiag))
  allocate(out2d(ib2d:ie2d,jb2d:je2d,nout2d), out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  allocate(tmp2d(ibtmp:ietmp,kbtmp:ketmp,ntmp2d), tmp1d(ibtmp:ietmp,ntmp1d))
  p3a = 0.0 ; p3o = 0.0 ; tdiag = 0.0 ; qdiag = 0.0 ; out2d = 0.0 ; out3d = 0.0 ; tmp2d = 0.0 ; tmp1d = 0.0
  getdbz = .false. ; getvt = .true.

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
                 .false.,dorad,.false.,.false.,.false.,  &
                 getdbz,getvt,.false.,.false.)

  vt(:,:,:,1) = qten(:,:,:,nqc)
  vt(:,:,:,2) = qten(:,:,:,nqr)
  vt(:,:,:,3) = qten(:,:,:,nqi)
  vt(:,:,:,4) = qten(:,:,:,nqs)
  vt(:,:,:,5) = qten(:,:,:,nqg)
end subroutine cm1f_morrison_mp_driver

! GRAUPEL_INIT's module variables, in the order of cm1k::morrison::Params.
subroutine cm1f_morrison_params(v) bind(C, name="cm1f_morrison_params")
  use iso_c_binding
  use cm1f_graupelcopy_module, only : ai, ac, as, ar, ag, bi, bc, bs, br, bg, r, rv, cp, ep_2, cv, cvv, cpv, &
      cpl, cpi, xlv0, xls0, lv1, lv2, ls1, ls2, tmelt, rhosu, rhow, rhoi, rhosn, rhog, aimm, bimm, ecr, dcs, &
      mi0, mg0, f1s, f2s, f1r, f2r, g, qsmall, qthr, eii, eci, cpw, ci, di, cs, ds, cg, dg, rin, mmult, &
      lammaxi, lammini, lammaxr, lamminr, d_b, lammaxs, lammins, lammaxg, lamming, k1, c1, mw, osm, vi, &
      epsm, rhoa, map, ma, rr, bact, rm1, sig1, nanew1, f11, f21, rm2, sig2, nanew2, f12, f22, &
      cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8, cons9, cons10, cons11, cons12, cons13, &
      cons14, cons15, cons16, cons17, cons18, cons19, cons20, cons21, cons22, cons23, cons24, cons25, &
      cons26, cons27, cons28, cons29, cons30, cons31, cons32, cons33, cons34, cons35, cons36, cons37, &
      cons38, cons39, cons40, cons41, inum, ndcnst, ihail
  implicit none
  real(c_double), intent(out) :: v(129)
  v(1:10) = (/ ai, ac, as, ar, ag, bi, bc, bs, br, bg /)
  v(11:25) = (/ r, rv, cp, ep_2, cv, cvv, cpv, cpl, cpi, xlv0, xls0, lv1, lv2, ls1, ls2 /)
  v(26:37) = (/ tmelt, rhosu, rhow, rhoi, rhosn, rhog, aimm, bimm, ecr, dcs, mi0, mg0 /)
  v(38:47) = (/ f1s, f2s, f1r, f2r, g, qsmall, qthr, eii, eci, cpw /)
  v(48:55) = (/ ci, di, cs, ds, cg, dg, rin, mmult /)
  v(56:64) = (/ lammaxi, lammini, lammaxr, lamminr, d_b, lammaxs, lammins, lammaxg, lamming /)
  v(65:75) = (/ k1, c1, mw, osm, vi, epsm, rhoa, map, ma, rr, bact /)
  v(76:85) = (/ rm1, sig1, nanew1, f11, f21, rm2, sig2, nanew2, f12, f22 /)
  v(86:126) = (/ cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8, cons9, cons10, cons11, cons12, &
      cons13, cons14, cons15, cons16, cons17, cons18, cons19, cons20, cons21, cons22, cons23, cons24, &
      cons25, cons26, cons27, cons28, cons29, cons30, cons31, cons32, cons33, cons34, cons35, cons36, &
      cons37, cons38, cons39, cons40, cons41 /)
  v(127) = inum
  v(128) = ndcnst
  v(129) = ihail
end subroutine cm1f_morrison_params

real(c_double) function cm1f_morrison_gamma(x) bind(C, name="cm1f_morrison_gamma")
  use iso_c_binding
  use cm1f_graupelcopy_module, only : mgamma => gamma
  implicit none
  real(c_double), value :: x
  cm1f_morrison_gamma = mgamma(x)
end function cm1f_morrison_gamma

real(c_double) function cm1f_morrison_derf1(x) bind(C, name="cm1f_morrison_derf1")
  use iso_c_binding
  use cm1f_graupelcopy_module, only : mderf1 => derf1
  implicit none
  real(c_double), value :: x
  cm1f_morrison_derf1 = mderf1(x)
end function cm1f_morrison_derf1

subroutine cm1f_morrison_satadj2(tnew, prs, qv, qc, qr, qi, qs, qg, rho, eqtset_, tsmall_) &
    bind(C, name="cm1f_morrison_satadj2")
  use iso_c_binding
  use cm1f_graupelcopy_module, only : satadj2
  implicit none
  real(c_double) :: tnew, prs, qv, qc
  real(c_double), value :: qr, qi, qs, qg, rho, tsmall_
  integer(c_int), value :: eqtset_
  call satadj2(tnew, prs, qv, qc, qr, qi, qs, qg, rho, eqtset_, tsmall_)
end subroutine cm1f_morrison_satadj2

! MP_GRAUPEL of the copy (INUM as set by cm1f_morrison_setup): q3d holds
! qv qc qr qi qs qg nci ncs ncr ncg [ncc]; t is the temperature (inout),
! p the pressure, dzq the layer depths. vt(:,:,:,1..5) = vtc vtr vti vts vtg.
subroutine cm1f_morrison_mp_graupel(dt_, dorad_, t, q3d, p, rho, dzq, w, rain, prate, vt, eff) &
    bind(C, name="cm1f_morrison_mp_graupel")
  use iso_c_binding
  use input
  use cm1f_graupelcopy_module, only : mp_graupel
  implicit none
  real(c_double), value :: dt_
  integer(c_int), value :: dorad_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: t, p, rho, dzq
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: w
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate
  real(c_double), dimension(ib:ie,jb:je,kb:ke,5) :: vt
  real(c_double), dimension(ibr:ier,jbr:jer,kbr:ker,6) :: eff

  real, allocatable :: pii(:,:,:), nc(:,:,:), ruh(:), rvh(:), rmh(:,:,:)
  real, allocatable :: tdiag(:,:,:,:), qdiag(:,:,:,:), out3d(:,:,:,:)
  double precision :: tcond, tevac, tevar, train
  real :: dt

  dt = dt_
  allocate(pii(ib:ie,jb:je,kb:ke), nc(ib:ie,jb:je,kb:ke), ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke))
  allocate(tdiag(ibdt:iedt,jbdt:jedt,kbdt:kedt,ntdiag), qdiag(ibdq:iedq,jbdq:jedq,kbdq:kedq,nqdiag))
  allocate(out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  pii = 1.0 ; ruh = 1.0 ; rvh = 1.0 ; rmh = 1.0 ; tdiag = 0.0 ; qdiag = 0.0 ; out3d = 0.0
  nc = 0.0
  if (numq >= 11) nc = q3d(ib:ie,jb:je,kb:ke,11)
  tcond = 0.0d0 ; tevac = 0.0d0 ; tevar = 0.0d0 ; train = 0.0d0

  call MP_GRAUPEL(1, t, pii, &
                  q3d(ib,jb,kb,1), q3d(ib,jb,kb,2), q3d(ib,jb,kb,3), &
                  q3d(ib,jb,kb,4), q3d(ib,jb,kb,5), q3d(ib,jb,kb,6), &
                  q3d(ib,jb,kb,7), q3d(ib,jb,kb,8), q3d(ib,jb,kb,9), &
                  q3d(ib,jb,kb,10), nc, &
                  p, rho, dt, dzq, w, rain, prate, &
                  eff(ibr,jbr,kbr,1), eff(ibr,jbr,kbr,2), eff(ibr,jbr,kbr,3), &
                  eff(ibr,jbr,kbr,4), eff(ibr,jbr,kbr,5), eff(ibr,jbr,kbr,6), &
                  tcond, tevac, tevar, train, &
                  ruh, rvh, rmh, rho, .false., &
                  vt(ib,jb,kb,1), vt(ib,jb,kb,2), vt(ib,jb,kb,3), vt(ib,jb,kb,4), vt(ib,jb,kb,5), &
                  .true., dorad_ /= 0, .false., .false., tdiag, qdiag, out3d)
  if (numq >= 11) q3d(ib:ie,jb:je,kb:ke,11) = nc
end subroutine cm1f_morrison_mp_graupel
