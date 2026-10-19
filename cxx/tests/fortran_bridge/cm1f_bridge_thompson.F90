! bind(C) shims around the Thompson microphysics (ptype=3), through the
! copies of thompson.F and mp_driver.F in cm1f_bridge_thomcopy.F90:
! thompson_init (expensive tables over a chosen km subset), the module's
! constants and tables (read and replace), mp_gt_driver and the ptype=3
! branch of mp_driver. Arrays are raw float64 pointers in Fortran order,
! with the bounds set by cm1f_thompson_setup.

subroutine cm1f_thompson_setup(ni_, nj_, nk_, ngxy_, ngz_, nrain_, testcase_, eqtset_, &
                               dx_, dy_, dz_) bind(C, name="cm1f_thompson_setup")
  use iso_c_binding
  use input
  use constants
  implicit none
  integer(c_int), value :: ni_, nj_, nk_, ngxy_, ngz_, nrain_, testcase_, eqtset_
  real(c_double), value :: dx_, dy_, dz_

  nx = ni_ ; ny = nj_ ; nz = nk_
  ni = ni_ ; nj = nj_ ; nk = nk_ ; nkp1 = nk + 1
  ib = 1 - ngxy_ ; ie = ni + ngxy_
  jb = 1 - ngxy_ ; je = nj + ngxy_
  kb = 1 - ngz_  ; ke = nk + ngz_
  ibm = ib ; iem = ie ; jbm = jb ; jem = je ; kbm = kb ; kem = ke
  nrain = nrain_

  ! the rest of mp_driver's argument shapes: unused for ptype=3 here
  ibp3 = 1 ; iep3 = 1 ; jbp3 = 1 ; jep3 = 1 ; kbp3 = 1 ; kep3 = 1 ; np3a = 1 ; np3o = 1
  ibr = 1 ; ier = 1 ; jbr = 1 ; jer = 1 ; kbr = 1 ; ker = 1
  ibdt = 1 ; iedt = 1 ; jbdt = 1 ; jedt = 1 ; kbdt = 1 ; kedt = 1 ; ntdiag = 1
  ibdq = 1 ; iedq = 1 ; jbdq = 1 ; jedq = 1 ; kbdq = 1 ; kedq = 1 ; nqdiag = 1
  ib2d = 1 ; ie2d = 1 ; jb2d = 1 ; je2d = 1 ; nout2d = 1
  ib3d = 1 ; ie3d = 1 ; jb3d = 1 ; je3d = 1 ; kb3d = 1 ; ke3d = 1 ; nout3d = 1
  ibtmp = 1 ; ietmp = 1 ; jbtmp = 1 ; jetmp = 1 ; kbtmp = 1 ; ketmp = 1 ; ntmp2d = 1 ; ntmp1d = 1

  ! param.F for imoist=1, ptype=3
  imoist = 1 ; ptype = 3 ; iice = 1
  numq = 8 ; nqv = 1 ; nql1 = 2 ; nql2 = 3 ; nqs1 = 4 ; nqs2 = 6
  nnc1 = 7 ; nnc2 = 8
  nqc = 2 ; nqr = 3 ; nqi = 4 ; nqs = 5 ; nqg = 6 ; nnci = 7 ; nncc = 0 ; nqi2 = 0 ; nqi3 = 0
  nbudget = 10
  eqtset = eqtset_ ; efall = 0
  pdscheme = 1
  axisymm = 0
  timestats = 0
  radopt = 0
  myid = 0
  stopit = .false.
  qd_dbz = 1 ; td_mp = 0 ; qd_mp = 0 ; td_efall = 0
  qd_vtc = 0 ; qd_vtr = 0 ; qd_vts = 0 ; qd_vtg = 0 ; qd_vti = 0
  output_fallvel = 0

  ! cm1.F (double precision)
  smeps = 1.0e-60
  tsmall = 0.000000000001
  qsmall = 1.0e-24

  dx = dx_ ; dy = dy_ ; dz = dz_
  rdx = 1.0/dx ; rdy = 1.0/dy ; rdz = 1.0/dz

  call set_constants(testcase_)
end subroutine cm1f_thompson_setup

! thompson_init (as param.F calls it), with qr_acr_qg, qr_acr_qs and
! freezeH2O computed only over the zero-based km ranges given (each
! inclusive; an empty range when e < s). The tables are built once per
! process (thompson_init's allocation guard); returns 1 if this call built
! them, 0 if they already existed (and the ranges were ignored).
function cm1f_thompson_init(nt_c_, kg_s, kg_e, ks_s, ks_e, kf_s, kf_e) result(built) &
    bind(C, name="cm1f_thompson_init")
  use iso_c_binding
  use cm1f_thomcopy_dm, only: decomp_s, decomp_e, decomp_calls
  use cm1f_thomcopy, only: thompson_init, tcg_racg
  implicit none
  real(c_double), value :: nt_c_
  integer(c_int), value :: kg_s, kg_e, ks_s, ks_e, kf_s, kf_e
  integer(c_int) :: built
  real :: hgt(1:3,1:3,1:3), ntc
  built = 0
  if (.not. allocated(tcg_racg)) built = 1
  decomp_s = (/ kg_s, ks_s, kf_s /)
  decomp_e = (/ kg_e, ks_e, kf_e /)
  decomp_calls = 0
  hgt = 0.0
  ntc = nt_c_
  call thompson_init(hgt=hgt, dowr=.false., nt_c_in=ntc,              &
                     ids=1, ide=2, jds=1, jde=2, kds=1, kde=2,        &
                     ims=1, ime=3, jms=1, jme=3, kms=1, kme=3,        &
                     its=1, ite=1, jts=1, jte=1, kts=1, kte=1)
end function cm1f_thompson_init

! The constants thompson_init derives, packed in the order of
! cm1k::thompson tests' pack_consts(). Returns the count.
function cm1f_thompson_consts(buf) result(n) bind(C, name="cm1f_thompson_consts")
  use iso_c_binding
  use cm1f_thomcopy
  implicit none
  real(c_double) :: buf(*)
  integer(c_int) :: n
  n = 0
  call bridge_put1(Nt_c) ; call bridge_put1(mu_c) ; call bridge_put1(Sc3) ; call bridge_put1(D0i) ; call bridge_put1(xm0s) ; call bridge_put1(xm0g)
  call bridge_putv(cce, size(cce)) ; call bridge_putv(ccg, size(ccg)) ; call bridge_putv(ocg1, 15) ; call bridge_putv(ocg2, 15)
  call bridge_putv(cie, 7) ; call bridge_putv(cig, 7) ; call bridge_put1(oig1) ; call bridge_put1(oig2) ; call bridge_put1(obmi)
  call bridge_putv(cre, 13) ; call bridge_putv(crg, 13)
  call bridge_put1(ore1) ; call bridge_put1(org1) ; call bridge_put1(org2) ; call bridge_put1(org3) ; call bridge_put1(obmr)
  call bridge_putv(cse, 18) ; call bridge_putv(csg, 18) ; call bridge_put1(oams) ; call bridge_put1(obms) ; call bridge_put1(ocms)
  call bridge_putv(cge, 12) ; call bridge_putv(cgg, 12)
  call bridge_put1(oge1) ; call bridge_put1(ogg1) ; call bridge_put1(ogg2) ; call bridge_put1(ogg3)
  call bridge_put1(oamg) ; call bridge_put1(obmg) ; call bridge_put1(ocmg)
  call bridge_put1(t1_qr_qc) ; call bridge_put1(t1_qr_qi) ; call bridge_put1(t2_qr_qi) ; call bridge_put1(t1_qg_qc)
  call bridge_put1(t1_qs_qc) ; call bridge_put1(t1_qs_qi) ; call bridge_put1(t1_qr_ev) ; call bridge_put1(t2_qr_ev)
  call bridge_put1(t1_qs_sd) ; call bridge_put1(t2_qs_sd) ; call bridge_put1(t1_qg_sd) ; call bridge_put1(t2_qg_sd)
  call bridge_put1(t1_qs_me) ; call bridge_put1(t2_qs_me) ; call bridge_put1(t1_qg_me) ; call bridge_put1(t2_qg_me)
  call bridge_put1(dble(nic1)) ; call bridge_put1(dble(nic2)) ; call bridge_put1(dble(nii2)) ; call bridge_put1(dble(nii3))
  call bridge_put1(dble(nir2)) ; call bridge_put1(dble(nir3)) ; call bridge_put1(dble(nis2)) ; call bridge_put1(dble(nig2))
  call bridge_put1(dble(nig3)) ; call bridge_put1(dble(niIN2))
  call bridge_putv(Dc, nbc) ; call bridge_putv(dtc, nbc) ; call bridge_putv(Di, nbi) ; call bridge_putv(dti, nbi)
  call bridge_putv(Dr, nbr) ; call bridge_putv(dtr, nbr) ; call bridge_putv(Ds, nbs) ; call bridge_putv(dts, nbs)
  call bridge_putv(Dg, nbg) ; call bridge_putv(dtg, nbg) ; call bridge_putv(t_Nc, nbc)
contains
  subroutine bridge_put1(x)
    double precision, intent(in) :: x
    n = n + 1
    buf(n) = x
  end subroutine bridge_put1
  subroutine bridge_putv(a, m)
    integer, intent(in) :: m
    double precision, intent(in) :: a(m)
    buf(n+1:n+m) = a(1:m)
    n = n + m
  end subroutine bridge_putv
end function cm1f_thompson_consts

! Copy one of the module's lookup tables out (dir=0) or in (dir=1), whole,
! in Fortran order. id: 1-6 tcg_racg, tmr_racg, tcr_gacr, tmg_gacr,
! tnr_racg, tnr_gacr; 7-18 tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2,
! tcr_sacr1, tms_sacr1, tcr_sacr2, tms_sacr2, tnr_racs1, tnr_racs2,
! tnr_sacr1, tnr_sacr2; 19-24 tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz,
! tpi_qcfz, tni_qcfz; 25-27 tps_iaus, tni_iaus, tpi_ide; 28 t_Efrw;
! 29 t_Efsw; 30 tpc_wev; 31 tnc_wev. Returns the table's size.
function cm1f_thompson_table(id, buf, dir) result(n) bind(C, name="cm1f_thompson_table")
  use iso_c_binding
  use cm1f_thomcopy
  implicit none
  integer(c_int), value :: id, dir
  real(c_double) :: buf(*)
  integer(c_int) :: n
  select case (id)
  case (1) ; n = size(tcg_racg) ; call bridge_cp(tcg_racg)
  case (2) ; n = size(tmr_racg) ; call bridge_cp(tmr_racg)
  case (3) ; n = size(tcr_gacr) ; call bridge_cp(tcr_gacr)
  case (4) ; n = size(tmg_gacr) ; call bridge_cp(tmg_gacr)
  case (5) ; n = size(tnr_racg) ; call bridge_cp(tnr_racg)
  case (6) ; n = size(tnr_gacr) ; call bridge_cp(tnr_gacr)
  case (7) ; n = size(tcs_racs1) ; call bridge_cp(tcs_racs1)
  case (8) ; n = size(tmr_racs1) ; call bridge_cp(tmr_racs1)
  case (9) ; n = size(tcs_racs2) ; call bridge_cp(tcs_racs2)
  case (10) ; n = size(tmr_racs2) ; call bridge_cp(tmr_racs2)
  case (11) ; n = size(tcr_sacr1) ; call bridge_cp(tcr_sacr1)
  case (12) ; n = size(tms_sacr1) ; call bridge_cp(tms_sacr1)
  case (13) ; n = size(tcr_sacr2) ; call bridge_cp(tcr_sacr2)
  case (14) ; n = size(tms_sacr2) ; call bridge_cp(tms_sacr2)
  case (15) ; n = size(tnr_racs1) ; call bridge_cp(tnr_racs1)
  case (16) ; n = size(tnr_racs2) ; call bridge_cp(tnr_racs2)
  case (17) ; n = size(tnr_sacr1) ; call bridge_cp(tnr_sacr1)
  case (18) ; n = size(tnr_sacr2) ; call bridge_cp(tnr_sacr2)
  case (19) ; n = size(tpi_qrfz) ; call bridge_cp(tpi_qrfz)
  case (20) ; n = size(tni_qrfz) ; call bridge_cp(tni_qrfz)
  case (21) ; n = size(tpg_qrfz) ; call bridge_cp(tpg_qrfz)
  case (22) ; n = size(tnr_qrfz) ; call bridge_cp(tnr_qrfz)
  case (23) ; n = size(tpi_qcfz) ; call bridge_cp(tpi_qcfz)
  case (24) ; n = size(tni_qcfz) ; call bridge_cp(tni_qcfz)
  case (25) ; n = size(tps_iaus) ; call bridge_cp(tps_iaus)
  case (26) ; n = size(tni_iaus) ; call bridge_cp(tni_iaus)
  case (27) ; n = size(tpi_ide) ; call bridge_cp(tpi_ide)
  case (28) ; n = size(t_Efrw) ; call bridge_cp(t_Efrw)
  case (29) ; n = size(t_Efsw) ; call bridge_cp(t_Efsw)
  case (30) ; n = size(tpc_wev) ; call bridge_cp(tpc_wev)
  case (31) ; n = size(tnc_wev) ; call bridge_cp(tnc_wev)
  case default ; n = 0
  end select
contains
  subroutine bridge_cp(a)
    double precision :: a(*)
    if (dir == 0) then
      buf(1:n) = a(1:n)
    else
      a(1:n) = buf(1:n)
    endif
  end subroutine bridge_cp
end function cm1f_thompson_table

! mp_gt_driver as mp_driver calls it, on interior columns 1..ni x 1..nj,
! with getvt = .true. (the vt* outputs are mass-weighted fall speeds).
subroutine cm1f_thompson_mp_gt_driver(dt_, t3d, p, w, dz3d, q3d, rain, prate, vtc, vtr, vti, vts, vtg) &
    bind(C, name="cm1f_thompson_mp_gt_driver")
  use iso_c_binding
  use input
  use cm1f_thomcopy, only: mp_gt_driver
  implicit none
  real(c_double), value :: dt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: t3d, p, dz3d, vtc, vtr, vti, vts, vtg
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: w
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate
  real :: dt
  double precision :: b1, b2, b3, b4
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:), rr(:,:,:), pii(:,:,:), d2(:,:,:), re(:,:,:,:)
  real, allocatable :: refl(:,:,:), out3d(:,:,:,:)
  allocate(ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke), pii(ib:ie,jb:je,kb:ke))
  allocate(d2(ib:ie,jb:je,2), re(ib:ie,jb:je,kb:ke,3), refl(ib:ie,jb:je,kb:ke), out3d(1,1,1,1))
  ruh = 1.0 ; rvh = 1.0 ; rmh = 1.0 ; rr = 1.0 ; pii = 1.0 ; d2 = 0.0 ; re = 0.0 ; refl = 0.0 ; out3d = 0.0
  b1 = 0.0d0 ; b2 = 0.0d0 ; b3 = 0.0d0 ; b4 = 0.0d0
  dt = dt_
  call mp_gt_driver(qv=q3d(ib,jb,kb,1), qc=q3d(ib,jb,kb,2), qr=q3d(ib,jb,kb,3), &
                    qi=q3d(ib,jb,kb,4), qs=q3d(ib,jb,kb,5), qg=q3d(ib,jb,kb,6), &
                    ni=q3d(ib,jb,kb,7), nr=q3d(ib,jb,kb,8),                     &
                    t3d=t3d, pii=pii, p=p, w=w, dz=dz3d, dt_in=dt, itimestep=1, &
                    RAINNC=rain(ib,jb,1), RAINNCV=d2(ib,jb,1), SR=d2(ib,jb,2),  &
                    refl_10cm=refl, diagflag=.false., do_radar_ref=1,           &
                    re_cloud=re(ib,jb,kb,1), re_ice=re(ib,jb,kb,2), re_snow=re(ib,jb,kb,3), &
                    has_reqc=0, has_reqi=0, has_reqs=0,                         &
                    nrain=nrain, dx=dx, dy=dy, cm1dz=dz,                        &
                    tcond=b1, tevac=b2, tevar=b3, train=b4,                     &
                    ruh=ruh, rvh=rvh, rmh=rmh, rr=rr, rain=rain, prate=prate,   &
                    ib3d=1, ie3d=1, jb3d=1, je3d=1, kb3d=1, ke3d=1,             &
                    nout3d=1, out3d=out3d, eqtset=eqtset, getvt=.true.,         &
                    vtc=vtc, vtr=vtr, vti=vti, vts=vts, vtg=vtg,                &
                    ids=1, ide=ni+1, jds=1, jde=nj+1, kds=1, kde=nk+1,          &
                    ims=ib, ime=ie, jms=jb, jme=je, kms=kb, kme=ke,             &
                    its=1, ite=ni, jts=1, jte=nj, kts=1, kte=nk)
end subroutine cm1f_thompson_mp_gt_driver

! Full mp_driver call (ptype=3, mp_driver_module's copy built against the
! scheme copy). Budgets use ruh=rvh=rmh=1 and are discarded; with getvt the
! fall speeds come back in vt(:,:,:,1..5) = qten(:,:,:,nqc..nqg).
subroutine cm1f_thompson_mp_driver(dt_, getvt_, mh, rmh, mf, pi0, th0, w3d, rho, prs, pp3d, th3d, q3d, &
                                   rain, prate, vt) bind(C, name="cm1f_thompson_mp_driver")
  use iso_c_binding
  use input
  use cm1f_mpdrvcopy, only: mp_driver
  implicit none
  real(c_double), value :: dt_
  integer(c_int), value :: getvt_
  real(c_double), dimension(ib:ie,jb:je,kb:ke) :: mh, rmh, pi0, th0, rho, prs, pp3d, th3d
  real(c_double), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w3d
  real(c_double), dimension(ibm:iem,jbm:jem,kbm:kem,numq) :: q3d
  real(c_double), dimension(ib:ie,jb:je,nrain) :: rain
  real(c_double), dimension(ib:ie,jb:je) :: prate
  real(c_double), dimension(ib:ie,jb:je,kb:ke,5) :: vt

  real :: dt
  double precision, allocatable :: qbudget(:), asq(:), bsq(:), d2(:,:,:)
  real, allocatable, dimension(:) :: xh, ruh, xf, ruf, yh, rvh, yf, rvf
  real, allocatable, dimension(:,:,:) :: c1, c2, zh, rmf, zf, rho0, prs0, qv0
  real, allocatable, dimension(:,:,:,:) :: dum, qa, qten
  real, allocatable :: p3a(:,:,:), p3o(:,:,:,:), eff(:,:,:,:), tdiag(:,:,:,:), qdiag(:,:,:,:)
  real, allocatable :: out2d(:,:,:), out3d(:,:,:,:), tmp2d(:,:,:), tmp1d(:,:)
  logical :: getdbz, getvt
  integer :: nstep, n

  dt = dt_
  nstep = 1
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
  allocate(p3a(ibp3:iep3,kbp3:kep3,np3a), p3o(ibp3:iep3,jbp3:jep3,kbp3:kep3,np3o), eff(ibr:ier,jbr:jer,kbr:ker,6))
  allocate(tdiag(ibdt:iedt,jbdt:jedt,kbdt:kedt,ntdiag), qdiag(ibdq:iedq,jbdq:jedq,kbdq:kedq,nqdiag))
  allocate(out2d(ib2d:ie2d,jb2d:je2d,nout2d), out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  allocate(tmp2d(ibtmp:ietmp,kbtmp:ketmp,ntmp2d), tmp1d(ibtmp:ietmp,ntmp1d))
  p3a = 0.0 ; p3o = 0.0 ; eff = 0.0 ; tdiag = 0.0 ; qdiag = 0.0 ; out2d = 0.0 ; out3d = 0.0 ; tmp2d = 0.0 ; tmp1d = 0.0
  getdbz = .false. ; getvt = (getvt_ /= 0)

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

  do n = 1, 5
    vt(:,:,:,n) = qten(:,:,:,n+1)
  enddo
end subroutine cm1f_thompson_mp_driver

! Recompute one of the expensive tables (which = 1: qr_acr_qg, 2: qr_acr_qs,
! 3: freezeH2O) over another zero-based km range, after thompson_init.
subroutine cm1f_thompson_recompute(which, km_s, km_e) bind(C, name="cm1f_thompson_recompute")
  use iso_c_binding
  use cm1f_thomcopy_dm, only: decomp_s, decomp_e, decomp_calls
  use cm1f_thomcopy, only: qr_acr_qg, qr_acr_qs, freezeH2O
  implicit none
  integer(c_int), value :: which, km_s, km_e
  decomp_calls = which - 1
  decomp_s(which) = km_s
  decomp_e(which) = km_e
  select case (which)
  case (1) ; call qr_acr_qg(.false.)
  case (2) ; call qr_acr_qs(.false.)
  case (3) ; call freezeH2O(.false.)
  end select
end subroutine cm1f_thompson_recompute
