! bind(C) shim around the GFS-EDMF PBL (ipbl=3): one call of
! module_bl_gfsedmf's BL_GFSEDMF on the whole (1..ni,1..nj) tile, with the
! arguments turb.F (sfc_and_turb, lines 2123-2160) passes, but for packed
! arrays. The diagnostics that never feed back (hpbl2d, evap2d, heat2d, brcr,
! wscale, wscaleu, prkpp, stau) go to scratch arrays.
!
! Packed arrays (must match gfs_tests.cpp):
!   a3d(ib:ie,jb:je,kb:ke,12) inputs: 1 dum1(u) 2 dum2(v) 3 dum3(th) 4 dum4(T)
!       5 divx(qv) 6 ppten(qc) 7 dum8(qi) 8 prs 9 dum7(pi) 10 dum5(dz8w) 11 zf
!       12 dum9(rthraten)
!   s2d(ib:ie,jb:je,16): 1 psfc 2 ust 3 hpbl 4 psim 5 psih 6 hfx 7 qfx 8 tsk
!       9 gz1oz0 10 wspd 11 br 12 kpbl2d (as real) 13 u10 14 v10 15 znt 16 xland
!   o3d(ib:ie,jb:je,kb:ke,10) outputs: 1 upten 2 vpten 3 thpten 4 qvpten
!       5 qcpten 6 qipten 7 xkzm 8 xkzh 9 dissten 10 dheat (thten)
!   rc(7): cp g rovcp rd rovg ep1 karman
!   rp(4): var_ric coef_ric_l coef_ric_s alpha
!   ic(3): p_qi p_first_scalar disheat

subroutine cm1f_gfsedmf(ni, nj, nk, ib, ie, jb, je, kb, ke, dt, rc, rp, ic, a3d, s2d, o3d) &
    bind(C, name="cm1f_gfsedmf")
  use iso_c_binding
  use module_bl_gfsedmf, only: bl_gfsedmf
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke
  real(c_double), value :: dt
  real(c_double), intent(in) :: rc(7), rp(4)
  integer(c_int), intent(in) :: ic(3)
  real(c_double), intent(in) :: a3d(ib:ie,jb:je,kb:ke,12)
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,16)
  real(c_double), intent(inout) :: o3d(ib:ie,jb:je,kb:ke,10)
  real, allocatable, dimension(:,:) :: hpbl2d, evap2d, heat2d, brcr, wscale, wscaleu, prkpp, stau
  integer, allocatable :: kpbl2d(:,:)
  real :: cp, g, rovcp, rd, rovg, ep1, karman, var_ric, coef_ric_l, coef_ric_s, gfs_alpha, ens_pblamp
  integer :: p_qi, p_first_scalar, ens_random_seed
  logical :: disheat, pert_pbl

  cp = rc(1); g = rc(2); rovcp = rc(3); rd = rc(4); rovg = rc(5); ep1 = rc(6); karman = rc(7)
  var_ric = rp(1); coef_ric_l = rp(2); coef_ric_s = rp(3); gfs_alpha = rp(4)
  p_qi = ic(1); p_first_scalar = ic(2); disheat = (ic(3).eq.1)
  pert_pbl = .false.
  ens_random_seed = 99
  ens_pblamp = 0.2
  allocate(hpbl2d(ib:ie,jb:je), evap2d(ib:ie,jb:je), heat2d(ib:ie,jb:je), brcr(ib:ie,jb:je), &
           wscale(ib:ie,jb:je), wscaleu(ib:ie,jb:je), prkpp(ib:ie,jb:je), stau(ib:ie,jb:je),  &
           kpbl2d(ib:ie,jb:je))
  hpbl2d = 0.0; evap2d = 0.0; heat2d = 0.0; brcr = 0.0; wscale = 0.0; wscaleu = 0.0
  prkpp = 0.0; stau = 0.0
  kpbl2d = nint(s2d(:,:,12))

        call  BL_GFSEDMF(U3D=a3d(:,:,:,1),V3D=a3d(:,:,:,2),TH3D=a3d(:,:,:,3),T3D=a3d(:,:,:,4),QV3D=a3d(:,:,:,5), &
                  QC3D=a3d(:,:,:,6),QI3D=a3d(:,:,:,7),P3D=a3d(:,:,:,8),PI3D=a3d(:,:,:,9),   &
                  RUBLTEN=o3d(:,:,:,1),RVBLTEN=o3d(:,:,:,2),RTHBLTEN=o3d(:,:,:,3),          &
                  RQVBLTEN=o3d(:,:,:,4),RQCBLTEN=o3d(:,:,:,5),RQIBLTEN=o3d(:,:,:,6),        &
                  CP=cp,G=g,ROVCP=rovcp,R=rd,ROVG=rovg,                             &
                  P_QI=p_qi,P_FIRST_SCALAR=p_first_scalar,                          &
                  dz8w=a3d(:,:,:,10),z=a3d(:,:,:,11),PSFC=s2d(:,:,1),               &
                  UST=s2d(:,:,2),PBL=s2d(:,:,3),PSIM=s2d(:,:,4),PSIH=s2d(:,:,5),    &
                  HFX=s2d(:,:,6),QFX=s2d(:,:,7),TSK=s2d(:,:,8),GZ1OZ0=s2d(:,:,9),   &
                  WSPD=s2d(:,:,10),BR=s2d(:,:,11),                                  &
                  DT=dt,KPBL2D=kpbl2d,EP1=ep1,KARMAN=karman,                        &
                  DISHEAT=disheat,                                                  &
                  RTHRATEN=a3d(:,:,:,12),                                           &
                  HPBL2D=hpbl2d, EVAP2D=evap2d, HEAT2D=heat2d,                      &
                  U10=s2d(:,:,13),V10=s2d(:,:,14),ZNT=s2d(:,:,15),                  &
                  DKU3D=o3d(:,:,:,7),DKT3D=o3d(:,:,:,8),                            &
                  VAR_RIC=var_ric,coef_ric_l=coef_ric_l,coef_ric_s=coef_ric_s,      &
                  alpha=gfs_alpha,xland=s2d(:,:,16), pert_pbl=pert_pbl,             &
                  ens_random_seed=ens_random_seed, ens_pblamp=ens_pblamp,           &
                  brcr=brcr,wscale=wscale,wscaleu=wscaleu,prkpp=prkpp,              &
                  stau=stau,dissten=o3d(:,:,:,9),dheat=o3d(:,:,:,10),               &
                  ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=nk+1 ,          &
                  ims=ib ,ime=ie   , jms=jb ,jme=je   , kms=kb ,kme=ke ,            &
                  its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=nk              )

  s2d(:,:,12) = kpbl2d
end subroutine cm1f_gfsedmf
