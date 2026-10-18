! bind(C) shim around the GFDL surface layer (sfcmodel=4): one call of
! module_sf_gfdl's SF_GFDL on the whole (1..ni,1..nj) tile, with the
! arguments turb.F (sfc_and_turb, lines 857-875) passes, but for packed
! arrays and the soil moisture: SF_GFDL reads smois(i,1,j) of a dummy
! declared (ims:ime,jms:jme,1:num_soil_layers), which is out of bounds for
! j > 1 with CM1's 2D smois and num_soil_layers = 1 (see
! cxx/src/cm1k/physics/sfc_gfdl/cm1k_sfc_gfdl.hpp). Here smois is
! (ib:ie,jb:je,1:nj) with smois(i,1,j) = the packed value at (i,j).
!
! Packed arrays (must match gfdl_tests.cpp):
!   a3d(ib:ie,jb:je,kb:ke,5) inputs: 1 dum1(u) 2 dum2(v) 3 dum4(T) 4 divx(qv) 5 prs
!   s2d(ib:ie,jb:je,37): 1 psfc 2 xland 3 gsw 4 glw 5 zkmax 6 smois 7 tsk 8 znt
!       9 mznt 10 ust 11 u10 12 v10 13 chs 14 chs2 15 cqs2 16 cpmm 17 psim 18 psih
!       19 hfx 20 qfx 21 lh 22 taux 23 tauy 24 flhc 25 flqc 26 qgh 27 qsfc
!       28 gz1oz0 29 swspd 30 br 31 cd_out 32 ch_out 33 mol 34 zol 35 z0t 36 fm 37 fh
!   rc(7): cp rovcp rd xlv ep1 ep2 karman
!   ic(4): ntsflg isfflx icoef_sf isltyp

subroutine cm1f_sf_gfdl(ni, nj, nk, ib, ie, jb, je, kb, ke, dt, sfenth, rc, ic, a3d, s2d) &
    bind(C, name="cm1f_sf_gfdl")
  use iso_c_binding
  use module_sf_gfdl, only: sf_gfdl
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke
  real(c_double), value :: dt, sfenth
  real(c_double), intent(in) :: rc(7)
  integer(c_int), intent(in) :: ic(4)
  real(c_double), intent(in) :: a3d(ib:ie,jb:je,kb:ke,5)
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,37)
  real, allocatable :: smois(:,:,:), charn(:,:), msang(:,:), scurx(:,:), scury(:,:)
  integer, allocatable :: isltyp(:,:)
  real :: cp, rovcp, rd, xlv, ep1, ep2, karman, ens_cdamp
  integer :: ntsflg, isfflx, icoef_sf, iwavecpl, ens_random_seed
  logical :: lcurr_sf, pert_cd
  integer :: i, j

  cp = rc(1); rovcp = rc(2); rd = rc(3); xlv = rc(4); ep1 = rc(5); ep2 = rc(6); karman = rc(7)
  ntsflg = ic(1); isfflx = ic(2); icoef_sf = ic(3)
  allocate(smois(ib:ie,jb:je,1:nj), isltyp(ib:ie,jb:je), charn(ib:ie,jb:je), msang(ib:ie,jb:je), &
           scurx(ib:ie,jb:je), scury(ib:ie,jb:je))
  smois = 0.0
  do j = 1, nj
    do i = ib, ie
      smois(i,1,j) = s2d(i,j,6)
    enddo
  enddo
  isltyp = ic(4)
  charn = 0.0185
  msang = 0.0
  scurx = 0.0
  scury = 0.0
  ! turb.F 484-500
  pert_cd = .false.
  ens_random_seed = 99
  ens_cdamp = 0.2
  iwavecpl = 0
  lcurr_sf = .false.

          call SF_GFDL(U3D=a3d(:,:,:,1),V3D=a3d(:,:,:,2),T3D=a3d(:,:,:,3),QV3D=a3d(:,:,:,4),P3D=a3d(:,:,:,5), &
                     CP=cp,ROVCP=rovcp,R=rd,XLV=xlv,PSFC=s2d(:,:,1),                           &
                     CHS=s2d(:,:,13),CHS2=s2d(:,:,14),CQS2=s2d(:,:,15), CPM=s2d(:,:,16),       &
                     DT=dt, SMOIS=smois,num_soil_layers=nj,ISLTYP=ISLTYP,ZNT=s2d(:,:,8),        &
                     MZNT=s2d(:,:,9),                                                          &
                     UST=s2d(:,:,10),PSIM=s2d(:,:,17),PSIH=s2d(:,:,18),                        &
                     XLAND=s2d(:,:,2),HFX=s2d(:,:,19),QFX=s2d(:,:,20),TAUX=s2d(:,:,22),        &
                     TAUY=s2d(:,:,23),LH=s2d(:,:,21),GSW=s2d(:,:,3),GLW=s2d(:,:,4),            &
                     TSK=s2d(:,:,7),FLHC=s2d(:,:,24),FLQC=s2d(:,:,25),                         &
                     QGH=s2d(:,:,26),QSFC=s2d(:,:,27),U10=s2d(:,:,11),V10=s2d(:,:,12),         &
                     ICOEF_SF=icoef_sf,IWAVECPL=iwavecpl,LCURR_SF=lcurr_sf,                    &
                     CHARN=charn,MSANG=msang,SCURX=scurx, SCURY=scury,                         &
                     pert_Cd=pert_cd, ens_random_seed=ens_random_seed, ens_Cdamp=ens_cdamp,    &
                     GZ1OZ0=s2d(:,:,28),WSPD=s2d(:,:,29),BR=s2d(:,:,30),ZKMAX=s2d(:,:,5),      &
                     ISFFLX=isfflx,                                                            &
                     EP1=ep1,EP2=ep2,KARMAN=karman,NTSFLG=ntsflg,SFENTH=sfenth,                &
                     Cd_out=s2d(:,:,31),Ch_out=s2d(:,:,32),mol_out=s2d(:,:,33),                &
                     zol_out=s2d(:,:,34),z0t_out=s2d(:,:,35),                                  &
                     fm_out=s2d(:,:,36),fh_out=s2d(:,:,37),                                    &
                  ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=nk+1 ,                     &
                  ims=ib ,ime=ie   , jms=jb ,jme=je   , kms=kb ,kme=ke ,                       &
                  its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=nk  )

end subroutine cm1f_sf_gfdl
