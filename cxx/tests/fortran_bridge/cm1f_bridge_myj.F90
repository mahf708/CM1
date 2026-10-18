! bind(C) shims around the MYJ surface layer (sfcmodel=7) and PBL (ipbl=6):
! module_sf_myjsfc's MYJSFCINIT (init_physics.F) and MYJSFC, and
! module_bl_myjpbl's MYJPBL, called one j row at a time with the arguments
! turb.F passes (sfc_and_turb lines 932-1013 and 2567-2640), on packed
! arrays. No `input` module state.
!
! Packed arrays (must match myj_tests.cpp); 3D arrays are (1:ni,1:nk+1,1:nj,n),
! i.e. WRF's (ims:ime,kms:kme,jms:jme) one row after the other; 2D arrays are
! (1:ni,1:nj,n):
!   MYJSFC  a3(...,10): 1 dz 2 pmid 3 pint 4 th 5 t 6 qv 7 qc 8 u 9 v 10 q2
!           s2(...,46): 1 ht 2 tsk 3 qsfc 4 thz0 5 qz0 6 uz0 7 vz0 8 xland
!             9 ustar 10 znt 11 z0base 12 pblh 13 mavail 14 rmol 15 akhs
!             16 akms 17 rib 18 chs 19 chs2 20 cqs2 21 hfx 22 qfx 23 flx_lh
!             24 flhc 25 flqc 26 qgh 27 cpm 28 ct 29 u10 30 v10 31 t02
!             32 th02 33 tshltr 34 th10 35 q02 36 qshltr 37 q10 38 pshltr
!             39 u10e 40 v10e 41 psim 42 psih 43 fm 44 fhh 45 z0out 46 rhosfc
!   MYJPBL  a3(...,12): 1 dz 2 pmid 3 pint 4 th 5 t 6 exner 7 qv 8 qcw 9 qci
!             10 u 11 v 12 rho
!           o3(...,10): 1 tke_myj 2 exch_h 3 exch_m 4 el_myj 5 rublten
!             6 rvblten 7 rthblten 8 rqvblten 9 rqcblten 10 rqiblten
!           s2(...,18): 1 ht 2 tsk 3 qsfc 4 chklowq 5 thz0 6 qz0 7 uz0 8 vz0
!             9 xland 10 ustar 11 znt 12 pblh 13 kpbl (as real) 14 ct
!             15 akhs 16 akms 17 elflx 18 mixht

! MYJSFCINIT (RESTART = .false.): the psi tables (module variables of
! module_sf_myjsfc). LOWLYR and USTAR are set on a dummy 1x1 grid.
subroutine cm1f_myjsfcinit() bind(C, name="cm1f_myjsfcinit")
  use iso_c_binding
  use module_sf_myjsfc, only: MYJSFCINIT
  implicit none
  integer :: lowlyr(1,1), ivgtyp(1,1)
  real :: ust(1,1), znt(1,1), xland(1,1), sice(1,1)
  lowlyr = 1; ivgtyp = 0; ust = 0.0; znt = 0.0; xland = 2.0; sice = 0.0
  call MYJSFCINIT(LOWLYR=lowlyr,USTAR=ust,Z0=znt                    &
                 ,SEAMASK=xland,XICE=sice,IVGTYP=ivgtyp,RESTART=.false. &
                 ,ALLOWED_TO_READ=.false.                           &
                 ,ids=1 ,ide=2 , jds=1 ,jde=2 , kds=1 ,kde=2 ,          &
                  ims=1 ,ime=1 , jms=1 ,jme=1 , kms=1 ,kme=2 ,          &
                  its=1 ,ite=1 , jts=1 ,jte=1 , kts=1 ,kte=1 )
end subroutine cm1f_myjsfcinit

subroutine cm1f_myjsfc(ni, nj, nk, itimestep, iusemyjpbl, isftcflx, a3, s2) bind(C, name="cm1f_myjsfc")
  use iso_c_binding
  use module_sf_myjsfc, only: MYJSFC
  implicit none
  integer(c_int), value :: ni, nj, nk, itimestep, iusemyjpbl, isftcflx
  real(c_double), intent(in) :: a3(1:ni,1:nk+1,1:nj,10)
  real(c_double), intent(inout) :: s2(1:ni,1:nj,46)
  integer :: lowlyr(1:ni,1), ivgtyp(1:ni,1), j
  logical :: usemyjpbl
  real :: p1000mb
  lowlyr = 1; ivgtyp = 0
  usemyjpbl = (iusemyjpbl .eq. 1)
  p1000mb = 100000.0
  do j=1,nj
          call   MYJSFC(ITIMESTEP=itimestep,  &
                        HT=s2(1,j,1),  &
                        DZ=a3(1,1,j,1),                               &
                        PMID=a3(1,1,j,2),  &
                        PINT=a3(1,1,j,3),  &
                        TH=a3(1,1,j,4),  &
                        T=a3(1,1,j,5),  &
                        QV=a3(1,1,j,6),  &
                        QC=a3(1,1,j,7),  &
                        U=a3(1,1,j,8),  &
                        V=a3(1,1,j,9),  &
                        Q2=a3(1,1,j,10)                    &
                       ,TSK=s2(1,j,2),QSFC=s2(1,j,3),THZ0=s2(1,j,4),QZ0=s2(1,j,5),UZ0=s2(1,j,6),VZ0=s2(1,j,7)     &
                       ,LOWLYR=lowlyr,XLAND=s2(1,j,8),IVGTYP=ivgtyp,ISURBAN=0,IZ0TLND=0            &
                       ,USTAR=s2(1,j,9),ZNT=s2(1,j,10),Z0BASE=s2(1,j,11),PBLH=s2(1,j,12),MAVAIL=s2(1,j,13),RMOL=s2(1,j,14) &
                       ,AKHS=s2(1,j,15),AKMS=s2(1,j,16)                                      &
                       ,RIB=s2(1,j,17)                                             &
                       ,CHS=s2(1,j,18),CHS2=s2(1,j,19),CQS2=s2(1,j,20),HFX=s2(1,j,21),QFX=s2(1,j,22),FLX_LH=s2(1,j,23) &
                       ,FLHC=s2(1,j,24),FLQC=s2(1,j,25)         &
                       ,QGH=s2(1,j,26),CPM=s2(1,j,27),CT=s2(1,j,28)                                     &
                       ,U10=s2(1,j,29),V10=s2(1,j,30),T02=s2(1,j,31),TH02=s2(1,j,32),TSHLTR=s2(1,j,33),TH10=s2(1,j,34) &
                       ,Q02=s2(1,j,35),QSHLTR=s2(1,j,36),Q10=s2(1,j,37),PSHLTR=s2(1,j,38)          &
                       ,P1000mb=p1000mb,U10E=s2(1,j,39),V10E=s2(1,j,40),                             &
                        psim=s2(1,j,41),psih=s2(1,j,42),fm=s2(1,j,43),fhh=s2(1,j,44),z0out=s2(1,j,45), &
                        usemyjpbl=usemyjpbl,rhosfc=s2(1,j,46),isftcflx=isftcflx,     &
                     ids=1  ,ide=ni+1 , jds= 1 ,jde= 2   , kds=1  ,kde=nk+1 ,                                   &
                     ims=1  ,ime=ni   , jms= 1 ,jme= 1   , kms=1  ,kme=nk+1 ,                                     &
                     its=1  ,ite=ni   , jts= 1 ,jte= 1   , kts=1  ,kte=nk  )
  enddo
end subroutine cm1f_myjsfc

subroutine cm1f_myjpbl(ni, nj, nk, dt, a3, o3, s2) bind(C, name="cm1f_myjpbl")
  use iso_c_binding
  use module_bl_myjpbl, only: MYJPBL
  implicit none
  integer(c_int), value :: ni, nj, nk
  real(c_double), value :: dt
  real(c_double), intent(in) :: a3(1:ni,1:nk+1,1:nj,12)
  real(c_double), intent(inout) :: o3(1:ni,1:nk+1,1:nj,10)
  real(c_double), intent(inout) :: s2(1:ni,1:nj,18)
  integer :: lowlyr(1:ni,1), kpbl(1:ni,1), j
  real :: sice(1:ni,1), snow(1:ni,1)
  real, allocatable :: z3(:,:,:)
  allocate(z3(1:ni,1:nk+1,6))
  lowlyr = 1; sice = 0.0; snow = 0.0
  do j=1,nj
    z3 = 0.0
    kpbl(:,1) = nint(s2(:,j,13))
        call     MYJPBL(dt=DT,STEPBL=1,HT=s2(1,j,1),DZ=a3(1,1,j,1)                     &
                       ,PMID=a3(1,1,j,2),PINT=a3(1,1,j,3),TH=a3(1,1,j,4),T=a3(1,1,j,5),EXNER=a3(1,1,j,6)  &
                       ,QV=a3(1,1,j,7),QCW=a3(1,1,j,8),QCI=a3(1,1,j,9),QCS=z3(1,1,1),QCR=z3(1,1,2),QCG=z3(1,1,3)    &
                       ,U=a3(1,1,j,10),V=a3(1,1,j,11),RHO=a3(1,1,j,12),TSK=s2(1,j,2),QSFC=s2(1,j,3),CHKLOWQ=s2(1,j,4) &
                       ,THZ0=s2(1,j,5),QZ0=s2(1,j,6),UZ0=s2(1,j,7),VZ0=s2(1,j,8)      &
                       ,LOWLYR=lowlyr,XLAND=s2(1,j,9),SICE=sice,SNOW=snow                         &
                       ,TKE_MYJ=o3(1,1,j,1),EXCH_H=o3(1,1,j,2),EXCH_M=o3(1,1,j,3),USTAR=s2(1,j,10),ZNT=s2(1,j,11) &
                       ,EL_MYJ=o3(1,1,j,4),PBLH=s2(1,j,12),KPBL=kpbl,CT=s2(1,j,14)   &
                       ,AKHS=s2(1,j,15),AKMS=s2(1,j,16),ELFLX=s2(1,j,17),MIXHT=s2(1,j,18)                          &
                       ,RUBLTEN=o3(1,1,j,5),RVBLTEN=o3(1,1,j,6),RTHBLTEN=o3(1,1,j,7),RQVBLTEN=o3(1,1,j,8),RQCBLTEN=o3(1,1,j,9)     &
                       ,RQIBLTEN=o3(1,1,j,10),RQSBLTEN=z3(1,1,4),RQRBLTEN=z3(1,1,5),RQGBLTEN=z3(1,1,6),           &
                  ids=1  ,ide=ni+1 , jds=1  ,jde=2    , kds=1  ,kde=nk+1 ,          &
                  ims=1 ,ime=ni      , jms=1  ,jme=1    , kms=1, kme=nk+1  ,            &
                  its=1  ,ite=ni   , jts=1  ,jte=1    , kts=1  ,kte=nk )
    s2(:,j,13) = real(kpbl(:,1))
  enddo
  deallocate(z3)
end subroutine cm1f_myjpbl
