! bind(C) shims around CM1's MYNN surface layer (sfcmodel=6),
! src/module_sf_mynn.F:
!   mynn_sf_init_driver   (init_physics.F) fills the psi lookup tables
!   SFCLAY_mynn           called exactly as turb.F's mynn_sf_j_loop does:
!                         one j row at a time, the column inputs packed into
!                         tmp2d(1:ni,1:nk+1,n) and passed by sequence
!                         association with ims=1,ime=ni,kms=1,kme=nk
! The constants come from CM1's `constants` module (set_constants(0)), and
! ep1/ep2 are computed as turb.F does.

subroutine cm1f_mynn_sf_init() bind(C, name="cm1f_mynn_sf_init")
  use iso_c_binding
  use constants, only: set_constants
  use module_sf_mynn, only: mynn_sf_init_driver
  implicit none
  call set_constants(0)
  call mynn_sf_init_driver(.true.)
end subroutine cm1f_mynn_sf_init

subroutine cm1f_mynn_psi_tables(ms, mu, hs, hu) bind(C, name="cm1f_mynn_psi_tables")
  use iso_c_binding
  use module_sf_mynn, only: psim_stab, psim_unstab, psih_stab, psih_unstab
  implicit none
  real(c_double), intent(out), dimension(0:1000) :: ms, mu, hs, hu
  ms = psim_stab; mu = psim_unstab; hs = psih_stab; hu = psih_unstab
end subroutine cm1f_mynn_psi_tables

! 2D arrays are (1:ni,1:nj); u3d, v3d, dz8w are (1:ni,1:nj,1:2) (k=1,2);
! t3d, qv3d, p3d, rho3d are (1:ni,1:nj) (k=1).
subroutine cm1f_mynn_sfclay(ni, nj, nk, itimestep, isfflx, isftcflx, iz0tlnd, dx, &
                            u3d, v3d, dz8w, t3d, qv3d, p3d, rho3d, &
                            psfc, hpbl, mavail, xland, tsk, snowh, &
                            ust, mol, qsfc, znt, ustm, hfx, qfx, zol, rmol, wspd, &
                            chs, chs2, cqs2, cpmm, regime, psim, psih, lh, flhc, flqc, qgh, &
                            u10, v10, th2, t2, q2, gz1oz0, br, ch, ck, cka, cd, cda) &
    bind(C, name="cm1f_mynn_sfclay")
  use iso_c_binding
  use constants, only: cp, g, rovcp, rd, rv, xlv, svp1, svp2, svp3, svpt0, karman
  use module_sf_mynn, only: sfclay_mynn
  implicit none
  integer(c_int), value :: ni, nj, nk, itimestep, isfflx, isftcflx, iz0tlnd
  real(c_double), value :: dx
  real(c_double), intent(in), dimension(ni,nj,2) :: u3d, v3d, dz8w
  real(c_double), intent(in), dimension(ni,nj) :: t3d, qv3d, p3d, rho3d
  real(c_double), intent(in), dimension(ni,nj) :: psfc, hpbl, mavail, xland, tsk, snowh
  real(c_double), intent(inout), dimension(ni,nj) :: ust, mol, qsfc, znt, ustm, hfx, qfx, zol, rmol, wspd, &
      chs, chs2, cqs2, cpmm, regime, psim, psih, lh, flhc, flqc, qgh, u10, v10, th2, t2, q2, gz1oz0, br, &
      ch, ck, cka, cd, cda
  real, allocatable :: tmp2d(:,:,:), qcg(:,:)
  real :: ep1, ep2
  integer :: i, j, k
  allocate(tmp2d(1:ni,1:nk+1,8), qcg(ni,nj))
  tmp2d = 0.0
  qcg = 0.0
  ep1 = rv/rd - 1.0
  ep2 = rd/rv
  DO j=1,nj
    do k=1,2
    do i=1,ni
      tmp2d(i,k,1) = u3d(i,j,k)
      tmp2d(i,k,2) = v3d(i,j,k)
      tmp2d(i,k,6) = dz8w(i,j,k)
    enddo
    enddo
    k = 1
    do i=1,ni
      tmp2d(i,k,3) = t3d(i,j)
      tmp2d(i,k,4) = qv3d(i,j)
      tmp2d(i,k,5) = p3d(i,j)
      tmp2d(i,k,7) = 0.0          ! th3d: not used by SFCLAY1D_mynn
      tmp2d(i,k,8) = rho3d(i,j)
    enddo
    call SFCLAY_mynn(                                                         &
               U3D=tmp2d(1,1,1), V3D=tmp2d(1,1,2), T3D=tmp2d(1,1,3),          &
               QV3D=tmp2d(1,1,4), P3D=tmp2d(1,1,5), dz8w=tmp2d(1,1,6),        &
               th3d=tmp2d(1,1,7), rho3d=tmp2d(1,1,8),                         &
               CP=cp,G=g,ROVCP=rovcp,R=rd,XLV=xlv,PSFCPA=psfc(1,j),           &
               CHS=chs(1,j),CHS2=chs2(1,j),CQS2=cqs2(1,j),CPM=cpmm(1,j),      &
               ZNT=znt(1,j),UST=ust(1,j),PBLH=hpbl(1,j),MAVAIL=mavail(1,j),   &
               ZOL=zol(1,j),MOL=mol(1,j),REGIME=regime(1,j),PSIM=psim(1,j),PSIH=psih(1,j), &
               XLAND=xland(1,j),HFX=hfx(1,j),QFX=qfx(1,j),LH=lh(1,j),TSK=tsk(1,j), &
               FLHC=flhc(1,j),FLQC=flqc(1,j),QGH=qgh(1,j),QSFC=qsfc(1,j),RMOL=rmol(1,j), &
               U10=u10(1,j),V10=v10(1,j),TH2=th2(1,j),T2=t2(1,j),Q2=q2(1,j),SNOWH=snowh(1,j), &
               GZ1OZ0=gz1oz0(1,j),WSPD=wspd(1,j),BR=br(1,j),ISFFLX=isfflx,DX=dx, &
               SVP1=svp1,SVP2=svp2,SVP3=svp3,SVPT0=svpt0,EP1=ep1,EP2=ep2,     &
               KARMAN=karman,itimestep=itimestep,                             &
               ch=ch(1,j),                                                    &
               qcg=qcg(1,j),                                                  &
               spp_pbl=0,                                                     &
               ids=1  ,ide=ni+1 , jds=1  ,jde=2    , kds=1  ,kde=nk+1 ,       &
               ims=1  ,ime=ni   , jms=1  ,jme=1    , kms=1  ,kme=nk ,         &
               its=1  ,ite=ni   , jts=1  ,jte=1    , kts=1  ,kte=nk  ,        &
               ustm=ustm(1,j),ck=ck(1,j),cka=cka(1,j),cd=cd(1,j),cda=cda(1,j), &
               isftcflx=isftcflx,iz0tlnd=iz0tlnd)
  ENDDO
end subroutine cm1f_mynn_sfclay

! turb.F after SFCLAY_mynn (update_sfc = .false.), verbatim: qsfc diagnosis,
! ch/cq/s10, sfcmodel=6 ch=cq=ck, thflux/qvflux. All arrays (1:ni,1:nj).
subroutine cm1f_mynn_sfc_finish(ni, nj, dosfcflx_in, divx, rf, qsfc, qfx, flqc, chs2, cqs2, ck, u10, v10, &
                                hfx, cpmm, ch, cq, s10, thflux, qvflux) bind(C, name="cm1f_mynn_sfc_finish")
  use iso_c_binding
  implicit none
  integer(c_int), value :: ni, nj, dosfcflx_in
  real(c_double), intent(in), dimension(ni,nj) :: divx, rf, qfx, flqc, chs2, cqs2, ck, u10, v10, hfx, cpmm
  real(c_double), intent(inout), dimension(ni,nj) :: qsfc, ch, cq, s10, thflux, qvflux
  integer :: i, j
  real :: qx
  logical :: dosfcflx
  dosfcflx = (dosfcflx_in .ne. 0)
          do j=1,nj
          do i=1,ni
            qx = divx(i,j)
            if ( FLQC(i,j) .ne. 0.) then
               QSFC(i,j)=QX+QFX(i,j)/FLQC(i,j)
            else
               QSFC(i,j) = QX
            end if
          enddo
          enddo
            do j=1,nj
            do i=1,ni
              ch(i,j) = chs2(i,j)
              cq(i,j) = cqs2(i,j)
              s10(i,j) = sqrt( u10(i,j)**2 + v10(i,j)**2 )
            enddo
            enddo
            do j=1,nj
            do i=1,ni
              ch(i,j) = ck(i,j)
              cq(i,j) = ck(i,j)
            enddo
            enddo
          IF( dosfcflx )THEN
            do j=1,nj
            do i=1,ni
              thflux(i,j) = hfx(i,j)/(CPMM(i,j)*rf(i,j))
              qvflux(i,j) = qfx(i,j)/rf(i,j)
            enddo
            enddo
          ENDIF
end subroutine cm1f_mynn_sfc_finish
