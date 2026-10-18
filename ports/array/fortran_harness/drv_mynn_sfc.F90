! Harness driver for the MYNN surface layer (src/module_sf_mynn.F,
! sfcmodel=6): mynn_sf_init_driver (psi tables), then SFCLAY_mynn called as
! turb.F's mynn_sf_j_loop does (one j row at a time, the column inputs packed
! into tmp2d(1:ni,1:nk+1,n), sequence association with ims=1,ime=ni).
!   params = [ni, nj, itimestep, isfflx, isftcflx, iz0tlnd, dx]
!   u3d, v3d, dz8w (ni,nj,2); t3d, qv3d, p3d, rho3d (ni,nj)
!   s2in (ni,nj,6): psfc, hpbl, mavail, xland, tsk, snowh
!   st (ni,nj,33) inout: ust mol qsfc znt ustm hfx qfx zol rmol wspd chs chs2
!      cqs2 cpmm regime psim psih lh flhc flqc qgh u10 v10 th2 t2 q2 gz1oz0
!      br ch ck cka cd cda
!   out: st, psi (0:1000,4): psim_stab, psim_unstab, psih_stab, psih_unstab
program drv_mynn_sfc
  use harness_io
  use constants, only: set_constants, cp, g, rovcp, r_d => rd, rv, xlv, svp1, svp2, svp3, svpt0, karman
  use module_sf_mynn, only: sfclay_mynn, mynn_sf_init_driver, psim_stab, psim_unstab, psih_stab, psih_unstab
  implicit none
  real :: params(7)
  integer :: ni, nj, nk, itimestep, isfflx, isftcflx, iz0tlnd, i, j, k
  real :: dx, ep1, ep2
  real, allocatable :: u3d(:,:,:), v3d(:,:,:), dz8w(:,:,:), t3d(:,:), qv3d(:,:), p3d(:,:), rho3d(:,:)
  real, allocatable :: s2in(:,:,:), st(:,:,:), tmp2d(:,:,:), qcg(:,:), psi(:,:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); itimestep = nint(params(3)); isfflx = nint(params(4))
  isftcflx = nint(params(5)); iz0tlnd = nint(params(6)); dx = params(7)
  nk = 4
  call set_constants(0)
  call mynn_sf_init_driver(.true.)
  allocate(u3d(ni,nj,2), v3d(ni,nj,2), dz8w(ni,nj,2), t3d(ni,nj), qv3d(ni,nj), p3d(ni,nj), rho3d(ni,nj))
  allocate(s2in(ni,nj,6), st(ni,nj,33), tmp2d(1:ni,1:nk+1,8), qcg(ni,nj), psi(0:1000,4))
  call rd('u3d', u3d, size(u3d)); call rd('v3d', v3d, size(v3d)); call rd('dz8w', dz8w, size(dz8w))
  call rd('t3d', t3d, size(t3d)); call rd('qv3d', qv3d, size(qv3d)); call rd('p3d', p3d, size(p3d))
  call rd('rho3d', rho3d, size(rho3d)); call rd('s2in', s2in, size(s2in)); call rd('st', st, size(st))
  qcg = 0.0
  ep1 = rv/r_d - 1.0
  ep2 = r_d/rv
  DO j=1,nj
    tmp2d = 0.0
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
      tmp2d(i,k,7) = 0.0
      tmp2d(i,k,8) = rho3d(i,j)
    enddo
    call SFCLAY_mynn(                                                         &
               U3D=tmp2d(1,1,1), V3D=tmp2d(1,1,2), T3D=tmp2d(1,1,3),          &
               QV3D=tmp2d(1,1,4), P3D=tmp2d(1,1,5), dz8w=tmp2d(1,1,6),        &
               th3d=tmp2d(1,1,7), rho3d=tmp2d(1,1,8),                         &
               CP=cp,G=g,ROVCP=rovcp,R=r_d,XLV=xlv,PSFCPA=s2in(1,j,1),         &
               CHS=st(1,j,11),CHS2=st(1,j,12),CQS2=st(1,j,13),CPM=st(1,j,14), &
               ZNT=st(1,j,4),UST=st(1,j,1),PBLH=s2in(1,j,2),MAVAIL=s2in(1,j,3), &
               ZOL=st(1,j,8),MOL=st(1,j,2),REGIME=st(1,j,15),PSIM=st(1,j,16),PSIH=st(1,j,17), &
               XLAND=s2in(1,j,4),HFX=st(1,j,6),QFX=st(1,j,7),LH=st(1,j,18),TSK=s2in(1,j,5), &
               FLHC=st(1,j,19),FLQC=st(1,j,20),QGH=st(1,j,21),QSFC=st(1,j,3),RMOL=st(1,j,9), &
               U10=st(1,j,22),V10=st(1,j,23),TH2=st(1,j,24),T2=st(1,j,25),Q2=st(1,j,26),SNOWH=s2in(1,j,6), &
               GZ1OZ0=st(1,j,27),WSPD=st(1,j,10),BR=st(1,j,28),ISFFLX=isfflx,DX=dx, &
               SVP1=svp1,SVP2=svp2,SVP3=svp3,SVPT0=svpt0,EP1=ep1,EP2=ep2,     &
               KARMAN=karman,itimestep=itimestep,                             &
               ch=st(1,j,29),                                                 &
               qcg=qcg(1,j),                                                  &
               spp_pbl=0,                                                     &
               ids=1  ,ide=ni+1 , jds=1  ,jde=2    , kds=1  ,kde=nk+1 ,       &
               ims=1  ,ime=ni   , jms=1  ,jme=1    , kms=1  ,kme=nk ,         &
               its=1  ,ite=ni   , jts=1  ,jte=1    , kts=1  ,kte=nk  ,        &
               ustm=st(1,j,5),ck=st(1,j,30),cka=st(1,j,31),cd=st(1,j,32),cda=st(1,j,33), &
               isftcflx=isftcflx,iz0tlnd=iz0tlnd)
  ENDDO
  psi(:,1) = psim_stab; psi(:,2) = psim_unstab; psi(:,3) = psih_stab; psi(:,4) = psih_unstab
  call wr('st', st, size(st))
  call wr('psi', psi, size(psi))
end program drv_mynn_sfc
