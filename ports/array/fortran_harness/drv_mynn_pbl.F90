! Harness driver for the MYNN-EDMF PBL (ipbl=4/5): the domynn block of
! src/turb.F, one j row at a time, mynnedmf_wrapper_run called with turb.F's
! arguments (tmp2d packing, sequence association with ims=1,ime=ni).
! Adapted from the cm1k bridge (cxx/tests/fortran_bridge/cm1f_bridge_mynn_pbl.F90).
!   params = [ni, nj, nk, nstep, ipbl, ptype, edmf, edmf_mom, dt, dx, rthr]
!   in3 (ni,nj,nk,12): dz,u,v,w,th,qv,qc,qi,p,exner,rho,t3d
!   st3 (ni,nj,nk,26) inout: qke,qke_adv,sh3d,sm3d,tsq,qsq,cov,el_pbl,qc_bl,
!       qi_bl,cldfra_bl,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,
!       sub_thl3d,sub_sqv3d,det_thl3d,det_sqv3d,dqke,qwt,qshear,qbuoy,qdiss
!   te3 (ni,nj,nk,9) out: upten,vpten,thpten,qvpten,qcpten,qipten,xkzh,xkzm,dheat
!   s2 (ni,nj,16) inout: znt,xland,tsk,qsfc,psfc,ust,ch,hfx,qfx,wspd,rmol,hpbl,
!       kpbl,nupdraft,maxmf,ktop_plume
program drv_mynn_pbl
  use harness_io
  use module_bl_mynn_wrapper, only: mynnedmf_wrapper_run
  implicit none
  real :: params(11)
  integer :: ni, nj, nk, nstep, ipbl, ptype, edmf, edmf_mom
  real :: dt, dx, rthr
  real, allocatable :: in3(:,:,:,:), st3(:,:,:,:), te3(:,:,:,:), s2(:,:,:)
  real, allocatable :: tmp2d(:,:,:), tmp1d(:,:)
  real, allocatable, dimension(:,:) :: znt, xland, tsk, qsfc, psfc, ust, ch, hfx, qfx, wspd, rmol, hpbl, maxmf
  integer, allocatable, dimension(:,:) :: kpbl2d, nupdraft, ktop_plume
  integer :: i, j, k, n, initflag
  logical :: restart, cycling
  logical :: FLAG_QI, FLAG_QNI, FLAG_QC, FLAG_QNC, FLAG_QNWFA, FLAG_QNIFA
  real :: bl_mynn_closure
  integer, parameter :: bl_mynn_tkebudget = 1, bl_mynn_cloudpdf = 2, bl_mynn_mixlength = 1, icloud_bl = 1, &
                        bl_mynn_edmf_tke = 0, bl_mynn_cloudmix = 1, bl_mynn_mixqt = 0, bl_mynn_output = 1, &
                        bl_mynn_mixscalars = 0, spp_pbl = 0
  logical, parameter :: bl_mynn_tkeadvect = .true.
  integer :: bl_mynn_edmf, bl_mynn_edmf_mom

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3)); nstep = nint(params(4))
  ipbl = nint(params(5)); ptype = nint(params(6)); edmf = nint(params(7)); edmf_mom = nint(params(8))
  dt = params(9); dx = params(10); rthr = params(11)
  allocate(in3(ni,nj,nk,12), st3(ni,nj,nk,26), te3(ni,nj,nk,9), s2(ni,nj,16))
  te3 = 0.0
  call rd('in3', in3, size(in3)); call rd('st3', st3, size(st3)); call rd('s2', s2, size(s2))
  allocate(tmp2d(1:ni,1:nk+1,59), tmp1d(1:ni,2))
  allocate(znt(ni,nj), xland(ni,nj), tsk(ni,nj), qsfc(ni,nj), psfc(ni,nj), ust(ni,nj), ch(ni,nj), hfx(ni,nj), &
           qfx(ni,nj), wspd(ni,nj), rmol(ni,nj), hpbl(ni,nj), maxmf(ni,nj))
  allocate(kpbl2d(ni,nj), nupdraft(ni,nj), ktop_plume(ni,nj))
  znt = s2(:,:,1); xland = s2(:,:,2); tsk = s2(:,:,3); qsfc = s2(:,:,4); psfc = s2(:,:,5); ust = s2(:,:,6)
  ch = s2(:,:,7); hfx = s2(:,:,8); qfx = s2(:,:,9); wspd = s2(:,:,10); rmol = s2(:,:,11); hpbl = s2(:,:,12)
  kpbl2d = nint(s2(:,:,13)); nupdraft = nint(s2(:,:,14)); maxmf = s2(:,:,15); ktop_plume = nint(s2(:,:,16))

  bl_mynn_edmf = edmf
  bl_mynn_edmf_mom = edmf_mom
  if( ipbl.eq.5 )then
    bl_mynn_closure = 3.0
  else
    bl_mynn_closure = 2.5
  endif

        FLAG_QC = .true.
        FLAG_QI = .true.
        if( ptype.eq.3 .or. ptype.eq.5 )then
          FLAG_QNI = .true.
        else
          FLAG_QNI = .false.
        endif
        FLAG_QNC = .false.
        FLAG_QNWFA = .false.
        FLAG_QNIFA = .false.

      mynn_j_loop:  &
      DO j=1,nj

        tmp1d = 0.0
        tmp2d = 0.0

        IF( nstep.eq.0 )THEN
          initflag=1
        ELSE
          initflag=0
        ENDIF
        restart = .false.
        cycling = .false.

        do k=1,nk
        do i=1,ni
          do n=1,8
            tmp2d(i,k,n) = in3(i,j,k,n)      ! dz,u,v,w,th,qv,qc,qi
          enddo
          tmp2d(i,k, 9) = 0.0            ! qnc
          tmp2d(i,k,10) = 0.0            ! qni
          tmp2d(i,k,11) = 0.0            ! qnwfa
          tmp2d(i,k,12) = 0.0            ! qnifa
          tmp2d(i,k,13) = 0.0            ! qnbca
          tmp2d(i,k,14) = in3(i,j,k,9)   ! p
          tmp2d(i,k,15) = in3(i,j,k,10)  ! exner
          tmp2d(i,k,16) = in3(i,j,k,11)  ! rho
          tmp2d(i,k,17) = in3(i,j,k,12)  ! t3d
          tmp2d(i,k,18) = st3(i,j,k,1)   ! qke
          tmp2d(i,k,19) = st3(i,j,k,2)   ! qke updated after advection
          do n=3,7
            tmp2d(i,k,17+n) = st3(i,j,k,n)   ! sh3d,sm3d,tsq,qsq,cov
          enddo
          do n=25,37
            tmp2d(i,k,n) = 0.0           ! tendencies, exch_h, exch_m
          enddo
          tmp2d(i,k,38) = st3(i,j,k,8)   ! el_pbl
          do n=22,26
            tmp2d(i,k,17+n) = st3(i,j,k,n)   ! dqke,qwt,qshear,qbuoy,qdiss
          enddo
          do n=9,21
            tmp2d(i,k,35+n) = st3(i,j,k,n)   ! qc_bl .. det_sqv3d
          enddo
          tmp2d(i,k,57) = rthr           ! rthraten
          tmp2d(i,k,58) = 0.0            ! pattern_spp_pbl
        enddo
        enddo

        do i=1,ni
          tmp1d(i, 1) = 0.0    ! uoce
          tmp1d(i, 2) = 0.0    ! voce
        enddo

      call mynnedmf_wrapper_run(                 &
        initflag=initflag,  &
        restart=restart,  &
        cycling=cycling,                &
        delt=dt,  &
        dz=tmp2d(1,1, 1),  &
        dxc=dx,  &
        znt=znt(1,j),                         &
        u=tmp2d(1,1, 2),  &
        v=tmp2d(1,1, 3),  &
        w=tmp2d(1,1, 4),  &
        th=tmp2d(1,1, 5),                                &
        qv=tmp2d(1,1, 6),  &
        qc=tmp2d(1,1, 7),  &
        qi=tmp2d(1,1, 8),  &
        qnc=tmp2d(1,1, 9),  &
        qni=tmp2d(1,1,10),  &
        qnwfa=tmp2d(1,1,11),  &
        qnifa=tmp2d(1,1,12),  &
        qnbca=tmp2d(1,1,13),      &
        p=tmp2d(1,1,14),  &
        exner=tmp2d(1,1,15),  &
        rho=tmp2d(1,1,16),  &
        t3d=tmp2d(1,1,17),                         &
        xland=xland(1,j),  &
        ts=tsk(1,j),  &
        qsfc=qsfc(1,j),  &
        ps=psfc(1,j),                        &
        ust=ust(1,j),  &
        ch=ch(1,j),  &
        hfx=hfx(1,j),  &
        qfx=qfx(1,j),  &
        rmol=rmol(1,j),  &
        wspd=wspd(1,j),                &
        uoce=tmp1d(1,1),  &
        voce=tmp1d(1,2),                               &
        qke=tmp2d(1,1,18),  &
        qke_adv=tmp2d(1,1,19),  &
        sh3d=tmp2d(1,1,20),  &
        sm3d=tmp2d(1,1,21),                   &
        Tsq=tmp2d(1,1,22),  &
        Qsq=tmp2d(1,1,23),  &
        Cov=tmp2d(1,1,24),                             &
        rublten=tmp2d(1,1,25),  &
        rvblten=tmp2d(1,1,26),  &
        rthblten=tmp2d(1,1,27),                &
        rqvblten=tmp2d(1,1,28),  &
        rqcblten=tmp2d(1,1,29),  &
        rqiblten=tmp2d(1,1,30),              &
        rqncblten=tmp2d(1,1,31),  &
        rqniblten=tmp2d(1,1,32),                     &
        rqnwfablten=tmp2d(1,1,33),  &
        rqnifablten=tmp2d(1,1,34),  &
        rqnbcablten=tmp2d(1,1,35),     &
        exch_h=tmp2d(1,1,36),  &
        exch_m=tmp2d(1,1,37),  &
        pblh=hpbl(1,j),  &
        kpbl=kpbl2d(1,j),  &
        el_pbl=tmp2d(1,1,38),          &
        dqke=tmp2d(1,1,39),  &
        qwt=tmp2d(1,1,40),  &
        qshear=tmp2d(1,1,41),  &
        qbuoy=tmp2d(1,1,42),  &
        qdiss=tmp2d(1,1,43),             &
        qc_bl=tmp2d(1,1,44),  &
        qi_bl=tmp2d(1,1,45),  &
        cldfra_bl=tmp2d(1,1,46),                   &
        edmf_a=tmp2d(1,1,47),  &
        edmf_w=tmp2d(1,1,48),  &
        edmf_qt=tmp2d(1,1,49),                   &
        edmf_thl=tmp2d(1,1,50),  &
        edmf_ent=tmp2d(1,1,51),  &
        edmf_qc=tmp2d(1,1,52),               &
        sub_thl3d=tmp2d(1,1,53),  &
        sub_sqv3d=tmp2d(1,1,54),                     &
        det_thl3d=tmp2d(1,1,55),  &
        det_sqv3d=tmp2d(1,1,56),                     &
        nupdraft=nupdraft(1,j),  &
        maxMF=maxmf(1,j),  &
        ktop_plume=ktop_plume(1,j),               &
        rthraten=tmp2d(1,1,57),                                &
        dheat3d=tmp2d(1,1,59),         &
        tke_budget=bl_mynn_tkebudget,         &
        bl_mynn_tkeadvect=bl_mynn_tkeadvect,   &
        bl_mynn_cloudpdf=bl_mynn_cloudpdf,   &
        bl_mynn_mixlength=bl_mynn_mixlength,   &
        icloud_bl=icloud_bl,          &
        bl_mynn_edmf=bl_mynn_edmf,        &
        bl_mynn_edmf_mom=bl_mynn_edmf_mom,   &
        bl_mynn_edmf_tke=bl_mynn_edmf_tke,    &
        bl_mynn_cloudmix=bl_mynn_cloudmix,   &
        bl_mynn_mixqt=bl_mynn_mixqt,       &
        bl_mynn_output=bl_mynn_output,     &
        bl_mynn_closure=bl_mynn_closure,     &
        bl_mynn_mixscalars=bl_mynn_mixscalars,                      &
        spp_pbl=spp_pbl,  &
        pattern_spp_pbl=tmp2d(1,1,58),                 &
        flag_qc=flag_qc,  &
        flag_qi=flag_qi,                         &
        flag_qnc=flag_qnc,  &
        flag_qni=flag_qni,                       &
        flag_qnwfa=.false.,  &
        flag_qnifa=.false.,  &
        flag_qnbca=.false.,        &
        ids=1,ide=ni+1,jds=1,jde=2,kds=1,kde=nk+1,   &
        ims=1,ime=ni  ,jms=1,jme=1,kms=1,kme=nk  ,   &
        its=1,ite=ni  ,jts=1,jte=1,kts=1,kte=nk      )

        do k=1,nk
        do i=1,ni
          st3(i,j,k,1) = tmp2d(i,k,18)
          st3(i,j,k,2) = tmp2d(i,k,19)
          do n=3,7
            st3(i,j,k,n) = tmp2d(i,k,17+n)
          enddo
          st3(i,j,k,8) = tmp2d(i,k,38)
          do n=22,26
            st3(i,j,k,n) = tmp2d(i,k,17+n)
          enddo
          do n=9,21
            st3(i,j,k,n) = tmp2d(i,k,35+n)
          enddo
          do n=1,6
            te3(i,j,k,n) = tmp2d(i,k,24+n)   ! upten .. qipten
          enddo
          te3(i,j,k,7) = tmp2d(i,k,36)       ! xkzh
          te3(i,j,k,8) = tmp2d(i,k,37)       ! xkzm
          te3(i,j,k,9) = tmp2d(i,k,59)       ! dheat
        enddo
        enddo

      ENDDO  mynn_j_loop

  s2(:,:,1) = znt; s2(:,:,6) = ust; s2(:,:,8) = hfx; s2(:,:,9) = qfx; s2(:,:,10) = wspd; s2(:,:,11) = rmol
  s2(:,:,12) = hpbl; s2(:,:,13) = dble(kpbl2d); s2(:,:,14) = dble(nupdraft); s2(:,:,15) = maxmf
  s2(:,:,16) = dble(ktop_plume)
  call wr('st3', st3, size(st3)); call wr('te3', te3, size(te3)); call wr('s2', s2, size(s2))
end program drv_mynn_pbl
