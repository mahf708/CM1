! bind(C) shims around the YSU PBL (ipbl=1): the ysu_j_loop of turb.F
! (sfc_and_turb, lines 1890-2048) calling mmm_physics_wrapper's ysu_wrapper
! (-> bl_ysu.F bl_ysu_run), verbatim but for packed arrays, and the YSU
! dissipative heating block (turb.F 2717-2752). No `input` module state.
!
! Packed arrays (must match ysu_tests.cpp):
!   a3d(ib:ie,jb:je,kb:ke,11) inputs:  1 dum1(u) 2 dum2(v) 3 dum4(T) 4 divx(qv)
!       5 ppten(qc) 6 dum8(qi) 7 prs 8 dum6(p8w, k=1..nk+1) 9 dum7(pi)
!       10 dum5(dz8w) 11 dum9(rthraten)
!   s2d(ib:ie,jb:je,16): 1 psfc 2 zntmp 3 ust 4 hpbl 5 fm 6 fh 7 xland 8 hfx
!       9 qfx 10 wspd 11 br 12 kpbl2d (as real) 13 wstar 14 delta 15 u10 16 v10
!   o3d(ib:ie,jb:je,kb:ke,8) outputs: 1 upten 2 vpten 3 thpten 4 qvpten
!       5 qcpten 6 qipten 7 xkzh 8 xkzm
!   rc(10): cp g rovcp rd rovg ep1 ep2 karman xlv rv

subroutine cm1f_ysu(ni, nj, nk, ib, ie, jb, je, kb, ke, itopdown, dt, rc, a3d, s2d, o3d) &
    bind(C, name="cm1f_ysu")
  use iso_c_binding
  use mmm_physics_wrapper_module, only: ysu_wrapper
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke, itopdown
  real(c_double), value :: dt
  real(c_double), intent(in) :: rc(10)
  real(c_double), intent(in), target :: a3d(ib:ie,jb:je,kb:ke,11)
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,16)
  real(c_double), intent(inout) :: o3d(ib:ie,jb:je,kb:ke,8)
  real :: cp, g, rovcp, rd, rovg, ep1, ep2, karman, xlv, rv
  real, allocatable :: tmp2d(:,:,:), tmp1d(:,:)
  integer, allocatable :: kpbl2d(:,:)
  logical :: ysu_topdown_pblmix
  character(len=256) :: errmsg
  integer :: errflg, i, j, k, ibtmp, kbtmp
  cp = rc(1); g = rc(2); rovcp = rc(3); rd = rc(4); rovg = rc(5); ep1 = rc(6); ep2 = rc(7)
  karman = rc(8); xlv = rc(9); rv = rc(10)
  ibtmp = 1; kbtmp = 1
  allocate(tmp2d(1:ni,1:nk+1,44), tmp1d(1:ni,5), kpbl2d(ib:ie,jb:je))
  kpbl2d = nint(s2d(:,:,12))

        ysu_topdown_pblmix = (itopdown.eq.1)

        ysu_j_loop:  &
        DO j=1,nj

        tmp2d = 0.0

        do k=1,nk
        do i=1,ni
          tmp2d(i,k, 1) =  a3d(i,j,k,1)   ! u
          tmp2d(i,k, 2) =  a3d(i,j,k,2)   ! v
          tmp2d(i,k, 3) =  a3d(i,j,k,3)   ! T
          tmp2d(i,k, 4) =  a3d(i,j,k,4)   ! qv
          tmp2d(i,k, 5) =  a3d(i,j,k,5)   ! qc
          tmp2d(i,k, 6) =  a3d(i,j,k,6)   ! qi
          tmp2d(i,k, 7) =  a3d(i,j,k,7)   ! p3d
          tmp2d(i,k, 9) =  a3d(i,j,k,9)   ! pi3d
          tmp2d(i,k,10) =  a3d(i,j,k,10)  ! dz8w
          tmp2d(i,k,11) =  a3d(i,j,k,11)  ! rthraten
        enddo
        enddo

        do k=1,nk+1
        do i=1,ni
          tmp2d(i,k,8) =  a3d(i,j,k,8)  ! p3di
        enddo
        enddo

        tmp1d = 0.0

        do i=1,ni
          tmp1d(i,1) =  1.0              ! ctopo
          tmp1d(i,2) =  1.0              ! ctopo2
          tmp1d(i,3) =  0.0              ! uoce
          tmp1d(i,4) =  0.0              ! voce
          tmp1d(i,5) =  0.0              ! frcurb
        enddo

        call ysu_wrapper(                                          &
                  ux=tmp2d(ibtmp,kbtmp,1),                        &
                  vx=tmp2d(ibtmp,kbtmp,2),                        &
                  tx=tmp2d(ibtmp,kbtmp,3),                        &
                  qvx=tmp2d(ibtmp,kbtmp,4),                       &
                  qcx=tmp2d(ibtmp,kbtmp,5),                       &
                  qix=tmp2d(ibtmp,kbtmp,6),                       &
                  nmix=1,                                         &
                  qmix=tmp2d(ibtmp,kbtmp,23),                     &
                  p2d=tmp2d(ibtmp,kbtmp,7),                       &
                  p2di=tmp2d(ibtmp,kbtmp,8),                      &
                  pi2d=tmp2d(ibtmp,kbtmp,9),                      &
                  f_qc=.true.,                                    &
                  f_qi=.true.,                                    &
                  utnp=tmp2d(ibtmp,kbtmp,12),                     &
                  vtnp=tmp2d(ibtmp,kbtmp,13),                     &
                  ttnp=tmp2d(ibtmp,kbtmp,14),                     &
                  qvtnp=tmp2d(ibtmp,kbtmp,15),                    &
                  qctnp=tmp2d(ibtmp,kbtmp,16),                    &
                  qitnp=tmp2d(ibtmp,kbtmp,17),                    &
                  qmixtnp=tmp2d(ibtmp,kbtmp,24),                  &
                  cp=cp,                                          &
                  g=g,                                            &
                  rovcp=rovcp,                                    &
                  rd=rd,                                          &
                  rovg=rovg,                                      &
                  ep1=ep1,                                        &
                  ep2=ep2,                                        &
                  karman=karman,                                  &
                  xlv=xlv,                                        &
                  rv=rv,                                          &
                  dz8w2d=tmp2d(ibtmp,kbtmp,10),                   &
                  psfcpa=s2d(1,j,1),                              &
                  znt=s2d(1,j,2),                                 &
                  ust=s2d(1,j,3),                                 &
                  hpbl=s2d(1,j,4),                                &
                  psim=s2d(1,j,5),                                &
                  psih=s2d(1,j,6),                                &
                  xland=s2d(1,j,7),                               &
                  hfx=s2d(1,j,8),                                 &
                  qfx=s2d(1,j,9),                                 &
                  wspd=s2d(1,j,10),                               &
                  br=s2d(1,j,11),                                 &
                  dt=dt,                                          &
                  kpbl1d=kpbl2d(1,j),                             &
                  exch_hx=tmp2d(ibtmp,kbtmp,18),                  &
                  exch_mx=tmp2d(ibtmp,kbtmp,19),                  &
                  wstar=s2d(1,j,13),                              &
                  delta=s2d(1,j,14),                              &
                  u10=s2d(1,j,15),                                &
                  v10=s2d(1,j,16),                                &
                  uox=tmp1d(ibtmp,3),                             &
                  vox=tmp1d(ibtmp,4),                             &
                  rthraten=tmp2d(ibtmp,kbtmp,11),                 &
                  ysu_topdown_pblmix=ysu_topdown_pblmix,          &
                  ctopo=tmp1d(ibtmp,1),                           &
                  ctopo2=tmp1d(ibtmp,2),                          &
                  a_u=tmp2d(ibtmp,kbtmp,31), &
                  a_v=tmp2d(ibtmp,kbtmp,32), &
                  a_t=tmp2d(ibtmp,kbtmp,33), &
                  a_q=tmp2d(ibtmp,kbtmp,34), &
                  a_e=tmp2d(ibtmp,kbtmp,35), &
                  b_u=tmp2d(ibtmp,kbtmp,36), &
                  b_v=tmp2d(ibtmp,kbtmp,37), &
                  b_t=tmp2d(ibtmp,kbtmp,38), &
                  b_q=tmp2d(ibtmp,kbtmp,39), &
                  b_e=tmp2d(ibtmp,kbtmp,40), &
                  sfk=tmp2d(ibtmp,kbtmp,41), &
                  vlk=tmp2d(ibtmp,kbtmp,42), &
                  dlu=tmp2d(ibtmp,kbtmp,43), &
                  dlg=tmp2d(ibtmp,kbtmp,44), &
                  frcurb=tmp1d(ibtmp,5), &
                  flag_bep = .false. ,                            &
                  its=1, ite=ni,                                  &
                  kts=1, kte=nk,                                  &
                  errmsg=errmsg ,                                 &
                  errflg=errflg )

        do k=1,nk
        do i=1,ni
          o3d(i,j,k,1) = tmp2d(i,k,12)
          o3d(i,j,k,2) = tmp2d(i,k,13)
          o3d(i,j,k,3) = tmp2d(i,k,14)
          o3d(i,j,k,4) = tmp2d(i,k,15)
          o3d(i,j,k,5) = tmp2d(i,k,16)
          o3d(i,j,k,6) = tmp2d(i,k,17)
        enddo
        enddo

          do k=1,nk
          do i=1,ni
            o3d(i,j,k,7) = tmp2d(i,k,18)
            o3d(i,j,k,8) = tmp2d(i,k,19)
          enddo
          enddo

        ENDDO  ysu_j_loop

  s2d(:,:,12) = kpbl2d
end subroutine cm1f_ysu

! turb.F 2717-2752 (ysudiss), verbatim
subroutine cm1f_ysu_dissten(ni, nj, nk, ib, ie, jb, je, kb, ke, dz, rdz, rmh, mf, upten, vpten, &
                            dum1, dum2, thten, thten1, dissten) bind(C, name="cm1f_ysu_dissten")
  use iso_c_binding
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke
  real(c_double), value :: dz, rdz
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: rmh, upten, vpten, dum1, dum2
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: thten, thten1
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke+1) :: dissten
  integer :: i, j, k
  real :: tem1, tem2

        DO j=1,nj

          do i=1,ni
            ! assume t13,t23 are zero at top of domain:
            !   thten = t13
            !   thten1 = t23
            thten(i,j,nk+1) = 0.0
            thten1(i,j,nk+1) = 0.0
          enddo

          do k=nk,2,-1
          do i=1,ni
            tem1 = dz*rmh(i,j,k)
            tem2 = rdz*mf(i,j,k)
            ! get stresses from u,v tendencies:
            thten(i,j,k) = thten(i,j,k+1)-upten(i,j,k)*tem1
            thten1(i,j,k) = thten1(i,j,k+1)-vpten(i,j,k)*tem1
            ! NOTE:  dissten is defined at w points:
            !  dum1 = u
            !  dum2 = v
            dissten(i,j,k) = dissten(i,j,k)                                 &
                          +( thten(i,j,k)*( dum1(i,j,k)-dum1(i,j,k-1) )*tem2 &
                            +thten1(i,j,k)*( dum2(i,j,k)-dum2(i,j,k-1) )*tem2 )
          enddo
          enddo

        ENDDO
end subroutine cm1f_ysu_dissten
