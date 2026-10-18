! bind(C) shims around the WRF/MMM surface layers CM1 uses with
! sfcmodel=2 (sfclay.F: SFCLAY, sfclayinit) and sfcmodel=3 (sf_sfclayrev.F
! through mmm_physics_wrapper's sf_sfclayrev_wrapper), plus verbatim copies of
! the turb.F (sfc_and_turb) code around them:
!   cm1f_wrf_prep     the dum1..dum7 block before the PBL call (and the ugr
!                     block it reads), turb.F ~305-330 and 1828-1849, and the
!                     divx/ppten/dum8 moisture copies
!   cm1f_sfclay_post  qsfc diagnosis (update_sfc=.false.), ch/cq/s10,
!                     thflux/qvflux, turb.F 1007-1110
! None of these read the `input` module: dimensions come as arguments.
!
! The surface arrays travel packed as s2d(ib:ie,jb:je,ns), in this order
! (must match sfclay_tests.cpp):
!   1 psfc  2 chs  3 chs2  4 cqs2  5 cpmm  6 znt  7 ust  8 hpbl  9 mavail
!  10 zol  11 mol  12 regime  13 psim  14 psih  15 fm  16 fh  17 xland
!  18 hfx  19 qfx  20 lh  21 tsk  22 flhc  23 flqc  24 qgh  25 qsfc  26 rmol
!  27 u10  28 v10  29 th2  30 t2  31 q2  32 gz1oz0  33 wspd  34 br  35 dsxy
!  36 ustm  37 ck  38 cka  39 cd  40 cda  41 z0t  42 z0q  43 ch  44 cq
!  45 s10  46 thflux  47 qvflux
! and the constants as rc(19):
!   cp g rovcp rd xlv lv1 lv2 svp1 svp2 svp3 svpt0 ep1 ep2 karman eomeg
!   stbolt p1000mb rv rovg

subroutine cm1f_sfclay_init(psimtb_out, psihtb_out) bind(C, name="cm1f_sfclay_init")
  use iso_c_binding
  use module_sf_sfclay, only: sfclayinit, psimtb, psihtb
  implicit none
  real(c_double), intent(out) :: psimtb_out(0:1000), psihtb_out(0:1000)
  call sfclayinit
  psimtb_out = psimtb
  psihtb_out = psihtb
end subroutine cm1f_sfclay_init

subroutine cm1f_sfclayrev_init() bind(C, name="cm1f_sfclayrev_init")
  use iso_c_binding
  use sf_sfclayrev, only: sf_sfclayrev_init
  implicit none
  character(len=256) :: errmsg
  integer :: errflg
  call sf_sfclayrev_init(errmsg=errmsg, errflg=errflg)
end subroutine cm1f_sfclayrev_init

! turb.F 732-751 (sfcmodel=2)
subroutine cm1f_sfclay(ni, nj, nk, ib, ie, jb, je, kb, ke, isftcflx, iz0tlnd, rc, &
                       dum1, dum2, dum4, divx, prs, dum5, rf, s2d) bind(C, name="cm1f_sfclay")
  use iso_c_binding
  use module_sf_sfclay, only: sfclay
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke, isftcflx, iz0tlnd
  real(c_double), intent(in) :: rc(19)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: dum1, dum2, dum4, divx, prs, dum5, rf
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,47)
  real :: CP, G, ROVCP, RD, XLV, lv1, lv2, SVP1, SVP2, SVP3, SVPT0, EP1, EP2, KARMAN, EOMEG, STBOLT, P1000mb
  integer :: isfflx
  CP = rc(1); G = rc(2); ROVCP = rc(3); RD = rc(4); XLV = rc(5); lv1 = rc(6); lv2 = rc(7)
  SVP1 = rc(8); SVP2 = rc(9); SVP3 = rc(10); SVPT0 = rc(11); EP1 = rc(12); EP2 = rc(13)
  KARMAN = rc(14); EOMEG = rc(15); STBOLT = rc(16); P1000mb = rc(17)
  isfflx = 1
  call SFCLAY(dum1,dum2,dum4,divx,prs,dum5,      &
               CP,G,ROVCP,RD,XLV,lv1,lv2,s2d(ib,jb,1),s2d(ib,jb,2),s2d(ib,jb,3),s2d(ib,jb,4),s2d(ib,jb,5), &
               s2d(ib,jb,6),s2d(ib,jb,7),s2d(ib,jb,8),s2d(ib,jb,9),s2d(ib,jb,10),s2d(ib,jb,11),s2d(ib,jb,12), &
               s2d(ib,jb,13),s2d(ib,jb,14), &
               s2d(ib,jb,15),s2d(ib,jb,16),                                        &
               s2d(ib,jb,17),s2d(ib,jb,18),s2d(ib,jb,19),s2d(ib,jb,20),s2d(ib,jb,21),s2d(ib,jb,22), &
               s2d(ib,jb,23),s2d(ib,jb,24),s2d(ib,jb,25),s2d(ib,jb,26), &
               s2d(ib,jb,27),s2d(ib,jb,28),s2d(ib,jb,29),s2d(ib,jb,30),s2d(ib,jb,31),rf(ib,jb,1), &
               s2d(ib,jb,32),s2d(ib,jb,33),s2d(ib,jb,34),isfflx,s2d(ib,jb,35), &
               SVP1,SVP2,SVP3,SVPT0,EP1,EP2,                 &
               KARMAN,EOMEG,STBOLT,                          &
               P1000mb,                                      &
               1  ,ni+1 , 1  ,nj+1 , 1  ,nk+1 ,              &
               ib ,ie , jb ,je , kb ,ke ,                    &
               1  ,ni , 1  ,nj , 1  ,nk ,                    &
               s2d(ib,jb,36),s2d(ib,jb,37),s2d(ib,jb,38),s2d(ib,jb,39),s2d(ib,jb,40), &
               isftcflx,iz0tlnd,s2d(ib,jb,41),s2d(ib,jb,42))
end subroutine cm1f_sfclay

! turb.F 753-839 (sfcmodel=3): sfclayrev_j_loop, verbatim but for the
! packed surface arrays
subroutine cm1f_sfclayrev(ni, nj, nk, ib, ie, jb, je, kb, ke, isftcflx, iz0tlnd, rc, &
                          dum1, dum2, dum4, divx, prs, dum5, s2d) bind(C, name="cm1f_sfclayrev")
  use iso_c_binding
  use mmm_physics_wrapper_module, only: sf_sfclayrev_wrapper
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke, isftcflx, iz0tlnd
  real(c_double), intent(in) :: rc(19)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: dum1, dum2, dum4, divx, prs, dum5
  real(c_double), intent(inout), target :: s2d(ib:ie,jb:je,47)
  real :: cp, g, rovcp, rd, xlv, svp1, svp2, svp3, svpt0, ep1, ep2, karman, p1000mb
  real, allocatable :: tmp2d(:,:,:), tmp1d(:,:)
  character(len=256) :: errmsg
  integer :: errflg, i, j, k, ibtmp
  cp = rc(1); g = rc(2); rovcp = rc(3); rd = rc(4); xlv = rc(5)
  svp1 = rc(8); svp2 = rc(9); svp3 = rc(10); svpt0 = rc(11); ep1 = rc(12); ep2 = rc(13)
  karman = rc(14); p1000mb = rc(17)
  ibtmp = 1
  allocate(tmp2d(1:ni,1:nk+1,44), tmp1d(1:ni,5))
  tmp2d = 0.0
  tmp1d = 0.0

        sfclayrev_j_loop:  &
        do j=1,nj

          do k=1,2
          do i=1,ni
            tmp2d(i,k,1) = dum1(i,j,k)  ! u3d
            tmp2d(i,k,2) = dum2(i,j,k)  ! v3d
            tmp2d(i,k,3) = dum4(i,j,k)  ! t3d
            tmp2d(i,k,4) = divx(i,j,k)  ! qv3d
            tmp2d(i,k,5) = prs(i,j,k)  ! p3d
            tmp2d(i,k,6) = dum5(i,j,k)  ! dz8w
          enddo
          enddo

          do i=1,ni
            tmp1d(i,1) = s2d(i,j,35)  ! dx
            tmp1d(i,2) = 0.0        ! water_depth
          enddo

          call sf_sfclayrev_wrapper(               &
                      ux=tmp2d(ibtmp,1,1),  &
                      vx=tmp2d(ibtmp,1,2),  &
                      t1d=tmp2d(ibtmp,1,3),  &
                      qv1d=tmp2d(ibtmp,1,4),  &
                      p1d=tmp2d(ibtmp,1,5),  &
                      dz8w1d=tmp2d(ibtmp,1,6),  &
                      cp=cp,  &
                      g=g,  &
                      rovcp=rovcp,  &
                      r=rd,  &
                      xlv=xlv,  &
                      psfcpa=s2d(1,j,1),  &
                      chs=s2d(1,j,2),  &
                      chs2=s2d(1,j,3),  &
                      cqs2=s2d(1,j,4),  &
                      cpm=s2d(1,j,5),      &
                      pblh=s2d(1,j,8),  &
                      rmol=s2d(1,j,26), &
                      znt=s2d(1,j,6),  &
                      ust=s2d(1,j,7),  &
                      mavail=s2d(1,j,9),  &
                      zol=s2d(1,j,10),  &
                      mol=s2d(1,j,11),  &
                      regime=s2d(1,j,12),  &
                      psim=s2d(1,j,13),  &
                      psih=s2d(1,j,14), &
                      fm=s2d(1,j,15),  &
                      fh=s2d(1,j,16),                                        &
                      xland=s2d(1,j,17),  &
                      hfx=s2d(1,j,18),  &
                      qfx=s2d(1,j,19),  &
                      tsk=s2d(1,j,21),  &
                      u10=s2d(1,j,27),  &
                      v10=s2d(1,j,28),  &
                      th2=s2d(1,j,29),  &
                      t2=s2d(1,j,30),  &
                      q2=s2d(1,j,31),                            &
                      flhc=s2d(1,j,22),  &
                      flqc=s2d(1,j,23),  &
                      qgh=s2d(1,j,24),  &
                      qsfc=s2d(1,j,25),  &
                      lh=s2d(1,j,20),  &
                      gz1oz0=s2d(1,j,32),  &
                      wspd=s2d(1,j,33),  &
                      br=s2d(1,j,34),  &
                      isfflx=.true.,  &
                      dx=tmp1d(ibtmp,1),                     &
                      svp1=svp1,  &
                      svp2=svp2,  &
                      svp3=svp3,  &
                      svpt0=svpt0,  &
                      ep1=ep1,  &
                      ep2=ep2,                 &
                      karman=karman,  &
                      p1000mb=p1000mb,                                &
                      shalwater_z0=.false.,  &
                      water_depth=tmp1d(ibtmp,2),     &
                      isftcflx=isftcflx,  &
                      iz0tlnd=iz0tlnd,          &
                      scm_force_flux=.false.,  &
                      ustm=s2d(1,j,36),  &
                      ck=s2d(1,j,37),  &
                      cka=s2d(1,j,38),  &
                      cd=s2d(1,j,39),  &
                      cda=s2d(1,j,40),  &
                      its=1,  &
                      ite=ni,  &
                      errmsg=errmsg,  &
                      errflg=errflg)
        enddo  sfclayrev_j_loop

end subroutine cm1f_sfclayrev

! turb.F 1007-1018 (update_sfc=.false.), 1039-1048, 1101-1110 (verbatim)
subroutine cm1f_sfclay_post(ni, nj, ib, ie, jb, je, kb, ke, set_fluxes, divx, rf, s2d) &
    bind(C, name="cm1f_sfclay_post")
  use iso_c_binding
  implicit none
  integer(c_int), value :: ni, nj, ib, ie, jb, je, kb, ke, set_fluxes
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: divx, rf
  real(c_double), intent(inout) :: s2d(ib:ie,jb:je,47)
  integer :: i, j
  real :: qx

          do j=1,nj
          do i=1,ni
            qx = divx(i,j,1)
            if ( s2d(i,j,23) .ne. 0.) then
               s2d(i,j,25)=QX+s2d(i,j,19)/s2d(i,j,23)
            else
               s2d(i,j,25) = QX
            end if
          enddo
          enddo

            do j=1,nj
            do i=1,ni
              s2d(i,j,43) = s2d(i,j,3)
              s2d(i,j,44) = s2d(i,j,4)
              s2d(i,j,45) = sqrt( s2d(i,j,27)**2 + s2d(i,j,28)**2 )
            enddo
            enddo

          IF( set_fluxes.eq.1 )THEN
            do j=1,nj
            do i=1,ni
              s2d(i,j,46) = s2d(i,j,18)/(s2d(i,j,5)*rf(i,j,1))
              s2d(i,j,47) = s2d(i,j,19)/rf(i,j,1)
            enddo
            enddo
          ENDIF
end subroutine cm1f_sfclay_post

! turb.F ~305-330 (ugr, imove=0) and 1828-1849 (dum1..dum7), verbatim; plus
! the moisture copies (divx: ~388-440, dum8/ppten: get_vars ~1764-1808).
subroutine cm1f_wrf_prep(ni, nj, nk, ib, ie, jb, je, kb, ke, pertflx, dz, cgt1, cgt2, cgt3, &
                         ua, va, u0, v0, th0, tha, pi0, ppi, prs, rmh, c1, c2, psfc, &
                         dum1, dum2, dum3, dum4, dum5, dum6, dum7, &
                         numq, imoist, nqv, nqc, nqi, qa, divx, ppten, dum8) bind(C, name="cm1f_wrf_prep")
  use iso_c_binding
  implicit none
  integer(c_int), value :: ni, nj, nk, ib, ie, jb, je, kb, ke, pertflx, numq, imoist, nqv, nqc, nqi
  real(c_double), value :: dz, cgt1, cgt2, cgt3
  real(c_double), intent(in), dimension(ib:ie+1,jb:je,kb:ke) :: ua, u0
  real(c_double), intent(in), dimension(ib:ie,jb:je+1,kb:ke) :: va, v0
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: th0, tha, pi0, ppi, prs, rmh, c1, c2
  real(c_double), intent(in) :: psfc(ib:ie,jb:je)
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: dum1, dum2, dum3, dum4, dum5, dum6, dum7, &
                                                                divx, ppten, dum8
  real(c_double), intent(in) :: qa(ib:ie,jb:je,kb:ke,numq)
  real, allocatable :: ugr(:,:,:), vgr(:,:,:)
  integer :: i, j, k
  allocate(ugr(ib:ie+1,jb:je,kb:ke), vgr(ib:ie,jb:je+1,kb:ke))
  ugr = 0; vgr = 0

        IF( pertflx.eq.1 )THEN
          do k=1,nk
          do j=jb,je
          do i=ib,ie
            ugr(i,j,k) = ua(i,j,k)-u0(i,j,k)
            vgr(i,j,k) = va(i,j,k)-v0(i,j,k)
          enddo
          enddo
          enddo
        ELSE
          do k=1,nk
          do j=jb,je
          do i=ib,ie
            ugr(i,j,k) = ua(i,j,k)
            vgr(i,j,k) = va(i,j,k)
          enddo
          enddo
          enddo
        ENDIF

        DO j=1,nj
          do k=1,nk
          do i=1,ni
            ! ground-relative winds:
            dum1(i,j,k)=0.5*(ugr(i,j,k)+ugr(i+1,j,k))
            dum2(i,j,k)=0.5*(vgr(i,j,k)+vgr(i,j+1,k))
            dum3(i,j,k)=th0(i,j,k)+tha(i,j,k)
            dum7(i,j,k)=pi0(i,j,k)+ppi(i,j,k)
            dum4(i,j,k)=dum3(i,j,k)*dum7(i,j,k)
            dum5(i,j,k) = dz*rmh(i,j,k)
            dum6(i,j,k) = c1(i,j,k)*prs(i,j,k-1)+c2(i,j,k)*prs(i,j,k)
          enddo
          enddo
          do i=1,ni
            ! surface:
            dum6(i,j,1) = psfc(i,j)
            ! top of model:
            dum6(i,j,nk+1)= cgt1*prs(i,j,nk)+cgt2*prs(i,j,nk-1)+cgt3*prs(i,j,nk-2)
          enddo
        ENDDO

      IF(imoist.eq.1)THEN
        DO k=1,nk
          do j=1,nj
          do i=1,ni
            divx(i,j,k) = qa(i,j,k,nqv)
          enddo
          enddo
        ENDDO
      ELSE
          DO k=1,nk
            do j=1,nj
            do i=1,ni
              divx(i,j,k) = 0.0
            enddo
            enddo
          ENDDO
      ENDIF

        DO k=1,nk
          if( nqi.ge.1 )then
            do j=1,nj
            do i=1,ni
              dum8(i,j,k) = qa(i,j,k,nqi)
            enddo
            enddo
          else
            do j=1,nj
            do i=1,ni
              dum8(i,j,k) = 0.0
            enddo
            enddo
          endif
        ENDDO

        if( nqc.ge.1 )then
          DO k=1,nk
          do j=1,nj
          do i=1,ni
            ppten(i,j,k) = qa(i,j,k,nqc)
          enddo
          enddo
          ENDDO
        else
          DO k=1,nk
          do j=1,nj
          do i=1,ni
            ppten(i,j,k) = 0.0
          enddo
          enddo
          ENDDO
        endif
end subroutine cm1f_wrf_prep
