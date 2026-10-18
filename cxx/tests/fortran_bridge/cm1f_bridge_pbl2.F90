! bind(C) shims around CM1's simple PBL scheme (cm1setup=2, ipbl=2) and the
! horizontal Smagorinsky-like scheme (horizturb=1):
!   sfcphys.F   gethpbl2 (bulk-Ri boundary-layer depth)
!   turb.F      turbparam_vert, turbparam_horiz (private in turb_module:
!               called through cm1f_turbcopy_module, cm1f_bridge_turbpriv.F90)
!               and the dobr09 block of sfc_and_turb (verbatim copy below,
!               calling turbtend.F's turbsz/turbuz/turbvz)
!   solve1.F    the use_pbl blocks (verbatim copies below)
!
! cm1f_turb_setup (cm1f_bridge_turb.F90) sets the grid bounds and the
! turbulence options, cm1f_sfc_options (cm1f_bridge_sfc.F90) the surface
! layer; cm1f_pbl2_options then switches the `input` module to cm1setup=2,
! ipbl=2 as param.F does. turbu/turbv/turbw/turbs of cm1f_bridge_turb.F90
! then run with dohturb = (horizturb = 1), dovturb = .false.

subroutine cm1f_pbl2_options(iopt, ropt) bind(C, name="cm1f_pbl2_options")
  use iso_c_binding
  use input
  implicit none
  !  iopt: 1 horizturb   ropt: 1 l_h 2 lhref1 3 lhref2 4 l_inf
  integer(c_int), intent(in) :: iopt(1)
  real(c_double), intent(in) :: ropt(4)
  cm1setup = 2; sgsmodel = 0; ipbl = 2; horizturb = iopt(1)
  idoles = .false.; iusetke = .false.; idopbl = .true.; use_pbl = .true.
  dohturb = (horizturb .eq. 1); dovturb = .false.
  tconfig = 2
  sfcmodel = 1; bbc = 3
  l_h = ropt(1); lhref1 = ropt(2); lhref2 = ropt(3); l_inf = ropt(4)
  output_dissten = 0
  ibb = ib; ieb = ie; jbb = jb; jeb = je; kbb = kb; keb = ke
  ibl = ib; iel = ie; jbl = jb; jel = je
end subroutine cm1f_pbl2_options

!-----------------------------------------------------------------------
! sfcphys.F

subroutine cm1f_pbl2_gethpbl2(psfc, qsfc, thflux, qvflux, ust, tsk, zh, th0, tha, divx, ugr, vgr, hpbl) &
    bind(C, name="cm1f_pbl2_gethpbl2")
  use iso_c_binding
  use input
  use sfcphys_module, only: gethpbl2
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: psfc, thflux, qvflux, ust, tsk
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: qsfc, hpbl
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: zh, th0, tha, divx
  real(c_double), intent(in) :: ugr(ib:ie+1,jb:je,kb:ke), vgr(ib:ie,jb:je+1,kb:ke)
  real, allocatable, dimension(:,:) :: thv1, thvsfc, govthv, brilast
  real, allocatable :: riout(:,:,:)
  allocate(thv1(ib:ie,jb:je), thvsfc(ib:ie,jb:je), govthv(ib:ie,jb:je), brilast(ib:ie,jb:je))
  allocate(riout(ib:ie,jb:je,kb:ke))
  thv1 = 0; thvsfc = 0; govthv = 0; brilast = 0; riout = 0
  call gethpbl2(psfc, qsfc, thflux, qvflux, ust, tsk, zh, th0, tha, divx, ugr, vgr, &
                thv1, thvsfc, govthv, brilast, hpbl, riout)
end subroutine cm1f_pbl2_gethpbl2

!-----------------------------------------------------------------------
! turb.F

subroutine cm1f_pbl2_turbparam_vert(zf, mf, nm, defv, kmv, khv, dissten, zntmp, ust) &
    bind(C, name="cm1f_pbl2_turbparam_vert")
  use iso_c_binding
  use input
  use cm1f_turbcopy_module, only: turbparam_vert
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: zf, mf, nm, defv
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmv, khv
  real(c_double), intent(inout) :: dissten(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: zntmp, ust
  call turbparam_vert(2, zf, 1.0, .false., one_h, onej_h, s1, mf, mf, s2, zero2, zero2, zero2, s3, &
                      nm, defv, defv, s4, kmv, khv, cm0_, dissten, out3d, zero2, zntmp, ust, zero2, &
                      c1d, c1d, c1d, c1d, c1d, c1d, c1d, c1d, &
                      cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
end subroutine cm1f_pbl2_turbparam_vert

subroutine cm1f_pbl2_turbparam_horiz(zf, mf, nm, defh, kmh, khh, dissten, xland, psfc) &
    bind(C, name="cm1f_pbl2_turbparam_horiz")
  use iso_c_binding
  use input
  use cm1f_turbcopy_module, only: turbparam_horiz
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: zf, mf, nm, defh
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, khh
  real(c_double), intent(inout) :: dissten(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: xland, psfc
  real, allocatable :: tlh(:,:)
  allocate(tlh(ib:ie,jb:je)); tlh = l_h
  call turbparam_horiz(2, zf, 1.0, .false., one_h, onej_h, s1, mf, mf, s2, zero2, zero2, zero2, s3, &
                       nm, defh, defh, s4, kmh, khh, cm0_, dissten, out3d, zero2, zero2, zero2, xland, psfc, tlh, &
                       c1d, c1d, c1d, c1d, c1d, c1d, c1d, c1d, &
                       cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
end subroutine cm1f_pbl2_turbparam_horiz

! turb.F sfc_and_turb, dobr09 block (verbatim, with the scratch arrays of
! sfc_and_turb local here: dum1..dum3, thten, thten1, sadv).
subroutine cm1f_pbl2_br09(dosfc, dt, ruh, uf, uh, rvh, vf, mh, mf, rmf, rho, rr, rf, th0, tha, qa, &
                          ua, va, wa, t13, t23, kmv, khv, thflux, qvflux, &
                          thpten, qvpten, qcpten, upten, vpten) bind(C, name="cm1f_pbl2_br09")
  use iso_c_binding
  use input
  use constants
  use turbtend_module, only: turbsz, turbuz, turbvz
  use cm1f_bridge_turb_mod
  implicit none
  integer(c_int), value :: dosfc
  real(c_double), value :: dt
  real(c_double), intent(in) :: ruh(ib:ie), uf(ib:ie+1), uh(ib:ie), rvh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, th0, tha, t13, t23
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, rmf, wa
  real(c_double), intent(in) :: qa(ibm:iem,jbm:jem,kbm:kem,numq)
  real(c_double), intent(in) :: ua(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmv, khv
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: thflux, qvflux
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: thpten, qvpten, qcpten, upten, vpten
  real, allocatable, dimension(:,:,:) :: dum1, dum2, dum3, thten, thten1, sadv, qipten
  logical :: dosfcflx
  integer :: i, j, k
  allocate(dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke), &
           thten(ib:ie,jb:je,kb:ke), thten1(ib:ie,jb:je,kb:ke), sadv(ib:ie,jb:je,kb:ke), &
           qipten(ib:ie,jb:je,kb:ke))
  dum1 = 0; dum2 = 0; dum3 = 0; thten = 0; thten1 = 0; sadv = 0; qipten = 0
  dosfcflx = (dosfc .ne. 0)

        if( doimpl.eq.1 )then
          !  Arrays for vimpl turbs:
          do k=1,nk
          do j=1,nj
          do i=1,ni
            thten(i,j,k)  = khv(i,j,k  )*mf(i,j,k  )*rf(i,j,k  )*mh(i,j,k)*rr(i,j,k)
            thten1(i,j,k) = khv(i,j,k+1)*mf(i,j,k+1)*rf(i,j,k+1)*mh(i,j,k)*rr(i,j,k)
          enddo
          enddo
          enddo
        endif

        do k=1,nk
        do j=0,nj+1
        do i=0,ni+1
          sadv(i,j,k) = (th0(i,j,k)-th0r)+tha(i,j,k)
        enddo
        enddo
        enddo

        call     turbsz(1,dt,dosfcflx,thflux,mh,mf,  &
                       thpten,dum1,dum2,dum3,rho,rr,rf,sadv,khv,thten,thten1,1)
        if( imoist.eq.1 )  &
        call     turbsz(1,dt,dosfcflx,qvflux,mh,mf,  &
                       qvpten,dum1,dum2,dum3,rho,rr,rf,qa(ib,jb,kb,nqv),khv,thten,thten1,2)
        if( imoist.eq.1 .and. nqc.ge.1 )  &
        call     turbsz(0,dt,dosfcflx,qvflux,mh,mf,  &
                       qcpten,dum1,dum2,dum3,rho,rr,rf,qa(ib,jb,kb,nqc),khv,thten,thten1,0)
        if( imoist.eq.1 .and. nqi.ge.1 )  &
        call     turbsz(0,dt,dosfcflx,qvflux,mh,mf,  &
                       qipten,dum1,dum2,dum3,rho,rr,rf,qa(ib,jb,kb,nqi),khv,thten,thten1,0)
        call     turbuz(dt,one_h,ruh,one_f,one_f,one_f,one_f,uf,onej_h,mh,mf,rmf,rho,rf,  &
                       zero2,one_t,one_t,one_t,one_t,rds_,rds_,rdsf_,rdsf_,one_t3,     &
                       upten,dum1,dum2,dum3,thten,thten1,ua,wa,t13,t13,t13,t13,kmv, &
                       kprof,kprof,kprof,dum2pt,dum2pt)
        call     turbvz(dt,one_h,one_h,one_h,one_h,uh,one_f,rvh,vf,mh,mf,rho,rr,rf,   &
                       zero2,one_t,one_t,one_t,one_t,rds_,rds_,rdsf_,rdsf_,one_t3,  &
                       vpten,dum1,dum2,dum3,thten,thten1,va,wa,t23,t23,t23,kmv, &
                       kprof,kprof,kprof,dum2pt,dum2pt)
end subroutine cm1f_pbl2_br09

!-----------------------------------------------------------------------
! solve1.F: the use_pbl blocks (thten1 before doturb; uten1, vten1, qten
! after it), verbatim

subroutine cm1f_pbl2_solve1(thpten, qvpten, qcpten, upten, vpten, thten1, uten1, vten1, qten) &
    bind(C, name="cm1f_pbl2_solve1")
  use iso_c_binding
  use input
  implicit none
  real(c_double), intent(in), dimension(ibb:ieb,jbb:jeb,kbb:keb) :: thpten, qvpten, qcpten, upten, vpten
  real(c_double), intent(inout) :: thten1(ib:ie,jb:je,kb:ke)
  real(c_double), intent(inout) :: uten1(ib:ie+1,jb:je,kb:ke), vten1(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(inout) :: qten(ibm:iem,jbm:jem,kbm:kem,numq)
  integer :: i, j, k, ip

      IF( use_pbl )THEN
        do k=1,nk
        do j=1,nj
        do i=1,ni
          thten1(i,j,k) = thten1(i,j,k) + thpten(i,j,k)
        enddo
        enddo
        enddo
      ENDIF

      if( use_pbl )then
        if( axisymm.eq.1 )then
          ip = 2
        else
          ip = 1
        endif
        do k=1,nk
        do j=1,nj
        do i=ip,ni+1
          uten1(i,j,k) = uten1(i,j,k) + 0.5*( upten(i-1,j,k)+ upten(i,j,k))
        enddo
        enddo
        enddo
      endif

      if( use_pbl )then
        do k=1,nk
        do j=1,nj+1
        do i=1,ni
          vten1(i,j,k) = vten1(i,j,k) + 0.5*( vpten(i,j-1,k)+ vpten(i,j,k))
        enddo
        enddo
        enddo
      endif

      if(imoist.eq.1)then
            if( use_pbl )then
              do k=1,nk
              do j=1,nj
              do i=1,ni
                qten(i,j,k,nqv) = qten(i,j,k,nqv) + qvpten(i,j,k)
              enddo
              enddo
              enddo
            endif
        IF( use_pbl )THEN
        if(nqc.ge.1)then
          do k=1,nk
          do j=1,nj
          do i=1,ni
            qten(i,j,k,nqc) = qten(i,j,k,nqc) + qcpten(i,j,k)
          enddo
          enddo
          enddo
        endif
        ENDIF
      endif
end subroutine cm1f_pbl2_solve1
