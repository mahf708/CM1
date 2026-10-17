! bind(C) shims around CM1's LES subgrid turbulence (sgsmodel=1):
! turb.F (calcnm, calcdef, tkebc; tkekm and gettau/sfcstress are in
! cm1f_bridge_turbpriv.F90), turbtend.F
! (turbs, turbt, turbu, turbv, turbw) and bc.F (bcw, bc2d).
!
! cm1f_turb_setup sets the `input` state the routines read (serial run, no
! terrain, no immersed boundaries, no budgets) and allocates the dummy
! arguments that only matter for options we do not use (terrain metrics,
! axisymmetric factors, MPI buffers, diagnostics). Arrays from C are
! contiguous, Fortran-ordered, with the bounds set here.
!
! The inline pieces of solve1.F (TKE source terms, dissipative heating with
! cvm, vimpl coefficient arrays) have no subroutine of their own; they are
! reproduced verbatim below (cm1f_turb_tke_source, cm1f_turb_dissheat,
! cm1f_turb_vimpl) so the C++ is compared against the Fortran expressions as
! compiled by gfortran.
module cm1f_bridge_turb_mod
  implicit none
  real, allocatable, dimension(:) :: one_h, one_f, onej_h, onej_f, rds_, rdsf_, kprof
  real, allocatable, dimension(:,:) :: one_t, zero2, cm0_
  real, allocatable, dimension(:,:,:) :: one_t3, s1, s2, s3, s4, s5, s6, s7, s8, s9, dum2pt
  real, allocatable, dimension(:,:,:,:) :: diag, out3d
  real, allocatable, dimension(:) :: c1d
  real, allocatable, dimension(:,:) :: cj, ci
  logical, allocatable :: bndy(:,:,:), cloudvar(:)
  integer, allocatable :: kbdy(:,:)
  !  iopt (must match turb_tests.cpp):
  !   1 tconfig 2 bcturbs 3 doimpl 4 wbc 5 ebc 6 sbc 7 nbc 8 imoist 9 ptype
  !  10 iice 11 numq 12 nqv 13 nqc 14 nqi 15 nql1 16 nql2 17 nqs1 18 nqs2
  !  19 eqtset 20 idiss 21 testcase
end module cm1f_bridge_turb_mod

subroutine cm1f_turb_setup(ni_, nj_, nk_, ngxy_, ngz_, iopt, ropt) bind(C, name="cm1f_turb_setup")
  use iso_c_binding
  use input
  use constants
  use ib_module, only: ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  use cm1f_bridge_turb_mod
  implicit none
  integer(c_int), value :: ni_, nj_, nk_, ngxy_, ngz_
  integer(c_int), intent(in) :: iopt(21)
  real(c_double), intent(in) :: ropt(3)

  nx = ni_; ny = nj_; nz = nk_
  ni = ni_; nj = nj_; nk = nk_; nkp1 = nk + 1
  ngxy = ngxy_; ngz = ngz_
  ib = 1 - ngxy; ie = ni + ngxy
  jb = 1 - ngxy; je = nj + ngxy
  kb = 1 - ngz;  ke = nk + ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  ibt = ib; iet = ie; jbt = jb; jet = je; kbt = kb; ket = ke + 1
  ibc = ib; iec = ie; jbc = jb; jec = je; kbc = kb; kec = ke + 1
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
  imp = max(1, ni); jmp = max(1, nj); kmp = max(2, nk); kmt = nk + 1; cmp = ngxy
  ibib = 1; ieib = 1; jbib = 1; jeib = 1; kbib = 1; keib = 1; kmaxib = 1
  ibdt = 1; iedt = 1; jbdt = 1; jedt = 1; kbdt = 1; kedt = 1; ntdiag = 1
  ibdq = 1; iedq = 1; jbdq = 1; jedq = 1; kbdq = 1; kedq = 1; nqdiag = 1
  ibdv = 1; iedv = 1; jbdv = 1; jedv = 1; kbdv = 1; kedv = 1
  nudiag = 1; nvdiag = 1; nwdiag = 1
  ib3d = 1; ie3d = 1; jb3d = 1; je3d = 1; kb3d = 1; ke3d = 1; nout3d = 1
  ib2pt = 1; ie2pt = 1; jb2pt = 1; je2pt = 1; kb2pt = 1; ke2pt = 1; ntwk = 1

  cm1setup = 1; sgsmodel = 1; ipbl = 0; horizturb = 0
  idoles = .true.; iusetke = .true.; dohturb = .true.; dovturb = .true.
  dot2p = .false.
  tconfig = iopt(1); bcturbs = iopt(2); doimpl = iopt(3)
  wbc = iopt(4); ebc = iopt(5); sbc = iopt(6); nbc = iopt(7)
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc .ne. 1) ibw = 1
  if (ebc .ne. 1) ibe = 1
  if (sbc .ne. 1) ibs = 1
  if (nbc .ne. 1) ibn = 1
  bbc = 1; tbc = 1
  imoist = iopt(8); ptype = iopt(9); iice = iopt(10)
  numq = iopt(11); nqv = iopt(12); nqc = iopt(13); nqi = iopt(14)
  nql1 = iopt(15); nql2 = iopt(16); nqs1 = iopt(17); nqs2 = iopt(18)
  eqtset = iopt(19); idiss = iopt(20)

  dx = ropt(1); dy = ropt(2); dz = ropt(3)
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz

  terrain_flag = .false.; axisymm = 0; do_ib = .false.
  timestats = 0; myid = 0
  smeps = 1.0e-60

  call set_constants(iopt(21))

  if (allocated(one_h)) then
    deallocate(one_h, one_f, onej_h, onej_f, rds_, rdsf_, kprof, one_t, zero2, cm0_, one_t3, &
               s1, s2, s3, s4, s5, s6, s7, s8, s9, dum2pt, diag, out3d, c1d, cj, ci, bndy, cloudvar, kbdy)
  end if
  allocate(one_h(ib:ie), one_f(ib:ie+1), onej_h(jb:je), onej_f(jb:je+1), rds_(kb:ke), rdsf_(kb:ke+1), kprof(kb:ke))
  allocate(one_t(itb:ite,jtb:jte), zero2(ib:ie,jb:je), cm0_(ib:ie,jb:je), one_t3(itb:ite,jtb:jte,ktb:kte))
  allocate(s1(ib:ie,jb:je,kb:ke), s2(ib:ie,jb:je,kb:ke), s3(ib:ie,jb:je,kb:ke), s4(ib:ie,jb:je,kb:ke), &
           s5(ib:ie,jb:je,kb:ke), s6(ib:ie,jb:je,kb:ke), s7(ib:ie,jb:je,kb:ke), s8(ib:ie,jb:je,kb:ke), &
           s9(ib:ie,jb:je,kb:ke), dum2pt(ib2pt:ie2pt,jb2pt:je2pt,kb2pt:ke2pt))
  allocate(diag(1,1,1,1), out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  allocate(c1d(kmt), cj(jmp,kmt), ci(imp,kmt), bndy(ibib:ieib,jbib:jeib,kbib:keib), cloudvar(maxq), &
           kbdy(ibib:ieib,jbib:jeib))
  one_h = 1; one_f = 1; onej_h = 1; onej_f = 1; rds_ = rdz; rdsf_ = rdz; kprof = 0
  one_t = 1; zero2 = 0; cm0_ = c_m; one_t3 = 0
  s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0; s8 = 0; s9 = 0; dum2pt = 0
  diag = 0; out3d = 0; c1d = 0; cj = 0; ci = 0; bndy = .false.; cloudvar = .false.; kbdy = 1
end subroutine cm1f_turb_setup

!-----------------------------------------------------------------------
! bc.F

subroutine cm1f_turb_bcw(w, flag) bind(C, name="cm1f_turb_bcw")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), intent(inout) :: w(ib:ie,jb:je,kb:ke+1)
  integer(c_int), value :: flag
  call bcw(w, flag)
end subroutine cm1f_turb_bcw

!-----------------------------------------------------------------------
! turb.F

subroutine cm1f_turb_calcnm(c1, c2, mf, pi0, th0, prs, pp, th, qa, nm) bind(C, name="cm1f_turb_calcnm")
  use iso_c_binding
  use input
  use turb_module, only: calcnm
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: c1, c2, pi0, th0, prs, pp, th
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: qa(ibm:iem,jbm:jem,kbm:kem,numq)
  real(c_double), intent(inout) :: nm(ib:ie,jb:je,kb:ke+1)
  logical, allocatable :: iamsat(:,:,:)
  allocate(iamsat(ib:ie,jb:je,kb:ke)); iamsat = .false.
  call calcnm(c1, c2, mf, pi0, s9, th0, cloudvar, nm, s1, s2, s3, s4, s5, s6, prs, pp, th, qa, iamsat)
end subroutine cm1f_turb_calcnm

subroutine cm1f_turb_calcdef(uh, uf, vh, vf, mh, c1, c2, mf, rho, rr, rf, ua, va, wa, &
                             s11, s12, s13, s22, s23, s33, defv, defh) bind(C, name="cm1f_turb_calcdef")
  use iso_c_binding
  use input
  use turb_module, only: calcdef
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho, rr, rf
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: ua(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke), wa(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: s11, s12, s13, s22, s23, s33
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke+1) :: defv, defh
  call calcdef(rds_, rds_, rdsf_, rdsf_, zero2, one_t, one_t, one_t, one_t, one_t, one_t, &
               one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, one_f, uf, vh, vf, mh, c1, c2, mf, defv, defh, &
               s1, s2, ua, va, wa, s11, s12, s13, s22, s23, s33, one_t3, one_t3, rho, rr, rf)
end subroutine cm1f_turb_calcdef

subroutine cm1f_turb_tkebc(tkea, kmh, kmv, khh, khv) bind(C, name="cm1f_turb_tkebc")
  use iso_c_binding
  use input
  use turb_module, only: tkebc
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(inout) :: tkea(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv, khh, khv
  call tkebc(tkea, kmh, kmv, khh, khv, zero2, zero2, zero2, &
             cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
end subroutine cm1f_turb_tkebc

!-----------------------------------------------------------------------
! turbtend.F

subroutine cm1f_turb_turbt(dt, uh, uf, vh, vf, mh, mf, rho, rr, rf, t, tten, kmh, kmv) &
    bind(C, name="cm1f_turb_turbt")
  use iso_c_binding
  use input
  use turbtend_module, only: turbt
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: t(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(inout) :: tten(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv
  call turbt(dt, one_h, one_h, uh, one_f, uf, vh, vf, mh, mf, rho, rr, rf, &
             rds_, rds_, one_t, one_t, one_t, one_t, one_t, one_t, &
             s1, s2, s3, s4, s5, s6, t, tten, kmh, kmv)
end subroutine cm1f_turb_turbt

subroutine cm1f_turb_turbu(dt, ruh, uf, vh, mh, mf, rmf, rho, rf, u, uten, w, t11, t12, t13, t22, kmv) &
    bind(C, name="cm1f_turb_turbu")
  use iso_c_binding
  use input
  use turbtend_module, only: turbu
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: ruh(ib:ie), uf(ib:ie+1), vh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rf, t11, t12, t13, t22
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, rmf, w
  real(c_double), intent(in) :: u(ib:ie+1,jb:je,kb:ke)
  real(c_double), intent(inout) :: uten(ib:ie+1,jb:je,kb:ke)
  real(c_double), intent(in) :: kmv(ibc:iec,jbc:jec,kbc:kec)
  call turbu(dt, one_h, ruh, one_f, one_f, one_f, one_f, uf, vh, mh, mf, rmf, rho, rf, &
             zero2, one_t, one_t, one_t, one_t, rds_, rds_, rdsf_, rdsf_, one_t3, &
             s1, s2, s3, s4, s5, s6, s7, s8, s9, u, uten, w, t11, t12, t13, t22, kmv, cm0_, &
             kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
end subroutine cm1f_turb_turbu

subroutine cm1f_turb_turbv(dt, uh, rvh, vf, mh, mf, rho, rr, rf, v, vten, w, t12, t22, t23, kmv) &
    bind(C, name="cm1f_turb_turbv")
  use iso_c_binding
  use input
  use turbtend_module, only: turbv
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), rvh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, t12, t22, t23
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w
  real(c_double), intent(in) :: v(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(inout) :: vten(ib:ie,jb:je+1,kb:ke)
  real(c_double), intent(in) :: kmv(ibc:iec,jbc:jec,kbc:kec)
  call turbv(dt, one_h, one_h, one_h, one_h, uh, one_f, rvh, vf, mh, mf, rho, rr, rf, &
             zero2, one_t, one_t, one_t, one_t, rds_, rds_, rdsf_, rdsf_, one_t3, &
             s1, s2, s3, s4, s5, s6, s7, s8, s9, v, vten, w, t12, t22, t23, kmv, cm0_, &
             kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
end subroutine cm1f_turb_turbv

subroutine cm1f_turb_turbw(dt, uh, vh, mh, mf, rho, rf, w, wten, t13, t23, t33, t22, kmh) &
    bind(C, name="cm1f_turb_turbw")
  use iso_c_binding
  use input
  use turbtend_module, only: turbw
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), vh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rf, t13, t23, t33, t22
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: mf, w
  real(c_double), intent(inout) :: wten(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: kmh(ibc:iec,jbc:jec,kbc:kec)
  call turbw(dt, one_h, one_h, one_h, one_h, uh, one_f, vh, mh, mf, rho, rf, one_t, one_t, one_t, rds_, rds_, &
             s1, s2, s3, s4, s5, s6, w, wten, t13, t23, t33, t22, kmh, cm0_, .false., diag)
end subroutine cm1f_turb_turbw

subroutine cm1f_turb_turbs(iflux, dt, uh, uf, vh, vf, mh, mf, rho, rr, rf, s, sten, khh, khv, dum7, dum8) &
    bind(C, name="cm1f_turb_turbs")
  use iso_c_binding
  use input
  use turbtend_module, only: turbs
  use cm1f_bridge_turb_mod
  implicit none
  integer(c_int), value :: iflux
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, rho, rr, rf, s, dum7, dum8
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: sten(ib:ie,jb:je,kb:ke)
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: khh, khv
  call turbs(iflux, dt, .false., one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, uf, vh, vf, zero2, &
             rds_, rds_, rdsf_, rdsf_, mh, mf, one_t, one_t, one_t, one_t, one_t, one_t, &
             one_t3, one_t3, one_t3, one_t3, &
             s1, s2, s3, s4, s5, s6, rho, rr, rf, s, sten, khh, khv, cm0_, dum7, dum8, &
             .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1)
end subroutine cm1f_turb_turbs

!-----------------------------------------------------------------------
! solve1.F inline code (verbatim copies)

subroutine cm1f_turb_tke_source(nm, defv, defh, dissten, kmh, kmv, khv, tketen) bind(C, name="cm1f_turb_tke_source")
  use iso_c_binding
  use input
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke+1) :: nm, defv, defh, dissten
  real(c_double), intent(in), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv, khv
  real(c_double), intent(inout) :: tketen(ibt:iet,jbt:jet,kbt:ket)
  integer :: i, j, k
        IF(tconfig.eq.1)THEN
          do k=2,nk
          do j=1,nj
          do i=1,ni
            tketen(i,j,k) = -khv(i,j,k)*nm(i,j,k)  &
                            -dissten(i,j,k)  &
                            +kmv(i,j,k)*max(0.0,(defv(i,j,k)+defh(i,j,k)))
          enddo
          enddo
          enddo
        ELSEIF(tconfig.eq.2)THEN
          do k=2,nk
          do j=1,nj
          do i=1,ni
            tketen(i,j,k) = -khv(i,j,k)*nm(i,j,k)  &
                            -dissten(i,j,k)  &
                            +kmv(i,j,k)*max(0.0,defv(i,j,k))   &
                            +kmh(i,j,k)*max(0.0,defh(i,j,k))
          enddo
          enddo
          enddo
        ENDIF
end subroutine cm1f_turb_tke_source

subroutine cm1f_turb_dissheat(pi0, ppi, qa, dissten, thten1) bind(C, name="cm1f_turb_dissheat")
  use iso_c_binding
  use input
  use constants
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: pi0, ppi
  real(c_double), intent(in) :: qa(ibm:iem,jbm:jem,kbm:kem,numq)
  real(c_double), intent(in) :: dissten(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: thten1(ib:ie,jb:je,kb:ke)
  real, allocatable, dimension(:,:,:) :: dum1, dum2, dum3
  integer :: i, j, k, n, k1
  real :: epsd, dheat
  integer, parameter :: rterm_ = 0
  allocate(dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke))
  dum1 = 0; dum2 = 0; dum3 = 0

      IF( eqtset.eq.2 .and. imoist.eq.1 .and. (idiss.eq.1.or.rterm_.eq.1) )THEN
        IF( nql1.ge.1 )THEN
          do k=1,nk
          do j=1,nj
          do i=1,ni
            dum2(i,j,k)=qa(i,j,k,nql1)
          enddo
          enddo
          enddo
          do n=nql1+1,nql2
            do k=1,nk
            do j=1,nj
            do i=1,ni
              dum2(i,j,k)=dum2(i,j,k)+qa(i,j,k,n)
            enddo
            enddo
            enddo
          enddo
        ELSE
          dum2 = 0.0
        ENDIF
        IF(iice.eq.1)THEN
          do k=1,nk
          do j=1,nj
          do i=1,ni
            dum3(i,j,k)=qa(i,j,k,nqs1)
          enddo
          enddo
          enddo
          do n=nqs1+1,nqs2
            do k=1,nk
            do j=1,nj
            do i=1,ni
              dum3(i,j,k)=dum3(i,j,k)+qa(i,j,k,n)
            enddo
            enddo
            enddo
          enddo
        ELSE
          dum3 = 0.0
        ENDIF
        DO k=1,nk
        do j=1,nj
        do i=1,ni
          dum1(i,j,k)=cv+cvv*qa(i,j,k,nqv)+cpl*dum2(i,j,k)+cpi*dum3(i,j,k)
        enddo
        enddo
        ENDDO
      ELSE
        DO k=1,nk
        do j=1,nj
        do i=1,ni
          dum1(i,j,k)=cv
        enddo
        enddo
        ENDDO
      ENDIF

      IF( idiss.eq.1 )THEN
        k1 = 1
        if(imoist.eq.1.and.eqtset.eq.2)then
          do k=k1,nk
          do j=1,nj
          do i=1,ni
            epsd = 0.5*(dissten(i,j,k)+dissten(i,j,k+1))
            dheat=epsd/( cpdcv*dum1(i,j,k)*(pi0(i,j,k)+ppi(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
        else
          do k=k1,nk
          do j=1,nj
          do i=1,ni
            epsd = 0.5*(dissten(i,j,k)+dissten(i,j,k+1))
            dheat=epsd/( cp*(pi0(i,j,k)+ppi(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
        endif
      ENDIF
end subroutine cm1f_turb_dissheat

subroutine cm1f_turb_vimpl(khv, mf, rf, mh, rr, dum7, dum8) bind(C, name="cm1f_turb_vimpl")
  use iso_c_binding
  use input
  implicit none
  real(c_double), intent(in) :: khv(ibc:iec,jbc:jec,kbc:kec), mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: rf, mh, rr
  real(c_double), intent(inout), dimension(ib:ie,jb:je,kb:ke) :: dum7, dum8
  integer :: i, j, k
          do k=1,nk
          do j=1,nj
          do i=1,ni
            dum7(i,j,k) = khv(i,j,k  )*mf(i,j,k  )*rf(i,j,k  )*mh(i,j,k)*rr(i,j,k)
            dum8(i,j,k) = khv(i,j,k+1)*mf(i,j,k+1)*rf(i,j,k+1)*mh(i,j,k)*rr(i,j,k)
          enddo
          enddo
          enddo
end subroutine cm1f_turb_vimpl

! turb.F sfc_and_turb, "facheck" block (verbatim): ramp of the turbulence
! "constants" with ramp_sgs>=1.
subroutine cm1f_turb_ramp(rtime, ramp_time_, cm0, cme, csm, ce1, ce2) bind(C, name="cm1f_turb_ramp")
  use iso_c_binding
  use input
  use constants
  implicit none
  real(c_double), value :: rtime, ramp_time_
  real(c_double), intent(in) :: cm0(ib:ie,jb:je)
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: cme, csm, ce1, ce2
  real :: turbfac
  integer :: i, j, k
  ramp_time = ramp_time_
        turbfac = (rtime-0.0)/ramp_time
        turbfac = max(0.0,turbfac)
        turbfac = min(1.0,turbfac)
        do k=1,nk+1
        do j=1,nj
        do i=1,ni
          if( cm0(i,j).gt.cmemin )then
            cme(i,j,k) = max( 0.001 , turbfac*cm0(i,j) )
            ce1(i,j,k) = cme(i,j,k) * c_l * c_l * ( 1.0 / ri_c - 1.0 )
            ce2(i,j,k) = max( 0.0 , cme(i,j,k) * pi * pi - ce1(i,j,k) )
            csm(i,j,k) = ( cme(i,j,k) * cme(i,j,k) * cme(i,j,k) / ( ce1(i,j,k) + ce2(i,j,k) ) )**0.25   ! Smagorinsky constant
          endif
        enddo
        enddo
        enddo
end subroutine cm1f_turb_ramp

! bc.F bc2d on one level of a w-level array (as tkebc calls it).
subroutine cm1f_turb_bc2d_level(w, k) bind(C, name="cm1f_turb_bc2d_level")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), intent(inout) :: w(ib:ie,jb:je,kb:ke+1)
  integer(c_int), value :: k
  call bc2d(w(ib,jb,k))
end subroutine cm1f_turb_bc2d_level

! solve2.F TKE step (verbatim, cm1setup/=4) followed by bcw(tke3d,1).
subroutine cm1f_turb_tke_update(dttmp, tkea, wten, tke3d) bind(C, name="cm1f_turb_tke_update")
  use iso_c_binding
  use input
  use bc_module
  implicit none
  real(c_double), value :: dttmp
  real(c_double), intent(in), dimension(ibt:iet,jbt:jet,kbt:ket) :: tkea, wten
  real(c_double), intent(inout) :: tke3d(ibt:iet,jbt:jet,kbt:ket)
  integer :: i, j, k
        do k=2,nk
          do j=1,nj
          do i=1,ni
            tke3d(i,j,k)=tkea(i,j,k)+dttmp*wten(i,j,k)
            if(tke3d(i,j,k).lt.1.0e-10) tke3d(i,j,k)=0.0
          enddo
          enddo
        enddo
          call bcw(tke3d,1)
end subroutine cm1f_turb_tke_update
