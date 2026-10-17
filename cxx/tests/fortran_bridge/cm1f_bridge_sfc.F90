! bind(C) shims around CM1's original surface layer (sfcmodel=1) and the
! pieces of the LES turbulence it feeds (bbc=3, surface fluxes):
!   sfcphys.F       getcecd, sfcflux, sfcdiags
!   init_surface.F  init_surface (+ getlanduse; reads ./LANDUSE.TBL)
!   turb.F          tkebc (bbc=3); the surface block and the "bc/comms"
!                   block of sfc_and_turb (verbatim copies below);
!                   gettau/sfcstress (bbc=3) is in cm1f_bridge_turbpriv.F90
!   turbtend.F      turbs with dosfcflx (surface flux of theta/qv)
!   solve1.F        dissipative heating with the bbc=3 surface term
!                   (verbatim copy)
!   init3d.F        random-number seeding and draws (verbatim copy)
!
! cm1f_turb_setup (cm1f_bridge_turb.F90) sets the grid bounds and the
! turbulence options first; cm1f_sfc_options then switches the `input`
! module to sfcmodel=1, bbc=3 with the surface options given.

subroutine cm1f_sfc_options(iopt, ropt) bind(C, name="cm1f_sfc_options")
  use iso_c_binding
  use input
  implicit none
  !  iopt (must match sfc_tests.cpp):
  !   1 isfcflx 2 cecd 3 set_znt 4 set_ust 5 set_flx 6 pertflx 7 lu0 8 season
  !  ropt: 1 cnstce 2 cnstcd 3 cnst_znt 4 cnst_ust 5 cnst_shflx 6 cnst_lhflx 7 tsk0 8 xland0
  integer(c_int), intent(in) :: iopt(8)
  real(c_double), intent(in) :: ropt(8)
  sfcmodel = 1; oceanmodel = 1; bbc = 3; tbc = 1; ipbl = 0; cm1setup = 1
  isfcflx = iopt(1); cecd = iopt(2); set_znt = iopt(3); set_ust = iopt(4); set_flx = iopt(5)
  pertflx = iopt(6); lu0 = iopt(7); season = iopt(8)
  cnstce = ropt(1); cnstcd = ropt(2); cnst_znt = ropt(3); cnst_ust = ropt(4)
  cnst_shflx = ropt(5); cnst_lhflx = ropt(6); tsk0 = ropt(7); xland0 = ropt(8)
  tmn0 = 297.28
  initsfc = 1; testcase = 0; imove = 0; use_avg_sfc = .false.; use_pbl = .false.
  ibl = ib; iel = ie; jbl = jb; jel = je
  ibmyj = 1; iemyj = 1; jbmyj = 1; jemyj = 1
  dowr = .false.; output_sfcflx = 0
end subroutine cm1f_sfc_options

!-----------------------------------------------------------------------
! sfcphys.F

subroutine cm1f_sfc_getcecd(za, u1, v1, s1, u10, v10, s10, xland, znt, ust, cd, ch, cq) &
    bind(C, name="cm1f_sfc_getcecd")
  use iso_c_binding
  use input
  use sfcphys_module, only: getcecd
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: za, xland
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: u1, v1, s1, u10, v10, s10, znt, ust, cd, ch, cq
  real, allocatable :: xh(:)
  double precision :: au, av, as, at
  allocate(xh(ib:ie)); xh = 0
  au = 0; av = 0; as = 0; at = 0
  call getcecd(xh, u1, v1, s1, za, u10, v10, s10, xland, znt, ust, cd, ch, cq, au, av, as, at)
end subroutine cm1f_sfc_getcecd

subroutine cm1f_sfc_sfcflux(dt, ruh, rvh, rf, psfc, ch, cq, th0, tsk, thflux, qvflux, mavail, &
                            s1, s10, tha, qva, qsfc, znt) bind(C, name="cm1f_sfc_sfcflux")
  use iso_c_binding
  use input
  use sfcphys_module, only: sfcflux
  implicit none
  real(c_double), value :: dt
  real(c_double), intent(in) :: ruh(ib:ie), rvh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: rf, th0, tha
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: ch, cq, tsk, mavail, s1
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: psfc, thflux, qvflux, s10, qsfc, znt
  real(c_double), intent(in) :: qva(ibm:iem,jbm:jem,kbm:kem)
  real, allocatable :: xf(:), d2(:,:), d3(:,:,:)
  double precision :: qb
  allocate(xf(ib:ie+1), d2(ib:ie,jb:je), d3(ib:ie,jb:je,kb:ke))
  xf = 0; d2 = 0; d3 = 0; qb = 0
  call sfcflux(dt, ruh, xf, rvh, d2, ch, cq, d3, d3, th0, d3, tsk, thflux, qvflux, mavail, &
               d3, rf, d2, d2, s1, d3, tha, qva, qb, psfc, d2, d2, s10, qsfc, znt, 0.0)
end subroutine cm1f_sfc_sfcflux

subroutine cm1f_sfc_sfcdiags(tsk, thflux, qvflux, cd, ch, cq, u1, v1, s1, wspd, xland, psfc, qsfc, &
                             u10, v10, hfx, qfx, cda, znt, gz1oz0, br, hpbl, dsxy, th2, t2, q2, &
                             za, th0, tha, rf, qa) bind(C, name="cm1f_sfc_sfcdiags")
  use iso_c_binding
  use input
  use sfcphys_module, only: sfcdiags
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: tsk, thflux, qvflux, cd, ch, cq, u1, v1, s1, &
                                                       xland, psfc, za
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: wspd, qsfc, u10, v10, hfx, qfx, cda, znt, &
                                                          gz1oz0, br, th2, t2, q2, hpbl, dsxy
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: th0, tha, rf, qa
  real, allocatable, dimension(:,:) :: zs, pi0s, psim, psih, zol, mol, fm, fh
  real, allocatable :: d3(:,:,:)
  allocate(zs(ib:ie,jb:je), pi0s(ib:ie,jb:je), psim(ib:ie,jb:je), psih(ib:ie,jb:je), zol(ib:ie,jb:je), &
           mol(ib:ie,jb:je), fm(ib:ie,jb:je), fh(ib:ie,jb:je), d3(ib:ie,jb:je,kb:ke))
  zs = 0; pi0s = 0; psim = 0; psih = 0; zol = 0; mol = 0; fm = 0; fh = 0; d3 = 0
  call sfcdiags(tsk, thflux, qvflux, cd, ch, cq, u1, v1, s1, wspd, &
                xland, psfc, qsfc, u10, v10, hfx, qfx, cda, znt, gz1oz0, &
                psim, psih, br, zol, mol, hpbl, dsxy, th2, t2, q2, fm, fh, &
                zs, za, pi0s, d3, th0, d3, tha, d3, rf, qa)
end subroutine cm1f_sfc_sfcdiags

!-----------------------------------------------------------------------
! init_surface.F (initsfc=1; getlanduse reads ./LANDUSE.TBL)

subroutine cm1f_sfc_init_surface(ruh, rvh, tsk, xland, znt, mavail, dsxy, dosfcflx_out) &
    bind(C, name="cm1f_sfc_init_surface")
  use iso_c_binding
  use input
  use init_surface_module, only: init_surface
  implicit none
  real(c_double), intent(in) :: ruh(ib:ie), rvh(jb:je)
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: tsk, xland, znt, mavail, dsxy
  integer(c_int), intent(out) :: dosfcflx_out
  integer, parameter :: nsl = 5
  logical :: dosfcflx
  integer, allocatable :: lu_index(:,:)
  real, allocatable :: h1(:), f1(:), hj(:), fj(:), rznt(:,:), d2(:,:), d3(:,:,:), tslb(:,:,:), z0base(:,:)
  real :: slab_zs(nsl), slab_dzs(nsl)
  real, allocatable, dimension(:,:) :: emiss, thc, albd, tmn, tml, t0ml, hml, h0ml, huml, hvml, tmoml
  allocate(lu_index(ibl:iel,jbl:jel), h1(ib:ie), f1(ib:ie+1), hj(jb:je), fj(jb:je+1), rznt(ib:ie,jb:je), &
           d2(ib:ie,jb:je), d3(ib:ie,jb:je,kb:ke), tslb(ibl:iel,jbl:jel,nsl), z0base(ibmyj:iemyj,jbmyj:jemyj))
  allocate(emiss(ibl:iel,jbl:jel), thc(ibl:iel,jbl:jel), albd(ibl:iel,jbl:jel), tmn(ibl:iel,jbl:jel), &
           tml(ibl:iel,jbl:jel), t0ml(ibl:iel,jbl:jel), hml(ibl:iel,jbl:jel), h0ml(ibl:iel,jbl:jel), &
           huml(ibl:iel,jbl:jel), hvml(ibl:iel,jbl:jel), tmoml(ibl:iel,jbl:jel))
  lu_index = 0; h1 = 0; f1 = 0; hj = 0; fj = 0; rznt = 0; d2 = 0; d3 = 0; tslb = 0; z0base = 0
  slab_zs = 0; slab_dzs = 0
  emiss = 0; thc = 0; albd = 0; tmn = 0; tml = 0; t0ml = 0; hml = 0; h0ml = 0; huml = 0; hvml = 0; tmoml = 0
  dosfcflx = .false.
  call init_surface(nsl, dosfcflx, h1, ruh, f1, hj, rvh, fj, &
                    lu_index, xland, tsk, slab_zs, slab_dzs, tslb, &
                    emiss, thc, albd, znt, rznt, mavail, dsxy, d2, d3, &
                    tmn, tml, t0ml, hml, h0ml, huml, hvml, tmoml, z0base)
  dosfcflx_out = 0
  if (dosfcflx) dosfcflx_out = 1
end subroutine cm1f_sfc_init_surface

!-----------------------------------------------------------------------
! turb.F

subroutine cm1f_sfc_tkebc(tkea, kmh, kmv, khh, khv, zntmp, ust) bind(C, name="cm1f_sfc_tkebc")
  use iso_c_binding
  use input
  use turb_module, only: tkebc
  use cm1f_bridge_turb_mod
  implicit none
  real(c_double), intent(inout) :: tkea(ibt:iet,jbt:jet,kbt:ket)
  real(c_double), intent(inout), dimension(ibc:iec,jbc:jec,kbc:kec) :: kmh, kmv, khh, khv
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: zntmp, ust
  call tkebc(tkea, kmh, kmv, khh, khv, zntmp, ust, zero2, &
             cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
end subroutine cm1f_sfc_tkebc

! sfc_and_turb, bbc=3 surface block (verbatim, imove=0, terrain_flag=.false.,
! kval=1): psfc, ground-relative winds, u1/v1/s1/t1, za.
subroutine cm1f_sfc_layer_k1(c_gs1, c_gs2, c_gs3, prs, ua, va, u0, v0, th0, tha, psfc, u1, v1, s1, t1, za) &
    bind(C, name="cm1f_sfc_layer_k1")
  use iso_c_binding
  use input
  implicit none
  real(c_double), value :: c_gs1, c_gs2, c_gs3
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: prs, th0, tha
  real(c_double), intent(in), dimension(ib:ie+1,jb:je,kb:ke) :: ua, u0
  real(c_double), intent(in), dimension(ib:ie,jb:je+1,kb:ke) :: va, v0
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: psfc, u1, v1, s1, t1, za
  real, allocatable :: ugr(:,:,:), vgr(:,:,:), zh(:,:,:)
  integer :: i, j, k, kval
  allocate(ugr(ib:ie+1,jb:je,kb:ke), vgr(ib:ie,jb:je+1,kb:ke), zh(ib:ie,jb:je,kb:ke))
  ugr = 0; vgr = 0; zh = 0

      do j=1,nj
      do i=1,ni
        psfc(i,j) = c_gs1*prs(i,j,1)+c_gs2*prs(i,j,2)+c_gs3*prs(i,j,3)
      enddo
      enddo

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

        kval = 1
          do j=1,nj
          do i=1,ni
            ! use ground-relative winds:
            u1(i,j) = 0.5*( ugr(i,j,kval) + ugr(i+1,j,kval) )
            v1(i,j) = 0.5*( vgr(i,j,kval) + vgr(i,j+1,kval) )
            s1(i,j) = sqrt(u1(i,j)**2+v1(i,j)**2)
            t1(i,j) = th0(i,j,kval)+tha(i,j,kval)
          enddo
          enddo
end subroutine cm1f_sfc_layer_k1

! sfc_and_turb "bc/comms" block for bbc=3 (serial, sfcmodel/=4).
subroutine cm1f_sfc_bcs(ust, u1, v1, s1, znt, wspd, zntmp) bind(C, name="cm1f_sfc_bcs")
  use iso_c_binding
  use input
  use bc_module, only: bc2d
  implicit none
  real(c_double), intent(inout), dimension(ib:ie,jb:je) :: ust, u1, v1, s1, znt, wspd, zntmp
  integer :: i, j
    call bc2d(ust)
    call bc2d(u1)
    call bc2d(v1)
    call bc2d(s1)
    call bc2d(znt)
    call bc2d(wspd)
      do j=0,nj+1
      do i=0,ni+1
        zntmp(i,j) = znt(i,j)
      enddo
      enddo
end subroutine cm1f_sfc_bcs

!-----------------------------------------------------------------------
! turbtend.F: turbs with surface fluxes (dosfcflx=.true.)

subroutine cm1f_sfc_turbs(iflux, dt, uh, uf, vh, vf, mh, mf, rho, rr, rf, s, sten, khh, khv, dum7, dum8, sflux) &
    bind(C, name="cm1f_sfc_turbs")
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
  real(c_double), intent(in) :: sflux(ib:ie,jb:je)
  call turbs(iflux, dt, .true., one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, uf, vh, vf, sflux, &
             rds_, rds_, rdsf_, rdsf_, mh, mf, one_t, one_t, one_t, one_t, one_t, one_t, &
             one_t3, one_t3, one_t3, one_t3, &
             s1, s2, s3, s4, s5, s6, rho, rr, rf, s, sten, khh, khv, cm0_, dum7, dum8, &
             .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1)
end subroutine cm1f_sfc_turbs

!-----------------------------------------------------------------------
! solve1.F dissipative heating with bbc=3 (verbatim; cvm as in
! cm1f_turb_dissheat)

subroutine cm1f_sfc_dissheat(pi0, ppi, qa, dissten, zf, ust, zntmp, thten1) bind(C, name="cm1f_sfc_dissheat")
  use iso_c_binding
  use input
  use constants
  implicit none
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: pi0, ppi
  real(c_double), intent(in) :: qa(ibm:iem,jbm:jem,kbm:kem,numq)
  real(c_double), intent(in) :: dissten(ib:ie,jb:je,kb:ke+1), zf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je) :: ust, zntmp
  real(c_double), intent(inout) :: thten1(ib:ie,jb:je,kb:ke)
  real, allocatable, dimension(:,:,:) :: dum1, dum2, dum3
  integer :: i, j, k, n, k1
  real :: epsd, dheat, dz1
  allocate(dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke))
  dum1 = 0; dum2 = 0; dum3 = 0

      IF( eqtset.eq.2 .and. imoist.eq.1 .and. idiss.eq.1 )THEN
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
        ENDIF
        DO k=1,nk
        do j=1,nj
        do i=1,ni
          dum1(i,j,k)=cv+cvv*qa(i,j,k,nqv)+cpl*dum2(i,j,k)+cpi*dum3(i,j,k)
        enddo
        enddo
        ENDDO
      ELSE
        dum1 = cv
      ENDIF

      IF( idiss.eq.1 )THEN
        ! note:  dissten array stores epsilon (dissipation rate) at w points
        if( bbc.eq.3 )then
          k1 = 2
        else
          k1 = 1
        endif
        if(imoist.eq.1.and.eqtset.eq.2)then
          ! moist, new equations:
          do k=k1,nk
          do j=1,nj
          do i=1,ni
            epsd = 0.5*(dissten(i,j,k)+dissten(i,j,k+1))
            dheat=epsd/( cpdcv*dum1(i,j,k)*(pi0(i,j,k)+ppi(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
          if( bbc.eq.3 )then
            k = 1
            do j=1,nj
            do i=1,ni
              dz1 = zf(i,j,2)-zf(i,j,1)
              epsd = (ust(i,j)**3)*alog((dz1+zntmp(i,j))/zntmp(i,j))/(karman*dz1)
              dheat=epsd/( cpdcv*dum1(i,j,k)*(pi0(i,j,k)+ppi(i,j,k)) )
              thten1(i,j,k)=thten1(i,j,k)+dheat
            enddo
            enddo
          endif
        else
          ! traditional cloud-modeling equations (also dry equations):
          do k=k1,nk
          do j=1,nj
          do i=1,ni
            epsd = 0.5*(dissten(i,j,k)+dissten(i,j,k+1))
            dheat=epsd/( cp*(pi0(i,j,k)+ppi(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
          if( bbc.eq.3 )then
            k = 1
            do j=1,nj
            do i=1,ni
              dz1 = zf(i,j,2)-zf(i,j,1)
              epsd = (ust(i,j)**3)*alog((dz1+zntmp(i,j))/zntmp(i,j))/(karman*dz1)
              dheat=epsd/( cp*(pi0(i,j,k)+ppi(i,j,k)) )
              thten1(i,j,k)=thten1(i,j,k)+dheat
            enddo
            enddo
          endif
        endif
      ENDIF
end subroutine cm1f_sfc_dissheat

!-----------------------------------------------------------------------
! init3d.F: seed gfortran's generator as CM1 does (use_truly_random_pert =
! .false.), then return n draws of random_number (the first one is init3d's
! "rand-1"). Also returns the seed size.

subroutine cm1f_random_cm1(n, out, seed_size) bind(C, name="cm1f_random_cm1")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(out) :: out(n)
  integer(c_int), intent(out) :: seed_size
  integer :: k, m
  integer, allocatable :: sand(:)
  real :: rand
        call random_seed(size=k)
        k = max(2,k)
        allocate( sand(k) )
        do m=1,k
          sand(m) = nint( 2.0e9*(2.0*float(m-1)/float(k-1)-1.0) )
        enddo
        call random_seed(put=sand(1:k))
        deallocate( sand )
  seed_size = k
  do m = 1, n
    call random_number(rand)
    out(m) = rand
  enddo
end subroutine cm1f_random_cm1
