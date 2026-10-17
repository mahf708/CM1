! bind(C) shims around CM1's advection (adv.F: advs, advu, advv, advw).
!
! cm1f_adv_setup sets the `input`/`ib_module` state the routines read
! (serial run, no terrain, no immersed boundaries); the other shims call the
! original routines with scratch/budget arguments allocated here. Arrays from
! C are contiguous, Fortran-ordered, with the bounds given by the setup.
module cm1f_bridge_adv_mod
  implicit none
  ! input order of iopt (must match adv_tests.cpp)
  !  1 hadvordrs 2 vadvordrs 3 hadvordrv 4 vadvordrv 5 advwenos 6 advwenov
  !  7 weno_order 8 idiff 9 mdiff 10 difforder 11 wbc 12 ebc 13 sbc 14 nbc
end module cm1f_bridge_adv_mod

subroutine cm1f_adv_setup(ni_, nj_, nk_, ngxy_, ngz_, iopt, ropt) bind(C, name="cm1f_adv_setup")
  use iso_c_binding
  use input
  use ib_module, only: ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  implicit none
  integer(c_int), value :: ni_, nj_, nk_, ngxy_, ngz_
  integer(c_int), intent(in) :: iopt(14)
  real(c_double), intent(in) :: ropt(4)

  nx = ni_; ny = nj_; nz = nk_
  ni = ni_; nj = nj_; nk = nk_; nkp1 = nk + 1
  ngxy = ngxy_; ngz = ngz_
  ib = 1 - ngxy; ie = ni + ngxy
  jb = 1 - ngxy; je = nj + ngxy
  kb = 1 - ngz;  ke = nk + ngz
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
  imp = max(1, ni); jmp = max(1, nj); kmp = max(2, nk); cmp = ngxy
  ibib = 1; ieib = 1; jbib = 1; jeib = 1; kbib = 1; keib = 1; kmaxib = 1
  ibdt = 1; iedt = 1; jbdt = 1; jedt = 1; kbdt = 1; kedt = 1; ntdiag = 1
  ibdq = 1; iedq = 1; jbdq = 1; jedq = 1; kbdq = 1; kedq = 1; nqdiag = 1
  ibdv = 1; iedv = 1; jbdv = 1; jedv = 1; kbdv = 1; kedv = 1
  nudiag = 1; nvdiag = 1; nwdiag = 1
  ib3d = 1; ie3d = 1; jb3d = 1; je3d = 1; kb3d = 1; ke3d = 1; nout3d = 1

  hadvordrs = iopt(1); vadvordrs = iopt(2); hadvordrv = iopt(3); vadvordrv = iopt(4)
  advwenos = iopt(5); advwenov = iopt(6); weno_order = iopt(7)
  idiff = iopt(8); mdiff = iopt(9); difforder = iopt(10)
  wbc = iopt(11); ebc = iopt(12); sbc = iopt(13); nbc = iopt(14)
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc .ne. 1) ibw = 1
  if (ebc .ne. 1) ibe = 1
  if (sbc .ne. 1) ibs = 1
  if (nbc .ne. 1) ibn = 1

  rdx = ropt(1); rdy = ropt(2); rdz = ropt(3); kdiff6 = ropt(4)
  dx = 1.0/rdx; dy = 1.0/rdy; dz = 1.0/rdz

  terrain_flag = .false.; axisymm = 0; do_ib = .false.; dolsw = .false.
  stat_qsrc = 0; pdscheme = 1; timestats = 0
  smeps = 1.0e-60
end subroutine cm1f_adv_setup

subroutine cm1f_advs(nrk, wflag, pdef, pdefweno, dt, weps, diffit, hadvorder, vadvorder, &
                     uh, vh, mh, c1, c2, rho0, rr0, rf0, rrf0, rru, rrv, rrw, divx, rho, rr, &
                     s0, s, sten) bind(C, name="cm1f_advs")
  use iso_c_binding
  use input
  use adv_module
  use ib_module, only: ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  implicit none
  integer(c_int), value :: nrk, wflag, pdef, pdefweno, diffit, hadvorder, vadvorder
  real(c_double), value :: dt, weps
  real(c_double), intent(in) :: uh(ib:ie), vh(jb:je)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho0, rr0, rf0, rrf0, divx, rho, rr, s0, s
  real(c_double), intent(in) :: rru(ib:ie+1,jb:je,kb:ke), rrv(ib:ie,jb:je+1,kb:ke), rrw(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: sten(ib:ie,jb:je,kb:ke)

  real, allocatable, dimension(:,:,:) :: advx, advy, advz, dumx, dumy, dumz, mass, subs
  logical, allocatable :: flag(:,:,:), bndy(:,:,:)
  real, allocatable, dimension(:,:,:) :: sw31, sw32, se31, se32, ss31, ss32, sn31, sn32
  real, allocatable :: gz(:,:), rdsf(:), wprof(:), diag(:,:,:,:), out3d(:,:,:,:), xf(:)
  double precision, allocatable :: dumk1(:), dumk2(:)
  integer, allocatable :: kbdy(:,:), hflx(:,:,:)
  double precision :: bsq

  allocate(advx(ib:ie,jb:je,kb:ke), advy(ib:ie,jb:je,kb:ke), advz(ib:ie,jb:je,kb:ke), &
           dumx(ib:ie,jb:je,kb:ke), dumy(ib:ie,jb:je,kb:ke), dumz(ib:ie,jb:je,kb:ke), &
           mass(ib:ie,jb:je,kb:ke), subs(ib:ie,jb:je,kb:ke), flag(ib:ie,jb:je,kb:ke))
  advx = 0; advy = 0; advz = 0; dumx = 0; dumy = 0; dumz = 0; mass = 0; subs = 0; flag = .false.
  allocate(sw31(cmp,jmp,kmp), sw32(cmp,jmp,kmp), se31(cmp,jmp,kmp), se32(cmp,jmp,kmp))
  allocate(ss31(imp,cmp,kmp), ss32(imp,cmp,kmp), sn31(imp,cmp,kmp), sn32(imp,cmp,kmp))
  allocate(gz(itb:ite,jtb:jte), rdsf(kb:ke+1), wprof(kb:ke), dumk1(kb:ke), dumk2(kb:ke))
  allocate(diag(1,1,1,1), out3d(1,1,1,1), bndy(1,1,1), kbdy(1,1), hflx(1,1,1), xf(ib:ie+1))
  xf = 0
  gz = 1; rdsf = rdz; wprof = 0; dumk1 = 0; dumk2 = 0; diag = 0; out3d = 0
  bndy = .false.; kbdy = 1; hflx = 0; bsq = 0

  call advs(nrk, wflag, 0, bsq, uh, uh, uh, uh, uh, uh, xf, vh, vh, gz, gz, mh, mh, &
            rho0, rr0, rf0, rrf0, advx, advy, advz, dumx, dumy, dumz, mass, subs, divx, &
            rru, rrv, rrw, s0, s, sten, pdef, pdefweno, dt, weps, &
            flag, sw31, sw32, se31, se32, ss31, ss32, sn31, sn32, rdsf, c1, c2, rho, rr, diffit, &
            .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1, 1, 1, 1, wprof, dumk1, dumk2, &
            hadvorder, vadvorder, kbdy, bndy, hflx, hflx, hflx, hflx, out3d)
end subroutine cm1f_advs

subroutine cm1f_advu(nrk, dt, uh, uf, vh, vf, mh, mf, c1, c2, rho0, rr0, rf0, rrf0, &
                     rru, rrv, rrw, divx, rho, u3d, v3d, w3d, uten) bind(C, name="cm1f_advu")
  use iso_c_binding
  use input
  use adv_module
  implicit none
  integer(c_int), value :: nrk
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho0, rr0, rf0, rrf0, divx, rho
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: rru(ib:ie+1,jb:je,kb:ke), rrv(ib:ie,jb:je+1,kb:ke), rrw(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: u3d(ib:ie+1,jb:je,kb:ke), v3d(ib:ie,jb:je+1,kb:ke), w3d(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: uten(ib:ie+1,jb:je,kb:ke)

  real, allocatable, dimension(:,:,:) :: d1, d2, d3, d4, d5, d6, d7
  real, allocatable :: gz(:,:), rdsf(:), wprof(:), diag(:,:,:,:)
  logical, allocatable :: bndy(:,:,:)
  integer, allocatable :: kbdy(:,:), hflx(:,:,:)

  allocate(d1(ib:ie,jb:je,kb:ke), d2(ib:ie,jb:je,kb:ke), d3(ib:ie,jb:je,kb:ke), d4(ib:ie,jb:je,kb:ke), &
           d5(ib:ie,jb:je,kb:ke), d6(ib:ie,jb:je,kb:ke), d7(ib:ie,jb:je,kb:ke))
  d1 = 0; d2 = 0; d3 = 0; d4 = 0; d5 = 0; d6 = 0; d7 = 0
  allocate(gz(itb:ite,jtb:jte), rdsf(kb:ke+1), wprof(kb:ke), diag(1,1,1,1), bndy(1,1,1), kbdy(1,1), hflx(1,1,1))
  gz = 1; rdsf = rdz; wprof = 0; diag = 0; bndy = .false.; kbdy = 1; hflx = 0

  call advu(nrk, uh, uh, uh, uf, uf, uf, uf, uf, vh, gz, gz, gz, mh, rho0, rr0, rf0, rrf0, &
            d1, d2, d3, d4, d5, d6, d7, divx, rru, u3d, uten, rrv, rrw, rdsf, c1, c2, rho, dt, &
            .false., diag, wprof, bndy, kbdy, hflx, hflx, hflx, hflx, vf, mf, v3d, w3d)
end subroutine cm1f_advu

subroutine cm1f_advv(nrk, dt, uh, uf, vh, vf, mh, mf, c1, c2, rho0, rr0, rf0, rrf0, &
                     rru, rrv, rrw, divx, rho, u3d, v3d, w3d, vten) bind(C, name="cm1f_advv")
  use iso_c_binding
  use input
  use adv_module
  implicit none
  integer(c_int), value :: nrk
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho0, rr0, rf0, rrf0, divx, rho
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: rru(ib:ie+1,jb:je,kb:ke), rrv(ib:ie,jb:je+1,kb:ke), rrw(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: u3d(ib:ie+1,jb:je,kb:ke), v3d(ib:ie,jb:je+1,kb:ke), w3d(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: vten(ib:ie,jb:je+1,kb:ke)

  real, allocatable, dimension(:,:,:) :: d1, d2, d3, d4, d5, d6, d7
  real, allocatable :: gz(:,:), rdsf(:), wprof(:), diag(:,:,:,:)
  logical, allocatable :: bndy(:,:,:)
  integer, allocatable :: kbdy(:,:), hflx(:,:,:)

  allocate(d1(ib:ie,jb:je,kb:ke), d2(ib:ie,jb:je,kb:ke), d3(ib:ie,jb:je,kb:ke), d4(ib:ie,jb:je,kb:ke), &
           d5(ib:ie,jb:je,kb:ke), d6(ib:ie,jb:je,kb:ke), d7(ib:ie,jb:je,kb:ke))
  d1 = 0; d2 = 0; d3 = 0; d4 = 0; d5 = 0; d6 = 0; d7 = 0
  allocate(gz(itb:ite,jtb:jte), rdsf(kb:ke+1), wprof(kb:ke), diag(1,1,1,1), bndy(1,1,1), kbdy(1,1), hflx(1,1,1))
  gz = 1; rdsf = rdz; wprof = 0; diag = 0; bndy = .false.; kbdy = 1; hflx = 0

  call advv(nrk, uh, uh, uh, uh, uh, uf, vh, vf, gz, gz, gz, mh, rho0, rr0, rf0, rrf0, &
            d1, d2, d3, d4, d5, d6, d7, divx, rru, rrv, v3d, vten, rrw, rdsf, c1, c2, rho, dt, &
            .false., diag, wprof, bndy, kbdy, hflx, hflx, hflx, hflx, uf, mf, u3d, w3d)
end subroutine cm1f_advv

subroutine cm1f_advw(nrk, dt, hadvorder, vadvorder, advweno, iamw_, uh, uf, vh, vf, mh, mf, c1, c2, &
                     rho0, rr0, rf0, rrf0, rru, rrv, rrw, divx, rho, u3d, v3d, w3d, wten) bind(C, name="cm1f_advw")
  use iso_c_binding
  use input
  use adv_module
  implicit none
  integer(c_int), value :: nrk, hadvorder, vadvorder, advweno, iamw_
  real(c_double), value :: dt
  real(c_double), intent(in) :: uh(ib:ie), uf(ib:ie+1), vh(jb:je), vf(jb:je+1)
  real(c_double), intent(in), dimension(ib:ie,jb:je,kb:ke) :: mh, c1, c2, rho0, rr0, rf0, rrf0, divx, rho
  real(c_double), intent(in) :: mf(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: rru(ib:ie+1,jb:je,kb:ke), rrv(ib:ie,jb:je+1,kb:ke), rrw(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(in) :: u3d(ib:ie+1,jb:je,kb:ke), v3d(ib:ie,jb:je+1,kb:ke), w3d(ib:ie,jb:je,kb:ke+1)
  real(c_double), intent(inout) :: wten(ib:ie,jb:je,kb:ke+1)

  real, allocatable, dimension(:,:,:) :: d1, d2, d3, d4, d5, d6, d7
  real, allocatable :: gz(:,:), rds(:), rdsf(:), diag(:,:,:,:)
  logical, allocatable :: bndy(:,:,:)
  integer, allocatable :: kbdy(:,:), hflx(:,:,:)
  logical :: iamw

  allocate(d1(ib:ie,jb:je,kb:ke), d2(ib:ie,jb:je,kb:ke), d3(ib:ie,jb:je,kb:ke), d4(ib:ie,jb:je,kb:ke), &
           d5(ib:ie,jb:je,kb:ke), d6(ib:ie,jb:je,kb:ke), d7(ib:ie,jb:je,kb:ke))
  d1 = 0; d2 = 0; d3 = 0; d4 = 0; d5 = 0; d6 = 0; d7 = 0
  allocate(gz(itb:ite,jtb:jte), rds(kb:ke), rdsf(kb:ke+1), diag(1,1,1,1), bndy(1,1,1), kbdy(1,1), hflx(1,1,1))
  gz = 1; rds = rdz; rdsf = rdz; diag = 0; bndy = .false.; kbdy = 1; hflx = 0
  iamw = (iamw_ .ne. 0)

  call advw(nrk, uh, uh, uh, uh, uh, uf, vh, gz, gz, mh, mf, rho0, rr0, rf0, rrf0, &
            d1, d2, d3, d4, d5, d6, d7, divx, rru, rrv, rrw, w3d, wten, rds, rdsf, c1, c2, rho, dt, &
            .false., diag, hadvorder, vadvorder, advweno, bndy, kbdy, uf, vf, u3d, v3d, &
            hflx, hflx, hflx, hflx, iamw)
end subroutine cm1f_advw
