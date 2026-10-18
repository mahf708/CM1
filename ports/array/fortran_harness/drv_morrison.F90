! Harness driver for the Morrison scheme (src/morrison.F), linked against
! the public variant of module_mp_GRAUPEL (tests/fortran.py public_module):
! the module's private routines (M2005MICRO_GRAUPEL via MP_GRAUPEL, satadj2,
! GAMMA, DERF1) and constants are reachable from here.
! params(1) = mode:
!   1  constants:  params = [1, ihail, ndcnst, inum]       -> consts(124)
!   2  GAMMA/DERF1: params = [2, n]; x(n)                  -> gam(n), erf(n)
!   3  satadj2:    params = [3, n, eqtset, tsmall]; t, p, qv, qc, qr, qi, qs,
!                  qg, rho (n each)                        -> t, p, qv, qc
!   4  MP_GRAUPEL: params = [4, ni, nj, nk, dx, dy, dz, dt, ihail, ndcnst,
!                  inum, eqtset, tsmall, testcase, nrain]; 3-D fields  -> 3-D fields
module morr_consts
  implicit none
contains
  subroutine dump_consts(c)
    use module_mp_graupel
    real, intent(out) :: c(124)
    c(1) = AI
    c(2) = AC
    c(3) = AS
    c(4) = AR
    c(5) = BI
    c(6) = BC
    c(7) = BS
    c(8) = BR
    c(9) = AG
    c(10) = BG
    c(11) = R
    c(12) = RV
    c(13) = CP
    c(14) = EP_2
    c(15) = CV
    c(16) = CVV
    c(17) = CPV
    c(18) = CPL
    c(19) = CPI
    c(20) = LV1
    c(21) = LV2
    c(22) = LS1
    c(23) = LS2
    c(24) = TMELT
    c(25) = RHOSU
    c(26) = RHOW
    c(27) = RHOI
    c(28) = RHOSN
    c(29) = RHOG
    c(30) = AIMM
    c(31) = BIMM
    c(32) = ECR
    c(33) = DCS
    c(34) = MI0
    c(35) = MG0
    c(36) = F1S
    c(37) = F2S
    c(38) = F1R
    c(39) = F2R
    c(40) = G
    c(41) = QSMALL
    c(42) = QTHR
    c(43) = EII
    c(44) = ECI
    c(45) = CPW
    c(46) = CI
    c(47) = DI
    c(48) = CS
    c(49) = DS
    c(50) = CG
    c(51) = DG
    c(52) = RIN
    c(53) = MMULT
    c(54) = LAMMAXI
    c(55) = LAMMINI
    c(56) = LAMMAXR
    c(57) = LAMMINR
    c(58) = D_B
    c(59) = LAMMAXS
    c(60) = LAMMINS
    c(61) = LAMMAXG
    c(62) = LAMMING
    c(63) = K1
    c(64) = C1
    c(65) = MW
    c(66) = OSM
    c(67) = VI
    c(68) = EPSM
    c(69) = RHOA
    c(70) = MAP
    c(71) = MA
    c(72) = RR
    c(73) = BACT
    c(74) = RM1
    c(75) = SIG1
    c(76) = NANEW1
    c(77) = F11
    c(78) = F21
    c(79) = RM2
    c(80) = SIG2
    c(81) = NANEW2
    c(82) = F12
    c(83) = F22
    c(84) = CONS1
    c(85) = CONS2
    c(86) = CONS3
    c(87) = CONS4
    c(88) = CONS5
    c(89) = CONS6
    c(90) = CONS7
    c(91) = CONS8
    c(92) = CONS9
    c(93) = CONS10
    c(94) = CONS11
    c(95) = CONS12
    c(96) = CONS13
    c(97) = CONS14
    c(98) = CONS15
    c(99) = CONS16
    c(100) = CONS17
    c(101) = CONS18
    c(102) = CONS19
    c(103) = CONS20
    c(104) = CONS21
    c(105) = CONS22
    c(106) = CONS23
    c(107) = CONS24
    c(108) = CONS25
    c(109) = CONS26
    c(110) = CONS27
    c(111) = CONS28
    c(112) = CONS29
    c(113) = CONS30
    c(114) = CONS31
    c(115) = CONS32
    c(116) = CONS33
    c(117) = CONS34
    c(118) = CONS35
    c(119) = CONS36
    c(120) = CONS37
    c(121) = CONS38
    c(122) = CONS39
    c(123) = CONS40
    c(124) = CONS41
  end subroutine dump_consts
  subroutine init_scheme(ihail, ndcnst, inum)
    use module_mp_graupel, only : graupel_init, minum => inum
    integer, intent(in) :: ihail, inum
    real, intent(in) :: ndcnst
    integer :: ii
    ii = 1
    call graupel_init(ihail, ii, ndcnst)
    minum = inum
  end subroutine init_scheme
  subroutine call_gamma(n, x, gm, ef)
    use module_mp_graupel, only : gamma_ => gamma, derf1
    integer, intent(in) :: n
    real, intent(in) :: x(n)
    real, intent(out) :: gm(n), ef(n)
    integer :: i
    do i = 1, n
      gm(i) = gamma_(x(i))
      ef(i) = derf1(x(i))
    enddo
  end subroutine call_gamma
  subroutine call_satadj(n, eqtset, tsmall, t, p, qv, qc, qr, qi, qs, qg, rho)
    use module_mp_graupel, only : satadj2
    integer, intent(in) :: n, eqtset
    real, intent(in) :: tsmall
    real, intent(inout) :: t(n), p(n), qv(n), qc(n)
    real, intent(in) :: qr(n), qi(n), qs(n), qg(n), rho(n)
    integer :: i
    do i = 1, n
      call satadj2(t(i), p(i), qv(i), qc(i), qr(i), qi(i), qs(i), qg(i), rho(i), eqtset, tsmall)
    enddo
  end subroutine call_satadj
end module morr_consts

program drv_morrison
  use harness_io
  use morr_consts
  use input, only : ni, nj, nk, dx, dy, dz, ib, ie, jb, je, kb, ke, nrain, eqtset, tsmall, &
       testcase, axisymm, timestats, ibr, ier, jbr, jer, kbr, ker, ibdt, iedt, jbdt, jedt, &
       kbdt, kedt, ntdiag, ibdq, iedq, jbdq, jedq, kbdq, kedq, nqdiag, ib3d, ie3d, jb3d, &
       je3d, kb3d, ke3d, nout3d, ngxy, ngz, qd_dbz, td_cond, td_evac, td_evar, td_melt, &
       td_frz, td_dep, td_subl, qd_cond, qd_evac, qd_evar, qd_dep, qd_subl, radopt
  use module_mp_graupel, only : mp_graupel
  implicit none
  real :: params(16)
  real, allocatable :: c(:), x(:), gm(:), ef(:), t(:), p(:), qv(:), qc(:), qr(:), qi(:), &
       qs(:), qg(:), rho1(:)
  integer :: mode, n
  real :: dt
  double precision :: tcond, tevac, tevar, train
  real :: bud(4)
  real, allocatable, dimension(:,:,:) :: t3d, pii, qv3, qc3, qr3, qi3, qs3, qg3, nni, ns3, &
       nr3, ng3, nc3, p3, rho3, ddz, w3, rmh, rr, vtc, vtr, vti, vts, vtg, rainnc, &
       effc, effi, effs, effr, effg, effis
  real, allocatable :: prate(:,:), ruh(:), rvh(:)
  real, allocatable :: tdiag(:,:,:,:), qdiag(:,:,:,:), out3d(:,:,:,:)

  params = 0.0
  call rd('params', params, 1)
  mode = nint(params(1))
  select case (mode)
  case (1)
    call rd('params', params, 4)
    call init_scheme(nint(params(2)), params(3), nint(params(4)))
    allocate(c(124))
    call dump_consts(c)
    call wr('consts', c, size(c))
  case (2)
    call rd('params', params, 2)
    n = nint(params(2))
    allocate(x(n), gm(n), ef(n))
    call rd('x', x, n)
    call init_scheme(0, 0.0, 1)
    call call_gamma(n, x, gm, ef)
    call wr('gam', gm, n); call wr('erf', ef, n)
  case (3)
    call rd('params', params, 4)
    n = nint(params(2))
    allocate(t(n), p(n), qv(n), qc(n), qr(n), qi(n), qs(n), qg(n), rho1(n))
    call rd('t', t, n); call rd('p', p, n); call rd('qv', qv, n); call rd('qc', qc, n)
    call rd('qr', qr, n); call rd('qi', qi, n); call rd('qs', qs, n); call rd('qg', qg, n)
    call rd('rho', rho1, n)
    call init_scheme(0, 0.0, 1)
    call call_satadj(n, nint(params(3)), params(4), t, p, qv, qc, qr, qi, qs, qg, rho1)
    call wr('t', t, n); call wr('p', p, n); call wr('qv', qv, n); call wr('qc', qc, n)
  case (4)
    call rd('params', params, 15)
    ni = nint(params(2)); nj = nint(params(3)); nk = nint(params(4))
    dx = params(5); dy = params(6); dz = params(7); dt = params(8)
    eqtset = nint(params(12)); tsmall = params(13); testcase = nint(params(14))
    nrain = nint(params(15))
    axisymm = 0; timestats = 0; radopt = 0
    ngxy = 3; ngz = 1
    ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
    ibr = 1; ier = 1; jbr = 1; jer = 1; kbr = 1; ker = 1
    ibdt = 1; iedt = 1; jbdt = 1; jedt = 1; kbdt = 1; kedt = 1; ntdiag = 1
    ibdq = 1; iedq = 1; jbdq = 1; jedq = 1; kbdq = 1; kedq = 1; nqdiag = 1
    ib3d = 1; ie3d = 1; jb3d = 1; je3d = 1; kb3d = 1; ke3d = 1; nout3d = 1
    qd_dbz = 0; td_cond = 0; td_evac = 0; td_evar = 0; td_melt = 0; td_frz = 0; td_dep = 0
    td_subl = 0; qd_cond = 0; qd_evac = 0; qd_evar = 0; qd_dep = 0; qd_subl = 0
    call init_scheme(nint(params(9)), params(10), nint(params(11)))
    allocate(t3d(ib:ie,jb:je,kb:ke), pii(ib:ie,jb:je,kb:ke), qv3(ib:ie,jb:je,kb:ke), &
         qc3(ib:ie,jb:je,kb:ke), qr3(ib:ie,jb:je,kb:ke), qi3(ib:ie,jb:je,kb:ke), &
         qs3(ib:ie,jb:je,kb:ke), qg3(ib:ie,jb:je,kb:ke), nni(ib:ie,jb:je,kb:ke), &
         ns3(ib:ie,jb:je,kb:ke), nr3(ib:ie,jb:je,kb:ke), ng3(ib:ie,jb:je,kb:ke), &
         nc3(ib:ie,jb:je,kb:ke), p3(ib:ie,jb:je,kb:ke), rho3(ib:ie,jb:je,kb:ke), &
         ddz(ib:ie,jb:je,kb:ke), w3(ib:ie,jb:je,kb:ke+1), rmh(ib:ie,jb:je,kb:ke), &
         rr(ib:ie,jb:je,kb:ke), vtc(ib:ie,jb:je,kb:ke), vtr(ib:ie,jb:je,kb:ke), &
         vti(ib:ie,jb:je,kb:ke), vts(ib:ie,jb:je,kb:ke), vtg(ib:ie,jb:je,kb:ke), &
         rainnc(ib:ie,jb:je,nrain), prate(ib:ie,jb:je), ruh(ib:ie), rvh(jb:je))
    allocate(effc(1,1,1), effi(1,1,1), effs(1,1,1), effr(1,1,1), effg(1,1,1), effis(1,1,1), &
         tdiag(1,1,1,1), qdiag(1,1,1,1), out3d(1,1,1,1))
    pii = 1.0; vtc = 0; vtr = 0; vti = 0; vts = 0; vtg = 0
    call rd('t3d', t3d, size(t3d)); call rd('qv', qv3, size(qv3)); call rd('qc', qc3, size(qc3))
    call rd('qr', qr3, size(qr3)); call rd('qi', qi3, size(qi3)); call rd('qs', qs3, size(qs3))
    call rd('qg', qg3, size(qg3)); call rd('nni', nni, size(nni)); call rd('ns', ns3, size(ns3))
    call rd('nr', nr3, size(nr3)); call rd('ng', ng3, size(ng3)); call rd('nc', nc3, size(nc3))
    call rd('p', p3, size(p3)); call rd('rho', rho3, size(rho3)); call rd('ddz', ddz, size(ddz))
    call rd('w', w3, size(w3)); call rd('rmh', rmh, size(rmh)); call rd('rr', rr, size(rr))
    call rd('rainnc', rainnc, size(rainnc)); call rd('prate', prate, size(prate))
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('bud', bud, 4)
    tcond = bud(1); tevac = bud(2); tevar = bud(3); train = bud(4)
    call mp_graupel(1, t3d, pii, qv3, qc3, qr3, qi3, qs3, qg3, nni, ns3, nr3, ng3, nc3, &
         p3, rho3, dt, ddz, w3, rainnc, prate, effc, effi, effs, effr, effg, effis, &
         tcond, tevac, tevar, train, ruh, rvh, rmh, rr, .false., &
         vtc, vtr, vti, vts, vtg, .true., .false., .false., .false., tdiag, qdiag, out3d)
    call wr('t3d', t3d, size(t3d)); call wr('qv', qv3, size(qv3)); call wr('qc', qc3, size(qc3))
    call wr('qr', qr3, size(qr3)); call wr('qi', qi3, size(qi3)); call wr('qs', qs3, size(qs3))
    call wr('qg', qg3, size(qg3)); call wr('nni', nni, size(nni)); call wr('ns', ns3, size(ns3))
    call wr('nr', nr3, size(nr3)); call wr('ng', ng3, size(ng3)); call wr('nc', nc3, size(nc3))
    call wr('rainnc', rainnc, size(rainnc)); call wr('prate', prate, size(prate))
    call wr('vtc', vtc, size(vtc)); call wr('vtr', vtr, size(vtr)); call wr('vti', vti, size(vti))
    call wr('vts', vts, size(vts)); call wr('vtg', vtg, size(vtg))
    bud(1) = tcond; bud(2) = tevac; bud(3) = tevar; bud(4) = train
    call wr('bud', bud, 4)
  end select
end program drv_morrison
