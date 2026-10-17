! Harness driver for the LES subgrid turbulence (cm1setup=1, sgsmodel=1):
! turb.F (calcnm, calcdef, tkekm, tkebc, gettau/sfcstress), turbtend.F
! (turbs, turbt, turbu, turbv, turbw), bc.F (bcw, bc2d) and the inline
! solve1/solve2 code (TKE source, dissipative heating, vimpl arrays, the
! ramp_sgs block of sfc_and_turb, the solve2 TKE step) copied verbatim.
!
! tkekm and gettau are private in turb_module, so turb.F is compiled a
! second time below under another module name with private -> public.
!
! params(33) = [mode, ni, nj, nk, dx, dy, dz, dt, tconfig, bcturbs, doimpl,
!   wbc, ebc, sbc, nbc, imoist, ptype, iice, numq, nqv, nqc, nqi, nql1, nql2,
!   nqs1, nqs2, eqtset, idiss, iflux, iarg (bcw flag / bc2d level),
!   rtime, ramp_time, dttmp]
! modes: 1 calcnm 2 calcdef 3 tkekm 4 tkebc 5 gettau 6 turbt 7 turbu
!   8 turbv 9 turbw 10 turbs 11 tke_source 12 dissheat 13 vimpl 14 ramp
!   15 bcw 16 bc2d(level) 17 tke_update 18 sfc_and_turb LES chain
#define turb_module cm1x_turbcopy_module
#define private public
#include "../../../src/turb.F"
#undef private
#undef turb_module

program drv_turb
  use harness_io
  use input
  use constants, only : set_constants, c_m, c_l, ri_c, cmemin, pi, cv, cvv, cpl, cpi, cpdcv, cp
  use ib_module, only : ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  use bc_module, only : bcw, bc2d
  use turbtend_module, only : turbs, turbt, turbu, turbv, turbw
  use cm1x_turbcopy_module, only : calcnm, calcdef, tkekm, tkebc, gettau
  implicit none
  real :: params(33), dt, rtime, dttmp, turbfac
  integer :: mode, iflux, iarg, i, j, k, n, k1
  real, allocatable, dimension(:) :: one_h, one_f, rds_, rdsf_, kprof, c1d
  real, allocatable, dimension(:,:) :: one_t, zero2, cm0, cj, ci, stau
  real, allocatable, dimension(:,:,:) :: one_t3, s1, s2, s3, s4, s5, s6, s7, s8, s9, dum2pt
  real, allocatable, dimension(:,:,:,:) :: diag, out3d, qa
  logical, allocatable :: bndy(:,:,:), cloudvar(:), iamsat(:,:,:)
  integer, allocatable :: kbdy(:,:)
  double precision :: avg
  real, allocatable :: uh(:), ruh(:), uf(:), vh(:), rvh(:), vf(:)
  real, allocatable, dimension(:,:,:) :: c1, c2, pi0, th0, prs, pp, th, mh, rmh, rho, rr, rf, &
       t11, t12, t13, t22, t23, t33, s, sten, dum7, dum8, thten1, dum1, dum2, dum3
  real, allocatable, dimension(:,:,:) :: mf, rmf, zf, nm, defv, defh, lenscl, dissten, wa, wten, &
       kmh, kmv, khh, khv, cme, csm, ce1, ce2, tkea, tke3d, tketen
  real, allocatable :: ua(:,:,:), uten(:,:,:), va(:,:,:), vten(:,:,:)
  real :: epsd, dheat

  call rd('params', params, size(params))
  mode = nint(params(1))
  ni = nint(params(2)); nj = nint(params(3)); nk = nint(params(4))
  dx = params(5); dy = params(6); dz = params(7); dt = params(8)
  tconfig = nint(params(9)); bcturbs = nint(params(10)); doimpl = nint(params(11))
  wbc = nint(params(12)); ebc = nint(params(13)); sbc = nint(params(14)); nbc = nint(params(15))
  imoist = nint(params(16)); ptype = nint(params(17)); iice = nint(params(18))
  numq = nint(params(19)); nqv = nint(params(20)); nqc = nint(params(21)); nqi = nint(params(22))
  nql1 = nint(params(23)); nql2 = nint(params(24)); nqs1 = nint(params(25)); nqs2 = nint(params(26))
  eqtset = nint(params(27)); idiss = nint(params(28)); iflux = nint(params(29)); iarg = nint(params(30))
  rtime = params(31); ramp_time = params(32); dttmp = params(33)

  nx = ni; ny = nj; nz = nk; nkp1 = nk + 1
  ngxy = 3; ngz = 1
  ib = 1 - ngxy; ie = ni + ngxy; jb = 1 - ngxy; je = nj + ngxy; kb = 1 - ngz; ke = nk + ngz
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
  idoles = .true.; iusetke = .true.; dohturb = .true.; dovturb = .true.; dot2p = .false.
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc .ne. 1) ibw = 1
  if (ebc .ne. 1) ibe = 1
  if (sbc .ne. 1) ibs = 1
  if (nbc .ne. 1) ibn = 1
  bbc = 1; tbc = 1
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  terrain_flag = .false.; axisymm = 0; do_ib = .false.
  timestats = 0; myid = 0; smeps = 1.0e-60
  call set_constants(0)

  allocate(one_h(ib:ie), one_f(ib:ie+1), rds_(kb:ke), rdsf_(kb:ke+1), kprof(kb:ke), c1d(kmt))
  allocate(one_t(itb:ite,jtb:jte), zero2(ib:ie,jb:je), cm0(ib:ie,jb:je), cj(jmp,kmt), ci(imp,kmt), &
           stau(ib:ie,jb:je), one_t3(itb:ite,jtb:jte,ktb:kte))
  allocate(s1(ib:ie,jb:je,kb:ke), s2(ib:ie,jb:je,kb:ke), s3(ib:ie,jb:je,kb:ke), s4(ib:ie,jb:je,kb:ke), &
           s5(ib:ie,jb:je,kb:ke), s6(ib:ie,jb:je,kb:ke), s7(ib:ie,jb:je,kb:ke), s8(ib:ie,jb:je,kb:ke), &
           s9(ib:ie,jb:je,kb:ke), dum2pt(ib2pt:ie2pt,jb2pt:je2pt,kb2pt:ke2pt))
  allocate(diag(1,1,1,1), out3d(ib3d:ie3d,jb3d:je3d,kb3d:ke3d,nout3d))
  allocate(bndy(ibib:ieib,jbib:jeib,kbib:keib), cloudvar(maxq), kbdy(ibib:ieib,jbib:jeib), &
           iamsat(ib:ie,jb:je,kb:ke))
  one_h = 1; one_f = 1; rds_ = rdz; rdsf_ = rdz; kprof = 0; c1d = 0
  one_t = 1; zero2 = 0; cm0 = c_m; cj = 0; ci = 0; stau = 0; one_t3 = 0
  s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0; s8 = 0; s9 = 0; dum2pt = 0
  diag = 0; out3d = 0; bndy = .false.; cloudvar = .false.; kbdy = 1; iamsat = .false.; avg = 0

  allocate(uh(ib:ie), ruh(ib:ie), uf(ib:ie+1), vh(jb:je), rvh(jb:je), vf(jb:je+1))
  allocate(c1(ib:ie,jb:je,kb:ke), c2(ib:ie,jb:je,kb:ke), pi0(ib:ie,jb:je,kb:ke), th0(ib:ie,jb:je,kb:ke), &
           prs(ib:ie,jb:je,kb:ke), pp(ib:ie,jb:je,kb:ke), th(ib:ie,jb:je,kb:ke), mh(ib:ie,jb:je,kb:ke), &
           rmh(ib:ie,jb:je,kb:ke), rho(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke), rf(ib:ie,jb:je,kb:ke), &
           t11(ib:ie,jb:je,kb:ke), t12(ib:ie,jb:je,kb:ke), t13(ib:ie,jb:je,kb:ke), t22(ib:ie,jb:je,kb:ke), &
           t23(ib:ie,jb:je,kb:ke), t33(ib:ie,jb:je,kb:ke), s(ib:ie,jb:je,kb:ke), sten(ib:ie,jb:je,kb:ke), &
           dum7(ib:ie,jb:je,kb:ke), dum8(ib:ie,jb:je,kb:ke), thten1(ib:ie,jb:je,kb:ke), &
           dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke))
  allocate(mf(ib:ie,jb:je,kb:ke+1), rmf(ib:ie,jb:je,kb:ke+1), zf(ib:ie,jb:je,kb:ke+1), &
           nm(ib:ie,jb:je,kb:ke+1), defv(ib:ie,jb:je,kb:ke+1), defh(ib:ie,jb:je,kb:ke+1), &
           lenscl(ib:ie,jb:je,kb:ke+1), dissten(ib:ie,jb:je,kb:ke+1), wa(ib:ie,jb:je,kb:ke+1), &
           wten(ib:ie,jb:je,kb:ke+1), kmh(ibc:iec,jbc:jec,kbc:kec), kmv(ibc:iec,jbc:jec,kbc:kec), &
           khh(ibc:iec,jbc:jec,kbc:kec), khv(ibc:iec,jbc:jec,kbc:kec), cme(ibc:iec,jbc:jec,kbc:kec), &
           csm(ibc:iec,jbc:jec,kbc:kec), ce1(ibc:iec,jbc:jec,kbc:kec), ce2(ibc:iec,jbc:jec,kbc:kec), &
           tkea(ibt:iet,jbt:jet,kbt:ket), tke3d(ibt:iet,jbt:jet,kbt:ket), tketen(ibt:iet,jbt:jet,kbt:ket))
  allocate(ua(ib:ie+1,jb:je,kb:ke), uten(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke), &
           vten(ib:ie,jb:je+1,kb:ke), qa(ibm:iem,jbm:jem,kbm:kem,max(1,numq)))
  zf = 0; rmh = 1; dum1 = 0; dum2 = 0; dum3 = 0

  select case (mode)
  case (1)
    call rd('c1', c1, size(c1)); call rd('c2', c2, size(c2)); call rd('mf', mf, size(mf))
    call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0)); call rd('prs', prs, size(prs))
    call rd('pp', pp, size(pp)); call rd('th', th, size(th)); call rd('qa', qa, size(qa))
    call rd('nm', nm, size(nm))
    call calcnm(c1, c2, mf, pi0, s9, th0, cloudvar, nm, s1, s2, s3, s4, s5, s6, prs, pp, th, qa, iamsat)
    call wr('nm', nm, size(nm))
  case (2)
    call rdgrid()
    call rd('c1', c1, size(c1)); call rd('c2', c2, size(c2))
    call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('ua', ua, size(ua)); call rd('va', va, size(va)); call rd('wa', wa, size(wa))
    call rdtau(); call rd('defv', defv, size(defv)); call rd('defh', defh, size(defh))
    call calcdef(rds_, rds_, rdsf_, rdsf_, zero2, one_t, one_t, one_t, one_t, one_t, one_t, &
                 one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, one_f, uf, vh, vf, mh, c1, c2, mf, &
                 defv, defh, s1, s2, ua, va, wa, t11, t12, t13, t22, t23, t33, one_t3, one_t3, rho, rr, rf)
    call wrtau(); call wr('defv', defv, size(defv)); call wr('defh', defh, size(defh))
  case (3)
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('mf', mf, size(mf)); call rd('rmf', rmf, size(rmf)); call rd('rf', rf, size(rf))
    call rd('nm', nm, size(nm)); call rd('defv', defv, size(defv)); call rd('defh', defh, size(defh))
    call rdk(); call rdconst(); call rd('tkea', tkea, size(tkea))
    call rd('lenscl', lenscl, size(lenscl)); call rd('dissten', dissten, size(dissten))
    call dotkekm()
    call wrk(); call wr('tkea', tkea, size(tkea))
    call wr('lenscl', lenscl, size(lenscl)); call wr('dissten', dissten, size(dissten))
  case (4)
    call rd('tkea', tkea, size(tkea)); call rdk()
    call dotkebc()
    call wr('tkea', tkea, size(tkea)); call wrk()
  case (5)
    call rd('rf', rf, size(rf)); call rd('mf', mf, size(mf)); call rdk()
    call rd('ua', ua, size(ua)); call rd('va', va, size(va)); call rd('wa', wa, size(wa))
    call rdtau()
    call dogettau()
    call wrtau()
  case (6)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('t', tkea, size(tkea)); call rd('tten', tketen, size(tketen)); call rdk()
    call turbt(dt, one_h, one_h, uh, one_f, uf, vh, vf, mh, mf, rho, rr, rf, &
               rds_, rds_, one_t, one_t, one_t, one_t, one_t, one_t, &
               s1, s2, s3, s4, s5, s6, tkea, tketen, kmh, kmv)
    call wr('tten', tketen, size(tketen))
  case (7)
    call rdgrid(); call rd('ruh', ruh, size(ruh)); call rd('rmf', rmf, size(rmf))
    call rd('rho', rho, size(rho)); call rd('rf', rf, size(rf))
    call rd('u', ua, size(ua)); call rd('uten', uten, size(uten)); call rd('w', wa, size(wa))
    call rdtau(); call rdk()
    call turbu(dt, one_h, ruh, one_f, one_f, one_f, one_f, uf, vh, mh, mf, rmf, rho, rf, &
               zero2, one_t, one_t, one_t, one_t, rds_, rds_, rdsf_, rdsf_, one_t3, &
               s1, s2, s3, s4, s5, s6, s7, s8, s9, ua, uten, wa, t11, t12, t13, t22, kmv, cm0, &
               kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
    call wr('uten', uten, size(uten))
  case (8)
    call rdgrid(); call rd('rvh', rvh, size(rvh))
    call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('v', va, size(va)); call rd('vten', vten, size(vten)); call rd('w', wa, size(wa))
    call rdtau(); call rdk()
    call turbv(dt, one_h, one_h, one_h, one_h, uh, one_f, rvh, vf, mh, mf, rho, rr, rf, &
               zero2, one_t, one_t, one_t, one_t, rds_, rds_, rdsf_, rdsf_, one_t3, &
               s1, s2, s3, s4, s5, s6, s7, s8, s9, va, vten, wa, t12, t22, t23, kmv, cm0, &
               kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
    call wr('vten', vten, size(vten))
  case (9)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rf', rf, size(rf))
    call rd('w', wa, size(wa)); call rd('wten', wten, size(wten)); call rdtau(); call rdk()
    call turbw(dt, one_h, one_h, one_h, one_h, uh, one_f, vh, mh, mf, rho, rf, one_t, one_t, one_t, rds_, rds_, &
               s1, s2, s3, s4, s5, s6, wa, wten, t13, t23, t33, t22, kmh, cm0, .false., diag)
    call wr('wten', wten, size(wten))
  case (10)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('s', s, size(s)); call rd('sten', sten, size(sten)); call rdk()
    call rd('dum7', dum7, size(dum7)); call rd('dum8', dum8, size(dum8))
    call turbs(iflux, dt, .false., one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, uf, vh, vf, zero2, &
               rds_, rds_, rdsf_, rdsf_, mh, mf, one_t, one_t, one_t, one_t, one_t, one_t, &
               one_t3, one_t3, one_t3, one_t3, &
               s1, s2, s3, s4, s5, s6, rho, rr, rf, s, sten, khh, khv, cm0, dum7, dum8, &
               .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1)
    call wr('sten', sten, size(sten))
  case (11)
    call rd('nm', nm, size(nm)); call rd('defv', defv, size(defv)); call rd('defh', defh, size(defh))
    call rd('dissten', dissten, size(dissten)); call rdk(); call rd('tketen', tketen, size(tketen))
    call tke_source()
    call wr('tketen', tketen, size(tketen))
  case (12)
    call rd('pi0', pi0, size(pi0)); call rd('ppi', pp, size(pp)); call rd('qa', qa, size(qa))
    call rd('dissten', dissten, size(dissten)); call rd('thten1', thten1, size(thten1))
    call dissheat()
    call wr('thten1', thten1, size(thten1))
  case (13)
    call rdk(); call rd('mf', mf, size(mf)); call rd('rf', rf, size(rf))
    call rd('mh', mh, size(mh)); call rd('rr', rr, size(rr))
    call rd('dum7', dum7, size(dum7)); call rd('dum8', dum8, size(dum8))
    call vimpl()
    call wr('dum7', dum7, size(dum7)); call wr('dum8', dum8, size(dum8))
  case (14)
    call rd('cm0', cm0, size(cm0)); call rdconst()
    call ramp()
    call wrconst()
  case (15)
    call rd('w', wa, size(wa))
    call bcw(wa, iarg)
    call wr('w', wa, size(wa))
  case (16)
    call rd('w', wa, size(wa))
    call bc2d(wa(ib,jb,iarg))
    call wr('w', wa, size(wa))
  case (17)
    call rd('tkea', tkea, size(tkea)); call rd('tten', wten, size(wten)); call rd('tke3d', tke3d, size(tke3d))
    do k=2,nk
      do j=1,nj
      do i=1,ni
        tke3d(i,j,k)=tkea(i,j,k)+dttmp*wten(i,j,k)
        if(tke3d(i,j,k).lt.1.0e-10) tke3d(i,j,k)=0.0
      enddo
      enddo
    enddo
    call bcw(tke3d,1)
    call wr('tke3d', tke3d, size(tke3d))
  case (18)
    ! sfc_and_turb, LES path (sgsmodel=1, no surface layer, no PBL), in CM1's order
    call rdgrid(); call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('c1', c1, size(c1)); call rd('c2', c2, size(c2)); call rd('rmf', rmf, size(rmf))
    call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0)); call rd('rho', rho, size(rho))
    call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf)); call rd('prs', prs, size(prs))
    call rd('ua', ua, size(ua)); call rd('va', va, size(va)); call rd('wa', wa, size(wa))
    call rd('ppi', pp, size(pp)); call rd('tha', th, size(th)); call rd('qa', qa, size(qa))
    call rd('tkea', tkea, size(tkea)); call rd('tke3d', tke3d, size(tke3d))
    call rd('cm0', cm0, size(cm0)); call rdconst(); call rdk(); call rdtau()
    call rd('nm', nm, size(nm)); call rd('defv', defv, size(defv)); call rd('defh', defh, size(defh))
    call rd('lenscl', lenscl, size(lenscl)); call rd('dissten', dissten, size(dissten))
    if (ramp_time .gt. 0.0) call ramp()
    do k=1,nk+1
    do j=1,nj
    do i=1,ni
      dissten(i,j,k)=0.0
    enddo
    enddo
    enddo
    iamsat = .false.
    call calcnm(c1, c2, mf, pi0, s9, th0, cloudvar, nm, s1, s2, s3, s4, s5, s6, prs, pp, th, qa, iamsat)
    call calcdef(rds_, rds_, rdsf_, rdsf_, zero2, one_t, one_t, one_t, one_t, one_t, one_t, &
                 one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, one_f, uf, vh, vf, mh, c1, c2, mf, &
                 defv, defh, s1, s2, ua, va, wa, t11, t12, t13, t22, t23, t33, one_t3, one_t3, rho, rr, rf)
    call dotkekm()
    call dotkebc()
    do j=0,nj+1
    do i=0,ni+1
      tke3d(i,j,1) = tkea(i,j,1)
      tke3d(i,j,nk+1) = tkea(i,j,nk+1)
    enddo
    enddo
    call dogettau()
    call wr('tkea', tkea, size(tkea)); call wr('tke3d', tke3d, size(tke3d)); call wrk(); call wrconst()
    call wr('nm', nm, size(nm)); call wr('defv', defv, size(defv)); call wr('defh', defh, size(defh))
    call wr('lenscl', lenscl, size(lenscl)); call wr('dissten', dissten, size(dissten)); call wrtau()
  case default
    stop 'drv_turb: bad mode'
  end select

contains

  subroutine rdgrid()
    call rd('uh', uh, size(uh)); call rd('uf', uf, size(uf))
    call rd('vh', vh, size(vh)); call rd('vf', vf, size(vf))
    call rd('mh', mh, size(mh)); call rd('mf', mf, size(mf))
  end subroutine rdgrid

  subroutine rdk()
    call rd('kmh', kmh, size(kmh)); call rd('kmv', kmv, size(kmv))
    call rd('khh', khh, size(khh)); call rd('khv', khv, size(khv))
  end subroutine rdk

  subroutine wrk()
    call wr('kmh', kmh, size(kmh)); call wr('kmv', kmv, size(kmv))
    call wr('khh', khh, size(khh)); call wr('khv', khv, size(khv))
  end subroutine wrk

  subroutine rdconst()
    call rd('cme', cme, size(cme)); call rd('csm', csm, size(csm))
    call rd('ce1', ce1, size(ce1)); call rd('ce2', ce2, size(ce2))
  end subroutine rdconst

  subroutine wrconst()
    call wr('cme', cme, size(cme)); call wr('csm', csm, size(csm))
    call wr('ce1', ce1, size(ce1)); call wr('ce2', ce2, size(ce2))
  end subroutine wrconst

  subroutine rdtau()
    call rd('t11', t11, size(t11)); call rd('t12', t12, size(t12)); call rd('t13', t13, size(t13))
    call rd('t22', t22, size(t22)); call rd('t23', t23, size(t23)); call rd('t33', t33, size(t33))
  end subroutine rdtau

  subroutine wrtau()
    call wr('t11', t11, size(t11)); call wr('t12', t12, size(t12)); call wr('t13', t13, size(t13))
    call wr('t22', t22, size(t22)); call wr('t23', t23, size(t23)); call wr('t33', t33, size(t33))
  end subroutine wrtau

  subroutine dotkekm()
    call tkekm(1, 0.0, 1.0, ruh, rvh, rmh, zf, mf, rmf, zero2, zero2, zero2, rf, &
               nm, defv, defh, s1, s2, s3, s4, s5, &
               kmh, kmv, khh, khv, cme, csm, ce1, ce2, tkea, lenscl, dissten, out3d, &
               c1d, c1d, c1d, c1d, c1d, c1d, c1d, c1d, &
               cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci, &
               cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
  end subroutine dotkekm

  subroutine dotkebc()
    call tkebc(tkea, kmh, kmv, khh, khv, zero2, zero2, zero2, &
               cj, cj, cj, cj, ci, ci, ci, ci, cj, cj, cj, cj, ci, ci, ci, ci)
  end subroutine dotkebc

  subroutine dogettau()
    call gettau(one_f, one_f, one_f, one_f, zero2, stau, zero2, zero2, zero2, zero2, zero2, zero2, zero2, &
                rf, mf, s1, s2, s3, s4, s5, s6, kmh, kmv, t11, t12, t13, t22, t23, t33, &
                ua, ua, va, va, wa, avg, avg, avg, avg, avg, bndy, kbdy)
  end subroutine dogettau

  ! turb.F sfc_and_turb, "facheck" block (verbatim)
  subroutine ramp()
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
            csm(i,j,k) = ( cme(i,j,k) * cme(i,j,k) * cme(i,j,k) / ( ce1(i,j,k) + ce2(i,j,k) ) )**0.25
          endif
        enddo
        enddo
        enddo
  end subroutine ramp

  ! solve1.F TKE source (verbatim)
  subroutine tke_source()
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
  end subroutine tke_source

  ! solve1.F cvm (dum1) and dissipative heating (verbatim, rterm=0, bbc/=3)
  subroutine dissheat()
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
            dheat=epsd/( cpdcv*dum1(i,j,k)*(pi0(i,j,k)+pp(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
        else
          do k=k1,nk
          do j=1,nj
          do i=1,ni
            epsd = 0.5*(dissten(i,j,k)+dissten(i,j,k+1))
            dheat=epsd/( cp*(pi0(i,j,k)+pp(i,j,k)) )
            thten1(i,j,k)=thten1(i,j,k)+dheat
          enddo
          enddo
          enddo
        endif
      ENDIF
  end subroutine dissheat

  ! solve1.F vimpl arrays (verbatim)
  subroutine vimpl()
          do k=1,nk
          do j=1,nj
          do i=1,ni
            dum7(i,j,k) = khv(i,j,k  )*mf(i,j,k  )*rf(i,j,k  )*mh(i,j,k)*rr(i,j,k)
            dum8(i,j,k) = khv(i,j,k+1)*mf(i,j,k+1)*rf(i,j,k+1)*mh(i,j,k)*rr(i,j,k)
          enddo
          enddo
          enddo
  end subroutine vimpl

end program drv_turb
