! Harness driver for the terrain (terrain_flag=.true.) branches of the LES
! subgrid turbulence: turb.F calcdef and turbtend.F turbs, turbt, turbu,
! turbv, turbw.  The metrics (rds, sigma, rdsf, sigmaf, gz, rgz, gzu, rgzu,
! gzv, rgzv, gx, gxu, gy, gyv) are inputs.  Same layout as drv_turb.F90.
!
! params(40) = drv_turb's 33 + [zt, cgs1, cgs2, cgs3, cgt1, cgt2, cgt3]
! modes: 2 calcdef 6 turbt 7 turbu 8 turbv 9 turbw 10 turbs
#define turb_module cm1x_turbcopy_module
#define private public
#include "../../../src/turb.F"
#undef private
#undef turb_module

program drv_turb_terrain
  use harness_io
  use input
  use constants, only : set_constants, c_m, c_l, ri_c, cmemin, pi, cv, cvv, cpl, cpi, cpdcv, cp
  use ib_module, only : ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  use bc_module, only : bcw, bc2d
  use turbtend_module, only : turbs, turbt, turbu, turbv, turbw
  use cm1x_turbcopy_module, only : calcnm, calcdef, tkekm, tkebc, gettau
  implicit none
  real :: params(40), dt, rtime, dttmp, turbfac
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
  real, allocatable, dimension(:) :: rds, sigma, rdsf, sigmaf
  real, allocatable, dimension(:,:) :: gz, rgz, gzu, rgzu, gzv, rgzv
  real, allocatable, dimension(:,:,:) :: gx, gxu, gy, gyv

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
  zt = params(34); rzt = 1.0/zt
  cgs1 = params(35); cgs2 = params(36); cgs3 = params(37)
  cgt1 = params(38); cgt2 = params(39); cgt3 = params(40)

  nx = ni; ny = nj; nz = nk; nkp1 = nk + 1
  ngxy = 3; ngz = 1
  ib = 1 - ngxy; ie = ni + ngxy; jb = 1 - ngxy; je = nj + ngxy; kb = 1 - ngz; ke = nk + ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  ibt = ib; iet = ie; jbt = jb; jet = je; kbt = kb; ket = ke + 1
  ibc = ib; iec = ie; jbc = jb; jec = je; kbc = kb; kec = ke + 1
  itb = ib; ite = ie; jtb = jb; jte = je; ktb = kb; kte = ke
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
  terrain_flag = .true.; axisymm = 0; do_ib = .false.
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
  allocate(rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1))
  allocate(gz(itb:ite,jtb:jte), rgz(itb:ite,jtb:jte), gzu(itb:ite,jtb:jte), rgzu(itb:ite,jtb:jte), &
           gzv(itb:ite,jtb:jte), rgzv(itb:ite,jtb:jte))
  allocate(gx(itb:ite,jtb:jte,ktb:kte), gxu(itb:ite,jtb:jte,ktb:kte), gy(itb:ite,jtb:jte,ktb:kte), &
           gyv(itb:ite,jtb:jte,ktb:kte))
  call rd('rds', rds, size(rds)); call rd('sigma', sigma, size(sigma))
  call rd('rdsf', rdsf, size(rdsf)); call rd('sigmaf', sigmaf, size(sigmaf))
  call rd('gz', gz, size(gz)); call rd('rgz', rgz, size(rgz))
  call rd('gzu', gzu, size(gzu)); call rd('rgzu', rgzu, size(rgzu))
  call rd('gzv', gzv, size(gzv)); call rd('rgzv', rgzv, size(rgzv))
  call rd('gx', gx, size(gx)); call rd('gxu', gxu, size(gxu))
  call rd('gy', gy, size(gy)); call rd('gyv', gyv, size(gyv))

  select case (mode)
  case (2)
    call rdgrid()
    call rd('c1', c1, size(c1)); call rd('c2', c2, size(c2))
    call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('ua', ua, size(ua)); call rd('va', va, size(va)); call rd('wa', wa, size(wa))
    call rdtau(); call rd('defv', defv, size(defv)); call rd('defh', defh, size(defh))
    call calcdef(rds, sigma, rdsf, sigmaf, zero2, gz, rgz, gzu, rgzu, gzv, rgzv, &
                 one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, one_f, uf, vh, vf, mh, c1, c2, mf, &
                 defv, defh, s1, s2, ua, va, wa, t11, t12, t13, t22, t23, t33, gx, gy, rho, rr, rf)
    call wrtau(); call wr('defv', defv, size(defv)); call wr('defh', defh, size(defh))
  case (6)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('t', tkea, size(tkea)); call rd('tten', tketen, size(tketen)); call rdk()
    call turbt(dt, one_h, one_h, uh, one_f, uf, vh, vf, mh, mf, rho, rr, rf, &
               rds, sigma, gz, rgz, gzu, rgzu, gzv, rgzv, &
               s1, s2, s3, s4, s5, s6, tkea, tketen, kmh, kmv)
    call wr('tten', tketen, size(tketen))
  case (7)
    call rdgrid(); call rd('ruh', ruh, size(ruh)); call rd('rmf', rmf, size(rmf))
    call rd('rho', rho, size(rho)); call rd('rf', rf, size(rf))
    call rd('u', ua, size(ua)); call rd('uten', uten, size(uten)); call rd('w', wa, size(wa))
    call rdtau(); call rdk()
    call turbu(dt, one_h, ruh, one_f, one_f, one_f, one_f, uf, vh, mh, mf, rmf, rho, rf, &
               zero2, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gxu, &
               s1, s2, s3, s4, s5, s6, s7, s8, s9, ua, uten, wa, t11, t12, t13, t22, kmv, cm0, &
               kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
    call wr('uten', uten, size(uten))
  case (8)
    call rdgrid(); call rd('rvh', rvh, size(rvh))
    call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('v', va, size(va)); call rd('vten', vten, size(vten)); call rd('w', wa, size(wa))
    call rdtau(); call rdk()
    call turbv(dt, one_h, one_h, one_h, one_h, uh, one_f, rvh, vf, mh, mf, rho, rr, rf, &
               zero2, gz, rgz, gzu, gzv, rds, sigma, rdsf, sigmaf, gyv, &
               s1, s2, s3, s4, s5, s6, s7, s8, s9, va, vten, wa, t12, t22, t23, kmv, cm0, &
               kprof, kprof, kprof, dum2pt, dum2pt, .false., diag)
    call wr('vten', vten, size(vten))
  case (9)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rf', rf, size(rf))
    call rd('w', wa, size(wa)); call rd('wten', wten, size(wten)); call rdtau(); call rdk()
    call turbw(dt, one_h, one_h, one_h, one_h, uh, one_f, vh, mh, mf, rho, rf, gz, rgzu, rgzv, rds, sigma, &
               s1, s2, s3, s4, s5, s6, wa, wten, t13, t23, t33, t22, kmh, cm0, .false., diag)
    call wr('wten', wten, size(wten))
  case (10)
    call rdgrid(); call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr)); call rd('rf', rf, size(rf))
    call rd('s', s, size(s)); call rd('sten', sten, size(sten)); call rdk()
    call rd('dum7', dum7, size(dum7)); call rd('dum8', dum8, size(dum8))
    call turbs(iflux, dt, .false., one_h, one_h, one_h, one_h, uh, one_f, one_f, one_f, uf, vh, vf, zero2, &
               rds, sigma, rdsf, sigmaf, mh, mf, gz, rgz, gzu, rgzu, gzv, rgzv, &
               gx, gxu, gy, gyv, &
               s1, s2, s3, s4, s5, s6, rho, rr, rf, s, sten, khh, khv, cm0, dum7, dum8, &
               .false., 1, 1, 1, 1, 1, 1, 1, diag, 1, 1, 1)
    call wr('sten', sten, size(sten))
  case default
    stop 'drv_turb_terrain: bad mode'
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

  subroutine rdtau()
    call rd('t11', t11, size(t11)); call rd('t12', t12, size(t12)); call rd('t13', t13, size(t13))
    call rd('t22', t22, size(t22)); call rd('t23', t23, size(t23)); call rd('t33', t33, size(t33))
  end subroutine rdtau

  subroutine wrtau()
    call wr('t11', t11, size(t11)); call wr('t12', t12, size(t12)); call wr('t13', t13, size(t13))
    call wr('t22', t22, size(t22)); call wr('t23', t23, size(t23)); call wr('t33', t33, size(t33))
  end subroutine wrtau

end program drv_turb_terrain
