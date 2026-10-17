! Harness driver for src/kessler.F routines (kessler, satadj, k_fallout,
! fallout, geterain, getefall).  params = [mode, ni, nj, nk, dx, dy, dz, dt,
!   eqtset, tsmall, nrain, nrk, setzero, cpx, lx1]
!   mode: 1 kessler, 2 satadj, 3 k_fallout, 4 fallout, 5 geterain, 6 getefall
program drv_kessler
  use harness_io
  use input
  use constants, only : set_constants
  use kessler_module
  implicit none
  real :: params(15), b1(1), b2(1), b3(1), dt, cpx, lx1
  double precision :: d1, d2, d3
  integer :: mode, nrk, setzero
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:), pi0(:,:,:), th0(:,:,:), tmp(:,:,:), &
       rho(:,:,:), rr(:,:,:), pp3d(:,:,:), th3d(:,:,:), prs(:,:,:), q3d(:,:,:,:), &
       qr(:,:,:), vr(:,:,:), zh(:,:,:), mh(:,:,:), mf(:,:,:), rain(:,:,:), prate(:,:), &
       t(:,:,:), cvm(:,:,:), tten(:,:,:)

  call rd('params', params, size(params))
  mode = nint(params(1))
  ni = nint(params(2)); nj = nint(params(3)); nk = nint(params(4))
  dx = params(5); dy = params(6); dz = params(7); dt = params(8)
  eqtset = nint(params(9)); tsmall = params(10); nrain = nint(params(11))
  nrk = nint(params(12)); setzero = nint(params(13)); cpx = params(14); lx1 = params(15)
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  numq = 3; nqv = 1; nqc = 2; nqr = 3; nql1 = 2; nql2 = 3; nqs1 = -2; nqs2 = -2
  iice = 0; timestats = 0; myid = 0
  call set_constants(0)

  allocate(ruh(ib:ie), rvh(jb:je))
  allocate(rmh(ib:ie,jb:je,kb:ke), pi0(ib:ie,jb:je,kb:ke), th0(ib:ie,jb:je,kb:ke), &
       tmp(ib:ie,jb:je,kb:ke), rho(ib:ie,jb:je,kb:ke), rr(ib:ie,jb:je,kb:ke), &
       pp3d(ib:ie,jb:je,kb:ke), th3d(ib:ie,jb:je,kb:ke), prs(ib:ie,jb:je,kb:ke), &
       qr(ib:ie,jb:je,kb:ke), vr(ib:ie,jb:je,kb:ke), zh(ib:ie,jb:je,kb:ke), &
       mh(ib:ie,jb:je,kb:ke), mf(ib:ie,jb:je,kb:ke+1), t(ib:ie,jb:je,kb:ke), &
       cvm(ib:ie,jb:je,kb:ke), tten(ib:ie,jb:je,kb:ke))
  allocate(q3d(ibm:iem,jbm:jem,kbm:kem,numq), rain(ib:ie,jb:je,nrain), prate(ib:ie,jb:je))

  select case (mode)
  case (1)
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('rmh', rmh, size(rmh)); call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0))
    call rd('tmp', tmp, size(tmp)); call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr))
    call rd('pp3d', pp3d, size(pp3d)); call rd('th3d', th3d, size(th3d)); call rd('prs', prs, size(prs))
    call rd('q3d', q3d, size(q3d))
    call rd('bud', b1, 1); d1 = b1(1); d2 = b1(1)*2; d3 = b1(1)*3
    call kessler(dt, d1, d2, d3, ruh, rvh, rmh, pi0, th0, tmp, rho, rr, pp3d, th3d, prs, &
         q3d(ib,jb,kb,1), q3d(ib,jb,kb,2), q3d(ib,jb,kb,3))
    call wr('tmp', tmp, size(tmp)); call wr('rho', rho, size(rho)); call wr('pp3d', pp3d, size(pp3d))
    call wr('th3d', th3d, size(th3d)); call wr('prs', prs, size(prs)); call wr('q3d', q3d, size(q3d))
    b1(1) = d1; b2(1) = d2; b3(1) = d3
    call wr('tauto', b1, 1); call wr('taccr', b2, 1); call wr('tevar', b3, 1)
  case (2)
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('rmh', rmh, size(rmh)); call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0))
    call rd('rho', rho, size(rho)); call rd('rr', rr, size(rr))
    call rd('pp3d', pp3d, size(pp3d)); call rd('prs', prs, size(prs)); call rd('th3d', th3d, size(th3d))
    call rd('q3d', q3d, size(q3d))
    call rd('bud', b1, 1); d1 = b1(1); d2 = b1(1)*2
    call satadj(nrk, dt, d1, d2, ruh, rvh, rmh, pi0, th0, rho, rr, pp3d, prs, th3d, q3d)
    call wr('rho', rho, size(rho)); call wr('pp3d', pp3d, size(pp3d))
    call wr('th3d', th3d, size(th3d)); call wr('prs', prs, size(prs)); call wr('q3d', q3d, size(q3d))
    b1(1) = d1; b2(1) = d2
    call wr('tcond', b1, 1); call wr('tevac', b2, 1)
  case (3)
    call rd('rho', rho, size(rho)); call rd('qr3d', qr, size(qr)); call rd('vr', vr, size(vr))
    call k_fallout(rho, qr, vr)
    call wr('vr', vr, size(vr))
  case (4)
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('zh', zh, size(zh)); call rd('mh', mh, size(mh)); call rd('mf', mf, size(mf))
    call rd('rain', rain, size(rain)); call rd('prate', prate, size(prate))
    call rd('rr', rr, size(rr)); call rd('rho', rho, size(rho))
    call rd('qr3d', qr, size(qr)); call rd('vr', vr, size(vr))
    call rd('bud', b1, 1); d1 = b1(1)
    call fallout(dt, d1, ruh, rvh, zh, mh, mf, rain, prate, rr, rho, qr, vr)
    call wr('rain', rain, size(rain)); call wr('prate', prate, size(prate))
    call wr('qr3d', qr, size(qr)); call wr('vr', vr, size(vr))
    b1(1) = d1; call wr('train', b1, 1)
  case (5)
    call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
    call rd('t', t, size(t)); call rd('rr', rr, size(rr))
    call rd('qr3d', qr, size(qr)); call rd('vr', vr, size(vr))
    call rd('bud', b1, 1); d1 = b1(1)
    call geterain(dt, cpx, lx1, d1, ruh, rvh, t, rr, qr, vr)
    b1(1) = d1; call wr('erain', b1, 1)
  case (6)
    call rd('mf', mf, size(mf)); call rd('t', t, size(t)); call rd('cvm', cvm, size(cvm))
    call rd('tten', tten, size(tten)); call rd('qr3d', qr, size(qr)); call rd('vr', vr, size(vr))
    call getefall(setzero, cpx, mf, t, cvm, tten, qr, vr)
    call wr('tten', tten, size(tten))
  end select
end program drv_kessler
