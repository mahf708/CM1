! Harness driver for the Kessler (ptype=1) branch of mp_driver (src/mp_driver.F).
! params = [ni, nj, nk, dx, dy, dz, dt, eqtset, tsmall, qsmall, nrain]
program drv_kessler_mp
  use harness_io
  use input
  use constants, only : set_constants
  use mp_driver_module, only : mp_driver
  implicit none
  real :: params(11), dt
  integer :: nstep
  double precision, allocatable :: qbudget(:), asq(:), bsq(:), dum2d1(:,:), dum2d2(:,:), &
       dum2d3(:,:), dum2d4(:,:), dum2d5(:,:)
  real, allocatable :: qb(:), aq(:)
  real, allocatable :: xh(:), ruh(:), xf(:), ruf(:), yh(:), rvh(:), yf(:), rvf(:)
  real, allocatable, dimension(:,:,:) :: mh, rmh, c1, c2, zh, mf, rmf, zf, rain, pi0, th0, &
       rho0, prs0, qv0, rho, prs, dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8, w3d, &
       ppi, pp3d, ppten, sten, tha, th3d, thten, p3a, effc, effi, effs, effr, effg, effis, &
       out2d, tmp2d
  real, allocatable :: prate(:,:), tmp1d(:,:)
  real, allocatable, dimension(:,:,:,:) :: qa, q3d, qten, p3o, tdiag, qdiag, out3d
  logical :: getdbz, getvt

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dx = params(4); dy = params(5); dz = params(6); dt = params(7)
  eqtset = nint(params(8)); tsmall = params(9); qsmall = params(10); nrain = nint(params(11))
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  numq = 3; nqv = 1; nqc = 2; nqr = 3; nql1 = 2; nql2 = 3; nqs1 = -2; nqs2 = -2
  iice = 0; timestats = 0; myid = 0; nbudget = 10
  imoist = 1; ptype = 1; pdscheme = 1; efall = 0; axisymm = 0; output_fallvel = 0
  qd_dbz = 0; td_mp = 0; qd_mp = 0; td_efall = 0; stopit = .false.
  ibp3 = 1; iep3 = 1; jbp3 = 1; jep3 = 1; kbp3 = 1; kep3 = 1; np3a = 1; np3o = 1
  ibr = 1; ier = 1; jbr = 1; jer = 1; kbr = 1; ker = 1
  ibdt = 1; iedt = 1; jbdt = 1; jedt = 1; kbdt = 1; kedt = 1; ntdiag = 1
  ibdq = 1; iedq = 1; jbdq = 1; jedq = 1; kbdq = 1; kedq = 1; nqdiag = 1
  ib2d = 1; ie2d = 1; jb2d = 1; je2d = 1; nout2d = 1
  ib3d = 1; ie3d = 1; jb3d = 1; je3d = 1; kb3d = 1; ke3d = 1; nout3d = 1
  ibtmp = 1; ietmp = 1; kbtmp = 1; ketmp = 1; ntmp2d = 1; ntmp1d = 1
  call set_constants(0)

  allocate(qbudget(nbudget), asq(numq), bsq(numq), qb(nbudget), aq(numq))
  allocate(dum2d1(ib:ie,jb:je), dum2d2(ib:ie,jb:je), dum2d3(ib:ie,jb:je), dum2d4(ib:ie,jb:je), &
       dum2d5(ib:ie,jb:je))
  allocate(xh(ib:ie), ruh(ib:ie), xf(ib:ie+1), ruf(ib:ie+1), yh(jb:je), rvh(jb:je), &
       yf(jb:je+1), rvf(jb:je+1))
  allocate(mh(ib:ie,jb:je,kb:ke), rmh(ib:ie,jb:je,kb:ke), c1(ib:ie,jb:je,kb:ke), &
       c2(ib:ie,jb:je,kb:ke), zh(ib:ie,jb:je,kb:ke), mf(ib:ie,jb:je,kb:ke+1), &
       rmf(ib:ie,jb:je,kb:ke+1), zf(ib:ie,jb:je,kb:ke+1), rain(ib:ie,jb:je,nrain), &
       prate(ib:ie,jb:je))
  allocate(pi0(ib:ie,jb:je,kb:ke), th0(ib:ie,jb:je,kb:ke), rho0(ib:ie,jb:je,kb:ke), &
       prs0(ib:ie,jb:je,kb:ke), qv0(ib:ie,jb:je,kb:ke), rho(ib:ie,jb:je,kb:ke), &
       prs(ib:ie,jb:je,kb:ke), dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), &
       dum3(ib:ie,jb:je,kb:ke), dum4(ib:ie,jb:je,kb:ke), dum5(ib:ie,jb:je,kb:ke), &
       dum6(ib:ie,jb:je,kb:ke), dum7(ib:ie,jb:je,kb:ke), dum8(ib:ie,jb:je,kb:ke), &
       w3d(ib:ie,jb:je,kb:ke+1), ppi(ib:ie,jb:je,kb:ke), pp3d(ib:ie,jb:je,kb:ke), &
       ppten(ib:ie,jb:je,kb:ke), sten(ib:ie,jb:je,kb:ke), tha(ib:ie,jb:je,kb:ke), &
       th3d(ib:ie,jb:je,kb:ke), thten(ib:ie,jb:je,kb:ke))
  allocate(qa(ibm:iem,jbm:jem,kbm:kem,numq), q3d(ibm:iem,jbm:jem,kbm:kem,numq), &
       qten(ibm:iem,jbm:jem,kbm:kem,numq))
  allocate(p3a(1,1,1), p3o(1,1,1,1), effc(1,1,1), effi(1,1,1), effs(1,1,1), effr(1,1,1), &
       effg(1,1,1), effis(1,1,1), tdiag(1,1,1,1), qdiag(1,1,1,1), out2d(1,1,1), &
       out3d(1,1,1,1), tmp2d(1,1,1), tmp1d(1,1))
  xh = 0; xf = 0; ruf = 1; yh = 0; yf = 0; rvf = 1; c1 = 0; c2 = 0; rmf = 1; zf = 0
  rho0 = 1; prs0 = 1e5; qv0 = 0; w3d = 0; ppi = 0; ppten = 0; sten = 0; tha = 0; qa = 0
  dum1 = 0; dum2 = 0; dum3 = 0; dum4 = 0; dum5 = 0; dum6 = 0; dum7 = 0; dum8 = 0
  thten = 0; qten = 0; bsq = 0

  call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh))
  call rd('mh', mh, size(mh)); call rd('rmh', rmh, size(rmh)); call rd('zh', zh, size(zh))
  call rd('mf', mf, size(mf)); call rd('rain', rain, size(rain)); call rd('prate', prate, size(prate))
  call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0))
  call rd('rho', rho, size(rho)); call rd('prs', prs, size(prs))
  call rd('pp3d', pp3d, size(pp3d)); call rd('th3d', th3d, size(th3d)); call rd('q3d', q3d, size(q3d))
  call rd('qbudget', qb, size(qb)); qbudget = qb
  call rd('asq', aq, size(aq)); asq = aq
  nstep = 1; getdbz = .false.; getvt = .false.

  call mp_driver(nstep, dt, qbudget, asq, bsq, xh, ruh, xf, ruf, yh, rvh, yf, rvf, &
       mh, rmh, c1, c2, zh, mf, rmf, zf, rain, prate, pi0, th0, rho0, prs0, qv0, &
       rho, prs, dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8, &
       w3d, ppi, pp3d, ppten, sten, tha, th3d, thten, qa, q3d, qten, &
       p3a, p3o, dum2d1, dum2d2, dum2d3, dum2d4, dum2d5, &
       effc, effi, effs, effr, effg, effis, &
       tdiag, qdiag, out2d, out3d, tmp2d, tmp1d, &
       .false., .false., .false., .false., .false., &
       getdbz, getvt, .false., .false.)

  call wr('rho', rho, size(rho)); call wr('prs', prs, size(prs)); call wr('pp3d', pp3d, size(pp3d))
  call wr('th3d', th3d, size(th3d)); call wr('q3d', q3d, size(q3d)); call wr('qten', qten, size(qten))
  call wr('thten', thten, size(thten)); call wr('dum1', dum1, size(dum1)); call wr('dum3', dum3, size(dum3))
  call wr('rain', rain, size(rain)); call wr('prate', prate, size(prate))
  qb = qbudget; call wr('qbudget', qb, size(qb))
  aq = asq; call wr('asq', aq, size(aq))
end program drv_kessler_mp
