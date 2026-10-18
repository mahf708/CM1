! Harness driver for base (src/base.F) with terrain_flag=.true. (zh, zf 3-D),
! single node; params as drv_base.F90 plus [wbc(=ebc), sbc(=nbc), testcase].
! params = [ni, nj, nk, dx, dy, dz, isnd, iwnd, imoist, ptype, psolver, imove,
!           umove, vmove, cgs1, cgs2, cgs3, cgt1, cgt2, cgt3, tsmall]
! inputs : zh, mh, rmh, c1, c2 (scalar pts), zf, mf (w pts)
! outputs: pi0 prs0 rho0 thv0 th0 rth0 qv0 qc0 qi0 rr0 rf0 rrf0 t0 rh0 u0 v0
!          rho0s pi0s prs0s rth0s
program drv_base_terrain
  use harness_io
  use input
  use constants, only : set_constants
  use base_module
  implicit none
  real :: params(24)
  real, allocatable, dimension(:,:,:) :: zh,mh,rmh,c1,c2,zf,mf
  real, allocatable, dimension(:,:) :: rho0s,pi0s,prs0s,rth0s
  real, allocatable, dimension(:) :: wprof,ufrc,vfrc,thfrc,qvfrc,ug,vg,dvdr,uavg,vavg,thavg,pavg,ulspg,vlspg
  real, allocatable :: qavg(:,:)
  real, allocatable, dimension(:,:,:) :: pi0,prs0,rho0,thv0,th0,rth0,qv0,qc0,qi0,rr0,rf0,rrf0,t0,rh0,u0,v0,thrd
  integer, allocatable :: reqs(:)
  real, allocatable :: b1(:), b3(:,:,:)

  call set_constants(0)
  call rd('params', params, 24)
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dx = params(4); dy = params(5); dz = params(6)
  isnd = nint(params(7)); iwnd = nint(params(8)); imoist = nint(params(9))
  ptype = nint(params(10)); psolver = nint(params(11)); imove = nint(params(12))
  umove = params(13); vmove = params(14)
  cgs1 = params(15); cgs2 = params(16); cgs3 = params(17)
  cgt1 = params(18); cgt2 = params(19); cgt3 = params(20)
  tsmall = params(21)
  nx = ni; ny = nj; nz = nk
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibb2 = 1; ibe2 = 1; jbb2 = 1; jbe2 = 1; kbb2 = 1; kbe2 = 1
  imp = 1; jmp = 1; kmp = 1; kmt = 1; rmp = 1; cmp = 1
  numq = 3; nqv = 1
  terrain_flag = .true.; testcase = nint(params(24)); lspgrad = 0; axisymm = 0
  wbc = nint(params(22)); ebc = wbc; sbc = nint(params(23)); nbc = sbc
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc.ne.1) ibw = 1
  if (ebc.ne.1) ibe = 1
  if (sbc.ne.1) ibs = 1
  if (nbc.ne.1) ibn = 1
  dowr = .false.; myid = 0; outfile = 6

  allocate(zh(ib:ie,jb:je,kb:ke),mh(ib:ie,jb:je,kb:ke),rmh(ib:ie,jb:je,kb:ke))
  allocate(c1(ib:ie,jb:je,kb:ke),c2(ib:ie,jb:je,kb:ke))
  allocate(zf(ib:ie,jb:je,kb:ke+1),mf(ib:ie,jb:je,kb:ke+1))
  allocate(rho0s(ib:ie,jb:je),pi0s(ib:ie,jb:je),prs0s(ib:ie,jb:je),rth0s(ib:ie,jb:je))
  allocate(wprof(kb:ke),ufrc(kb:ke),vfrc(kb:ke),thfrc(kb:ke),qvfrc(kb:ke),ug(kb:ke),vg(kb:ke), &
           dvdr(kb:ke),uavg(kb:ke),vavg(kb:ke),thavg(kb:ke),pavg(kb:ke),ulspg(kb:ke),vlspg(kb:ke))
  allocate(qavg(kb:ke,numq))
  wprof=0; ufrc=0; vfrc=0; thfrc=0; qvfrc=0; ug=0; vg=0; dvdr=0; uavg=0; vavg=0
  thavg=0; pavg=0; ulspg=0; vlspg=0; qavg=0
  allocate(pi0(ib:ie,jb:je,kb:ke),prs0(ib:ie,jb:je,kb:ke),rho0(ib:ie,jb:je,kb:ke), &
           thv0(ib:ie,jb:je,kb:ke),th0(ib:ie,jb:je,kb:ke),rth0(ib:ie,jb:je,kb:ke), &
           qv0(ib:ie,jb:je,kb:ke),qc0(ib:ie,jb:je,kb:ke),qi0(ib:ie,jb:je,kb:ke),  &
           rr0(ib:ie,jb:je,kb:ke),rf0(ib:ie,jb:je,kb:ke),rrf0(ib:ie,jb:je,kb:ke), &
           t0(ib:ie,jb:je,kb:ke),rh0(ib:ie,jb:je,kb:ke))
  allocate(u0(ib:ie+1,jb:je,kb:ke),v0(ib:ie,jb:je+1,kb:ke),thrd(ibb2:ibe2,jbb2:jbe2,kbb2:kbe2))
  pi0=0; prs0=0; rho0=0; thv0=0; th0=0; rth0=0; qv0=0; qc0=0; qi0=0
  rr0=0; rf0=0; rrf0=0; t0=0; rh0=0; u0=0; v0=0; thrd=0
  rho0s=0; pi0s=0; prs0s=0; rth0s=0
  allocate(reqs(rmp), b1(kmt), b3(2,2,kmt+2))
  reqs=0; b1=0; b3=0

  call rd('zh',zh,size(zh)); call rd('mh',mh,size(mh)); call rd('rmh',rmh,size(rmh))
  call rd('c1',c1,size(c1)); call rd('c2',c2,size(c2))
  call rd('zf',zf,size(zf)); call rd('mf',mf,size(mf))

  call base(zh,mh,rmh,c1,c2,zf,mf,rho0s,pi0s,prs0s,rth0s,         &
            wprof,ufrc,vfrc,thfrc,qvfrc,ug,vg,dvdr,               &
            uavg,vavg,thavg,pavg,qavg,ulspg,vlspg,                &
            pi0,prs0,rho0,thv0,th0,rth0,qv0,u0,v0,thrd,           &
            qc0,qi0,rr0,rf0,rrf0,t0  ,rh0 ,                       &
            reqs,reqs,reqs,b1,b1,b1,b1,b1,b1,b1,b1,               &
            b3,b3,b3,b3,b3,b3,b3,b3,                              &
            b3,b3,b3,b3,b3,b3,b3,b3,                              &
            b3,b3,b3,b3,b3,b3,b3,b3,                              &
            b3,b3,b3,b3,b3,b3,b3,b3)

  call wr('pi0',pi0,size(pi0)); call wr('prs0',prs0,size(prs0)); call wr('rho0',rho0,size(rho0))
  call wr('thv0',thv0,size(thv0)); call wr('th0',th0,size(th0)); call wr('rth0',rth0,size(rth0))
  call wr('qv0',qv0,size(qv0)); call wr('qc0',qc0,size(qc0)); call wr('qi0',qi0,size(qi0))
  call wr('rr0',rr0,size(rr0)); call wr('rf0',rf0,size(rf0)); call wr('rrf0',rrf0,size(rrf0))
  call wr('t0',t0,size(t0)); call wr('rh0',rh0,size(rh0))
  call wr('u0',u0,size(u0)); call wr('v0',v0,size(v0))
  call wr('rho0s',rho0s,size(rho0s)); call wr('pi0s',pi0s,size(pi0s))
  call wr('prs0s',prs0s,size(prs0s)); call wr('rth0s',rth0s,size(rth0s))
end program drv_base_terrain
