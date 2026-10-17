! Harness driver for init3d (src/init3d.F): no terrain, periodic, single node,
! Kessler moisture (numq=3: qv,qc,qr) or dry.
! params = [ni, nj, nk, dx, dy, dz, iinit, ibalance, imoist, ptype, isnd, psolver,
!           irandp, centerx, centery]
! inputs : xh(ib:ie), yh(jb:je), zh, pi0, prs0, th0, thv0, qv0, rho0, rf0, mh (scalar),
!          mf, zf (w pts), u0 (u pts), v0 (v pts)
! outputs: ua, u3d, va, v3d, tha, qa(ib:ie,jb:je,kb:ke,numq), ppi
program drv_init3d
  use harness_io
  use input
  use constants, only : set_constants
  use init3d_module
  implicit none
  real :: params(15)
  real, allocatable, dimension(:) :: xh,rxh,uh,ruh,xf,rxf,uf,ruf,yh,vh,rvh,yf,vf,rvf
  real, allocatable, dimension(:) :: xfref,xhref,yfref,yhref,sigma
  real, allocatable, dimension(:,:) :: gz,zs,rho0s,pi0s,prs0s,d2
  real, allocatable, dimension(:,:,:) :: c1,c2,zh,mh,rmh,zf,mf,rmf
  real, allocatable, dimension(:,:,:) :: pi0,prs0,rho0,thv0,th0,rth0,qv0,qc0,qi0,rr0,rf0,rrf0
  real, allocatable, dimension(:,:,:) :: swath
  real, allocatable, dimension(:,:,:) :: dum1,dum2,dum3,dum4,divx,rho,prs
  real, allocatable, dimension(:,:,:) :: u0,rru,ua,u3d,uten,uten1
  real, allocatable, dimension(:,:,:) :: v0,rrv,va,v3d,vten,vten1
  real, allocatable, dimension(:,:,:) :: rrw,wa,w3d,wten,wten1
  real, allocatable, dimension(:,:,:) :: ppi,pp3d,ppten,sten,tha,th3d,thten,thten1
  real, allocatable, dimension(:,:,:,:) :: qa,q3d,qten,pta
  real, allocatable, dimension(:,:,:) :: small3
  real, allocatable, dimension(:,:) :: pdata
  real, allocatable, dimension(:) :: k1
  complex, allocatable, dimension(:,:,:) :: cx3
  complex, allocatable, dimension(:,:) :: cx2
  integer :: nq

  call set_constants(0)
  call rd('params', params, 15)
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dx = params(4); dy = params(5); dz = params(6)
  iinit = nint(params(7)); ibalance = nint(params(8)); imoist = nint(params(9))
  ptype = nint(params(10)); isnd = nint(params(11)); psolver = nint(params(12))
  irandp = nint(params(13)); centerx = params(14); centery = params(15)
  nx = ni; ny = nj; nz = nk
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  ibc = 1; iec = 1; jbc = 1; jec = 1; kbc = 1; kec = 1
  ibt = 1; iet = 1; jbt = 1; jet = 1; kbt = 1; ket = 1
  ibp = 1; iep = 1; jbp = 1; jep = 1; kbp = 1; kep = 1
  itb = 1; ite = 1; jtb = 1; jte = 1
  ipb = 1; ipe = 1; jpb = 1; jpe = 1; kpb = 1; kpe = 1
  nrain = 1; npt = 1; nparcels = 1; npvals = 1
  if (imoist == 1) then
    numq = 3; nqv = 1; nqc = 2; nqr = 3; nql1 = 2; nql2 = 3
  else
    numq = 1; nqv = 1; nqc = 0; nqr = 0; nql1 = 1; nql2 = 0
  endif
  iice = 0; iptra = 0; iprcl = 0; testcase = 0; pdcomp = .false.
  idoles = .false.; iusetke = .false.
  terrain_flag = .false.; axisymm = 0
  wbc = 1; ebc = 1; sbc = 1; nbc = 1; ibw = 0; ibe = 0; ibs = 0; ibn = 0
  dowr = .false.; myid = 0; outfile = 6; myi1 = 1; myj1 = 1
  stretch_x = 0; stretch_y = 0

  allocate(xh(ib:ie),rxh(ib:ie),uh(ib:ie),ruh(ib:ie),xf(ib:ie+1),rxf(ib:ie+1),uf(ib:ie+1),ruf(ib:ie+1))
  allocate(yh(jb:je),vh(jb:je),rvh(jb:je),yf(jb:je+1),vf(jb:je+1),rvf(jb:je+1))
  allocate(xfref(1-ngxy:nx+ngxy+1),xhref(1-ngxy:nx+ngxy+1),yfref(1-ngxy:ny+ngxy+1),yhref(1-ngxy:ny+ngxy+1))
  allocate(sigma(kb:ke))
  rxh=0; uh=1; ruh=1; xf=0; rxf=0; uf=1; ruf=1; vh=1; rvh=1; yf=0; vf=1; rvf=1
  xfref=0; xhref=0; yfref=0; yhref=0; sigma=0
  allocate(gz(itb:ite,jtb:jte),zs(ib:ie,jb:je),rho0s(ib:ie,jb:je),pi0s(ib:ie,jb:je),prs0s(ib:ie,jb:je),d2(ib:ie,jb:je))
  gz=1; zs=0; rho0s=0; pi0s=0; prs0s=0; d2=0
  allocate(c1(ib:ie,jb:je,kb:ke),c2(ib:ie,jb:je,kb:ke),zh(ib:ie,jb:je,kb:ke),mh(ib:ie,jb:je,kb:ke),rmh(ib:ie,jb:je,kb:ke))
  allocate(zf(ib:ie,jb:je,kb:ke+1),mf(ib:ie,jb:je,kb:ke+1),rmf(ib:ie,jb:je,kb:ke+1))
  c1=0.5; c2=0.5; rmh=1; rmf=1
  allocate(pi0(ib:ie,jb:je,kb:ke),prs0(ib:ie,jb:je,kb:ke),rho0(ib:ie,jb:je,kb:ke),thv0(ib:ie,jb:je,kb:ke), &
           th0(ib:ie,jb:je,kb:ke),rth0(ib:ie,jb:je,kb:ke),qv0(ib:ie,jb:je,kb:ke),qc0(ib:ie,jb:je,kb:ke), &
           qi0(ib:ie,jb:je,kb:ke),rr0(ib:ie,jb:je,kb:ke),rf0(ib:ie,jb:je,kb:ke),rrf0(ib:ie,jb:je,kb:ke))
  rth0=0; qc0=0; qi0=0; rr0=0; rrf0=0
  allocate(swath(ib:ie,jb:je,nrain))
  allocate(dum1(ib:ie,jb:je,kb:ke),dum2(ib:ie,jb:je,kb:ke),dum3(ib:ie,jb:je,kb:ke),dum4(ib:ie,jb:je,kb:ke), &
           divx(ib:ie,jb:je,kb:ke),rho(ib:ie,jb:je,kb:ke),prs(ib:ie,jb:je,kb:ke))
  dum1=0; dum2=0; dum3=0; dum4=0; divx=0; rho=0; prs=0
  allocate(u0(ib:ie+1,jb:je,kb:ke),rru(ib:ie+1,jb:je,kb:ke),ua(ib:ie+1,jb:je,kb:ke),u3d(ib:ie+1,jb:je,kb:ke), &
           uten(ib:ie+1,jb:je,kb:ke),uten1(ib:ie+1,jb:je,kb:ke))
  allocate(v0(ib:ie,jb:je+1,kb:ke),rrv(ib:ie,jb:je+1,kb:ke),va(ib:ie,jb:je+1,kb:ke),v3d(ib:ie,jb:je+1,kb:ke), &
           vten(ib:ie,jb:je+1,kb:ke),vten1(ib:ie,jb:je+1,kb:ke))
  allocate(rrw(ib:ie,jb:je,kb:ke+1),wa(ib:ie,jb:je,kb:ke+1),w3d(ib:ie,jb:je,kb:ke+1), &
           wten(ib:ie,jb:je,kb:ke+1),wten1(ib:ie,jb:je,kb:ke+1))
  rru=0; ua=0; u3d=0; uten=0; uten1=0; rrv=0; va=0; v3d=0; vten=0; vten1=0
  rrw=0; wa=0; w3d=0; wten=0; wten1=0
  allocate(ppi(ib:ie,jb:je,kb:ke),pp3d(ib:ie,jb:je,kb:ke),ppten(ib:ie,jb:je,kb:ke),sten(ib:ie,jb:je,kb:ke), &
           tha(ib:ie,jb:je,kb:ke),th3d(ib:ie,jb:je,kb:ke),thten(ib:ie,jb:je,kb:ke),thten1(ib:ie,jb:je,kb:ke))
  ppi=0; pp3d=0; ppten=0; sten=0; tha=0; th3d=0; thten=0; thten1=0
  allocate(qa(ibm:iem,jbm:jem,kbm:kem,numq),q3d(ibm:iem,jbm:jem,kbm:kem,numq),qten(ibm:iem,jbm:jem,kbm:kem,numq))
  qa=0; q3d=0; qten=0
  allocate(pta(ibp:iep,jbp:jep,kbp:kep,npt),small3(1,1,1),pdata(nparcels,npvals),k1(kpb:kpe))
  pta=0; small3=0; pdata=0; k1=0
  allocate(cx3(ipb:ipe,jpb:jpe,kpb:kpe),cx2(ipb:ipe,jpb:jpe))
  cx3=0; cx2=0

  call rd('xh',xh,size(xh)); call rd('yh',yh,size(yh)); call rd('zh',zh,size(zh))
  call rd('mh',mh,size(mh)); call rd('mf',mf,size(mf)); call rd('zf',zf,size(zf))
  call rd('pi0',pi0,size(pi0)); call rd('prs0',prs0,size(prs0)); call rd('th0',th0,size(th0))
  call rd('thv0',thv0,size(thv0)); call rd('qv0',qv0,size(qv0)); call rd('rho0',rho0,size(rho0))
  call rd('rf0',rf0,size(rf0)); call rd('u0',u0,size(u0)); call rd('v0',v0,size(v0))

  call init3d(xh,rxh,uh,ruh,xf,rxf,uf,ruf,yh,vh,rvh,yf,vf,rvf,  &
              xfref,xhref,yfref,yhref,sigma,c1,c2,gz,zs,        &
              zh,mh,rmh,zf,mf,rmf,rho0s,pi0s,prs0s,             &
              pi0,prs0,rho0,thv0,th0,rth0,qv0,                  &
              u0,v0,qc0,qi0,rr0,rf0,rrf0,                       &
              swath,swath,swath,swath,swath,swath,swath,swath,  &
              d2,d2,d2,d2,d2,d2,                                &
              dum1,dum2,dum3,dum4,divx,rho,prs,                 &
              rru,ua,u3d,uten,uten1,rrv,va,v3d,vten,vten1,      &
              rrw,wa,w3d,wten,wten1,ppi,pp3d,ppten,sten,        &
              tha,th3d,thten,thten1,qa,q3d,qten,                &
              small3,small3,small3,small3,small3,small3,small3, &
              pta,pta,pta,                                      &
              pdata,small3,k1,k1,k1,k1,cx3,cx3,cx3,cx2,cx2)

  call wr('ua',ua,size(ua)); call wr('u3d',u3d,size(u3d))
  call wr('va',va,size(va)); call wr('v3d',v3d,size(v3d))
  call wr('tha',tha,size(tha)); call wr('ppi',ppi,size(ppi))
  call wr('qa',qa,size(qa))
end program drv_init3d
