! Harness driver for adv.F: advs, advu, advv, advw (+ misclibs getdivx).
! Configuration: 5th-order flux advection, no WENO, no terrain, periodic,
! Cartesian, no diffusion, no budgets (reference case kessler_periodic).
program drv_adv
  use harness_io
  use input
  use constants, only : epsilon
  use ib_module, only : ibib, ieib, jbib, jeib, kbib, keib, kmaxib
  use adv_module
  use misclibs, only : getdivx
  implicit none

  real :: p(12)
  integer :: nrk, pdef, wflag
  real :: dt
  double precision :: bsq, weps
  real, allocatable, dimension(:) :: xh,rxh,arh1,arh2,uh,ruh,xf,rxf,arf1,arf2,uf,vh,rvh,vf, &
                                      rds,rdsf,sigma,sigmaf,wprof
  double precision, allocatable :: dumk1(:),dumk2(:)
  real, allocatable, dimension(:,:) :: gz,rgz,gzu,gzv,rgzu,rgzv,dzdx,dzdy
  real, allocatable, dimension(:,:,:) :: mh,rmh,mf,rho0,rr0,rf0,rrf0,c1,c2,rho,ri, &
       advx,advy,advz,dumx,dumy,dumz,mass,subs,divx,s0,s,sten, &
       rru,rrv,rrw,u3d,uten,v3d,vten,w3d,wten,rru2,rrv2,rrw2,divx2
  real, allocatable :: sw31(:,:,:),sw32(:,:,:),se31(:,:,:),se32(:,:,:)
  real, allocatable :: ss31(:,:,:),ss32(:,:,:),sn31(:,:,:),sn32(:,:,:)
  logical, allocatable :: flag(:,:,:), bndy(:,:,:)
  integer, allocatable :: kbdy(:,:), hflxw(:,:,:),hflxe(:,:,:),hflxs(:,:,:),hflxn(:,:,:)
  real, allocatable :: diag(:,:,:,:), out3d(:,:,:,:), udiag(:,:,:,:), vdiag(:,:,:,:), wdiag(:,:,:,:)

  call rd('params', p, size(p))
  ni = nint(p(1)); nj = nint(p(2)); nk = nint(p(3))
  nrk = nint(p(4)); pdef = nint(p(5)); wflag = nint(p(6))
  dt = p(7); dx = p(8); dy = p(9); dz = p(10)

  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  itb = 1; ite = 1; jtb = 1; jte = 1
  cmp = ngxy; imp = ni; jmp = nj; kmp = max(2,nk)
  ibib = 1; ieib = 1; jbib = 1; jeib = 1; kbib = 1; keib = 1; kmaxib = 1
  ib3d = 1; ie3d = 1; jb3d = 1; je3d = 1; kb3d = 1; ke3d = 1; nout3d = 1
  ibdv = 1; iedv = 1; jbdv = 1; jedv = 1; kbdv = 1; kedv = 1; nudiag = 1; nvdiag = 1; nwdiag = 1

  terrain_flag = .false.; axisymm = 0
  wbc = 1; ebc = 1; sbc = 1; nbc = 1; ibw = 0; ibe = 0; ibs = 0; ibn = 0
  advwenos = 0; advwenov = 0; weno_order = 5
  hadvordrs = 5; vadvordrs = 5; hadvordrv = 5; vadvordrv = 5
  idiff = 0; difforder = 6; mdiff = 0
  do_ib = .false.; dolsw = .false.; stat_qsrc = 0; pdscheme = 1; timestats = 0
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  smeps = 1.0e-30

  allocate(xh(ib:ie),rxh(ib:ie),arh1(ib:ie),arh2(ib:ie),uh(ib:ie),ruh(ib:ie))
  allocate(xf(ib:ie+1),rxf(ib:ie+1),arf1(ib:ie+1),arf2(ib:ie+1),uf(ib:ie+1))
  allocate(vh(jb:je),rvh(jb:je),vf(jb:je+1))
  allocate(rds(kb:ke),sigma(kb:ke),wprof(kb:ke),rdsf(kb:ke+1),sigmaf(kb:ke+1))
  allocate(dumk1(kb:ke),dumk2(kb:ke))
  allocate(gz(itb:ite,jtb:jte),rgz(itb:ite,jtb:jte),gzu(itb:ite,jtb:jte),gzv(itb:ite,jtb:jte), &
           rgzu(itb:ite,jtb:jte),rgzv(itb:ite,jtb:jte),dzdx(itb:ite,jtb:jte),dzdy(itb:ite,jtb:jte))
  xh=0; rxh=0; arh1=1; arh2=1; ruh=1; xf=0; rxf=0; arf1=1; arf2=1; rvh=1
  rds=0; sigma=0; wprof=0; rdsf=0; sigmaf=0; dumk1=0; dumk2=0
  gz=1; rgz=1; gzu=1; gzv=1; rgzu=1; rgzv=1; dzdx=0; dzdy=0

  allocate(mh(ib:ie,jb:je,kb:ke),rmh(ib:ie,jb:je,kb:ke),rho0(ib:ie,jb:je,kb:ke),rr0(ib:ie,jb:je,kb:ke), &
           rf0(ib:ie,jb:je,kb:ke),rrf0(ib:ie,jb:je,kb:ke),c1(ib:ie,jb:je,kb:ke),c2(ib:ie,jb:je,kb:ke), &
           rho(ib:ie,jb:je,kb:ke),ri(ib:ie,jb:je,kb:ke),advx(ib:ie,jb:je,kb:ke),advy(ib:ie,jb:je,kb:ke), &
           advz(ib:ie,jb:je,kb:ke),dumx(ib:ie,jb:je,kb:ke),dumy(ib:ie,jb:je,kb:ke),dumz(ib:ie,jb:je,kb:ke), &
           mass(ib:ie,jb:je,kb:ke),subs(ib:ie,jb:je,kb:ke),divx(ib:ie,jb:je,kb:ke),s0(ib:ie,jb:je,kb:ke), &
           s(ib:ie,jb:je,kb:ke),sten(ib:ie,jb:je,kb:ke),divx2(ib:ie,jb:je,kb:ke),flag(ib:ie,jb:je,kb:ke))
  allocate(mf(ib:ie,jb:je,kb:ke+1),rrw(ib:ie,jb:je,kb:ke+1),w3d(ib:ie,jb:je,kb:ke+1), &
           wten(ib:ie,jb:je,kb:ke+1),rrw2(ib:ie,jb:je,kb:ke+1))
  allocate(rru(ib:ie+1,jb:je,kb:ke),u3d(ib:ie+1,jb:je,kb:ke),uten(ib:ie+1,jb:je,kb:ke),rru2(ib:ie+1,jb:je,kb:ke))
  allocate(rrv(ib:ie,jb:je+1,kb:ke),v3d(ib:ie,jb:je+1,kb:ke),vten(ib:ie,jb:je+1,kb:ke),rrv2(ib:ie,jb:je+1,kb:ke))
  allocate(sw31(cmp,jmp,kmp),sw32(cmp,jmp,kmp),se31(cmp,jmp,kmp),se32(cmp,jmp,kmp))
  allocate(ss31(imp,cmp,kmp),ss32(imp,cmp,kmp),sn31(imp,cmp,kmp),sn32(imp,cmp,kmp))
  allocate(bndy(ibib:ieib,jbib:jeib,kbib:keib),kbdy(ibib:ieib,jbib:jeib))
  allocate(hflxw(ibib:ieib,jbib:jeib,kmaxib),hflxe(ibib:ieib,jbib:jeib,kmaxib), &
           hflxs(ibib:ieib,jbib:jeib,kmaxib),hflxn(ibib:ieib,jbib:jeib,kmaxib))
  allocate(diag(1,1,1,1),out3d(1,1,1,1),udiag(1,1,1,1),vdiag(1,1,1,1),wdiag(1,1,1,1))
  rmh=1; rho=1; ri=1; advx=0; advy=0; advz=0; dumx=0; dumy=0; dumz=0; mass=0; subs=0; flag=.false.
  sw31=0; sw32=0; se31=0; se32=0; ss31=0; ss32=0; sn31=0; sn32=0
  bndy=.false.; kbdy=1; hflxw=0; hflxe=0; hflxs=0; hflxn=0
  diag=0; out3d=0; udiag=0; vdiag=0; wdiag=0

  call rd('uh', uh, size(uh));   call rd('uf', uf, size(uf))
  call rd('vh', vh, size(vh));   call rd('vf', vf, size(vf))
  call rd('mh', mh, size(mh));   call rd('mf', mf, size(mf))
  call rd('rho0', rho0, size(rho0)); call rd('rr0', rr0, size(rr0))
  call rd('rf0', rf0, size(rf0));    call rd('rrf0', rrf0, size(rrf0))
  call rd('c1', c1, size(c1));   call rd('c2', c2, size(c2))
  call rd('rru', rru, size(rru)); call rd('rrv', rrv, size(rrv)); call rd('rrw', rrw, size(rrw))
  call rd('divx', divx, size(divx))
  call rd('s0', s0, size(s0)); call rd('s', s, size(s)); call rd('sten', sten, size(sten))
  call rd('u3d', u3d, size(u3d)); call rd('uten', uten, size(uten))
  call rd('v3d', v3d, size(v3d)); call rd('vten', vten, size(vten))
  call rd('w3d', w3d, size(w3d)); call rd('wten', wten, size(wten))

  bsq = 0.0d0
  weps = 0.01d0*epsilon

  call advs(nrk,wflag,0,bsq,xh,rxh,arh1,arh2,uh,ruh,xf,vh,rvh,gz,rgz,mh,rmh,            &
            rho0,rr0,rf0,rrf0,advx,advy,advz,dumx,dumy,dumz,mass,subs,divx,              &
            rru,rrv,rrw,s0,s,sten,pdef,0,dt,weps,                                        &
            flag,sw31,sw32,se31,se32,ss31,ss32,sn31,sn32,rdsf,c1,c2,rho,ri,0,            &
            .false.,1,1,1,1,1,1,1,diag,1,1,1,                                            &
            1,1,1,wprof,dumk1,dumk2,hadvordrs,vadvordrs,kbdy,bndy,hflxw,hflxe,hflxs,hflxn,out3d)
  call wr('sten', sten, size(sten))

  call advu(nrk,arh1,arh2,uh,xf,rxf,arf1,arf2,uf,vh,gz,rgz,gzu,mh,rho0,rr0,rf0,rrf0,     &
            dumx,dumy,dumz,advx,advy,advz,subs,divx,                                     &
            rru,u3d,uten,rrv,rrw,rdsf,c1,c2,rho,dt,.false.,udiag,wprof,bndy,kbdy,        &
            hflxw,hflxe,hflxs,hflxn,vf,mf,v3d,w3d)
  call wr('uten', uten, size(uten))

  call advv(nrk,xh,rxh,arh1,arh2,uh,xf,vh,vf,gz,rgz,gzv,mh,rho0,rr0,rf0,rrf0,            &
            dumx,dumy,dumz,advx,advy,advz,subs,divx,                                     &
            rru,rrv,v3d,vten,rrw,rdsf,c1,c2,rho,dt,.false.,vdiag,wprof,bndy,kbdy,        &
            hflxw,hflxe,hflxs,hflxn,uf,mf,u3d,w3d)
  call wr('vten', vten, size(vten))

  call advw(nrk,xh,rxh,arh1,arh2,uh,xf,vh,gz,rgz,mh,mf,rho0,rr0,rf0,rrf0,                &
            dumx,dumy,dumz,advx,advy,advz,subs,divx,                                     &
            rru,rrv,rrw,w3d,wten,rds,rdsf,c1,c2,rho,dt,                                  &
            .false.,wdiag,hadvordrv,vadvordrv,advwenov,bndy,kbdy,uf,vf,u3d,v3d,          &
            hflxw,hflxe,hflxs,hflxn,.true.)
  call wr('wten', wten, size(wten))

  ! getdivx on copies (treat u3d/v3d/w3d as the time-averaged velocities)
  rru2 = u3d; rrv2 = v3d; rrw2 = w3d; divx2 = divx
  call getdivx(arh1,arh2,uh,vh,mh,rho0,rf0,rru2,rrv2,rrw2,divx2,  &
               rds,rdsf,sigma,sigmaf,gz,rgzu,rgzv,dzdx,dzdy)
  call wr('rru2', rru2, size(rru2)); call wr('rrv2', rrv2, size(rrv2))
  call wr('rrw2', rrw2, size(rrw2)); call wr('divx2', divx2, size(divx2))
end program drv_adv
