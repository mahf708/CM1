! Harness driver for sound (src/sound.F), no terrain, psolver=3.
! params = [ni, nj, nk, nsound, nrk, dt, alph, kdiv, dx, dy, dz, smeps, wbc, ebc, sbc, nbc]
program drv_sound
  use harness_io
  use input
  use ib_module, only : ibib,ieib,jbib,jeib,kbib,keib
  use sound_module
  use constants, only : set_constants
  implicit none
  real :: params(16), dt, rtime
  double precision :: mtime
  integer :: nrk
  logical :: get_time_avg
  real, allocatable, dimension(:) :: xh,arh1,arh2,uh,ruh,xf,uf,yh,vh,rvh,yf,vf,rds,sigma,rdsf,sigmaf
  real, allocatable, dimension(:,:,:) :: zh,mh,rmh,c1,c2,mf,zf,pi0,rho0,rr0,rf0,rrf0,th0,rth0
  real, allocatable, dimension(:,:) :: zs,gz,dzdx,radbcw,radbce,radbcs,radbcn,dtu,dtu0,dtv,dtv0
  real, allocatable, dimension(:,:,:) :: gx
  real, allocatable, dimension(:,:,:) :: dum1,dum2,dum3,dum4,mm,bk,ppd,fpk,qk,pk1,pk2,wtens
  real, allocatable, dimension(:,:,:) :: u0,rru,ua,u3d,uten,v0,rrv,va,v3d,vten,rrw,wa,w3d,wten
  real, allocatable, dimension(:,:,:) :: ppi,pp3d,piadv,ppten,ppx,thv,ppterm
  logical, allocatable :: bndy(:,:,:)
  integer, allocatable :: kbdy(:,:), reqs_p(:)
  real, allocatable :: pw(:,:), ps(:,:)

  call set_constants(0)
  call rd('params', params, 16)
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  nsound = nint(params(4)); nrk = nint(params(5)); dt = params(6)
  alph = params(7); kdiv = params(8)
  dx = params(9); dy = params(10); dz = params(11); smeps = params(12)
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
  ibib = 1; ieib = 1; jbib = 1; jeib = 1; kbib = 1; keib = 1
  imp = 1; jmp = 1; kmp = 1; rmp = 1
  terrain_flag = .false.; axisymm = 0; irbc = 4; roflux = 0
  wbc = nint(params(13)); ebc = nint(params(14)); sbc = nint(params(15)); nbc = nint(params(16))
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc.ne.1) ibw = 1
  if (ebc.ne.1) ibe = 1
  if (sbc.ne.1) ibs = 1
  if (nbc.ne.1) ibn = 1
  convinit = 0; wnudge = 0; do_ib = .false.; timestats = 0
  rtime = 0.0; mtime = 0.0d0; get_time_avg = .true.

  allocate(xh(ib:ie),arh1(ib:ie),arh2(ib:ie),uh(ib:ie),ruh(ib:ie),xf(ib:ie+1),uf(ib:ie+1))
  allocate(yh(jb:je),vh(jb:je),rvh(jb:je),yf(jb:je+1),vf(jb:je+1))
  allocate(rds(kb:ke),sigma(kb:ke),rdsf(kb:ke+1),sigmaf(kb:ke+1))
  xh=0; arh1=0; arh2=0; ruh=0; xf=0; yh=0; rvh=0; yf=0; rds=0; sigma=0; rdsf=0; sigmaf=0
  allocate(zh(ib:ie,jb:je,kb:ke),rmh(ib:ie,jb:je,kb:ke),pi0(ib:ie,jb:je,kb:ke),rho0(ib:ie,jb:je,kb:ke))
  allocate(rrf0(ib:ie,jb:je,kb:ke),th0(ib:ie,jb:je,kb:ke),rth0(ib:ie,jb:je,kb:ke),zf(ib:ie,jb:je,kb:ke+1))
  zh=0; rmh=0; pi0=0; rho0=0; rrf0=0; th0=0; rth0=0; zf=0
  allocate(mh(ib:ie,jb:je,kb:ke),c1(ib:ie,jb:je,kb:ke),c2(ib:ie,jb:je,kb:ke),mf(ib:ie,jb:je,kb:ke+1))
  allocate(rr0(ib:ie,jb:je,kb:ke),rf0(ib:ie,jb:je,kb:ke))
  allocate(zs(ib:ie,jb:je),gz(1,1),dzdx(1,1),gx(1,1,1))
  zs=0; gz=1; dzdx=0; gx=0
  allocate(radbcw(jb:je,kb:ke),radbce(jb:je,kb:ke),radbcs(ib:ie,kb:ke),radbcn(ib:ie,kb:ke))
  call rd('radbcw',radbcw,size(radbcw)); call rd('radbce',radbce,size(radbce))
  call rd('radbcs',radbcs,size(radbcs)); call rd('radbcn',radbcn,size(radbcn))
  allocate(dtu(ib:ie,jb:je),dtu0(ib:ie,jb:je),dtv(ib:ie,jb:je),dtv0(ib:ie,jb:je))
  allocate(dum1(ib:ie,jb:je,kb:ke),dum2(ib:ie,jb:je,kb:ke),dum3(ib:ie,jb:je,kb:ke),dum4(ib:ie,jb:je,kb:ke))
  allocate(mm(ib:ie,jb:je,kb:ke),bk(ib:ie,jb:je,kb:ke),ppd(ib:ie,jb:je,kb:ke),fpk(ib:ie,jb:je,kb:ke))
  allocate(qk(ib:ie,jb:je,kb:ke),pk1(ib:ie,jb:je,kb:ke),pk2(ib:ie,jb:je,kb:ke),wtens(ib:ie,jb:je,kb:ke))
  dum3=0; dum4=0; bk=0; wtens=0
  allocate(u0(ib:ie+1,jb:je,kb:ke),rru(ib:ie+1,jb:je,kb:ke),ua(ib:ie+1,jb:je,kb:ke),u3d(ib:ie+1,jb:je,kb:ke),uten(ib:ie+1,jb:je,kb:ke))
  allocate(v0(ib:ie,jb:je+1,kb:ke),rrv(ib:ie,jb:je+1,kb:ke),va(ib:ie,jb:je+1,kb:ke),v3d(ib:ie,jb:je+1,kb:ke),vten(ib:ie,jb:je+1,kb:ke))
  allocate(rrw(ib:ie,jb:je,kb:ke+1),wa(ib:ie,jb:je,kb:ke+1),w3d(ib:ie,jb:je,kb:ke+1),wten(ib:ie,jb:je,kb:ke+1))
  allocate(ppi(ib:ie,jb:je,kb:ke),pp3d(ib:ie,jb:je,kb:ke),piadv(ib:ie,jb:je,kb:ke),ppten(ib:ie,jb:je,kb:ke))
  allocate(ppx(ib:ie,jb:je,kb:ke),thv(ib:ie,jb:je,kb:ke),ppterm(ib:ie,jb:je,kb:ke))
  u0=0; v0=0
  allocate(bndy(1,1,1),kbdy(1,1),reqs_p(1),pw(1,1),ps(1,1))
  bndy=.false.; kbdy=0; reqs_p=0; pw=0; ps=0

  call rd('uh',uh,size(uh)); call rd('uf',uf,size(uf)); call rd('vh',vh,size(vh)); call rd('vf',vf,size(vf))
  call rd('mh',mh,size(mh)); call rd('mf',mf,size(mf)); call rd('c1',c1,size(c1)); call rd('c2',c2,size(c2))
  call rd('rr0',rr0,size(rr0)); call rd('rf0',rf0,size(rf0))
  call rd('dtu0',dtu0,size(dtu0)); call rd('dtv0',dtv0,size(dtv0))
  call rd('dtu',dtu,size(dtu)); call rd('dtv',dtv,size(dtv))
  call rd('ua',ua,size(ua)); call rd('u3d',u3d,size(u3d)); call rd('uten',uten,size(uten))
  call rd('va',va,size(va)); call rd('v3d',v3d,size(v3d)); call rd('vten',vten,size(vten))
  call rd('wa',wa,size(wa)); call rd('w3d',w3d,size(w3d)); call rd('wten',wten,size(wten))
  call rd('rru',rru,size(rru)); call rd('rrv',rrv,size(rrv)); call rd('rrw',rrw,size(rrw))
  call rd('ppi',ppi,size(ppi)); call rd('pp3d',pp3d,size(pp3d)); call rd('piadv',piadv,size(piadv))
  call rd('ppten',ppten,size(ppten)); call rd('ppx',ppx,size(ppx))
  call rd('thv',thv,size(thv)); call rd('ppterm',ppterm,size(ppterm))
  call rd('ppd',ppd,size(ppd)); call rd('fpk',fpk,size(fpk)); call rd('qk',qk,size(qk))
  call rd('pk1',pk1,size(pk1)); call rd('pk2',pk2,size(pk2)); call rd('mm',mm,size(mm))
  call rd('dum1',dum1,size(dum1)); call rd('dum2',dum2,size(dum2))

  call sound( dt,xh,arh1,arh2,uh,ruh,xf,uf,yh,vh,rvh,yf,vf,     &
              rds,sigma,rdsf,sigmaf,zh,mh,rmh,c1,c2,mf,zf,      &
              pi0,rho0,rr0,rf0,rrf0,th0,rth0,zs,                &
              gz,gz,gz,gz,gz,gz,                                &
              dzdx,dzdx,gx,gx,gx,gx,                            &
              radbcw,radbce,radbcs,radbcn,dtu,dtu0,dtv,dtv0,    &
              dum1,dum2,dum3,dum4,mm  ,bk  ,                    &
              ppd ,fpk ,qk ,pk1,pk2,wtens,                      &
              u0,rru,ua,u3d,uten,                               &
              v0,rrv,va,v3d,vten,                               &
              rrw,wa,w3d,wten,                                  &
              ppi,pp3d,piadv,ppten,ppx,                         &
              thv,ppterm,nrk,dt,rtime,mtime,get_time_avg,       &
              bndy,kbdy,                                        &
              pw,pw,pw,pw,ps,ps,ps,ps,reqs_p)

  call wr('u3d',u3d,size(u3d)); call wr('v3d',v3d,size(v3d)); call wr('w3d',w3d,size(w3d))
  call wr('pp3d',pp3d,size(pp3d)); call wr('ppd',ppd,size(ppd)); call wr('ppx',ppx,size(ppx))
  call wr('rru',rru,size(rru)); call wr('rrv',rrv,size(rrv)); call wr('rrw',rrw,size(rrw))
  call wr('dtu',dtu,size(dtu)); call wr('dtv',dtv,size(dtv))
  call wr('fpk',fpk,size(fpk)); call wr('qk',qk,size(qk)); call wr('pk1',pk1,size(pk1))
  call wr('pk2',pk2,size(pk2)); call wr('mm',mm,size(mm))
  call wr('dum1',dum1,size(dum1)); call wr('dum2',dum2,size(dum2))
end program drv_sound
