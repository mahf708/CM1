! Harness driver for bc.F lateral boundary routines (non-MPI) and the
! open-boundary helpers (radbcew4/radbcns4, ssopenbc*), plus calccflquick.
! params = [ni, nj, nk, wbc, ebc, sbc, nbc, dx, dy, dz, dt, dts]
program drv_bc
  use harness_io
  use input
  use constants, only : set_constants
  use bc_module
  use misclibs, only : calccflquick
  implicit none
  real :: params(12), dt, dts, cfl(1)
  integer :: reqc
  real, allocatable, dimension(:,:,:) :: s,p,u,v,w0,w1,ua,u3d,va,v3d,uten,vten,mh,dum1
  real, allocatable, dimension(:,:,:) :: uo,vo,gx
  real, allocatable, dimension(:,:) :: s2d,radbcw,radbce,radbcs,radbcn,gz
  real, allocatable, dimension(:) :: ruf,rvf,uh,vh,rds,sigma,rdsf,sigmaf

  call set_constants(0)
  call rd('params', params, 12)
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  wbc = nint(params(4)); ebc = nint(params(5)); sbc = nint(params(6)); nbc = nint(params(7))
  dx = params(8); dy = params(9); dz = params(10); dt = params(11); dts = params(12)
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  nx = ni; ny = nj; nz = nk
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  itb = 1; ite = 1; jtb = 1; jte = 1; ktb = 1; kte = 1
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc.ne.1) ibw = 1
  if (ebc.ne.1) ibe = 1
  if (sbc.ne.1) ibs = 1
  if (nbc.ne.1) ibn = 1
  terrain_flag = .false.; axisymm = 0; timestats = 0

  allocate(s(ib:ie,jb:je,kb:ke), p(ib:ie,jb:je,kb:ke), mh(ib:ie,jb:je,kb:ke), dum1(ib:ie,jb:je,kb:ke))
  allocate(u(ib:ie+1,jb:je,kb:ke), ua(ib:ie+1,jb:je,kb:ke), u3d(ib:ie+1,jb:je,kb:ke), uten(ib:ie+1,jb:je,kb:ke), uo(ib:ie+1,jb:je,kb:ke))
  allocate(v(ib:ie,jb:je+1,kb:ke), va(ib:ie,jb:je+1,kb:ke), v3d(ib:ie,jb:je+1,kb:ke), vten(ib:ie,jb:je+1,kb:ke), vo(ib:ie,jb:je+1,kb:ke))
  allocate(w0(ib:ie,jb:je,kb:ke+1), w1(ib:ie,jb:je,kb:ke+1))
  allocate(s2d(ib:ie,jb:je), radbcw(jb:je,kb:ke), radbce(jb:je,kb:ke), radbcs(ib:ie,kb:ke), radbcn(ib:ie,kb:ke))
  allocate(ruf(ib:ie+1), rvf(jb:je+1), uh(ib:ie), vh(jb:je))
  allocate(rds(kb:ke), sigma(kb:ke), rdsf(kb:ke+1), sigmaf(kb:ke+1), gz(1,1), gx(1,1,1))
  rds=0; sigma=0; rdsf=0; sigmaf=0; gz=1; gx=0; dum1=0

  call rd('s',s,size(s)); call rd('p',p,size(p)); call rd('u',u,size(u)); call rd('v',v,size(v))
  call rd('w0',w0,size(w0)); call rd('w1',w1,size(w1)); call rd('s2d',s2d,size(s2d))
  call rd('ua',ua,size(ua)); call rd('u3d',u3d,size(u3d)); call rd('va',va,size(va)); call rd('v3d',v3d,size(v3d))
  call rd('uten',uten,size(uten)); call rd('vten',vten,size(vten)); call rd('mh',mh,size(mh))
  call rd('ruf',ruf,size(ruf)); call rd('rvf',rvf,size(rvf)); call rd('uh',uh,size(uh)); call rd('vh',vh,size(vh))
  call rd('radbcw',radbcw,size(radbcw)); call rd('radbce',radbce,size(radbce))
  call rd('radbcs',radbcs,size(radbcs)); call rd('radbcn',radbcn,size(radbcn))

  call bcs(s); call bcp(p); call bcu(u); call bcv(v); call bcw(w0,0); call bcw(w1,1); call bc2d(s2d)
  call wr('s',s,size(s)); call wr('p',p,size(p)); call wr('u',u,size(u)); call wr('v',v,size(v))
  call wr('w0',w0,size(w0)); call wr('w1',w1,size(w1)); call wr('s2d',s2d,size(s2d))

  ! radiative phase speeds (u1 = ua, u2 = u3d)
  call radbcew4(ruf,radbcw,radbce,ua,u3d,dt)
  call radbcns4(rvf,radbcs,radbcn,va,v3d,dt)
  call wr('radbcw',radbcw,size(radbcw)); call wr('radbce',radbce,size(radbce))
  call wr('radbcs',radbcs,size(radbcs)); call wr('radbcn',radbcn,size(radbcn))

  ! small-step open-boundary velocity updates (with the new phase speeds)
  uo = u3d; vo = v3d
  if (wbc.eq.2) call ssopenbcw(uh,rds,sigma,rdsf,sigmaf,gz,gz,gx,radbcw,dum1,uo,uten,dts)
  if (ebc.eq.2) call ssopenbce(uh,rds,sigma,rdsf,sigmaf,gz,gz,gx,radbce,dum1,uo,uten,dts)
  if (sbc.eq.2) call ssopenbcs(vh,rds,sigma,rdsf,sigmaf,gz,gz,gx,radbcs,dum1,vo,vten,dts)
  if (nbc.eq.2) call ssopenbcn(vh,rds,sigma,rdsf,sigmaf,gz,gz,gx,radbcn,dum1,vo,vten,dts)
  call wr('uo',uo,size(uo)); call wr('vo',vo,size(vo))

  call calccflquick(dt,uh,vh,mh,ua,va,w1,reqc)
  cfl(1) = cflmax
  call wr('cfl',cfl,1)
end program drv_bc
