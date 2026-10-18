! Harness driver for the terrain-following grid: init_terrain (itern = 1..3,
! which calls adv_routines' zsgrad and bc2d), bcwsfc, and the terrain
! branches of ssopenbcw/e/s/n (bc.F), non-MPI.
! params = [ni, nj, nk, dx, dy, itern, wbc(=ebc), sbc(=nbc), maxx, maxy,
!           maxz, dts, cgs1, cgs2, cgs3, cgt1, cgt2, cgt3, hadvordrs, hadvordrv]
program drv_terrain
  use harness_io
  use input
  use constants, only : set_constants
  use bc_module
  use init_terrain_module, only : init_terrain
  implicit none
  real :: p(20), dts
  real, allocatable, dimension(:) :: xh,uh,xf,uf,yh,vh,yf,vf,rds,sigma,rdsf,sigmaf
  real, allocatable, dimension(:,:) :: zs,gz,rgz,gzu,rgzu,gzv,rgzv,dzdx,dzdy
  real, allocatable, dimension(:,:) :: radbcw,radbce,radbcs,radbcn
  real, allocatable, dimension(:,:,:) :: zh,zf,gx,gxu,gy,gyv,u,v,w,u3d,v3d,uten,vten,dum1
  integer :: reqs(8)
  real :: k8(8)
  real, allocatable, dimension(:,:,:) :: c3a,c3b,c3c,c3d,c3e,c3f
  real, allocatable, dimension(:,:) :: c2a,c2b

  call set_constants(0)
  call rd('params', p, size(p))
  ni = nint(p(1)); nj = nint(p(2)); nk = nint(p(3))
  dx = p(4); dy = p(5); itern = nint(p(6))
  wbc = nint(p(7)); ebc = wbc; sbc = nint(p(8)); nbc = sbc
  maxx = p(9); maxy = p(10); maxz = p(11); dts = p(12)
  cgs1 = p(13); cgs2 = p(14); cgs3 = p(15); cgt1 = p(16); cgt2 = p(17); cgt3 = p(18)
  hadvordrs = nint(p(19)); hadvordrv = nint(p(20))
  rdx = 1.0/dx; rdy = 1.0/dy
  nx = ni; ny = nj; nz = nk
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  itb = ib; ite = ie; jtb = jb; jte = je; ktb = kb; kte = ke
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc.ne.1) ibw = 1
  if (ebc.ne.1) ibe = 1
  if (sbc.ne.1) ibs = 1
  if (nbc.ne.1) ibn = 1
  terrain_flag = .true.; axisymm = 0; timestats = 0; dowr = .false.; myid = 0
  cmp = ngxy; imp = ni; jmp = nj; kmp = max(2,nk); kmt = 1; rmp = 8

  allocate(xh(ib:ie),uh(ib:ie),xf(ib:ie+1),uf(ib:ie+1),yh(jb:je),vh(jb:je),yf(jb:je+1),vf(jb:je+1))
  allocate(rds(kb:ke),sigma(kb:ke),rdsf(kb:ke+1),sigmaf(kb:ke+1))
  allocate(zs(ib:ie,jb:je),gz(ib:ie,jb:je),rgz(ib:ie,jb:je),gzu(ib:ie,jb:je),rgzu(ib:ie,jb:je), &
           gzv(ib:ie,jb:je),rgzv(ib:ie,jb:je),dzdx(ib:ie,jb:je),dzdy(ib:ie,jb:je))
  allocate(zh(ib:ie,jb:je,kb:ke),zf(ib:ie,jb:je,kb:ke+1))
  allocate(gx(ib:ie,jb:je,kb:ke),gxu(ib:ie,jb:je,kb:ke),gy(ib:ie,jb:je,kb:ke),gyv(ib:ie,jb:je,kb:ke))
  allocate(u(ib:ie+1,jb:je,kb:ke),v(ib:ie,jb:je+1,kb:ke),w(ib:ie,jb:je,kb:ke+1))
  allocate(u3d(ib:ie+1,jb:je,kb:ke),uten(ib:ie+1,jb:je,kb:ke),v3d(ib:ie,jb:je+1,kb:ke),vten(ib:ie,jb:je+1,kb:ke))
  allocate(dum1(ib:ie,jb:je,kb:ke+1))
  allocate(radbcw(jb:je,kb:ke),radbce(jb:je,kb:ke),radbcs(ib:ie,kb:ke),radbcn(ib:ie,kb:ke))
  allocate(c3a(cmp,jmp+1,kmp),c3b(imp+1,cmp,kmp),c3c(cmp,jmp+1,kmp),c3d(imp+1,cmp,kmp))
  allocate(c3e(cmp,jmp+1,kmp),c3f(imp+1,cmp,kmp),c2a(cmp,jmp),c2b(imp,cmp))

  ! param.F's initial values for points init_terrain does not set
  zs=0; gz=1; rgz=1; gzu=1; rgzu=1; gzv=1; rgzv=1; dzdx=0; dzdy=0
  gx=0; gxu=0; gy=0; gyv=0; zh=0; zf=0; dum1=0

  call rd('xh',xh,size(xh)); call rd('uh',uh,size(uh)); call rd('xf',xf,size(xf)); call rd('uf',uf,size(uf))
  call rd('yh',yh,size(yh)); call rd('vh',vh,size(vh)); call rd('yf',yf,size(yf)); call rd('vf',vf,size(vf))
  call rd('rds',rds,size(rds)); call rd('sigma',sigma,size(sigma))
  call rd('rdsf',rdsf,size(rdsf)); call rd('sigmaf',sigmaf,size(sigmaf))

  call init_terrain(xh,uh,xf,uf,yh,vh,yf,vf,rds,sigma,rdsf,sigmaf,  &
                    zh,zf,zs,gz,rgz,gzu,rgzu,gzv,rgzv,              &
                    dzdx,dzdy,gx,gxu,gy,gyv,                        &
                    reqs,reqs,reqs,reqs,                            &
                    k8,k8,k8,k8,k8,k8,k8,k8,                        &
                    c3a,c3a,c3a,c3a,c3b,c3b,c3b,c3b,                &
                    c3a,c3a,c3a,c3a,c3d,c3d,c3d,c3d,                &
                    c3e,c3e,c3e,c3e,c3f,c3f,c3f,c3f,                &
                    c2a,c2a,c2a,c2a,c2b,c2b,c2b,c2b)
  call wr('zs',zs,size(zs)); call wr('zh',zh,size(zh)); call wr('zf',zf,size(zf))
  call wr('gz',gz,size(gz)); call wr('rgz',rgz,size(rgz))
  call wr('gzu',gzu,size(gzu)); call wr('rgzu',rgzu,size(rgzu))
  call wr('gzv',gzv,size(gzv)); call wr('rgzv',rgzv,size(rgzv))
  call wr('dzdx',dzdx,size(dzdx)); call wr('dzdy',dzdy,size(dzdy))
  call wr('gx',gx,size(gx)); call wr('gxu',gxu,size(gxu))
  call wr('gy',gy,size(gy)); call wr('gyv',gyv,size(gyv))
  call wr('rds',rds,size(rds)); call wr('rdsf',rdsf,size(rdsf))

  ! kinematic lower boundary condition
  call rd('u',u,size(u)); call rd('v',v,size(v)); call rd('w',w,size(w))
  call bcwsfc(gz,dzdx,dzdy,u,v,w)
  call wr('w',w,size(w))

  ! small-step open boundaries (terrain branch)
  call rd('u3d',u3d,size(u3d)); call rd('uten',uten,size(uten))
  call rd('v3d',v3d,size(v3d)); call rd('vten',vten,size(vten))
  call rd('radbcw',radbcw,size(radbcw)); call rd('radbce',radbce,size(radbce))
  call rd('radbcs',radbcs,size(radbcs)); call rd('radbcn',radbcn,size(radbcn))
  if (wbc.eq.2) then
    call ssopenbcw(uh,rds,sigma,rdsf,sigmaf,gz,rgzu,gx,radbcw,dum1,u3d,uten,dts)
    call ssopenbce(uh,rds,sigma,rdsf,sigmaf,gz,rgzu,gx,radbce,dum1,u3d,uten,dts)
  endif
  if (sbc.eq.2) then
    call ssopenbcs(vh,rds,sigma,rdsf,sigmaf,gz,rgzv,gy,radbcs,dum1,v3d,vten,dts)
    call ssopenbcn(vh,rds,sigma,rdsf,sigmaf,gz,rgzv,gy,radbcn,dum1,v3d,vten,dts)
  endif
  call wr('uo',u3d,size(u3d)); call wr('vo',v3d,size(v3d))
end program drv_terrain
