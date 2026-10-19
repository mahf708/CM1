! bind(C) shim around base (src/base.F) for the native-initialization tests:
! every isnd / iwnd of base.F on a grid supplied by the caller (serial, no
! terrain, lateral boundaries wbc..nbc). The caller passes the namelist
! values base.F reads (iparams / rparams below) and the grid arrays; the
! base state comes back on the full Fortran bounds. The testcase forcing
! profiles base.F also sets (wprof, ug, vg, ...) are computed and dropped.
!
! iparams: ni nj nk isnd iwnd imoist ptype psolver imove testcase eqtset iice
!          wbc ebc sbc nbc lspgrad
! rparams: dx dy dz umove vmove tsk0 hurr_vg hurr_rotate hurr_rad hurr_vgpl
!          cgs1 cgs2 cgs3 cgt1 cgt2 cgt3 tsmall
subroutine cm1f_base(iparams, rparams, zh_in, mh_in, rmh_in, c1_in, c2_in, zf_in, mf_in, &
                     pi0_o, prs0_o, rho0_o, thv0_o, th0_o, rth0_o, qv0_o, qc0_o, qi0_o,   &
                     rr0_o, rf0_o, rrf0_o, u0_o, v0_o) bind(C, name="cm1f_base")
  use iso_c_binding
  use input
  use constants, only : set_constants
  use base_module, only : base
  use goddard_module, only : T0K, T00K, RT0
  implicit none
  integer(c_int), intent(in) :: iparams(17)
  real(c_double), intent(in) :: rparams(17)
  real(c_double), intent(in) :: zh_in(*), mh_in(*), rmh_in(*), c1_in(*), c2_in(*), zf_in(*), mf_in(*)
  real(c_double), intent(out) :: pi0_o(*), prs0_o(*), rho0_o(*), thv0_o(*), th0_o(*), rth0_o(*)
  real(c_double), intent(out) :: qv0_o(*), qc0_o(*), qi0_o(*), rr0_o(*), rf0_o(*), rrf0_o(*)
  real(c_double), intent(out) :: u0_o(*), v0_o(*)

  real, allocatable, dimension(:,:,:) :: zh,mh,rmh,c1,c2,zf,mf
  real, allocatable, dimension(:,:) :: rho0s,pi0s,prs0s,rth0s
  real, allocatable, dimension(:) :: wprof,ufrc,vfrc,thfrc,qvfrc,ug,vg,dvdr,uavg,vavg,thavg,pavg,ulspg,vlspg
  real, allocatable :: qavg(:,:)
  real, allocatable, dimension(:,:,:) :: pi0,prs0,rho0,thv0,th0,rth0,qv0,qc0,qi0,rr0,rf0,rrf0,t0,rh0,u0,v0,thrd
  integer, allocatable :: reqs(:)
  real, allocatable :: b1(:), b3(:,:,:)
  integer :: nsz, nwz, nuz, nvz

  ni = iparams(1); nj = iparams(2); nk = iparams(3)
  isnd = iparams(4); iwnd = iparams(5); imoist = iparams(6); ptype = iparams(7)
  psolver = iparams(8); imove = iparams(9); testcase = iparams(10)
  eqtset = iparams(11); iice = iparams(12)
  wbc = iparams(13); ebc = iparams(14); sbc = iparams(15); nbc = iparams(16)
  lspgrad = iparams(17)
  dx = rparams(1); dy = rparams(2); dz = rparams(3)
  umove = rparams(4); vmove = rparams(5); tsk0 = rparams(6)
  hurr_vg = rparams(7); hurr_rotate = rparams(8); hurr_rad = rparams(9); hurr_vgpl = rparams(10)
  ! param.F sets hurr_angle for lspgrad = 3 only
  pi_dp = 4.0_dp*atan(1.0_dp)   ! as param.F (kind dp: quad precision)
  hurr_angle = 0.0
  if (lspgrad == 3) hurr_angle = dble(hurr_rotate) * ( pi_dp / 180.0d0 )
  cgs1 = rparams(11); cgs2 = rparams(12); cgs3 = rparams(13)
  cgt1 = rparams(14); cgt2 = rparams(15); cgt3 = rparams(16)
  tsmall = rparams(17)
  call set_constants(testcase)
  ! goddard.F's consat (param.F calls it for ptype = 2 only) sets the
  ! temperatures base.F uses for isnd = 10 with ice; otherwise they stay 0
  if (ptype == 2) then
    T0K = 273.15; T00K = 233.15; RT0 = 1./(T0K-T00K)
  else
    T0K = 0.0; T00K = 0.0; RT0 = 0.0
  endif
  nx = ni; ny = nj; nz = nk
  rdx = 1.0/dx; rdy = 1.0/dy; rdz = 1.0/dz
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibb2 = 1; ibe2 = 1; jbb2 = 1; jbe2 = 1; kbb2 = 1; kbe2 = 1
  imp = 1; jmp = 1; kmp = 1; kmt = 1; rmp = 1; cmp = 1
  numq = 3; nqv = 1
  terrain_flag = .false.; axisymm = 0
  ibw = 0; ibe = 0; ibs = 0; ibn = 0
  if (wbc /= 1) ibw = 1
  if (ebc /= 1) ibe = 1
  if (sbc /= 1) ibs = 1
  if (nbc /= 1) ibn = 1
  dowr = .false.; myid = 0; outfile = 6

  allocate(zh(ib:ie,jb:je,kb:ke),mh(ib:ie,jb:je,kb:ke),rmh(ib:ie,jb:je,kb:ke))
  allocate(c1(ib:ie,jb:je,kb:ke),c2(ib:ie,jb:je,kb:ke))
  allocate(zf(ib:ie,jb:je,kb:ke+1),mf(ib:ie,jb:je,kb:ke+1))
  nsz = size(zh); nwz = size(zf)
  zh = reshape(zh_in(1:nsz), shape(zh)); mh = reshape(mh_in(1:nsz), shape(mh))
  rmh = reshape(rmh_in(1:nsz), shape(rmh)); c1 = reshape(c1_in(1:nsz), shape(c1))
  c2 = reshape(c2_in(1:nsz), shape(c2))
  zf = reshape(zf_in(1:nwz), shape(zf)); mf = reshape(mf_in(1:nwz), shape(mf))

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

  pi0_o(1:nsz) = reshape(pi0, [nsz]); prs0_o(1:nsz) = reshape(prs0, [nsz])
  rho0_o(1:nsz) = reshape(rho0, [nsz]); thv0_o(1:nsz) = reshape(thv0, [nsz])
  th0_o(1:nsz) = reshape(th0, [nsz]); rth0_o(1:nsz) = reshape(rth0, [nsz])
  qv0_o(1:nsz) = reshape(qv0, [nsz]); qc0_o(1:nsz) = reshape(qc0, [nsz])
  qi0_o(1:nsz) = reshape(qi0, [nsz]); rr0_o(1:nsz) = reshape(rr0, [nsz])
  rf0_o(1:nsz) = reshape(rf0, [nsz]); rrf0_o(1:nsz) = reshape(rrf0, [nsz])
  nuz = size(u0); nvz = size(v0)
  u0_o(1:nuz) = reshape(u0, [nuz]); v0_o(1:nvz) = reshape(v0, [nvz])
end subroutine cm1f_base
