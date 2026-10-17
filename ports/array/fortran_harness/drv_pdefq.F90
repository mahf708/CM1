! Harness driver for pdefq (src/misclibs.F).
! params = [ni, nj, nk, dx, dy, dz, pdscheme, rmax]
program drv_pdefq
  use harness_io
  use input
  use constants, only : set_constants
  use misclibs, only : pdefq
  implicit none
  real :: params(8), b1(1), rmax
  double precision :: asq
  real, allocatable :: ruh(:), rvh(:), rmh(:,:,:), rho(:,:,:), q(:,:,:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dx = params(4); dy = params(5); dz = params(6); pdscheme = nint(params(7)); rmax = params(8)
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  timestats = 0
  call set_constants(0)
  allocate(ruh(ib:ie), rvh(jb:je), rmh(ib:ie,jb:je,kb:ke), rho(ib:ie,jb:je,kb:ke), q(ib:ie,jb:je,kb:ke))
  call rd('ruh', ruh, size(ruh)); call rd('rvh', rvh, size(rvh)); call rd('rmh', rmh, size(rmh))
  call rd('rho', rho, size(rho)); call rd('q3d', q, size(q))
  call rd('asq', b1, 1); asq = b1(1)
  call pdefq(rmax, asq, ruh, rvh, rmh, rho, q)
  call wr('q3d', q, size(q))
  b1(1) = asq; call wr('asq', b1, 1)
end program drv_pdefq
