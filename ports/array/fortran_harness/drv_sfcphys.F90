! Harness driver for getcecd (src/sfcphys.F): the neutral surface layer of
! sfcmodel=1 (isfcflx=0, no use_avg_sfc).
! params = [ni, nj, set_znt, set_ust, cecd, cnst_ust, cnstcd]
program drv_sfcphys
  use harness_io
  use input
  use constants, only : set_constants
  use sfcphys_module, only : getcecd
  implicit none
  real :: params(7)
  real, allocatable :: xh(:)
  real, allocatable, dimension(:,:) :: u1, v1, s1, za, u10, v10, s10, xland, znt, ust, &
       cd, ch, cq
  double precision :: avgsfcu, avgsfcv, avgsfcs, avgsfct

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2))
  set_znt = nint(params(3)); set_ust = nint(params(4)); cecd = nint(params(5))
  cnst_ust = params(6); cnstcd = params(7)
  ngxy = 3
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy
  ibl = ib; iel = ie; jbl = jb; jel = je
  use_avg_sfc = .false.; isfcflx = 0; timestats = 0; myid = 0
  call set_constants(0)
  allocate(xh(ib:ie))
  allocate(u1(ib:ie,jb:je), v1(ib:ie,jb:je), s1(ib:ie,jb:je), za(ib:ie,jb:je), &
       u10(ib:ie,jb:je), v10(ib:ie,jb:je), s10(ib:ie,jb:je), xland(ib:ie,jb:je), &
       znt(ib:ie,jb:je), ust(ib:ie,jb:je), cd(ib:ie,jb:je), ch(ib:ie,jb:je), cq(ib:ie,jb:je))
  xh = 0; u10 = 0; v10 = 0; s10 = 0; ust = 0; cd = 0; ch = 0; cq = 0
  avgsfcu = 0; avgsfcv = 0; avgsfcs = 0; avgsfct = 0
  call rd('u1', u1, size(u1)); call rd('v1', v1, size(v1)); call rd('s1', s1, size(s1))
  call rd('za', za, size(za)); call rd('xland', xland, size(xland)); call rd('znt', znt, size(znt))
  call getcecd(xh,u1,v1,s1,za,u10,v10,s10,xland,znt,ust,cd,ch,cq,avgsfcu,avgsfcv,avgsfcs,avgsfct)
  call wr('cd', cd, size(cd)); call wr('znt', znt, size(znt)); call wr('ust', ust, size(ust))
  call wr('u10', u10, size(u10)); call wr('v10', v10, size(v10)); call wr('s10', s10, size(s10))
end program drv_sfcphys
