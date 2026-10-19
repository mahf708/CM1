! Harness driver for testcase_simple_phys (src/testcase_simple_phys.F):
! simple radiation (testcase 4, 5), the lspgrad=3 mesoscale tendencies, the
! testcase 14 SGP forcing and the prescribed tendencies, on Kessler moisture
! (numq = 3: qv, qc, qr).
! params = [ni, nj, nk, testcase, lspgrad, dx, dy, dz, hurr_rad, hurr_angle, mtime]
program drv_testcase
  use harness_io
  use input
  use constants, only : set_constants
  use simple_phys_module, only : testcase_simple_phys
  implicit none
  real :: params(11)
  real, allocatable, dimension(:,:,:) :: mh, rho0, rr0, rf0, th0, zh, zf, u0, v0, ua, va, tha, &
       uten1, vten1, thten1, dum1, dum2, dum3, dum4, dum5, dum6, frad, thraten
  real, allocatable :: zir(:,:), qa(:,:,:,:), qten(:,:,:,:)
  real, allocatable, dimension(:) :: ufrc, vfrc, thfrc, qvfrc, ug, vg, dvdr, uavg, vavg, thavg, &
       ruh, ruf, rvh, rvf
  real, allocatable :: qavg(:,:)
  double precision, allocatable :: cavg(:,:)
  double precision :: mt

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  testcase = nint(params(4)); lspgrad = nint(params(5))
  dx = params(6); dy = params(7); dz = params(8); rdz = 1.0/dz
  hurr_rad = params(9); hurr_angle = params(10); mt = params(11)
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  ibr = ib; ier = ie; jbr = jb; jer = je; kbr = kb; ker = ke
  numq = 3; nqv = 1; nql1 = 2; nql2 = 3; imoist = 1
  minx = 0.0; maxx = ni*dx; miny = 0.0; maxy = nj*dy
  myid = 0; timestats = 0
  call set_constants(testcase)
  allocate(mh(ib:ie,jb:je,kb:ke), rho0(ib:ie,jb:je,kb:ke), rr0(ib:ie,jb:je,kb:ke), rf0(ib:ie,jb:je,kb:ke), &
       th0(ib:ie,jb:je,kb:ke), zh(ib:ie,jb:je,kb:ke), zf(ib:ie,jb:je,kb:ke+1), u0(ib:ie+1,jb:je,kb:ke), &
       v0(ib:ie,jb:je+1,kb:ke), ua(ib:ie+1,jb:je,kb:ke), va(ib:ie,jb:je+1,kb:ke), tha(ib:ie,jb:je,kb:ke), &
       uten1(ib:ie+1,jb:je,kb:ke), vten1(ib:ie,jb:je+1,kb:ke), thten1(ib:ie,jb:je,kb:ke), &
       dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke), dum4(ib:ie,jb:je,kb:ke), &
       dum5(ib:ie,jb:je,kb:ke), dum6(ib:ie,jb:je,kb:ke), frad(ib:ie,jb:je,kb:ke), thraten(ib:ie,jb:je,kb:ke), &
       zir(ib:ie,jb:je), qa(ib:ie,jb:je,kb:ke,numq), qten(ib:ie,jb:je,kb:ke,numq))
  allocate(ufrc(kb:ke), vfrc(kb:ke), thfrc(kb:ke), qvfrc(kb:ke), ug(kb:ke), vg(kb:ke), dvdr(kb:ke), &
       uavg(kb:ke), vavg(kb:ke), thavg(kb:ke), qavg(kb:ke,numq), cavg(kb:ke,3+numq), &
       ruh(ib:ie), ruf(ib:ie+1), rvh(jb:je), rvf(jb:je+1))
  rr0 = 0; rf0 = 0; u0 = 0; v0 = 0; frad = 0; thraten = 0; zir = 0
  ! simplerad's scratch: zero, as in cm1k / cm1x (CM1 never sets ql, qt at k = nk+1)
  dum1 = 0; dum2 = 0; dum3 = 0; dum4 = 0; dum5 = 0; dum6 = 0
  uavg = 0; vavg = 0; thavg = 0; qavg = 0; cavg = 0
  call rd('mh', mh, size(mh)); call rd('rho0', rho0, size(rho0)); call rd('th0', th0, size(th0))
  call rd('zh', zh, size(zh)); call rd('zf', zf, size(zf))
  call rd('ua', ua, size(ua)); call rd('va', va, size(va)); call rd('tha', tha, size(tha))
  call rd('qa', qa, size(qa))
  call rd('uten1', uten1, size(uten1)); call rd('vten1', vten1, size(vten1))
  call rd('thten1', thten1, size(thten1)); call rd('qten', qten, size(qten))
  call rd('ufrc', ufrc, size(ufrc)); call rd('vfrc', vfrc, size(vfrc))
  call rd('thfrc', thfrc, size(thfrc)); call rd('qvfrc', qvfrc, size(qvfrc))
  call rd('ug', ug, size(ug)); call rd('vg', vg, size(vg)); call rd('dvdr', dvdr, size(dvdr))
  call rd('ruh', ruh, size(ruh)); call rd('ruf', ruf, size(ruf))
  call rd('rvh', rvh, size(rvh)); call rd('rvf', rvf, size(rvf))
  call testcase_simple_phys(mh,rho0,rr0,rf0,th0,u0,v0,     &
       zh,zf,dum1,dum2,dum3,dum4,dum5,dum6,              &
       ufrc,vfrc,thfrc,qvfrc,ug,vg,dvdr,                 &
       uavg,vavg,thavg,qavg,cavg,                        &
       ua,va,tha,qa,uten1,vten1,thten1,qten,             &
       frad,thraten,zir,ruh,ruf,rvh,rvf,mt)
  call wr('uten1', uten1, size(uten1)); call wr('vten1', vten1, size(vten1))
  call wr('thten1', thten1, size(thten1)); call wr('qten', qten, size(qten))
  call wr('ufrc', ufrc, size(ufrc)); call wr('vfrc', vfrc, size(vfrc))
  call wr('thfrc', thfrc, size(thfrc)); call wr('qvfrc', qvfrc, size(qvfrc))
  call wr('frad', frad, size(frad)); call wr('thraten', thraten, size(thraten))
  call wr('zir', zir, size(zir))
end program drv_testcase
