! Harness driver for the radopt=2 (RRTMG) branch of radiation_driver
! (src/radiation_driver.F) with Kessler (ptype=1) moisture: rrtmg_lwinit /
! rrtmg_swinit as init_physics.F calls them, then one radiation_driver call.
! params = [ni, nj, nk, dz, dt, dtrad, mtime, year, month, day, hour, minute,
!           second, ctrlat, ctrlon, eqtset, cgs1, cgs2, cgs3, cgt1, cgt2, cgt3]
program drv_radiation_driver
  use harness_io
  use input
  use constants, only : set_constants
  use radtrns3d_module, only : julday, nrad2d
  use radiation_module, only : radiation_driver
  use module_ra_rrtmg_lw, only : rrtmg_lwinit
  use module_ra_rrtmg_sw, only : rrtmg_swinit
  implicit none
  real :: params(22), dt, p_top, b1(1)
  double precision :: mtime, radtim
  integer :: nstep, rbufsz
  real, allocatable :: xh(:), yh(:), xf(:), yf(:)
  real, allocatable, dimension(:,:,:) :: zf, rmh, c1, c2, swten, lwten, swtenc, lwtenc, &
       cldfra, o30, thraten, effc, effi, effs, effr, effg, effis, dum1, dum2, dum3, dum4, &
       dum5, dum6, dum7, dum8, dum9, dum10, dum11, prs0, pi0, th0, prs, ppi, tha, rho, &
       qc_bl, qi_bl, cldfra_bl
  integer, allocatable :: cldfra1_flag(:,:,:)
  real, allocatable, dimension(:,:) :: radsw, rnflx, radswnet, radlwin, dsr, olr, &
       lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc, &
       swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc, lwcf, swcf, coszr, &
       xice, xsnow, xlat, xlong, coszen, swddir, swddni, swddif, hrang, &
       rth0s, prs0s, rho0s, tsk, albd, glw, gsw, emiss, xland
  real, allocatable :: rad2d(:,:,:), qa(:,:,:,:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dz = params(4); dt = params(5); dtrad = params(6); mtime = params(7)
  year = nint(params(8)); month = nint(params(9)); day = nint(params(10))
  hour = nint(params(11)); minute = nint(params(12)); second = nint(params(13))
  ctrlat = params(14); ctrlon = params(15); eqtset = nint(params(16))
  cgs1 = params(17); cgs2 = params(18); cgs3 = params(19)
  cgt1 = params(20); cgt2 = params(21); cgt3 = params(22)
  ngxy = 3; ngz = 1
  ib = 1-ngxy; ie = ni+ngxy; jb = 1-ngxy; je = nj+ngxy; kb = 1-ngz; ke = nk+ngz
  ibm = ib; iem = ie; jbm = jb; jem = je; kbm = kb; kem = ke
  ibr = ib; ier = ie; jbr = jb; jer = je; kbr = kb; ker = ke
  ibl = ib; iel = ie; jbl = jb; jel = je
  ibmynn = 1; iemynn = 1; jbmynn = 1; jemynn = 1; kbmynn = 1; kemynn = 1
  numq = 3; nqv = 1; nqc = 2; nqr = 3; nqi = 0; nqs = 0; nqg = 0; nqi2 = 0; nqi3 = 0
  imoist = 1; ptype = 1; radopt = 2; testcase = 0; idopbl = .false.; ipbl = 0
  myid = 0; timestats = 0
  call set_constants(0)
  call julday(year, month, day, jday)
  rbufsz = 1; nstep = 1; radtim = mtime

  allocate(xh(ib:ie), yh(jb:je), xf(ib:ie+1), yf(jb:je+1))
  allocate(zf(ib:ie,jb:je,kb:ke+1))
  allocate(rmh(ib:ie,jb:je,kb:ke), c1(ib:ie,jb:je,kb:ke), c2(ib:ie,jb:je,kb:ke), &
       prs0(ib:ie,jb:je,kb:ke), pi0(ib:ie,jb:je,kb:ke), th0(ib:ie,jb:je,kb:ke), &
       prs(ib:ie,jb:je,kb:ke), ppi(ib:ie,jb:je,kb:ke), tha(ib:ie,jb:je,kb:ke), &
       rho(ib:ie,jb:je,kb:ke))
  allocate(dum1(ib:ie,jb:je,kb:ke), dum2(ib:ie,jb:je,kb:ke), dum3(ib:ie,jb:je,kb:ke), &
       dum4(ib:ie,jb:je,kb:ke), dum5(ib:ie,jb:je,kb:ke), dum6(ib:ie,jb:je,kb:ke), &
       dum7(ib:ie,jb:je,kb:ke), dum8(ib:ie,jb:je,kb:ke), dum9(ib:ie,jb:je,kb:ke), &
       dum10(ib:ie,jb:je,kb:ke), dum11(ib:ie,jb:je,kb:ke))
  allocate(swten(ibr:ier,jbr:jer,kbr:ker), lwten(ibr:ier,jbr:jer,kbr:ker), &
       swtenc(ibr:ier,jbr:jer,kbr:ker), lwtenc(ibr:ier,jbr:jer,kbr:ker), &
       cldfra(ibr:ier,jbr:jer,kbr:ker), o30(ibr:ier,jbr:jer,kbr:ker), &
       thraten(ibr:ier,jbr:jer,kbr:ker), cldfra1_flag(ibr:ier,jbr:jer,kbr:ker))
  allocate(effc(ibr:ier,jbr:jer,kbr:ker), effi(ibr:ier,jbr:jer,kbr:ker), &
       effs(ibr:ier,jbr:jer,kbr:ker), effr(ibr:ier,jbr:jer,kbr:ker), &
       effg(ibr:ier,jbr:jer,kbr:ker), effis(ibr:ier,jbr:jer,kbr:ker))
  allocate(qc_bl(1,1,1), qi_bl(1,1,1), cldfra_bl(1,1,1))
  allocate(radsw(ni,nj), rnflx(ni,nj), radswnet(ni,nj), radlwin(ni,nj), dsr(ni,nj), &
       olr(ni,nj), rad2d(ni,nj,nrad2d))
  allocate(lwupt(ib:ie,jb:je), lwuptc(ib:ie,jb:je), lwdnt(ib:ie,jb:je), lwdntc(ib:ie,jb:je), &
       lwupb(ib:ie,jb:je), lwupbc(ib:ie,jb:je), lwdnb(ib:ie,jb:je), lwdnbc(ib:ie,jb:je), &
       swupt(ib:ie,jb:je), swuptc(ib:ie,jb:je), swdnt(ib:ie,jb:je), swdntc(ib:ie,jb:je), &
       swupb(ib:ie,jb:je), swupbc(ib:ie,jb:je), swdnb(ib:ie,jb:je), swdnbc(ib:ie,jb:je), &
       lwcf(ib:ie,jb:je), swcf(ib:ie,jb:je), coszr(ib:ie,jb:je), xice(ib:ie,jb:je), &
       xsnow(ib:ie,jb:je), xlat(ib:ie,jb:je), xlong(ib:ie,jb:je), coszen(ib:ie,jb:je), &
       swddir(ib:ie,jb:je), swddni(ib:ie,jb:je), swddif(ib:ie,jb:je), hrang(ib:ie,jb:je), &
       rth0s(ib:ie,jb:je), prs0s(ib:ie,jb:je), rho0s(ib:ie,jb:je), tsk(ib:ie,jb:je), &
       albd(ib:ie,jb:je), glw(ib:ie,jb:je), gsw(ib:ie,jb:je), emiss(ib:ie,jb:je), &
       xland(ib:ie,jb:je))
  allocate(qa(ibm:iem,jbm:jem,kbm:kem,numq))
  xh = 0; yh = 0; xf = 0; yf = 0; zf = 0
  swten = 0; lwten = 0; lwtenc = 0; cldfra = 0; o30 = 0; thraten = 0
  effc = 0; effi = 0; effs = 0; effr = 0; effg = 0; effis = 0
  dum1 = 0; dum2 = 0; dum3 = 0; dum4 = 0; dum5 = 0; dum6 = 0; dum7 = 0; dum8 = 0
  dum9 = 0; dum10 = 0; dum11 = 0
  lwupt = 0; lwuptc = 0; lwdnt = 0; lwdntc = 0; lwupb = 0; lwupbc = 0; lwdnb = 0; lwdnbc = 0
  swupt = 0; swuptc = 0; swdnt = 0; swdntc = 0; swupb = 0; swupbc = 0; swdnb = 0; swdnbc = 0
  lwcf = 0; swcf = 0; coszr = 0; xice = 0; xsnow = 0; xlat = 0; xlong = 0; coszen = 0
  swddir = 0; swddni = 0; swddif = 0; hrang = 0; glw = 0; gsw = 0
  rth0s = 0; prs0s = 0; rho0s = 0; qc_bl = 0; qi_bl = 0; cldfra_bl = 0

  call rd('rmh', rmh, size(rmh)); call rd('c1', c1, size(c1)); call rd('c2', c2, size(c2))
  call rd('prs0', prs0, size(prs0)); call rd('pi0', pi0, size(pi0)); call rd('th0', th0, size(th0))
  call rd('prs', prs, size(prs)); call rd('ppi', ppi, size(ppi)); call rd('tha', tha, size(tha))
  call rd('rho', rho, size(rho)); call rd('qa', qa, size(qa))
  call rd('tsk', tsk, size(tsk)); call rd('albd', albd, size(albd))
  call rd('emiss', emiss, size(emiss)); call rd('xland', xland, size(xland))
  call rd('swtenc', swtenc, size(swtenc))

  ! init_physics.F
  p_top = cgt1*prs0(1,1,nk)+cgt2*prs0(1,1,nk-1)+cgt3*prs0(1,1,nk-2)
  call rrtmg_lwinit(p_top=p_top, allowed_to_read=.true., myid=myid,          &
       ids=1, ide=ni+1, jds=1, jde=nj+1, kds=1, kde=nk+1,                    &
       ims=ib, ime=ie, jms=jb, jme=je, kms=kb, kme=ke,                       &
       its=1, ite=ni, jts=1, jte=nj, kts=1, kte=nk)
  call rrtmg_swinit(allowed_to_read=.true., myid=myid,                       &
       ids=1, ide=ni+1, jds=1, jde=nj+1, kds=1, kde=nk+1,                    &
       ims=ib, ime=ie, jms=jb, jme=je, kms=kb, kme=ke,                       &
       its=1, ite=ni, jts=1, jte=nj, kts=1, kte=nk)

  call radiation_driver(mtime,radtim,dt,rbufsz,xh,yh,xf,yf,zf,rmh,c1,c2,     &
       swten,lwten,swtenc,lwtenc,cldfra,o30,thraten,                         &
       radsw,rnflx,radswnet,radlwin,dsr,olr,rad2d,                           &
       effc,effi,effs,effr,effg,effis,                                       &
       lwupt,lwuptc,lwdnt,lwdntc,lwupb,lwupbc,lwdnb,lwdnbc,                  &
       swupt,swuptc,swdnt,swdntc,swupb,swupbc,swdnb,swdnbc,                  &
       lwcf,swcf,coszr,                                                      &
       xice,xsnow,xlat,xlong,coszen,swddir,swddni,swddif,hrang,              &
       cldfra1_flag,dum1,dum2,dum3,dum4,dum5,dum6,dum7,dum8,dum9,dum10,dum11, &
       prs0,pi0,th0,prs,ppi,tha,rho,qa,                                      &
       rth0s,prs0s,rho0s,tsk,albd,glw,gsw,emiss,xland,nstep,                 &
       qc_bl,qi_bl,cldfra_bl)

  call wr('swten', swten, size(swten)); call wr('lwten', lwten, size(lwten))
  call wr('swtenc', swtenc, size(swtenc)); call wr('lwtenc', lwtenc, size(lwtenc))
  call wr('cldfra', cldfra, size(cldfra)); call wr('thraten', thraten, size(thraten))
  call wr('glw', glw, size(glw)); call wr('gsw', gsw, size(gsw)); call wr('olr', olr, size(olr))
  call wr('lwcf', lwcf, size(lwcf)); call wr('swcf', swcf, size(swcf))
  call wr('lwupt', lwupt, size(lwupt)); call wr('lwdnb', lwdnb, size(lwdnb))
  call wr('swupt', swupt, size(swupt)); call wr('swdnb', swdnb, size(swdnb))
  call wr('swddir', swddir, size(swddir)); call wr('swddif', swddif, size(swddif))
  call wr('coszen', coszen, size(coszen)); call wr('hrang', hrang, size(hrang))
  b1 = rad_solcon; call wr('solcon', b1, 1)
  b1 = rad_declin; call wr('declin', b1, 1)
  b1 = rad_jday; call wr('julian', b1, 1)
  b1 = real(jday); call wr('jday', b1, 1)
end program drv_radiation_driver
