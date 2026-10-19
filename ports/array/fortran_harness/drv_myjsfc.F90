! Harness driver for the MYJ surface layer (src/module_sf_myjsfc.F):
! MYJSFCINIT (the psi tables) then MYJSFC on ni x nj columns at once
! (ims:ime = 1:ni, jms:jme = 1:nj, kms:kme = 1:nk+1, kts:kte = 1:nk), as
! turb.F calls it (LOWLYR = 1, ISURBAN = 0, P1000mb = 1e5).
! params = [ni, nj, nk, itimestep, usemyjpbl, isftcflx]
! 3-D inputs arrive as Python [k, j, i] = Fortran (i, j, k), k = 1..nk
! (pint: 1..nk+1); 2-D inputs as (i, j).
program drv_myjsfc
  use harness_io
  use module_sf_myjsfc, only : myjsfc, myjsfcinit
  implicit none
  real :: params(6)
  integer :: ni, nj, nk, itimestep, isftcflx, i, j, k
  logical :: usemyjpbl
  real, allocatable, dimension(:,:,:) :: buf, dz, pmid, pint, th, t, qv, qc, u, v, q2
  real, allocatable, dimension(:,:) :: ht, tsk, qsfc, thz0, qz0, uz0, vz0, xland, ust, znt, &
       z0base, pblh, mavail, rmol, akhs, akms, rib, chs, chs2, cqs2, hfx, qfx, flx_lh, flhc, &
       flqc, qgh, cpm, ct, u10, v10, t02, th02, tshltr, th10, q02, qshltr, q10, pshltr, &
       u10e, v10e, psim, psih, fm, fhh, z0out, rhosfc, xice, ustd, z0d
  integer, allocatable, dimension(:,:) :: lowlyr, ivgtyp

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  itimestep = nint(params(4)); usemyjpbl = nint(params(5)) /= 0; isftcflx = nint(params(6))

  allocate(buf(ni,nj,nk+1))
  allocate(dz(ni,nk+1,nj), pmid(ni,nk+1,nj), pint(ni,nk+1,nj), th(ni,nk+1,nj), t(ni,nk+1,nj), &
       qv(ni,nk+1,nj), qc(ni,nk+1,nj), u(ni,nk+1,nj), v(ni,nk+1,nj), q2(ni,nk+1,nj))
  allocate(ht(ni,nj), tsk(ni,nj), qsfc(ni,nj), thz0(ni,nj), qz0(ni,nj), uz0(ni,nj), vz0(ni,nj), &
       xland(ni,nj), ust(ni,nj), znt(ni,nj), z0base(ni,nj), pblh(ni,nj), mavail(ni,nj), &
       rmol(ni,nj), akhs(ni,nj), akms(ni,nj), rib(ni,nj), chs(ni,nj), chs2(ni,nj), cqs2(ni,nj), &
       hfx(ni,nj), qfx(ni,nj), flx_lh(ni,nj), flhc(ni,nj), flqc(ni,nj), qgh(ni,nj), cpm(ni,nj), &
       ct(ni,nj), u10(ni,nj), v10(ni,nj), t02(ni,nj), th02(ni,nj), tshltr(ni,nj), th10(ni,nj), &
       q02(ni,nj), qshltr(ni,nj), q10(ni,nj), pshltr(ni,nj), u10e(ni,nj), v10e(ni,nj), &
       psim(ni,nj), psih(ni,nj), fm(ni,nj), fhh(ni,nj), z0out(ni,nj), rhosfc(ni,nj), &
       xice(ni,nj), ustd(ni,nj), z0d(ni,nj), lowlyr(ni,nj), ivgtyp(ni,nj))
  dz = 0; pmid = 0; pint = 0; th = 0; t = 0; qv = 0; qc = 0; u = 0; v = 0; q2 = 0
  xice = 0; ivgtyp = 0; ustd = 0; z0d = 0

  call rd3('dz', dz, nk); call rd3('pmid', pmid, nk); call rd3('pint', pint, nk+1)
  call rd3('th', th, nk); call rd3('t', t, nk); call rd3('qv', qv, nk); call rd3('qc', qc, nk)
  call rd3('u', u, nk); call rd3('v', v, nk); call rd3('q2', q2, nk)
  call rd('ht', ht, size(ht)); call rd('tsk', tsk, size(tsk)); call rd('qsfc', qsfc, size(qsfc))
  call rd('thz0', thz0, size(thz0)); call rd('qz0', qz0, size(qz0)); call rd('uz0', uz0, size(uz0))
  call rd('vz0', vz0, size(vz0)); call rd('xland', xland, size(xland)); call rd('ust', ust, size(ust))
  call rd('znt', znt, size(znt)); call rd('z0base', z0base, size(z0base)); call rd('pblh', pblh, size(pblh))
  call rd('mavail', mavail, size(mavail)); call rd('akhs', akhs, size(akhs)); call rd('akms', akms, size(akms))
  call rd('rhosfc', rhosfc, size(rhosfc))
  ! out-only arrays start at a sentinel so a missed write shows
  rmol = -999.; rib = -999.; chs = -999.; chs2 = -999.; cqs2 = -999.; hfx = -999.; qfx = -999.
  flx_lh = -999.; flhc = -999.; flqc = -999.; qgh = -999.; cpm = -999.; ct = -999.
  u10 = -999.; v10 = -999.; t02 = -999.; th02 = -999.; tshltr = -999.; th10 = -999.; q02 = -999.
  qshltr = -999.; q10 = -999.; pshltr = -999.; u10e = -999.; v10e = -999.
  psim = -999.; psih = -999.; fm = -999.; fhh = -999.; z0out = -999.

  call MYJSFCINIT(LOWLYR=lowlyr,USTAR=ustd,Z0=z0d,SEAMASK=xland,XICE=xice,IVGTYP=ivgtyp, &
       RESTART=.false.,ALLOWED_TO_READ=.false.,                                         &
       ids=1,ide=ni+1,jds=1,jde=nj+1,kds=1,kde=nk+1,                                    &
       ims=1,ime=ni,jms=1,jme=nj,kms=1,kme=nk+1,                                        &
       its=1,ite=ni,jts=1,jte=nj,kts=1,kte=nk)

  call MYJSFC(ITIMESTEP=itimestep,HT=ht,DZ=dz,PMID=pmid,PINT=pint,TH=th,T=t,QV=qv,QC=qc,  &
       U=u,V=v,Q2=q2,TSK=tsk,QSFC=qsfc,THZ0=thz0,QZ0=qz0,UZ0=uz0,VZ0=vz0,                 &
       LOWLYR=lowlyr,XLAND=xland,IVGTYP=ivgtyp,ISURBAN=0,IZ0TLND=0,                        &
       USTAR=ust,ZNT=znt,Z0BASE=z0base,PBLH=pblh,MAVAIL=mavail,RMOL=rmol,                 &
       AKHS=akhs,AKMS=akms,RIB=rib,CHS=chs,CHS2=chs2,CQS2=cqs2,HFX=hfx,QFX=qfx,           &
       FLX_LH=flx_lh,FLHC=flhc,FLQC=flqc,QGH=qgh,CPM=cpm,CT=ct,                           &
       U10=u10,V10=v10,T02=t02,TH02=th02,TSHLTR=tshltr,TH10=th10,Q02=q02,QSHLTR=qshltr,   &
       Q10=q10,PSHLTR=pshltr,P1000mb=100000.,U10E=u10e,V10E=v10e,                          &
       psim=psim,psih=psih,fm=fm,fhh=fhh,z0out=z0out,usemyjpbl=usemyjpbl,rhosfc=rhosfc,   &
       isftcflx=isftcflx,                                                                 &
       ids=1,ide=ni+1,jds=1,jde=nj+1,kds=1,kde=nk+1,                                      &
       ims=1,ime=ni,jms=1,jme=nj,kms=1,kme=nk+1,                                          &
       its=1,ite=ni,jts=1,jte=nj,kts=1,kte=nk)

  call wr('qsfc', qsfc, size(qsfc)); call wr('thz0', thz0, size(thz0)); call wr('qz0', qz0, size(qz0))
  call wr('uz0', uz0, size(uz0)); call wr('vz0', vz0, size(vz0)); call wr('ustar', ust, size(ust))
  call wr('znt', znt, size(znt)); call wr('pblh', pblh, size(pblh)); call wr('akhs', akhs, size(akhs))
  call wr('akms', akms, size(akms)); call wr('rmol', rmol, size(rmol)); call wr('rib', rib, size(rib))
  call wr('chs', chs, size(chs)); call wr('chs2', chs2, size(chs2)); call wr('cqs2', cqs2, size(cqs2))
  call wr('hfx', hfx, size(hfx)); call wr('qfx', qfx, size(qfx)); call wr('flx_lh', flx_lh, size(flx_lh))
  call wr('flhc', flhc, size(flhc)); call wr('flqc', flqc, size(flqc)); call wr('qgh', qgh, size(qgh))
  call wr('cpm', cpm, size(cpm)); call wr('ct', ct, size(ct)); call wr('u10', u10, size(u10))
  call wr('v10', v10, size(v10)); call wr('t02', t02, size(t02)); call wr('th02', th02, size(th02))
  call wr('tshltr', tshltr, size(tshltr)); call wr('th10', th10, size(th10)); call wr('q02', q02, size(q02))
  call wr('qshltr', qshltr, size(qshltr)); call wr('q10', q10, size(q10)); call wr('pshltr', pshltr, size(pshltr))
  call wr('u10e', u10e, size(u10e)); call wr('v10e', v10e, size(v10e)); call wr('psim', psim, size(psim))
  call wr('psih', psih, size(psih)); call wr('fm', fm, size(fm)); call wr('fhh', fhh, size(fhh))
  call wr('z0out', z0out, size(z0out))

contains

  ! read a Python [k, j, i] array of nl levels into the WRF (i, k, j) layout
  subroutine rd3(name, a, nl)
    character(*), intent(in) :: name
    integer, intent(in) :: nl
    real, intent(inout) :: a(ni, nk+1, nj)
    call rd(name, buf, ni*nj*nl)
    do k = 1, nl
      do j = 1, nj
        do i = 1, ni
          a(i, k, j) = buf(i, j, k)
        enddo
      enddo
    enddo
  end subroutine rd3

end program drv_myjsfc
