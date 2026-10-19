! Harness driver for the MYJ PBL (src/module_bl_myjpbl.F): MYJPBL on
! ni x nj columns at once (ims:ime = 1:ni, jms:jme = 1:nj, kms:kme = 1:nk+1,
! kts:kte = 1:nk), as turb.F calls it (STEPBL = 1, LOWLYR = 1, QCI present,
! QCS/QCR/QCG and SICE/SNOW zero).
! params = [ni, nj, nk, dt]
! 3-D inputs arrive as Python [k, j, i] = Fortran (i, j, k), k = 1..nk
! (pint: 1..nk+1); 2-D inputs as (i, j).  3-D outputs go back the same way.
program drv_myjpbl
  use harness_io
  use module_bl_myjpbl, only : myjpbl
  implicit none
  real :: params(4), dt
  integer :: ni, nj, nk, i, j, k
  real, allocatable, dimension(:,:,:) :: buf, dz, pmid, pint, th, t, exner, qv, qcw, qci, &
       qcs, qcr, qcg, u, v, rho, tke, exch_h, exch_m, el, rub, rvb, rthb, rqvb, rqcb, rqib, &
       rqsb, rqrb, rqgb
  real, allocatable, dimension(:,:) :: ht, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0, xland, &
       sice, snow, ust, znt, pblh, ct, akhs, akms, elflx, mixht, kpblr
  integer, allocatable, dimension(:,:) :: lowlyr, kpbl

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3)); dt = params(4)

  allocate(buf(ni,nj,nk+1))
  allocate(dz(ni,nk+1,nj), pmid(ni,nk+1,nj), pint(ni,nk+1,nj), th(ni,nk+1,nj), t(ni,nk+1,nj), &
       exner(ni,nk+1,nj), qv(ni,nk+1,nj), qcw(ni,nk+1,nj), qci(ni,nk+1,nj), qcs(ni,nk+1,nj), &
       qcr(ni,nk+1,nj), qcg(ni,nk+1,nj), u(ni,nk+1,nj), v(ni,nk+1,nj), rho(ni,nk+1,nj), &
       tke(ni,nk+1,nj), exch_h(ni,nk+1,nj), exch_m(ni,nk+1,nj), el(ni,nk+1,nj), &
       rub(ni,nk+1,nj), rvb(ni,nk+1,nj), rthb(ni,nk+1,nj), rqvb(ni,nk+1,nj), rqcb(ni,nk+1,nj), &
       rqib(ni,nk+1,nj), rqsb(ni,nk+1,nj), rqrb(ni,nk+1,nj), rqgb(ni,nk+1,nj))
  allocate(ht(ni,nj), tsk(ni,nj), qsfc(ni,nj), chklowq(ni,nj), thz0(ni,nj), qz0(ni,nj), &
       uz0(ni,nj), vz0(ni,nj), xland(ni,nj), sice(ni,nj), snow(ni,nj), ust(ni,nj), znt(ni,nj), &
       pblh(ni,nj), ct(ni,nj), akhs(ni,nj), akms(ni,nj), elflx(ni,nj), mixht(ni,nj), &
       kpblr(ni,nj), lowlyr(ni,nj), kpbl(ni,nj))
  dz = 0; pmid = 0; pint = 0; th = 0; t = 0; exner = 0; qv = 0; qcw = 0; qci = 0
  qcs = 0; qcr = 0; qcg = 0; u = 0; v = 0; rho = 0; tke = 0; exch_h = 0; exch_m = 0
  el = -999.; rub = -999.; rvb = -999.; rthb = -999.; rqvb = -999.; rqcb = -999.; rqib = -999.
  rqsb = 0; rqrb = 0; rqgb = 0
  sice = 0; snow = 0; znt = 0; lowlyr = 1; pblh = -999.; mixht = -999.; kpbl = -999

  call rd3('dz', dz, nk); call rd3('pmid', pmid, nk); call rd3('pint', pint, nk+1)
  call rd3('th', th, nk); call rd3('t', t, nk); call rd3('exner', exner, nk)
  call rd3('qv', qv, nk); call rd3('qcw', qcw, nk); call rd3('qci', qci, nk)
  call rd3('u', u, nk); call rd3('v', v, nk); call rd3('tke_myj', tke, nk)
  call rd('ht', ht, size(ht)); call rd('tsk', tsk, size(tsk)); call rd('qsfc', qsfc, size(qsfc))
  call rd('chklowq', chklowq, size(chklowq)); call rd('thz0', thz0, size(thz0))
  call rd('qz0', qz0, size(qz0)); call rd('uz0', uz0, size(uz0)); call rd('vz0', vz0, size(vz0))
  call rd('xland', xland, size(xland)); call rd('ust', ust, size(ust)); call rd('ct', ct, size(ct))
  call rd('akhs', akhs, size(akhs)); call rd('akms', akms, size(akms)); call rd('elflx', elflx, size(elflx))

  call MYJPBL(DT=dt,STEPBL=1,HT=ht,DZ=dz,PMID=pmid,PINT=pint,TH=th,T=t,EXNER=exner,QV=qv,     &
       QCW=qcw,QCI=qci,QCS=qcs,QCR=qcr,QCG=qcg,U=u,V=v,RHO=rho,TSK=tsk,QSFC=qsfc,               &
       CHKLOWQ=chklowq,THZ0=thz0,QZ0=qz0,UZ0=uz0,VZ0=vz0,LOWLYR=lowlyr,XLAND=xland,             &
       SICE=sice,SNOW=snow,TKE_MYJ=tke,EXCH_H=exch_h,EXCH_M=exch_m,USTAR=ust,ZNT=znt,          &
       EL_MYJ=el,PBLH=pblh,KPBL=kpbl,CT=ct,AKHS=akhs,AKMS=akms,ELFLX=elflx,MIXHT=mixht,        &
       RUBLTEN=rub,RVBLTEN=rvb,RTHBLTEN=rthb,RQVBLTEN=rqvb,RQCBLTEN=rqcb,                       &
       RQIBLTEN=rqib,RQSBLTEN=rqsb,RQRBLTEN=rqrb,RQGBLTEN=rqgb,                                 &
       ids=1,ide=ni+1,jds=1,jde=nj+1,kds=1,kde=nk+1,                                            &
       ims=1,ime=ni,jms=1,jme=nj,kms=1,kme=nk+1,                                                &
       its=1,ite=ni,jts=1,jte=nj,kts=1,kte=nk)

  kpblr = real(kpbl)
  call wr3('tke_myj', tke); call wr3('el_myj', el); call wr3('exch_h', exch_h)
  call wr3('exch_m', exch_m); call wr3('rublten', rub); call wr3('rvblten', rvb)
  call wr3('rthblten', rthb); call wr3('rqvblten', rqvb); call wr3('rqcblten', rqcb)
  call wr3('rqiblten', rqib)
  call wr('pblh', pblh, size(pblh)); call wr('kpbl', kpblr, size(kpblr)); call wr('mixht', mixht, size(mixht))
  call wr('ct', ct, size(ct)); call wr('qsfc', qsfc, size(qsfc)); call wr('thz0', thz0, size(thz0))
  call wr('qz0', qz0, size(qz0))

contains

  ! Python [k, j, i] array of nl levels <-> the WRF (i, k, j) layout
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

  subroutine wr3(name, a)
    character(*), intent(in) :: name
    real, intent(in) :: a(ni, nk+1, nj)
    do k = 1, nk
      do j = 1, nj
        do i = 1, ni
          buf(i, j, k) = a(i, k, j)
        enddo
      enddo
    enddo
    call wr(name, buf, ni*nj*nk)
  end subroutine wr3

end program drv_myjpbl
