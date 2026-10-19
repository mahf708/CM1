! Harness driver for SF_GFDL (src/module_sf_gfdl.F): the GFDL surface layer
! (sfcmodel=4) on ni x nj points, as turb.F calls it (constants from CM1's
! constants module; iwavecpl=0, lcurr_sf=.false., pert_cd=.false.).
! params = [ni, nj, dt, ntsflg, sfenth, icoef_sf, isfflx, isltyp]
! 2-D arrays (1:ni,1:nj); the k=1 inputs as (1:ni,1:nj,1:1).  smois is passed
! with num_soil_layers = nj so that the smois(i,1,j) SF_GFDL reads is the
! input sm(i,j) (CM1 passes a 2-D array there: out of bounds for j > 1).
program drv_sfc_gfdl
  use harness_io
  use constants, only : set_constants, cp, rovcp, rdc => rd, rv, xlv, karman
  use module_sf_gfdl, only : sf_gfdl
  implicit none
  real :: params(8)
  integer :: ni, nj, ntsflg, icoef_sf, isfflx, isl, i, j
  real :: dt, sfenth, ep1, ep2
  real, allocatable, dimension(:,:,:) :: u3d, v3d, t3d, qv3d, p3d, smois
  real, allocatable, dimension(:,:) :: psfc, chs, chs2, cqs2, cpm, znt, mznt, ust, psim, &
       psih, xland, hfx, qfx, taux, tauy, lh, gsw, glw, tsk, flhc, flqc, qgh, qsfc, u10, &
       v10, charn, msang, scurx, scury, gz1oz0, wspd, br, zkmax, cd_out, ch_out, mol, &
       zol, z0t, fm, fh, sm
  integer, allocatable :: isltyp(:,:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); dt = params(3); ntsflg = nint(params(4))
  sfenth = params(5); icoef_sf = nint(params(6)); isfflx = nint(params(7)); isl = nint(params(8))
  call set_constants(0)
  ep1 = rv/rdc - 1.0
  ep2 = rdc/rv

  allocate(u3d(ni,nj,1), v3d(ni,nj,1), t3d(ni,nj,1), qv3d(ni,nj,1), p3d(ni,nj,1), &
       smois(ni,nj,nj))
  allocate(psfc(ni,nj), chs(ni,nj), chs2(ni,nj), cqs2(ni,nj), cpm(ni,nj), znt(ni,nj), &
       mznt(ni,nj), ust(ni,nj), psim(ni,nj), psih(ni,nj), xland(ni,nj), hfx(ni,nj), &
       qfx(ni,nj), taux(ni,nj), tauy(ni,nj), lh(ni,nj), gsw(ni,nj), glw(ni,nj), tsk(ni,nj), &
       flhc(ni,nj), flqc(ni,nj), qgh(ni,nj), qsfc(ni,nj), u10(ni,nj), v10(ni,nj), &
       charn(ni,nj), msang(ni,nj), scurx(ni,nj), scury(ni,nj), gz1oz0(ni,nj), wspd(ni,nj), &
       br(ni,nj), zkmax(ni,nj), cd_out(ni,nj), ch_out(ni,nj), mol(ni,nj), zol(ni,nj), &
       z0t(ni,nj), fm(ni,nj), fh(ni,nj), sm(ni,nj), isltyp(ni,nj))
  chs = 0; chs2 = 0; cqs2 = 0; cpm = 0; mznt = 0; psim = 0; psih = 0; hfx = 0; qfx = 0
  taux = 0; tauy = 0; lh = 0; flhc = 0; flqc = 0; qgh = 0; qsfc = 0; charn = 0; msang = 0
  scurx = 0; scury = 0; gz1oz0 = 0; wspd = 0; br = 0; cd_out = 0; ch_out = 0; mol = 0
  zol = 0; z0t = 0; fm = 0; fh = 0; ust = 0; smois = 0
  isltyp = isl

  call rd('u3d', u3d, size(u3d)); call rd('v3d', v3d, size(v3d))
  call rd('t3d', t3d, size(t3d)); call rd('qv3d', qv3d, size(qv3d))
  call rd('p3d', p3d, size(p3d)); call rd('psfc', psfc, size(psfc))
  call rd('xland', xland, size(xland)); call rd('gsw', gsw, size(gsw))
  call rd('glw', glw, size(glw)); call rd('tsk', tsk, size(tsk))
  call rd('znt', znt, size(znt)); call rd('u10', u10, size(u10)); call rd('v10', v10, size(v10))
  call rd('zkmax', zkmax, size(zkmax)); call rd('smois', sm, size(sm))
  do j = 1, nj
    do i = 1, ni
      smois(i,1,j) = sm(i,j)
    enddo
  enddo

  call SF_GFDL(U3D=u3d,V3D=v3d,T3D=t3d,QV3D=qv3d,P3D=p3d,                         &
       CP=cp,ROVCP=rovcp,R=rdc,XLV=xlv,PSFC=psfc,                                &
       CHS=chs,CHS2=chs2,CQS2=cqs2, CPM=cpm,                                      &
       DT=dt, SMOIS=smois,num_soil_layers=nj,ISLTYP=isltyp,ZNT=znt,               &
       MZNT=mznt,                                                                 &
       UST=ust,PSIM=psim,PSIH=psih,                                               &
       XLAND=xland,HFX=hfx,QFX=qfx,TAUX=taux,TAUY=tauy,LH=lh,GSW=gsw,GLW=glw,     &
       TSK=tsk,FLHC=flhc,FLQC=flqc,                                               &
       QGH=qgh,QSFC=qsfc,U10=u10,V10=v10,                                         &
       ICOEF_SF=icoef_sf,IWAVECPL=0,LCURR_SF=.false.,                             &
       CHARN=charn,MSANG=msang,SCURX=scurx, SCURY=scury,                          &
       pert_Cd=.false., ens_random_seed=99, ens_Cdamp=0.2,                        &
       GZ1OZ0=gz1oz0,WSPD=wspd,BR=br,ZKMAX=zkmax, ISFFLX=isfflx,                  &
       EP1=ep1,EP2=ep2,KARMAN=karman,NTSFLG=ntsflg,SFENTH=sfenth,                 &
       Cd_out=cd_out,Ch_out=ch_out,mol_out=mol,zol_out=zol,z0t_out=z0t,           &
       fm_out=fm,fh_out=fh,                                                       &
       ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=2 ,                      &
       ims=1 ,ime=ni   , jms=1 ,jme=nj   , kms=1 ,kme=1 ,                         &
       its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=1  )

  call wr('tsk', tsk, size(tsk)); call wr('znt', znt, size(znt)); call wr('mznt', mznt, size(mznt))
  call wr('ust', ust, size(ust)); call wr('u10', u10, size(u10)); call wr('v10', v10, size(v10))
  call wr('chs', chs, size(chs)); call wr('chs2', chs2, size(chs2)); call wr('cqs2', cqs2, size(cqs2))
  call wr('cpm', cpm, size(cpm)); call wr('psim', psim, size(psim)); call wr('psih', psih, size(psih))
  call wr('hfx', hfx, size(hfx)); call wr('qfx', qfx, size(qfx)); call wr('lh', lh, size(lh))
  call wr('taux', taux, size(taux)); call wr('tauy', tauy, size(tauy))
  call wr('flhc', flhc, size(flhc)); call wr('flqc', flqc, size(flqc)); call wr('qgh', qgh, size(qgh))
  call wr('qsfc', qsfc, size(qsfc)); call wr('gz1oz0', gz1oz0, size(gz1oz0))
  call wr('wspd', wspd, size(wspd)); call wr('br', br, size(br))
  call wr('cd_out', cd_out, size(cd_out)); call wr('ch_out', ch_out, size(ch_out))
  call wr('mol', mol, size(mol)); call wr('zol', zol, size(zol)); call wr('z0t', z0t, size(z0t))
  call wr('fm', fm, size(fm)); call wr('fh', fh, size(fh))
end program drv_sfc_gfdl
