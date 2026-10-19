! Harness driver for BL_GFSEDMF (src/module_bl_gfsedmf.F): the GFS-EDMF PBL
! (ipbl=3) on ni x nj columns of nk levels, as turb.F calls it (constants
! from CM1's constants module; pert_pbl = .false.).
! params = [ni, nj, nk, dt, p_qi, alpha, var_ric, coef_ric_l, coef_ric_s, disheat]
! Arrays are (1:ni,1:nj,1:nk) / (1:ni,1:nj); kpbl2d is written as real.
program drv_pbl_gfs
  use harness_io
  use constants, only : set_constants, cp, g, rovcp, rdc => rd, rv, karman
  use module_bl_gfsedmf, only : bl_gfsedmf
  implicit none
  real :: params(10)
  integer :: ni, nj, nk, p_qi
  logical :: disheat
  real :: dt, alpha, var_ric, coef_ric_l, coef_ric_s, ep1, rovg
  real, allocatable, dimension(:,:,:) :: u3d, v3d, th3d, t3d, qv3d, qc3d, qi3d, p3d, pi3d, &
       dz8w, z, rthraten, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, &
       dku3d, dkt3d, dissten, dheat
  real, allocatable, dimension(:,:) :: psfc, ust, pbl, psim, psih, hfx, qfx, tsk, gz1oz0, &
       wspd, br, hpbl2d, evap2d, heat2d, u10, v10, znt, xland, brcr, wscale, wscaleu, &
       prkpp, stau, kpblr
  integer, allocatable :: kpbl2d(:,:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3))
  dt = params(4); p_qi = nint(params(5)); alpha = params(6); var_ric = params(7)
  coef_ric_l = params(8); coef_ric_s = params(9); disheat = nint(params(10)) == 1
  call set_constants(0)
  ep1 = rv/rdc - 1.0
  rovg = rdc/g

  allocate(u3d(ni,nj,nk), v3d(ni,nj,nk), th3d(ni,nj,nk), t3d(ni,nj,nk), qv3d(ni,nj,nk), &
       qc3d(ni,nj,nk), qi3d(ni,nj,nk), p3d(ni,nj,nk), pi3d(ni,nj,nk), dz8w(ni,nj,nk), &
       z(ni,nj,nk), rthraten(ni,nj,nk), rublten(ni,nj,nk), rvblten(ni,nj,nk), &
       rthblten(ni,nj,nk), rqvblten(ni,nj,nk), rqcblten(ni,nj,nk), rqiblten(ni,nj,nk), &
       dku3d(ni,nj,nk), dkt3d(ni,nj,nk), dissten(ni,nj,nk), dheat(ni,nj,nk))
  allocate(psfc(ni,nj), ust(ni,nj), pbl(ni,nj), psim(ni,nj), psih(ni,nj), hfx(ni,nj), &
       qfx(ni,nj), tsk(ni,nj), gz1oz0(ni,nj), wspd(ni,nj), br(ni,nj), hpbl2d(ni,nj), &
       evap2d(ni,nj), heat2d(ni,nj), u10(ni,nj), v10(ni,nj), znt(ni,nj), xland(ni,nj), &
       brcr(ni,nj), wscale(ni,nj), wscaleu(ni,nj), prkpp(ni,nj), stau(ni,nj), &
       kpblr(ni,nj), kpbl2d(ni,nj))
  th3d = 0; rublten = 0; rvblten = 0; rthblten = 0; rqvblten = 0; rqcblten = 0
  rqiblten = 0; dku3d = 0; dkt3d = 0; dissten = 0; dheat = 0; qi3d = 0
  ust = 0; pbl = 0; brcr = 0; wscale = 0; wscaleu = 0; prkpp = 0; stau = 0

  call rd('u3d', u3d, size(u3d)); call rd('v3d', v3d, size(v3d))
  call rd('t3d', t3d, size(t3d)); call rd('qv3d', qv3d, size(qv3d))
  call rd('qc3d', qc3d, size(qc3d))
  if (p_qi >= 1) call rd('qi3d', qi3d, size(qi3d))
  call rd('p3d', p3d, size(p3d)); call rd('pi3d', pi3d, size(pi3d))
  call rd('dz8w', dz8w, size(dz8w)); call rd('z', z, size(z))
  call rd('rthraten', rthraten, size(rthraten))
  call rd('psfc', psfc, size(psfc)); call rd('psim', psim, size(psim))
  call rd('psih', psih, size(psih)); call rd('hfx', hfx, size(hfx))
  call rd('qfx', qfx, size(qfx)); call rd('tsk', tsk, size(tsk))
  call rd('gz1oz0', gz1oz0, size(gz1oz0)); call rd('wspd', wspd, size(wspd))
  call rd('br', br, size(br)); call rd('u10', u10, size(u10)); call rd('v10', v10, size(v10))
  call rd('znt', znt, size(znt)); call rd('xland', xland, size(xland))

  call BL_GFSEDMF(U3D=u3d,V3D=v3d,TH3D=th3d,T3D=t3d,QV3D=qv3d,                 &
       QC3D=qc3d,QI3D=qi3d,P3D=p3d,PI3D=pi3d,                                  &
       RUBLTEN=rublten,RVBLTEN=rvblten,RTHBLTEN=rthblten,                      &
       RQVBLTEN=rqvblten,RQCBLTEN=rqcblten,RQIBLTEN=rqiblten,                  &
       CP=cp,G=g,ROVCP=rovcp,R=rdc,ROVG=rovg,                                  &
       P_QI=p_qi,P_FIRST_SCALAR=1,                                             &
       dz8w=dz8w,z=z,PSFC=psfc,                                                &
       UST=ust,PBL=pbl,PSIM=psim,PSIH=psih,                                    &
       HFX=hfx,QFX=qfx,TSK=tsk,GZ1OZ0=gz1oz0,WSPD=wspd,BR=br,                  &
       DT=dt,KPBL2D=kpbl2d,EP1=ep1,KARMAN=karman,                              &
       DISHEAT=disheat,                                                        &
       RTHRATEN=rthraten,                                                      &
       HPBL2D=hpbl2d, EVAP2D=evap2d, HEAT2D=heat2d,                            &
       U10=u10,V10=v10,ZNT=znt,                                                &
       DKU3D=dku3d,DKT3D=dkt3d,                                                &
       VAR_RIC=var_ric,coef_ric_l=coef_ric_l,coef_ric_s=coef_ric_s,            &
       alpha=alpha,xland=xland, pert_pbl=.false.,                              &
       ens_random_seed=99, ens_pblamp=0.2,                                     &
       brcr=brcr,wscale=wscale,wscaleu=wscaleu,prkpp=prkpp,                    &
       stau=stau,dissten=dissten,dheat=dheat,                                  &
       ids=1  ,ide=ni+1 , jds= 1 ,jde=nj+1 , kds=1  ,kde=nk+1 ,                &
       ims=1 ,ime=ni   , jms=1 ,jme=nj   , kms=1 ,kme=nk ,                     &
       its=1  ,ite=ni   , jts=1  ,jte=nj   , kts=1  ,kte=nk              )

  kpblr = real(kpbl2d)
  call wr('rublten', rublten, size(rublten)); call wr('rvblten', rvblten, size(rvblten))
  call wr('rthblten', rthblten, size(rthblten)); call wr('rqvblten', rqvblten, size(rqvblten))
  call wr('rqcblten', rqcblten, size(rqcblten)); call wr('rqiblten', rqiblten, size(rqiblten))
  call wr('dku3d', dku3d, size(dku3d)); call wr('dkt3d', dkt3d, size(dkt3d))
  call wr('dissten', dissten, size(dissten)); call wr('dheat', dheat, size(dheat))
  call wr('ust', ust, size(ust)); call wr('wspd', wspd, size(wspd))
  call wr('hpbl', pbl, size(pbl)); call wr('kpbl2d', kpblr, size(kpblr))
end program drv_pbl_gfs
