! Harness driver for the YSU PBL (src/bl_ysu.F bl_ysu_run) called as turb.F's
! ysu_j_loop calls it: ysu_wrapper (src/mmm_physics_wrapper.F) row by row.
! params = [ni, nj, nk, dt]
! 3-D inputs (ni,nj,nk): ux vx tx qvx qcx qix p2d pi2d dz8w rthraten; p2di (ni,nj,nk+1)
! 2-D inputs (ni,nj): psfc znt ust psim psih xland hfx qfx wspd br u10 v10
program drv_ysu
  use harness_io
  use constants, only : set_constants, cp, g, rovcp, rdc => rd, karman, xlv, rv
  use mmm_physics_wrapper_module, only : ysu_wrapper
  implicit none
  real :: params(4)
  integer :: ni, nj, nk, i, j, k, errflg
  character(len=256) :: errmsg
  real :: dt, ep1, ep2, rovg
  real, allocatable, dimension(:,:,:) :: ux, vx, tx, qvx, qcx, qix, p2d, pi2d, dz8w, rth, p2di, &
       utnp, vtnp, ttnp, qvtnp, qctnp, qitnp, exch_hx, exch_mx
  real, allocatable, dimension(:,:) :: psfc, znt, ust, psim, psih, xland, hfx, qfx, wspd, br, &
       u10, v10, hpbl, wstar, delta, rkpbl
  real, allocatable :: t2(:,:,:), t1(:,:), qmix(:,:,:), qmixtnp(:,:,:)
  integer, allocatable :: kpbl(:)

  call rd('params', params, size(params))
  ni = nint(params(1)); nj = nint(params(2)); nk = nint(params(3)); dt = params(4)
  call set_constants(0)
  ep1 = rv/rdc - 1.0
  ep2 = rdc/rv
  rovg = rdc/g

  allocate(ux(ni,nj,nk), vx(ni,nj,nk), tx(ni,nj,nk), qvx(ni,nj,nk), qcx(ni,nj,nk), qix(ni,nj,nk), &
       p2d(ni,nj,nk), pi2d(ni,nj,nk), dz8w(ni,nj,nk), rth(ni,nj,nk), p2di(ni,nj,nk+1), &
       utnp(ni,nj,nk), vtnp(ni,nj,nk), ttnp(ni,nj,nk), qvtnp(ni,nj,nk), qctnp(ni,nj,nk), &
       qitnp(ni,nj,nk), exch_hx(ni,nj,nk), exch_mx(ni,nj,nk))
  allocate(psfc(ni,nj), znt(ni,nj), ust(ni,nj), psim(ni,nj), psih(ni,nj), xland(ni,nj), &
       hfx(ni,nj), qfx(ni,nj), wspd(ni,nj), br(ni,nj), u10(ni,nj), v10(ni,nj), hpbl(ni,nj), &
       wstar(ni,nj), delta(ni,nj), rkpbl(ni,nj))
  allocate(t2(ni,nk+1,24), t1(ni,5), qmix(ni,nk,1), qmixtnp(ni,nk,1), kpbl(ni))

  call rd('ux', ux, size(ux)); call rd('vx', vx, size(vx)); call rd('tx', tx, size(tx))
  call rd('qvx', qvx, size(qvx)); call rd('qcx', qcx, size(qcx)); call rd('qix', qix, size(qix))
  call rd('p2d', p2d, size(p2d)); call rd('pi2d', pi2d, size(pi2d)); call rd('dz8w', dz8w, size(dz8w))
  call rd('rthraten', rth, size(rth)); call rd('p2di', p2di, size(p2di))
  call rd('psfc', psfc, size(psfc)); call rd('znt', znt, size(znt)); call rd('ust', ust, size(ust))
  call rd('psim', psim, size(psim)); call rd('psih', psih, size(psih)); call rd('xland', xland, size(xland))
  call rd('hfx', hfx, size(hfx)); call rd('qfx', qfx, size(qfx)); call rd('wspd', wspd, size(wspd))
  call rd('br', br, size(br)); call rd('u10', u10, size(u10)); call rd('v10', v10, size(v10))

  do j = 1, nj
    t2 = 0.0; t1 = 0.0; qmix = 0.0
    t1(:,1) = 1.0; t1(:,2) = 1.0
    call ysu_wrapper(ux=ux(:,j,:), vx=vx(:,j,:), tx=tx(:,j,:), qvx=qvx(:,j,:), qcx=qcx(:,j,:), &
         qix=qix(:,j,:), nmix=1, qmix=qmix, p2d=p2d(:,j,:), p2di=p2di(:,j,:), pi2d=pi2d(:,j,:), &
         f_qc=.true., f_qi=.true., &
         utnp=t2(:,1:nk,1), vtnp=t2(:,1:nk,2), ttnp=t2(:,1:nk,3), qvtnp=t2(:,1:nk,4), &
         qctnp=t2(:,1:nk,5), qitnp=t2(:,1:nk,6), qmixtnp=qmixtnp, &
         cp=cp, g=g, rovcp=rovcp, rd=rdc, rovg=rovg, ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=rv, &
         dz8w2d=dz8w(:,j,:), psfcpa=psfc(:,j), znt=znt(:,j), ust=ust(:,j), hpbl=hpbl(:,j), &
         psim=psim(:,j), psih=psih(:,j), xland=xland(:,j), hfx=hfx(:,j), qfx=qfx(:,j), &
         wspd=wspd(:,j), br=br(:,j), dt=dt, kpbl1d=kpbl, &
         exch_hx=t2(:,:,7), exch_mx=t2(:,:,8), wstar=wstar(:,j), delta=delta(:,j), &
         u10=u10(:,j), v10=v10(:,j), uox=t1(:,3), vox=t1(:,4), rthraten=rth(:,j,:), &
         ysu_topdown_pblmix=.false., ctopo=t1(:,1), ctopo2=t1(:,2), &
         flag_bep=.false., its=1, ite=ni, kts=1, kte=nk, errmsg=errmsg, errflg=errflg)
    do k = 1, nk
      utnp(:,j,k) = t2(:,k,1); vtnp(:,j,k) = t2(:,k,2); ttnp(:,j,k) = t2(:,k,3)
      qvtnp(:,j,k) = t2(:,k,4); qctnp(:,j,k) = t2(:,k,5); qitnp(:,j,k) = t2(:,k,6)
      exch_hx(:,j,k) = t2(:,k,7); exch_mx(:,j,k) = t2(:,k,8)
    enddo
    do i = 1, ni
      rkpbl(i,j) = kpbl(i)
    enddo
  enddo

  call wr('utnp', utnp, size(utnp)); call wr('vtnp', vtnp, size(vtnp)); call wr('ttnp', ttnp, size(ttnp))
  call wr('qvtnp', qvtnp, size(qvtnp)); call wr('qctnp', qctnp, size(qctnp)); call wr('qitnp', qitnp, size(qitnp))
  call wr('exch_hx', exch_hx, size(exch_hx)); call wr('exch_mx', exch_mx, size(exch_mx))
  call wr('hpbl', hpbl, size(hpbl)); call wr('kpbl', rkpbl, size(rkpbl)); call wr('wstar', wstar, size(wstar))
  call wr('delta', delta, size(delta)); call wr('u10', u10, size(u10)); call wr('v10', v10, size(v10))
end program drv_ysu
